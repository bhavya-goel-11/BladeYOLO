"""G29 (research_log.md): RF-DETR with a frozen DINOv3 prior ("FP").

RF-DETR-M fine-tunes its DINOv2-S encoder end to end. FP adds a second, frozen encoder: DINOv3 ViT-B/16 (blocks 4/8/12,
final norm, stride 16, the same blocks BladeYOLO-v2 ViT-B fused). Its features are added to every output level of
RF-DETR's projector, i.e. to the memory the transformer encodes, selects its two-stage proposals from and decodes from,
through a zero-initialised fusion layer:

    feat' = feat + Out(DW3x3(GELU(LN(Mix([feat, GELU(Reduce(LN(prior)))])))))     Out = 1x1 conv, weights and bias 0

so at step 0 the model is exactly RF-DETR (tests/test_g29_prior.py), and training learns how much prior to mix in.

Why (research_log.md, G29 pre-registration): frozen DINOv3 carried all of BladeYOLO-v2's advantage at every label
fraction (§22, §24; ViT-B > ViT-S); on WTBlade-group the frozen-DINOv3 ViT-B YOLO (18.0) beats fine-tuned RF-DETR-M
(16.2 / 16.9, G27 A-2), and on WTBD DINOv3 ViT-B box embeddings re-rank RF-DETR's outputs (+0.025, H18/H18-R). Fine-
tuning moves RF-DETR's own features away from generic neighbourhoods (G22-O) and adapting its encoder further
collapsed it (G27 D1), so the prior is kept frozen and separate instead of being written into the fine-tuned encoder.

The DINOv3 model is held in a process-wide cache, not as a submodule: it is not in the optimizer, not in the EMA copy,
and not in checkpoints (those carry only the fusion layers, 0.79M parameters on Medium). The input is RF-DETR's own
ImageNet-normalised tensor, which is DINOv3's normalisation too; patch 16 on both, so the grids coincide.

  install(size="b")   patches rfdetr.models.backbone.backbone.Backbone (__init__, forward, forward_export) for the rest
                      of the process; call it before the RF-DETR model is built (train and predict alike)
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE = {}  # (size, device) -> frozen DINOv3


def _dino_weights():
    """models/dino_weights.py without importing the models package (which needs Ultralytics, absent on RF-DETR
    sessions)."""
    sys.path.insert(0, os.path.join(ROOT, "models"))
    try:
        import dino_weights
    finally:
        sys.path.pop(0)
    return dino_weights


def prior_spec(size):
    _, dim, _, blocks = _dino_weights().SIZES[size]
    return dim, blocks


def prior_model(size, device):
    key = (size, str(device))
    if key not in _CACHE:
        m = _dino_weights().build_dinov3(size).to(device).eval()
        m.requires_grad_(False)
        _CACHE[key] = m
    return _CACHE[key]


@torch.no_grad()
def prior_features(x, size):
    """(B, 3, H, W) ImageNet-normalised -> (B, len(blocks) * dim, H/16, W/16), frozen, no gradient."""
    _, blocks = prior_spec(size)
    m = prior_model(size, x.device)
    feats = m.get_intermediate_layers(x, n=blocks, reshape=True, norm=True)
    return torch.cat(list(feats), 1)


class LayerNorm2d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class PriorFuse(nn.Module):
    """feat + zero-initialised projection of [feat, reduced prior]; identity at initialisation."""

    def __init__(self, c_prior, c):
        super().__init__()
        self.prior_norm = LayerNorm2d(c_prior)
        self.reduce = nn.Conv2d(c_prior, c, 1)
        self.mix = nn.Conv2d(2 * c, c, 1)
        self.mix_norm = LayerNorm2d(c)
        self.dw = nn.Conv2d(c, c, 3, padding=1, groups=c)
        self.out = nn.Conv2d(c, c, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, feat, prior):
        prior = prior.to(feat.dtype)
        if prior.shape[-2:] != feat.shape[-2:]:  # P3 / P5 levels of larger RF-DETR sizes
            prior = F.interpolate(prior, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        s = F.gelu(self.reduce(self.prior_norm(prior)))
        h = F.gelu(self.mix_norm(self.mix(torch.cat([feat, s], 1))))
        return feat + self.out(F.gelu(self.dw(h)))


def install(size="b"):
    """Patch RF-DETR's Backbone so every model built afterwards carries `prior_fuse` and adds the frozen prior."""
    from rfdetr.models.backbone.backbone import Backbone
    from rfdetr.utilities.tensors import NestedTensor

    if getattr(Backbone, "_g29_size", None) is not None:
        assert Backbone._g29_size == size, (Backbone._g29_size, size)
        return
    orig_init, orig_forward, orig_export = Backbone.__init__, Backbone.forward, Backbone.forward_export
    dim, blocks = prior_spec(size)

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        c = kwargs.get("out_channels", 256)
        self.prior_fuse = nn.ModuleList(PriorFuse(dim * len(blocks), c) for _ in self.projector_scale)

    def forward(self, tensor_list):
        out, cross = orig_forward(self, tensor_list)
        prior = prior_features(tensor_list.tensors, size)
        out = [NestedTensor(f(nt.tensors, prior), nt.mask, nt.no_padding) for f, nt in zip(self.prior_fuse, out)]
        return out, cross

    def forward_export(self, tensors):
        feats, masks, cross = orig_export(self, tensors)
        prior = prior_features(tensors, size)
        return [f(x, prior) for f, x in zip(self.prior_fuse, feats)], masks, cross

    Backbone.__init__, Backbone.forward, Backbone.forward_export = __init__, forward, forward_export
    Backbone._g29_size = size
    Backbone._g29_orig = (orig_init, orig_forward, orig_export)  # tests: the unpatched path, for the identity check
    print(f"G29 frozen prior installed: DINOv3 ViT-{size.upper()}/16 blocks {[b + 1 for b in blocks]}, "
          f"{dim * len(blocks)} channels into every projector level", flush=True)
