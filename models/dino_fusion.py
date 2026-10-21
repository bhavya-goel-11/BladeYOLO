"""Frozen DINOv3 semantics fused into a pretrained CNN backbone (BladeYOLO-v2).

The YOLOv12-L backbone keeps its layer indices, so its COCO weights load unchanged (see models.trainer). DINOv3 needs
the raw image, which Ultralytics only gives to layer 0, so layer 0 is a Conv that also runs the ViT and attaches its
feature maps to its output tensor. DinoFuse layers take that output plus a backbone level and add the (resized,
projected) DINOv3 features to it through a zero-initialised projection: at the start of training the backbone
features pass through unchanged, and the network learns how much semantics to mix in.

Optionally (train.py --dino-lora R) DINOv3 is adapted with LoRA: every attention qkv and output projection gets a
trainable rank-R update whose up-projection starts at zero, so training starts from exactly the frozen features.

Optionally (train.py --retrieval) layer 0 also attaches `.retr`, a per-class prior from the nearest labelled
patches (models/retrieval.py), and DinoFuse takes it as extra input channels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv

from .dino_weights import SIZES, build_dinov3, dino_lora_rank, dino_size
from .retrieval import retrieval_config, retrieval_prior


class LoRA(nn.Module):
    """Low-rank update x -> B(A(x)) * alpha / r; B starts at zero, so the update starts at zero."""

    def __init__(self, d_in, d_out, r, alpha=None):
        super().__init__()
        self.A = nn.Linear(d_in, r, bias=False)
        self.B = nn.Linear(r, d_out, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=5**0.5)
        nn.init.zeros_(self.B.weight)
        self.scale = (alpha or r) / r

    def forward(self, x):
        return self.B(self.A(x)) * self.scale


class LoRALinear(nn.Module):
    """A frozen DINOv3 linear layer plus a LoRA update.

    The update is registered on DinoConv (`dino_lora`), not here: the trainer freezes every parameter under
    `0.dino.`, and `0.dino_lora.` falls outside that pattern. This module only holds a reference to it.
    """

    def __init__(self, base, lora):
        super().__init__()
        self.base = base
        self.__dict__["lora"] = lora  # unregistered reference; owned by DinoConv.dino_lora
        self.in_features, self.out_features = base.in_features, base.out_features  # read by DINOv3's attention

    def forward(self, x):
        return self.base(x) + self.lora(x)


class DinoConv(Conv):
    """Conv layer 0 of the backbone that also extracts frozen DINOv3 features from the input image.

    The ViT size comes from BLADEYOLO_DINO (train.py --dino; ViT-S/16 by default). The features (three blocks at 1/3,
    2/3 and 3/3 depth, stride 16; ViT-S/B: blocks 4/8/12) are attached to the output as `.dino` for DinoFuse layers.
    The ViT lives under `.dino`, so the trainer's freeze pattern "0.dino" keeps it frozen; train() keeps it in eval
    mode. Size, width and blocks are stored per instance; checkpoints from before (ViT-S) fall back to the class values.
    With a LoRA rank (BLADEYOLO_DINO_LORA) the attention layers get trainable updates in `.dino_lora` and the features
    are computed with gradients while training.
    """

    dino_blocks = (3, 7, 11)
    dino_dim = 384

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__(c1, c2, k, s, p, g, d, act)
        self.dino_size = dino_size()
        _, self.dino_dim, _, self.dino_blocks = SIZES[self.dino_size]
        self.dino = build_dinov3(self.dino_size)
        self.dino.requires_grad_(False)
        r = dino_lora_rank()
        self.dino_lora = None  # frozen (older checkpoints have no attribute: _semantics treats that as None)
        if r > 0:
            self.dino_lora = nn.ModuleList()
            for blk in self.dino.blocks:
                for name in ("qkv", "proj"):
                    base = getattr(blk.attn, name)
                    lora = LoRA(base.in_features, base.out_features, r)
                    self.dino_lora.append(lora)
                    setattr(blk.attn, name, LoRALinear(base, lora))
        path, nc = retrieval_config()
        self.retr_nc = 0  # no retrieval prior (older checkpoints have no attribute: treated as 0)
        if nc:
            mem = torch.load(path, map_location="cpu")
            if mem["nc"] != nc or mem["size"] != self.dino_size:
                raise ValueError(f"retrieval memory {path} is for nc={mem['nc']}, DINOv3 {mem['size']!r}")
            self.register_buffer("mem_feat", mem["feat"])  # persistent: checkpoints carry the memory to test time
            self.register_buffer("mem_cls", mem["cls"])
            self.retr_nc, self.retr_w_bg, self.retr_cap, self.retr_k = nc, mem["w_bg"], mem["cap"], mem["k"]
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def train(self, mode=True):
        super().train(mode)
        self.dino.eval()  # frozen: no dropout / drop-path
        return self

    def _semantics(self, x):
        with torch.set_grad_enabled(getattr(self, "dino_lora", None) is not None and torch.is_grad_enabled()):
            return self.dino.get_intermediate_layers((x - self.mean) / self.std, n=self.dino_blocks, reshape=True, norm=True)

    def _attach(self, y, x):
        y.dino = self._semantics(x)
        if getattr(self, "retr_nc", 0):
            y.retr = retrieval_prior(y.dino[-1].detach(), self.mem_feat, self.mem_cls, self.retr_w_bg, self.retr_nc,
                                     self.retr_cap, self.retr_k)
        return y

    def forward(self, x):
        return self._attach(super().forward(x), x)

    def forward_fuse(self, x):  # Ultralytics swaps this in after fusing Conv+BN for inference
        return self._attach(super().forward_fuse(x), x)


class DinoFuse(nn.Module):
    """x = [layer-0 output carrying .dino, backbone feature map]: feature + zero-init projection of DINOv3 block `level`
    (and of the retrieval prior `.retr`, when layer 0 has one)."""

    def __init__(self, level, c):
        super().__init__()
        self.level = level
        self.retr_nc = retrieval_config()[1]
        self.proj = Conv(c + SIZES[dino_size()][1] + self.retr_nc, c, 1)  # the width of the DINOv3 that DinoConv builds
        self.mix = Conv(c, c, 3, g=c)
        self.out = nn.Conv2d(c, c, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        src, feat = x
        sem = src.dino[self.level].to(feat.dtype)
        if getattr(self, "retr_nc", 0):
            sem = torch.cat([sem, src.retr.to(feat.dtype)], 1)
        if sem.shape[2:] != feat.shape[2:]:
            sem = F.interpolate(sem, size=feat.shape[2:], mode="bilinear", align_corners=False)
        return feat + self.out(self.mix(self.proj(torch.cat([feat, sem], 1))))
