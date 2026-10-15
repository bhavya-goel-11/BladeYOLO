"""Frozen DINOv3 semantics fused into a pretrained CNN backbone (BladeYOLO-v2).

The YOLOv12-L backbone keeps its layer indices, so its COCO weights load unchanged (see models.trainer). DINOv3 needs
the raw image, which Ultralytics only gives to layer 0, so layer 0 is a Conv that also runs the ViT and attaches its
feature maps to its output tensor. DinoFuse layers take that output plus a backbone level and add the (resized,
projected) DINOv3 features to it through a zero-initialised projection: at the start of training the backbone
features pass through unchanged, and the network learns how much semantics to mix in.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv

from .dino_weights import build_dinov3_vits16


class DinoConv(Conv):
    """Conv layer 0 of the backbone that also extracts frozen DINOv3 ViT-S/16 features from the input image.

    The features (blocks 4/8/12, stride 16) are attached to the output as `.dino` for DinoFuse layers. The ViT lives
    under `.dino`, so the trainer's freeze pattern "0.dino" keeps it frozen; train() keeps it in eval mode.
    """

    dino_blocks = (3, 7, 11)
    dino_dim = 384

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__(c1, c2, k, s, p, g, d, act)
        self.dino = build_dinov3_vits16()
        self.dino.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def train(self, mode=True):
        super().train(mode)
        self.dino.eval()  # frozen: no dropout / drop-path
        return self

    def _semantics(self, x):
        with torch.no_grad():
            return self.dino.get_intermediate_layers((x - self.mean) / self.std, n=self.dino_blocks, reshape=True, norm=True)

    def forward(self, x):
        y = super().forward(x)
        y.dino = self._semantics(x)
        return y

    def forward_fuse(self, x):  # Ultralytics swaps this in after fusing Conv+BN for inference
        y = super().forward_fuse(x)
        y.dino = self._semantics(x)
        return y


class DinoFuse(nn.Module):
    """x = [layer-0 output carrying .dino, backbone feature map]: feature + zero-init projection of DINOv3 block `level`."""

    def __init__(self, level, c):
        super().__init__()
        self.level = level
        self.proj = Conv(c + DinoConv.dino_dim, c, 1)
        self.mix = Conv(c, c, 3, g=c)
        self.out = nn.Conv2d(c, c, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        src, feat = x
        sem = src.dino[self.level].to(feat.dtype)
        if sem.shape[2:] != feat.shape[2:]:
            sem = F.interpolate(sem, size=feat.shape[2:], mode="bilinear", align_corners=False)
        return feat + self.out(self.mix(self.proj(torch.cat([feat, sem], 1))))
