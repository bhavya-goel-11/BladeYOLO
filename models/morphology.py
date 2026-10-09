import math

import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d
from ultralytics.nn.modules.conv import Conv


class DeformConv(nn.Module):
    """Modulated deformable convolution (DCNv2) + BN + SiLU.

    Each kernel tap learns a 2D offset and a [0, 1] modulation weight, so the sampling grid can bend
    along irregular defect contours and ignore background taps. Offsets start at zero and modulation
    at 0.5, i.e. the layer starts as a (scaled) regular convolution.
    """

    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        self.k, self.s, self.p = k, s, k // 2
        self.offset_mask = nn.Conv2d(c1, 3 * k * k, k, s, self.p)
        nn.init.zeros_(self.offset_mask.weight)
        nn.init.zeros_(self.offset_mask.bias)
        self.weight = nn.Parameter(torch.empty(c2, c1, k, k))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        offset, mask = self.offset_mask(x).split([2 * self.k * self.k, self.k * self.k], 1)
        x = deform_conv2d(x, offset, self.weight, stride=self.s, padding=self.p, mask=mask.sigmoid())
        return self.act(self.bn(x))


class DeformBottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, e=1.0):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 3)
        self.cv2 = DeformConv(c_, c2, 3)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2fMorph(nn.Module):
    """C2f CSP block whose bottlenecks use deformable convolutions (defect-contour tracing)."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(DeformBottleneck(self.c, self.c, shortcut) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
