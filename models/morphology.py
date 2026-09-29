import torch
import torch.nn as nn
import math
from torchvision.ops import deform_conv2d
from ultralytics.nn.modules.conv import Conv

class MorphologicalConv(nn.Module):
    """
    Morphological Deformable Convolution.
    Learns 2D offsets for the convolution kernel to perfectly trace irregular 
    defect edges like branching cracks and peeling contours, eliminating rigid 
    background noise.
    """
    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, act=True):
        super().__init__()
        self.in_channels = c1
        self.out_channels = c2
        self.kernel_size = k
        self.stride = s
        self.padding = p
        self.groups = g
        
        # Learnable offsets for the deformable kernel (2 * k * k channels)
        self.offset_conv = nn.Conv2d(c1, 2 * k * k, kernel_size=k, stride=s, padding=p, bias=True)
        # Initialize offsets to 0 (acts as standard conv initially)
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)
        
        self.weight = nn.Parameter(torch.Tensor(c2, c1 // g, k, k))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.Tensor(c2))
        nn.init.constant_(self.bias, 0)
        
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        offsets = self.offset_conv(x)
        x = deform_conv2d(x, offsets, self.weight, self.bias, 
                          stride=self.stride, padding=self.padding, 
                          dilation=1, mask=None)
        return self.act(self.bn(x))

class MorphologicalBottleneck(nn.Module):
    """Standard Bottleneck but with MorphologicalConv instead of standard Conv."""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = MorphologicalConv(c_, c2, k[1], 1, p=k[1]//2, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f_Morph(nn.Module):
    """CSP Bottleneck with 2 convolutions, using MorphologicalBottleneck."""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(MorphologicalBottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    
    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
