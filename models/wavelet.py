import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv


class WaveletDown(nn.Module):
    """Stride-2 downsampling through a learnable 2x2 wavelet analysis.

    Each channel is split into four Haar-initialised sub-bands (LL, HL, LH, HH) whose filters are
    then learned. All sub-bands are kept and mixed by a 1x1 conv, so the high-frequency detail that
    pooling or strided convs discard (thin cracks, pitting edges) stays available downstream.
    """

    def __init__(self, c1, c2):
        super().__init__()
        haar = torch.tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],  # LL
                [[-1.0, -1.0], [1.0, 1.0]],  # HL
                [[-1.0, 1.0], [-1.0, 1.0]],  # LH
                [[1.0, -1.0], [-1.0, 1.0]],  # HH
            ]
        ) / 2.0
        self.dwt = nn.Conv2d(c1, 4 * c1, 2, 2, groups=c1, bias=False)
        self.dwt.weight.data.copy_(haar.repeat(c1, 1, 1).unsqueeze(1))
        self.mix = Conv(4 * c1, c2, 1)
        self.local = Conv(c2, c2, 3, g=c2)

    def forward(self, x):
        return self.local(self.mix(self.dwt(x)))
