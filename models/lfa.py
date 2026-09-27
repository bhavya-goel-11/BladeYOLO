import torch
import torch.nn as nn
import math

class HaarDWT(nn.Module):
    """
    Physics-aware Discrete Wavelet Transform (Haar) using fixed convolutions.
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # Haar Wavelet basis functions (2x2)
        ll = torch.tensor([[1.0, 1.0], [1.0, 1.0]]) / 2.0
        hl = torch.tensor([[-1.0, -1.0], [1.0, 1.0]]) / 2.0
        lh = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]]) / 2.0
        hh = torch.tensor([[1.0, -1.0], [-1.0, 1.0]]) / 2.0
        
        weight = torch.zeros(channels * 4, 1, 2, 2)
        
        for i in range(channels):
            weight[i * 4 + 0, 0] = ll
            weight[i * 4 + 1, 0] = hl
            weight[i * 4 + 2, 0] = lh
            weight[i * 4 + 3, 0] = hh
            
        self.register_buffer('weight', weight)
        
    def forward(self, x):
        out = nn.functional.conv2d(x, self.weight, stride=2, groups=self.channels)
        return out

class LFA(nn.Module):
    """
    Localized Frequency Attention.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.dwt = HaarDWT(in_channels)
        
        self.attention_generator = nn.Sequential(
            nn.Conv2d(in_channels * 4, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        freq_features = self.dwt(x)
        attn_mask = self.attention_generator(freq_features)
        x_down = self.pool(x)
        out = x_down * attn_mask
        return out
