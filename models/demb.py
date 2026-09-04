import torch
import torch.nn as nn
import torch.nn.functional as F

from .util import wavelet
class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)

class WTConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1',padding='same'):
        super(WTConv2d, self).__init__()

        self.channel_proj = None
        if in_channels != out_channels:

             self.channel_proj = nn.Conv2d(in_channels, out_channels, 1, bias=False)
             temp_in_channels = in_channels
        else:
             temp_in_channels = in_channels

        self.in_channels = temp_in_channels
        self.out_channels = out_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter = nn.Parameter(torch.rand(4, 1, 2, 2).repeat(self.in_channels, 1, 1, 1), requires_grad=False)
        self.iwt_filter = nn.Parameter(torch.rand(4, 1, 2, 2).repeat(self.in_channels, 1, 1, 1), requires_grad=False)

        self.base_conv = nn.Conv2d(self.in_channels, self.in_channels, kernel_size, padding='same', stride=1, dilation=1, groups=self.in_channels, bias=bias)
        self.base_scale = _ScaleModule([1,self.in_channels,1,1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(self.in_channels*4, self.in_channels*4, kernel_size, padding='same', stride=1, dilation=1, groups=self.in_channels*4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1,self.in_channels*4,1,1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:

            self.do_stride = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)
        else:
            self.do_stride = None

    def forward(self, x):

        if self.channel_proj is not None:
             x_temp = x
        else:
             x_temp = x

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x_temp

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)

            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = torch.zeros(curr_x_ll.shape[0], curr_x_ll.shape[1], 4, curr_x_ll.shape[2]//2, curr_x_ll.shape[3]//2).to(curr_x_ll.device)

            curr_x_ll = curr_x[:,:,0,:,:]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:,:,0,:,:])
            x_h_in_levels.append(curr_x_tag[:,:,1:4,:,:])

        next_x_ll = 0

        for i in range(self.wt_levels-1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)

            next_x_ll = torch.zeros(curr_x.shape[0], curr_x.shape[1], curr_shape[2], curr_shape[3]).to(curr_x.device)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x_temp = self.base_scale(self.base_conv(x_temp))
        x_temp = x_temp + x_tag

        if self.do_stride is not None:
            x_temp = self.do_stride(x_temp)

        if self.channel_proj is not None:
             x = self.channel_proj(x_temp)
        else:
             x = x_temp

        return x

class MultiReceptiveFieldBranch(nn.Module):

    def __init__(self, in_channels=3, base_channels=16):
        super().__init__()
        self.base_channels = base_channels

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        self.fine_branch = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),

            WTConv2d(
                in_channels=base_channels,
                out_channels=base_channels,
                kernel_size=3, padding='same',
                wt_levels=1, stride=1, bias=False
            ),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
        )

        self.medium_branch = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.GELU(),

            nn.Conv2d(base_channels * 2, base_channels * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.GELU(),

            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.GELU(),
        )

        self.large_branch = nn.Sequential(

            WTConv2d(
                in_channels=base_channels,
                out_channels=base_channels * 2,
                kernel_size=5, padding='same',
                wt_levels=2, stride=2, bias=False
            ),
            nn.BatchNorm2d(base_channels * 2),
            nn.GELU(),

            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.GELU(),

            nn.Conv2d(base_channels * 4, base_channels * 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.GELU(),
        )

    def forward(self, x):
        stem_out = self.stem(x)
        fine_feat = self.fine_branch(stem_out)
        medium_feat = self.medium_branch(stem_out)
        large_feat = self.large_branch(stem_out)
        return [fine_feat, medium_feat, large_feat]

class concat(nn.Module):
    def __init__(self, sem_channels, det_channels, hidden_dim):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(sem_channels + det_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )

    def forward(self, semantic_feat, detail_feat):
        fused = torch.cat([semantic_feat, detail_feat], dim=1)
        return self.fusion(fused)
