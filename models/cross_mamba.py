import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, List
from functools import partial

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics_modules.transformer import LayerNorm2d

try:
    from vmamba.models.vssm import CrossSS2D, LSBlock
    from timm.models.layers import DropPath
    
    def sync_devices(x, y):
        return x, y.to(x.device)
except ImportError:
    import warnings
    warnings.warn("vmamba or timm not found! CrossSS2D, LSBlock, and DropPath will use dummy implementations. Install vmamba for actual Cross-Mamba functionality.")
    
    class DropPath(nn.Module):
        def __init__(self, drop_prob=None):
            super(DropPath, self).__init__()
            self.drop_prob = drop_prob
        def forward(self, x):
            return x

    class LSBlock(nn.Module):
        def __init__(self, in_features, hidden_features, act_layer, drop=0.0):
            super().__init__()
            self.identity = nn.Identity()
        def forward(self, x):
            return self.identity(x)
            
    class CrossSS2D(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
        def forward(self, x, guide_x):
            return x

    def sync_devices(x, y):
        return x, y.to(x.device)

class CrossVSSBlock(nn.Module):
    def __init__(
            self,
            in_channels: int = 0,
            guide_channels: int = 0,
            hidden_dim: int = 0,
            drop_path: float = 0.1,
            norm_layer: Callable[..., nn.Module] = partial(LayerNorm2d, eps=1e-6),
            ssm_d_state: int = 16,
            ssm_ratio: float = 2.0,
            ssm_rank_ratio: float = 2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer: Callable = nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias: bool = True,
            ssm_drop_rate: float = 0.0,
            forward_type: str = "v2",
            guide_upsample_scale: float = 2.0,
            **kwargs,
    ):
        super().__init__()

        self.ssm_branch = ssm_ratio > 0
        self.hidden_dim = hidden_dim
        self.guide_upsample_scale = guide_upsample_scale

        self.guide_adapt = nn.Sequential(
            nn.Conv2d(guide_channels, hidden_dim, kernel_size=1, bias=False),
            nn.Upsample(
                scale_factor=guide_upsample_scale,
                mode="bilinear",
                align_corners=False
            ),
            norm_layer(hidden_dim),
            ssm_act_layer()
        )

        self.proj_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=True),
            nn.BatchNorm2d(hidden_dim),
            ssm_act_layer()
        )

        self.lsblock = LSBlock(
            in_features=hidden_dim,
            hidden_features=hidden_dim,
            act_layer=ssm_act_layer,
            drop=ssm_drop_rate
        )

        if self.ssm_branch:
            self.norm_x = norm_layer(hidden_dim)
            self.norm_guide = norm_layer(hidden_dim)

            self.cross_ssm = CrossSS2D(
                guide_dim=hidden_dim,
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_rank_ratio=ssm_rank_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,
                forward_type=forward_type,
                **kwargs
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, guide_x: torch.Tensor):
        x, guide_x = sync_devices(x, guide_x)

        x_proj = self.proj_conv(x)
        guide_adapted = self.guide_adapt(guide_x)

        if guide_adapted.shape[2:] != x_proj.shape[2:]:
            guide_adapted = F.interpolate(
                guide_adapted,
                size=x_proj.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        x_local = self.lsblock(x_proj)

        if self.ssm_branch:
            x_norm = self.norm_x(x_local)
            guide_norm = self.norm_guide(guide_adapted)

            x_norm, guide_norm = sync_devices(x_norm, guide_norm)

            x_ssm = self.cross_ssm(
                x=x_norm,
                guide_x=guide_norm
            )

            x_proj = x_proj + self.drop_path(x_ssm)

        return x_proj



class CrossScaleStateBlock(nn.Module):

    """优化版跨尺度状态增强模块 - P5→P4→P3语义指导"""
    def __init__(self,
                 in_channels: int = 384,
                 cross_ssm_d_state: int = 16,
                 cross_ssm_ratio: float = 2.0,
                 cross_ssm_rank_ratio: float = 2.0,
                 cross_ssm_dt_rank: Any = "auto",
                 cross_ssm_act_layer: Callable = nn.SiLU,
                 cross_ssm_conv: int = 3,
                 cross_ssm_conv_bias: bool = True,
                 cross_ssm_drop_rate: float = 0.0,
                 cross_mlp_ratio: float = 4.0,
                 cross_mlp_act_layer: Callable = nn.GELU,
                 cross_mlp_drop_rate: float = 0.0,
                 cross_drop_path: float = 0.1):
        super().__init__()
        self.in_channels = in_channels

        self.cross_vss_blocks = nn.ModuleList([
            CrossVSSBlock(
                in_channels=in_channels,
                guide_channels=in_channels,
                hidden_dim=in_channels,
                drop_path=cross_drop_path,
                ssm_d_state=cross_ssm_d_state,
                ssm_ratio=cross_ssm_ratio,
                ssm_rank_ratio=cross_ssm_rank_ratio,
                ssm_dt_rank=cross_ssm_dt_rank,
                ssm_act_layer=cross_ssm_act_layer,
                ssm_conv=cross_ssm_conv,
                ssm_conv_bias=cross_ssm_conv_bias,
                ssm_drop_rate=cross_ssm_drop_rate,
                forward_type="v2",
                mlp_ratio=cross_mlp_ratio,
                mlp_act_layer=cross_mlp_act_layer,
                mlp_drop_rate=cross_mlp_drop_rate,
                guide_upsample_scale=2.0,
            ),
            CrossVSSBlock(
                in_channels=in_channels,
                guide_channels=in_channels,
                hidden_dim=in_channels,
                drop_path=cross_drop_path,
                ssm_d_state=cross_ssm_d_state,
                ssm_ratio=cross_ssm_ratio,
                ssm_rank_ratio=cross_ssm_rank_ratio,
                ssm_dt_rank=cross_ssm_dt_rank,
                ssm_act_layer=cross_ssm_act_layer,
                ssm_conv=cross_ssm_conv,
                ssm_conv_bias=cross_ssm_conv_bias,
                ssm_drop_rate=cross_ssm_drop_rate,
                forward_type="v2",
                mlp_ratio=cross_mlp_ratio,
                mlp_act_layer=cross_mlp_act_layer,
                mlp_drop_rate=cross_mlp_drop_rate,
                guide_upsample_scale=2.0,
            )
        ])

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        if len(feats) != 3:
            raise ValueError(f"要求输入3个尺度特征，实际收到{len(feats)}个")
        P3, P4, P5 = feats

        for i, feat in enumerate([P3, P4, P5]):
            if feat.shape[1] != self.in_channels:
                raise RuntimeError(f"第{i+1}个特征通道数{feat.shape[1]}与配置{self.in_channels}不匹配")

        P4_enhanced = self.cross_vss_blocks[0](x=P4, guide_x=P5)
        P3_enhanced = self.cross_vss_blocks[1](x=P3, guide_x=P4_enhanced)

        return [P3_enhanced, P4_enhanced, P5]
