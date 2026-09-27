import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# DINOv3 Patching (Clean, No Style Injection)
from dinov3.models.vision_transformer import DinoVisionTransformer

def patched_get_intermediate_layers(self_model, x, n=1, reshape=False, return_class_token=False, norm=False):
    if hasattr(self_model, 'prepare_tokens_with_masks'):
        out = self_model.prepare_tokens_with_masks(x)
        x = out[0]
    else:
        x = self_model.patch_embed(x)
        x = torch.cat((self_model.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self_model.interpolate_pos_encoding(x, x.shape[1], x.shape[2])

    outputs = []
    for i, blk in enumerate(self_model.blocks):
        x = blk(x)
        if i in n:
            outputs.append(x)
            
    if norm and hasattr(self_model, 'norm'):
        outputs = [self_model.norm(out) for out in outputs]
        
    if not return_class_token:
        num_extra = getattr(self_model, 'n_storage_tokens', 0) + 1
        outputs = [out[:, num_extra:] for out in outputs]
        
    if reshape:
        B, _, C = outputs[0].shape
        H = W = int(math.sqrt(outputs[0].shape[1]))
        outputs = [out.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous() for out in outputs]
        
    return outputs

DinoVisionTransformer.patched_get_intermediate_layers = patched_get_intermediate_layers

from .lfa import LFA

class FeatureFusion(nn.Module):
    def __init__(self, semantic_channels, detail_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(semantic_channels + detail_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, groups=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
    def forward(self, semantic, detail):
        if semantic.shape[2:] != detail.shape[2:]:
            semantic = F.interpolate(semantic, size=detail.shape[2:], mode='bilinear', align_corners=False)
        fused = torch.cat([semantic, detail], dim=1)
        return self.conv(fused)

class PhysicsAwareBackbone(nn.Module):
    """
    Clean, Physics-Aware Backbone replacing the buggy BladeYOLO backbone.
    1. Extracts pure semantics from DINOv3 (Blocks 4, 8, 12).
    2. Extracts high-res physical details via CNN Stem + LFA.
    3. Fuses them robustly into P3, P4, P5 for YOLOv12 Neck.
    """
    def __init__(self, in_channels=3, p3_channels=256, p4_channels=512, p5_channels=512, freeze_dino=True):
        super().__init__()
        
        # 1. DINOv3 Branch
        dino_model = torch.hub.load('facebookresearch/dinov3', 'dinov3_vits14', trust_repo=True)
        self.dino = dino_model
        if freeze_dino:
            for param in self.dino.parameters():
                param.requires_grad = False
                
        dino_dim = 384 # ViT-S dimension
        
        # 2. Physics / LFA Branch
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU()
        )
        
        # P3 Detail (Stride 8)
        self.p3_conv = nn.Sequential(
            nn.Conv2d(128, p3_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(p3_channels),
            nn.GELU()
        )
        self.p3_lfa = LFA(p3_channels)
        
        # P4 Detail (Stride 16)
        self.p4_conv = nn.Sequential(
            nn.Conv2d(p3_channels, p4_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(p4_channels),
            nn.GELU()
        )
        self.p4_lfa = LFA(p4_channels)
        
        # P5 Detail (Stride 32)
        self.p5_conv = nn.Sequential(
            nn.Conv2d(p4_channels, p4_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(p4_channels),
            nn.GELU()
        )
        
        # 3. Fusion Blocks
        self.fuse3 = FeatureFusion(dino_dim, p3_channels, p3_channels)
        self.fuse4 = FeatureFusion(dino_dim, p4_channels, p4_channels)
        self.fuse5 = FeatureFusion(dino_dim, p4_channels, p5_channels)

    def forward(self, x):
        # 1. DINOv3 Semantics (Frozen)
        with torch.no_grad():
            dino_feats = self.dino.patched_get_intermediate_layers(x, n=[3, 7, 11], reshape=True)
            
        # 2. Physics Details
        stem_out = self.stem(x)
        
        p3_det = self.p3_conv(stem_out)
        p3_det = self.p3_lfa(p3_det)
        
        p4_det = self.p4_conv(p3_det)
        p4_det = self.p4_lfa(p4_det)
        
        p5_det = self.p5_conv(p4_det)
        
        # 3. Fusion
        p3_out = self.fuse3(dino_feats[0], p3_det)
        p4_out = self.fuse4(dino_feats[1], p4_det)
        p5_out = self.fuse5(dino_feats[2], p5_det)
        
        return [p3_out, p4_out, p5_out]
