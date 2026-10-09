"""Hybrid backbone: frozen DINOv3 ViT-S/16 semantics fused with a trainable wavelet detail branch."""

import os
import sys
import zipfile

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv

from .wavelet import WaveletDown

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DINOV3_REPO_ZIP = "https://github.com/facebookresearch/dinov3/archive/refs/heads/main.zip"
DINOV3_WEIGHT_PATHS = [
    os.environ.get("DINOV3_WEIGHTS", ""),  # explicit override
    os.path.join(ROOT_DIR, "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"),  # local: project folder
    "/kaggle/input/models/shamskarib/dinov3-vits/pytorch/default/1/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
]


def ensure_dinov3_importable():
    """Put the DINOv3 source on sys.path, downloading it into the torch hub cache if needed.

    The repo's hubconf pulls in segmentation/eval dependencies we do not need, so the backbone
    is imported from dinov3.hub.backbones directly. The path must stay on sys.path because
    checkpoints pickle the DINOv3 classes by reference (and DDP workers inherit sys.path).
    """
    try:
        import dinov3  # noqa: F401  (pip-installed or already on the path)

        return
    except ImportError:
        pass
    repo = os.path.join(torch.hub.get_dir(), "dinov3-main")
    if not os.path.isdir(repo):
        os.makedirs(torch.hub.get_dir(), exist_ok=True)
        zip_path = repo + ".zip"
        torch.hub.download_url_to_file(DINOV3_REPO_ZIP, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(torch.hub.get_dir())
        os.remove(zip_path)
    sys.path.insert(0, repo)


def build_dinov3_vits16():
    ensure_dinov3_importable()
    from dinov3.hub.backbones import dinov3_vits16

    model = dinov3_vits16(pretrained=False)
    path = next((p for p in DINOV3_WEIGHT_PATHS if p and os.path.exists(p)), None)
    if not path:
        raise FileNotFoundError(f"DINOv3 ViT-S/16 weights not found in {DINOV3_WEIGHT_PATHS[1:]}. Set DINOV3_WEIGHTS=/path/to/file.pth.")
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    state = {k.removeprefix("backbone."): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    params = dict(model.named_parameters())
    if [k for k in missing if k in params] or unexpected:
        raise RuntimeError(f"DINOv3 weights do not match the model. Missing: {missing}. Unexpected: {unexpected}.")
    print(f"[BladeYOLO] Loaded DINOv3 ViT-S/16 weights from {path}")
    return model.float()


class FeatureFusion(nn.Module):
    """Concatenate (resized) DINOv3 semantics with detail features and mix them."""

    def __init__(self, c_sem, c_det, c_out):
        super().__init__()
        self.proj = Conv(c_sem + c_det, c_out, 1)
        self.mix = Conv(c_out, c_out, 3, g=c_out)

    def forward(self, sem, det):
        if sem.shape[2:] != det.shape[2:]:
            sem = F.interpolate(sem, size=det.shape[2:], mode="bilinear", align_corners=False)
        return self.mix(self.proj(torch.cat([sem, det], 1)))


class BladeBackbone(nn.Module):
    """Returns [P3, P4, P5] (strides 8/16/32) with channels (256, 512, 512).

    1. Semantic branch: frozen DINOv3 ViT-S/16, blocks 4/8/12 (stride 16), resized to each level.
    2. Detail branch: CNN stem to stride 4, then learnable-wavelet downsampling for strides 8/16/32,
       so high-frequency sub-bands are kept as channels instead of being pooled away.
    3. Per-level fusion of the two.
    """

    channels = (256, 512, 512)
    dino_blocks = (3, 7, 11)
    dino_dim = 384

    def __init__(self, *args):
        super().__init__()
        self.dino = build_dinov3_vits16()
        self.dino.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

        c3, c4, c5 = self.channels
        self.stem = nn.Sequential(Conv(3, 64, 3, 2), Conv(64, 128, 3, 2))  # stride 4
        self.down3 = nn.Sequential(WaveletDown(128, c3), Conv(c3, c3, 3))  # stride 8
        self.down4 = WaveletDown(c3, c4)  # stride 16
        self.down5 = WaveletDown(c4, c5)  # stride 32
        self.fuse = nn.ModuleList(FeatureFusion(self.dino_dim, c, c) for c in self.channels)

    def train(self, mode=True):
        super().train(mode)
        self.dino.eval()  # frozen: no dropout / drop-path
        return self

    def forward(self, x):
        with torch.no_grad():
            sem = self.dino.get_intermediate_layers(
                (x - self.mean) / self.std, n=self.dino_blocks, reshape=True, norm=True
            )
        d3 = self.down3(self.stem(x))
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        return [f(s, d) for f, s, d in zip(self.fuse, sem, (d3, d4, d5))]
