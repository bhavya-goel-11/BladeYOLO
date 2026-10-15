"""Frozen DINOv3 ViT-S/16 for BladeYOLO-v2 (models/dino_fusion.py): source on the path, weights from disk."""

import os
import sys
import zipfile

import torch

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
