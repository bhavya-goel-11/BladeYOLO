"""Retrieval prior from the labelled images (train.py --retrieval; Phase 28, H3).

Before training, the last-block DINOv3 patch features of every labelled training image (unaugmented, 640 px) are
stored with each patch's class (the smallest box containing the patch centre; -1 for background, which is
subsampled and re-weighted). In the model, each patch of the input looks up its k nearest stored patches by cosine
and gets per-class vote shares: an nc-channel map that DinoFuse takes next to the DINOv3 features.

While training, the query image itself (and its scene's duplicates) is in memory, and the model cannot tell which
image it is under mosaic. Stored patches more similar than `cap` are therefore ignored, at training and test alike.
cap = 0.8 makes the prior on augmented training images about as informative as on test images (AUROC 0.84 / 0.83,
mean defect score 0.41 / 0.41; runs/analysis/priors/retrieval_leak.txt), where without the cap it is near-perfect
in training (0.92 augmented, 0.996 clean) and the detector would learn to trust it more than it deserves at test.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F

K_SEARCH = 64  # neighbours fetched per patch; those above the cap are dropped, the first k of the rest vote


def retrieval_config():
    """(memory file, nc) from train.py --retrieval; ("", 0) when off."""
    return os.environ.get("BLADEYOLO_RETRIEVAL", ""), int(os.environ.get("BLADEYOLO_RETRIEVAL_NC", "0") or 0)


def patch_classes(gt, h, w, stride=16):
    """(h*w,) class of each patch centre's smallest containing box, -1 background. gt: (n, 5) cls, x1, y1, x2, y2 px."""
    cx, cy = np.meshgrid((np.arange(w) + 0.5) * stride, (np.arange(h) + 0.5) * stride)
    lab, area = -np.ones(h * w, int), np.full(h * w, np.inf)
    for k, x1, y1, x2, y2 in gt:
        inside = ((cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2)).ravel()
        upd = inside & ((x2 - x1) * (y2 - y1) < area)
        lab[upd], area[upd] = int(k), (x2 - x1) * (y2 - y1)
    return lab


def read_labels(img_path, W, H, s):
    """YOLO label file of an image (Ultralytics' images/ -> labels/ convention), boxes in the resized frame."""
    parts = os.path.splitext(img_path)[0].split(os.sep)
    i = len(parts) - 1 - parts[::-1].index("images")
    lab = os.sep.join(parts[:i] + ["labels"] + parts[i + 1:]) + ".txt"
    rows = [list(map(float, l.split()[:5])) for l in open(lab) if l.strip()] if os.path.exists(lab) else []
    return np.array([[c, (x - w / 2) * W * s, (y - h / 2) * H * s, (x + w / 2) * W * s, (y + h / 2) * H * s]
                     for c, x, y, w, h in rows]).reshape(-1, 5)


@torch.no_grad()
def build_memory(images, nc, size, out, imgsz=640, bg_max=60000, cap=0.8, k=10, seed=0, device="cpu"):
    """Encode the labelled images and save the memory (features fp16, classes, background weight, settings)."""
    import cv2

    from .dino_weights import SIZES, build_dinov3

    blocks = SIZES[size][3]
    vit = build_dinov3(size).eval().to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    feats, labs = [], []
    for p in images:
        im = cv2.imread(p)
        H, W = im.shape[:2]
        s = imgsz / max(H, W)
        h, w = max(1, round(H * s / 16)), max(1, round(W * s / 16))
        x = torch.from_numpy(cv2.resize(im, (w * 16, h * 16))[:, :, ::-1].copy()).to(device).permute(2, 0, 1)[None]
        y = vit.get_intermediate_layers((x.float() / 255 - mean) / std, n=blocks, reshape=True, norm=True)[-1]
        feats.append(F.normalize(y[0].flatten(1).T, dim=1).half().cpu())
        labs.append(torch.from_numpy(patch_classes(read_labels(p, W, H, s), h, w)))
    feats, labs = torch.cat(feats), torch.cat(labs)
    bg = torch.where(labs < 0)[0]
    n_bg = min(len(bg), bg_max)
    keep_bg = bg[torch.randperm(len(bg), generator=torch.Generator().manual_seed(seed))[:n_bg]]
    keep = torch.cat([torch.where(labs >= 0)[0], keep_bg])
    mem = dict(feat=feats[keep].contiguous(), cls=labs[keep].to(torch.int8), w_bg=len(bg) / max(n_bg, 1), nc=nc,
               cap=cap, k=k, size=size, images=len(images))
    torch.save(mem, out)
    print(f"[BladeYOLO] retrieval memory: {int((labs >= 0).sum())} defect + {n_bg} of {len(bg)} background patches "
          f"from {len(images)} images -> {out}")
    return mem


@torch.no_grad()
def retrieval_prior(sem, mem_feat, mem_cls, w_bg, nc, cap, k, chunk=4096):
    """sem (B, D, h, w) last-block DINOv3 features -> (B, nc, h, w) per-class vote shares of the k nearest stored
    patches with cosine <= cap (background votes weighted by w_bg for its subsampling)."""
    B, D, h, w = sem.shape
    q = F.normalize(sem.flatten(2).transpose(1, 2).reshape(-1, D).float(), dim=1)
    M = mem_feat.to(torch.float16 if q.is_cuda else torch.float32)
    out = []
    for c in q.split(chunk):
        s, j = (c.to(M.dtype) @ M.T).topk(min(K_SEARCH, len(M)), dim=1)
        ok = s <= cap
        use = ok & (ok.cumsum(1) <= k)
        cls = mem_cls[j].long()
        wt = torch.where(cls >= 0, 1.0, float(w_bg)) * use
        votes = torch.zeros(len(c), nc + 1, device=c.device).scatter_add_(1, cls + 1, wt.float())
        out.append(votes[:, 1:] / votes.sum(1, keepdim=True).clamp_min(1e-6))
    return torch.cat(out).view(B, h, w, nc).permute(0, 3, 1, 2).contiguous()
