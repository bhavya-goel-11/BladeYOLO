"""FreeSel (Xie et al., NeurIPS 2023, arXiv 2309.17342) on WTBD-group: the published label-free selection baseline for
H6 (tools/select_scenes.py), plus a diagnostic of its attention filter on small defects.

Reimplemented from the paper (object-detection settings, its §4 and appendix C.1):
  1. DINO ViT-S/16 (DeiT-S, facebookresearch/dino) on the image resized to SIZE x SIZE (paper: 224).
  2. Attention filter: last-layer [CLS] attention (mean over heads), patches sorted by it; keep the first t whose
     attention sums to <= tau (0.5) of the total.
  3. Spectral clustering of the kept patches into K = 5 "semantic patterns": affinity = last-layer patch-to-patch
     attention (mean over heads, symmetrised) between patches within Chebyshev distance 2, zero diagonal; normalised
     Laplacian, K smallest eigenvectors, rows normalised, k-means. A pattern = mean feature of its patches.
  4. Selection: a random first image; then repeatedly a pattern is drawn with probability proportional to its squared
     cosine distance to the nearest selected pattern, and its whole image is selected; until the image budget.
FreeSel selects images, not scenes (as published).

Diagnostic (--diagnose): on the training images, the share of all patches and of labelled defect patches (patch
centre inside a box) that the tau filter keeps. If defects fall outside the [CLS] attention, FreeSel never sees them.

Usage:
    python tools/freesel.py --data external/WTBD-group --budget 189 --prefix wtbd_group_train_f25 --size 224 --seed 0
    python tools/freesel.py --data external/WTBD-group --diagnose --size 224 640
"""

import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import bootstrap_ci as bc  # noqa: E402

TAU, K, D0 = 0.5, 5, 2


def load_dino():
    torch.set_num_threads(os.cpu_count())
    return torch.hub.load("facebookresearch/dino:main", "dino_vits16", verbose=False).eval()


@torch.no_grad()
def encode(model, path, size):
    """Last-layer patch features (n, 384), [CLS] attention (n,), patch attention (n, n); grid side size // 16."""
    import cv2

    x = torch.from_numpy(cv2.resize(cv2.imread(path), (size, size))[:, :, ::-1].copy()).permute(2, 0, 1)[None].float() / 255
    x = (x - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    attn = model.get_last_selfattention(x)[0].mean(0)  # (n+1, n+1), mean over heads
    feats = model.get_intermediate_layers(x, n=1)[0][0, 1:]
    ca = attn[0, 1:]
    return torch.nn.functional.normalize(feats, dim=-1).numpy(), (ca / ca.sum()).numpy(), attn[1:, 1:].numpy()


def kept(ca, tau=TAU):
    """Indices of the attention filter (Eq. 2): highest-attention patches whose attention sums to <= tau."""
    order = np.argsort(-ca)
    t = max(int(np.searchsorted(np.cumsum(ca[order]), tau, side="right")), K)
    return order[:t]


def patterns(feats, ca, pa, side, rng):
    """K semantic patterns (K, d) of one image (Alg. 1 and appendix A)."""
    from sklearn.cluster import KMeans

    idx = kept(ca)
    yx = np.stack([idx // side, idx % side], 1)
    near = np.abs(yx[:, None] - yx[None]).max(-1) <= D0
    A = pa[np.ix_(idx, idx)]
    A = (A + A.T) / 2 * near
    np.fill_diagonal(A, 0)
    d = A.sum(1)
    dinv = 1 / np.sqrt(np.maximum(d, 1e-12))
    L = np.eye(len(idx)) - dinv[:, None] * A * dinv[None]
    _, vec = np.linalg.eigh(L)
    V = vec[:, :K]
    V /= np.linalg.norm(V, axis=1, keepdims=True).clip(1e-12)
    lab = KMeans(K, n_init=10, random_state=int(rng.integers(1 << 30))).fit_predict(V)
    P = np.stack([feats[idx][lab == j].mean(0) if (lab == j).any() else feats[idx].mean(0) for j in range(K)])
    return P / np.linalg.norm(P, axis=1, keepdims=True)


def select(P, budget, rng):
    """Alg. 2: D^2 sampling over patterns; P (n_images, K, d) -> selected image indices."""
    n = len(P)
    flat = P.reshape(-1, P.shape[-1])
    owner = np.repeat(np.arange(n), K)
    first = int(rng.integers(n))
    chosen = [first]
    dmin = 1 - (flat @ P[first].T).max(1)
    while len(chosen) < budget:
        p = np.clip(dmin, 0, None) ** 2
        p[np.isin(owner, chosen)] = 0
        j = owner[rng.choice(len(flat), p=p / p.sum())]
        chosen.append(int(j))
        dmin = np.minimum(dmin, 1 - (flat @ P[j].T).max(1))
    return chosen


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--budget", type=int, default=189)
    ap.add_argument("--prefix", default="wtbd_group_train_f25")
    ap.add_argument("--size", type=int, nargs="+", default=[224])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--diagnose", action="store_true")
    args = ap.parse_args()
    data = os.path.abspath(args.data)
    model = load_dino()
    out = os.path.join(ROOT, "runs", "analysis", "select_scenes")
    os.makedirs(out, exist_ok=True)
    if args.diagnose:
        from models.retrieval import patch_classes  # noqa: E402  (repo root on the path below)

        tr = bc.load_yolo_split(data, "train")
        lines = []
        for size in args.size:
            side = size // 16
            n_all = n_keep = d_all = d_keep = 0
            per_img = []
            for p, v in tr.items():
                _, ca, _ = encode(model, p, size)
                k = np.zeros(side * side, bool)
                k[kept(ca)] = True
                lab = patch_classes(v["gt"] * np.array([1, size / 640, size / 640, size / 640, size / 640]), side, side)
                n_all += k.size
                n_keep += k.sum()
                d_all += (lab >= 0).sum()
                d_keep += (k & (lab >= 0)).sum()
                if (lab >= 0).any():
                    per_img.append((k & (lab >= 0)).sum() / (lab >= 0).sum())
            lines.append(f"size {size}: filter keeps {n_keep / n_all:.1%} of patches but {d_keep / max(d_all, 1):.1%} of defect "
                         f"patches ({d_all} defect patches); images with no defect patch kept: "
                         f"{np.mean(np.array(per_img) == 0):.1%} of {len(per_img)}")
            print(lines[-1], flush=True)
        open(os.path.join(out, "freesel_attention_filter.txt"), "w").write("\n".join(lines) + "\n")
        return
    files = sorted(os.listdir(os.path.join(data, "train", "images")))
    paths = [os.path.join(data, "train", "images", f) for f in files]
    rng = np.random.default_rng(args.seed)
    for size in args.size:
        cache = os.path.join(out, f"freesel_patterns_{size}_s{args.seed}.npy")
        if os.path.exists(cache):
            P = np.load(cache)
        else:
            P = []
            for i, p in enumerate(paths):
                feats, ca, pa = encode(model, p, size)
                P.append(patterns(feats, ca, pa, size // 16, rng))
                if i % 100 == 0:
                    print(f"  patterns {size}px {i}/{len(paths)}", flush=True)
            P = np.stack(P)
            np.save(cache, P)
        sel = sorted(files[i] for i in select(P, args.budget, rng))
        name = f"{args.prefix}_freesel{size}_s{args.seed}.txt"
        open(os.path.join(ROOT, "tools", "splits", name), "w").write("\n".join(sel) + "\n")
        print(f"{name}: {len(sel)} images")


if __name__ == "__main__":
    sys.path.insert(0, ROOT)
    main()
