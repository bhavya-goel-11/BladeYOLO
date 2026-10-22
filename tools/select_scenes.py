"""Label-free choice of which scenes to label (Phase 28, H6): patch-coverage selection on frozen DINOv3 features.

The 25% subsets so far (tools/scene_fractions.py) are random scene orders chosen to match the class mix, i.e. they
use the labels. Here no label is read. Defects cover a few patches of a blade photo, so an image-level embedding
(dominated by blade and sky) cannot see them; the selection works on patches instead:
  1. DINOv3 last-block patch features (640 px, L2-normalised) of every training image; 128 random patches per image.
  2. k-means (1,024 centroids) on those patches: the modes of the training distribution, common and rare alike.
  3. Greedy selection of whole scenes (same grouping as scene_fractions.py) until the image budget is reached: each
     step adds the scene with the largest gain per image in soft coverage, sum over centroids of the best cosine of
     any selected patch to the centroid. All centroids weigh the same, so a rare pattern counts as much as a common one.
Also writes plain random scene draws of the same budget (--random N, no class matching) as controls for luck.

Usage:
    python tools/select_scenes.py --data external/WTBD-group --budget 189 --prefix wtbd_group_train_f25 --random 2
Writes tools/splits/<prefix>_cover.txt, <prefix>_rand<k>.txt and runs/analysis/select_scenes/<prefix>.txt (summary).
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import matching as mt  # noqa: E402

PER_IMAGE = 128
K = 1024


def patch_features(paths, size, cache, seed=0):
    """(n_images, PER_IMAGE, D) random patches of each image's last-block DINOv3 features, cached."""
    if os.path.exists(cache):
        return np.load(cache)
    import cv2
    import torch

    from models.dino_weights import SIZES, build_dinov3

    torch.set_num_threads(os.cpu_count())
    vit = build_dinov3(size).eval()
    blocks = SIZES[size][3]
    mean, std = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    rng = np.random.default_rng(seed)
    out = []
    for i, p in enumerate(paths):
        x = torch.from_numpy(cv2.resize(cv2.imread(p), (640, 640))[:, :, ::-1].copy()).permute(2, 0, 1)[None].float() / 255
        with torch.no_grad():
            y = vit.get_intermediate_layers((x - mean) / std, n=blocks, reshape=False, norm=True)[-1][0]
        y = torch.nn.functional.normalize(y, dim=-1).numpy()
        out.append(y[rng.choice(len(y), PER_IMAGE, replace=False)].astype(np.float16))
        if i % 50 == 0:
            print(f"  features {i}/{len(paths)}", flush=True)
    out = np.stack(out)
    np.save(cache, out)
    return out


def scenes(paths, cache, workers):
    E = mt.dinov2_embeddings(paths, os.path.join(cache, "dinov2_vits14.npz"))
    nn_idx, nn_sim = mt.knn(E, mt.KNN)
    pairs = sorted({(min(a, int(b)), max(a, int(b))) for a in range(len(paths)) for b in nn_idx[a] if a != int(b)})
    inl = mt.verify_pairs(paths, pairs, workers)
    dsu = mt.DSU(len(paths))
    for a in range(len(paths)):
        for b, s in zip(nn_idx[a], nn_sim[a]):
            b = int(b)
            if a != b and (s >= mt.DINO_GROUP or inl[(min(a, b), max(a, b))][0] >= mt.MIN_INLIERS):
                dsu.union(a, b)
    return dsu.groups()


def greedy_cover(F, groups, budget):
    """Scenes chosen greedily by soft-coverage gain per image. F: (n_images, P, D); returns scene indices in order."""
    from sklearn.cluster import KMeans

    X = F.reshape(-1, F.shape[-1]).astype(np.float32)
    C = KMeans(K, n_init=1, random_state=0).fit(X).cluster_centers_
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    S = np.stack([(F[g].reshape(-1, F.shape[-1]).astype(np.float32) @ C.T).max(0) for g in groups])  # (scenes, K)
    best = np.full(K, -1.0)
    chosen, used = [], 0
    sizes = np.array([len(g) for g in groups])
    while True:
        gain = (np.maximum(S, best) - best).sum(1) / sizes
        gain[chosen] = -np.inf
        gain[used + sizes > budget * 1.05] = -np.inf  # may overshoot the budget by 5% at most
        j = int(np.argmax(gain))
        if not np.isfinite(gain[j]):
            break
        chosen.append(j)
        used += sizes[j]
        best = np.maximum(best, S[j])
        if used >= budget:
            break
    return chosen, float(best.mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--budget", type=int, required=True, help="labelled images (189 = the 25%% subset)")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--random", type=int, default=2, help="plain random scene draws of the same budget")
    ap.add_argument("--dino", default="b")
    ap.add_argument("--cache", default=os.path.join(ROOT, ".dataset_cache"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    import yaml

    names = yaml.safe_load(open(os.path.join(args.data, "data.yaml")))["names"]
    img_dir = os.path.join(args.data, "train", "images")
    files = sorted(os.listdir(img_dir))
    paths = [os.path.join(img_dir, f) for f in files]
    groups = scenes(paths, args.cache, args.workers)
    out = os.path.join(ROOT, "runs", "analysis", "select_scenes")
    os.makedirs(out, exist_ok=True)
    F = patch_features(paths, args.dino, os.path.join(out, f"{args.prefix}_patches_{args.dino}.npy"))
    chosen, cov = greedy_cover(F, groups, args.budget)

    def boxes(sel):
        cls = []
        for f in sel:
            lab = os.path.join(args.data, "train", "labels", os.path.splitext(f)[0] + ".txt")
            cls += [int(l.split()[0]) for l in open(lab) if l.strip()] if os.path.exists(lab) else []
        return np.bincount(cls, minlength=len(names))

    lines = [f"{len(files)} training images in {len(groups)} scenes; budget {args.budget} images"]
    subsets = {"cover": sorted(files[i] for g in chosen for i in groups[g])}
    rng = np.random.default_rng(0)
    for k in range(1, args.random + 1):
        sel, used = [], 0
        for g in rng.permutation(len(groups)):
            if used + len(groups[g]) <= args.budget * 1.05:
                sel.append(g)
                used += len(groups[g])
            if used >= args.budget:
                break
        subsets[f"rand{k}"] = sorted(files[i] for g in sel for i in groups[g])
    base = os.path.join(ROOT, "tools", "splits", f"{args.prefix}.txt")
    if os.path.exists(base):
        subsets["class-matched random (existing)"] = sorted(n.strip() for n in open(base) if n.strip())
    for name, sel in subsets.items():
        if not name.startswith("class-matched"):
            open(os.path.join(ROOT, "tools", "splits", f"{args.prefix}_{name}.txt"), "w").write("\n".join(sel) + "\n")
        n_sc = len({gi for gi, g in enumerate(groups) for i in g if files[i] in set(sel)})
        lines.append(f"{name}: {n_sc} scenes, {len(sel)} images, boxes " + ", ".join(f"{n} {c}" for n, c in zip(names, boxes(sel))))
    lines.append(f"cover: mean best cosine to the {K} centroids {cov:.3f}")
    open(os.path.join(out, f"{args.prefix}.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
