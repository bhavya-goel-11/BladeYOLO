"""Nested scene-level subsets of a training split, for learning curves (how much does more labelled data help?).

Training images are grouped into scenes (tools/matching.py: images of one surface stay together), and whole scenes
are drawn: an image-level subset would keep near-copies of every scene and overstate how much data it holds. One
random order of scenes is chosen out of --trials so that every subset is as close as possible to the full set's class
mix (box shares) at its fraction; subsets are prefixes of that order, so each fraction contains the smaller ones.
Writes <out>/<prefix>_f<NN>.txt (image file names) for train.py --train-subset, and a summary.

Usage:
    python tools/scene_fractions.py --data external/WTBD-group --fractions 0.25 0.5 --prefix wtbd_group_train
"""

import argparse
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import matching as mt  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="YOLO-layout dataset with train/images and train/labels")
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5])
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "tools", "splits"))
    ap.add_argument("--cache", default=os.path.join(ROOT, ".dataset_cache"))
    ap.add_argument("--trials", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    names = yaml.safe_load(open(os.path.join(args.data, "data.yaml")))["names"]
    img_dir = os.path.join(args.data, "train", "images")
    files = sorted(os.listdir(img_dir))
    paths = [os.path.join(img_dir, f) for f in files]
    boxes = []
    for f in files:
        lab = os.path.join(args.data, "train", "labels", os.path.splitext(f)[0] + ".txt")
        cls = [int(l.split()[0]) for l in open(lab) if l.strip()] if os.path.exists(lab) else []
        boxes.append(np.bincount(cls, minlength=len(names)))
    boxes = np.array(boxes)

    E = mt.dinov2_embeddings(paths, os.path.join(args.cache, "dinov2_vits14.npz"))
    nn_idx, nn_sim = mt.knn(E, mt.KNN)
    pairs = sorted({(min(a, int(b)), max(a, int(b))) for a in range(len(paths)) for b in nn_idx[a] if a != int(b)})
    inl = mt.verify_pairs(paths, pairs, args.workers)
    dsu = mt.DSU(len(paths))
    for a in range(len(paths)):
        for b, s in zip(nn_idx[a], nn_sim[a]):
            b = int(b)
            if a != b and (s >= mt.DINO_GROUP or inl[(min(a, b), max(a, b))][0] >= mt.MIN_INLIERS):
                dsu.union(a, b)
    groups = dsu.groups()
    gbox = np.array([boxes[g].sum(0) for g in groups])
    gimg = np.array([len(g) for g in groups])
    share = boxes.sum(0) / boxes.sum()

    rng = np.random.default_rng(args.seed)
    best, best_score = None, np.inf
    for _ in range(args.trials):
        order = rng.permutation(len(groups))
        cum_img, cum_box = np.cumsum(gimg[order]), np.cumsum(gbox[order], 0)
        score, cuts = 0.0, []
        for fr in args.fractions:
            k = int(np.searchsorted(cum_img, fr * len(paths))) + 1
            b = cum_box[k - 1]
            if (b == 0).any():  # every class must be present
                score = np.inf
                break
            score += np.abs(b / b.sum() - share).sum()
            cuts.append(k)
        if score < best_score:
            best, best_score, best_cuts = order, score, cuts
    os.makedirs(args.out, exist_ok=True)
    lines = [f"{len(paths)} training images in {len(groups)} scenes; class box shares: "
             + ", ".join(f"{n} {s:.1%}" for n, s in zip(names, share))]
    for fr, k in zip(args.fractions, best_cuts):
        sel = sorted(files[i] for g in best[:k] for i in groups[g])
        b = boxes[[files.index(f) for f in sel]].sum(0)
        path = os.path.join(args.out, f"{args.prefix}_f{round(fr * 100):02d}.txt")
        open(path, "w").write("\n".join(sel) + "\n")
        lines.append(f"{os.path.basename(path)}: {k} scenes, {len(sel)} images, boxes " +
                     ", ".join(f"{n} {int(c)}" for n, c in zip(names, b)))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
