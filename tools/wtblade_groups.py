"""Leak-free group split of WTBlade-Defect (fengChe), the test bed for multi-view test-time methods (Phase 28, H4).

WTBlade's 10,660 images are augmented copies (Roboflow names "<source>_jpg.rf.<hash>") of 2,794 source frames, many
of them consecutive drone-video frames, and its own split leaks (§5). Same scene, hence same split, if any of:
  - same source frame (name prefix)
  - a verified pair of the view pool (tools/view_pool.py pairs.csv: copies and real second views)
  - DINOv2 neighbours with cosine >= DINO_GROUP or >= MIN_INLIERS ORB/RANSAC inliers (as tools/wtbd_groups.py)
Images that match WTBD (pool.csv excluded = 1) are left out, so the two benchmarks share nothing. Groups are split
70/15/15, class-stratified (matching.split_groups), and audited like WTBD-group until no held-out image matches a
train image. The official 7 classes are kept.
Writes <out>/split_groups.csv (image path relative to the repo, subset, group, real_views = number of partners in
the same subset that are a different view, i.e. a verified non-copy pair covering < SAME_VIEW) and report.json.
Usage:
    python tools/wtblade_groups.py WTBlade-Defect --out runs/analysis/wtblade
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import matching as mt  # noqa: E402

POOL = os.path.join(ROOT, "runs", "analysis", "novel", "view_pool")


def source_of(path):
    return os.path.basename(path).split("_jpg.rf.")[0] if "_jpg.rf." in path else os.path.basename(path)


def classes_of(path):
    lab = os.path.splitext(path.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep))[0] + ".txt"
    return [int(l.split()[0]) for l in open(lab) if l.strip()] if os.path.exists(lab) else []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="WTBlade-Defect folder (relative to the repo root, as in the view pool)")
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "analysis", "wtblade"))
    ap.add_argument("--cache", default=os.path.join(ROOT, ".dataset_cache"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.chdir(ROOT)  # the embedding and verification caches are keyed by repo-relative paths

    import yaml

    names = yaml.safe_load(open(os.path.join(args.src, "data.yaml")))["names"]
    excluded = {r["image"] for r in csv.DictReader(open(os.path.join(POOL, "pool.csv"))) if r["excluded"] == "1"}
    paths = sorted(os.path.join(args.src, s, "images", f) for s in ("train", "valid", "test")
                   for f in os.listdir(os.path.join(args.src, s, "images")))
    n_all = len(paths)
    paths = [p for p in paths if p not in excluded]
    pos = {p: i for i, p in enumerate(paths)}
    N = len(paths)
    boxes = [classes_of(p) for p in paths]
    print(f"{N} images ({n_all - N} matching WTBD left out), {sum(map(len, boxes))} boxes")

    dsu = mt.DSU(N)
    first = {}
    for i, p in enumerate(paths):
        dsu.union(i, first.setdefault(source_of(p), i))
    real = {}  # image -> partners that are a different view
    n_pool = 0
    for r in csv.DictReader(open(os.path.join(POOL, "pairs.csv"))):
        a, b = pos.get(r["image_a"]), pos.get(r["image_b"])
        if a is None or b is None:
            continue
        dsu.union(a, b)
        n_pool += 1
        if r["same_photo"] == "0" and float(r["view_coverage"]) < mt.SAME_VIEW:
            real.setdefault(a, set()).add(b), real.setdefault(b, set()).add(a)
    print(f"source frames {len(first)}, view-pool pairs {n_pool}, images with a different view {len(real)}")

    E = mt.dinov2_embeddings(paths, os.path.join(args.cache, "dinov2_vits14.npz"))
    vcache = os.path.join(args.cache, "verify.json")
    known = json.load(open(vcache)) if os.path.exists(vcache) else {}
    key = lambda a, b: f"{min(a, b)}\t{max(a, b)}"  # noqa: E731

    def verified(pairs):
        todo = sorted({(min(a, b), max(a, b)) for a, b in pairs if key(a, b) not in known})
        if todo:
            print(f"  verifying {len(todo)} new pairs", flush=True)
            idx = sorted({p for t in todo for p in t})
            ipos = {p: i for i, p in enumerate(idx)}
            res = mt.verify_pairs(idx, [(ipos[a], ipos[b]) for a, b in todo], args.workers)
            known.update({key(idx[a], idx[b]): list(v) for (a, b), v in res.items()})
            json.dump(known, open(vcache, "w"))
        return {(a, b): known[key(a, b)][0] for a, b in pairs}

    nn_idx, nn_sim = mt.knn(E, mt.KNN)
    inl = verified([(paths[a], paths[int(b)]) for a in range(N) for b in nn_idx[a] if a != int(b)])
    for a in range(N):
        for b, s in zip(nn_idx[a], nn_sim[a]):
            b = int(b)
            if a != b and (s >= mt.DINO_GROUP or inl[(paths[a], paths[b])] >= mt.MIN_INLIERS):
                dsu.union(a, b)

    for rnd in range(1, 13):
        groups = dsu.groups()
        counts = np.array([np.bincount([c for i in g for c in boxes[i]], minlength=len(names)) for g in groups])
        assign = mt.split_groups(counts, np.array([len(g) for g in groups]), args.seed)
        split_of, group_of = [None] * N, [None] * N
        for gi, (g, s) in enumerate(zip(groups, assign)):
            for i in g:
                split_of[i], group_of[i] = s, gi
        train = np.array([i for i in range(N) if split_of[i] == "train"])
        held = np.array([i for i in range(N) if split_of[i] != "train"])
        top = np.concatenate([train[np.argsort(-(E[held[j:j + 1024]] @ E[train].T), axis=1)[:, :10]]
                              for j in range(0, len(held), 1024)])
        v = verified([(paths[int(a)], paths[int(b)]) for a, bs in zip(held, top) for b in bs])
        sims = {(paths[int(a)], paths[int(b)]): float(E[a] @ E[b]) for a, bs in zip(held, top) for b in bs}
        bad = [(a, b) for (a, b), n in v.items() if n >= mt.MIN_INLIERS or sims[(a, b)] >= mt.DINO_GROUP]
        print(f"round {rnd}: {len(groups)} groups, {len(bad)} held-out/train matches", flush=True)
        if not bad:
            break
        for a, b in bad:
            dsu.union(pos[a], pos[b])
    else:
        raise SystemExit("audit did not converge in 12 rounds")

    views = [len({j for j in real.get(i, ()) if split_of[j] == split_of[i]}) for i in range(N)]
    report = dict(images=N, left_out_matching_wtbd=n_all - N, source_frames=len(first), groups=len(groups),
                  largest_groups=sorted(Counter(group_of).values())[-5:], audit_rounds=rnd)
    report["split"] = {s: dict(images=sum(x == s for x in split_of), groups=len({g for g, x in zip(group_of, split_of) if x == s}),
                               images_with_a_different_view=sum(1 for i in range(N) if split_of[i] == s and views[i]),
                               boxes=dict(zip(names, np.bincount([c for i in range(N) if split_of[i] == s for c in boxes[i]],
                                                                 minlength=len(names)).tolist())))
                       for s in ("train", "val", "test")}
    with open(os.path.join(args.out, "split_groups.csv"), "w") as f:
        f.write("image,subset,group,real_views\n")
        f.writelines(f"{p},{s},{g},{v}\n" for p, s, g, v in zip(paths, split_of, group_of, views))
    json.dump(report, open(os.path.join(args.out, "report.json"), "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
