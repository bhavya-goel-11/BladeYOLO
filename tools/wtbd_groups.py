"""Leakage audit of WTBD's published split, and a group split that keeps near-duplicate images together.

WTBD's 1,065 images are 1024 px crops of ~2,500 raw drone photos with turbine metadata removed, and the authors'
split is a seeded random shuffle of image indices, so neighbouring crops of one blade can sit in train and test.
Same matching as tools/build_dataset.py: DINOv2 retrieval (k nearest neighbours), flip-aware ORB/RANSAC verification;
two images are the same scene if verified (>= MIN_INLIERS inliers) or embedding cosine >= DINO_GROUP.
  1. authors' split: held-out images with a same-scene image in train
  2. group split: connected same-scene groups assigned whole to train/val/test (class-stratified), audited until no
     held-out image matches a train image; written as <out>/split_groups.csv for tools/convert_wtbd.py --split-file
  3. overlap with WindSurface-Defect-v3 (any split): WTBD images matching a v3 image
Usage:
    python tools/wtbd_groups.py "external/WTBD/WT blade defect dataset" --out runs/analysis/wtbd
"""

import argparse
import collections
import csv
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import build_dataset as B  # noqa: E402
from convert_wtbd import CLASSES  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "analysis", "wtbd"))
    ap.add_argument("--v3", default=os.path.join(ROOT, "WindSurface-Defect-v3"))
    ap.add_argument("--cache", default=os.path.join(ROOT, ".dataset_cache"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    authors = {r["ImageID"]: r["Subset"] for r in csv.DictReader(open(os.path.join(args.src, "train_val_test_split.txt")))}
    names = sorted(authors, key=lambda n: int(n.split(".")[0]))
    paths = [os.path.abspath(os.path.join(args.src, "JPEGImages", n)) for n in names]
    boxes = []
    for n in names:
        root = ET.parse(os.path.join(args.src, "Annotations", n.replace(".jpg", ".xml"))).getroot()
        boxes.append([CLASSES.index(o.find("name").text.strip()) for o in root.iter("object")])
    N = len(paths)

    v3 = [os.path.join(args.v3, r["file"]) for r in csv.DictReader(open(os.path.join(args.v3, "manifest.csv")))]
    v3_split = {os.path.join(args.v3, r["file"]): r["split"] for r in csv.DictReader(open(os.path.join(args.v3, "manifest.csv")))}
    allp = paths + v3
    E = B.dinov2_embeddings(allp, os.path.join(args.cache, "dinov2_vits14.npz"))
    Ew, Ev = E[:N], E[N:]

    vcache = os.path.join(args.cache, "verify.json")
    known = json.load(open(vcache)) if os.path.exists(vcache) else {}

    def verified(pairs):
        """pairs of (path_a, path_b) -> {(a, b): inliers}, cached."""
        key = lambda a, b: f"{min(a, b)}\t{max(a, b)}"  # noqa: E731
        todo = sorted({(min(a, b), max(a, b)) for a, b in pairs if key(a, b) not in known})
        if todo:
            idx = sorted({p for t in todo for p in t})
            pos = {p: i for i, p in enumerate(idx)}
            res = B.verify_pairs(idx, [(pos[a], pos[b]) for a, b in todo], args.workers)
            known.update({key(idx[a], idx[b]): list(v) for (a, b), v in res.items()})
            json.dump(known, open(vcache, "w"))
        return {(a, b): known[key(a, b)][0] for a, b in pairs}

    # same-scene graph inside WTBD
    nn_idx, nn_sim = B.knn(Ew, B.KNN)
    pairs = [(paths[a], paths[int(b)]) for a in range(N) for b in nn_idx[a]]
    inl = verified(pairs)
    dsu = B.DSU(N)
    edges = set()
    for a in range(N):
        for b, s in zip(nn_idx[a], nn_sim[a]):
            b = int(b)
            if s >= B.DINO_GROUP or inl[(paths[a], paths[b])] >= B.MIN_INLIERS:
                dsu.union(a, b)
                edges.add((min(a, b), max(a, b)))

    # 1. authors' split
    def leaks(split_of):
        out = []
        for a, b in edges:
            sa, sb = split_of[a], split_of[b]
            if sa != sb and "train" in (sa, sb):
                out.append((a, b) if sa != "train" else (b, a))
        return out

    authors_of = [authors[n] for n in names]
    al = leaks(authors_of)
    held_leaky = {a for a, _ in al}
    report = dict(images=N, same_scene_pairs=len(edges), groups=len(dsu.groups()),
                  authors_split={s: sum(x == s for x in authors_of) for s in ("train", "val", "test")},
                  authors_split_heldout_images_with_train_match={s: sum(authors_of[a] == s for a in held_leaky)
                                                                for s in ("val", "test")})

    # 2. group split with audit
    for rnd in range(1, 13):
        groups = dsu.groups()
        counts = np.array([np.bincount([c for i in g for c in boxes[i]], minlength=len(CLASSES)) for g in groups])
        assign = B.split_groups(counts, np.array([len(g) for g in groups]), args.seed)
        split_of = [None] * N
        for g, s in zip(groups, assign):
            for i in g:
                split_of[i] = s
        train = np.array([i for i in range(N) if split_of[i] == "train"])
        held = np.array([i for i in range(N) if split_of[i] != "train"])
        top = train[np.argsort(-(Ew[held] @ Ew[train].T), axis=1)[:, :10]]
        ap_ = [(paths[int(a)], paths[int(b)]) for a, bs in zip(held, top) for b in bs]
        v = verified(ap_)
        bad = [(a, b) for (a, b), n in v.items() if n >= B.MIN_INLIERS]
        print(f"round {rnd}: {len(groups)} groups, {len(bad)} held-out/train matches")
        if not bad:
            break
        pos = {p: i for i, p in enumerate(paths)}
        for a, b in bad:
            dsu.union(pos[a], pos[b])
            edges.add((min(pos[a], pos[b]), max(pos[a], pos[b])))
    report["group_split"] = {s: dict(images=sum(x == s for x in split_of),
                                     boxes=dict(zip(CLASSES, np.bincount([c for i in range(N) if split_of[i] == s
                                                                          for c in boxes[i]], minlength=len(CLASSES)).tolist())))
                             for s in ("train", "val", "test")}
    with open(os.path.join(args.out, "split_groups.csv"), "w") as f:
        f.write("ImageID,Subset\n")
        f.writelines(f"{n},{s}\n" for n, s in zip(names, split_of))

    # 3. overlap with v3
    sims = Ew @ Ev.T
    top = np.argsort(-sims, axis=1)[:, :5]
    cand = [(paths[a], v3[int(b)]) for a in range(N) for b in top[a]]
    v = verified(cand)
    match = collections.defaultdict(set)
    for (a, b), n in v.items():
        if n >= B.MIN_INLIERS:
            match[a].add(v3_split[b])
    for a in range(N):
        if sims[a, top[a, 0]] >= B.DINO_GROUP:
            match[paths[a]].add(v3_split[v3[int(top[a, 0])]])
    report["overlap_with_v3"] = dict(wtbd_images_matching_v3=len(match),
                                     by_v3_split=dict(collections.Counter(s for ss in match.values() for s in ss)))
    json.dump(report, open(os.path.join(args.out, "report.json"), "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
