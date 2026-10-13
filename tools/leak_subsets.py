"""Split WTBD's published test set into images that have a same-scene twin in its train set and images that do not.

Same matching as tools/wtbd_groups.py: for every test image, its 10 nearest train images by DINOv2 embedding are
checked; a twin is a pair with embedding cosine >= DINO_GROUP or >= MIN_INLIERS flip-aware ORB/RANSAC inliers.
Scoring a model trained on the published split on both subsets separates memorisation from detection: if the
inflation comes from the duplicates, the leaked subset scores high and the clean subset near the leak-free split.
Writes <out>/{leaked,clean}/test/{images,labels} (hard links) with data.yaml, and <out>/test_leak.csv.

Usage:
    python tools/leak_subsets.py --official WTBD-official --out runs/analysis/wtbd_leak
    python train.py --data runs/analysis/wtbd_leak/leaked/data.yaml --test <run>/weights/best.pt
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import build_dataset as B  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=os.path.join(ROOT, "external", "WTBD", "WT blade defect dataset"))
    ap.add_argument("--official", default=os.path.join(ROOT, "external", "WTBD-official"))
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "analysis", "wtbd_leak"))
    ap.add_argument("--cache", default=os.path.join(ROOT, ".dataset_cache"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    names = {s: sorted(os.listdir(os.path.join(args.official, s, "images"))) for s in ("train", "test")}
    # embeddings and verifications are cached under the source image paths (tools/wtbd_groups.py)
    src = {s: [os.path.abspath(os.path.join(args.src, "JPEGImages", n)) for n in names[s]] for s in names}
    E = B.dinov2_embeddings(src["train"] + src["test"], os.path.join(args.cache, "dinov2_vits14.npz"))
    Etr, Ete = E[:len(src["train"])], E[len(src["train"]):]
    sims = Ete @ Etr.T
    top = np.argsort(-sims, axis=1)[:, :B.KNN]

    vcache = os.path.join(args.cache, "verify.json")
    known = json.load(open(vcache)) if os.path.exists(vcache) else {}
    key = lambda a, b: f"{min(a, b)}\t{max(a, b)}"  # noqa: E731
    pairs = [(src["test"][i], src["train"][int(j)]) for i in range(len(top)) for j in top[i]]
    todo = sorted({(min(a, b), max(a, b)) for a, b in pairs if key(a, b) not in known})
    if todo:
        idx = sorted({p for t in todo for p in t})
        pos = {p: i for i, p in enumerate(idx)}
        res = B.verify_pairs(idx, [(pos[a], pos[b]) for a, b in todo], args.workers)
        known.update({key(idx[a], idx[b]): list(v) for (a, b), v in res.items()})
        json.dump(known, open(vcache, "w"))

    rows = []
    for i, n in enumerate(names["test"]):
        twins = [(names["train"][int(j)], float(sims[i, j]), known[key(src["test"][i], src["train"][int(j)])][0])
                 for j in top[i]]
        twins = [t for t in twins if t[1] >= B.DINO_GROUP or t[2] >= B.MIN_INLIERS]
        rows.append((n, bool(twins), twins[0] if twins else None))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "test_leak.csv"), "w") as f:
        f.write("image,leaked,twin_in_train,cosine,inliers\n")
        f.writelines(f"{n},{int(l)},{t[0] if t else ''},{t[1] if t else ''},{t[2] if t else ''}\n" for n, l, t in rows)
    yaml_head = open(os.path.join(args.official, "data.yaml")).read()
    for sub, want in (("leaked", True), ("clean", False)):
        for d in ("images", "labels"):
            os.makedirs(os.path.join(args.out, sub, "test", d), exist_ok=True)
        for n, l, _ in rows:
            if l != want:
                continue
            stem = os.path.splitext(n)[0]
            for d, f in (("images", n), ("labels", stem + ".txt")):
                dst = os.path.join(args.out, sub, "test", d, f)
                if not os.path.exists(dst):
                    os.link(os.path.realpath(os.path.join(args.official, "test", d, f)), dst)
        with open(os.path.join(args.out, sub, "data.yaml"), "w") as f:
            f.write(yaml_head.replace("train: train/images", "train: test/images").replace("val: val/images",
                                                                                          "val: test/images"))
    n_leak = sum(l for _, l, _ in rows)
    print(f"test images: {len(rows)}, with a twin in train: {n_leak}, without: {len(rows) - n_leak}")


if __name__ == "__main__":
    main()
