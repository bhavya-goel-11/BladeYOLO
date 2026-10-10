"""How much of a candidate YOLO dataset is new relative to WindSurface-Defect-v3, and is any of it in our val/test?

Uses the dataset builder's matching: source-name stems and parent photos from tile names, DINOv2 retrieval and
flip-aware ORB/RANSAC verification. Every candidate image gets one status:
  held_out  same content (verified match or embedding >= DINO_GROUP), same source stem or same parent photo as a
            v3 val/test image: must never be added to training
  in_train  same content or same source stem as a v3 train image: adds nothing
  new       neither. New images are then de-duplicated among themselves (Roboflow copies, flips, near-identical
            frames) and counted per class as unique images/boxes.

Usage:
    python tools/check_overlap.py wind-turbine.v1i.yolov12
Writes runs/analysis/overlap_<name>/{report.md, report.json, images.csv}.
"""

import argparse
import collections
import csv
import hashlib
import json
import os
import sys
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import numpy as np  # noqa: E402

import build_dataset as B  # noqa: E402


def candidate_items(root):
    import yaml

    names = yaml.safe_load(open(os.path.join(root, "data.yaml")))["names"]
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else names  # YOLO allows {id: name}
    items = []
    for split in sorted(os.listdir(root)):
        img_dir, lab_dir = os.path.join(root, split, "images"), os.path.join(root, split, "labels")
        if not os.path.isdir(img_dir):
            continue
        for f in sorted(os.listdir(img_dir)):
            stem = os.path.splitext(f)[0]
            src = B.source_stem(stem)
            items.append(dict(path=os.path.relpath(os.path.join(img_dir, f), ROOT), split=split, stem=src,
                              photo=B.photo_key(src), boxes=B.read_boxes(os.path.join(lab_dir, stem + ".txt"))))
    return items, names


def reference_items(v3):
    """v3 images, addressed by their original source file (whose embedding/verification are already cached)."""
    import re

    items = []
    for r in csv.DictReader(open(os.path.join(v3, "manifest.csv"))):
        stem = os.path.splitext(os.path.basename(r["original"]))[0]
        src = B.source_stem(re.sub(r"_[01]$", "", stem) if r["source"] == "ws" else stem)
        items.append(dict(path=r["original"], split=r["split"], group=r["group"], stem=src, photo=B.photo_key(src)))
    return items


def file_md5(path):
    return hashlib.md5(open(os.path.join(ROOT, path), "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("candidate")
    ap.add_argument("--v3", default=os.path.join(ROOT, "WindSurface-Defect-v3"))
    ap.add_argument("--cache", default=os.path.join(ROOT, ".dataset_cache"))
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--knn", type=int, default=10)
    args = ap.parse_args()
    os.chdir(ROOT)  # cached embeddings/verifications are keyed by project-relative paths

    cand, names = candidate_items(os.path.relpath(os.path.abspath(args.candidate), ROOT))
    ref = reference_items(args.v3)
    nc, nr = len(cand), len(ref)
    paths = [c["path"] for c in cand] + [r["path"] for r in ref]  # candidate i -> i, reference j -> nc + j
    print(f"candidate {nc} images ({len(names)} classes), reference v3 {nr} images")

    print("1/4 embeddings (DINOv2, cached per file)")
    E = B.dinov2_embeddings(paths, os.path.join(args.cache, "dinov2_vits14.npz"))
    Ec, Er = E[:nc], E[nc:]

    print("2/4 retrieval: candidate -> v3 and candidate -> candidate")
    sims_r = np.zeros((nc, args.knn), np.float32)
    idx_r = np.zeros((nc, args.knn), int)
    for i in range(0, nc, 2048):
        s = Ec[i:i + 2048] @ Er.T
        j = np.argpartition(-s, args.knn, axis=1)[:, :args.knn]
        idx_r[i:i + 2048], sims_r[i:i + 2048] = j, np.take_along_axis(s, j, 1)
    idx_c, sims_c = B.knn(Ec, args.knn)

    print("3/4 geometric verification (ORB + RANSAC, flip-aware; cached per file pair)")
    vcache = os.path.join(args.cache, "verify.json")
    known = json.load(open(vcache)) if os.path.exists(vcache) else {}
    pairs = {(a, nc + int(b)) for a in range(nc) for b in idx_r[a]} | {(min(a, int(b)), max(a, int(b))) for a in range(nc) for b in idx_c[a]}
    lookup = {}
    todo = []
    for a, b in pairs:
        k1, k2 = f"{paths[a]}\t{paths[b]}", f"{paths[b]}\t{paths[a]}"
        v = known.get(k1) or known.get(k2)
        if v is None:
            todo.append((a, b))
        else:
            lookup[(a, b)] = tuple(v)
    print(f"  {len(pairs)} pairs, {len(todo)} to verify")
    if todo:
        new = B.verify_pairs(paths, sorted(todo), args.workers)
        lookup.update(new)
        known.update({f"{paths[a]}\t{paths[b]}": v for (a, b), v in new.items()})
        json.dump(known, open(vcache, "w"))

    print("4/4 classification")
    ref_by_stem, ref_by_photo = collections.defaultdict(list), collections.defaultdict(list)
    for j, r in enumerate(ref):
        ref_by_stem[r["stem"]].append(j)
        if r["photo"]:
            ref_by_photo[r["photo"]].append(j)
    status, reason, match = [], [], []
    for i, c in enumerate(cand):
        hits = []  # (reference index, why)
        hits += [(j, "same source stem") for j in ref_by_stem.get(c["stem"], [])]
        hits += [(j, "same parent photo") for j in (ref_by_photo.get(c["photo"], []) if c["photo"] else [])]
        for j, s in zip(idx_r[i], sims_r[i]):
            inl, _ = lookup.get((i, nc + int(j)), (0, 0.0))
            if inl >= B.MIN_INLIERS:
                hits.append((int(j), f"verified match ({inl} inliers)"))
            elif s >= B.DINO_GROUP:
                hits.append((int(j), f"embedding {s:.3f}"))
        held = [(j, w) for j, w in hits if ref[j]["split"] != "train"]
        same_train = [(j, w) for j, w in hits if ref[j]["split"] == "train" and w != "same parent photo"]
        if held:
            status.append("held_out"), reason.append(held[0][1]), match.append(ref[held[0][0]]["path"])
        elif same_train:
            status.append("in_train"), reason.append(same_train[0][1]), match.append(ref[same_train[0][0]]["path"])
        else:
            photo_train = [j for j, w in hits if w == "same parent photo"]
            status.append("new"), reason.append("same parent photo as v3 train" if photo_train else ""), match.append("")

    # unique new images: collapse Roboflow copies, exact files, same-view matches and near-identical embeddings
    new_idx = [i for i in range(nc) if status[i] == "new"]
    dsu = B.DSU(nc)
    by_key = collections.defaultdict(list)
    for i in new_idx:
        by_key["stem:" + cand[i]["stem"]].append(i)
    with Pool(args.workers) as pool:
        for i, h in zip(new_idx, pool.map(file_md5, [cand[i]["path"] for i in new_idx], chunksize=64)):
            by_key["md5:" + h].append(i)
    for idx in by_key.values():
        for j in idx[1:]:
            dsu.union(idx[0], j)
    new_set = set(new_idx)
    for a in new_idx:
        for b, s in zip(idx_c[a], sims_c[a]):
            b = int(b)
            if b not in new_set:
                continue
            inl, cov = lookup.get((min(a, b), max(a, b)), (0, 0.0))
            if s >= B.DINO_DUP or (inl >= B.MIN_INLIERS and cov >= B.SAME_VIEW):
                dsu.union(a, b)
    clusters = collections.defaultdict(list)
    for i in new_idx:
        clusters[dsu.find(i)].append(i)
    # representative: the copy with the most boxes (ties: first by name)
    reps = [max(sorted(cl, key=lambda i: cand[i]["path"]), key=lambda i: len(cand[i]["boxes"])) for cl in clusters.values()]

    def per_class(indices):
        boxes, imgs, photos = collections.Counter(), collections.Counter(), collections.defaultdict(set)
        for i in indices:
            cs = {b[0] for b in cand[i]["boxes"]}
            for b in cand[i]["boxes"]:
                boxes[b[0]] += 1
            for c in cs:
                imgs[c] += 1
                photos[c].add(cand[i]["photo"] or cand[i]["stem"])
        return {names[k]: dict(boxes=boxes[k], images=imgs[k], parent_photos=len(photos[k])) for k in range(len(names))}

    counts = collections.Counter(status)
    report = dict(
        candidate=args.candidate, images=nc, unique_source_stems=len({c["stem"] for c in cand}),
        status_counts=dict(counts), unique_new_images=len(reps),
        new_reasons=dict(collections.Counter(reason[i] or "no relation to v3" for i in new_idx)),
        per_class_all=per_class(range(nc)), per_class_unique_new=per_class(reps),
        per_class_held_out=per_class([i for i in range(nc) if status[i] == "held_out"]),
    )
    out = os.path.join(ROOT, "runs", "analysis", "overlap_" + os.path.basename(os.path.normpath(args.candidate)))
    os.makedirs(out, exist_ok=True)
    json.dump(report, open(os.path.join(out, "report.json"), "w"), indent=2)
    rep_set = set(reps)
    with open(os.path.join(out, "images.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "status", "reason", "matched_v3_image", "unique_new_representative", "boxes"])
        for i, c in enumerate(cand):
            w.writerow([c["path"], status[i], reason[i], match[i], i in rep_set, len(c["boxes"])])
    lines = [f"# Overlap of `{args.candidate}` with WindSurface-Defect-v3", "",
             f"- images: {nc} ({report['unique_source_stems']} unique source stems)",
             f"- held_out (matches v3 val/test, must not be used): {counts['held_out']}",
             f"- in_train (duplicates of v3 train): {counts['in_train']}",
             f"- new: {counts['new']} images -> **{len(reps)} unique** after removing copies", "",
             "| class | all boxes | all photos | unique-new boxes | unique-new images | unique-new photos | held-out boxes |",
             "|---|---|---|---|---|---|---|"]
    for n in names:
        a, u, h = report["per_class_all"][n], report["per_class_unique_new"][n], report["per_class_held_out"][n]
        lines.append(f"| {n} | {a['boxes']} | {a['parent_photos']} | {u['boxes']} | {u['images']} | {u['parent_photos']} | {h['boxes']} |")
    open(os.path.join(out, "report.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}/report.md, report.json, images.csv")


if __name__ == "__main__":
    main()
