"""WTBlade-group: one image per source frame on the leak-free group split of tools/wtblade_groups.py, plus the
registered real views between frames of the same split (Phase 28, H4).

WTBlade's Roboflow copies are offline augmentations of 2,791 source frames; keeping them would turn every scene into
test-time augmentation supplied by the dataset (§24b, H1). Each source frame keeps its first copy (sorted name).
Frames of one split that the view pool links (a verified non-copy pair of their copies) are registered again on the
kept images (view_label_audit.transform_job, both directions; >= MIN_INLIERS inliers) and written to views.json:
{target name: [[source name, inliers, flip, M (2x3, 480 px frames), coverage], ...]}. coverage < SAME_VIEW is a
different view of the surface; above it, a neighbouring video frame of nearly the same view.
Writes <out>/{train,val,test}/{images,labels}, data.yaml (the official 7 classes), groups.csv, views_{val,test}.json.
Usage:
    python tools/build_wtblade_group.py WTBlade-Defect --out external/WTBlade-group
"""

import argparse
import csv
import json
import os
import shutil
import sys
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import matching as mt  # noqa: E402
import view_label_audit as va  # noqa: E402
from wtblade_groups import POOL, source_of  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("--split", default=os.path.join(ROOT, "runs", "analysis", "wtblade", "split_groups.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "external", "WTBlade-group"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    os.chdir(ROOT)
    import yaml

    rows = list(csv.DictReader(open(args.split)))
    keep, split_of, group_of = {}, {}, {}
    for r in sorted(rows, key=lambda r: r["image"]):
        s = source_of(r["image"])
        keep.setdefault(s, r["image"])
        split_of[s], group_of[s] = r["subset"], r["group"]
    for split in ("train", "val", "test"):
        for d in ("images", "labels"):
            os.makedirs(os.path.join(args.out, split, d), exist_ok=True)
    for s, p in keep.items():
        name = os.path.basename(p)
        shutil.copyfile(p, os.path.join(args.out, split_of[s], "images", name))
        lab = os.path.splitext(p.replace("/images/", "/labels/"))[0] + ".txt"
        shutil.copyfile(lab, os.path.join(args.out, split_of[s], "labels", os.path.splitext(name)[0] + ".txt"))
    names = yaml.safe_load(open(os.path.join(args.src, "data.yaml")))["names"]
    yaml.safe_dump(dict(path=".", train="train/images", val="val/images", test="test/images", nc=len(names), names=names),
                   open(os.path.join(args.out, "data.yaml"), "w"), sort_keys=False)
    with open(os.path.join(args.out, "groups.csv"), "w") as f:
        f.write("image,subset,group\n")
        f.writelines(f"{os.path.basename(keep[s])},{split_of[s]},{group_of[s]}\n" for s in sorted(keep))

    links = set()
    for r in csv.DictReader(open(os.path.join(POOL, "pairs.csv"))):
        a, b = source_of(r["image_a"]), source_of(r["image_b"])
        if r["source"] == "wtblade" and r["same_photo"] == "0" and a != b and a in keep and b in keep \
                and split_of[a] == split_of[b] and split_of[a] != "train":
            links.add((min(a, b), max(a, b)))
    jobs = [(keep[a], keep[b]) for a, b in sorted(links)] + [(keep[b], keep[a]) for a, b in sorted(links)]
    print(f"{len(keep)} frames: " + ", ".join(f"{s} {sum(v == s for v in split_of.values())}" for s in ("train", "val", "test"))
          + f"; registering {len(jobs)} directed pairs of linked val/test frames", flush=True)
    with Pool(args.workers) as pool:
        res = pool.map(va.transform_job, jobs, chunksize=16)
    views = {"val": {}, "test": {}}
    n_other = {"val": 0, "test": 0}
    for (pa, pb), (inl, fl, M, _) in zip(jobs, res):
        if inl < mt.MIN_INLIERS:
            continue
        cov = va.view_coverage((640, 640), (640, 640), fl, M)  # every WTBlade image is 640 x 640
        s = split_of[source_of(pb)]
        views[s].setdefault(os.path.basename(pb), []).append([os.path.basename(pa), inl, fl, M, round(cov, 3)])
        n_other[s] += cov < mt.SAME_VIEW
    for s in ("val", "test"):
        json.dump(views[s], open(os.path.join(args.out, f"views_{s}.json"), "w"))
        other = {t for t, v in views[s].items() if any(c < mt.SAME_VIEW for *_, c in v)}
        print(f"{s}: {sum(map(len, views[s].values()))} registered directed pairs ({n_other[s]} different-view) into "
              f"{len(views[s])} frames; {len(other)} frames have a different view")


if __name__ == "__main__":
    main()
