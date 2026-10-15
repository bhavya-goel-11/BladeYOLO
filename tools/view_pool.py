"""Unlabelled pool of registered view pairs for real-view consistency training (research log, Phases 19-21).

Pool images come from other blade datasets, used without their labels. Steps:
  1. exclusion  a pool image is dropped if it matches any WTBD image (train, val or test): embedding cosine >=
                DINO_GROUP with one of its 5 nearest WTBD images, or >= MIN_INLIERS verified ORB/RANSAC inliers.
                Val/test matches would leak; train matches add nothing new.
  2. pairs      verified same-surface pairs between remaining pool images, from the cross-view label audit's
                transforms (tools/view_label_audit.py: flip + similarity transform, >= MIN_INLIERS inliers).
  3. view type  a pair is a real second view if its two images come from different source photographs (other
                frame, other shot): photo_id() reads the source photo from the file name (Roboflow copies share
                <source>_jpg.rf.*, Wind Surface twins differ in _0/_1, tiles encode their parent DJI photo). Copies and
                tiles of one photo are synthetic views. Alignment similarity (ZNCC of intensities and of gradient
                magnitudes over the overlap) is reported too, but cannot separate the two: Roboflow augmentations
                (cutouts, colour shifts) lower it as much as a real change of view does.
Writes runs/analysis/novel/view_pool/{pool.csv, pairs.csv, report.md, zncc_hist.png, examples_*.jpg}.
--export DIR writes a self-contained copy for training elsewhere (Kaggle): DIR/pairs.csv with the real pairs only and
paths relative to DIR, and DIR/images/ with just the images those pairs use.

Usage:
    python tools/view_pool.py --sources WTBlade-Defect:wtblade WindSurface-Defect:windsurface
    python tools/view_pool.py --export view-pool        # after the run above: upload view-pool/ as a Kaggle dataset
"""

import argparse
import collections
import csv
import json
import os
import re
import sys
from multiprocessing import Pool

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import matching as mt  # noqa: E402
import view_label_audit as va  # noqa: E402

OUT = os.path.join(ROOT, "runs", "analysis", "novel", "view_pool")
WTBD = os.path.join(ROOT, "external", "WTBD-group")


TILE_NAMES = (
    r"^(DJI_\d+)_\d+_\d+(?:_\d+)?$",  # DJI_<photo>_<row>_<col>[_<n>]
    r"^(\d+_\d+)_\d+_\d+_\d+_\d+_\d+_(\d+)_(\d+)$",  # <set>_<photo>_<x>_<y>_<w>_<h>_<?>_<W>_<H>
    r"^(\d+)_\d+_\d+_\d+_\d+_\d+_(\d+)_(\d+)$",  # <photo>_<x>_<y>_<w>_<h>_<?>_<W>_<H>
)


def photo_id(path):
    """Source photograph of an image file: Roboflow exports are <source>_jpg.rf.<hash> (format suffixes such as _png_jpg
    dropped), Wind Surface twins add _0/_1, and tiles name their parent photo (DJI_0744_03_05 -> DJI_0744)."""
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = re.sub(r"_[01]$", "", stem) if "WindSurface" in path else stem
    src = re.sub(r"(_(png|jpe?g))+$", "", re.split(r"\.rf\.", stem)[0], flags=re.IGNORECASE).lstrip("_")
    src = re.sub(r"_[01]$", "", src) if "WindSurface" in path else src
    for pattern in TILE_NAMES:
        m = re.match(pattern, src)
        if m:
            return "photo:" + "_".join(m.groups())
    return src


def warp_pair(a, b, fl, M):
    """Greyscale a (480 px) flipped and warped into b's 480 px frame, b, and the overlap mask."""
    ga, gb = mt._gray(a), mt._gray(b)
    if fl is not None:
        ga = cv2.flip(ga, fl)
    h, w = gb.shape
    wa = cv2.warpAffine(ga, np.array(M, np.float32), (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    mask = cv2.warpAffine(np.full(ga.shape, 255, np.uint8), np.array(M, np.float32), (w, h), flags=cv2.INTER_NEAREST) > 0
    mask = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0  # drop interpolated borders
    return wa, gb, mask


def zncc(x, y, mask):
    x, y = x[mask].astype(np.float64), y[mask].astype(np.float64)
    if len(x) < 500:
        return float("nan")
    x, y = x - x.mean(), y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else float("nan")


def grad(g):
    g = g.astype(np.float32)
    return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))


def pair_job(args):
    a, b, fl, M = args
    wa, gb, mask = warp_pair(a, b, fl, M)
    return zncc(wa, gb, mask), zncc(grad(wa), grad(gb), mask), float(mask.mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", default=None,
                    help="<dataset root>:<label-audit name> (the audit's transforms_v3.json supplies the pairs)")
    ap.add_argument("--export", default=None, metavar="DIR", help="export the real pairs and their images to DIR")
    ap.add_argument("--cache", default=os.path.join(ROOT, ".dataset_cache"))
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    if args.export:
        return export(args.export)
    if not args.sources:
        ap.error("--sources is required unless --export")
    os.makedirs(OUT, exist_ok=True)

    # 1. exclusion against every WTBD image
    wtbd = [os.path.abspath(os.path.join(WTBD, s, "images", f)) for s in ("train", "val", "test")
            for f in sorted(os.listdir(os.path.join(WTBD, s, "images")))]
    wtbd_split = {p: p.split(os.sep)[-3] for p in wtbd}
    pool, src_of, items_of = [], {}, {}
    for spec in args.sources:
        root, name = spec.split(":")
        items = va.load(root)
        items_of[name] = items
        for p, _, _ in items:
            pool.append(p)
            src_of[p] = name
    E = mt.dinov2_embeddings(wtbd + pool, os.path.join(args.cache, "dinov2_vits14.npz"))
    Ew, Ep = E[:len(wtbd)], E[len(wtbd):]
    S = Ep @ Ew.T
    top = np.argsort(-S, axis=1)[:, :5]
    vcache = os.path.join(args.cache, "verify.json")
    known = json.load(open(vcache)) if os.path.exists(vcache) else {}
    key = lambda a, b: f"{min(a, b)}\t{max(a, b)}"  # noqa: E731
    cand = sorted({(min(pool[i], wtbd[j]), max(pool[i], wtbd[j])) for i in range(len(pool)) for j in top[i]})
    todo = [c for c in cand if key(*c) not in known]
    if todo:
        idx = sorted({x for c in todo for x in c})
        pos = {p: k for k, p in enumerate(idx)}
        res = mt.verify_pairs(idx, [(pos[a], pos[b]) for a, b in todo], args.workers)
        known.update({key(idx[a], idx[b]): list(v) for (a, b), v in res.items()})
        json.dump(known, open(vcache, "w"))
    excluded = {}
    for i, p in enumerate(pool):
        for j in top[i]:
            if S[i, j] >= mt.DINO_GROUP or known[key(p, wtbd[j])][0] >= mt.MIN_INLIERS:
                excluded.setdefault(p, set()).add(wtbd_split[wtbd[j]])
    with open(os.path.join(OUT, "pool.csv"), "w") as f:
        f.write("image,source,excluded,matches_wtbd_split\n")
        f.writelines(f"{p},{src_of[p]},{int(p in excluded)},{'|'.join(sorted(excluded.get(p, ())))}\n" for p in pool)

    # 2. verified pairs among kept pool images, 3. view type
    jobs, meta = [], []
    for name, items in items_of.items():
        T = json.load(open(os.path.join(ROOT, "runs", "analysis", "novel", "label_audit", name, "transforms_v3.json")))
        for k, (inl, fl, M, _) in T.items():
            x, y = map(int, k.split())
            a, b = items[x][0], items[y][0]
            if inl < mt.MIN_INLIERS or M is None or a in excluded or b in excluded:
                continue
            cov = va.view_coverage(items[x][1], items[y][1], fl, M)
            ang = np.degrees(np.arctan2(M[1][0], M[0][0])) % 90
            jobs.append((a, b, fl, M))
            meta.append((name, a, b, inl, fl, cov, min(ang, 90 - ang)))
    print(f"{len(pool)} pool images, {len(excluded)} excluded; {len(jobs)} directed verified pairs to score")
    with Pool(args.workers) as workers:
        scores = []
        for n, r in enumerate(workers.imap(pair_job, jobs, chunksize=32)):
            scores.append(r)
            if (n + 1) % 5000 == 0:
                print(f"  scored {n + 1}/{len(jobs)}", flush=True)
    with open(os.path.join(OUT, "pairs.csv"), "w") as f:
        w = csv.writer(f)
        w.writerow(["source", "image_a", "image_b", "same_photo", "inliers", "flip", "M", "view_coverage",
                    "angle_from_axis", "zncc", "zncc_grad", "overlap"])
        for (name, a, b, inl, fl, cov, ang), (z, zg, ov), (_, _, _, M) in zip(meta, scores, jobs):
            w.writerow([name, a, b, int(photo_id(a) == photo_id(b)), inl, fl, json.dumps(np.round(M, 6).tolist()),
                        f"{cov:.3f}", f"{ang:.1f}", f"{z:.4f}", f"{zg:.4f}", f"{ov:.3f}"])

    # summary, histogram, example sheets per ZNCC band
    z = np.array([s[0] for s in scores])
    zg = np.array([s[1] for s in scores])
    by_src = collections.Counter(src_of[p] for p in excluded)
    by_split = collections.Counter(s for v in excluded.values() for s in v)
    md = ["# Unlabelled view pool", "",
          f"Pool images: {len(pool)} (" + ", ".join(f"{n} {sum(1 for p in pool if src_of[p] == n)}" for n in items_of) + ")",
          f"Excluded (match a WTBD image): {len(excluded)} (" + ", ".join(f"{n} {c}" for n, c in by_src.items()) +
          "); matched WTBD splits: " + ", ".join(f"{s} {c}" for s, c in by_split.items()),
          f"Verified directed pairs between kept images: {len(jobs)}; real second views (different source photos): "
          f"{sum(photo_id(j[0]) != photo_id(j[1]) for j in jobs)} (" + ", ".join(
              f"{n} {sum(photo_id(j[0]) != photo_id(j[1]) for j, m in zip(jobs, meta) if m[0] == n)}" for n in items_of) + ")",
          "Source photos: " + ", ".join(f"{n} {len({photo_id(p) for p in pool if src_of[p] == n and p not in excluded})}"
                                         for n in items_of), "",
          "| ZNCC band (intensity) | pairs | median gradient ZNCC |", "|---|---|---|"]
    bands = [(-1, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 0.98), (0.98, 1.01)]
    for lo, hi in bands:
        m = (z >= lo) & (z < hi)
        md.append(f"| [{lo:.2f}, {hi:.2f}) | {int(m.sum())} | {np.nanmedian(zg[m]) if m.any() else float('nan'):.3f} |")
    open(os.path.join(OUT, "report.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), dpi=130)
    ax[0].hist(z[np.isfinite(z)], bins=100)
    ax[0].set_xlabel("intensity ZNCC after alignment")
    ax[1].hist(zg[np.isfinite(zg)], bins=100)
    ax[1].set_xlabel("gradient ZNCC after alignment")
    for a_ in ax:
        a_.set_ylabel("verified pairs")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "zncc_hist.png"))
    rng = np.random.default_rng(0)
    for lo, hi in bands:
        idx = np.where((z >= lo) & (z < hi))[0]
        idx = idx[rng.permutation(len(idx))][:8]
        tiles = []
        for i in idx:
            a, b, fl, M = jobs[i]
            wa, gb, mask = warp_pair(a, b, fl, M)
            tile = np.hstack([cv2.resize(cv2.imread(a), (240, 240)), cv2.resize(cv2.imread(b), (240, 240)),
                              cv2.resize(cv2.cvtColor(np.where(mask, wa, 0).astype(np.uint8), cv2.COLOR_GRAY2BGR), (240, 240))])
            cv2.putText(tile, f"zncc {z[i]:.3f} grad {zg[i]:.3f}", (5, 20), 0, 0.6, (0, 0, 255), 2)
            tiles.append(tile)
        if tiles:
            cv2.imwrite(os.path.join(OUT, f"examples_zncc_{lo:.2f}_{hi:.2f}.jpg"), np.vstack(tiles))


def export(out):
    """Real pairs (different source photos) of pairs.csv, with their images copied under out/images/."""
    import shutil

    rows = [r for r in csv.DictReader(open(os.path.join(OUT, "pairs.csv"))) if r["same_photo"] == "0"]
    rel = lambda p: os.path.join("images", p.replace("/images/", "/"))  # noqa: E731  <dataset>/<split>/<file>
    for p in sorted({r["image_a"] for r in rows} | {r["image_b"] for r in rows}):
        dst = os.path.join(out, rel(p))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):
            shutil.copy2(os.path.join(ROOT, p), dst)
    with open(os.path.join(out, "pairs.csv"), "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({**r, "image_a": rel(r["image_a"]), "image_b": rel(r["image_b"])})
    n_img = len({r["image_a"] for r in rows} | {r["image_b"] for r in rows})
    print(f"exported {len(rows)} real pairs and {n_img} images to {out}")


if __name__ == "__main__":
    main()
