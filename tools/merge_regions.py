"""Labelling convention "one box per connected damage region": merge same-class boxes that overlap or touch.

The source datasets cut one damage region into several adjacent boxes, inconsistently; a detector that boxes the whole
region is then scored as a miss plus false positives (runs/analysis/merge_experiment.txt: about +0.05 mAP50 on test
for the same predictions once both sides follow one convention). This applies the convention to every split, so
training, validation and test agree:
  same class, boxes within --gap pixels (at the 640 input) of each other -> one box, their union, as long as the
  boxes cover at least --min-fill of that union box (a diagonal staircase of boxes along a slanted edge is a
  deliberate tight cover and is left alone; merging it would produce a box that is mostly background)
Images are hard-linked; data.yaml, ignore_regions.json and the manifest (box counts updated) are carried over and every
merge is logged in <out>/merge_log.csv. Scores on the output are not comparable with the input dataset: report both.

Usage:
    python tools/merge_regions.py --data WindSurface-Defect-v3.1 --out WindSurface-Defect-v3.2
"""

import argparse
import csv
import os
import shutil

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fill(boxes, idx):
    """Fraction of the union bounding box covered by the boxes idx (rasterised on a 1 px grid)."""
    b = np.array([boxes[i] for i in idx])
    x0, y0 = b[:, :2].min(0)
    W, H = int(np.ceil(b[:, 2].max() - x0)) + 1, int(np.ceil(b[:, 3].max() - y0)) + 1
    m = np.zeros((H, W), bool)
    for x1, y1, x2, y2 in b:
        m[int(y1 - y0):int(np.ceil(y2 - y0)), int(x1 - x0):int(np.ceil(x2 - x0))] = True
    return m.mean()


def components(boxes, gap, min_fill):
    """Agglomerative merging of boxes (x1, y1, x2, y2): repeatedly join the two groups that are within gap of each
    other and whose union box is best covered by their boxes, while that coverage is >= min_fill. Overlapping blobs
    merge; a diagonal staircase of boxes along an edge (a deliberate tight cover of a slanted structure) does not,
    because its union box would be mostly background."""
    groups = [[i] for i in range(len(boxes))]

    def near(g, h):
        return any(not (boxes[i][2] + gap < boxes[j][0] or boxes[j][2] + gap < boxes[i][0] or
                        boxes[i][3] + gap < boxes[j][1] or boxes[j][3] + gap < boxes[i][1]) for i in g for j in h)

    while True:
        best = max(((fill(boxes, g + h), a, b) for a, g in enumerate(groups) for b, h in enumerate(groups)
                    if a < b and near(g, h)), default=None)
        if best is None or best[0] < min_fill:
            return sorted(groups)
        _, a, b = best
        groups[a] = groups[a] + groups[b]
        del groups[b]


def merge_labels(rows, W, H, gap, min_fill):
    """rows: [cls, x, y, w, h] normalised -> (merged rows, list of merged groups)."""
    s = 640 / max(W, H)
    px = [(r[0], (r[1] - r[3] / 2) * W * s, (r[2] - r[4] / 2) * H * s, (r[1] + r[3] / 2) * W * s, (r[2] + r[4] / 2) * H * s)
          for r in rows]
    out, merged = [], []
    for c in sorted({r[0] for r in px}):
        cb = [r[1:] for r in px if r[0] == c]
        for g in components(cb, gap, min_fill):
            x1, y1 = min(cb[i][0] for i in g), min(cb[i][1] for i in g)
            x2, y2 = max(cb[i][2] for i in g), max(cb[i][3] for i in g)
            out.append([c, (x1 + x2) / 2 / (W * s), (y1 + y2) / 2 / (H * s), (x2 - x1) / (W * s), (y2 - y1) / (H * s)])
            if len(g) > 1:
                merged.append((c, len(g)))
    return out, merged


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(ROOT, "WindSurface-Defect-v3.1"))
    ap.add_argument("--out", default=os.path.join(ROOT, "WindSurface-Defect-v3.2"))
    ap.add_argument("--gap", type=float, default=8.0, help="max gap in pixels at the 640 input (0: overlap/touch only)")
    ap.add_argument("--min-fill", type=float, default=0.6,
                    help="merge only while the boxes cover at least this fraction of their union box")
    args = ap.parse_args()

    data, out = os.path.abspath(args.data), os.path.abspath(args.out)
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; remove it first")
    rows = list(csv.DictReader(open(os.path.join(data, "manifest.csv"))))
    log, counts = [], {}
    for r in rows:
        split, stem = r["split"], os.path.splitext(os.path.basename(r["file"]))[0]
        src_img, dst_img = os.path.join(data, r["file"]), os.path.join(out, r["file"])
        os.makedirs(os.path.dirname(dst_img), exist_ok=True)
        os.link(os.path.realpath(src_img), dst_img)
        lab = os.path.join(data, "labels", split, stem + ".txt")
        old = [[int(v[0])] + [float(x) for x in v[1:]] for v in (l.split() for l in open(lab)) if len(v) == 5]
        new, merged = merge_labels(old, int(r["width"]), int(r["height"]), args.gap, args.min_fill)
        os.makedirs(os.path.join(out, "labels", split), exist_ok=True)
        with open(os.path.join(out, "labels", split, stem + ".txt"), "w") as f:
            f.writelines(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in new)
        log += [dict(file=r["file"], split=split, cls=c, boxes_merged=n) for c, n in merged]
        b = counts.setdefault(split, [0, 0])
        b[0], b[1] = b[0] + len(old), b[1] + len(new)
        r["boxes"] = len(new)
    for name in ("data.yaml", "ignore_regions.json"):
        if os.path.exists(os.path.join(data, name)):
            shutil.copy(os.path.join(data, name), out)
    with open(os.path.join(out, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out, "merge_log.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "split", "cls", "boxes_merged"])
        w.writeheader()
        w.writerows(log)
    summary = "\n".join(f"- {s}: {a} -> {b} boxes" for s, (a, b) in sorted(counts.items()))
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write(f"# {os.path.basename(out)}\n\n{os.path.basename(data)} with the labelling convention \"one box per "
                f"connected damage region\": same-class boxes within {args.gap:g} px (at the 640 input) of each other "
                f"merged into their union while they cover >= {args.min_fill:g} of it, in every split (tools/merge_regions.py). Merges are listed in merge_log.csv."
                f"\n\n{summary}\n")
    print(summary)


if __name__ == "__main__":
    main()
