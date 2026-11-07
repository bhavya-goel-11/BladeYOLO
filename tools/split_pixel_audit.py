"""Residual-leak audit of our own leak-free splits (research_log.md, G27 pool L finding): for every val and test image of
WTBD-group and WTBlade-group, its best pixel correlation with a training image of the same split (48-px grey,
8 flips/rotations; the Wind Surface twin test of tools/ws_pixel_twins.py, twin >= 0.99). The splits were built with
DINOv2 + ORB/RANSAC grouping, which misses photometric copies (re-exposed, tinted, rotated).

  .venv/bin/python tools/split_pixel_audit.py   -> runs/analysis/split_audit/{report.md, twins.csv, <split>_<sub>.jpg}
"""

import csv
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from g27_pool import dihedral, gray48, unit  # noqa: E402

SPLITS = {"WTBD-group": "data/blade-defect-benchmark/leakfree/WTBD-group", "WTBlade-group": "data/blade-defect-benchmark/leakfree/WTBlade-group"}
TWIN = 0.99
OUT = os.path.join(ROOT, "runs", "analysis", "split_audit")


def best_match(query, train):
    T = unit(np.stack([gray48(p) for p in train]))
    Q = np.stack([gray48(p) for p in query])
    best, arg = np.full(len(query), -1.0, np.float32), np.zeros(len(query), int)
    for v in dihedral(Q):
        s = unit(np.ascontiguousarray(v)) @ T.T
        j = s.argmax(1)
        m = s[np.arange(len(s)), j]
        up = m > best
        best[up], arg[up] = m[up], j[up]
    return best, arg


def main():
    os.makedirs(OUT, exist_ok=True)
    lines = ["# Pixel-twin audit of the leak-free splits (tools/split_pixel_audit.py)", "",
             f"Best correlation of each held-out image with a training image of the same split (48 px grey, 8 "
             f"flips/rotations). Twin: >= {TWIN}. Contact sheets: `<split>_<subset>.jpg` (held-out | train).", "",
             "| split | subset | images | twins >= 0.99 | >= 0.98 | median best |", "|---|---|---|---|---|---|"]
    rows = []
    for name, root in SPLITS.items():
        tr = [os.path.join(root, "train", "images", f) for f in sorted(os.listdir(os.path.join(ROOT, root, "train", "images")))]
        for sub in ("val", "test"):
            q = [os.path.join(root, sub, "images", f) for f in sorted(os.listdir(os.path.join(ROOT, root, sub, "images")))]
            best, arg = best_match(q, tr)
            lines.append(f"| {name} | {sub} | {len(q)} | {(best >= TWIN).sum()} | {(best >= 0.98).sum()} | {np.median(best):.3f} |")
            tiles = []
            for i in np.argsort(-best):
                if best[i] < 0.98:
                    break
                rows.append(dict(split=name, subset=sub, image=os.path.basename(q[i]), train=os.path.basename(tr[arg[i]]),
                                 corr=f"{best[i]:.4f}"))
                if len(tiles) < 24:
                    pair = [cv2.resize(cv2.imread(os.path.join(ROOT, p)), (160, 160)) for p in (q[i], tr[arg[i]])]
                    cv2.putText(pair[0], f"{best[i]:.3f}", (4, 16), 0, 0.5, (0, 0, 255), 2)
                    tiles.append(np.concatenate(pair + [np.full((160, 6, 3), 255, np.uint8)], 1))
            if tiles:
                tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 4)
                cv2.imwrite(os.path.join(OUT, f"{name}_{sub}.jpg"),
                            np.concatenate([np.concatenate(tiles[k:k + 4], 1) for k in range(0, len(tiles), 4)], 0))
    with open(os.path.join(OUT, "twins.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, ["split", "subset", "image", "train", "corr"])
        w.writeheader()
        w.writerows(rows)
    open(os.path.join(OUT, "report.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
