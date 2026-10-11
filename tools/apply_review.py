"""Apply label-review decisions (tools/review_labels.py) to WindSurface-Defect-v3, producing v3.1.

Only the reviewer's decisions change labels:
  unlabelled candidate  "damage: <class>"   -> train: box added (the predicted box) with that class
                                               val/test: ignore region (ignore_regions.json): the box was drawn by
                                               one of the evaluated models, so as a label it would hand that model a
                                               free match; tools/error_analysis.py neither rewards nor penalises
                                               predictions there
                        "damage, unclear"   -> --unclear skip (default): not added, counted; predicted: added as the
                                               model's class
                        no damage / not on a blade / unsure -> nothing
  existing label        keep / unsure / "damage, unclear" -> unchanged
                        "damage: <class>"   -> class changed (unchanged if it is the labelled class)
                        no damage / not on a blade -> removed
Images are hard-linked from v3 (no extra disk space); images left without boxes stay as background images.
Splits without a review file are copied unchanged. Every change is logged in <out>/review_changes.csv.

Usage:
    python tools/apply_review.py --out WindSurface-Defect-v3.1
"""

import argparse
import collections
import csv
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_labels(path):
    return [[int(v[0])] + [float(x) for x in v[1:]] for v in (l.split() for l in open(path)) if len(v) == 5]


def to_yolo(box, W, H):
    """640-input pixel box (x1, y1, x2, y2) -> normalised (x, y, w, h) of the original W x H image."""
    s = 640 / max(W, H)
    x1, y1, x2, y2 = (max(0.0, box[0] / (W * s)), max(0.0, box[1] / (H * s)), min(1.0, box[2] / (W * s)), min(1.0, box[3] / (H * s)))
    return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]


def iou_xywh(a, b):
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-12)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(ROOT, "WindSurface-Defect-v3"))
    ap.add_argument("--reviews", default=os.path.join(ROOT, "runs", "analysis", "review"),
                    help="folder holding <split>/review_<split>.json")
    ap.add_argument("--out", default=os.path.join(ROOT, "WindSurface-Defect-v3.1"))
    ap.add_argument("--unclear", choices=("skip", "predicted"), default="skip")
    args = ap.parse_args()

    data, out = os.path.abspath(args.data), os.path.abspath(args.out)
    if os.path.exists(out):
        shutil.rmtree(out)
    rows = list(csv.DictReader(open(os.path.join(data, "manifest.csv"))))
    size = {r["file"]: (int(r["width"]), int(r["height"])) for r in rows}
    labels = {}
    for r in rows:
        stem = os.path.splitext(os.path.basename(r["file"]))[0]
        labels[r["file"]] = read_labels(os.path.join(data, "labels", r["split"], stem + ".txt"))

    log, stats, ignore = [], collections.Counter(), collections.defaultdict(list)
    for split in ("train", "val", "test"):
        path = os.path.join(args.reviews, split, f"review_{split}.json")
        if not os.path.exists(path):
            print(f"{split}: no review file, unchanged")
            continue
        rev = json.load(open(path))
        names = rev["names"]
        for d in rev["decisions"]:
            f = d["image"]
            W, H = size[f]
            box = to_yolo(d["box"], W, H)
            dec = d["decision"]
            new_cls = int(dec.split(":")[1]) if dec.startswith("cls:") else None
            stats[f"{split} {d['kind']} {dec.split(':')[0]}"] += 1
            if d["kind"] == "unlabelled":
                if new_cls is None and dec == "damage" and args.unclear == "predicted":
                    new_cls = d["cls"]
                if new_cls is not None and split == "train":
                    labels[f].append([new_cls] + box)
                    log.append((split, f, "added", "", names[new_cls], d["id"], d.get("note", "")))
                elif new_cls is not None:
                    ignore[f].append(d["box"] + [new_cls])
                    log.append((split, f, "ignore region", "", names[new_cls], d["id"], d.get("note", "")))
                continue
            # existing label: find it by its box
            j = max(range(len(labels[f])), key=lambda j: iou_xywh(labels[f][j][1:], box), default=None)
            if j is None or labels[f][j][0] != d["cls"] or iou_xywh(labels[f][j][1:], box) < 0.98:
                raise SystemExit(f"{split} item {d['id']}: label not found in {f} (was the dataset rebuilt?)")
            old = names[labels[f][j][0]]
            if dec in ("none", "off_blade"):
                labels[f].pop(j)
                log.append((split, f, "removed" if dec == "none" else "removed (off blade)", old, "", d["id"], d.get("note", "")))
            elif new_cls is not None and new_cls != labels[f][j][0]:
                labels[f][j][0] = new_cls
                log.append((split, f, "class changed", old, names[new_cls], d["id"], d.get("note", "")))

    # write the dataset: hard-linked images, rewritten labels, same manifest
    for r in rows:
        stem = os.path.splitext(os.path.basename(r["file"]))[0]
        img = os.path.join(out, r["file"])
        os.makedirs(os.path.dirname(img), exist_ok=True)
        os.makedirs(os.path.join(out, "labels", r["split"]), exist_ok=True)
        try:
            os.link(os.path.join(data, r["file"]), img)
        except OSError:
            shutil.copy2(os.path.join(data, r["file"]), img)
        with open(os.path.join(out, "labels", r["split"], stem + ".txt"), "w") as fh:
            fh.writelines(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in labels[r["file"]])
    for f in ("manifest.csv", "data.yaml"):
        shutil.copy2(os.path.join(data, f), os.path.join(out, f))
    json.dump(ignore, open(os.path.join(out, "ignore_regions.json"), "w"), indent=1)  # [x1, y1, x2, y2] at 640 px, class
    with open(os.path.join(out, "review_changes.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "image", "change", "old_class", "new_class", "review_item", "note"])
        w.writerows(log)

    import yaml

    names = yaml.safe_load(open(os.path.join(data, "data.yaml")))["names"]
    counts = {s: collections.Counter() for s in ("train", "val", "test")}
    for r in rows:
        for b in labels[r["file"]]:
            counts[r["split"]][b[0]] += 1
    changes = collections.Counter((s, c) for s, _, c, *_ in log)
    md = ["# Wind Surface Defect v3.1", "",
          "WindSurface-Defect-v3 (same images, same split) with labels corrected by a human review of model-flagged",
          "candidates (`tools/review_labels.py`, `tools/apply_review.py`). Candidates were pooled from several models;",
          "only reviewer decisions changed labels. Every change is listed in `review_changes.csv`.", "",
          f"Unclear-class damage: {args.unclear}.", "",
          "| split | " + " | ".join(sorted({c for _, c in changes})) + " |", "|---|" + "---|" * len({c for _, c in changes})]
    for s in ("train", "val", "test"):
        md.append(f"| {s} | " + " | ".join(str(changes[(s, c)]) for c in sorted({c for _, c in changes})) + " |")
    md += ["", "Review decisions:", ""] + [f"- {k}: {v}" for k, v in sorted(stats.items())]
    md += ["", "| class | train | val | test |", "|---|---|---|---|"]
    md += [f"| {n} | {counts['train'][i]} | {counts['val'][i]} | {counts['test'][i]} |" for i, n in enumerate(names)]
    open(os.path.join(out, "README.md"), "w").write("\n".join(md) + "\n\nOriginal v3 README:\n\n" + open(os.path.join(data, "README.md")).read())
    print("\n".join(md))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
