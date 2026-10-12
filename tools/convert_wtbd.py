"""Convert WTBD (Ji et al., Scientific Data 13:396, 2026; figshare 10.6084/m9.figshare.30210175, CC BY 4.0) to YOLO.

Source: <src>/JPEGImages/<i>.jpg, <src>/Annotations/<i>.xml (PASCAL VOC, primary annotator) and
<src>/train_val_test_split.txt (the authors' split: a seeded random shuffle of image indices, 70/15/15).
Output: <out>/{train,val,test}/{images,labels}, data.yaml. Images are hard-linked. --split-file replaces the
authors' split, e.g. with tools/splits/wtbd_group_split.csv (tools/wtbd_groups.py), where augmented copies and other
views of one defect stay in one split (the authors' random split puts 122 of 161 test images' twins in train).

Usage:
    python tools/convert_wtbd.py "external/WTBD/WT blade defect dataset" --out WTBD-group \
        --split-file tools/splits/wtbd_group_split.csv
"""

import argparse
import csv
import json
import os
import xml.etree.ElementTree as ET

CLASSES = ["craze", "corrosion", "surface_injure", "thunderstrike", "crack", "hide_craze"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split-file", default=None, help="CSV ImageID,Subset (default: the authors' split)")
    args = ap.parse_args()

    split_file = args.split_file or os.path.join(args.src, "train_val_test_split.txt")
    split = {r["ImageID"]: r["Subset"] for r in csv.DictReader(open(split_file))}
    out = os.path.abspath(args.out)
    counts = {}
    for name, sub in sorted(split.items(), key=lambda kv: int(kv[0].split(".")[0])):
        stem = os.path.splitext(name)[0]
        root = ET.parse(os.path.join(args.src, "Annotations", stem + ".xml")).getroot()
        W, H = int(root.find("size/width").text), int(root.find("size/height").text)
        rows = []
        for o in root.iter("object"):
            b = o.find("bndbox")
            x1, y1, x2, y2 = (float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            rows.append(f"{CLASSES.index(o.find('name').text.strip())} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                        f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n")
        for d in ("images", "labels"):
            os.makedirs(os.path.join(out, sub, d), exist_ok=True)
        dst = os.path.join(out, sub, "images", name)
        if not os.path.exists(dst):
            os.link(os.path.realpath(os.path.join(args.src, "JPEGImages", name)), dst)
        open(os.path.join(out, sub, "labels", stem + ".txt"), "w").writelines(rows)
        c = counts.setdefault(sub, [0, 0])
        c[0], c[1] = c[0] + 1, c[1] + len(rows)
    with open(os.path.join(out, "data.yaml"), "w") as f:
        f.write(f"path: .\ntrain: train/images\nval: val/images\ntest: test/images\nnc: {len(CLASSES)}\n"
                f"names: {json.dumps(CLASSES)}\n")
    print({k: dict(images=a, boxes=b) for k, (a, b) in counts.items()})


if __name__ == "__main__":
    main()
