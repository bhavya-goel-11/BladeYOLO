"""Side-by-side crops of one class from each source dataset of WindSurface-Defect-v3, plus box statistics.

Usage:
    python tools/class_crops.py corrosion                                   # WindSurface-Defect-v3, by source
    python tools/class_crops.py surface_oil --data wind-turbine.v1i.yolov12  # any raw YOLO export
Writes runs/analysis/<class>_crops.png (crops with context) and <class>_scenes.png (whole images with boxes).
"""

import argparse
import csv
import os
import random
import statistics

from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCES = {"ws": "Wind Surface Defect", "wt": "WTBlade-Defect", "bj": "Beijing wind-turbine"}


def boxes_by_source(data, cls_id):
    if not os.path.exists(os.path.join(data, "manifest.csv")):
        return {os.path.basename(os.path.normpath(data)): raw_yolo_boxes(data, cls_id)}
    out = {s: [] for s in SOURCES}
    for r in csv.DictReader(open(os.path.join(data, "manifest.csv"))):
        stem = os.path.splitext(os.path.basename(r["file"]))[0]
        for line in open(os.path.join(data, "labels", r["split"], stem + ".txt")):
            c, x, y, w, h = line.split()
            if int(c) == cls_id:
                out[r["source"]].append(dict(img=os.path.join(data, r["file"]), group=r["group"], split=r["split"],
                                             box=tuple(map(float, (x, y, w, h)))))
    return out


def raw_yolo_boxes(data, cls_id):
    """Any Roboflow-style YOLO export (<split>/images, <split>/labels); groups = parent photo from tile names."""
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from build_dataset import photo_key, source_stem

    out = []
    for split in sorted(os.listdir(data)):
        img_dir = os.path.join(data, split, "images")
        if not os.path.isdir(img_dir):
            continue
        for f in sorted(os.listdir(img_dir)):
            stem = os.path.splitext(f)[0]
            src = source_stem(stem)
            for line in open(os.path.join(data, split, "labels", stem + ".txt")):
                v = line.split()
                if len(v) == 5 and int(v[0]) == cls_id:
                    out.append(dict(img=os.path.join(img_dir, f), group=photo_key(src) or src, split=split,
                                    box=tuple(map(float, v[1:]))))
    return out


def sample_diverse(boxes, n, rng):
    """At most one box per split group first, so the sheet is not ten crops of one photo."""
    by_group = {}
    for b in rng.sample(boxes, len(boxes)):
        by_group.setdefault(b["group"], []).append(b)
    picked = [g[0] for g in by_group.values()][:n]
    rest = [b for g in by_group.values() for b in g[1:]]
    return picked + rest[: n - len(picked)]


def crop(b, size, pad=1.5):
    im = Image.open(b["img"]).convert("RGB")
    W, H = im.size
    x, y, w, h = b["box"]
    x1, y1 = (x - w / 2 - w * pad) * W, (y - h / 2 - h * pad) * H
    x2, y2 = (x + w / 2 + w * pad) * W, (y + h / 2 + h * pad) * H
    c = im.crop((max(0, x1), max(0, y1), min(W, x2), min(H, y2)))
    scale = size / max(c.size)  # enlarge tiny defects too, so a 6-px pinhole is visible
    c = c.resize((max(1, round(c.width * scale)), max(1, round(c.height * scale))), Image.NEAREST if scale > 4 else Image.BICUBIC)
    tile = Image.new("RGB", (size, size), (255, 255, 255))
    tile.paste(c, ((size - c.width) // 2, (size - c.height) // 2))
    return tile


def scene(b, size):
    im = Image.open(b["img"]).convert("RGB")
    W, H = im.size
    d = ImageDraw.Draw(im)
    x, y, w, h = b["box"]
    d.rectangle(((x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H), outline=(255, 0, 0), width=max(2, W // 200))
    im.thumbnail((size, size))
    return im


def sheet(rows, size, title):
    pad, label_h = 6, 24
    width = max(len(r[1]) for r in rows) * (size + pad) + pad
    S = Image.new("RGB", (width, len(rows) * (size + label_h + pad) + 30), (255, 255, 255))
    d = ImageDraw.Draw(S)
    d.text((pad, 8), title, fill=(0, 0, 0))
    y = 30
    for label, tiles in rows:
        d.text((pad, y + 4), label, fill=(0, 0, 0))
        for i, t in enumerate(tiles):
            S.paste(t, (pad + i * (size + pad), y + label_h))
        y += size + label_h + pad
    return S


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cls")
    ap.add_argument("--data", default=os.path.join(ROOT, "WindSurface-Defect-v3"))
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-px", type=float, default=0, help="only boxes whose shorter side is at least this many pixels")
    args = ap.parse_args()

    import yaml

    names = yaml.safe_load(open(os.path.join(args.data, "data.yaml")))["names"]
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else names  # YOLO allows {id: name}
    boxes = boxes_by_source(args.data, names.index(args.cls))
    if args.min_px:
        def big(b):
            W, H = Image.open(b["img"]).size
            return min(b["box"][2] * W, b["box"][3] * H) >= args.min_px
        boxes = {s: [b for b in bs if big(b)] for s, bs in boxes.items()}
    rng = random.Random(args.seed)
    out = os.path.join(ROOT, "runs", "analysis")
    os.makedirs(out, exist_ok=True)

    crop_rows, scene_rows = [], []
    for s, bs in boxes.items():
        label = SOURCES.get(s, s)
        if not bs:
            continue
        pick = sample_diverse(bs, args.n, rng)
        areas = [b["box"][2] * b["box"][3] * 100 for b in bs]
        aspects = [max(b["box"][2] / b["box"][3], b["box"][3] / b["box"][2]) for b in bs if b["box"][3] > 0]
        imgs = {b["img"] for b in bs}
        stats = (f"{label}: {len(bs)} boxes in {len(imgs)} images, {len({b['group'] for b in bs})} groups | "
                 f"box area % of image: median {statistics.median(areas):.2f}, p90 {sorted(areas)[int(len(areas) * .9)]:.2f} | "
                 f"boxes/image {len(bs) / len(imgs):.1f} | elongation median {statistics.median(aspects):.1f}")
        print(stats)
        half = (len(pick) + 1) // 2
        crop_rows += [(f"{label} ({len(bs)} boxes)", [crop(b, 150) for b in pick[:half]]),
                      (f"{label} (cont.)", [crop(b, 150) for b in pick[half:]])]
        scene_rows.append((label, [scene(b, 220) for b in pick[:6]]))
    tag = (args.cls + (f"_min{args.min_px:g}px" if args.min_px else "")) if os.path.exists(os.path.join(args.data, "manifest.csv")) else f"{os.path.basename(os.path.normpath(args.data))}_{args.cls}" + (f"_min{args.min_px:g}px" if args.min_px else "")
    sheet(crop_rows, 150, f"'{args.cls}' crops (box + 150% context, enlarged) by source").save(os.path.join(out, f"{tag}_crops.png"))
    sheet(scene_rows, 220, f"'{args.cls}' boxes (red) in whole images by source").save(os.path.join(out, f"{tag}_scenes.png"))
    print("wrote", os.path.join(out, f"{tag}_crops.png"), "and", os.path.join(out, f"{tag}_scenes.png"))


if __name__ == "__main__":
    main()
