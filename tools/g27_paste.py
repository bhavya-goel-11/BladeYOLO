"""G27 recipe component (research_log.md): class-balancing copy-paste of real defects, the matched copy-paste of H10
(+0.037 mAP50-95 for ViT-B at 25%) generalised to every class and every protocol's full training set.

Rule (fixed before any G27 run): in each protocol's training set, every class with fewer boxes than the median class
count gets min(median - n, n) extra boxes, i.e. up to the median and at most doubled. Each new image is a training
image with up to 2 extra boxes of one class, pasted from real boxes of that class in the same training set (resized
x0.8-1.2, feathered edge), centred on a blade pixel (SAM 2.1-tiny blade mask prompted at the image's labelled box
centres, tools/synth_defects.blade_masks), >= 80% on the blade, overlapping no box. Only training images are read,
so validation and test stay untouched; plans are made locally and committed, the paste itself runs where the data is.

  .venv/bin/python tools/g27_paste.py plan --protocol wtbd_group          -> tools/splits/g27_cp_wtbd_group.csv
  python tools/g27_paste.py paste --protocol wtbd_group --dataset <prepared rfdetr dataset>
     (adds cp_*.jpg + labels to <dataset>/train/; run after rfdetr_v1.py prepare-official, before train)
"""

import argparse
import collections
import csv
import os
import random
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import synth_defects as sd  # noqa: E402

PROTOCOLS = {"wtbd_group": "external/WTBD-group", "wtbd_official": "external/WTBD-official",
             "wtblade_group": "external/WTBlade-group", "wtblade_defect": "WTBlade-Defect",
             "windsurface": "WindSurface-Defect"}
SAM = os.path.join(ROOT, "runs", "analysis", "sam_refine", "sam2.1_t.pt")
PER_IMAGE, MIN_BLADE, TRIES = 2, 0.8, 300
FIELDS = ["i", "image", "cls", "cx", "cy", "w", "h", "src_image", "src_cx", "src_cy", "src_w", "src_h"]


def plan_path(protocol):
    return os.path.join(ROOT, "tools", "splits", f"g27_cp_{protocol}.csv")


def train_names(protocol):
    """Training images as the rfdetr_v1 prepared dataset has them (Wind Surface: minus the seeded 10% carve)."""
    d = os.path.join(ROOT, PROTOCOLS[protocol], "train", "images")
    names = sorted(os.listdir(d))
    if protocol == "windsurface":
        held = set(random.Random(0).sample(names, round(0.1 * len(names))))
        names = [n for n in names if n not in held]
    return names, d, os.path.join(os.path.dirname(d), "labels")


def deficits(labels, nc):
    n = np.bincount([int(b[0]) for v in labels.values() for b in v], minlength=nc)
    med = float(np.median(n))
    return n, med, {c: int(min(med - n[c], n[c])) for c in range(nc) if n[c] < med}


class Masks:
    """synth_defects.blade_masks with one SAM predictor kept loaded (same prompts, same PNG cache)."""

    def __init__(self, img_dir, cache):
        self.img_dir, self.cache, self.sam = img_dir, cache, None
        os.makedirs(cache, exist_ok=True)

    def __call__(self, m, labels):
        p = os.path.join(self.cache, os.path.splitext(m)[0] + ".png")
        if not os.path.exists(p):
            if self.sam is None:
                from ultralytics.models.sam import SAM2Predictor

                self.sam = SAM2Predictor(overrides=dict(task="segment", mode="predict", imgsz=1024, model=SAM, conf=0.0,
                                                        save=False, verbose=False, device="cpu"))
                self.sam.args.conf = -1.0
            f = os.path.join(self.img_dir, m)
            H, W = cv2.imread(f).shape[:2]
            self.sam.set_image(f)
            blade = np.zeros((H, W), bool)
            for _, x, y, _, _ in labels:
                k = self.sam(points=[[x * W, y * H]], labels=[1], multimask_output=True)[0].masks.data.cpu().numpy() > 0.5
                blade |= k[k.reshape(len(k), -1).sum(1).argmax()]
            self.sam.reset_image()
            cv2.imwrite(p, cv2.resize(blade.astype(np.uint8) * 255, (sd.MASK_SIDE, sd.MASK_SIDE), interpolation=cv2.INTER_AREA))
        return cv2.imread(p, cv2.IMREAD_GRAYSCALE) > 127


def plan(protocol, seed=0):
    import yaml

    names, img_dir, lab_dir = train_names(protocol)
    nc = len(yaml.safe_load(open(os.path.join(ROOT, PROTOCOLS[protocol], "data.yaml")))["names"])
    labels = {m: sd.read_labels(lab_dir, m) for m in names}
    n, med, need = deficits(labels, nc)
    print(f"{protocol}: boxes per class {n.tolist()}, median {med}, extra {need}", flush=True)
    rng = np.random.default_rng(seed)
    pool = {c: [(m, b) for m in names for b in labels[m] if int(b[0]) == c] for c in need}
    order = list(rng.permutation([m for m in names if len(labels[m])]))
    masks = Masks(img_dir, os.path.join(ROOT, "runs", "analysis", "g27", "blade_masks", protocol))
    rows, i, k = [], 0, 0
    todo = [c for c in sorted(need) for _ in range(need[c])]
    todo = [todo[j:j + PER_IMAGE] for j in range(0, len(todo), PER_IMAGE)]
    todo = [g for g in todo if len(set(g)) == 1] + [[c] for g in todo if len(set(g)) > 1 for c in g]  # one class per image
    for group in todo:
        while True:
            assert k < len(order) * 3, f"ran out of target images at new image {i}"
            img = order[k % len(order)]
            k += 1
            mask = masks(img, labels[img])
            if not mask.any():
                continue
            blade = (np.argwhere(mask)[:, ::-1] + 0.5) / sd.MASK_SIDE
            placed, have = [], labels[img]
            for c in group:
                for _ in range(TRIES):
                    src, sb = pool[c][rng.integers(len(pool[c]))]
                    s = rng.uniform(0.8, 1.2)
                    w, h = min(sb[3] * s, 0.95), min(sb[4] * s, 0.95)
                    cx, cy = blade[rng.integers(len(blade))]
                    cx, cy = float(np.clip(cx, w / 2, 1 - w / 2)), float(np.clip(cy, h / 2, 1 - h / 2))
                    if not sd.overlaps((cx, cy, w, h), have) and sd.on_blade((cx, cy, w, h), mask) >= MIN_BLADE:
                        placed.append(dict(i=i, image=img, cls=c, cx=round(cx, 5), cy=round(cy, 5), w=round(w, 5),
                                           h=round(h, 5), src_image=src, src_cx=sb[1], src_cy=sb[2], src_w=sb[3], src_h=sb[4]))
                        have = np.vstack([have, [c, cx, cy, w, h]])
                        break
            if len(placed) == len(group):
                rows += placed
                i += 1
                break
        if i % 50 == 0:
            print(f"  {protocol}: {i}/{len(todo)} new images", flush=True)
    with open(plan_path(protocol), "w", newline="") as f:
        wr = csv.DictWriter(f, FIELDS)
        wr.writeheader()
        wr.writerows(rows)
    got = collections.Counter(int(r["cls"]) for r in rows)
    assert got == collections.Counter({c: v for c, v in need.items() if v > 0}), (got, need)
    print(f"{protocol}: {i} new images, {len(rows)} boxes {dict(sorted(got.items()))} -> {plan_path(protocol)}")


def paste(protocol, dataset):
    img_dir, lab_dir = os.path.join(dataset, "train", "images"), os.path.join(dataset, "train", "labels")
    rows = list(csv.DictReader(open(plan_path(protocol))))
    by_i = collections.defaultdict(list)
    for r in rows:
        by_i[int(r["i"])].append(r)
    have = set(os.listdir(img_dir))
    miss = {r[k] for r in rows for k in ("image", "src_image")} - have
    assert not miss, f"{len(miss)} plan images not in {img_dir}, e.g. {sorted(miss)[:3]}"
    assert not any(f.startswith("cp_") for f in have), f"{img_dir} already has pasted images"
    for i, rs in sorted(by_i.items()):
        img = cv2.imread(os.path.join(img_dir, rs[0]["image"]))
        H, W = img.shape[:2]
        for r in rs:
            box = sd.box_px(float(r["cx"]), float(r["cy"]), float(r["w"]), float(r["h"]), H, W)
            src = cv2.imread(os.path.join(img_dir, r["src_image"]))
            s = sd.box_px(*(float(r[k]) for k in ("src_cx", "src_cy", "src_w", "src_h")), *src.shape[:2])
            patch = cv2.resize(src[s[1]:s[3], s[0]:s[2]], (box[2] - box[0], box[3] - box[1]), interpolation=cv2.INTER_AREA)
            img = sd.blend(img, patch, box)
        stem = f"cp_{i:05d}__{os.path.splitext(rs[0]['image'])[0]}"
        cv2.imwrite(os.path.join(img_dir, stem + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        lab = sd.read_labels(lab_dir, rs[0]["image"]).tolist() + [[int(r["cls"]), *(float(r[k]) for k in ("cx", "cy", "w", "h"))] for r in rs]
        with open(os.path.join(lab_dir, stem + ".txt"), "w") as f:
            f.write("".join(f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in lab))
    print(f"{protocol}: {len(by_i)} pasted images ({len(rows)} boxes) -> {img_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("plan")
    a.add_argument("--protocol", required=True, choices=list(PROTOCOLS) + ["all"])
    a = sp.add_parser("paste")
    a.add_argument("--protocol", required=True, choices=PROTOCOLS)
    a.add_argument("--dataset", required=True)
    args = ap.parse_args()
    if args.cmd == "plan":
        for p in (PROTOCOLS if args.protocol == "all" else [args.protocol]):
            plan(p)
    else:
        paste(args.protocol, args.dataset)


if __name__ == "__main__":
    main()
