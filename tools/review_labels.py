"""Build a local, keyboard-driven review page for auditing WindSurface-Defect-v3 labels.

Candidates come from the predictions of every given run (tools/error_analysis.py caches them), pooled so no model
is favoured. Only the reviewer's decisions change labels; nothing is accepted automatically.
  unlabelled  confident prediction (>= --conf) with no label nearby (IoU < 0.1): real damage that was never labelled?
  missed      label that no run found (no same- or other-class prediction with IoU >= 0.1): blank box or real miss?
  class       label that a run found with IoU >= 0.5 but as another class: which class is right?

Usage:
    python tools/review_labels.py runs/detect/BladeYOLO_WindSurface/yolo12l_v3 runs/detect/BladeYOLO_WindSurface/bladeyolo_v3 --split test
Answers per item: damage of class 1-6, damage of unclear class, no damage, not on a blade, keep as labelled, unsure.
Open runs/analysis/review/<split>/index.html in a browser. Decisions autosave in the browser and are exported with
the "Export" button as review_<split>.json (keep it next to index.html).
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import error_analysis as E  # noqa: E402


def dedupe(items, thr=0.5):
    """Merge candidates on the same image whose boxes overlap (IoU >= thr); keeps the highest confidence."""
    out = []
    for it in sorted(items, key=lambda it: -it["conf"]):
        if not any(o["img"] == it["img"] and E.iou_matrix(np.array([o["box"]]), np.array([it["box"]]))[0, 0] >= thr for o in out):
            out.append(it)
    return out


def render(it, gt, out_dir, idx):
    """Crop with context (enlarged) and the whole image, both with the item's box (yellow) and other labels (green)."""
    from PIL import Image, ImageDraw

    im = Image.open(it["img"]).convert("RGB")
    s = 640 / max(im.size)
    full = im.resize((round(im.width * s), round(im.height * s)))
    d = ImageDraw.Draw(full)
    for g in gt:
        if list(g[1:]) != it["box"]:
            d.rectangle(tuple(g[1:]), outline=(0, 220, 0), width=1)
    b = it["box"]
    d.rectangle(tuple(b), outline=(255, 230, 0), width=3)
    w, h = b[2] - b[0], b[3] - b[1]
    pad = max(w, h) * 1.0 + 24
    x1, y1, x2, y2 = max(0, b[0] - pad), max(0, b[1] - pad), min(full.width, b[2] + pad), min(full.height, b[3] + pad)
    # crop from an undrawn copy, then draw a thin box so the defect itself stays visible
    c = im.resize(full.size).crop((x1, y1, x2, y2))
    k = 420 / max(c.size)
    c = c.resize((round(c.width * k), round(c.height * k)), Image.LANCZOS)
    ImageDraw.Draw(c).rectangle(((b[0] - x1) * k, (b[1] - y1) * k, (b[2] - x1) * k, (b[3] - y1) * k), outline=(255, 230, 0), width=2)
    crop_p, full_p = f"crops/{idx:04d}_c.jpg", f"crops/{idx:04d}_f.jpg"
    c.save(os.path.join(out_dir, crop_p), quality=92)
    full.save(os.path.join(out_dir, full_p), quality=85)
    return crop_p, full_p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--data", default=os.path.join(ROOT, "WindSurface-Defect-v3"))
    ap.add_argument("--conf", type=float, default=0.2, help="confidence for unlabelled-damage candidates")
    args = ap.parse_args()

    import yaml

    data = os.path.abspath(args.data)
    names = yaml.safe_load(open(os.path.join(data, "data.yaml")))["names"]
    imgs = E.load_split(data, args.split, names)
    preds = {}
    for r in args.runs:
        p = os.path.join(ROOT, "runs", "analysis", "errors", os.path.basename(os.path.normpath(r)), f"{args.split}_preds.json")
        if not os.path.exists(p):
            sys.exit(f"missing {p}: run tools/error_analysis.py on this split first")
        preds[r] = json.load(open(p))

    unl, gt_status = [], {}
    for r, P in preds.items():
        gts, ps = E.classify(imgs, P, args.conf, len(names))
        unl += [dict(img=p["img"], box=p["box"], conf=p["conf"], cls=p["cls"]) for p in ps if p["type"] == "bkg"]
        for g in gts:
            gt_status.setdefault((g["img"], tuple(g["box"])), []).append(g)
    items = [dict(it, kind="unlabelled") for it in dedupe(unl)]
    for (f, box), gs in gt_status.items():
        if all(g["type"] == "missed" for g in gs):
            items.append(dict(img=f, box=list(box), cls=gs[0]["cls"], conf=0.0, kind="missed"))
        elif any(g["type"] == "cls" for g in gs) and not any(g["type"] == "tp" for g in gs):
            pc = next(g["pred_cls"] for g in gs if g["type"] == "cls")
            items.append(dict(img=f, box=list(box), cls=gs[0]["cls"], pred_cls=pc, conf=0.0, kind="class"))
    items.sort(key=lambda it: (it["kind"], it["img"], it["box"]))

    out = os.path.join(ROOT, "runs", "analysis", "review", args.split)
    os.makedirs(os.path.join(out, "crops"), exist_ok=True)
    rows = []
    for i, it in enumerate(items):
        crop_p, full_p = render(it, imgs[it["img"]]["gt"], out, i)
        rows.append(dict(id=i, kind=it["kind"], image=os.path.relpath(it["img"], data), box=[round(v, 1) for v in it["box"]],
                         cls=it["cls"], pred_cls=it.get("pred_cls"), conf=round(it["conf"], 3), source=imgs[it["img"]]["source"],
                         size=round(E.size_of(it["box"])), crop=crop_p, full=full_p))
    json.dump(dict(split=args.split, names=names, conf=args.conf, runs=[os.path.basename(os.path.normpath(r)) for r in args.runs],
                   items=rows), open(os.path.join(out, "items.json"), "w"), indent=1)
    page = open(os.path.join(ROOT, "tools", "review_page.html")).read()
    page = page.replace("/*ITEMS*/null", json.dumps(dict(split=args.split, names=names, items=rows)))
    open(os.path.join(out, "index.html"), "w").write(page)
    counts = {k: sum(r["kind"] == k for r in rows) for k in ("unlabelled", "missed", "class")}
    print(f"{len(rows)} items {counts} -> {os.path.join(out, 'index.html')}")


if __name__ == "__main__":
    main()
