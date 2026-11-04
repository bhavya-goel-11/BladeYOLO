"""Rescore every cached WTBD-group / WTBlade-group test prediction without the held-out images that have a pixel
twin in train (tools/split_pixel_audit.py). Output: runs/analysis/split_audit/rescore.md.
  .venv/bin/python tools/split_rescore.py   (after tools/split_pixel_audit.py)"""
import csv, glob, json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "runs", "analysis", "split_audit")
sys.path.insert(0, os.path.join(ROOT, "tools"))
import bootstrap_ci as bc
tw = list(csv.DictReader(open(os.path.join(OUT, "twins.csv"))))
out = ["# Leak-free test scores with and without the pixel-twin test images", "",
       "mAP50-95 (bootstrap_ci evaluator, conf 0.001 caches). 'clean' drops the test images with a training pixel twin.", ""]
for name, root, nc in (("WTBD-group", "external/WTBD-group", 6), ("WTBlade-group", "external/WTBlade-group", 7)):
    data = os.path.join(ROOT, root)
    imgs = bc.load_yolo_split(data, "test")
    files = sorted(imgs)
    drop = {os.path.join(data, "test", "images", r["image"]) for r in tw if r["split"] == name and r["subset"] == "test" and float(r["corr"]) >= 0.99}
    drop = {os.path.abspath(d) for d in drop}
    assert drop <= set(files), drop - set(files)
    clean = [f for f in files if f not in drop]
    rows = []
    for p in sorted(glob.glob(os.path.join(ROOT, "runs", "analysis", "bootstrap", "**", "*.json"), recursive=True)):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if not isinstance(d, dict) or set(d) != set(files):
            continue
        preds = {k: np.array(v).reshape(-1, 6) for k, v in d.items()}
        m = bc.per_image_matches(imgs, preds, nc)
        a, c = bc.score(m, files, nc), bc.score(m, clean, nc)
        rows.append((os.path.relpath(p, os.path.join(ROOT, "runs", "analysis", "bootstrap")), a[0], a[1], c[0], c[1]))
    d50 = np.array([r[3] - r[1] for r in rows]); d95 = np.array([r[4] - r[2] for r in rows])
    out += [f"## {name}: {len(files)} test images, {len(drop)} with a training pixel twin; {len(rows)} cached runs", "",
            f"Δ (clean − all), percent points: mAP50 mean {100*d50.mean():+.2f} [min {100*d50.min():+.2f}, max {100*d50.max():+.2f}]; "
            f"mAP50-95 mean {100*d95.mean():+.2f} [min {100*d95.min():+.2f}, max {100*d95.max():+.2f}]", "",
            "| run | all mAP50 / 50-95 | clean mAP50 / 50-95 |", "|---|---|---|"]
    out += [f"| {r[0]} | {100*r[1]:.1f} / {100*r[2]:.1f} | {100*r[3]:.1f} / {100*r[4]:.1f} |" for r in rows]
    out.append("")
open(os.path.join(OUT, "rescore.md"), "w").write("\n".join(out) + "\n")
print("\n".join(l for l in out if not l.startswith("| ") or "group.json" in l or "wtblade_group" in l[:60]))
