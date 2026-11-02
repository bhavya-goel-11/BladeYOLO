"""G25 (research_log.md): RF-DETR-M on the official (published, leaky) protocols, scored with the bootstrap-ci COCO
evaluator, next to the numbers BladeYOLO (IEEE TGRS 2026) and our earlier runs report on the same protocols.

  WTBD-official   authors' split, test 161           rfdetrm640_wtbd_official, rfdetrm768_wtbd_official
  Wind Surface    public copy, val 714 = test (4:1)  rfdetrm640_windsurface_official
  WTBlade-Defect  Roboflow split, test 553           rfdetrm640_wtblade_defect_official

Image-level bootstrap (official splits have no scene groups; 1,000 replicates) for intervals. Predictions are the
Kaggle ones (conf 0.001, 640-input pixels, keyed by file name).
  .venv/bin/python tools/g25_official.py -> runs/analysis/g25/report.md
"""

import csv
import json
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import bootstrap_ci as bc  # noqa: E402

RUNS = os.path.join(ROOT, "runs", "detect", "BladeYOLO_WindSurface")
OUT = os.path.join(ROOT, "runs", "analysis", "g25")
N_BOOT = 1000
SETS = [  # name, data dir, test folder, runs, published references (mAP50, mAP50-95; percent)
    ("WTBD-official", os.path.join(ROOT, "external", "WTBD-official"), "test",
     ["rfdetrm640_wtbd_official", "rfdetrm768_wtbd_official"],
     [("BladeYOLO-v2 (ours, Ultralytics eval, §12)", 84.0, 49.3), ("YOLO12-L (ours, §12)", 70.4, 38.8)]),
    ("Wind Surface Defect", os.path.join(ROOT, "WindSurface-Defect"), "val",
     ["rfdetrm640_windsurface_official"],
     [("BladeYOLO (paper, Table V)", 86.5, 57.4), ("BladeYOLO-L, our best leaky reproduction (§2, Run 6, val)", 78.2, 46.9)]),
    ("WTBlade-Defect", os.path.join(ROOT, "WTBlade-Defect"), "test",
     ["rfdetrm640_wtblade_defect_official"],
     [("BladeYOLO (paper, Table I, 3 seeds; its own 9:1 re-split of 5,326 images)", 85.8, 74.6),
      ("YOLOv12-L (paper, Table III baseline; same re-split)", 83.3, 71.9)]),
]


def names_of(data):
    n = yaml.safe_load(open(os.path.join(data, "data.yaml")))["names"]
    return list(n.values()) if isinstance(n, dict) else list(n)


def last_val(run):
    rows = [r for r in csv.DictReader(open(os.path.join(RUNS, run, "metrics.csv"))) if r.get("val/mAP_50_95")]
    best = max(rows, key=lambda r: float(r["val/mAP_50_95"]))
    return float(best["val/mAP_50"]), float(best["val/mAP_50_95"]), int(best["epoch"])


def main():
    os.makedirs(OUT, exist_ok=True)
    lines = ["# G25: RF-DETR-M on the official (published, leaky) protocols (research_log.md)", "",
             "COCO 101-point AP (tools/bootstrap_ci.py), predictions at conf 0.001; image-level bootstrap 95% CI "
             f"({N_BOOT} replicates). Published numbers are copied from the papers / earlier sections, not re-run.", ""]
    res = {}
    for name, data, split, runs, refs in SETS:
        names = names_of(data)
        imgs = bc.load_yolo_split(data, split)
        files = sorted(imgs)
        lines += [f"## {name} ({len(files)} test images, {sum(len(v['gt']) for v in imgs.values())} boxes)", "",
                  "| model | mAP50 | mAP50-95 | source |", "|---|---|---|---|"]
        rng = np.random.default_rng(0)
        draws = [rng.integers(0, len(files), len(files)) for _ in range(N_BOOT)]
        for run in runs:
            raw = json.load(open(os.path.join(RUNS, run, f"preds_test.json")))
            preds = {os.path.join(data, split, "images", k): np.array(v).reshape(-1, 6) for k, v in raw.items()}
            preds = {os.path.abspath(k): v for k, v in preds.items()}
            missing = [f for f in files if f not in preds]
            assert not missing, f"{run}: {len(missing)} test images without predictions, e.g. {missing[:2]}"
            m = bc.per_image_matches(imgs, preds, len(names))
            m50, m5095, pc50, pc = bc.score(m, files, len(names))
            bs = np.array([bc.score(m, [files[i] for i in d], len(names))[:2] for d in draws])
            v50, v5095, ep = last_val(run)
            res[run] = dict(map50=m50, map5095=m5095, pc50=dict(zip(names, map(float, pc50))))
            lines.append(f"| **{run}** | **{100 * m50:.1f}** [{100 * np.percentile(bs[:, 0], 2.5):.1f}, "
                         f"{100 * np.percentile(bs[:, 0], 97.5):.1f}] | **{100 * m5095:.1f}** "
                         f"[{100 * np.percentile(bs[:, 1], 2.5):.1f}, {100 * np.percentile(bs[:, 1], 97.5):.1f}] | "
                         f"this work (RF-DETR's own val: {100 * v50:.1f} / {100 * v5095:.1f} at epoch {ep}) |")
        lines += [f"| {r} | {a:.1f} | {b:.1f} | published / earlier |" for r, a, b in refs]
        lines += ["", "| class | " + " | ".join(runs) + " |", "|---|" + "---|" * len(runs)]
        lines += [f"| {c} | " + " | ".join(f"{100 * res[r]['pc50'][c]:.1f}" for r in runs) + " |" for c in names]
        lines.append("")
    json.dump(res, open(os.path.join(OUT, "report.json"), "w"), indent=1)
    open(os.path.join(OUT, "report.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
