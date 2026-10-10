"""Evaluate a checkpoint on one split of WindSurface-Defect-v3, overall and per source dataset.

Usage:
    python tools/eval_by_source.py runs/detect/BladeYOLO_WindSurface/yolo12l_baseline/weights/best.pt --split test
Outputs (plots + metrics.json) go to runs/eval/<run>/<split>_<source>/.
"""

import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCES = {"ws": "Wind Surface Defect", "wt": "WTBlade-Defect"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights")
    ap.add_argument("--data", default=os.path.join(ROOT, "WindSurface-Defect-v3"))
    ap.add_argument("--split", default="test", choices=("val", "test"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    sys.path.insert(0, ROOT)
    import models  # noqa: F401  (needed to load BladeYOLO checkpoints)
    import yaml
    from ultralytics import YOLO

    from train import save_metrics

    data = os.path.abspath(args.data)
    names = yaml.safe_load(open(os.path.join(data, "data.yaml")))["names"]
    rows = [r for r in csv.DictReader(open(os.path.join(data, "manifest.csv"))) if r["split"] == args.split]
    run = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(args.weights))))
    out = os.path.join(ROOT, "runs", "eval", run)
    os.makedirs(out, exist_ok=True)

    model = YOLO(args.weights)
    for src in ("all", *SOURCES):
        files = [os.path.join(data, r["file"]) for r in rows if src == "all" or r["source"] == src]
        name = f"{args.split}_{src}"
        with open(os.path.join(out, f"{name}.txt"), "w") as f:
            f.write("\n".join(files) + "\n")
        with open(os.path.join(out, f"{name}.yaml"), "w") as f:
            yaml.safe_dump({"path": out, "train": f"{name}.txt", "val": f"{name}.txt", "names": names}, f)
        print(f"\n=== {args.split} / {SOURCES.get(src, 'all sources')}: {len(files)} images ===")
        metrics = model.val(data=os.path.join(out, f"{name}.yaml"), split="val", batch=args.batch, imgsz=640,
                            device=args.device, plots=False, project=out, name=name, exist_ok=True)
        save_metrics(metrics, os.path.join(out, name, "metrics.json"), weights=os.path.abspath(args.weights),
                     split=args.split, source=src, images=len(files))


if __name__ == "__main__":
    main()
