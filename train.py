"""Train BladeYOLO (or a stock Ultralytics baseline) on the Wind Surface Defect dataset.

Examples:
    python train.py --model bladeyolo2-l.yaml --name bladeyolo2_v3 --epochs 100   # BladeYOLO-v2-L
    python train.py --model yolo12l.pt --name yolo12l_v3_r90 --box-loss ciou --epochs 100   # stock baseline
    python train.py                                  # BladeYOLO-L (v1), both GPUs if available
    python train.py --resume runs/detect/BladeYOLO_WindSurface/bladeyolo_l/weights/last.pt
"""

import argparse
import os
import sys

import yaml

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# Default: the merged, deduplicated, group-split dataset built by tools/build_dataset.py, copied into the
# project folder (locally and on Kaggle), else read straight from the Kaggle input.
# The original Wind Surface Defect split leaks augmented twins into val; use it only explicitly via --data.
DATA_CANDIDATES = [
    os.path.join(ROOT_DIR, "WindSurface-Defect-v3", "data.yaml"),
    "/kaggle/input/datasets/beegee11/wind-surface-defect/WindSurface-Defect-v3/data.yaml",
]
# Optimisation and augmentation recipe, shared by every run (and tools/diagnose_nan.py).
RECIPE = dict(
    deterministic=False,  # deterministic mode slows training and deform_conv2d's backward has no deterministic kernel
    optimizer="AdamW",
    lr0=0.002,
    lrf=0.001,
    cos_lr=True,
    warmup_epochs=5,
    weight_decay=0.01,  # AdamW decay; v2 runs overfit after ~100 epochs with 0.0005
    mosaic=1.0,
    mixup=0.15,
    flipud=0.5,  # drone views of blades have no canonical "up"
    close_mosaic=10,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=os.path.join(ROOT_DIR, "bladeyolo-l.yaml"), help="model YAML")
    p.add_argument("--data", default=None, help="dataset YAML (default: WindSurface-Defect-v3, local or Kaggle input)")
    p.add_argument("--name", default="bladeyolo_l", help="run name under runs/detect/BladeYOLO_WindSurface")
    p.add_argument("--resume", default=None, help="last.pt of an interrupted run to resume")
    p.add_argument("--epochs", type=int, default=150, help="v2 runs peaked at epochs ~90-170")
    p.add_argument("--seed", type=int, default=0, help="change for repeat runs to measure run-to-run variance")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16, help="total batch size across GPUs")
    p.add_argument("--workers", type=int, default=4, help="dataloader workers per GPU")
    p.add_argument("--device", default=None, help="e.g. '0' or '0,1' (default: all visible GPUs)")
    p.add_argument("--cache", default="ram", help="image cache: ram, disk or false")
    p.add_argument("--box-loss", choices=("wiou", "ciou"), default="wiou")
    p.add_argument("--rot90", type=float, default=0.5,
                   help="probability of a random quarter turn (with the flips: all 8 orientations); 0 disables")
    p.add_argument("--no-amp", action="store_true", help="train in fp32 (slower; for debugging)")
    return p.parse_args()


def resolve_data(path):
    """Write a copy of the dataset YAML with an absolute root (the original may be read-only, e.g. on Kaggle)."""
    path = path or next((p for p in DATA_CANDIDATES if os.path.exists(p)), None)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Dataset YAML not found. Pass --data. Looked in: {DATA_CANDIDATES}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["path"] = root = os.path.abspath(os.path.join(os.path.dirname(path), cfg.get("path") or "."))
    for split in ("train", "val"):  # the Kaggle zips unpack to <split>/<split>/images, local copies to <split>/images
        if not os.path.isdir(os.path.join(root, cfg[split])) and os.path.isdir(os.path.join(root, split, "images")):
            cfg[split] = os.path.join(split, "images")
    out = os.path.join(ROOT_DIR, "runs", "data.yaml")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out


def main():
    args = parse_args()
    os.environ["BLADEYOLO_BOX_LOSS"] = args.box_loss  # read by `import models`, inherited by DDP workers
    os.environ["BLADEYOLO_ROT90"] = str(args.rot90)  # read by BladeTrainer.build_dataset, inherited by DDP workers
    sys.path.insert(0, ROOT_DIR)  # DDP workers copy sys.path, so they can import `models` too

    import torch
    from ultralytics import YOLO

    from models.trainer import BladeTrainer

    test_device = 0 if torch.cuda.is_available() else "cpu"
    if args.resume:
        model = YOLO(args.resume)
        model.train(resume=True, trainer=BladeTrainer)
        evaluate_test(model.trainer.best, model.trainer.args.data, args.batch, test_device)
        return

    device = args.device or ",".join(str(i) for i in range(max(torch.cuda.device_count(), 1)))
    data = resolve_data(args.data)
    model = YOLO(args.model)
    model.train(
        trainer=BladeTrainer,
        data=data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device if torch.cuda.is_available() else "cpu",
        workers=args.workers,
        cache=False if args.cache == "false" else args.cache,
        amp=not args.no_amp,
        project=os.path.join(ROOT_DIR, "runs", "detect", "BladeYOLO_WindSurface"),
        name=args.name,
        seed=args.seed,
        **RECIPE,
    )
    evaluate_test(model.trainer.best, data, args.batch, test_device, box_loss=args.box_loss, rot90=args.rot90)


def evaluate_test(weights, data, batch, device, **info):
    """Report the held-out test split once, with the checkpoint selected on val; also with test-time augmentation
    (flips and rescaling, merged predictions), reported separately as metrics_tta.json."""
    with open(data) as f:
        if "test" not in yaml.safe_load(f):
            return
    from ultralytics import YOLO

    print(f"\nTest-split evaluation of {weights}")
    out = os.path.join(os.path.dirname(os.path.dirname(weights)), "test")
    metrics = YOLO(weights).val(data=data, split="test", batch=batch, imgsz=640, device=device,
                                project=os.path.dirname(out), name="test", exist_ok=True)
    save_metrics(metrics, os.path.join(out, "metrics.json"), weights=str(weights), split="test", **info)
    print("\nTest-split evaluation with test-time augmentation")
    metrics = YOLO(weights).val(data=data, split="test", batch=batch, imgsz=640, device=device, augment=True,
                                project=os.path.dirname(out), name="test_tta", exist_ok=True, plots=False)
    save_metrics(metrics, os.path.join(out, "metrics_tta.json"), weights=str(weights), split="test", tta=True, **info)


def save_metrics(metrics, path, **info):
    """Write an Ultralytics validation result (overall + per class) to JSON, so results never live only in logs."""
    import json

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({**info, "overall": metrics.results_dict, "per_class": metrics.summary(normalize=False)},
                  f, indent=2, default=float)


if __name__ == "__main__":
    main()
