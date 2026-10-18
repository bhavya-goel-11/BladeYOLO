"""Train BladeYOLO (or a stock Ultralytics baseline) on a dataset with its official classes, e.g. WTBD.

Examples (WTBD = the folder holding WTBD-group/ and WTBD-official/):
    python train.py --data WTBD/WTBD-group/data.yaml --model bladeyolo2-l.yaml --name bladeyolo2_wtbd_group --epochs 150
    python train.py --data WTBD/WTBD-group/data.yaml --model yolo12l.pt --box-loss ciou --name yolo12l_wtbd_group --epochs 150
    python train.py --data WTBD/WTBD-group/data.yaml --model bladeyolo2-l.yaml --tolerant --name bladeyolo2_wtbd_group_tolerant
    python train.py --data WTBD/WTBD-group/data.yaml --tolerant-classes corrosion,hide_craze --name bladeyolo2_wtbd_group_tolsel_s0
    python train.py --data WTBD/WTBD-group/data.yaml --dino b --train-subset tools/splits/wtbd_group_train_f25.txt --name bladeyolo2b_wtbd_group_f25_s0
    python train.py --data WTBD/WTBD-group/data.yaml --train-subset tools/splits/wtbd_group_train_f25.txt --name bladeyolo2_wtbd_group_f25
    python train.py --data WTBD/WTBD-group/data.yaml --test runs/detect/BladeYOLO_WindSurface/bladeyolo2_wtbd_group/weights/best.pt
    python train.py --data WTBD/WTBD-group/data.yaml --train-subset tools/splits/wtbd_group_train_f25.txt --ssl synthetic \
        --ssl-unlabelled tools/splits/wtbd_group_train_f25_rest.txt --name bladeyolo2_wtbd_group_f25_sslin
    python train.py --resume runs/detect/BladeYOLO_WindSurface/bladeyolo2_wtbd_group/weights/last.pt
"""

import argparse
import os
import shutil
import sys

import yaml

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# Optimisation and augmentation recipe, shared by every run (and tools/probe_nan.py).
RECIPE = dict(
    deterministic=False,  # deterministic mode slows training and deform_conv2d's backward has no deterministic kernel
    optimizer="AdamW",
    # v1/v3 from-scratch runs used lr0=0.002 and Ultralytics' default warmup_bias_lr=0.1. With pretrained weights
    # (BladeYOLO-v2, yolo12l.pt) that disrupted the detector during warmup and diverged to NaN at epoch 4; 0.1 is an
    # SGD value (Ultralytics' own AdamW setting is 0.0 and lr0 ~0.001 for 6 classes).
    lr0=0.001,
    warmup_bias_lr=0.0,
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
    p.add_argument("--model", default=os.path.join(ROOT_DIR, "bladeyolo2-l.yaml"), help="model YAML or .pt")
    p.add_argument("--data", default=None, help="dataset YAML with the dataset's official classes (required unless --resume)")
    p.add_argument("--name", default="bladeyolo2", help="run name under runs/detect/BladeYOLO_WindSurface")
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
    p.add_argument("--train-subset", default=None, metavar="LIST",
                   help="train only on the images named in LIST (e.g. tools/splits/wtbd_group_train_f25.txt)")
    p.add_argument("--ssl", choices=("real", "synthetic"), default=None,
                   help="teacher-student training on unlabelled view pairs (models/view_consistency.py): 'real' = another "
                        "photograph of the surface (different source photo), 'synthetic' = an augmentation of the same "
                        "image (baseline)")
    p.add_argument("--ssl-pairs", default=os.path.join(ROOT_DIR, "runs", "analysis", "novel", "view_pool", "pairs.csv"),
                   help="pairs.csv from tools/view_pool.py")
    p.add_argument("--ssl-unlabelled", default=None, metavar="LIST",
                   help="--ssl synthetic only: unlabelled images from the training split instead of the view pool, "
                        "e.g. tools/splits/wtbd_group_train_f25_rest.txt (the training images outside --train-subset)")
    p.add_argument("--ssl-ratio", type=float, default=0.25, help="unlabelled pairs per labelled image in a batch")
    p.add_argument("--ssl-tau", type=float, default=0.5, help="teacher confidence for a pseudo-label")
    p.add_argument("--ssl-balance", action="store_true",
                   help="class-balanced thresholds: classes the teacher rarely labels get lower ones, down to --ssl-tau-min")
    p.add_argument("--ssl-tau-min", type=float, default=0.25, help="lowest class threshold with --ssl-balance")
    p.add_argument("--ssl-start", type=int, default=30, help="epochs of supervised burn-in before pseudo-labels")
    p.add_argument("--dino", choices=("s", "b", "l"), default="s",
                   help="frozen DINOv3 size for BladeYOLO-v2 models: ViT-S/16 (default), ViT-B/16 or ViT-L/16")
    p.add_argument("--tolerant", action="store_true",
                   help="split-tolerant box targets: a label or the region of touching same-class labels (models/tolerant.py)")
    p.add_argument("--tolerant-classes", default=None, metavar="NAMES",
                   help="comma-separated class names (from --data): split tolerance for these classes only; implies --tolerant")
    p.add_argument("--test", default=None, metavar="WEIGHTS",
                   help="only evaluate WEIGHTS (e.g. a run's best.pt) on the test split of --data, then exit")
    args = p.parse_args()
    if not args.data and not args.resume:
        p.error("--data is required (a resumed run keeps its own dataset)")
    return args


def local_copy(root):
    """A read-only dataset folder (e.g. a Kaggle input) is copied once into the project folder, so Ultralytics can write
    its label caches next to the labels; later runs reuse the copy."""
    if os.access(root, os.W_OK):
        return root
    dst = os.path.join(ROOT_DIR, os.path.basename(root))
    if not os.path.isdir(dst):
        print(f"Copying read-only dataset {root} -> {dst}")
        tmp = dst + ".partial"  # renamed only when complete
        shutil.copytree(root, tmp, copy_function=shutil.copyfile, dirs_exist_ok=True)
        for d, _, _ in os.walk(tmp):  # copytree keeps the source's read-only folder modes
            os.chmod(d, 0o755)
        os.replace(tmp, dst)
    return dst


def resolve_data(path, train_subset=None):
    """Write a copy of the dataset YAML with an absolute root, on a writable copy of the dataset (see local_copy).
    train_subset: file of training image names (tools/scene_fractions.py); training then uses only those images."""
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Dataset YAML not found: {path!r}. Pass --data <dataset>/data.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["path"] = root = local_copy(os.path.abspath(os.path.join(os.path.dirname(path), cfg.get("path") or ".")))
    for split in ("train", "val", "test"):  # the Kaggle zips unpack to <split>/<split>/images, local copies to <split>/images
        if split in cfg and not os.path.isdir(os.path.join(root, cfg[split])) and os.path.isdir(os.path.join(root, split, "images")):
            cfg[split] = os.path.join(split, "images")
    out = os.path.join(ROOT_DIR, "runs", "data.yaml")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if train_subset:
        names = [n.strip() for n in open(train_subset) if n.strip()]
        imgs = [os.path.join(root, cfg["train"], n) for n in names]
        missing = [p for p in imgs if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"{len(missing)} images of {train_subset} not in the training split, e.g. {missing[0]}")
        cfg["train"] = os.path.join(ROOT_DIR, "runs", "train_subset.txt")
        with open(cfg["train"], "w") as f:
            f.write("\n".join(imgs) + "\n")
        print(f"Training on {len(imgs)} images listed in {train_subset}")
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out


def tolerant_class_ids(args):
    """--tolerant-classes names -> "i,j" indices in the dataset YAML's class order."""
    with open(args.data) as f:
        names = yaml.safe_load(f)["names"]
    names = list(names.values()) if isinstance(names, dict) else list(names)
    want = [c.strip() for c in args.tolerant_classes.split(",") if c.strip()]
    unknown = [c for c in want if c not in names]
    if unknown:
        raise SystemExit(f"--tolerant-classes: {unknown} not in the dataset classes {names}")
    ids = [names.index(c) for c in want]
    print(f"[BladeYOLO] split tolerance for {want} only (class ids {ids})")
    return ",".join(map(str, ids))


def main():
    args = parse_args()
    os.environ["BLADEYOLO_BOX_LOSS"] = args.box_loss  # read by `import models`, inherited by DDP workers
    os.environ["BLADEYOLO_ROT90"] = str(args.rot90)  # read by BladeTrainer.build_dataset, inherited by DDP workers
    os.environ["BLADEYOLO_DINO"] = args.dino  # read when DinoConv/DinoFuse are built, inherited by DDP workers
    os.environ["BLADEYOLO_TOLERANT"] = "1" if args.tolerant or args.tolerant_classes else "0"  # read by `import models`, inherited by DDP workers
    os.environ["BLADEYOLO_TOLERANT_CLASSES"] = tolerant_class_ids(args) if args.tolerant_classes else ""  # models/tolerant.py
    os.environ["BLADEYOLO_SSL"] = args.ssl or ""  # read by BladeTrainer (models/view_consistency.config), inherited by DDP workers
    if args.ssl_unlabelled and args.ssl != "synthetic":
        raise SystemExit("--ssl-unlabelled needs --ssl synthetic (a plain image list has no second views)")
    if args.ssl:
        if not args.ssl_unlabelled and not os.path.exists(args.ssl_pairs):
            raise FileNotFoundError(f"--ssl needs the view-pair list from tools/view_pool.py: {args.ssl_pairs}")
        os.environ.update(BLADEYOLO_SSL_PAIRS=os.path.abspath(args.ssl_pairs), BLADEYOLO_SSL_RATIO=str(args.ssl_ratio),
                          BLADEYOLO_SSL_TAU=str(args.ssl_tau), BLADEYOLO_SSL_START=str(args.ssl_start),
                          BLADEYOLO_SSL_BALANCE="1" if args.ssl_balance else "0", BLADEYOLO_SSL_TAU_MIN=str(args.ssl_tau_min))
    sys.path.insert(0, ROOT_DIR)  # DDP workers copy sys.path, so they can import `models` too

    import torch
    from ultralytics import YOLO

    from models.trainer import BladeTrainer

    test_device = 0 if torch.cuda.is_available() else "cpu"
    if args.test:
        evaluate_test(args.test, resolve_data(args.data), args.batch, test_device, imgsz=args.imgsz)
        return
    if args.resume:
        model = YOLO(args.resume)
        model.train(resume=True, trainer=BladeTrainer)
        test_in_new_process(model.trainer.best, model.trainer.args.data, args.batch, model.trainer.args.imgsz)
        return

    device = args.device or ",".join(str(i) for i in range(max(torch.cuda.device_count(), 1)))
    data = resolve_data(args.data, args.train_subset)
    if args.ssl_unlabelled:  # image names of the training split -> absolute paths (before DDP workers start)
        with open(data) as f:
            cfg = yaml.safe_load(f)
        img_dir = os.path.join(cfg["path"], "train", "images")
        names = [n.strip() for n in open(args.ssl_unlabelled) if n.strip()]
        paths = [os.path.join(img_dir, n) for n in names]
        missing = [q for q in paths if not os.path.exists(q)]
        if missing:
            raise FileNotFoundError(f"{len(missing)} images of {args.ssl_unlabelled} not in {img_dir}, e.g. {missing[0]}")
        if args.train_subset:
            overlap = set(names) & {n.strip() for n in open(args.train_subset) if n.strip()}
            if overlap:
                raise ValueError(f"{len(overlap)} unlabelled images are also in --train-subset, e.g. {sorted(overlap)[0]}")
        out = os.path.join(ROOT_DIR, "runs", "ssl_unlabelled.txt")
        with open(out, "w") as f:
            f.write("\n".join(paths) + "\n")
        os.environ["BLADEYOLO_SSL_PAIRS"] = out
        print(f"Unlabelled images for teacher-student training: {len(paths)} from {args.ssl_unlabelled}")
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
    test_in_new_process(model.trainer.best, data, args.batch, args.imgsz)


def test_in_new_process(weights, data, batch, imgsz):
    """Run the test evaluation (train.py --test) in a fresh process. In the training process, after DDP, it twice died
    on Kaggle right after loading the model, with no output or traceback, and the run ended without test metrics."""
    import subprocess

    sys.stdout.flush()
    cmd = [sys.executable, "-u", os.path.abspath(__file__), "--test", str(weights), "--data", str(data),
           "--batch", str(batch), "--imgsz", str(imgsz)]
    code = subprocess.run(cmd).returncode
    if code:
        print(f"Test evaluation exited with code {code}; rerun it with:\n  {' '.join(cmd)}", flush=True)


def evaluate_test(weights, data, batch, device, imgsz=640, **info):
    """Report the held-out test split once, with the checkpoint selected on val; also with test-time augmentation
    (flips and rescaling, merged predictions), reported separately as metrics_tta.json. A failure is printed and the
    other evaluation still runs (rerun with train.py --test)."""
    import traceback

    with open(data) as f:
        if "test" not in yaml.safe_load(f):
            return
    from ultralytics import YOLO

    out = os.path.join(os.path.dirname(os.path.dirname(weights)), "test")
    for tta in (False, True):
        print(f"\nTest-split evaluation of {weights} at {imgsz}px{' with test-time augmentation' if tta else ''}")
        try:
            metrics = YOLO(weights).val(data=data, split="test", batch=batch, imgsz=imgsz, device=device, augment=tta,
                                        project=os.path.dirname(out), name="test_tta" if tta else "test",
                                        exist_ok=True, plots=not tta)
            save_metrics(metrics, os.path.join(out, "metrics_tta.json" if tta else "metrics.json"), weights=str(weights),
                         split="test", imgsz=imgsz, tta=tta, **info)
        except Exception:
            traceback.print_exc()
            print(f"Test-split evaluation{' with TTA' if tta else ''} FAILED (see traceback above)")


def save_metrics(metrics, path, **info):
    """Write an Ultralytics validation result (overall + per class) to JSON, so results never live only in logs."""
    import json

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({**info, "overall": metrics.results_dict, "per_class": metrics.summary(normalize=False)},
                  f, indent=2, default=float)


if __name__ == "__main__":
    main()
