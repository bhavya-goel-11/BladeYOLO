"""V1 (Phase 31): RF-DETR, a foundation-native detector fine-tuned end to end, on the same 25% subsets as the
BladeYOLO-v2 ViT-B runs, scored with the scene bootstrap (tools/bootstrap_ci.py) on the same test images.

  prepare  (Kaggle) YOLO-layout copy with only the subset in train/; val -> valid/, test/ unchanged
  train    (Kaggle) RF-DETR fine-tune (pip install "rfdetr[train]==1.11.2")
  predict  (Kaggle) test and val predictions, conf 0.001, keyed by file name, boxes in 640-input pixels
  import   (local)  file-name keys -> local paths in the bootstrap cache, so bootstrap_ci.py scores the run like any
                    other (its run argument only names the cache file)
  score    (local)  the pre-registered paired comparison with ViT-B seed 0 on each subset -> runs/analysis/v1/report.md

Kaggle, per subset:
  python tools/rfdetr_v1.py prepare --data <WTBD-group> --subset tools/splits/wtbd_group_train_f25_rand1.txt --out ds_rand1
  python tools/rfdetr_v1.py train --dataset ds_rand1 --out runs/detect/BladeYOLO_WindSurface/rfdetrm_rand1_wtbd_group_f25
  python tools/rfdetr_v1.py predict --run runs/detect/BladeYOLO_WindSurface/rfdetrm_rand1_wtbd_group_f25 --data <WTBD-group>
Local:
  python tools/rfdetr_v1.py import --run runs/detect/BladeYOLO_WindSurface/rfdetrm_rand1_wtbd_group_f25
  (WTBlade-group runs: add --dataset wtblade; the cache follows the run name's fraction, f25 or f08)
"""

import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
DATA = os.path.join(ROOT, "data", "blade-defect-benchmark", "leakfree", "WTBD-group")
CACHE = {"test": os.path.join(ROOT, "runs", "analysis", "bootstrap", "seeds_f25", "preds"),
         "val": os.path.join(ROOT, "runs", "analysis", "bootstrap", "v1_val_preds")}
WTBLADE = os.path.join(ROOT, "data", "blade-defect-benchmark", "leakfree", "WTBlade-group")  # H18-W, cross-dataset replication


def wtblade_cache(run):
    """H18-W runs (*_f25) -> wtblade_f25/, H19-W runs (*_f08) -> wtblade_f08/."""
    frac = os.path.basename(os.path.normpath(run)).rsplit("_", 1)[-1]
    return {s: os.path.join(ROOT, "runs", "analysis", "bootstrap", f"wtblade_{frac}", s) for s in ("test", "val")}
SIZES = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium", "base": "RFDETRBase",
         "large": "RFDETRLarge"}
BEST = ["checkpoint_best_total.pth", "checkpoint_best_ema.pth", "checkpoint_best_regular.pth"]


def prepare(data, subset, out):
    """Copy (not symlink: RF-DETR rejects split folders resolving outside the dataset root) a YOLO-layout dataset."""
    keep = set(open(subset).read().split())
    names = yaml.safe_load(open(os.path.join(data, "data.yaml")))["names"]
    for src, dst in [("train", "train"), ("val", "valid"), ("test", "test")]:
        for d in ("images", "labels"):
            os.makedirs(os.path.join(out, dst, d), exist_ok=True)
        imgs = sorted(os.listdir(os.path.join(data, src, "images")))
        if src == "train":
            missing = keep - set(imgs)
            assert not missing, f"{len(missing)} subset images not in train/, e.g. {sorted(missing)[:3]}"
            imgs = [f for f in imgs if f in keep]
        for f in imgs:
            shutil.copy2(os.path.join(data, src, "images", f), os.path.join(out, dst, "images", f))
            lab = os.path.splitext(f)[0] + ".txt"
            p = os.path.join(data, src, "labels", lab)
            if os.path.exists(p):
                shutil.copy2(p, os.path.join(out, dst, "labels", lab))
        print(f"{dst}: {len(imgs)} images", flush=True)
    yaml.safe_dump(dict(path=".", train="train/images", val="valid/images", test="test/images", nc=len(names),
                        names=list(names)), open(os.path.join(out, "data.yaml"), "w"), sort_keys=False)


def prepare_official(data, out, carve=0.0, seed=0, agnostic=False):
    """Official (published, leaky) protocol: copy the dataset's own split unchanged. Folders train/ + val|valid/ +
    test/ map to train/valid/test. A dataset without test/ (Wind Surface Defect: train/val, the paper's 4:1 protocol)
    keeps val/ as test/, and `carve` > 0 moves that fraction of train/ (random, seeded) to valid/ for checkpoint
    selection, so the reported test set is never used for selection. `agnostic` (G28): every box becomes class 0,
    "defect"."""
    import random

    names = yaml.safe_load(open(os.path.join(data, "data.yaml")))["names"]
    names = list(names.values()) if isinstance(names, dict) else list(names)
    has = {d: os.path.isdir(os.path.join(data, d, "images")) for d in ("train", "val", "valid", "test")}
    val_src = "valid" if has["valid"] else "val"
    train = sorted(os.listdir(os.path.join(data, "train", "images")))
    if has["test"]:
        plan = {"train": [("train", f) for f in train], "valid": [(val_src, f) for f in sorted(os.listdir(os.path.join(data, val_src, "images")))],
                "test": [("test", f) for f in sorted(os.listdir(os.path.join(data, "test", "images")))]}
    else:
        assert carve > 0, "no test/ split: pass --carve to hold out a validation set from train/"
        rng = random.Random(seed)
        held = set(rng.sample(train, round(carve * len(train))))
        plan = {"train": [("train", f) for f in train if f not in held], "valid": [("train", f) for f in sorted(held)],
                "test": [(val_src, f) for f in sorted(os.listdir(os.path.join(data, val_src, "images")))]}
    for dst, items in plan.items():
        for d in ("images", "labels"):
            os.makedirs(os.path.join(out, dst, d), exist_ok=True)
        for src, f in items:
            shutil.copy2(os.path.join(data, src, "images", f), os.path.join(out, dst, "images", f))
            lab = os.path.join(data, src, "labels", os.path.splitext(f)[0] + ".txt")
            if os.path.exists(lab):
                dst_lab = os.path.join(out, dst, "labels", os.path.basename(lab))
                if agnostic:
                    rows = [l.split() for l in open(lab) if l.strip()]
                    open(dst_lab, "w").write("".join(" ".join(["0"] + r[1:]) + "\n" for r in rows))
                else:
                    shutil.copy2(lab, dst_lab)
        print(f"{dst}: {len(items)} images", flush=True)
    names = ["defect"] if agnostic else names
    yaml.safe_dump(dict(path=".", train="train/images", val="valid/images", test="test/images", nc=len(names),
                        names=names), open(os.path.join(out, "data.yaml"), "w"), sort_keys=False)


def model_class(size):
    import rfdetr

    return getattr(rfdetr, SIZES[size])


def train(dataset, out, size, epochs, batch, accum, seed, freeze_encoder=False, resolution=None, lr_schedule=None,
          lr_drop=None, init=None, slot_mask=None, keep=()):
    if any(os.path.exists(os.path.join(out, b)) for b in BEST):
        print(f"{out} already has a best checkpoint; not retraining")
        return
    os.makedirs(out, exist_ok=True)
    opts = dict(dataset=os.path.abspath(dataset), size=size, epochs=epochs, batch_size=batch, grad_accum_steps=accum,
                seed=seed, freeze_encoder=freeze_encoder, resolution=resolution, lr_schedule=lr_schedule, lr_drop=lr_drop,
                init=os.path.abspath(init) if init else None, slot_mask=os.path.abspath(slot_mask) if slot_mask else None,
                keep=list(keep))
    json.dump(opts, open(os.path.join(out, "v1_options.json"), "w"), indent=2)
    if slot_mask:  # G27 direction 2, stage 1: each image's loss only on its own dataset's slots
        import g27_slots

        g27_slots.install(slot_mask)
    # periodic checkpoints only at the end: four runs of 10-epoch checkpoints filled Kaggle's disk (v69)
    # G22 control: freeze_encoder keeps the DINOv2 encoder frozen (projector and decoder still train)
    # G27: init = an adapted RF-DETR checkpoint (direction 1 encoder, or the direction 2 stage-1 model) instead of COCO
    kw = model_kwargs(freeze_encoder, resolution, init)
    if init:  # RF-DETR keeps a smaller-than-COCO checkpoint head (the 34-slot stage 1) unless num_classes is explicit
        names = yaml.safe_load(open(os.path.join(dataset, "data.yaml")))["names"]
        kw["num_classes"] = len(names)
    model_class(size)(**kw).train(dataset_dir=os.path.abspath(dataset), output_dir=os.path.abspath(out), epochs=epochs,
                              batch_size=batch, grad_accum_steps=accum, seed=seed, checkpoint_interval=epochs,
                              **schedule_kwargs(lr_schedule, lr_drop, epochs))
    for f in os.listdir(out):
        if f.endswith((".ckpt", ".pth")) and f != best_checkpoint(out).split(os.sep)[-1] and f not in keep:
            os.remove(os.path.join(out, f))


def model_kwargs(freeze_encoder=False, resolution=None, init=None):
    """G22 control (frozen DINOv2 encoder), G24 (input resolution, divisible by patch 16 x 2 windows = 32), G27 init."""
    kw = {"freeze_encoder": True} if freeze_encoder else {}
    if init:
        assert os.path.exists(init), init
        kw["pretrain_weights"] = os.path.abspath(init)
    if resolution:
        assert resolution % 32 == 0, resolution
        kw["resolution"] = resolution
    return kw


def schedule_kwargs(lr_schedule=None, lr_drop=None, epochs=100):
    """G26: RF-DETR's default is a 10x step decay after lr_drop = 100 epochs, which a run of <= 100 epochs never
    reaches (constant LR throughout, every run before G26). 'cosine' anneals to 0 over the run; lr_drop moves the step."""
    if lr_schedule in (None, "step") and lr_drop is None:
        return {}
    if lr_schedule == "cosine":
        assert lr_drop is None, "lr_drop applies to the step schedule only"
        return {"lr_scheduler": "cosine"}
    assert lr_schedule in (None, "step") and 0 < lr_drop < epochs, (lr_schedule, lr_drop, epochs)
    return {"lr_scheduler": "step", "lr_scheduler_kwargs": {"lr_drop": lr_drop}}


def run_resolution(run):
    """The training resolution recorded in v1_options.json (None = the model default, 576 for Medium)."""
    p = os.path.join(run, "v1_options.json")
    return json.load(open(p)).get("resolution") if os.path.exists(p) else None


def best_checkpoint(run):
    for b in BEST:
        if os.path.exists(os.path.join(run, b)):
            return os.path.join(run, b)
    raise FileNotFoundError(f"no {BEST} in {run}: {sorted(glob.glob(os.path.join(run, '*.pth')))}")


def to_rows(xyxy, conf, cls, scale):
    """Our cache format: (n, 6) x1 y1 x2 y2 conf cls in 640-input pixels."""
    return np.c_[np.asarray(xyxy, float).reshape(-1, 4) * scale, np.asarray(conf, float), np.asarray(cls, float)].round(3)


def predict(run, data, size, splits, agnostic=False):
    opts = os.path.join(run, "v1_options.json")
    size = json.load(open(opts)).get("size", size) if os.path.exists(opts) else size  # G26: Large runs
    from PIL import Image

    names = ["defect"] if agnostic else yaml.safe_load(open(os.path.join(data, "data.yaml")))["names"]  # G28: 1 class
    model = model_class(size)(pretrain_weights=best_checkpoint(run), num_classes=len(names),
                              **model_kwargs(resolution=run_resolution(run)))
    for split in splits:
        d = os.path.join(data, split, "images")
        out, dropped = {}, 0
        for i, f in enumerate(sorted(os.listdir(d))):
            im = Image.open(os.path.join(d, f)).convert("RGB")
            det = model.predict(im, threshold=0.001, include_source_image=False)
            cls = np.asarray(det.class_id)
            assert len(cls) == 0 or (cls.min() >= 0 and cls.max() <= len(names)), f"class ids {np.unique(cls)}"
            keep = cls < len(names)  # the head has num_classes + 1 outputs; the last is the no-object slot
            dropped += int((~keep).sum())
            out[f] = to_rows(np.asarray(det.xyxy)[keep], np.asarray(det.confidence)[keep], cls[keep],
                             640 / max(im.size)).tolist()
            if (i + 1) % 50 == 0:
                print(f"  {split}: {i + 1}", flush=True)
        json.dump(out, open(os.path.join(run, f"preds_{split}.json"), "w"))
        print(f"{split}: {len(out)} images, {sum(len(v) for v in out.values())} boxes ({dropped} no-object boxes dropped) "
              f"-> {run}/preds_{split}.json")


def import_preds(run, data=DATA, cache=CACHE):
    """preds_<split>.json (file-name keys) -> <cache[split]>/<run name>.json with local absolute paths."""
    name = os.path.basename(os.path.normpath(run))
    for split, c in cache.items():
        src = os.path.join(run, f"preds_{split}.json")
        if not os.path.exists(src):
            print(f"skip {split}: no {src}")
            continue
        p = json.load(open(src))
        local = sorted(os.listdir(os.path.join(data, split, "images")))
        assert sorted(p) == local, f"{split}: prediction files differ from {data}/{split}/images"
        os.makedirs(c, exist_ok=True)
        json.dump({os.path.abspath(os.path.join(data, split, "images", f)): p[f] for f in local},
                  open(os.path.join(c, f"{name}.json"), "w"))
        print(f"{split}: {len(local)} images -> {c}/{name}.json")


PAIRS = [("class-matched", "rfdetrm_wtbd_group_f25", "bladeyolo2b_wtbd_group_f25_s0")] + [
    (f"rand{k}", f"rfdetrm_rand{k}_wtbd_group_f25", f"bladeyolo2b_rand{k}_wtbd_group_f25_s0") for k in (1, 2, 3)]
PAIRS_S1 = [(s, a.replace("rfdetrm_", "rfdetrm_s1_"), b) for s, a, b in PAIRS]  # seed 1 (H18-R), same ViT-B s0
CLASSES = ["craze", "corrosion", "surface_injure", "thunderstrike", "crack", "hide_craze"]


def score_v1(n_boot=2000, seed=0, pairs=PAIRS, tag=""):
    """Scene bootstrap over WTBD-group test: per subset RF-DETR - ViT-B mAP50-95, and the mean over the subsets present.
    Pre-registered bar: mean >= +0.03, positive on >= 3 of 4 subsets, pooled P(mean <= 0) < 0.05."""
    import bootstrap_ci as bc

    imgs = bc.load_yolo_split(DATA, "test")
    files = sorted(imgs)
    groups = [[files[i] for i in g] for g in bc.scene_groups(files, os.path.join(ROOT, "data", "cache"), 4)]
    nc = len(CLASSES)
    load = lambda n: {k: np.array(v).reshape(-1, 6) for k, v in json.load(open(os.path.join(CACHE["test"], n + ".json"))).items()}  # noqa: E731
    have = [p for p in pairs if os.path.exists(os.path.join(CACHE["test"], p[1] + ".json"))]
    m = {n: bc.per_image_matches(imgs, load(n), nc) for _, a, b in have for n in (a, b)}
    point = {n: bc.score(v, files, nc) for n, v in m.items()}
    rng = np.random.default_rng(seed)
    d = np.zeros((n_boot, len(have)))
    for i in range(n_boot):
        draw = [f for g in rng.integers(0, len(groups), len(groups)) for f in groups[g]]
        d[i] = [bc.score(m[a], draw, nc)[1] - bc.score(m[b], draw, nc)[1] for _, a, b in have]
    diff = np.array([point[a][1] - point[b][1] for _, a, b in have])
    mean, pm = float(diff.mean()), float(np.mean(d.mean(1) <= 0))
    ok = [mean >= 0.03, int((diff > 0).sum()) >= 3, pm < 0.05]
    complete = len(have) == len(pairs)
    ci = lambda x: f"[{np.percentile(x, 2.5):+.3f}, {np.percentile(x, 97.5):+.3f}]"  # noqa: E731
    lines = ["# V1: RF-DETR Medium vs BladeYOLO-v2 ViT-B (seed 0), 25% subsets, WTBD-group test", "",
             f"{len(files)} test images in {len(groups)} scene groups, {n_boot} scene-bootstrap replicates.", "",
             "| subset | RF-DETR mAP50 | RF-DETR mAP50-95 | ViT-B mAP50 | ViT-B mAP50-95 | diff mAP50-95 [95% CI] | P(diff <= 0) |",
             "|---|---|---|---|---|---|---|"]
    for k, (s_, a, b) in enumerate(have):
        lines.append(f"| {s_} | {point[a][0]:.3f} | {point[a][1]:.3f} | {point[b][0]:.3f} | {point[b][1]:.3f} | "
                     f"{diff[k]:+.3f} {ci(d[:, k])} | {np.mean(d[:, k] <= 0):.3f} |")
    lines += ["", f"Mean over {len(have)} subsets: {mean:+.3f} {ci(d.mean(1))}, P(<= 0) {pm:.3f}; positive on "
              f"{int((diff > 0).sum())}/{len(have)}.", "",
              f"Bar (mean >= +0.03; >= 3/4 positive; P < 0.05): {ok} -> "
              + ("**clearly beats**" if complete and all(ok) else "does not clearly beat" if complete
                 else f"incomplete ({len(have)}/{len(pairs)} subsets)"), "",
              "| subset | model | " + " | ".join(CLASSES) + " |", "|---|---|" + "---|" * nc]
    for s_, a, b in have:
        for n in (a, b):
            lines.append(f"| {s_} | {n} | " + " | ".join(f"{x:.3f}" for x in point[n][3]) + " |")
    out = os.path.join(ROOT, "runs", "analysis", "v1")
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, f"report{tag}.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("prepare")
    a.add_argument("--data", required=True)
    a.add_argument("--subset", required=True)
    a.add_argument("--out", required=True)
    a = sp.add_parser("prepare-official")
    a.add_argument("--data", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--carve", type=float, default=0.0, help="fraction of train/ held out as valid/ when there is no test/")
    a.add_argument("--agnostic", action="store_true", help="G28: every box becomes class 0 'defect'")
    a = sp.add_parser("train")
    a.add_argument("--dataset", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--size", default="medium", choices=SIZES)
    a.add_argument("--epochs", type=int, default=100)
    a.add_argument("--batch", type=int, default=4)
    a.add_argument("--accum", type=int, default=4)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--freeze-encoder", action="store_true", help="G22 control: frozen DINOv2 encoder")
    a.add_argument("--resolution", type=int, default=None, help="G24: input resolution (default: model's, 576)")
    a.add_argument("--lr-schedule", choices=["step", "cosine"], default=None, help="G26 (default: RF-DETR's step)")
    a.add_argument("--lr-drop", type=int, default=None, help="G26: epoch of the 10x step decay (default 100)")
    a.add_argument("--init", default=None, help="G27: RF-DETR checkpoint to start from instead of COCO")
    a.add_argument("--slot-mask", default=None, help="G27 stage 1: slots.json of a multi-dataset pool")
    a.add_argument("--keep", nargs="*", default=[], help="checkpoint files to keep besides the best (e.g. last_ema.pth)")
    a = sp.add_parser("predict")
    a.add_argument("--run", required=True)
    a.add_argument("--data", required=True)
    a.add_argument("--size", default="medium", choices=SIZES)
    a.add_argument("--splits", nargs="+", default=["test", "val"])
    a.add_argument("--agnostic", action="store_true", help="G28: the run is a 1-class 'defect' model")
    a = sp.add_parser("import")
    a.add_argument("--run", required=True)
    a.add_argument("--dataset", default="wtbd", choices=["wtbd", "wtblade"])
    a = sp.add_parser("score")
    a.add_argument("--seed1", action="store_true", help="score the seed-1 runs (rfdetrm_s1_*) -> report_s1.md")
    args = ap.parse_args()
    if args.cmd == "prepare":
        prepare(args.data, args.subset, args.out)
    elif args.cmd == "prepare-official":
        prepare_official(args.data, args.out, args.carve, agnostic=args.agnostic)
    elif args.cmd == "train":
        train(args.dataset, args.out, args.size, args.epochs, args.batch, args.accum, args.seed, args.freeze_encoder, args.resolution,
              args.lr_schedule, args.lr_drop, args.init, args.slot_mask, args.keep)
    elif args.cmd == "predict":
        predict(args.run, args.data, args.size, args.splits, args.agnostic)
    elif args.cmd == "import":
        import_preds(args.run) if args.dataset == "wtbd" else import_preds(args.run, WTBLADE, wtblade_cache(args.run))
    else:
        score_v1(pairs=PAIRS_S1, tag="_s1") if args.seed1 else score_v1()


if __name__ == "__main__":
    main()
