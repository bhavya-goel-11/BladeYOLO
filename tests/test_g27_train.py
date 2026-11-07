"""G27 end to end on CPU (tiny): rfdetr_v1.train with --slot-mask on a 2-dataset pool (stage 1), keeping last_ema.pth,
then a 6-class fine-tune from it with --init, then copy-paste into a prepared dataset.
  .venv-rfdetr/bin/python tests/test_g27_train.py
Checks: the slot mask is called inside RF-DETR's real training loop; stage 1 leaves last_ema.pth with 13 + 1 outputs;
the fine-tune loads it (its encoder starts from stage 1, not COCO) and ends with a 6 + 1 head; g27_paste.paste writes
the planned images and labels and refuses to paste twice."""
import csv
import json
import os
import shutil
import sys

import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import rfdetr_v1 as v1  # noqa: E402

CKPT = os.path.join(ROOT, "data", "weights", "rfdetr", "rf-detr-medium.pth")
OUT = os.path.join(ROOT, "runs", "tests", "g27_train")
WTBD, WTBL = os.path.join(ROOT, "data", "blade-defect-benchmark", "leakfree", "WTBD-group"), os.path.join(ROOT, "data", "blade-defect-benchmark", "leakfree", "WTBlade-group")


def labelled(root, split, n):
    d = os.path.join(root, split, "images")
    out = []
    for f in sorted(os.listdir(d)):
        p = os.path.join(root, split, "labels", os.path.splitext(f)[0] + ".txt")
        if os.path.exists(p) and open(p).read().strip():
            out.append((os.path.join(d, f), p))
        if len(out) == n:
            return out


def write_ds(ds, items, names):
    for split, its in items.items():
        for k in ("images", "labels"):
            os.makedirs(os.path.join(ds, split, k), exist_ok=True)
        for img, lab, off, tag in its:
            name = f"{tag}__{os.path.basename(img)}"
            shutil.copy2(img, os.path.join(ds, split, "images", name))
            rows = [l.split() for l in open(lab) if l.strip()]
            with open(os.path.join(ds, split, "labels", os.path.splitext(name)[0] + ".txt"), "w") as f:
                f.write("".join(f"{int(r[0]) + off} {' '.join(r[1:])}\n" for r in rows))
    yaml.safe_dump(dict(path=".", train="train/images", val="valid/images", test="test/images", nc=len(names),
                        names=names), open(os.path.join(ds, "data.yaml"), "w"), sort_keys=False)


def main():  # DataLoader workers start with forkserver (Python 3.14): they re-import this file
    shutil.rmtree(OUT, ignore_errors=True)
    # stage 1: 2 datasets, slots 0-5 (WTBD) and 6-12 (WTBlade)
    pool = os.path.join(OUT, "pool")
    a, b = labelled(WTBD, "train", 3), labelled(WTBL, "train", 3)
    va, vb = labelled(WTBD, "val", 1), labelled(WTBL, "val", 1)
    wn = yaml.safe_load(open(os.path.join(WTBD, "data.yaml")))["names"]
    bn = yaml.safe_load(open(os.path.join(WTBL, "data.yaml")))["names"]
    split = lambda x, y: [(i, l, 0, "wtbd") for i, l in x] + [(i, l, 6, "wtblade") for i, l in y]  # noqa: E731
    write_ds(pool, {"train": split(a, b), "valid": split(va, vb), "test": split(va, vb)},
             [f"wtbd:{n}" for n in wn] + [f"wtblade:{n}" for n in bn])
    slots = os.path.join(pool, "slots.json")
    json.dump({"wtbd": {"offset": 0, "names": wn}, "wtblade": {"offset": 6, "names": bn}}, open(slots, "w"))

    s1 = os.path.join(OUT, "stage1")
    v1.train(pool, s1, "medium", 1, 2, 1, 0, resolution=192, init=CKPT, slot_mask=slots, keep=["last_ema.pth"])
    from rfdetr.models.criterion import SetCriterion  # noqa: E402
    calls = SetCriterion.forward.calls
    assert calls >= 3, f"slot mask called {calls} times in training"
    last = os.path.join(s1, "last_ema.pth")
    assert os.path.exists(last), os.listdir(s1)
    st1 = torch.load(last, map_location="cpu", weights_only=False)["model"]
    assert st1["class_embed.bias"].shape[0] == 14, st1["class_embed.bias"].shape
    coco = torch.load(CKPT, map_location="cpu", weights_only=False)["model"]
    k = "backbone.0.encoder.encoder.encoder.layer.5.mlp.fc1.weight"
    assert not torch.equal(st1[k], coco[k]), "stage 1 must have trained the encoder"
    SetCriterion.forward = SetCriterion.forward._g27_orig  # the fine-tune runs without the mask

    # stage 2: a 6-class WTBD fine-tune from the stage-1 checkpoint
    ft_ds = os.path.join(OUT, "wtbd")
    write_ds(ft_ds, {"train": [(i, l, 0, "x") for i, l in a], "valid": [(i, l, 0, "x") for i, l in va],
                     "test": [(i, l, 0, "x") for i, l in va]}, wn)
    m = v1.model_class("medium")(**v1.model_kwargs(resolution=192, init=last))
    sd = m.model.model.state_dict()
    assert torch.equal(sd[k], st1[k]), "the fine-tune must start from the stage-1 encoder"
    ft = os.path.join(OUT, "ft")
    v1.train(ft_ds, ft, "medium", 1, 2, 1, 0, resolution=192, init=last)
    best = torch.load(v1.best_checkpoint(ft), map_location="cpu", weights_only=False)["model"]
    assert best["class_embed.bias"].shape[0] == 7, best["class_embed.bias"].shape
    opts = json.load(open(os.path.join(ft, "v1_options.json")))
    assert opts["init"] == os.path.abspath(last) and opts["slot_mask"] is None

    # copy-paste into a prepared dataset (the committed WTBD-group plan; only its rows' images are needed)
    import g27_paste  # noqa: E402

    rows = list(csv.DictReader(open(g27_paste.plan_path("wtbd_group"))))
    need = {r[c] for r in rows for c in ("image", "src_image")}
    prep = os.path.join(OUT, "prepared")
    for d in ("images", "labels"):
        os.makedirs(os.path.join(prep, "train", d))
    for n in need:
        shutil.copy2(os.path.join(WTBD, "train", "images", n), os.path.join(prep, "train", "images", n))
        lab = os.path.join(WTBD, "train", "labels", os.path.splitext(n)[0] + ".txt")
        if os.path.exists(lab):
            shutil.copy2(lab, os.path.join(prep, "train", "labels"))
    g27_paste.paste("wtbd_group", prep)
    new = sorted(f for f in os.listdir(os.path.join(prep, "train", "images")) if f.startswith("cp_"))
    n_img = len({r["i"] for r in rows})
    assert len(new) == n_img, (len(new), n_img)
    added = 0
    for f in new:
        stem = os.path.splitext(f)[0]
        tgt = stem.split("__", 1)[1]
        lines = open(os.path.join(prep, "train", "labels", stem + ".txt")).read().split("\n")
        base = open(os.path.join(WTBD, "train", "labels", tgt + ".txt")).read().strip().split("\n")
        added += len([l for l in lines if l.strip()]) - len([l for l in base if l.strip()])
    assert added == len(rows), (added, len(rows))
    try:
        g27_paste.paste("wtbd_group", prep)
        raise SystemExit("pasted twice")
    except AssertionError:
        pass
    print(f"stage 1: slot mask called {calls}x, last_ema 14 outputs; fine-tune from it: 7 outputs; "
          f"copy-paste: {n_img} images, {added} boxes")
    print("ok")


if __name__ == "__main__":
    main()
