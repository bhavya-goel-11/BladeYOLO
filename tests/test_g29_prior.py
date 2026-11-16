"""G29 frozen DINOv3 prior on RF-DETR (tools/g29_prior.py), on CPU:
  .venv-rfdetr/bin/python tests/test_g29_prior.py [unit|e2e]     (no argument: both, each in its own process, so the
                                                                 unit model is freed before training; 6 GB machine)
Unit (RF-DETR-M from the COCO checkpoint, prior installed):
  - the prior's grid equals the projector's at 320 and 768 px (both patch 16), so no resampling happens on Medium;
  - at initialisation the patched backbone returns exactly the unpatched backbone's features (zero-init Out);
  - the frozen DINOv3 is outside the model: not in parameters() (optimizer, EMA) and not in state_dict() (checkpoints);
    the fusion layers are in both;
  - once Out is non-zero, gradients reach the fusion layers and never DINOv3.
End to end (rfdetr_v1.train --prior s, 1 epoch on 4 WTBD-group images, 192 px, then rfdetr_v1.predict; ViT-S keeps
it inside this 6 GB machine: the code path is the one ViT-B takes, which the unit part checks):
  - v1_options.json records the prior; the best checkpoint holds trained (non-zero) fusion weights;
  - a model rebuilt from the checkpoint the way predict builds it carries exactly those weights (nothing silently
    dropped by RF-DETR's strict=False loading), and predict writes predictions for every test image.
"""
import json
import os
import shutil
import sys

import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import g29_prior  # noqa: E402
import rfdetr_v1 as v1  # noqa: E402

OUT = os.path.join(ROOT, "runs", "tests", "g29_prior")
WTBD = os.path.join(ROOT, "data", "blade-defect-benchmark", "leakfree", "WTBD-group")


def backbone_of(m):
    return m.model.model.backbone[0]


def unit():
    g29_prior.install("b")
    from rfdetr.models.backbone.backbone import Backbone
    from rfdetr.utilities.tensors import NestedTensor

    m = v1.model_class("medium")(resolution=320)
    bb = backbone_of(m).eval()
    model = m.model.model
    dino = g29_prior.prior_model("b", "cpu")
    sd = model.state_dict()
    fuse_keys = [k for k in sd if ".prior_fuse." in k]
    assert fuse_keys and all(k.startswith("backbone.0.prior_fuse.0.") for k in fuse_keys), fuse_keys[:3]
    assert len(bb.prior_fuse) == len(bb.projector_scale) == 1
    assert not torch.any(sd["backbone.0.prior_fuse.0.out.weight"]) and not torch.any(sd["backbone.0.prior_fuse.0.out.bias"])
    dino_ids = {id(p) for p in dino.parameters()}
    assert not dino_ids & {id(p) for p in model.parameters()}, "DINOv3 must not be a model parameter"
    n_sd_dino = sum(1 for k in sd if k.startswith("backbone.0.") and ".blocks." in k and "encoder" not in k)
    assert n_sd_dino == 0, n_sd_dino
    n_fuse = sum(p.numel() for n, p in model.named_parameters() if ".prior_fuse." in n)

    torch.manual_seed(0)
    for res in (320, 768):
        x = torch.randn(1, 3, res, res)
        nt = NestedTensor(x, torch.zeros(1, res, res, dtype=torch.bool), True)
        with torch.no_grad():
            got, _ = bb(nt)
            ref, _ = Backbone._g29_orig[1](bb, nt)
            prior = g29_prior.prior_features(x, "b")
        assert prior.shape == (1, 3 * 768, res // 16, res // 16), prior.shape
        assert got[0].tensors.shape[-2:] == prior.shape[-2:], (got[0].tensors.shape, prior.shape)
        assert torch.equal(got[0].tensors, ref[0].tensors), "zero-init fusion must be the identity"
        assert torch.equal(got[0].mask, ref[0].mask)

    bb.train()
    with torch.no_grad():
        bb.prior_fuse[0].out.weight.normal_(0, 0.01)
    x = torch.randn(1, 3, 320, 320)
    nt = NestedTensor(x, torch.zeros(1, 320, 320, dtype=torch.bool), True)
    got, _ = bb(nt)
    got[0].tensors.sum().backward()
    assert all(p.grad is not None and torch.any(p.grad) for p in bb.prior_fuse.parameters())
    assert all(p.grad is None for p in dino.parameters())
    print(f"unit: grids match at 320/768, identity at init, {n_fuse / 1e6:.2f}M fusion parameters, DINOv3 outside "
          f"the model and its checkpoints")


def labelled(split, n):
    d = os.path.join(WTBD, split, "images")
    out = []
    for f in sorted(os.listdir(d)):
        p = os.path.join(WTBD, split, "labels", os.path.splitext(f)[0] + ".txt")
        if os.path.exists(p) and open(p).read().strip():
            out.append((os.path.join(d, f), p))
        if len(out) == n:
            return out


def end_to_end():
    shutil.rmtree(OUT, ignore_errors=True)
    ds = os.path.join(OUT, "ds")
    for split, src, n in (("train", "train", 4), ("valid", "val", 1), ("test", "test", 2)):
        for k in ("images", "labels"):
            os.makedirs(os.path.join(ds, split, k))
        for img, lab in labelled(src, n):
            shutil.copy2(img, os.path.join(ds, split, "images"))
            shutil.copy2(lab, os.path.join(ds, split, "labels"))
    names = yaml.safe_load(open(os.path.join(WTBD, "data.yaml")))["names"]
    yaml.safe_dump(dict(path=".", train="train/images", val="valid/images", test="test/images", nc=len(names),
                        names=list(names)), open(os.path.join(ds, "data.yaml"), "w"), sort_keys=False)
    run = os.path.join(OUT, "run")
    v1.train(ds, run, "medium", 1, 2, 1, 0, resolution=192, prior="s")
    assert json.load(open(os.path.join(run, "v1_options.json")))["prior"] == "s"
    best = torch.load(v1.best_checkpoint(run), map_location="cpu", weights_only=False)["model"]
    k = "backbone.0.prior_fuse.0.out.weight"
    assert k in best and torch.any(best[k]), "the checkpoint must hold trained fusion weights"
    m = v1.model_class("medium")(pretrain_weights=v1.best_checkpoint(run), num_classes=len(names),
                                 **v1.model_kwargs(resolution=192))
    sd = m.model.model.state_dict()
    fuse = [x for x in best if ".prior_fuse." in x]
    assert all(torch.equal(sd[x], best[x]) for x in fuse), "fusion weights must load from the checkpoint"
    v1.predict(run, ds, "medium", ["test"])
    p = json.load(open(os.path.join(run, "preds_test.json")))
    assert sorted(p) == sorted(os.listdir(os.path.join(ds, "test", "images"))), sorted(p)
    print(f"end to end: trained 1 epoch with the prior, {len(fuse)} fusion tensors saved and reloaded, "
          f"{sum(len(v) for v in p.values())} test boxes predicted")


def main():  # DataLoader workers start with forkserver (Python 3.14): they re-import this file
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "all":
        import subprocess

        for m in ("unit", "e2e"):
            subprocess.run([sys.executable, os.path.abspath(__file__), m], check=True)
        print("ok")
        return
    os.chdir(os.path.join(ROOT, "data", "weights", "rfdetr"))  # RF-DETR's default 'rf-detr-medium.pth' is cwd-relative
    unit() if mode == "unit" else end_to_end()


if __name__ == "__main__":
    main()
