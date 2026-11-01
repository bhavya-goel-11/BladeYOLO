"""G22: RFDETRMedium(freeze_encoder=True) must leave the DINOv2 encoder without gradients and the decoder trainable.
  .venv-rfdetr/bin/python tests/test_rfdetr_freeze.py   (uses a local checkpoint, no download)"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import rfdetr_v1 as v1

ck = v1.best_checkpoint(os.path.join(ROOT, "runs", "detect", "BladeYOLO_WindSurface", "rfdetrm_wtbd_group_f25"))
for freeze in (False, True):
    m = v1.model_class("medium")(pretrain_weights=ck, num_classes=6, **({"freeze_encoder": True} if freeze else {}))
    net = m.model.model
    enc = [p.requires_grad for n, p in net.named_parameters() if ".encoder." in n]
    rest = [p.requires_grad for n, p in net.named_parameters() if ".encoder." not in n]
    print(f"freeze={freeze}: encoder params {len(enc)} trainable {sum(enc)}; other {len(rest)} trainable {sum(rest)}")
    assert enc and rest and all(rest)
    assert (sum(enc) == 0) if freeze else all(enc)
print("ok")
