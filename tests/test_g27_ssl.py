"""G27 direction 1: tools/g27_ssl.py on CPU (tiny run).
  .venv-rfdetr/bin/python tests/test_g27_ssl.py
Checks: block masks cover exactly the ratio; the mask wrapper changes only masked calls and is removed afterwards;
a 3-step run writes a checkpoint that RF-DETR loads, whose encoder is the EMA teacher and whose decoder is COCO's."""
import os
import shutil
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import g27_ssl as S  # noqa: E402

CKPT = os.path.join(ROOT, "data", "weights", "rfdetr", "rf-detr-medium.pth")
OUT = os.path.join(ROOT, "runs", "tests", "g27_ssl")
IMGS = os.path.join(OUT, "imgs")

rng = np.random.default_rng(0)
for r in (0.3, 0.5, 0.7):
    m = S.block_mask(36, r, rng)
    assert m.shape == (1296,) and m.sum() == round(r * 1296), (r, m.sum())

shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(IMGS)
src = os.path.join(ROOT, "data", "blade-defect-benchmark", "leakfree", "WTBD-group", "train", "images")
for f in sorted(os.listdir(src))[:6]:
    shutil.copy2(os.path.join(src, f), IMGS)

enc = S.build_encoder(CKPT).eval()
x = torch.randn(2, 3, 192, 192)
with torch.no_grad():
    a = S.forward_masked(enc, x)
    b = S.forward_masked(enc, x, torch.zeros(2, 144, dtype=torch.bool))
    mk = torch.zeros(2, 144, dtype=torch.bool)
    mk[:, :72] = True
    c = S.forward_masked(enc, x, mk)
    d = S.forward_masked(enc, x)
assert len(a) == 4 and a[-1].shape == (2, 384, 12, 12), [t.shape for t in a]
assert all(torch.allclose(p, q, atol=1e-5) for p, q in zip(a, b)), "an all-false mask must change nothing"
assert not torch.allclose(a[-1], c[-1], atol=1e-3), "masking must change the features"
assert all(torch.equal(p, q) for p, q in zip(a, d)), "the wrapper must be removed after the masked call"
assert "forward" not in vars(enc.embeddings)
t = S.target_of(a)
assert t.shape == (2, 12, 12, 384) and torch.isfinite(t).all()

path = S.train(IMGS, OUT, steps=3, batch=2, side=192, warmup=1, workers=0, ckpt=CKPT, log_every=1)
new = torch.load(path, map_location="cpu", weights_only=False)["model"]
old = torch.load(CKPT, map_location="cpu", weights_only=False)["model"]
assert set(new) == set(old)
enc_keys = [k for k in old if k.startswith(S.ENC)]
assert len(enc_keys) == 223
changed = [k for k in enc_keys if not torch.equal(new[k], old[k])]
assert len(changed) > 100, f"only {len(changed)} encoder tensors changed"
assert all(torch.equal(new[k], old[k]) for k in old if not k.startswith(S.ENC)), "non-encoder weights must stay COCO's"
drift = max((new[k].float() - old[k].float()).abs().max().item() for k in changed)
assert drift < 0.05, f"3 EMA steps moved a weight by {drift}"

from rfdetr import RFDETRMedium  # noqa: E402

m = RFDETRMedium(pretrain_weights=path, num_classes=6).model.model
sd = m.state_dict()
assert all(torch.equal(sd[k], new[k]) for k in enc_keys), "RF-DETR must load the adapted encoder"
print(f"masks ok; wrapper ok; 3 steps: {len(changed)}/{len(enc_keys)} encoder tensors changed (max {drift:.2e}); "
      f"RF-DETR loads the checkpoint")
print("ok")
