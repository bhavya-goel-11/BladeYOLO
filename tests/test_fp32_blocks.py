"""Residual refinement blocks must stay finite under fp16 autocast even when their inner activations grow.

Their output conv feeds a BatchNorm, so training does not bound the scale entering that conv; in the BladeYOLO-v2
run it reached ~2.4e3 and the conv output overflowed fp16 (65504) -> NaN from epoch 17 on.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.morphology import C2fMorphRes  # noqa: E402
from models.ss2d import C2fSS2DRes  # noqa: E402


def _check(block):
    torch.manual_seed(0)
    m = block(128, 128).train()
    with torch.no_grad():
        m.cv2.bn.weight.fill_(1.0)  # trained: no longer the zero-init identity
        m.cv2.conv.weight.mul_(1e5)  # inner scale drift, as in the failing run
    x = torch.randn(2, 128, 16, 16)
    ref = m(x)
    with torch.autocast("cpu", dtype=torch.float16):
        y = m(x.half())
    assert torch.isfinite(y).all()
    assert torch.allclose(y.float(), ref, atol=0.05, rtol=0.01)


def test_half_model_keeps_block_fp32():
    m = C2fSS2DRes(128, 128).eval().half()  # Ultralytics validates and saves checkpoints with model.half()
    assert all(p.dtype == torch.float32 for p in m.parameters())
    y = m(torch.randn(1, 128, 8, 8).half())
    assert y.dtype == torch.float16 and torch.isfinite(y).all()


def test_ss2d_res_fp16():
    _check(C2fSS2DRes)


def test_morph_res_fp16():
    _check(C2fMorphRes)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
