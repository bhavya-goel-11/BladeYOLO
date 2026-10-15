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
from ultralytics.nn.modules.block import AAttn  # noqa: E402


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


def test_area_attention_scores_fp32():
    """YOLOv12 area attention: identical in fp32; in fp16 no worse than stock below overflow, finite above it."""

    def make(scale):
        torch.manual_seed(0)
        m = AAttn(dim=256, num_heads=8, area=4).eval()
        with torch.no_grad():
            m.qkv.conv.weight.mul_(scale)  # activations of a few hundred, as in the late-training NaN batches
        return m

    x = torch.randn(2, 256, 16, 16)
    with torch.no_grad():
        m = make(1)
        assert torch.equal(m(x), AAttn._stock_forward(m, x))
        m = make(40)  # qkv absmax ~130: stock fp16 still finite
        ref = m(x)
        mh = m.half()
        err_fixed = (mh(x.half()).float() - ref).abs().max()
        err_stock = (AAttn._stock_forward(mh, x.half()).float() - ref).abs().max()
        assert err_fixed <= err_stock * 1.01, (err_fixed, err_stock)
        mh = make(200).half()  # qkv absmax ~660: stock fp16 scores overflow (inf -> NaN)
        assert not torch.isfinite(AAttn._stock_forward(mh, x.half())).all()
        assert torch.isfinite(mh(x.half())).all()

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
