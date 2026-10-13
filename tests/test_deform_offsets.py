"""DeformConv must not crash on non-finite offsets and must not change outputs for valid ones.

torchvision's deform_conv2d bounds-checks sampling positions with comparisons that NaN passes, so NaN/huge offsets
read outside the tensor and segfault. Ultralytics' FLOPs probe feeds an uninitialised torch.empty image; on Kaggle this
killed the post-training test evaluation twice without any traceback.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.morphology import DeformConv  # noqa: E402


def test_non_finite_offsets():
    torch.manual_seed(0)
    m = DeformConv(8, 8, 3).eval()
    for v in (float("nan"), float("inf"), -float("inf"), 1e30):
        with torch.no_grad():
            m.offset_mask.bias.fill_(v)
            y = m(torch.randn(1, 8, 16, 16))
        assert y.shape == (1, 8, 16, 16)
    print("ok test_non_finite_offsets")


def test_valid_offsets_unchanged():
    from torchvision.ops import deform_conv2d

    torch.manual_seed(0)
    m = DeformConv(8, 8, 3).eval()
    with torch.no_grad():
        m.offset_mask.weight.normal_(0, 0.5)
        x = torch.randn(2, 8, 16, 16)
        y = m(x)
        offset, mask = m.offset_mask(x).split([18, 9], 1)  # unclamped reference
        ref = m.act(m.bn(deform_conv2d(x, offset, m.weight, stride=1, padding=1, mask=mask.sigmoid())))
    assert torch.equal(y, ref)
    print("ok test_valid_offsets_unchanged")


if __name__ == "__main__":
    test_non_finite_offsets()
    test_valid_offsets_unchanged()
