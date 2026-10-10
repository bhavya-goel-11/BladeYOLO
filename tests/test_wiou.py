"""WIoU v3: correct reduction to IoU loss behaviour and robustness of its running mean."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import wiou  # noqa: E402


def _boxes(n=8):
    torch.manual_seed(0)
    xy = torch.rand(n, 2) * 100
    wh = torch.rand(n, 2) * 50 + 5
    t = torch.cat([xy, xy + wh], 1)
    p = t + torch.randn(n, 4) * 3
    return p.requires_grad_(), t


def test_perfect_boxes_zero_loss():
    _, t = _boxes()
    wiou._RunningMean.value = None
    assert torch.allclose(1 - wiou.wiou_v3(t, t, xywh=False, CIoU=True), torch.zeros(len(t), 1), atol=1e-6)


def test_running_mean_survives_nan_batch():
    """Regression: one NaN batch poisoned the running mean, so the box loss stayed NaN after NaN recovery."""
    p, t = _boxes()
    wiou._RunningMean.value = None
    wiou.wiou_v3(p, t, xywh=False, CIoU=True)
    before = wiou._RunningMean.value.clone()
    bad = p.detach().clone()
    bad[0] = float("nan")
    wiou.wiou_v3(bad, t, xywh=False, CIoU=True)
    assert torch.isfinite(wiou._RunningMean.value) and torch.equal(wiou._RunningMean.value, before)
    loss = 1 - wiou.wiou_v3(p, t, xywh=False, CIoU=True)
    loss.sum().backward()
    assert torch.isfinite(loss).all() and torch.isfinite(p.grad).all()


def test_validation_does_not_update_mean():
    p, t = _boxes()
    wiou._RunningMean.value = None
    wiou.wiou_v3(p, t, xywh=False, CIoU=True)
    before = wiou._RunningMean.value.clone()
    with torch.no_grad():
        wiou.wiou_v3(p * 1.5, t, xywh=False, CIoU=True)
    assert torch.equal(wiou._RunningMean.value, before)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
