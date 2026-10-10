"""SS2D correctness: the chunked SSD scan must equal the step-by-step selective-scan recurrence."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.ss2d import SS2D, _merge_orders, _scan_orders, ssd_scan  # noqa: E402


def test_ssd_matches_recurrence():
    torch.manual_seed(0)
    b, L, h, p, n = 2, 150, 3, 4, 5  # L not a multiple of the chunk: exercises padding
    x = torch.randn(b, L, h, p, dtype=torch.float64)
    a = -torch.rand(b, L, h, dtype=torch.float64)
    B = torch.randn(b, L, n, dtype=torch.float64)
    C = torch.randn(b, L, n, dtype=torch.float64)
    state, ref = torch.zeros(b, h, p, n, dtype=torch.float64), []
    for t in range(L):
        state = state * torch.exp(a[:, t])[..., None, None] + x[:, t, :, :, None] * B[:, t, None, None, :]
        ref.append((state * C[:, t, None, None, :]).sum(-1))
    assert torch.allclose(ssd_scan(x, a, B, C, chunk=32), torch.stack(ref, 1), atol=1e-10)


def test_scan_orders_roundtrip():
    t = torch.randn(2, 5, 7, 3)
    assert torch.equal(_merge_orders(_scan_orders(t), 5, 7), 4 * t)


def test_ss2d_shapes_and_grads():
    m = SS2D(64)
    x = torch.randn(2, 9, 11, 64, requires_grad=True)
    y = m(x)
    y.sum().backward()
    assert y.shape == x.shape and torch.isfinite(x.grad).all()


def test_ss2d_half_precision_large_activations():
    """Regression: the scan output exceeded fp16 range and was cast before the norm -> inf -> NaN under AMP."""
    torch.manual_seed(0)
    m = SS2D(256).half()
    x = (torch.randn(1, 40, 40, 256) * 1000).half().requires_grad_()
    y = m(x)
    y.float().sum().backward()
    assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
