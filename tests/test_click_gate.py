"""G17 pieces (tools/click_gate.py): clicks fall inside their boxes; the pseudo-box is the best same-class detection
containing the click, else a median-size box centred on the click and clipped."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import click_gate as cg  # noqa: E402


def test_clicks_inside():
    gt = np.array([[0, 10, 20, 30, 60], [2, 100, 100, 101, 140]], float)
    c = cg.clicks(np.repeat(gt, 500, 0), np.random.default_rng(0))
    b = np.repeat(gt, 500, 0)
    assert ((c[:, 0] >= b[:, 1]) & (c[:, 0] <= b[:, 3]) & (c[:, 1] >= b[:, 2]) & (c[:, 1] <= b[:, 4])).all()
    print("ok test_clicks_inside")


def test_pseudo_box():
    pred = np.array([[0, 0, 50, 50, 0.9, 1],    # wrong class
                     [0, 0, 40, 40, 0.3, 0],    # contains the click
                     [5, 5, 30, 30, 0.6, 0],    # contains it, more confident
                     [100, 100, 120, 120, 0.99, 0],  # does not contain it
                     [0, 0, 40, 40, 0.04, 0]])  # below MIN_CONF
    box, det = cg.pseudo_box((20, 20), 0, pred, (10, 10), (640, 640))
    assert det and np.allclose(box, [5, 5, 30, 30])
    box, det = cg.pseudo_box((3, 300), 0, pred, (10, 20), (640, 640))
    assert not det and np.allclose(box, [0, 290, 8, 310])  # centred, clipped at x = 0
    print("ok test_pseudo_box")


if __name__ == "__main__":
    test_clicks_inside()
    test_pseudo_box()
