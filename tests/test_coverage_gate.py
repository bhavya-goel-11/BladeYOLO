"""G16 pieces (tools/coverage_gate.py): same-class proximity, the per-box outcome, the two-way fixed-effects slope and
patch coverage."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import coverage_gate as cg  # noqa: E402


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def test_proximity_same_class_only():
    bank = unit([[1, 0], [0, 1], [1, 1]])
    q = unit([[1, 0], [0, 1]])
    x = cg.proximity(q, np.array([0, 2]), bank, np.array([1, 0, 0]))
    assert abs(x[0] - np.sqrt(0.5)) < 1e-9  # class 0 bank: [0,1] and [1,1]; best is [1,1]
    assert np.isnan(x[1])  # class 2 absent
    x3 = cg.proximity(q[:1], np.array([0]), bank, np.array([1, 0, 0]), k=3)
    assert abs(x3[0] - (0 + np.sqrt(0.5)) / 2) < 1e-9  # top-k capped at the 2 available
    print("ok test_proximity_same_class_only")


def test_outcome():
    gt = np.array([[0, 0, 0, 10, 10], [1, 20, 20, 30, 30]], float)
    pred = np.array([[0, 0, 10, 10, 0.9, 1], [0, 0, 10, 10, 0.4, 0], [0, 0, 6, 10, 0.8, 0], [20, 20, 30, 30, 0.7, 1]])
    assert np.allclose(cg.outcome(gt, pred), [0.8, 0.7])  # 0.9 is the wrong class; the [0,0,6,10] box has IoU 0.6
    assert np.allclose(cg.outcome(gt, pred, any_class=True), [0.9, 0.7])
    assert np.allclose(cg.outcome(gt, pred[:, :].copy() * [1, 1, 1, 1, 1, 0] + [0, 0, 0, 0, 0, 5]), [0, 0])
    print("ok test_outcome")


def test_fe_slope_recovers_coefficient():
    rng = np.random.default_rng(0)
    nb, ns = 60, 8
    box, sub = np.repeat(np.arange(nb), ns), np.tile(np.arange(ns), nb)
    x = rng.normal(size=nb * ns) + 0.5 * sub  # x correlated with subset quality
    n = rng.integers(1, 30, nb * ns)
    y = 0.3 * x - 0.1 * np.log1p(n) + rng.normal(size=nb)[box] + 1.0 * sub + 0.01 * rng.normal(size=nb * ns)
    b = cg.fe_slope(y, np.c_[x, np.log1p(n)], box, sub)
    assert abs(b[0] - 0.3) < 0.01 and abs(b[1] + 0.1) < 0.01, b
    print("ok test_fe_slope_recovers_coefficient")


def test_coverage():
    C = unit([[1, 0], [0, 1]])
    assert abs(cg.coverage(C, unit([[1, 0]])) - 0.5) < 1e-9
    assert abs(cg.coverage(C, unit([[1, 0], [0, 1]])) - 1.0) < 1e-9
    print("ok test_coverage")


if __name__ == "__main__":
    test_proximity_same_class_only()
    test_outcome()
    test_fe_slope_recovers_coefficient()
    test_coverage()
