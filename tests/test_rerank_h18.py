"""H18 pieces (tools/rerank_h18.py): fractional AP50-95 targets under greedy matching, the feature matrix, and the
fractional-target logistic fit."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import rerank_h18 as h  # noqa: E402


def test_targets():
    gt = np.array([[0, 0, 0, 10, 10]], float)
    pred = np.array([[0, 0, 10, 10, 0.5, 0],   # exact: TP at every threshold
                     [0, 0, 10, 10, 0.9, 1],   # wrong class
                     [0, 0, 10, 10, 0.3, 0],   # duplicate, lower confidence: never matched
                     [0, 0, 10, 8, 0.2, 0]])
    t = h.targets(gt, pred)
    assert np.allclose(t[:3], [1, 0, 0]) and t[3] == 0
    t2 = h.targets(gt, np.array([[0, 0, 10, 8.2, 0.9, 0]]))  # IoU 0.82: TP at 0.50..0.80 = 7 of 10 thresholds
    assert abs(t2[0] - 0.7) < 1e-9, t2
    print("ok test_targets")


def test_features():
    bank = np.eye(3)[[0, 0, 1]].astype(float)
    bank_cls = np.array([0, 0, 1])
    pred = np.array([[0, 0, 1, 1, 0.5, 0], [0, 0, 1, 1, 0.5, 2]])
    emb = np.eye(3)[[0, 2]].astype(float)
    X = h.features(pred, emb, bank, bank_cls, nc=3)
    assert X.shape == (2, 10)
    assert abs(X[0, 0]) < 1e-9 and abs(X[0, 2] - 1) < 1e-9 and X[0, 6] == 0  # logit(0.5) = 0; cos 1; class present
    assert X[1, 6] == 1 and X[1, 1] == 0 and X[1, 2] == 0  # class 2 has no labelled boxes
    assert abs(X[0, 3] - np.log(3)) < 1e-9 and X[1, 9] == 1
    print("ok test_features")


def test_fit():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 2))
    t = 1 / (1 + np.exp(-(2 * X[:, 0])))
    f = h.fit(X, t, C=100)
    assert np.corrcoef(f(X), t)[0, 1] > 0.99
    print("ok test_fit")


if __name__ == "__main__":
    test_targets()
    test_features()
    test_fit()
