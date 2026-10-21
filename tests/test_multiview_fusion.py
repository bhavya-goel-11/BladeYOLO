"""Multi-view inference: the transform matches the label audit's point mapping, and fused scores average over the
views that see a location."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import multiview_fusion as mf  # noqa: E402
import view_label_audit as vla  # noqa: E402


def test_affine_matches_label_audit():
    rng = np.random.default_rng(0)
    M = [[0.9, -0.2, 30.0], [0.2, 0.9, -12.0]]
    pts = rng.uniform(0, 1000, (20, 2))
    for fl in (None, 1, 0, -1):
        A = mf.affine((1024, 768), (900, 1200), fl, M)
        ref = vla.move(pts, (1024, 768), (900, 1200), fl, M)
        assert np.allclose(np.c_[pts, np.ones(20)] @ A[:2].T, ref), fl
    print("ok test_affine_matches_label_audit")


def test_fuse_scores():
    """b and a: a is b shifted 300 px right (a's x = b's x - 300), so a sees only b's x >= 300."""
    A_ab = np.array([[1, 0, 300], [0, 1, 0], [0, 0, 1.0]])  # a -> b
    preds = {"b": np.array([[400, 100, 450, 150, 0.8, 1],    # seen by both; a also finds it (0.4)
                            [50, 100, 100, 150, 0.7, 1]]),   # only b sees it
             "a": np.array([[100, 100, 150, 150, 0.4, 1],    # = b's first box
                            [200, 300, 260, 360, 0.6, 2]])}  # b missed it (b's 500..560)
    graph = {"b": [("a", A_ab, "other")], "a": [("b", np.linalg.inv(A_ab), "other")]}
    out = mf.fuse(preds, graph, iou=0.5, w=1.0, geom="own", partners="other")["b"]
    by = {tuple(np.round(r[:4])): r[4] for r in out}
    assert np.isclose(by[(400, 100, 450, 150)], (0.8 + 0.4) / 2)
    assert np.isclose(by[(50, 100, 100, 150)], 0.7)  # a cannot see it: not diluted
    assert np.isclose(by[(500, 300, 560, 360)], 0.6 / 2)  # recovered from a, diluted by b's miss
    half = mf.fuse(preds, graph, 0.5, 0.5, "own", "other")["b"]
    assert np.isclose(half[np.all(np.round(half[:, :4]) == [400, 100, 450, 150], 1), 4][0], (0.8 + 0.5 * 0.4) / 1.5)
    assert len(mf.fuse(preds, graph, 0.5, 1.0, "own", "all")["b"]) == 3
    assert np.allclose(mf.fuse(preds, {"b": [("a", A_ab, "same")]}, 0.5, 1.0, "own", "other")["b"], preds["b"])
    print("ok test_fuse_scores")


if __name__ == "__main__":
    test_affine_matches_label_audit()
    test_fuse_scores()
