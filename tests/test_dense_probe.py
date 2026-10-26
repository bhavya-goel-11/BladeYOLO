"""Dense probe (tools/dense_probe.py, gates G8/G9): pixel targets follow the smallest covering box, and AP from score
histograms matches the exact average precision.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import dense_probe as dp  # noqa: E402


def test_pixel_targets_smallest_box_wins():
    gt = np.array([[1, 0, 0, 640, 640], [4, 100, 100, 200, 200]], float)
    t = dp.pixel_targets(gt)
    assert t[0, 0] == 2 and t[150 // dp.STRIDE, 150 // dp.STRIDE] == 5
    assert (t > 0).all()
    print("ok test_pixel_targets_smallest_box_wins")


def test_ap_from_histograms_matches_exact_ap():
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(0)
    target = rng.integers(0, 7, 5000)
    prob = rng.dirichlet(np.ones(6), 5000) * 0.5
    prob[np.arange(5000), np.maximum(target - 1, 0)] += 0.5 * (target > 0)
    h = dp.histograms(prob, target)
    for c in range(6):
        exact = average_precision_score(target == c + 1, prob[:, c])
        assert abs(dp.ap_from_hist(h[c]) - exact) < 0.01, (c, dp.ap_from_hist(h[c]), exact)
    print("ok test_ap_from_histograms_matches_exact_ap")


def test_hf_channels_shape_and_edge_response():
    grey = np.zeros((640, 640), np.float32)
    grey[:, 320:] = 1.0
    x = dp.hf_channels(grey)
    assert x.shape == (dp.SIDE, dp.SIDE, 7)
    assert x[80, 80, 1] > 10 * x[80, 10, 1]
    print("ok test_hf_channels_shape_and_edge_response")


if __name__ == "__main__":
    test_pixel_targets_smallest_box_wins()
    test_ap_from_histograms_matches_exact_ap()
    test_hf_channels_shape_and_edge_response()
