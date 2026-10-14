"""Granularity-tolerant matching and region AP on hand-built cases."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import granularity_eval as ge  # noqa: E402


def test_one_box_over_two_segments():
    g = np.array([[0, 0, 100, 50], [100, 0, 200, 50]], float)  # one crack labelled as two segments
    p = np.array([[0, 0, 200, 50, 0.9]])  # predicted as one region
    assert ge.match_image(g, p, 0.5, tolerant=False) == [(0, 0, True)]  # standard: false positive, both missed
    assert ge.match_image(g, p, 0.5, tolerant=True) == [(2, 1, True)]  # tolerant: both segments found
    print("ok test_one_box_over_two_segments")


def test_fragment_of_found_defect_ignored():
    g = np.array([[0, 0, 200, 50]], float)
    p = np.array([[0, 0, 190, 50, 0.9], [0, 0, 80, 50, 0.8]])  # found, plus a fragment inside it
    assert ge.match_image(g, p, 0.5, tolerant=False) == [(1, 1, True), (0, 0, True)]
    assert ge.match_image(g, p, 0.5, tolerant=True) == [(1, 1, True), (0, 0, False)]
    print("ok test_fragment_of_found_defect_ignored")


def test_exact_match_and_background_identical():
    g = np.array([[10, 10, 60, 60]], float)
    p = np.array([[10, 10, 60, 60, 0.9], [300, 300, 350, 350, 0.5]])
    assert ge.match_image(g, p, 0.5, False) == ge.match_image(g, p, 0.5, True) == [(1, 1, True), (0, 0, True)]
    print("ok test_exact_match_and_background_identical")


def test_region_ap_perfect_and_split_invariant():
    imgs = {"a": dict(gt=np.array([[0, 0, 0, 200, 40], [0, 200, 0, 400, 40]], float))}
    one = {"a": np.array([[0, 0, 400, 40, 0.9, 0]])}  # same region as one box
    s = ge.scores(ge.per_image(imgs, one, 1, np.array([0.5])), ["a"], 1)
    assert abs(s["region"][0] - 1.0) < 1e-9
    assert s["standard50"][0] == 0.0 and abs(s["tolerant50"][0] - 1.0) < 1e-9
    s2 = ge.scores(ge.per_image(imgs, one, 1, np.array([0.5])), ["a", "a"], 1)  # a bootstrap draw repeating the image
    assert abs(s2["tolerant50"][0] - 1.0) < 1e-9 and abs(s2["region"][0] - 1.0) < 1e-9
    print("ok test_region_ap_perfect_and_split_invariant")


if __name__ == "__main__":
    test_one_box_over_two_segments()
    test_fragment_of_found_defect_ignored()
    test_exact_match_and_background_identical()
    test_region_ap_perfect_and_split_invariant()
