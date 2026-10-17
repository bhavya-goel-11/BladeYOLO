"""SAM box refinement: the cached SAM boxes are applied to the right detections, with the agreement gate and blend."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import sam_refine as sr  # noqa: E402


def test_scale_box():
    assert np.allclose(sr.scale_box(np.array([10.0, 20, 30, 60]), 1.5), [5, 10, 35, 70])


def test_refine_gate_and_blend():
    p = {"a": np.array([[0, 0, 100, 100, 0.9, 1],      # SAM agrees (IoU 0.64): refined
                        [200, 200, 300, 300, 0.6, 2],  # SAM box elsewhere: kept
                        [0, 0, 50, 50, 0.01, 0]])}     # below min_conf: never prompted, kept
    sb = np.zeros((2, len(sr.EXPAND), 6))
    sb[0, :, :4] = [10, 10, 90, 90]
    sb[1, :, :4] = [0, 0, 20, 20]
    out = sr.refine(p, {"a": sb}, min_conf=0.05, max_det=30, k=0, r=0.5, a=1.0)["a"]
    assert np.allclose(out[0, :4], [10, 10, 90, 90]) and np.allclose(out[1:, :4], p["a"][1:, :4])
    half = sr.refine(p, {"a": sb}, 0.05, 30, 0, 0.5, 0.5)["a"]
    assert np.allclose(half[0, :4], [5, 5, 95, 95])
    assert np.allclose(sr.refine(p, {"a": sb}, 0.05, 30, 0, 0.0, 1.0)["a"][1, :4], [0, 0, 20, 20])  # no gate
    assert np.allclose(p["a"][0, :4], [0, 0, 100, 100])  # input not modified


if __name__ == "__main__":
    test_scale_box()
    test_refine_gate_and_blend()
    print("ok test_sam_refine")
