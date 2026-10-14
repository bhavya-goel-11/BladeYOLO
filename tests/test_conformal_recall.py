"""Coverage-based flagging and the conformal threshold on synthetic data."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import conformal_recall as cr  # noqa: E402


def test_flag_conf():
    gt = np.array([[0, 0, 0, 100, 100]], float)
    full = np.array([[0, 0, 100, 100, 0.8, 3]])  # any class counts
    part = np.array([[0, 0, 40, 100, 0.9, 0]])  # 40% only
    two = np.array([[0, 0, 40, 100, 0.9, 0], [40, 0, 100, 100, 0.3, 1]])  # 40% + 60%: flagged once both are on
    assert cr.flag_conf(gt, full)[0] == 0.8 and cr.flag_conf(gt, part)[0] == 0 and cr.flag_conf(gt, two)[0] == 0.3
    print("ok test_flag_conf")


def test_calibration_controls_risk():
    rng = np.random.default_rng(0)
    pool = [(rng.uniform(0, 1, 3), np.zeros(0)) for _ in range(20000)]  # defects' flag thresholds ~ U(0, 1)
    grid = np.linspace(0, 1, 1001)
    misses = []
    for _ in range(200):
        cal = [pool[i] for i in rng.choice(len(pool), 100, replace=False)]
        lam = cr.calibrate(cal, 0.1, grid)
        misses.append(cr.loss(pool, lam).mean())
    assert np.mean(misses) <= 0.1 + 0.005, np.mean(misses)  # the guarantee holds in expectation
    assert np.mean(misses) > 0.07  # and is not trivially conservative
    print(f"ok test_calibration_controls_risk (mean miss {np.mean(misses):.3f} at alpha 0.1)")


if __name__ == "__main__":
    test_flag_conf()
    test_calibration_controls_risk()
