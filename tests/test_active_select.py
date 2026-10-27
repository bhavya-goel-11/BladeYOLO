"""H14 selection (tools/active_select.py): whole scenes are added within the 5% overshoot, start scenes are kept,
uncertainty follows the most confident detection, and scenes are ranked by mean uncertainty.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import active_select as al  # noqa: E402


def test_add_scenes_budget_and_start_kept():
    groups = [["a", "b"], ["c"], ["d", "e", "f"], ["g"], ["h", "i"]]
    sel = al.add_scenes([2, 0, 4, 1, 3], groups, [1], 4)
    assert sel[0] == 1 and sel == [1, 2]  # 1 + 3 = 4 images
    sel = al.add_scenes([4, 2, 0, 3], groups, [], 3)
    assert sel == [4, 3]  # scene 2 (3 images) would reach 5 > 3.15; scene 0 likewise; scene 3 completes 3
    print("ok test_add_scenes_budget_and_start_kept")


def test_uncertainty_and_ranking():
    assert al.uncertainty([]) == 1.0 and al.uncertainty([0.01, 0.04]) == 1.0
    assert abs(al.uncertainty([0.3, 0.9, 0.02]) - 0.1) < 1e-9
    groups = [["a"], ["b", "c"], ["d"]]
    order, score = al.rank_scenes(groups, [0, 1, 2], {"a": 0.2, "b": 0.9, "c": 0.5, "d": 0.7})
    assert order == [1, 2, 0] and abs(score[1] - 0.7) < 1e-9
    print("ok test_uncertainty_and_ranking")


if __name__ == "__main__":
    test_add_scenes_budget_and_start_kept()
    test_uncertainty_and_ranking()
