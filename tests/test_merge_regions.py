"""Region-merging convention: overlapping same-class blobs merge, a diagonal staircase along an edge does not."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
from merge_regions import merge_labels  # noqa: E402


def px(c, x1, y1, x2, y2, W=640, H=640):
    return [c, (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H]


def test_overlapping_blobs_merge():
    rows = [px(1, 100, 100, 200, 200), px(1, 180, 110, 290, 210)]
    out, merged = merge_labels(rows, 640, 640, gap=8, min_fill=0.6)
    assert len(out) == 1 and merged == [(1, 2)]


def test_diagonal_staircase_kept():
    rows = [px(0, 40 * i, 40 * i, 40 * i + 50, 40 * i + 50) for i in range(6)]  # boxes along a 45 degree edge
    out, merged = merge_labels(rows, 640, 640, gap=8, min_fill=0.6)
    assert len(out) == 6 and not merged


def test_other_class_and_far_boxes_untouched():
    rows = [px(1, 100, 100, 200, 200), px(2, 150, 150, 250, 250), px(1, 400, 400, 450, 450)]
    out, merged = merge_labels(rows, 640, 640, gap=8, min_fill=0.6)
    assert len(out) == 3 and not merged


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
