"""RandomRot90 keeps every box exactly on its object, for pixel and normalised boxes."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _check(normalized):
    from ultralytics.utils.instance import Instances

    from models.augment import RandomRot90

    h, w = 120, 200
    img = np.zeros((h, w, 3), np.uint8)
    boxes = np.array([[10, 20, 60, 50], [150, 70, 190, 115]], np.float32)  # xyxy pixels
    for v, (x1, y1, x2, y2) in enumerate(boxes.astype(int), 1):
        img[y1:y2, x1:x2] = 50 * v
    b = boxes / np.array([w, h, w, h], np.float32) if normalized else boxes.copy()
    ins = Instances(b, segments=np.zeros((0, 1000, 2), np.float32), bbox_format="xyxy", normalized=normalized)
    out = RandomRot90(p=1.0)({"img": img, "instances": ins})
    im, ins = out["img"], out["instances"]
    assert im.shape[:2] == (w, h)
    ins.convert_bbox("xyxy")
    bb = ins.bboxes * (np.array([h, w, h, w]) if normalized else 1)
    for v, (x1, y1, x2, y2) in enumerate(np.round(bb).astype(int), 1):
        region = im[y1:y2, x1:x2, 0]
        assert region.size and (region == 50 * v).all(), f"box {v} not on its object after rotation"
        assert (im[..., 0] == 50 * v).sum() == region.size, f"box {v} does not cover its whole object"


def test_rot90_pixel_boxes():
    _check(normalized=False)


def test_rot90_normalized_boxes():
    _check(normalized=True)


if __name__ == "__main__":
    test_rot90_pixel_boxes()
    test_rot90_normalized_boxes()
    print("ok test_augment")
