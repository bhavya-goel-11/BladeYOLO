"""Training augmentation for drone images of blades, which have no canonical orientation."""

import random

import numpy as np


class RandomRot90:
    """Rotate the image and its boxes by 90 degrees (counter-clockwise) with probability p.

    Together with Ultralytics' horizontal and vertical flips this covers all 8 orientations of the image square. Unlike
    the arbitrary-angle `degrees` augmentation, a quarter turn maps an axis-aligned box onto an exactly tight
    axis-aligned box, so it adds orientation variety without loosening the box targets.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, labels):
        if random.random() >= self.p:
            return labels
        img = labels["img"]
        h, w = img.shape[:2]
        labels["img"] = np.ascontiguousarray(np.rot90(img))  # pixel (x, y) -> (y, w - x); new size h x w -> w x h
        ins = labels["instances"]
        ins.convert_bbox(format="xywh")
        W = 1.0 if ins.normalized else w
        b = ins.bboxes.copy()
        ins._bboxes.bboxes = np.stack([b[:, 1], W - b[:, 0], b[:, 3], b[:, 2]], 1).astype(b.dtype) if len(b) else b
        if ins.segments is not None and len(ins.segments):
            s = ins.segments
            ins.segments = np.stack([s[..., 1], W - s[..., 0]], -1).astype(s.dtype)
        labels["instances"] = ins
        return labels


def add_rot90(dataset, p):
    """Insert RandomRot90 before the final Format step of a training dataset, now and whenever its transforms are
    rebuilt (Ultralytics rebuilds them when mosaic is switched off for the last epochs)."""
    build = dataset.build_transforms

    def build_with_rot90(hyp):
        t = build(hyp)
        if dataset.augment:
            t.insert(len(t.transforms) - 1, RandomRot90(p))
        return t

    dataset.build_transforms = build_with_rot90
    if dataset.augment:
        dataset.transforms.insert(len(dataset.transforms.transforms) - 1, RandomRot90(p))
    return dataset
