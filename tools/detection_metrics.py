"""Box metrics shared by the evaluation tools: IoU, 101-point COCO AP, and a run's inference size and device."""

import os

import numpy as np

IOU_THRS = np.linspace(0.5, 0.95, 10)


def iou_matrix(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), -1)
    area = lambda x: np.prod(x[:, 2:] - x[:, :2], -1)  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def ap_from(tp, conf, n_gt):
    """101-point interpolated AP (COCO), tp: (n,) bool, conf: (n,)."""
    if n_gt == 0:
        return float("nan")
    o = np.argsort(-conf)
    tpc = np.cumsum(tp[o])
    rec, prec = tpc / n_gt, tpc / np.arange(1, len(tp) + 1)
    prec = np.maximum.accumulate(prec[::-1])[::-1] if len(prec) else prec
    x = np.linspace(0, 1, 101)
    idx = np.searchsorted(rec, x, side="left")
    return float(np.mean([prec[i] if i < len(prec) else 0 for i in idx]))


def run_imgsz(run):
    """Inference size of a run: the size it was trained at (args.yaml), default 640. Boxes are always reported in
    640-input pixels, whatever the inference size."""
    import yaml

    path = os.path.join(run, "args.yaml")
    return int(yaml.safe_load(open(path)).get("imgsz", 640)) if os.path.exists(path) else 640


def device():
    import torch

    return "0" if torch.cuda.is_available() else "cpu"
