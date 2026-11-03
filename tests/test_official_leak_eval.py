"""tools/official_leak_eval.py: the numpy matcher equals Ultralytics' own DetectionValidator.match_predictions, and the
Ultralytics-style and COCO-style scores of the same predictions agree within 1 point on WTBD-official test. The
logged Ultralytics `val` numbers of §12 (BladeYOLO-v2 0.840 / 0.493, YOLO12-L 0.704 / 0.388) are 1-2.5 points higher
than either: that gap comes from `val`'s own inference path (rectangular letterbox batches), not from the AP formula,
so it is printed, not asserted.
.venv/bin/python tests/test_official_leak_eval.py"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import numpy as np
import torch
from ultralytics.models.yolo.detect import DetectionValidator

import bootstrap_ci as bc
import detection_metrics as dm
import official_leak_eval as ole

rng = np.random.default_rng(0)
val = DetectionValidator()
val.iouv = torch.tensor(dm.IOU_THRS, dtype=torch.float32)
for _ in range(200):
    g = rng.integers(0, 5)
    p = rng.integers(0, 12)
    xy = rng.uniform(0, 500, (g, 2))
    gt = np.c_[rng.integers(0, 3, g), xy, xy + rng.uniform(10, 120, (g, 2))]
    j = rng.integers(0, max(g, 1), p)
    base = gt[j, 1:] if g else rng.uniform(0, 500, (p, 4))
    box = base + rng.normal(0, 12, (p, 4))
    box[:, 2:] = np.maximum(box[:, 2:], box[:, :2] + 1)
    pred = np.c_[box, rng.uniform(0, 1, p), rng.integers(0, 3, p)]
    ours = ole.ultra_match(gt, pred)
    if g and p:
        iou = torch.tensor(dm.iou_matrix(gt[:, 1:], pred[:, :4]), dtype=torch.float32)
        ref = val.match_predictions(torch.tensor(pred[:, 5]), torch.tensor(gt[:, 0]), iou).numpy()
        assert (ours == ref).all()
print("matcher identical to ultralytics on 200 random cases")

data = os.path.join(ROOT, "external", "WTBD-official")
imgs = bc.load_yolo_split(data, "test")
for run, logged in (("bladeyolo2_wtbd_official", (0.840, 0.493)), ("yolo12l_wtbd_official", (0.704, 0.388))):
    preds = ole.load_preds(run, data, "test")
    m50, m5095, _ = ole.ultra_score(ole.ultra_stats(imgs, preds), sorted(imgs), 6)
    c50, c5095 = bc.score(bc.per_image_matches(imgs, preds, 6), sorted(imgs), 6)[:2]
    print(f"{run}: ultra {m50:.3f} / {m5095:.3f}, coco {c50:.3f} / {c5095:.3f}, logged val {logged[0]:.3f} / "
          f"{logged[1]:.3f}")
    assert abs(m50 - c50) < 0.01 and abs(m5095 - c5095) < 0.01
# weighted bootstrap == scoring the resampled multiset directly
m = bc.per_image_matches(imgs, preds, 6)
files = sorted(imgs)
prep = ole.weighted_prep(m, files, 6)
assert np.allclose(ole.weighted_score(prep, np.ones(len(files)), 6), bc.score(m, files, 6)[:2], atol=1e-3)  # tie order of equal confidences
for _ in range(3):
    d = rng.integers(0, len(files), len(files))
    w = np.bincount(d, minlength=len(files)).astype(float)
    a, b = ole.weighted_score(prep, w, 6), bc.score(m, [files[i] for i in d], 6)[:2]
    assert np.allclose(a, b, atol=2e-3), (a, b)
print("weighted bootstrap matches resampled scoring")
print("ok")
