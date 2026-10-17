"""Real-view consistency: the pixel maps that move pseudo-labels between views must be exact."""

import csv
import glob
import json
import os
import random
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import view_label_audit as va  # noqa: E402
from models import view_consistency as vc  # noqa: E402

OUT = os.path.join(ROOT, "runs", "tests", "view_consistency")


def marker_image(h=500, w=700, box=(400, 120, 460, 170)):
    """Grey noise (equal channels, like the letterbox border) with a pure green marker box."""
    img = np.repeat(np.random.default_rng(0).integers(0, 60, (h, w, 1), dtype=np.uint8), 3, axis=2)
    x1, y1, x2, y2 = box
    img[y1:y2, x1:x2] = (0, 255, 0)
    return img, np.array([box], float)


def bright_box(img):
    """Bounding box of the green marker: green well above red and blue (grey pixels have equal channels)."""
    g = img[..., 1].astype(int) - np.maximum(img[..., 0], img[..., 2]).astype(int)
    ys, xs = np.nonzero(g > 60)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], float)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def test_audit_matrix_matches_projection():
    os.makedirs(OUT, exist_ok=True)
    src = sorted(glob.glob(os.path.join(ROOT, "external", "WTBD-official", "train", "images", "*.jpg")))[0]
    a = cv2.imread(src)
    H, W = a.shape[:2]
    R = cv2.getRotationMatrix2D((W / 2, H / 2), 25, 0.85)
    b = cv2.warpAffine(cv2.flip(a, 0), R, (W + 100, H + 80))
    pa, pb = os.path.join(OUT, "a.png"), os.path.join(OUT, "b.png")
    cv2.imwrite(pa, a)
    cv2.imwrite(pb, b)
    _, fl, M, _ = va.transform_job((pa, pb))
    box = np.array([[W * 0.2, H * 0.3, W * 0.5, H * 0.6]])
    want = va.project(box, (H, W), b.shape[:2], fl, M)
    got = vc.move_boxes(box, vc.audit_matrix((H, W), b.shape[:2], fl, M))
    assert np.allclose(got, want, atol=1e-6), (got, want)
    print("ok test_audit_matrix_matches_projection")


def test_synthetic_view_map():
    img, box = marker_image()
    for seed in range(20):
        out, T = vc.synthetic_view(img, random.Random(seed))
        moved = vc.move_boxes(box, T)[0]
        h, w = out.shape[:2]
        if moved[0] < 0 or moved[1] < 0 or moved[2] > w or moved[3] > h:  # shifted (partly) out of the frame
            continue
        found = bright_box(out)
        assert iou(moved, found) > 0.85, (seed, moved, found)
    print("ok test_synthetic_view_map")


def test_dataset_letterbox_chain():
    os.makedirs(OUT, exist_ok=True)
    img, box = marker_image()
    p = os.path.join(OUT, "marker.png")
    cv2.imwrite(p, img)
    pairs = os.path.join(OUT, "pairs.csv")
    with open(pairs, "w") as f:
        w = csv.writer(f)
        w.writerow(["image_a", "image_b", "same_photo", "flip", "M"])
        w.writerow([p, p, 0, "None", json.dumps([[1, 0, 0], [0, 1, 0]])])
    ds = vc.PairDataset("synthetic", pairs, size=640, seed=3)
    for i in range(5):
        ds.epoch = i
        A, B, T, mask, _ = ds[0]
        A, B, T = A.numpy().transpose(1, 2, 0), B.numpy().transpose(1, 2, 0), T.numpy().astype(np.float64)
        a_box, b_box = bright_box(A), bright_box(B)
        assert iou(vc.move_boxes(a_box[None], T)[0], b_box) > 0.8
    ds_real = vc.PairDataset("real", pairs, size=640)  # identity transform between two letterboxed copies
    A, B, T, mask, grads = ds_real[0]
    assert np.allclose(T.numpy(), np.eye(3), atol=1e-5) and mask.numpy().mean() > 0.5
    g = grads.float().numpy()
    assert vc.local_zncc(g[0], g[1], (300, 200, 400, 300)) > 0.99  # identical views agree everywhere
    print("ok test_dataset_letterbox_chain")


def test_transfer_filters():
    T = np.eye(3)
    mask = np.zeros((640, 640), bool)
    mask[100:500, 100:500] = True
    det = np.array([[150, 150, 250, 250, 0.9, 2],   # inside: kept
                    [150, 150, 250, 250, 0.3, 1],   # below tau: dropped
                    [450, 450, 600, 600, 0.9, 0],   # mostly outside the valid area: dropped
                    [300, 300, 302, 400, 0.9, 3]])  # too thin: dropped
    lab, drop = vc.transfer(det, T, mask, 640, tau=0.5)
    assert lab.shape == (1, 5) and lab[0, 0] == 2
    assert drop.tolist() == [[450, 450, 500, 500]]  # the half-outside box, clipped to the view: ignored (the 2 px one is negligible)
    assert np.allclose(lab[0, 1:], [200 / 640, 200 / 640, 100 / 640, 100 / 640])
    print("ok test_transfer_filters")


def test_local_check_drops_moved_content():
    """A box whose content differs between warped A and B (a blade that turned between frames) is not moved."""
    rng = np.random.default_rng(1)
    ga = rng.random((640, 640)).astype(np.float32)
    gb = ga.copy()
    gb[300:400, 300:400] = rng.random((100, 100))  # content changed inside the second box only
    det = np.array([[100, 100, 200, 200, 0.9, 0], [300, 300, 400, 400, 0.9, 1]])
    mask = np.ones((640, 640), bool)
    lab, drop = vc.transfer(det, np.eye(3), mask, 640, 0.5, np.stack([ga, gb]))
    assert lab.shape == (1, 5) and lab[0, 0] == 0, lab
    assert drop.tolist() == [[300, 300, 400, 400]]  # ignored in B, not trained as background
    print("ok test_local_check_drops_moved_content")


def test_unlabelled_image_list():
    """An image list (training images without labels) gives synthetic pairs; it has no second views for real mode."""
    os.makedirs(OUT, exist_ok=True)
    img, _ = marker_image()
    p = os.path.join(OUT, "u.png")
    cv2.imwrite(p, img)
    lst = os.path.join(OUT, "unlabelled.txt")
    open(lst, "w").write(p + "\n")
    ds = vc.PairDataset("synthetic", lst, size=320)
    A, B, T, mask, _ = ds[0]
    assert len(ds) == 1 and tuple(A.shape) == (3, 320, 320)
    try:
        vc.PairDataset("real", lst)
        raise AssertionError("real mode accepted an image list")
    except ValueError:
        pass
    print("ok test_unlabelled_image_list")


def test_class_balanced_thresholds():
    thr = vc.class_thresholds([100, 50, 10, 0], tau=0.5, tau_min=0.25)
    assert abs(thr[0] - 0.5) < 1e-9  # the class the teacher labels most keeps tau
    assert abs(thr[1] - 0.5 * 0.5 / 1.5) < 1e-9 or thr[1] == 0.25  # convex mapping, floored
    assert thr[2] == thr[3] == 0.25 and (np.diff(thr) <= 0).all()  # rarely / never labelled classes: the floor
    assert (vc.class_thresholds([0, 0], 0.5, 0.25) == 0.5).all()  # no counts yet: tau everywhere
    det = np.array([[100, 100, 200, 200, 0.4, 0], [300, 300, 400, 400, 0.4, 3]])
    lab, _ = vc.transfer(det, np.eye(3), np.ones((640, 640), bool), 640, thr)
    assert lab[:, 0].tolist() == [3]  # 0.4 passes the rare class's threshold (0.25), not the common one's (0.5)
    print("ok test_class_balanced_thresholds")


if __name__ == "__main__":
    test_audit_matrix_matches_projection()
    test_synthetic_view_map()
    test_dataset_letterbox_chain()
    test_transfer_filters()
    test_local_check_drops_moved_content()
    test_unlabelled_image_list()
    test_class_balanced_thresholds()
