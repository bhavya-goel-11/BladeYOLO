"""The recovered transform must carry a box onto the same surface in a flipped, rotated and rescaled copy."""

import glob
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import error_analysis as ea  # noqa: E402
import view_label_audit as va  # noqa: E402


def test_projection_through_flip_rotation_scale():
    src = sorted(glob.glob(os.path.join(ROOT, "external", "WTBD-official", "train", "images", "*.jpg")))[0]
    a = cv2.imread(src)
    H, W = a.shape[:2]
    out = os.path.join(ROOT, "runs", "tests", "view_audit")
    os.makedirs(out, exist_ok=True)
    # b = horizontal flip of a, rotated 20 degrees about the centre and scaled 0.8, on a larger canvas
    R = cv2.getRotationMatrix2D((W / 2, H / 2), 20, 0.8)
    R[:, 2] += [100, 60]
    b = cv2.warpAffine(cv2.flip(a, 1), R, (W + 200, H + 120))
    pa, pb = os.path.join(out, "a.png"), os.path.join(out, "b.png")
    cv2.imwrite(pa, a)
    cv2.imwrite(pb, b)
    inl, fl, M, pts = va.transform_job((pa, pb))
    assert inl >= 25, inl
    box = np.array([[W * 0.3, H * 0.3, W * 0.6, H * 0.55]])
    got = va.project(box, (H, W), b.shape[:2], fl, M)[0]
    c = np.array([[box[0, 0], box[0, 1]], [box[0, 2], box[0, 1]], [box[0, 2], box[0, 3]], [box[0, 0], box[0, 3]]])
    c[:, 0] = W - 1 - c[:, 0]  # the flip
    c = np.c_[c, np.ones(4)] @ R.T
    want = np.array([c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max()])
    iou = ea.iou_matrix(got[None], want[None])[0, 0]
    assert iou > 0.9, (iou, got, want, fl)
    # a label on the matched texture is supported; one on blank surface is not
    s = max(H, W) / 480
    q = np.array(pts, float) * s
    tex = [*np.percentile(q, 25, axis=0), *np.percentile(q, 75, axis=0)]
    assert va.local_support(tex, (H, W), pts) >= va.LOCAL_MIN
    far = [q[:, 0].min() - 0.3 * W, 0, q[:, 0].min() - 0.15 * W, 0.1 * H]
    assert va.local_support(far, (H, W), pts) == 0, far
    print(f"ok test_projection_through_flip_rotation_scale (IoU {iou:.3f}, {inl} inliers, flip {fl})")


if __name__ == "__main__":
    test_projection_through_flip_rotation_scale()
