"""Alignment similarity separates an augmented copy from a different image."""

import glob
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import view_label_audit as va  # noqa: E402
import view_pool as vp  # noqa: E402


def test_copy_aligns_and_other_image_does_not():
    imgs = sorted(glob.glob(os.path.join(ROOT, "external", "WTBD-official", "train", "images", "*.jpg")))
    a = cv2.imread(imgs[0])
    H, W = a.shape[:2]
    out = os.path.join(ROOT, "runs", "tests", "view_pool")
    os.makedirs(out, exist_ok=True)
    R = cv2.getRotationMatrix2D((W / 2, H / 2), 15, 0.9)
    copy = cv2.convertScaleAbs(cv2.warpAffine(cv2.flip(a, 1), R, (W, H)), alpha=1.2, beta=15)  # flip, rotate, brighter
    pa, pc = os.path.join(out, "a.png"), os.path.join(out, "copy.png")
    cv2.imwrite(pa, a)
    cv2.imwrite(pc, copy)
    inl, fl, M, _ = va.transform_job((pa, pc))
    z, zg, ov = vp.pair_job((pa, pc, fl, M))
    assert z > 0.9 and zg > 0.8 and ov > 0.3, (z, zg, ov)
    other = imgs[len(imgs) // 2]  # a different image forced through the same transform
    z2, zg2, _ = vp.pair_job((other, pc, fl, M))
    assert z2 < 0.6 and zg2 < 0.5, (z2, zg2)
    print(f"ok test_copy_aligns_and_other_image_does_not (copy {z:.3f}/{zg:.3f}, other {z2:.3f}/{zg2:.3f})")


if __name__ == "__main__":
    test_copy_aligns_and_other_image_does_not()
