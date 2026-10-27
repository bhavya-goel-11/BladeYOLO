"""G15 rescoring (tools/proto_rescore.py): box embeddings average the right patches, class probabilities follow the
nearest labelled boxes, and rescoring keeps classes and leaves classes without labelled boxes untouched.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import proto_rescore as pr  # noqa: E402


def test_box_embed_means_patches_inside():
    rng = np.random.default_rng(0)
    fmap = rng.normal(size=(pr.GRID, pr.GRID, 8)).astype(np.float32)
    e = pr.box_embed(fmap, [[32, 48, 80, 96], [100, 100, 104, 104]])
    # patch centres 16i + 8 inside [32, 80] x [48, 96]: columns 2..4, rows 3..5
    ref = fmap[3:6, 2:5].reshape(-1, 8).mean(0)
    assert np.allclose(e[0], ref / np.linalg.norm(ref), atol=1e-5)
    ref = fmap[6, 6]  # tiny box: the patch containing its centre (102 // 16 = 6)
    assert np.allclose(e[1], ref / np.linalg.norm(ref), atol=1e-5)
    print("ok test_box_embed_means_patches_inside")


def test_class_probs_and_rescore():
    bank = np.eye(4, 8, dtype=np.float32)  # boxes of classes 0, 1, 3, 3
    cls = np.array([0, 1, 3, 3])
    det = np.eye(4, 8, dtype=np.float32)[[0, 2]]
    p = pr.class_probs(det, bank, cls)
    assert p[0, 0] > 0.99 and np.isnan(p[:, [2, 4, 5]]).all()
    assert abs(np.nansum(p[1]) - 1) < 1e-6
    pred = np.array([[0, 0, 10, 10, 0.64, 0], [0, 0, 10, 10, 0.5, 4], [0, 0, 10, 10, 0.5, 1]])
    out = pr.rescore(pred, np.eye(4, 8, dtype=np.float32)[[0, 1, 2]], bank, cls)
    assert np.isclose(out[0, 4], 0.64 ** 0.5 * p[0, 0] ** 0.5, atol=1e-4)
    assert out[1, 4] == 0.5  # class 4 has no labelled boxes: unchanged
    assert out[2, 4] < 0.5 and (out[:, 5] == pred[:, 5]).all()
    print("ok test_class_probs_and_rescore")


if __name__ == "__main__":
    test_box_embed_means_patches_inside()
    test_class_probs_and_rescore()
