"""Retrieval prior (train.py --retrieval, models/retrieval.py): the cap keeps a query's own stored patches out, votes
are re-weighted for the background subsampling, and the memory is built from labelled images, used by DinoFuse and
carried by checkpoints. Uses the randomly initialised ViT-S/16 of test_model.py, so it checks wiring, not accuracy.
"""

import io
import os
import sys

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from test_model import _fake_dinov3_weights  # noqa: E402

from models.retrieval import build_memory, patch_classes, retrieval_prior  # noqa: E402

OUT = os.path.join(ROOT, "runs", "tests", "retrieval")


def test_patch_classes():
    lab = patch_classes(np.array([[2, 0, 0, 32, 32], [4, 0, 0, 16, 16]]), 3, 3)  # smaller box wins where they overlap
    assert lab.reshape(3, 3).tolist() == [[4, 2, -1], [2, 2, -1], [-1, -1, -1]]
    print("ok test_patch_classes")


def test_votes_and_cap():
    """Memory: 2 class-1 patches near e0, 1 background patch at e0 itself. A query at e0 (cosine 1 with the background
    patch) must not see that patch under cap 0.99, and background votes count w_bg times."""
    e = torch.eye(4)
    near = torch.nn.functional.normalize(e[0] + 0.3 * e[1], dim=0)
    mem = torch.stack([near, near, e[0]])
    cls = torch.tensor([1, 1, -1], dtype=torch.int8)
    q = e[0].view(1, 4, 1, 1)
    capped = retrieval_prior(q, mem, cls, w_bg=3.0, nc=2, cap=0.99, k=10)
    assert torch.allclose(capped.flatten(), torch.tensor([0.0, 1.0]))
    open_ = retrieval_prior(q, mem, cls, w_bg=3.0, nc=2, cap=1.01, k=10)
    assert torch.allclose(open_.flatten(), torch.tensor([0.0, 2 / 5]))  # 2 defect votes vs 1 background vote x 3
    assert torch.allclose(retrieval_prior(q, mem, cls, 3.0, 2, cap=-1, k=10), torch.zeros(1, 2, 1, 1))  # nothing left
    print("ok test_votes_and_cap")


def _dataset():
    """Two 64x96 images in images/labels layout, one box each."""
    rng = np.random.default_rng(0)
    imgs = []
    for i, row in enumerate(("1 0.25 0.5 0.4 0.6", "0 0.7 0.3 0.2 0.3")):
        for d in ("images", "labels"):
            os.makedirs(os.path.join(OUT, "data", d), exist_ok=True)
        p = os.path.join(OUT, "data", "images", f"{i}.jpg")
        cv2.imwrite(p, rng.integers(0, 255, (64, 96, 3), dtype=np.uint8))
        open(os.path.join(OUT, "data", "labels", f"{i}.txt"), "w").write(row + "\n")
        imgs.append(p)
    return imgs


def test_memory_model_and_checkpoint():
    _fake_dinov3_weights()
    mem_path = os.path.join(OUT, "memory.pt")
    mem = build_memory(_dataset(), nc=2, size="s", out=mem_path, imgsz=96, bg_max=10)
    n_def = int((mem["cls"] >= 0).sum())
    assert n_def > 0 and len(mem["cls"]) == n_def + 10 and mem["w_bg"] > 1 and mem["feat"].dtype == torch.float16
    assert set(mem["cls"].tolist()) == {-1, 0, 1}

    import models  # noqa: F401
    from ultralytics.nn.tasks import DetectionModel

    from models.dino_fusion import DinoFuse

    os.environ.update(BLADEYOLO_RETRIEVAL=mem_path, BLADEYOLO_RETRIEVAL_NC="2")
    try:
        m = DetectionModel(os.path.join(ROOT, "bladeyolo2-l.yaml"), nc=2, verbose=False).eval()
    finally:
        os.environ.pop("BLADEYOLO_RETRIEVAL"), os.environ.pop("BLADEYOLO_RETRIEVAL_NC")
    fuses = [f for f in m.model if isinstance(f, DinoFuse)]
    assert fuses and all(f.retr_nc == 2 for f in fuses)
    assert "model.0.mem_feat" in m.state_dict() and torch.equal(m.model[0].mem_feat, mem["feat"])
    x = torch.rand(1, 3, 128, 160)
    y0 = m.model[0](x)
    assert y0.retr.shape == (1, 2, 8, 10) and (y0.retr >= 0).all() and (y0.retr.sum(1) <= 1 + 1e-5).all()
    for f in fuses:  # zero-init output: perturb, as one optimiser step would, so the prior changes the result
        torch.nn.init.normal_(f.out.weight, std=0.01)
    with torch.no_grad():
        ref = m(x)[0]
    buf = io.BytesIO()
    torch.save(m, buf)
    buf.seek(0)
    m2 = torch.load(buf, weights_only=False)  # checkpoints pickle the model: the memory must come with it
    with torch.no_grad():
        assert torch.equal(ref, m2(x)[0])
        m2.model[0].mem_cls.fill_(-1)  # all background: the prior goes to zero and the output changes
        assert not torch.equal(ref, m2(x)[0])
    print("ok test_memory_model_and_checkpoint")


if __name__ == "__main__":
    test_patch_classes()
    test_votes_and_cap()
    test_memory_model_and_checkpoint()
