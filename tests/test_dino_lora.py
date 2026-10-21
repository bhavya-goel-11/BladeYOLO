"""LoRA adaptation of DINOv3 (train.py --dino-lora): starts as the frozen model, survives the trainer's freeze rule,
receives gradient, and stays wired after the copies Ultralytics makes (EMA deepcopy, checkpoint pickle).

Uses the randomly initialised ViT-S/16 of test_model.py, so it checks wiring, not accuracy.
"""

import copy
import io
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from test_model import _fake_dinov3_weights  # noqa: E402

RANK = 4


def _build(rank, cls="DinoConv"):
    _fake_dinov3_weights()
    import models  # noqa: F401
    from models import dino_fusion

    os.environ["BLADEYOLO_DINO_LORA"] = str(rank)
    try:
        return getattr(dino_fusion, cls)(3, 64, 3, 2).eval()
    finally:
        os.environ.pop("BLADEYOLO_DINO_LORA")


def _features(conv, x):
    with torch.no_grad():
        return conv._semantics(x)


def test_starts_as_frozen_model():
    frozen, adapted = _build(0), _build(RANK)
    assert frozen.dino_lora is None and len(adapted.dino_lora) == 2 * len(adapted.dino.blocks)
    x = torch.rand(1, 3, 96, 128)
    assert all(torch.equal(a, b) for a, b in zip(_features(frozen, x), _features(adapted, x)))
    print("ok test_starts_as_frozen_model")


def test_freeze_rule_and_gradients():
    """The trainer freezes names containing 'model.0.dino.' (Ultralytics' substring rule): every original DINOv3
    weight is frozen, no LoRA weight is, and after a detection loss only the LoRA weights in DINOv3 get gradient.
    DinoFuse's output projection starts at zero, which blocks gradient to DINOv3 until the first update has moved it;
    it is perturbed here as one optimiser step would."""
    _fake_dinov3_weights()
    import models  # noqa: F401
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import DetectionModel

    os.environ["BLADEYOLO_DINO_LORA"] = str(RANK)
    try:
        m = DetectionModel(os.path.join(ROOT, "bladeyolo2-l.yaml"), nc=5, verbose=False)
    finally:
        os.environ.pop("BLADEYOLO_DINO_LORA")
    names = dict(m.named_parameters(prefix="model"))
    frozen = {k for k in names if "model.0.dino." in k}
    lora = {k for k in names if ".dino_lora." in k}
    assert frozen and len(lora) == 4 * len(m.model[0].dino.blocks) and not frozen & lora
    assert {id(p) for p in m.model[0].dino.parameters()} == {id(names[k]) for k in frozen}
    for k, p in names.items():
        p.requires_grad_(k not in frozen)
    from models.dino_fusion import DinoFuse

    for f in m.model:
        if isinstance(f, DinoFuse):
            torch.nn.init.normal_(f.out.weight, std=0.01)
    m.args = get_cfg()
    m.train()
    batch = {"img": torch.rand(2, 3, 128, 128), "batch_idx": torch.tensor([0.0, 1.0]), "cls": torch.tensor([[0.0], [3.0]]),
             "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.1], [0.3, 0.6, 0.1, 0.3]])}
    m.loss(batch)[0].sum().backward()
    assert all(names[k].grad is None for k in frozen)
    b_grads = [names[k].grad for k in lora if k.endswith(".B.weight")]
    assert all(g is not None and torch.isfinite(g).all() for g in b_grads) and any(g.abs().sum() > 0 for g in b_grads)
    print("ok test_freeze_rule_and_gradients")


def test_copies_keep_lora_wired():
    """EMA (deepcopy) and checkpoints (pickle) must keep each wrapped layer pointing at its own model's LoRA."""
    conv = _build(RANK)
    for lora in conv.dino_lora:
        torch.nn.init.normal_(lora.B.weight, std=0.02)
    x = torch.rand(1, 3, 96, 128)
    ref = _features(conv, x)
    assert not all(torch.equal(a, b) for a, b in zip(ref, _features(_build(0), x)))  # the update is applied
    buf = io.BytesIO()
    torch.save(conv, buf)
    buf.seek(0)
    for c in (copy.deepcopy(conv), torch.load(buf, weights_only=False)):
        assert c.dino.blocks[0].attn.qkv.lora is c.dino_lora[0] and c.dino.blocks[-1].attn.proj.lora is c.dino_lora[-1]
        assert all(torch.equal(a, b) for a, b in zip(ref, _features(c, x)))
        with torch.no_grad():
            c.dino_lora[0].B.weight.zero_()  # changing the copy changes the copy's features, not the original's
        assert not all(torch.equal(a, b) for a, b in zip(ref, _features(c, x)))
    assert all(torch.equal(a, b) for a, b in zip(ref, _features(conv, x)))
    print("ok test_copies_keep_lora_wired")


if __name__ == "__main__":
    test_starts_as_frozen_model()
    test_freeze_rule_and_gradients()
    test_copies_keep_lora_wired()
