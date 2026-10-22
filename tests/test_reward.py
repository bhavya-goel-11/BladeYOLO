"""Reward fine-tuning (train.py --reward, models/reward.py): the reward ignores how a region is cut into boxes, the
policy gradient raises the score of a correct detection and lowers a false one, and the full model trains with it.
"""

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from models import reward  # noqa: E402


def test_reward_is_granularity_invariant():
    gt, gc = torch.tensor([[0.0, 0.0, 64.0, 32.0]]), torch.tensor([1])
    halves = torch.tensor([[[0.0, 0.0, 32.0, 32.0], [32.0, 0.0, 64.0, 32.0]]])  # one sample, two boxes
    both = torch.ones(1, 2, dtype=torch.bool)
    r = lambda boxes, cls, keep, g=gt, c=gc: reward.region_f1(boxes, cls, keep, g, c, nc=2, h=40, w=40).item()  # noqa: E731
    assert abs(r(halves, torch.tensor([1, 1]), both) - 1) < 1e-6  # split into two boxes: same union, full reward
    assert abs(r(gt[None], gc, both[:, :1]) - 1) < 1e-6  # one box
    assert r(halves, torch.tensor([1, 1]), torch.tensor([[True, False]])) < 0.8  # half the region missing
    assert r(halves, torch.tensor([0, 0]), both) == 0  # right place, wrong class
    empty = torch.zeros(0, 4)
    assert r(halves, torch.tensor([1, 1]), torch.zeros(1, 2, dtype=torch.bool), empty, torch.zeros(0, dtype=torch.long)) == 1
    print("ok test_reward_is_granularity_invariant")


def test_gradient_direction():
    """Two candidates: one on the labelled defect, one on background. Lowering the loss must raise the first logit
    and lower the second (many samples, so the sign is not noise)."""
    S = reward.S
    reward.S = 512
    try:
        scores = torch.full((1, 2, 1), -2.0)
        scores[0, 0, 0], scores[0, 1, 0] = 0.0, 0.1  # the false one even starts slightly higher
        scores.requires_grad_(True)
        boxes = torch.tensor([[[16.0, 16.0, 48.0, 48.0], [96.0, 96.0, 128.0, 128.0]]], requires_grad=True)
        gt_l, gt_b = torch.zeros(1, 1, 1), torch.tensor([[[16.0, 16.0, 48.0, 48.0]]])
        loss, r, k = reward.reward_loss(scores, boxes, gt_l, gt_b, torch.ones(1, 1, 1), torch.tensor([160, 160]),
                                        generator=torch.Generator().manual_seed(0))
        loss.backward()
    finally:
        reward.S = S
    g = scores.grad[0, :, 0]
    assert g[0] < 0 < g[1], g  # gradient descent raises the correct box's logit, lowers the false one's
    assert torch.isfinite(boxes.grad).all() and 0 < r < 1 and 0 < k <= 2
    print("ok test_gradient_direction")


def test_model_trains_with_reward():
    from test_model import _fake_dinov3_weights

    _fake_dinov3_weights()
    os.environ.update(BLADEYOLO_REWARD="1.0", BLADEYOLO_REWARD_START="0")
    try:
        import models  # noqa: F401
        from ultralytics.cfg import get_cfg
        from ultralytics.nn.tasks import DetectionModel

        reward.install()
        m = DetectionModel(os.path.join(ROOT, "bladeyolo2-l.yaml"), nc=5, verbose=False)
        m.args = get_cfg()
        m.train()
        reward.STATE["epoch"] = 0
        batch = {"img": torch.rand(2, 3, 128, 128), "batch_idx": torch.tensor([0.0, 1.0]),
                 "cls": torch.tensor([[0.0], [3.0]]), "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.1], [0.3, 0.6, 0.1, 0.3]])}
        assert isinstance(m.init_criterion(), reward.RewardDetectionLoss)
        loss, items = m.loss(batch)
        loss.sum().backward()
        assert torch.isfinite(loss).all() and reward.STATE["stats"][0][0] == 1
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.model[-1].parameters())
    finally:
        os.environ.pop("BLADEYOLO_REWARD"), os.environ.pop("BLADEYOLO_REWARD_START")
    print("ok test_model_trains_with_reward")


if __name__ == "__main__":
    test_reward_is_granularity_invariant()
    test_gradient_direction()
    test_model_trains_with_reward()
