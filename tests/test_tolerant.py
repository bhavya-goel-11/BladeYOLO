"""Split-tolerant box targets: region construction, target choice, assignment, and the full loss on a real model."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["BLADEYOLO_BOX_LOSS"] = "ciou"  # plain CIoU: WIoU's running mean would make the box loss history-dependent
from models import tolerant  # noqa: E402


def boxes(*b):
    return torch.tensor([b], dtype=torch.float32)


def test_region_boxes():
    g = boxes([0, 0, 100, 40], [100, 0, 200, 40], [300, 300, 340, 340], [0, 0, 0, 0])  # two touching, one apart, pad
    lab = torch.tensor([[[0], [0], [0], [0]]], dtype=torch.float32)
    mask = torch.tensor([[[1], [1], [1], [0]]], dtype=torch.float32)
    r = tolerant.region_boxes(lab, g, mask)
    assert r[0, 0].tolist() == r[0, 1].tolist() == [0, 0, 200, 40]  # each half's region is the whole crack
    assert r[0, 2].tolist() == [300, 300, 340, 340] and r[0, 3].tolist() == [0, 0, 0, 0]
    lab2 = torch.tensor([[[0], [1], [0], [0]]], dtype=torch.float32)  # different classes are not merged
    assert tolerant.region_boxes(lab2, g, mask)[0, 0].tolist() == [0, 0, 100, 40]
    stair = boxes([0, 0, 40, 40], [40, 40, 80, 80], [80, 80, 120, 120])  # diagonal: union is mostly background
    lab3 = torch.zeros(1, 3, 1)
    assert torch.equal(tolerant.region_boxes(lab3, stair, torch.ones(1, 3, 1)), stair)
    print("ok test_region_boxes")


def test_class_selective_regions():
    """Only the selected classes get regions; other classes keep their own box, exactly as without tolerance."""
    g = boxes([0, 0, 100, 40], [100, 0, 200, 40], [0, 100, 100, 140], [100, 100, 200, 140])
    lab = torch.tensor([[[1], [1], [2], [2]]], dtype=torch.float32)  # two touching pairs, classes 1 and 2
    mask = torch.ones(1, 4, 1)
    assert tolerant.region_boxes(lab, g, mask, classes=None)[0, 2].tolist() == [0, 100, 200, 140]
    r = tolerant.region_boxes(lab, g, mask, classes=[1])
    assert r[0, 0].tolist() == r[0, 1].tolist() == [0, 0, 200, 40]  # class 1 merged
    assert torch.equal(r[0, 2:], g[0, 2:])  # class 2 untouched
    os.environ["BLADEYOLO_TOLERANT_CLASSES"] = "2"
    try:
        r = tolerant.region_boxes(lab, g, mask)  # read from the environment, as in training
        assert torch.equal(r[0, :2], g[0, :2]) and r[0, 3].tolist() == [0, 100, 200, 140]
    finally:
        os.environ.pop("BLADEYOLO_TOLERANT_CLASSES")
    print("ok test_class_selective_regions")


class _Assigner:
    last_alt_targets = None


def test_box_loss_picks_the_closer_target():
    a = _Assigner()
    loss = tolerant.TolerantBboxLoss(16, a)
    seg, region = torch.tensor([[[0.0, 0, 10, 4]]]), torch.tensor([[[0.0, 0, 20, 4]]])
    a.last_alt_targets = region  # grid units with stride 1
    fg, ts = torch.ones(1, 1, dtype=torch.bool), torch.ones(1, 1, 1)
    pd = torch.zeros(1, 1, 64)
    anc = torch.tensor([[5.0, 2.0]])
    args = (pd, None, anc, seg, ts, ts.sum(), fg, torch.tensor([640.0, 640]), torch.ones(1, 1))
    whole = loss(pd, region.clone(), *args[2:])[0]  # prediction = whole region: no box loss
    half = loss(pd, seg.clone(), *args[2:])[0]  # prediction = labelled half: no box loss either
    std = tolerant.BboxLoss(16)(pd, region.clone(), *args[2:])[0]  # the standard loss penalises the whole region
    assert whole < 1e-6 and half < 1e-6 and std > 0.3, (whole, half, std)
    print("ok test_box_loss_picks_the_closer_target")


def test_assigner_scores_region_predictions():
    lab = torch.zeros(1, 2, 1)
    g = boxes([0, 0, 64, 32], [64, 0, 128, 32])
    mask = torch.ones(1, 2, 1)
    anc = torch.tensor([[16.0 + 32 * i, 16.0] for i in range(4)])
    pd_b = torch.tensor([[[0.0, 0, 128, 32]] * 4])  # every anchor predicts the whole region
    pd_s = torch.full((1, 4, 1), 0.9)
    std = tolerant.TaskAlignedAssigner(topk=4, num_classes=1, stride=[32])
    tol = tolerant.TolerantAssigner(topk=4, num_classes=1, stride=[32])
    s_std = std(pd_s, pd_b, anc, lab, g, mask)[2].sum()
    s_tol = tol(pd_s, pd_b, anc, lab, g, mask)[2].sum()
    assert s_tol > s_std, (s_tol, s_std)
    assert tol.last_alt_targets[0, 0].tolist() == [0, 0, 128, 32]
    print(f"ok test_assigner_scores_region_predictions (classification target mass {s_std:.3f} -> {s_tol:.3f})")


def test_full_loss_on_a_model():
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils import DEFAULT_CFG

    torch.manual_seed(0)
    m = DetectionModel("yolov8n.yaml", nc=2, verbose=False)
    m.args = DEFAULT_CFG
    m.train()
    img = torch.rand(2, 3, 128, 128)
    apart = dict(img=img, batch_idx=torch.tensor([0.0, 0, 1]), cls=torch.tensor([[0.0], [1], [0]]),
                 bboxes=torch.tensor([[0.2, 0.2, 0.2, 0.2], [0.7, 0.7, 0.2, 0.2], [0.5, 0.5, 0.3, 0.3]]))
    touching = dict(apart, cls=torch.tensor([[0.0], [0], [0]]),
                    bboxes=torch.tensor([[0.3, 0.5, 0.2, 0.1], [0.5, 0.5, 0.2, 0.1], [0.5, 0.5, 0.3, 0.3]]))
    preds = m(img)
    out = {}
    for name, crit in (("std", tolerant.v8DetectionLoss(m)), ("tol", tolerant.TolerantDetectionLoss(m))):
        for case, batch in (("apart", apart), ("touching", touching)):
            loss, _ = crit(preds, batch)
            assert torch.isfinite(loss).all()
            out[name, case] = loss.detach()
    assert torch.allclose(out["std", "apart"], out["tol", "apart"])  # nothing to merge: identical to standard
    assert not torch.allclose(out["std", "touching"], out["tol", "touching"])
    loss, _ = tolerant.TolerantDetectionLoss(m)(preds, touching)
    loss.sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    print("ok test_full_loss_on_a_model")


if __name__ == "__main__":
    test_region_boxes()
    test_class_selective_regions()
    test_box_loss_picks_the_closer_target()
    test_assigner_scores_region_predictions()
    test_full_loss_on_a_model()
