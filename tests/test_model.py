"""BladeYOLO end to end on CPU: build through Ultralytics, forward, WIoU loss, backward, frozen DINOv3.

DINOv3 weights are gated, so a randomly initialised ViT-S/16 checkpoint is written under runs/ and used
in their place; this checks wiring and key matching, not accuracy.
"""

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
FAKE_WEIGHTS = os.path.join(ROOT, "runs", "tests", "dinov3_vits16_random.pth")


def _fake_dinov3_weights():
    if not os.path.exists(FAKE_WEIGHTS):
        from dinov3.hub.backbones import dinov3_vits16

        os.makedirs(os.path.dirname(FAKE_WEIGHTS), exist_ok=True)
        torch.save(dinov3_vits16(pretrained=False).state_dict(), FAKE_WEIGHTS)
    os.environ["DINOV3_WEIGHTS"] = FAKE_WEIGHTS


def test_forward_loss_backward():
    _fake_dinov3_weights()
    import models  # noqa: F401
    import ultralytics.utils.loss as yolo_loss
    import ultralytics.utils.tal as yolo_tal
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import DetectionModel

    assert yolo_loss.bbox_iou.__name__ == "wiou_v3" and yolo_tal.bbox_iou.__name__ == "bbox_iou"
    m = DetectionModel(os.path.join(ROOT, "bladeyolo-l.yaml"), nc=5, verbose=False)
    m.args = get_cfg()
    m.train()
    x = torch.rand(2, 3, 320, 256)
    batch = {
        "img": x,
        "batch_idx": torch.tensor([0.0, 0.0, 1.0]),
        "cls": torch.tensor([[0.0], [3.0], [4.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.1], [0.3, 0.6, 0.05, 0.3], [0.7, 0.2, 0.4, 0.4]]),
    }
    loss, _ = m.loss(batch)
    loss.sum().backward()
    assert torch.isfinite(loss).all()
    assert all(p.grad is None for p in m.model[0].dino.parameters())
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    m.eval()
    with torch.no_grad():
        assert m(torch.rand(1, 3, 192, 320))[0].shape[1] == 4 + 5


if __name__ == "__main__":
    test_forward_loss_backward()
    print("ok test_forward_loss_backward")
