"""BladeYOLO-v2 end to end on CPU: build through Ultralytics, forward, WIoU loss, backward, frozen DINOv3.

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
    m = DetectionModel(os.path.join(ROOT, "bladeyolo2-l.yaml"), nc=5, verbose=False)
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


def test_v2_starts_as_pretrained_yolo12l():
    """BladeYOLO-v2 at initialisation is exactly the COCO-pretrained YOLOv12-L: every pretrained tensor is loaded and
    the DINOv3 fusion and the deformable / selective-scan refinements are zero-initialised residuals, so the neck
    outputs and the box branch of the head equal those of the stock checkpoint. Then: DINOv3 stays frozen, every
    added branch receives gradient, and the fused (inference) forward works."""
    _fake_dinov3_weights()
    import models  # noqa: F401
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import load_checkpoint

    from models.trainer import BladeTrainer

    trainer = BladeTrainer.__new__(BladeTrainer)  # get_model only needs args/data
    trainer.args, trainer.data = get_cfg(), {"nc": 6, "channels": 3, "names": {i: str(i) for i in range(6)}}
    m = trainer.get_model(cfg=os.path.join(ROOT, "bladeyolo2-l.yaml"), verbose=False)
    ref, _ = load_checkpoint(os.path.join(ROOT, "yolo12l.pt"))
    ref, m = ref.float().eval(), m.eval()
    x = torch.rand(1, 3, 256, 320)

    def outputs(model):
        y = []
        for layer in model.model[:-1]:  # everything but Detect
            xx = (y[layer.f] if isinstance(layer.f, int) else [y[j] for j in layer.f]) if layer.f != -1 else (y[-1] if y else x)
            y.append(layer(xx))
        return y

    with torch.no_grad():
        a, b = outputs(ref), outputs(m)
        for i, j in ((4, 9), (6, 10), (8, 11), (14, 17), (17, 20), (20, 23), (14, 24), (17, 25), (20, 26)):
            assert torch.allclose(a[i], b[j], atol=1e-4), f"layer {j} differs from YOLOv12-L layer {i}"
        for k in range(3):  # box-regression branch of the head
            assert torch.allclose(ref.model[-1].cv2[k](a[(14, 17, 20)[k]]), m.model[-1].cv2[k](b[24 + k]), atol=1e-4)

    m.args = get_cfg()
    m.train()
    batch = {
        "img": torch.rand(2, 3, 320, 256),
        "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.tensor([[0.0], [5.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.1], [0.3, 0.6, 0.05, 0.3]]),
    }
    loss, _ = m.loss(batch)
    loss.sum().backward()
    assert torch.isfinite(loss).all()
    assert all(p.grad is None for p in m.model[0].dino.parameters())
    for i in (9, 10, 11):
        assert m.model[i].out.weight.grad.abs().sum() > 0, f"DINOv3 fusion {i} gets no gradient"
    for i in (24, 25, 26):
        assert m.model[i].cv2.bn.weight.grad.abs().sum() > 0, f"refinement {i} gets no gradient"
    m.eval()
    with torch.no_grad():
        raw = m(torch.rand(1, 3, 192, 320))[0]
        m.fuse(verbose=False)  # the summary profiles FLOPs with thop, which crashes deform_conv2d on some CPU builds
        assert m(torch.rand(1, 3, 192, 320))[0].shape == raw.shape


def test_nodino_ablation_starts_as_pretrained_yolo12l():
    """The ablation config (bladeyolo2-l-nodino.yaml) has no DINOv3 at all and, like BladeYOLO-v2, starts as the stock
    YOLOv12-L: every backbone and neck output matches the checkpoint, and the refinements are zero-initialised."""
    import models  # noqa: F401
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import load_checkpoint

    from models.trainer import BladeTrainer

    trainer = BladeTrainer.__new__(BladeTrainer)
    trainer.args, trainer.data = get_cfg(), {"nc": 6, "channels": 3, "names": {i: str(i) for i in range(6)}}
    m = trainer.get_model(cfg=os.path.join(ROOT, "bladeyolo2-l-nodino.yaml"), verbose=False)
    assert not any("dino" in n.lower() for n, _ in m.named_modules()) and len(m.model) == 25
    ref, _ = load_checkpoint(os.path.join(ROOT, "yolo12l.pt"))
    ref, m = ref.float().eval(), m.eval()
    x = torch.rand(1, 3, 256, 320)

    def outputs(model):
        y = []
        for layer in model.model[:-1]:
            xx = (y[layer.f] if isinstance(layer.f, int) else [y[j] for j in layer.f]) if layer.f != -1 else (y[-1] if y else x)
            y.append(layer(xx))
        return y

    with torch.no_grad():
        a, b = outputs(ref), outputs(m)
        for i in range(21):  # backbone and neck: same indices as the checkpoint
            assert torch.allclose(a[i], b[i], atol=1e-4), f"layer {i} differs from YOLOv12-L"
        for k in range(3):  # refinements are identity at init; box branch of the head matches
            assert torch.allclose(a[(14, 17, 20)[k]], b[21 + k], atol=1e-4)
            assert torch.allclose(ref.model[-1].cv2[k](a[(14, 17, 20)[k]]), m.model[-1].cv2[k](b[21 + k]), atol=1e-4)


if __name__ == "__main__":
    test_forward_loss_backward()
    print("ok test_forward_loss_backward")
    test_v2_starts_as_pretrained_yolo12l()
    print("ok test_v2_starts_as_pretrained_yolo12l")
    test_nodino_ablation_starts_as_pretrained_yolo12l()
    print("ok test_nodino_ablation_starts_as_pretrained_yolo12l")
