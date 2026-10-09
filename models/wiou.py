"""Wise-IoU v3 box-regression loss (Tong et al., 2023) for Ultralytics.

Ultralytics' BboxLoss computes `1 - bbox_iou(pred, target, CIoU=True)`, using the name it imported
into ultralytics.utils.loss. We replace that name only, so the loss becomes WIoU v3 while the
task-aligned assigner (ultralytics.utils.tal) keeps matching anchors with plain CIoU.
"""

import torch
import ultralytics.utils.loss as yolo_loss
from ultralytics.utils.metrics import bbox_iou as _bbox_iou

ALPHA, DELTA = 1.9, 3.0  # focusing hyper-parameters recommended by the paper
MOMENTUM = 1 - 0.5 ** (1 / 7000)  # running-mean momentum used by the reference implementation


class _RunningMean:
    value = None  # mean IoU loss, tracked on the training device


def wiou_v3(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """Drop-in for bbox_iou. Returns 1 - L_WIoUv3 so that BboxLoss's `1 - iou` equals the WIoU v3 loss."""
    if not CIoU:
        return _bbox_iou(box1, box2, xywh=xywh, GIoU=GIoU, DIoU=DIoU, eps=eps)
    box1, box2 = box1.float(), box2.float()
    l_iou = 1 - _bbox_iou(box1, box2, xywh=xywh, eps=eps)
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1 / 2, x1 + w1 / 2, y1 - h1 / 2, y1 + h1 / 2
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2 / 2, x2 + w2 / 2, y2 - h2 / 2, y2 + h2 / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)

    # v1: distance attention; the enclosing-box size is detached so it does not fight the IoU term.
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
    r_wiou = torch.exp(rho2 / (cw**2 + ch**2 + eps).detach())

    # v3: non-monotonic focusing on the outlier degree beta = L_IoU / running mean(L_IoU).
    l_det = l_iou.detach()
    if torch.is_grad_enabled() and l_det.numel():  # update only on training steps, not validation loss
        batch_mean = l_det.mean()
        if _RunningMean.value is None or _RunningMean.value.device != batch_mean.device:
            _RunningMean.value = batch_mean.clone()
        else:
            _RunningMean.value.lerp_(batch_mean, MOMENTUM)
    mean = _RunningMean.value if _RunningMean.value is not None else l_det.mean()
    beta = l_det / (mean + eps)
    r = beta / (DELTA * ALPHA ** (beta - DELTA))
    return 1 - r * r_wiou * l_iou


def install():
    yolo_loss.bbox_iou = wiou_v3
