import torch
import math
import ultralytics.utils.metrics as metrics

original_bbox_iou = metrics.bbox_iou

class WIoUTracker:
    iou_mean = 1.0
    momentum = 1 - 0.999

def wiou_bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Overridden bbox_iou to calculate WIoU v3 instead of CIoU.
    WIoU dynamically focuses gradients on average-quality anchors (tightening boxes).
    """
    # 1. Get standard IoU from original function (disable CIoU to get raw IoU first)
    iou = original_bbox_iou(box1, box2, xywh, GIoU=False, DIoU=False, CIoU=False, eps=eps)
    
    if not CIoU:
        return iou
        
    # 2. Extract coordinates for WIoU math
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    if xywh:
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Distance of centers
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    
    c2 = cw ** 2 + ch ** 2 + eps
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
    
    # Distance penalty R_WIoU
    r_wiou = torch.exp((rho2) / c2)
    
    # Outlier degree beta
    L_iou = 1 - iou
    
    # Update running mean of L_iou
    with torch.no_grad():
        WIoUTracker.iou_mean = (1 - WIoUTracker.momentum) * WIoUTracker.iou_mean + WIoUTracker.momentum * L_iou.mean().item()
        
    beta = L_iou / (WIoUTracker.iou_mean + eps)
    
    # Non-monotonic focusing factor (WIoU v3)
    alpha, delta = 1.9, 3.0
    r = beta / (delta * (alpha ** (beta - delta)))
    
    # WIoU = R_WIoU * L_iou, but we return a value that YOLO will do `1 - val` on.
    # YOLO does: loss = 1.0 - iou
    # So we must return `1.0 - (r * r_wiou * L_iou)`
    wiou_loss = r * r_wiou * L_iou
    return 1.0 - wiou_loss

# Inject it!
metrics.bbox_iou = wiou_bbox_iou
print("✅ [BladeYOLO] Injected Wise-IoU v3 (WIoU) into Ultralytics core metrics.")
