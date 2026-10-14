"""Split-tolerant box targets for amorphous defects (BLADEYOLO_TOLERANT=1, train.py --tolerant).

Annotators cut one crack, crazing or eroded edge into several boxes, or draw one box over it, inconsistently even
across copies of the same image (research log, Phase 16). The standard loss then penalises a prediction that covers
the whole defect when the label happens to be split. Here every label also has an alternative target: the union of
the touching same-class labels it belongs to (its "region"), if that union is not mostly background (member area
>= MIN_FILL of the union, the rule tools/merge_regions.py uses against diagonal staircases). Both places where boxes
are compared accept either target:
  assignment  the task-aligned assigner scores an anchor by its better overlap with the label or with its region
  box loss    each positive anchor regresses towards whichever of the two its prediction already overlaps more
Classification targets are unchanged. Labels without touching same-class neighbours behave exactly as before.
"""

import math

import torch
from ultralytics.utils.loss import BboxLoss, v8DetectionLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import TaskAlignedAssigner

GAP = 8.0  # px at the training input: same-class labels this close are parts of one region
MIN_FILL = 0.6


def region_boxes(gt_labels, gt_bboxes, mask_gt, gap=GAP, min_fill=MIN_FILL):
    """(b, n, 4): for each label the bounding box of its connected same-class touching labels (itself if alone or
    if the union is mostly background). gt_labels (b, n, 1), gt_bboxes (b, n, 4) xyxy, mask_gt (b, n, 1)."""
    b, n, _ = gt_bboxes.shape
    if n <= 1:
        return gt_bboxes
    valid = mask_gt[..., 0].bool()
    x1, y1, x2, y2 = gt_bboxes.unbind(-1)
    ix = torch.minimum(x2[:, :, None], x2[:, None, :]) - torch.maximum(x1[:, :, None], x1[:, None, :]) + gap
    iy = torch.minimum(y2[:, :, None], y2[:, None, :]) - torch.maximum(y1[:, :, None], y1[:, None, :]) + gap
    same = gt_labels[..., 0][:, :, None] == gt_labels[..., 0][:, None, :]
    eye = torch.eye(n, dtype=torch.bool, device=gt_bboxes.device)[None]
    R = ((ix > 0) & (iy > 0) & same & valid[:, :, None] & valid[:, None, :]) | eye
    for _ in range(math.ceil(math.log2(n)) + 1):  # transitive closure by repeated squaring
        R = torch.bmm(R.float(), R.float()) > 0
    big = torch.finfo(gt_bboxes.dtype).max
    ux1 = torch.where(R, x1[:, None, :], big).amin(-1)
    uy1 = torch.where(R, y1[:, None, :], big).amin(-1)
    ux2 = torch.where(R, x2[:, None, :], -big).amax(-1)
    uy2 = torch.where(R, y2[:, None, :], -big).amax(-1)
    area = ((x2 - x1) * (y2 - y1)).clamp(min=0)
    fill = (R.float() * area[:, None, :]).sum(-1) / ((ux2 - ux1) * (uy2 - uy1)).clamp(min=1e-9)
    alt = torch.stack((ux1, uy1, ux2, uy2), -1)
    return torch.where((fill >= min_fill)[..., None] & valid[..., None], alt, gt_bboxes)


class TolerantAssigner(TaskAlignedAssigner):
    """Task-aligned assignment where an anchor's overlap with a label is the better of the label and its region."""

    last_alt_targets = None  # (b, anchors, 4): region box of each anchor's assigned label, for TolerantBboxLoss

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        out = super().forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
        target_bboxes, target_gt_idx = out[1], out[4]
        if gt_bboxes.shape[1] == 0:
            self.last_alt_targets = target_bboxes
        else:
            alt = region_boxes(gt_labels, gt_bboxes, mask_gt)
            idx = target_gt_idx.long() + torch.arange(alt.shape[0], device=alt.device)[:, None] * alt.shape[1]
            self.last_alt_targets = alt.view(-1, 4)[idx]
        return out

    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        self._alt = region_boxes(gt_labels, gt_bboxes, mask_gt)  # from this call's labels (also per-image fallback)
        return super()._forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        align, overlaps = super().get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt)
        alt = getattr(self, "_alt", None)
        if alt is None or alt.shape != gt_bboxes.shape or torch.equal(alt, gt_bboxes):
            return align, overlaps
        align2, overlaps2 = super().get_box_metrics(pd_scores, pd_bboxes, gt_labels, alt, mask_gt)
        return torch.maximum(align, align2), torch.maximum(overlaps, overlaps2)


class TolerantBboxLoss(BboxLoss):
    """Box loss towards the label or its region, whichever the (detached) prediction overlaps more."""

    def __init__(self, reg_max, assigner):
        super().__init__(reg_max)
        self.assigner = assigner

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum,
                fg_mask, imgsz, stride):
        alt = self.assigner.last_alt_targets
        if alt is not None and alt.shape == target_bboxes.shape:
            alt = alt / stride  # target_bboxes arrive in grid units
            p = pred_bboxes.detach()
            better = bbox_iou(p, alt, xywh=False) > bbox_iou(p, target_bboxes, xywh=False)
            target_bboxes = torch.where(better, alt, target_bboxes)
        return super().forward(pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum,
                               fg_mask, imgsz, stride)


class TolerantDetectionLoss(v8DetectionLoss):
    def __init__(self, model, tal_topk=10, tal_topk2=None):
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        a = self.assigner
        self.assigner = TolerantAssigner(topk=a.topk, num_classes=a.num_classes, alpha=a.alpha, beta=a.beta,
                                         stride=self.stride.tolist(), topk2=getattr(a, "topk2", None))
        self.bbox_loss = TolerantBboxLoss(self.reg_max, self.assigner).to(self.device)


def install():
    """Use the tolerant loss for every DetectionModel built in this process (DDP workers read the same env var)."""
    from ultralytics.nn.tasks import DetectionModel

    original = DetectionModel.init_criterion

    def init_criterion(self):
        if getattr(self.model[-1], "one2one_cv2", None) is not None:  # end-to-end heads keep their own loss
            return original(self)
        return TolerantDetectionLoss(self)

    DetectionModel.init_criterion = init_criterion
