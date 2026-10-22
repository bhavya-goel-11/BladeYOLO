"""Reward fine-tuning with a granularity-invariant reward (Phase 28, H5; train.py --reward W --reward-start E).

Annotators cut one amorphous defect (corrosion, crazing, an eroded edge) into boxes inconsistently, so a detector is
scored, and trained, against an arbitrary partition. Here the set of boxes a detector outputs is treated as a
stochastic policy and rewarded only for *where* it puts defects, not how it cuts them into boxes:

  policy   per image, the top K detections after NMS (detached geometry): each is kept with probability
           sigmoid(its class logit), and a kept box is drawn from a Gaussian around the predicted box
           (sd SIGMA x its width/height): log pi = sum of Bernoulli and Gaussian log-likelihoods.
  reward   per class, F1 between the union of the kept boxes and the union of the labelled boxes, rasterised on a
           GRID-px grid; averaged over the classes present in either. Splitting or merging boxes over the same
           region leaves it unchanged. An image with no labels and no kept box scores 1.
  update   S samples per image, leave-one-out baseline (RLOO): loss = -mean (R_s - mean of the others) * log pi_s / K,
           added to the standard detection loss with weight W from epoch E on (the supervised loss keeps training).

Per-epoch statistics (mean reward, kept boxes) go to <run>/reward_stats.csv.
"""

import os

import torch
import torchvision
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors

K = 30  # candidate detections per image
S = 8  # samples per image
SIGMA = 0.1  # Gaussian sd of a kept box's edges, as a share of its width / height
GRID = 4  # px of the reward raster at the training input
NMS_IOU = 0.6

STATE = {"epoch": -1, "stats": {}}  # set by BladeTrainer.preprocess_batch


def config():
    """(weight, start epoch) from train.py --reward, or None."""
    w = float(os.environ.get("BLADEYOLO_REWARD", "0") or 0)
    return (w, int(os.environ.get("BLADEYOLO_REWARD_START", "0") or 0)) if w > 0 else None


def raster(boxes, h, w):
    """(..., n, 4) xyxy px -> (..., h, w) bool union on the GRID raster (cell centres)."""
    ys = (torch.arange(h, device=boxes.device) + 0.5) * GRID
    xs = (torch.arange(w, device=boxes.device) + 0.5) * GRID
    x1, y1, x2, y2 = (boxes[..., i, None, None] for i in range(4))
    inside = (xs >= x1) & (xs <= x2) & (ys[:, None] >= y1) & (ys[:, None] <= y2)  # (..., n, h, w)
    return inside.any(-3)


def region_f1(pred_boxes, pred_cls, keep, gt_boxes, gt_cls, nc, h, w):
    """Reward of S kept sets. pred_boxes (S, k, 4), pred_cls (k,), keep (S, k) bool; gt (g, 4), (g,) -> (S,)."""
    f1, n = torch.zeros(len(keep), device=keep.device), torch.zeros(len(keep), device=keep.device)
    for c in torch.unique(torch.cat([pred_cls, gt_cls])).tolist():
        g = raster(gt_boxes[gt_cls == c], h, w) if (gt_cls == c).any() else torch.zeros(h, w, dtype=torch.bool, device=keep.device)
        sel = pred_cls == c
        kb = pred_boxes[:, sel].clone()
        kb[~keep[:, sel]] = -1.0  # dropped boxes rasterise to nothing
        p = raster(kb, h, w) if sel.any() else torch.zeros(len(keep), h, w, dtype=torch.bool, device=keep.device)
        tp = (p & g).flatten(1).sum(1).float()
        denom = p.flatten(1).sum(1).float() + g.sum().float()
        present = (p.flatten(1).any(1)) | g.any()
        f1 += torch.where(present, 2 * tp / denom.clamp(min=1), torch.zeros_like(tp))
        n += present.float()
    return torch.where(n > 0, f1 / n.clamp(min=1), torch.ones_like(f1))


def reward_loss(pred_scores, pred_bboxes, gt_labels, gt_bboxes, mask_gt, imgsz, generator=None):
    """RLOO policy-gradient loss over the batch. pred_scores (b, a, nc) logits, pred_bboxes (b, a, 4) xyxy px (both
    with gradient); gt in px. Returns (loss, mean reward, mean kept boxes)."""
    b, a, nc = pred_scores.shape
    h, w = int(imgsz[0]) // GRID, int(imgsz[1]) // GRID
    losses, rewards, kept = [], [], []
    for i in range(b):
        prob = pred_scores[i].detach().sigmoid()
        top = prob.flatten().topk(min(K * 10, prob.numel())).indices
        anc, cls = top // nc, top % nc
        boxes = pred_bboxes[i, anc].detach().float()
        k = torchvision.ops.batched_nms(boxes, prob.flatten()[top].float(), cls, NMS_IOU)[:K]
        anc, cls = anc[k], cls[k]
        logit = pred_scores[i, anc, cls].float()
        mu = pred_bboxes[i, anc].float()
        wh = (mu[:, 2:] - mu[:, :2]).detach().clamp(min=2.0)
        sd = (SIGMA * torch.cat([wh, wh], 1)).clamp(min=1.0)
        p = logit.sigmoid().detach()
        keep = torch.rand(S, len(k), device=p.device, generator=generator) < p
        draw = mu.detach() + sd * torch.randn(S, len(k), 4, device=p.device, generator=generator)
        g = mask_gt[i, :, 0].bool()
        R = region_f1(draw, cls, keep, gt_bboxes[i][g].float(), gt_labels[i][g, 0].long(), nc, h, w)
        adv = R - (R.sum() - R) / max(S - 1, 1)
        logp_keep = torch.where(keep, torch.nn.functional.logsigmoid(logit), torch.nn.functional.logsigmoid(-logit)).sum(1)
        logp_box = (-0.5 * ((draw - mu) / sd) ** 2).sum(2)  # Gaussian log-density up to a constant (sd fixed)
        logp = logp_keep + (logp_box * keep).sum(1)
        losses.append(-(adv.detach() * logp).mean() / max(len(k), 1))
        rewards.append(R.mean()), kept.append(keep.float().sum(1).mean())
    return torch.stack(losses).mean(), torch.stack(rewards).mean().item(), torch.stack(kept).mean().item()


class RewardDetectionLoss(v8DetectionLoss):
    """Standard detection loss + weight x the policy-gradient loss above (from the configured epoch on). The reward
    term is added to the classification slot of the loss vector, so the logged 'cls_loss' includes it."""

    def loss(self, preds, batch):
        batch_size = preds["boxes"].shape[0]
        _, loss, items = self.get_assigned_targets_and_loss(preds, batch)
        cfg = config()
        if cfg and STATE["epoch"] >= cfg[1] and torch.is_grad_enabled():
            pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
            pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
            anchor_points, stride = make_anchors(preds["feats"], self.stride, 0.5)
            imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device) * self.stride[0]
            targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]].to(pred_scores.dtype))
            gt_labels, gt_bboxes = targets.split((1, 4), 2)
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
            pred_bboxes = self.bbox_decode(anchor_points, pred_distri) * stride
            rl, r, n = reward_loss(pred_scores, pred_bboxes, gt_labels, gt_bboxes, mask_gt, imgsz)
            loss = loss.clone()
            loss[1] = loss[1] + cfg[0] * rl
            st = STATE["stats"].setdefault(STATE["epoch"], [0, 0.0, 0.0])
            st[0] += 1
            st[1] += r
            st[2] += n
        return loss * batch_size, items


def install():
    """Use the reward loss for every DetectionModel built in this process (DDP workers read the same env vars)."""
    from ultralytics.nn.tasks import DetectionModel

    original = DetectionModel.init_criterion

    def init_criterion(self):
        if getattr(self.model[-1], "one2one_cv2", None) is not None:
            return original(self)
        return RewardDetectionLoss(self)

    DetectionModel.init_criterion = init_criterion
