"""G27 direction 2 (research_log.md): dataset-partitioned class slots for RF-DETR's multi-dataset stage 1.

Each source dataset keeps its own official classes in its own block of slots (tools/g27_pool.py slots.json); same-named
classes of different datasets are NOT merged, because their labelling conventions differ (research_log §10: mutual
crop retrieval <= 0.33, boxes on shared images rarely agree). An image is labelled only in its own dataset's
taxonomy, so the other blocks' classes are unknown, not absent: their classification loss is masked out per image
(Zhou, Koltun & Kraehenbuehl, "Simple multi-dataset detection", CVPR 2022, partitioned heads). Implemented by
filling the other blocks' logits with a large negative constant before the criterion (matching and every loss see
p = 0 there: the IA-BCE loss and its gradient vanish for those entries). The image's dataset is read off its labels
(blocks are disjoint), so pool images all carry >= 1 box (g27_pool drops empty images).

  install(slots_json)  patches rfdetr.models.criterion.SetCriterion.forward for the rest of the process
"""

import json

import torch

FILL = -1e4  # finite in fp16; sigmoid() == 0 and the IA-BCE terms are exactly 0 there (no 0 * inf)


def load_ranges(slots_json):
    s = json.load(open(slots_json))
    ranges = sorted((v["offset"], v["offset"] + len(v["names"])) for v in s.values())
    assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:])) and ranges[0][0] == 0, ranges
    return ranges


def allowed(targets, ranges, n_out, device):
    """(B, n_out) bool: the image's own block, plus any output beyond the real slots (RF-DETR's extra logit)."""
    n_real = ranges[-1][1]
    out = torch.zeros(len(targets), n_out, dtype=torch.bool, device=device)
    out[:, n_real:] = True
    for b, t in enumerate(targets):
        lab = t["labels"]
        if len(lab) == 0:
            out[b] = True  # unknown dataset: unmasked (g27_pool exports none of these)
            continue
        lo, hi = next(r for r in ranges if r[0] <= int(lab[0]) < r[1])
        assert bool(((lab >= lo) & (lab < hi)).all()), f"an image mixes slot blocks: {lab.tolist()}"
        out[b, lo:hi] = True
    return out


def mask_outputs(outputs, targets, ranges):
    """Shallow copy of outputs with every layer's pred_logits filled outside each image's own block."""
    def one(d):
        if "pred_logits" not in d:
            return d
        lg = d["pred_logits"]
        keep = allowed(targets, ranges, lg.shape[-1], lg.device)[:, None, :]
        return {**d, "pred_logits": lg.masked_fill(~keep, FILL)}

    out = one(outputs)
    if "aux_outputs" in outputs:
        out["aux_outputs"] = [one(a) for a in outputs["aux_outputs"]]
    if "enc_outputs" in outputs:
        out["enc_outputs"] = one(outputs["enc_outputs"])
    return out


def install(slots_json):
    from rfdetr.models.criterion import SetCriterion

    ranges = load_ranges(slots_json)
    if getattr(SetCriterion.forward, "_g27_slots", None) is not None:
        SetCriterion.forward = SetCriterion.forward._g27_orig
    orig = SetCriterion.forward

    def forward(self, outputs, targets, num_boxes=None):
        forward.calls += 1
        return orig(self, mask_outputs(outputs, targets, ranges), targets, num_boxes)

    forward.calls, forward._g27_slots, forward._g27_orig = 0, ranges, orig
    SetCriterion.forward = forward
    print(f"G27 slot mask installed: {len(ranges)} blocks, {ranges[-1][1]} slots", flush=True)
    return forward
