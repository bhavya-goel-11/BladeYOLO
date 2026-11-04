"""G27 direction 2: tools/g27_slots.py against RF-DETR's real SetCriterion (train mode, group DETR, aux + enc layers).
  .venv-rfdetr/bin/python tests/test_g27_slots.py
Two fake datasets: A = slots 0-5, B = slots 6-12. With the mask installed, an A image's B-slot logits get exactly zero
gradient on every layer and do not change the loss; its own slots still train. Without it, they do get gradient."""
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import g27_slots  # noqa: E402
from rfdetr.config import RFDETRMediumConfig, TrainConfig  # noqa: E402
from rfdetr.models.lwdetr import build_criterion_from_config  # noqa: E402

OUT = os.path.join(ROOT, "runs", "tests", "g27_slots")
os.makedirs(OUT, exist_ok=True)
SLOTS = os.path.join(OUT, "slots.json")
json.dump({"a": {"offset": 0, "names": list("abcdef")}, "b": {"offset": 6, "names": list("ghijklm")}}, open(SLOTS, "w"))
assert g27_slots.load_ranges(SLOTS) == [(0, 6), (6, 13)]

crit, _ = build_criterion_from_config(RFDETRMediumConfig(num_classes=13), TrainConfig(dataset_dir=OUT, output_dir=OUT))
crit.train()
G = crit.group_detr
B, Q, C = 2, 10 * G, 14  # num_classes + 1 outputs
torch.manual_seed(0)


def make():
    def layer():
        return {"pred_logits": torch.randn(B, Q, C, requires_grad=True),
                "pred_boxes": torch.rand(B, Q, 4).mul(0.5).add(0.25).requires_grad_(True)}
    out = layer()
    out["aux_outputs"] = [layer() for _ in range(3)]
    out["enc_outputs"] = layer()
    return out


targets = [{"labels": torch.tensor([1, 3]), "boxes": torch.tensor([[0.3, 0.3, 0.2, 0.2], [0.7, 0.6, 0.1, 0.3]])},
           {"labels": torch.tensor([8]), "boxes": torch.tensor([[0.5, 0.5, 0.3, 0.3]])}]


def run(outputs):
    ld = crit(outputs, targets)
    loss = sum(ld[k] * crit.weight_dict[k] for k in ld if k in crit.weight_dict)
    loss.backward()
    return loss.item(), [outputs["pred_logits"].grad] + [a["pred_logits"].grad for a in outputs["aux_outputs"]] + \
        [outputs["enc_outputs"]["pred_logits"].grad]


torch.manual_seed(1)
_, g0 = run(make())
assert g0[0][0, :, 6:13].abs().sum() > 0, "control: unmasked, an A image's B slots get gradient"

fwd = g27_slots.install(SLOTS)
torch.manual_seed(1)
o1 = make()
l1, g1 = run(o1)
for g in g1:
    assert g[0, :, 6:13].abs().sum() == 0 and g[1, :, 0:6].abs().sum() == 0, "masked slots must get zero gradient"
    assert g[0, :, 0:6].abs().sum() > 0 and g[1, :, 6:13].abs().sum() > 0, "own slots must still train"
torch.manual_seed(1)
o2 = make()
with torch.no_grad():
    for d in [o2] + o2["aux_outputs"] + [o2["enc_outputs"]]:
        d["pred_logits"][0, :, 6:13] += 5.0
        d["pred_logits"][1, :, 0:6] -= 3.0
l2, _ = run(o2)
assert abs(l1 - l2) < 1e-4, f"loss must not depend on the masked logits: {l1} vs {l2}"
assert fwd.calls == 2
bad = [{"labels": torch.tensor([1, 8]), "boxes": torch.rand(2, 4)}]
try:
    g27_slots.allowed(bad, [(0, 6), (6, 13)], C, "cpu")
    raise SystemExit("accepted an image with labels from two blocks")
except AssertionError:
    pass
g27_slots.install(SLOTS)  # reinstalling must not stack wrappers
from rfdetr.models.criterion import SetCriterion  # noqa: E402
assert SetCriterion.forward._g27_orig is fwd._g27_orig
print(f"group_detr {G}; masked loss {l1:.4f} == perturbed {l2:.4f}; zero grad on other blocks in all 5 layers")
print("ok")
