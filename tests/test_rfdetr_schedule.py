"""G26: rfdetr_v1.schedule_kwargs builds TrainConfigs that RF-DETR accepts, and the managed scheduler they select
really decays inside the run (the default step at epoch 100 never fires in a <= 100-epoch run).
.venv-rfdetr/bin/python tests/test_rfdetr_schedule.py"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import torch
from rfdetr.config import TrainConfig
from rfdetr.training.module_model import _build_managed_scheduler
import rfdetr_v1 as v1

EPOCHS, SPE = 100, 10  # epochs, optimizer steps per epoch
DS = os.path.join(ROOT, "runs", "analysis", "g26", "test_ds")


def factors(kw):
    cfg = TrainConfig(dataset_dir=DS, output_dir=DS, epochs=EPOCHS, **kw)
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    s = _build_managed_scheduler(opt, cfg, total_steps=EPOCHS * SPE, steps_per_epoch=SPE, warmup_steps=0)
    return [s.lr_lambda(e * SPE) if hasattr(s, "lr_lambda") else s.lr_lambdas[0](e * SPE) for e in range(EPOCHS)]


os.makedirs(DS, exist_ok=True)
assert v1.schedule_kwargs() == {} and v1.schedule_kwargs("step") == {}
default = factors(v1.schedule_kwargs())
assert set(default) == {1.0}, "default step schedule should stay constant through epoch 99"
step = factors(v1.schedule_kwargs(lr_drop=75, epochs=EPOCHS))
assert step[74] == 1.0 and step[75] == 0.1 and step[-1] == 0.1, step[70:80]
cos = factors(v1.schedule_kwargs("cosine", epochs=EPOCHS))
assert cos[0] == 1.0 and abs(cos[50] - 0.5) < 0.02 and cos[-1] < 0.01, (cos[0], cos[50], cos[-1])
for bad in (dict(lr_schedule="cosine", lr_drop=75), dict(lr_drop=100), dict(lr_drop=0)):
    try:
        v1.schedule_kwargs(epochs=EPOCHS, **bad); raise SystemExit(f"accepted {bad}")
    except AssertionError:
        pass
print(f"default constant {default[0]}..{default[-1]}; step@75 {step[74]}->{step[75]}; cosine {cos[0]:.2f}/{cos[50]:.2f}/{cos[-1]:.3f}")
print("ok")
