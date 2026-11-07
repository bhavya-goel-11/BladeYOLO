"""G24: an RF-DETR-M built at resolution 768 (from a 576 checkpoint) predicts without shape errors, and
run_resolution reads the recorded value. .venv-rfdetr/bin/python tests/test_rfdetr_resolution.py"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import numpy as np
from PIL import Image
import rfdetr_v1 as v1

run = os.path.join(ROOT, "runs", "detect", "BladeYOLO_WindSurface", "rfdetrm_wtbd_group_f25")
assert v1.run_resolution(run) is None
img = Image.open(os.path.join(ROOT, "data", "blade-defect-benchmark", "leakfree", "WTBlade-group", "test", "images",
                              sorted(os.listdir(os.path.join(ROOT, "data", "blade-defect-benchmark", "leakfree", "WTBlade-group", "test", "images")))[0])).convert("RGB")
for res in (None, 768):
    m = v1.model_class("medium")(pretrain_weights=v1.best_checkpoint(run), num_classes=6, **v1.model_kwargs(resolution=res))
    det = m.predict(img, threshold=0.001)
    print(f"resolution {res or 'default'}: model {m.model_config.resolution}, {len(det.xyxy)} boxes")
    assert m.model_config.resolution == (res or 576)
out = os.path.join(ROOT, "runs", "analysis", "g24", "test_opts"); os.makedirs(out, exist_ok=True)
json.dump({"resolution": 768}, open(os.path.join(out, "v1_options.json"), "w"))
assert v1.run_resolution(out) == 768
try:
    v1.model_kwargs(resolution=700); raise SystemExit("700 accepted")
except AssertionError:
    pass
print("ok")
