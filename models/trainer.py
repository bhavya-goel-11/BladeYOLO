from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG

import models  # noqa: F401  (registers BladeYOLO layers and WIoU in this process, incl. DDP workers)


class BladeTrainer(DetectionTrainer):
    """DetectionTrainer that keeps the DINOv3 branch frozen.

    Ultralytics re-enables requires_grad on every parameter not listed in `freeze`, so the DINOv3
    weights are named there (layer 0 is BladeBackbone, its ViT lives under `.dino`). For models
    without that layer, e.g. a stock YOLO baseline, the pattern matches nothing.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        overrides = {**(overrides or {}), "freeze": ["0.dino"]}
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
