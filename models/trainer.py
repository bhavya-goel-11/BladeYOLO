import os

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG, LOGGER

import models  # noqa: F401  (registers BladeYOLO layers and WIoU in this process, incl. DDP workers)


def load_pretrained(model, weights, layer_map):
    """Copy layers of a stock Ultralytics checkpoint (e.g. yolo12l.pt) into `model`, re-indexed by `layer_map`.

    layer_map: [[first, last, offset], ...]: checkpoint layers first..last go to layer index + offset in `model`.
    Every mapped tensor must match by name and shape, except the class-score convs of the Detect head, whose output
    size depends on the number of classes: a silent partial transfer would make comparisons meaningless.
    """
    from ultralytics.nn.modules import Detect
    from ultralytics.nn.tasks import load_checkpoint

    src, _ = load_checkpoint(weights)
    csd = src.float().state_dict()
    # Released YOLOv12 checkpoints carry a conv bias in the attention position encoding (attn.pe) that the current
    # code no longer has. A BatchNorm follows that conv, so the bias folds exactly into its running mean.
    for k in [k for k in csd if k.endswith("attn.pe.conv.bias")]:
        csd[k.removesuffix("conv.bias") + "bn.running_mean"] -= csd.pop(k)
    own = model.state_dict()
    detect = len(model.model) - 1 if isinstance(model.model[-1], Detect) else None
    out, skipped, bad = {}, [], []
    for k, v in csd.items():
        i = int(k.split(".")[1])
        dst = next((i + off for a, b, off in layer_map if a <= i <= b), None)
        if dst is None:
            continue
        k2 = ".".join(["model", str(dst), *k.split(".")[2:]])
        if k2 in own and own[k2].shape == v.shape:
            out[k2] = v
        elif dst == detect and ".cv3." in k2 and k2 in own:  # class-score layers: nc differs
            skipped.append(k2)
        else:
            bad.append(f"{k} -> {k2}")
    if bad:
        raise RuntimeError(f"{weights}: {len(bad)} tensors do not match the model, e.g. {bad[:3]}")
    model.load_state_dict(out, strict=False)
    LOGGER.info(f"[BladeYOLO] Loaded {len(out)} pretrained tensors from {weights} (layer map {layer_map}); "
                f"{len(skipped)} class-score tensors re-initialised for {model.yaml['nc']} classes")
    return out


class BladeTrainer(DetectionTrainer):
    """DetectionTrainer that keeps DINOv3 frozen and initialises pretrained backbones.

    Ultralytics re-enables requires_grad on every parameter not listed in `freeze`, so the DINOv3 weights are named
    there (layer 0 is BladeBackbone or DinoConv, its ViT lives under `.dino`). For models without them, e.g. a stock
    YOLO baseline, the pattern matches nothing. A model YAML with a `pretrained` section (weights, layer map) is
    initialised from that checkpoint when training starts from the YAML (not when resuming or fine-tuning a .pt).
    BLADEYOLO_ROT90=p adds random quarter turns to the training augmentation (an environment variable, so DDP
    workers get it too).
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        overrides = {**(overrides or {}), "freeze": ["0.dino"]}
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)

    def build_dataset(self, img_path, mode="train", batch=None):
        dataset = super().build_dataset(img_path, mode=mode, batch=batch)
        p = float(os.environ.get("BLADEYOLO_ROT90", 0))
        if mode == "train" and p > 0:
            from .augment import add_rot90

            add_rot90(dataset, p)
        return dataset

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        pre = model.yaml.get("pretrained")
        if weights is None and pre:
            load_pretrained(model, pre["weights"], pre["map"])
        return model
