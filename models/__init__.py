"""BladeYOLO modules and their registration with Ultralytics.

Importing this package makes the custom layers usable by name in model YAMLs and switches the box
loss to WIoU v3. DDP workers import it too, because they unpickle BladeTrainer from models.trainer.
"""

import functools
import os

from ultralytics.nn import tasks

from . import wiou
from .backbone import BladeBackbone, ensure_dinov3_importable
from .dino_fusion import DinoConv, DinoFuse
from .morphology import C2fMorph, C2fMorphRes
from .ss2d import C2fSS2D, C2fSS2DRes

__all__ = ("BladeBackbone", "C2fMorph", "C2fMorphRes", "C2fSS2D", "C2fSS2DRes", "DinoConv", "DinoFuse")

# parse_model only infers channels (and inserts the repeat count) for blocks it recognises by identity, so while a
# model is being parsed our blocks are bound to built-in slots with the same call signature that our YAMLs never use:
#   C3x, C3Ghost, C3TR, C2 (c1, c2, n, shortcut, g, e)  CSP blocks
#   GhostConv    (c1, c2, k, s)               conv layer, output width scaled like Conv
#   CBFuse       list input, output channels = those of the last input
_SLOTS = {"C3x": C2fMorph, "C3Ghost": C2fSS2D, "C3TR": C2fMorphRes, "C2": C2fSS2DRes, "GhostConv": DinoConv,
          "CBFuse": DinoFuse}


def _register():
    if getattr(tasks, "_bladeyolo_registered", False):
        return
    for cls in (BladeBackbone, C2fMorph, C2fMorphRes, C2fSS2D, C2fSS2DRes, DinoConv, DinoFuse):
        setattr(tasks, cls.__name__, cls)

    parse_model = tasks.parse_model

    @functools.wraps(parse_model)
    def parse_model_with_blade_blocks(*args, **kwargs):
        originals = {name: getattr(tasks, name) for name in _SLOTS}
        try:
            for name, cls in _SLOTS.items():
                setattr(tasks, name, cls)
            return parse_model(*args, **kwargs)
        finally:
            for name, cls in originals.items():
                setattr(tasks, name, cls)

    tasks.parse_model = parse_model_with_blade_blocks
    ensure_dinov3_importable()  # saved checkpoints reference DINOv3 classes by module path
    if os.environ.get("BLADEYOLO_BOX_LOSS", "wiou").lower() == "wiou":  # env so DDP workers agree
        wiou.install()
    tasks._bladeyolo_registered = True


_register()
