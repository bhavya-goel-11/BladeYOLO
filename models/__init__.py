"""BladeYOLO modules and their registration with Ultralytics.

Importing this package makes the custom layers usable by name in model YAMLs and switches the box
loss to WIoU v3. DDP workers import it too, because they unpickle BladeTrainer from models.trainer.
"""

import functools
import os

from ultralytics.nn import tasks

from . import wiou
from .backbone import BladeBackbone, ensure_dinov3_importable
from .morphology import C2fMorph
from .ss2d import C2fSS2D

__all__ = ("BladeBackbone", "C2fMorph", "C2fSS2D")

# parse_model only infers input channels and inserts the repeat count for blocks it recognises by
# identity. Our CSP blocks share C3x's signature (c1, c2, n, shortcut, g, e), so while a model is
# being parsed we bind them to two built-in slots our YAMLs never use.
_CSP_SLOTS = {"C3x": C2fMorph, "C3Ghost": C2fSS2D}


def _register():
    if getattr(tasks, "_bladeyolo_registered", False):
        return
    for cls in (BladeBackbone, C2fMorph, C2fSS2D):
        setattr(tasks, cls.__name__, cls)

    parse_model = tasks.parse_model

    @functools.wraps(parse_model)
    def parse_model_with_blade_blocks(*args, **kwargs):
        originals = {name: getattr(tasks, name) for name in _CSP_SLOTS}
        try:
            for name, cls in _CSP_SLOTS.items():
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
