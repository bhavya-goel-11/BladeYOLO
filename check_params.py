import torch
import torch.nn as nn
from ultralytics import YOLO
import ultralytics.nn.modules as modules
import ultralytics.nn.tasks as tasks

from models.backbone import DINO3Backbone

class BladeYOLOBackbone(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.backbone = DINO3Backbone(
            use_mrf=True, 
            use_cross_scale=True, 
            use_aqua_style=True
        ).to(torch.float32)
        
    def forward(self, x):
        return self.backbone(x)

class GetIndex(nn.Module):
    def __init__(self, c1, c2, index):
        super().__init__()
        self.index = index
        self.proj = nn.Conv2d(384, c2, kernel_size=1, bias=False) if 384 != c2 else nn.Identity()
        
    def forward(self, x):
        return self.proj(x[self.index])

# Inject modules
setattr(modules, 'BladeYOLOBackbone', BladeYOLOBackbone)
setattr(tasks, 'BladeYOLOBackbone', BladeYOLOBackbone)
setattr(modules, 'GhostConv', GetIndex)
setattr(tasks, 'GhostConv', GetIndex)

try:
    from ultralytics.nn.modules.block import A2C2f, C3k2
    setattr(modules, 'A2C2f', A2C2f)
    setattr(tasks, 'A2C2f', A2C2f)
    setattr(modules, 'C3k2', C3k2)
    setattr(tasks, 'C3k2', C3k2)
except ImportError:
    pass

# Load YAML
model = YOLO('bladeyolo.yaml')
model.info(detailed=True)

total_params = sum(p.numel() for p in model.model.parameters())
trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)

print("="*50)
print(f"Total Parameters: {total_params / 1e6:.2f} M")
print(f"Trainable Parameters: {trainable_params / 1e6:.2f} M")
print("="*50)
