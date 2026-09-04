import os
import sys
import torch
import torch.nn as nn

# 1. Ensure absolute/relative imports resolve properly from the project root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

# Important: Ultralytics imports MUST happen after sys.path manipulation
from ultralytics import YOLO
from ultralytics.nn import modules, tasks

# Import our custom restructured modules
from models.backbone import DINO3Backbone


# 2. Define standard wrappers for the Ultralytics YAML Parser
class BladeYOLOBackbone(nn.Module):
    """Wrapper for DINO3Backbone to handle Ultralytics auto-arguments."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        # Initialize with paper's default architectural settings
        self.backbone = DINO3Backbone(
            use_mrf=True, 
            use_cross_scale=True, 
            use_aqua_style=True
        )
        
    def forward(self, x):
        # Outputs [F3_enh, F4_enh, F5] (or equivalent P3, P4, P5 scales)
        return self.backbone(x)

class GetIndex(nn.Module):
    """Extracts a specific tensor from a list output."""
    def __init__(self, c1, c2, index):
        super().__init__()
        self.index = index
        print(f'GetIndex initialized with c1={c1}, c2={c2}, index={index}')
        
    def forward(self, x):
        return x[self.index]


# 3. Dynamically inject into Ultralytics namespace!
# This allows us to use custom modules without modifying pip-installed code.
setattr(modules, 'BladeYOLOBackbone', BladeYOLOBackbone)
setattr(tasks, 'BladeYOLOBackbone', BladeYOLOBackbone)

# We hijack 'GhostConv' in the YAML to bypass channel inference issues.
# Ultralytics will read `GhostConv, [256, 0]`, set c2=256, and pass index=0.
setattr(modules, 'GhostConv', GetIndex)
setattr(tasks, 'GhostConv', GetIndex)


def main():
    # Relative paths for robust Kaggle execution
    data_path = os.path.join(ROOT_DIR, 'WindSurface-Defect', 'data.yaml')
    yaml_path = os.path.join(ROOT_DIR, 'bladeyolo.yaml')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset YAML not found at: {data_path}")
        
    print(f"===========================================================")
    print(f" Initializing BladeYOLO via: {yaml_path}")
    print(f" Target Dataset: {data_path}")
    print(f"===========================================================")

    # Initialize the model using our custom YAML structure
    model = YOLO(yaml_path)

    # Detect dual GPUs (Kaggle T4x2 uses indices 0 and 1)
    num_gpus = torch.cuda.device_count()
    devices = [0, 1] if num_gpus >= 2 else (0 if num_gpus == 1 else 'cpu')

    # Start Training (strictly following IEEE TGRS 2026 params)
    results = model.train(
        data=data_path,
        epochs=300,
        batch=10,             # Splits to 5 per GPU if dual T4
        imgsz=640,
        device=devices,
        amp=True,             # Mixed precision (essential for fitting in VRAM)
        
        # Optimizer Params
        optimizer='SGD',
        lr0=0.01,
        cos_lr=True,
        
        # Augmentations (No complex tricks, just standard augmentations)
        fliplr=0.5,
        flipud=0.5,
        hsv_v=0.2,            # Random brightness
        mosaic=0.0,           # Disable default Ultralytics mosaic
        mixup=0.0,            # Disable default Ultralytics mixup
        copy_paste=0.0,
        
        project='BladeYOLO_WindSurface',
        name='tgrs_paper_reproduction'
    )
    
    print("Training successfully completed.")

if __name__ == '__main__':
    main()
