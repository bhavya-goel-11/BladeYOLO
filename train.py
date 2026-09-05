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
    """Extracts a specific tensor from a list output and projects it to the expected channels."""
    def __init__(self, c1, c2, index):
        super().__init__()
        self.index = index
        # The DINO3Backbone outputs 384 channels. Project them to c2 (e.g., 256, 512, 1024)
        self.proj = nn.Conv2d(384, c2, kernel_size=1, bias=False) if 384 != c2 else nn.Identity()
        
    def forward(self, x):
        return self.proj(x[self.index])


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
        
    # --- DYNAMICALLY FIX DATASET PATH ---
    # Ultralytics needs the absolute path in data.yaml, so we rewrite it at runtime
    import yaml
    with open(data_path, 'r') as f:
        data_cfg = yaml.safe_load(f)
    data_cfg['path'] = os.path.dirname(data_path)
    with open(data_path, 'w') as f:
        yaml.dump(data_cfg, f, default_flow_style=False)
    # ------------------------------------
        
    print(f"===========================================================")
    print(f" Initializing BladeYOLO via: {yaml_path}")
    print(f" Target Dataset: {data_path}")
    print(f"===========================================================")

    # Initialize the model using our custom YAML structure
    model = YOLO(yaml_path)

    # Detect dual GPUs (Kaggle T4x2 uses indices 0 and 1)
    num_gpus = torch.cuda.device_count()
    
    # --- DISTRIBUTED DATA PARALLEL (DDP) NATIVE FIX ---
    # Because Ultralytics spawns a separate temp file for DDP, dynamic runtime 
    # patches are lost. We permanently inject our custom modules into the pip 
    # installation on disk for the duration of this Kaggle session.
    import ultralytics.nn.tasks as tasks_module
    tasks_file = tasks_module.__file__
    
    with open(tasks_file, 'r', encoding='utf-8') as f:
        tasks_content = f.read()
        
    patch_code = f"""
# --- BLADEYOLO CUSTOM MODULE INJECTION ---
import sys
if '{ROOT_DIR}' not in sys.path:
    sys.path.append('{ROOT_DIR}')
try:
    from models.backbone import DINO3Backbone
    import torch.nn as nn
    class BladeYOLOBackbone(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.backbone = DINO3Backbone(use_mrf=True, use_cross_scale=True, use_aqua_style=True)
        def forward(self, x):
            return self.backbone(x)

    class GetIndex(nn.Module):
        def __init__(self, c1, c2, index):
            super().__init__()
            self.index = index
            self.proj = nn.Conv2d(384, c2, kernel_size=1, bias=False) if 384 != c2 else nn.Identity()
        def forward(self, x):
            return self.proj(x[self.index])
            
    # Explicitly add to module globals so parse_model() can find them via globals()[m]
    globals()['BladeYOLOBackbone'] = BladeYOLOBackbone
    globals()['GhostConv'] = GetIndex
except Exception as e:
    print(f"BladeYOLO DDP Injection warning: {{e}}")
# ------------------------------------------
"""
    if "# --- BLADEYOLO CUSTOM MODULE INJECTION ---" not in tasks_content:
        print(f"Injecting BladeYOLO modules into Ultralytics core: {tasks_file}")
        with open(tasks_file, 'a', encoding='utf-8') as f:
            f.write("\n" + patch_code)

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
