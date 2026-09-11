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

# --- KAGGLE DDP FIX ---
# DDP creates fresh python subprocesses that don't inherit dynamic monkey-patches.
# We must inject our custom classes directly into the installed ultralytics tasks.py file.
import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules as modules
import os

tasks_file = tasks.__file__
with open(tasks_file, 'r') as f:
    tasks_code = f.read()

if "BladeYOLOBackbone" not in tasks_code:
    print(f"Injecting BladeYOLO modules into Ultralytics core: {tasks_file}")
    inject_code = """
import sys
import torch
import torch.nn as nn
sys.path.append("/kaggle/working/BladeYOLO")
try:
    from models.backbone import DINO3Backbone
    
    class BladeYOLOBackbone(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            weight_path = None
            for p in ['dinov3_vits16.pth', '/kaggle/input/dinov3/dinov3_vits16.pth', '/kaggle/input/models/shamskarib/dinov3-vits/pytorch/default/1/dinov3_vits16_pretrain_lvd1689m-08c60483.pth']:
                import os
                if os.path.exists(p):
                    weight_path = p
                    break
            self.backbone = DINO3Backbone(
                use_mrf=True, 
                use_cross_scale=True, 
                use_aqua_style=True,
                model_path=weight_path
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

    import ultralytics.nn.modules as modules
    import ultralytics.nn.tasks as tasks
    setattr(modules, 'GhostConv', GetIndex)
    setattr(tasks, 'GhostConv', GetIndex)
except Exception as e:
    print(f"Failed to load BladeYOLO dependencies in DDP subprocess: {e}")
"""
    with open(tasks_file, 'a') as f:
        f.write(inject_code)
# ----------------------


# Import our custom restructured modules
from models.backbone import DINO3Backbone


# 2. Define standard wrappers for the Ultralytics YAML Parser
class BladeYOLOBackbone(nn.Module):
    """Wrapper for DINO3Backbone to handle Ultralytics auto-arguments."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        # Initialize with paper's default architectural settings
        weight_path = None
        for p in ['dinov3_vits16.pth', '/kaggle/input/dinov3/dinov3_vits16.pth', '/kaggle/input/models/shamskarib/dinov3-vits/pytorch/default/1/dinov3_vits16_pretrain_lvd1689m-08c60483.pth']:
            import os
            if os.path.exists(p):
                weight_path = p
                break
        self.backbone = DINO3Backbone(
            use_mrf=True, 
            use_cross_scale=True, 
            use_aqua_style=True,
            model_path=weight_path
        ).to(torch.float32)  # Force FP32 to prevent BFloat16 EMA crashes
        
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

# Also expose GetIndex directly so the PyTorch unpickler can find it when loading last.pt
setattr(modules, 'GetIndex', GetIndex)
setattr(tasks, 'GetIndex', GetIndex)












# --- DDP SURVIVAL PATCH FOR FREEZING ---
import ultralytics.engine.trainer as trainer_mod
trainer_file = trainer_mod.__file__
with open(trainer_file, 'r') as f:
    trainer_code = f.read()

if "✅ [BladeYOLO]" not in trainer_code:
    print(f"Injecting BladeYOLO freeze patch into Ultralytics core: {trainer_file}")
    import re
    # Match def build_optimizer(...) regardless of its default arguments
    pattern = r"(def build_optimizer\([^{:]+\):)"
    
    replacement = r'''\1
        # [BladeYOLO DDP Patch] Re-apply freezing logic after Ultralytics unfreezes
        if hasattr(model, 'model') and hasattr(model.model[0], 'backbone'):
            model.model[0].backbone.freeze_backbone_layers()
            print("✅ [BladeYOLO] Re-applied DINOv3 freezing logic inside DDP subprocess.")
'''
    if re.search(pattern, trainer_code):
        trainer_code = re.sub(pattern, replacement, trainer_code)
        with open(trainer_file, 'w') as f:
            f.write(trainer_code)
# ---------------------------------------

def main():
    # Relative paths for robust Kaggle execution
    # Prioritize Kaggle input path for dataset
    yaml_path = os.path.join(ROOT_DIR, 'bladeyolo.yaml')
    kaggle_data_path = "/kaggle/input/datasets/beegee11/wind-surface-defect/data.yaml"
    local_data_path = os.path.join(ROOT_DIR, 'WindSurface-Defect', 'data.yaml')
    
    is_kaggle = os.path.exists(kaggle_data_path)
    original_data_path = kaggle_data_path if is_kaggle else local_data_path
    
    if not os.path.exists(original_data_path):
        raise FileNotFoundError(f"Dataset YAML not found at: {original_data_path}")
        
    # --- DYNAMICALLY FIX DATASET PATH ---
    # Ultralytics needs the absolute path in data.yaml. 
    # If on Kaggle, the input directory is read-only, so we copy it to a writable temp file.
    import yaml
    import shutil
    
    with open(original_data_path, 'r') as f:
        data_cfg = yaml.safe_load(f)
        
    data_cfg['path'] = os.path.dirname(original_data_path)
    
    # Write to a local writable file
    data_path = os.path.join(ROOT_DIR, 'active_data.yaml')
    with open(data_path, 'w') as f:
        yaml.dump(data_cfg, f, default_flow_style=False)
    # ------------------------------------
        
    print(f"===========================================================")
    print(f" Target Dataset: {data_path}")
    print(f"===========================================================")

    # Prioritize the uploaded Kaggle dataset path, fallback to local
    kaggle_last_pt = "/kaggle/input/datasets/beegee11/wind-surface-defect/runs/runs/detect/BladeYOLO_WindSurface/tgrs_paper_reproduction/weights/last.pt"
    local_last_pt = os.path.join(ROOT_DIR, "runs", "detect", "BladeYOLO_WindSurface", "tgrs_paper_reproduction", "weights", "last.pt")
    
    last_pt = kaggle_last_pt if os.path.exists(kaggle_last_pt) else local_last_pt
    
    devices = '0,1' if torch.cuda.device_count() > 1 else '0'

    if os.path.exists(last_pt):
        print(f"Found checkpoint! Resuming training from: {last_pt}")
        model = YOLO(last_pt)
        results = model.train(resume=True)
    else:
        print("No checkpoint found. Starting fresh training run...")
        model = YOLO(yaml_path)
        
        results = model.train(
            data=data_path,
            epochs=300,
            batch=10,             # Splits to 5 per GPU if dual T4
            imgsz=640,
            device=devices,
            optimizer='SGD',
            lr0=0.01,
            cos_lr=True,
            
            # Augmentations
            mosaic=0.0,           # Disabled to recreate the 77.7% run
            mixup=0.0,            # Disabled to recreate the 77.7% run
            copy_paste=0.0,
            
            project='BladeYOLO_WindSurface',
            name='tgrs_paper_reproduction',
            
            amp=False             # MUST be False to prevent cuFFT crashes on Kaggle T4
        )
if __name__ == '__main__':
    main()
