import os
import models.loss
import sys
import glob
import argparse
import torch
import torch.nn as nn
import __main__

# 1. Ensure absolute/relative imports resolve properly from the project root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Important: Ultralytics imports MUST happen after sys.path manipulation
from ultralytics import YOLO
from ultralytics.nn import modules, tasks

# --- KAGGLE DDP FIX ---
# DDP creates fresh python subprocesses that don't inherit dynamic monkey-patches.
# We must inject our custom classes directly into the installed ultralytics tasks.py file.
tasks_file = tasks.__file__
with open(tasks_file, 'r') as f:
    tasks_code = f.read()

inject_marker = "# --- BLADEYOLO INJECTION START ---"
inject_code = f"""{inject_marker}
import sys
import os
import models.loss
import torch
import torch.nn as nn

for _p in [r"{ROOT_DIR}", "/kaggle/working/BladeYOLO"]:
    if os.path.exists(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from models.backbone import PhysicsAwareBackbone
    
    class BladeYOLOBackbone(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.backbone = PhysicsAwareBackbone().to(torch.float32)
            
        def forward(self, x):
            return self.backbone(x)

    class GetIndex(nn.Module):
        def __init__(self, c1, c2, index):
            super().__init__()
            self.index = index
            real_c1 = 256 if index == 0 else 512
            self.proj = nn.Conv2d(real_c1, c2, kernel_size=1, bias=False) if real_c1 != c2 else nn.Identity()
            
        def forward(self, x):
            return self.proj(x[self.index])

    import ultralytics.nn.modules as modules
    import ultralytics.nn.tasks as tasks
    import __main__
    
    setattr(modules, 'BladeYOLOBackbone', BladeYOLOBackbone)
    setattr(tasks, 'BladeYOLOBackbone', BladeYOLOBackbone)
    setattr(__main__, 'BladeYOLOBackbone', BladeYOLOBackbone)
    
    setattr(modules, 'GetIndex', GetIndex)
    setattr(tasks, 'GetIndex', GetIndex)
    setattr(__main__, 'GetIndex', GetIndex)
    
    setattr(modules, 'GhostConv', GetIndex)
    setattr(tasks, 'GhostConv', GetIndex)
    setattr(__main__, 'GhostConv', GetIndex)
    
    try:
        from ultralytics.nn.modules.block import A2C2f, C3k2
        setattr(modules, 'A2C2f', A2C2f)
        setattr(tasks, 'A2C2f', A2C2f)
        setattr(modules, 'C3k2', C3k2)
        setattr(tasks, 'C3k2', C3k2)
    except ImportError:
        pass
        
    try:
        from models.morphology import C2f_Morph
        from models.cross_mamba import C2f_CrossMamba
        setattr(modules, 'C2f', C2f_Morph)
        setattr(tasks, 'C2f', C2f_Morph)
        setattr(__main__, 'C2f', C2f_Morph)
        setattr(modules, 'C3', C2f_CrossMamba)
        setattr(tasks, 'C3', C2f_CrossMamba)
        setattr(__main__, 'C3', C2f_CrossMamba)
        setattr(modules, 'C2f_Morph', C2f_Morph)
        setattr(tasks, 'C2f_Morph', C2f_Morph)
        setattr(__main__, 'C2f_Morph', C2f_Morph)
    except ImportError:
        pass
except Exception as e:
    print(f"Failed to load BladeYOLO dependencies in tasks.py: {{e}}")
# --- BLADEYOLO INJECTION END ---
"""

if inject_marker not in tasks_code:
    print(f"Injecting BladeYOLO modules into Ultralytics core: {tasks_file}")
    # Strip any old legacy injection if present
    cut_marker = '    return "detect"  # assume detect\n'
    idx = tasks_code.find(cut_marker)
    if idx != -1:
        base_code = tasks_code[:idx + len(cut_marker)]
    else:
        base_code = tasks_code
    with open(tasks_file, 'w') as f:
        f.write(base_code + "\n" + inject_code)
# ----------------------

# Import our custom restructured modules
from models.backbone import PhysicsAwareBackbone

# 2. Define standard wrappers for the Ultralytics YAML Parser
class BladeYOLOBackbone(nn.Module):
    """Wrapper for PhysicsAwareBackbone to handle Ultralytics auto-arguments."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.backbone = PhysicsAwareBackbone().to(torch.float32)  # Force FP32 to prevent BFloat16 EMA crashes
        
    def forward(self, x):
        # Outputs [P3, P4, P5] from LFA+DINOv3 fusion
        return self.backbone(x)

class GetIndex(nn.Module):
    """Extracts a specific tensor from a list output and projects it to the expected channels."""
    def __init__(self, c1, c2, index):
        super().__init__()
        self.index = index
        real_c1 = 256 if index == 0 else 512
        self.proj = nn.Conv2d(real_c1, c2, kernel_size=1, bias=False) if real_c1 != c2 else nn.Identity()
        
    def forward(self, x):
        return self.proj(x[self.index])

# 3. Dynamically inject into Ultralytics and __main__ namespaces!
setattr(modules, 'BladeYOLOBackbone', BladeYOLOBackbone)
setattr(tasks, 'BladeYOLOBackbone', BladeYOLOBackbone)
setattr(__main__, 'BladeYOLOBackbone', BladeYOLOBackbone)

# We hijack 'GhostConv' in the YAML to bypass channel inference issues.
setattr(modules, 'GhostConv', GetIndex)
setattr(tasks, 'GhostConv', GetIndex)
setattr(__main__, 'GhostConv', GetIndex)

# Also expose GetIndex directly so the PyTorch unpickler can find it when loading last.pt
setattr(modules, 'GetIndex', GetIndex)
setattr(tasks, 'GetIndex', GetIndex)
setattr(__main__, 'GetIndex', GetIndex)

try:
    from ultralytics.nn.modules.block import A2C2f, C3k2
    setattr(modules, 'A2C2f', A2C2f)
    setattr(tasks, 'A2C2f', A2C2f)
    setattr(modules, 'C3k2', C3k2)
    setattr(tasks, 'C3k2', C3k2)
except ImportError:
    pass

try:
    from models.morphology import C2f_Morph
    setattr(modules, 'C3', C2f_Morph)
    setattr(tasks, 'C3', C2f_Morph)
    setattr(__main__, 'C3', C2f_Morph)
    setattr(modules, 'C2f_Morph', C2f_Morph)
    setattr(tasks, 'C2f_Morph', C2f_Morph)
    setattr(__main__, 'C2f_Morph', C2f_Morph)
except ImportError:
    pass


# --- DDP SURVIVAL PATCH FOR FREEZING ---
import ultralytics.engine.trainer as trainer_mod
trainer_file = trainer_mod.__file__
with open(trainer_file, 'r') as f:
    trainer_code = f.read()

if "✅ [BladeYOLO]" not in trainer_code:
    print(f"Injecting BladeYOLO freeze patch into Ultralytics core: {trainer_file}")
    target_string = "if not any(v.requires_grad for v in self.model.parameters()):"
    replacement = '''
        # [BladeYOLO DDP Patch] Re-apply freezing logic after Ultralytics _setup_train unfreezes it
        if hasattr(self.model, 'model') and hasattr(self.model.model[0], 'backbone'):
            if hasattr(self.model.model[0].backbone, 'dino'):
                for param in self.model.model[0].backbone.dino.parameters():
                    param.requires_grad = False
            print("✅ [BladeYOLO] Re-applied DINOv3 freezing logic inside DDP subprocess.")
            
        if not any(v.requires_grad for v in self.model.parameters()):
'''
    if target_string in trainer_code:
        trainer_code = trainer_code.replace(target_string, replacement)
        with open(trainer_file, 'w') as f:
            f.write(trainer_code)
# ---------------------------------------

def find_checkpoint(project, name, explicit_path=None):
    """Find the best checkpoint to resume from."""
    if explicit_path:
        if os.path.exists(explicit_path):
            return explicit_path
        raise FileNotFoundError(f"Specified checkpoint not found: {explicit_path}")

    candidates = [
        # Current active run
        os.path.join(ROOT_DIR, "runs", "detect", project, name, "weights", "last.pt"),
        f"/kaggle/input/datasets/beegee11/wind-surface-defect/runs/runs/detect/{project}/{name}/weights/last.pt",
        # Legacy run names
        os.path.join(ROOT_DIR, "runs", "detect", project, "tgrs_paper_reproduction", "weights", "last.pt"),
        f"/kaggle/input/datasets/beegee11/wind-surface-defect/runs/runs/detect/{project}/tgrs_paper_reproduction/weights/last.pt",
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    # Fallback: search for any last.pt under runs/ sorted by modification time (most recent first)
    all_runs = sorted(
        glob.glob(os.path.join(ROOT_DIR, "runs", "**", "last.pt"), recursive=True),
        key=os.path.getmtime,
        reverse=True
    )
    if all_runs:
        return all_runs[0]

    return None

def main():
    parser = argparse.ArgumentParser(description="BladeYOLO Training and Resumption")
    parser.add_argument("--resume", nargs="?", const=True, default=True, help="Resume training (default: True). Pass False to train fresh, or path to checkpoint.")
    parser.add_argument("--weights", type=str, default=None, help="Explicit weights/checkpoint path to resume from.")
    parser.add_argument("--data", type=str, default=None, help="Path to data.yaml.")
    parser.add_argument("--device", type=str, default=None, help="Device(s) to train on (e.g. '0' or '0,1').")
    parser.add_argument("--epochs", type=int, default=400, help="Total epochs (default: 400).")
    parser.add_argument("--batch", type=int, default=10, help="Batch size (default: 10).")
    args, _ = parser.parse_known_args()

    project = 'BladeYOLO_WindSurface'
    name = 'bladeyolo_l_sota'
    yaml_path = os.path.join(ROOT_DIR, 'bladeyolo-l.yaml')
    
    local_data_path = os.path.join(ROOT_DIR, 'WindSurface-Defect', 'data.yaml')
    kaggle_data_path = "/kaggle/input/datasets/beegee11/wind-surface-defect/data.yaml"
    
    if args.data and os.path.exists(args.data):
        data_path = args.data
    else:
        data_path = local_data_path if os.path.exists(local_data_path) else kaggle_data_path

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset YAML not found at: {data_path}")
        
    # Ensure data.yaml does not use 'path: .' which misleads Ultralytics into resolving relative to cwd
    import yaml
    try:
        with open(data_path, 'r') as f:
            data_cfg = yaml.safe_load(f)
        if isinstance(data_cfg, dict) and data_cfg.get('path') == '.':
            data_cfg.pop('path', None)
            with open(data_path, 'w') as f:
                yaml.dump(data_cfg, f, default_flow_style=False)
    except Exception as e:
        print(f"Warning: could not inspect/update path in {data_path}: {e}")
        
    print("=" * 60)
    print(f" Target Dataset: {data_path}")
    print("=" * 60)

    devices = args.device if args.device is not None else ('0,1' if torch.cuda.device_count() > 1 else '0')

    # Determine resume mode and checkpoint path
    resume_requested = True
    explicit_ckpt = args.weights
    if isinstance(args.resume, str):
        if args.resume.lower() in ('false', '0', 'no'):
            resume_requested = False
        else:
            explicit_ckpt = args.resume
    elif args.resume is False:
        resume_requested = False

    last_pt = None
    if resume_requested or explicit_ckpt:
        last_pt = find_checkpoint(project, name, explicit_path=explicit_ckpt)

    if last_pt and os.path.exists(last_pt):
        print(f"Found checkpoint! Resuming training from: {last_pt}")
        model = YOLO(last_pt)
        results = model.train(resume=True, data=data_path, device=devices)
    else:
        print("No checkpoint found. Starting fresh training run...")
        model = YOLO(yaml_path)
        
        results = model.train(
            data=data_path,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=640,
            device=devices,
            optimizer='AdamW',
            lr0=0.002,
            lrf=0.001,
            cos_lr=True,
            warmup_epochs=5,
            warmup_bias_lr=0.1,
            momentum=0.937,
            weight_decay=0.0005,
            flipud=0.0,
            mosaic=1.0,
            mixup=0.15,
            copy_paste=0.0,
            project=project,
            name=name,
            amp=False
        )

if __name__ == '__main__':
    main()
