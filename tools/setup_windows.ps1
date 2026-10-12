# Set up BladeYOLO on a Windows workstation with an older NVIDIA GPU (tested target: GeForce MX230, 2 GB, Pascal,
# driver 556 / CUDA 12.5). Run from the repository root in PowerShell:
#     powershell -ExecutionPolicy Bypass -File tools\setup_windows.ps1
# Put audit_bundle.zip and project_bundle.zip (made on the Linux machine) in the repository root first.
#
# torch 2.6 + CUDA 12.4: runs on drivers >= 12.4 and still ships Pascal (sm_61) kernels (CUDA 12.8 builds dropped them).
# The GPU is for inference, analysis and the dataset audit; training stays on Kaggle (2 GB is far too small).

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

function Step($msg) { Write-Host "`n== $msg" -ForegroundColor Cyan }

Step "Python 3.12 virtual environment"
if (-not (Test-Path .venv)) { py -3.12 -m venv .venv }
$py = ".venv\Scripts\python.exe"
& $py -m pip install --upgrade pip

Step "PyTorch 2.6 (CUDA 12.4 build, Pascal-compatible)"
& $py -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

Step "Project requirements (Ultralytics, DINOv3 code) and test tools"
& $py -m pip install -r requirements.txt
& $py -m pip install pytest

Step "GPU check"
& $py -c "import torch; ok = torch.cuda.is_available(); print('CUDA available:', ok); print(torch.cuda.get_device_name(0) if ok else 'NO GPU - check the NVIDIA driver'); x = torch.randn(256, 256, device='cuda' if ok else 'cpu'); print('matmul ok:', float((x @ x).sum()) == float((x @ x).sum()))"

Step "Unpacking data bundles"
foreach ($b in "audit_bundle.zip", "project_bundle.zip") {
    if (Test-Path $b) { Write-Host "extracting $b"; tar -xf $b } else { Write-Host "$b not found, skipped" -ForegroundColor Yellow }
}

Step "Rebuilding derived datasets (deterministic, from v3 + committed review decisions / splits)"
if ((Test-Path WindSurface-Defect-v3) -and -not (Test-Path WindSurface-Defect-v3.1)) { & $py tools\apply_review.py --out WindSurface-Defect-v3.1 }
if ((Test-Path WindSurface-Defect-v3.1) -and -not (Test-Path WindSurface-Defect-v3.2)) { & $py tools\merge_regions.py --data WindSurface-Defect-v3.1 --out WindSurface-Defect-v3.2 }
$wtbd = "external\WTBD\WT blade defect dataset"
if ((Test-Path $wtbd) -and -not (Test-Path WTBD-group)) {
    & $py tools\convert_wtbd.py $wtbd --out WTBD-group --split-file tools\splits\wtbd_group_split.csv
    & $py tools\convert_wtbd.py $wtbd --out WTBD-official
}

Step "Unit tests"
$env:DINOV3_WEIGHTS = "runs\tests\dinov3_vits16_random.pth"
foreach ($t in "test_wiou", "test_augment", "test_ss2d", "test_fp32_blocks", "test_merge_regions") { & $py tests\$t.py }

Write-Host "`nDone. For DINOv2-based tools (dataset audit/builder) set:  `$env:TORCH_HOME = `"$PWD\.dataset_cache\torch`"" -ForegroundColor Green
