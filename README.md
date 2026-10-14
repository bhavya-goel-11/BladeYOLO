<h1 align="center">
BladeYOLO: Wind Turbine Blade Defect Detection with Limited Annotations and Weak-Saliency Awareness
</h1>


<p align="center">
    Yabin Xu<sup>1</sup>,
    Fangtao Zhang<sup>1</sup>,
    Fan Wang<sup>1</sup>,
    Zhan Wang<sup>2</sup>,
    Honghua Chen<sup>3</sup>,
    Mingqiang Wei<sup>4</sup>,
    Haoran Xie<sup>3</sup>,
    Sam Kwong<sup>3</sup>
</p>


<p align="center">
    <sup>1</sup>School of Mechanical Engineering, Zhejiang Sci-Tech University, Hangzhou, China
    <br>
    <sup>2</sup>Department of Artificial Intelligence and Robotics, Zhejiang Energy Digital Technology Co., Ltd., Hangzhou, China
    <br>
    <sup>3</sup>Lingnan University, Hong Kong SAR, China
    <br>
    <sup>4</sup>School of Computer Science and Technology, Nanjing University of Aeronautics and Astronautics, Nanjing, China
</p>


<p align="center">
    <i>IEEE Transactions on Geoscience and Remote Sensing (TGRS), 2026</i>
</p>


## Training (Kaggle 2×T4)

1. Build the dataset locally with `python tools/build_dataset.py --out WindSurface-Defect-v3` (see `WindSurface-Defect-v3/README.md`)
   and upload `WindSurface-Defect-v3/` as a Kaggle dataset.
   - Dataset: copy it into the project folder (`BladeYOLO/WindSurface-Defect-v3`), locally and on Kaggle;
     otherwise it is read from `/kaggle/input/datasets/beegee11/wind-surface-defect/WindSurface-Defect-v3`.
   - DINOv3 ViT-S/16 weights: locally `BladeYOLO/dinov3_vits16_pretrain_lvd1689m-08c60483.pth`; on Kaggle
     the `shamskarib/dinov3-vits` model input is used in place. Override with `DINOV3_WEIGHTS=/path/to/file.pth`.
2. `pip install -r requirements.txt`
3. Train (both GPUs, AMP, total batch 16; `python train.py -h` lists all options):
   - BladeYOLO-v2-L: `python train.py --model bladeyolo2-l.yaml --name bladeyolo2_v3 --epochs 100`
   - stock baseline under the same pipeline: `python train.py --model yolo12l.pt --name yolo12l_v3_r90 --box-loss ciou --epochs 100`
   - BladeYOLO-L (v1, wavelet backbone trained from scratch): `python train.py`

   Test metrics are written to `<run>/test/metrics.json`, with test-time augmentation to `metrics_tta.json`.

### Current recipe: v3.2 labels and an ensemble

`WindSurface-Defect-v3.2` (`python tools/merge_regions.py --data WindSurface-Defect-v3.1 --out WindSurface-Defect-v3.2`)
is v3.1 (reviewed test labels, ignore regions) with one labelling convention in every split: same-class boxes within
8 px of each other are merged into one region box while they cover at least 60% of it (diagonal staircases of boxes
along a slanted edge are left alone). Upload it as its own Kaggle dataset and pass its `data.yaml` with `--data`.
Three diverse models, one Kaggle session each:

    python train.py --data <v3.2>/data.yaml --model bladeyolo2-l.yaml --name bladeyolo2_v32 --epochs 100
    python train.py --data <v3.2>/data.yaml --model yolo12l.pt --name yolo12l_v32_832 --box-loss ciou --imgsz 832 --batch 12 --epochs 100
    python train.py --data <v3.2>/data.yaml --name bladeyolo_v32 --epochs 150

then fuse them (weighted box fusion, fusion IoU chosen on val, test reported once) with
`python tools/ensemble.py <run> <run> <run> --data WindSurface-Defect-v3.2 [--tta]`, which writes
`runs/analysis/ensemble/<name>/metrics.json`. Scores on v3.2 are not comparable with v3; report both.

### WTBD benchmark (single dataset, leak-free split)

WTBD (Ji et al., *Scientific Data* 13:396, 2026; [figshare 10.6084/m9.figshare.30210175](https://doi.org/10.6084/m9.figshare.30210175),
CC BY 4.0): 1,065 UAV images (1024 px) of one coastal wind farm, 6 classes (craze, corrosion, surface_injure,
thunderstrike, crack, hide_craze), double-annotated (Cohen's kappa 0.897). Its published split is a seeded random
shuffle of images, but the images include augmented copies (re-lit, rotated, recoloured views of the same defect):
`tools/wtbd_groups.py` finds 122 of the 161 official test images geometrically verified (ORB/RANSAC) as the same
surface as a train image, and 1,065 images form only ~330 distinct scenes. It writes a group split that keeps each
scene in one split (class-stratified; audited to zero held-out/train matches), committed as
`tools/splits/wtbd_group_split.csv`.

    python tools/convert_wtbd.py "<WTBD>/WT blade defect dataset" --out WTBD-group --split-file tools/splits/wtbd_group_split.csv
    python tools/convert_wtbd.py "<WTBD>/WT blade defect dataset" --out WTBD-official      # authors' split, for comparison
    python train.py --data WTBD-group/data.yaml --model bladeyolo2-l.yaml --name bladeyolo2_wtbd_group --epochs 150
    python train.py --data WTBD-official/data.yaml --model bladeyolo2-l.yaml --name bladeyolo2_wtbd_official --epochs 120
    python train.py --data WTBD-group/data.yaml --model yolo12l.pt --box-loss ciou --name yolo12l_wtbd_group --epochs 150
    python train.py --data WTBD-official/data.yaml --model yolo12l.pt --box-loss ciou --name yolo12l_wtbd_official --epochs 120

The group-split result is the one to report; the official-split run measures how much the leak inflates scores.
Test results (Ultralytics mAP50 / mAP50-95; same recipe and seed for all four runs):

| model | WTBD-group (leak-free) | WTBD official (random split) |
|---|---|---|
| BladeYOLO-v2 | 0.548 / 0.245 | 0.840 / 0.493 |
| YOLO12-L | 0.438 / 0.206 | 0.704 / 0.388 |

`tools/bootstrap_ci.py` gives 95% intervals by resampling test scene groups, and paired intervals for the difference
between two runs (`runs/analysis/bootstrap/<name>/report.md`); the group test set has only 58 independent scenes.
`tools/leak_subsets.py` scores a model separately on official test images with and without a twin in train.
(WTBs2025, *Scientific Data* 13:1310, 2026, was also audited with `tools/build_dataset.py --native wtbs`: of 7,544
images, 1,486 unique, learnable ones remain after removing augmented copies, GAN images and sub-16 px defects.)

BladeYOLO-v2 (`bladeyolo2-l.yaml`) is the COCO-pretrained YOLOv12-L (backbone, neck and head; `yolo12l.pt` is downloaded
automatically) with BladeYOLO's components added as zero-initialised residual branches, so training starts exactly
from the pretrained detector: frozen DINOv3 semantics fused into P3/P4/P5, deformable-conv refinement of the P3/P4
outputs and selective-scan refinement of the P5 output. Training adds random quarter turns (`--rot90`, default 0.5),
which with the flips cover all 8 orientations without loosening boxes.

Kaggle sessions stop after 12 h. To continue an interrupted run, attach its `runs/` folder and run
`python train.py --resume <run>/weights/last.pt` with the same `--box-loss` and `--rot90` as the original run.

## Local GPU workstation (Windows, older NVIDIA GPU)

For inference, error analysis, ensembling, the dataset audit and the label tools on a Windows machine with a small
Pascal-generation GPU (e.g. GeForce MX230, 2 GB; training stays on Kaggle). Datasets, trained weights and caches are
not in git: make `audit_bundle.zip` (raw source datasets + embedding/verification caches) and `project_bundle.zip`
(WindSurface-Defect-v3, WTBD, trained `best.pt` files, cached predictions, review decisions) on the Linux machine, put
both in the repository root, then in PowerShell:

    powershell -ExecutionPolicy Bypass -File tools\setup_windows.ps1

It creates `.venv` with torch 2.6 + CUDA 12.4 (the newest build that still runs Pascal GPUs on a CUDA 12.5 driver),
installs the requirements, checks the GPU, unpacks the bundles, rebuilds v3.1 / v3.2 / WTBD-group / WTBD-official
deterministically, and runs the unit tests. Tools pick the GPU automatically; caches use OS-independent keys, so caches
made on Linux are reused. Clone to a short path (e.g. `C:\BladeYOLO`): some Roboflow file names are long.

Error analysis and label review: `tools/error_analysis.py` (per class, size, error type; galleries),
`tools/ensemble.py` (weighted box fusion of several runs, scored on val and test), `tools/merge_regions.py` (region
labelling convention), `tools/probe_nan.py` (first non-finite layer of a checkpoint on training batches; during
training, `BladeTrainer` drops non-finite batches and writes the same per-layer report to `<run>/nan_batches/`),
`tools/review_labels.py` (local review page for model-flagged label candidates) and `tools/apply_review.py`
(applies the decisions as a new dataset version).
