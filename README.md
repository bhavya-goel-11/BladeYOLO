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

Datasets are used with their **official classes** (WTBD: craze, corrosion, surface_injure, thunderstrike, crack,
hide_craze). The full history of experiments, results and corrections is in `research_log.md`.

1. Setup: `pip install -r requirements.txt`. DINOv3 weights (ViT-S/16 by default; `--dino b|l` for ViT-B/16, ViT-L/16):
   the official Hugging Face releases `facebook/dinov3-vit{s,b,l}16-pretrain-lvd1689m` (gated; DINOv3 License, which
   asks publications to acknowledge DINOv3). Locally `hf download facebook/dinov3-vits16-pretrain-lvd1689m
   --local-dir external/dinov3/vits16` (likewise vitb16, vitl16); on Kaggle the `beegee11/vits16`, `beegee11/vitb16`,
   `beegee11/vitl16` model inputs are used in place. Override with `DINOV3_WEIGHTS=/path/to/model.safetensors`.
2. Dataset: build WTBD-group / WTBD-official as below and upload them (zipped) as a Kaggle dataset. `--data` is required;
   a read-only input (Kaggle) is copied once into the project folder so label caches can be written.
3. Train (both GPUs, AMP, total batch 16; `python train.py -h` lists all options). Test metrics are written to
   `<run>/test/metrics.json`, with test-time augmentation to `metrics_tta.json`; `python train.py --data ... --test
   <best.pt>` re-runs the test evaluation alone. Kaggle sessions stop after 12 h: continue with
   `python train.py --resume <run>/weights/last.pt` (same `--box-loss`, `--rot90`, `--tolerant` as the original run).

### WTBD benchmark (leak-free split)

WTBD (Ji et al., *Scientific Data* 13:396, 2026; [figshare 10.6084/m9.figshare.30210175](https://doi.org/10.6084/m9.figshare.30210175),
CC BY 4.0): 1,065 UAV images (1024 px) of one wind farm, 6 classes, double-annotated (Cohen's kappa 0.897). Its
published split is a seeded random shuffle, but the images include augmented copies: `tools/wtbd_groups.py` finds that
122 of the 161 official test images show the same surface as a train image (ORB/RANSAC-verified), and the 1,065 images
form only ~330 distinct scenes. It writes a group split that keeps each scene in one split (class-stratified, audited to
zero held-out/train matches), committed as `tools/splits/wtbd_group_split.csv`.

    python tools/convert_wtbd.py "<WTBD>/WT blade defect dataset" --out WTBD-group --split-file tools/splits/wtbd_group_split.csv
    python tools/convert_wtbd.py "<WTBD>/WT blade defect dataset" --out WTBD-official      # authors' split, for comparison
    python train.py --data WTBD-group/data.yaml --model bladeyolo2-l.yaml --name bladeyolo2_wtbd_group --epochs 150
    python train.py --data WTBD-official/data.yaml --model bladeyolo2-l.yaml --name bladeyolo2_wtbd_official --epochs 120
    python train.py --data WTBD-group/data.yaml --model yolo12l.pt --box-loss ciou --name yolo12l_wtbd_group --epochs 150
    python train.py --data WTBD-official/data.yaml --model yolo12l.pt --box-loss ciou --name yolo12l_wtbd_official --epochs 120

Test results (Ultralytics mAP50 / mAP50-95; same recipe and seed). The group split is the one to report; the official
split measures how much the leak inflates scores.

| model | WTBD-group (leak-free) | WTBD official (random split) |
|---|---|---|
| BladeYOLO-v2 | 0.548 / 0.245 | 0.840 / 0.493 |
| BladeYOLO-v2, `--tolerant` | 0.523 / 0.249 | – |
| YOLO12-L | 0.438 / 0.206 | 0.704 / 0.388 |

Learning curves on WTBD-group (`--train-subset tools/splits/wtbd_group_train_f{25,50}.txt`, scene-level subsets from
`tools/scene_fractions.py`): BladeYOLO-v2 0.232 / 0.413 / 0.548 and YOLO12-L 0.106 / 0.244 / 0.438 test mAP50 at
25 / 50 / 100% of the labelled scenes.

### Model

BladeYOLO-v2 (`bladeyolo2-l.yaml`) is the COCO-pretrained YOLOv12-L (backbone, neck and head; `yolo12l.pt` is downloaded
automatically) with BladeYOLO's components added as zero-initialised residual branches, so training starts exactly from
the pretrained detector: frozen DINOv3 semantics fused into P3/P4/P5 (`models/dino_fusion.py`), deformable-conv
refinement of the P3/P4 outputs (`models/morphology.py`) and selective-scan refinement of the P5 output
(`models/ss2d.py`). Training uses WIoU v3 (`models/wiou.py`; `--box-loss ciou` for the stock loss) and random quarter
turns (`--rot90`). `--tolerant` uses split-tolerant box targets (`models/tolerant.py`): a label or the region of
touching same-class labels, whichever the prediction is closer to.

`--ssl real|synthetic --ssl-pairs <view-pool>/pairs.csv` adds teacher-student training on unlabelled view pairs
(`models/view_consistency.py`): the EMA teacher's detections on one view are moved into the other through the verified
transform and become its labels; `real` pairs are different photographs of the same surface, `synthetic` the same
images with random augmentations (baseline). See the research log, Phases 19-21.

Mixed precision: the refinement blocks run in fp32 and YOLOv12's area-attention scores are computed in fp32
(`models/precision.py`; the stock fp16 scores overflow late in training). As a last line of defence `BladeTrainer`
(`models/trainer.py`) drops a non-finite batch, restores the BatchNorm statistics and writes a per-layer report to
`<run>/nan_batches/`; `tools/probe_nan.py` runs the same diagnosis on a checkpoint.

## Evaluation and analysis tools

| tool | purpose |
|---|---|
| `tools/matching.py` | DINOv2 retrieval + flip-aware ORB/RANSAC verification: which images show the same surface |
| `tools/wtbd_groups.py` | leakage audit of WTBD's published split; the leak-free group split |
| `tools/convert_wtbd.py` | WTBD (VOC) to YOLO, with the authors' or the group split |
| `tools/leak_subsets.py` | official test images with and without a twin in train, as separate test sets |
| `tools/bootstrap_ci.py` | 95% intervals by resampling test scene groups; paired intervals for differences between runs |
| `tools/granularity_eval.py` | standard vs granularity-tolerant vs region AP (how defects are cut into boxes) |
| `tools/view_label_audit.py` | label consistency across verified views of the same surface |
| `tools/conformal_recall.py` | recall guarantees by conformal risk control; calibrated on seen vs unseen scenes; operating curve |
| `tools/detection_metrics.py` | shared box IoU and COCO AP |
| `tools/view_pool.py` | unlabelled view pairs from other blade datasets (WTBD matches excluded); real views = different source photos; `--export` for Kaggle |

Unit tests: `python tests/<test>.py` in the project `.venv` (outputs under `runs/`).

## Earlier work (retired)

The paper reproduction, the merged WindSurface-Defect-v2/v3 datasets with a custom 6-class taxonomy, the v1 model
(BladeYOLO-L, wavelet backbone), label review, region merging, ensembling and the Windows workstation setup are
documented in `research_log.md` (Phases 1–12). Their code was removed from the working tree after commit `b374f4c`
(`git checkout b374f4c -- <path>` restores it); records of the retired runs are kept in `runs/analysis/archive/`.
