<h1 align="center">Memorised Blades</h1>

<p align="center">
<b>Data leakage in wind-turbine blade defect benchmarks, leak-free splits, and a frozen foundation prior for detection transformers</b>
</p>

<p align="center">
<a href="paper/ieee/main.pdf">Paper (IEEE journal format, PDF)</a> ·
<a href="https://www.kaggle.com/datasets/beegee11/blade-defect-benchmark">Dataset on Kaggle</a> ·
<a href="research_log.md">Research log</a> ·
<a href="#5-quick-start">Quick start</a> ·
<a href="#7-reproducing-every-experiment">Reproduce</a>
</p>

This repository holds the code, split files, tests, paper sources and full research log of a study of drone-image
defect detection on wind-turbine blades. It started as a reproduction of **BladeYOLO** (Xu et al., *IEEE TGRS* 2026).
It became an audit of the benchmarks that paper and others report on, and it ends with a pre-registered method
experiment.

**What the study found:**

- **The public benchmarks leak.** 78%, 84% and 92% of the official test images of WTBD, Wind Surface Defect and
  WTBlade-Defect have a near-copy or same-scene image in their training split. The data were augmented or tiled
  *before* a random split.
- **The leak inflates the scores.** With the same detector and recipe, swapping only WTBD's random split for a
  scene-grouped one drops RF-DETR-M from 90.4 / 61.2 to 55.3 / 33.2 mAP50 / mAP50-95. On WTBlade-Defect, test copies of
  training frames score 98.2 mAP50, and test images with no training twin score 50.4.
- **Tuning on the leaky splits rewards memorisation.** One recipe-tuning step raised scores on leaked test images and
  lowered them on unseen ones, on all three datasets.
- **Two leak-free benchmarks are released:** WTBD-group and WTBlade-group, scene-grouped and audited, with scene-level
  bootstrap intervals.
- **Frozen DINOv3 features are the whole advantage of a DINOv3-fused YOLO.** This holds at 25, 50 and 100% labels.
  The deformable, selective-scan and loss components around them add nothing measurable.
- **RF-DETR-FP** (proposed) adds a *frozen* DINOv3 ViT-B prior beside RF-DETR's fine-tuned encoder, through a
  zero-initialised fusion layer of 0.79 M parameters.
  - It passes its pre-registered paired test narrowly: +1.45 leak-free mAP50-95, P(>0) = 0.958.
  - It gives the best WTBD-group result of any model (34.4 mAP50-95).
  - Validation does not confirm the gain, and it is not a low-label method.

**Why our reproduction of BladeYOLO gave different numbers.** BladeYOLO's split is unpublished, our public copy of Wind
Surface is 220 images smaller, our early code had four defects, and every random split of these data is dominated by
twins. Section VI-A of the paper gives the full account.

---

## Contents

1. [Results at a glance](#1-results-at-a-glance)
2. [Repository layout](#2-repository-layout)
3. [Data and pretrained models (links, licences)](#3-data-and-pretrained-models)
4. [Setup: local GPU, Kaggle, CPU-only](#4-setup)
5. [Quick start](#5-quick-start)
6. [The models and how to train each](#6-the-models-and-how-to-train-each)
7. [Reproducing every experiment (sessions, commands, times)](#7-reproducing-every-experiment)
8. [Evaluation, leak analysis and scoring](#8-evaluation-leak-analysis-and-scoring)
9. [Building the leak-free splits and pools from raw data](#9-building-the-leak-free-splits-and-pools-from-raw-data)
10. [Paper: figures, tables and PDF](#10-paper-figures-tables-and-pdf)
11. [Tests](#11-tests)
12. [Estimated run times and hardware](#12-estimated-run-times-and-hardware)
13. [Troubleshooting](#13-troubleshooting)
14. [Citation, licences and acknowledgements](#14-citation-licences-and-acknowledgements)

---

## 1. Results at a glance

mAP50 / mAP50-95 on the test set of each protocol. Wind Surface has no separate test split, so its validation set
serves as test. The official protocols are leaky, and the leak-free ones are ours. Values are two-seed means where two
seeds exist, from the COCO-style evaluator (`tools/bootstrap_ci.py`).

| Model | WTBD-official | Wind Surface | WTBlade-Defect | **WTBD-group** | **WTBlade-group** |
|---|---|---|---|---|---|
| BladeYOLO (paper) | – | **86.5 / 57.4** | 85.8 / **74.6** ‡ | – | – |
| YOLOv12-L (as reported by BladeYOLO) | – | – | 83.3 / 71.9 ‡ | – | – |
| Faithful BladeYOLO reimplementation (ours, Run 3) | – | 52.6 / 28.0 | – | – | – |
| BladeYOLO-L (ours, Run 6, leaky) | – | 78.2 / 46.9 | – | – | – |
| YOLO12-L | 69.0 / 37.7 | – | – | 40.0 / 18.5 | 32.2 / 13.3 |
| BladeYOLO-v2 without DINOv3 | – | – | – | 31.8 / 15.2 | – |
| BladeYOLO-v2 (frozen DINOv3) | 81.5 / 48.1 | – | – | 56.1 / 24.4 | **41.8 / 18.0** |
| RF-DETR-L (G26) | 88.9 / 59.5 | – | – | – | – |
| RF-DETR-M (G26 / A0) | 90.4 / **61.2** | 81.0 / 53.3 | **87.2** / 55.1 | 54.3 / 32.2 | 40.0 / 16.6 |
| RF-DETR-M + copy-paste (G27 A1) | – | – | – | 55.0 / 31.7 | 40.6 / 16.9 |
| **RF-DETR-FP (ours, G29)** | **90.9** / 59.9 | 82.1 / 51.7 | – | **58.9 / 34.4** | 41.0 / 17.3 |

‡ This is BladeYOLO's own unpublished 9:1 re-split, not the official 553-image test set. RF-DETR-FP was not run on the
WTBlade-Defect official protocol: as pre-registered, it does not fit a 12 h T4 session.

**The same detector, scored on leaked vs unseen official test images** (mAP50, RF-DETR-M G26):

| | all | leaked | no twin |
|---|---|---|---|
| WTBD-official | 90.4 | 91.5 | 81.6 |
| Wind Surface | 81.0 | 83.3 | 59.2 |
| WTBlade-Defect | 87.2 | 90.8 (copies of a train frame: 98.2) | 50.4 |

Every number, interval and decision is in [`research_log.md`](research_log.md) and the paper,
[`paper/ieee/main.pdf`](paper/ieee/main.pdf).

---

## 2. Repository layout

```
train.py                     YOLO-family training (YOLO12-L, BladeYOLO-v2) on Ultralytics; all options: python train.py -h
bladeyolo2-l.yaml            BladeYOLO-v2 (YOLOv12-L + frozen DINOv3 fusion + DCNv2 + selective scan, zero-initialised)
bladeyolo2-l-nodino.yaml     the same without DINOv3 (the ablation)
models/                      BladeYOLO-v2 modules and training hooks
  dino_fusion.py             DinoConv / DinoFuse: frozen DINOv3 features into P3/P4/P5 (optional LoRA)
  dino_weights.py            finds DINOv3 weights (DINOV3_WEIGHTS, Kaggle model input, data/weights/dinov3/...)
  morphology.py, ss2d.py     DCNv2 refinement; four-direction selective scan (Mamba-2 chunked form)
  wiou.py, tolerant.py       WIoU v3 loss; split-tolerant targets
  precision.py, trainer.py   fp32 islands under AMP; BladeTrainer (frozen DINOv3, NaN guard, DDP)
  view_consistency.py, reward.py, retrieval.py, augment.py   semi-supervision, reward fine-tuning, retrieval prior, Rot90
tools/
  rfdetr_v1.py               RF-DETR driver: prepare | prepare-official | train | predict | import | score
  g29_prior.py               RF-DETR-FP: frozen DINOv3 prior + PriorFuse, patched into RF-DETR's backbone
  sessions.sh                every training queue of the final phase (Kaggle or any GPU machine), with a 12 h time guard
  get_data.sh                downloads the dataset (Kaggle CLI) and the pretrained weights
  bootstrap_ci.py            COCO-style evaluator + scene bootstrap (paired) — the scorer of every table
  official_leak_eval.py      official protocols: two evaluators, leaked vs no-twin subsets, paired differences
  g27_score.py, g28_score.py, g29_score.py   pre-registered scorers of the final experiments
  matching.py, wtbd_groups.py, wtblade_groups.py, build_wtblade_group.py, split_pixel_audit.py, ws_pixel_twins.py
                             the leakage audit and the leak-free splits
  g27_pool.py, g27_paste.py, g27_ssl.py, g27_slots.py   extra-data pools, copy-paste, encoder SSL, multi-dataset slots
  paper_figures{,2,3}.py, paper_evidence.py, build_ieee_tables.py, build_paper.py   paper figures and tables
  pull_kaggle_runs.py        merges a Kaggle notebook's runs/ output into ./runs
  ...                        one tool per hypothesis (H1–H20, G8–G29); each file's docstring says what and how
  splits/                    split files and labelled subsets (e.g. wtbd_group_train_f25{,_rand1,_rand2,_rand3}.txt)
tests/                       unit and end-to-end tests (CPU), one file per component
paper/ieee/                  the paper: main.tex + sections/ + tikz/ diagrams + references.bib (build: latexmk -pdf)
paper/figures/               every figure of the paper (generated)
paper/paper.md               the earlier pandoc draft (8 Nov), kept for traceability
research_log.md              the full record: every phase, hypothesis, pre-registration, result and correction
requirements.txt             full GPU-machine environment (pinned)
```

Not in git, created by the code: `data/` (dataset and weights), `runs/` (training runs, cached predictions,
analyses; about 18 GB here) and `*.pt` / `*.pth` checkpoints.

---

## 3. Data and pretrained models

### 3.1 The benchmark bundle (one download)

Everything this repository reads is assembled in one public Kaggle dataset:
**[`beegee11/blade-defect-benchmark`](https://www.kaggle.com/datasets/beegee11/blade-defect-benchmark)**. It is a
5.9 GB download, 6.8 GB unpacked, 223,622 files, each listed with its SHA-256 in `MANIFEST.csv`.

```
raw/                          the original datasets, files unchanged
  WTBD-original/              WTBD as published (images + Pascal VOC XML)
  WTBD-official/              the same, YOLO format, the authors' random split (train 745 / val 159 / test 161)
  WTBlade-Defect/             "fengChe" Roboflow export, 7 classes, train/valid/test (9,010 / 1,097 / 553)
  WindSurface-Defect/         Liu & Liu (2025), 5 classes, train/val (2,856 / 714; val = test in the 4:1 protocol)
  Beijing-wind-turbine/, Beijing-wind_turbine_check/, WTBs2025/   auxiliary sources (pools and audit only)
leakfree/
  WTBD-group/                 WTBD regrouped by scene: 745 / 159 / 161 images, 224 / 60 / 58 scenes, 6 classes;
                              split.csv, groups_val.csv, groups_test.csv (scene groups for the bootstrap)
  WTBlade-group/              WTBlade-Defect, one frame per source: 2,136 / 323 / 332 frames, 177 test scenes,
                              7 classes; groups.csv
pools/
  g27-pool-L/                 extra training data, twin-free against both leak-free val + test (G27)
  g28-x-wtbd/, g28-x-wtblade/ leave-one-dataset-out pools, twin-free against ALL of the target (G28)
checkpoints/g27-s1/           RF-DETR-M after in-domain SSL (rfdetrm_g27ssl.pth) and multi-dataset stage 1 (last_ema.pth)
LICENSES.md, README.md, MANIFEST.csv
```

**Download it**, in either of these ways:

- **Automatically**, with the Kaggle CLI. This needs an API token (section 4.1):
  ```bash
  bash tools/get_data.sh            # dataset -> data/blade-defect-benchmark/, plus RF-DETR-M and YOLO12-L weights
  bash tools/get_data.sh --verify   # the same, then checks every file against MANIFEST.csv
  ```
- **Manually**, in the same way with the CLI:
  ```bash
  kaggle datasets download beegee11/blade-defect-benchmark -p data
  unzip -q data/blade-defect-benchmark.zip -d data/blade-defect-benchmark && rm data/blade-defect-benchmark.zip
  ```
  Or use **Download** on the dataset page and unzip the archive to `data/blade-defect-benchmark/`.

On Kaggle there is nothing to download: add the dataset as a notebook input and it appears at
`/kaggle/input/datasets/beegee11/blade-defect-benchmark`. `tools/sessions.sh` uses that path automatically. To use
another location, set `BENCH=/path/to/blade-defect-benchmark`.

### 3.2 Original dataset sources

| Dataset | Source | Licence |
|---|---|---|
| WTBD (Ji, Cheng, Wu, *Scientific Data* 13:396, 2026) | figshare [10.6084/m9.figshare.30210175](https://doi.org/10.6084/m9.figshare.30210175) | CC BY 4.0 |
| WTBlade-Defect ("fengChe") | Roboflow Universe [detr-swsa0/fengche-evxno](https://universe.roboflow.com/detr-swsa0/fengche-evxno) | page says CC BY 4.0, **but it contains DTU NordTank images, which are CC BY-NC** |
| Wind Surface Defect (Liu & Liu, *Applied Soft Computing* 185:113914, 2025) | the authors' repository [Yae666/GFD-YOLOX](https://github.com/Yae666/GFD-YOLOX) (Google Drive link) | no data licence stated |
| DTU drone inspection images (Shihavuddin & Chen, 2018) | Mendeley Data [10.17632/hd96prn3nc.2](https://doi.org/10.17632/hd96prn3nc.2) | CC BY-NC 4.0 |
| WTBs2025 (Zhang, *Scientific Data* 13:1310, 2026) | figshare [10.6084/m9.figshare.28876391](https://doi.org/10.6084/m9.figshare.28876391.v1) | CC BY 4.0 |
| Beijing wind-turbine | Roboflow [beijing-university-9d61y/wind-turbine-ebl65](https://universe.roboflow.com/beijing-university-9d61y/wind-turbine-ebl65) | CC BY 4.0 |
| wind_turbine_check | Roboflow [bhavya-goel/wind_turbine_check-sly4r](https://universe.roboflow.com/bhavya-goel/wind_turbine_check-sly4r) | CC BY 4.0 |

Use the bundle **for non-commercial research only**. WTBlade-Defect and everything derived from it (WTBlade-group, the
pools, the G27 checkpoints) contain CC BY-NC images. Cite every source you use (section 14).

### 3.3 Pretrained models

| Model | Used by | How to get it | Licence |
|---|---|---|---|
| RF-DETR-M, COCO (`rf-detr-medium.pth`) | every RF-DETR run | downloaded by `rfdetr` on first use, or `bash tools/get_data.sh`. Code: [roboflow/rf-detr](https://github.com/roboflow/rf-detr), `pip install "rfdetr[train]==1.11.2"` | Apache-2.0 |
| YOLO12-L, COCO (`yolo12l.pt`) | YOLO12-L and BladeYOLO-v2 | downloaded by Ultralytics on first use ([docs](https://docs.ultralytics.com/models/yolo12/)) | AGPL-3.0 |
| DINOv3 ViT-S/16 | BladeYOLO-v2 (default) | [facebook/dinov3-vits16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m) | DINOv3 License (gated) |
| DINOv3 ViT-B/16 | **RF-DETR-FP**, BladeYOLO-v2 `--dino b` | [facebook/dinov3-vitb16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) | DINOv3 License (gated) |
| DINOv3 ViT-L/16 | BladeYOLO-v2 `--dino l` (ablation) | [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) | DINOv3 License (gated) |
| DINOv3 code | all DINOv3 users | [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3), pinned at commit `6876159a` | DINOv3 License |
| DINOv2 ViT-S/14 | the leakage audit (`tools/matching.py`) | `torch.hub` (`facebookresearch/dinov2`), downloaded automatically | Apache-2.0 |
| OWLv2 base/16 ensemble | open-vocabulary baseline (`tools/openvocab_eval.py`) | [google/owlv2-base-patch16-ensemble](https://huggingface.co/google/owlv2-base-patch16-ensemble), automatically | Apache-2.0 |
| SAM 2.1 tiny (`sam2.1_t.pt`) | SAM box refinement (`tools/sam_refine.py`) | Ultralytics asset, automatically | Apache-2.0 |

**DINOv3 is gated.** First accept the licence on the model page. Then download it with the Hugging Face CLI:

```bash
hf auth login                                  # or: export HF_TOKEN=hf_...
bash tools/get_data.sh --no-data --dino b      # ViT-B/16 -> data/weights/dinov3/vitb16/ (s,b,l for all three)
# equivalent: hf download facebook/dinov3-vitb16-pretrain-lvd1689m --local-dir data/weights/dinov3/vitb16
```

The code looks for the weights in this order:
1. `$DINOV3_WEIGHTS`, a path to a `model.safetensors`.
2. A Kaggle model input at `/kaggle/input/models/<owner>/vit{s,b,l}16/*/default/*/model.safetensors`. The owner is
   hard-coded as `beegee11` in `models/dino_weights.py`.
3. `data/weights/dinov3/vit{s,b,l}16/model.safetensors`.

On Kaggle, set `DINOV3_WEIGHTS`, or download into `data/weights/dinov3/` inside the notebook using an `HF_TOKEN`
secret (section 4.2).

---

## 4. Setup

### 4.1 Local GPU machine (Linux, NVIDIA driver with CUDA 12.8, Python 3.10–3.13)

```bash
git clone https://github.com/bhavya-goel-11/BladeYOLO.git && cd BladeYOLO
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt     # torch 2.7.1+cu128, ultralytics 8.4.174, rfdetr 1.11.2, dinov3 (pinned)
```

Kaggle API token, used by `get_data.sh` and `pull_kaggle_runs.py`:
1. On kaggle.com, go to **Settings → API → Create New Token**.
2. Save the `kaggle.json` it downloads to `~/.kaggle/kaggle.json` and run `chmod 600 ~/.kaggle/kaggle.json`.
3. Alternatively, `export KAGGLE_USERNAME=... KAGGLE_KEY=...`.

Then get the data and weights:

```bash
bash tools/get_data.sh --dino b       # ~6 GB download; DINOv3 needs the HF licence and login (section 3.3)
```

Use a venv. Inside an NVIDIA NGC container, installing into the system Python replaces the container's own torch.

### 4.2 Kaggle notebook (free GPU T4 ×2, 12 h per session)

**Notebook settings:** Accelerator **GPU T4 ×2**, Internet **on**. Then **Save Version → Save & Run All**, so the
run continues with the browser closed.

**Inputs** (Add Input):
- Dataset `beegee11/blade-defect-benchmark`.
- DINOv3 weights, for RF-DETR-FP and BladeYOLO-v2, in either of two ways:
  - Attach your own Kaggle model with `model.safetensors` and set `DINOV3_WEIGHTS` to its path.
  - Add an `HF_TOKEN` secret (Add-ons → Secrets) and download in the notebook, as in the setup cell below.

**Setup cell:**

```python
!git clone https://github.com/bhavya-goel-11/BladeYOLO.git
%cd BladeYOLO
!pip install -q "rfdetr[train]==1.11.2"
!pip install -q --no-deps "dinov3 @ git+https://github.com/facebookresearch/dinov3.git@6876159a11b4df116f30f667f8c9888617df0751"
# DINOv3 from Hugging Face (only if you did not attach a model input; needs the HF_TOKEN secret):
from kaggle_secrets import UserSecretsClient
import os; os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
!hf download facebook/dinov3-vitb16-pretrain-lvd1689m --local-dir data/weights/dinov3/vitb16
```

Do **not** `pip install -r requirements.txt` on Kaggle: it would replace Kaggle's preinstalled torch. RF-DETR sessions
need only the two lines above. YOLO-family runs (`train.py`) use Kaggle's preinstalled Ultralytics plus the `dinov3`
line.

**Run a queue and collect the results:**

```python
!bash tools/sessions.sh final1      # e.g. the G29 session 1 (two lanes, one per T4); see section 7
```

When the version finishes, merge its `runs/` output into your local clone:

```bash
python tools/pull_kaggle_runs.py --kernel <your-user>/<notebook-slug>        # --dry-run to preview
```

Alternatively, download the notebook output, unzip `runs/` into the repository and keep `runs/logs/`.

### 4.3 CPU-only (tests, scoring, figures)

The analysis tools and tests run on CPU from the same `requirements.txt`. The RF-DETR end-to-end test uses ViT-S at
192 px, so it fits in 6 GB of RAM. Use `OMP_NUM_THREADS=2` on small machines.

---

## 5. Quick start

These commands train and score RF-DETR-FP on the leak-free WTBD-group benchmark, on one GPU. That is about 4 h on a T4
and about 1.3 h on an A100.

```bash
D=data/blade-defect-benchmark/leakfree/WTBD-group
R=runs/detect/BladeYOLO_WindSurface

# 1. RF-DETR folder layout (train/ valid/ test/), split unchanged
python tools/rfdetr_v1.py prepare-official --data $D --out runs/work/wtbd_group_rf

# 2. train: the G26 recipe (Medium, 768 px, 100 epochs, cosine, batch 4 x accumulation 4) + the frozen ViT-B prior
python tools/rfdetr_v1.py train --dataset runs/work/wtbd_group_rf --out $R/rfdetrm768cos_fp_wtbd_group_full \
    --size medium --batch 4 --accum 4 --lr-schedule cosine --resolution 768 --epochs 100 --prior b

# 3. predict test and val (best EMA checkpoint, conf 0.001) -> preds_test.json, preds_val.json
python tools/rfdetr_v1.py predict --run $R/rfdetrm768cos_fp_wtbd_group_full --data $D

# 4. score with the scene bootstrap
python tools/bootstrap_ci.py $R/rfdetrm768cos_fp_wtbd_group_full --data $D --name fp_wtbd_group
```

Drop `--prior b` for the RF-DETR-M baseline (A0). For WTBlade-group, use
`D=data/blade-defect-benchmark/leakfree/WTBlade-group` with `--resolution 640 --epochs 60`.

---

## 6. The models and how to train each

### 6.1 RF-DETR-FP: frozen DINOv3 prior beside a fine-tuned encoder (proposed; `tools/g29_prior.py`)

RF-DETR-M (`rfdetr` 1.11.2) fine-tunes a windowed DINOv2-S/16 encoder end to end. A two-stage head selects 300 queries
from its P4 projector memory, which has 256 channels at stride 16. RF-DETR-FP runs a frozen DINOv3 ViT-B/16 on the
same normalised input. Its blocks 4, 8 and 12 (with the final norm, 2,304 channels, on the same 16-px grid) are added
to the projector output before query selection and decoding:

```
S  = GELU(Reduce1x1(LN(P)))                        P = [LN(Φ4(I)); LN(Φ8(I)); LN(Φ12(I))], no gradient
Z  = GELU(LN(Mix1x1([F; S])))
F' = F + Out1x1(GELU(DW3x3(Z)))                    Out initialised to zero: step 0 is exactly RF-DETR
```

- The fusion has 0.79 M trainable parameters.
- DINOv3 sits in a process-wide cache, outside the optimizer, the EMA copy and the checkpoints. Checkpoints hold only
  the fusion layers.
- `install("b")` patches RF-DETR's `Backbone`; `rfdetr_v1.py train/predict --prior b` call it.
- Use `--prior s` or `--prior l` for ViT-S or ViT-L.

### 6.2 RF-DETR-M baselines (`tools/rfdetr_v1.py`)

| Protocol | Command (after `prepare` / `prepare-official`) |
|---|---|
| WTBD-group (A0) | `train --resolution 768 --epochs 100 --lr-schedule cosine --batch 4 --accum 4` |
| WTBlade-group (A0) | `train --resolution 640 --epochs 60 --lr-schedule cosine --batch 4 --accum 4` |
| WTBD-official (G26) | `prepare-official --data raw/WTBD-official`; `train --resolution 768 --epochs 100 --lr-schedule cosine --batch 2 --accum 8` |
| Wind Surface (G26) | `prepare-official --data raw/WindSurface-Defect --carve 0.1` (10% of train becomes validation, seed 0); `train --resolution 640 --epochs 100 --lr-schedule cosine`; `predict --splits test valid` |
| WTBlade-Defect official (G26) | `prepare-official --data raw/WTBlade-Defect`; `train --resolution 640 --epochs 40 --lr-schedule cosine` |
| 25% labels (V1) | `prepare --data leakfree/WTBD-group --subset tools/splits/wtbd_group_train_f25.txt`; `train` (576 px, RF-DETR's own schedule) |
| + copy-paste (G27 A1) | after `prepare-official`: `python tools/g27_paste.py paste --protocol wtbd_group --dataset <prepared>` |
| class-agnostic (G28) | `prepare-official --agnostic`; `predict --agnostic` |

Other `train` options:
- `--seed`: the training seed.
- `--freeze-encoder`: the G22 control.
- `--init <ckpt>`: G27 D1/D2, starting from `checkpoints/g27-s1/*.pth`.
- `--size large`: RF-DETR-L.
- `--keep`: extra checkpoint files to keep.

### 6.3 YOLO family (`train.py`, Ultralytics 8.4.174)

```bash
D=data/blade-defect-benchmark/leakfree/WTBD-group
python train.py --data $D/data.yaml --model bladeyolo2-l.yaml --name bladeyolo2_wtbd_group --epochs 150             # BladeYOLO-v2 (ViT-S)
python train.py --data $D/data.yaml --model bladeyolo2-l.yaml --dino b --name bladeyolo2b_wtbd_group --epochs 150   # with ViT-B
python train.py --data $D/data.yaml --model bladeyolo2-l-nodino.yaml --name bladeyolo2nodino_wtbd_group --epochs 150 # ablation
python train.py --data $D/data.yaml --model yolo12l.pt --box-loss ciou --name yolo12l_wtbd_group --epochs 150      # stock YOLO12-L
python train.py --data data/blade-defect-benchmark/raw/WTBD-official/data.yaml --model bladeyolo2-l.yaml --name bladeyolo2_wtbd_official --epochs 120
python train.py --data $D/data.yaml --model bladeyolo2-l.yaml --train-subset tools/splits/wtbd_group_train_f25.txt --name ..._f25  # 25% of scenes
```

- Runs use every visible GPU (DDP), AMP, a total batch of 16, and AdamW at lr 0.001 with weight decay 0.01 and a cosine
  schedule.
- Augmentation is mosaic, flips and random quarter turns (`--rot90`).
- Test metrics go to `<run>/test/metrics.json`, and with TTA to `metrics_tta.json`.
- Kaggle stops a session after 12 h. Resume with `--resume <run>/weights/last.pt`, plus the run's other options.
- `--test <best.pt>` evaluates only.
- Options from the research phases: `--tolerant`, `--tolerant-classes`, `--ssl real|synthetic`, `--reward`,
  `--retrieval`, `--dino-lora R` and `--seed`. `python train.py -h` documents each one.

---

## 7. Reproducing every experiment

The research log is the master record. Each hypothesis (H1–H20, G8–G29) has a pre-registration with its exact
commands, followed by its result. The final-phase training queues are packaged in **`tools/sessions.sh`**:

```bash
bash tools/sessions.sh <session> [<session> ...]     # any machine, in the requirements.txt environment
!bash tools/sessions.sh <session>                    # Kaggle notebook cell
```

| Session | What it trains | T4 time | A100 time |
|---|---|---|---|
| `s2` | G27 A0 and A1 on WTBD-group; A0 on WTBlade-group | ~9 h | ~3 h |
| `s3` | G27 A2, A3 on WTBD-group; A1 on WTBlade-group (needs `checkpoints/g27-s1`) | ~9 h | ~3 h |
| `s4` | G27 A2, A3 on WTBlade-group | ~7 h | ~2.5 h |
| `a0s1`, `a1s1` | seed 1 of A0 and A1 on both leak-free benchmarks | ~6 h each | ~2 h each |
| `g28id` | G28 ID arms (class-agnostic) on both targets | ~6 h | ~2 h |
| `g28x_wtbd`, `g28x_wtbl` | G28 X arms on the twin-free pools, 12 epochs each | ~5–9 h | ~1.5–3 h |
| `yolo12l_wtbl` | YOLO12-L on WTBlade-group, 50 epochs | ~3 h | ~1 h |
| `all` | `a0s1 a1s1 g28id g28x_wtbd g28x_wtbl yolo12l_wtbl` in order | – | 13 h (measured) |
| `fp_s0`, `fp_s1` | **G29 primary**: RF-DETR-FP seed 0/1 on WTBD-group (768, 100 ep) and WTBlade-group (640, 60 ep) | 9.7 h | ~3 h |
| `fp_ws` | G29: RF-DETR-FP on Wind Surface official (640, 100 ep, 10% carve) | 10.8 h | ~3.4 h |
| `fp_wtbdoff` | G29: RF-DETR-FP on WTBD-official (768, 100 ep, batch 2 × 8) | 4.9 h | ~1.5 h |
| `fp_f25` | G29 secondary: RF-DETR-FP on the four 25% subsets (576 px) | ~4 h | ~1.3 h |
| **`final1`** | **two lanes in parallel:** GPU 0 `fp_ws` · GPU 1 `fp_s0` | 10.8 h | – |
| **`final2`** | **two lanes in parallel:** GPU 0 `fp_wtbdoff fp_f25` · GPU 1 `fp_s1` | ~9.7 h | – |

T4 times for the G29 sessions (`fp_*`, `final1`, `final2`) are measured, as are the A100 time of `all` and the
T4 per-run times in section 12. The other session times are estimates (`~`) built from measured per-epoch speeds.

**To reproduce G29 (the paper's final experiment) on free Kaggle GPUs:** run two notebooks (section 4.2), each with
GPU T4 ×2, one with `!bash tools/sessions.sh final1` and the other with `!bash tools/sessions.sh final2`. Both finish
inside 12 h. The baselines they are paired with come from `s2`, `a0s1`, the G26 official runs (section 6.2) and the
V1 25% runs.

**How the queue behaves:**
- **Time guard.** No run starts with less than 30 min left of an `BUDGET_H=11.3` h budget. Training stops (SIGINT)
  15 min before the budget ends. A stopped run is renamed `<run>_STOPPED` and is never scored. On a machine without a
  time limit, set `BUDGET_H=1000`.
- **Resumable.** Finished runs (with `preds_test.json` and `preds_val.json`, or `test/metrics.json`) are skipped, so
  rerunning the same line resumes the queue.
- **Two GPUs.** `final1` and `final2` need two GPUs and run one lane per GPU with `CUDA_VISIBLE_DEVICES`. Lane logs go
  to `runs/logs/lane{0,1}_*.log`, with a status line every 15 min.
- **Paths.** Set them with `BENCH=...` (the dataset), `WORK=...` (prepared copies; default `runs/work/` or
  `/kaggle/working/`) and `RUNS=...` (run folders; default `runs/detect/BladeYOLO_WindSurface`).
- **Records.** Every run writes `machine.txt` (the GPU) and `v1_options.json` (its arguments).

**Earlier experiments** (Phases 1–31, H1–H20, G8–G24) are run from their own tools: one per hypothesis, named in the
log entry with its exact command. Their training runs use `train.py` and `tools/rfdetr_v1.py` as above.

---

## 8. Evaluation, leak analysis and scoring

Every analysis tool runs on CPU from cached predictions under `runs/`. Its outputs go to `runs/analysis/<topic>/`.

| Tool | Output |
|---|---|
| `tools/bootstrap_ci.py <run> [<run> ...] --data <split> --name <n>` | 101-point COCO mAP; scene-bootstrap 95% CIs; paired differences |
| `tools/official_leak_eval.py` / `... paired` | official protocols: two evaluators, leaked vs no-twin subsets, paired Δ and interaction → `runs/analysis/paper/{official,paired}.{md,json}` |
| `tools/g27_score.py`, `tools/g28_score.py`, `tools/g29_score.py` | the pre-registered decisions of G27, G28, G29 → `runs/analysis/g2{7,8,9}/score.md` |
| `tools/granularity_eval.py` | standard vs tolerant vs region AP |
| `tools/error_anatomy.py`, `tools/extent_ceiling.py` | hit / ranked low / other class / no box; box extent against the annotator ceiling |
| `tools/view_label_audit.py` | annotator self-consistency on aligned copies |
| `tools/conformal_recall.py` | conformal recall guarantees, calibrated on seen vs unseen scenes |
| `tools/leak_subsets.py`, `tools/ws_pixel_twins.py`, `tools/split_pixel_audit.py` | per-image twin flags; the residual-leak audit of our own splits |

Example: score G29 after the four sessions have finished.

```bash
python tools/g29_score.py                  # -> runs/analysis/g29/score.md: P (primary), S (secondary), per class, anatomy
python tools/official_leak_eval.py         # official tables incl. G29 rows
python tools/official_leak_eval.py paired  # paired differences on leaked and no-twin images
```

The scorers expect the run folder names of section 7. The G29 scorer reads the seed-1 A0 twins from
`runs/analysis/archive/kaggle_a0s1/`, which are the T4 runs.

---

## 9. Building the leak-free splits and pools from raw data

The bundle already contains the splits and pools. These commands rebuild them from `raw/`, and each step audits its
output.

```bash
B=data/blade-defect-benchmark
python tools/wtbd_groups.py "$B/raw/WTBD-original/WT blade defect dataset" --out runs/analysis/wtbd        # audit + group split
python tools/convert_wtbd.py "$B/raw/WTBD-original/WT blade defect dataset" --out WTBD-group --split-file tools/splits/wtbd_group_split.csv
python tools/wtblade_groups.py $B/raw/WTBlade-Defect --out runs/analysis/wtblade                         # 841 groups, 70/15/15
python tools/build_wtblade_group.py $B/raw/WTBlade-Defect --out WTBlade-group                           # one frame per source
python tools/split_pixel_audit.py                                                                     # residual pixel twins
python tools/scene_fractions.py --data $B/leakfree/WTBD-group --fractions 0.25 0.5 --prefix wtbd_group_train
python tools/g27_pool.py build --pool L      # (and XWTBD, XWTBL); then: audit, export
```

The matching step embeds every image with DINOv2 and verifies candidate pairs with ORB + RANSAC. On a CPU it takes
roughly 10–30 min per dataset. Results are cached under `runs/analysis/`.

---

## 10. Paper: figures, tables and PDF

```bash
sudo apt install -y texlive-latex-extra texlive-publishers texlive-science texlive-fonts-recommended latexmk
python tools/paper_figures.py && python tools/paper_figures2.py && python tools/paper_figures3.py   # paper/figures/
python tools/build_ieee_tables.py                                                                 # appendix tables
cd paper/ieee && latexmk -pdf main.tex                                                            # -> main.pdf
```

The paper is IEEEtran, journal mode, 32 pages. Each section is its own file in `paper/ieee/sections/`. The
architecture diagrams are TikZ, in `paper/ieee/tikz/`. The bibliography is `paper/ieee/references.bib`.

---

## 11. Tests

```bash
python tests/test_g29_prior.py         # RF-DETR-FP: identity at init, DINOv3 outside optimizer/EMA/checkpoint, gradients,
                                       # 1-epoch end-to-end train + reload + predict (CPU, ~10 min, 6 GB RAM is enough)
python tests/test_model.py             # BladeYOLO-v2 builds and equals pretrained YOLOv12-L at initialisation
python tests/test_ss2d.py              # selective scan matches the reference recurrence (1.8e-14)
python tests/test_official_leak_eval.py  # evaluator equivalence and bootstrap
```

`tests/` has one test file per component. Each runs on its own, writes only under `runs/tests/`, and runs on CPU.

---

## 12. Estimated run times and hardware

The T4 times were measured on Kaggle Tesla T4s (fp16, one GPU per run unless noted). On one A100 (bf16), RF-DETR
runs measured about 3.2× faster than on a T4 (28.5 s against 91 s per epoch on WTBD-group). The A100 column is
derived from that ratio except where marked measured, and a `~` marks an estimate.

| Run | T4 | A100 |
|---|---|---|
| RF-DETR-M, WTBD-group, 768 px, 100 ep | 2.5 h (91 s/epoch) | 48 min (measured) |
| RF-DETR-FP, same | 4.2 h* | ~1.3 h |
| RF-DETR-M, WTBlade-group, 640 px, 60 ep | ~3.5 h | ~1.1 h |
| RF-DETR-FP, same | 5.5 h* | ~1.7 h |
| RF-DETR-M, Wind Surface official, 640 px, 100 ep | 7 h (4.2 min/epoch) | ~2.2 h |
| RF-DETR-FP, same | 10.8 h* | ~3.4 h |
| RF-DETR-M, WTBD-official, 768 px, 100 ep | 3 h | ~1 h |
| RF-DETR-FP, same | 4.9 h* | ~1.5 h |
| RF-DETR-M, WTBlade-Defect official, 640 px, 40 ep | ~9.7 h (14.6 min/epoch) | ~3 h |
| RF-DETR-M / FP, 25% subset, 576 px, 100 ep | 35 min / ~1 h* | ~11 / ~20 min |
| RF-DETR-M X arm (G28), 12 ep on a 14–26 k-image pool | ~5–9 h | 1.5–2.8 h (measured) |
| BladeYOLO-v2, WTBD-group, 150 ep (T4 ×2) | 2.2 h | – |
| BladeYOLO-v2, WTBD-official, 120 ep (T4 ×2) | 1.7 h | – |
| BladeYOLO-v2 25% (ViT-S / B / L), 150 ep (T4 ×2) | 0.8 / 1.0 / 1.7 h | – |
| CPU: scoring one experiment (2,000 bootstrap replicates) | 2–15 min | |

\* With a second job running on the other T4 of the same session, which adds about 10–20%.

Memory: RF-DETR-M at 768 px uses batch 4 with accumulation 4 on a 16 GB T4. WTBD-official uses 2 × 8. The frozen
ViT-B adds about 1.5–1.7× to the time and fits in the same memory.

---

## 13. Troubleshooting

| Symptom | Fix |
|---|---|
| `403` from the Kaggle CLI | Token missing or not readable: check `~/.kaggle/kaggle.json` is mode 600, or set `KAGGLE_USERNAME` and `KAGGLE_KEY`. |
| `No space left` while downloading | The dataset needs about 14 GB while unpacking. If `/tmp` is a small tmpfs, `export TMPDIR=$PWD/runs/work/tmp`. |
| DINOv3 download fails (401/403) | Accept the licence on the Hugging Face model page, then `hf auth login`. |
| `FileNotFoundError: model.safetensors` for DINOv3 | Set `DINOV3_WEIGHTS=/path/model.safetensors`, or download to `data/weights/dinov3/vitb16/`. |
| RF-DETR cannot find `rf-detr-medium.pth` | It is looked up relative to the working directory: run from the repo root, or `bash tools/get_data.sh`. |
| Kaggle session killed at 12 h | Use `tools/sessions.sh` (time guard) and rerun the same line in a new version; finished runs are skipped. |
| `lanes need 2 GPUs` | `final1`/`final2` need GPU T4 ×2; on one GPU, run the sessions one after another (`bash tools/sessions.sh fp_ws fp_s0`). |
| NaN in BladeYOLO-v2 training | Fixed by fp32 blocks and fp32 area-attention scores (`models/precision.py`); `tools/probe_nan.py <ckpt>` diagnoses a checkpoint. |
| CPU tests killed (out of memory) | Run `tests/test_g29_prior.py unit` and `... e2e` separately, with `OMP_NUM_THREADS=2`. |
| Different numbers from `ultralytics val` | Its rectangular-batch path differs by 1–2.5 points from cached predictions; the paper uses `tools/bootstrap_ci.py` (within 0.4 of Ultralytics' AP on identical predictions). |

---

## 14. Citation, licences and acknowledgements

If you use the leak-free splits, the audit or RF-DETR-FP, please cite this work, together with the datasets and models
you use:

```bibtex
@misc{goel2026memorisedblades,
  author = {Goel, Bhavya},
  title  = {Memorised Blades: Data Leakage Inflates Wind-Turbine Blade Defect Benchmarks, Leak-Free Splits Reset the
            Field, and a Frozen Foundation Prior for Detection Transformers},
  year   = {2026},
  howpublished = {\url{https://github.com/bhavya-goel-11/BladeYOLO}},
  note   = {Data: \url{https://www.kaggle.com/datasets/beegee11/blade-defect-benchmark}}
}
@article{bladeyolo,  author={Xu, Yabin and others}, title={{BladeYOLO}: Wind Turbine Blade Defect Detection With Limited
  Annotations and Weak-Saliency Awareness}, journal={IEEE Trans. Geosci. Remote Sens.}, volume={64}, year={2026}}
@article{wtbd, author={Ji, Lipeng and Cheng, Junjie and Wu, Shilong}, title={Multiclass Dataset for Intelligent Detection
  of Wind Turbine Blade Defects Using Drone Imagery}, journal={Scientific Data}, volume={13}, pages={396}, year={2026}}
@article{gfdyolox, author={Liu, Dandan and Liu, Mingjie}, title={Wind turbine blades defect detection based on global and
  local attention with multi-feature fusion}, journal={Applied Soft Computing}, volume={185}, pages={113914}, year={2025}}
@article{rfdetr, author={Robinson, Isaac and others}, title={{RF-DETR}: Neural Architecture Search for Real-Time
  Detection Transformers}, journal={arXiv:2511.09554}, year={2025}}
@article{dinov3, author={Sim{\'e}oni, Oriane and others}, title={{DINOv3}}, journal={arXiv:2508.10104}, year={2025}}
```

The complete reference list, with 96 entries, is `paper/ieee/references.bib`.

**Licences.**
- **Code:** the code in this repository is released for research. It builds on Ultralytics (AGPL-3.0), RF-DETR
  (Apache-2.0) and DINOv3 (DINOv3 License). Checkpoints that contain DINOv3 weights must ship with that licence.
  RF-DETR-FP checkpoints contain only the fusion layers.
- **Data:** see section 3.2 and the bundle's `LICENSES.md`. Use it for non-commercial research only.

**Acknowledgements.** We thank the creators of WTBD, Wind Surface Defect, WTBlade-Defect, WTBs2025 and the DTU drone
inspection images for releasing their data, and the DINOv3, DINOv2, RF-DETR and Ultralytics authors for their models
and code. The experiments ran on Kaggle GPU sessions and one A100.
