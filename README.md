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

BladeYOLO-v2 (`bladeyolo2-l.yaml`) is the COCO-pretrained YOLOv12-L (backbone, neck and head; `yolo12l.pt` is downloaded
automatically) with BladeYOLO's components added as zero-initialised residual branches, so training starts exactly
from the pretrained detector: frozen DINOv3 semantics fused into P3/P4/P5, deformable-conv refinement of the P3/P4
outputs and selective-scan refinement of the P5 output. Training adds random quarter turns (`--rot90`, default 0.5),
which with the flips cover all 8 orientations without loosening boxes.

Kaggle sessions stop after 12 h. To continue an interrupted run, attach its `runs/` folder and run
`python train.py --resume <run>/weights/last.pt` with the same `--box-loss` and `--rot90` as the original run.

Error analysis and label review: `tools/error_analysis.py` (per class, size, error type; galleries),
`tools/ensemble.py` (weighted box fusion of several runs, scored on val and test), `tools/merge_regions.py` (region
labelling convention), `tools/probe_nan.py` (first non-finite layer of a checkpoint on training batches),
`tools/review_labels.py` (local review page for model-flagged label candidates) and `tools/apply_review.py`
(applies the decisions as a new dataset version).
