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
3. `python train.py`. This uses both GPUs, AMP, a total batch of 16 and 150 epochs. Run `python train.py -h` to see all options.

Kaggle sessions stop after 12 h. To continue an interrupted run, attach its `runs/` folder and run
`python train.py --resume <run>/weights/last.pt`.

To train a stock baseline under the same pipeline:
`python train.py --model yolo12l.pt --name yolo12l_baseline --box-loss ciou` (COCO-pretrained).
