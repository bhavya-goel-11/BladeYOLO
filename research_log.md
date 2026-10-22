# BladeYOLO Research Log

**Wind-turbine blade surface defect detection: from a paper reproduction to a leak-free benchmark**

| | |
|---|---|
| Researcher | Bhavya Goel |
| Period | 3 Sep 2026 – 17 Oct 2026 |
| Compute | Kaggle 2× Tesla T4 (all training since 9 Oct); A100 for Runs 5–7; local CPU machine (analysis only) |
| Repository | github.com/bhavya-goel-11/BladeYOLO (`master`) |
| Reference papers | BladeYOLO (IEEE TGRS 2026, `BladeYOLO.pdf`); GFD-YOLOX / Wind Surface Defect dataset (Appl. Soft Comput. 2025, `1-s2.0-S156849462501227X-main.pdf`); WTBD (Ji et al., *Scientific Data* 13:396, 2026, `s41597-026-06762-x.pdf`); WTBs2025 (*Scientific Data* 13:1310, 2026) |

This log covers the whole project. Phases 1–5 (Sep 3 – Oct 6) summarise the original tentative research log (commit `6f6638a`, 6 Oct). That commit was removed from `master` by the 9 Oct reset but is still in the local git object store: `git show 6f6638a:research_log.md`. Later phases come from the working sessions, the git history and the run folders under `runs/detect/BladeYOLO_WindSurface/`. Where a later audit corrected an earlier conclusion, the correction appears in that phase and in the corrections section (§26).

---

## Contents

1. [Headline results](#1-headline-results)
2. [Phases 1–5: the reproduction attempt and the BladeYOLO-L architecture (3 Sep – 6 Oct)](#2-phases-15-the-reproduction-attempt-and-the-bladeyolo-l-architecture-3-sep--6-oct)
3. [Phase 6: code audit and repair (9 Oct)](#3-phase-6-code-audit-and-repair-9-oct)
4. [Phase 7: leakage in the Wind Surface Defect dataset (9 Oct)](#4-phase-7-leakage-in-the-wind-surface-defect-dataset-9-oct)
5. [Phase 8: dataset v2, the first leak-free baselines (9–10 Oct)](#5-phase-8-dataset-v2-the-first-leak-free-baselines-910-oct)
6. [Phase 9: industrial taxonomy and dataset v3 (10–11 Oct)](#6-phase-9-industrial-taxonomy-and-dataset-v3-1011-oct)
7. [Phase 10: v3 runs, error analysis and label review (11 Oct)](#7-phase-10-v3-runs-error-analysis-and-label-review-11-oct)
8. [Phase 11: BladeYOLO-v2 and the NaN investigations (11–12 Oct)](#8-phase-11-bladeyolo-v2-and-the-nan-investigations-1112-oct)
9. [Phase 12: the v3 ceiling: region merging, ensembling, honest limits (12 Oct)](#9-phase-12-the-v3-ceiling-region-merging-ensembling-honest-limits-12-oct)
10. [Phase 13: new datasets and the content-based audit (12–13 Oct)](#10-phase-13-new-datasets-and-the-content-based-audit-1213-oct)
11. [Phase 14: WTBD training: leak-free vs published split (13 Oct)](#11-phase-14-wtbd-training-leak-free-vs-published-split-13-oct)
12. [Phase 15: YOLO12-L baselines and scene-level confidence intervals (14 Oct)](#12-phase-15-yolo12-l-baselines-and-scene-level-confidence-intervals-14-oct)
13. [Phase 16: evaluation experiments: granularity, label consistency, recall guarantees (14 Oct)](#13-phase-16-evaluation-experiments-granularity-label-consistency-recall-guarantees-14-oct)
14. [Phase 17: follow-ups: split-tolerant training, scene-level guarantees, label audits on two more datasets (15 Oct)](#14-phase-17-follow-ups-split-tolerant-training-scene-level-guarantees-label-audits-on-two-more-datasets-15-oct)
15. [Phase 18: split-tolerant training run and the YOLOv12 attention fp16 overflow (15 Oct)](#15-phase-18-split-tolerant-training-run-and-the-yolov12-attention-fp16-overflow-15-oct)
16. [Phase 19: repository cleanup and a new direction: real-view consistency for data-limited detection (15 Oct)](#16-phase-19-repository-cleanup-and-a-new-direction-real-view-consistency-for-data-limited-detection-15-oct)
17. [Phase 20: learning curves: labelled scenes still pay, and BladeYOLO-v2's advantage is a low-data advantage (15 Oct)](#17-phase-20-learning-curves-labelled-scenes-still-pay-and-bladeyolo-v2s-advantage-is-a-low-data-advantage-15-oct)
18. [Phase 21: the unlabelled view pool and the real-view teacher–student implementation (15–16 Oct)](#18-phase-21-the-unlabelled-view-pool-and-the-real-view-teacherstudent-implementation-1516-oct)
19. [Phase 22: first semi-supervised runs at 25%: no effect, and why (16–17 Oct)](#19-phase-22-first-semi-supervised-runs-at-25-no-effect-and-why-1617-oct)
20. [Phase 23: second semi-supervised round: collapse fixed, small consistent gains, no real-view advantage (17 Oct)](#20-phase-23-second-semi-supervised-round-collapse-fixed-small-consistent-gains-no-real-view-advantage-17-oct)
21. [Phase 24: seeds and class-balanced in-domain pseudo-labels: semi-supervision is dropped (17–18 Oct)](#21-phase-24-seeds-and-class-balanced-in-domain-pseudo-labels-semi-supervision-is-dropped-1718-oct)
22. [Phase 25: the DINOv3 ablation: frozen foundation features carry the entire low-data advantage (18 Oct)](#22-phase-25-the-dinov3-ablation-frozen-foundation-features-carry-the-entire-low-data-advantage-18-oct)
23. [Phase 26: class-selective split tolerance at 100%: the targeted corrosion gain replicates, the aggregate does not move (18–19 Oct)](#23-phase-26-class-selective-split-tolerance-at-100-the-targeted-corrosion-gain-replicates-the-aggregate-does-not-move-1819-oct)
24. [Phase 27: scaling the frozen prior and the full DINOv3 learning curve: a larger backbone helps up to ViT-B, and the DINOv3 gain does not shrink with data (19–20 Oct)](#24-phase-27-scaling-the-frozen-prior-and-the-full-dinov3-learning-curve-a-larger-backbone-helps-up-to-vit-b-and-the-dinov3-gain-does-not-shrink-with-data-1920-oct)
25. [Engineering problems solved](#25-engineering-problems-solved)
26. [Corrections to earlier conclusions](#26-corrections-to-earlier-conclusions)
27. [Run registry](#27-run-registry)
28. [Datasets and artefacts](#28-datasets-and-artefacts)
29. [Open work and next steps](#29-open-work-and-next-steps)
30. [Working conventions](#30-working-conventions)

---

## 1. Headline results

| Claim | Evidence |
|---|---|
| The BladeYOLO paper's numbers could not be reproduced. | Paper: 86.5 mAP50 / 57.4 mAP50-95. Closest faithful reproduction (Run 3): ~52.6 / ~28. |
| The high scores on Wind Surface Defect (Runs 6–7, ~78 / ~47) were leakage. | 532 of 714 val images (75%) have a near-identical twin in train. Of 1,785 augmented `_0`/`_1` pairs, 550 are split across train and val. |
| Once leakage was removed, every model fell to 0.25–0.37 mAP50 on the merged datasets. | v2 and v3 runs (§5–§9). The ceiling on v3 is about 0.40–0.45 mAP50, limited by annotation granularity and a small number of independent scenes. |
| WTBD (2026) is the most suitable public dataset, and its published split leaks. | 125 of 161 official test images have a train twin. A group split, audited to zero matches, was built (`tools/splits/wtbd_group_split.csv`). |
| **BladeYOLO-v2, leak-free WTBD-group test: 0.548 mAP50 / 0.245 mAP50-95 (0.574 / 0.279 with test-time augmentation).** | `runs/.../bladeyolo2_wtbd_group/test/` |
| **The same model on the published (leaky) WTBD split: 0.840 / 0.493 (0.830 / 0.517 with test-time augmentation).** | `runs/.../bladeyolo2_wtbd_official/test/` |
| The random split inflates mAP50 by +0.29 and doubles mAP50-95. | Same model, same recipe, only the split differs. |
| Stock YOLO12-L, same recipe: 0.438 / 0.206 leak-free, 0.704 / 0.388 published split. The inflation (+0.27 mAP50) is model-independent. | `runs/.../yolo12l_wtbd_*/test/` |
| Annotation granularity (how a defect is cut into boxes) accounts for about +0.12 AP50 on the leak-free test, and WTBD's own copies of the same image agree on only 86% of labels (corrosion 65%; no class conflicts). | Phase 16, Experiments 1 and 3 |
| A recall guarantee calibrated on scenes seen in training fails on new scenes (promised ≤ 10% missed, actual 31.5%); calibrated on unseen scenes it holds (6.3%). | Phase 16, Experiment 2 |
| BladeYOLO-v2 beats YOLO12-L on the leak-free test: +0.161 mAP50 [95% CI 0.025, 0.284], +0.059 mAP50-95 [0.005, 0.169]. **But without the two classes seen in ≤ 3 test scenes the gain is +0.052 [−0.027, 0.135]: not established.** | Paired bootstrap over 58 test scene groups (§12) |
| **Frozen DINOv3 features are the whole advantage, at every data size.** Without them BladeYOLO-v2 equals or trails YOLO12-L; with them it gains +0.039 / +0.088 / +0.090 mAP50-95 at 25 / 50 / 100% of the labelled scenes (all intervals exclude zero). At 25%, a frozen ViT-B/16 adds a further +0.028 [+0.004, +0.096] over ViT-S/16; ViT-L/16 adds nothing beyond ViT-B. | Paired scene bootstrap (§22, §24) |

The project's main contribution is now methodological: a reproducible, leakage-audited protocol for blade defect detection, measured evidence of how much random splits of augmented data inflate published numbers, granularity-aware evaluation, a cross-view label audit, and the finding that leaked calibration breaks recall guarantees. The architecture is BladeYOLO-v2. Against stock YOLO12-L it wins on both splits, but on the leak-free split the win rests largely on one class with 3 test scenes (thunderstrike), so an architectural advantage is suggested, not demonstrated (§12). The DINOv3 ablations (§22, §24) settle it: the advantage comes from the frozen DINOv3 features, not from the architecture around them.

---

## 2. Phases 1–5: the reproduction attempt and the BladeYOLO-L architecture (3 Sep – 6 Oct)

*Summarised from the 6 Oct log (`6f6638a`). Results are on the original Wind Surface Defect split: 2,856 train / 714 val, 5 classes (corrosion, dirt, hide_craze, surface_eye, thunderstrike).*

### Goal
Reproduce BladeYOLO (IEEE TGRS 2026), then extend it beyond the paper's claimed 86.5 / 57.4.

### Paper-faithful architecture (Runs 1–3)
- Frozen DINOv3 ViT-S/16, tapped at blocks 4/8/12.
- Fourier-amplitude Style Injection (SI) into those ViT blocks.
- A detail branch with WTConv (one level in the fine branch, two in the large branch).
- Cross-Mamba P5→P4→P3 guidance, using an in-repo VMamba `CrossSS2D` (commit `2936e46`).
- YOLOv12-L neck and head (commit `9b980e1`).

### Runs

| Run | Dates | Hardware | Architecture | Optimiser | Epochs | mAP50 | mAP50-95 | Outcome |
|---|---|---|---|---|---|---|---|---|
| 1 | 5–7 Sep | T4×2 | Paper-faithful | SGD 0.01 | crashed | — | — | bf16/AMP, DINOv3 unfreezing and DDP failures |
| 2 | 8–13 Sep | T4×2 | Paper-faithful + mosaic | SGD | ~200 | — | — | NaN from FFT gradients in Style Injection |
| 3 | 14–22 Sep | T4×2 | Paper-faithful, FFT detached | SGD 0.001 | ~283 | ~52.6 | ~28 | Stagnated. **Closest reproduction.** |
| 4 | 27–29 Sep | T4×2 | BladeYOLO-S: SI and Mamba removed, LFA (Haar DWT) + A2C2f | SGD | 400 | 64.3 | 37.3 | ~8M params; "information bottleneck" at ~64% |
| 5 | 29 Sep | A100 | Same as Run 4 | AdamW 0.002 | ~9 | — | — | Same ceiling expected; abandoned |
| 6 | 30 Sep – 1 Oct | A100 | BladeYOLO-L: deformable "morphological" convs + adaptive (learnable) wavelets, 57.2M params | AdamW 0.002, cosine | 400 | 78.2 | 46.9 | `bladeyolo_l_sota` (best val 0.781 / 0.469, epoch 392) |
| 7 | 2–6 Oct | A100 | Run 6 + "Cross-Scale Vision Mamba" + WIoU v3 | AdamW 0.002 | 400 | 77.0 (peak) | 47.8 | `bladeyolo_l_sota-2` (best val 0.769 / 0.478, epoch 372) |

### Findings recorded at the time (6 Oct)
- **Reproduction gap.** Run 3 sat 33.9 pp (mAP50) and 29.4 pp (mAP50-95) below the paper. That is far beyond run-to-run variance.
- **Unreproducible from the published information.** The paper reports 3,790 images, but the public Kaggle copy has 3,570. The paper gives no split seed, image list or code. The log stated that the evidence does not establish fabrication.
- **Optimiser.** SGD → AdamW was credited as a breakthrough. Run 4 → Run 6 also scaled 8M → 57M parameters, so the two effects are confounded.
- **Run 6 vs Run 7.** WIoU + Mamba was credited with +0.9 mAP50-95 (Run 7 vs Run 6).

### Conclusions as of 6 Oct, superseded
"BladeYOLO-L reaches ~78 / ~47; the remaining gap is locality bias." Phase 6 and Phase 7 overturned both halves:
- the Run 7 modules were not doing what the log assumed (§3)
- the ~78 / ~47 numbers were inflated by leakage (§4)

---

## 3. Phase 6: code audit and repair (9 Oct)

**Action.** The repo was reset to `4f95ba1` (the Run 7 code) and force-pushed. The four later commits, including `6f6638a` (the log), were removed from `master`. Then both papers, all model code and both 400-epoch `results.csv` files were audited.

### Findings: the Run 7 model was not the model described
| Component | Problem | Consequence |
|---|---|---|
| WIoU v3 | Ultralytics' loss imports `bbox_iou` under its own name, so the monkey-patch never reached it. The formula also had three errors: gradient flowed through β, the distance term wasn't detached, and `.item()` forced a GPU sync every step. | **Run 7 trained with plain CIoU.** Its "+0.9 mAP50-95 from WIoU" is not valid. |
| "Cross-Scale Vision Mamba" (`SimpleSS2D`) | It did not perform a selective scan. | Run 7's "global context" claim is not valid. |
| DINOv3 branch | The custom extractor skipped RoPE positional embeddings and fed unnormalised pixels. | Every run before 9 Oct used weakened DINOv3 features. |
| Wavelet (LFA) | Produced a single-channel sigmoid mask and then average-pooled. | High-frequency detail was thrown away. |
| `train.py` | Overwrote Ultralytics' installed `tasks.py` / `trainer.py` on disk, and silently resumed from any old `last.pt`. | Fragile and irreproducible. |

### Repairs (commit `dfdb5f4` and follow-ups)
- **WIoU v3:** `models/wiou.py` patches only the loss namespace; the TAL assigner keeps CIoU.
- **SS2D:** `models/ss2d.py` is a real VMamba-style four-direction selective scan, using the Mamba-2 chunked formulation in plain PyTorch. It matches a step-by-step reference recurrence to 1.8e-14.
- **DINOv3:** loaded through the official `get_intermediate_layers` with ImageNet normalisation. Weight loading is strict, so missing weights are an error. The source is pinned in `requirements.txt` at commit `6876159a`.
- **Detail branch:** `WaveletDown` keeps all four sub-bands. The deformable convs are now DCNv2 (modulated).
- **Ultralytics integration:** `import models` registers the layers; `models/trainer.py` (`BladeTrainer`) keeps DINOv3 frozen (`freeze=["0.dino"]`) and works under DDP.
- **`train.py`:** AMP on, batch 16, RAM cache, `--resume` explicit, `--box-loss ciou|wiou`, a shared `RECIPE`.

### Conclusion
Runs 6 and 7 measured a different model from the one described. Any comparison between them, or with the paper, has to be redone on fixed code and clean data.

---

## 4. Phase 7: leakage in the Wind Surface Defect dataset (9 Oct)

### Findings
- **Augmented before splitting.** The dataset (from the GFD-YOLOX paper) is 1,785 originals plus one augmented copy each, split at random. Files come in `<name>_0` / `<name>_1` pairs, and **550 pairs are split across train and val**.
- **Pixel-level check.** Every val image was compared with every train image, including flips. **532 of 714 val images (75%)** have a near-identical train image (correlation ≥ 0.99; median best match 0.9988).
- **Second leak.** About 2,274 images are tiles of drone photos (`DJI_xxxx_r_c`, up to 44 tiles per photo). 386 of 882 source photos have tiles in both train and val.
- **Effective size.** The later DINOv2/ORB deduplication found only **1,159 unique source images** among the 3,570 files, with up to 8 copies each, and about 81 independent scenes (§10).

### Conclusions
1. **Runs 6–7 are inflated.** Three-quarters of the val set was effectively seen in training. The val loss was still falling at epoch 400, consistent with memorisation.
2. **The published numbers on this dataset (GFD-YOLOX, BladeYOLO) are probably inflated in the same way.** BladeYOLO reports a random 4:1 split. This may explain "too good" results without any fabrication.
3. **The leak rewards larger models**, because they memorise better.
4. **Methodology from here on:** group-aware splits, a locked test set, model selection on val only, and results reported on test only.

---

## 5. Phase 8: dataset v2, the first leak-free baselines (9–10 Oct)

### Dataset v2: Wind Surface Defect + WTBlade-Defect (fengChe)
Built with `tools/build_dataset.py`:
- **Matching:** DINOv2 retrieval → flip-aware ORB + RANSAC verification.
- **Deduplication:** non-chaining.
- **Grouping:** union-find over filenames, MD5s, parent photos (parsed from tile names), verified matches and DINO cosine (≥ 0.93 = same scene).
- **Split:** 70/15/15, stratified by class over 1,000 trials, then a leak-audit loop until no val or test image matches train.

| | Value |
|---|---|
| Unique images | 2,606 (train 1,823 / val 394 / test 389) |
| Duplicates removed | 9,075 of 11,681 usable images |
| Boxes dropped | 8,967 boxes of WTBlade-only classes (deformity, oil, peeling) |

**WTBlade-Defect:** despite its README, it had been augmented before upload. Its 10,660 images come from 2,794 sources, and its own Roboflow split leaks.

**surface_eye can't be tested.** All 103 of its boxes are tiles of one photo, so it stays in train only. Any surface_eye score on the original dataset, including the papers', was measured on tiles of a training photo.

**Augmentation decision:** no offline augmentation, since pre-split augmentation is what caused the leak. Online mosaic/mixup is used instead, plus `flipud=0.5` (drone views have no canonical up).

### Runs on v2
| Run | Model | Val best (mAP50 / mAP50-95) | Test (mAP50 / mAP50-95) |
|---|---|---|---|
| `yolo12l_baseline` | Stock YOLO12-L (COCO), CIoU | 0.290 / 0.122 (epoch 166 of 266) | 0.358 / 0.149 |
| `bladeyolo_l` | BladeYOLO-L (repaired), WIoU | 0.344 / 0.123 (epoch 84 of 167) | 0.400 / 0.146 at epoch 124 |

- **NaN at epoch 9 under AMP:** the SS2D output overflowed fp16 before its LayerNorm. The LayerNorm now runs in fp32, confirmed with `tools/diagnose_nan.py` on Kaggle (since removed; superseded by `tools/probe_nan.py`). You ruled out skipping batches ("do not skip batches, let's resolve the issue"), so the root cause was fixed instead.
- The BladeYOLO run was stopped because val stagnated.

### Conclusions
- **The leak-free numbers are about half the leaky ones** (~0.35–0.40 vs ~0.78 mAP50). This is the honest starting point.
- **BladeYOLO-L beats the baseline on mAP50 (+4 pp) but not on mAP50-95.** It has higher recall and looser boxes.
- **Data is the binding constraint:** few independent scenes, and one class untestable.

---

## 6. Phase 9: industrial taxonomy and dataset v3 (10–11 Oct)

> **Retired (15 Oct).** All work from Phase 13 on uses each dataset's official classes: WTBD's six classes for every model and evaluation, and each dataset's own classes in the label audits. The custom taxonomy below exists only in the merged v3/v3.1/v3.2 datasets. Their tools stay so the earlier runs can be reproduced. `train.py` no longer falls back to v3 and requires `--data`.

### Taxonomy
The research (and your choice of option B) adopted an industrial blade-surface taxonomy with 6 classes: `leading_edge_erosion`, `contamination`, `crack`, `pitting`, `lightning_damage`, `coating_damage`. Erosion and coating damage are separated by location: coating loss on the leading edge is erosion, elsewhere it is coating damage. Classes outside the taxonomy were dropped.

### Sources assessed
| Source | Decision |
|---|---|
| Wind Surface Defect (WS) | Included |
| WTBlade-Defect / fengChe (WT) | Included (mapped classes) |
| Beijing wind-turbine (Roboflow, BJ) | Included (`surface_oil` → contamination) |
| `wind_turbine_check` | Byte-identical to the Beijing export; not added twice |
| WTBs2025 | Included with rules: drop classes 2 and 8, minimum box 16 px, no GAN images |
| Foster YOLO tiles (586×371) | Overlaps WS; its "damage" class is too generic |
| Wind Turbine Blade Surfaces | 3D metrology, no 2D labels; rejected |

### Dataset v3
5,777 images (train 4,050 / val 863 / test 864), with a per-class independence table in the dataset README.

| class | train boxes / groups | val | test |
|---|---|---|---|
| erosion | 984 / 310 | 211 / 128 | 211 / 107 |
| contamination | 2,042 / 193 | 437 / 190 | 440 / 199 |
| crack | 861 / 98 | 184 / 108 | 187 / 115 |
| pitting | 1,557 / **1** | 53 / 9 | 79 / 7 |
| lightning | 285 / 81 | 61 / 35 | 61 / 35 |
| coating | 1,445 / 159 | 315 / 177 | 309 / 150 |

**Recipe changes:** 150 epochs, `flipud=0.5`, AdamW weight decay 0.01, `--seed` (commit `58db1d1`).

### Conclusion
v3 is far larger in files, but some classes rest on very few independent sources; pitting has a single training group. The scene counts in §10 later confirmed this as the dataset's structural ceiling.

---

## 7. Phase 10: v3 runs, error analysis and label review (11 Oct)

### Runs on v3 (150 epochs)
| Run | Val best | Test (Ultralytics) | Test (own evaluator) | Class-agnostic AP50 |
|---|---|---|---|---|
| `yolo12l_v3` | 0.290 / 0.141 (epoch 104) | 0.263 / 0.109 | 0.245 / 0.106 | 0.283 |
| `bladeyolo_v3` (v1) | 0.356 / 0.156 (epoch 93) | 0.317 / 0.124 | 0.312 / 0.121 | 0.329 |

### Error analysis (`tools/error_analysis.py`)
The analysis breaks errors down TIDE-style, measures AP and recall by box size, and compares coverage recall with IoU-0.5 recall and the recall ceiling. Its report is `runs/analysis/errors/report.md`.
- **Both models plateau after ~100 epochs,** so 150 is enough.
- **Coverage far exceeds IoU-0.5 recall** (pitting: 0.87 coverage vs 0.27–0.38 at IoU 0.5). One region is labelled as several arbitrary sub-boxes, so a correct detection of the whole region counts as a localisation error. This is the **annotation-granularity problem**.
- **Small objects:** recall is 8% for 16–32 px boxes, 20% for medium and 34% for large (baseline, val).
- **Main confusion:** coating_damage → leading_edge_erosion (32 cases), which the location rule partly explains.
- **Real misses remain:** many crack, coating and lightning defects are genuinely missed.

### Label review (`tools/review_labels.py`, `tools/review_page.html`, `tools/apply_review.py`)
- 763 test candidates were reviewed by you: unlabelled detections, labels every model missed, and class conflicts. You observed that many "false positives" were real damage, and many "missed" boxes were blank.
- **Option A:** confirmed unlabelled damage in val/test becomes ignore regions, and train is unchanged. The result is v3.1: 86 class changes, 66 removals and 166 ignore regions in 137 test images.
- **Re-scored test, v3 → v3.1:** baseline 0.245 → 0.272, BladeYOLO 0.312 → 0.341 mAP50.

### Conclusions
- **Label noise is real, but worth only about +0.03 mAP50.** The poor performance is mainly a model and data problem. As you put it: "we first need to train a very good model."
- **BladeYOLO-v1's detail branch is trained from scratch and too shallow.** It gives higher recall but no gain on small or medium boxes. This motivated v2.

---

## 8. Phase 11: BladeYOLO-v2 and the NaN investigations (11–12 Oct)

### BladeYOLO-v2 design (`bladeyolo2-l.yaml`, commit `e77dc0d`)
- **Base:** COCO-pretrained YOLOv12-L for backbone, neck and head, loaded by layer map `[[0,8,0],[9,20,3],[21,21,6]]`. 1,239 tensors transfer; 6 class-score tensors are re-initialised.
- **Additions, all as zero-initialised residual branches,** so training starts exactly at the pretrained detector (verified by `tests/test_model.py`):
  - **`DinoConv` + `DinoFuse`:** frozen DINOv3 semantics fused into P3/P4/P5.
  - **`C2fMorphRes`:** deformable-conv refinement of P3/P4 for irregular contours.
  - **`C2fSS2DRes`:** selective-scan refinement of P5 for global context.
- **Augmentation:** `RandomRot90` (exact box transform; with flips covers all 8 orientations), on with p = 0.5 by default.
- **Evaluation:** a test-time-augmentation pass, saved separately.
- **Size:** 53.7M parameters, 32.1M trainable, 233.5 GFLOPs.

### NaN round 1 (epoch 4)
**Causes:**
- The WIoU running mean is a module-level global. One NaN batch poisoned it permanently, surviving Ultralytics' NaN recovery.
- The recipe (lr0 0.002, `warmup_bias_lr` 0.1, an SGD value) disrupted the pretrained weights during warmup.

**Fix (`2efd069`):** the running mean only updates from finite values; lr0 is 0.001 and `warmup_bias_lr` is 0.0.

### NaN round 2 (epoch 17)
- **Diagnosis (`tools/probe_nan.py`, `b75fd91`):** the probe re-runs the failing batch layer by layer in fp16 and fp32. It found an fp16 overflow in `model.26.cv2.conv` (inside C2fSS2DRes); its input absmax was 2,364, and the fp32 forward was finite.
- **Root cause:** the conv feeds a BatchNorm, so nothing in training bounds its input scale.
- **Fix (`08f9a4b`):** `models/precision.py::FP32Block`, a mixin that keeps C2fSS2DRes and C2fMorphRes in fp32 under AMP and through `.half()`. The run after it was stable.

### `bladeyolo2_v3` result
| | mAP50 | mAP50-95 |
|---|---|---|
| Val (best, epoch 79) | 0.369 | 0.172 |
| Test, v3 labels (evaluator) | 0.296 | 0.123 |
| Test, v3.2 labels | 0.340 | 0.144 |

The test evaluation failed silently on Kaggle, which wasn't understood until §11. Test numbers came from the local evaluator.

### Conclusion
A stronger initialisation did not break the v3 ceiling. v2 matched v1 on test (0.296 vs 0.312 mAP50; 0.123 vs 0.121 mAP50-95). The limit is the data and labels, not the architecture.

---

## 9. Phase 12: the v3 ceiling: region merging, ensembling, honest limits (12 Oct)

- **Merge experiment:** scoring with a merged-region convention gives about +0.04–0.06 mAP50, quantifying the granularity effect.
- **v3.2 (`tools/merge_regions.py`, `b856b3c`):**
  - The first version merged diagonal "staircase" erosion boxes into huge boxes that were mostly background. This was caught visually and fixed with a fill-ratio rule (`--min-fill 0.6 --gap 8`).
  - Box counts: train 7,174 → 6,344, val 1,261 → 1,188, test 1,221 → 1,133.
  - Re-scored test: v1 0.365 / 0.143, v2 0.340 / 0.144, YOLO12-L 0.281 / 0.123.
- **Ensemble (`tools/ensemble.py`, weighted box fusion):**
  - Baseline + v1 + v2 reaches 0.370 / 0.140 at fusion IoU 0.55.
  - With the fusion IoU chosen honestly on val (0.7), it reaches 0.339 / 0.143.
  - About 0.411 / 0.168 on merged labels.
- **Confounder:** the YOLO12-L v3 baseline used the old, harmful recipe (lr 0.002 + 0.1 bias warmup).

### Conclusions
- **About 0.40–0.45 mAP50 is the realistic ceiling on v3.**
- **80–85% mAP50 isn't reachable on these labels.** The earlier high scores came from leakage.

You then set the goal that drove Phase 13: "there seriously has to be a way to produce high accuracy without leakage … a genuinely robust model that can actually be used industrially … without seed roulette or leakage."

---

## 10. Phase 13: new datasets and the content-based audit (12–13 Oct)

### Literature and dataset search
| Dataset | Facts | Verdict |
|---|---|---|
| **WTBD** (Ji et al., *Sci. Data* 13:396, 2026; figshare 10.6084/m9.figshare.30210175, CC BY 4.0) | 1,065 images (1024 px); 6 classes: craze, corrosion, surface_injure, thunderstrike, crack, hide_craze. Double-annotated, κ = 0.897. Authors' split is a seeded random shuffle (seed 42). **No detection baselines in the paper**; its validation is LBP/HOG + t-SNE. | **Chosen** |
| WTBs2025 (*Sci. Data* 13:1310, 2026) | 7,544 images, 9 classes, GA-DCGAN lightning images. Provenance resolved. | Mostly tiny boxes (see the audit) |
| DTU NordTank 2017/18 + Foster annotations | ~13k tiles, CC BY-NC | Possible temporal split; not downloaded |
| Blade30, AQUADA-GO | No download link / segmentation only | Not usable |
| WTBD818-DET (SASED-YOLO, *Sci. Rep.* 2026) | A different "WTBD" with 8 classes | Name collision; don't compare |

### WTBD leakage audit (`tools/wtbd_groups.py`, `565a943`)
- 1,065 images form about 330–368 scene groups (2,034 same-scene pairs).
- **The authors' split leaks:** 115 of 159 val and 123 of 161 test images have a same-scene train image.
- **Group split** (`tools/splits/wtbd_group_split.csv`): train 745 / val 159 / test 161 images, with 1,109 / 240 / 235 boxes. Audited until zero held-out images matched train. Labels are visually clean; median box ~100–200 px at 640.
- 202 WTBD images also appear in v3, and 184 in the Beijing set, so WTBD shares a source with Beijing.

### WTBs2025-clean (`build_dataset.py --native wtbs`)
Only 1,486 of 7,544 images are unique and learnable.
- Removed: 3,169 images with boxes under 16 px, 1,596 pinhole / localised-damage images, 1,235 duplicates and 57 GAN images.
- Oil leakage is 520 files from 29 originals.

### Content-based audit of all six datasets (`tools/audit_datasets.py`; report `runs/analysis/datasets/report.md`)
| Dataset | Files | Unique images | Independent scenes | Median box px @640 | Crop purity |
|---|---|---|---|---|---|
| Wind Surface (v3 source) | 3,570 | 954 | **81** | 54–257 | 0.0–0.92 |
| Beijing / check | 8,922 / 6,563 | 2,196 / 2,005 | **164 / 144** | 52–250 | 0.11–0.89 |
| WTBlade / fengChe | 10,660 | 1,146 | 803 | 57–255 | 0.35–0.80 |
| WTBs2025 | 7,544 | 4,145 | 3,403 | **14–43** | 0.14–0.68 |
| **WTBD** | 1,065 | 597 | 550 | **103–206** | **0.56–0.84** |

**Findings:**
- **v3's ceiling is structural.** Several v3 classes rest on about 10 real scenes, and thunderstrike crops resemble hide_craze 72% of the time.
- **Same-named classes don't match across datasets.** Mutual crop retrieval is ≤ 0.33 apart from bj ↔ bjc, which are the same data, and boxes on shared images rarely agree. Merging classes across datasets would mix labelling conventions.
- **WTBs2025 absorbed fengChe under other class names:** 1,499 shared images, e.g. peeling → erosion.

**Conclusion:** use WTBD alone, with the group split, keeping its 6 native classes. Do not add Beijing, which contains WTBD test photos.

### Effective size of WTBD
WTBD's 745 training images contain more independent defects than v3's 4,050: about 550 scenes in WTBD versus 81 in Wind Surface.

---

## 11. Phase 14: WTBD training: leak-free vs published split (13 Oct)

### Run A: `bladeyolo2_wtbd_group`, first attempt (lost)
- **Progress:** it reached val 0.649 / 0.379 at epoch 97, still rising.
- **Failure at epoch 99:** one NaN batch. The training loss recovered, but **every later validation scored 0**. The NaN had been written into the BatchNorm running statistics, and the EMA copied them. Ultralytics' recovery doesn't trigger on a finite 0 score.
- The run folder was lost when the interactive Kaggle session ended.

**Fixes (`91ae01f`):** in `models/trainer.py`, after every forward pass, the BatchNorm buffers and the loss are checked.
- A poisoned set is restored from the last finite copy, and that optimizer step is skipped (also in fp32, where GradScaler wouldn't skip it). The batch is excluded from the logged loss.
- The first bad batches are saved to `nan_batches/` with a per-layer fp16/fp32 finiteness report.
- Last resort: non-finite model or EMA state before validation triggers a reload of `last.pt`.
- Tested with an injected overflow: exactly one batch dropped, correct layer reported, training continued normally.
- Cost of a dropped batch at the observed rate: ~0.06% of training.

Also added:
- `train.py --test WEIGHTS`
- copying read-only Kaggle datasets into the project folder, so label caches can be written (`1d58ea2`)
- resolving nested test splits

### Run B: `bladeyolo2_wtbd_group` (150 epochs, 2.2 h, no NaN events)
| | mAP50 | mAP50-95 |
|---|---|---|
| Val, best (epoch 124) | 0.634 (final validation of `best.pt`) | 0.366 |
| **Test** | **0.548** | **0.245** |
| **Test with test-time augmentation** | **0.574** | **0.279** |

Per class, test mAP50: thunderstrike 0.98, surface_injure 0.75, hide_craze 0.57, corrosion 0.54, crack 0.38, **craze 0.08**.

**Findings:**
- **Plateau from about epoch 120.** The last 30 epochs averaged ~0.63 val mAP50; the mosaic-off final 10 added nothing. The earlier 0.649 was val noise.
- **Craze is one scene.** All 15 test craze images are rotated, flipped or recoloured copies of one long leading-edge split (2018/09/20 15:51). It was annotated as 2–3 segment boxes, and the model draws one box over two segments. Of 39 boxes: 17 are found at IoU < 0.5, 8 are called corrosion, 13 are missed. Without craze, the other five classes average 0.643 on test, matching val.
- **The weakness is box tightness.** mAP50-95 drops more than mAP50 from val to test; corrosion and crack are worst.

**The Kaggle test evaluation died silently, again.** The cause was found locally (`b8721bc`):
- Ultralytics' FLOPs profiler runs the model on an uninitialised `torch.empty` image.
- The garbage values become NaN offsets in torchvision's `deform_conv2d`. Its bounds check passes NaN, so it reads outside the tensor and segfaults, with no traceback.
- It is random because it depends on memory contents. This also explains the missing test results of `bladeyolo2_v3` (§8).

**Fixes (`b8721bc`):**
- Offsets are made finite and clamped just beyond the image, so valid outputs are bit-identical (`tests/test_deform_offsets.py`: the old code segfaults, the new code doesn't).
- The post-training test runs in a fresh process and prints a rerun command if it fails.

### Run C: `bladeyolo2_wtbd_official` (published random split, 120 epochs, 1.7 h)
| | mAP50 | mAP50-95 |
|---|---|---|
| Val (final validation of `best.pt`) | 0.832 | 0.476 |
| **Test** | **0.840** | **0.493** |
| **Test with test-time augmentation** | **0.830** | **0.517** |

My local CPU evaluation of the official test reproduced the Kaggle number exactly, which validates the local evaluation path.

### Leakage inflation: same model, same recipe, only the split differs
| class (test mAP50) | official | group | |
|---|---|---|---|
| crack | 0.991 | 0.381 | inflated |
| craze | 0.909 | 0.076 | inflated |
| hide_craze | 0.800 | 0.569 | inflated |
| surface_injure | 0.816 | 0.747 | slightly |
| thunderstrike | 0.995 | 0.980 | same |
| corrosion | 0.527 | 0.538 | **not inflated** |
| **all** | **0.840** | **0.548** | **+0.29** (mAP50-95 ×2.0) |

### Leaked vs clean subsets of the official test (`tools/leak_subsets.py`)
125 test images have a train twin and 36 do not.

| subset | mAP50 | mAP50-95 | mAP50 with test-time aug. | mAP50-95 with test-time aug. |
|---|---|---|---|---|
| leaked (125) | 0.851 | 0.498 | 0.844 | 0.530 |
| clean (36) | 0.825 | 0.508 | 0.814 | 0.567 |

**Inconclusive.** The clean subset has only one box each of craze, crack and thunderstrike. Those classes score 0.995 there and inflate the subset's mean, and they are exactly the classes behind the inflation, so the subset can't test it.

On the three classes with enough boxes, leaked vs clean:

| class | leaked | clean |
|---|---|---|
| corrosion | 0.61 | 0.38 |
| surface_injure | 0.87 | 0.69 |
| hide_craze | 0.73 | 0.90 |

Directionally consistent with leakage, but only 50 boxes.

### Conclusions
1. **The published WTBD protocol overstates detection performance by about 0.29 mAP50.** The inflation is concentrated in classes whose test images are copies of a few training scenes.
2. **BladeYOLO-v2's honest WTBD performance is 0.55–0.57 mAP50.** It is strong on visually distinctive classes (thunderstrike, surface_injure) and weak on box tightness for irregular shapes.
3. **The 161-image test set is small.** Per-class numbers rest on few scenes (craze: one), so confidence intervals over scene groups are needed.
4. **The leaked/clean subset test can't separate the effect at this sample size.** The group-vs-official comparison is the evidence.

---

## 12. Phase 15: YOLO12-L baselines and scene-level confidence intervals (14 Oct)

### Baseline runs
Stock COCO-pretrained YOLO12-L with its own CIoU loss, on the same recipe, seed, image size, epochs and augmentation as BladeYOLO-v2 (`--model yolo12l.pt --box-loss ciou`). Both runs completed on Kaggle with test and test-time-augmentation metrics, so the segfault fix (`b8721bc`) held.

| model | WTBD-group test | with test-time aug. | WTBD official test | with test-time aug. | val best (group / official) |
|---|---|---|---|---|---|
| BladeYOLO-v2 | **0.548 / 0.245** | 0.574 / 0.279 | 0.840 / 0.493 | 0.830 / 0.517 | 0.650 / 0.368 · 0.845 / 0.475 |
| YOLO12-L | **0.438 / 0.206** | 0.440 / 0.230 | 0.704 / 0.388 | 0.734 / 0.438 | 0.419 / 0.269 · 0.665 / 0.394 |

Per-class test mAP50 (Ultralytics):

| class | BladeYOLO-v2 group | YOLO12-L group | BladeYOLO-v2 official | YOLO12-L official |
|---|---|---|---|---|
| craze | 0.076 | 0.065 | 0.909 | 0.622 |
| corrosion | 0.538 | 0.608 | 0.527 | 0.460 |
| surface_injure | 0.747 | 0.629 | 0.816 | 0.711 |
| thunderstrike | 0.980 | 0.418 | 0.995 | 0.938 |
| crack | 0.381 | 0.384 | 0.991 | 0.854 |
| hide_craze | 0.569 | 0.526 | 0.800 | 0.639 |

**Leakage inflation is model-independent.** The published split adds +0.266 mAP50 for YOLO12-L and +0.292 for BladeYOLO-v2; mAP50-95 rises ×1.9 and ×2.0.

### Scene-level confidence intervals (`tools/bootstrap_ci.py`)
Test images are grouped into scenes with the split's matching (DINOv2 + ORB). Each replicate redraws scene groups with replacement and scores both models on the same draw (2,000 replicates; COCO 101-point AP). Reports are in `runs/analysis/bootstrap/{wtbd_group,wtbd_official}/report.md`. This evaluator's point estimates differ slightly from Ultralytics': BladeYOLO-v2 0.561 vs 0.548 and YOLO12-L 0.400 vs 0.438 on the group test.

| test set | scene groups | BladeYOLO-v2 mAP50 [95% CI] | YOLO12-L mAP50 [95% CI] | paired difference mAP50 | paired difference mAP50-95 |
|---|---|---|---|---|---|
| WTBD-group | **58** (craze 2, thunderstrike 3, crack 8, surface_injure 14, corrosion 18, hide_craze 19) | 0.561 [0.484, 0.809] | 0.400 [0.340, 0.675] | **+0.161 [0.025, 0.284]**, P(≤0) = 0.005 | +0.059 [0.005, 0.169], P(≤0) = 0.017 |
| WTBD-group without craze and thunderstrike | 58 | 0.567 | 0.515 | **+0.052 [−0.027, 0.135]**, P(≤0) = 0.077 | +0.015 [−0.032, 0.067], P(≤0) = 0.236 |
| WTBD official | 108 | 0.815 [0.775, 0.857] | 0.690 [0.635, 0.752] | +0.125 [0.074, 0.176], P(≤0) < 0.001 | +0.103 [0.065, 0.139], P(≤0) < 0.001 |

The class-subset check is `runs/analysis/bootstrap/robust_check.py`. The upper CI bounds on the full group test are skewed high: replicates that miss the two craze scenes drop the weakest class from the mean.

### Findings
1. **BladeYOLO-v2 is better than YOLO12-L on the leak-free test overall, and the interval excludes zero.** Most of the gain is thunderstrike (1.00 vs 0.27 in the bootstrap evaluator), seen in only 3 test scenes. On the four classes with 8–19 scenes the gain is +0.05 mAP50, with an interval that includes zero.
2. **On the official split BladeYOLO-v2 wins every class with a tight interval.** That split rewards memorising scenes seen in training, which a larger model with frozen DINOv3 features does well. It is not evidence of better generalisation.
3. **The leak-free test set is small in the dimension that matters.** It has 161 images but only 58 independent scenes, versus 108 in the official test, because the group split keeps whole scenes together. Per-class results for craze, thunderstrike and crack rest on 2–8 scenes.
4. **Box tightness remains the shared weakness.** Corrosion mAP50-95 is 0.13 for BladeYOLO-v2 and 0.22 for YOLO12-L on group test. Corrosion is the one class where YOLO12-L is ahead on both mAP50 and mAP50-95.

### Conclusions
- **The leakage result is robust:** both models, both metrics, about +0.27–0.29 mAP50 from the random split.
- **The honest claim for the architecture:** "BladeYOLO-v2 scores 0.55 vs 0.44 mAP50 on a leak-free WTBD split (paired bootstrap +0.16, 95% CI 0.03–0.28). The gain is concentrated in lightning damage; on the other classes the evidence is inconclusive."
- **More independent test scenes are needed** before per-class or architectural claims can be made with confidence. Options are group k-fold cross-validation or a second dataset (§29).

### Cleanup in this phase
- Removed `tools/diagnose_nan.py`: a one-off investigation of the fixed SS2D fp16 bug, superseded by `tools/probe_nan.py` and the trainer's per-layer NaN report.
- Removed `tools/eval_by_source.py`: per-source evaluation on v3, which you judged not useful.
- Removed the unused function `find_image` (`build_dataset.py`), two unused parameters of `gallery` (`error_analysis.py`) and an unused variable (`probe_nan.py`). `vulture` now reports no dead code.
- Kept the v1 modules (`backbone.py`, `wavelet.py`, `C2fMorph`, `C2fSS2D`). `bladeyolo-l.yaml` and the v1 checkpoints need them.

---

## 13. Phase 16: evaluation experiments: granularity, label consistency, recall guarantees (14 Oct)

**Motivation.** Earlier phases repeatedly found that box-level scores penalise *how* a defect is cut into boxes (v3 coverage vs IoU recall, §7; the merge experiment, §9; the craze test scene, §11). A literature check found:
- many-to-one matching for evaluation in scene-text detection (TedEval, 2019)
- box-noise-aware AP (Murrugarra-Llerena et al., CVPRW 2022 / WACV 2025)
- conformal object detection (Andéol et al., 2023 / 2025), and conformal segmentation for steel defects (2025)

Nothing combines these for amorphous surface defects, with leakage-aware calibration, or cross-view label audits. Three experiments, none needing new training:

### Experiment 1: granularity-invariant evaluation (`tools/granularity_eval.py`)
Three scores on the same cached test predictions:
- **standard:** one-to-one, IoU ≥ 0.5
- **tolerant:** one prediction may match several same-class labels whose union it fits at IoU ≥ 0.5 (each label counts towards recall), and fragments of an already-found defect are ignored
- **region:** pixel-level precision/recall of the union of predicted boxes vs the union of labelled boxes per class (4 px grid)

Scene-group bootstrap: 500 replicates. Unit tests: `tests/test_granularity_eval.py`. Reports: `runs/analysis/novel/granularity/{wtbd_group,wtbd_official}/report.md`.

| test set | model | standard AP50 | tolerant AP50 | region AP | standard AP50-95 | tolerant AP50-95 |
|---|---|---|---|---|---|---|
| WTBD-group | BladeYOLO-v2 | 0.561 [0.483, 0.791] | **0.680** [0.604, 0.831] | 0.603 [0.552, 0.743] | 0.244 | 0.306 |
| WTBD-group | YOLO12-L | 0.400 [0.339, 0.678] | **0.524** [0.465, 0.740] | 0.441 [0.402, 0.592] | 0.185 | 0.246 |
| WTBD-group | difference | +0.161 [0.023, 0.284] | +0.156 [0.001, 0.237] | **+0.162 [0.077, 0.236]** | +0.059 | +0.061 |
| WTBD official | BladeYOLO-v2 | 0.815 | 0.889 | 0.771 | 0.481 | 0.509 |
| WTBD official | YOLO12-L | 0.690 | 0.800 | 0.666 | 0.377 | 0.433 |

Per class on WTBD-group, BladeYOLO-v2 AP50 standard / tolerant / region:

| class | standard | tolerant | region |
|---|---|---|---|
| craze | 0.099 | **0.661** | 0.427 |
| corrosion | 0.626 | 0.676 | 0.684 |
| surface_injure | 0.759 | 0.816 | 0.689 |
| thunderstrike | 1.000 | 1.000 | 0.682 |
| crack | 0.352 | 0.356 | 0.491 |
| hide_craze | 0.530 | 0.568 | 0.645 |

**Findings:**
1. **Box splitting accounts for about +0.12 AP50 for both models.** That is roughly 27% of BladeYOLO-v2's apparent AP50 shortfall on the leak-free test. Craze goes from 0.099 to 0.661: the "craze failure" of §11 is almost entirely the split-segment labelling.
2. **The model comparison is stable across all three scores** (+0.16 each). The region metric gives the tightest interval for the difference ([0.077, 0.236]). It also shows a crack gap (region 0.49 vs 0.25) that box AP hides (0.35 vs 0.39).
3. **Hypothesis (tested in Phase 17, step 14: not supported): part of the leakage inflation is memorised box splitting.** The gap between the official and leak-free splits shrinks as the metric becomes split-invariant:

   | | standard | tolerant | region |
   |---|---|---|---|
   | BladeYOLO-v2 | +0.254 | +0.209 | +0.168 |
   | YOLO12-L | +0.290 | +0.276 | +0.225 |

   A test image whose twin was in training also carries that twin's box splitting, and a model can reproduce the splitting, not only find the defect. This is consistent with Experiment 3, but not proven.

### Experiment 3: cross-view label audit (`tools/view_label_audit.py`)
**Method:**
- For every pair of WTBD images showing the same surface (DINOv2 nearest neighbours, then flip-aware ORB + RANSAC similarity transform, ≥ 25 inliers), project each label into the other image.
- Classify it as agree (same class, IoU ≥ 0.5), class conflict, split (cut into boxes differently), loose, or missing.
- **Local verification:** a label is audited only if ≥ 8 of the match's inlier keypoints fall inside its own box. Without this, matches on burned-in captions and sky linked different blades: blade numbers 324 vs 322 and 326 were seen on example sheets, and "missing" fell from 5.0% to 1.4% once the check was added.
- **Strata:** same view (the moved frame covers ≥ 70% of the other image: augmented copies, exact alignment) vs other view (perspective not modelled: an upper bound on disagreement).

Unit test: `tests/test_view_label_audit.py` (recovers a known flip + rotation + scale at IoU 0.985). *(Phase 17 refines the strata: on axis-aligned same-view copies WTBD agreement is 87.3%; rotated copies 77.9%, where "loose" includes the geometric enlargement of rotated boxes.)* Example sheets were checked by eye: `runs/analysis/novel/label_audit/wtbd/examples_*.jpg`.

| class | same-view agreement | split | loose | missing | other-view agreement |
|---|---|---|---|---|---|
| corrosion | **65.4%** | 115 | 43 | 1 | 33.5% |
| crack | 94.6% | 0 | 46 | 0 | 84.4% |
| craze | 91.2% | 5 | 60 | 4 | 66.7% |
| hide_craze | 82.9% | 5 | 25 | 3 | 40.0% |
| surface_injure | 81.4% | 44 | 77 | 40 | 68.5% |
| thunderstrike | 94.7% | 0 | 15 | 0 | 95.8% |
| **all** | **86.0%** (2,957 of 3,440) | 4.9% | 7.7% | 1.4% | 64.1% (733 of 1,144) |

The table covers 2,908 same-view and 1,133 other-view verified directed pairs.

**Findings:**
1. **WTBD's augmented copies were annotated independently, and inconsistently in extent, never in class.** 14% of labels on exactly aligned copies of the same image disagree, and there are **zero class conflicts**. The disagreement is about where damage starts and ends, and how it is cut into boxes. Corrosion along the blade edge is one long box in one copy and 3–4 pieces in another; the thunderstrike patch's edge strips are labelled in some copies and not others.
2. **Self-consistency bounds what box AP can measure.**
   - Corrosion copies agree only 65% of the time, and the models score 0.54–0.63 AP50. That is near the annotation's own consistency, so further corrosion "improvement" is mostly unmeasurable with these labels.
   - Crack copies agree 95%, while the models score 0.35–0.39. **Crack is a genuine model weakness.**
3. **This supports Experiment 1.** The label noise is the kind the tolerant and region metrics are designed to absorb (extent and splitting). It also explains how leakage can inflate box metrics beyond defect recognition.
4. **Caveat:** disagreements concentrate in a few heavily duplicated scenes (e.g. the thunderstrike patch scene). Counts are per label occurrence, not per independent scene.

### Experiment 2: conformal recall guarantees and leaked calibration (`tools/conformal_recall.py`)
Conformal risk control picks a confidence threshold so that the expected share of missed defects is ≤ α. A defect counts as flagged if ≥ 50% of its box is covered by predictions above the threshold, of any class. Unit tests: `tests/test_conformal_recall.py` (the guarantee holds on synthetic data: mean miss 0.091 at α = 0.1).

The comparison:
- calibrating on **seen scenes** (group-train for the group model, or the official val split, 115 of 159 with train twins) vs on **unseen scenes** (group val)
- deploying on unseen scenes in every case (group test, or the 36 official test images without a twin)

Predictions were cached for the group-train, group-val and official-val splits (`runs/analysis/novel/predict_all.py`). Each setting uses 500 draws of 100 calibration images. Report: `runs/analysis/novel/conformal/report.md`.

| model | calibrated on | deployed on | target α | realised miss rate | draws violating α | predictions / image (*not* false alarms, see the Phase 17 correction) |
|---|---|---|---|---|---|---|
| BladeYOLO-v2 (group) | seen scenes (group-train) | unseen (group test) | 0.05 | 0.188 | 100% | 1.4 |
| | | | **0.10** | **0.315** | **100%** | 1.0 |
| | | | 0.20 | 0.570 | 100% | 0.6 |
| BladeYOLO-v2 (group) | unseen scenes (group val) | unseen (group test) | 0.05 | 0.031 | 0% | 3.7 |
| | | | **0.10** | **0.063** | **0%** | 2.4 |
| | | | 0.20 | 0.116 | 0% | 1.7 |
| YOLO12-L (group) | seen scenes | unseen | 0.10 | 0.376 | 100% | 1.1 |
| YOLO12-L (group) | unseen scenes | unseen | 0.10 | 0.048 | 0% | 4.5 |
| BladeYOLO-v2 (official) | official val (leaky) | test with a train twin (125) | 0.10 | 0.126 | 91% | 1.5 |
| | | test without a twin (36) | **0.10** | **0.259** | **100%** | 1.3 |
| YOLO12-L (official) | official val (leaky) | test with a twin | 0.10 | 0.071 | 8% | 3.1 |
| | | test without a twin | 0.10 | 0.116 | 100% | 2.5 |

**Findings:**
1. **Calibrating on scenes the model has seen invalidates the guarantee.** Promised at most 10% missed defects, BladeYOLO-v2 misses 31.5% on new scenes, in every draw. Models are overconfident on what they memorised, so the calibrated threshold is far too high for new data.
2. **Calibrating on unseen scenes restores it.** 6.3% missed at α = 0.1, zero violations, slightly conservative. *(Corrected in Phase 17: the price in real false alarms is 0.2 instead of 0.1 per image, not 2.4 instead of 1.0.)*
3. **The published-split workflow fails in the same way.** Calibrating on the leaky official val split looks almost acceptable on test images that have a twin in train. On genuinely new images, the 36 without a twin, it misses 2.6× the promised rate. A leaky test set would hide the failure.
4. **Practical rule for inspection systems:** calibration data must be split by scene from the training data, as test data must. A random split of augmented drone imagery gives a recall guarantee that does not hold in the field.

**Caveats:**
- Calibration units are images. Images of one scene are correlated, so group-level calibration (scene groups as units) would be the strict version.
- The deployment sets are small: 161 images in 58 scenes, and 36 images.
- Every deployment set comes from one wind farm.

### Conclusions of Phase 16
- **Box-level AP on amorphous defects mixes detection quality with annotation convention.** About 0.12 AP50 on WTBD is box splitting (Experiment 1), and WTBD's own copies of the same image agree on only 86% of labels, 65% for corrosion (Experiment 3). Granularity-tolerant and region scores should be reported alongside AP.
- **Leakage does more than inflate scores.** Part of the inflation looks like memorised box splitting (Experiment 1, a hypothesis). Most consequentially, it breaks the statistical guarantees an industrial system would rely on (Experiment 2).
- **Together with Phases 14–15,** this is a coherent methodological contribution: leakage in published blade datasets, measured inflation, granularity-aware evaluation, a label-consistency audit, and scene-aware calibration.

---

## 14. Phase 17: follow-ups: split-tolerant training, scene-level guarantees, label audits on two more datasets (15 Oct)

Steps 10–14 of the Phase 16 next steps. Steps 12–14 need no training. Step 10 is implemented and tested, and its Kaggle run is pending. Step 11 is deferred by design (below).

### Correction to Phase 16, Experiment 2
The "false alarms per image" column in Phase 16 counted **all** predictions above the threshold, correct detections included. `tools/conformal_recall.py` now counts only predictions that lie on no labelled defect (IoU < 0.1 with every label, and < 50% of the prediction inside labels). With real false alarms, image units, unseen-scene calibration, α = 0.1: BladeYOLO-v2 has 0.2 false alarms per image (not 2.4), and YOLO12-L 0.8. The miss rates and violation rates are unchanged.

### Step 14: is part of the leakage inflation memorised box splitting? Not supported.
Prediction: if leaky models reproduce the annotator's box splitting on test images whose twin was in training, the tolerant score should add *less* on those images. Test: `tools/granularity_eval.py --subsets runs/analysis/wtbd_leak/test_leak.csv` (official models, official test). It compares only classes with ≥ 5 labels in both subsets (corrosion, surface_injure, hide_craze); scene bootstrap, 300 replicates.

| model | tolerant − standard AP50, test images with a train twin (125) | without a twin (36) | difference |
|---|---|---|---|
| BladeYOLO-v2 | +0.131 [0.074, 0.184] | +0.124 [0.032, 0.224] | +0.007 [−0.112, 0.111] |
| YOLO12-L | +0.206 [0.115, 0.266] | +0.087 [0.012, 0.178] | +0.119 [−0.008, 0.217] (opposite direction) |

The gain is the same (BladeYOLO-v2), or larger on twin images (YOLO12-L). **The hypothesis is not supported.** The shrinking leakage gap under split-invariant metrics (Phase 16) is better explained by the two test sets' composition: the leak-free test contains the heavily split craze scene. Power is limited: 36 clean images, 3 classes. Report: `runs/analysis/novel/granularity/wtbd_official_twins/report.md`.

### Step 12: scene-level and class-aware recall guarantees, operating curve
`tools/conformal_recall.py` now takes scene groups as units (images of one surface pooled; 224 train, 60 val and 58 test groups) and optionally class-aware flagging (only predictions of the defect's own class count). It also writes an operating curve: realised recall vs false alarms over α, calibrated on unseen scenes. Report and figure: `runs/analysis/novel/conformal/{report.md, curve.png, curve.csv}`.

| model | units, flagging | calibrated on | α | realised miss rate on new scenes | violations | false alarms / image |
|---|---|---|---|---|---|---|
| BladeYOLO-v2 | scene, any class | seen scenes | 0.10 | **0.288** | 100% | 0.1 |
| BladeYOLO-v2 | scene, any class | unseen scenes | 0.10 | **0.047** | 1% | 0.2 |
| YOLO12-L | scene, any class | seen scenes | 0.10 | 0.274 | 100% | 0.1 |
| YOLO12-L | scene, any class | unseen scenes | 0.10 | 0.123 | 71% | 0.4 |
| BladeYOLO-v2 | scene, class-aware | unseen scenes | 0.10 | 0.075 | 7% | 0.6 |
| YOLO12-L | scene, class-aware | unseen scenes | 0.10 | 0.085 | 11% | 4.1 |

Operating points (scene units, calibrated on unseen scenes, any class):

| α | BladeYOLO-v2 recall / false alarms per image | YOLO12-L recall / false alarms per image |
|---|---|---|
| 0.1 | **0.953 / 0.23** | 0.873 / 0.41 |
| 0.2 | 0.866 / 0.08 | 0.805 / 0.17 |
| 0.3 | 0.705 / 0.05 | 0.651 / 0.08 |

**Findings:**
1. **Seen-scene calibration breaks the guarantee with scene units too** (28.8% missed against a promised 10%, every draw). The Phase 16 result does not depend on how the units are defined.
2. **With unseen-scene calibration and scene units,** BladeYOLO-v2 keeps the guarantee (4.7% missed, 1% of draws violate). YOLO12-L misses 12.3% on average, violated in 71% of draws. Conformal guarantees hold in expectation over exchangeable calibration and deployment scenes; with only 60 calibration and 58 deployment scenes from one site, the val and test scene groups are not perfectly exchangeable. Treat the YOLO12-L figure as a real-world shortfall, not a bug.
3. **The operating curve is the most practical comparison so far.** At every operating point up to about 1 false alarm per image, BladeYOLO-v2 finds more defects than YOLO12-L: at 0.1 false alarms per image, about 90% vs about 71% of defects on new scenes. Requiring the correct class costs BladeYOLO-v2 little, but YOLO12-L a lot (4.1 false alarms per image at α = 0.1). Caveat: one test set of 58 scenes, no interval on the curve yet.

### Step 13: cross-view label audit on Wind Surface Defect and WTBlade-Defect (done)
`tools/view_label_audit.py` changes in this phase:
- **Paths kept as given,** so cached embeddings are reused.
- **Only each label's local keypoint support is kept,** not the keypoints themselves. Stored keypoints would have needed several GB of RAM for WTBlade's 143,344 transforms; the WTBD report recomputed this way is byte-for-byte identical.
- **Same-view pairs are split by rotation.** The example sheets showed that Roboflow and WTBD copies include arbitrary-angle rotations; the axis-aligned box of a rotated box is larger, which inflates "loose". The **axis-aligned** stratum (rotation within 3° of a multiple of 90°, flips included) is now the headline: there boxes map to boxes and every difference is an annotation difference.

Example sheets were checked by eye for every dataset and outcome.

| dataset | same-view axis-aligned labels | agree | split | loose | missing | class conflicts | same-view rotated agreement | other-view agreement |
|---|---|---|---|---|---|---|---|---|
| WTBD | 2,965 | **87.3%** | 4.6% | 6.7% | 1.5% | 0 | 77.9% | 64.1% |
| WTBlade-Defect | 42,812 | **86.0%** (85.9% without dirt) | 2.5% | 8.2% | 3.1% (1.1% without dirt) | 91 (0.2%) | 80.7% | 59.8% |
| Wind Surface Defect | 10,261 | **83.8%** | **9.2%** | 5.9% | 1.1% | 0 | (18 labels only) | 44.2% |

Agreement by class on axis-aligned copies. The amorphous classes disagree most; compact, distinct defects least.

| dataset | lowest | highest |
|---|---|---|
| WTBD | corrosion 69.0%, surface_injure 82.7%, hide_craze 84.4% | thunderstrike 96.4%, crack 95.1%, craze 93.7% |
| WTBlade | crack 81.5%, peeling 83.7%, rust 85.2% | deformity 96.9%, oil 96.1%, burn 91.7% |
| Wind Surface | surface_eye 79.2%, corrosion 80.7%, dirt 81.2% | hide_craze 97.0%, thunderstrike 87.8% |

**Findings from the WTBlade sheets** (not documented by its Roboflow page):
1. **Copy-paste synthetic images.** Several images are a blade with lettering ("NON…") and defect patches pasted onto it. The same pasted patch is labelled *peeling* in one copy and *crack* in another; these are most of the 91 class conflicts.
2. **DTU NordTank imagery.** "NORDTANK" is legible on many nacelles, so the set contains the DTU drone inspection images (Shihavuddin et al.), whose licence is CC BY-NC, not the CC BY 4.0 the Roboflow page states. **Any publication using WTBlade must cite and respect the original licence.**
3. **Drone video of a rotating rotor.** Consecutive frames align on the static ground and tower while the blades move, so a dirt label on a blade lands on empty ground in the next frame. That is a matching artefact, not an annotation error. It accounts for most WTBlade "missing" (dirt: 1,080 of 1,320), which is why results are also given without dirt.

Wind Surface's same-view pairs are mostly overlapping tiles of one drone photo at different zoom, each labelled independently. Leading-edge erosion is drawn as a different chain of boxes in each tile, hence the high split share.

**Conclusion:** on exactly aligned, unrotated copies of the same image, **13–16% of labels disagree** in all three datasets: WTBD (2026, double-annotated), WTBlade and Wind Surface (the original BladeYOLO paper's dataset). Almost all of it is extent and box splitting, not class. Box-level AP on these datasets therefore has a ceiling well below 1.0 that differs by class. The amorphous classes (corrosion, peeling, crack) are the least self-consistent.

### Step 10: split-tolerant training (`models/tolerant.py`, `train.py --tolerant`), ready for Kaggle
Every label gets an alternative target: the union of the touching same-class labels it belongs to (gap ≤ 8 px), used only if the union is not mostly background (member area ≥ 60% of the union, the v3.2 staircase rule).
- The task-aligned assigner scores an anchor by its better overlap with the label or its region.
- The box loss regresses towards whichever of the two the (detached) prediction overlaps more.
- Classification targets and labels without touching neighbours are unchanged.
- It composes with WIoU, and is switched on by the env var `BLADEYOLO_TOLERANT` so DDP workers agree.

Tests: `tests/test_tolerant.py`.
- Regions merge touching same-class labels only, and diagonal staircases are rejected.
- The box loss accepts the whole region or the labelled half.
- When every anchor predicts the whole region, the tolerant assigner gives it 4.0 classification target mass where the standard one gives 1.76.
- On a real YOLO model the loss is identical to standard when no labels touch, and differs when they do.
- A 2-epoch CPU training with BladeTrainer + WIoU runs, uses `TolerantDetectionLoss`, and the box loss falls.

Kaggle command (one run, ~2.2 h):
```
python train.py --data .../WTBD-group/data.yaml --model bladeyolo2-l.yaml --tolerant --name bladeyolo2_wtbd_group_tolerant --epochs 150
```
Then score with `tools/bootstrap_ci.py` and `tools/granularity_eval.py` against `bladeyolo2_wtbd_group` (standard, tolerant and region AP, paired intervals), and with the conformal operating curve. Expected: higher tolerant and region scores, and standard AP roughly unchanged or slightly lower. Whether it pays off is an open question.

### Step 11: reward fine-tuning (RL), deferred by design
REINFORCE fine-tuning on the region or tolerant score ("Tuning computer vision models with task rewards", 2023) is the expensive, high-variance way to optimise the same objective that step 10 targets directly. Plan: run step 10 first.
- **If the tolerant run already moves the region score:** RL is unnecessary.
- **If it doesn't:** fine-tune `bladeyolo2_wtbd_group/best.pt` for 10–20 epochs. Sample detections from the per-anchor class scores, use reward = region AP on the image (with a running-mean baseline), add the log-likelihood gradient to the standard loss, and keep a KL anchor to the pretrained model.

---

## 15. Phase 18: split-tolerant training run and the YOLOv12 attention fp16 overflow (15 Oct)

### The run
`bladeyolo2_wtbd_group_tolerant`: BladeYOLO-v2 on WTBD-group with `--tolerant` (models/tolerant.py), otherwise the same recipe and seed as `bladeyolo2_wtbd_group`. 150 epochs on Kaggle. The log is `runs/detect/BladeYOLO_WindSurface/bladeyolo2_wtbd_group_tolerant/bladeyolo2_wtbd_group_tolerant.log`.

**Five non-finite batches** (epochs 111, 113, 140, 148, 149, on both GPUs) were each dropped by the per-batch guard (§11). No reload was needed, and the run completed with test and test-time-augmentation metrics.

### Root cause of the late-training NaN: YOLOv12 area attention in fp16
**Diagnosis:**
- The guard's per-layer reports (`nan_batches/*.json`) put the first non-finite output at **layer 6 (stock `A2C2f`, P4)** in all five batches, in fp16 only (fp32 finite). The block's input was small (absmax 50–70).
- Replaying the saved batches through layer 6 (`runs/analysis/nan_probe/layer6.py`, final weights) reproduces three of the five. The NaN first appears at `m.1.0.attn.proj`, whose input, the attention output, is already non-finite.

**Cause:** Ultralytics' `AAttn.forward` forms the scores as an explicit fp16 product, `(q * d^-0.5)^T @ k`. With q/k activations around 270 and 32-dimensional heads, a score reaches about 270/√32 × 270 × 32 ≈ 4×10⁵ > 65,504 (the fp16 maximum). It becomes inf, and the softmax returns NaN. This is a latent fp16 bug in stock YOLOv12 that appears late in training as activations grow. It is not caused by the tolerant loss (the layer is stock YOLO12-L), and it is the likely cause of the lost run's epoch-99 NaN (§11).

**Fix (`models/precision.py`):** `AAttn.forward` is replaced at import (8.4.174 only, version-guarded) by the same code with the scores and softmax computed in fp32. The probabilities are ≤ 1, so the product with v stays in fp16. Attention runs only in backbone layers 6 and 8, on ≤ 400 tokens per area, so the cost is negligible.

**Tests** (`tests/test_fp32_blocks.py::test_area_attention_scores_fp32`):
- identical to stock in fp32
- in fp16 below overflow, no further from fp32 than stock
- above overflow, stock is non-finite and the fix is finite

All five real NaN batches pass layer 6 finite with the fix.

### Results: split-tolerant vs standard training
Test, Ultralytics metrics: tolerant **0.523 / 0.249** (with test-time augmentation 0.552 / 0.282); standard 0.548 / 0.245 (0.574 / 0.279). Best val epoch 83 vs 124.

Paired scene bootstrap (`runs/analysis/bootstrap/wtbd_group_tolerant/`, `runs/analysis/novel/granularity/wtbd_group_tolerant/`):

| score (tolerant − standard) | difference | 95% interval |
|---|---|---|
| standard AP50 | −0.031 | [−0.157, 0.038] |
| tolerant AP50 | +0.015 | [−0.111, 0.064] |
| region AP | −0.015 | [−0.093, 0.042] |
| standard AP50-95 | −0.003 | [−0.054, 0.032] |
| tolerant AP50-95 | +0.014 | [−0.041, 0.046] |

Per class, standard / tolerant / region AP50:

| class | split-tolerant training | standard training |
|---|---|---|
| corrosion | 0.498 / **0.836** / **0.757** | 0.626 / 0.676 / 0.684 |
| crack | 0.399 / 0.409 / **0.572** | 0.352 / 0.356 / 0.506 |
| surface_injure | 0.698 / 0.748 / **0.532** | 0.759 / 0.816 / 0.735 |
| hide_craze | 0.545 / 0.617 / 0.644 | 0.530 / 0.568 / 0.646 |
| craze | 0.095 / 0.605 / 0.423 | 0.099 / 0.661 / 0.429 |
| thunderstrike | 0.945 / 0.950 / 0.662 | 1.000 / 1.000 / 0.681 |

Ultralytics per-class mAP50-95 for corrosion: 0.13 → **0.26**.

**Findings:**
1. **No measurable overall effect** on any score: every interval spans zero.
2. **The effect is class-specific, in the direction the label audit predicts.** Corrosion, the least self-consistent class (69% agreement on aligned copies), improves strongly on the split-invariant scores, and its mAP50-95 doubles. Crack improves on region AP. **surface_injure gets worse** (region 0.74 → 0.53): its touching labels are more often genuinely distinct defects, so the region target is wrong for it.
3. **One run and one seed:** per-class shifts cannot yet be separated from run-to-run variance.

**Conclusions:**
- Blanket split tolerance is not an improvement.
- The evidence points to **class-selective tolerance**: apply it only to classes whose labels are inconsistently split (corrosion; possibly craze and hide_craze), chosen from the label audit, not from test results.
- Step 11 (RL fine-tuning on the region score) stays deferred: it would optimise the same blanket objective, which did not move the aggregate.

---

## 16. Phase 19: repository cleanup and a new direction: real-view consistency for data-limited detection (15 Oct)

### Decision
The analysis results of Phases 13–18 (leakage, label self-consistency, broken guarantees) are solid, but they are evaluation work. The goal from here is a **computer-vision method contribution**: reliable defect detectors from limited labelled data.

What is no longer new, from a literature check:
- **Box-to-dense supervision with foundation models:** Boxes2Pixels (2026) turns boxes into SAM masks and trains on frozen DINOv2, evaluated on wind-turbine inspection.
- **DINOv3 as an inspection backbone:** benchmarked against ImageNet pretraining (2026).
- **Training-free few-shot defect segmentation** with DINO and SAM (2025).

### What limits us (from Phases 13–18)
1. **Few independent scenes, not few images.** WTBD has about 330 scenes. Synthetic augmentation adds pixels, not new views of a defect.
2. **Drone inspection produces overlapping real views of the same surface,** and `tools/matching.py` registers them with verified similarity transforms (about 4,000 WTBD pairs, about 70,000 WTBlade pairs). Until now they were only a leakage hazard.
3. **Most available imagery is unlabelled** (sweeps, video frames, DTU NordTank).

### Method to develop: real-view consistency (semi-supervised detection)
In teacher–student semi-supervised detection, a teacher (an exponential moving average of the student) pseudo-labels unlabelled images, and the student learns to match them under synthetic augmentations.

**Here, the second view is a genuinely different photograph of the same surface.** The teacher's detections on view A are mapped through the verified transform into view B, and become pseudo-labels for the student on B. Optionally, a feature-consistency loss is added at matched locations. Only regions with local keypoint support count, as in the label audit.

**Hypotheses:**
- real viewpoint, lighting and blur changes teach invariances that synthetic augmentation cannot
- every unlabelled overlapping image becomes training signal
- detecting a defect consistently across views favours whole-defect regions over arbitrary splits

**Novelty check:** multi-view pseudo-labelling exists for aerial detection with synchronised cameras (MAVREC, CVPR 2024), and correspondence-based self-supervision exists for dense features. I found no teacher–student detection that uses geometric registration of real overlapping views as the consistency signal, evaluated on scene-leak-free splits. A deeper literature pass is due before writing.

### Evaluation protocol
- **Leak-free scene splits** (WTBD-group, then a WTBlade group split), several seeds, scene-level bootstrap intervals (`tools/bootstrap_ci.py`).
- **Reliability as well as AP:** the conformal recall guarantee and the recall / false-alarm curve on new scenes (`tools/conformal_recall.py`).
- **Baselines:**
  - supervised only
  - standard teacher–student with synthetic augmentations only
  - real-view consistency
- **Learning curves first** (step 1 below): if accuracy is flat between 50% and 100% of the labelled scenes, extra (pseudo-)labelled views will not pay, and the method would be redirected before any heavy engineering.

### Cleanup
All results and decisions are documented in this log. The working tree now holds only what the current work uses.

**Code**
- **Removed after `b374f4c`** (restore with `git checkout b374f4c -- <path>`):
  - the merged-dataset builder and audit: `build_dataset.py`, `audit_datasets.py`, `check_overlap.py`, `class_crops.py`
  - the v3 analysis and label tools: `error_analysis.py`, `review_labels.py`, `review_page.html`, `apply_review.py`, `merge_regions.py`, `ensemble.py`
  - `setup_windows.ps1`
  - the v1 model: `bladeyolo-l.yaml`, `models/backbone.py`'s `BladeBackbone`, `models/wavelet.py`
  - `tests/test_merge_regions.py`
- **Moved into small modules:** the shared image matching to `tools/matching.py`, the box metrics to `tools/detection_metrics.py`, and the DINOv3 loader to `models/dino_weights.py`.
- **Checked:**
  - `tools/wtbd_groups.py` reproduces the committed WTBD split byte-for-byte
  - the bootstrap, granularity, conformal and leak tools reproduce their logged numbers
  - all 28 unit tests pass

**Data and artefacts deleted** (about 9 GB):
- transfer bundles and upload zips
- the derived builds of the retired v3 taxonomy (v2, v3, v3.1, v3.2, WTBs2025-clean)
- the WTBD download archive (re-downloadable from figshare) and its unpacked copy
- scratch, debug and per-source evaluation outputs
- the v3 error-analysis, review and ensemble artefacts
- the weights of retired runs

**Kept:**
- every raw source dataset (WindSurface-Defect, WTBlade-Defect, the two Beijing exports, WTBs2025, WTBD raw, WTBD-group/-official)
- the five WTBD runs
- `runs/analysis/archive/` (`results.csv`, `args.yaml` and test metrics of the seven retired runs, plus the human v3 label review `review_test_v3.json`)
- the embedding and verification caches
- the reference papers

The root `bladeyolo.log` moved into `runs/detect/BladeYOLO_WindSurface/bladeyolo2_wtbd_group/` as that run's log.

### Learning curves (step 1, ready)
`tools/scene_fractions.py` draws nested subsets of WTBD-group's 224 training scenes: the best of 1,000 random scene orders, so each subset matches the full class mix and every class is present. The lists are committed. `train.py --train-subset` trains on them with the unchanged Kaggle dataset.

| subset | scenes | images | craze / corrosion / surface_injure / thunderstrike / crack / hide_craze boxes |
|---|---|---|---|
| `tools/splits/wtbd_group_train_f25.txt` | 51 | 189 | 45 / 38 / 79 / 10 / 39 / 60 |
| `tools/splits/wtbd_group_train_f50.txt` | 107 | 373 | 77 / 90 / 168 / 33 / 75 / 122 |
| full train | 224 | 745 | 180 / 180 / 286 / 66 / 157 / 240 |

---

## 17. Phase 20: learning curves: labelled scenes still pay, and BladeYOLO-v2's advantage is a low-data advantage (15 Oct)

**Runs:** four Kaggle runs on nested scene subsets of WTBD-group (Phase 19): BladeYOLO-v2 and YOLO12-L at 25% (51 scenes, 189 images) and 50% (107 scenes, 373 images), same recipe and seed, full val and test. The full-data runs are `bladeyolo2_wtbd_group` / `yolo12l_wtbd_group`. Combined log: `runs/detect/BladeYOLO_WindSurface/learning_curve_runs.log`.
- Training times: BladeYOLO-v2 0.80 h and 1.29 h; YOLO12-L 0.49 h and 0.79 h.
- **No non-finite batches in any run:** the fp32 attention-score fix of Phase 18 holds.

Test results, Ultralytics mAP50 / mAP50-95:

| model | 25% | 50% | 100% |
|---|---|---|---|
| BladeYOLO-v2 | 0.232 / 0.082 | 0.413 / 0.183 | 0.548 / 0.245 |
| YOLO12-L | 0.106 / 0.037 | 0.244 / 0.105 | 0.438 / 0.206 |

Paired scene bootstrap (`tools/bootstrap_ci.py`, 2,000 replicates; reports `runs/analysis/bootstrap/lc_*/report.md`; COCO 101-point AP, so values differ slightly from Ultralytics):

| comparison | mAP50 difference [95% CI] | mAP50-95 difference [95% CI] |
|---|---|---|
| BladeYOLO-v2, 100% − 50% | **+0.148** [0.052, 0.259] | +0.069 [0.031, 0.142] |
| BladeYOLO-v2, 100% − 25% | +0.354 [0.241, 0.567] | +0.166 [0.119, 0.348] |
| YOLO12-L, 100% − 50% | **+0.175** [0.120, 0.338] | +0.086 [0.054, 0.207] |
| YOLO12-L, 100% − 25% | +0.301 [0.234, 0.542] | +0.143 [0.111, 0.332] |
| BladeYOLO-v2 − YOLO12-L at 25% | **+0.108** [0.041, 0.233] | +0.037 [0.011, 0.099] |
| BladeYOLO-v2 − YOLO12-L at 50% | **+0.189** [0.066, 0.359] | +0.077 [0.027, 0.172] |
| BladeYOLO-v2 at 50% − YOLO12-L at 100% | +0.013 [−0.182, 0.154] | −0.009 [−0.073, 0.051] |

Without the classes seen in ≤ 3 test scenes (craze, thunderstrike; `runs/analysis/bootstrap/robust_check_lc.py`, 1,000 replicates):

| BladeYOLO-v2 − YOLO12-L | 25% | 50% | 100% |
|---|---|---|---|
| mAP50 on corrosion, surface_injure, crack, hide_craze | **+0.075** [0.018, 0.232] | **+0.094** [0.043, 0.254] | +0.052 [−0.027, 0.134] |

**Findings:**
1. **The curves are steep, not saturating.** Doubling the labelled scenes from 50% to 100% adds +0.15 to +0.18 mAP50, with intervals well above zero, for both detectors. Data is the binding constraint on WTBD, so additional (pseudo-)labelled views can pay off. This passes the go/no-go check of Phase 19 for the real-view semi-supervised method.
2. **BladeYOLO-v2 is markedly more data-efficient.** It beats YOLO12-L at every fraction. With half the labelled scenes it is statistically indistinguishable from YOLO12-L with all of them; the interval is wide, so this is "no detectable difference", not proven equivalence.
3. **The architectural advantage is a low-data advantage.** On the well-represented classes it is clear at 25% and 50%, and not distinguishable from noise at 100%. The full-data result of Phase 15 (overall win carried by thunderstrike) fits the same pattern. A plausible explanation is that frozen DINOv3 semantics substitute for labelled examples when those are scarce. Testing it directly needs the component ablation (step 6).

**Conclusion:** the new direction is well motivated, and its base model is BladeYOLO-v2. The architecture handles label scarcity, and real-view semi-supervision should add effective data. The method's claims should be reported as **learning curves** (25 / 50 / 100% of the labelled scenes), not only at full data.

---

## 18. Phase 21: the unlabelled view pool and the real-view teacher–student implementation (15–16 Oct)

### The unlabelled view pool (`tools/view_pool.py`)
WTBD has no unlabelled images, so the pool comes from other blade datasets, used **without their labels**: WTBlade-Defect (10,660 files) and Wind Surface Defect (3,570). The Beijing exports are left out entirely: they share 184 images with WTBD (Phase 13), too high a leakage risk to filter.

1. **Exclusion.** A pool image is dropped if it matches *any* WTBD image: embedding cosine ≥ 0.93 with one of its 5 nearest WTBD images, or ≥ 25 verified ORB/RANSAC inliers. Val/test matches would leak; train matches add nothing new. **363 of 14,230 pool images were excluded** (WTBlade 133, Wind Surface 230), matching WTBD train 191, **val 89 and test 97** times. The check was needed.
2. **Pairs.** The verified same-surface transforms of the label audit (§14) between the remaining images: **77,530 directed pairs**.
3. **Real second view vs augmented copy.**
   - **First attempt: photometric similarity after alignment** (ZNCC of intensities and of gradient magnitudes). It does not work. The example sheets per band show:
     - Roboflow copies with cutout squares and colour shifts fall as low as 0.7.
     - Video frames of a turning rotor score high on intensity, because the ground dominates, while the blades have moved.
     - Near-identical consecutive frames sit at ≥ 0.98.
     - So photometric similarity cannot separate copies from new views in augmented Roboflow data.
   - **Adopted: the source photograph, read from the file name** (`photo_id`). Roboflow copies share `<source>_jpg.rf.*`, Wind Surface twins differ in `_0`/`_1`, and tiles encode their parent DJI photo. A **real second view** is a verified same-surface pair from **different source photos** (another frame or shot). Copies and tiles of one photo are synthetic views. `pairs.csv` carries `same_photo`, the transform matrix, and the ZNCC values for description.
   - **Result:** **49,886 real-view pairs** (WTBlade 45,777, Wind Surface 4,109) from 2,791 and 315 source photos. Copies make up the other 27,644 pairs.
   - **Visual check** (`runs/analysis/novel/view_pool/examples_real_pairs.jpg`, 16 random pairs): these are genuinely different photographs.
     - video frames at different moments (rotor in different positions)
     - separate shots of the same blade tip
     - frames from different points of a DTU NordTank flight
     - some WTBlade sources were uploaded already augmented (cutout squares), but they are still different frames
4. **Balanced sampling.** Video sources have many neighbouring frames, so uniform pair sampling would be dominated by a few videos. Pairs are drawn so that every view-A image is equally likely (weights 1/31 to 1; 7,090 distinct A images). The synthetic mode uses the same 7,384 images, one item each.
5. **Export:** `python tools/view_pool.py --export view-pool` gives 358 MB with paths relative to `pairs.csv`, to upload as a Kaggle dataset. It contains WTBlade imagery including DTU NordTank frames (CC BY-NC): use it for research only.

### Method implementation (`models/view_consistency.py`, `BladeTrainer`, `train.py --ssl`)
- **Teacher:** the trainer's EMA model, which exists and is updated identically on every DDP rank.
- **Each batch, from epoch `--ssl-start` (default 30, a supervised burn-in):**
  1. `--ssl-ratio` × the per-GPU batch of unlabelled pairs is drawn (default 0.25: 2 pairs next to 8 labelled images).
  2. The teacher detects on view A, and detections with confidence ≥ `--ssl-tau` (0.5) are moved into view B.
  3. B is appended to the labelled batch with those pseudo-labels, and the unchanged loss trains on both.
- **Safeguards:**
  - **B is masked to the area A covers,** so defects outside A's view are not trained as background.
  - **A moved box is kept only if** ≥ 50% of it lies in that area, its sides are ≥ 4 px, and **the warped A and B agree inside the box** (gradient ZNCC ≥ 0.5). This is per-box local verification: the global transform fits the static scene, but labels on a blade that turned between frames must not move (the same artefact the label audit found).
- **Exact pixel maps:** A's letterbox → A's pixels → 480-px matching frame → flip → transform → B's pixels → B's letterbox, composed as 3×3 matrices.
- **Synthetic baseline (`--ssl synthetic`):** **the same images** as the real pairs, each paired with a random augmentation of itself (flip, quarter turn, scale 0.75–1.25, shift, colour, blur) with its exact matrix. The two modes differ only in where the second view comes from.
- **Statistics:** per epoch, pairs used, confident teacher boxes, transferred pseudo-labels and their classes (`<run>/ssl_stats.csv`).
- **`--export DIR`** writes the real pairs with relative paths plus only their images, as a self-contained Kaggle dataset.

### Tests
`tests/test_view_pool.py`: an augmented copy aligns at ZNCC 0.987 (gradients 0.979), a different image at ≈ 0.

`tests/test_view_consistency.py`:
- the composed matrix equals the audit's verified projection
- synthetic augmentations move a marker exactly where their matrix says (20 random draws)
- the letterbox chain of the dataset is exact
- the transfer filter drops low-confidence, outside and degenerate boxes
- the local check drops a box whose content changed between views

CPU smoke runs through `BladeTrainer`, fine-tuned from a trained checkpoint so the teacher detects:
- **synthetic:** 10 pairs → 14 teacher boxes → 14 pseudo-labels
- **real:** 10 pairs → 10 → 10
- finite losses in both

From a random initialisation, no pseudo-labels appear even at τ = 0.01, as expected. That is why `--ssl-start` gives a supervised burn-in.

---

## 19. Phase 22: first semi-supervised runs at 25%: no effect, and why (16–17 Oct)

**Runs:** BladeYOLO-v2 on 25% of WTBD-group's labelled scenes (189 images), with the view pool of Phase 21: `--ssl synthetic` and `--ssl real`. Defaults: ratio 0.25 (2 unlabelled pairs next to 8 labelled images per GPU), τ = 0.5, pseudo-labels from epoch 31. Same recipe and seed as the supervised `bladeyolo2_wtbd_group_f25`.

| run | test mAP50 / 50-95 | with test-time aug. | val best | pseudo-labels over training (GPU 0) |
|---|---|---|---|---|
| supervised | 0.232 / 0.082 | 0.269 / 0.105 | 0.444 / 0.226 | – |
| synthetic views | 0.221 / 0.071 | 0.233 / 0.089 | 0.427 / 0.203 | 2,880 pairs → 751 teacher boxes → 678 pseudo-labels (90% kept) |
| real views | 0.213 / 0.085 | 0.268 / 0.107 | 0.450 / 0.210 | 2,880 pairs → 124 teacher boxes → **66** pseudo-labels (53% kept) |

Paired scene bootstrap, supervised − semi-supervised (`runs/analysis/bootstrap/ssl_f25/report.md`):
- **synthetic:** +0.028 mAP50 [−0.188, 0.069]
- **real:** +0.018 [−0.068, 0.057]
- mAP50-95: +0.010 and −0.000

**No detectable effect either way.**

**Why the hypothesis was not actually tested:**
1. **The unlabelled signal was tiny.** With 189 labelled images an epoch is about 12 batches, so about 2,900 pairs per GPU over the whole run, and only a few hundred pseudo-labels, against 189 labelled images seen 150 times.
2. **The real-view teacher collapsed.** Confident teacher boxes per epoch on the pool fell from 2.4 (first third) to 0.3–0.4 (last two thirds). Cause, a design flaw: an unlabelled image whose pseudo-labels were all removed (low confidence, or the local and area checks, which real pairs fail more often: 53% kept vs 90%) still joined the batch as an image *without defects*. The student learned that pool images contain no damage, which made the teacher less confident, a feedback loop.
3. **Domain shift.** The pool comes from other wind farms, and the WTBD-trained teacher is rarely confident there at τ = 0.5. The synthetic run had the same domain gap but kept 90% of its few boxes.

**Fix implemented** (`models/view_consistency.py`, `BladeTrainer`):
- an unlabelled image with no surviving pseudo-label is no longer used
- the regions of confident teacher boxes that fail a check are greyed out in B, so they are ignored, not trained as background
- `ssl_stats.csv` now also records images used and ignored boxes

Tests updated (`tests/test_view_consistency.py`: dropped boxes are returned for ignoring).

**Still open:**
- **Throughput:** more unlabelled pairs per step does not fit in T4 memory (8 + 2 images ≈ 11 GB of 15).
- **Domain:** the standard protocol draws unlabelled data from the same dataset (the 75% of WTBD training scenes outside the labelled subset). But WTBD's numbered file names carry no source-photo provenance, so its real second views cannot be separated from its augmented copies the way WTBlade's can (§18).

**Next round (decided 17 Oct, one Kaggle session, 3 runs at 25%):**
- **A:** rerun `--ssl synthetic` and `--ssl real` on the view pool with the fix.
- **B:** the in-domain protocol. `--ssl synthetic --ssl-unlabelled tools/splits/wtbd_group_train_f25_rest.txt`, i.e. the 556 WTBD training images outside the 25% subset, used without labels; val and test untouched. This tests whether teacher–student training helps on WTBD at all. Real views cannot be identified inside WTBD, so B has no real-view counterpart.
- **Implementation:** `train.py --ssl-unlabelled` resolves the names against `--data` and refuses overlap with `--train-subset` and use with `--ssl real`; `PairDataset` accepts an image list in synthetic mode.
- **Tests:** list mode and refusals; a 1-epoch CPU run through the full command line; trained-teacher smoke runs in both modes after the fix (synthetic 11 teacher boxes → 7 pseudo-labels, 4 ignored; real 8 → 8; 7 of 10 images used each).

---

## 20. Phase 23: second semi-supervised round: collapse fixed, small consistent gains, no real-view advantage (17 Oct)

**Runs:** BladeYOLO-v2, 25% of WTBD-group's labelled scenes, with the fix of Phase 22 (images without pseudo-labels skipped, regions of filtered teacher boxes ignored). Three Kaggle runs:
- `…_sslsyn2`: synthetic views of the pool images
- `…_sslreal2`: real views from the pool
- `…_sslin`: in-domain, synthetic views of the 556 WTBD training images outside the labelled subset

No non-finite batches; the setup prints were confirmed by the user.

| run | test mAP50 / 50-95 | with test-time aug. | val best | pool images used | pseudo-labels | ignored | teacher boxes per epoch (thirds) |
|---|---|---|---|---|---|---|---|
| supervised | 0.232 / 0.082 | 0.269 / 0.105 | 0.444 / 0.226 | – | – | – | – |
| synthetic, pool | 0.224 / 0.078 | 0.259 / 0.106 | 0.448 / 0.220 | 1,217 | 2,748 | 204 | 17.8 / 24.8 / 33.5 |
| real, pool | 0.192 / 0.077 | 0.248 / 0.101 | 0.382 / 0.184 | 561 | 961 | 582 | 16.9 / 12.5 / 10.9 |
| in-domain | 0.209 / 0.080 | 0.245 / 0.101 | **0.457** / 0.227 | 987 | 1,294 | 38 | 13.0 / 8.2 / 12.3 |

(Statistics are GPU 0 only.)

Pseudo-labels per class:

| run | craze | corrosion | surface_injure | thunderstrike | crack | hide_craze |
|---|---|---|---|---|---|---|
| synthetic, pool | 19 | 118 | **2,495 (91%)** | 3 | 8 | 105 |
| real, pool | 288 | 340 | 243 | 0 | 77 | 13 |
| in-domain | 377 | 89 | 476 | 49 | 152 | 151 |

**Paired scene bootstrap** (`runs/analysis/bootstrap/ssl_f25_r2/report.md`), semi-supervised − supervised mAP50, all six classes:

| run | mAP50 difference [95% CI] |
|---|---|
| synthetic | −0.019 [−0.105, 0.197] |
| real | −0.021 [−0.073, 0.171] |
| in-domain | +0.018 [−0.064, 0.144] |

Per class (COCO evaluator), the semi-supervised runs gain on surface_injure (0.29 → 0.35–0.43), corrosion and crack, and lose on **thunderstrike** (0.32 → 0.07–0.26; 3 test scenes, almost never pseudo-labelled).

On the four classes with ≥ 8 test scenes (corrosion, surface_injure, crack, hide_craze; `runs/analysis/bootstrap/robust_check_ssl.py`, 1,000 replicates):

| run | mAP50 difference [95% CI] | mAP50-95 difference [95% CI] |
|---|---|---|
| synthetic | +0.030 [−0.077, 0.105] | +0.012 [−0.027, 0.034] |
| real | +0.029 [−0.020, 0.066] | +0.012 [−0.011, 0.025] |
| in-domain | **+0.042** [−0.023, 0.117] | **+0.013** [−0.005, 0.036] (P(≤0) = 0.046) |

**Findings:**
1. **The background-collapse fix works.** Teacher output rose from 66–751 to 961–3,046 boxes and no longer collapses. Real-view teacher boxes decline only mildly (16.9 → 10.9 per epoch).
2. **Teacher–student training gives a small, consistent, not yet significant gain on well-represented classes:** +0.03 to +0.04 mAP50 for all three variants, largest in-domain. On the overall mean it is offset by the rare class thunderstrike.
3. **Pseudo-labels are class-biased, depending on the domain.** On images from other wind farms the teacher labels mostly surface_injure (91% in the synthetic run), and the bias grows as the student learns it. In-domain pseudo-labels are the most balanced. **Rare classes (thunderstrike) are almost never pseudo-labelled,** so semi-supervision dilutes them.
4. **No evidence for the real-view hypothesis.** Real views perform like synthetic views on the robust classes (+0.029 vs +0.030) and worse on the overall mean and val. They also lose many more boxes to the local and area checks (582 ignored vs 204). On the evidence so far, the second view's source does not matter. **Using unlabelled data from the target distribution (in-domain) matters more.**

**Conclusions:**
- **The real-view direction is not supported** by two rounds of experiments. It should not be pursued further without a new reason, e.g. genuinely multi-view data from the target farm.
- **The in-domain teacher–student result is the most promising:** small positive effect, balanced pseudo-labels. Two obvious improvements: class-balanced pseudo-labelling (per-class thresholds or quotas, so rare classes are not drowned) and seeds to resolve a +0.04 effect from noise.

**Next round (decided 17 Oct): in-domain teacher–student with class-balanced pseudo-labels, with seeds.**

*Class-balanced thresholds* (`--ssl-balance`, `view_consistency.class_thresholds`):
- FlexMatch's mapping (Zhang et al., 2021), applied to detection.
- Per class, the confident teacher detections (≥ τ) of the previous epoch are counted; β_c = count_c / max count; τ_c = τ · β_c / (2 − β_c), floored at `--ssl-tau-min` (0.25).
- The class the teacher labels most keeps τ; classes it rarely labels (thunderstrike) get lower thresholds.
- The thresholds are logged per epoch in `ssl_stats.csv`.

*Seeds:* `--seed` now also changes the unlabelled sampling sequence. Before, it was fixed per GPU, so different seeds would have replayed the same unlabelled data.

*Tests:*
- the threshold mapping by hand, and per-class filtering in `transfer`
- a 2-epoch trained-teacher smoke run: all thresholds 0.25 in epoch 1; in epoch 2 the most-labelled class keeps 0.25 and the others drop to the floor; pseudo-labels 7 → 15, and craze appears

*Plan:* 5 runs at 25%, about 6 h: supervised seeds 1 and 2 (seed 0 exists), and balanced in-domain seeds 0, 1 and 2.

*Stopping rule:* if the gain on the well-represented classes is still within noise with three seeds per arm, semi-supervision is dropped too.

---

## 21. Phase 24: seeds and class-balanced in-domain pseudo-labels: semi-supervision is dropped (17–18 Oct)

**Runs** (Kaggle, commit `8592077`; no non-finite batches): supervised seeds 1 and 2 at 25% (`…_f25_s1`, `…_f25_s2`; seed 0 is `…_f25`), and in-domain teacher–student with class-balanced thresholds (`--ssl-balance`), seeds 0–2 (`…_f25_sslinb_s0/1/2`). Outputs pulled with the new `tools/pull_kaggle_runs.py` (below).

| arm (3 seeds) | test mAP50 | mAP50-95 | precision | recall | TTA mAP50 / 50-95 |
|---|---|---|---|---|---|
| supervised | 0.213 ± 0.026 (0.232, 0.224, 0.184) | 0.079 ± 0.010 | 0.254 | 0.248 | 0.247 / 0.102 |
| balanced in-domain | 0.163 ± 0.020 (0.143, 0.163, 0.183) | 0.066 ± 0.006 | 0.139 | 0.281 | 0.192 / 0.083 |

(COCO evaluator, mean ± s.d. over seeds.) **Seed noise alone is ±0.026 mAP50 at 25%**: supervised seeds span 0.184–0.232, as wide as every semi-supervised effect measured in Phases 22–23.

**Paired scene bootstrap, three seeds per arm** (`runs/analysis/bootstrap/robust_check_seeds.py`, log `seeds_f25.log`; each replicate scores every run on the same scene draw, and an arm is the mean of its seeds; bootstrap evaluator, so absolute values differ slightly from the COCO numbers):

| comparison | all 6 classes, mAP50 [95% CI] | 4 robust classes, mAP50 [95% CI] |
|---|---|---|
| balanced − supervised | **−0.042 [−0.121, −0.003]**, P(≥0) = 0.020 | **−0.028 [−0.088, −0.006]**, P(≥0) = 0.011 |
| unbalanced in-domain (`…_sslin`, 1 seed) − supervised | +0.023 [−0.063, +0.089] | +0.027 [−0.012, +0.096], P(≥0) = 0.91 |

Per class (mean over seeds, test mAP50), balanced vs supervised: craze 0.098 vs 0.038, surface_injure 0.404 vs 0.349, but **thunderstrike 0.042 vs 0.304**, hide_craze 0.293 vs 0.383, corrosion 0.100 vs 0.166.

**What the balancing did** (`ssl_stats.csv`):
- With per-epoch counts of only tens of confident boxes, β_c is tiny for every class but the most frequent one, so **five of six classes sat at the 0.25 floor almost every epoch**; the "balanced" scheme was in effect a global threshold cut from 0.5 to 0.25.
- Pseudo-labels per run: 5,957 / 28,742 / 4,372 (vs 1,294 unbalanced). **Seed 1 ran away** (epochs 33–62): up to ~2,300 pseudo-boxes per epoch from 24 images, 55% hide_craze and 31% crack, a confirmation-bias loop; seed 0 had 6 such epochs. Both never recovered on val (best 0.344 and 0.352 vs 0.444 supervised); seed 2 had none and matched supervised on val (0.445).
- Thunderstrike received 8× more pseudo-labels (384 in seed 0 vs 49), yet its test AP collapsed.

**Pseudo-label precision measured directly** (`runs/analysis/bootstrap/pseudo_precision.py`, log `pseudo_precision.log`). The 556 "unlabelled" images have hidden ground truth; a pseudo-box is correct if it overlaps a same-class box at IoU ≥ 0.5:

| labeller | τ = 0.5: boxes, correct | τ = 0.25: boxes, correct |
|---|---|---|
| supervised 25% model (a strong stand-in for the teacher) | 236, 66% | 699, 40% |
| balanced run, seed 0 (final) | 318, 46% | 1,169, 23% (crack 10%) |
| balanced run, seed 2 (final) | 504, 43% | 1,175, 27% |

Lowering τ triples the pseudo-labels but most of the added boxes are wrong, and training on them makes the model's own labels worse (crack 10% correct in seed 0). Even at τ = 0.5 only two thirds are correct, partly because of box extent (below).

**Conclusions:**
1. **Class-balanced thresholds are harmful here** (significant on both class sets): with a weak teacher, lower thresholds admit mostly wrong boxes, and the loop amplifies them. FlexMatch's mapping assumes a teacher that is already reasonably accurate.
2. **By the stopping rule (§20), semi-supervision is dropped.** The unbalanced in-domain variant remains at +0.027 on the robust classes with one seed, inside the seed noise; it is noted but not pursued ahead of the directions below.
3. **Methodological lesson:** at 25%, any training-time change needs ≥ 3 seeds per arm; single-run comparisons of Phases 22–23 could not have resolved effects below ~0.05.

### Where the errors are (diagnostics on cached test predictions)

- **Not resolution.** Defect boxes are large: median short side at 640 px is 62–148 px by class, none below 8 px, and the images are only 1024 px natively. Tiling or higher input resolution is not expected to help.
- **Not classification** (`runs/analysis/bootstrap/agnostic_ap.py`, log `agnostic_ap.log`). Class-agnostic AP50 ≈ class-aware mAP50: BladeYOLO-v2 100% 0.563 vs 0.561; 25% 0.210 vs 0.207 and 0.259 vs 0.196 (two seeds); YOLO12-L 100% 0.485 vs 0.400. The detector rarely confuses defect types; what fails is finding and boxing them.
- **Mostly box extent.** Share of the 235 test defects matched by any prediction with conf ≥ 0.1, class ignored:

| model | hit at IoU 0.5 | region found (IoU ≥ 0.1 or half-covered) | missed entirely |
|---|---|---|---|
| BladeYOLO-v2, 100% | 66% | 94% | 6% |
| YOLO12-L, 100% | 59% | 87% | 13% |
| BladeYOLO-v2, 25% (2 seeds) | 39–41% | 75–77% | 23–25% |

At 25%, ~36% of defects are located but fail IoU 0.5, more than are missed outright; the gap shrinks slowly with data (28 points at 100%). This matches the label audit (annotators agree on only ~87% of boxes across copies, §14). The "region found" criterion is generous (a large box half-covering a defect counts) and will be tightened.

**Hypothesis:** in low-data defect detection, recognising *that* something is a defect is learned quickly; the expensive part is the annotators' extent convention (where a defect ends, one box or several).

### Next round (decided 18 Oct)

1. **Training-free SAM 2.1 box refinement** (`tools/sam_refine.py`, test `tests/test_sam_refine.py`): each detection (conf ≥ 0.05, at most 30 per image) prompts SAM 2.1-tiny; the mask's bounding box replaces or blends with the detection. Prompt expansion (1.0 / 1.15 / 1.3), an agreement gate (IoU(original, SAM box) ≥ 0 / 0.3 / 0.5 / 0.7) and the blend (0.5 / 1.0) are chosen on val (one setting for all runs, by mean val mAP50-95), then applied once to test with a paired scene bootstrap. Runs: the three supervised 25% seeds and the 100% model. Being post-hoc, it is evaluated on existing seeds at no training cost. Smoke test: a loose detection [425, 143, 533, 249] was snapped to [459, 147, 522, 243] against ground truth [454, 140, 528, 248]. **Result: negative (below).**
2. **DINOv3 ablation** (done, §22: DINOv3 carries the whole low-data advantage) (`bladeyolo2-l-nodino.yaml`, test in `tests/test_model.py`): BladeYOLO-v2 without DINOv3 (layer 0 a plain Conv, no DinoFuse), otherwise identical; it starts exactly as YOLOv12-L (1,239 pretrained tensors). Tests whether the 2.2× low-data advantage over YOLO12-L at 25% (0.232 vs 0.106) comes from the frozen foundation features. Two seeds at 25%, against the three supervised seeds.

### Result: SAM 2.1 box refinement does not help (18 Oct)

`runs/analysis/sam_refine/report.md`, `val_grid.csv`. Best val setting (all four runs): prompt as is (expand 1.0), no agreement gate, blend 0.5: val mean mAP50 0.463 → 0.469, mAP50-95 0.225 → 0.244. On test (paired scene bootstrap):

| run | original mAP50 / 50-95 | refined | mAP50 difference [95% CI] | mAP50-95 difference [95% CI] |
|---|---|---|---|---|
| 25%, seed 0 | 0.207 / 0.078 | 0.202 / 0.084 | −0.005 [−0.015, +0.030] | +0.005 [−0.001, +0.015] |
| 25%, seed 1 | 0.196 / 0.078 | 0.201 / 0.082 | +0.006 [−0.007, +0.019] | +0.004 [−0.018, +0.010] |
| 25%, seed 2 | 0.203 / 0.071 | 0.178 / 0.075 | −0.025 [−0.157, +0.011] | +0.003 [−0.010, +0.014] |
| 100% | 0.561 / 0.244 | 0.539 / 0.242 | −0.022 [−0.108, −0.004] | −0.002 [−0.026, +0.012] |

The val gain does not transfer; at most a +0.004 mAP50-95 tightening at 25%, within noise, and a small mAP50 loss at 100%. **Training-free SAM refinement is negative.**

**Why: what the near-misses are** (`runs/analysis/bootstrap/near_miss.py`, log `near_miss.log`; stricter than the first diagnostic; class ignored, conf ≥ 0.1; for each ground-truth box without an IoU ≥ 0.5 prediction, the best-overlapping prediction decides: *part* = prediction ≥ 80% inside the ground truth and < half its area; *merged* = ground truth ≥ 80% inside a prediction > 2× its area; *shifted* = other overlap with IoU ≥ 0.1; *missed* = none):

| model | hit | part | merged | shifted | missed | ground truths covered by ≥ 2 parts |
|---|---|---|---|---|---|---|
| 25%, seeds 0 / 1 / 2 | 39 / 41 / 45% | 9 / 11 / 10% | 8 / 7 / 9% | 17 / 17 / 17% | 27 / 24 / 19% | 10 / 12 / 17 |
| 100% | 66% | 2% | 12% | 12% | 8% | 0 |

Per class (25% seed 0 → 100%):

| class | hit | missed | main near-miss |
|---|---|---|---|
| surface_injure | 36% → 87% | **49% → 5%** | – |
| corrosion | 53% → 82% | **32% → 0%** | – |
| thunderstrike | 62% → 100% | 0% → 0% | merged 23% → 0% |
| hide_craze | 63% → 65% | 20% → 18% | part 12% → 0%, merged 4% → 12% |
| crack | 21% → 55% | 6% → 3% | **shifted 45% → 33%**, part 21% → 3% |
| craze | **5% → 21%** | 23% → 13% | **merged 23% → 46%**, shifted 41% → 18% |

**Findings (revising the first diagnostic):**
1. With the stricter criterion, at 25% near-misses (34–36%) and outright misses (19–27%) are of similar size; "mostly extent" was overstated.
2. **The near-misses are granularity and geometry, not loose boxes**: about half are part/merged (how many boxes a defect is), the rest misaligned, concentrated in the thin, elongated classes (crack, craze). SAM tightens a box around one object; it cannot decide whether a craze field is one box or five, which is why it does not help.
3. **Two kinds of class:** *recognition-limited* (surface_injure, corrosion: misses vanish with data, so more labels or better features pay) and *structure-limited* (craze, crack: still 21% / 55% hits at full data, dominated by merged and shifted boxes; more data of the same annotation style does not fix them).

**Conclusions:** the recognition-limited classes are where data efficiency (foundation features, the DINOv3 ablation) can pay. For craze and crack the limit is the axis-aligned box itself: the annotators' grouping is inconsistent (the label audit) and IoU on thin diagonal boxes is unstable, so their scores reflect the annotation format as much as the detector.

### Next after the DINOv3 ablation: class-selective split tolerance (decided 18 Oct)

Phase 18's blanket tolerance helped corrosion (mAP50-95 0.13 → 0.26) and hurt surface_injure. The near-miss diagnostic, **rerun on val** so that test plays no part in the choice (`near_miss.py val`, log `near_miss_val.log`), shows where "merged" errors (a prediction spanning several touching labels, which is the case tolerance addresses) occur:

| class (val) | merged, 25% seeds / 100% | part, 25% / 100% | label-audit agreement (§14) |
|---|---|---|---|
| corrosion | 18–23% / 31% | 3–5% / 3% | 69.0% |
| hide_craze | 0–10% / 8% | 2% / 4% | 84.4% |
| crack | 0–3% / 0% | 26–50% / 12% | 95.1% |
| surface_injure | 0–2% / 0% | 0% / 0% | 82.7% |

**Pre-registered selection rule:** audit agreement < 85% and a non-trivial merged rate on val → **corrosion and hide_craze**. surface_injure is excluded by the same rule (no merged errors; Phase 18 hurt it). Crack's errors are *part* errors (a prediction covering a piece of one label), which region targets do not address, so it is excluded too.

**Implementation:** `train.py --tolerant-classes corrosion,hide_craze` (names from `--data`, implies `--tolerant`; unknown names stop the run) sets `BLADEYOLO_TOLERANT_CLASSES`; `models/tolerant.region_boxes` then builds regions only for those classes, and every other label keeps itself as its only target, as in standard training. Test: `tests/test_tolerant.py::test_class_selective_regions`.

**Runs** (100% of WTBD-group, as in Phase 18): class-selective tolerance with seeds 0 and 1, and a second supervised seed at 100% so seed noise can be put on the comparison. **Primary score, fixed in advance:** corrosion mAP50-95. Secondary: overall mAP50-95, and surface_injure (should no longer be hurt).

### Tooling

- `tools/pull_kaggle_runs.py`: downloads only the `runs/` files of the latest version of the Kaggle notebook (`beegee11/bladeyolo` by default; the output also holds the repo clone and dataset copy), stages them in `.kaggle_pull/` (gitignored, removed afterwards) and merges file by file: new files copied, identical skipped, differing local files kept and reported (`--overwrite` replaces). The notebook log is kept as `runs/kaggle_logs/<kernel>_<UTC time>.log`. `--dry-run`, `--kernel owner/name[/version]`. The Kaggle CLI is in `.venv` (`requirements.txt`).
- Engineering note: two CPU inference jobs at once exceed this 6 GB machine (OOM kill); analyses run sequentially.

---

## 22. Phase 25: the DINOv3 ablation: frozen foundation features carry the entire low-data advantage (18 Oct)

**Runs** (Kaggle, commit `389155e`; no non-finite batches): `bladeyolo2nodino_wtbd_group_f25_s0/s1`, BladeYOLO-v2 without DINOv3 (`bladeyolo2-l-nodino.yaml`: layer 0 a plain Conv, no DinoFuse; same YOLOv12-L initialisation, deformable and selective-scan refinements, WIoU loss and recipe), 25% of WTBD-group's labelled scenes, seeds 0 and 1.

| at 25% | test mAP50 (COCO evaluator) | mAP50-95 | val best (mAP50 / 50-95) |
|---|---|---|---|
| BladeYOLO-v2, 3 seeds | 0.213 (0.232, 0.224, 0.184) | 0.079 | 0.411–0.444 / 0.199–0.226 |
| **without DINOv3, 2 seeds** | **0.097** (0.114, 0.081) | **0.038** | 0.229, 0.186 / 0.126, 0.105 |
| YOLO12-L, 1 run | 0.106 | 0.037 | 0.233 / 0.139 |

**Paired scene bootstrap** (`runs/analysis/bootstrap/robust_check_nodino.py`, log `nodino_f25.log`; bootstrap evaluator, an arm is the mean of its seeds):

| comparison | all 6 classes: mAP50 [95% CI] | mAP50-95 [95% CI] | 4 robust classes: mAP50 [95% CI] |
|---|---|---|---|
| without DINOv3 − BladeYOLO-v2 | **−0.108 [−0.240, −0.077]** | **−0.039 [−0.101, −0.025]** | **−0.099 [−0.216, −0.059]** |
| YOLO12-L − BladeYOLO-v2 | −0.103 [−0.239, −0.070] | −0.035 [−0.098, −0.016] | −0.091 [−0.226, −0.046] |

Point values: BladeYOLO-v2 0.202, without DINOv3 0.094, YOLO12-L 0.099. Per class (AP50, bootstrap evaluator, BladeYOLO-v2 / without DINOv3 / YOLO12-L): craze 0.035 / 0.017 / 0.014, corrosion 0.175 / 0.089 / 0.106, surface_injure 0.348 / 0.304 / 0.285, thunderstrike 0.234 / 0.001 / 0.000, crack 0.038 / 0.003 / 0.011, hide_craze 0.382 / 0.152 / 0.178.

**Findings:**
1. **Removing DINOv3 removes the whole low-data advantage.** Without it, BladeYOLO-v2 falls to YOLO12-L's level (0.094 vs 0.099 mAP50), on the aggregate and class by class. The deformable and selective-scan refinements and the WIoU loss contribute nothing measurable at 25%.
2. **Significant and not a thunderstrike artefact:** the four well-represented classes alone lose −0.099 [−0.216, −0.059]. Earlier (§12) the 100% advantage over YOLO12-L rested mainly on thunderstrike; at 25% the DINOv3 effect is broad.
3. **Largest on the rare and distinctive classes** (thunderstrike 0.23 → 0.00, hide_craze 0.38 → 0.15, corrosion halved), smallest on the most frequent class (surface_injure 0.35 → 0.30). In terms of §21's two kinds of class: frozen foundation features are what makes the *recognition-limited* classes learnable from few labels.

**Conclusion:** BladeYOLO-v2's data efficiency comes from injecting frozen DINOv3 features, not from the architecture around it. This is the first strong, positive, causal result of the method work (Phases 19–25), and it identifies the lever: **the foundation-model prior.** It also settles the "architecture vs pretrained features" question left open in §12.

**Next (decided 18 Oct):**
1. **Complete the curve:** BladeYOLO-v2 without DINOv3 at 50% and 100%, to measure how the DINOv3 benefit changes with labelled data (25%: +0.11 mAP50, 2.2×; with YOLO12-L as the stand-in, 100%: +0.11, 1.25×).
2. **Scale the prior:** a larger frozen DINOv3 (ViT-B/16, ViT-L/16) in place of ViT-S/16 at 25%. Hypothesis: the low-data gain grows with the foundation model. Needs the model size to be configurable in `models/dino_weights.py` / `DinoConv` (`dino_dim`, block indices) and the gated weights on Kaggle.
3. Class-selective split tolerance at 100%: done (§23).

**Implementation of item 2 (18 Oct):** `train.py --dino s|b|l` (`BLADEYOLO_DINO`; ViT-S/16 default, so earlier runs and checkpoints are unchanged).
- Weights: the official Hugging Face releases (`facebook/dinov3-vit{s,b,l}16-pretrain-lvd1689m`, gated; downloaded to `external/dinov3/vit?16/`). They are in the transformers layout; `models/dino_weights.hf_to_dinov3` converts them on load (q/k/v fused into qkv with a zero k bias; layer scales, MLP and norms renamed), and a `config.json` next to the file is checked so a wrong size is refused.
- Two buffers are not stored by Hugging Face and are set as in Meta's release (`hf_buffers`): **the RoPE periods rounded to bfloat16** (Meta's checkpoint stores them in bf16, so the model was trained with rounded periods; unrounded periods change the features) and the qkv bias mask (all qkv biases are zero in this release, so it only matters for safety).
- **Verified:** converted ViT-S/16 has all 175 weight tensors identical to Meta's original `dinov3_vits16_pretrain_lvd1689m-08c60483.pth` (SHA-256 prefix `08c60483`, matching its filename) and gives bit-identical features. ViT-B/16 (12 blocks, width 768) and ViT-L/16 (24 blocks, width 1024) build and fuse; ViT-L uses blocks 8/16/24 (the same relative depths as ViT-S/B's 4/8/12). Tests: `tests/test_dino_sizes.py`.
- Weight search order (from 19 Oct, for all three sizes): `DINOV3_WEIGHTS`, the Kaggle models `beegee11/vit{s,b,l}16` (the official Hugging Face files, `/kaggle/input/models/beegee11/vit?16/<framework>/default/<version>/model.safetensors`, latest version first), then `external/dinov3/vit?16/model.safetensors`. ViT-S now also loads from the Hugging Face release instead of Meta's .pth; the two are bit-identical (test above), so earlier ViT-S runs remain comparable.

---

## 23. Phase 26: class-selective split tolerance at 100%: the targeted corrosion gain replicates, the aggregate does not move (18–19 Oct)

**Runs** (Kaggle, commit `ac09b2c`; no non-finite batches): `bladeyolo2_wtbd_group_tolsel_s0/s1` (`--tolerant-classes corrosion,hide_craze`, classes pre-registered in §21) and a second supervised seed at 100%, `bladeyolo2_wtbd_group_s1` (interrupted by the user at epoch 77 and resumed in the same session with `--resume`; 150 epochs).

| run (100%) | test mAP50 / 50-95 | TTA | val best | corrosion 50-95 | hide_craze 50-95 | thunderstrike 50-95 |
|---|---|---|---|---|---|---|
| supervised s0 | 0.548 / 0.245 | 0.574 / 0.279 | 0.650 / 0.368 | 0.134 | 0.275 | 0.336 |
| supervised s1 | 0.554 / 0.245 | 0.559 / 0.269 | 0.630 / 0.370 | 0.111 | 0.281 | 0.345 |
| class-selective s0 | 0.567 / 0.245 | 0.568 / 0.258 | 0.656 / 0.381 | **0.268** | 0.316 | 0.232 |
| class-selective s1 | 0.561 / 0.245 | 0.555 / 0.264 | 0.704 / 0.380 | **0.232** | 0.321 | 0.233 |
| blanket tolerance (Phase 18) | 0.523 / 0.249 | 0.552 / 0.282 | 0.609 / 0.352 | 0.256 | 0.270 | 0.322 |

(COCO evaluator.) Seed noise at 100% is small on the aggregate: the two supervised seeds agree to 0.006 mAP50 and exactly on mAP50-95.

**Paired scene bootstrap** (`runs/analysis/bootstrap/robust_check_tolsel.py`, log `tolsel.log`; mean of 2 seeds per arm, bootstrap evaluator, 1,000 replicates), class-selective − supervised:

| class | AP50-95 | difference [95% CI] | P(>0) | AP50 difference [95% CI] |
|---|---|---|---|---|
| **corrosion (primary)** | 0.139 → 0.232 | **+0.093 [−0.003, +0.246]** | 0.97 | −0.009 [−0.204, +0.222] |
| hide_craze | 0.270 → 0.306 | +0.035 [−0.041, +0.073] | 0.82 | +0.004 [−0.107, +0.061] |
| surface_injure | 0.446 → 0.404 | −0.042 [−0.084, +0.079] | 0.34 | −0.041 [−0.088, +0.085] |
| thunderstrike | 0.349 → 0.227 | −0.122 [−0.157, −0.104] | 0.00 | −0.098 [−0.119, 0.000] |
| craze / crack | 0.037 → 0.040 / 0.208 → 0.214 | +0.003 / +0.007 | – | +0.008 / +0.001 |
| mean (6 classes) | 0.242 → 0.237 | −0.004 [−0.026, +0.046] | 0.54 | −0.022 [−0.057, +0.042] |

**Findings:**
1. **The pre-registered effect replicates in direction and size:** corrosion mAP50-95 roughly doubles in both seeds (0.13 / 0.11 → 0.27 / 0.23), as in Phase 18 (0.26). Both tolerant seeds exceed both supervised seeds, but with 2 seeds and 58 test scenes the interval just touches zero (+0.093 [−0.003, +0.246], P(>0) = 0.97): **suggestive, not significant at 95%.**
2. **The gain is at strict IoU only:** corrosion AP50 is unchanged (0.54 → 0.53). Region targets do not make the detector find more corrosion; they make its boxes agree with the annotators' extent, which is what mAP50-95 rewards. This fits §21: corrosion's errors were "merged" boxes, a granularity disagreement.
3. **Thunderstrike drops** (0.35 → 0.23 in both seeds; AP50 1.00 → 0.90) although its targets are unchanged. Its interval excludes zero, but it rests on 13 boxes in 3 test scenes, where a scene bootstrap cannot represent the uncertainty (three clusters give almost no resampling variety). The blanket run (Phase 18), which *did* apply tolerance to thunderstrike, kept 0.32. Treated as unexplained and under-sampled, not as an effect of tolerance.
4. **The aggregate does not move** (mean AP50-95 −0.004 [−0.026, +0.046]): the corrosion and hide_craze gains are offset by thunderstrike and surface_injure.

**Conclusions:**
- Class-selective tolerance is a **targeted, annotation-aware fix**: it raises strict-IoU scores on the class whose labels are inconsistently split, in the direction and size the label audit predicted, chosen without test data. It is not a general improvement, and with two seeds it stops just short of significance.
- Together with §21–§22 the picture is consistent: **recognition** of defects is what limited data costs and what foundation features (DINOv3) buy; **extent and granularity** on the amorphous classes is an annotation-convention problem, addressed (partly) by tolerant targets and not by more data or SAM.

---

## 24. Phase 27: scaling the frozen prior and the full DINOv3 learning curve: a larger backbone helps up to ViT-B, and the DINOv3 gain does not shrink with data (19–20 Oct)

**Runs** (Kaggle, two sessions, commit `8018925`; DINOv3 from the official Hugging Face release via the `beegee11/vit{s,b,l}16` Kaggle models, checkpoint sizes 234 MB for ViT-B and 650 MB for ViT-L confirm the backbones; no non-finite batches):
- 25% of WTBD-group's labelled scenes, frozen **ViT-B/16** (`bladeyolo2b_wtbd_group_f25_s0/s1`) and **ViT-L/16** (`bladeyolo2l_wtbd_group_f25_s0/s1`), `--dino b|l`, otherwise BladeYOLO-v2 unchanged.
- BladeYOLO-v2 **without DINOv3** at 50% (`bladeyolo2nodino_wtbd_group_f50`) and 100% (`bladeyolo2nodino_wtbd_group`), one seed each, completing the curve of Phase 25.
- Training time on 2×T4: ViT-B 0.9–1.1 h, ViT-L 1.5–1.8 h (ViT-S 0.8 h); without DINOv3 1.1 h (50%) and 1.7 h (100%).

### A. Backbone size at 25%

| at 25% | test mAP50 / 50-95 (Ultralytics) | TTA | val best (mAP50 / 50-95) |
|---|---|---|---|
| ViT-S/16, 3 seeds | 0.232 / 0.082, 0.224 / 0.087, 0.184 / 0.068 | – | 0.444 / 0.226, 0.411 / 0.199, 0.435 / 0.203 |
| **ViT-B/16, 2 seeds** | **0.252 / 0.102, 0.283 / 0.116** | 0.313 / 0.127, 0.272 / 0.132 | 0.383 / 0.176, 0.416 / 0.217 |
| **ViT-L/16, 2 seeds** | **0.337 / 0.115, 0.301 / 0.109** | 0.357 / 0.153, 0.348 / 0.146 | 0.438 / 0.219, 0.484 / 0.235 |
| without DINOv3, 2 seeds | 0.114 / 0.044, 0.081 / 0.031 | – | 0.229 / 0.126, 0.186 / 0.105 |

**Paired scene bootstrap** (`runs/analysis/bootstrap/robust_check_scale.py`, log `scale_f25.log`; bootstrap evaluator, an arm is the mean of its seeds, 1,000 replicates over the 58 test scenes):

| comparison | all 6 classes: mAP50 [95% CI] | mAP50-95 [95% CI] | 4 robust classes: mAP50 [95% CI] | mAP50-95 [95% CI] |
|---|---|---|---|---|
| ViT-B − ViT-S | +0.051 [−0.000, +0.148] | **+0.028 [+0.004, +0.096]** | **+0.089 [+0.030, +0.124]** | **+0.037 [+0.007, +0.059]** |
| ViT-L − ViT-S | +0.086 [−0.009, +0.184] | **+0.031 [+0.005, +0.113]** | +0.050 [−0.012, +0.093] | +0.024 [−0.006, +0.042] |
| ViT-L − ViT-B | +0.035 [−0.096, +0.107] | +0.003 [−0.040, +0.031] | −0.040 [−0.084, +0.005] | −0.013 [−0.039, +0.007] |
| without DINOv3 − ViT-S | −0.108 [−0.240, −0.077] | −0.039 [−0.101, −0.025] | −0.099 [−0.216, −0.059] | −0.044 [−0.111, −0.023] |

Point values (all 6 classes, mAP50 / 50-95): ViT-S 0.202 / 0.076, ViT-B 0.253 / 0.104, ViT-L 0.288 / 0.106, without DINOv3 0.094 / 0.037. Per class (AP50-95, ViT-S / ViT-B / ViT-L / without): craze 0.008 / 0.008 / 0.005 / 0.006, corrosion 0.052 / 0.103 / 0.072 / 0.021, surface_injure 0.172 / 0.254 / 0.228 / 0.131, thunderstrike 0.056 / 0.076 / 0.146 / 0.000, crack 0.012 / 0.007 / 0.014 / 0.001, hide_craze 0.155 / 0.177 / 0.174 / 0.062.

**Findings:**
1. **A larger frozen backbone helps at 25%.** ViT-B and ViT-L both beat ViT-S on mAP50-95 with intervals that exclude zero (+0.028 and +0.031, about +40%), ViT-B also on the four robust classes alone. The gain is in the well-represented classes (corrosion, surface_injure, hide_craze), not only thunderstrike.
2. **It saturates after ViT-B.** ViT-L is no better than ViT-B (+0.003 [−0.040, +0.031]; on the robust classes slightly worse, P(>0) = 0.09). ViT-L's higher mAP50 comes mostly from thunderstrike (3 test scenes). The hypothesis of Phase 25 ("the low-data gain grows with the foundation model") holds from S to B, not beyond, at this data size.
3. **Validation disagrees with test on ViT-B:** on val, ViT-B is not better than ViT-S (mAP50-95 0.197 vs 0.209 seed mean), while ViT-L is best (0.227). Val has 159 images; the seed spread on val (ViT-B 0.176–0.217) is as large as the differences. Under the selection rule (val decides, test reports), **ViT-L is the val-selected backbone**; on test it is indistinguishable from ViT-B, which trains 1.6× faster.
4. **With ViT-B or ViT-L, 25% of the labels beat 50% without DINOv3** (test mAP50-95 0.104–0.106 vs 0.088, and YOLO12-L at 50%: 0.099).

### B. The learning curve with and without DINOv3

Bootstrap evaluator, all 6 classes (`robust_check_scale.py curve`, log `curve.log`), mAP50 / mAP50-95:

| | 25% | 50% | 100% |
|---|---|---|---|
| BladeYOLO-v2 (ViT-S) | 0.202 / 0.076 (3 seeds) | 0.413 / 0.175 | 0.560 / 0.242 (2 seeds) |
| without DINOv3 | 0.094 / 0.037 (2 seeds) | 0.230 / 0.088 | 0.318 / 0.152 |
| YOLO12-L | 0.099 / 0.041 | 0.225 / 0.099 | 0.400 / 0.185 |
| DINOv3 gain, mAP50-95 [95% CI] | +0.039 [+0.025, +0.101] | +0.088 [+0.040, +0.221] | +0.090 [+0.040, +0.205] |
| same, 4 robust classes | +0.044 [+0.023, +0.111] | +0.061 [+0.022, +0.179] | +0.049 [−0.018, +0.103] |

(The gain is ViT-S − without DINOv3, the negation of the logged difference.) Ultralytics test for the new runs: without DINOv3 0.225 / 0.093 at 50% (TTA 0.244 / 0.111) and 0.361 / 0.170 at 100% (TTA 0.385 / 0.210).

**Findings:**
5. **The DINOv3 gain does not shrink with data in absolute terms:** +0.039, +0.088, +0.090 mAP50-95, intervals excluding zero at every fraction (on the robust classes at 100% the interval just includes zero). In relative terms it is 2.0×, 2.0× and 1.6×. So "a low-data advantage" (§17, §22) was too narrow: frozen DINOv3 features help at every data size measured, relatively most at low data.
6. **ViT-S at 50% beats no DINOv3 at 100% on all six classes** (0.175 vs 0.152 mAP50-95), but not on the four robust classes (0.185 vs 0.217), where thunderstrike (0.282 vs 0.012 at 50%) is excluded. "DINOv3 is worth more than doubling the labels" holds only with thunderstrike, so it is not claimed.
7. **Without DINOv3, BladeYOLO-v2's architecture is no better than YOLO12-L at any fraction, and at 100% slightly worse** (YOLO12-L − without DINOv3: +0.004, +0.011, +0.033 [−0.002, +0.131] mAP50-95, P(>0) = 0.70, 0.89, 0.97). The deformable, selective-scan and WIoU components contribute nothing measurable at any data size; the whole advantage over YOLO12-L is the DINOv3 fusion.

**Conclusions:**
- **The lever is the quality of the frozen prior, and it pays at every data size.** Going from ViT-S to ViT-B adds about +40% mAP50-95 at 25% at modest cost; ViT-L adds nothing further on test.
- **The architecture around DINOv3 is dead weight.** The simplest defensible method is "frozen DINOv3 features fused into a standard detector".

**Next (proposed 20 Oct, not yet decided):**
1. **YOLO12-L + DINOv3 fusion only** (DinoConv/DinoFuse on stock YOLO12-L, no deformable/selective-scan/WIoU), ViT-S and the val-selected ViT-L at 25%, to confirm the simplification costs nothing.
2. **The val-selected ViT-L at 100%** (2 seeds, about 4 h each), to see whether the size gain persists with full data.
3. **Light adaptation of DINOv3** (LoRA or the last blocks unfrozen) at 25%, since the frozen prior is the lever.

### Programme from here (decided 21 Oct): improve the model, not diagnose it

Twenty-seven phases have mapped where the headroom is; from now on every run is meant to improve results. The evidence:
frozen DINOv3 features are the whole advantage at every data size (§22, §24); a better prior pays (ViT-B +40% at 25%)
but size saturates; at 25% the losses are misses and localisation, not class confusion (class-agnostic AP is close
to class-aware AP; 19–27% of defects missed); methods that act on labels or boxes failed (pseudo-labels, SAM).
Two things were never tried: **adapting DINOv3 to blade imagery** (it has been frozen in every run) and **the native
1024 px resolution** (every run is at 640, so a DINOv3 patch covers 25 native pixels).

> **Dropped (22 Oct, your decision):** Stage 1 is not run. LoRA and higher input resolution are standard techniques, so even a pass would not be a contribution; the novelty search continues with the Phase 28 hypotheses (§24b). `--dino-lora` stays in the code (tested) but has no runs.

**Stage 1: is the prior adaptable?** ViT-B/16 at 25%, two seeds per arm, against `bladeyolo2b_wtbd_group_f25_s0/s1`:
- **A. LoRA:** rank-16 updates on every attention qkv and output projection of DINOv3 (`--dino-lora 16`). The up-projection
  starts at zero, so training starts from the frozen features (`tests/test_dino_lora.py`).
- **B. Native resolution:** `--imgsz 1024` (DINOv3 grid 64×64 instead of 40×40), `--batch 8` against memory; Ultralytics
  accumulates to a nominal batch of 64 either way.
- **Success bar, fixed before the runs:** test mAP50-95 (bootstrap evaluator, mean of 2 seeds) at least **+0.02 over
  frozen ViT-B with P(>0) ≥ 0.95**, and the same direction on val. Arms that pass are combined.
- Screening on ViT-B instead of the val-selected ViT-L uses test information (ViT-B ties ViT-L on test at 0.6× the
  cost); stated here so it can be stated in the paper.

**Stage 2: label-free in-domain adaptation from real views** (if Stage 1 shows the prior is adaptable): LoRA-adapt
DINOv3 on the 14,000 unlabelled blade images of the view pool (§18) with a dense correspondence loss across the
49,886 verified real-view pairs (features at corresponding points of two photographs of one surface must agree),
anchored to the original model against collapse; control: the same with synthetic views. Then detection at 25% and
50%. Unlike the pseudo-label methods of §19–§21, it needs no labels on the unlabelled images. Needs a literature check
before any novelty claim.

### Literature check (21 Oct): what is and is not new

| Our element | Closest prior work | Verdict |
|---|---|---|
| DINOv3 in YOLOv12-L for blade defects with few labels | **The BladeYOLO paper itself** (Xu et al., arXiv 2607.28065, 30 Jul 2026; the paper of §2): DINOv3 ViT-S/16 (blocks 4/8/12) replaces YOLOv12-L's backbone, plus wavelet/Cross-Mamba and Style-Injector modules; random 10–70% label fractions. Also DINO-YOLO (arXiv 2510.25140): DINOv3 fused into YOLOv12 at P0/P3 for small civil-engineering datasets, nine DINOv3 variants | **Not new** |
| Leakage in the blade benchmarks | No audit of WTBD, WTBlade or Wind Surface found. The BladeYOLO paper augments before a random split, has no val set, and reports on WTBlade and Wind Surface (75% of Wind Surface val have a train twin, §4). Generic precedents: CIFAR near-duplicates (1902.00423), CrowdAI (2304.02296) | **New for this domain** |
| Leak-free group split of WTBD, scene bootstrap, causal ablations (§12, §22, §24) | None found for blade defects; the BladeYOLO paper and DINO-YOLO report single splits without scene-level intervals, and neither shows the gain coming only from DINOv3 | **New** (rigour and findings, not a method) |
| LoRA / higher resolution for DINOv3 (Stage 1) | Standard techniques | Not new; done for results |
| Dense multi-view correspondence fine-tuning of a foundation model (Stage 2) | MEF (ICLR 2025, 2411.19458): fine-tunes ViTs (DINOv2) for multiview feature equivariance; FiT3D (ECCV 2024): 3D-aware fine-tuning of DINOv2; Splat and Distill (2602.06032) finds MEF can hurt depth. In-domain continued SSL of DINOv2 is known (cell imaging, semiconductors) | **The loss is not new.** New only as the setting: casually captured inspection photos registered in 2D (no 3D), for detection, with a real-vs-synthetic-view control |

**Consequence for the paper:** the defensible contribution is (1) the leakage audit and leak-free benchmark of the blade datasets the BladeYOLO paper reports on, (2) the controlled evidence that the low-label gain comes from frozen DINOv3 features alone and holds across the learning curve, and (3) whatever Stage 1–2 add. Stage 2 is an application of MEF-style training, worth running only if it gives a large gain.

**On the paper's own modules:** the paper-faithful reimplementation was tried in Runs 1–3 (§2: Style Injection, WTConv detail branch, Cross-Mamba on frozen DINOv3 ViT-S/16; best ~52.6 / ~28 vs the paper's 86.5 / 57.4). Those runs predate the Phase 6 audit, so they used the weakened DINOv3 extractor (no RoPE, unnormalised input, §3), detached Style-Injection FFT gradients, SGD and the leaky Wind Surface split. They therefore support "not reproducible from the published information", not "the paper's modules add nothing beyond DINOv3". The second claim holds for our components (§22, §24); for the paper's exact modules it would need one corrected faithful run on the leak-free split, deferred unless a reviewer asks.

---

## 24b. Phase 28: hypotheses for a genuinely new method (21 Oct)

The user's decision (21 Oct): framing the paper on leakage and evaluation is the last resort; risky hypotheses are welcome. Candidates, in order of testing cost:
1. **Multi-view inference:** fuse detections across registered views of the same surface at test time (no training).
2. **Normal-blade anomaly prior:** distance of DINOv3 patches from normal blade surface, as an extra detector input.
3. **Retrieval prior:** nearest labelled defect patches in DINOv3 space, as a class-likelihood map.
4. **Multi-view test-time training:** LoRA adapted on each flight by cross-view agreement.
5. **RL with a granularity-invariant reward:** policy-gradient fine-tuning whose reward ignores how a defect is cut into boxes.
6. **Label-free choice of which scenes to label** (cold-start selection on DINOv3 features).

### H1 result: multi-view inference cannot be tested fairly on WTBD

`tools/multiview_fusion.py` (tests: `tests/test_multiview_fusion.py`): each view's detections are moved into the target view through the label audit's verified transforms and clustered per class; a cluster's score is the mean, over the views that see its location, of each view's best score there (a view that sees the spot and detects nothing counts 0). Settings chosen on val, applied once to test; three ViT-S 25% seeds; log `runs/analysis/multiview_f25.log`, report `runs/analysis/multiview/`.

- **WTBD has almost no genuine second views:** only 12/159 val and 26/161 test images have a partner that is a different photograph (moved frame covers < 70%). Most partners are copies of the same photo: of 510 test same-view pairs, 38% are near-exact (scale 1 ± 0.02, axis-aligned), the rest re-crops (scale 0.67–1.5) or rotations at arbitrary angles, consistent with WTBD's own augmentation.
- **Other-view partners only:** no effect (mean of 3 seeds +0.000 mAP50 [−0.003, +0.011]).
- **All partners (copies included):** +0.023 mAP50 [+0.010, +0.063], +0.006 mAP50-95. Not a valid gain: it is test-time augmentation supplied by the dataset's duplicates, which a real flight does not produce, and val did not prefer it.
- **Verdict:** untested rather than negative. A fair test needs real flights, e.g. a WTBlade benchmark (45,777 real-view pairs).

### H2/H3 diagnostic: frozen-feature priors before building them into the detector

`runs/analysis/priors/prior_diag.py` (report `report.txt`, chance level `chance.py` / `chance.log`): frozen ViT-B/16 patch features at 640 (40×40 per image), three priors scored on the 161 test images (235 boxes; 120 of them missed by ViT-B f25 s0, i.e. no prediction with conf ≥ 0.25 at IoU ≥ 0.1). A box counts as hit if any of its patches passes the threshold that flags 1% / 5% of background patches. Because boxes are large (median 111 patches), random scores already hit many boxes, so the chance level is reported alongside.

| prior | patch AUROC | box recall at 1% (missed) | at 5% (missed) |
|---|---|---|---|
| chance (random scores) | 0.50 | 0.59 (0.60) | 0.85 (0.87) |
| retrieval: kNN-10 over labelled 25% patches | 0.830 | 0.88 (0.85) | 0.97 (0.97) |
| anom_lab: distance to labelled background | 0.797 | 0.31 (0.32) | 0.67 (0.72) |
| anom_pool: distance to 800 unlabelled pool images | 0.737 | 0.22 (0.17) | 0.52 (0.48) |

- **H2 (anomaly prior) fails:** both anomaly scores rank defect patches above background overall, but their most anomalous patches are not on defects, and box recall is *below chance*. Distance from normal surface picks out other unusual content (edges, sky, tower, lighting) before defects. Weakest on crack and hide_craze (0.06–0.10 at 1%).
- **H3 (retrieval prior) is above chance but not decisive:** at 1% false alarms it hits 85% of the boxes the detector missed, against 60% by chance. So the labelled-patch memory holds information the detector does not use at its 0.25 threshold, but the same features and labels already train the detector, and a per-patch hit is far from a box.
- **Bug fixed on the way:** the first run was OOM-killed at the kNN step (4096-row fp32 blocks against 200k patches, with 2.8 GB of features in RAM); features are now memory-mapped and scored in 512-row blocks.
- **Decision:** drop H2. H3 is only worth a training run as a cheap add-on (retrieval map as an extra DinoFuse input); it is not a headline method by itself.

### H3 implementation: retrieval prior as a detector input (`train.py --retrieval`)

**Leakage while training** (`runs/analysis/priors/retrieval_leak.py` → `retrieval_leak.txt`). In training, the query image and its scene's duplicates are in the labelled memory, and under mosaic the model cannot know which image it is. Retrieval AUROC on 60 labelled images (memory: all 27,678 defect + 60,000 background patches of the 189):

| rule | clean train | augmented train | test |
|---|---|---|---|
| none | 0.996 (defect mean 0.89) | 0.923 (0.61) | 0.830 (0.415) |
| oracle: drop own scene group | 0.696 (0.011) | 0.712 (0.098) | — |
| cap 0.9 | 0.867 | 0.940 (0.60) | 0.834 |
| **cap 0.8** | 0.705 | **0.841 (0.41)** | **0.831 (0.41)** |
| cap 0.7 | 0.702 | 0.709 (0.07) | 0.785 (0.27) |

- Without a rule the prior is near-perfect in training (it reads back the image's own labels), so a detector would learn to trust it far more than it deserves at test.
- **Chosen: ignore stored patches with cosine > 0.8, at training and test alike.** Training always sees augmented images, and with this cap the prior there is as informative as at test (AUROC 0.84 vs 0.83, mean defect score 0.41 vs 0.41) while test loses nothing (0.831 vs 0.830). Test top-1 similarity: median 0.80, 99th percentile 0.94.
- **Open question:** leaving out the query's own scene is *weaker* than test (0.70 vs 0.83, mean defect score 0.01 vs 0.42): labelled training scenes resemble each other less than test scenes resemble them. Not explained yet (43 scene groups among the 189 images; a few large groups may hold most examples of a class).

**Implementation** (`models/retrieval.py`, hooks in `models/dino_fusion.py`, `train.py --retrieval [--retrieval-cap 0.8]`; tests `tests/test_retrieval.py`):
- Before the model is built, `train.py` encodes the labelled training images (unaugmented, 640 px, last DINOv3 block, L2-normalised): every defect patch and up to 60,000 background patches (re-weighted by background / kept), saved as `runs/retrieval_memory_<name>.pt`.
- DinoConv registers it as persistent buffers (fp16), so checkpoints carry the memory to validation and test. ViT-B, 25%: about 88k × 768 patches, +135 MB per checkpoint.
- Per input patch, under `no_grad`: 64 nearest stored patches by cosine, those above the cap dropped, the first 10 vote; output is per-class vote shares (6 channels), attached as `.retr`. Each DinoFuse concatenates it with its DINOv3 level, through the existing zero-initialised projection.
- Refused with `--dino-lora` (the stored features would go stale as DINOv3 adapts).
- Verified locally: unit tests (cap, background weighting, memory build, DinoFuse wiring, pickle round trip) and an end-to-end CPU run (16 images, 1 epoch, 320 px, ViT-B): memory build → training → val → `best.pt` → standalone fused test with TTA. The automatic test step was OOM-killed on this 7 GB machine (training process and test process both resident), not on Kaggle's 30 GB.

**Pre-registered (same bar as Stage 1):** ViT-B, 25%, seeds 0 and 1, against frozen ViT-B s0/s1. Passes if test mAP50-95 rises ≥ +0.02 with P(>0) ≥ 0.95 (paired scene bootstrap, both seeds) and val agrees in direction.

### H3 result: the retrieval prior does not help (fails the bar)

Runs (Kaggle, commit `733eb67`, 150 epochs each): `bladeyolo2b_retr_wtbd_group_f25_s0/s1`. Paired scene bootstrap against frozen ViT-B s0/s1 (`runs/analysis/bootstrap/robust_check_scale.py h3` → `h3_retrieval.log`; 1,000 replicates over the 58 test scenes; an arm is the mean of its seeds):

| | ViT-B | ViT-B + retrieval | difference [95% CI], P(>0) |
|---|---|---|---|
| test mAP50, 6 classes | 0.253 | 0.258 | +0.004 [−0.036, +0.078], 0.64 |
| **test mAP50-95, 6 classes** | 0.104 | 0.105 | **+0.001 [−0.020, +0.030], 0.59** |
| test mAP50-95, 4 robust classes | 0.135 | 0.141 | +0.006 [−0.018, +0.024], 0.73 |
| val mAP50-95 (best epoch), s0 / s1 | 0.176 / 0.217 | 0.235 / 0.194 | +0.059 / −0.023 |

- **Fails:** far from +0.02, and val disagrees between seeds.
- The prior marks 85% of the missed defects at 1% false alarms as a patch map, but the detector, which already sees the same DINOv3 features, gains nothing from it. Per class, hide_craze rises (0.177 → 0.211) and thunderstrike falls (0.076 → 0.055); both are within seed noise.
- **Closed.** Knowledge from the 25% labels is already used; a non-parametric copy of it adds no new information.

### H4 test bed: WTBlade-group (multi-view test-time methods)

WTBD cannot test multi-view methods (H1: 26 test images with a genuine second view). WTBlade can, once its leaks and copies are removed.

**Group split** (`tools/wtblade_groups.py` → `runs/analysis/wtblade/{split_groups.csv, report.json}`):
- 10,527 images; the 133 that match WTBD are left out, so the two benchmarks share nothing.
- Same scene if same source frame (Roboflow name prefix; 2,791 frames), a verified view-pool pair (copies and real views, 68,076), or DINOv2 neighbours with cosine ≥ 0.93 / ≥ 25 ORB inliers.
- Class-stratified 70/15/15 over groups, then the WTBD-group audit loop (held-out vs top-10 train neighbours, ORB-verified or cosine ≥ 0.93). **Converged after 11 rounds with 841 groups** and no held-out/train match. Largest groups 396–1,793 images (drone videos). Official 7 classes, all present in every split.

**Dataset** (`tools/build_wtblade_group.py` → `external/WTBlade-group`, zip `runs/exports/WTBlade-group.zip`, 101 MB):
- **One image per source frame** (first copy by name). The Roboflow copies are offline augmentations, and H1 showed copies act as free TTA.
- Frames: **train 2,136 / val 323 / test 332**; all 640 × 640.
- Real views: same-split frames linked in the view pool, registered again on the kept copies (ORB/RANSAC similarity + flip, both directions, ≥ 25 inliers) → `views_{val,test}.json`. **Test: 155 frames have at least one registered partner frame, 41 a different view (coverage < 0.7); val: 161 and 36.** Partner coverage median 0.82 (test), 0.84 (val). All partners are distinct captures (different source photos; Phase 21's visual check).
- **Power caveat:** the multi-view test frames sit in few scene groups (16 for the 100 linked at pool level), so scene-bootstrap intervals will be wide. A multi-view method has to show a large effect to pass.
- Licence: contains DTU NordTank frames (CC BY-NC). Kaggle dataset private; research use only.

**Plan, in order (each step gates the next):**
1. Base detector: BladeYOLO-v2 ViT-B on all 2,136 training frames, 50 epochs (about as many optimiser steps as the WTBD 150-epoch runs).
2. **H1 on WTBlade** (no training): multi-view fusion of the base detector's outputs over the registered partners, settings chosen on val. If fusion does not make better detections than a single view, test-time adaptation has no signal to learn from, and H4 stops here.
3. **H4: per-scene test-time adaptation** on pseudo-labels fused across registered real views; reset per scene.
   - Control: the same procedure with synthetic partner views (augmentations of the frame itself, same count).
   - Pre-registered: passes if real-view adaptation beats the synthetic-view control by ≥ +0.02 mAP50-95 with P(>0) ≥ 0.95 on test frames with a registered partner, and val agrees.

### H4 result: the gate fails, since real views do not make better detections even on WTBlade

**Base detector** `bladeyolo2b_wtblade_group` (Kaggle, ViT-B, all 2,136 training frames, 50 epochs; val best at epoch 42, 0.426 / 0.172):
- Test mAP50 / mAP50-95: **0.422 / 0.182** (Ultralytics).
- Per class AP50-95: peeling 0.358, deformity 0.286, dirt 0.265, rust 0.133, burn 0.093, oil 0.090, crack 0.047.
- It is the first leak-free WTBlade result.

**H1 on WTBlade** (`tools/multiview_fusion.py --views external/WTBlade-group …` → `runs/analysis/multiview_wtblade/`, log `runs/analysis/multiview_wtblade.log`; settings chosen on val, applied once to test; bootstrap evaluator, 1,000 replicates over the test scene groups):

| test frames | partners | setting (val) | single view | fused | Δ mAP50 | Δ mAP50-95 |
|---|---|---|---|---|---|---|
| all 332 | different view | IoU 0.55, w 0.5, mean | 0.419 / 0.179 | 0.407 / 0.171 | −0.011 [−0.022, +0.001] | −0.008 [−0.015, +0.000], P(>0) 0.10 |
| 41 with a different view | different view | same | 0.548 / 0.249 | 0.512 / 0.217 | −0.036 [−0.041, +0.025] | −0.032 [−0.040, +0.062], 0.28 |
| all 332 | any (incl. neighbouring frames) | IoU 0.4, w 0.5, own | 0.419 / 0.179 | 0.423 / 0.177 | +0.005 [−0.008, +0.018] | −0.002 [−0.008, +0.004], 0.23 |
| 155 with any partner | any | same | 0.443 / 0.197 | 0.457 / 0.195 | +0.015 [−0.005, +0.069], 0.93 | −0.001 [−0.014, +0.020], 0.40 |

- **On val, different-view fusion lowers every setting** (0.163–0.167 against 0.168). The best any-partner setting adds +0.002.
- **Reading:** where a second view genuinely differs, moving its detections through a 2D similarity transform and averaging scores dilutes correct detections (the other view sees the spot at another scale and angle, or not at all) more than it recovers misses. Neighbouring frames behave like test-time augmentation: a small mAP50 gain that does not survive at mAP50-95.
- **Verdict:** H1 fails on a fair test bed, and with it the gate for H4. Pseudo-labels fused across real views are no better than single-view ones, so per-scene test-time adaptation would have no extra signal to learn from. **H4 is not run.** Together with Phases 22–23 (real-view teacher–student training no better than synthetic views), real second views have now failed as a signal at training time, at inference and as a source of pseudo-labels.
- **Still useful:** WTBlade-group is a second leak-free benchmark (841 scene groups) for any later method.

### Round 2 of Phase 28 (22 Oct): everything that could add something unexplored

Your direction (22 Oct): "let's do literally everything that can contribute something previously unexplored, as much experimentation as possible." Stage 1 (LoRA, 1024 px) is dropped as not novel. Status of the candidates:

| # | idea | status |
|---|---|---|
| 1 | copies of an image as a second annotator (consensus-aware targets) | measured below, too little signal at 25% |
| H5 | reward fine-tuning with a granularity-invariant reward | implemented, for Kaggle |
| H6 | label-free choice of the scenes to label (patch coverage) | implemented, for Kaggle |
| 4 | zero-shot open-vocabulary detectors on the leak-free splits | next |
| 5 | transfer between WTBD-group and WTBlade-group | next |

**Idea 1 measured (label audit pairs, same-view copies):**
- 25% subset: 138 of 189 images have a same-view copy in the subset. Of their 524 projected labels, 89% agree, 8% are loose, 3% split and 0.2% missing. **Only 24 images have any disagreement**, and none is about class.
- All 745 training images: 573 have a copy, 120 have a disagreement (15% of projected labels).
- The model already trains on both versions of each disputed box. Targets built from consensus would change 24 images at 25%, which cannot plausibly move test mAP50-95 by the +0.02 bar. **Not run at 25%**; reconsider only if a 100% experiment is made.

### H5: reward fine-tuning with a granularity-invariant reward (`models/reward.py`, `train.py --reward W --reward-start E`)

- **Idea:** the boxes a detector outputs are a stochastic policy, and the reward depends only on *where* defects are claimed, not on how they are cut into boxes. Unlike the split-tolerant loss (Phase 18), which changes per-anchor targets, this optimises a set-level, non-differentiable score directly. Related work: task-reward tuning of vision models (Pinto et al., ICML 2023) used standard rewards; a partition-invariant reward for amorphous defects is the new part.
- **Policy, per image:** top 30 detections after NMS. Each is kept with probability sigmoid(its class logit), and a kept box is drawn from a Gaussian around the predicted box (sd 10% of width and height).
- **Reward:** per class, F1 between the union of kept boxes and the union of labelled boxes on a 4 px raster, averaged over the classes present.
- **Update:** 8 samples per image, leave-one-out baseline (RLOO); weight W on top of the unchanged supervised loss from epoch E. Per-epoch reward in `<run>/reward_stats.csv`.
- **Tests** (`tests/test_reward.py`): a region split into two boxes earns the full reward; the policy gradient raises a correct box's logit and lowers a false one's (512 samples); the full model's loss is finite and trains.
- **Runs:** ViT-B, 25%, `--reward 1.0 --reward-start 100`, seeds 0 and 1, 150 epochs. Before epoch 100 the run is identical to `bladeyolo2b_wtbd_group_f25_s0/s1`.
- **Pre-registered:**
  - Weights compared: `last.pt` against the baselines' `last.pt`, because val-mAP checkpoint selection could otherwise pick a pre-reward epoch.
  - Primary: **region AP50** (`tools/granularity_eval.py`) at least +0.02 with P(>0) ≥ 0.95 (paired scene bootstrap, mean of 2 seeds), same direction on val.
  - Guard: standard test mAP50-95 not more than 0.01 below the baseline (point estimate).
  - W = 1.0 is fixed in advance; no tuning.

### H6: label-free scene selection (`tools/select_scenes.py`)

- **Question:** which scenes should be labelled, before any label exists? The 25% subset used so far was chosen using labels (random scene orders scored on class-mix match), so it is a strong baseline.
- **Method:**
  1. 128 random DINOv3 ViT-B last-block patches per training image.
  2. k-means into 1,024 centroids.
  3. Greedy choice of whole scenes for the largest gain per image in soft coverage, i.e. the sum over centroids of the best cosine of any selected patch. Every centroid weighs the same, so rare patterns (defects cover few patches) count as much as common ones.
  4. Same budget: 189 images, ≤ 5% over.
- **Controls:** two plain random scene draws of the same budget (no labels), for subset luck.
- **Novelty:** cold-start selection on self-supervised features is known in classification (ProbCover, TypiClust). Patch-level coverage for scenes in detection with tiny objects is the new part, but modest.
- **Runs:** ViT-B, seeds 0 and 1 for `_cover`, seed 0 for `_rand1` and `_rand2`.
- **Pre-registered:**
  - Passes if `_cover` reaches test mAP50-95 at least +0.02 above the class-matched subset (`bladeyolo2b_wtbd_group_f25_s0/s1`) with P(>0) ≥ 0.95, and val agrees.
  - Also reported: `_cover` against the mean of the two plain random draws (a label-free selector should at least beat label-free random).
- **Subsets** (`runs/analysis/select_scenes/wtbd_group_train_f25.txt`; 745 training images in 224 scenes):

| subset | scenes | images | boxes craze / corrosion / surface_injure / thunderstrike / crack / hide_craze |
|---|---|---|---|
| class-matched random (existing, uses labels) | 51 | 189 | 45 / 38 / 79 / 10 / 39 / 60 |
| **cover** (label-free) | **85** | 195 | 39 / 83 / 51 / **31** / 9 / 52 |
| rand1 (label-free) | 84 | 189 | 24 / 49 / 94 / 12 / 8 / 100 |
| rand2 (label-free) | 55 | 193 | 29 / 74 / 74 / 25 / 1 / 70 |

  The coverage selector prefers many small scenes and, without labels, reaches every class with a more even spread (three times the thunderstrike boxes), but it finds few cracks (9 against 39).

### H6 result: label-free selection passes on test (robustness pending)

Runs (Kaggle, 150 epochs, ViT-B): `bladeyolo2b_cover_wtbd_group_f25_s0/s1`, `bladeyolo2b_rand{1,2}_wtbd_group_f25_s0`. Paired scene bootstrap (`runs/analysis/bootstrap/robust_check_scale.py h6` → `h6_select.log`; 1,000 replicates over the 58 test scenes; an arm is the mean of its runs):

| arm | test mAP50 | test mAP50-95 | val mAP50-95 (best epoch) |
|---|---|---|---|
| class-matched 25% (uses labels) | 0.253 | 0.104 | 0.176 / 0.217 |
| plain random (2 draws) | 0.318 | 0.124 | 0.235 / 0.265 |
| **cover (label-free)** | **0.420** | **0.182** | 0.220 / 0.230 |

- **cover − class-matched:** mAP50-95 **+0.078 [−0.002, +0.217], P(>0) 0.972**; mAP50 +0.167 [+0.042, +0.356]. Val is in the same direction (+0.029). **Passes the pre-registered bar.**
- **cover − plain random:** mAP50-95 +0.058 [+0.015, +0.196], P 0.997. **Val disagrees:** both random draws beat cover on val.
- **plain random − class-matched:** +0.020 [−0.047, +0.038]. The class-matched subset is an unlucky draw on test.
- **Where the gain is:** per-class AP50-95 rises for thunderstrike from 0.076 to 0.337 and for crack from 0.007 to 0.159 (cover has 3× the thunderstrike boxes, but fewer crack boxes: 9 against 39). On the 4 classes with ≥ 8 test scenes the gain is +0.041 [−0.006, +0.120], P 0.951.
- **Not yet robust.** Two random draws cannot measure subset luck, and val ranks the random draws above cover. Robustness runs are needed: 6 more random draws and 2 cover subsets with other k-means seeds (`tools/select_scenes.py --seed`).

### H6 robustness round: pre-registered before the runs (23 Oct)

**Subsets** (`runs/analysis/robust_subsets.sh`; all 189–197 images; `tools/splits/wtbd_group_train_f25_*`):
- `rand3`–`rand8`: 6 more plain random scene draws (same generator; `rand1`/`rand2` reproduced byte for byte). With rand1/2: 8 draws. Their class counts vary widely, e.g. crack 1–93 boxes, thunderstrike 9–35.
- `cover_k1`, `cover_k2`: the cover selector with k-means seeds 1 and 2 (89 and 95 scenes; thunderstrike 30 / 29, crack 17 / 15).
- `freesel224_s0/s1`, `freesel640_s0/s1`: FreeSel (`tools/freesel.py`), as published at 224 px and at our 640 px, two sampling seeds each. It selects images, not scenes.

**Runs:** ViT-B, 150 epochs, training seed 0, one run per subset (12 runs).

**Pre-registered analysis** (bootstrap evaluator, paired scene bootstrap over the 58 test scenes):
- *cover family* = mean of `cover` s0, s1, `cover_k1`, `cover_k2`; *random* = mean of the 8 random draws; *FreeSel-224* and *FreeSel-640* = mean of their two subsets.
- **H6 holds against random** if cover family − random is ≥ +0.02 test mAP50-95 with P(>0) ≥ 0.95, and on val the cover family's mean (best-epoch mAP50-95) is above the random draws' mean. Also reported: cover's rank among the 8 random draws, on test and on val.
- **H6 improves on FreeSel** if cover family − FreeSel-224 is ≥ +0.02 with P(>0) ≥ 0.95 (and the same direction on val). FreeSel-640 is reported as a stronger variant.
- If cover fails against random, the first H6 result is attributed to subset luck.

### H5 result: region AP passes on test; val check pending

Runs (Kaggle, 150 epochs, ViT-B, `--reward 1.0 --reward-start 100`): `bladeyolo2b_reward_wtbd_group_f25_s0/s1`. As pre-registered, `last.pt` against the baselines' `last.pt` (`runs/analysis/h5_last/`, symlinked run folders). Scored with `tools/granularity_eval.py --arms` (new option: difference of arm means; 500 replicates) → `runs/analysis/novel/granularity/h5_reward_last/report.md`:

| test, mean of 2 seeds | reward − baseline [95% CI] | P(>0) |
|---|---|---|
| **region AP (primary)** | **+0.081 [−0.007, +0.102]** | **0.97** |
| standard AP50 | +0.011 [−0.037, +0.082] | 0.74 |
| tolerant AP50 | +0.017 [−0.032, +0.078] | 0.79 |
| standard AP50-95 (guard: ≥ −0.01) | −0.005 [−0.035, +0.013] | 0.37 |
| tolerant AP50-95 | −0.002 [−0.033, +0.015] | 0.44 |

- Region AP per run: reward 0.490 / 0.492, baseline 0.375 / 0.444.
- **Primary metric passes on test and the guard holds.** The detector learned what the reward pays for: where defects are, not how they are cut into boxes. Standard and tolerant AP are unchanged.
- **Val condition not yet checked:** the training logs only have standard val mAP (final epoch: s0 +0.030, s1 −0.033). Val region AP still has to be computed.
- Training: mean reward per epoch rose from 0.49 (epoch 101) to 0.62 (epoch 150); kept boxes per image fell from ~2.0 to ~1.3.
- Caveat: before the reward starts, seed 1 had already diverged from its baseline (val 0.384 vs 0.403 at epoch 100). GPU nondeterminism alone gives ~0.02 between nominally identical runs.

**Literature check for H5 (23 Oct):**
- **Policy-gradient detector training on a set-level reward is prior art:**
  - Rao, Lu and Zhou, *Learning Globally Optimized Object Detector via Policy Gradient*, CVPR 2018, treat the set of detections as the action and optimise a non-differentiable mAP-based reward with REINFORCE.
  - Pinto et al., ICML 2023, tune vision models with task rewards.
- **Granularity for amorphous defects:** crack work handles it in the annotation (masks for cracks in Rail-5k, centre-line labels, weak and coarse labels; Xu et al., CVPRW 2023, study label errors for thin cracks). No detector training with a reward that ignores how a region is cut into boxes was found.
- **What is new:** the reward, i.e. union-of-boxes F1 per class, invariant to splitting and merging, for detectors on amorphous defects, together with the evidence that it raises region AP without hurting box AP. RL fine-tuning of detectors itself is not new.

### Literature check for H6 (23 Oct)

| work | what it does | relation to H6 |
|---|---|---|
| **FreeSel** (Xie et al., NeurIPS 2023, arXiv 2309.17342) | Single pass, label-free. DINO ViT-S/16 last-layer patch features, filtered by [CLS] attention (keeps the patches holding τ of the attention mass), spectral clustering per image into K "semantic patterns"; images chosen by k-means++-style D² sampling over patterns. Detection on PASCAL VOC (SSD), Cityscapes segmentation, CIFAR | **Same core idea**: patch-level self-supervised features for choosing what to label for detection. Patch-level label-free selection is therefore **not new** |
| **OFDS** (Popp et al., arXiv 2412.10032, 2024/25) | Object-level features: Grounding DINO prompted with the **class names** + SAM 2 object pointers; per-class k-means, rare classes first. VOC, Cityscapes, LoveDA; beats FreeSel, K-Centers, TypiClust-style baselines; largest gains on rare classes; tiny proposals filtered out | Needs an open-vocabulary proposer that recognises the classes. On blades OWLv2 finds almost nothing (item 4: class-agnostic AP50 0.057), so OFDS's premise fails here |
| TypiClust, ProbCover, USL (Wang et al. 2022), CALR, Chen et al. 2023 "hard-to-contrast" | Cold-start selection from self-supervised **image-level** embeddings, classification | Image-level; OFDS reports image-level K-Centers below random on VOC detection |
| DINOv2 + k-means coresets (agriculture, Frontiers 2025) | Image-level clustering for classification | Image-level |
| Foundation-assisted AL for detection (arXiv 2607.16671, Jul 2026) | Remote sensing; UPN + SAM 2 + DINOv2 object-level diversity plus uncertainty; iterative | Object-level, needs a proposer |
| Blade defects | No active-learning or annotation-selection work found; BladeYOLO (2607.28065) uses random label fractions | Domain gap |

**What could still be new in H6 (each needs evidence):**
1. **Tiny defects defeat saliency filtering.** FreeSel keeps only the patches the [CLS] token attends to. Defects cover a few patches of a blade photo, so they may be filtered out before selection. H6 keeps an unbiased patch sample. *Testable now:* the share of labelled defect patches that FreeSel's attention filter keeps on WTBD train. If it drops most of them, that is a clean, new negative finding about the established method.
2. **No proposer and no class names**, in a domain where open-vocabulary proposers fail (item 4), unlike OFDS.
3. **Scene-level selection** that respects near-duplicate groups (FreeSel and OFDS select images).
4. **Equal-weight coverage of global patch modes** (a deterministic facility-location objective) instead of D² sampling, which chases outliers.
5. **Domain evidence:** the first annotation-selection result for blade inspection, on a leak-free benchmark.

**Claim 1 tested and falsified** (`tools/freesel.py --diagnose` → `runs/analysis/select_scenes/freesel_attention_filter.txt`; DINO ViT-S/16, τ = 0.5, WTBD-group training images):

| input | patches kept | defect patches kept | images losing every defect patch |
|---|---|---|---|
| 224 px (FreeSel's setting) | 16.8% | **50.1%** of 18,950 | 0.6% of 726 |
| 640 px | 12.7% | **39.7%** of 151,454 | 0.0% of 745 |

The [CLS] attention of DINO concentrates on blade defects about 3× above chance. FreeSel's filter therefore keeps defects rather than discarding them, and FreeSel is a stronger, fairer baseline than expected. The case for H6 now rests on claims 2–5 and on the head-to-head runs.

**Consequence:** H6 can only be claimed as a contribution *against FreeSel (and OFDS) as baselines*, on WTBD and ideally WTBlade. Without that comparison it is an application of a known idea. Next: (1) the attention-filter diagnostic, which costs nothing; (2) FreeSel subsets (code is public) and OFDS subsets as Kaggle arms next to the robustness runs.

### Item 5: transfer between the two leak-free benchmarks (`tools/transfer_eval.py`)

- **Models:** WTBD-group 100% (ViT-S/16, 2 seeds) and WTBlade-group 100% (ViT-B/16), each tested on both test sets.
- **Scoring:** damage only (WTBlade's dirt and oil dropped), class-agnostic; plus the two classes that map one to one (crack ↔ crack, thunderstrike ↔ burn). Paired scene bootstrap, 1,000 replicates. Report `runs/analysis/transfer/report.md`. The run was interrupted twice by machine restarts; prediction caches let it finish.

| test set | in-domain agnostic AP50 / AP50-95 | transferred agnostic AP50 / AP50-95 | transferred − in-domain AP50 [95% CI] |
|---|---|---|---|
| WTBD-group (161 images, 58 scenes) | 0.563 / 0.250, 0.580 / 0.258 | 0.052 / 0.011 (WTBlade model) | −0.525 [−0.694, −0.329] |
| WTBlade-group (332 frames, 177 scenes) | 0.475 / 0.202 | 0.040 / 0.016, 0.022 / 0.007 (WTBD models) | −0.439 [−0.496, −0.346] |

- **No transfer in either direction:** a detector trained on one blade benchmark finds about 5–9% of what the in-domain detector finds on the other, even ignoring class. Mapped classes get nothing: crack AP50 is 0.000 both ways, lightning 0.073 at most.
- **Possible causes, not yet separated:**
  - imaging: WTBD crops 1024 px close-ups, while WTBlade is 640 px drone and video frames, often whole blades;
  - annotation conventions: what counts as "crack", box extent and splitting (§14: labels disagree even within each dataset);
  - taxonomy.
- **Consequence:** results on one blade dataset say little about another, which supports reporting every method on both leak-free benchmarks. It is a measurement, not a method.

### Item 4: open-vocabulary detection without training (`tools/openvocab_eval.py`)

- **Model:** OWLv2 (`google/owlv2-base-patch16-ensemble`, local copy in `external/owlv2`; the Hugging Face Xet download stalled, so the weights came from the CDN with curl).
- **Queries:**
  - *text*: three fixed descriptions per class, written before any result; class score = best description.
  - *proto*: few-shot from the 25% labels without training. For each labelled box, the class embedding of the patch whose predicted box overlaps it most (exemplars: craze 45, corrosion 38, surface_injure 79, thunderstrike 10, crack 39, hide_craze 60); per class the normalised mean is the query.
- **Scoring:** bootstrap evaluator on the WTBD-group test split (1,000 scene replicates), against the trained ViT-B 25% detector. Report `runs/analysis/openvocab/report.md`.

| | mAP50 | mAP50-95 | class-agnostic mAP50 / mAP50-95 |
|---|---|---|---|
| OWLv2 text | 0.033 | 0.014 | 0.057 / 0.025 |
| OWLv2 proto (25% exemplars) | 0.006 | 0.002 | 0.005 / 0.002 |
| ViT-B 25% s0 / s1 (trained) | 0.250 / 0.257 | 0.099 / 0.108 | 0.314 / 0.297 |

- text − trained: mAP50-95 −0.122 [−0.207, −0.075]; proto − trained: −0.137 [−0.225, −0.089].
- **Result:** a web-scale open-vocabulary detector finds almost none of the blade defects, even ignoring class (0.057 class-agnostic AP50, against 0.30 for a detector trained on 189 images). Text descriptions do best on corrosion (0.08 AP50) and crack (0.06).
- Mean exemplar prototypes are worse than text. Presumably the box-averaged embedding mostly encodes blade surface and fires on background. Variants (nearest exemplar, several prototypes per class) were not tried, since they would be chosen after seeing the result.
- **Contribution:** a measurement, not a method. It is the first leak-free number for open-vocabulary detection on blade defects, and it shows the 25%-label regime is far from solved by foundation detectors.

---

## 25. Engineering problems solved

| Problem | Root cause | Fix | Commit |
|---|---|---|---|
| WIoU never active | Patched the wrong namespace | Patch the loss namespace | `dfdb5f4` |
| NaN in SS2D (epoch 9) | fp16 overflow before the LayerNorm | fp32 LayerNorm | `2332cf4`, `4cf2c85` |
| NaN epoch 4 (v2) | Poisoned WIoU global + aggressive warmup | Finite-only running mean; lr0 0.001, `warmup_bias_lr` 0 | `2efd069` |
| NaN epoch 17 (v2) | fp16 overflow, layer 26 conv into BN | `FP32Block` for the refinement blocks | `08f9a4b` |
| NaN epoch 99 (WTBD) leaving val at 0 | NaN batch poisons BN running stats + EMA | Per-batch BN repair + step skip + per-layer report | `91ae01f` |
| NaN batches late in training (epochs 99–149) | YOLOv12 `AAttn` forms attention scores in fp16; they overflow (~4e5) | Scores and softmax in fp32 (`models/precision.py`), version-guarded | Phase 18 |
| Test evaluation silently missing (twice) | `deform_conv2d` segfault on NaN offsets from the FLOPs probe | Finite, clamped offsets; test in a fresh process | `b8721bc` |
| Strict pretrained loading mismatch | YOLOv12 checkpoint carries `attn.pe.conv.bias` | Fold the bias exactly into BN `running_mean` | `e77dc0d` |
| Ultralytics label cache on a read-only Kaggle input | Input mount not writable | Copy the dataset into the project folder once | `1d58ea2` |
| Background processes, `!cmd &` on Kaggle | Notebook limitation | `subprocess.Popen` / Save & Run All | — |
| Windows MX230 workstation | Pascal GPU, CUDA 12.5 driver | torch 2.6 + cu124; inference/analysis only | `091cb8a` |
| LoRA weights inside DINOv3 would be frozen by the trainer | Ultralytics freezes every parameter whose name contains `model.0.dino.` | Updates registered on DinoConv as `dino_lora`; the wrapped layers only reference them; tested through deepcopy and pickle | Phase 27 |

---

## 26. Corrections to earlier conclusions

| Earlier conclusion | Status | Why |
|---|---|---|
| Run 6/7 reach ~78 / ~47 on Wind Surface Defect | **Invalid** | 75% of val had train twins (§4) |
| WIoU v3 gave +0.9 mAP50-95 (Run 7) | **Invalid** | WIoU was never active; Run 7 trained with CIoU (§3) |
| Mamba added global context (Run 7) | **Invalid** | `SimpleSS2D` did not scan (§3) |
| SGD → AdamW was the breakthrough | **Confounded** | It changed together with an 8M → 57M scale-up |
| BladeYOLO-L's remaining gap is locality bias | **Superseded** | The gap was leakage plus data structure |
| v3 is a better dataset because it is larger | **Superseded** | Scenes, not files, matter: WS 81, WTBD 550 (§10) |
| Mislabelling explains the poor v3 scores | **Rejected** | The review was worth only +0.03 mAP50 (§7) |
| The first WTBD run's 0.649 val is its ceiling | **Corrected** | The completed run plateaus at ~0.63; 0.649 was noise |
| The local thop segfault is a CPU-only quirk | **Corrected** | It is a real NaN-offset bug that also hit Kaggle (§11) |
| BladeYOLO-v2's leak-free win over YOLO12-L shows the architecture generalises better | **Qualified** | The win rests mainly on thunderstrike (3 test scenes); without it the interval includes zero (§12) |
| BladeYOLO-v2's advantage comes from its architecture (deformable, selective-scan, WIoU) | **Rejected** | Without DINOv3 it equals YOLO12-L at 25% and 50% and is slightly worse at 100%; the frozen DINOv3 features carry the whole advantage (§22, §24) |
| Honest calibration costs 2.4 instead of 1.0 false alarms per image (Phase 16) | **Corrected** | That column counted all predictions; real false alarms are 0.2 vs 0.1 per image (§14) |
| WTBD copies agree on 86.0% of labels (Phase 16) | **Refined** | Rotated copies inflate "loose"; axis-aligned copies agree on 87.3% (§14) |
| WTBlade is CC BY 4.0 (Roboflow page) | **Questioned** | It contains DTU NordTank images (CC BY-NC) and synthetic copy-paste images (§14) |
| Part of the leakage inflation is memorised box splitting (Phase 16 hypothesis) | **Not supported** | The tolerant gain is not smaller on test images with a train twin (§14, step 14) |
| At 25%, most lost defects are found but boxed differently (first diagnostic, Phase 24) | **Refined** | With a stricter criterion, near-misses (34–36%) and misses (19–27%) are of similar size; near-misses are granularity and misalignment, not loose boxes (§21) |
| Split tolerance helps corrosion (Phase 18, one run) | **Replicated, not significant** | Two class-selective seeds: corrosion mAP50-95 0.13 → 0.25, +0.093 [−0.003, +0.246]; aggregate unchanged (§23) |
| BladeYOLO-v2's DINOv3 advantage is a low-data advantage (§17, §22) | **Corrected** | Without DINOv3 at 50% and 100% the gap stays as large in absolute terms (+0.039 / +0.088 / +0.090 mAP50-95); only the relative gain shrinks (2.0× → 1.6×) (§24) |
| The low-data gain grows with the size of the frozen DINOv3 (Phase 25 hypothesis) | **Partly supported** | ViT-B beats ViT-S at 25% (+0.028 [+0.004, +0.096] mAP50-95); ViT-L is no better than ViT-B (§24) |
| Teacher–student training gives a small, consistent gain on well-represented classes (Phase 23) | **Qualified** | Single seeds; seed noise at 25% is ±0.026 mAP50, as large as the effects. The class-balanced variant is significantly worse with 3 seeds (§21) |

---

## 27. Run registry

| Run folder | Data | Model | Epochs | Val best (mAP50 / 50-95) | Test (mAP50 / 50-95) | Notes |
|---|---|---|---|---|---|---|
| Runs 1–5 | WS original | Paper-faithful / BladeYOLO-S | — | see §2 | — | Logs on Kaggle only |
| `bladeyolo_l_sota` (Run 6) | WS original (**leaky**) | BladeYOLO-L | 400 | 0.781 / 0.469 | — | Inflated |
| `bladeyolo_l_sota-2` (Run 7) | WS original (**leaky**) | BladeYOLO-L + "Mamba" + (inactive) WIoU | 400 | 0.769 / 0.478 | — | Inflated |
| `yolo12l_baseline` | v2 | YOLO12-L | 266 | 0.290 / 0.122 | 0.358 / 0.149 | lr0 0.002 |
| `bladeyolo_l` | v2 | BladeYOLO-L (repaired) | 167 (stopped) | 0.344 / 0.123 | 0.400 / 0.146 (epoch 124) | |
| `yolo12l_v3` | v3 | YOLO12-L | 150 | 0.290 / 0.141 | 0.263 / 0.109 | Old harmful recipe |
| `bladeyolo_v3` | v3 | BladeYOLO-L v1 | 150 | 0.356 / 0.156 | 0.317 / 0.124 | |
| `bladeyolo2_v3` | v3 | BladeYOLO-v2 | 100 | 0.369 / 0.172 | 0.296 / 0.123 (evaluator) | Test lost on Kaggle (segfault) |
| WBF ensemble | v3 / v3.2 | 3 models | — | — | 0.339 / 0.143 (v3); ~0.411 / 0.168 (merged) | Fusion IoU chosen on val |
| `bladeyolo2_wtbd_group` (attempt 1) | WTBD-group | BladeYOLO-v2 | 99 of 150 | 0.649 / 0.379 | — | NaN at epoch 99; folder lost |
| **`bladeyolo2_wtbd_group`** | WTBD-group (leak-free) | BladeYOLO-v2 | 150 | 0.650 / 0.368 (epoch 124) | **0.548 / 0.245**; TTA 0.574 / 0.279 | Test run locally |
| **`bladeyolo2_wtbd_official`** | WTBD official (**leaky**) | BladeYOLO-v2 | 120 | 0.845 / 0.475 (epoch 105) | **0.840 / 0.493**; TTA 0.830 / 0.517 | TTA run locally |
| **`bladeyolo2_wtbd_group_tolerant`** | WTBD-group (leak-free) | BladeYOLO-v2, split-tolerant targets | 150 | 0.609 / 0.352 (epoch 83) | **0.523 / 0.249**; TTA 0.552 / 0.282 | 5 NaN batches dropped (attention overflow, since fixed) |
| `bladeyolo2_wtbd_group_f25` | WTBD-group, 25% of train scenes | BladeYOLO-v2 | 150 | 0.444 / 0.226 (epoch 126) | **0.232 / 0.082**; TTA 0.269 / 0.105 | Learning curve |
| `bladeyolo2_wtbd_group_f50` | WTBD-group, 50% of train scenes | BladeYOLO-v2 | 150 | 0.472 / 0.263 (epoch 140) | **0.413 / 0.183**; TTA 0.473 / 0.211 | Learning curve |
| `yolo12l_wtbd_group_f25` | WTBD-group, 25% of train scenes | YOLO12-L | 150 | 0.233 / 0.139 (epoch 150) | **0.106 / 0.037**; TTA 0.144 / 0.059 | Learning curve |
| `yolo12l_wtbd_group_f50` | WTBD-group, 50% of train scenes | YOLO12-L | 150 | 0.279 / 0.167 (epoch 120) | **0.244 / 0.105**; TTA 0.259 / 0.125 | Learning curve |
| `bladeyolo2_wtbd_group_f25_sslsyn` | WTBD-group 25% + view pool | BladeYOLO-v2, synthetic-view teacher–student | 150 | 0.427 / 0.203 (epoch 142) | **0.221 / 0.071**; TTA 0.233 / 0.089 | Phase 22 |
| `bladeyolo2_wtbd_group_f25_sslreal` | WTBD-group 25% + view pool | BladeYOLO-v2, real-view teacher–student | 150 | 0.450 / 0.210 (epoch 107) | **0.213 / 0.085**; TTA 0.268 / 0.107 | Phase 22; teacher collapsed |
| `bladeyolo2_wtbd_group_f25_sslsyn2` | WTBD-group 25% + view pool | BladeYOLO-v2, synthetic views (fixed) | 150 | 0.448 / 0.220 (epoch 122) | **0.224 / 0.078**; TTA 0.259 / 0.106 | Phase 23 |
| `bladeyolo2_wtbd_group_f25_sslreal2` | WTBD-group 25% + view pool | BladeYOLO-v2, real views (fixed) | 150 | 0.382 / 0.184 (epoch 127) | **0.192 / 0.077**; TTA 0.248 / 0.101 | Phase 23 |
| `bladeyolo2_wtbd_group_f25_sslin` | WTBD-group 25% + other 75% unlabelled | BladeYOLO-v2, in-domain teacher–student | 150 | 0.457 / 0.227 (epoch 131) | **0.209 / 0.080**; TTA 0.245 / 0.101 | Phase 23 |
| `bladeyolo2_wtbd_group_f25_s1` | WTBD-group, 25% of train scenes | BladeYOLO-v2, seed 1 | 150 | 0.411 / 0.199 (epoch 129) | **0.224 / 0.087**; TTA 0.242 / 0.105 | Phase 24 |
| `bladeyolo2_wtbd_group_f25_s2` | WTBD-group, 25% of train scenes | BladeYOLO-v2, seed 2 | 150 | 0.435 / 0.203 (epoch 143) | **0.184 / 0.068**; TTA 0.229 / 0.096 | Phase 24 |
| `bladeyolo2_wtbd_group_f25_sslinb_s0` | WTBD-group 25% + other 75% unlabelled | BladeYOLO-v2, in-domain, class-balanced, seed 0 | 150 | 0.344 / 0.154 (epoch 75) | **0.143 / 0.060**; TTA 0.183 / 0.079 | Phase 24 |
| `bladeyolo2_wtbd_group_f25_sslinb_s1` | WTBD-group 25% + other 75% unlabelled | same, seed 1 | 150 | 0.352 / 0.167 (epoch 92) | **0.163 / 0.065**; TTA 0.180 / 0.075 | Phase 24; pseudo-label runaway, epochs 33–62 |
| `bladeyolo2_wtbd_group_f25_sslinb_s2` | WTBD-group 25% + other 75% unlabelled | same, seed 2 | 150 | 0.445 / 0.205 (epoch 148) | **0.183 / 0.073**; TTA 0.213 / 0.095 | Phase 24 |
| `bladeyolo2nodino_wtbd_group_f25_s0` | WTBD-group, 25% of train scenes | BladeYOLO-v2 without DINOv3, seed 0 | 150 | 0.229 / 0.126 (epoch 106) | **0.114 / 0.044**; TTA 0.153 / 0.063 | Phase 25 |
| `bladeyolo2nodino_wtbd_group_f25_s1` | WTBD-group, 25% of train scenes | BladeYOLO-v2 without DINOv3, seed 1 | 150 | 0.186 / 0.105 (epoch 89) | **0.081 / 0.031**; TTA 0.126 / 0.053 | Phase 25 |
| `bladeyolo2_wtbd_group_s1` | WTBD-group (leak-free) | BladeYOLO-v2, seed 1 | 150 | 0.630 / 0.370 (epoch 128) | **0.554 / 0.245**; TTA 0.559 / 0.269 | Phase 26; resumed at epoch 78 |
| `bladeyolo2_wtbd_group_tolsel_s0` | WTBD-group (leak-free) | BladeYOLO-v2, tolerance for corrosion + hide_craze, seed 0 | 150 | 0.656 / 0.381 (epoch 140) | **0.567 / 0.245**; TTA 0.568 / 0.258 | Phase 26 |
| `bladeyolo2_wtbd_group_tolsel_s1` | WTBD-group (leak-free) | same, seed 1 | 150 | 0.704 / 0.380 (epoch 119) | **0.561 / 0.245**; TTA 0.555 / 0.264 | Phase 26 |
| `bladeyolo2b_wtbd_group_f25_s0` | WTBD-group, 25% of train scenes | BladeYOLO-v2, frozen DINOv3 ViT-B/16, seed 0 | 150 | 0.383 / 0.176 (epoch 127) | **0.252 / 0.102**; TTA 0.313 / 0.127 | Phase 27 |
| `bladeyolo2b_wtbd_group_f25_s1` | WTBD-group, 25% of train scenes | same, seed 1 | 150 | 0.416 / 0.217 (epoch 146) | **0.283 / 0.116**; TTA 0.272 / 0.132 | Phase 27 |
| `bladeyolo2l_wtbd_group_f25_s0` | WTBD-group, 25% of train scenes | BladeYOLO-v2, frozen DINOv3 ViT-L/16, seed 0 | 150 | 0.438 / 0.219 (epoch 107) | **0.337 / 0.115**; TTA 0.357 / 0.153 | Phase 27 |
| `bladeyolo2l_wtbd_group_f25_s1` | WTBD-group, 25% of train scenes | same, seed 1 | 150 | 0.484 / 0.235 (epoch 81) | **0.301 / 0.109**; TTA 0.348 / 0.146 | Phase 27 |
| `bladeyolo2nodino_wtbd_group_f50` | WTBD-group, 50% of train scenes | BladeYOLO-v2 without DINOv3 | 150 | 0.293 / 0.166 (epoch 105) | **0.225 / 0.093**; TTA 0.244 / 0.111 | Phase 27 |
| `bladeyolo2nodino_wtbd_group` | WTBD-group (leak-free) | BladeYOLO-v2 without DINOv3 | 150 | 0.404 / 0.247 (epoch 108) | **0.361 / 0.170**; TTA 0.385 / 0.210 | Phase 27 |
| **`yolo12l_wtbd_group`** | WTBD-group (leak-free) | YOLO12-L | 150 | 0.419 / 0.269 (epoch 130) | **0.438 / 0.206**; TTA 0.440 / 0.230 | Same recipe |
| **`yolo12l_wtbd_official`** | WTBD official (**leaky**) | YOLO12-L | 120 | 0.665 / 0.394 (epoch 104) | **0.704 / 0.388**; TTA 0.734 / 0.438 | Same recipe |

Val best is the best epoch in `results.csv` by fitness (0.1 × mAP50 + 0.9 × mAP50-95); the final validation of `best.pt` can differ slightly. Since Phase 19 the retired runs (everything before WTBD) keep only `results.csv`, `args.yaml` and test metrics, in `runs/analysis/archive/run_<name>/`; their weights were deleted.

---

## 28. Datasets and artefacts

| Item | Location | Notes |
|---|---|---|
| WTBD raw | `external/WTBD/` | figshare 10.6084/m9.figshare.30210175 |
| WTBD-group / WTBD-official | `external/WTBD-group/`, `external/WTBD-official/`; Kaggle `beegee11/wind-surface-defect/WTBD-yolo` | `tools/convert_wtbd.py` |
| WTBD group split, scene fractions | `tools/splits/` (committed) | Zero-leak audited split; nested 25% / 50% training subsets |
| Other raw datasets | `WindSurface-Defect/`, `WTBlade-Defect/`, `wind-turbine.v1i.yolov12/`, `wind_turbine_check.v1i.yolov12/`, `WTBs2025/` | Official classes only |
| Current runs | `runs/detect/BladeYOLO_WindSurface/` | `bladeyolo2_wtbd_{group,official,group_tolerant}`, `yolo12l_wtbd_{group,official}`, learning-curve, seed and SSL runs at 25%, DINOv3 size (`bladeyolo2{b,l}_…`) and without-DINOv3 (`bladeyolo2nodino_…`) runs |
| SAM refinement | `runs/analysis/sam_refine/` | SAM 2.1-tiny weights, cached predictions and SAM boxes, `val_grid.csv`, `report.md` |
| Kaggle logs | `runs/kaggle_logs/` | One per `pull_kaggle_runs.py` pull |
| DINOv3 weights | `external/dinov3/vit{s,b,l}16/` (Hugging Face, gated), `external/dinov3/official/` (Meta ViT-S .pth) | DINOv3 License (Meta, 19 Aug 2025; `external/dinov3/vits16/LICENSE.md`): use, modification and redistribution allowed, but anything containing the weights (including our trained checkpoints) must ship with the license; publications must acknowledge DINOv3; military/ITAR uses prohibited |
| Retired runs and the v3 label review | `runs/analysis/archive/` | `results.csv`, `args.yaml`, test metrics; `review_test_v3.json` |
| Dataset audit (Phase 13) | `runs/analysis/datasets/report.md` | |
| WTBD leakage and craze diagnosis | `runs/analysis/{wtbd, wtbd_leak, wtbd_craze}/` | |
| Scene bootstrap | `runs/analysis/bootstrap/` | Reports, scene groups, cached predictions |
| Evaluation experiments | `runs/analysis/novel/` | `granularity/`, `label_audit/{wtbd,windsurface,wtblade}/`, `conformal/`, cached predictions |
| NaN replay | `runs/analysis/nan_probe/` | Layer-6 replay of the tolerant run's NaN batches |
| Caches | `.dataset_cache/` | DINOv2 embeddings, ORB verifications |

**Tools:**
- matching and splits: `matching.py`, `wtbd_groups.py`, `convert_wtbd.py`, `scene_fractions.py`, `view_pool.py`
- box refinement: `sam_refine.py`
- Kaggle: `pull_kaggle_runs.py`
- evaluation: `detection_metrics.py`, `bootstrap_ci.py`, `leak_subsets.py`, `granularity_eval.py`, `view_label_audit.py`, `conformal_recall.py`
- debugging: `probe_nan.py`

Everything retired is restorable from commit `b374f4c`.

**Tests** (`tests/`): `test_ss2d`, `test_wiou`, `test_model`, `test_augment`, `test_fp32_blocks` (incl. fp16 attention), `test_deform_offsets`, `test_granularity_eval`, `test_view_label_audit`, `test_conformal_recall`, `test_tolerant`, `test_view_pool`, `test_view_consistency`, `test_sam_refine`, `test_dino_sizes`, `test_dino_lora`. All pass in `.venv`.

---

## 29. Open work and next steps

**A. Real-view consistency for data-limited detection (Phase 19)**
1. **Learning curves: done (§17).** Steep and significant for both detectors; BladeYOLO-v2 is the more data-efficient base model. Commands used:
   ```
   python train.py --data .../WTBD-group/data.yaml --model bladeyolo2-l.yaml --train-subset tools/splits/wtbd_group_train_f25.txt --name bladeyolo2_wtbd_group_f25 --epochs 150
   python train.py --data .../WTBD-group/data.yaml --model bladeyolo2-l.yaml --train-subset tools/splits/wtbd_group_train_f50.txt --name bladeyolo2_wtbd_group_f50 --epochs 150
   python train.py --data .../WTBD-group/data.yaml --model yolo12l.pt --box-loss ciou --train-subset tools/splits/wtbd_group_train_f25.txt --name yolo12l_wtbd_group_f25 --epochs 150
   python train.py --data .../WTBD-group/data.yaml --model yolo12l.pt --box-loss ciou --train-subset tools/splits/wtbd_group_train_f50.txt --name yolo12l_wtbd_group_f50 --epochs 150
   ```
2. **Registered-view pair index: done (§18),** with real views identified by source photo, not photometric similarity:
   - verified view pairs within the training images, from the label audit's transforms
   - an unlabelled pool: WTBlade and Wind Surface frames, and DTU NordTank raw imagery after a licence check (CC BY-NC, fine for research)
   - no test or validation scene, and no image matching one, enters the pool (checked with `tools/matching.py`)
3. **Teacher–student training: implemented and tested (§18):**
   - an EMA teacher, with pseudo-labels transferred through the verified transform and confidence filtering
   - the synthetic-augmentation variant as the baseline
   - unit tests for the box transfer and the EMA
4. **Runs:** supervised vs synthetic teacher–student vs real-view teacher–student, with BladeYOLO-v2 as the base model, at 25 / 50 / 100% of the labelled scenes on WTBD-group (2 seeds), then on a WTBlade group split. Scored with the scene bootstrap and the conformal operating curve. **First: 25% of the labelled scenes** (supervised run exists: `bladeyolo2_wtbd_group_f25`):
   ```
   python train.py --data .../WTBD-group/data.yaml --model bladeyolo2-l.yaml --train-subset tools/splits/wtbd_group_train_f25.txt --ssl synthetic --ssl-pairs .../view-pool/pairs.csv --name bladeyolo2_wtbd_group_f25_sslsyn --epochs 150
   python train.py --data .../WTBD-group/data.yaml --model bladeyolo2-l.yaml --train-subset tools/splits/wtbd_group_train_f25.txt --ssl real --ssl-pairs .../view-pool/pairs.csv --name bladeyolo2_wtbd_group_f25_sslreal --epochs 150
   ```
   - **The unlabelled pool needs care.** WTBD has no unlabelled images, so the pool comes from the other datasets used without their labels. Beijing shares 184 images with WTBD, so anything matching a WTBD val or test scene is excluded.
   - **Real views vs augmented copies:** most WTBlade "same-surface" pairs are Roboflow augmented copies (synthetic). Real second views are video frames and overlapping photos, and must be told apart (e.g. by the photometric residual after alignment), or the method silently becomes synthetic consistency.

**A3. Foundation-feature prior (§22)**
1. **Done (§24):** BladeYOLO-v2 without DINOv3 at 50% and 100%:
   ```
   python train.py --data .../WTBD-group/data.yaml --model bladeyolo2-l-nodino.yaml --train-subset tools/splits/wtbd_group_train_f50.txt --name bladeyolo2nodino_wtbd_group_f50 --epochs 150
   python train.py --data .../WTBD-group/data.yaml --model bladeyolo2-l-nodino.yaml --name bladeyolo2nodino_wtbd_group --epochs 150
   ```
2. **Done (§24):** larger frozen DINOv3 at 25%, two seeds each (weights: the Kaggle models `beegee11/vit{s,b,l}16`):
   ```
   python train.py --data .../WTBD-group/data.yaml --model bladeyolo2-l.yaml --dino b --train-subset tools/splits/wtbd_group_train_f25.txt --seed 0 --name bladeyolo2b_wtbd_group_f25_s0 --epochs 150
   python train.py --data .../WTBD-group/data.yaml --model bladeyolo2-l.yaml --dino b --train-subset tools/splits/wtbd_group_train_f25.txt --seed 1 --name bladeyolo2b_wtbd_group_f25_s1 --epochs 150
   python train.py --data .../WTBD-group/data.yaml --model bladeyolo2-l.yaml --dino l --train-subset tools/splits/wtbd_group_train_f25.txt --seed 0 --name bladeyolo2l_wtbd_group_f25_s0 --epochs 150
   python train.py --data .../WTBD-group/data.yaml --model bladeyolo2-l.yaml --dino l --train-subset tools/splits/wtbd_group_train_f25.txt --seed 1 --name bladeyolo2l_wtbd_group_f25_s1 --epochs 150
   ```
   Ran in two Kaggle sessions (seed 0 of B and L plus no DINOv3 at 50%, then seed 1 plus no DINOv3 at 100%).
3. **Done (§24):** ViT-B beats ViT-S at 25% (+0.028 mAP50-95), ViT-L is no better than ViT-B; the DINOv3 gain holds at 50% and 100%; without DINOv3 the architecture equals or trails YOLO12-L. Analysis: `.venv/bin/python runs/analysis/bootstrap/robust_check_scale.py [curve]`.
4. **Dropped 22 Oct (not novel; see §24).** ~~Stage 1 of the programme (§24, decided 21 Oct):~~ ViT-B at 25%, LoRA and native resolution, two seeds each:
   ```
   python train.py --data .../WTBD-group/data.yaml --model bladeyolo2-l.yaml --dino b --dino-lora 16 --train-subset tools/splits/wtbd_group_train_f25.txt --seed 0 --name bladeyolo2b_lora16_wtbd_group_f25_s0 --epochs 150
   python train.py --data .../WTBD-group/data.yaml --model bladeyolo2-l.yaml --dino b --imgsz 1024 --batch 8 --train-subset tools/splits/wtbd_group_train_f25.txt --seed 0 --name bladeyolo2b_1024_wtbd_group_f25_s0 --epochs 150
   (same with --seed 1 and names …_s1)
   ```
   Then Stage 2 (label-free real-view adaptation) if Stage 1 shows the prior is adaptable. The simplified model (YOLO12-L + DINOv3 fusion only) and ViT-L at 100% follow once the method is fixed.

**A2. Current round (§21): box extent and foundation features**
1. **Done, negative (§21):** SAM 2.1 box refinement on the three supervised 25% seeds and the 100% model (local, no training):
   ```
   .venv/bin/python tools/sam_refine.py --runs runs/detect/BladeYOLO_WindSurface/bladeyolo2_wtbd_group_f25 runs/detect/BladeYOLO_WindSurface/bladeyolo2_wtbd_group_f25_s1 runs/detect/BladeYOLO_WindSurface/bladeyolo2_wtbd_group_f25_s2 runs/detect/BladeYOLO_WindSurface/bladeyolo2_wtbd_group
   ```
   Near-misses are granularity (part/merged) and misalignment on thin classes, not loose boxes; the training-time SAM variant is not pursued.
2. DINOv3 ablation at 25%, two seeds (Kaggle):
   ```
   python train.py --data .../WTBD-group/data.yaml --model bladeyolo2-l-nodino.yaml --train-subset tools/splits/wtbd_group_train_f25.txt --seed 0 --name bladeyolo2nodino_wtbd_group_f25_s0 --epochs 150
   python train.py --data .../WTBD-group/data.yaml --model bladeyolo2-l-nodino.yaml --train-subset tools/splits/wtbd_group_train_f25.txt --seed 1 --name bladeyolo2nodino_wtbd_group_f25_s1 --epochs 150
   ```
   If DINOv3 carries the low-data advantage: larger DINOv3 backbones or light adapters as the data-efficiency lever.
4. **Done (§23):** class-selective split tolerance at 100%: corrosion mAP50-95 +0.093 [−0.003, +0.246], aggregate unchanged. Commands used:
   ```
   python train.py --data .../WTBD-group/data.yaml --model bladeyolo2-l.yaml --tolerant-classes corrosion,hide_craze --seed 0 --name bladeyolo2_wtbd_group_tolsel_s0 --epochs 150
   python train.py --data .../WTBD-group/data.yaml --model bladeyolo2-l.yaml --tolerant-classes corrosion,hide_craze --seed 1 --name bladeyolo2_wtbd_group_tolsel_s1 --epochs 150
   python train.py --data .../WTBD-group/data.yaml --model bladeyolo2-l.yaml --seed 1 --name bladeyolo2_wtbd_group_s1 --epochs 150
   ```
3. **Done (§21):** stricter near-miss diagnostic, per class. Open: the same at 50% of the scenes, and whether craze/crack scores are limited by the axis-aligned box format (e.g. region-level or granularity-tolerant evaluation for those classes only, §14).

**B. Carried over**
4c. **Semi-supervised round (§20–§21, done; semi-supervision dropped):** in-domain + class-balanced, three seeds per arm:
   ```
   python train.py --data .../WTBD-group/data.yaml --model bladeyolo2-l.yaml --train-subset tools/splits/wtbd_group_train_f25.txt --seed 1 --name bladeyolo2_wtbd_group_f25_s1 --epochs 150
   python train.py --data .../WTBD-group/data.yaml --model bladeyolo2-l.yaml --train-subset tools/splits/wtbd_group_train_f25.txt --seed 2 --name bladeyolo2_wtbd_group_f25_s2 --epochs 150
   python train.py --data .../WTBD-group/data.yaml --model bladeyolo2-l.yaml --train-subset tools/splits/wtbd_group_train_f25.txt --ssl synthetic --ssl-unlabelled tools/splits/wtbd_group_train_f25_rest.txt --ssl-balance --seed 0 --name bladeyolo2_wtbd_group_f25_sslinb_s0 --epochs 150
   (same with --seed 1 / --seed 2 and names …_sslinb_s1 / …_sslinb_s2)
   ```
4b. **Previous round (§19–§20, done):** A + B at 25%:
   ```
   python train.py --data .../WTBD-group/data.yaml --model bladeyolo2-l.yaml --train-subset tools/splits/wtbd_group_train_f25.txt --ssl synthetic --ssl-pairs .../view-pool/pairs.csv --name bladeyolo2_wtbd_group_f25_sslsyn2 --epochs 150
   python train.py --data .../WTBD-group/data.yaml --model bladeyolo2-l.yaml --train-subset tools/splits/wtbd_group_train_f25.txt --ssl real --ssl-pairs .../view-pool/pairs.csv --name bladeyolo2_wtbd_group_f25_sslreal2 --epochs 150
   python train.py --data .../WTBD-group/data.yaml --model bladeyolo2-l.yaml --train-subset tools/splits/wtbd_group_train_f25.txt --ssl synthetic --ssl-unlabelled tools/splits/wtbd_group_train_f25_rest.txt --name bladeyolo2_wtbd_group_f25_sslin --epochs 150
   ```
5. **Class-selective split tolerance:** moved to A2 item 4 (corrosion and hide_craze, chosen from the label audit and val).
6. **Seeds** for the existing WTBD-group pair, the **loss vs architecture separation** (`bladeyolo2-l.yaml --box-loss ciou`), and a **component ablation** (without DINOv3 fusion) to test whether the low-data advantage comes from the DINOv3 features.
7. **WTBlade group split** with its official classes, for a second benchmark. Licence: it contains DTU NordTank (CC BY-NC) and synthetic copy-paste images (§14).

## 30. Working conventions

- **Classes:** datasets are used with their official classes only; the custom v3 taxonomy (Phase 9) is retired.
- **Testing:** tests and analyses run in the project's own `.venv`, with outputs under `runs/` and never `/tmp` ("research projects can have their own testing environments").
- **NaN handling:** fix the root cause. Batches are only dropped by the guard as a last line of defence, logged and reported.
- **Selection and reporting:** model selection on val, results reported on test. Never tune on test.
- **Git:** push only on request. Commits end with "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>". Datasets, weights, zips and `runs/` are gitignored.
- **Kaggle:** run long jobs with **Save Version → Save & Run All (Commit)** so `/kaggle/working` survives. Download run folders into `runs/detect/BladeYOLO_WindSurface/`.
