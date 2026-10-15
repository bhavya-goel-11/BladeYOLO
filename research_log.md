# BladeYOLO Research Log

**Wind-turbine blade surface defect detection: from a paper reproduction to a leak-free benchmark**

| | |
|---|---|
| Researcher | Bhavya Goel |
| Period | 3 Sep 2026 – 14 Oct 2026 |
| Compute | Kaggle 2× Tesla T4 (all training since 9 Oct); A100 for Runs 5–7; local CPU machine (analysis only) |
| Repository | github.com/bhavya-goel-11/BladeYOLO (`master`) |
| Reference papers | BladeYOLO (IEEE TGRS 2026, `BladeYOLO.pdf`); GFD-YOLOX / Wind Surface Defect dataset (Appl. Soft Comput. 2025, `1-s2.0-S156849462501227X-main.pdf`); WTBD (Ji et al., *Scientific Data* 13:396, 2026, `s41597-026-06762-x.pdf`); WTBs2025 (*Scientific Data* 13:1310, 2026) |

This log covers the whole project. Phases 1–5 (Sep 3 – Oct 6) summarise the original tentative research log (commit `6f6638a`, 6 Oct). That commit was removed from `master` by the 9 Oct reset but is still in the local git object store: `git show 6f6638a:research_log.md`. Later phases come from the working sessions, the git history and the run folders under `runs/detect/BladeYOLO_WindSurface/`. Where a later audit corrected an earlier conclusion, the correction appears in that phase and in the corrections section (§16).

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
15. [Engineering problems solved](#15-engineering-problems-solved)
16. [Corrections to earlier conclusions](#16-corrections-to-earlier-conclusions)
17. [Run registry](#17-run-registry)
18. [Datasets and artefacts](#18-datasets-and-artefacts)
19. [Open work and next steps](#19-open-work-and-next-steps)
20. [Working conventions](#20-working-conventions)

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

The project's main contribution is now methodological: a reproducible, leakage-audited protocol for blade defect detection, measured evidence of how much random splits of augmented data inflate published numbers, granularity-aware evaluation, a cross-view label audit, and the finding that leaked calibration breaks recall guarantees. The architecture is BladeYOLO-v2. Against stock YOLO12-L it wins on both splits, but on the leak-free split the win rests largely on one class with 3 test scenes (thunderstrike), so an architectural advantage is suggested, not demonstrated (§12).

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
- **More independent test scenes are needed** before per-class or architectural claims can be made with confidence. Options are group k-fold cross-validation or a second dataset (§19).

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

## 15. Engineering problems solved

| Problem | Root cause | Fix | Commit |
|---|---|---|---|
| WIoU never active | Patched the wrong namespace | Patch the loss namespace | `dfdb5f4` |
| NaN in SS2D (epoch 9) | fp16 overflow before the LayerNorm | fp32 LayerNorm | `2332cf4`, `4cf2c85` |
| NaN epoch 4 (v2) | Poisoned WIoU global + aggressive warmup | Finite-only running mean; lr0 0.001, `warmup_bias_lr` 0 | `2efd069` |
| NaN epoch 17 (v2) | fp16 overflow, layer 26 conv into BN | `FP32Block` for the refinement blocks | `08f9a4b` |
| NaN epoch 99 (WTBD) leaving val at 0 | NaN batch poisons BN running stats + EMA | Per-batch BN repair + step skip + per-layer report | `91ae01f` |
| Test evaluation silently missing (twice) | `deform_conv2d` segfault on NaN offsets from the FLOPs probe | Finite, clamped offsets; test in a fresh process | `b8721bc` |
| Strict pretrained loading mismatch | YOLOv12 checkpoint carries `attn.pe.conv.bias` | Fold the bias exactly into BN `running_mean` | `e77dc0d` |
| Ultralytics label cache on a read-only Kaggle input | Input mount not writable | Copy the dataset into the project folder once | `1d58ea2` |
| Background processes, `!cmd &` on Kaggle | Notebook limitation | `subprocess.Popen` / Save & Run All | — |
| Windows MX230 workstation | Pascal GPU, CUDA 12.5 driver | torch 2.6 + cu124; inference/analysis only | `091cb8a` |

---

## 16. Corrections to earlier conclusions

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
| Honest calibration costs 2.4 instead of 1.0 false alarms per image (Phase 16) | **Corrected** | That column counted all predictions; real false alarms are 0.2 vs 0.1 per image (§14) |
| WTBD copies agree on 86.0% of labels (Phase 16) | **Refined** | Rotated copies inflate "loose"; axis-aligned copies agree on 87.3% (§14) |
| WTBlade is CC BY 4.0 (Roboflow page) | **Questioned** | It contains DTU NordTank images (CC BY-NC) and synthetic copy-paste images (§14) |
| Part of the leakage inflation is memorised box splitting (Phase 16 hypothesis) | **Not supported** | The tolerant gain is not smaller on test images with a train twin (§14, step 14) |

---

## 17. Run registry

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
| **`yolo12l_wtbd_group`** | WTBD-group (leak-free) | YOLO12-L | 150 | 0.419 / 0.269 (epoch 130) | **0.438 / 0.206**; TTA 0.440 / 0.230 | Same recipe |
| **`yolo12l_wtbd_official`** | WTBD official (**leaky**) | YOLO12-L | 120 | 0.665 / 0.394 (epoch 104) | **0.704 / 0.388**; TTA 0.734 / 0.438 | Same recipe |

Val best is the best epoch in `results.csv` by fitness (0.1 × mAP50 + 0.9 × mAP50-95); the final validation of `best.pt` can differ slightly.

---

## 18. Datasets and artefacts

| Item | Location | Notes |
|---|---|---|
| WindSurface-Defect (original) | `WindSurface-Defect/` | Leaky random split |
| v2 / v3 / v3.1 / v3.2 | `WindSurface-Defect-v*/` | v3.1 = test review applied; v3.2 = region-merged labels |
| WTBD raw | `external/WTBD/` | figshare download |
| WTBD-group / WTBD-official | `external/WTBD-group/`, `external/WTBD-official/`; Kaggle `beegee11/wind-surface-defect/WTBD-yolo` | Built by `tools/convert_wtbd.py` |
| WTBD group split | `tools/splits/wtbd_group_split.csv` (committed) | Zero-leak audited |
| WTBs2025-clean | `WTBs2025-clean/` | Native 7-class audit set |
| Dataset audit | `runs/analysis/datasets/report.md` | Scenes, purity, cross-dataset equivalence |
| Error analysis (v3) | `runs/analysis/errors/report.md` | |
| Label review | `runs/analysis/review/` | `review_test.json` |
| WTBD craze diagnosis | `runs/analysis/wtbd_craze/` | Contact sheets of test vs train craze |
| WTBD leak subsets | `runs/analysis/wtbd_leak/` | `test_leak.csv`, subset metrics |
| Scene bootstrap | `runs/analysis/bootstrap/` | Reports per split, scene groups, cached predictions, class-subset check |
| Evaluation experiments | `runs/analysis/novel/` | `granularity/`, `label_audit/wtbd/` (with example sheets), `conformal/`, cached val/train predictions |
| Caches | `.dataset_cache/` | DINOv2 embeddings (38,324 images), ORB verifications, crop embeddings |

**Tools:**
- dataset building and auditing: `build_dataset.py`, `check_overlap.py`, `audit_datasets.py`, `wtbd_groups.py`, `convert_wtbd.py`, `leak_subsets.py`, `bootstrap_ci.py`, `view_label_audit.py`
- labels and evaluation: `error_analysis.py`, `review_labels.py`, `apply_review.py`, `merge_regions.py`, `ensemble.py`, `class_crops.py`, `granularity_eval.py`, `conformal_recall.py`
- debugging and setup: `probe_nan.py`, `setup_windows.ps1` (`diagnose_nan.py` and `eval_by_source.py` were removed on 14 Oct; they are in the git history)

**Tests** (`tests/`): `test_ss2d`, `test_wiou`, `test_model`, `test_augment`, `test_fp32_blocks`, `test_merge_regions`, `test_deform_offsets`, `test_granularity_eval`, `test_view_label_audit`, `test_conformal_recall`, `test_tolerant`. All pass in `.venv`.

---

## 19. Open work and next steps

Ordered by how much each step strengthens the main claims, with rough Kaggle cost (2×T4; one WTBD run ≈ 2 h).

**A. Make the architecture claim decidable**
1. **Group k-fold cross-validation on WTBD.** Use 5 folds of whole scene groups, each fold's test about 70 scenes, so every one of the ~330–368 scenes is tested once. Pooling all folds gives about 6× more test scenes than the single split, and craze and thunderstrike get enough scenes to judge. Cost: 5 folds × 2 models × ~2 h ≈ 20 h, two to three sessions. Needs a fold option in `tools/wtbd_groups.py`.
2. **Separate the loss from the architecture.** BladeYOLO-v2 trained with WIoU and YOLO12-L with CIoU, so they differ in both. Run `bladeyolo2-l.yaml --box-loss ciou` on WTBD-group (~2 h). This also tests whether WIoU explains YOLO12-L's lead on corrosion.
3. **Component ablation.** Train BladeYOLO-v2 without DINOv3 fusion, and separately without the deformable/selective-scan refinements, to see which branch carries the thunderstrike gain. That is 2 runs (~4 h) and needs two small yaml variants.
4. **Seeds.** Repeat the group-split pair with `--seed 1` and `--seed 2` (~8 h) to measure run-to-run variance next to the scene variance.

**B. A second, larger leak-free benchmark**
5. **WTBlade / fengChe with its own classes.** It has 803 independent scenes (WTBD-group test has 58) and is CC BY 4.0. Its 7 native classes all have 80–425 scenes. Build a group split with the existing tools: `build_dataset.py` matching and `wtbd_groups.py`-style auditing. Running both models on it (~2 × 2–3 h) would give much tighter intervals and a second dataset for the claims.
6. **Cross-dataset check, no training.** Run the WTBD models class-agnostically ("is there damage here?") on fengChe, which shares only 3 images with WTBD. This tests generalisation to another source.

**C. Improve the model where it is weak**
7. **Box tightness:** train at 832 or 1024 px (WTBD is natively 1024 px). Corrosion mAP50-95 (0.13) and crack are the weakest. Cost ~3–4 h per run at 1024 with a smaller batch.
8. **Elongated defects:** the craze test scene is labelled as 2–3 segment boxes and predicted as one. Score with a region-merged convention (`tools/merge_regions.py`) as a diagnostic, and report it separately.
9. **Optional pretraining** on fengChe + WTBs2025, then fine-tuning on WTBD, excluding Beijing, which contains WTBD test photos. Only worth it if cross-validation shows overfitting.

**D. Follow-ups to the evaluation experiments (Phase 16), status after Phase 17**
10. **Split-tolerant training** (`--tolerant`): implemented and tested; **Kaggle run pending** (§14).
11. **Reward fine-tuning (RL):** deferred until step 10's result (§14).
12. **Scene-level and class-aware conformal calibration, operating curve:** done (§14).
13. **Label audit on other datasets:** done. Axis-aligned agreement: WTBD 87.3%, WTBlade 86.0%, Wind Surface 83.8%. WTBlade contains synthetic copy-paste images and DTU NordTank imagery (§14).
14. **Memorised box splitting:** tested, not supported (§14).

**E. Write-up**
15. **Results summary for the professor.** Cover the reproduction gap, the leakage evidence on two datasets, the leak-free protocol, the 2×2 table with intervals, the honest architecture claim, and the Phase 16 evaluation findings. Optionally send the WTBD authors the group-split CSV.

## 20. Working conventions

- **Classes:** datasets are used with their official classes only; the custom v3 taxonomy (Phase 9) is retired.
- **Testing:** tests and analyses run in the project's own `.venv`, with outputs under `runs/` and never `/tmp` ("research projects can have their own testing environments").
- **NaN handling:** fix the root cause. Batches are only dropped by the guard as a last line of defence, logged and reported.
- **Selection and reporting:** model selection on val, results reported on test. Never tune on test.
- **Git:** push only on request. Commits end with "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>". Datasets, weights, zips and `runs/` are gitignored.
- **Kaggle:** run long jobs with **Save Version → Save & Run All (Commit)** so `/kaggle/working` survives. Download run folders into `runs/detect/BladeYOLO_WindSurface/`.
