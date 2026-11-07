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

### H5 result: passes all pre-registered conditions; the val effect is small

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
- **Val condition, checked** (`runs/analysis/h5_val.sh` → `runs/analysis/novel/granularity/h5_reward_last_val/report.md`; val scene groups built with the same matcher, `runs/analysis/bootstrap/h5_last_val/groups.csv`; 500 replicates): region AP reward 0.435 / 0.463 vs baseline 0.411 / 0.451, **+0.018 [−0.056, +0.071], P(>0) 0.57**. Both seeds are positive, so the direction agrees. Standard AP50 +0.031 [−0.006, +0.080], standard AP50-95 −0.000, tolerant AP50-95 −0.010.
- **Verdict: H5 passes all three pre-registered conditions** (test region AP ≥ +0.02 with P ≥ 0.95; val same direction; standard AP50-95 not below −0.01). **But the effect is fragile:** the val gain is a quarter of the test gain and its interval includes 0, so +0.081 is likely an optimistic estimate from 58 test scenes. Before relying on it: more seeds, and a replication on WTBlade-group.
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

### H6 robustness result: holds against random, does not improve on FreeSel (24 Oct)

Runs (Kaggle, ViT-B, 150 epochs, training seed 0): `bladeyolo2b_rand{3..8}_wtbd_group_f25_s0`, `bladeyolo2b_coverk{1,2}_wtbd_group_f25_s0`, `bladeyolo2b_freesel{224,640}s{0,1}_wtbd_group_f25`. With the earlier cover s0/s1, rand1/2 and class-matched runs, 18 runs at the same 25% budget. Paired scene bootstrap as pre-registered: `runs/analysis/bootstrap/robust_check_scale.py h6r` → `h6r.log` (1,000 replicates over the 58 test scenes; an arm is the mean of its runs). Val = best-epoch mAP50-95 from `results.csv`.
- The `freesel640s0` weights arrived as `best.zip` / `last.zip`: ordinary PyTorch checkpoints (a `.pt` is a zip archive) under the wrong extension, renamed to `.pt`. The first bootstrap attempt was killed by memory pressure from other applications; predictions are cached per run, so the reruns resumed.

| arm (runs) | test mAP50 | test mAP50-95 | per-run test mAP50-95 | val mAP50-95 | rank among 8 random (test / val) |
|---|---|---|---|---|---|
| cover family (4) | 0.406 | **0.177** | 0.189 0.174 0.173 0.171 | 0.236 | 2/9 · 3/9 |
| FreeSel-224 (2) | 0.429 | 0.156 | 0.156 0.156 | **0.271** | 2/9 · 1/9 |
| FreeSel-640 (2) | 0.344 | 0.145 | 0.144 0.145 | 0.266 | 2/9 · 2/9 |
| random (8) | 0.313 | 0.125 | 0.112 0.135 0.118 0.123 0.090 0.100 0.124 0.196 | 0.216 | — |
| class-matched (2) | 0.253 | 0.104 | 0.099 0.108 | 0.197 | 7/9 · 7/9 |

Paired differences, test mAP50-95 (all 6 classes):

| comparison | Δ [95% CI] | P(>0) | val Δ |
|---|---|---|---|
| cover family − random | **+0.052 [+0.019, +0.105]** | **0.998** | +0.020 |
| cover family − FreeSel-224 | +0.021 [−0.012, +0.091] | 0.917 | −0.035 |
| cover family − FreeSel-640 | +0.032 [−0.007, +0.070] | 0.957 | −0.030 |
| FreeSel-224 − random | +0.031 [−0.013, +0.071] | 0.900 | +0.055 |
| FreeSel-640 − random | +0.020 [−0.000, +0.088] | 0.972 | +0.050 |
| cover family − class-matched | +0.073 [+0.006, +0.170] | 0.986 | +0.039 |

On the 4 robust classes (craze and thunderstrike excluded, as in earlier phases): cover − random +0.029 [−0.005, +0.073], P 0.937; cover − FreeSel-224 −0.001, P 0.44 (mAP50 −0.058, P 0.04); FreeSel-224 − random +0.030 [+0.012, +0.072], P 1.000.

- **Against random: H6 holds** (pre-registered bar met: +0.052 ≥ +0.02, P 0.998, val higher). Stable across k-means seeds: all 4 cover subsets score 0.171–0.189, above 7 of the 8 random draws. The first H6 result was not subset luck.
- **Against FreeSel: H6 fails.** Against FreeSel-224 (the pre-registered baseline), test +0.021 with P 0.917 < 0.95 and val in the opposite direction (−0.035). Against FreeSel-640, P 0.957 on test but val again opposite. Test favours cover and val favours FreeSel; on the 4 robust classes they are equal on mAP50-95, and FreeSel is ahead on mAP50. Cover's test edge comes from thunderstrike (AP50-95 0.315 vs 0.209) and crack (0.174 vs 0.105), classes with few test instances.
- FreeSel at 640 px is no better than at 224 px, so keeping more defect patches in the attention filter (claim 1 diagnostic) does not translate into better subsets.
- **Consequence for novelty:** cover is not a method contribution over FreeSel; the two are equivalent within the noise of this benchmark. What remains:
  1. *Domain evidence:* label-free patch-level selection (either method) beats random scene selection on blade defects. Cover passes against random; FreeSel-224 is ahead of random on val and on the robust classes (P 1.000).
  2. *Subset luck:* eight random 25% subsets with an identical model, recipe and seed span **0.090–0.196** test mAP50-95 (bootstrap evaluator), a factor of 2, larger than any model effect measured in this project (DINOv3, ViT size, H3, H5). The class-matched subset used for all earlier 25% comparisons is a below-average draw (7th of 9). Low-label results reported on one random subset, including the BladeYOLO paper's 10–70% fractions, are therefore largely luck; our earlier 25% comparisons are paired and internally valid but tied to one subset.
  3. Test and val disagree on which selector is better, a further sign that one split cannot rank selectors at this scale.

### H7: what makes a labelled subset good? Pre-registered before computing (24 Oct)

**Question.** 18 runs trained the same model on 16 distinct 189-image subsets (class-matched, cover, cover_k1/k2, rand1–8, FreeSel-224 s0/s1, FreeSel-640 s0/s1) and their test scores span 0.09–0.20. If a property computable *without labels* predicts that score, a selector that maximises it directly is a new, evidence-based method; if none does, subset luck is unexplained noise, which strengthens H6 finding 2.

**Data.** Per subset: test mAP50-95 from the bootstrap evaluator (mean of its runs when two runs share a subset), val = best-epoch mAP50-95. Features: the cached DINOv3 ViT-B patch features of `tools/select_scenes.py` (128 random patches per training image, 640 px).

**Label-free properties (fixed now; candidates for a selector):**
- P1 number of scenes
- P2 soft coverage, K = 1,024 centroids (the cover objective), averaged over k-means seeds 0, 1, 2
- P3 soft coverage at K = 64 (coarse modes)
- P4 tail coverage: soft coverage restricted to the 25% smallest of the 1,024 clusters (rare patterns)
- P5 image diversity: mean pairwise cosine distance of the images' mean-pooled features
- P6 distribution match: MMD² (RBF on cosine distance) between subset patches and all training patches (lower = more representative)
- P7 typicality: mean cosine of each subset image (mean-pooled) to its 10 nearest training images
- P8 FreeSel pattern coverage: mean over all training images' FreeSel-224 patterns (seed 0) of the best cosine to a pattern of a subset image
- P9 atypical-patch mass: mean distance of subset patches to their nearest of the 1,024 centroids

**Label-based statistics (explanatory only, not usable for selection):** total boxes, the smallest per-class box count, thunderstrike boxes, crack boxes, classes with ≥ 10 boxes, class entropy.

**Decision rule.** A label-free property is a candidate if all three hold:
(a) Spearman ρ with test mAP50-95 over the 16 subsets has |ρ| ≥ 0.6 and permutation p < 0.05 / 9 (Bonferroni over P1–P9);
(b) ρ with val has the same sign and |ρ| ≥ 0.4;
(c) within the 8 random draws alone (not shaped by any selector), ρ with test has the same sign and |ρ| ≥ 0.5.
A candidate is only a hypothesis: the selector that maximises it must then beat cover, FreeSel-224 and random in new pre-registered runs before it is claimed. If no property passes, H7 is closed.

### H7 result: no label-free property predicts subset quality; H7 closed (24 Oct)

`tools/subset_properties.py` → `runs/analysis/subset_properties/{table.csv, report.md}` (745 training images in 224 scenes; Spearman ρ, two-sided permutation p, 20,000 permutations; Bonferroni threshold p < 0.0056):

| property | ρ test | p | ρ val | ρ test, 8 random draws |
|---|---|---|---|---|
| P1 scenes | +0.61 | 0.014 | +0.66 | +0.10 |
| P2 coverage K1024 | +0.43 | 0.100 | +0.52 | −0.43 |
| P3 coverage K64 | +0.47 | 0.063 | +0.46 | +0.31 |
| P4 tail coverage | +0.54 | 0.030 | +0.69 | +0.02 |
| P5 image diversity | −0.01 | 0.96 | −0.16 | −0.36 |
| P6 MMD² to pool | −0.34 | 0.19 | −0.64 | −0.05 |
| P7 typicality | −0.65 | 0.008 | −0.60 | −0.31 |
| P8 FreeSel pattern coverage | +0.38 | 0.15 | +0.48 | −0.24 |
| P9 atypical-patch mass | +0.47 | 0.067 | +0.84 | +0.29 |
| *label-based:* boxes / min class / thunderstrike / crack / classes ≥ 10 / entropy | −0.32 / −0.13 / −0.05 / −0.36 / +0.10 / −0.06 | ≥ 0.17 | | |

- **No property meets the rule.** The closest, P7 typicality (fewer typical images is better) and P1 number of scenes, miss the Bonferroni threshold and have no effect within the random draws (−0.31, +0.10). Their correlation over all 16 subsets is carried by the selector-versus-random split: the label-free selectors pick more scenes and less typical images *and* score higher, but among random draws these properties do not predict which draw is good.
- **Class composition does not explain subset luck either:** no box-count statistic correlates with test score (|ρ| ≤ 0.36). Even rand8, the best draw (0.196), has unremarkable properties on every measure.
- **Subset quality is partly real, not evaluation noise:** test and val agree across subsets (ρ = 0.65, p = 0.006; within the random draws 0.55, p = 0.16).
- **Verdict: H7 is closed; no new selector follows from it.** The variation between subsets is consistent across splits but invisible to every subset-level statistic tried, label-free or label-based. This supports finding 2 of the H6 robustness round: low-label results depend on which particular scenes are labelled in a way that current selection criteria cannot predict, so they must be reported over several subsets.

### H5 replication: pre-registered before the runs (24 Oct)

H5 is the only method candidate left, and its first result is fragile: test region AP +0.081, val only +0.018. Two replications, with the method and W = 1.0 unchanged:

**R1: more seeds on WTBD-group 25%** (ViT-B, 150 epochs, `--reward-start 100`, class-matched subset): baseline and reward at seeds 2 and 3 (`bladeyolo2b_wtbd_group_f25_s{2,3}`, `bladeyolo2b_reward_wtbd_group_f25_s{2,3}`).
**R2: a second dataset, WTBlade-group, all 2,136 training frames** (ViT-B, 50 epochs, the base recipe of `bladeyolo2b_wtblade_group`): baseline seed 1 (`bladeyolo2b_wtblade_group_s1`; seed 0 exists) and reward at seeds 0 and 1 (`bladeyolo2b_reward_wtblade_group_s{0,1}`). The reward runs over the last third of training, as on WTBD (epochs 100–150 of 150): `--reward-start 33` of 50. Fixed now, not tuned.

**Analysis (same as H5):** `last.pt` against the baselines' `last.pt`; `tools/granularity_eval.py --arms`, paired scene bootstrap; an arm is the mean of its seeds.
- R1 passes if, over all 4 WTBD seeds (s0–s3), test region AP50 rises ≥ +0.02 with P(>0) ≥ 0.95, val in the same direction, and standard test mAP50-95 drops by no more than 0.01. The new seeds s2/s3 alone are also reported.
- R2 passes on the same three conditions on WTBlade-group (all 7 classes; groups from `external/WTBlade-group/groups.csv`).
- **H5 is claimed as a method only if both R1 and R2 pass.** If R1 passes and R2 fails, it is reported as a WTBD-specific effect; if R1 fails, H5 is closed.

### H5 replication, R1 (WTBD, 4 seeds, test): fails; H5 closed (26 Oct)

`runs/analysis/h5r.sh` → `runs/analysis/novel/granularity/h5r_wtbd_4seeds/report.md`. Reward − baseline, mean of 4 seeds, `last.pt` vs `last.pt`: region AP **+0.036 [−0.016, +0.060], P(>0) 0.896**; standard AP50 +0.019 (P 0.928); standard AP50-95 +0.000 (guard holds). Per seed, region AP reward / baseline: s0 0.489 / 0.378, s1 0.493 / 0.446, s2 0.468 / 0.433, s3 0.415 / 0.462. The new seeds average about −0.006: the first +0.081 was an optimistic two-seed estimate. **R1 fails the primary condition (P < 0.95), so by the pre-registered rule H5 is closed.** The remaining reports (val, s2/s3 alone, WTBlade-group R2) are added below when the CPU evaluation finishes; they cannot reopen H5.

**Remaining reports (27 Oct)**, reward − baseline, region AP [95% CI], P(>0); standard AP50-95 in brackets:

| report | region AP | standard AP50-95 |
|---|---|---|
| WTBD 4 seeds, val | +0.000 [−0.048, +0.044], P 0.42 | (−0.001) |
| WTBD s2/s3 only, test | −0.005 [−0.048, +0.046], P 0.50 | (+0.005) |
| WTBD s2/s3 only, val | −0.010 [−0.057, +0.039], P 0.29 | (−0.002) |
| **R2: WTBlade-group, test** | **−0.006 [−0.018, +0.012], P 0.22** | (+0.003) |
| R2: WTBlade-group, val | −0.031 [−0.044, +0.006], P 0.05 | (+0.003) |

- **R2 fails as well:** on WTBlade-group the reward slightly lowers region AP on test and val, with standard AP unchanged.
- **Final verdict on H5: closed, no effect.** The new seeds and the second dataset show nothing; only the first two WTBD seeds on test ever showed a gain. The reward neither helps nor harms the standard metrics. This is the clearest example in the project of a pre-registered two-seed "pass" that does not replicate.
- Reports: `runs/analysis/novel/granularity/h5r_*`. The H5 run folders were then archived as in §27: 9 runs (`runs/analysis/archive/archived_27oct.txt`). Weights are kept for 5 reference runs only: `bladeyolo2b_wtbd_group_f25_s0/s1` (the ViT-B 25% baselines for Phase 29), `bladeyolo2b_wtblade_group`, `bladeyolo2_wtbd_group` and `yolo12l_wtbd_group` (1.8 GB in all).

## Phase 29: new hypotheses after H6 (26 Oct)

**What the data say about where detection fails** (constraints for any new idea):
- Defects are not small: images are 788–1,024 px, and median box short sides at 640 px input are 62–148 px (only 5% of hide_craze boxes are under 16 px). Tiling, slicing and super-resolution have nothing to recover, so they are not pursued.
- The weak classes are found as regions but missed as boxes: at 25%, craze and crack standard AP50 0.01–0.08 against region AP 0.13–0.47 (H5 reports). They are texture-defined and amorphous, not small.
- Rare classes are starved: thunderstrike has 79 boxes in all of train and test.
- Already tried and closed: SAM 2.1 box refinement, split-tolerant loss, semi-supervision with real or synthetic views, retrieval and anomaly priors, multi-view fusion, set-level reward, patch-coverage selection, OWLv2. New ideas must not repeat them.

**Literature (searched 26 Oct):**
- *Learned feature upsamplers* for frozen foundation features, 2025–26: JAFAR, LoftUp, AnyUp (ICLR 2026; encoder-agnostic), UPLiFT (CVPR 2026), ViT-Up (arXiv 2606.14024). They are evaluated on segmentation, depth and probing; **no object-detection result was found**, and small or fine structures are noted as their weak point.
- *Frequency and wavelet branches* for thin, low-contrast cracks: FD2-YOLO (Sensors 2025), YOLO11-FR (PLOS ONE 2026), TDDC-YOLO (Sci. Rep. 2026), instance-normalisation wavelet layer (Sensors 2025). These are prior art on concrete, wood and belts, not on blades and not on top of foundation features.
- *Few-shot defect synthesis by diffusion*: AnomalyDiffusion (AAAI 2024), DualAnoDiff (CVPR 2025), DefectFill (CVPR 2025), Anomagic (2025), SynSur (2026); for blades, a diffusion-built dataset (Sci. Data 2026). These are evaluated mostly on classification or generation metrics, rarely on detection under label scarcity.
- *Training-free VLM blade inspection*: "Seeing the Unseen" (arXiv 2510.22868) and a few-shot retrieval-grounded VLM (Results in Engineering 2026). Their claims have not been checked on a leak-free split.
- *Box-prompted SAM masks for defects*: Boxes2Pixels (arXiv 2604.11162) trains segmentation, including on a wind-turbine benchmark. This is close to our closed SAM 2.1 refinement, so it is not pursued.

**Hypotheses** (each has a cheap gate; a detector run is made only if its gate passes; every bar is fixed now):

- **H8: learned feature upsampling of frozen DINOv3 features for texture-defined defects.** Our detector reads DINOv3 at stride 16; an encoder-agnostic learned upsampler (AnyUp, `torch.hub` `wimmerth/anyup`, the paper model) could restore the fine texture that separates craze, crack and hide_craze from blade surface. This would be the first detection use of these upsamplers.
  - *Gate G8 (CPU, no detector training)*, `tools/dense_probe.py`: a dense linear probe (multinomial logistic regression, 7 outputs = 6 classes + background) on DINOv3 ViT-B last-layer features at stride 4 (160 × 160 at 640 px). Pixel targets are box unions (the smallest covering box decides the class). It is trained on the class-matched 25% subset (256 background and 256 defect pixels sampled per image) and scored on the test split by per-class pixel AP, pooled over images, with a 1,000-replicate scene bootstrap; val is reported for direction.
  - Arms: *bilinear* (standard upsampling, the control) and *AnyUp*.
  - **G8 passes** if AnyUp − bilinear mean pixel AP over the 6 classes is ≥ +0.02 with P(>0) ≥ 0.95, the mean over the weak classes (craze, crack, hide_craze) is positive, and val agrees in direction. If it passes, H8 becomes a detector change (AnyUp features fused at the stride-8 level) with its own pre-registration.
- **H9: high-frequency texture cues on top of foundation features.** Retrofits the crack-detection frequency branches to blades, but tests whether they add anything *beyond* DINOv3, which those papers never did.
  - *Gate G9* uses the same probe. Arm *bilinear + HF* adds 7 hand-crafted high-frequency channels to the bilinear features: |difference of Gaussians| at σ = 1, 2, 4, gradient magnitude at σ = 1, 2, 4, and local standard deviation (5 × 5), on the grey 640 px image, average-pooled to stride 4 and standardised.
  - **G9 passes** if (bilinear + HF) − bilinear mean pixel AP over craze, crack and hide_craze is ≥ +0.02 with P(>0) ≥ 0.95 and val agrees.
- **H10: few-shot generative synthesis for rare classes, judged under subset luck.** Inpaint thunderstrike and crack into defect-free blade regions with a few-shot diffusion inpainter fitted on the labelled 25% only, against a copy-paste control (pasting real defect crops) and no augmentation, each on 3 random 25% subsets. The new elements: a detection evaluation under scarcity on a leak-free split, and a comparison made across subsets rather than on one draw. Needs a GPU; its design and bar are pre-registered after G8/G9.
- **H11: does a training-free VLM claim survive a leak-free split?** Re-test the knowledge-augmented VLM inspection of arXiv 2510.22868 on WTBD-group test. This is a measurement, like item 4; it needs a GPU.
- **H12: do foundation features reduce subset luck?** The 8 random draws span 0.090–0.196 with DINOv3 ViT-B. The same 8 draws trained without DINOv3 would show whether stronger features shrink the dependence on which scenes are labelled. This is a measurement that sharpens the main methodological finding: 8 Kaggle runs of the cheaper no-DINOv3 model.

**Order:** G8 and G9 now, on CPU after the H5 evaluation finishes (the machine has 7 GB). H12 runs are cheap and can go to Kaggle in parallel. H10 and H11 follow.

### H12 result: DINOv3 does not reduce subset luck; draw quality carries over between models (27 Oct)

8 Kaggle runs of `bladeyolo2-l-nodino.yaml` on `tools/splits/wtbd_group_train_f25_rand{1..8}.txt`, seed 0, 150 epochs (`bladeyolo2nodino_rand{k}_wtbd_group_f25`), paired with the ViT-B runs on the same draws (`bladeyolo2b_rand{k}_wtbd_group_f25_s0`). Test mAP50-95 from the bootstrap evaluator (58 test scenes, 1,000 scene resamples), val is the best-epoch val mAP50-95. Script `runs/analysis/bootstrap/h12.py` → `runs/analysis/h12.log`.

| draw | ViT-B test | no DINOv3 test | gain | ViT-B val | no DINOv3 val |
|---|---|---|---|---|---|
| rand1 | 0.112 | 0.056 | +0.056 | 0.235 | 0.168 |
| rand2 | 0.135 | 0.067 | +0.069 | 0.265 | 0.154 |
| rand3 | 0.118 | 0.046 | +0.072 | 0.211 | 0.097 |
| rand4 | 0.123 | 0.063 | +0.060 | 0.211 | 0.077 |
| rand5 | 0.090 | 0.026 | +0.064 | 0.200 | 0.076 |
| rand6 | 0.100 | 0.056 | +0.044 | 0.172 | 0.086 |
| rand7 | 0.124 | 0.049 | +0.076 | 0.160 | 0.096 |
| rand8 | 0.196 | 0.072 | +0.125 | 0.270 | 0.140 |

| | ViT-B | no DINOv3 |
|---|---|---|
| test mean | 0.125 | 0.054 |
| test range (width) | 0.090–0.196 (0.107) | 0.026–0.072 (0.045) |
| test SD [scene-bootstrap 95% CI] | 0.032 [0.014, 0.066] | 0.014 [0.012, 0.042] |
| test SD / mean | 0.26 | 0.26 |
| val mean, SD | 0.216, 0.040 | 0.112, 0.037 |

- **Spread:** SD ratio ViT-B / no DINOv3 = 2.27 [0.59, 3.06], P(<1) 0.18. DINOv3 does not shrink the dependence on which scenes are labelled. In absolute terms the spread is, if anything, larger; relative to the mean it is the same (0.26 for both). On val the absolute SDs are equal (0.040 vs 0.037).
- **DINOv3 gain is draw-independent:** +0.071 [+0.051, +0.126], P(>0) 1.000, positive on 8/8 draws (+0.044 to +0.125). The low-data advantage of the frozen foundation features (Phase 27) therefore does not depend on a lucky subset.
- **Draw quality is a property of the draw:** Spearman ρ between the two models' draw rankings is +0.71 on test (p 0.047) and +0.62 on val; rand8 is the best draw and rand5 the worst-or-near-worst for both. Together with H7 (no label-free property predicts it), subset luck is real, model-independent and not removed by stronger features.

**Verdict.** H12 answered: foundation features add a constant multiplicative gain but do not reduce subset luck. Consequence for reporting: any low-label claim on WTBD-group must be made across several labelled subsets, whatever the backbone; a single draw can move test mAP50-95 by ±40% of its mean. The 8 no-DINOv3 runs are archived (metrics kept, weights deleted; test predictions cached in `runs/analysis/bootstrap/seeds_f25/preds/`).

Housekeeping: a stray `runs/detect/runs/detect/BladeYOLO_WindSurface/smoke_retrieval/` (smoke-test plots from 21 Oct) came from `train.py --test` with a relative weights path, which Ultralytics resolved under its default `runs/detect/`. The test output path is now made absolute (`47a6bfc`); the stray folder and two Kaggle leftovers in `runs/detect/` (`data.yaml`, `train_subset.txt`, pointing to `/kaggle/working`) were deleted.

**G8/G9 status:** restarted 27 Oct 19:02 after the first start never got past loading; results below.

### G8 and G9 results: both fail; H8 and H9 closed (27 Oct)

`tools/dense_probe.py`, all three arms, run 27 Oct 19:02–21:15 → `runs/analysis/dense_probe/report.md`. Probe trained on the class-matched 25% subset (189 images, 96,029 pixels); per-class pixel AP at stride 4, scene bootstrap of 1,000 replicates.

| arm | test mean AP (6) | test weak (3) | val mean AP (6) | val weak (3) |
|---|---|---|---|---|
| bilinear | 0.203 | 0.164 | 0.226 | 0.204 |
| AnyUp | 0.144 | 0.118 | 0.206 | 0.190 |
| bilinear + HF | 0.206 | 0.163 | 0.230 | 0.207 |

- **G8 fails, in the wrong direction.** Test: AnyUp − bilinear = −0.059 [−0.081, −0.008], P(>0) 0.010. Weak classes: −0.046 [−0.099, +0.002]. Val: −0.020 and −0.013, same direction. AnyUp loses most on craze (0.155 → 0.053) and thunderstrike (0.166 → 0.056); only hide_craze gains (+0.017 test, +0.039 val). This is a reliable negative, not a lack of power.
- **G9 fails as an informative null.** Weak classes, test: (bilinear + HF) − bilinear = −0.001 [−0.006, +0.005], P(>0) 0.354; val +0.003 [−0.005, +0.007]. All 6 classes: +0.003 (test), +0.004 (val). The interval excludes the +0.02 bar by a wide margin. Hand-crafted high-frequency cues add nothing that DINOv3 features do not already carry.
- **Caveat (not a rescue).** The pixel targets are box unions, not masks. Edge-aligned upsampling (AnyUp's purpose) sharpens features to object boundaries, while the targets fill the whole box. That alone could penalise it, so G8 shows only that AnyUp does not help a box-level task like detection; it is silent on segmentation. The defects' size also matters (median short side 62–148 px at 640): stride-16 features already resolve them, so finer features had little to recover.

**Verdict.** H8 and H9 closed: neither feature-side change is taken to a detector. Together with the earlier closures, the feature side (stronger backbones aside, Phase 27) is not where the remaining error is. That points back to the label side: few examples of the rare classes (H10), and the box-vs-region mismatch of the weak classes.

### H10: few-shot generated rare-class defects vs matched copy-paste, across 3 subsets. Pre-registered before any run (27 Oct)

**Question.** Does a generative model fitted on a few labelled defects add training signal that pasting those same real defects does not? A naïve "synthesis helps" claim is confounded by the extra images, by where they are placed, and by subset luck (H12: one draw moves mAP50-95 by ±40% of its mean). The design removes all three.

**Matched arms** (`tools/synth_defects.py`, tests `tests/test_synth_defects.py`). Each augmented arm adds the same 96 images to a 25% subset: a labelled image of the subset with one extra box (64 thunderstrike, 32 crack). The plan fixes the target image, class, box size and place, and is committed (`tools/splits/h10_plan_rand{1,2,3}.csv`); only the box content differs between arms.
- *Placement.* Box size: a real box of that class in the subset, ×0.8–1.2. Place: no overlap (>10%) with the image's boxes, and ≥ 80% of the box on the blade. The blade mask comes from SAM 2.1-t, prompted with a point at each labelled box centre, keeping the largest of its 3 masks. A first version without a blade mask put many boxes on sky and ground. Prompts beside the box let sky in, so they were dropped; the centre prompt sometimes returns only the defect, and then that image just has no free blade area.
- *paste* (control): the real box the size was drawn from, resized, with an 8% feathered edge.
- *synth*: Stable Diffusion 1.5 inpainting (`stable-diffusion-v1-5/stable-diffusion-inpainting`) with a rank-16 LoRA on the UNet attention, fitted per subset on that subset's thunderstrike and crack boxes only (1,500 steps, batch 4, lr 1e-4; box masks, random crops, flips and quarter turns; one fixed prompt per class). It inpaints each planned box (30 steps, guidance 7.5, seed 0 + row) and composites it with the same feathered edge.
- *none*: the existing ViT-B runs on the same draws, `bladeyolo2b_rand{1,2,3}_wtbd_group_f25_s0` (same model, recipe and seed).
- Training data available to fit: thunderstrike / crack boxes are 12 / 8 (rand1), 25 / 1 (rand2), 35 / 6 (rand3). Crack in rand2 is one-shot.
- Plans as generated: 96 rows each over 76 / 75 / 61 distinct target images (rand1/2/3; at most 3 rows per image). Of each subset's images, 189/193/189 have masks (all labelled images). On the paste contact sheet nearly every box sits on blade surface; one or two of 16 still touch sky (mask errors).
- Known mismatch: many WTBD thunderstrike boxes are whole rotors damaged by lightning, at turbine scale. Pasted onto a close-up blade surface, they are semantically odd; the synth arm inherits the same box sizes. Both arms share this, so the comparison stays matched.

**Runs** (Kaggle, `bladeyolo2-l.yaml --dino b`, seed 0, 150 epochs): `bladeyolo2b_{paste,synth}_rand{k}_wtbd_group_f25`, k = 1, 2, 3.

**Decision rule.** Target AP = mean AP50-95 of thunderstrike and crack (test has only 13 thunderstrike and 33 crack boxes, so neither alone has the power). Arm = mean over the 3 subsets; paired scene bootstrap over the 58 test scenes, 1,000 replicates (bootstrap evaluator). **H10 passes** if all of the following hold:
1. synth − paste target AP ≥ +0.02 with P(>0) ≥ 0.95;
2. synth − paste all-class mAP50-95 ≥ −0.01 (no harm elsewhere);
3. the target-AP difference is positive on at least 2 of the 3 subsets;
4. val agrees in direction (target AP on val from the same evaluator).

Reported, not decisive: per class; synth − none and paste − none (the augmented arms see 1.5× the images per epoch, so these mix augmentation with extra steps); the contact sheets (before | after) of both arms.

**Expected power.** Low. 46 target test boxes and 3 subsets; a real +0.02 may not reach P ≥ 0.95. A failure will be reported with its interval, as "not detected at this power", not as "no effect".

### H10 result: generated defects fail; matched copy-paste helps (28 Oct)

Runs (Kaggle, ViT-B, seed 0, 150 epochs): `bladeyolo2b_{paste,synth}_rand{1,2,3}_wtbd_group_f25`.

**Integrity** (`runs/analysis/bootstrap/h10_integrity.py`):
- All 6 arms have 96 images and 96 labels each.
- Each train list is exactly its subset plus the generated images, in plan order.
- Every generated label is the source labels plus the planned box.
- Pixels outside the box are unchanged (mean difference ≤ 0.47).
- All three LoRA fits ran 1,500 finite steps.
- `synth_rand3` stopped early at epoch 142 (best epoch 42, patience 100).

**Defects of the synth arm itself:**
- 19 of 288 synth boxes (7%) are almost unchanged (mean pixel change < 5), so they are labels on clean surface. The worst case is rand2 crack, 7 of 32.
- The generated thunderstrikes repeat one memorised burn shape.
- The generated cracks come out as dark smears.

Scored with `runs/analysis/bootstrap/h10.py` → `runs/analysis/h10.log`: bootstrap evaluator, 1,000 scene replicates. Target AP is the mean AP50-95 of thunderstrike and crack.

| test | target AP (rand1 / 2 / 3, mean) | mAP50-95 (mean) |
|---|---|---|
| synth | 0.139 / 0.111 / 0.039, **0.096** | 0.134 |
| paste | 0.156 / 0.210 / 0.128, **0.164** | 0.159 |
| none | 0.079 / 0.081 / 0.099, **0.086** | 0.122 |

| comparison | target AP [95% CI], P(>0); per subset | mAP50-95 [95% CI], P(>0) |
|---|---|---|
| **synth − paste (primary)** | **−0.068 [−0.123, +0.109], 0.16**; −0.017 −0.098 −0.089 | −0.025 [−0.086, +0.032], 0.11 |
| synth − none | +0.010 [−0.026, +0.126], 0.68; +0.060 +0.030 −0.060 | +0.012 [−0.014, +0.048], 0.85 |
| **paste − none** | **+0.078 [+0.000, +0.110], 0.975**; +0.077 +0.128 +0.029 | **+0.037 [+0.011, +0.081], 0.999** |
| val: synth − paste | −0.050 [−0.106, +0.007], 0.048; +0.002 −0.126 −0.025 | −0.009 [−0.030, +0.015], 0.20 |

**Verdict: H10 fails.** All four conditions fail. Generated defects are worse than pasting the same real defects, on test and on val, on all 3 subsets on test, and on 2 of 3 on val. Synthesis is discarded.

**Scope check (what survives).** Matched copy-paste of the subset's own thunderstrike and crack boxes onto blade-masked places beats no augmentation:
- target AP +0.078, positive on all 3 subsets;
- all-class mAP50-95 +0.037 [+0.011, +0.081], about +30%.

Caveats:
- The augmented arms train on 285 instead of 189 images, so they take 1.5× the steps per epoch.
- The "none" arm has no val predictions (its weights are archived).
- Copy-paste is known (Simple Copy-Paste, Ghiasi et al., CVPR 2021), so this is a recipe gain, not a contribution. It is noted for later recipes. No further runs are spent on it.

The 6 runs are archived (`runs/analysis/archive/archived_28oct_h10.txt`); test and val predictions are cached.

## Phase 30: hypotheses after G8/G9 (27 Oct)

**Diagnosis carried forward.** The feature side is exhausted: AnyUp hurts, high-frequency cues add nothing beyond DINOv3 (G8, G9), and retrieval as an input did not help (H3). What remains is on the label side (few rare-class examples, which subset is labelled) and on the decision side (how the detector scores what it already finds). **Reporting rule from now on:** every verdict states its interval, so a failure reads either as "an effect above X is excluded" or as "the test cannot see effects below X". A result that passes on WTBD-group is replicated on WTBlade-group before it is claimed.

Literature check (27 Oct): SAM 3 (arXiv 2511.16719) reports exemplar prompts on general benchmarks; we found no evaluation on industrial defects. Prototype classification with DINOv2 exists for few-shot detection in satellite imagery (arXiv 2403.05381), where prototypes label proposals inside the detector. We found no work on rescoring a trained detector's outputs with foundation-feature prototypes of its own labelled set. Cold-start active learning for detection is studied (e.g. arXiv 2205.07708), but we found no study that measures it across several initial labelled sets.

- **H15: post-hoc prototype rescoring with frozen DINOv3 (training-free, CPU).** A detector trained on few labels often finds the right region but ranks it poorly against false alarms, and that ranking decides AP. Each detection is embedded by DINOv3 ViT-B: the mean of the last-layer patch tokens inside its box, at 640 px. It is compared with the embeddings of the labelled boxes of the model's own training subset (cosine similarity; class score = mean of the top-3 similarities to that class's boxes; softmax over the 6 classes with τ = 0.05). Rescored confidence = conf^0.5 · p(class of the detection)^0.5. The class is never changed, and a class without labelled boxes keeps its confidence. α = 0.5, k = 3 and τ = 0.05 are fixed now; none is tuned.
  - *Data:* cached test predictions of 20 trained models with known subsets: ViT-B on rand1–8 and on the class-matched subset (seeds 0, 1), and no-DINOv3 on the same 10 subsets. Tool `tools/proto_rescore.py` → `runs/analysis/proto_rescore/`.
  - **G15 passes** if, for the 10 ViT-B models (arm = mean over models; paired scene bootstrap, 1,000 replicates), all of the following hold:
    1. rescored − original mean AP50-95 over thunderstrike and crack ≥ +0.02 with P(>0) ≥ 0.95;
    2. all-class mAP50-95 ≥ −0.005;
    3. rare-class AP improves on ≥ 7 of 10 models;
    4. val agrees in direction for the two class-matched models (the only ViT-B runs whose weights remain).
  - The no-DINOv3 family is reported separately (secondary: does a detector without DINOv3 gain more?). If G15 passes, the next step is a learned version (prototype head trained jointly), pre-registered separately, with a WTBlade replication.
- **H14: does detector-guided active learning reduce subset luck? (Kaggle, no-DINOv3 model, which H12 showed has the same relative luck at a third of the cost.)** Start from 3 random scene-level 10% draws. Train on each, then add 15% of the scenes chosen either by the detector's uncertainty (scenes ranked by the mean, over their images, of 1 − the highest detection confidence above 0.05, so images with no confident detection count as uncertain) or at random, and retrain at 25%. That is 3 runs at 10% and 6 runs at 25%.
  - **H14 passes** if, over the 3 draws (paired scene bootstrap), all of the following hold:
    1. uncertainty − random test mAP50-95 ≥ +0.02 with P(>0) ≥ 0.95;
    2. it is positive on at least 2 of 3 draws;
    3. val agrees in direction.
  - Reported: the SD across draws in each arm (does guided selection shrink the spread?).
  - Tooling: `tools/active_select.py` (tests `tests/test_active_select.py`). The scene of every training image (745 images, 224 scenes, `select_scenes.py` grouping) is committed as `tools/splits/wtbd_group_train_scenes.csv`. The start draws `tools/splits/wtbd_group_train_f10_al{1,2,3}.txt` have 76, 78 and 75 images (32, 21 and 29 scenes; seeds 11–13).
  - Runs: `bladeyolo2nodino_al{k}_wtbd_group_f10`, then `bladeyolo2nodino_al{k}{unc,rnd}_wtbd_group_f25`, all at seed 0 and 150 epochs. The 25% splits are written on Kaggle by `extend` and downloaded with the runs.
- **H13: SAM 3 for rare-class defects on a leak-free split (Kaggle; measurement first).** Stage A: SAM 3 zero-shot with a text prompt per class, on WTBD-group test, scored by the bootstrap evaluator. It is compared with OWLv2 (item 4) and with BladeYOLO-ViT-B at 25%. Stage B, a few-shot fine-tune on rand1–3, is designed only if Stage A beats OWLv2 on at least one rare class. The checkpoint (`facebook/sam3`) is gated on Hugging Face; the user has access (28 Oct). Tool `tools/sam3_eval.py`: `predict` on a Kaggle GPU (transformers `Sam3Model`; OWLv2's 18 fixed prompts, which the tool checks are identical; vision encoder once per image; instances with score ≥ 0.01, at most 100 per prompt, pooled per class, NMS at IoU 0.5, top 300 per image), then `score` locally with the bootstrap evaluator against OWLv2 text and the trained ViT-B 25% detector, class-aware and class-agnostic. The report states the Stage B condition.

*H13 update (28 Oct).*
- **Prior work:** the BladeYOLO paper (§V-G, Fig. 11) already ran SAM 3 zero-shot on blade defects, with and without "expert" text prompts. The analysis was qualitative only, with no numbers. They found redundant boxes, localisation deviations and category confusion, and concluded that the model needs task-specific adaptation.
- **Stage A** is therefore a measurement, not a hypothesis. It gives the first quantitative, leak-free number for their qualitative claim, and it is the zero-shot row of V1. It runs because it is ready and costs about 30 minutes of GPU.
- **Stage B** (fine-tuning on our labels) no longer depends on Stage A. It moves into V1, to be decided after G16.

*H13 Stage A result (29 Oct).*
- **Run:** Kaggle T4, `transformers` `Sam3Model`, 161 test images, 47,951 boxes, about 18 min. The predictions were fetched from the committed output of the `beegee11/bladeyolo` notebook (`kaggle kernels output`). Report: `runs/analysis/sam3/report.md`.
- **Class-aware mAP50-95:**
  - SAM 3 text: **0.042** [0.019, 0.072] (mAP50 0.078).
  - OWLv2 text: 0.014.
  - ViT-B 25%: s0 0.099, s1 0.108.
- **Differences:**
  - SAM 3 − OWLv2: +0.029 [−0.004, +0.055], P(>0) 0.94.
  - SAM 3 − ViT-B 25% (mean of the two seeds): about −0.062, interval [−0.189, −0.033].
- **Per class (AP50-95):** SAM 3 gets nearly all of its score from hide_craze (0.142) and surface_injure (0.091). The other classes are close to zero: craze 0.002, corrosion 0.010, thunderstrike 0.003, crack 0.005.
- **Class-agnostic mAP50-95:**
  - SAM 3 0.066, OWLv2 0.025, ViT-B 0.125 / 0.127.
  - SAM 3 − OWLv2: +0.041 [+0.002, +0.108].
  - SAM 3 − ViT-B: interval [−0.123, −0.003].
- **Reading:** this is the first quantitative, leak-free number for BladeYOLO's qualitative SAM 3 claim. Zero-shot SAM 3 beats OWLv2 but stays far below a detector trained on 189 labelled images. Most of the gap is class assignment from text: class-agnostic SAM 3 reaches about half of ViT-B. Everything here agrees with "needs task-specific adaptation".
- **Stage B condition:** formally met on thunderstrike (0.003 vs 0.001), but both values are effectively zero, so it carries no signal. Stage B is not triggered by Stage A. A SAM 3 fine-tune is decided in V1 only, after RF-DETR.

**Order:** G15 now (CPU, local, about an hour). H14 is a Kaggle session after the H10 sessions. H13 Stage A starts once access to the SAM 3 weights is granted.

### G15 result: fails the bar; a small, reliable training-free gain (27 Oct)

`tools/proto_rescore.py` → `runs/analysis/proto_rescore/report.md`. The bank holds 1,109 labelled boxes in 745 training images; each model uses only its own subset's boxes. Rescored − original, test (paired scene bootstrap, family = mean over models):

| family | all-class mAP50-95 | rare (thunderstrike, crack) AP50-95 | rare improved |
|---|---|---|---|
| ViT-B (10 models) | +0.005 [−0.001, +0.017], P 0.950 | **+0.013** [+0.003, +0.049], P 0.996 | 9/10 |
| no DINOv3 (10 models) | +0.012 [+0.007, +0.026], P 1.000 | +0.010 [+0.003, +0.034], P 0.999 | 10/10 |
| val, ViT-B class-matched (2) | −0.005 [−0.015, +0.013] | +0.002 [−0.009, +0.046], P 0.83 | 1/2 |

- ViT-B per class on test (family mean, original → rescored): thunderstrike 0.116 → 0.137, crack 0.071 → 0.076, surface_injure 0.264 → 0.271, hide_craze 0.160 → 0.155, craze and corrosion about unchanged. Nearly all of the rare-class gain is thunderstrike.
- No DINOv3: every class gains (craze +0.004, corrosion +0.008, surface_injure +0.018, thunderstrike +0.016, crack +0.003, hide_craze +0.022). As expected, the frozen features add more to a detector that has none.

**Verdict.** G15 fails condition 1 (+0.013 < +0.02). Conditions 2 and 3 hold; val is flat (condition 4 only nominally positive, +0.002). The learned prototype head is not pursued. What stands: a parameter-free post-processing step, with settings fixed in advance, gives a small rare-class gain that is reliable on test across 19 of 20 models. Its size is below what we pre-registered as worth a method, and val does not confirm it on the two models that could be checked.

Engineering: the val stage was killed twice by the out-of-memory killer (the ViT-B detector and the DINOv3 embedder together, with about 2.5 GB free). Val predictions are now made in a separate process, and each split is merged into `report.json` as it finishes.

## Phase 31: one variable behind the large effects? The coverage hypothesis (proposed 28 Oct)

**Why.** Thirty phases have closed most single-mechanism ideas: the feature side (G8, G9, H3), the label side (semi-supervision, H5), the decision side (G15, SAM refinement) and real views (H1, H4). The largest effect measured is also unexplained: which 25% of the scenes is labelled moves test mAP50-95 by a factor of 2 (0.090–0.196 over 8 random draws, H6 robustness round), more than DINOv3 does (+0.071, H12), and no property of a subset predicts it (H7). Each of the other large effects in this log can be read as one quantity, **how close a test defect is, in foundation-feature space, to a labelled defect of its class**:
- Leakage: a test image with a train twin is at proximity ≈ 1, and the published split inflates mAP50 by +0.29 (§11).
- Transfer: a detector trained on one blade dataset finds 5–9% of what the in-domain detector finds on the other (item 5).
- Rare classes: thunderstrike AP50-95 goes from 0.08 (class-matched subset) to 0.34 (cover), while box counts do not predict subset quality (H7).
- The closed priors: retrieval as an input (H3) and prototype rescoring (G15) hand the detector exactly this information, and gain little. That is expected if the trained detector already behaves like a proximity function.

H7 measured every subset property against the training pool, never against the scenes the detector is tested on. That link is untested.

**H16 (coverage hypothesis).** In low-label detection, whether a held-out defect is detected is governed by its proximity, in frozen DINOv3 space, to the nearest labelled defect of its class. Subset luck is the variation of that coverage between draws. Testable consequences:
- (a) Instance level: for the same test defect across subsets, closer exemplars give a higher detection score, beyond the number of exemplars.
- (b) Subset level: coverage of the test set predicts subset quality, including within random draws, where all nine pool properties of H7 failed.
- (c) A label-free selector that targets the deployment images (transductive) beats pool coverage (cover) and FreeSel.

**Literature check (28 Oct).**

| work | relation |
|---|---|
| Feldman & Zhang, *What neural networks memorize and why* (NeurIPS 2020, arXiv 2008.03703) | Long-tail theory: rare test examples are got right through memorised similar training examples, measured by subsampled influence. Classification only; closest theory |
| TFA (Wang et al., ICML 2020, arXiv 2003.06957); VFA (arXiv 2301.13411); RF100-VL (arXiv 2505.20612) | Few-shot detection results vary strongly with the sampled shots; TFA re-ran over multiple sample groups, VFA models support-set variance, RF100-VL reports seed variance. None explains which sample sets are good |
| Transductive experimental design (Yu et al., ICML 2006); ITL (Hübotter et al., NeurIPS 2024, arXiv 2402.15441) | Select examples informative about a known target set. Classification fine-tuning (MNIST, CIFAR-100), not detection |
| FreeSel, OFDS (2412.10032), foundation-assisted AL (2607.16671) | Cold-start selection for detection; targets the unlabelled pool, not the deployment set |
| BoS (ICLR 2024, arXiv 2403.13803), PCR (arXiv 2508.12082) | Label-free performance estimation for detectors from the model's outputs, not from proximity to labelled data |
| Data attribution for detectors | None found (searches 28 Oct) |

**Two new blade papers (read 28 Oct; PDFs at the repo root).**
- **KGP-YOLO** (Li et al., *IEEE Sensors J.* 26(2), Jan 2026; `11290107.pdf`).
  - Method: a keypoint network locates the blade in 1920 × 1080 UAV frames and crops it into 4 blocks; YOLOv11 with BiFormer routing attention then detects defects in the blocks.
  - Results: 87.3 / 92.4 / 88.5 mAP50 on SY-PLUS / DTU / Blade30.
  - Protocol: each dataset is augmented (flips, brightness, saturation) *before* the train/val/test split, e.g. DTU goes from 130 images to 780, split 624/78/78. Results are single runs.
- **GA-YOLOv5 + Unity synthetic data** (Zhang et al., *IEEE Trans. Ind. Appl.* 61(1), 2025; `10717455.pdf`).
  - Data: 432 annotated DTU NordTank images (3 classes), doubled by CLAHE copies, with a random 60/40 split of a dataset that holds repeated 2017/2018 shots of the same turbine.
  - Results: a genetic search over 30 hyperparameters adds +0.040 mAP50. Rendered Unity images add up to +0.022, with the best of six amounts read off the test set.
  - Protocol: single runs.

**How they bear on this programme:**
1. They add two more cases of the protocol §4 and §11 measured: augmentation or duplication before a random split, single runs, no intervals. This strengthens the leakage and rigour finding.
2. Their gains (+0.02–0.04 mAP50) are smaller than the seed noise measured here at 25% (±0.026 mAP50, §21) and far smaller than subset luck.
3. KGP-YOLO's blade cropping answers a small-defect problem in 1080p whole-blade frames. WTBD's defects are not small (median short side 62–148 px at 640, §29), so it does not change the WTBD direction. It could matter for WTBlade-type frames.
4. Synthetic data: H10 found that generated defects lose to pasting the same real defects. That fits the coverage hypothesis below, since rendered or generated defects are far from real test defects.

The direction is unchanged.

**What would be new:**
- an instance-level law for detection, measured across many labelled subsets of one benchmark (about 50 trained models with known subsets already exist here);
- an explanation of subset luck where pool statistics failed;
- the link to leakage and transfer;
- a deployment-targeted selector whose gain is pre-registered.

The idea that test items similar to training items are predicted well is folklore. The claim stands or falls on how well it predicts.

### G16: the coverage gate (CPU, cached data only). Pre-registered before computing (28 Oct)

**Data.**
- Cached test predictions of ViT-B models on 16 subsets: 18 models (class-matched s0/s1, rand1–8, cover s0/s1, cover_k1, cover_k2, FreeSel-224 s0/s1, FreeSel-640 s0/s1).
- Cached test predictions of no-DINOv3 models on 9 subsets: 10 models (class-matched s0/s1, rand1–8).
- Models sharing a subset are averaged.
- Embeddings are DINOv3 ViT-B last-layer patch tokens at 640 px: G15's cached test and val maps and its bank of training-box embeddings. A box embedding is the mean of the tokens inside the box, L2-normalised.

**Quantities.**
- x(b, S): proximity of test box b to subset S, the highest cosine similarity between b and any labelled box of b's class in S. It is undefined if S has no box of that class.
- y(b, m): the highest confidence among model m's predictions of b's class with IoU ≥ 0.5 to b; 0 if there is none.
- n(c, S): the number of labelled boxes of class c in S.

**Conditions.**
1. *Instance-level law (decides whether H16 holds).*
   - Fit: least squares of y on x and log(1 + n), with test-box and subset fixed effects. Pairs where x is undefined are left out.
   - Pass: the coefficient of x is positive with P(>0) ≥ 0.95 under a 1,000-replicate cluster bootstrap over the 58 test scene groups. This must hold in the ViT-B family and, separately, in the no-DINOv3 family.
2. *Subset luck explained.* Subset coverage X(S) is the mean over the 6 classes of the mean x(b, S) over that class's test boxes, with x = 0 where undefined. Spearman ρ between X(S) and test mAP50-95 (bootstrap evaluator, full test set) must meet all of:
   - ≥ 0.6 over the 16 ViT-B subsets, with permutation p < 0.05;
   - ≥ 0.5 within the 8 random draws, for ViT-B and for no-DINOv3;
   - on val (val boxes, best-epoch val mAP50-95), the same sign and ≥ 0.4 over the 16 subsets.

   This is H7's bar; the best pool property there reached +0.31 within random draws.
3. *A label-free proxy exists (decides whether H16b runs).* Transductive coverage T(S) is the cover objective with the training pool replaced by the unlabelled test images:
   - 128 random patches per test image (seed 0), clustered by k-means into 1,024 centroids (seed 0);
   - T(S) = the mean over centroids of the highest cosine to any of the subset's training patches (128 per image).

   Spearman ρ with test mAP50-95 must be ≥ 0.6 over the 16 subsets (p < 0.05) and ≥ 0.5 within the 8 random draws. On val (target = val images) it must have the same sign and be ≥ 0.4.

**Reported, not decisive:**
- the top-3 mean instead of the maximum;
- class-agnostic and region-found outcomes;
- detection when the class is absent from the subset;
- the law along the nested ViT-S curve (25 / 50 / 100%);
- proximity of the official-split test boxes to the official train set, against their inflation.

**Verdict.**
- H16 holds if condition 1 passes in both families.
- H16 explains subset luck if condition 2 passes.
- H16b runs only if condition 3 passes. If 1 and 2 pass but 3 fails, the selector needs a defect-focused proxy, designed and pre-registered separately.

### H16b: deployment-targeted scene selection (Kaggle; run only if G16 condition 3 passes). Pre-registered now

**Selector.** `tools/select_scenes.py`'s greedy whole-scene coverage, with the 1,024 target centroids taken from the unlabelled test images (*trans-test*) or the unlabelled val images (*trans-val*). It uses k-means seeds 0 and 1, selects 189 images (at most 5% over) and uses no labels.

**Runs.** ViT-B, seed 0, 150 epochs: 4 runs. The baselines already exist: cover (4 runs), FreeSel (4), random (8), class-matched (2).

**Passes if** all of the following hold:
1. trans-test family − cover family ≥ +0.02 test mAP50-95 with P(>0) ≥ 0.95 (paired scene bootstrap);
2. the trans-val family is above the cover family on val (best-epoch val mAP50-95, the split it targets);
3. both trans families are above the random family on their own target split.

Cross-target results (trans-test on val, trans-val on test) are reported. They separate "targeting this deployment set" from "a better selector in general".

**Framing.** This is transductive, like active learning with a known target set: the unlabelled images of a new inspection campaign exist before anything is labelled. No test label is used.

### G16 result: the instance-level law holds; it does not explain subset luck; H16b is not run (28 Oct)

`tools/coverage_gate.py` (tests `tests/test_coverage_gate.py`) → `runs/analysis/coverage_gate/{report.md, pairs.csv, subsets.csv}`. 235 test boxes × 16 ViT-B subsets (3,760 pairs) and × 9 no-DINOv3 subsets (2,115).

**Condition 1, passes.** The fit is box and subset fixed effects with a log(1 + n) covariate; the interval is a cluster bootstrap over the 58 test scenes.

| family | outcome, proximity | coefficient of x [95% CI] | P(>0) |
|---|---|---|---|
| ViT-B | class-aware, max (primary) | **+0.770 [+0.448, +0.982]** | 1.000 |
| ViT-B | class-aware, top-3 mean | +0.974 [+0.620, +1.346] | 1.000 |
| ViT-B | class-agnostic, max | +0.711 [+0.375, +0.895] | 1.000 |
| no DINOv3 | class-aware, max (primary) | **+0.376 [+0.176, +0.834]** | 1.000 |
| no DINOv3 | class-aware, top-3 mean | +0.582 [+0.289, +1.204] | 1.000 |
| no DINOv3 | class-agnostic, max | +0.321 [+0.152, +0.710] | 1.000 |

The comparison is the same test defect across subsets, with the subset's overall quality and the number of same-class exemplars held fixed. A closer labelled exemplar raises the detection confidence, in both families and for every variant of the statistic. The effect is twice as large with DINOv3 in the detector. This fits the closed priors H3 and G15: the detector already uses proximity in DINOv3 space.

**Conditions 2 and 3 fail.** They are directionally positive but below the bar:

| statistic | ρ test (16 subsets), p | within 8 random draws | ρ val |
|---|---|---|---|
| X: test-box coverage (uses labels; explanatory) | +0.43, 0.098 | +0.38 (ViT-B), +0.69 (no DINOv3) | +0.59 |
| T: transductive patch coverage (label-free) | +0.54, 0.032 | +0.45 (ViT-B) | +0.56 |

Subset-level coverage varies over a narrow range (X 0.738–0.760). Unlike every H7 pool property, both statistics are positive on test, within the random draws and on val. T reaches +0.45 within the random draws, where H7's best was +0.31. Neither meets the pre-registered bar (ρ ≥ 0.6, ≥ 0.5 within the random draws).

**Verdict.**
- H16 holds at the instance level.
- It does not explain subset luck at the subset level.
- By the pre-registered rule, H16b (transductive selection) is **not run**.

**Scope check.** What survives is a mechanism-level finding for the paper, with no run attached. Per defect, low-label detection is a proximity function in foundation-feature space, and DINOv3 doubles its slope. The weak subset-level link means subset luck is not just aggregated coverage: it depends on *which* defects are covered. No rescue runs are made (fail-fast rule).

### G17: click gate (CPU). Pre-registered before computing (28 Oct)

**Question.** If an annotator clicks once on each defect of the 556 training images outside the class-matched 25% subset (`tools/splits/wtbd_group_train_f25_rest.txt`, 838 ground-truth boxes), and the click gives location and class, how often can the 25% detector supply a correct box?

**Inputs.**
- The kept ViT-B class-matched models `bladeyolo2b_wtbd_group_f25_s0/s1` predict the 556 images (conf 0.001, NMS 0.7, as `tools/bootstrap_ci.py`).
- One click per ground-truth box, drawn uniformly inside the box (seed 0).

**Pseudo-box.**
- The highest-confidence prediction of the clicked class whose box contains the click, with conf ≥ 0.05.
- If there is none (fallback), a box of the class's median width and height in the 25% subset, centred on the click and clipped to the image.

**G17 passes** if, averaged over the two models, **≥ 60% of clicks** yield a pseudo-box with IoU ≥ 0.5 to the clicked ground-truth box.

**Reported, not decisive:**
- the rate per class and per model;
- the rate without the fallback;
- mean IoU;
- AP50-95-style quality (the share at IoU ≥ 0.75);
- context: plain pseudo-labelling at τ = 0.5 on the same images, i.e. the share of ground truth recovered and the precision.

If G17 passes, H17 (25% boxes + 75% click-derived boxes vs 25% boxes, on several subsets) is pre-registered separately.

**Result (28 Oct): G17 fails.** Clicks with a pseudo-box at IoU ≥ 0.5: **0.420** (s0 0.422, s1 0.418), against the 0.60 bar. Outputs: `runs/analysis/click_gate/{report.md, clicks.csv}` (`tools/click_gate.py`).

- IoU ≥ 0.75: 0.173; mean IoU 0.439.
- The detector supplies a box for 62% of clicks. 66.5% of those boxes reach IoU ≥ 0.5; the median-size fallback almost never does (1.6%). Without the fallback: 0.414.
- Per class, IoU ≥ 0.5: thunderstrike 0.625, surface_injure 0.476, corrosion 0.430, hide_craze 0.394, craze 0.359, crack 0.322.
- Context, plain pseudo-labels at τ = 0.5: 327/357 boxes, precision 0.676/0.658, ground truth recovered 0.223/0.267.

**Scope check.** Clicks raise recall: 0.41 of the ground truth gets a correct box, against 0.22–0.27 for plain pseudo-labels. Precision does not improve, though: 0.665 for detector boxes chosen by a click, versus 0.66–0.68 at τ = 0.5. So once a click has settled existence and class, the 25% detector still gets the *extent* wrong about a third of the time. That is the same weakness that sank semi-supervision (§21), and this time it is in localisation rather than existence. Training on these boxes would add about 350 correct boxes and 230 wrong ones, the noise mix that already failed. Point-supervised detection is prior art (Point DETR, P2BNet), so a better click-to-box head would not be new either. **H17 is dropped and no runs are made.** One note survives: on these defects the label bottleneck at 25% is box extent, not existence, which agrees with §14 (13–16% extent disagreement). Next: V1.

### Other directions (proposed, not pre-registered)

- **H17: clicks on many scenes instead of boxes on few.**
  - Evidence: labelled scenes are what pay (§17, H6). Box extent is the least reliable part of the labels (13–16% disagreement on identical images, §14). At 25%, misses are as common as near-misses (§21), and class confusion is rare.
  - Proposal: one click per defect on the other 75% of the scenes, with the box drawn by the 25% detector (its highest-scoring box of the clicked class that contains the click). Plain pseudo-labelling failed on existence (66% precision at τ = 0.5, §21); a click settles existence and class, and the detector only supplies extent.
  - Prior art: mixing points and boxes under a budget is known (UFO², arXiv 2010.10804: about 3.9 point-labelled images per box; Point DETR, arXiv 2104.07434).
  - New here: amorphous defects whose extent is a convention anyway, scored under subset luck.
  - *Gate G17 (CPU):*
    - Setup: the kept class-matched ViT-B models predict the 556 images outside the subset, and one click per ground-truth box is simulated (uniform inside the box).
    - Pseudo-box: the highest-confidence prediction of the clicked class containing the click (conf ≥ 0.05). Without one, a box of the class's median size centred on the click.
    - Pass: ≥ 60% of clicks yield a pseudo-box with IoU ≥ 0.5.
- **V1: a foundation-native detector as the base (validity, not novelty).**
  - Candidates: RF-DETR (fine-tuned DINOv2 backbone; state of the art on RF100-VL), DEIMv2 (DINOv3 backbones, arXiv 2509.20787) and SAM 3 (H13).
  - FT-FSOD (CVPR 2026, arXiv 2603.28182) reports that fine-tuning decides cross-domain few-shot detection: RF100-VL 10-shot 41.9 against SAM 3's 35.7. Every run here kept DINOv3 frozen.
  - If one of these clearly beats BladeYOLO-v2 ViT-B at 25%, method work should move to it, and the label-side methods that failed with a weak teacher should be re-checked on it.
- **R1 (high risk): transfer between blade datasets.**
  - If G16 holds, its cross-dataset proximity separates two explanations of the transfer failure: distance in appearance (which coverage can fix) or annotation convention (which it cannot).
  - Method idea: a detector conditioned on a dataset (convention) token, calibrated on a few boxes of a new dataset.
  - Multi-dataset detectors with dataset-specific heads exist; few-shot convention calibration is not known to us.
  - Needs at least 3 leak-free datasets. Only after G16.
- **H14 deferred.** At cold start with a weak model, uncertainty sampling usually loses to random (TypiClust). If G16 passes, the deployment-targeted selector replaces its uncertainty arm.

**Order.**
1. G16 on CPU once the H10 scorer has finished. It uses cached data only and takes minutes of compute.
2. Kaggle in the meantime: H13 stage A (SAM 3).
3. Then, depending on the results: H16b (4 runs), V1 (RF-DETR at 25% on rand1–3 and the class-matched subset), and G17.
4. **Status (28 Oct):** G16 holds at the instance level only (H16b not run); G17 fails (H17 dropped). Next: V1.

### V1: RF-DETR as the base detector (Kaggle). Pre-registered before any run (28 Oct)

**Question.** Does a foundation-native detector, fine-tuned end to end, clearly beat BladeYOLO-v2 ViT-B (frozen DINOv3) at 25% on the same subsets and test images?

**Arms.** RF-DETR Medium (`rfdetr[train]==1.11.2`, fine-tuned DINOv2 backbone, 576 px), with the package's default fine-tuning recipe: 100 epochs, lr 1e-4 (encoder 1.5e-4), batch 4 × accumulation 4, EMA, seed 0. The best checkpoint is chosen on our val split by RF-DETR's own val mAP. Subsets: class-matched (`wtbd_group_train_f25.txt`) and rand1–3. Each is paired with the cached ViT-B seed-0 run on the same subset (`bladeyolo2b_wtbd_group_f25_s0`, `bladeyolo2b_rand{1,2,3}_wtbd_group_f25_s0`).

**Scoring.** Test predictions at conf 0.001 go through `tools/bootstrap_ci.py` (scene bootstrap, 2,000 replicates) with mAP50-95 as the metric. Tool `tools/rfdetr_v1.py` (prepare / train / predict on Kaggle; import locally), tests `tests/test_rfdetr_v1.py`.

**"Clearly beats"** (all three must hold):
- mean RF-DETR − ViT-B mAP50-95 over the 4 subsets ≥ +0.03;
- positive on ≥ 3 of 4 subsets;
- pooled bootstrap P(difference ≤ 0) < 0.05.

**Decision.**
- *If it clearly beats ViT-B:* method work moves to RF-DETR. Semi-supervision (§20–21) and the click gate (G17) are re-checked with the RF-DETR teacher, using cached predictions first.
- *If it does not:* the frozen-DINOv3 base is defensible, and this becomes a reported baseline row.
- *Reported, not decisive:* val mAP50-95, per-class AP, and the spread across subsets (does the end-to-end fine-tune reduce subset luck?).

**Attempt 1 (Kaggle v68, 29 Oct): stopped at prediction.**
- Class-matched training finished: 100 epochs in 35 min, best EMA val mAP50-95 **0.253** (RF-DETR's COCO evaluator, our val split).
- `predict` then failed: the predictions contained class id 6. RF-DETR's head has `num_classes + 1` outputs, the last being a no-object slot, and RF-DETR's own evaluator ignores it because it isn't a category.
- Fix: `predict` drops that slot and logs the count. `train` now skips a run that already has a best checkpoint.
- No test numbers yet; rerun.

**Attempt 2 (Kaggle v69, 29 Oct): 3 of 4 subsets finished.**
- Class-matched, rand1 and rand2 finished (best EMA val 0.263 / 0.288 / 0.260). Prediction dropped 0.9–1.7k no-object boxes per split.
- rand3 died at epoch ~29 with `No space left on device`: the 10-epoch `.ckpt` files of all runs filled `/kaggle/working`. Fix: `checkpoint_interval = epochs`, and every checkpoint except the best is deleted after training.
- The class-matched best checkpoint was recovered from the v69 output (`runs/detect/BladeYOLO_WindSurface/rfdetrm_wtbd_group_f25/checkpoint_best_total.pth`, gitignored) as the teacher for follow-ups.
- New subcommand `tools/rfdetr_v1.py score` → `runs/analysis/v1/report.md`.

**Interim test result (3/4 subsets; the verdict waits for rand3).** Test mAP50-95, RF-DETR vs ViT-B s0:

| subset | RF-DETR | ViT-B | diff [95% CI] | P(≤ 0) |
|---|---|---|---|---|
| class-matched | 0.189 | 0.099 | +0.090 [−0.026, +0.155] | 0.057 |
| rand1 | 0.205 | 0.112 | +0.093 [+0.028, +0.215] | 0.003 |
| rand2 | 0.215 | 0.135 | +0.080 [+0.019, +0.154] | 0.006 |

- **Mean +0.087 [+0.034, +0.158], P 0.001, positive on 3/3.** All three parts of the bar already hold on these subsets.
- RF-DETR at 25% (0.19–0.22) approaches our 100% ViT-S number (0.245, Ultralytics evaluator).
- The gains are largest on hide_craze (+0.19 to +0.22 AP50-95) and surface_injure. craze goes from about 0 to 0.04–0.09. crack is mixed: +0.157 on class-matched, −0.087 on rand2.

**Final result (Kaggle v70 adds rand3, 29 Oct): V1 clearly beats ViT-B.** rand3 trained cleanly: best EMA val 0.234, 751 no-object boxes dropped on test.

| subset | RF-DETR | ViT-B | diff [95% CI] | P(≤ 0) |
|---|---|---|---|---|
| class-matched | 0.189 | 0.099 | +0.090 [−0.026, +0.155] | 0.057 |
| rand1 | 0.205 | 0.112 | +0.093 [+0.028, +0.215] | 0.003 |
| rand2 | 0.215 | 0.135 | +0.080 [+0.019, +0.154] | 0.006 |
| rand3 | 0.147 | 0.118 | +0.029 [−0.061, +0.079] | 0.304 |

- **Mean +0.073 [+0.023, +0.119], P 0.002, positive on 4/4.** The pre-registered bar holds: mean ≥ +0.03, ≥ 3/4 positive, P < 0.05.
- **Subset luck is not reduced:** RF-DETR spans 0.147–0.215 (range 0.068) against ViT-B's 0.099–0.135 (0.036).
- **rand3 caveat:** thunderstrike collapses for RF-DETR (0.004 vs ViT-B 0.186), although rand3 has the most thunderstrike boxes (35). So the end-to-end fine-tune can lose a class that the frozen-feature model keeps. The cause is not investigated.
- **Decision (pre-registered):** method work moves to RF-DETR. First, G17 (click → box) is re-checked with the RF-DETR class-matched teacher, using the same rule and bar as before. Semi-supervision is re-checked only if the box precision clearly exceeds 66%.
- **Paper:** RF-DETR at 25% becomes the base row. The frozen-DINOv3 YOLO is a baseline. BladeYOLO's claims are compared on the same leak-free split.

**G17 recheck with the RF-DETR teacher (29 Oct): still fails.**
- RF-DETR class-matched predictions on the 745 training images were made locally on CPU from the recovered checkpoint, in a separate project venv `.venv-rfdetr` (rfdetr 1.11.2, transformers 5.22, torch 2.15 CPU). That took about 6 min; 6,980 no-object boxes were dropped.
- `tools/click_gate.py --models rfdetrm_wtbd_group_f25 --tag rfdetr` → `runs/analysis/click_gate/rfdetr/report.md`. Same rule and bar as before.
- **Clicks with a pseudo-box at IoU ≥ 0.5: 0.519** (ViT-B 0.420; bar 0.60).
  - IoU ≥ 0.75: 0.327 (ViT-B 0.173). Mean IoU 0.519.
  - The detector supplies a box for 79% of clicks; 65.1% of those boxes are correct (ViT-B 66.5%).
  - Per class: thunderstrike 0.804, surface_injure 0.589, craze 0.511, crack 0.492, hide_craze 0.444, corrosion 0.430.
  - Plain pseudo-labels at τ = 0.5: precision 0.502, ground truth recovered 0.288.
- **Scope check (diagnostic, not a gate).** The upper bound picks the best-IoU candidate among the conf ≥ 0.05 boxes containing the click. Same class: 0.661 (ViT-B 0.483). Any class, with the class taken from the click: 0.735 (ViT-B 0.556). Even a perfect same-class selector would only just clear the bar, and a realistic one would land below it. Training would still see roughly 35–45% wrong boxes, and point-supervised detection is prior art.
- **Decision.** H17 stays dropped. The semi-supervision recheck is not triggered: detector-box precision 0.651 does not clearly exceed 66%, and τ = 0.5 precision is 0.502.
- **What survives:** the stronger base mainly sharpens extents (IoU ≥ 0.75 nearly doubles), but at 25% it does not make click-derived labels clean enough.

### G16-R: the proximity law on RF-DETR (CPU). Pre-registered before computing (30 Oct)

**Question.** Does G16's instance-level law (a test defect is detected if a labelled same-class defect is close in frozen DINOv3 ViT-B space) also hold for RF-DETR, which fine-tunes its own DINOv2 backbone end to end? Is the slope flatter than for the frozen-feature model?

**Setup.**
- Same definitions as G16 condition 1: x = max cosine to the subset's same-class boxes; y = highest same-class confidence at IoU ≥ 0.5; y ~ x + log(1+n) with box and subset fixed effects; cluster bootstrap over the 58 test scenes, 1,000 replicates.
- Subsets: class-matched and rand1–3, where RF-DETR runs exist. RF-DETR family: `rfdetrm_*`. Comparison: ViT-B seed 0 on the same 4 subsets.
- Slope difference RF-DETR − ViT-B is computed on the same scene draws.
- Caveat: the confidence scales differ (sigmoid focal vs YOLO). So the comparison is also reported with y binarised (detected at conf ≥ 0.05).

**Readout.**
- The law **holds for RF-DETR** if the class-aware slope is positive with P(>0) ≥ 0.95.
- **Flatter** if the binarised slope difference RF-DETR − ViT-B < 0 with P ≥ 0.95. That would mean end-to-end fine-tuning generalises beyond the labelled neighbours.
- Reported, not decisive: top-3 proximity, class-agnostic outcome, and the raw-confidence difference.
- No runs depend on this. It decides whether "detection = proximity to labelled defects" is a detector-independent finding.

**Result (30 Oct): the law holds for RF-DETR's confidence, but no longer for whether a defect is found at all.** Report: `runs/analysis/coverage_gate/rfdetr/report.md` (`tools/coverage_gate.py rfdetr`). 235 test boxes × 4 subsets, 58 scene clusters.

| detector | class-aware conf | class-agnostic conf | detected (conf ≥ 0.05) |
|---|---|---|---|
| RF-DETR | +0.478 [+0.169, +0.929], P 0.994 | +0.363, P 0.976 | **−0.016 [−0.886, +0.603], P 0.498** |
| ViT-B s0 | +0.593 [−0.022, +0.921], P 0.972 | +0.662, P 0.994 | +1.098 [+0.113, +1.684], P 0.984 |

- **Difference RF-DETR − ViT-B:**
  - binarised −1.114 [−2.342, +0.182], P(<0) 0.957: **flatter**, by the pre-registered rule;
  - class-aware confidence −0.116, P 0.60;
  - class-agnostic −0.299, P 0.85.
- **Detection rate:** at conf ≥ 0.05, RF-DETR detects 64.1% of test defects, ViT-B 48.2%.
- **Reading:**
  - For the frozen-feature detector, *whether* a defect is found at all depends on a close labelled neighbour.
  - The end-to-end fine-tuned detector finds defects regardless of proximity: the binarised slope is about 0. Proximity still shapes *how confidently* it scores them.
  - So in RF-DETR, coverage moves from recall to ranking. Far-from-labelled defects are found but ranked low, which is the regime where AP is lost to ranking against false alarms.
- **Caveats:** 4 subsets only, and one seed per detector. Proximity is measured in DINOv3 ViT-B space, not in RF-DETR's own DINOv2 features.
- **Implication (not pre-registered):** a ranking-side intervention is better motivated on RF-DETR than it was on ViT-B. G15 (prototype rescoring, +0.013 on ViT-B) is the cheapest instance and can be re-checked on the cached RF-DETR test predictions, with its settings fixed as in G15.

### G15-R: prototype rescoring on RF-DETR (CPU). Pre-registered before computing (30 Oct)

**Setup.**
- G15's method unchanged: DINOv3 ViT-B box embeddings at 640 px; each model's own training-subset bank; K 3, τ 0.05, α 0.5; class never changed.
- Applied to the 4 RF-DETR models (class-matched, rand1–3), on test and on val (the cached val predictions of all 4 exist).
- Paired scene bootstrap, 1,000 replicates; family = mean over the 4 models.

**G15-R passes** under G15's rule with 4 models. All of these must hold:
1. rare-class (thunderstrike, crack) AP50-95 gain ≥ +0.02 with P(>0) ≥ 0.95;
2. all-class mAP50-95 ≥ −0.005;
3. rare-class AP improves on ≥ 3 of 4 models;
4. val family mean rare gain > 0.

**Secondary (reported, motivated by G16-R):** the all-class mAP50-95 gain and its interval. If coverage now acts on ranking, the gain should exceed G15's ViT-B +0.005.

**If it passes:** a learned, coverage-calibrated ranking head on RF-DETR is pre-registered separately. **If it fails:** one scope check, then the box-extent direction.

**Result (30 Oct): G15-R passes.** Report: `runs/analysis/proto_rescore/rfdetr/report.md` (`tools/proto_rescore.py rfdetr`). Rescored − original, family mean of 4 models:

| split | rare (thunderstrike, crack) AP50-95 | all-class mAP50-95 | rare improved |
|---|---|---|---|
| test | **+0.046** [+0.013, +0.109], P 0.999 | +0.011 [−0.002, +0.058], P 0.952 | 4/4 |
| val | **+0.037** [+0.008, +0.076], P 0.995 | +0.013 [−0.001, +0.043], P 0.967 | 4/4 |

- **Per model (test mAP50-95):** class-matched 0.189 → 0.209; rand1 0.205 → 0.217; rand2 0.215 → 0.223; rand3 0.147 → 0.153.
- **Rare AP (test):** 0.133 → 0.202, 0.066 → 0.106, 0.052 → 0.090, 0.006 → 0.044. rand3's thunderstrike collapse is partly repaired.
- **Per class (test, family mean):**
  - gains: thunderstrike 0.072 → 0.125, crack 0.056 → 0.097, hide_craze +0.012;
  - small losses: surface_injure −0.023, craze −0.008, corrosion −0.005.
- **Against G15 on ViT-B:** rare +0.046 vs +0.013 (3.5×), all-class +0.011 vs +0.005, and val now agrees (4/4 on val vs 1/2 before).
- **Mechanism, consistent with G16-R:** RF-DETR finds far-from-labelled defects but ranks them low. A coverage signal from the labelled exemplars re-ranks them. ViT-B never found them, so there was little to re-rank.
- **Caveats:**
  - The all-class gain is small and its interval includes 0.
  - Frequent classes lose a little: the fixed α = 0.5 applies the same pull to every class.
  - Only 4 models, one seed each.
  - Prototype/kNN rescoring with foundation features has prior art in few-shot detection, so the novelty must be checked before any method claim.
- **Next (as pre-registered):**
  1. A literature check on prototype/exemplar re-ranking of detector outputs.
  2. Then a learned, coverage-calibrated ranking head on RF-DETR, pre-registered separately.

**Literature check (30 Oct).** Scoring detections by similarity to labelled support instances is established in few-shot and cross-domain detection:
- the NTIRE 2025 CD-FSOD report (arXiv 2504.10685): one team uses cosine similarity to support instances as the confidence;
- prototype similarity on DINOv2 (arXiv 2606.23069, 2026);
- prototype calibration of pseudo-labels (Calibrate Before Adapt, 2026);
- detector calibration in FSOD (ECCV 2022).

G15-R's rescoring is therefore **not a contribution in itself**. What we did not find:
- a measured account of *when* exemplar re-ranking pays off: fine-tuning decouples recall from coverage (G16-R), so the gain appears on a fine-tuned base and barely on a frozen-feature one (G15 vs G15-R);
- its use as a coverage signal under subset luck.

Any method claim must rest on that mechanism, plus a learned or adaptive version that beats the fixed rule, not on rescoring as such.

### H18: a learned coverage-aware re-ranker on RF-DETR (CPU). Pre-registered before computing (31 Oct)

**Idea.** G15-R pulls every class equally (fixed α 0.5): rare classes gain, frequent ones lose a little. H18 learns how far to trust the coverage signal, per class and per proximity level. It is a small calibrator over detection-level features, fitted on val, the split already used for model selection, the way temperature scaling is. Test stays untouched.

**Features** (fixed now, per detection; prototype quantities exactly as in G15, DINOv3 ViT-B at 640 px, the model's own subset bank):
- z1 = logit(RF-DETR confidence);
- z2 = log p(detected class) from the prototype softmax (K 3, τ 0.05);
- z3 = top-3 mean cosine to the detected class's labelled boxes;
- z4 = log(1 + labelled boxes of the class in the subset);
- interactions z1·z3 and z2·z4;
- one-hot class (6).

A class without labelled boxes gets z2 = z3 = 0 and a missing-indicator.

**Target.** The fraction of IoU thresholds 0.50:0.95 at which the detection is a true positive, under greedy matching in RF-DETR's original confidence order (the AP50-95 notion of correct).

**Model.** Logistic regression with fractional targets (two weighted rows per detection), L2 with C = 1, standardised features. One calibrator pooled over the 4 models' val predictions, applied to each model's test predictions; the new confidence is the predicted probability. No other knobs.

**Scoring.** Test, paired scene bootstrap, 1,000 replicates, family = mean over the 4 models. Arms: original, fixed rule (G15-R), H18. Val is reported with 2-fold scene cross-fitting, as a sanity check only.

**H18 passes** if all of these hold:
1. all-class mAP50-95, H18 − fixed rule ≥ +0.01 with P(>0) ≥ 0.95;
2. rare (thunderstrike, crack) AP50-95, H18 − original ≥ +0.03;
3. no class loses more than 0.01 AP50-95 against the original (family mean);
4. H18 > fixed rule in all-class mAP50-95 on ≥ 3 of 4 models.

**If it fails:** one scope check (which condition failed, and why), then the fixed rule plus the mechanism stands as the result, and work moves to the box-extent direction.

**Result (31 Oct): H18 passes on test. The val cross-fit sanity check disagrees.** Tool `tools/rerank_h18.py` (tests `tests/test_rerank_h18.py`, 3 pass) → `runs/analysis/rerank_h18/report.md`. The calibrator was fitted on 186,654 val detections.

| test, family mean | mAP50-95 | rare AP50-95 |
|---|---|---|
| original | 0.189 | 0.064 |
| fixed rule | 0.200 | 0.111 |
| **H18** | **0.214** | **0.113** |

- **Per model (original → fixed → H18):**
  - class-matched 0.189 → 0.209 → 0.213;
  - rand1 0.205 → 0.217 → 0.235;
  - rand2 0.215 → 0.223 → 0.239;
  - rand3 0.147 → 0.153 → 0.168.
- **The four conditions:**
  1. H18 − fixed rule, all-class: +0.013 [+0.003, +0.021], P 0.987 ✓
  2. Rare, H18 − original: +0.048 [+0.016, +0.110] ✓
  3. Every class gains against the original: craze +0.010, corrosion +0.009, surface_injure +0.001, thunderstrike +0.045, crack +0.052, hide_craze +0.031 ✓
  4. H18 beats the fixed rule on 4/4 models ✓
- H18 − original, all-class: +0.025 [+0.012, +0.070].
- **Val cross-fit (sanity check only, as pre-registered):** H18 0.262, fixed rule 0.270, original 0.257. That is H18 − fixed −0.009 [−0.023, +0.008], with H18 ahead on 1/4 models; surface_injure −0.034, thunderstrike −0.004.
- **Robustness check** (`tools/rerank_h18_halfval.py`, run after the result and reported whatever it showed): calibrators fitted on either val half alone still beat the fixed rule on test, +0.0117 and +0.0116 (full val +0.0132). So the test gain does not depend on fitting on all of val. The val disagreement is a property of the val halves: when a calibrator is fitted on one scene fold and scored on the other, it does not beat the fixed rule. That shift between scene folds is not explained.
- **Verdict:**
  - By the pre-registered rule, **H18 passes**.
  - Honest strength: robust on test (4/4 models, every class, robust to halving the fitting data), but not reproduced within val.
  - Before a paper claim it needs an independent check: RF-DETR seed 1 on the same subsets, or a second dataset (WTBlade). It runs only on Kaggle, and is pre-registered separately.

### H18-R: replication on new RF-DETR models (Kaggle). Pre-registered before any run (31 Oct)

**Runs.** RF-DETR Medium, **seed 1**, on the same 4 subsets with the same recipe (`rfdetrm_s1_*`).

**Frozen transfer.** The H18 calibrator is the one already fitted on the seed-0 models' val predictions, refitted deterministically from the same data. It is applied unchanged to the seed-1 models' test predictions, with no refitting on seed 1. The fixed rule is applied as in G15-R.

**H18-R replicates** if, on the 4 seed-1 models (test, paired scene bootstrap, family mean), all of these hold:
1. H18 − fixed rule, all-class mAP50-95 > 0 with P(>0) ≥ 0.90;
2. H18 − original, all-class > 0 with P ≥ 0.95;
3. H18 beats the fixed rule on ≥ 3 of 4 models.

Also reported: the same comparison on seed-1 val (the calibrator was never fitted on these models), and the V1 comparison of seed 1 against ViT-B.

**If it replicates:** H18 is the method result; WTBlade follows as the cross-dataset check. **If not:** H18 is reported as a single-seed finding, and the fixed rule plus the mechanism is the result.

### H18-R result (31 Oct; Kaggle v72, `rfdetrm_s1_*`)

v72 trained the 4 seed-1 models without errors. The log has only the old harmless `rfdetr.__version__` traceback, and `v1_options.json` shows seed 1 for all 4. Commands: `tools/rfdetr_v1.py import --run <run dir>`, then `tools/rfdetr_v1.py score --seed1` (→ `runs/analysis/v1/report_s1.md`) and `tools/rerank_h18.py replicate` (→ `runs/analysis/rerank_h18/replicate/report.md`).

**V1 on seed 1 (RF-DETR s1 vs ViT-B s0, test mAP50-95):**

| subset | RF-DETR s1 | ViT-B | diff [95% CI] | P(≤0) |
|---|---|---|---|---|
| class-matched | 0.221 | 0.099 | +0.122 [+0.057, +0.227] | 0.000 |
| rand1 | 0.188 | 0.112 | +0.076 [+0.019, +0.199] | 0.004 |
| rand2 | 0.220 | 0.135 | +0.085 [+0.028, +0.205] | 0.003 |
| rand3 | 0.169 | 0.118 | +0.051 [−0.055, +0.120] | 0.172 |

- Mean +0.083 [+0.045, +0.150], P 0.000, 4/4 positive. This replicates seed 0's mean of +0.073.
- Seed noise for the same subset is about ±0.03 (seed 0 → seed 1: class-matched 0.189 → 0.221, rand3 0.147 → 0.169).
- Seed 0's rand3 thunderstrike collapse (0.004) does not recur (0.062). It was at least partly seed noise on 13 boxes in 3 scenes.

**H18-R (frozen seed-0 calibrator, seed-1 models, test):**

| model | original | fixed rule | H18 |
|---|---|---|---|
| class-matched | 0.221 | 0.234 | 0.235 |
| rand1 | 0.188 | 0.207 | 0.217 |
| rand2 | 0.220 | 0.231 | 0.240 |
| rand3 | 0.169 | 0.175 | 0.185 |
| **family mean** | **0.200** | **0.212** | **0.219** |

1. H18 − fixed: +0.0078 [−0.0027, +0.0214], P(>0) **0.937** (bar 0.90) ✓
2. H18 − original: +0.0197 [+0.0075, +0.0449], P **1.000** (bar 0.95) ✓
3. H18 beats the fixed rule on **4/4** models ✓

→ **H18-R replicates.**

- **Per class vs original:** every class gains (craze 0.070→0.077, corrosion 0.183→0.189, surface_injure 0.432→0.438, thunderstrike 0.082→0.110, crack 0.030→0.077, hide_craze 0.401→0.425).
- **Per class vs fixed rule:** H18 gains on the frequent classes (surface_injure +0.021, hide_craze +0.020) and on crack (+0.013). It loses on thunderstrike (0.130 → 0.110). Rare AP is about equal (H18 0.094, fixed 0.097). So on seed 1, H18's margin over the fixed rule comes from not hurting the frequent classes, not from extra rare-class gain.
- **Effect size:** the gain over the fixed rule shrinks from +0.013 (seed 0) to +0.008 (seed 1). Its seed-1 CI includes 0, so it passes only on the pre-registered P ≥ 0.90 bar.
- **Seed-1 val** (reported, not a condition): H18 − fixed +0.012 [−0.005, +0.025], P 0.931, 4/4. H18 − original +0.012, P 0.996. These are new model outputs on the images the calibrator was fitted on, so they are not fully independent. Even so, the seed-0 cross-fit disagreement (−0.009) does not recur.
- **Verdict:**
  - By the pre-registered rule, H18 replicates across training seeds.
  - Honest size: about +0.01 over the fixed rule and +0.02 over raw RF-DETR, consistent in sign across 8 models.
  - The test set is still the same 58 scenes, so the remaining open question is cross-dataset transfer. Per the pre-registration, **WTBlade-group is next** (RF-DETR 25% runs plus a pre-registered H18 transfer).

### H18-W: cross-dataset replication on WTBlade-group. Pre-registered before any run (31 Oct)

**Why WTBlade-group.**
- It is leak-free (§H4 test bed) and comes from a different source with different classes: burn, crack, deformity, dirt, oil, peeling, rust.
- Its test split has 332 frames in 177 scene groups, three times WTBD-group's 58. So it tests whether the H18 gain survives scene sampling, which no WTBD seed can.

**Subsets** (`tools/wtblade_subsets.py` → `tools/splits/wtblade_group_train_f25{,_rand1,_rand2,_rand3}.txt`, summary `runs/analysis/wtblade/subsets_f25.txt`):
- Budget: 25% of the 2,136 training frames, i.e. 534 images. Scenes come from `external/WTBlade-group/groups.csv` (522 train scenes).
- They are drawn exactly like the WTBD subsets: one class-matched draw (best of 1,000 random scene orders) and 3 plain random draws (rng 0).
- Scenes that would overshoot the budget by > 5% are skipped. This excludes one 899-image video scene: 42% of train, holding dirt 1,185 and rust 1,065 of all train boxes. So no subset contains it, and the class-matched draw cannot match rust's full-set share (56 rust boxes vs 1,249 in train).

| subset | scenes | images | burn | crack | deformity | dirt | oil | peeling | rust |
|---|---|---|---|---|---|---|---|---|---|
| class-matched | 174 | 534 | 47 | 95 | 121 | 446 | 66 | 436 | 56 |
| rand1 | 211 | 534 | 67 | 129 | 157 | 344 | 74 | 481 | 70 |
| rand2 | 323 | 534 | 86 | 240 | 222 | 380 | 84 | 71 | 116 |
| rand3 | 227 | 547 | 66 | 134 | 157 | 336 | 57 | 463 | 69 |

**Runs (Kaggle).** RF-DETR Medium, seed 0, with the V1 recipe unchanged (100 epochs, batch 4 × accum 4) on the 4 subsets: `rfdetrm_wtblade_group_f25{,_rand1,_rand2,_rand3}`. Test and val predictions at conf 0.001 come from `tools/rfdetr_v1.py predict`. They are imported with `import --dataset wtblade` into `runs/analysis/bootstrap/wtblade_f25/{test,val}`.

**Frozen from WTBD; nothing tuned on WTBlade.**
- Fixed rule: α 0.5, k 3, τ 0.05, DINOv3 ViT-B box embeddings, prototype bank = the model's own labelled subset.
- H18 recipe: the same 8 feature families (z1 logit conf, z2 log p_proto, z3 top-3 cos, z4 log(1+n), z1·z3, z2·z4, missing-class indicator, class one-hot, now 7 classes), the fractional AP50-95 TP target and logistic C = 1.
- Rare classes are defined by the same rule as WTBD: the two classes with the fewest full-train boxes, burn (141) and oil (149).

**Primary test, H18-W.**
- The class one-hot ties a calibrator to its label set, so the WTBD calibrator cannot be applied to WTBlade. What transfers is the recipe.
- The H18 calibrator is fitted once on the 4 WTBlade models' val predictions (val, 323 frames) and applied unchanged to WTBlade test.
- Paired scene bootstrap over the 177 test groups from `groups.csv`, 2,000 replicates, family mean over the 4 models.

**H18-W replicates** if all three of these hold (the H18-R bars):
1. H18 − fixed rule, all-class mAP50-95 > 0 with P(>0) ≥ 0.90;
2. H18 − original, all-class > 0 with P ≥ 0.95;
3. H18 beats the fixed rule on ≥ 3 of 4 models.

**Reported, not gating:**
- the fixed rule vs original on WTBlade (does G15-R's rescoring gain transfer?);
- rare (burn, oil) and per-class AP;
- val 2-fold scene cross-fit as on WTBD;
- the absolute RF-DETR 25% scores against the 100% ViT-B reference `bladeyolo2b_wtblade_group` (0.172 test mAP50-95).

**Consequences.**
- *Replicates:* H18 is a cross-dataset method result. The paper's method claim is "a coverage-aware re-ranker learned on val transfers across seeds and datasets".
- *Fails:* H18 is a WTBD-specific finding. The paper rests on the mechanism (G16/G16-R) and the fixed rule, with H18 reported as not transferring. No rescue runs (§30).

### H18-W result (1 Nov; Kaggle run 31 Oct 12:05–18:13 UTC, `rfdetrm_*wtblade_group_f25`)

**Run checks.**
- All 4 models trained on their subsets (534/534/534/547 images, seed 0) and predicted all 332 test and 323 val frames. The only traceback is the old `rfdetr.__version__` cell.
- RF-DETR's own val mAP50-95 plateaued at about 0.13 by epoch 10–20 on every subset. mAP75 stayed at 0.06–0.08, so localisation is the limit, as on WTBD.
- Commands: `tools/rfdetr_v1.py import --dataset wtblade`, then `tools/rerank_h18.py wtblade` → `runs/analysis/rerank_h18/wtblade/report.md` (log `runs/analysis/rerank_h18/wtblade.log`). The calibrator was fitted on 384,212 val detections.

**Test** (177 scenes, 2,000 paired bootstrap replicates):

| model | original | fixed rule | H18 |
|---|---|---|---|
| class-matched | 0.132 | 0.126 | 0.134 |
| rand1 | 0.139 | 0.129 | 0.140 |
| rand2 | 0.131 | 0.130 | 0.133 |
| rand3 | 0.128 | 0.115 | 0.129 |
| **family mean** | **0.132** | **0.125** | **0.134** |

1. H18 − fixed: +0.0090 [+0.0032, +0.0125], P 0.996 ✓
2. H18 − original: +0.0018 [−0.0009, +0.0045], P **0.907** (bar 0.95) ✗
3. H18 beats the fixed rule on 4/4 models ✓

→ **H18-W does not replicate.**

- **The decisive number is the fixed rule, not H18.** G15-R's prototype rescoring *lowers* WTBlade test mAP50-95: fixed − original −0.0072 [−0.0124, +0.0001], P(>0) 0.028, on 4/4 models. It also lowers rare AP (burn, oil: 0.085 → 0.076) and peeling (0.233 → 0.202).
- **H18 only recovers what the fixed rule loses.** It ends up at the original detector's level. So on WTBlade, H18's "gain over the fixed rule" means it learned to ignore a harmful signal, not that the coverage signal helps.
- **Val cross-fit agrees:**
  - fixed − original −0.016, P(>0) 0.013;
  - H18 − original −0.010, P(>0) 0.039;
  - H18 − fixed +0.006.
- **Verdict:**
  - Per the pre-registered consequence, H18 is a WTBD-specific finding and the method claim is dropped. On WTBlade, neither the fixed rule (G15-R) nor H18 helps RF-DETR.
  - What does transfer is only the statement "the learned calibrator is never worse than the fixed rule". On its own, that is not a contribution.
  - A scope check under the fail-fast rule (§30) follows.

**Scope check** (`tools/h18w_scope.py` → `runs/analysis/proto_rescore/wtblade/scope.md`). Criterion fixed before looking: Spearman ρ(log labelled boxes of the class in the model's subset, fixed − original test AP50-95) < 0 with one-sided permutation P < 0.05, pooled over WTBD (4 seed-0 models) and WTBlade (4 models).

- Pooled: 52 (model, class) points, ρ −0.285, P 0.022 → **scope found by the criterion.**

| labelled boxes of the class | points | mean Δ AP50-95 | share > 0 |
|---|---|---|---|
| < 30 | 8 | **+0.037** | 0.88 |
| 30–60 | 10 | −0.002 | 0.40 |
| 60–120 | 20 | −0.007 | 0.50 |
| 120–250 | 7 | +0.002 | 0.43 |
| ≥ 250 | 7 | −0.012 | 0.14 |

- **Honest strength: weak.**
  - All 8 points below 30 boxes are WTBD (crack 1–8, thunderstrike 10–25, craze 24–29), because WTBlade's rarest class has 47 boxes per subset.
  - Within WTBlade there is no slope (ρ +0.09); within WTBD ρ −0.39.
  - So box count and dataset are confounded. The boundary rests on 8 points from 4 models and 3 classes.
- **Reading:** a boundary condition, not a method. Exemplar rescoring pays only when a class has fewer than about 30 labelled boxes, i.e. when coverage limits ranking (G16-R). Above that it is neutral to harmful. This is consistent with G15 vs G15-R and with H18-W.
- **What would confirm it:** a pre-registered test on WTBlade in the low-count regime. Take smaller WTBlade subsets so that some classes fall below 30 boxes, and predict a positive fixed − original gain for exactly those classes. That breaks the dataset confound. One short Kaggle run (small subsets train fast).

### H19-W: the low-count boundary within WTBlade-group. Pre-registered before any run (1 Nov)

**Question.** The H18-W scope check found prototype rescoring (G15-R fixed rule) helps only classes with < 30 labelled boxes, but every such point was WTBD, so box count and dataset were confounded. H19-W puts WTBlade classes below 30 boxes and asks whether the gain appears for exactly those classes, within one dataset, with the predictions fixed now.

**Subsets** (`tools/wtblade_subsets.py --fraction 0.08 --prefix wtblade_group_train_f08` → `tools/splits/wtblade_group_train_f08{,_rand1,_rand2,_rand3}.txt`, summary `runs/analysis/wtblade/subsets_f08.txt`). Same drawing rule as H18-W (one class-matched draw, best of 1,000; 3 random draws, rng 0; > 5% overshoot skipped), budget 8% = 171 images. 8% was picked before any training as the fraction that splits WTBlade's (model, class) points roughly 1 : 2 into below and above 30 boxes (6% puts almost everything low, 10% almost everything high).

| subset | scenes | images | burn | crack | deformity | dirt | oil | peeling | rust |
|---|---|---|---|---|---|---|---|---|---|
| class-matched | 92 | 171 | **8** | 63 | 45 | 128 | **15** | 63 | **23** |
| rand1 | 115 | 171 | **26** | 75 | 90 | 112 | 34 | 44 | 31 |
| rand2 | 89 | 176 | **15** | 57 | 71 | 121 | **26** | **23** | 50 |
| rand3 | 110 | 172 | 34 | 55 | 66 | 149 | **23** | **27** | **21** |

Bold = low points (1 ≤ n < 30): **10 low, 18 high, 0 with n = 0.** The low set is not only the globally rare classes: it includes peeling and rust, which are common in full train.

**Runs (Kaggle).** RF-DETR Medium, seed 0, V1 recipe unchanged (100 epochs, batch 4 × accum 4): `rfdetrm_wtblade_group_f08{,_rand1,_rand2,_rand3}`, test + val predictions at conf 0.001, imported with `rfdetr_v1.py import --dataset wtblade` (cache `runs/analysis/bootstrap/wtblade_f08/`).

**Frozen; nothing tuned.** Fixed rule α 0.5, k 3, τ 0.05, DINOv3 ViT-B box embeddings, bank = the model's own 8% subset. The threshold 30 comes from the H18-W scope table and is not moved.

**Scoring** (`tools/h19w_lowcount.py`, committed with this entry; dry run on the f25 models reproduces H18-W: fixed − original −0.0072, ρ +0.088, no low points). Δ = fixed − original test AP50-95 per (model, class); paired scene bootstrap over the 177 test groups, 2,000 replicates.

**H19-W confirms the boundary** if all three hold:
1. mean Δ over the 10 low points > 0 with P(>0) ≥ 0.95;
2. mean Δ(low) − mean Δ(high) > 0 with P ≥ 0.95;
3. Δ > 0 on ≥ 70% of low points (≥ 7 of 10).

**Reported, not gating:** the count-gated rule (rescore only classes with n < 30, original elsewhere) vs original on all-class mAP50-95, here and retro-actively on WTBD f25 and WTBlade f25 (where it is the identity); Spearman ρ(log n, Δ) within f08; absolute f08 scores vs f25.

**Consequences.**
- *Confirmed:* the boundary is a property of label count, not of WTBD. The paper gets a mechanism-backed usage rule: exemplar rescoring pays only below ~30 labelled boxes per class (coverage-limited ranking, G16-R), and the count-gated rule is the practical form.
- *Not confirmed:* the boundary was a WTBD artefact; the rescoring line is closed. Next direction: box extent (mAP75 stuck at 0.06–0.08 on both datasets), starting with a prior-art check. No rescue runs (§30).

### H19-W result (1 Nov; Kaggle run 06:34–09:14 UTC, `rfdetrm_*wtblade_group_f08`)

**Run checks.** All 4 models trained for 100 epochs (1,100 steps) on 171/171/176/172 images, about 40 min each. They predicted all 332 test and 323 val frames. The only traceback is the old `rfdetr.__version__` cell. Imported to `runs/analysis/bootstrap/wtblade_f08/`; scored with `tools/h19w_lowcount.py` → `runs/analysis/proto_rescore/wtblade/h19w_f08.md` (+ `.json`, `.log`). Original all-class test mAP50-95 at 8%: 0.093 / 0.102 / 0.084 / 0.088 (mean 0.092, vs 0.132 at 25%).

| statistic | point | 95% CI | P(>0) |
|---|---|---|---|
| mean Δ, 10 low points | −0.0019 | [−0.0126, +0.0071] | 0.286 |
| mean Δ, 18 high points | −0.0076 | [−0.0122, +0.0001] | 0.026 |
| low − high | +0.0056 | [−0.0102, +0.0149] | 0.764 |
| fixed rule − original, all-class | −0.0056 | [−0.0094, +0.0003] | 0.031 |
| count-gated rule − original, all-class | −0.0007 | [−0.0045, +0.0025] | 0.286 |

1. low > 0, P ≥ 0.95: ✗ (−0.0019, P 0.286)
2. low − high > 0, P ≥ 0.95: ✗ (P 0.764)
3. ≥ 7/10 low points positive: ✗ (5/10)

→ **H19-W does not confirm the boundary.** Within WTBlade, ρ(log n, Δ) = −0.07: no count slope. Below 30 boxes the gain is a coin flip (burn 8 boxes +0.034, oil 15 boxes −0.022). The WTBD < 30 gains were a WTBD property, not a label-count law.

- Descriptive only (not a rescue): rescoring hurts peeling at every count (−0.022 to −0.058 on all 4 models); without peeling the low mean is +0.0035. A class-specific failure, not a count effect.
- The count-gated rule is not a usable recipe: −0.0007 on WTBlade f08.
- **Verdict (pre-registered consequence):** the rescoring line (G15-R fixed rule, H18, the boundary) is closed as a contribution. What stands from it is WTBD-only: G15-R/H18 on WTBD, and G16/G16-R as a mechanism description there. Next direction: box extent (mAP75 stuck at 0.06–0.08 on both datasets), starting with a prior-art check.

### Box-extent direction: prior-art check and ceiling diagnostic (1 Nov). Discarded before any run

**Prior art.**
- Box uncertainty: KL-Loss with variance voting (He et al., CVPR 2019) models each box side as a Gaussian. Variance voting needs NMS, which RF-DETR does not have.
- Repeated labels: "Drawing the Same Bounding Box Twice?" (Tschirschwitz et al., 2023) estimates ground truth from repeated box labels; earlier work fuses annotators with weighted box fusion (WBF); a medical-detection paper re-weights the loss by expert agreement. Inaccurate-box training: OA-MIL (ECCV 2022).
- Wind turbines: oriented boxes for blade defects (EUSIPCO 2024, Heriot-Watt), and the DTU re-annotations, which note the one-box-or-several ambiguity.
- Our own earlier attempts: SAM 2.1 refinement was negative (§21); split-tolerant targets helped corrosion only (§23).
- The one open niche was WTBD's *accidental* repeated labels (independently annotated augmented copies, §14) as an extent signal. The diagnostic below checks whether extent is worth attacking at all.

**Diagnostic** (`tools/extent_ceiling.py` → `runs/analysis/extent/ceiling.md`). Two IoU distributions on WTBD:
- *annotators:* the best same-class IoU of a label projected into its independently labelled copy (same-view, axis-aligned pairs, exact transform from the §14 audit cache);
- *models:* the best same-class IoU of a prediction with conf ≥ 0.1, per WTBD-group test box.

| source | boxes | IoU ≥ 0.5 | of those, ≥ 0.75 | median IoU of those | IoU ≥ 0.75 |
|---|---|---|---|---|---|
| annotators, all copies | 2,965 | 0.87 | 0.70 | 0.83 | 0.61 |
| annotators, test copies | 392 | 0.91 | 0.75 | 0.84 | 0.68 |
| **RF-DETR-M 25%** (4 subsets) | 940 | **0.55** | **0.67** | **0.83** | 0.37 |
| ViT-B 25% (4 subsets) | 940 | 0.41 | 0.45 | 0.73 | 0.19 |
| BladeYOLO-v2 100% | 235 | 0.63 | 0.47 | 0.74 | 0.29 |

**Finding: RF-DETR is at the annotator ceiling for extent.**
- Once RF-DETR finds a defect, its box agrees with the label as tightly as a second annotator's box does: 0.67 vs 0.70 reach IoU ≥ 0.75, median IoU 0.83 vs 0.83.
- Its whole strict-IoU gap comes from finding the defect at all (IoU ≥ 0.5: 0.55 vs 0.87).
- The YOLO-family detectors (ViT-B, and BladeYOLO-v2 even at 100%) *were* extent-limited (0.45–0.47). The "extent bottleneck" of §21 was a property of those detectors, and RF-DETR removed it.
- So better box refinement can buy RF-DETR at most a few points of IoU ≥ 0.75 among found defects. The extent direction is discarded without spending a run (§30).

**Where RF-DETR fails instead** (test boxes of the weak classes, 4 subsets, conf ≥ 0.1):

| class | boxes | hit (IoU ≥ 0.5, same class) | located but another class | missed (no same-class overlap) | merged / shifted / part |
|---|---|---|---|---|---|
| crack | 132 | 0.17 | **0.30** | **0.42** | 0.11 |
| thunderstrike | 52 | 0.38 | **0.54** | 0.00 | 0.08 |
| craze | 156 | 0.33 | 0.14 | 0.13 | 0.39 (merged 0.20) |

- The remaining failures are **recognition of the rare classes**. Crack and thunderstrike have 1–39 labelled boxes per subset: they are either missed or found with the wrong class.
- Class-agnostic ≈ class-aware held for the YOLO family (§21); it does not hold for RF-DETR's rare classes.
- Craze is the only class with a granularity component (20% merged).
- This points back to coverage (G16): what limits RF-DETR at 25% is too few labelled examples of the rare classes, not where the box edges go. On WTBlade, the H18-W/H19-W rescoring did not fix rare-class recognition.
- Caveat: WTBlade has no group-split copy audit, so the ceiling comparison is WTBD-only.

### Error anatomy: at 25% labels, RF-DETR's right boxes are among its outputs and scored low (1 Nov)

`tools/error_anatomy.py` → `runs/analysis/extent/anatomy.md`. Per test box: is a same-class box at IoU ≥ 0.5 among the outputs (conf ≥ 0.001), and how is it scored? Outcomes:
- *hit*: such a box with conf ≥ 0.1;
- *ranked low*: one exists only below 0.1;
- *other cls*: only other-class boxes at IoU ≥ 0.5;
- *no box*: nothing at IoU ≥ 0.5.

Candidate recall = hit + ranked low. YOLO-family runs keep fewer boxes after NMS (~50–80 per image) than RF-DETR (~300), so recall within each image's top-50 is also given.

| dataset | detector | hit | ranked low | other cls | no box | candidate recall | in top-50 |
|---|---|---|---|---|---|---|---|
| WTBD | **RF-DETR-M 25%** (4 subsets) | 0.55 | **0.37** | 0.04 | 0.04 | **0.92** | 0.81 |
| WTBD | ViT-B 25% (4 subsets) | 0.41 | 0.26 | 0.15 | 0.18 | 0.67 | 0.66 |
| WTBD | BladeYOLO-v2 100% (2 seeds) | 0.63 | 0.17 | 0.08 | 0.12 | 0.80 | 0.80 |
| WTBlade | **RF-DETR-M 25%** (4 subsets) | 0.55 | 0.28 | 0.03 | 0.14 | **0.83** | 0.72 |
| WTBlade | RF-DETR-M 8% (4 subsets) | 0.43 | 0.33 | 0.07 | 0.16 | 0.77 | 0.62 |
| WTBlade | ViT-B 100% | 0.58 | 0.27 | 0.06 | 0.09 | 0.85 | 0.81 |

**Findings.**
1. **RF-DETR at 25% already proposes the right box for most defects.**
   - WTBD: candidate recall 0.92, above BladeYOLO-v2 trained on 100% (0.80), even within the top-50 boxes (0.81 vs 0.80).
   - WTBlade: 0.83 at 25%, the level of ViT-B trained on 100% (0.85).
   - Thunderstrike on WTBD: 1.00 candidate recall, but 62% ranked low.
2. **Its low-label deficit is ranking.**
   - On WTBD, 37% of test defects have a correct box scored below 0.1 (median score 0.025).
   - Extent is at the annotator ceiling (previous entry), and "no box" is 4%.
   - The true class is usually the top or second class at that place (median rank 1–2). WTBD's rare classes are the exception: thunderstrike rank 4, crack rank 3, confused with craze, surface_injure and hide_craze.
   - So the loss is mostly true boxes scored below other images' false positives, not wrong classes.
3. **Explains the H18 dataset split.**
   - WTBD's rare-class errors are re-rankable confusions: the right class is present at rank 3–4.
   - On WTBlade, rare-class misses split between ranked-low and no box. Crack is 44% no box even for ViT-B at 100% (42%); WTBlade crack also has the lowest copy agreement (81.5%, §13). No re-ranker can recover a box that was never proposed.
4. **Fewer labels mostly add ranked-low and no-box cases** (WTBlade 25% → 8%: hit 0.55 → 0.43, ranked low 0.28 → 0.33, no box 0.14 → 0.16).

**Consolidated reading** (with G16-R and the extent ceiling): in a fine-tuned foundation detector, label scarcity shows up as *scoring*, not as *finding* or *boxing*. That is the opposite of the YOLO-family picture (§21: extent- and proposal-limited). The next method question is therefore: which label-free signal separates ranked-low true boxes from false positives?

### G20: query-consensus rescoring (CPU, cached predictions). Pre-registered before computing (1 Nov)

**Idea.** From the anatomy: RF-DETR proposes the right box but scores it low. One-to-one matching trains a DETR to give one query per object a high score and push its duplicates down. At low labels the "winner" itself is under-confident, but the duplicates still pile up on real defects, while background false positives tend to be isolated. The score mass of the other queries at the same place is a label-free objectness signal, with no exemplars, prototypes or classes involved (unlike G15-R/H18):

  s'_p = s_p + λ · Σ_{q ≠ p, IoU(p, q) ≥ 0.5} s_q   (q of any class; boxes unchanged)

**Prior art to check if it passes:** box/score voting (Gidaris & Komodakis 2015), weighted box fusion (single-model use), Softer-NMS variance voting. Those fuse *boxes* after NMS in NMS-based detectors; scoring a DETR's own suppressed duplicates as evidence is the possibly new part.

**Protocol** (`tools/g20_consensus.py`, committed with this entry):
- λ ∈ {0.25, 0.5, 1, 2} is chosen per dataset on val (family mean mAP50-95 of the 4 RF-DETR 25% models; WTBD val `v1_val_preds`, WTBlade `wtblade_f25/val`) and applied once to test.
- Paired scene bootstrap, 2,000 replicates.

**G20 passes** only if it holds on **both** WTBD-group and WTBlade-group test (the H18-W lesson):
- consensus − original family mean mAP50-95 > 0 with P(>0) ≥ 0.95;
- better on ≥ 3 of 4 models.

**Reported, not gating:** WTBlade 8% models (with WTBlade's λ); per class; the val curve.

**Consequences.**
- *Pass:* a prior-art check, then replication on the seed-1 WTBD models, and a write-up as a training-free DETR re-scorer.
- *Fail:* one scope check, then discard (§30).

### G20 result: fails on both datasets; discarded (1 Nov)

`runs/analysis/g20/report.md` (log `g20.log`). Val picked the smallest λ (0.25) on both datasets, and val mAP fell monotonically with λ.

| test | original | consensus (λ 0.25) | Δ [95% CI], P(>0) | models better |
|---|---|---|---|---|
| WTBD 25% | 0.189 | 0.144 | −0.046 [−0.068, −0.002], 0.022 | 0/4 |
| WTBlade 25% | 0.132 | 0.112 | −0.021 [−0.027, −0.014], 0.000 | 0/4 |
| WTBlade 8% (reported) | 0.092 | 0.085 | −0.008 [−0.012, −0.003], 0.004 | 0/4 |

Every class loses on WTBD (crack 0.056 → 0.008).

**Scope check** (`runs/analysis/g20/scope.py` → `scope.log`, val only). The obvious flaw is that under AP matching the duplicates of a true box are false positives, and the rule raised them too. So the check boosted only the top-scoring box of each overlap cluster:
- WTBD val: −0.011 at λ 0.25, falling to −0.089 at λ 2;
- WTBlade val: −0.005, falling to −0.053.
No scope.

**Reading:** the premise is false. Score mass from overlapping queries does not separate RF-DETR's ranked-low true boxes from its false positives: false positives pile up just as much (on defect-like texture and on the true defects' surroundings). **G20 discarded** (§30). The ranking signal, if one exists, has to come from outside the detector's own query set.

### G21: flip agreement as a ranking signal (CPU inference). Pre-registered before any flipped prediction (1 Nov)

**Idea.** G20 showed that RF-DETR's own query set carries no ranking signal. An outside view of the same image might. Texture-triggered false positives depend on the exact pixels, while real defects should survive a left-right mirror. A box ranks high only if *both* views score it:
- *agreement* (G21's arm): s' = √(s · s_f), where s_f is the score of the same-class box from the mirrored image (mapped back) with the highest IoU ≥ 0.5, and 0 if none.
- *tta-mean* (the prior-art baseline): standard score-averaging test-time augmentation, s' = (s + s_f)/2.
- Boxes are unchanged in both arms.

**Models.** Only checkpoints still available are used (every Kaggle download returns the latest version, so the f25 WTBlade and the other WTBD checkpoints are gone):
- WTBlade-group 8%: 4 subsets;
- WTBD-group 25%: class-matched and rand3.
Flipped predictions are made locally in `.venv-rfdetr` (`tools/g21_flip.py predict`); original predictions are the Kaggle ones.

**G21 passes** if, on **both** datasets' test splits, agreement − original family mean mAP50-95 > 0 with paired scene bootstrap P(>0) ≥ 0.95 (2,000 replicates), and agreement is better on all models but at most one.

**Contribution test (beyond standard TTA):** agreement − tta-mean > 0 with P ≥ 0.90 on both datasets.
- If G21 passes but this does not, the gain is ordinary TTA: a recipe note, not a contribution.

**Reported:** val (direction), tta-mean − original.

**Fail:** one scope check, then discard (§30).

### G21 result: fails on both datasets; plain flip TTA also hurts RF-DETR; discarded (2 Nov)

`runs/analysis/g21/report.md` (logs `predict.log`, `score.log`). Test, family mean mAP50-95:

| test | original | tta-mean | agreement | agreement − original [95% CI], P(>0), better | agreement − tta-mean, P(>0) |
|---|---|---|---|---|---|
| WTBD 25% (2) | 0.168 | 0.143 | 0.139 | −0.029 [−0.076, −0.019], 0.000, 0/2 | −0.004, 0.067 |
| WTBlade 8% (4) | 0.092 | 0.087 | 0.086 | −0.006 [−0.013, −0.002], 0.005, 1/4 | −0.000, 0.296 |

Val agrees in direction on both datasets. Standard tta-mean also loses (WTBD −0.025, WTBlade −0.0055, 0/6 models better), and agreement adds nothing beyond it.

**Sanity check** (`runs/analysis/g21/sanity.py` → `sanity.log`): the unflip is correct. Every original box with s ≥ 0.2 finds a same-class mirrored twin, with median IoU 0.95 on WTBD and 0.92 on WTBlade. The twin's score is about half of s, which is regression to the mean from conditioning on high s; below s 0.05 the ratio is ≥ 1.

**Scope check** (`scope.py` → `scope.log`; `scope_1to1.py` → `scope_1to1.log`, post hoc):
- **The mirrored view alone is as good as the original.** Flip-only − original: WTBD test −0.016 and +0.017; WTBlade test −0.008 to +0.011; val within ±0.008. So fusion is losing despite two equally good views.
- **Why: many-to-one matching undoes DETR's duplicate suppression.** With max-IoU matching, an original box that duplicates a true box borrows its twin's high flipped score. It then climbs the ranking and counts as a false positive. Greedy one-to-one matching (original boxes in score order, each flipped box used at most once) recovers about half the loss:
  - tta-mean − original: WTBD −0.025 → −0.013, WTBlade −0.0055 → −0.0016;
  - it still never beats the original on either dataset (val or test);
  - agreement stays below tta-mean (WTBD test 0.142 vs 0.155).
- No scope.

**Reading:**
- Requiring agreement between views is not a ranking signal for RF-DETR at low labels.
- A more general recipe note: naive box-matched score-averaging TTA hurts an NMS-free set-prediction detector, because cross-view matching reintroduces the duplicates that one-to-one training suppressed. One-to-one matching halves the damage but does not turn it into a gain.
- With G20, this closes "rescore RF-DETR's own outputs" from inside the query set (G20) and from a second view of the same image (G21). The ranking deficit found in the error anatomy (37% of WTBD defects ranked < 0.1) is not fixable by label-free post-hoc rescoring of this kind.
- **G21 discarded** (§30).

### Crop re-classifier on RF-DETR candidates: discarded before any run (2 Nov)

**Proposal** (after G21): re-rank and re-label RF-DETR's top-k boxes with a classifier on frozen DINOv3 features of each box, trained only on the labelled crops.

**Already covered:**
- G15-R re-ranks with DINOv3 ViT-B box embeddings against the labelled boxes, using the fixed rule.
- H18 is the learned version over the same features.
- Together they give WTBD test +0.025 mAP50-95, but the result does not hold on WTBlade (H18-W, H19-W).

**What the proposal would add, and why each is moot:**
- **Re-labelling:** RF-DETR already emits a box per class hypothesis (the true class has median rank 1–2 in the error anatomy), so changing a box's class reduces to re-scoring an existing box.
- **A trained head instead of kNN prototypes:** this is H18's change, and the one that failed on WTBlade.
- **The WTBlade gap is not mainly ranking:** crack is 44% no box, even for ViT-B at 100%, so no re-scoring can reach it.

Not run. Taken together, label-free post-hoc rescoring (G20, G21) and exemplar rescoring (G15-R, H18) are closed. What remains of the low-label deficit needs a training-time or data-side change.

### WTBlade crack misses do not depend on the label budget (2 Nov; diagnostic, cached predictions)

`runs/analysis/extent/crack_nobox.py` → `crack_nobox.log`; `crack_nearmiss.py` → `crack_nearmiss.log`. All WTBlade test images are 640×640 (Roboflow export), so native-resolution inference has nothing to offer.

| detector | crack boxes without a same-class IoU ≥ 0.5 box |
|---|---|
| RF-DETR 8% (4 subsets) | 0.47 |
| RF-DETR 25% (4 subsets) | 0.49 |
| ViT-B 100% | 0.43 |

**The crack miss rate is essentially flat from 8% to 100% of labels.** Unlike WTBD's rare-class deficit, which is ranking, this one is not fixed by more labels or by any rescoring.

- **Geometry: compact, not thin.** Aspect ratio is 1.4.
  - Short side: found 52 px [36–72], missed 39 px [28–51].
  - Miss rate is 1.00 below 16 px (5% of cracks), 0.65 below 32 px (30%), 0.56 below 64 px, and 0.22 above.
- **What the misses are (RF-DETR 25%; 8% is the same):**
  - 34% have no crack box nearby (IoU < 0.1);
  - 25% have IoU 0.1–0.3;
  - 41% are near-misses at IoU 0.3–0.5;
  - only 10% are covered by another class at IoU ≥ 0.5.
- **The near-misses are offsets, not granularity:**
  - shifted 44%, larger box 29%, part 17%, merged with a neighbouring crack 11%;
  - median predicted/label area 1.2.
  - This differs from WTBD, where RF-DETR's extent matches the annotator ceiling and the misses are class confusion.
  - WTBlade has no independent re-annotations (its "same-surface" pairs are Roboflow copies with transformed labels), so whether the offsets are label noise cannot be tested here.

**Reading:** the low-label deficit has two forms, separable by diagnostics alone:
- **WTBD rare classes:** ranking (the box is there, scored low). Exemplar rescoring helps here (G15-R, H18).
- **WTBlade crack:** localisation of small compact cracks plus possible label offset. It is insensitive to the label budget, so it is a ceiling of the data or annotation rather than of the training recipe.

This accounts for H18 / H18-W / H19-W from the error side.

### Novelty audit of the surviving findings (2 Nov; web literature check, not exhaustive)

**Item 2, the instance-level proximity law (G16): partly prior art.**
- Its premise is the core-set assumption of active learning (Sener & Savarese 2018; instance-level core-set for detection, e.g. IDEAL-M3D 2025).
- Its design (the same test example tracked across random training subsets) is the datamodels design (Ilyas et al., ICML 2022), though only in classification.
- Nearest-neighbour distance to the labelled set as a per-sample accuracy estimator: OCCAM (arXiv 2406.04508), also classification.
- Found nothing that holds a test *object* fixed across training subsets in detection. Even so, the law alone is a detection instance of a known premise, not a paper.

**Item 1, fine-tuning moves the role of coverage from recall to ranking (G16 → G16-R): open.** Nearest work:
- LP-FT (Kumar et al., ICLR 2022): fine-tuning distorts pretrained features; classification, OOD;
- Attention-RPN (Fan et al., CVPR 2020): novel objects get low-scored good boxes;
- frozen-backbone prototype FSOD (arXiv 2606.23069, 2026);
- F-VLM: fine-tuning helps base classes, not novel ones.

None states that end-to-end fine-tuning decouples *whether* a defect is found from its proximity to labelled exemplars while keeping the dependence in its *score*, or uses that to predict when exemplar rescoring pays off.

**The weakness of item 1 as it stands:**
- WTBD only, with 4 subsets and one seed; the "flatter" test passed at P 0.957, just over the bar.
- The contrast was RF-DETR against BladeYOLO-v2 ViT-B, which differ in architecture and head as well as in fine-tuning.
- G22 below fixes both.

**Third dataset, from the local sources** (audit `runs/analysis/datasets/report.md`): **WTBs2025** (*Sci. Data* 13:1310, 2026) is the only candidate with enough independent scenes.
- 3,403 scenes from 4,145 unique images; 9 official classes.
- WindSurface has 81 scenes. Beijing and its check export share WTBD test photos.
- Caveats:
  - Small boxes: median 14–43 px at 640; pinholes 2 px, localized damage 6.5 px.
  - 782 of its images duplicate WTBlade images (1,371 same-scene), and must be removed from its test split for independence.
  - 114 GAN lightning images.
  - It ships no split.

It is prepared and pre-registered separately (G23) once G22 stage B has decided whether item 1 survives.

### G22: does fine-tuning move coverage from recall to ranking? Controlled replication (CPU + Kaggle). Pre-registered before computing (2 Nov)

**Claim under test (item 1).** With a frozen backbone, whether a test defect is found at all depends on its proximity to the labelled same-class defects. End-to-end fine-tuning removes that dependence for being found and keeps it for the score.

**Definitions** (G16/G16-R unchanged):
- x = max cosine, in frozen DINOv3 ViT-B box-embedding space, between the test box and the model's training-subset boxes of its class;
- y = highest same-class confidence at IoU ≥ 0.5 (0 if none);
- detected = y ≥ 0.05;
- y ~ x + log(1 + n) with test-box and subset fixed effects;
- cluster bootstrap over test scenes, 1,000 replicates.

**The control.** RF-DETR-M with `freeze_encoder=True`: the DINOv2 encoder is frozen; projector and decoder train. Recipe, subsets and seed are otherwise identical to the fine-tuned runs (100 epochs, batch 4 × accum 4, seed 0). Fine-tuning is then the only difference, which removes G16-R's architecture confound.

**Stage A (CPU, now, cached predictions):** WTBlade test, 8 fine-tuned RF-DETR models (f25 class-matched and rand1–3; f08 the same).
- **A1:** class-aware confidence slope > 0 with P(>0) ≥ 0.95, on f25 alone and on f25 + f08 pooled.
- **Reported:**
  - the binarised slope;
  - candidate recall (any same-class box at IoU ≥ 0.5, conf ≥ 0.001), so a ceiling effect of 300 queries can be judged.

**Stage B (Kaggle):** frozen-encoder RF-DETR on the 4 WTBD f25 subsets of G16-R and the 4 WTBlade f25 subsets (8 runs, `rfdetrmfz_*`). Per dataset, paired with the fine-tuned models on the same subsets and the same scene draws.
- **B1:** frozen binarised slope > 0 with P ≥ 0.95. Coverage governs recall without fine-tuning.
- **B2:** binarised slope, fine-tuned − frozen < 0 with P ≥ 0.95. Fine-tuning flattens it.
- **B3:** fine-tuned class-aware confidence slope > 0 with P ≥ 0.95. On WTBD this is G16-R's +0.478, recomputed; on WTBlade it is A1.

**Item 1 replicates** if B1–B3 hold on **both** datasets. If they hold on one dataset only, it is a dataset-specific observation and not a claim.

**Reported, not decisive:**
- The G15-R fixed-rule rescoring gain (all-class mAP50-95) on frozen vs fine-tuned models. Item 1 predicts a larger gain on the fine-tuned models, because only they find far-from-labelled defects to re-rank.
- The detection rate and mAP of the frozen models.

**Fail:** one scope check, then item 1 is dropped as a claim. The paper would then rest on the diagnostic decomposition alone.

### G22 stage A result: on WTBlade, fine-tuned RF-DETR's recall does depend on proximity (2 Nov)

`tools/g22_coverage.py stageA` → `runs/analysis/g22/stageA.md`. 748 test boxes, 177 scenes.

**Estimator check:** fixed effects absorbed by alternating demeaning (`check`) reproduce G16-R's dummy-variable OLS exactly, at +0.4775 and −0.0159.

| WTBlade, fine-tuned RF-DETR | detected / candidate rate | class-aware conf slope | detected (≥ 0.05) slope | candidate (≥ 0.001) slope |
|---|---|---|---|---|
| f25 (4) | 0.636 / 0.829 | +1.308 [+0.918, +1.768], P 1.000 | **+1.772 [+1.050, +2.625], P 1.000** | +0.605, P 0.996 |
| f25 + f08 (8) | 0.570 / 0.798 | +1.182 [+0.970, +1.378], P 1.000 | **+1.608 [+1.257, +2.059], P 1.000** | +0.763, P 1.000 |

- **A1 holds:** the confidence slope is positive.
- **But the claim's "fine-tuning removes the recall dependence" does not generalise.** On WTBD the fine-tuned detected slope was −0.016 [−0.886, +0.603]. On WTBlade it is +1.77 with P 1.000.
- Stage B still decides the weaker form, B2 ("fine-tuning flattens it"), with the frozen-encoder control on both datasets. The control is now more informative, not less.

**What this does line up with:**
- WTBD: recall independent of proximity, so far-from-labelled defects are found but ranked low, and exemplar rescoring helps (G15-R, H18).
- WTBlade: recall depends on proximity, so far defects are never found, and rescoring fails (H18-W, H19-W; crack 44–49% no box at any budget).

**Candidate (not a claim; two datasets are two points):** the fine-tuned detector's recall-vs-coverage dependence, measured on a held-out split, predicts whether exemplar rescoring pays off on that dataset. The test is a third dataset with the prediction made before rescoring is run (G23, WTBs2025). Cosine scales differ between datasets, so G23 must use a scale-free statistic. Defined at its pre-registration.

**Engineering:**
- `tools/rfdetr_v1.py train --freeze-encoder`;
- `tests/test_rfdetr_freeze.py` passes (encoder 0/223 parameters trainable, all 286 others trainable).
- "Frozen" means frozen at RF-DETR's COCO-pretrained encoder weights.

### Per-class check of the "diagnostic predicts rescoring payoff" candidate: no relation (2 Nov; exploratory, post hoc)

`runs/analysis/g22/perclass.py` → `perclass.log`. The per-class G15-R gains had already been seen (G15-R, H18-W, H19-W), so this is a consistency check, not a test. RF-DETR 25% on WTBD (6 classes) and WTBlade (7), plus WTBlade 8% (7).

Spearman correlation of each class's fixed-rule gain with:

| predictor | 13 classes | 20 rows |
|---|---|---|
| standardised detected-vs-proximity slope | −0.15 | −0.11 |
| near − far tercile detection gap | −0.16 | −0.18 |
| ranked-low share (error anatomy) | +0.01 | +0.11 |
| no-box share | −0.04 | +0.08 |

All p > 0.4.

- **Counterexample:** WTBD craze has the highest ranked-low share (0.63) and still loses under rescoring (−0.008). The gains are concentrated in thunderstrike (+0.052) and crack (+0.041).
- Per-class AP over about 50 boxes is noisy, so this cannot rule a relation out, but it gives the candidate no support.
- **The dataset-level match** (WTBD flat recall dependence with rescoring gain; WTBlade steep with no gain) **stays two points without a class-level mechanism.** Not a claim; G23's predictor role is dropped. Fail-fast: no scope at this level.

### WTBs2025 audit for G23: weak as a test bed (2 Nov)

The labels are complete; a naive `cat | awk` count under-reads them because files lack final newlines. Recount, one folder per class:
- coating detachment 3,495 boxes; erosion 4,526; lightning 354; localized damage 1,254; oil 563; paint cracks 503; pinholes 892; protective film 632; stains 298.

**Each folder labels only its own class.** Oil carries 11 paint-crack boxes and protective film 9 oil boxes. Any other defect type in a photo is unlabelled, and only 18 files appear in two folders.

**Boxes are mostly below one RF-DETR patch.** Median short side at 640 px:

| class | median short side | share ≥ 16 px |
|---|---|---|
| pinholes | 1.5 px | 0.05 |
| localized damage | 5 px | 0.06 |
| coating detachment | 9 px | 0.28 |
| protective film damage | 10.5 px | 0.32 |
| paint cracks, stains | 13.5 px | 0.42–0.43 |
| erosion | 17 px | 0.51 |
| oil | 27.5 px | 0.90 |
| lightning | 36 px | 0.88 |

7,024 of 7,544 images are 640×640, so no higher resolution exists.

**Consequence:** a detector on WTBs2025 fails mainly through tiny-object localisation, and its unlabelled co-occurring defects count as false positives. It can still serve as a replication point for G22-type questions on the large-box classes (oil, lightning, erosion). It cannot discriminate between hypotheses about ranking. Not prepared further until a question needs it.

### H20: is subset luck annotation-convention match? Pre-registered before computing (2 Nov)

**Idea.** H7 found subset quality real (test–val ρ 0.65) but invisible to every appearance and class-count statistic. A property H7 never measured is *how* a subset's labellers cut defects into boxes. Earlier phases found WTBD scores sensitive to box granularity: craze merged vs split (§9, §11), and 13–16% extent disagreement (§14). A detector learns the box convention of its subset. If that convention differs from the held-out labels' convention, correct detections are scored as errors.

**Statistics,** per subset S, per class c, over S's labelled boxes against a reference set's boxes of c:
- **G1 (primary):** 1-D Wasserstein distance between log box areas (px² at 640), averaged over the 6 classes. This is a granularity proxy: merged boxes are large, split ones small.
- **G2:** the same for log aspect ratio (w/h).
- **G3:** the same for the number of c-boxes per image containing c (split vs merged directly).
- Classes absent from S are skipped.

**References:** the test boxes (explanatory, uses test labels) and the val boxes (usable in practice, since val labels exist).

**Data:** the 16 H7 subsets with their test and val mAP50-95 (`runs/analysis/subset_properties/table.csv`).

**H20 passes** (G1) if all hold:
- (a) Spearman ρ(G1 vs test, test mAP50-95) ≤ −0.6 with two-sided permutation p < 0.05 (20,000 permutations);
- (b) ρ(G1 vs val, val mAP50-95) ≤ −0.4;
- (c) within the 8 random draws, ρ(G1 vs test, test) ≤ −0.5;
- (d) the practical version, ρ(G1 vs val, test mAP50-95), ≤ −0.4.

G2 and G3 are reported only.

**If it passes:** a pre-registered convention-matched subset selector on Kaggle (choose scenes whose box convention matches val), against random and cover. **If it fails:** one scope check, then closed.

### H20 result: fails; annotation-convention distance does not explain subset luck; closed (2 Nov)

`tools/h20_convention.py` → `runs/analysis/h20/report.md`. 16 H7 subsets.

| statistic | reference | ρ vs test score (perm p) | ρ vs val score | ρ within 8 random draws |
|---|---|---|---|---|
| **G1 log-area W1 (primary)** | test | **−0.27 (0.30)** | −0.35 | −0.29 |
| G1 | val | −0.26 (0.32) | −0.32 | −0.10 |
| G2 log aspect | test / val | −0.45 (0.08) / −0.49 (0.05) | −0.25 / −0.35 | −0.12 / −0.12 |
| G3 boxes per image | test / val | −0.41 (0.12) / −0.09 (0.75) | −0.43 / −0.21 | −0.31 / −0.02 |

- Conditions (a), (c) and (d) fail; **H20 fails.**
- All 12 correlations are negative, but they share the same 16 subsets, so that is not independent evidence.

**Scope check** (`runs/analysis/h20/scope.py` → `scope.log`): per-class convention distance against per-class test AP, across the 16 ViT-B subsets.
- **Craze**, the class where granularity was shown to matter: ρ +0.09 (area) and +0.11 (boxes per image).
- **Other classes are mixed:**
  - surface_injure boxes per image −0.54 (p 0.04), but thunderstrike +0.56 (p 0.03), in the opposite direction;
  - hide_craze area −0.45;
  - crack within the random draws −0.86, which is 8 points with no support from the full set (−0.17).
- No consistent per-class mechanism.

**Closed.** Subset luck is not visible in appearance (H7), class counts (H7), coverage (G16) or box convention (H20).

### From diagnoses to fixes: what is already done, and the crossed test (2 Nov)

| diagnosis | the fix it implies | status |
|---|---|---|
| WTBD rare classes: right box ranked low | exemplar rescoring | done: G15-R +0.011, H18 +0.025 on WTBD, nothing on WTBlade; the method is prior art |
| WTBlade crack: small compact cracks missed at any label budget | more pixels per defect | **G24 below** |
| flip TTA hurts DETR through many-to-one matching | one-to-one matching | done (G21 scope): halves the loss, no gain |
| subset luck | none (H7, G16, H20 all negative) | — |

No single fix is new. What can be new is a **double dissociation**: the diagnosis predicts which fix works where. Ranking regime: rescoring helps WTBD but not WTBlade, which is already measured. Localisation regime: resolution should help WTBlade more than WTBD.

### G24: input resolution as the localisation-regime fix (Kaggle). Pre-registered before any run (2 Nov)

**Runs:** RF-DETR-M at resolution **768** (default 576; tokens × 1.78), otherwise the V1 recipe:
- 100 epochs, seed 0;
- batch 2 × accum 8 (effective 16, as 4 × 4), to fit memory;
- on the same subsets as existing 576 runs: WTBD f25 (class-matched, rand1–3; `rfdetrmr768_*_wtbd_group_f25`) and WTBlade f08 (`rfdetrmr768_*_wtblade_group_f08`).

`tools/rfdetr_v1.py train --resolution 768`; predictions use the recorded resolution. `tests/test_rfdetr_resolution.py` passes.

**Prediction from the diagnosis** (WTBlade's deficit is localisation of small compact cracks; WTBD's is ranking):

| | condition |
|---|---|
| **R1** | WTBlade f08 test mAP50-95, 768 − 576 > 0 with paired scene bootstrap P ≥ 0.95 (2,000 replicates), better on ≥ 3/4 models |
| **R2** | WTBlade crack no-box rate (error anatomy) lower at 768 than 576 on ≥ 3/4 models |
| **R3** | relative gain (Δ / 576 mAP) larger on WTBlade than on WTBD, bootstrap P ≥ 0.90 |

- **Double dissociation confirmed** if R1–R3 hold. Together with the rescoring results, the diagnosis then picks the working fix on each dataset.
- **Confound, stated now:** WTBD images are 1024 px natively, so 768 gives WTBD real extra detail, while WTBlade (640 native) is upsampled. This works *against* R3.
- **Reported:** per-class Δ; WTBD crack/thunderstrike confusion; G15-R rescoring on the 768 models.

**Fail:** one scope check, then the resolution arm is closed and the dissociation is not claimed.

### G22-O: proximity in the detector's own feature space (CPU). Pre-registered before computing (2 Nov)

**Caveat being addressed:** G16-R and G22 measure proximity in frozen DINOv3 ViT-B space, not in the space the detector uses.

**Method:**
- For each RF-DETR model with a local checkpoint, a forward hook on its backbone projector output (stride 16, the decoder's input) gives a box embedding: the mean of the tokens whose centres lie in the box, L2-normalised.
- Embedded: test boxes, and the model's own training-subset boxes.
- x_own = max cosine to the subset's same-class boxes. Otherwise the G22 estimator is unchanged.

**Models:**
- WTBD f25 class-matched and rand3 (the other checkpoints are no longer downloadable);
- WTBlade f08, all 4;
- later, the stage-B frozen models.

**Readout** (descriptive, no decision depends on it): own-space vs DINOv3-space slopes (confidence, detected) on the same models. The dataset contrast of G22 stage A (WTBD detected slope ≈ 0; WTBlade > 0) **holds in own space** if WTBD's detected-slope interval includes 0 and WTBlade's is > 0 with P ≥ 0.95.

### G22-O result: in the detector's own feature space, recall depends on proximity on both datasets (2 Nov)

`tools/g22_ownspace.py` → `runs/analysis/g22/ownspace.md` (logs `own_embed.log`, `own_score.log`). Token pooling unit-checked. Same models, boxes and scene draws for both spaces.

| models | proximity space | ρ(own, DINOv3 proximity) | class-aware conf slope | detected (≥ 0.05) slope |
|---|---|---|---|---|
| WTBD f25 class-matched + rand3 | DINOv3 ViT-B | +0.36 | +0.758 [−0.187, +1.542], P 0.952 | −0.003 [−2.355, +1.014], P 0.519 |
| | **own (projector)** | | +4.186 [+0.741, +8.623], P 0.990 | **+11.167 [+1.521, +16.268], P 0.985** |
| WTBlade f08 (4) | DINOv3 ViT-B | +0.73 | +0.948, P 1.000 | +1.552 [+0.995, +1.996], P 1.000 |
| | own | | +5.856, P 1.000 | +10.222 [+8.796, +11.816], P 1.000 |

Own-space slopes are larger because own-space cosines span a narrower range; magnitudes are not comparable across spaces.

**Readout (pre-registered): the dataset contrast does not hold in own space.** WTBD's detected-slope interval excludes 0.

**Reading:**
- The WTBD "found regardless of proximity" result (G16-R, G22) is a statement about proximity in *generic* foundation-feature space only.
- In the space the fine-tuned detector actually uses, whether a defect is found depends on how close it is to labelled defects, on both datasets.
- The two spaces agree much less on WTBD (ρ 0.36) than on WTBlade (0.73). One coherent reading: fine-tuning on WTBD re-organises the feature space away from DINOv3's neighbourhoods. It changes *which* neighbours matter rather than removing the dependence.

**Caveat:**
- Own-space proximity is partly circular. A box the detector cannot find probably also has non-defect-like features in its own space, so a positive own-space slope is expected to some degree by construction.
- Only 2 WTBD models.

**Consequence for item 1:** the claim can at most be "fine-tuning decouples recall from generic-feature coverage", not "from coverage".
- G22 stage B (pre-registered in DINOv3 space) still decides that narrower form.
- Its frozen-encoder models also get the own-space check once their checkpoints are downloaded.

### G24 result: resolution does not fix WTBlade's localisation deficit; it helps WTBD instead; dissociation not confirmed (2 Nov)

Kaggle, 1 Nov:
- WTBlade f08 at 768 on account 1 (`kaggle_out_acc1`), about 67 min per run;
- WTBD f25 at 768 on account 2, notebook v41 (`kaggle_out_acc2`).

All `v1_options.json` record resolution 768, batch 2 × accum 8. `tools/g24_resolution.py` → `runs/analysis/g24/report.md`.

| test, family mean of 4 | 576 | 768 | Δ [95% CI], P(>0) | better | relative |
|---|---|---|---|---|---|
| WTBlade f08 | 0.0920 | 0.0907 | −0.0012 [−0.0091, +0.0056], 0.312 | 1/4 | −1.3% |
| WTBD f25 | 0.1890 | 0.1993 | **+0.0103 [−0.0022, +0.0378], 0.958** | 3/4 | +5.4% |

- **R1 ✗, R2 ✗** (WTBlade crack no-box lower on 1/4 models), **R3 ✗** (P(WTBlade > WTBD) 0.066).
- **WTBlade per class:** crack +0.008, every other class −0.002 to −0.007.
- **WTBD:** every class gains (hide_craze +0.032, crack +0.010).

**Scope check** (`runs/analysis/g24/scope.py` → `scope.log`): WTBlade crack no-box rate by size, 576 → 768.
- short side < 32 px: 0.58 → 0.60;
- 32–64 px: 0.46 → 0.54;
- ≥ 64 px: 0.33 → 0.28.

Upsampling a 640-px source does not help even the small cracks. No scope.

**Reading:**
- WTBlade's crack deficit is not fixed by labels (8% → 100%, earlier) or by resolution. It is a ceiling of the data or annotation.
- Resolution helps where real pixels exist (WTBD, 1024-px originals): the confound stated at pre-registration, and the larger effect.
- **The "diagnosis picks the fix" double dissociation is not confirmed.** The localisation-regime fix fails on the localisation-regime dataset. G24 closed.

### G22 stage B result: the frozen-encoder control confirms the effect on WTBD, not on WTBlade; item 1 does not replicate (2 Nov)

Kaggle, account 2, notebook v40 (downloaded from the web UI as `runs_v40.zip` → `runs/analysis/v1/kaggle_out_v40`). All 8 `rfdetrmfz_*` runs record `freeze_encoder: true`. `tools/g22_coverage.py stageB` → `runs/analysis/g22/stageB.md`.

| | fine-tuned detected slope | frozen detected slope | fine-tuned − frozen | B1 / B2 / B3 |
|---|---|---|---|---|
| WTBD (235 boxes, 58 scenes) | −0.016 [−0.886, +0.603] | **+0.898 [+0.040, +1.796], P 0.977** | **−0.914 [−1.934, −0.267], P(<0) 0.993** | ✓ ✓ ✓ |
| WTBlade (748 boxes, 177 scenes) | +1.772, P 1.000 | +2.050, P 1.000 | −0.278 [−1.030, +0.686], P(<0) 0.745 | ✓ ✗ ✓ |

**Item 1 does not replicate** (pre-registered: B1–B3 on both datasets).

- **On WTBD the controlled design reproduces G16-R without its architecture confound.** Freezing only the encoder restores recall's dependence on DINOv3 proximity.
- **On WTBlade it does not.**

**Reported numbers** (`stageB_map.py` → `stageB_map.log`):

| | detection rate (≥ 0.05), fine-tuned / frozen | test mAP50-95, fine-tuned / frozen |
|---|---|---|
| WTBD | 0.641 / 0.528 | 0.189 / 0.123 |
| WTBlade | 0.636 / 0.626 | 0.132 / 0.087 |

On WTBlade, fine-tuning raises mAP without raising recall: the gain is in scores and boxes.

**Scope check** (`stageB_scope.py` → `stageB_scope.log`, post hoc): where fine-tuning's added recall falls, by within-class proximity tercile.
- WTBD: far +0.13, mid +0.13, near +0.09; far − near +0.04, P 0.72. Not concentrated on far-from-labelled defects.
- WTBlade: +0.01 in every tercile.
- No scope for a reformulation.

**Item 1 dropped as a claim.** It stays a WTBD observation, now with a clean control, and a caveat from G22-O: it only holds for generic-feature proximity.

### G25: official-protocol benchmark against BladeYOLO, with the leakage reported beside it (decided 2 Nov)

**Direction.** Stay on wind-turbine blades. Report our strongest detector on the **official (published) protocols** for direct comparison with BladeYOLO (IEEE TGRS 2026) and the other published numbers. Report the leak-free results and every audit finding beside them. The leak-free columns are the honest caveat on all official numbers, ours included.

**Protocols** (each dataset's own published split, unchanged; `tools/rfdetr_v1.py prepare-official`, tested on all three layouts, no train/valid name overlap):

| dataset | split | published reference |
|---|---|---|
| WTBD-official | authors' seeded random split, train 745 / val 159 / test 161 | no detection baselines in the dataset paper. Ours on this split: BladeYOLO-v2 0.840 / 0.493, YOLO12-L 0.704 / 0.388 (§12) |
| Wind Surface Defect | public copy, train 2,856 / val 714 (the 4:1 protocol of [57] and BladeYOLO). Val is the test set; 10% of train (seed 0) is carved out for checkpoint selection, so the test set is never used for selection | BladeYOLO 86.5 / 57.4 mAP50 / mAP50-95 (paper Table V) |
| WTBlade-Defect | Roboflow split, train 9,010 / valid 1,097 / test 553 | BladeYOLO 85.8 / 74.6 (Table I, 3 seeds). **Protocol mismatch:** the paper re-split 5,326 images 9:1 (4,766 / 560) and published no image list, so its exact split cannot be rebuilt |

**Runs** (Kaggle):
- RF-DETR-M at **640 px**, the paper's input size for all methods (RF-DETR's default is 576), seed 0;
- 100 epochs on WTBD-official, plus a 768-px run (G24: resolution helps WTBD's 1024-px originals);
- 60 epochs on Wind Surface;
- 25 epochs on WTBlade-Defect (about 3 Roboflow copies per frame, so about 75 passes per source frame).

The paper trained BladeYOLO for 300 epochs with SGD at 640.

**Scored** with both the scene-bootstrap evaluator and the RF-DETR/COCO evaluator, for comparability with published figures. Per-class AP50 is reported.

**Reported with it:**
- the leak-free result of the same detector family on the same data (WTBD-group, WTBlade-group);
- the measured leakage of each official split (§4, §11);
- the audit findings: granularity, label agreement, the frozen-vs-fine-tuned control, inert modules, error anatomy.

**Not a hypothesis test.** It is a benchmark entry under the community's protocol. No claim of generalisation is made from the official numbers.

### G25 result: RF-DETR-M on the official protocols (3 Nov)

Kaggle, 2 Nov, three sessions; root zips `runs.zip`, `runs (1).zip`, `runs (2).zip` → `runs/analysis/v1/kaggle_out_g25`. All `v1_options.json` match the plan.

`tools/g25_official.py` → `runs/analysis/g25/report.md`. COCO 101-point evaluator; image-level bootstrap 95% CI.

| official protocol | RF-DETR-M, this work: mAP50 / mAP50-95 | published or earlier, same protocol |
|---|---|---|
| WTBD-official (test 161) | **90.5 [86.3, 94.4] / 60.2 [55.5, 65.3]** at 768; 87.3 / 60.1 at 640 | BladeYOLO-v2 84.0 / 49.3, YOLO12-L 70.4 / 38.8 (ours, Ultralytics, §12); the dataset paper has no detection baselines |
| Wind Surface Defect (val 714 = test) | 81.8 [78.0, 85.4] / 51.4 [48.3, 55.0] | **BladeYOLO paper 86.5 / 57.4**; our best leaky reproduction 78.2 / 46.9 |
| WTBlade-Defect (Roboflow test 553) | **87.4** [84.4, 90.9] / 52.5 [49.6, 56.2] | BladeYOLO paper 85.8 / **74.6**, its YOLOv12-L 83.3 / 71.9, on its own 9:1 re-split of 5,326 images |

**Per class AP50:**
- WTBD at 768: crack 99.8, craze 94.9, thunderstrike 93.1, surface_injure 90.6, hide_craze 85.7, corrosion 79.2.
- Wind Surface: dirt 92.2, surface_eye 90.7, hide_craze 86.8, thunderstrike 69.7, corrosion 69.5.
- WTBlade-Defect: burn 92.4, rust 92.3, oil 91.7, deformity 84.6, peeling 84.2, crack 83.5, dirt 83.3.

**RF-DETR's own validation mAP agrees** (91.1 / 63.2, 83.0 / 52.5, 89.1 / 52.4).

**Reading:**
- **WTBD-official:** best result on this protocol by +10.9 mAP50-95 over our BladeYOLO-v2. These are the first detection numbers on WTBD's published split from a foundation detector. Resolution 768 adds +3.2 mAP50 (crack +7.4, hide_craze +7.3), consistent with G24.
- **Wind Surface:** below the BladeYOLO paper (−4.7 / −6.0) and above our own reproduction of it. Our public copy has 3,570 images against the paper's 3,790, so the splits are not identical.
- **WTBlade-Defect:** mAP50 above the paper (+1.6). mAP50-95 is 22 points below.
  - The protocols differ (10,660-image Roboflow split vs the paper's unpublished 5,326-image re-split).
  - The paper's ratio mAP50-95 / mAP50 = 0.87 (YOLOv12-L 0.86) is far above every leak-free blade result we have measured (0.42–0.47). It fits near-duplicate Roboflow copies falling on both sides of a random split. This is a hypothesis; the paper's split is not available to test it.
- **Two runs ended while still improving:** Wind Surface's best validation epoch was 58 of 60, WTBlade-Defect's 24 of 25 (paper: 300 epochs). Their official numbers are lower bounds for this detector.
- **Evaluators differ:** ours is COCO 101-point; the paper's is presumably Ultralytics. They differed by 1–3 points on WTBD in §12.

**Still missing for the "leakage beside it" columns:** the same detector, at full data, on the leak-free splits (WTBD-group, WTBlade-group). Existing RF-DETR leak-free runs use 25% or 8% of the labels only.

### G26: a tuned RF-DETR recipe for all three official benchmarks. Pre-registered before any run (3 Nov)

**Goal (user, 3 Nov):** match or beat BladeYOLO on every official protocol, even though the leaky protocols partly reward memorisation; the leak and the leak-free twins are reported beside every number.

**Finding that motivates it: no RF-DETR run so far ever decayed its learning rate.** RF-DETR 1.11.2's default schedule is a 10× step after `lr_drop = 100` epochs (`current_step < lr_drop * steps_per_epoch`), so runs of ≤ 100 epochs train at a constant 1e-4 throughout. The logged `train/lr` of the Wind Surface run is 1e-4 at every epoch. This applies to every RF-DETR run in §31–G25; the comparisons between them stay like for like, but the late-training gain of an annealed rate was never used.

**Code:** `rfdetr_v1.py train --lr-schedule {step,cosine} --lr-drop N` (function `schedule_kwargs`, recorded in `v1_options.json`); `predict` now reads the model size from `v1_options.json` (Large runs). `tests/test_rfdetr_schedule.py` (RF-DETR's own `_build_managed_scheduler`): default constant through epoch 99; `--lr-drop 75` drops at 75; cosine 1.00 → 0.50 → 0.000. The resolution and freeze tests still pass.

**Measured Kaggle speed (single GPU, from the event files):** Medium at 640: WTBD-official 1.2 min/epoch, Wind Surface 4.2, WTBlade-Defect 14.6; Medium at 768 on WTBD 1.8.

**Recipe changes, one session per protocol (12 h limit):**

| protocol | G25 recipe | G26 recipe |
|---|---|---|
| WTBD-official | M, 768, 100 ep, constant LR | (a) M, 768, 100 ep, cosine; (b) **Large**, 704 (native), 100 ep, cosine |
| Wind Surface | M, 640, 60 ep, constant | M, 640, 100 ep, cosine (≈ 7 h) |
| WTBlade-Defect | M, 640, 25 ep, constant | M, 640, 40 ep, cosine (≈ 9.7 h) |

**Selection rule:** model selection on validation only (WTBD: official val; Wind Surface: the seeded 10% carved from train; WTBlade-Defect: Roboflow valid). If Large beats Medium on WTBD validation mAP50-95, Large is used for the other two in a later round; the test set is scored once per run, never for selection.

**Reporting rule:** the final recipe is then trained on the leak-free splits (WTBD-group, WTBlade-group, 100% labels), so each official number has a same-recipe leak-free twin. No tuning on the leak-free splits.

### G26 result, sessions 1–2: the annealed schedule adds about +1–2 mAP50-95 and nothing at mAP50; Large does not help on test (3 Nov)

Kaggle, 3 Nov: session 1 (WTBD-official, Medium 768 cosine, then Large 704 cosine) and session 2 (Wind Surface, Medium 640, 100 epochs, cosine). Root zips extracted to `runs/analysis/v1/kaggle_out_g26/{wtbd,ws}`, then deleted. All `v1_options.json` match the plan. The logged LR now anneals (1e-4 → 5e-5 at mid-run → ~0).

Scored with `tools/g25_official.py g26` → `runs/analysis/g26/report.md` (same evaluator and bootstrap as G25).

| protocol | run | val mAP50 / mAP50-95 (best epoch) | **test** mAP50 / mAP50-95 [95% CI] |
|---|---|---|---|
| WTBD-official | M 768, constant LR (G25) | 90.8 / 62.4 (38) | 90.5 / 60.2 |
| | **M 768, cosine** | 92.5 / 62.0 (48) | **90.4** [85.8, 94.7] / **61.2** [56.4, 66.2] |
| | L 704, cosine | 90.3 / **63.4** (89) | 88.9 [84.8, 93.3] / 59.5 [54.6, 65.3] |
| Wind Surface | M 640, 60 ep, constant (G25) | 83.0 / 52.5 (58) | 81.8 / 51.4 |
| | **M 640, 100 ep, cosine** | 82.1 / 54.4 (93) | **81.0** [77.0, 84.9] / **53.3** [50.0, 57.0] |
| | BladeYOLO paper | | 86.5 / 57.4 |

**Reading:**
- **The schedule fix is worth +1.0 (WTBD) to +1.9 (Wind Surface) mAP50-95 and nothing at mAP50** (−0.1, −0.8). Every change lies inside the bootstrap intervals. Annealing tightens boxes; it does not find more defects.
- **Wind Surface: the gap to the paper is not a training-length or schedule problem.** It stays at −5.5 mAP50 / −4.1 mAP50-95. My extrapolation from the 60-epoch curve (test ≈ 83–85 / 54) overestimated mAP50. The weak classes are unchanged: corrosion 66.7, thunderstrike 68.7 AP50.
- **Large vs Medium:** Large wins on validation mAP50-95 (+1.4) and loses on test (−1.7 mAP50-95, −1.5 mAP50). The two splits disagree in sign by margins inside the intervals: no evidence that Large helps. Thunderstrike (99.1) and crack (99.7) improve, while corrosion, surface_injure and hide_craze fall.
- **Pre-registered selection rule:** it formally selects Large (validation mAP50-95 0.634 > 0.620). Running Large on Wind Surface and WTBlade-Defect costs about twice the GPU time for an effect that does not survive the test split. This is noted as a decision point, not taken silently; dropping a run cannot inflate any reported number.
- **Best official numbers so far (single seed):** WTBD-official 90.4 / 61.2 (M 768 cosine); Wind Surface 81.0 / 53.3 (cosine) or 81.8 / 51.4 (constant), depending on the metric. WTBlade-Defect, session 3 (A100, bf16 AMP rather than Kaggle's fp16), is still running.

### G26 result, session 3: WTBlade-Defect follows the same pattern (4 Nov)

Run on the user's A100 (NGC container, torch 2.7.1+cu128, numpy < 2, bf16 AMP; Kaggle runs used fp16 on T4/P100), not on Kaggle. Zip `rfdetrm640cos_wtblade_defect_official.zip` → `runs/analysis/v1/kaggle_out_g26/wtblade`, then deleted. `v1_options.json` matches the plan (Medium, 640, 40 epochs, cosine, 4 × 4); the LR anneals 1e-4 → 5e-5 → ~0.

| run | val mAP50 / mAP50-95 (best epoch) | **test** mAP50 / mAP50-95 [95% CI] |
|---|---|---|
| M 640, 25 ep, constant (G25) | 89.1 / 52.4 (24 of 25) | 87.4 [84.4, 90.9] / 52.5 [49.6, 56.2] |
| **M 640, 40 ep, cosine** | 89.4 / 55.7 (39 of 40) | **87.2** [83.9, 90.6] / **55.1** [52.0, 59.1] |
| BladeYOLO paper (own 9:1 re-split of 5,326 images) | | 85.8 / 74.6 |

Per class AP50 (cosine): rust 93.8, burn 92.2, oil 89.5, deformity 87.2, crack 83.2, dirt 82.9, peeling 81.9.

**G26 summary over the three protocols:** the annealed schedule adds +1.0, +1.9 and +2.6 mAP50-95 (WTBD-official, Wind Surface, WTBlade-Defect) and −0.1, −0.8, −0.2 mAP50. The direction is consistent; each single change lies within its interval. Against BladeYOLO's paper, a generic detector with a standard recipe:
- is ahead at mAP50 on WTBlade-Defect (+1.4);
- is behind on Wind Surface (−5.5 / −4.1);
- is 19.5 points behind at mAP50-95 on WTBlade-Defect, a protocol the paper's unpublished re-split makes non-comparable (its mAP50-95 / mAP50 ratio 0.87 vs ours 0.63).

### Paper preparation, local analyses: evaluator check, leaked vs no-twin test images, and what the G26 tuning bought (4 Nov; CPU, cached predictions, post hoc)

GPU work stops here (user, 4 Nov); the paper is written from the runs that exist. Two local analyses on the official-protocol predictions. Both are **post hoc** (not pre-registered); they are reported as measurements.

**Tool:** `tools/official_leak_eval.py` → `runs/analysis/paper/official.{md,json}`; `... paired` → `paired.{md,json}`. Tests: `tests/test_official_leak_eval.py`:
- the numpy matcher is identical to Ultralytics 8.4's `DetectionValidator.match_predictions` on 200 random cases;
- the weighted bootstrap (image multiplicities over a once-sorted detection list) equals scoring the resampled multiset (≤ 2e-3).

**1. Evaluator: settled, the AP formula does not matter.** On identical predictions, Ultralytics-style (`ap_per_class`) and our COCO-style AP agree within 0.4 points on every run and subset (e.g. RF-DETR-M 768 cosine on WTBD-official: 90.7 / 61.4 vs 90.4 / 61.2).
- The 1–2.5-point gap between §12's logged Ultralytics `val` numbers (BladeYOLO-v2 84.0 / 49.3) and the same models' cached predictions (81.6 / 48.0) comes from `val`'s own inference path (rectangular letterbox batches), not from the metric.
- So the RF-DETR official numbers are directly comparable to Ultralytics-reported figures: the paper's 86.5 / 57.4 (Wind Surface) and 85.8 / 74.6 (WTBlade-Defect) gaps are not an evaluator artefact.

**2. Leak flags per official test image** (a twin among the images the run actually trained on):

| protocol | criterion | leaked | no twin | other |
|---|---|---|---|---|
| WTBD-official | same scene (DINOv2 + ORB/RANSAC, `runs/analysis/wtbd_leak/test_leak.csv`) | 125 / 161 (78%) | 36 | |
| Wind Surface | pixel twin: corr ≥ 0.99 at 48 px over the 8 flips/rotations, against the 2,570 trained images (carve excluded) (`tools/ws_pixel_twins.py`) | 600 / 714 (84%) | 114 | (flips only, as in §4: 532; with rotations 611 against all train) |
| WTBlade-Defect | same scene group (`runs/analysis/wtblade/split_groups.csv`) | 499 / 542 grouped (92%) | 43 | 360 are Roboflow copies of a train frame; 11 WTBD-matching images ungrouped, left out |

The Wind Surface "no twin" images have no near-identical copy in train, but most still come from drone photos that have other tiles in train (§4), so it is the cleanest subset available, not a leak-free one.

**3. Score by subset** (coco evaluator, image bootstrap 95% CI; full tables with per-class AP and the ultra column in `official.md`):

| protocol, run | all | leaked | no twin |
|---|---|---|---|
| WTBD-official, RF-DETR-M 768 cosine | 90.4 / 61.2 | 91.5 / 63.9 | 81.6 [69.1, 93.2] / 49.9 [40.4, 61.1] |
| WTBD-official, BladeYOLO-v2 | 81.5 / 48.1 | 82.6 / 48.9 | 81.5 / 47.8 |
| Wind Surface, RF-DETR-M cosine | 81.0 / 53.3 | 83.3 / 55.1 | **59.2 [49.0, 68.1] / 37.6 [29.9, 45.4]** |
| WTBlade-Defect, RF-DETR-M cosine | 87.2 / 55.1 | 90.8 / 59.4 | **50.4 [38.5, 64.0] / 17.2 [13.5, 23.4]** |
| WTBlade-Defect, copies of a train frame (360) | | **98.2 / 69.8** | |
| WTBlade-Defect, same scene, not a copy (139) | | 65.7 / 30.1 | |

- **WTBlade-Defect:** a test image that is a Roboflow copy of a training frame scores 98.2 mAP50; one without a same-scene train image scores 50.4. The official number is mostly the copies. The copies' mAP50-95 / mAP50 ratio is 0.71 (no twin 0.34). This supports the hypothesis that the paper's 0.87 ratio on its own random re-split reflects copies across the split (still not testable without its split).
- **Wind Surface:** −24 mAP50 / −18 mAP50-95 from leaked to no-twin. Thunderstrike AP50 on no-twin images is 3 (12 boxes); corrosion 42.
- **WTBD-official:** the no-twin subset has 1 box each of craze, thunderstrike and crack, so its mean is unstable (as in §11). Only corrosion falls clearly (AP50 97 leaked → 54 no twin).

**4. What the G26 tuning bought: memorisation** (paired image bootstrap, cosine/longer − G25 recipe, percent points; interaction = Δ leaked − Δ no twin, subsets resampled independently):

| protocol | Δ mAP50-95, all | leaked | no twin | interaction [95% CI], P(>0) |
|---|---|---|---|---|
| WTBD-official (cosine − constant, M 768) | +1.0 [−1.2, +3.0] | +1.8 [−0.1, +4.0] | **−8.1 [−13.4, +0.2]** | +9.9 [+1.1, +15.8], 0.993 |
| Wind Surface (100 ep cosine − 60 ep constant) | **+1.8 [+0.5, +3.1]** | +2.3 [+0.9, +3.8] | −1.5 [−4.2, +1.4] | +3.8 [+0.7, +7.0], 0.988 |
| WTBlade-Defect (40 ep cosine − 25 ep constant) | **+2.6 [+1.1, +3.9]** | +3.5 [+2.0, +5.0]; copies +5.4 [+3.6, +7.0] | **−4.0 [−7.2, +0.1]** | +7.4 [+3.0, +11.1], 1.000 |

- Paired, the G26 gains on the whole official test are real on Wind Surface and WTBlade-Defect (the unpaired G26 intervals hid that).
- **On every protocol the gain sits on the leaked images, and the same change lowers the score on the images without a twin.** The interaction excludes zero on all three. Annealing and longer training tighten boxes on what the model has effectively seen; on new images they do not help and probably hurt.
- Confound: on Wind Surface and WTBlade-Defect the recipe also trained longer (60→100, 25→40 epochs); on WTBD only the schedule changed, and the pattern is the same.
- **Consequence:** tuning a recipe against a leaky benchmark's numbers selects for memorisation, measurably, in a single tuning step. This is the strongest single finding for the paper's argument, and it applies to our own G26 numbers. The G26 rows stay in the paper, labelled as what they are.

**Other paired comparisons (WTBD-official):**
- RF-DETR-M 768 cosine − BladeYOLO-v2: +13.2 [+8.2, +17.4] mAP50-95 on all images, but +2.1 [−5.3, +19.8] on the 36 no-twin images. The foundation detector's lead on the official split is not established on unseen images.
- BladeYOLO-v2 − YOLO12-L: +10.3 overall and +12.6 [−0.6, +15.6] (P 0.965) on no-twin images. That ordering survives, consistent with the leak-free result (§12).

**Paper materials** (from this point on, the write-up):
- `paper/README.md`: framing (benchmark audit with a strong baseline, not a module paper), outline mapped to evidence, abstract draft, limitations, and the two optional GPU runs (RF-DETR full-label leak-free twins);
- `paper/tables.md`: T1–T10 with sources;
- `paper/figures/` from `tools/paper_figures.py`.

### G27: one RF-DETR recipe for leak-free and official protocols, with two training-time additions. Pre-registered before any G27 run (4 Nov)

**Goal (user, 4 Nov):** the paper says that the field reports leaked benchmarks, that a generic detector (RF-DETR) matches the published models on them, and what leak-free scores look like. It should also show how we tried to improve those scores, ideally with one recipe that beats the other architectures on both the leak-free and the official protocols, even by a small margin.

**Why these additions, and not modules.** The leak-free diagnostics say where RF-DETR loses points:
- extent is at the annotator ceiling;
- candidate recall is already 0.83–0.92;
- the loss is the scoring of rare classes (error anatomy, 1 Nov);
- label-free and exemplar post-hoc rescoring are closed (G15-R, H18, G20, G21).

What is left must change what the model learns from. The encoder has never seen blade imagery: DINOv3 was frozen in every BladeYOLO run, and RF-DETR starts from generic DINOv2 → COCO.

**Components:**
- **CP, recipe component** (`tools/g27_paste.py`): H10's matched copy-paste, generalised. In each protocol's training set, every class below the median class count gets min(median − n, n) extra boxes. These are real boxes of that class from the same training set, at most 2 per new image, centred on a SAM 2.1 blade pixel. Plans are committed (`tools/splits/g27_cp_<protocol>.csv`):

  | protocol | new images | boxes added |
  |---|---|---|
  | WTBD-group | 45 | 89 (thunderstrike +66, crack +23) |
  | WTBD-official | 50 | 95 |
  | Wind Surface | 316 | 630 (surface_eye +305, thunderstrike +325) |
  | WTBlade-group | 168 | 332 (burn +141, deformity +42, oil +149) |
  | WTBlade-Defect | 973 | 1,941 (burn +911, deformity +202, oil +828; phase B only) |

  Visual check: some pastes land off the blade where SAM's largest mask includes sky, and box-level pastes carry their source background (as in H10). Kept as H10 ran it; the A1 − A0 comparison measures it.
- **D1, in-domain self-supervised encoder adaptation** (`tools/g27_ssl.py`):
  - Masked feature distillation (data2vec-style) on RF-DETR-M's own DINOv2-S encoder, starting from the COCO checkpoint.
  - An EMA teacher sees the crop. The student sees it with 50% of its 16-px patches replaced by the encoder's mask token (block masking), and regresses the teacher's mean layer-normalised stage-3/6/9/12 features at the masked patches.
  - 8,000 steps × 16 crops at 576, lr 5e-5 cosine (500 warm-up steps), EMA 0.999 → 0.9999.
  - The teacher encoder is written back into the COCO checkpoint; projector, decoder and heads stay COCO's.
  - The pool's labels are unused.
- **D2, multi-dataset supervised stage 1 with partitioned class slots** (`tools/g27_slots.py`, `rfdetr_v1.py train --slot-mask`):
  - RF-DETR-M trained on the pool with every dataset's official classes in its own block of slots (34 slots).
  - Same-named classes are not merged, because conventions differ (§10).
  - Each image's classification loss is restricted to its own block. Logits outside it are filled with −1e4 before the criterion, so matching and every loss see p = 0 there (Zhou et al., CVPR 2022).
  - 12 epochs, cosine, 576, 4 × 4, seed 0, COCO init.
  - The final EMA weights (`last_ema.pth`, no selection) initialise the fine-tunes; the head is re-initialised to the target classes.
- **D1 and D2 use the same pool images:** images only for D1, images plus labels for D2.

**Pools** (`tools/g27_pool.py`; the twin rule is in its docstring):
- **Pool L, for the leak-free protocols, is scene-level clean** against WTBD-group and WTBlade-group val + test:
  - WTBD images only from WTBD-group train;
  - WTBlade images only from scene groups of WTBlade-group train;
  - all other sources minus anything matching a val/test image: same file, DINOv2 ≥ 0.90, ORB ≥ 25 inliers (evaluation-side top 30 with cosine ≥ 0.70, pool-side top 3 with cosine ≥ 0.80), pixel correlation ≥ 0.97 over 8 flips/rotations, and the closure of all of these.
  - The exclusion thresholds sit below the audit's twin thresholds (DINOv2 0.93, pixel 0.99) on purpose (next bullet).
- **Pool O, for the official protocols, is image-level clean** against their val/test (and the Wind Surface carve). It is built only if a component enters the recipe.
- **How the rule got there (logged, not hidden):**
  - The first all-protocol pool (DINOv2 + ORB only) kept 830 pixel twins, which the independent audit found.
  - With pixel twins added, the audit's closest remaining pairs were still neighbouring video frames and re-exposed copies of evaluation scenes. Hence the scene rule and the pool-side ORB.
  - With those, at exactly 0.93 / 0.99 the closest kept pairs still included a brightened WTBD crack copy (pixel 0.98) and re-tinted or neighbouring drone frames (cosine 0.92–0.93). Hence the margin, 0.90 / 0.97.
  - Archived versions: `runs/analysis/g27/{g27_pool_v1_dino_orb_only.csv, g27_pool_v2_all_protocols_pixel.csv, archive/g27_pool_L_v3_093_099.csv}`.
- **Final pool L** (`tools/splits/g27_pool_L.csv`, `runs/analysis/g27/pool_L.md`): **22,318 images, 45,281 boxes, 34 slots**.

  | source | kept |
  |---|---|
  | WTBD | 692 |
  | WTBlade | 6,303 |
  | Wind Surface | 2,404 |
  | Beijing | 7,771 |
  | WTBs2025 | 5,148 |

- **Audit** (`pool_L_audit.md`, contact sheets `pool_L_audit_{pixel,dino}.jpg`):
  - max pixel correlation to any evaluation image 0.969, max DINOv2 cosine 0.900; no kept image at the audit's twin thresholds;
  - by eye, the closest pairs are different views of similar blades from the same wind farms. A few crack close-ups may be the same physical defect from another flight, which cannot be settled without capture metadata. Hence the proximity check below.
- **Export for Kaggle:** `tools/g27_pool.py export --pool L --out g27-pool-L` (1.3 GB, gitignored). Every label is in its own dataset's slot block (checked).

**Phase A: leak-free protocols, 100% labels** (RF-DETR-M; WTBD-group at 768 for 100 epochs, WTBlade-group at 640 for 60 epochs; cosine, 4 × 4):

| arm | recipe |
|---|---|
| A0 | G26 recipe, COCO init (the missing full-label leak-free RF-DETR twin, `paper/README.md` §5) |
| A1 | A0 + CP |
| A2 | A1 + D1 init |
| A3 | A1 + D2 init |

- **A-1 (seed 0, both datasets, all 4 arms; validation only).**
  - R = the arm with the highest mean val mAP50-95 over the two datasets, among the arms that beat A0 on val on both datasets (point estimates, paired scene bootstrap reported). If none does, R = A0.
  - This replaces a per-component rule written earlier the same day, before any run: if CP failed while a D passed, that rule asked for an "A0 + D" arm that is not run.
  - Each component's own val contrast is reported: CP is A1 − A0, D1 is A2 − A1, D2 is A3 − A1.
  - A combination (D2 stage 1 started from D1) is a later, separately registered step.
  - Scorer, written before any run: `tools/g27_score.py` → `runs/analysis/g27/score.md`.
- **A-2 (seed 1):** A0 and R on both datasets.
- **Primary test claim (scene bootstrap, paired, mean of the two seeds):**
  - R − A0 mAP50-95 > 0 on each dataset;
  - the mean over the two datasets has P(> 0) ≥ 0.95.
  - Secondary, reported: each component's seed-0 test effect, mAP50, per class, and the error anatomy (does D1/D2 move ranked-low defects to hits?).
  - **Pool-proximity check (reported with the claim):** R − A0 on the test images split at the median of their highest DINOv2 cosine to any pool L image. If the gain sits only on the near half, it is reported as possible leakage through the pool, not as a method gain.
- **Other architectures on the same leak-free tests:**
  - WTBD-group: BladeYOLO-v2 (ViT-S), YOLO12-L, and the no-DINOv3 ablation (existing runs);
  - WTBlade-group: BladeYOLO-v2 ViT-B (existing), plus one new YOLO12-L run (Ultralytics, the §12 recipe).
  - "Beats the other architectures" means R's test mAP50-95 point estimate is above every one of them on both datasets, with the paired CI reported.

**Phase B: official protocols, after A-2.**
- R is trained once on WTBD-official, Wind Surface and WTBlade-Defect with the G26 per-protocol settings. D1/D2 use pool O if accepted, so official numbers never see an official evaluation image.
- No selection on test.
- Reported:
  - beside the G26 RF-DETR rows, the BladeYOLO paper and our YOLO baselines;
  - with the leaked / no-twin breakdown (`official_leak_eval.py`). G27 changes are judged on the no-twin subset as much as on the total, given that tuning on these splits rewarded memorisation (previous section).

**Finding while building the pool: a small residual leak in our own leak-free splits** (`tools/split_pixel_audit.py`, `tools/split_rescore.py` → `runs/analysis/split_audit/`).
- The pixel-twin test (48 px grey, 8 flips/rotations, ≥ 0.99), run between each split's held-out images and its own train, finds:

  | split | test | val |
  |---|---|---|
  | WTBD-group | 6 / 161 | 6 / 159 |
  | WTBlade-group | 1 / 332 | 1 / 323 |

- All 6 WTBD-group test pairs are real copies by eye: re-exposed, darkened, tinted, or rotated 90°. WTBD ships photometrically augmented copies (§14), and DINOv2 + ORB grouping does not catch them.
- **Effect** (`runs/analysis/split_audit/rescore.md`, every cached test prediction rescored without those images):
  - WTBD-group, 116 runs: mAP50-95 −0.62 points on average (range −1.09 to +0.33); mAP50 −0.70 (−1.40 to +0.69);
  - WTBlade-group, 20 runs: ±0.03.
- So our leak-free numbers are inflated by under a point. This goes into the paper's limitations, with the pixel test recommended next to scene grouping when building splits.
- The splits are not changed, so every earlier comparison stays like for like. G27 reports WTBD-group test both with and without the 6 images.

**Not in scope:** new architectures; 25% labels (the pools include target training scenes); a transductive arm (test images in the D1 pool), which is listed as an optional leakage measurement only.

**Tests (CPU, all pass):**
- `tests/test_g27_ssl.py`: block masks exact; the mask wrapper only changes masked calls and is removed afterwards; a 3-step run writes a checkpoint RF-DETR loads, with the encoder changed and everything else COCO's.
- `tests/test_g27_slots.py`, against RF-DETR's real `SetCriterion` (train mode, group DETR 13, aux + encoder layers): other blocks get exactly zero gradient and do not change the loss, while the image's own block trains.
- `tests/test_g27_train.py`, end to end:
  - stage 1 with the slot mask inside RF-DETR's training loop, keeping `last_ema.pth` (14 outputs);
  - a 6-class fine-tune from it with a 7-output head;
  - copy-paste into a prepared dataset (45 images, 89 boxes, refuses a second paste).

  This surfaced one bug, now fixed. RF-DETR keeps a checkpoint head that is smaller than COCO's unless `num_classes` is explicit, so `train --init` now passes the dataset's class count.

### G27 Kaggle commands (phase A)

**Inputs and timings.**
- Private Kaggle datasets, uploaded with the Kaggle CLI on 6 Nov (`--dir-mode zip`; Kaggle unpacks each folder):
  - `beegee11/g27-pool-l`: the `g27-pool-L/` export (`data.yaml`, `slots.json`, `train/ valid/ test/`, `g27_pool_L.csv`).
  - `beegee11/blade-group-splits`: `external/WTBlade-group/` unchanged, at the dataset root (`data.yaml`, `groups.csv`, `views_*.json`, `train/ val/ test/`).
  - WTBD-group is the existing `beegee11/wind-surface-defect` dataset's `WTBD-group/` (file list and `data.yaml` identical to `external/WTBD-group/`, checked 6 Nov).
- `<S1 output>` is session 1's output, attached as input (the only placeholder left).
- Rough timings on a T4: SSL about 1 h; stage 1 about 6 h; WTBD-group run about 3 h; WTBlade-group run about 3.5 h.

**Session 1, pretraining (D1 and D2 on pool L, about 7–8 h):**
```
!python tools/g27_ssl.py --images /kaggle/input/datasets/beegee11/g27-pool-l/train/images --out runs/ssl/g27_d1_L --steps 8000
!python tools/rfdetr_v1.py train --dataset /kaggle/input/datasets/beegee11/g27-pool-l --out runs/detect/BladeYOLO_WindSurface/rfdetrm576_g27_stage1_L --size medium --epochs 12 --batch 4 --accum 4 --lr-schedule cosine --slot-mask /kaggle/input/datasets/beegee11/g27-pool-l/slots.json --keep last_ema.pth
```
Both steps read the pool straight from the read-only input; RF-DETR writes nothing into its dataset folder. Copying the pool to `/kaggle/working` first (the original command, 46,567 files) is silent for minutes, and interrupting it does not stop the cell: IPython ends only the running `!` command and goes on, so the later steps would run on a partial pool. That happened in the first S1 attempt (6 Nov), which was restarted without the copy.
Its outputs are `runs/ssl/g27_d1_L/rfdetrm_g27ssl.pth` (D1) and `runs/detect/BladeYOLO_WindSurface/rfdetrm576_g27_stage1_L/last_ema.pth` (D2).

**Prepared datasets (in any session that needs them):**
```
!python tools/rfdetr_v1.py prepare-official --data /kaggle/input/datasets/beegee11/wind-surface-defect/WTBD-group --out /kaggle/working/wtbd_group_rf
!python tools/rfdetr_v1.py prepare-official --data /kaggle/input/datasets/beegee11/wind-surface-defect/WTBD-group --out /kaggle/working/wtbd_group_cp
!python tools/g27_paste.py paste --protocol wtbd_group --dataset /kaggle/working/wtbd_group_cp
!python tools/rfdetr_v1.py prepare-official --data /kaggle/input/datasets/beegee11/blade-group-splits --out /kaggle/working/wtblade_group_rf
!python tools/rfdetr_v1.py prepare-official --data /kaggle/input/datasets/beegee11/blade-group-splits --out /kaggle/working/wtblade_group_cp
!python tools/g27_paste.py paste --protocol wtblade_group --dataset /kaggle/working/wtblade_group_cp
```

**Arms.** WTBD-group runs use `--resolution 768 --epochs 100`; WTBlade-group runs use `--resolution 640 --epochs 60`. Every run uses `--size medium --batch 4 --accum 4 --lr-schedule cosine`. Run names are fixed in `tools/g27_score.py`.
```
# A0
!python tools/rfdetr_v1.py train --dataset /kaggle/working/wtbd_group_rf --out runs/detect/BladeYOLO_WindSurface/rfdetrm768cos_wtbd_group_full --size medium --resolution 768 --epochs 100 --batch 4 --accum 4 --lr-schedule cosine
!python tools/rfdetr_v1.py train --dataset /kaggle/working/wtblade_group_rf --out runs/detect/BladeYOLO_WindSurface/rfdetrm640cos_wtblade_group_full --size medium --resolution 640 --epochs 60 --batch 4 --accum 4 --lr-schedule cosine
# A1 (copy-paste)
!python tools/rfdetr_v1.py train --dataset /kaggle/working/wtbd_group_cp --out runs/detect/BladeYOLO_WindSurface/rfdetrm768cos_cp_wtbd_group_full --size medium --resolution 768 --epochs 100 --batch 4 --accum 4 --lr-schedule cosine
!python tools/rfdetr_v1.py train --dataset /kaggle/working/wtblade_group_cp --out runs/detect/BladeYOLO_WindSurface/rfdetrm640cos_cp_wtblade_group_full --size medium --resolution 640 --epochs 60 --batch 4 --accum 4 --lr-schedule cosine
# A2 (copy-paste + D1)
!python tools/rfdetr_v1.py train --dataset /kaggle/working/wtbd_group_cp --out runs/detect/BladeYOLO_WindSurface/rfdetrm768cos_cp_d1_wtbd_group_full --size medium --resolution 768 --epochs 100 --batch 4 --accum 4 --lr-schedule cosine --init <S1 output>/runs/ssl/g27_d1_L/rfdetrm_g27ssl.pth
!python tools/rfdetr_v1.py train --dataset /kaggle/working/wtblade_group_cp --out runs/detect/BladeYOLO_WindSurface/rfdetrm640cos_cp_d1_wtblade_group_full --size medium --resolution 640 --epochs 60 --batch 4 --accum 4 --lr-schedule cosine --init <S1 output>/runs/ssl/g27_d1_L/rfdetrm_g27ssl.pth
# A3 (copy-paste + D2)
!python tools/rfdetr_v1.py train --dataset /kaggle/working/wtbd_group_cp --out runs/detect/BladeYOLO_WindSurface/rfdetrm768cos_cp_d2_wtbd_group_full --size medium --resolution 768 --epochs 100 --batch 4 --accum 4 --lr-schedule cosine --init <S1 output>/runs/detect/BladeYOLO_WindSurface/rfdetrm576_g27_stage1_L/last_ema.pth
!python tools/rfdetr_v1.py train --dataset /kaggle/working/wtblade_group_cp --out runs/detect/BladeYOLO_WindSurface/rfdetrm640cos_cp_d2_wtblade_group_full --size medium --resolution 640 --epochs 60 --batch 4 --accum 4 --lr-schedule cosine --init <S1 output>/runs/detect/BladeYOLO_WindSurface/rfdetrm576_g27_stage1_L/last_ema.pth
```

**After each run, predict val and test:**
```
!python tools/rfdetr_v1.py predict --run runs/detect/BladeYOLO_WindSurface/<run name> --data <the run's WTBD-group or WTBlade-group path above>
```

**Baseline for the architecture comparison:** YOLO12-L on WTBlade-group, the §12 recipe, 50 epochs like the ViT-B base run:
```
!python train.py --data /kaggle/input/datasets/beegee11/blade-group-splits/data.yaml --model yolo12l.pt --box-loss ciou --name yolo12l_wtblade_group --epochs 50
```

**Suggested sessions (12 h limit; two accounts in parallel):**

| session | runs |
|---|---|
| S1 | pretraining |
| S2 | A0 and A1 on WTBD-group, then A0 on WTBlade-group |
| S3 (after S1) | A2 and A3 on WTBD-group, then A1 on WTBlade-group |
| S4 | A2 and A3 on WTBlade-group |
| S5 | YOLO12-L on WTBlade-group, plus the seed-1 runs of A0 and R (`--seed 1`, names + `_s1`) once A-1 has picked R |

Locally, pull the run folders and run `.venv/bin/python tools/g27_score.py`.

**From 7 Nov, S2–S4 run through `tools/sessions.sh`** (one notebook line after the setup cell: `!bash tools/sessions.sh s2`, `s3` or `s4`). It runs exactly the arms above.
- **Why:** the first S2 attempt (7 Nov) set its paths as shell variables but ran the lines as separate `!` commands. Each line ran in a fresh shell with empty variables (`--data: expected one argument`, outputs under `/`), and the cell carried on past every failure. Nothing trained.
- **What the script fixes:** every path is set inside it, with `set -euo pipefail`, so a failed step stops the session.
- **Inputs are checked first.** For s3/s4 that includes the S1 checkpoints, from the private dataset `beegee11/g27-s1` by default, or from `D1=`/`D2=`.
- **Each prepared folder is rebuilt from scratch,** because `g27_paste.py` refuses a folder that already has pastes.
- **Resumable:** a run that already has both `preds_test.json` and `preds_val.json` is skipped, so rerunning a timed-out session continues it.
- **Tested locally** with train/predict stubbed (`runs/tests/g27_session/`):
  - real prepares and pastes: 45 / 168 images, as planned;
  - all 8 run names match `tools/g27_score.py`;
  - A2 gets D1 and A3 gets D2;
  - a rerun skips finished runs;
  - missing checkpoints stop s3 before any work.
- **`a0s1` (added 7 Nov):** A-2's seed-1 A0 runs on both datasets (`rfdetrm768cos_wtbd_group_full_s1`, `rfdetrm640cos_wtblade_group_full_s1`, about 7 h). They run alongside S2–S4 because A0 is in A-2 whatever R turns out to be, and R is picked on seed-0 val only, so these runs cannot influence the choice. The seed-1 R runs still wait for A-1.

### G27 S1 result: both pretraining checkpoints are usable; D1 narrowed the last-stage features (7 Nov)

Kaggle, 6–7 Nov, restarted without the pool copy. Both steps read `/kaggle/input/datasets/beegee11/g27-pool-l` (recorded in `options.json` and `v1_options.json`), so the interrupted partial copy was not used. The output arrived as `runs.zip` → `runs/ssl/g27_d1_L/`, `runs/detect/BladeYOLO_WindSurface/rfdetrm576_g27_stage1_L/`.

**D1, SSL (8,000 steps, 72 min):** loss 3.25 → 0.18 (step 500) → 0.059 (2,000) → 0.039 (end). The curve flattens from step 3,000; there is no NaN or spike.

**D2, stage 1 (12 epochs, about 6.5 h):**
- Train loss falls every epoch, 10.29 → 7.25; no NaN.
- Val (WTBD-group val + WTBlade-group val, all 34 slots competing at inference) mAP50-95 / mAP50: 0.128 / 0.288 (epoch 0), 0.183 / 0.385 (epoch 2), then flat at 0.17–0.18 / 0.39–0.41 to 0.180 / 0.398 at epoch 11.
- `last_ema.pth` was kept as planned (it initialises A3). `checkpoint_best_total.pth` is also there and unused.

**D1 collapse check** (post hoc, CPU; `runs/analysis/g27/d1_check.py` → `d1_check.md`). COCO encoder vs D1 teacher encoder on 96 held-out images (48 WTBD-group val + 48 WTBlade-group val, none in pool L), 576 px:

| metric | COCO | D1 |
|---|---|---|
| SSL target variance, total | 133.4 | 131.1 |
| SSL target variance per patch | 0.521 | 0.497 |
| last-stage effective rank | 257 | 135 |
| mean pairwise cosine of image-mean features | 0.72 | 0.90 |

- Encoder weights moved 2.8% (relative L2).
- The final loss is 8% of the target's per-patch variance, so the student predicts most of the masked targets.
- **Reading:** no full collapse (the target keeps its variance), but the last stage became lower-dimensional and more alike across images (effective rank halved). This is the partial dimensional collapse that masked feature distillation is known for. It may or may not survive the end-to-end fine-tune, which updates the encoder at 1.5e-4.
- **Decision: run as pre-registered.** A2 is judged only by A2 − A1. The check is not a gate, and changing D1 after seeing it would be a post-hoc change. If A2 fails, this check is the first suspect, and any D1 variant is registered separately.

### G27 A-1 result: R = A1 (copy-paste) by 0.04 points on WTBD-group val; D1 and D2 rejected (8 Nov)

All 10 runs of S2, S3, S4 and `a0s1` completed with no NaN, each from its intended init (`v1_options.json`). `tools/g27_score.py` → `runs/analysis/g27/score.md`. A first scorer run was killed by a 30-minute timeout I set, and its "pending" output was the stale 5 Nov dry run; the rerun had no time limit.

| val mAP50-95, seed 0 | WTBD-group | WTBlade-group | Δ vs A0, WTBD [95% CI] | Δ vs A0, WTBlade [95% CI] |
|---|---|---|---|---|
| A0 | 40.11 | 15.68 | | |
| A1 (+ CP) | 40.15 | 17.10 | +0.04 [−5.7, +4.4] | +1.4 [−0.5, +5.5] |
| A2 (+ CP + D1) | 34.2 | 15.6 | −5.9 [−14.1, +0.5] | −0.0 [−5.6, +1.4] |
| A3 (+ CP + D2) | 33.4 | 17.7 | −6.8 [−9.9, −0.9] | +2.0 [−1.7, +3.4] |

- **R = A1.** It is the only arm whose val point estimate is above A0's on both datasets, as the rule requires. On WTBD the margin is +0.0004 mAP50-95. That is smaller than A0's own seed-to-seed difference on val (0.08 points), so it is a tie in practice. The rule settles such ties on test over two seeds (A-2), which is why A-2 exists.
- **D1 and D2 are rejected.** Their component contrasts against A1 on WTBD val are −5.9 and −6.8 points (D2's CI excludes 0). This is consistent with the S1 check (D1 halved the encoder's last-stage effective rank).
- **Test, seed 0 (secondary, not a decision):**

  | model | WTBD-group | WTBlade-group |
  |---|---|---|
  | A0 seed 0 | 33.2 | 16.2 |
  | A0 seed 1 | 31.2 | 16.9 |
  | A1 seed 0 | 30.7 | 17.3 |
  | A2 seed 0 | 27.4 | 16.1 |
  | A3 seed 0 | 33.2 | 16.6 |
  | BladeYOLO-v2 (ViT-S) | 24.4 | – |
  | YOLO12-L | 18.5 | – |
  | BladeYOLO-v2 ViT-B | – | 18.0 |

  On seed 0, A1 is 2.5 points below A0 on WTBD test and 1.1 above on WTBlade.
- **Next (A-2):** A1 seed 1 on both datasets (`tools/sessions.sh a1s1`). Then the primary claim on test: R − A0 > 0 on each dataset over seeds 0 and 1, with pooled P ≥ 0.95.
- **Every RF-DETR arm beats BladeYOLO-v2 on WTBD-group test by 3–9 points** (secondary, point estimates). On WTBlade-group, RF-DETR's 16.1–17.3 is below the ViT-B BladeYOLO-v2's 18.0, so the "generic detector is strongest" framing holds on WTBD only, so far.

### All remaining runs on one A100 (8 Nov): `tools/sessions.sh all`

The session script is renamed `tools/sessions.sh` (it now serves G27, G28 and the baselines). The user's DGX workstation has a single A100 (40 GB) and no time limit. `bash tools/sessions.sh all` queues, one after another:
- `a0s1`, `a1s1`: the two seed-1 arms of G27 A-2;
- `g28id`, `g28x_wtbd`, `g28x_wtbl`: all of G28;
- `yolo12l_wtbl`: the YOLO12-L baseline on WTBlade-group (Ultralytics, `train.py`, the §12 recipe, 50 epochs).

The recipe is unchanged everywhere (RF-DETR batch 4 × accum 4, cosine), so DGX runs stay comparable in everything except the hardware. The GPU and precision of a run are in its training log, e.g. "Using bfloat16 Automatic Mixed Precision" on the A100 and fp16 on Kaggle T4s.

**Environment.** `tools/sessions.sh` calls plain `python`; the environment is the user's: a venv with `requirements.txt` installed, activated before the call.
- `requirements.txt` is the full GPU-machine install: torch 2.7.1+cu128 from the PyTorch index (the build of the G26 A100 run), RF-DETR, Ultralytics and the rest. A dry-run resolve on Linux/Python 3.12 gave 133 packages, transformers 5.22, numpy 2.5.
- Why a venv and not the NGC container's system Python: installing there broke it. NVIDIA's torch build (`2.6.0a0+…nv24.12`) reads as below 2.6 to `transformers` 5, which then disables PyTorch, and the NumPy 2 upgrade conflicts with that build.
- On Kaggle only `pip install "rfdetr[train]==1.11.2"` is used; the requirements file would replace Kaggle's own torch.

**Speed** (first run, `rfdetrm768cos_wtbd_group_full_s1`): about 28.5 s per epoch on the A100 alone, against 91 s on a Kaggle T4 (152 min for 100 epochs). That is about 3.2× faster, putting the whole queue at roughly 10 h. The G26 A100 run was slower than Kaggle only because 3–4 other jobs shared that GPU.

**Decision, before any A1 seed-1 result exists: A0 seed 1 is rerun on the A100.**
- A-2 pairs R and A0 within each seed. The existing A0 seed 1 ran on a Kaggle T4 in fp16.
- An A100 A1 seed 1 against it would mix the method with the GPU and precision change.
- Rerunning A0 seed 1 on the A100 makes each pair like for like: seed 0 on T4, both arms; seed 1 on A100, both arms.
- When the DGX runs are copied back, the Kaggle A0 seed-1 runs move to `runs/analysis/archive/kaggle_a0s1/` (same run names), and are reported as a hardware check: the same seed on two GPUs.

### G28: does a blade-defect detector work on a dataset it never saw? Leave-one-dataset-out, twin-free (Kaggle). Pre-registered before any pool was built or run started (7 Nov)

**Why (user, 7 Nov):** G27's components did not beat plain RF-DETR (validation, seed 0; the formal A-1 result is logged separately), and the diagnostics put the leak-free ceiling in the data (annotator extent ceiling, rare-class scoring). So the next direction is a question where label count is not the bottleneck. A literature search found no study that tests blade-defect detectors across datasets under conflicting taxonomies. Papers claiming "cross-dataset" results test one other set or non-blade data. Our datasets share images with each other (WTBD ↔ Wind Surface ↔ Beijing; WTBs2025 ⊃ fengChe = WTBlade), so a naive leave-one-dataset-out test would leak. The twin machinery of pool L is what makes the test honest.

**Question.** For a target dataset T, how close does a detector trained only on the other datasets (X) come to one trained on T itself (ID)? How much of the gap is how T's annotators draw boxes (convention), and how much is not finding the defect at all?

**Targets:** WTBD-group and WTBlade-group, the two leak-free splits with scene groups. Scoring is on their test sets (WTBD 161 images / 58 scenes; WTBlade 332).

**Arms** (RF-DETR-M, `rfdetr[train]==1.11.2`, cosine schedule, batch 4 × accum 4, seed 0, 640 px for every arm):

| arm | trained on | checkpoint selection |
|---|---|---|
| ID-T | T's group train, class-agnostic (every box → `defect`); WTBD 100 epochs, WTBlade 60 (G27 A0's epochs) | best on T's group val (RF-DETR default) |
| X-T | every other source, class-agnostic, twin-free against **all** of T (12 epochs, as G27 stage 1) | best on a held-out 5% of X-T's own images (training-domain validation, DomainBed); T is never seen, not even for selection |

- **Class-agnostic is required:** the taxonomies differ, and mapping them is the later method question, not the first test. One consequence, noted in advance: X learns to box things T does not label (e.g. dirt, oil, which WTBD has no class for). That shows up as false positives against T's labels, so the test is conservative for X. To make this visible, recall (AR) is reported beside AP, since recall is unaffected by scope.
- **X-T pools** (`tools/g27_pool.py`, new pool types X-WTBD and X-WTBlade):
  - T's whole source is left out (all of `external/WTBD-official`, or all 10,660 WTBlade-Defect images).
  - Every image of T is an evaluation key for the pool-L twin rule (same file, DINOv2 ≥ 0.90, ORB ≥ 25 inliers, pixel ≥ 0.97, closure), so copies of T inside other datasets are removed.
  - WTBs2025 GAN images and box-less images are dropped, as in pool L.
  - Each pool gets the same independent audit as pool L before export.

**Scores** (T's test, scene bootstrap, 2,000 replicates, paired):
- standard class-agnostic AP50 and AP50-95 (`tools/bootstrap_ci.py`, nc = 1);
- tolerant AP and region AP (`tools/granularity_eval.py`). Region AP is pixel-level and does not depend on how a defect is split into boxes.
- Gaps: `G_std = ID − X` (standard AP50) and `G_reg = ID − X` (region AP). Convention share = 1 − G_reg / G_std.

**Decision, per target, on point estimates, with CIs reported:**
- **Alive, direction "convention-adaptive cross-dataset detection"** if on both targets X reaches ≥ 0.80 × ID in region AP (it finds T's defects), and the convention share is ≥ 0.5 (most of the standard-AP gap is how boxes are drawn). The method question is then learning a target's annotation convention from a few target labels. That is registered separately.
- **Already generalizes** if X ≥ ID in standard AP50 on a target. That is a finding in itself; whether pooling also helps in-domain is a separate registration.
- **Dead otherwise:** drop the direction and move to defect detection without defect labels (normal-only anomaly detection, the next candidate of 7 Nov).
- The 0.80 and 0.5 bars are judgment calls, fixed here before any number exists.

**Reported, not decisive:**
- per-source composition of each X pool and how many images the twin rule removed (expected large for WTBD via Wind Surface/Beijing, and for WTBlade via WTBs2025);
- G27 A0 (class-aware, 768/640) scored class-agnostically, as a reference for ID-T.

**Sessions (estimates, T4):**
- X-WTBD and X-WTBlade, each about 7–9 h at 640 for 12 epochs, one session each;
- ID-WTBD and ID-WTBlade together, about 6 h, one session.
- Three sessions, all parallel.

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

**Cleanup of 26 Oct:** 44 more runs whose results are logged were archived the same way (`runs/analysis/archive/run_<name>/`: `results.csv`, `args.yaml`, `bladeyolo_options.json`, `reward_stats.csv`, test metrics; list in `runs/analysis/archive/archived_26oct.txt`), and their weights and plots deleted (17 GB). Their test predictions stay cached under `runs/analysis/` (`bootstrap/*/preds`, `transfer/`, `openvocab/`), so every logged bootstrap result can be recomputed without the weights; tools that read `results.csv` from `runs/detect/...` (e.g. `tools/subset_properties.py`) must be pointed at the archive. Weights kept for 14 runs: the 12 H5 runs (WTBD baselines and reward runs s0–s3, WTBlade-group baselines and reward runs s0/s1) and the two 100% reference detectors `bladeyolo2_wtbd_group` and `yolo12l_wtbd_group`.

---

## 28. Datasets and artefacts

**Layout from 8 Nov: all data lives under `data/` (gitignored).** Earlier entries in this log use the old paths. They map as follows:

| old path | new path |
|---|---|
| `external/WTBD/` | `data/blade-defect-benchmark/raw/WTBD-original/` |
| `external/WTBD-official/` | `data/blade-defect-benchmark/raw/WTBD-official/` |
| `WTBlade-Defect/`, `WindSurface-Defect/`, `WTBs2025/` | `data/blade-defect-benchmark/raw/<same name>/` |
| `wind-turbine.v1i.yolov12/`, `wind_turbine_check.v1i.yolov12/` | `data/blade-defect-benchmark/raw/Beijing-wind-turbine/`, `…/raw/Beijing-wind_turbine_check/` |
| `external/WTBD-group/`, `external/WTBlade-group/` | `data/blade-defect-benchmark/leakfree/<same name>/` |
| `g27-pool-L/` | `data/blade-defect-benchmark/pools/g27-pool-L/` (with `g28-x-wtbd/`, `g28-x-wtblade/`) |
| `external/{dinov3,owlv2,rfdetr}/` | `data/weights/{dinov3,owlv2,rfdetr}/` |
| `.dataset_cache/` | `data/cache/` |

- `data/blade-defect-benchmark/` is the single dataset that is published on Kaggle (`README.md`, `LICENSES.md` inside it). `data/weights/` and `data/cache/` stay local.
- The migration (`runs/analysis/cleanup/migrate_layout.py`, 8 Nov) rewrote every code path, and every stored path, in 47 code files, 263 csv/json files under `runs/analysis` and `tools/splits`, and 43,864 DINOv2 cache keys. The previous stored files are in `runs/analysis/cleanup/stored_paths_backup.tar.gz`.
- **Removed as redundant** (each checked file by file before deletion):
  - the old dataset folders: every file was a hard link of a bundle file;
  - `external/data.zip`: a zip of the group splits;
  - `view-pool/`: the retired real-view pool, rebuildable with `tools/view_pool.py --export`;
  - `runs/exports/`: copies of the official splits, plus their zips;
  - `runs/tests/`: test scratch;
  - `runs/analysis/v1/kaggle_out_*`: every file byte-identical to one in `runs/detect/`; the notebook logs were kept as `runs/kaggle_logs/v1_*`.

| Item | Location | Notes |
|---|---|---|
| WTBD raw | `data/blade-defect-benchmark/raw/WTBD-original/` | figshare 10.6084/m9.figshare.30210175 |
| WTBD-group / WTBD-official | `data/blade-defect-benchmark/{leakfree/WTBD-group,raw/WTBD-official}/` | `tools/convert_wtbd.py` |
| WTBD group split, scene fractions | `tools/splits/` (committed) | Zero-leak audited split; nested 25% / 50% training subsets |
| Other raw datasets | `data/blade-defect-benchmark/raw/` | Official classes only |
| Current runs | `runs/detect/BladeYOLO_WindSurface/` | `bladeyolo2_wtbd_{group,official,group_tolerant}`, `yolo12l_wtbd_{group,official}`, learning-curve, seed and SSL runs at 25%, DINOv3 size (`bladeyolo2{b,l}_…`) and without-DINOv3 (`bladeyolo2nodino_…`) runs |
| SAM refinement | `runs/analysis/sam_refine/` | SAM 2.1-tiny weights, cached predictions and SAM boxes, `val_grid.csv`, `report.md` |
| Kaggle logs | `runs/kaggle_logs/` | One per `pull_kaggle_runs.py` pull |
| DINOv3 weights | `data/weights/dinov3/vit{s,b,l}16/` (Hugging Face, gated), `data/weights/dinov3/official/` (Meta ViT-S .pth) | DINOv3 License (Meta, 19 Aug 2025; `data/weights/dinov3/vits16/LICENSE.md`): use, modification and redistribution allowed, but anything containing the weights (including our trained checkpoints) must ship with the license; publications must acknowledge DINOv3; military/ITAR uses prohibited |
| Retired runs and the v3 label review | `runs/analysis/archive/` | `results.csv`, `args.yaml`, test metrics; `review_test_v3.json` |
| Dataset audit (Phase 13) | `runs/analysis/datasets/report.md` | |
| WTBD leakage and craze diagnosis | `runs/analysis/{wtbd, wtbd_leak, wtbd_craze}/` | |
| Scene bootstrap | `runs/analysis/bootstrap/` | Reports, scene groups, cached predictions |
| Evaluation experiments | `runs/analysis/novel/` | `granularity/`, `label_audit/{wtbd,windsurface,wtblade}/`, `conformal/`, cached predictions |
| NaN replay | `runs/analysis/nan_probe/` | Layer-6 replay of the tolerant run's NaN batches |
| Caches | `data/cache/` | DINOv2 embeddings, ORB verifications |

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
- **Failed hypotheses (from 28 Oct):** after a bad result, run one scope check, i.e. whether any part of the result is usable. If there is none, discard and move to the next hypothesis. No rescue runs.
- **Git:** push only on request. Commits end with "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>". Datasets, weights, zips and `runs/` are gitignored.
- **Kaggle:** run long jobs with **Save Version → Save & Run All (Commit)** so `/kaggle/working` survives. Download run folders into `runs/detect/BladeYOLO_WindSurface/`.
