# Paper tables (draft numbers, all traceable)

Percent unless noted. "coco" = our 101-point COCO-style evaluator (`tools/bootstrap_ci.py`); "Ultralytics" = numbers
reported by Ultralytics `val`. On identical predictions the two AP formulas agree within 0.4 points
(`tests/test_official_leak_eval.py`); differences between rows of different origin come from the inference path,
not the metric. Brackets are 95% bootstrap intervals: image-level on official splits, scene-level on group splits.

---

## T1. The three benchmarks and their official splits (§4, §10, `runs/analysis/datasets/report.md`, `runs/analysis/paper/official.md`)

| dataset | files | unique images | independent scenes | official split (train / val / test) | test images with a training twin | how the twin arises |
|---|---|---|---|---|---|---|
| WTBD (Ji et al., *Sci. Data* 2026) | 1,065 | 597 | ~550 | 745 / 159 / 161 | **125 / 161 (78%)** | duplicate and augmented copies of the same photo, split at random |
| Wind Surface Defect (GFD-YOLOX; used by BladeYOLO) | 3,570 | 954 | **81** | 2,856 / 714 (4:1; val = test) | **600 / 714 (84%)** pixel twins (corr ≥ 0.99) | 1,785 originals + 1 augmented copy each (550 pairs split across train/val); up to 44 tiles per drone photo (386 of 882 photos on both sides) |
| WTBlade-Defect (Roboflow "fengChe"; used by BladeYOLO) | 10,660 | 1,146 | 803 | 9,010 / 1,097 / 553 | **499 / 542 (92%)** same scene; **360 (66%)** are copies of a training frame | ~3 Roboflow augmentation copies per source frame; consecutive drone-video frames |

Also augmenting before a random split: KGP-YOLO (*IEEE Sensors J.* 2026; DTU 130 → 780 images, then 624/78/78) and
GA-YOLOv5 (*IEEE TIA* 2025; CLAHE copies, random 60/40 split of repeated shots of the same turbine) (§Phase 31 note).

## T2a. Official protocols: this work vs published (`runs/analysis/g26/report.md`, `official.md`)

mAP50 / mAP50-95, single seed. Model selection on validation only. For Wind Surface, a seeded 10% of train is
carved out for validation, because the test set is the val split.

| protocol | model | mAP50 / mAP50-95 [95% CI] |
|---|---|---|
| WTBD-official (test 161) | YOLO12-L (ours, Ultralytics) | 70.4 / 38.8 |
| | BladeYOLO-v2 (ours, Ultralytics) | 84.0 / 49.3 |
| | RF-DETR-M 768, constant LR (G25) | 90.5 [86.3, 94.4] / 60.2 [55.6, 65.3] |
| | **RF-DETR-M 768, cosine (G26)** | **90.4** [85.8, 94.7] / **61.2** [56.5, 66.1] |
| | RF-DETR-L 704, cosine (G26; val-selected over M) | 88.9 / 59.5 |
| Wind Surface (val 714) | BladeYOLO (paper, Table V) | **86.5 / 57.4** |
| | our best leaky BladeYOLO-L reproduction | 78.2 / 46.9 |
| | RF-DETR-M 640, 60 ep constant (G25) | 81.8 [78.0, 85.4] / 51.4 [48.3, 55.0] |
| | RF-DETR-M 640, 100 ep cosine (G26) | 81.0 [77.1, 84.9] / 53.3 [50.0, 57.0] |
| WTBlade-Defect (test 553) | BladeYOLO (paper, Table I; own 9:1 re-split of 5,326 images, unpublished) | 85.8 / **74.6** |
| | YOLOv12-L (paper baseline, same re-split) | 83.3 / 71.9 |
| | RF-DETR-M 640, 25 ep constant (G25) | 87.4 [84.4, 90.9] / 52.5 [49.6, 56.2] |
| | **RF-DETR-M 640, 40 ep cosine (G26)** | **87.2** [84.0, 90.6] / 55.1 [52.0, 59.1] |

Ratio mAP50-95 / mAP50: BladeYOLO paper on WTBlade 0.87 (its YOLOv12-L 0.86); RF-DETR on the WTBlade copies of
training frames 0.71, on all official test images 0.63, on no-twin images 0.34; leak-free blade results 0.42–0.47.

## T2b. The same official runs on leaked vs no-twin test images (`runs/analysis/paper/official.md`)

coco, mAP50 / mAP50-95 [95% CI].

| protocol, run | all | leaked | no twin |
|---|---|---|---|
| WTBD-official, YOLO12-L | 69.0 / 37.7 | 69.7 / 38.5 | 70.0 / 35.2 |
| WTBD-official, BladeYOLO-v2 | 81.5 / 48.1 | 82.6 / 48.9 | 81.5 / 47.8 |
| WTBD-official, RF-DETR-M 768 cosine | 90.4 / 61.2 | 91.5 / 63.9 | 81.6 [69.1, 93.2] / 49.9 [40.4, 61.1] |
| Wind Surface, RF-DETR-M cosine | 81.0 / 53.3 | 83.3 / 55.1 | **59.2** [49.0, 68.1] / **37.6** [29.9, 45.4] |
| WTBlade-Defect, RF-DETR-M cosine | 87.2 / 55.1 | 90.8 / 59.4 | **50.4** [38.5, 64.0] / **17.2** [13.5, 23.4] |
| WTBlade-Defect, copies of a training frame (360 images) | | **98.2** [97.7, 98.8] / **69.8** [67.8, 72.5] | |
| WTBlade-Defect, same scene, not a copy (139) | | 65.7 / 30.1 | |

Subset sizes (images / boxes): WTBD 125 / 185 leaked, 36 / 53 no twin (no-twin holds 1 box each of craze,
thunderstrike and crack). Wind Surface 600 / 1,169 and 114 / 158. WTBlade 499 / 1,235 and 43 / 91.
Cross-check: WTBlade-Defect's no-twin score (50.4 / 17.2) matches the independently built leak-free WTBlade-group
benchmark (42.2 / 18.2, T5).

## T3. Tuning on the official splits rewards memorisation (`runs/analysis/paper/paired.md`; post hoc)

Paired image bootstrap, Δ mAP50-95 (points) = G26 recipe − G25 recipe, same images. Interaction = Δ leaked −
Δ no twin, with the two subsets resampled independently.

| protocol (change) | Δ all | Δ leaked | Δ no twin | interaction, P(>0) |
|---|---|---|---|---|
| WTBD-official (constant → cosine LR) | +1.0 [−1.2, +3.0] | +1.8 [−0.1, +4.0] | −8.1 [−13.4, +0.2] | +9.9 [+1.1, +15.8], 0.993 |
| Wind Surface (60 ep constant → 100 ep cosine) | +1.8 [+0.5, +3.1] | +2.3 [+0.9, +3.8] | −1.5 [−4.2, +1.4] | +3.8 [+0.7, +7.0], 0.988 |
| WTBlade-Defect (25 ep constant → 40 ep cosine) | +2.6 [+1.1, +3.9] | +3.5 [+2.0, +5.0]; copies +5.4 | −4.0 [−7.2, +0.1] | +7.4 [+3.0, +11.1], 1.000 |

The same comparison at mAP50: all −0.2 to −0.8; no twin −5.0 to −6.0.

## T4. Same model, same recipe, only the WTBD split differs (§11, §12; Ultralytics, test mAP50)

| class | BladeYOLO-v2 official | BladeYOLO-v2 group | YOLO12-L official | YOLO12-L group |
|---|---|---|---|---|
| crack | 99.1 | 38.1 | 85.4 | 38.4 |
| craze | 90.9 | 7.6 | 62.2 | 6.5 |
| hide_craze | 80.0 | 56.9 | 63.9 | 52.6 |
| surface_injure | 81.6 | 74.7 | 71.1 | 62.9 |
| thunderstrike | 99.5 | 98.0 | 93.8 | 41.8 |
| corrosion | 52.7 | 53.8 | 46.0 | 60.8 |
| **all (mAP50 / mAP50-95)** | **84.0 / 49.3** | **54.8 / 24.5** | **70.4 / 38.8** | **43.8 / 20.6** |

Inflation: +29.2 (BladeYOLO-v2) and +26.6 (YOLO12-L) mAP50; mAP50-95 ×2.0 and ×1.9.

## T5. Leak-free benchmarks, full labels (§12, §22, §24, H4; Ultralytics unless noted)

| benchmark | model | mAP50 / mAP50-95 |
|---|---|---|
| WTBD-group (test 161 images, 58 scenes) | YOLO12-L | 43.8 / 20.6 |
| | BladeYOLO-v2 without DINOv3 | 36.1 / 17.0 |
| | BladeYOLO-v2 (frozen DINOv3 ViT-S), 2 seeds | 54.8 / 24.5, 55.4 / 24.5 |
| | BladeYOLO-v2 − YOLO12-L (scene bootstrap) | +16.1 [2.5, 28.4] / +5.9 [0.5, 16.9]; without the 2 classes in ≤ 3 test scenes +5.2 [−2.7, +13.5] |
| WTBlade-group (test 332 frames, 177 scenes) | BladeYOLO-v2 ViT-B | 42.2 / 18.2 |
| low-label RF-DETR-M (coco) | WTBD-group 25%, mean of 4 subsets × 2 seeds | mAP50-95 18.9 (s0), 20.0 (s1) |
| | WTBlade-group 25% / 8%, mean of 4 subsets | mAP50-95 13.2 / 9.2 |

## T6. Where the advantage comes from: DINOv3 ablation and learning curve (§22, §24; WTBD-group, scene bootstrap, mAP50-95)

| | 25% | 50% | 100% |
|---|---|---|---|
| BladeYOLO-v2 (frozen DINOv3 ViT-S) | 7.6 (3 seeds) | 17.5 | 24.2 (2 seeds) |
| without DINOv3 (deformable + selective scan + WIoU kept) | 3.7 (2 seeds) | 8.8 | 15.2 |
| YOLO12-L (stock) | 4.1 | 9.9 | 18.5 |
| **DINOv3 gain** | **+3.9 [+2.5, +10.1]** | **+8.8 [+4.0, +22.1]** | **+9.0 [+4.0, +20.5]** |
| YOLO12-L − without DINOv3 (stock YOLO12-L vs our modules without DINOv3) | +0.4 | +1.1 | +3.3 [−0.2, +13.1] |

**T6b. Backbone size at 25%** (2 seeds each): ViT-S 7.6, ViT-B 10.4, ViT-L 10.6. ViT-B − ViT-S +2.8 [+0.4, +9.6];
ViT-L − ViT-B +0.3 [−4.0, +3.1].

## T7. Low labels: fine-tuned vs frozen foundation features, and subset luck

**T7a. RF-DETR-M (end-to-end fine-tuned DINOv2) vs BladeYOLO-v2 ViT-B (frozen DINOv3), 25% of WTBD-group, same 4 subsets** (V1, H18-R; test mAP50-95, coco)

| | class-matched | rand1 | rand2 | rand3 | mean Δ [95% CI] |
|---|---|---|---|---|---|
| ViT-B s0 | 9.9 | 11.2 | 13.5 | 11.8 | |
| RF-DETR s0 | 18.9 | 20.5 | 21.5 | 14.7 | +7.3 [+2.3, +11.9], 4/4 |
| RF-DETR s1 | 22.1 | 18.8 | 22.0 | 16.9 | +8.3 [+4.5, +15.0], 4/4 |
| RF-DETR, encoder frozen (G22 control) | | | | | WTBD 12.3 vs fine-tuned 18.9; WTBlade 8.7 vs 13.2 |

**T7b. Subset luck** (H6, H12; 8 random 25% scene draws, same model, recipe and seed): ViT-B test mAP50-95 9.0–19.6
(×2.2), without DINOv3 2.6–7.2. DINOv3 gain +7.1 [+5.1, +12.6], positive on 8/8 draws. Draw quality is shared across
models (Spearman ρ 0.71). No label-free subset property predicts it (H7, G16, H20). Label-free scene selection
(patch coverage) beats random by +5.2 [+1.9, +10.5] and ties FreeSel (+2.1, P 0.917, val opposite).

## T8. Evaluation ceilings and deployment

**T8a. Label self-consistency on exactly aligned copies of the same image** (§14 step 13): WTBD 87.3%, WTBlade
86.0%, Wind Surface 83.8% agreement (same class, IoU ≥ 0.5); almost no class conflicts. Lowest: corrosion 69.0%
(WTBD), crack 81.5% (WTBlade), surface_eye 79.2% (Wind Surface).

**T8b. Granularity** (§13 Exp. 1, WTBD-group): split-tolerant AP50 − standard AP50 = +11.9 (BladeYOLO-v2,
56.1 → 68.0) and +12.4 (YOLO12-L). Craze 9.9 → 66.1.

**T8c. Conformal recall guarantee, target ≤ 10% missed** (§13 Exp. 2, §14 step 12; BladeYOLO-v2):

| calibrated on | deployed on | realised miss rate | draws violating |
|---|---|---|---|
| seen scenes (group train), image units | unseen scenes | **31.5%** | 100% |
| seen scenes, scene units | unseen scenes | 28.8% | 100% |
| unseen scenes (group val), image units | unseen scenes | 6.3% | 0% |
| unseen scenes, scene units | unseen scenes | 4.7% | 1% |
| official val (leaky) | official test, no-twin images | **25.9%** | 100% |

## T9. What did not close the leak-free gap (pre-registered; WTBD-group unless noted; Δ test mAP50-95 unless noted)

| idea | result | verdict |
|---|---|---|
| Semi-supervised teacher–student (synthetic / real views / in-domain, class-balanced), 3 seeds | balanced in-domain 6.6 vs supervised 7.9 | fails |
| Real second views: multi-view fusion at test time (WTBlade-group) | −0.8 [−1.5, 0.0] | fails |
| SAM 2.1 box refinement | +0.3 to +0.5 at 25%, −0.2 at 100% | fails |
| Retrieval prior as detector input (H3) | +0.1 [−2.0, +3.0] | fails |
| Feature upsampling AnyUp (G8) / high-frequency cues (G9) | −5.9 / −0.1 (mean AP) | fail |
| Reward fine-tuning, granularity-invariant reward (H5), 4 seeds + WTBlade | region AP +3.6, P 0.896; WTBlade −0.6 | fails on replication |
| Generated rare-class defects vs matched copy-paste (H10) | synth − paste −2.5; paste − none +3.7 [+1.1, +8.1] | generation fails; copy-paste helps |
| Prototype rescoring (G15 / G15-R) and learned re-ranker (H18) on RF-DETR | WTBD +1.1 / +2.5; WTBlade −0.7 / +0.2 | WTBD only; prior art |
| Query-consensus rescoring (G20) | −4.6 (WTBD), −2.1 (WTBlade) | fails |
| Flip TTA / flip agreement (G21) on RF-DETR | −2.9 (WTBD), −0.6 (WTBlade) | fails; TTA hurts DETR |
| Input resolution 768 vs 576 (G24) | WTBD +1.0 (P 0.958), WTBlade −0.1 | helps where native pixels exist |
| Class-selective split-tolerant training (corrosion, hide_craze), 2 seeds | corrosion +9.3 [−0.3, +24.6]; mean −0.4 | targeted only |

## T10. Generalisation outside one dataset (§Phase 28 items 4–5)

| | result |
|---|---|
| Cross-dataset transfer, class-agnostic AP50 | WTBlade model on WTBD-group 5.2 (in-domain 56–58); WTBD models on WTBlade-group 2.2–4.0 (in-domain 47.5) |
| Open-vocabulary OWLv2, no training (WTBD-group) | text queries 3.3 mAP50 (class-agnostic 5.7); 25%-trained ViT-B 25.0–25.7 |
