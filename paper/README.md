# Preprint plan: framing, outline, and where every number comes from

Status (4 Nov 2026): experimentation stopped. Every number below exists in `research_log.md` and, for the new ones, in
`runs/analysis/paper/`. Tables: [`tables.md`](tables.md). Figures: `figures/` (regenerate with
`.venv/bin/python tools/paper_figures.py`).

## 1. Framing

**Recommended framing: a benchmark audit with a strong baseline. Not a new architecture.**

Working title:
> **Memorised Blades: data leakage inflates wind-turbine blade defect benchmarks, and tuning on them rewards it**

Alternative, more neutral:
> *What do blade-defect detection benchmarks measure? Leakage, annotation consistency and foundation features on three public datasets*

**Thesis in one paragraph.** On the three public datasets used to rank blade-defect detectors (WTBD, Wind Surface
Defect, WTBlade-Defect), 78–92% of the official test images have a near-copy or same-scene image in the training
split. A generic foundation detector (RF-DETR) with a standard recipe reaches the level of the published, specialised
architectures on two of the three protocols. But it scores 24 and 40 mAP50 points lower on Wind Surface's and
WTBlade-Defect's test images that have no training twin (10 on WTBD, whose no-twin subset is small). Tuning the recipe against the official numbers raised the score on leaked images and lowered it on the others,
on all three datasets, so leaderboard tuning on these splits measurably selects for memorisation. On leak-free
group splits that we build and release, the field sits at about 42–55 mAP50 / 18–25 mAP50-95. Controlled ablations
show that the only component that helps there is frozen foundation-model features (DINOv3). The task-specific modules
around them (deformable convolutions, selective-scan/Mamba-style blocks, WIoU) contribute nothing measurable at any
data size.

**Why this framing and not a module paper.**
- The module route was tested and did not hold up. Our BladeYOLO-v2 carries the same kind of add-ons the BladeYOLO
  paper claims (Mamba-style selective scan, deformable refinement, a new box loss). With the DINOv3 features removed,
  it equals or trails stock YOLO12-L at 25/50/100% labels (§22, §24). Thirty-odd pre-registered method hypotheses
  (§19–G24) produced no method that survives a second dataset. A paper claiming a novel module would need a module
  that works, and we do not have one. Writing up one that we know does not work is not an option.
- The audit is new for this domain (literature check §24, 21 Oct): no audit of WTBD, Wind Surface or WTBlade exists.
  It also directly explains the field's "too good" numbers, including the BladeYOLO paper's mAP50-95 / mAP50 ratio
  of 0.87. Our leak-free ratio is 0.42–0.47, while test images that are copies of a training frame reach 0.71.
- Benchmark audits with released splits are a recognised paper type (CIFAR near-duplicates; NeurIPS Datasets &
  Benchmarks). It is also the only story in which every result we have counts, including the negative ones.
- The user's goal of "a good score across benchmarks" is kept. We report state-of-the-art-level official numbers
  (WTBD-official 90.4 / 61.2; WTBlade-Defect 87.2 mAP50, above the paper's 85.8), and we show what they are made of.
  That is a stronger position than the leaderboard number alone.

**Tone.** Factual, not accusatory. The BladeYOLO and GFD-YOLOX authors followed common practice (augmenting, then
splitting at random). Say that the protocol, not the authors, inflates the numbers. Ask the BladeYOLO authors for
their 9:1 WTBlade split before submission; if they send it, the 0.87-ratio hypothesis becomes testable.

**Claims we can make, and claims we must not make.**

| claim | status |
|---|---|
| 78% / 84% / 92% of the official test images have a training twin | measured (T1) |
| Same model and recipe, only the split differs: +0.29 mAP50, mAP50-95 ×2, model-independent | measured, 2 models (T4) |
| Leaked test images score far higher than no-twin ones (WTBlade copies 98.2 vs no twin 50.4 mAP50) | measured, post hoc (T2b) |
| Tuning on official splits buys memorisation (interaction > 0, P ≥ 0.988 on all 3) | measured, post hoc, one tuning step; epochs confounded on 2 of 3 (T3) |
| A generic detector matches specialised published architectures on the official protocols | yes on WTBD and WTBlade-Defect (mAP50); **no on Wind Surface** (−5.5 / −4.1) |
| Frozen DINOv3 carries all of BladeYOLO-v2's advantage; the modules add nothing | measured on WTBD-group at 3 fractions (T6). **For our modules, not the BladeYOLO paper's exact modules** (§24 note) |
| Leaked calibration breaks conformal recall guarantees | measured, WTBD (T8) |
| "BladeYOLO's numbers are memorisation" | **not claimable**: its split is unpublished. Say "consistent with" and give the 0.87 vs 0.71 ratio evidence |
| A method contribution | **none**; the rescoring and label-free selection results are prior art or fail to transfer (T9) |

## 2. Outline (section → evidence → table / figure → log section)

1. **Introduction.** Blade inspection, the YOLO-variant literature, near-saturated reported numbers (85–92 mAP50),
   three questions: what do the benchmarks measure, how much is memorisation, what actually helps.
2. **Related work.** Blade defect detection (BladeYOLO, GFD-YOLOX, KGP-YOLO, GA-YOLOv5, DINO-YOLO); dataset
   leakage and near-duplicates (CIFAR 1902.00423, CrowdAI 2304.02296); foundation detectors (RF-DETR, DINOv2/v3);
   label noise and granularity in evaluation (TedEval, box-noise-aware AP); conformal detection; low-label subset
   variance (TFA, RF100-VL).
3. **Datasets and leakage audit.** DINOv2 retrieval plus flip-aware ORB/RANSAC verification, pixel twins, Roboflow
   copy names, DJI tiles, video frames. Unique images and independent scenes per dataset. Protocols of five
   published papers. → **T1**, §4, §10, §29 (Phase 31 note on KGP-YOLO and GA-YOLOv5).
4. **Leak-free benchmarks.** WTBD-group (group split, audited to zero matches; 58 test scenes) and WTBlade-group
   (841 groups; one frame per source; 177 test scenes); scene bootstrap; evaluator note (AP formula irrelevant,
   T2 note). → §10–§12, H4 test bed.
5. **Official protocols: scores and what they are made of.**
   - 5.1 Our numbers vs published → **T2a**.
   - 5.2 Leaked vs no-twin test images → **T2b, Fig. 1**.
   - 5.3 Same model, only the split differs → **T4, Fig. 3**.
   - 5.4 Tuning rewards memorisation → **T3, Fig. 2**.
   - 5.5 Consequence for deployment: leaked calibration breaks recall guarantees → **T8c**.
6. **Leak-free results: what helps.**
   - 6.1 Full data → **T5**.
   - 6.2 Component ablation and learning curve: DINOv3 is the whole advantage → **T6, Fig. 4**.
   - 6.3 Backbone size (ViT-S < ViT-B = ViT-L) → **T6b**.
   - 6.4 Fine-tuned (RF-DETR) vs frozen-feature detector at 25% labels, 2 seeds → **T7a**.
7. **Why leak-free scores are low.**
   - 7.1 Label self-consistency ceiling: 84–87% agreement on aligned copies; granularity accounts for +0.12 AP50 → **T8a, T8b**.
   - 7.2 Subset luck: ×2 between random 25% draws, larger than any model effect → **T7b, Fig. 5**.
   - 7.3 Error anatomy: the fine-tuned DETR proposes the right box and ranks it low; WTBlade cracks are missed at every budget → **Fig. 6**.
   - 7.4 No cross-dataset transfer; open-vocabulary detection near zero → **T10**.
8. **What did not work** (pre-registered, with bars and outcomes) → **T9**. Short. It shows the leak-free gap is not
   closed by the usual add-ons.
9. **Recommendations.** Group or scene splits; publish image lists; deduplicate before augmenting; report leaked /
   no-twin breakdowns; scene-bootstrap intervals; several labelled subsets for low-label claims; calibrate on
   unseen scenes.
10. **Limitations** (§4 below).
11. **Release:** split CSVs (`tools/splits/`, `runs/analysis/wtblade/split_groups.csv`), per-image leak flags,
    audit tools, evaluation code. Respect the dataset licences: WTBlade contains DTU NordTank imagery (CC BY-NC).

## 3. Abstract (draft)

> Public benchmarks for wind-turbine blade defect detection report 85–92 mAP50. We audit the three most used:
> WTBD, Wind Surface Defect and WTBlade-Defect. Using image retrieval with geometric verification, we find that 78%,
> 84% and 92% of their official test images have a near-copy or same-scene image in the training split, because the
> datasets were augmented or tiled before being split at random. Trained on the official splits, a generic
> foundation detector (RF-DETR) with a standard recipe gives the best WTBD result we know of (90.4 mAP50) and
> exceeds the published BladeYOLO on WTBlade-Defect (87.2 vs 85.8 mAP50), though not on Wind Surface (81.0 vs 86.5).
> On test images without a training twin it scores 24 and 40 points lower on Wind Surface and WTBlade-Defect: 50.4
> mAP50 on WTBlade-Defect's no-twin images against 98.2 on its copies of training frames. With model and recipe fixed, changing only WTBD's
> split from random to scene-grouped lowers mAP50 by 0.27–0.29 and halves mAP50-95. Tuning the training recipe
> against the official numbers improved leaked test images and degraded unseen ones on all three datasets, so
> leaderboard tuning on these splits rewards memorisation. We release leak-free group splits of WTBD and
> WTBlade-Defect, with scene-level bootstrap intervals. On them, controlled ablations show that frozen DINOv3 features
> account for the entire advantage of a modern YOLO variant at every label budget, while its deformable,
> selective-scan and loss components add nothing measurable. We further measure an annotation self-consistency
> ceiling of 84–87%, a factor-of-two dependence of low-label results on which scenes are labelled, and a failure of
> conformal recall guarantees calibrated on leaked data (31.5% missed against a promised 10%).

## 4. Limitations to state

- **Official-protocol runs are single seed.** Paired intervals are image-level, since the official splits have no
  scene groups.
- **No full-data RF-DETR on the leak-free splits.** The leak-free full-data rows are BladeYOLO-v2 / YOLO12-L
  (WTBD-group) and BladeYOLO-v2 ViT-B (WTBlade-group); RF-DETR leak-free rows are at 25% and 8% labels. *This is
  the one gap a reviewer will ask about* (see §5).
- **The G26 tuning analysis is post hoc and covers one tuning step.** On Wind Surface and WTBlade-Defect the epoch
  count changed with the schedule.
- **Wind Surface:** our public copy has 3,570 images against the paper's 3,790. Its "no twin" subset has no
  near-identical copy in training but shares drone photos (tiles) with it, so it is cleaner, not leak-free.
- **WTBD's no-twin subset** (36 images) holds a single box of each of three classes, so its mean is unstable.
  Corrosion is the only class with a clear drop.
- **The BladeYOLO paper's exact modules** (Style Injection, WTConv, Cross-Mamba) were reimplemented only before
  the Phase 6 audit (§2–§3). The "modules add nothing" claim is for our components.
- **One wind farm per dataset; small test sets** (58 and 177 test scenes).

## 5. GPU runs before submission: G27 (pre-registered 4 Nov; `research_log.md`, G27)

G27 supersedes the two optional runs listed here earlier. Its arm A0 *is* the RF-DETR-M full-label leak-free twin
(WTBD-group, WTBlade-group, G26 recipe), and its other arms test the improvement directions the paper reports:
copy-paste (recipe), in-domain self-supervised encoder adaptation (D1), and multi-dataset stage 1 with partitioned
class slots (D2), each from a leak-checked extra-data pool. Phase B then trains the selected recipe once on the three
official protocols. Kaggle commands: `research_log.md`, "G27 Kaggle commands".

If G27 finds nothing, the paper still gets A0 and its section on what was tried; if it finds a recipe, the paper gets
"one recipe, best on both protocol families" with the leak breakdown beside the official numbers.

## 6. Regenerating the paper numbers

```
.venv/bin/python tools/g25_official.py g26                    # official scores, G25 + G26 runs
.venv/bin/python -I tools/ws_pixel_twins.py WindSurface-Defect # Wind Surface pixel twins
.venv/bin/python tools/official_leak_eval.py                  # T2a/T2b: two evaluators, leaked vs no twin
.venv/bin/python tools/official_leak_eval.py paired           # T3: paired differences and interaction
.venv/bin/python tools/paper_figures.py                       # figures/
.venv/bin/python tests/test_official_leak_eval.py             # evaluator equivalence and bootstrap check
```
