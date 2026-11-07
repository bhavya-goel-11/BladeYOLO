"""G21 (research_log.md): flip agreement as a ranking signal for RF-DETR. Pre-registered before any flipped prediction
was made (1 Nov).

  predict  (.venv-rfdetr) each image mirrored left-right, predicted at conf 0.001, boxes mirrored back, saved as
           <run>/preds_<split>_flip.json in the cache format of tools/rfdetr_v1.py (640-input pixels)
  score    (.venv) arms on the original boxes, each matched to the same-class flipped box of highest IoU >= 0.5
           (s_f = its score, 0 if none):
             original   s
             tta-mean   (s + s_f) / 2        standard score-averaging TTA (the prior-art baseline)
             agreement  sqrt(s * s_f)        a box only ranks high if both views score it (G21's arm)
           -> runs/analysis/g21/report.md

Models with checkpoints: WTBlade-group 8% (4 subsets) and WTBD-group 25% class-matched + rand3.

  .venv-rfdetr/bin/python tools/g21_flip.py predict
  .venv/bin/python tools/g21_flip.py score
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
RUNS = os.path.join(ROOT, "runs", "detect", "BladeYOLO_WindSurface")
WTBD = os.path.join(ROOT, "data", "blade-defect-benchmark", "leakfree", "WTBD-group")
WTBLADE = os.path.join(ROOT, "data", "blade-defect-benchmark", "leakfree", "WTBlade-group")
MODELS = {"WTBD": (WTBD, ["rfdetrm_wtbd_group_f25", "rfdetrm_rand3_wtbd_group_f25"]),
          "WTBlade": (WTBLADE, ["rfdetrm_wtblade_group_f08"] + [f"rfdetrm_rand{k}_wtblade_group_f08" for k in (1, 2, 3)])}
N_BOOT = 2000
OUT = os.path.join(ROOT, "runs", "analysis", "g21")


def unflip(rows, width640):
    """Mirror (n, 6) boxes in 640-input pixels back: x1' = W - x2, x2' = W - x1."""
    r = np.array(rows, float).reshape(-1, 6)
    r[:, [0, 2]] = width640 - r[:, [2, 0]]
    return r


def predict():
    import yaml
    from PIL import Image, ImageOps
    import rfdetr_v1 as v1

    for _, (data, runs) in MODELS.items():
        names = yaml.safe_load(open(os.path.join(data, "data.yaml")))["names"]
        for run in runs:
            run = os.path.join(RUNS, run)
            model = v1.model_class("medium")(pretrain_weights=v1.best_checkpoint(run), num_classes=len(names))
            for split in ("val", "test"):
                path = os.path.join(run, f"preds_{split}_flip.json")
                if os.path.exists(path):
                    continue
                d = os.path.join(data, split, "images")
                out = {}
                for i, f in enumerate(sorted(os.listdir(d))):
                    im = ImageOps.mirror(Image.open(os.path.join(d, f)).convert("RGB"))
                    det = model.predict(im, threshold=0.001, include_source_image=False)
                    cls = np.asarray(det.class_id)
                    keep = cls < len(names)
                    s = 640 / max(im.size)
                    rows = v1.to_rows(np.asarray(det.xyxy)[keep], np.asarray(det.confidence)[keep], cls[keep], s)
                    out[f] = unflip(rows, im.size[0] * s).round(3).tolist()
                    if (i + 1) % 50 == 0:
                        print(f"  {os.path.basename(run)} {split} {i + 1}", flush=True)
                json.dump(out, open(path, "w"))
                print(f"{os.path.basename(run)} {split}: {len(out)} images -> {path}", flush=True)


def fuse(p, q, arm):
    """Rescore original boxes p (n, 6) with flipped boxes q."""
    import detection_metrics as dm

    if arm == "original" or not len(p):
        return p
    sf = np.zeros(len(p))
    if len(q):
        iou = dm.iou_matrix(p[:, :4], q[:, :4])
        iou[p[:, 5][:, None] != q[:, 5][None, :]] = 0
        best = iou.argmax(1)
        ok = iou[np.arange(len(p)), best] >= 0.5
        sf[ok] = q[best[ok], 4]
    r = p.copy()
    r[:, 4] = (p[:, 4] + sf) / 2 if arm == "tta-mean" else np.sqrt(p[:, 4] * sf)
    return r


def score():
    import bootstrap_ci as bc
    import proto_rescore as pr

    arms = ("original", "tta-mean", "agreement")
    os.makedirs(OUT, exist_ok=True)
    lines = ["# G21: flip agreement as a ranking signal (RF-DETR)", ""]
    verdict = {}
    for ds, (data, runs) in MODELS.items():
        if ds == "WTBlade":
            pr.use_wtblade("f08")
        for split in ("val", "test"):
            imgs = bc.load_yolo_split(data, split)
            files = sorted(imgs)
            ms = {}
            for run in runs:
                orig = {os.path.join(data, split, "images", k): np.array(v).reshape(-1, 6)
                        for k, v in json.load(open(os.path.join(RUNS, run, f"preds_{split}.json"))).items()}
                flip = {os.path.join(data, split, "images", k): np.array(v).reshape(-1, 6)
                        for k, v in json.load(open(os.path.join(RUNS, run, f"preds_{split}_flip.json"))).items()}
                ms[run] = {a: bc.per_image_matches(imgs, {f: fuse(orig[f], flip.get(f, np.zeros((0, 6))), a)
                                                         for f in orig}, pr.NC) for a in arms}
            pt = {run: {a: pr.metrics(ms[run][a], files)[0] for a in arms} for run in runs}
            lines += [f"## {ds} {split}", "", "| model | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
            lines += [f"| {r} | " + " | ".join(f"{pt[r][a]:.4f}" for a in arms) + " |" for r in runs]
            fam = {a: np.mean([pt[r][a] for r in runs]) for a in arms}
            lines += ["| **family mean** | " + " | ".join(f"**{fam[a]:.4f}**" for a in arms) + " |", ""]
            if split == "test":
                groups = pr.scenes("test", files)
                pairs = [("agreement", "original"), ("tta-mean", "original"), ("agreement", "tta-mean")]
                rng = np.random.default_rng(0)
                bs = {pq: [] for pq in pairs}
                for _ in range(N_BOOT):
                    dr = [f for gi in rng.integers(0, len(groups), len(groups)) for f in groups[gi]]
                    m = {a: np.mean([pr.metrics(ms[r][a], dr)[0] for r in runs]) for a in arms}
                    for x, y in pairs:
                        bs[(x, y)].append(m[x] - m[y])
                for (x, y), v in bs.items():
                    v = np.array(v)
                    wins = sum(pt[r][x] > pt[r][y] for r in runs)
                    lines.append(f"- {x} − {y}: {fam[x] - fam[y]:+.4f} [{np.percentile(v, 2.5):+.4f}, "
                                 f"{np.percentile(v, 97.5):+.4f}], P(>0) {np.mean(v > 0):.3f}, better on {wins}/{len(runs)}")
                    verdict[(ds, x, y)] = (fam[x] - fam[y], np.mean(v > 0), wins, len(runs))
                lines.append("")
    ok = all(verdict[(ds, "agreement", "original")][0] > 0 and verdict[(ds, "agreement", "original")][1] >= 0.95 and
             verdict[(ds, "agreement", "original")][2] >= verdict[(ds, "agreement", "original")][3] - 1 for ds in MODELS)
    novel = all(verdict[(ds, "agreement", "tta-mean")][0] > 0 and verdict[(ds, "agreement", "tta-mean")][1] >= 0.90
                for ds in MODELS)
    lines += [f"**G21 {'passes' if ok else 'fails'}**; agreement beyond standard TTA: {'yes' if novel else 'no'}"]
    open(os.path.join(OUT, "report.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["predict", "score"])
    predict() if ap.parse_args().cmd == "predict" else score()
