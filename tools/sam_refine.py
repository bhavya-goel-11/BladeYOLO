"""Training-free box refinement with SAM 2.1: snap each detection to the extent of the defect SAM segments inside it.

Motivation (research_log Phase 24): at 25% of the labelled scenes the detector finds the region of ~76% of test
defects but boxes only ~40% at IoU 0.5, so box extent, not detection, is the larger loss.

For every image the SAM 2.1 image encoder runs once; each detection with conf >= --min-conf is used as a box prompt
(as is, and expanded by each factor in --expand). The mask's bounding box, the mask's fill of that box and SAM's
predicted IoU are cached per detection. Refinement settings are then searched on val and applied once to test:

  expand e   prompt box scaled by e around its centre
  agree r    keep the original box unless IoU(original, SAM box) >= r (SAM boxes far from the detection are rejected)
  alpha a    refined = a * SAM box + (1 - a) * original

Scores are class-aware mAP50 / mAP50-95 (tools/bootstrap_ci.py), with a paired scene bootstrap on test.

  .venv/bin/python tools/sam_refine.py --runs runs/detect/BladeYOLO_WindSurface/bladeyolo2_wtbd_group_f25 ...
"""

import argparse
import csv
import itertools
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import bootstrap_ci as bc  # noqa: E402
import detection_metrics as dm  # noqa: E402

EXPAND = (1.0, 1.15, 1.3)
AGREE = (0.0, 0.3, 0.5, 0.7)
ALPHA = (0.5, 1.0)


def scale_box(b, e):
    c, wh = (b[:2] + b[2:]) / 2, (b[2:] - b[:2]) * e / 2
    return np.r_[c - wh, c + wh]


def sam_boxes(preds, imgs, out, weights, min_conf, max_det):
    """{run: {image: (n, len(EXPAND), 6) [x1, y1, x2, y2, fill, sam_iou] in 640-input pixels}} for detections >= min_conf."""
    import cv2
    from ultralytics.models.sam import SAM2Predictor

    cache = {}
    if os.path.exists(out):
        cache = {r: {k: np.array(v).reshape(-1, len(EXPAND), 6) for k, v in d.items()} for r, d in json.load(open(out)).items()}
    todo = [f for f in sorted(imgs) if any(f not in cache.get(r, {}) for r in preds)]
    if todo:
        sam = SAM2Predictor(overrides=dict(task="segment", mode="predict", imgsz=1024, model=weights, conf=0.0,
                                           save=False, verbose=False, device=dm.device()))
        sam.args.conf = -1.0  # keep every mask (it filters scores > conf), so masks stay aligned with prompts
    for n, f in enumerate(todo):
        s = imgs[f]["scale"]
        H, W = cv2.imread(f).shape[:2]
        sam.set_image(f)
        for r, p in preds.items():
            d = p.get(f, np.zeros((0, 6)))
            d = d[d[:, 4] >= min_conf][:max_det]
            res = np.zeros((len(d), len(EXPAND), 6))
            for k, e in enumerate(EXPAND):
                if not len(d):
                    break
                prompts = np.stack([np.clip(scale_box(b[:4] / s, e), 0, [W, H, W, H]) for b in d])
                out_r = sam(bboxes=prompts.tolist())[0]
                res[:, k] = np.c_[d[:, :4], np.zeros((len(d), 2))]  # no mask: the original box, fill 0
                if out_r.masks is None:
                    continue
                masks = out_r.masks.data.cpu().numpy() > 0.5
                scores, order = out_r.boxes.conf.cpu().numpy(), out_r.boxes.cls.cpu().numpy().astype(int)
                for i, m, sc in zip(order, masks, scores):  # cls carries the prompt index
                    ys, xs = np.nonzero(m)
                    if len(xs):
                        box = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], float)
                        res[i, k] = np.r_[box * s, m.sum() / max(np.prod(box[2:] - box[:2]), 1), sc]
            cache.setdefault(r, {})[f] = res
        sam.reset_image()
        if (n + 1) % 10 == 0 or n + 1 == len(todo):
            print(f"  SAM {n + 1}/{len(todo)}", flush=True)
            json.dump({r: {k: v.tolist() for k, v in d.items()} for r, d in cache.items()}, open(out, "w"))
    return cache


def refine(p, sb, min_conf, max_det, k, r, a):
    """Predictions with the first max_det detections >= min_conf refined by expansion index k, agreement r, blend a."""
    out = {}
    for f, d in p.items():
        d = d.copy()
        idx = np.where(d[:, 4] >= min_conf)[0][:max_det]
        s = sb.get(f)
        if s is not None and len(idx):
            new = s[: len(idx), k, :4]
            ok = np.diag(dm.iou_matrix(d[idx, :4], new)) >= r
            d[idx[ok], :4] = a * new[ok] + (1 - a) * d[idx[ok], :4]
        out[f] = d
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="run folders (weights/best.pt)")
    ap.add_argument("--data", default=os.path.join(ROOT, "external", "WTBD-group"))
    ap.add_argument("--groups", default=os.path.join(ROOT, "runs", "analysis", "bootstrap", "wtbd_group", "groups.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "analysis", "sam_refine"))
    ap.add_argument("--weights", default=os.path.join(ROOT, "runs", "analysis", "sam_refine", "sam2.1_t.pt"))
    ap.add_argument("--min-conf", type=float, default=0.05)
    ap.add_argument("--max-det", type=int, default=30)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    names = [os.path.basename(os.path.normpath(r)) for r in args.runs]

    S, results = {}, {}
    for split in ("val", "test"):
        imgs = bc.load_yolo_split(args.data, split)
        preds = {n: bc.predict(r, imgs, os.path.join(args.out, f"preds_{split}")) for n, r in zip(names, args.runs)}
        print(f"{split}: SAM boxes for {len(imgs)} images")
        sb = sam_boxes(preds, imgs, os.path.join(args.out, f"sam_{split}.json"), args.weights, args.min_conf, args.max_det)
        files = sorted(imgs)
        ms = lambda pp: bc.per_image_matches(imgs, pp, 6)
        S[split] = dict(imgs=imgs, preds=preds, sb=sb, files=files, base={n: ms(preds[n]) for n in names})
        if split == "val":  # one setting shared by all runs, chosen by mean val mAP50-95
            grid = []
            for k, r, a in itertools.product(range(len(EXPAND)), AGREE, ALPHA):
                sc = [bc.score(ms(refine(preds[n], sb[n], args.min_conf, args.max_det, k, r, a)), files, 6)[:2] for n in names]
                grid.append(((k, r, a), np.mean(sc, 0)))
            base = np.mean([bc.score(S[split]["base"][n], files, 6)[:2] for n in names], 0)
            with open(os.path.join(args.out, "val_grid.csv"), "w") as fh:
                w = csv.writer(fh)
                w.writerow(["expand", "agree", "alpha", "mAP50", "mAP50-95", "d_mAP50", "d_mAP50-95"])
                for (k, r, a), v in grid:
                    w.writerow([EXPAND[k], r, a, f"{v[0]:.4f}", f"{v[1]:.4f}", f"{v[0] - base[0]:+.4f}", f"{v[1] - base[1]:+.4f}"])
            best = max(grid, key=lambda g: g[1][1])
            k, r, a = best[0]
            print(f"val: original {base[0]:.3f} / {base[1]:.3f}; best setting expand {EXPAND[k]} agree {r} alpha {a}: "
                  f"{best[1][0]:.3f} / {best[1][1]:.3f}")
            results["setting"] = dict(expand=EXPAND[k], agree=r, alpha=a)

    T = S["test"]
    g = {}
    for row in csv.DictReader(open(args.groups)):
        g.setdefault(row["group"], []).append(os.path.join(os.path.abspath(args.data), "test", "images", row["image"]))
    groups = list(g.values())
    rng = np.random.default_rng(0)
    draws = [[f for gi in rng.integers(0, len(groups), len(groups)) for f in groups[gi]] for _ in range(1000)]
    lines = [f"# SAM 2.1 box refinement (setting from val: expand {EXPAND[k]}, agree {r}, alpha {a})", "",
             "| run | original mAP50 / 50-95 | refined | difference mAP50 [95% CI] | difference mAP50-95 [95% CI] |",
             "|---|---|---|---|---|"]
    for n in names:
        ref = bc.per_image_matches(T["imgs"], refine(T["preds"][n], T["sb"][n], args.min_conf, args.max_det, k, r, a), 6)
        o, s_ = bc.score(T["base"][n], T["files"], 6)[:2], bc.score(ref, T["files"], 6)[:2]
        d = np.array([np.subtract(bc.score(ref, dr, 6)[:2], bc.score(T["base"][n], dr, 6)[:2]) for dr in draws])
        ci = [f"{s_[j] - o[j]:+.3f} [{np.percentile(d[:, j], 2.5):+.3f}, {np.percentile(d[:, j], 97.5):+.3f}]" for j in (0, 1)]
        lines.append(f"| {n} | {o[0]:.3f} / {o[1]:.3f} | {s_[0]:.3f} / {s_[1]:.3f} | {ci[0]} | {ci[1]} |")
        results[n] = dict(original=o, refined=s_)
    open(os.path.join(args.out, "report.md"), "w").write("\n".join(lines) + "\n")
    json.dump(results, open(os.path.join(args.out, "report.json"), "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
