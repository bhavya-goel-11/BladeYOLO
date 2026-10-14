"""Recall guarantees by conformal risk control, and what calibrating on leaked scenes does to them.

Goal of an inspection system: flag at least 1 - alpha of the defects. Conformal risk control (Angelopoulos et al.,
2022) picks the confidence threshold lambda from a calibration set so that the expected miss rate on new data is
<= alpha, provided calibration and deployment units are exchangeable:
    lambda_hat = max { lambda : (n * mean_i L_i(lambda) + 1) / (n + 1) <= alpha },
L_i(lambda) = share of unit i's labelled defects not flagged (units with at least one defect). A defect is flagged if
>= 50% of its box is covered by predictions with confidence >= lambda: of any class (an inspection system must find
damage; the box split is secondary, see tools/granularity_eval.py) or, with class-aware flagging, of its own class.
Units are images, or scene groups (images of one surface pooled; tools/bootstrap_ci.py grouping): images of a scene
are not exchangeable with each other, so scene units are the strict version of the guarantee.

Experiments (each over --draws random calibration subsets):
  group model    calibrate on group-train (scenes seen in training: what a random split of augmented data gives) vs
                 on group-val (unseen scenes); deploy on group-test (unseen scenes)
  official model calibrate on the official val split (most images have a train twin); deploy on official test images
                 with and without a train twin (runs/analysis/wtbd_leak/test_leak.csv)
Reported per alpha: mean realised miss rate on deployment, share of draws violating alpha, false alarms per image
(predictions above the threshold that lie on no labelled defect).
Operating curve: for the honest setting (calibrated on unseen scenes) the realised recall and false alarms per image
over a range of alpha, for each model (curve.csv, curve.png).

Usage (predictions cached by runs/analysis/novel/predict_all.py and tools/bootstrap_ci.py):
    python tools/conformal_recall.py
Writes runs/analysis/novel/conformal/{report.md, report.json, curve.csv, curve.png, groups_*.json}.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import bootstrap_ci as bc  # noqa: E402
import error_analysis as ea  # noqa: E402

RUNS = os.path.join(ROOT, "runs", "detect", "BladeYOLO_WindSurface")
OUT = os.path.join(ROOT, "runs", "analysis", "novel", "conformal")
ALPHAS = (0.05, 0.1, 0.2)
CURVE_ALPHAS = np.round(np.arange(0.02, 0.41, 0.02), 2)
COVER = 0.5
CONFIGS = (("image", False), ("scene", False), ("scene", True))  # (unit, class-aware flagging)
N_CAL = dict(image=100, scene=40)  # calibration units per draw


def flag_conf(gt, preds, class_aware=False):
    """Per labelled box: the highest confidence threshold at which predictions cover >= COVER of it (of any class,
    or of the box's own class)."""
    p = preds[np.argsort(-preds[:, 4])] if len(preds) else preds
    out = np.zeros(len(gt))
    for k, (c, x1, y1, x2, y2) in enumerate(gt):
        S = 64
        m = np.zeros((S, S), bool)
        sx, sy = S / max(x2 - x1, 1e-6), S / max(y2 - y1, 1e-6)
        for q in p:
            if class_aware and int(q[5]) != int(c):
                continue
            a1, b1 = int(np.clip((q[0] - x1) * sx, 0, S)), int(np.clip((q[1] - y1) * sy, 0, S))
            a2, b2 = int(np.clip(np.ceil((q[2] - x1) * sx), 0, S)), int(np.clip(np.ceil((q[3] - y1) * sy), 0, S))
            if a2 > a1 and b2 > b1:
                m[b1:b2, a1:a2] = True
                if m.mean() >= COVER:
                    out[k] = q[4]
                    break
    return out


def false_alarm_conf(gt, preds):
    """Confidences of predictions that lie on no labelled defect (IoU < 0.1 with every label and < 50% of the
    prediction inside labels): what an inspector would review for nothing."""
    if not len(preds):
        return np.zeros(0)
    if not len(gt):
        return preds[:, 4]
    p, g = preds[:, :4], gt[:, 1:]
    iou = ea.iou_matrix(p, g).max(1)
    x1 = np.maximum(p[:, None, 0], g[None, :, 0])
    y1 = np.maximum(p[:, None, 1], g[None, :, 1])
    x2 = np.minimum(p[:, None, 2], g[None, :, 2])
    y2 = np.minimum(p[:, None, 3], g[None, :, 3])
    inside = ((x2 - x1).clip(0) * (y2 - y1).clip(0)).sum(1) / np.maximum((p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1]), 1e-9)
    return preds[(iou < 0.1) & (inside < 0.5), 4]


def units(imgs, preds, groups=None, class_aware=False):
    """Units with defects: (flag confidences of their defects, confidences of their false alarms, images).
    groups: lists of image paths (scene units); None: every image is a unit."""
    per = {f: (flag_conf(im["gt"], preds[f], class_aware), false_alarm_conf(im["gt"], preds[f])) for f, im in imgs.items()}
    groups = groups or [[f] for f in imgs]
    out = []
    for g in groups:
        g = [f for f in g if f in per]
        fc = np.concatenate([per[f][0] for f in g]) if g else np.zeros(0)
        if len(fc):
            out.append((fc, np.concatenate([per[f][1] for f in g]), len(g)))
    return out


def loss(u, lam):
    return np.array([np.mean(fc < lam) for fc, *_ in u])


def loss_matrix(u, grid):
    """(units, len(grid)): share of each unit's defects missed at each threshold (ascending grid)."""
    return np.array([np.searchsorted(np.sort(fc), grid, side="left") / len(fc) for fc, *_ in u])


def alarm_matrix(u, grid):
    """(units, len(grid)): false alarms per image at or above each threshold."""
    return np.array([(len(c) - np.searchsorted(np.sort(c), grid, side="left")) / n for _, c, n in u])


def calibrate(u, alpha, grid):
    """Largest lambda on the (ascending) grid whose adjusted calibration risk is <= alpha."""
    return grid[calibrate_idx(loss_matrix(u, grid), alpha)]


def calibrate_idx(L, alpha):
    """Index of the largest threshold with (n * mean loss + 1) / (n + 1) <= alpha (loss rises with the threshold)."""
    n = len(L)
    ok = (n * L.mean(0) + 1) / (n + 1) <= alpha
    bad = np.where(~ok)[0]
    return max(int(bad[0]) - 1, 0) if len(bad) else len(ok) - 1


def experiment(cal, deploys, n_cal, draws, grid, rng, alphas=ALPHAS):
    """For each alpha: thresholds from random calibration subsets, realised deployment miss rate and false alarms."""
    Lc = loss_matrix(cal, grid)
    dep = {k: (loss_matrix(v, grid).mean(0), alarm_matrix(v, grid).mean(0)) for k, v in deploys.items()}
    out = {}
    for alpha in alphas:
        idx = np.array([calibrate_idx(Lc[rng.choice(len(cal), min(n_cal, len(cal)), replace=False)], alpha)
                        for _ in range(draws)])
        res = dict(lambda_median=float(np.median(grid[idx])), calibration_units=len(cal))
        for name, (miss_curve, alarm_curve) in dep.items():
            miss = miss_curve[idx]
            res[name] = dict(miss_mean=float(miss.mean()), violation_rate=float((miss > alpha).mean()),
                             alarms_per_image=float(alarm_curve[idx].mean()), units=len(deploys[name]))
        out[float(alpha)] = res
    return out


def scene_groups(imgs, name, workers):
    """Scene groups of a split, cached as runs/analysis/novel/conformal/groups_<name>.json."""
    path = os.path.join(OUT, f"groups_{name}.json")
    if os.path.exists(path):
        return json.load(open(path))
    files = sorted(imgs)
    groups = [[files[i] for i in g] for g in bc.scene_groups(files, os.path.join(ROOT, ".dataset_cache"), workers)]
    json.dump(groups, open(path, "w"))
    return groups


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draws", type=int, default=500)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    grid = np.concatenate([[0.0], np.geomspace(1e-3, 0.95, 400)])

    def load(run, data, split, cache):
        d = os.path.abspath(os.path.join(ROOT, data))
        imgs = bc.load_yolo_split(d, split)
        return imgs, bc.predict(os.path.join(RUNS, run), imgs, os.path.join(ROOT, cache))

    leaked = {r["image"]: r["leaked"] == "1" for r in csv.DictReader(
        open(os.path.join(ROOT, "runs", "analysis", "wtbd_leak", "test_leak.csv")))}
    report, curve = {}, []
    for unit, aware in CONFIGS:
        cfg = f"{unit} units, {'class-aware' if aware else 'any-class'} flagging"
        report[cfg] = {}
        for model in ("bladeyolo2", "yolo12l"):
            rng = np.random.default_rng(0)
            run = f"{model}_wtbd_group"
            split = {}
            for s, cache in (("train", "runs/analysis/novel/preds/WTBD-group/train"),
                             ("val", "runs/analysis/novel/preds/WTBD-group/val"),
                             ("test", "runs/analysis/bootstrap/wtbd_group/preds")):
                imgs, preds = load(run, "external/WTBD-group", s, cache)
                groups = scene_groups(imgs, f"group_{s}", args.workers) if unit == "scene" else None
                split[s] = units(imgs, preds, groups, aware)
            n = N_CAL[unit]
            report[cfg][run] = dict(
                seen_scenes=experiment(split["train"], {"test": split["test"]}, n, args.draws, grid, rng),
                unseen_scenes=experiment(split["val"], {"test": split["test"]}, n, args.draws, grid, rng))
            if unit == "scene":
                c = experiment(split["val"], {"test": split["test"]}, n, args.draws, grid, rng, CURVE_ALPHAS)
                curve += [dict(model=model, flagging="class-aware" if aware else "any-class", alpha=a,
                               threshold=v["lambda_median"], recall=1 - v["test"]["miss_mean"],
                               alarms_per_image=v["test"]["alarms_per_image"]) for a, v in c.items()]

            run = f"{model}_wtbd_official"
            imgs, preds = load(run, "external/WTBD-official", "val", "runs/analysis/novel/preds/WTBD-official/val")
            val = units(imgs, preds, scene_groups(imgs, "official_val", args.workers) if unit == "scene" else None, aware)
            imgs, preds = load(run, "external/WTBD-official", "test", "runs/analysis/bootstrap/wtbd_official/preds")
            groups = scene_groups(imgs, "official_test", args.workers) if unit == "scene" else [[f] for f in imgs]
            dep = {}
            for name, want in (("test_with_twin", True), ("test_without_twin", False)):
                sub = {f: im for f, im in imgs.items() if leaked[os.path.basename(f)] == want}
                dep[name] = units(sub, preds, [[f for f in g if f in sub] for g in groups], aware)
            report[cfg][run] = dict(official_val=experiment(val, dep, n, args.draws, grid, rng))

    json.dump(report, open(os.path.join(OUT, "report.json"), "w"), indent=2)
    with open(os.path.join(OUT, "curve.csv"), "w") as f:
        w = csv.DictWriter(f, fieldnames=list(curve[0]))
        w.writeheader()
        w.writerows(curve)
    plot_curve(curve)

    md = ["# Conformal recall guarantees on WTBD", "",
          f"A defect is flagged if >= {COVER:.0%} of its box is covered by predictions above the threshold. {args.draws} "
          f"calibration draws of {N_CAL['image']} images or {N_CAL['scene']} scene groups. 'violations' = share of draws "
          "whose realised deployment miss rate (mean over deployment units) exceeds alpha.", ""]
    for cfg, runs in report.items():
        md += [f"## {cfg}", "", "| model | calibrated on | deployed on | alpha | median threshold | realised miss rate | "
               "violations | false alarms / image |", "|---|---|---|---|---|---|---|---|"]
        for run, exps in runs.items():
            for cal, res in exps.items():
                for alpha, r in res.items():
                    for dep, v in r.items():
                        if isinstance(v, dict):
                            md.append(f"| {run} | {cal} ({r['calibration_units']}) | {dep} ({v['units']}) | {alpha} | "
                                      f"{r['lambda_median']:.3f} | {v['miss_mean']:.3f} | {v['violation_rate']:.0%} | "
                                      f"{v['alarms_per_image']:.1f} |")
        md.append("")
    md += ["## Operating curve (scene units, calibrated on unseen scenes, deployed on group test)", "",
           "| model | flagging | alpha | threshold | realised recall | false alarms / image |", "|---|---|---|---|---|---|"]
    md += [f"| {c['model']} | {c['flagging']} | {c['alpha']} | {c['threshold']:.3f} | {c['recall']:.3f} | "
           f"{c['alarms_per_image']:.2f} |" for c in curve if c["alpha"] in (0.05, 0.1, 0.2, 0.3)]
    open(os.path.join(OUT, "report.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


def plot_curve(curve):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=150)
    styles = {"bladeyolo2": "#1f77b4", "yolo12l": "#d62728"}
    for model in ("bladeyolo2", "yolo12l"):
        for flag, ls in (("any-class", "-"), ("class-aware", "--")):
            c = [x for x in curve if x["model"] == model and x["flagging"] == flag]
            ax.plot([x["alarms_per_image"] for x in c], [x["recall"] for x in c], ls, marker="o", ms=3,
                    color=styles[model], label=f"{'BladeYOLO-v2' if model == 'bladeyolo2' else 'YOLO12-L'}, {flag}")
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel("false alarms per image (predictions on no labelled defect; symlog)")
    ax.set_ylabel("realised defect recall on new scenes")
    ax.set_title("Conformal operating points, WTBD-group test")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "curve.png"))


if __name__ == "__main__":
    main()
