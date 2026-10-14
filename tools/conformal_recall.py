"""Recall guarantees by conformal risk control, and what calibrating on leaked scenes does to them.

Goal of an inspection system: flag at least 1 - alpha of the defects. Conformal risk control (Angelopoulos et al.,
2022) picks the confidence threshold lambda from a calibration set so that the expected miss rate on new data is
<= alpha, provided calibration and deployment units are exchangeable:
    lambda_hat = max { lambda : (n * mean_i L_i(lambda) + 1) / (n + 1) <= alpha },
L_i(lambda) = fraction of image i's labelled defects not flagged (images with at least one defect); a defect is
flagged if >= 50% of its box is covered by predictions with confidence >= lambda, of any class (an inspection system
must find damage; the box split and class are secondary, see tools/granularity_eval.py).

Experiments (each over --draws random calibration subsets of --n-cal images):
  group model   calibrate on group-train images (scenes seen in training: what a random split of augmented data
                gives) vs on group-val images (unseen scenes); deploy on group-test (unseen scenes)
  official model calibrate on the official val split (most images have a train twin) vs deploy on official test
                images with and without a train twin (runs/analysis/wtbd_leak/test_leak.csv)
Reported per alpha: mean realised miss rate on deployment, share of draws violating alpha, false alarms per image.

Usage (predictions cached by runs/analysis/novel/predict_all.py and tools/bootstrap_ci.py):
    python tools/conformal_recall.py
Writes runs/analysis/novel/conformal/{report.md, report.json}.
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

RUNS = os.path.join(ROOT, "runs", "detect", "BladeYOLO_WindSurface")
ALPHAS = (0.05, 0.1, 0.2)
COVER = 0.5


def flag_conf(gt, preds):
    """Per labelled box: the highest confidence threshold at which predictions (any class) cover >= COVER of it."""
    p = preds[np.argsort(-preds[:, 4])] if len(preds) else preds
    out = np.zeros(len(gt))
    for k, (_, x1, y1, x2, y2) in enumerate(gt):
        S = 64
        m = np.zeros((S, S), bool)
        sx, sy = S / max(x2 - x1, 1e-6), S / max(y2 - y1, 1e-6)
        for q in p:
            a1, b1 = int(np.clip((q[0] - x1) * sx, 0, S)), int(np.clip((q[1] - y1) * sy, 0, S))
            a2, b2 = int(np.clip(np.ceil((q[2] - x1) * sx), 0, S)), int(np.clip(np.ceil((q[3] - y1) * sy), 0, S))
            if a2 > a1 and b2 > b1:
                m[b1:b2, a1:a2] = True
                if m.mean() >= COVER:
                    out[k] = q[4]
                    break
    return out


def units(imgs, preds):
    """Per image with defects: (flag confidences of its defects, confidences of all its predictions)."""
    return [(flag_conf(im["gt"], preds[f]), preds[f][:, 4]) for f, im in imgs.items() if len(im["gt"])]


def loss(u, lam):
    return np.array([np.mean(fc < lam) for fc, _ in u])


def loss_matrix(u, grid):
    """(units, len(grid)): share of each image's defects missed at each threshold (ascending grid)."""
    return np.array([np.searchsorted(np.sort(fc), grid, side="left") / len(fc) for fc, _ in u])


def alarm_matrix(u, grid):
    """(units, len(grid)): predictions per image at or above each threshold."""
    return np.array([len(c) - np.searchsorted(np.sort(c), grid, side="left") for _, c in u])


def calibrate(u, alpha, grid):
    """Largest lambda on the (ascending) grid whose adjusted calibration risk is <= alpha."""
    return grid[calibrate_idx(loss_matrix(u, grid), alpha)]


def calibrate_idx(L, alpha):
    """Index of the largest threshold with (n * mean loss + 1) / (n + 1) <= alpha (loss rises with the threshold)."""
    n = len(L)
    ok = (n * L.mean(0) + 1) / (n + 1) <= alpha
    bad = np.where(~ok)[0]
    return max(int(bad[0]) - 1, 0) if len(bad) else len(ok) - 1


def experiment(cal, deploys, n_cal, draws, grid, rng):
    """For each alpha: thresholds from random calibration subsets, realised deployment miss rate and false alarms."""
    Lc = loss_matrix(cal, grid)
    dep = {k: (loss_matrix(v, grid).mean(0), alarm_matrix(v, grid).mean(0)) for k, v in deploys.items()}
    out = {}
    for alpha in ALPHAS:
        idx = np.array([calibrate_idx(Lc[rng.choice(len(cal), min(n_cal, len(cal)), replace=False)], alpha)
                        for _ in range(draws)])
        res = dict(lambda_median=float(np.median(grid[idx])))
        for name, (miss_curve, alarm_curve) in dep.items():
            miss = miss_curve[idx]
            res[name] = dict(miss_mean=float(miss.mean()), violation_rate=float((miss > alpha).mean()),
                             alarms_per_image=float(alarm_curve[idx].mean()))
        out[alpha] = res
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-cal", type=int, default=100, help="calibration images per draw")
    ap.add_argument("--draws", type=int, default=500)
    args = ap.parse_args()
    out = os.path.join(ROOT, "runs", "analysis", "novel", "conformal")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(0)
    grid = np.concatenate([[0.0], np.geomspace(1e-3, 0.95, 400)])

    def load(run, data, split, cache):
        d = os.path.abspath(os.path.join(ROOT, data))
        imgs = bc.load_yolo_split(d, split)
        return imgs, bc.predict(os.path.join(RUNS, run), imgs, os.path.join(ROOT, cache))

    report, md = {}, ["# Conformal recall guarantees on WTBD", "",
                      f"Loss: share of an image's defects not covered (>= {COVER:.0%}) by predictions above the threshold "
                      f"(any class). Calibration subsets of {args.n_cal} images, {args.draws} draws. "
                      "'violations' = share of draws whose realised deployment miss rate exceeds alpha.", ""]
    for model in ("bladeyolo2", "yolo12l"):
        run = f"{model}_wtbd_group"
        tr = units(*load(run, "external/WTBD-group", "train", "runs/analysis/novel/preds/WTBD-group/train"))
        va = units(*load(run, "external/WTBD-group", "val", "runs/analysis/novel/preds/WTBD-group/val"))
        te = units(*load(run, "external/WTBD-group", "test", "runs/analysis/bootstrap/wtbd_group/preds"))
        report[run] = dict(seen_scenes=experiment(tr, {"test": te}, args.n_cal, args.draws, grid, rng),
                           unseen_scenes=experiment(va, {"test": te}, args.n_cal, args.draws, grid, rng))

        run = f"{model}_wtbd_official"
        offv = units(*load(run, "external/WTBD-official", "val", "runs/analysis/novel/preds/WTBD-official/val"))
        imgs, preds = load(run, "external/WTBD-official", "test", "runs/analysis/bootstrap/wtbd_official/preds")
        leaked = {r["image"]: r["leaked"] == "1" for r in csv.DictReader(
            open(os.path.join(ROOT, "runs", "analysis", "wtbd_leak", "test_leak.csv")))}
        sub = lambda want: units({f: im for f, im in imgs.items() if leaked[os.path.basename(f)] == want}, preds)  # noqa: E731
        report[run] = dict(official_val=experiment(offv, {"test_with_twin": sub(True), "test_without_twin": sub(False)},
                                                   args.n_cal, args.draws, grid, rng))

    json.dump(report, open(os.path.join(out, "report.json"), "w"), indent=2)
    md += ["| model | calibrated on | deployed on | alpha | median threshold | realised miss rate | violations | "
           "false alarms / image |", "|---|---|---|---|---|---|---|---|"]
    for run, exps in report.items():
        for cal, res in exps.items():
            for alpha, r in res.items():
                for dep, v in r.items():
                    if dep == "lambda_median":
                        continue
                    md.append(f"| {run} | {cal} | {dep} | {alpha} | {r['lambda_median']:.3f} | {v['miss_mean']:.3f} | "
                              f"{v['violation_rate']:.0%} | {v['alarms_per_image']:.1f} |")
    open(os.path.join(out, "report.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
