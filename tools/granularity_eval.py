"""How much of a detector's measured error is annotation granularity? Three scores on the same predictions.

Amorphous defects (cracks, crazing, erosion) have no natural object boundary: annotators split one defect into several
boxes or draw one box over several, and a detector that draws the region differently is scored as a miss plus a false
positive. Per class, at IoU 0.5 (or averaged over 0.50:0.95):
  standard      COCO-style one-to-one matching (101-point AP)
  tolerant      many-to-one: a prediction matches the set of unmatched same-class labels it overlaps (IoU >= t, or
                >= 50% of the label inside it) if its IoU with the bounding box of that set is >= t; recall counts
                every label in the set. A prediction lying mostly (>= 50% of its area) inside an already-matched label
                is a fragment of a found defect and is ignored. (Scene-text evaluation uses the same idea: TedEval.)
  region        pixel level, independent of how regions are split into boxes: at each confidence threshold, the union
                of predicted boxes vs the union of labelled boxes (pooled over images, 4 px grid at 640 input);
                region AP = area under the precision/recall curve.
Scene-group bootstrap intervals (groups.csv from tools/bootstrap_ci.py) for each score and for the difference between
the first two runs.

Usage:
    python tools/granularity_eval.py runs/.../bladeyolo2_wtbd_group runs/.../yolo12l_wtbd_group \
        --data external/WTBD-group --preds runs/analysis/bootstrap/wtbd_group --name wtbd_group
Writes runs/analysis/novel/granularity/<name>/{report.md, report.json}.
"""

import argparse
import collections
import csv
import json
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import bootstrap_ci as bc  # noqa: E402
import error_analysis as ea  # noqa: E402

GRID = 4  # px per region cell at the 640 input
N_TAU = 60  # confidence thresholds for the region curve


def inside_frac(a, b):
    """(n, m): fraction of each box in a that lies inside each box in b."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    area = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    return inter / np.maximum(area, 1e-9)


def weighted_ap(conf, tp_weight, is_tp, n_gt):
    """101-point AP where precision counts predictions and recall counts labels (a prediction can cover several)."""
    if n_gt == 0:
        return float("nan")
    o = np.argsort(-conf)
    rec = np.cumsum(tp_weight[o]) / n_gt
    prec = np.cumsum(is_tp[o]) / np.arange(1, len(o) + 1)
    prec = np.maximum.accumulate(prec[::-1])[::-1] if len(prec) else prec
    idx = np.searchsorted(rec, np.linspace(0, 1, 101), side="left")
    return float(np.mean([prec[i] if i < len(prec) else 0 for i in idx]))


def match_image(g, p, t, tolerant):
    """g: (n,4) labels, p: (m,5) [x1,y1,x2,y2,conf] sorted by conf. Returns per prediction (recall weight, is_tp, keep)."""
    used = np.zeros(len(g), bool)
    out = []
    iou = ea.iou_matrix(p[:, :4], g) if len(p) and len(g) else np.zeros((len(p), len(g)))
    gin = inside_frac(g, p[:, :4]).T if len(p) and len(g) else np.zeros((len(p), len(g)))  # label inside prediction
    pin = inside_frac(p[:, :4], g) if len(p) and len(g) else np.zeros((len(p), len(g)))  # prediction inside label
    for i in range(len(p)):
        free = ~used
        if not tolerant:
            cand = np.where((iou[i] >= t) & free)[0]
            if len(cand):
                used[cand[np.argmax(iou[i, cand])]] = True
                out.append((1, 1, True))
            else:
                out.append((0, 0, True))
            continue
        best = None
        single = np.where((iou[i] >= t) & free)[0]
        if len(single):
            best = [single[np.argmax(iou[i, single])]]
        group = np.where(((iou[i] >= t) | (gin[i] >= 0.5)) & free)[0]
        if len(group) > 1:
            u = np.array([g[group, 0].min(), g[group, 1].min(), g[group, 2].max(), g[group, 3].max()])
            if ea.iou_matrix(p[i:i + 1, :4], u[None])[0, 0] >= t:
                best = list(group)
        if best is not None:
            used[best] = True
            out.append((len(best), 1, True))
        elif len(g) and (pin[i][used] >= 0.5).any():
            out.append((0, 0, False))  # fragment of a found defect: neither true nor false positive
        else:
            out.append((0, 0, True))
    return out


def per_image(imgs, preds, nc, taus):
    """Matches and region counts of every image, computed once; a score over any multiset of images pools them.
    Returns (box, region): box[mode][f][(c, k)] = (conf, recall weight, is_tp, n labels) at IoU_THRS[k];
    region[f] = (nc, len(taus), 3) cell counts (intersection, predicted, labelled)."""
    S = 640 // GRID
    box = {m: {} for m in ("standard", "tolerant")}
    region = {}
    for f, im in imgs.items():
        g_all, p_all = im["gt"], preds[f]
        st = np.zeros((nc, len(taus), 3))
        for m in box:
            box[m][f] = {}
        for c in range(nc):
            g = g_all[g_all[:, 0] == c][:, 1:]
            p = p_all[p_all[:, 5] == c]
            p = p[np.argsort(-p[:, 4])]
            for m in box:
                for k, t in enumerate(ea.IOU_THRS):
                    r = [(q[4], w, tp) for (w, tp, keep), q in zip(match_image(g, p[:, :5], t, m == "tolerant"), p) if keep]
                    x = np.array(r, float).reshape(-1, 3)
                    box[m][f][(c, k)] = (x[:, 0], x[:, 1], x[:, 2], len(g))
            gm = np.zeros((S, S), bool)
            for bb in g:
                x1, y1, x2, y2 = np.clip(np.round(bb / GRID).astype(int), 0, S)
                gm[y1:y2, x1:x2] = True
            if not len(p) and not gm.any():
                continue
            pm, j = np.zeros((S, S), bool), 0
            for k in np.argsort(-taus):  # descending threshold: add boxes incrementally
                while j < len(p) and p[j, 4] >= taus[k]:
                    x1, y1, x2, y2 = np.clip(np.round(p[j, :4] / GRID).astype(int), 0, S)
                    pm[y1:y2, x1:x2] = True
                    j += 1
                st[c, k] = ((pm & gm).sum(), pm.sum(), gm.sum())
        region[f] = st
    return box, region


def box_ap(pre, files, nc, mode):
    """(nc, len(IOU_THRS)) AP over a multiset of images from precomputed matches."""
    aps = np.full((nc, len(ea.IOU_THRS)), np.nan)
    for c in range(nc):
        for k in range(len(ea.IOU_THRS)):
            ms = [pre[0][mode][f][(c, k)] for f in files]
            n = sum(m[3] for m in ms)
            if n:
                aps[c, k] = weighted_ap(*(np.concatenate([m[i] for m in ms]) for i in range(3)), n)
    return aps


def region_ap(st):
    """Region AP per class from pooled counts (precision envelope over recall, 101 points)."""
    out = []
    for c in range(len(st)):
        inter, pred, lab = st[c, :, 0], st[c, :, 1], st[c, :, 2]
        if lab.max() == 0:
            out.append(float("nan"))
            continue
        rec = inter / lab
        prec = np.where(pred > 0, inter / np.maximum(pred, 1), 1.0)
        o = np.argsort(rec)
        rec, prec = rec[o], prec[o]
        env = np.maximum.accumulate(prec[::-1])[::-1]
        idx = np.searchsorted(rec, np.linspace(0, 1, 101), side="left")
        out.append(float(np.mean([env[i] if i < len(env) else 0 for i in idx])))
    return np.array(out)


def scores(pre, files, nc):
    std, tol = box_ap(pre, files, nc, "standard"), box_ap(pre, files, nc, "tolerant")
    reg = region_ap(sum(pre[1][f] for f in files))
    return dict(standard50=std[:, 0], standard=std.mean(1), tolerant50=tol[:, 0], tolerant=tol.mean(1), region=reg)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--preds", required=True, help="folder of a tools/bootstrap_ci.py report (preds/, groups.csv)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=500, help="bootstrap replicates")
    args = ap.parse_args()

    out = os.path.join(ROOT, "runs", "analysis", "novel", "granularity", args.name)
    os.makedirs(out, exist_ok=True)
    data = os.path.abspath(args.data)
    names = yaml.safe_load(open(os.path.join(data, "data.yaml")))["names"]
    nc = len(names)
    imgs = bc.load_yolo_split(data, args.split)
    files = sorted(imgs)
    labels = [os.path.basename(os.path.normpath(r)) for r in args.runs]
    preds = [bc.predict(os.path.abspath(r), imgs, os.path.join(args.preds, "preds")) for r in args.runs]
    taus = np.unique(np.quantile(np.concatenate([p[:, 4] for P in preds for p in P.values() if len(p)]),
                                 np.linspace(0, 1, N_TAU)))
    pre = [per_image(imgs, P, nc, taus) for P in preds]
    point = [scores(q, files, nc) for q in pre]

    groups = collections.defaultdict(list)
    for r in csv.DictReader(open(os.path.join(args.preds, "groups.csv"))):
        groups[r["group"]].append(os.path.join(data, args.split, "images", r["image"]))
    groups = list(groups.values())
    keys = ["standard50", "tolerant50", "region", "standard", "tolerant"]
    rng = np.random.default_rng(0)
    boot = np.zeros((args.n, len(preds), len(keys)))
    for b in range(args.n):
        draw = [f for gi in rng.integers(0, len(groups), len(groups)) for f in groups[gi]]
        for k, q in enumerate(pre):
            s = scores(q, draw, nc)
            boot[b, k] = [np.nanmean(s[x]) for x in keys]
        if (b + 1) % 50 == 0:
            print(f"  bootstrap {b + 1}/{args.n}", flush=True)
    ci = lambda x: [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]  # noqa: E731

    report = dict(data=args.data, split=args.split, metrics=dict(
        standard50="AP50, one-to-one", tolerant50="AP50, many-to-one with fragments ignored",
        region="region AP (pixel union)", standard="AP50-95, one-to-one", tolerant="AP50-95, many-to-one"),
        runs={}, per_class={})
    for k, r in enumerate(labels):
        report["runs"][r] = {x: dict(value=float(np.nanmean(point[k][x])), ci=ci(boot[:, k, j])) for j, x in enumerate(keys)}
        report["per_class"][r] = {names[c]: {x: float(point[k][x][c]) for x in keys} for c in range(nc)}
    if len(preds) > 1:
        d = boot[:, 0] - boot[:, 1]
        report["difference"] = {x: dict(value=float(np.nanmean(point[0][x]) - np.nanmean(point[1][x])), ci=ci(d[:, j]),
                                         p_le_0=float(np.mean(d[:, j] <= 0))) for j, x in enumerate(keys)}
    json.dump(report, open(os.path.join(out, "report.json"), "w"), indent=2)

    f = lambda v: f"{v['value']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]"  # noqa: E731
    md = [f"# Granularity-invariant evaluation: {args.name}", "",
          f"{len(files)} {args.split} images, {len(groups)} scene groups, {args.n} bootstrap replicates (95% intervals).", "",
          "| run | standard AP50 | tolerant AP50 | region AP | standard AP50-95 | tolerant AP50-95 |", "|---|---|---|---|---|---|"]
    md += [f"| {r} | " + " | ".join(f(report["runs"][r][x]) for x in keys) + " |" for r in labels]
    if "difference" in report:
        md += [f"| {labels[0]} − {labels[1]} | " + " | ".join(
            f(report["difference"][x]) + f" P(≤0)={report['difference'][x]['p_le_0']:.2f}" for x in keys) + " |"]
    md += ["", "| class | " + " | ".join(f"{r} std / tol / region" for r in labels) + " |",
           "|---|" + "---|" * len(labels)]
    md += [f"| {names[c]} | " + " | ".join(
        f"{point[k]['standard50'][c]:.3f} / {point[k]['tolerant50'][c]:.3f} / {point[k]['region'][c]:.3f}"
        for k in range(len(labels))) + " |" for c in range(nc)]
    open(os.path.join(out, "report.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
