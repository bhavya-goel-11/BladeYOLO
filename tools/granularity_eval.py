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

--subsets CSV (columns: image, <subset>) compares image subsets, e.g. official test images with and without a train
twin (runs/analysis/wtbd_leak/test_leak.csv): the gain of the tolerant and region scores over the standard one per
subset, over the classes with >= MIN_LABELS labels in every subset, and the paired difference of gains.

Usage:
    python tools/granularity_eval.py runs/.../bladeyolo2_wtbd_group runs/.../yolo12l_wtbd_group \
        --data external/WTBD-group --preds runs/analysis/bootstrap/wtbd_group --name wtbd_group
    python tools/granularity_eval.py runs/.../bladeyolo2_wtbd_official runs/.../yolo12l_wtbd_official \
        --data external/WTBD-official --preds runs/analysis/bootstrap/wtbd_official --name wtbd_official_twins \
        --subsets runs/analysis/wtbd_leak/test_leak.csv
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
import detection_metrics as dm  # noqa: E402

GRID = 4  # px per region cell at the 640 input
N_TAU = 60  # confidence thresholds for the region curve
MIN_LABELS = 5  # labels of a class in every subset for --subsets comparisons


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
    iou = dm.iou_matrix(p[:, :4], g) if len(p) and len(g) else np.zeros((len(p), len(g)))
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
            if dm.iou_matrix(p[i:i + 1, :4], u[None])[0, 0] >= t:
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
                for k, t in enumerate(dm.IOU_THRS):
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
    aps = np.full((nc, len(dm.IOU_THRS)), np.nan)
    for c in range(nc):
        for k in range(len(dm.IOU_THRS)):
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
    ap.add_argument("--arms", action="store_true",
                    help="runs are two arms of equal size (first half vs second half, e.g. seeds): difference of arm means")
    ap.add_argument("--subsets", default=None, help="CSV image,<subset>: compare score gains between image subsets")
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
        h = len(preds) // 2 if args.arms else 1
        A, B = (slice(0, h), slice(h, 2 * h)) if args.arms else (slice(0, 1), slice(1, 2))
        d = boot[:, A].mean(1) - boot[:, B].mean(1)
        pa = lambda sl, x: float(np.mean([np.nanmean(point[k][x]) for k in range(len(preds))[sl]]))  # noqa: E731
        report["difference"] = {x: dict(value=pa(A, x) - pa(B, x), ci=ci(d[:, j]),
                                         p_le_0=float(np.mean(d[:, j] <= 0))) for j, x in enumerate(keys)}
        if args.arms:
            labels_diff = f"mean({', '.join(labels[A])}) − mean({', '.join(labels[B])})"
    if args.subsets:
        report["subsets"] = subset_gains(args, imgs, pre, groups, labels, names, data)
    json.dump(report, open(os.path.join(out, "report.json"), "w"), indent=2)

    f = lambda v: f"{v['value']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]"  # noqa: E731
    md = [f"# Granularity-invariant evaluation: {args.name}", "",
          f"{len(files)} {args.split} images, {len(groups)} scene groups, {args.n} bootstrap replicates (95% intervals).", "",
          "| run | standard AP50 | tolerant AP50 | region AP | standard AP50-95 | tolerant AP50-95 |", "|---|---|---|---|---|---|"]
    md += [f"| {r} | " + " | ".join(f(report["runs"][r][x]) for x in keys) + " |" for r in labels]
    if "difference" in report:
        md += [f"| {labels_diff if args.arms else labels[0] + ' − ' + labels[1]} | " + " | ".join(
            f(report["difference"][x]) + f" P(≤0)={report['difference'][x]['p_le_0']:.2f}" for x in keys) + " |"]
    md += ["", "| class | " + " | ".join(f"{r} std / tol / region" for r in labels) + " |",
           "|---|" + "---|" * len(labels)]
    md += [f"| {names[c]} | " + " | ".join(
        f"{point[k]['standard50'][c]:.3f} / {point[k]['tolerant50'][c]:.3f} / {point[k]['region'][c]:.3f}"
        for k in range(len(labels))) + " |" for c in range(nc)]
    if args.subsets:
        sg = report["subsets"]
        g = lambda v: f"{v['value']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]"  # noqa: E731
        md += ["", f"## Score gains by subset ({sg['column']}; classes with >= {MIN_LABELS} labels in every subset: "
               + ", ".join(sg["classes"]) + ")", "",
               "| run | subset | images | standard AP50 | tolerant − standard | region − standard |", "|---|---|---|---|---|---|"]
        for r in labels:
            for v, x in sg["runs"][r]["subsets"].items():
                md.append(f"| {r} | {v} | {x['images']} | {x['standard50']:.3f} | {g(x['tolerant_gain'])} | "
                          f"{g(x['region_gain'])} |")
            d = sg["runs"][r]["difference"]
            md.append(f"| {r} | {d['label']} | | | {g(d['tolerant_gain'])} P(≤0)={d['tolerant_gain']['p_le_0']:.2f} | "
                      f"{g(d['region_gain'])} P(≤0)={d['region_gain']['p_le_0']:.2f} |")
    open(os.path.join(out, "report.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


def subset_gains(args, imgs, pre, groups, labels, names, data):
    """Gain of the tolerant / region score over the standard one per image subset, and the paired difference."""
    col = None
    sub = {}
    for r in csv.DictReader(open(args.subsets)):
        col = col or [k for k in r if k != "image"][0]
        sub[os.path.join(data, args.split, "images", r["image"])] = r[col]
    values = sorted(set(sub.values()))
    files = {v: [f for f in sorted(imgs) if sub.get(f) == v] for v in values}
    nc = len(names)
    keep = [c for c in range(nc) if all(sum(int((imgs[f]["gt"][:, 0] == c).sum()) for f in files[v]) >= MIN_LABELS
                                        for v in values)]

    def gains(q, fs):
        s = scores(q, fs, nc)
        st, to, re = (np.nanmean(s[x][keep]) for x in ("standard50", "tolerant50", "region"))
        return st, to - st, re - st

    rng = np.random.default_rng(1)
    draws = [[f for gi in rng.integers(0, len(groups), len(groups)) for f in groups[gi]] for _ in range(args.n)]
    ci = lambda x: [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]  # noqa: E731
    out = dict(column=col, classes=[names[c] for c in keep], runs={})
    for k, r in enumerate(labels):
        boot = {v: np.array([gains(pre[k], [f for f in d if sub.get(f) == v]) for d in draws]) for v in values}
        res = {}
        for v in values:
            st, tg, rg = gains(pre[k], files[v])
            res[v] = dict(images=len(files[v]), standard50=float(st), tolerant_gain=dict(value=float(tg), ci=ci(boot[v][:, 1])),
                          region_gain=dict(value=float(rg), ci=ci(boot[v][:, 2])))
        a, b = values[-1], values[0]
        dt, dr = boot[a][:, 1] - boot[b][:, 1], boot[a][:, 2] - boot[b][:, 2]
        diff = dict(label=f"{col}={a} − {col}={b}",
                    tolerant_gain=dict(value=res[a]["tolerant_gain"]["value"] - res[b]["tolerant_gain"]["value"], ci=ci(dt),
                                       p_le_0=float(np.mean(dt <= 0))),
                    region_gain=dict(value=res[a]["region_gain"]["value"] - res[b]["region_gain"]["value"], ci=ci(dr),
                                     p_le_0=float(np.mean(dr <= 0))))
        out["runs"][r] = dict(subsets=res, difference=diff)
        print(f"  subsets done for {r}", flush=True)
    return out


if __name__ == "__main__":
    main()
