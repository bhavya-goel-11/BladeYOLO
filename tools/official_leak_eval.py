"""Paper tables for the official (published, leaky) protocols: every official-protocol run scored with two evaluators,
on the whole test set and on the test images with and without a training twin (research_log.md, "Paper preparation").

Evaluators:
  coco   tools/bootstrap_ci.py (101-point COCO sampling, greedy matching in confidence order), as G25/G26
  ultra  Ultralytics 8.4 DetectionValidator.match_predictions (IoU-sorted unique matching) + utils.metrics.ap_per_class
         (PR envelope, 101-point trapezoidal area), i.e. the numbers an Ultralytics `val` reports; the BladeYOLO paper
         and the other YOLO papers on these datasets report this kind of number.

Leak flags (a test image is "leaked" if a training image that the run actually trained on shows the same scene):
  WTBD-official    runs/analysis/wtbd_leak/test_leak.csv (DINOv2 + flip-aware ORB/RANSAC, tools/leak_subsets.py)
  Wind Surface     pixel twin: correlation >= 0.99 of 48x48 grey images over the 8 flips/rotations, against the 2,570
                   training images left after the seeded 10% carve (rfdetr_v1.prepare_official, seed 0)
                   (tools/ws_pixel_twins.py -> ws_pixel.npz)
  data/blade-defect-benchmark/raw/WTBlade-Defect   scene groups of tools/wtblade_groups.py (runs/analysis/wtblade/split_groups.csv): leaked if the test
                   image's group holds an official-train image; "copy" if a Roboflow copy of the same source frame is
                   in train. 11 test images matched WTBD and were never grouped; they are left out of both subsets.

Image-level bootstrap 95% CI on the coco scores, 1,000 replicates (official splits have no scene groups); the ultra
scores are point values (on identical predictions they agree with coco within 0.3 points, tests/test_official_leak_eval.py).
  .venv/bin/python tools/official_leak_eval.py          -> runs/analysis/paper/official.{md,json}
  .venv/bin/python tools/official_leak_eval.py paired   -> runs/analysis/paper/paired.{md,json}: paired differences
      between two runs on the same subset, and the interaction (difference on leaked images − difference on no-twin
      images; the two subsets resampled independently within one replicate)
"""

import csv
import json
import os
import random
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import bootstrap_ci as bc  # noqa: E402
import detection_metrics as dm  # noqa: E402
from g25_official import names_of  # noqa: E402

RUNS = os.path.join(ROOT, "runs", "detect", "BladeYOLO_WindSurface")
OUT = os.path.join(ROOT, "runs", "analysis", "paper")
YOLO_CACHE = os.path.join(ROOT, "runs", "analysis", "bootstrap", "wtbd_official", "preds")
N_BOOT = 1000
PIXEL_TWIN = 0.99
THR_COL = {round(float(t), 2): i for i, t in enumerate(dm.IOU_THRS)}

PROTOCOLS = [  # name, data dir, test folder, runs (label, run)
    ("WTBD-official", os.path.join(ROOT, "data", "blade-defect-benchmark", "raw", "WTBD-official"), "test",
     [("YOLO12-L", "yolo12l_wtbd_official"), ("BladeYOLO-v2 (ours)", "bladeyolo2_wtbd_official"),
      ("RF-DETR-M 640, const", "rfdetrm640_wtbd_official"), ("RF-DETR-M 768, const", "rfdetrm768_wtbd_official"),
      ("RF-DETR-M 768, cosine", "rfdetrm768cos_wtbd_official"), ("RF-DETR-L 704, cosine", "rfdetrl704cos_wtbd_official")]),
    ("Wind Surface Defect", os.path.join(ROOT, "data", "blade-defect-benchmark", "raw", "WindSurface-Defect"), "val",
     [("RF-DETR-M 640, 60 ep const", "rfdetrm640_windsurface_official"),
      ("RF-DETR-M 640, 100 ep cosine", "rfdetrm640cos_windsurface_official")]),
    ("WTBlade-Defect", os.path.join(ROOT, "data", "blade-defect-benchmark", "raw", "WTBlade-Defect"), "test",
     [("RF-DETR-M 640, 25 ep const", "rfdetrm640_wtblade_defect_official"),
      ("RF-DETR-M 640, 40 ep cosine", "rfdetrm640cos_wtblade_defect_official")]),
]


# ---------------------------------------------------------------- Ultralytics-style evaluator
def ultra_match(gt, pred):
    """Ultralytics DetectionValidator.match_predictions (use_scipy=False), numpy: (n_pred, 10) correct flags."""
    correct = np.zeros((len(pred), len(dm.IOU_THRS)), bool)
    if not len(gt) or not len(pred):
        return correct
    iou = dm.iou_matrix(gt[:, 1:], pred[:, :4]) * (gt[:, :1] == pred[None, :, 5])
    for i, t in enumerate(dm.IOU_THRS):
        m = np.array(np.nonzero(iou >= t)).T
        if m.shape[0]:
            if m.shape[0] > 1:
                m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
                m = m[np.unique(m[:, 1], return_index=True)[1]]
                m = m[np.unique(m[:, 0], return_index=True)[1]]
            correct[m[:, 1].astype(int), i] = True
    return correct


def ultra_stats(imgs, preds):
    return {f: (ultra_match(im["gt"], preds[f]), preds[f][:, 4], preds[f][:, 5], im["gt"][:, 0])
            for f, im in imgs.items()}


def ultra_score(stats, files, nc):
    from ultralytics.utils.metrics import ap_per_class

    tp, conf, pcls, tcls = (np.concatenate([stats[f][k] for f in files]) for k in range(4))
    if not len(tcls):
        return float("nan"), float("nan"), np.full(nc, np.nan)
    r = ap_per_class(tp, conf, pcls, tcls)
    ap, cls = r[5], r[6].astype(int)
    pc50 = np.full(nc, np.nan)
    pc50[cls] = ap[:, 0]
    return float(ap[:, 0].mean()), float(ap.mean()), pc50


# ---------------------------------------------------------------- leak flags
def leak_flags(name, data, files):
    """{abs test path: True (leaked) / False (no twin in train) / None (not assessed)}, plus a note."""
    base = {os.path.basename(f): f for f in files}
    if name == "WTBD-official":
        rows = csv.DictReader(open(os.path.join(ROOT, "runs", "analysis", "wtbd_leak", "test_leak.csv")))
        return {base[r["image"]]: r["leaked"] == "1" for r in rows}, {}, "same scene in train (DINOv2 + ORB/RANSAC)"
    if name == "Wind Surface Defect":
        z = np.load(os.path.join(OUT, "ws_pixel.npz"))
        tr = sorted(os.listdir(os.path.join(data, "train", "images")))
        held = set(random.Random(0).sample(tr, round(0.1 * len(tr))))
        assert not set(z["train"][z["arg"]]) & held, "twin search must exclude the carved validation images"
        return ({base[n]: bool(b >= PIXEL_TWIN) for n, b in zip(z["names"], z["best_trained"])}, {},
                f"pixel twin (corr >= {PIXEL_TWIN}, 8 flips/rotations) among the trained images")
    rows = list(csv.DictReader(open(os.path.join(ROOT, "runs", "analysis", "wtblade", "split_groups.csv"))))
    split = lambda r: r["image"].split("/")[1]  # noqa: E731
    src = lambda p: os.path.basename(p).split("_jpg.rf.")[0]  # noqa: E731
    g_tr = {r["group"] for r in rows if split(r) == "train"}
    s_tr = {src(r["image"]) for r in rows if split(r) == "train"}
    te = {os.path.basename(r["image"]): r for r in rows if split(r) == "test"}
    flags = {f: (te[b]["group"] in g_tr) if b in te else None for b, f in base.items()}
    copy = {f: src(b) in s_tr for b, f in base.items() if b in te}
    return flags, copy, "same scene group in train (tools/wtblade_groups.py)"


# ---------------------------------------------------------------- main
def load_preds(run, data, split):
    if run.startswith(("bladeyolo2", "yolo12l")):
        raw = json.load(open(os.path.join(YOLO_CACHE, f"{run}.json")))
        return {os.path.abspath(k): np.array(v).reshape(-1, 6) for k, v in raw.items()}
    raw = json.load(open(os.path.join(RUNS, run, "preds_test.json")))
    return {os.path.abspath(os.path.join(data, split, "images", k)): np.array(v).reshape(-1, 6) for k, v in raw.items()}


def weighted_prep(matches, files, nc):
    """Detections of all `files` per (class, IoU) sorted once by confidence, so a bootstrap replicate is a vector of
    image multiplicities: the same 101-point COCO AP as bootstrap_ci.score on the resampled multiset, without
    re-concatenating (a duplicated image's detections are tied in confidence, so weighting equals duplicating)."""
    prep = {}
    for c in range(nc):
        for t in dm.IOU_THRS:
            k = (c, round(float(t), 2))
            tp = np.concatenate([matches[f][k][0] for f in files])
            conf = np.concatenate([matches[f][k][1] for f in files])
            img = np.concatenate([np.full(len(matches[f][k][0]), i) for i, f in enumerate(files)]).astype(int)
            o = np.argsort(-conf, kind="stable")
            prep[k] = (tp[o].astype(float), img[o], np.array([matches[f][k][2] for f in files], float))
    return prep


def weighted_score(prep, w, nc):
    x = np.linspace(0, 1, 101)
    ap = np.full((nc, len(dm.IOU_THRS)), np.nan)
    for (c, t), (tp, img, ngt) in prep.items():
        n = float(w @ ngt)
        if n == 0:
            continue
        wd = w[img]
        tpc, dc = np.cumsum(wd * tp), np.cumsum(wd)
        keep = wd > 0
        rec, prec = tpc[keep] / n, tpc[keep] / dc[keep]
        prec = np.maximum.accumulate(prec[::-1])[::-1] if len(prec) else prec
        idx = np.searchsorted(rec, x, side="left")
        vals, ok = np.zeros(len(x)), idx < len(prec)
        vals[ok] = prec[idx[ok]]
        ap[c, THR_COL[t]] = vals.mean()
    return float(np.nanmean(ap[:, 0])), float(np.nanmean(ap.mean(1)))


def ci(prep, members, nc):
    """Image bootstrap within the subset `members` (indices into the prepared files)."""
    rng = np.random.default_rng(0)
    v = []
    for _ in range(N_BOOT):
        w = np.bincount(members[rng.integers(0, len(members), len(members))], minlength=len(prep[(0, 0.5)][2]))
        v.append(weighted_score(prep, w.astype(float), nc))
    return np.percentile(np.array(v), [2.5, 97.5], axis=0).T  # [[lo50, hi50], [lo5095, hi5095]]


def main():
    os.makedirs(OUT, exist_ok=True)
    res, md = {}, ["# Official protocols: two evaluators, leaked vs no-twin test images (research_log.md)", "",
                   f"Image-level bootstrap 95% CI, {N_BOOT} replicates. Percent. coco = our evaluator (G25/G26); "
                   "ultra = Ultralytics `val`-style (matching + ap_per_class of ultralytics 8.4).", ""]
    for name, data, split, runs in PROTOCOLS:
        names = names_of(data)
        nc = len(names)
        imgs = bc.load_yolo_split(data, split)
        files = sorted(imgs)
        flags, copy, how = leak_flags(name, data, files)
        subsets = {"all": files, "leaked": [f for f in files if flags.get(f) is True],
                   "no twin": [f for f in files if flags.get(f) is False]}
        if copy:
            subsets["copy of a train frame"] = [f for f in files if copy.get(f)]
            subsets["leaked, not a copy"] = [f for f in files if flags.get(f) and not copy.get(f)]
        nbox = {k: int(sum(len(imgs[f]["gt"]) for f in v)) for k, v in subsets.items()}
        md += [f"## {name}", "", f"Leak criterion: {how}. " + ", ".join(
            f"{k}: {len(v)} images / {nbox[k]} boxes" for k, v in subsets.items()) + ".", ""]
        cls_counts = {k: [int(sum((imgs[f]["gt"][:, 0] == c).sum() for f in v)) for c in range(nc)]
                      for k, v in subsets.items()}
        md += ["| boxes per class | " + " | ".join(names) + " |", "|---|" + "---|" * nc]
        md += [f"| {k} | " + " | ".join(map(str, cls_counts[k])) + " |" for k in subsets] + [""]
        res[name] = dict(subsets={k: len(v) for k, v in subsets.items()}, boxes=nbox, class_boxes=cls_counts, runs={})
        md += ["| run | subset | coco mAP50 | coco mAP50-95 | ultra mAP50 | ultra mAP50-95 |", "|---|---|---|---|---|---|"]
        for label, run in runs:
            preds = load_preds(run, data, split)
            assert all(f in preds for f in files), f"{run}: missing test predictions"
            m = bc.per_image_matches(imgs, preds, nc)
            u = ultra_stats(imgs, preds)
            prep = weighted_prep(m, files, nc)
            pos = {f: i for i, f in enumerate(files)}
            out = {}
            for k, fs in subsets.items():
                c50, c5095, cpc50, _ = bc.score(m, fs, nc)
                u50, u5095, upc50 = ultra_score(u, fs, nc)
                cc = ci(prep, np.array([pos[f] for f in fs]), nc)
                out[k] = dict(coco=[c50, c5095], coco_ci=cc.tolist(), ultra=[u50, u5095],
                              coco_pc50=dict(zip(names, map(float, cpc50))), ultra_pc50=dict(zip(names, map(float, upc50))))
                fmt = lambda v, c: f"{100 * v:.1f} [{100 * c[0]:.1f}, {100 * c[1]:.1f}]"  # noqa: E731
                md.append(f"| {label} | {k} | {fmt(c50, cc[0])} | {fmt(c5095, cc[1])} | {100 * u50:.1f} | "
                          f"{100 * u5095:.1f} |")
                print(f"{name} | {label} | {k}: coco {100 * c50:.1f}/{100 * c5095:.1f} ultra {100 * u50:.1f}/"
                      f"{100 * u5095:.1f}", flush=True)
            res[name]["runs"][run] = dict(label=label, **out)
        md += ["", "Per-class AP50 (ultra), all / leaked / no twin:", "",
               "| run | " + " | ".join(names) + " |", "|---|" + "---|" * nc]
        for label, run in runs:
            r = res[name]["runs"][run]
            md.append(f"| {label} | " + " | ".join(
                " / ".join("–" if np.isnan(r[k]["ultra_pc50"][c]) else f"{100 * r[k]['ultra_pc50'][c]:.0f}"
                           for k in ("all", "leaked", "no twin")) for c in names) + " |")
        md.append("")
    json.dump(res, open(os.path.join(OUT, "official.json"), "w"), indent=1)
    open(os.path.join(OUT, "official.md"), "w").write("\n".join(md) + "\n")
    print(f"wrote {os.path.join(OUT, 'official.md')}")


PAIRS = {  # protocol: [(label, run A, run B)]: reported as B − A
    "WTBD-official": [("G26 recipe: cosine − constant LR (M 768)", "rfdetrm768_wtbd_official", "rfdetrm768cos_wtbd_official"),
                      ("RF-DETR-M 768 cosine − BladeYOLO-v2", "bladeyolo2_wtbd_official", "rfdetrm768cos_wtbd_official"),
                      ("BladeYOLO-v2 − YOLO12-L", "yolo12l_wtbd_official", "bladeyolo2_wtbd_official")],
    "Wind Surface Defect": [("G26 recipe: 100 ep cosine − 60 ep constant", "rfdetrm640_windsurface_official",
                             "rfdetrm640cos_windsurface_official")],
    "WTBlade-Defect": [("G26 recipe: 40 ep cosine − 25 ep constant", "rfdetrm640_wtblade_defect_official",
                        "rfdetrm640cos_wtblade_defect_official")],
}


def paired():
    res, md = {}, ["# Official protocols: paired differences by leak subset (research_log.md)", "",
                   f"coco evaluator, image bootstrap {N_BOOT} replicates, percent points. Interaction = Δ(leaked) − "
                   "Δ(no twin), the two subsets resampled independently in each replicate.", "",
                   "| protocol | comparison | subset | Δ mAP50 [95% CI] | Δ mAP50-95 [95% CI] | P(Δ50-95 > 0) |",
                   "|---|---|---|---|---|---|"]
    for name, data, split, runs in PROTOCOLS:
        nc = len(names_of(data))
        imgs = bc.load_yolo_split(data, split)
        files = sorted(imgs)
        pos = {f: i for i, f in enumerate(files)}
        flags, copy, _ = leak_flags(name, data, files)
        sub = {"all": [f for f in files], "leaked": [f for f in files if flags.get(f) is True],
               "no twin": [f for f in files if flags.get(f) is False]}
        if copy:
            sub["copy of a train frame"] = [f for f in files if copy.get(f)]
        sub = {k: np.array([pos[f] for f in v]) for k, v in sub.items()}
        preps = {}
        for _, a, b in PAIRS[name]:
            for r in (a, b):
                if r not in preps:
                    preps[r] = weighted_prep(bc.per_image_matches(imgs, load_preds(r, data, split), nc), files, nc)
        for label, a, b in PAIRS[name]:
            rng = np.random.default_rng(0)
            point = {k: np.subtract(weighted_score(preps[b], np.bincount(v, minlength=len(files)).astype(float), nc),
                                    weighted_score(preps[a], np.bincount(v, minlength=len(files)).astype(float), nc))
                     for k, v in sub.items()}
            reps = {k: [] for k in sub}
            for _ in range(N_BOOT):
                for k, v in sub.items():
                    w = np.bincount(v[rng.integers(0, len(v), len(v))], minlength=len(files)).astype(float)
                    reps[k].append(np.subtract(weighted_score(preps[b], w, nc), weighted_score(preps[a], w, nc)))
            reps = {k: np.array(v) for k, v in reps.items()}
            reps["interaction"] = reps["leaked"] - reps["no twin"]
            point["interaction"] = point["leaked"] - point["no twin"]
            out = {}
            for k in list(sub) + ["interaction"]:
                lo, hi = np.percentile(reps[k], [2.5, 97.5], axis=0)
                pgt = float((reps[k][:, 1] > 0).mean())
                out[k] = dict(delta=point[k].tolist(), lo=lo.tolist(), hi=hi.tolist(), p_gt0_5095=pgt)
                md.append(f"| {name} | {label} | {k} | {100 * point[k][0]:+.1f} [{100 * lo[0]:+.1f}, {100 * hi[0]:+.1f}] | "
                          f"{100 * point[k][1]:+.1f} [{100 * lo[1]:+.1f}, {100 * hi[1]:+.1f}] | {pgt:.3f} |")
                print(md[-1], flush=True)
            res[f"{name} | {label}"] = out
    json.dump(res, open(os.path.join(OUT, "paired.json"), "w"), indent=1)
    open(os.path.join(OUT, "paired.md"), "w").write("\n".join(md) + "\n")


if __name__ == "__main__":
    paired() if sys.argv[1:] == ["paired"] else main()
