"""Where a trained detector fails on WindSurface-Defect-v3: error types, object size, localisation, confusions.

Stage 1 (slow, cached): predictions of each checkpoint on val and test at conf 0.001 (as Ultralytics val).
Stage 2: every ground-truth box and every prediction above the operating confidence (the F1-optimal confidence on
val, reused unchanged on test) is classified, TIDE-style:
  GT   tp       a prediction of the same class with IoU >= 0.5
       loc      only a same-class prediction with 0.1 <= IoU < 0.5 (found but badly boxed)
       cls      a prediction of another class with IoU >= 0.5 (found but misclassified)
       missed   nothing above 0.1 IoU
  pred tp, dup (second prediction on an already matched GT), loc, cls, bkg (on no object)
plus AP50 per object-size bucket (COCO-style area ranges), class-agnostic AP50 (what classification costs), the
IoU distribution of true positives (what limits mAP50-95), and galleries of misses and false positives.

Usage:
    python tools/error_analysis.py runs/detect/BladeYOLO_WindSurface/yolo12l_v3 runs/detect/BladeYOLO_WindSurface/bladeyolo_v3
Writes runs/analysis/errors/<run>/{<split>_preds.json, report.json, galleries/*.jpg} and runs/analysis/errors/report.md
(with --data <relabelled copy>, the cached predictions are reused and reports go to runs/analysis/errors/<dataset>/)
"""

import argparse
import collections
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIZES = {"tiny <16px": (0, 16), "small 16-32px": (16, 32), "medium 32-96px": (32, 96), "large >=96px": (96, 1e9)}
ELONG = {"compact <2:1": (0, 2), "elongated 2-4:1": (2, 4), "thin >=4:1": (4, 1e9)}
IOU_THRS = np.linspace(0.5, 0.95, 10)


def size_of(b):
    """sqrt(area) in pixels of the 640-px network input (images are letterboxed to 640 on their long side)."""
    return float(np.sqrt((b[2] - b[0]) * (b[3] - b[1])))


def bucket(v, table):
    return next(k for k, (lo, hi) in table.items() if lo <= v < hi)


def iou_matrix(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), -1)
    area = lambda x: np.prod(x[:, 2:] - x[:, :2], -1)  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def load_split(data, split, names):
    """{image path: dict(gt=(n,5) [cls, x1, y1, x2, y2] in 640-input pixels, scale)}"""
    import csv

    out = {}
    for r in csv.DictReader(open(os.path.join(data, "manifest.csv"))):
        if r["split"] != split:
            continue
        W, H = int(r["width"]), int(r["height"])
        s = 640 / max(W, H)
        stem = os.path.splitext(os.path.basename(r["file"]))[0]
        rows = [list(map(float, l.split())) for l in open(os.path.join(data, "labels", split, stem + ".txt")) if l.strip()]
        g = np.array([[c, (x - w / 2) * W * s, (y - h / 2) * H * s, (x + w / 2) * W * s, (y + h / 2) * H * s]
                      for c, x, y, w, h in rows]).reshape(-1, 5)
        out[os.path.join(data, r["file"])] = dict(gt=g, scale=s, source=r["source"])
    return out


def predict(run, data, split, imgs, batch):
    path = os.path.join(ROOT, "runs", "analysis", "errors", os.path.basename(run), f"{split}_preds.json")
    if os.path.exists(path):
        return json.load(open(path))
    sys.path.insert(0, ROOT)
    import models  # noqa: F401  (BladeYOLO modules for unpickling)
    from ultralytics import YOLO

    model = YOLO(os.path.join(run, "weights", "best.pt"))
    preds, files = {}, sorted(imgs)
    for i in range(0, len(files), batch):
        for f, r in zip(files[i:i + batch], model.predict(files[i:i + batch], imgsz=640, conf=0.001, iou=0.7, max_det=300,
                                                          verbose=False, device="cpu")):
            b = r.boxes
            s = imgs[f]["scale"]
            preds[f] = np.concatenate([b.xyxy.numpy() * s, b.conf.numpy()[:, None], b.cls.numpy()[:, None]], 1).round(3).tolist()
        print(f"  {os.path.basename(run)} {split}: {min(i + batch, len(files))}/{len(files)}", flush=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(preds, open(path, "w"))
    return preds


def drop_ignored(imgs, preds, data):
    """Remove predictions inside reviewed ignore regions (confirmed but unlabelled damage, see tools/apply_review.py)
    unless they match a label (IoU >= 0.5): there they are neither true nor false positives."""
    path = os.path.join(data, "ignore_regions.json")
    if not os.path.exists(path):
        return preds, 0
    ign = {os.path.join(data, k): np.array([r[:4] for r in v]) for k, v in json.load(open(path)).items()}
    out, n = {}, 0
    for f, P in preds.items():
        p = np.array(P).reshape(-1, 6)
        if f in ign and len(p):
            g = imgs[f]["gt"]
            on_gt = iou_matrix(p[:, :4], g[:, 1:]).max(1) >= 0.5 if len(g) else np.zeros(len(p), bool)
            r = ign[f]
            hit = (iou_matrix(p[:, :4], r).max(1) >= 0.5) | np.array([inter_frac(q[None], rr)[0] >= 0.5 for q in p[:, :4] for rr in r]).reshape(len(p), len(r)).any(1)
            keep = on_gt | ~hit
            n += int((~keep).sum())
            p = p[keep]
        out[f] = p.tolist()
    return out, n


def ap_from(tp, conf, n_gt):
    """101-point interpolated AP (COCO), tp: (n,) bool, conf: (n,)."""
    if n_gt == 0:
        return float("nan")
    o = np.argsort(-conf)
    tpc = np.cumsum(tp[o])
    rec, prec = tpc / n_gt, tpc / np.arange(1, len(tp) + 1)
    prec = np.maximum.accumulate(prec[::-1])[::-1] if len(prec) else prec
    x = np.linspace(0, 1, 101)
    idx = np.searchsorted(rec, x, side="left")
    return float(np.mean([prec[i] if i < len(prec) else 0 for i in idx]))


def match_ap(imgs, preds, nc, thr=0.5, keep_gt=None, keep_pred=None, agnostic=False):
    """Per-class AP at one IoU threshold. keep_gt/keep_pred restrict to an area range (COCO 'ignore' semantics:
    predictions matched to an ignored GT are dropped, unmatched predictions outside the range are dropped)."""
    tps, confs, ngt = collections.defaultdict(list), collections.defaultdict(list), collections.Counter()
    for f, im in imgs.items():
        g = im["gt"]
        p = np.array(preds[f]).reshape(-1, 6)
        for c in ([0] if agnostic else range(nc)):
            gc = g if agnostic else g[g[:, 0] == c]
            pc = p if agnostic else p[p[:, 5] == c]
            pc = pc[np.argsort(-pc[:, 4])]
            ign = np.array([not keep_gt(b[1:]) for b in gc], bool) if keep_gt else np.zeros(len(gc), bool)
            ngt[c] += int((~ign).sum())
            iou = iou_matrix(pc[:, :4], gc[:, 1:])
            used = np.zeros(len(gc), bool)
            for i in range(len(pc)):
                cand = np.where((iou[i] >= thr) & ~used)[0] if len(gc) else []
                if len(cand):
                    # prefer a non-ignored GT, as COCO does
                    j = max(cand, key=lambda j: (not ign[j], iou[i, j]))
                    used[j] = True
                    if ign[j]:
                        continue
                    tps[c].append(True)
                elif keep_pred and not keep_pred(pc[i, :4]):
                    continue
                else:
                    tps[c].append(False)
                confs[c].append(pc[i, 4])
    return {c: ap_from(np.array(tps[c], bool), np.array(confs[c]), ngt[c]) for c in ([0] if agnostic else range(nc))}, ngt


def classify(imgs, preds, conf_thr, nc):
    """TIDE-style classification of GT boxes and of predictions above conf_thr."""
    gts, ps = [], []
    for f, im in imgs.items():
        g = im["gt"]
        p = np.array(preds[f]).reshape(-1, 6)
        p = p[p[:, 4] >= conf_thr]
        p = p[np.argsort(-p[:, 4])]
        iou = iou_matrix(p[:, :4], g[:, 1:])
        same = (p[:, 5:6] == g[None, :, 0]) if len(p) and len(g) else np.zeros_like(iou, bool)
        used = np.zeros(len(g), bool)
        best_iou_tp = np.zeros(len(g))
        ptype = []
        for i in range(len(p)):  # greedy matching, highest confidence first (as in AP)
            s_iou = np.where(same[i], iou[i], 0)
            j = int(np.argmax(np.where(used, 0, s_iou))) if len(g) else -1
            if len(g) and s_iou[j] >= 0.5 and not used[j]:
                used[j], best_iou_tp[j] = True, s_iou[j]
                ptype.append("tp")
            elif len(g) and s_iou.max() >= 0.5:
                ptype.append("dup")
            elif len(g) and np.where(~same[i], iou[i], 0).max() >= 0.5:
                ptype.append("cls")
            elif len(g) and s_iou.max() >= 0.1:
                ptype.append("loc")
            elif len(g) and iou[i].max() >= 0.1:
                ptype.append("cls+loc")
            else:
                ptype.append("bkg")
        for i, t in enumerate(ptype):
            j = int(np.argmax(iou[i])) if len(g) else -1
            ps.append(dict(img=f, box=p[i, :4].tolist(), conf=float(p[i, 4]), cls=int(p[i, 5]), type=t,
                           gt_cls=int(g[j, 0]) if j >= 0 and iou[i, j] >= 0.1 else -1,
                           covers=int(((iou_matrix(g[:, 1:], p[i:i + 1, :4])[:, 0] > 0) & same[i]
                                       & (inter_frac(g[:, 1:], p[i, :4]) > 0.5)).sum()) if len(g) else 0))
        for j in range(len(g)):
            col = iou[:, j] if len(p) else np.zeros(0)
            s_col = np.where(same[:, j], col, 0) if len(p) else col
            o_col = np.where(~same[:, j], col, 0) if len(p) else col
            if used[j]:
                t = "tp"
            elif len(p) and s_col.max() >= 0.5:
                t = "tp"  # matched only through a duplicate: the object was found
            elif len(p) and o_col.max() >= 0.5:
                t = "cls"
            elif len(p) and s_col.max() >= 0.1:
                t = "loc"
            elif len(p) and o_col.max() >= 0.1:
                t = "cls+loc"
            else:
                t = "missed"
            b = g[j, 1:]
            w, h = b[2] - b[0], b[3] - b[1]
            gts.append(dict(img=f, box=b.tolist(), cls=int(g[j, 0]), type=t, iou=float(best_iou_tp[j] or (s_col.max() if len(p) else 0)),
                            pred_cls=int(p[int(np.argmax(o_col)), 5]) if t in ("cls", "cls+loc") else -1,
                            size=size_of(b), elong=float(max(w, h) / max(min(w, h), 1e-6)),
                            n_gt_img=len(g), source=imgs[f]["source"]))
    return gts, ps


def inter_frac(g, p):
    """Fraction of each GT box g (n,4) inside box p (4,)."""
    tl = np.maximum(g[:, :2], p[:2])
    br = np.minimum(g[:, 2:], p[2:])
    inter = np.prod(np.clip(br - tl, 0, None), -1)
    return inter / (np.prod(g[:, 2:] - g[:, :2], -1) + 1e-9)


def f1_conf(imgs, preds, nc):
    """Confidence that maximises overall F1 at IoU 0.5 (class-aware) on this split."""
    best = (0, 0.25)
    for t in np.arange(0.05, 0.8, 0.025):
        gts, ps = classify(imgs, preds, t, nc)
        tp = sum(p["type"] == "tp" for p in ps)
        P, R = tp / max(len(ps), 1), sum(g["type"] == "tp" for g in gts) / max(len(gts), 1)
        f1 = 2 * P * R / max(P + R, 1e-9)
        best = max(best, (f1, float(t)))
    return best[1], best[0]


def gallery(items, imgs_root, path, title, n=40, ctx=1.0, size=150, pred_boxes=None):
    """Crops around boxes (green = GT, red = prediction), with a caption, n per sheet."""
    from PIL import Image, ImageDraw

    tiles = []
    for it in items[:n]:
        im = Image.open(it["img"]).convert("RGB")
        s = 640 / max(im.size)
        b = np.array(it["box"]) / s  # back to original pixels
        w, h = b[2] - b[0], b[3] - b[1]
        pad = max(w, h) * ctx + 24 / s
        x1, y1, x2, y2 = b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad
        c = im.crop((max(0, x1), max(0, y1), min(im.width, x2), min(im.height, y2)))
        d = ImageDraw.Draw(c)
        ox, oy = max(0, x1), max(0, y1)
        for o in it.get("others", []):  # other boxes in view: (box, colour)
            ob = np.array(o[0]) / s
            d.rectangle((ob[0] - ox, ob[1] - oy, ob[2] - ox, ob[3] - oy), outline=o[1], width=1)
        d.rectangle((b[0] - ox, b[1] - oy, b[2] - ox, b[3] - oy), outline=it["colour"], width=2)
        k = size / max(c.size)
        c = c.resize((max(1, round(c.width * k)), max(1, round(c.height * k))))
        t = Image.new("RGB", (size, size + 26), (255, 255, 255))
        t.paste(c, ((size - c.width) // 2, 0))
        dt = ImageDraw.Draw(t)
        dt.text((2, size + 1), it["caption"][0], fill=(0, 0, 0))
        dt.text((2, size + 13), it["caption"][1], fill=(0, 0, 0))
        tiles.append(t)
    if not tiles:
        return
    cols = 8
    rows = (len(tiles) + cols - 1) // cols
    S = Image.new("RGB", (cols * (size + 4) + 4, rows * (size + 30) + 24), (255, 255, 255))
    ImageDraw.Draw(S).text((4, 6), title, fill=(0, 0, 0))
    for i, t in enumerate(tiles):
        S.paste(t, (4 + (i % cols) * (size + 4), 24 + (i // cols) * (size + 30)))
    S.save(path, quality=90)


def rs_tot(rs):
    return sum(a for a, _ in rs.values()) / max(sum(b for _, b in rs.values()), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--data", default=os.path.join(ROOT, "WindSurface-Defect-v3"))
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--predict-only", action="store_true")
    args = ap.parse_args()

    import yaml

    data = os.path.abspath(args.data)
    names = yaml.safe_load(open(os.path.join(data, "data.yaml")))["names"]
    nc = len(names)
    splits = {s: load_split(data, s, names) for s in ("val", "test")}
    # predictions depend only on the images, so a relabelled copy of the dataset (v3.1) reuses them
    preds = {(r, s): {os.path.join(data, k[k.rindex("images" + os.sep):]): v
                      for k, v in predict(os.path.abspath(r), data, s, splits[s], args.batch).items()}
             for r in args.runs for s in splits}
    for k in preds:
        preds[k], n = drop_ignored(splits[k[1]], preds[k], data)
        if n:
            print(f"{os.path.basename(k[0])} {k[1]}: {n} predictions inside ignore regions not scored")
    if args.predict_only:
        return
    out_root = os.path.join(ROOT, "runs", "analysis", "errors")
    if os.path.basename(data) != "WindSurface-Defect-v3":
        out_root = os.path.join(out_root, os.path.basename(data))
    md = ["# Error analysis on WindSurface-Defect-v3", ""]

    # ground-truth size distribution per split (train included: does test look like train?)
    train = load_split(data, "train", names)
    md += ["## Ground-truth box size (sqrt area, px at 640 input)", "",
           "| class | split | " + " | ".join(SIZES) + " | median |", "|---|---|" + "---|" * (len(SIZES) + 1)]
    for c in range(nc):
        for s, im in (("train", train), ("val", splits["val"]), ("test", splits["test"])):
            sz = [size_of(b[1:]) for v in im.values() for b in v["gt"] if b[0] == c]
            cnt = collections.Counter(bucket(x, SIZES) for x in sz)
            md.append(f"| {names[c]} | {s} | " + " | ".join(f"{100 * cnt[k] / max(len(sz), 1):.0f}%" for k in SIZES)
                      + f" | {np.median(sz):.0f} |")
    md.append("")

    for r in args.runs:
        run = os.path.basename(os.path.normpath(r))
        out = os.path.join(out_root, run)
        os.makedirs(os.path.join(out, "galleries"), exist_ok=True)
        conf, f1v = f1_conf(splits["val"], preds[(r, "val")], nc)
        report = dict(run=run, conf=conf, val_f1=f1v)
        md += [f"## {run}", "", f"Operating confidence {conf:.3f} (F1-optimal on val, F1 {f1v:.3f}), reused on test.", ""]
        for s, imgs in splits.items():
            P = preds[(r, s)]
            rep = report[s] = {}
            # AP at several IoUs, per class, class-agnostic
            aps = {t: match_ap(imgs, P, nc, t)[0] for t in IOU_THRS}
            rep["AP50"] = {names[c]: aps[0.5][c] for c in range(nc)}
            rep["AP50_95"] = {names[c]: float(np.mean([aps[t][c] for t in IOU_THRS])) for c in range(nc)}
            rep["AP75"] = {names[c]: aps[IOU_THRS[5]][c] for c in range(nc)}
            agn = match_ap(imgs, P, nc, 0.5, agnostic=True)[0][0]
            rep["agnostic_AP50"] = agn
            # AP50 per size bucket (COCO-style)
            rep["AP50_by_size"] = {}
            for k, (lo, hi) in SIZES.items():
                keep = lambda b, lo=lo, hi=hi: lo <= size_of(b) < hi  # noqa: E731
                a, n = match_ap(imgs, P, nc, 0.5, keep, keep)
                rep["AP50_by_size"][k] = {names[c]: (a[c], n[c]) for c in range(nc)}
            gts, ps = classify(imgs, P, conf, nc)
            rep["gt_types"] = {names[c]: dict(collections.Counter(g["type"] for g in gts if g["cls"] == c)) for c in range(nc)}
            rep["pred_types"] = {names[c]: dict(collections.Counter(p["type"] for p in ps if p["cls"] == c)) for c in range(nc)}
            rep["recall_by_size"] = {names[c]: {k: [sum(g["type"] == "tp" for g in gts if g["cls"] == c and bucket(g["size"], SIZES) == k),
                                                    sum(g["cls"] == c and bucket(g["size"], SIZES) == k for g in gts)] for k in SIZES} for c in range(nc)}
            rep["recall_by_elong"] = {names[c]: {k: [sum(g["type"] == "tp" for g in gts if g["cls"] == c and bucket(g["elong"], ELONG) == k),
                                                     sum(g["cls"] == c and bucket(g["elong"], ELONG) == k for g in gts)] for k in ELONG} for c in range(nc)}
            rep["recall_by_crowding"] = {k: [sum(g["type"] == "tp" for g in gts if lo <= g["n_gt_img"] < hi), sum(lo <= g["n_gt_img"] < hi for g in gts)]
                                         for k, (lo, hi) in {"1 box/img": (1, 2), "2-3": (2, 4), "4-9": (4, 10), ">=10": (10, 10**6)}.items()}
            rep["confusions"] = dict(collections.Counter(f"{names[g['cls']]} -> {names[g['pred_cls']]}" for g in gts if g["pred_cls"] >= 0).most_common())
            rep["fp_confusions"] = dict(collections.Counter(f"pred {names[p['cls']]} on {names[p['gt_cls']]}" for p in ps
                                                            if p["type"] in ("cls", "cls+loc") and p["gt_cls"] >= 0).most_common())
            tp_iou = [g["iou"] for g in gts if g["type"] == "tp"]
            rep["tp_iou"] = dict(mean=float(np.mean(tp_iou)), hist=np.histogram(tp_iou, bins=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])[0].tolist())
            rep["tp_iou_by_class"] = {names[c]: float(np.mean([g["iou"] for g in gts if g["cls"] == c and g["type"] == "tp"] or [0])) for c in range(nc)}
            # granularity disagreement: a loc-error prediction that covers >= 2 same-class GTs (merged), or a loc-error GT
            # with >= 2 same-class predictions inside it (split)
            loc_p = [p for p in ps if p["type"] == "loc"]
            rep["loc_preds_merging_gts"] = [sum(p["covers"] >= 2 for p in loc_p), len(loc_p)]
            # recall if "found" means a same-class prediction covers >= half of the GT (box-convention independent),
            # and the ceiling: recall at IoU 0.5 with every prediction down to conf 0.001
            cover, ceiling = collections.Counter(), collections.Counter()
            for f, im in imgs.items():
                p = np.array(P[f]).reshape(-1, 6)
                for b in im["gt"]:
                    same = p[p[:, 5] == b[0]]
                    hi = same[same[:, 4] >= conf]
                    cover[int(b[0])] += bool(len(hi) and max(inter_frac(b[None, 1:], q)[0] for q in hi[:, :4]) >= 0.5)
                    ceiling[int(b[0])] += bool(len(same) and iou_matrix(same[:, :4], b[None, 1:]).max() >= 0.5)
            n_gt = collections.Counter(g["cls"] for g in gts)
            rep["recall_coverage"] = {names[c]: cover[c] / max(n_gt[c], 1) for c in range(nc)}
            rep["recall_ceiling"] = {names[c]: ceiling[c] / max(n_gt[c], 1) for c in range(nc)}

            # galleries (test and val): misses by class, background false positives, confusions, loc errors
            by_img = collections.defaultdict(lambda: ([], []))
            for g in gts:
                by_img[g["img"]][0].append(g)
            for p in ps:
                by_img[p["img"]][1].append(p)

            def others(f):
                gg, pp = by_img[f]
                return [(g["box"], (0, 200, 0)) for g in gg] + [(p["box"], (230, 0, 0)) for p in pp]

            short = {n: n.split("_")[0][:8] for n in names}
            for c in range(nc):
                miss = sorted([g for g in gts if g["cls"] == c and g["type"] != "tp"], key=lambda g: (g["img"], g["box"]))
                rng = np.random.default_rng(0)
                miss = [miss[i] for i in rng.permutation(len(miss))]
                gallery([dict(g, colour=(0, 255, 0), others=others(g["img"]),
                              caption=(f"{g['type']} {g['size']:.0f}px" + (f" ->{short[names[g['pred_cls']]]}" if g["pred_cls"] >= 0 else ""),
                                       f"{g['source']} iou{g['iou']:.2f}")) for g in miss], data,
                        os.path.join(out, "galleries", f"{s}_{c}_{names[c]}_missed.jpg"),
                        f"{run} {s} {names[c]}: GT not detected (thick green = this GT, thin green = other GT, red = predictions >= {conf:.2f})")
                fps = sorted([p for p in ps if p["cls"] == c and p["type"] != "tp"], key=lambda p: -p["conf"])
                gallery([dict(p, colour=(255, 0, 0), others=others(p["img"]),
                              caption=(f"{p['type']} c{p['conf']:.2f}" + (f" on {short[names[p['gt_cls']]]}" if p["gt_cls"] >= 0 else ""),
                                       f"{imgs[p['img']]['source']} {size_of(p['box']):.0f}px")) for p in fps], data,
                        os.path.join(out, "galleries", f"{s}_{c}_{names[c]}_false_pos.jpg"),
                        f"{run} {s} predicted {names[c]}: highest-confidence false positives (thick red = this prediction)")
        json.dump(report, open(os.path.join(out, "report.json"), "w"), indent=1, default=float)

        # markdown summary
        for s in ("val", "test"):
            rep = report[s]
            md += [f"### {run} / {s}", "",
                   f"mAP50 {np.mean(list(rep['AP50'].values())):.3f}, mAP50-95 {np.mean(list(rep['AP50_95'].values())):.3f}, "
                   f"class-agnostic AP50 {rep['agnostic_AP50']:.3f}, mean IoU of true positives {rep['tp_iou']['mean']:.3f} "
                   f"(IoU histogram 0.5/0.6/0.7/0.8/0.9: {rep['tp_iou']['hist']})", "",
                   "| class | AP50 | AP75 | AP50-95 | TP IoU | recall (IoU .5 / coverage / ceiling) | GT: tp / loc / cls / missed | preds: tp / dup / loc / cls / bkg | "
                   + " | ".join(f"R {k}" for k in SIZES) + " | " + " | ".join(f"AP50 {k}" for k in SIZES) + " |",
                   "|---|---|---|---|---|---|---|---|" + "---|" * (2 * len(SIZES))]
            for c, n in enumerate(names):
                gt, pt = rep["gt_types"][n], rep["pred_types"][n]
                rs = rep["recall_by_size"][n]
                aps = rep["AP50_by_size"]
                md.append(f"| {n} | {rep['AP50'][n]:.3f} | {rep['AP75'][n]:.3f} | {rep['AP50_95'][n]:.3f} | {rep['tp_iou_by_class'][n]:.2f} | "
                          f"{rs_tot(rs):.2f} / {rep['recall_coverage'][n]:.2f} / {rep['recall_ceiling'][n]:.2f} | "
                          f"{gt.get('tp', 0)} / {gt.get('loc', 0)} / {gt.get('cls', 0) + gt.get('cls+loc', 0)} / {gt.get('missed', 0)} | "
                          f"{pt.get('tp', 0)} / {pt.get('dup', 0)} / {pt.get('loc', 0)} / {pt.get('cls', 0) + pt.get('cls+loc', 0)} / {pt.get('bkg', 0)} | "
                          + " | ".join(f"{a}/{b}" for a, b in rs.values()) + " | "
                          + " | ".join(("-" if aps[k][n][1] == 0 else f"{aps[k][n][0]:.2f}") for k in SIZES) + " |")
            tot = collections.Counter()
            for k in SIZES:
                for n in names:
                    tot[k] += rep["recall_by_size"][n][k][0]
                    tot[k + "_n"] += rep["recall_by_size"][n][k][1]
            md += ["", "Recall by size (all classes): " + ", ".join(f"{k} {tot[k]}/{tot[k + '_n']} ({100 * tot[k] / max(tot[k + '_n'], 1):.0f}%)" for k in SIZES),
                   "Recall by elongation: " + ", ".join(f"{k} " + "/".join(str(sum(rep['recall_by_elong'][n][k][i] for n in names)) for i in (0, 1)) for k in ELONG),
                   "Recall by boxes per image: " + ", ".join(f"{k} {a}/{b}" for k, (a, b) in rep["recall_by_crowding"].items()),
                   f"Localisation-error predictions covering >= 2 same-class GTs (annotation granularity): {rep['loc_preds_merging_gts'][0]}/{rep['loc_preds_merging_gts'][1]}",
                   "GT confusions (GT -> predicted): " + ", ".join(f"{k} {v}" for k, v in list(rep["confusions"].items())[:8]), ""]
    open(os.path.join(out_root, "report.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
