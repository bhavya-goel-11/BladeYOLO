"""Ensemble several detectors with weighted box fusion (WBF) and score the result on val and test.

Each checkpoint predicts the split once (conf 0.001, optionally with Ultralytics test-time augmentation); predictions
are cached in runs/analysis/ensemble/. Per image and class, boxes of all models are clustered (IoU >= --iou with the
cluster's fused box); a cluster's box is the confidence-weighted mean and its confidence the summed confidence divided
by the number of models, so boxes found by only one model are down-weighted. The fusion IoU is chosen on val (best
mAP50-95, then mAP50) and reused unchanged on test. Scoring is tools/error_analysis.py's (101-point COCO AP; ignore
regions of the dataset honoured).

Usage (Kaggle GPU):
    python tools/ensemble.py runs/.../bladeyolo2_v32 runs/.../bladeyolo_v32 runs/.../yolo12l_v32 \
        --data WindSurface-Defect-v3.2 --tta
Writes runs/analysis/ensemble/<name>/metrics.json.
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import error_analysis as ea  # noqa: E402

FUSION_IOUS = (0.5, 0.55, 0.6, 0.7)


def predict(run, data, split, imgs, batch, tta, device):
    """Cached predictions of run's best.pt: {image path: [[x1, y1, x2, y2, conf, cls], ...]} in 640-input pixels; inference at
    the size the run was trained at."""
    name = os.path.basename(os.path.normpath(run))
    path = os.path.join(ROOT, "runs", "analysis", "ensemble", "preds", f"{name}_{split}{'_tta' if tta else ''}.json")
    if os.path.exists(path):
        raw = json.load(open(path))
        return {os.path.join(data, k[k.rindex("images" + os.sep):]): v for k, v in raw.items()}
    sys.path.insert(0, ROOT)
    import models  # noqa: F401  (BladeYOLO modules for unpickling)
    from ultralytics import YOLO

    model = YOLO(os.path.join(run, "weights", "best.pt"))
    preds, files = {}, sorted(imgs)
    for i in range(0, len(files), batch):
        for f, r in zip(files[i:i + batch], model.predict(files[i:i + batch], imgsz=ea.run_imgsz(run), conf=0.001, iou=0.7, max_det=300,
                                                          augment=tta, verbose=False, device=device)):
            b = r.boxes.cpu()
            s = imgs[f]["scale"]
            preds[f] = np.concatenate([b.xyxy.numpy() * s, b.conf.numpy()[:, None], b.cls.numpy()[:, None]], 1).round(3).tolist()
        print(f"  {name} {split}{' tta' if tta else ''}: {min(i + batch, len(files))}/{len(files)}", flush=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(preds, open(path, "w"))
    return preds


def wbf(per_model, iou_thr, top_k=150):
    """per_model: list of (n, 6) arrays -> fused (m, 6)."""
    n_models = len(per_model)
    p = np.concatenate([np.asarray(a, float).reshape(-1, 6)[:top_k] for a in per_model])
    out = []
    for c in np.unique(p[:, 5]):
        pc = p[p[:, 5] == c]
        pc = pc[np.argsort(-pc[:, 4])]
        fused, members = np.zeros((0, 4)), []
        for b in pc:
            j = -1
            if len(fused):
                iou = ea.iou_matrix(b[None, :4], fused)[0]
                j = int(iou.argmax()) if iou.max() >= iou_thr else -1
            if j < 0:
                fused = np.vstack([fused, b[:4]])
                members.append([b])
            else:
                members[j].append(b)
                m = np.array(members[j])
                fused[j] = (m[:, :4] * m[:, 4:5]).sum(0) / m[:, 4].sum()
        for fb, m in zip(fused, members):
            out.append([*fb, sum(x[4] for x in m) / n_models, c])
    return np.array(out).reshape(-1, 6)


def evaluate(imgs, preds, nc, names):
    aps = {t: ea.match_ap(imgs, preds, nc, t)[0] for t in ea.IOU_THRS}
    per_class = {names[c]: dict(AP50=aps[0.5][c], AP50_95=float(np.mean([aps[t][c] for t in ea.IOU_THRS])))
                 for c in range(nc)}
    return dict(mAP50=float(np.nanmean([v["AP50"] for v in per_class.values()])),
                mAP50_95=float(np.nanmean([v["AP50_95"] for v in per_class.values()])),
                agnostic_AP50=ea.match_ap(imgs, preds, nc, 0.5, agnostic=True)[0][0], per_class=per_class)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run folders (weights/best.pt)")
    ap.add_argument("--data", default=os.path.join(ROOT, "WindSurface-Defect-v3"))
    ap.add_argument("--tta", action="store_true", help="also use test-time augmentation for every model")
    ap.add_argument("--name", default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import yaml

    data = os.path.abspath(args.data)
    names = yaml.safe_load(open(os.path.join(data, "data.yaml")))["names"]
    nc = len(names)
    device = args.device or ("0" if __import__("torch").cuda.is_available() else "cpu")
    splits = {s: ea.load_split(data, s, names) for s in ("val", "test")}
    preds = {s: [ea.drop_ignored(splits[s], predict(os.path.abspath(r), data, s, splits[s], args.batch, args.tta, device),
                                 data)[0] for r in args.runs] for s in splits}

    report = dict(runs=[os.path.basename(os.path.normpath(r)) for r in args.runs], data=os.path.basename(data), tta=args.tta,
                  single={}, fusion_iou_on_val={})
    for i, r in enumerate(report["runs"]):
        report["single"][r] = {s: evaluate(splits[s], preds[s][i], nc, names) for s in splits}
        print(f"{r}: " + "  ".join(f"{s} mAP50 {v['mAP50']:.3f} mAP50-95 {v['mAP50_95']:.3f}"
                                   for s, v in report["single"][r].items()))
    if len(args.runs) > 1:
        fused = {}
        for t in FUSION_IOUS:
            fv = {f: wbf([p[f] for p in preds["val"]], t).tolist() for f in splits["val"]}
            report["fusion_iou_on_val"][t] = evaluate(splits["val"], fv, nc, names)
            fused[t] = fv
            print(f"fusion IoU {t}: val mAP50 {report['fusion_iou_on_val'][t]['mAP50']:.3f} "
                  f"mAP50-95 {report['fusion_iou_on_val'][t]['mAP50_95']:.3f}")
        t = max(FUSION_IOUS, key=lambda t: (round(report["fusion_iou_on_val"][t]["mAP50_95"], 3),
                                             report["fusion_iou_on_val"][t]["mAP50"]))
        ft = {f: wbf([p[f] for p in preds["test"]], t).tolist() for f in splits["test"]}
        report["ensemble"] = dict(fusion_iou=t, val=report["fusion_iou_on_val"][t], test=evaluate(splits["test"], ft, nc, names))
        e = report["ensemble"]
        print(f"ENSEMBLE (fusion IoU {t}, chosen on val): val mAP50 {e['val']['mAP50']:.3f} mAP50-95 "
              f"{e['val']['mAP50_95']:.3f} | test mAP50 {e['test']['mAP50']:.3f} mAP50-95 {e['test']['mAP50_95']:.3f}")
    name = args.name or "+".join(report["runs"]) + ("_tta" if args.tta else "") + "_" + report["data"]
    out = os.path.join(ROOT, "runs", "analysis", "ensemble", name)
    os.makedirs(out, exist_ok=True)
    json.dump(report, open(os.path.join(out, "metrics.json"), "w"), indent=2, default=float)
    print(f"written {out}/metrics.json")


if __name__ == "__main__":
    main()
