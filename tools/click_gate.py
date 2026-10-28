"""Gate G17 (Phase 31): one click per defect (location + class) on the training images outside the class-matched 25%
subset; the 25% detector supplies the box. Definitions and the bar are pre-registered in research_log.md (G17).

pseudo-box = highest-confidence prediction of the clicked class containing the click (conf >= MIN_CONF); otherwise a
box of the class's median size in the 25% subset, centred on the click and clipped to the image.

  .venv/bin/python tools/click_gate.py  -> runs/analysis/click_gate/{report.md, clicks.csv}
"""

import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import bootstrap_ci as bc  # noqa: E402
import detection_metrics as dm  # noqa: E402

DATA = os.path.join(ROOT, "external", "WTBD-group")
SPLITS = os.path.join(ROOT, "tools", "splits")
RUNS = os.path.join(ROOT, "runs", "detect", "BladeYOLO_WindSurface")
MODELS = ["bladeyolo2b_wtbd_group_f25_s0", "bladeyolo2b_wtbd_group_f25_s1"]
OUT = os.path.join(ROOT, "runs", "analysis", "click_gate")
CLASSES = ["craze", "corrosion", "surface_injure", "thunderstrike", "crack", "hide_craze"]
MIN_CONF, PASS = 0.05, 0.60


def clicks(gt, rng):
    """(n, 2) one point drawn uniformly inside each [cls, x1, y1, x2, y2] box."""
    return np.c_[rng.uniform(gt[:, 1], gt[:, 3]), rng.uniform(gt[:, 2], gt[:, 4])] if len(gt) else np.zeros((0, 2))


def pseudo_box(click, cls, pred, size, img_wh, min_conf=MIN_CONF):
    """(box [x1, y1, x2, y2], from_detector) for one click; pred rows are x1 y1 x2 y2 conf cls."""
    x, y = click
    if len(pred):
        p = pred[(pred[:, 5] == cls) & (pred[:, 4] >= min_conf) & (pred[:, 0] <= x) & (pred[:, 2] >= x)
                 & (pred[:, 1] <= y) & (pred[:, 3] >= y)]
        if len(p):
            return p[np.argmax(p[:, 4]), :4], True
    w, h = size
    W, H = img_wh
    return np.array([max(x - w / 2, 0), max(y - h / 2, 0), min(x + w / 2, W), min(y + h / 2, H)]), False


def main():
    os.makedirs(OUT, exist_ok=True)
    rest = set(open(os.path.join(SPLITS, "wtbd_group_train_f25_rest.txt")).read().split())
    sub = set(open(os.path.join(SPLITS, "wtbd_group_train_f25.txt")).read().split())
    tr = bc.load_yolo_split(DATA, "train")
    imgs = {k: v for k, v in tr.items() if os.path.basename(k) in rest}
    lab = np.concatenate([v["gt"] for k, v in tr.items() if os.path.basename(k) in sub])
    size = {c: np.median(lab[lab[:, 0] == c][:, 3:5] - lab[lab[:, 0] == c][:, 1:3], 0) for c in range(len(CLASSES))}
    print(f"{len(imgs)} images, {sum(len(v['gt']) for v in imgs.values())} boxes", flush=True)

    rows, ctx = [], []
    for m in MODELS:
        pred = bc.predict(os.path.join(RUNS, m), imgs, os.path.join(OUT, "preds"), batch=2)
        rng = np.random.default_rng(0)
        hit_gt = 0
        n_conf, n_correct = 0, 0
        for f in sorted(imgs):
            g, p = imgs[f]["gt"], pred.get(f, np.zeros((0, 6)))
            side = 640  # WTBD images are square; boxes are in 640-input pixels
            for gb, ck in zip(g, clicks(g, rng)):
                box, det = pseudo_box(ck, int(gb[0]), p, size[int(gb[0])], (side, side))
                iou = float(dm.iou_matrix(box[None], gb[None, 1:])[0, 0])
                rows.append(dict(model=m, image=os.path.basename(f), cls=CLASSES[int(gb[0])], from_detector=det,
                                 iou=iou))
            # context: plain pseudo-labels at tau 0.5
            q = p[p[:, 4] >= 0.5]
            n_conf += len(q)
            if len(q) and len(g):
                iou = dm.iou_matrix(q[:, :4], g[:, 1:]) * (q[:, 5:6] == g[None, :, 0])
                n_correct += int((iou.max(1) >= 0.5).sum())
                hit_gt += int((iou.max(0) >= 0.5).sum())
        ctx.append((m, n_conf, n_correct, hit_gt))
    with open(os.path.join(OUT, "clicks.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    iou = np.array([r["iou"] for r in rows])
    det = np.array([r["from_detector"] for r in rows])
    mod = np.array([r["model"] for r in rows])
    cls = np.array([r["cls"] for r in rows])
    per_model = [float((iou[mod == m] >= 0.5).mean()) for m in MODELS]
    rate = float(np.mean(per_model))
    n_gt = int((mod == MODELS[0]).sum())
    lines = ["# G17: click gate (pre-registered, research_log.md Phase 31)", "",
             f"{len(imgs)} images outside the class-matched 25% subset, {n_gt} ground-truth boxes, one click each.", "",
             f"**Clicks with a pseudo-box at IoU ≥ 0.5: {rate:.3f}** (s0 {per_model[0]:.3f}, s1 {per_model[1]:.3f}); "
             f"bar {PASS:.2f} → G17 {'**passes**' if rate >= PASS else 'fails'}.", "",
             f"- IoU ≥ 0.75: {(iou >= 0.75).mean():.3f}; mean IoU {iou.mean():.3f}",
             f"- box from the detector: {det.mean():.3f} of clicks; IoU ≥ 0.5 among those {(iou[det] >= 0.5).mean():.3f}, "
             f"among fallback boxes {(iou[~det] >= 0.5).mean():.3f}",
             f"- without the fallback (fallback counted as a miss): {((iou >= 0.5) & det).mean():.3f}", ""]
    for m, n, c, h in ctx:
        lines.append(f"- context, plain pseudo-labels at τ = 0.5 ({m}): {n} boxes, precision {c / max(n, 1):.3f}, "
                     f"ground truth recovered {h / n_gt:.3f}")
    lines += ["", "| class | clicks per model | IoU ≥ 0.5 | IoU ≥ 0.75 | from detector |", "|---|---|---|---|---|"]
    for c in CLASSES:
        s = cls == c
        lines.append(f"| {c} | {s.sum() // len(MODELS)} | {(iou[s] >= 0.5).mean():.3f} | {(iou[s] >= 0.75).mean():.3f} | "
                     f"{det[s].mean():.3f} |")
    open(os.path.join(OUT, "report.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
