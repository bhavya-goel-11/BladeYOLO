"""Evidence figures for the paper, part 2 (part 1: tools/paper_figures.py). Images are drawn from the dataset bundle with a
fixed seed, not picked by hand; numbers come from runs/analysis (paper/evidence.json, g27, g28, label audit, conformal)
or, where an analysis predates a JSON output, from research_log.md (section given per constant).
  .venv/bin/python tools/paper_figures2.py [fig ...]   -> paper/figures/fig{7..}_*.{pdf,png}
"""

import csv
import glob
import json
import os
import random
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from paper_figures import C, GRID, INK, INK2, OUT, SURF  # noqa: E402  (shared palette and rcParams)

B = os.path.join(ROOT, "data", "blade-defect-benchmark")
A = os.path.join(ROOT, "runs", "analysis")
RUNS = os.path.join(ROOT, "runs", "detect", "BladeYOLO_WindSurface")
EV = os.path.join(A, "paper", "evidence.json")
BOX = [(214, 120, 42), (52, 104, 235), (122, 175, 27), (0, 161, 237), (170, 80, 170), (60, 60, 60), (40, 170, 170)]  # BGR


def save(fig, name, png_only=False):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=220, bbox_inches="tight")
    if not png_only:
        fig.savefig(os.path.join(OUT, f"{name}.pdf"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def save_image(im, name):
    os.makedirs(OUT, exist_ok=True)
    cv2.imwrite(os.path.join(OUT, f"{name}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("wrote", name)


def read_labels(path):
    return [list(map(float, l.split()[:5])) for l in open(path) if l.strip()] if os.path.exists(path) else []


def label_path(img):
    return os.path.splitext(img.replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"))[0] + ".txt"


def fit(im, size):
    h, w = im.shape[:2]
    s = size / max(h, w)
    r = cv2.resize(im, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    out = np.full((size, size, 3), 255, np.uint8)
    y, x = (size - r.shape[0]) // 2, (size - r.shape[1]) // 2
    out[y:y + r.shape[0], x:x + r.shape[1]] = r
    return out, s, x, y


def caption(im, text, y=0, scale=0.5, bg=(255, 255, 255), fg=(20, 20, 20)):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(im, (0, y), (w + 6, y + h + 7), bg, -1)
    cv2.putText(im, text, (3, y + h + 3), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, 1, cv2.LINE_AA)


def tile(path, size=320, names=None, boxes=None, width=2):
    """An image with its YOLO labels (or explicit [(cls, x1, y1, x2, y2, text)] in pixels at the 640-long-side frame)."""
    im, s, ox, oy = fit(cv2.imread(path), size)
    H, W = cv2.imread(path).shape[:2]
    if boxes is None:
        for c, x, y, w, h in read_labels(label_path(path)):
            x1, y1, x2, y2 = (x - w / 2) * W * s + ox, (y - h / 2) * H * s + oy, (x + w / 2) * W * s + ox, (y + h / 2) * H * s + oy
            col = BOX[int(c) % len(BOX)]
            cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), col, width)
            if names:
                cv2.putText(im, names[int(c)][:14], (int(x1) + 2, max(int(y1) - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
    else:
        k = size / 640
        for c, x1, y1, x2, y2, txt in boxes:
            col = BOX[int(c) % len(BOX)]
            cv2.rectangle(im, (int(x1 * k + ox), int(y1 * k + oy)), (int(x2 * k + ox), int(y2 * k + oy)), col, width)
            if txt:
                cv2.putText(im, txt, (int(x1 * k + ox) + 2, max(int(y1 * k + oy) - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
    return im


def grid(rows, gap=4):
    h = rows[0][0].shape[0]
    out = []
    for r in rows:
        line = []
        for j, t in enumerate(r):
            line.append(t)
            if j < len(r) - 1:
                line.append(np.full((h, gap, 3), 255, np.uint8))
        out.append(np.hstack(line))
        out.append(np.full((gap, out[-1].shape[1], 3), 255, np.uint8))
    w = max(o.shape[1] for o in out)
    out = [np.hstack([o, np.full((o.shape[0], w - o.shape[1], 3), 255, np.uint8)]) if o.shape[1] < w else o for o in out]
    return np.vstack(out[:-1])


def fig7_pipeline():
    """How the audit, the benchmarks and the scoring fit together."""
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 50)
    def box(x, y, w, h, title, sub, col):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=SURF, ec=col, lw=1.3))
        ax.text(x + w / 2, y + h - 1.8, title, ha="center", va="top", fontsize=6.8, color=INK, fontweight="bold")
        ax.text(x + w / 2, y + h - 5.4, sub, ha="center", va="top", fontsize=5.5, color=INK2, linespacing=1.25)
    def arrow(a, b):
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9))
    box(0, 34, 21, 15, "Public datasets", "WTBD, Wind Surface,\nWTBlade-Defect, Beijing,\nWTBs2025", C[0])
    box(26, 34, 21, 15, "Content audit", "DINOv2 retrieval, ORB and\nRANSAC with flips, pixel\ncorrelation, file names", C[0])
    box(52, 34, 21, 15, "Scene groups", "union-find: unique images,\nindependent scenes,\nper-image twin flags", C[0])
    box(78, 34, 22, 15, "Leak-free splits", "WTBD-group (58 test scenes),\nWTBlade-group (177),\nscene groups kept whole", C[2])
    box(0, 15, 30, 14, "Official protocols", "train as published; score leaked and\nno-twin test images separately", C[1])
    box(35, 15, 30, 14, "Ablations and method tests", "DINOv3 on/off, backbone size, label\nbudget, pre-registered tests, seeds", C[2])
    box(70, 15, 30, 14, "Leave-one-dataset-out", "train on the other datasets, twin-free\nagainst the target, class-agnostic", C[3])
    box(0, 0, 100, 10, "Scoring", "scene bootstrap (image bootstrap on official splits); standard, tolerant and region AP; recall guarantees; error anatomy", INK2)
    for a, b in (((21, 41.5), (26, 41.5)), ((47, 41.5), (52, 41.5)), ((73, 41.5), (78, 41.5))):
        arrow(a, b)
    arrow((58, 34), (15, 29)); arrow((86, 34), (50, 29)); arrow((89, 34), (85, 29))
    for x in (15, 50, 85):
        arrow((x, 15), (x, 10))
    save(fig, "fig7_pipeline")


def fig8_gallery():
    """Five labelled training images of each dataset, drawn with a fixed seed."""
    rng = random.Random(7)
    sets = [("WTBD", os.path.join(B, "raw", "WTBD-official"), "train/images/*"),
            ("Wind Surface", os.path.join(B, "raw", "WindSurface-Defect"), "train/images/*"),
            ("WTBlade-Defect", os.path.join(B, "raw", "WTBlade-Defect"), "train/images/*"),
            ("Beijing", os.path.join(B, "raw", "Beijing-wind-turbine"), "train/images/*"),
            ("WTBs2025", os.path.join(B, "raw", "WTBs2025"), "*/images/*")]
    rows = []
    for lab, d, pat in sets:
        names = yaml.safe_load(open(os.path.join(d, "data.yaml")))["names"]
        names = list(names.values()) if isinstance(names, dict) else names
        fs = sorted(glob.glob(os.path.join(d, pat)))
        rng.shuffle(fs)
        row, seen = [], set()
        for f in fs:
            if len(row) == 5:
                break
            lb = read_labels(label_path(f))
            cl = {int(x[0]) for x in lb}
            if not lb or (len(row) < 3 and cl <= seen):  # the first three show classes not yet shown
                continue
            seen |= cl
            t = tile(f, 300, names)
            if not row:
                caption(t, lab, 0, 0.55, (30, 30, 30), (255, 255, 255))
            row.append(t)
        rows.append(row)
    save_image(grid(rows), "fig8_dataset_gallery")


def fig9_effective_size():
    """Files vs unique images vs independent scenes (runs/analysis/datasets/report.md), and boxes per class."""
    d = [("WTBD", 1065, 597, 550), ("Wind Surface", 3570, 954, 81), ("WTBlade", 10660, 1146, 803), ("Beijing", 8922, 2196, 164),
         ("Beijing check", 6563, 2005, 144), ("WTBs2025", 7544, 4145, 3403)]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = axes[0]
    x = np.arange(len(d))
    for j, (lab, col) in enumerate((("files", C[0]), ("unique images", C[1]), ("independent scenes", C[2]))):
        v = [r[j + 1] for r in d]
        xx = x + (j - 1) * 0.27
        ax.bar(xx, v, 0.25, color=col, edgecolor=SURF, label=lab)
        for a, b in zip(xx, v):
            ax.text(a, b * 1.08, f"{b:,}", ha="center", va="bottom", fontsize=4.9, color=INK, rotation=90)
    ax.set_yscale("log"); ax.set_ylim(40, 90000); ax.set_xticks(x); ax.set_xticklabels([r[0] for r in d], fontsize=6.2, rotation=15)
    ax.set_ylabel("count (log scale)"); ax.legend(fontsize=6, ncol=3, loc="upper left", columnspacing=0.8, handlelength=1)
    ax.set_title("(a) What the file count hides", fontsize=8, loc="left")
    # independent scenes per class, audit report (scenes with a learnable box)
    rows = []
    for line in open(os.path.join(A, "datasets", "report.md")):
        p = [s.strip() for s in line.split("|")]
        if len(p) > 10 and p[1] in ("ws", "wt", "wtbd") and p[2] != "class":
            rows.append((p[1], p[2], int(p[9].strip("*"))))
    lab = {"ws": "Wind Surface", "wt": "WTBlade", "wtbd": "WTBD"}
    ax = axes[1]
    pos, ticks = [], []
    y = 0
    for k, col in (("wtbd", C[0]), ("ws", C[1]), ("wt", C[2])):
        rs = sorted([r for r in rows if r[0] == k], key=lambda r: r[2])
        ys = np.arange(y, y + len(rs))
        ax.barh(ys, [r[2] for r in rs], 0.8, color=col, edgecolor=SURF, label=lab[k])
        for a, r in zip(ys, rs):
            ax.text(r[2] * 1.08, a, str(r[2]), va="center", fontsize=5.2, color=INK)
        pos += list(ys); ticks += [r[1] for r in rs]
        y += len(rs) + 1
    ax.set_xscale("log"); ax.set_yticks(pos); ax.set_yticklabels(ticks, fontsize=5.3); ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
    ax.set_xlabel("independent scenes with a learnable box"); ax.legend(fontsize=6, loc="center right")
    ax.set_title("(b) Scenes per class", fontsize=8, loc="left")
    fig.tight_layout()
    save(fig, "fig9_effective_size")


def fig10_twins():
    """Official test images and their training twins: WTBD same scene, Wind Surface _0/_1 pairs, WTBlade Roboflow copies."""
    rng = random.Random(11)
    size = 260
    wtbd = os.path.join(B, "raw", "WTBD-official")
    rows = list(csv.DictReader(open(os.path.join(A, "wtbd_leak", "test_leak.csv"))))
    rows = [r for r in rows if r["leaked"] == "1"]
    rng.shuffle(rows)
    pairs = []
    for r in rows[:4]:
        pairs.append((os.path.join(wtbd, "test", "images", r["image"]), os.path.join(wtbd, "train", "images", r["twin_in_train"]),
                      f"WTBD: cos {float(r['cosine']):.2f}, {r['inliers']} inliers"))
    ws = os.path.join(B, "raw", "WindSurface-Defect")
    val = sorted(os.listdir(os.path.join(ws, "val", "images")))
    train = set(os.listdir(os.path.join(ws, "train", "images")))
    cand = []
    for f in val:
        st, ext = os.path.splitext(f)
        if st.endswith(("_0", "_1")):
            o = st[:-1] + ("1" if st.endswith("_0") else "0") + ext
            if o in train:
                cand.append((f, o))
    rng.shuffle(cand)
    for a, b in cand[:4]:
        pairs.append((os.path.join(ws, "val", "images", a), os.path.join(ws, "train", "images", b), "Wind Surface: _0/_1 pair"))
    wt = os.path.join(B, "raw", "WTBlade-Defect")
    tr = {}
    for f in os.listdir(os.path.join(wt, "train", "images")):
        tr.setdefault(f.split(".rf.")[0], []).append(f)
    cand = [f for f in sorted(os.listdir(os.path.join(wt, "test", "images"))) if f.split(".rf.")[0] in tr]
    rng.shuffle(cand)
    for f in cand[:4]:
        pairs.append((os.path.join(wt, "test", "images", f), os.path.join(wt, "train", "images", tr[f.split(".rf.")[0]][0]), "WTBlade: Roboflow copy"))
    grid_rows = []
    for i in range(0, 12, 4):
        row = []
        for a, b, cap in pairs[i:i + 4]:
            ta, tb = tile(a, size), tile(b, size)
            caption(ta, "test", 0, 0.42, (214, 120, 42), (255, 255, 255)); caption(tb, "train", 0, 0.42, (40, 40, 40), (255, 255, 255))
            caption(ta, cap, size - 20, 0.36)
            row += [ta, tb]
        grid_rows.append(row)
    print("candidates: wtbd", len(rows), "ws", len(val), "wtblade", len(cand))
    save_image(grid(grid_rows), "fig10_twins")


def fig11_ambiguity():
    """Crops of the label audit's example sheets (random order, fixed at audit time): each pair is one image and its aligned
    copy. Left: a label (green). Right: its exact projection (yellow) and what the copy's annotator drew (red)."""
    def pairs(ds, kind, idx):
        s = cv2.imread(os.path.join(A, "novel", "label_audit", ds, f"examples_same_view_axis_{kind}.jpg"))
        out = []
        for k in idx:
            r, c = k // 3, k % 3
            t = s[r * 256:(r + 1) * 256, c * 2 * 256:(c * 2 + 2) * 256].copy()
            out.append(t)
        return out
    rows = []
    for ds, lab, kind, idx in (("wtbd", "WTBD, split", "split", [0, 3, 10]), ("wtbd", "WTBD, loose", "loose", [0, 1, 2]),
                               ("wtblade", "WTBlade, split", "split", [0, 1, 2]), ("windsurface", "Wind Surface, split", "split", [0, 1, 2])):
        ts = pairs(ds, kind, idx)
        caption(ts[0], lab, 236, 0.5, (30, 30, 30), (255, 255, 255))
        rows.append(ts)
    save_image(grid(rows), "fig11_label_ambiguity")


def fig12_agreement():
    """Per-class agreement on axis-aligned copies (label audit reports) and the extent ceiling (log, 'Box-extent direction')."""
    ds = [("WTBD", "wtbd", C[0]), ("Wind Surface", "windsurface", C[1]), ("WTBlade", "wtblade", C[2])]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw=dict(width_ratios=[1.45, 1]))
    ax = axes[0]
    y, pos, ticks = 0, [], []
    for lab, key, col in ds:
        st = json.load(open(os.path.join(A, "novel", "label_audit", key, "report.json")))["strata"]["same_view_axis"]
        v = sorted(((k, 100 * x["agree"] / max(sum(x.values()), 1)) for k, x in st["per_class"].items()), key=lambda t: t[1])
        ys = np.arange(y, y + len(v))
        ax.barh(ys, [t[1] for t in v], 0.8, color=col, edgecolor=SURF, label=lab)
        for a, t in zip(ys, v):
            ax.text(t[1] + 0.6, a, f"{t[1]:.0f}", va="center", fontsize=5.2, color=INK)
        pos += list(ys); ticks += [t[0] for t in v]
        y += len(v) + 1
    ax.set_yticks(pos); ax.set_yticklabels(ticks, fontsize=5.4); ax.set_xlim(50, 122); ax.set_xticks(range(50, 101, 10)); ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
    ax.set_xlabel("labels agreeing with the aligned copy, %"); ax.legend(fontsize=6, loc="lower right")
    ax.set_title("(a) The annotators against themselves", fontsize=8, loc="left")
    ax = axes[1]
    rows = [("annotators", 87, 70), ("RF-DETR-M 25%", 55, 67), ("ViT-B 25%", 41, 45), ("BladeYOLO-v2 100%", 63, 47)]
    x = np.arange(len(rows))
    for j, (k, lab, col) in enumerate(((1, "IoU ≥ 0.5, of all boxes", C[0]), (2, "IoU ≥ 0.75, of those", C[1]))):
        xx = x + (j - 0.5) * 0.38
        ax.bar(xx, [r[k] for r in rows], 0.36, color=col, edgecolor=SURF, label=lab)
        for a, r in zip(xx, rows):
            ax.text(a, r[k] + 1, str(r[k]), ha="center", fontsize=5.8, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=5.8, rotation=15); ax.set_ylim(0, 112); ax.set_ylabel("%")
    ax.legend(fontsize=5.8, loc="upper right", ncol=1)
    ax.set_title("(b) Box agreement, WTBD", fontsize=8, loc="left")
    fig.tight_layout()
    save(fig, "fig12_label_agreement")


def fig13_craze():
    """All 15 test craze images are copies of one scene (log §11). Sheet made at the time (red: labels; blue: predictions)."""
    im = cv2.imread(os.path.join(A, "wtbd_craze", "test_craze.jpg"))
    save_image(im[:640], "fig13_craze_scene")


def fig14_conformal():
    """Conformal recall: realised miss rate (log, Phase 16-17) and the operating curve (runs/analysis/novel/conformal/curve.csv)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    ax = axes[0]
    rows = [("seen,\nimage units", 31.5), ("seen,\nscene units", 28.8), ("official val\n(leaky)", 25.9), ("unseen,\nimage units", 6.3), ("unseen,\nscene units", 4.7)]
    cols = [C[1], C[1], C[1], C[2], C[2]]
    ax.bar(range(5), [r[1] for r in rows], 0.6, color=cols, edgecolor=SURF)
    for i, r in enumerate(rows):
        ax.text(i, r[1] + 0.6, f"{r[1]:.1f}", ha="center", fontsize=6.5, color=INK)
    ax.axhline(10, color=INK, lw=0.9, ls="--"); ax.text(4.45, 10.6, "promised ≤ 10%", ha="right", fontsize=6.2, color=INK)
    ax.set_xticks(range(5)); ax.set_xticklabels([r[0] for r in rows], fontsize=5.8); ax.set_ylabel("defects missed on new scenes, %")
    ax.set_title("(a) The guarantee depends on the calibration set", fontsize=8, loc="left")
    ax = axes[1]
    rows = list(csv.DictReader(open(os.path.join(A, "novel", "conformal", "curve.csv"))))
    for m, lab, col in (("bladeyolo2", "BladeYOLO-v2", C[0]), ("yolo12l", "YOLO12-L", C[1])):
        r = [x for x in rows if x["model"].startswith(m) and x["flagging"] == "any-class"]
        r = sorted(r, key=lambda x: float(x["alarms_per_image"]))
        ax.plot([float(x["alarms_per_image"]) for x in r], [100 * float(x["recall"]) for x in r], color=col, lw=1.6, label=lab)
    ax.set_xscale("log"); ax.set_xlim(0.02, 10); ax.set_ylim(40, 101); ax.set_xlabel("false alarms per image (log scale)"); ax.set_ylabel("defects flagged, %")
    ax.legend(fontsize=6.5, loc="lower right"); ax.set_title("(b) Operating curve, unseen-scene calibration", fontsize=8, loc="left")
    fig.tight_layout()
    save(fig, "fig14_conformal")


def curve(run, key="val/ema_mAP_50_95"):
    rows = [r for r in csv.DictReader(open(os.path.join(RUNS, run, "metrics.csv"))) if r.get(key)]
    return [int(r["epoch"]) + 1 for r in rows], [100 * float(r[key]) for r in rows]


def train_loss(run):
    e = {}
    for r in csv.DictReader(open(os.path.join(RUNS, run, "metrics.csv"))):
        if r.get("train/loss"):
            e.setdefault(int(r["epoch"]), []).append(float(r["train/loss"]))
    k = sorted(e)
    return [i + 1 for i in k], [np.mean(e[i]) for i in k]


def fig15_training():
    """Validation curves from every run's metrics.csv / results.csv."""
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.1))
    arms = [("A0", "", C[0], "-"), ("A0 s1", "_s1", C[0], ":"), ("A1", "cp_", C[1], "-"), ("A1 s1", "cp_|_s1", C[1], ":"), ("A2", "cp_d1_", C[2], "-"), ("A3", "cp_d2_", C[3], "-")]
    for ax, base, tail, title in ((axes[0, 0], "rfdetrm768cos_", "wtbd_group_full", "(a) WTBD-group, 768 px"), (axes[0, 1], "rfdetrm640cos_", "wtblade_group_full", "(b) WTBlade-group, 640 px")):
        for lab, k, col, ls in arms:
            pre, suf = (k.split("|") + [""])[:2] if k.startswith("cp") else ("", k)
            x, y = curve(base + pre + tail + suf)
            ax.plot(x, y, color=col, ls=ls, lw=1.0, label=lab)
        ax.set_title(title, fontsize=8, loc="left"); ax.set_xlabel("epoch"); ax.set_ylabel("val mAP50-95, %")
    axes[0, 0].legend(fontsize=5.5, ncol=2, loc="lower right")
    ax = axes[0, 2]
    for run, lab, col in (("rfdetrm768cos_wtbd_group_full", "WTBD A0", C[0]), ("rfdetrm640cos_wtblade_group_full", "WTBlade A0", C[1])):
        x, y = train_loss(run)
        ax.plot(x, y, color=col, lw=1.2, label=lab)
    ax.set_title("(c) Training loss", fontsize=8, loc="left"); ax.set_xlabel("epoch"); ax.set_ylabel("mean train loss"); ax.legend(fontsize=6)
    ax = axes[1, 0]
    for run, lab, col, ls in (("rfdetrm640ag_id_wtbd_group", "ID-WTBD (target val)", C[0], "-"), ("rfdetrm640ag_id_wtblade_group", "ID-WTBlade (target val)", C[1], "-")):
        x, y = curve(run)
        ax.plot(x, y, color=col, lw=1.2, ls=ls, label=lab)
    ax.set_title("(d) G28 ID arms", fontsize=8, loc="left"); ax.set_xlabel("epoch"); ax.set_ylabel("val mAP50-95, %"); ax.legend(fontsize=5.8)
    ax = axes[1, 1]
    for run, lab, col in (("rfdetrm640ag_x_wtbd_group", "X-WTBD", C[0]), ("rfdetrm640ag_x_wtblade_group", "X-WTBlade", C[1])):
        x, y = curve(run)
        ax.plot(x, y, color=col, lw=1.4, marker="o", ms=2.5, label=lab)
    ax.set_title("(e) G28 X arms, own 5% val", fontsize=8, loc="left"); ax.set_xticks([1, 3, 6, 9, 12]); ax.set_xlabel("epoch"); ax.set_ylabel("val mAP50-95, %"); ax.legend(fontsize=6)
    ax = axes[1, 2]
    rows = list(csv.DictReader(open(os.path.join(RUNS, "yolo12l_wtblade_group", "results.csv"))))
    ax.plot([int(r["epoch"]) for r in rows], [100 * float(r["metrics/mAP50-95(B)"]) for r in rows], color=C[0], lw=1.2, label="mAP50-95")
    ax.plot([int(r["epoch"]) for r in rows], [100 * float(r["metrics/mAP50(B)"]) for r in rows], color=C[1], lw=1.2, label="mAP50")
    ax.set_title("(f) YOLO12-L, WTBlade-group", fontsize=8, loc="left"); ax.set_xlabel("epoch"); ax.set_ylabel("val, %"); ax.legend(fontsize=6)
    fig.tight_layout()
    save(fig, "fig15_training_curves")


def fig16_g27():
    """G27: runs/analysis/g27/score.json (points); intervals from runs/analysis/g27/score.md, copied."""
    s = json.load(open(os.path.join(A, "g27", "score.json")))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw=dict(width_ratios=[1, 1.1]))
    ax = axes[0]
    rows = [("A1 CP, WTBD", 0.04, -5.7, 4.4, C[1]), ("A1 CP, WTBlade", 1.4, -0.5, 5.5, C[1]), ("A2 CP + D1, WTBD", -5.9, -14.1, 0.5, C[2]), ("A2 CP + D1, WTBlade", -0.0, -5.6, 1.4, C[2]),
            ("A3 CP + D2, WTBD", -6.8, -9.9, -0.9, C[3]), ("A3 CP + D2, WTBlade", 2.0, -1.7, 3.4, C[3])][::-1]
    for i, (lab, p, lo, hi, col) in enumerate(rows):
        ax.plot([lo, hi], [i, i], color=INK2, lw=1); ax.plot(p, i, "o", color=col, ms=5, mec=SURF)
    ax.axvline(0, color=INK, lw=0.8); ax.set_yticks(range(6)); ax.set_yticklabels([r[0] for r in rows], fontsize=6.2)
    ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True); ax.set_xlabel("Δ val mAP50-95 vs A0, points [95% CI]")
    ax.set_title("(a) A-1: selection on validation, seed 0", fontsize=8, loc="left")
    ax = axes[1]
    pos, ticks = [], []
    y = 0
    for ds, col in (("WTBlade-group", C[1]), ("WTBD-group", C[0])):
        for arm, sd, m in (("A1", 1, "s"), ("A1", 0, "s"), ("A0", 1, "o"), ("A0", 0, "o")):
            v = 100 * s[ds][f"{arm}|{sd}|test"]
            ax.plot(v, y, m, color=col, ms=5, mec=SURF)
            ax.text(v + 0.5, y, f"{v:.1f}", va="center", fontsize=5.8, color=INK)
            pos.append(y); ticks.append(f"{ds.split('-')[0]} {arm} seed {sd}")
            y += 1
        y += 0.7
    ax.set_yticks(pos); ax.set_yticklabels(ticks, fontsize=5.8); ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
    ax.set_xlabel("test mAP50-95, %"); ax.set_xlim(12, 37)
    ax.set_title("(b) A-2: copy-paste vs plain, two seeds", fontsize=8, loc="left")
    fig.tight_layout()
    save(fig, "fig16_g27")


def fig17_benchmark():
    """The benchmark: leak-free test by model (evidence.json) and official vs leak-free for one detector."""
    e = json.load(open(EV))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw=dict(width_ratios=[1, 1, 1.05]))
    for ax, ds, order, title in ((axes[0], "WTBD-group", ["YOLO12-L", "BladeYOLO-v2 without DINOv3", "BladeYOLO-v2 (ViT-S)", "RF-DETR-M A0 seed 0", "RF-DETR-M A0 seed 1"], "(a) WTBD-group test"),
                                 (axes[1], "WTBlade-group", ["YOLO12-L", "BladeYOLO-v2 (ViT-B)", "RF-DETR-M A0 seed 0", "RF-DETR-M A0 seed 1"], "(b) WTBlade-group test")):
        r = e["models"][ds]["runs"]
        x = np.arange(len(order))
        for j, (k, lab, col) in enumerate((("mAP50", "mAP50", C[0]), ("mAP", "mAP50-95", C[1]))):
            xx = x + (j - 0.5) * 0.38
            v = [100 * r[m][k] for m in order]
            ax.bar(xx, v, 0.36, color=col, edgecolor=SURF, label=lab)
            for a, b in zip(xx, v):
                ax.text(a, b + 0.8, f"{b:.1f}", ha="center", fontsize=4.9, color=INK)
        short = [m.replace("BladeYOLO-v2", "BYv2").replace(" without DINOv3", " no DINO").replace("RF-DETR-M A0 seed ", "RF-DETR s").replace(" (ViT-S)", " ViT-S").replace(" (ViT-B)", " ViT-B") for m in order]
        ax.set_xticks(x); ax.set_xticklabels(short, fontsize=5.1, rotation=35, ha="right"); ax.set_ylim(0, 68)
        ax.set_title(title, fontsize=8, loc="left")
    axes[0].set_ylabel("test score, %"); axes[0].legend(fontsize=5.8, loc="upper left", ncol=2, columnspacing=0.8, handlelength=1)
    ax = axes[2]
    a0 = lambda ds, k: 100 * e["models"][ds]["runs"]["RF-DETR-M A0 seed 0"][k]
    lines = [("WTBD mAP50", 90.4, a0("WTBD-group", "mAP50"), C[0], "-"), ("WTBD mAP50-95", 61.2, a0("WTBD-group", "mAP"), C[0], "--"),
             ("WTBlade mAP50", 87.2, a0("WTBlade-group", "mAP50"), C[1], "-"), ("WTBlade mAP50-95", 55.1, a0("WTBlade-group", "mAP"), C[1], "--")]
    off = {"WTBD mAP50": 1.5, "WTBlade mAP50": -1.5, "WTBD mAP50-95": 1.3, "WTBlade mAP50-95": -1.3}
    for lab, o, l, col, ls in lines:
        ax.plot([0, 1], [o, l], color=col, ls=ls, lw=1.4, marker="o", ms=3.5)
        ax.text(-0.04, o + off[lab] * 0.3, f"{o:.1f}", ha="right", va="center", fontsize=5.6, color=INK)
        ax.text(1.04, l + off[lab] * 0.3, f"{l:.1f}", ha="left", va="center", fontsize=5.6, color=INK)
    ax.plot([], [], color=C[0], lw=1.4, label="WTBD"); ax.plot([], [], color=C[1], lw=1.4, label="WTBlade")
    ax.plot([], [], color=INK2, lw=1.2, ls="-", label="mAP50"); ax.plot([], [], color=INK2, lw=1.2, ls="--", label="mAP50-95")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["official split", "leak-free split"], fontsize=6); ax.set_xlim(-0.3, 1.3); ax.set_ylim(0, 100)
    ax.legend(fontsize=5.4, loc="lower left", ncol=2, columnspacing=0.8, handlelength=1.4)
    ax.set_title("(c) RF-DETR-M, split only", fontsize=8, loc="left")
    fig.tight_layout()
    save(fig, "fig17_benchmark")


def fig18_g28():
    """G28: runs/analysis/g28/score.json, the whole-image baseline (scope_region_baseline.log), PR curves from evidence.json."""
    s = json.load(open(os.path.join(A, "g28", "score.json")))
    e = json.load(open(EV))
    trivial = {"WTBD-group": (0.0, 14.0), "WTBlade-group": (1.0, 24.2)}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.5))
    for row, ds in enumerate(("WTBD-group", "WTBlade-group")):
        ax = axes[row, 0]
        p = s[ds]["point"]
        met = [("std50", "AP50"), ("tol50", "tolerant\nAP50"), ("region", "region\nAP"), ("ar50", "recall\nAR50")]
        x = np.arange(len(met))
        for j, (a, lab, col) in enumerate((("ID", "ID (target labels)", C[0]), ("X", "X (other datasets)", C[1]))):
            xx = x + (j - 0.5) * 0.38
            v = [100 * p[a][m[0]] for m in met]
            ax.bar(xx, v, 0.36, color=col, edgecolor=SURF, label=lab)
            for u, w in zip(xx, v):
                ax.text(u, w + 1.5, f"{w:.1f}", ha="center", fontsize=5.8, color=INK)
        t50, treg = trivial[ds]
        ax.plot([-0.45, 0.45], [t50, t50], color=INK, lw=1.2, ls="--")
        ax.plot([1.55, 2.45], [treg, treg], color=INK, lw=1.2, ls="--", label="whole-image box")
        ax.set_xticks(x); ax.set_xticklabels([m[1] for m in met], fontsize=6.2); ax.set_ylim(0, 118); ax.set_ylabel("%")
        ax.set_title(f"({'ac'[row]}) {ds} test", fontsize=8, loc="left")
        ax.legend(fontsize=5.8, ncol=3, loc="upper left", columnspacing=0.8, handlelength=1.4)
        ax = axes[row, 1]
        r = e["r_grid"]
        for a, lab, col in (("ID", "ID", C[0]), ("ref", "G27 A0, classes merged", C[2]), ("X", "X", C[1])):
            pr = e["g28_pr"][ds][a]
            mr = pr["max_recall"]
            xs = [100 * v for v in r if v <= mr]
            ax.plot(xs, [100 * v for v in pr["precision"][:len(xs)]], color=col, lw=1.5, label=f"{lab}: best F1 {pr['best_f1']:.2f}")
        ax.set_xlim(0, 100); ax.set_ylim(0, 102); ax.set_xlabel("recall at IoU 0.5, %"); ax.set_ylabel("precision, %"); ax.legend(fontsize=5.8, loc="upper right", frameon=True, framealpha=0.92, edgecolor=SURF)
        ax.set_title(f"({'bd'[row]}) Class-agnostic PR, {ds}", fontsize=8, loc="left")
    fig.tight_layout()
    save(fig, "fig18_g28")


def rf_boxes(run, path, conf, nc_names=None):
    d = json.load(open(os.path.join(RUNS, run, "preds_test.json")))
    k = next((x for x in (path, os.path.basename(path)) if x in d), None)
    return [] if k is None else [r for r in d[k] if r[4] >= conf]


def fig19_predictions():
    """Ground truth, ID and X on seeded random test images. Boxes with confidence >= 0.3."""
    rng = random.Random(3)
    rows = []
    sets = (("WTBD-group", "rfdetrm640ag_id_wtbd_group", "rfdetrm640ag_x_wtbd_group"), ("WTBlade-group", "rfdetrm640ag_id_wtblade_group", "rfdetrm640ag_x_wtblade_group"))
    per = []
    for ds, idr, xr in sets:
        d = os.path.join(B, "leakfree", ds, "test", "images")
        fs = sorted(os.listdir(d))
        fs = [f for f in fs if read_labels(label_path(os.path.join(d, f)))]
        rng.shuffle(fs)
        tiles = []
        for f in fs[:4]:
            p = os.path.join(d, f)
            gt = tile(p, 240)
            h, w = cv2.imread(p).shape[:2]
            s = 640 / max(h, w)
            caption(gt, f"{ds.split('-')[0]}: labels", 0, 0.4)
            row = [gt]
            for run, lab in ((idr, "ID"), (xr, "X")):
                pr = json.load(open(os.path.join(RUNS, run, "preds_test.json")))
                key = next(k for k in pr if os.path.basename(k) == f)
                rows_ = [r for r in np.array(pr[key]).reshape(-1, 6).tolist() if r[4] >= 0.3]
                # stored boxes are at the 640-long-side frame used by the evaluator
                t = tile(p, 240, boxes=[(1 if lab == "ID" else 2, r[0], r[1], r[2], r[3], f"{r[4]:.2f}") for r in rows_])
                caption(t, f"{lab}: {len(rows_)} box{'' if len(rows_) == 1 else 'es'}", 0, 0.4)
                row.append(t)
            tiles.append(row)
        per.append(tiles)
    all_rows = [per[0][i] + per[1][i] for i in range(4)]
    save_image(grid(all_rows), "fig19_predictions")


def fig20_pools():
    """G28 pools: composition and what the twin rule removed (evidence.json)."""
    e = json.load(open(EV))["pools"]
    src = [("wtbd", "WTBD"), ("wtblade", "WTBlade"), ("ws", "Wind Surface"), ("bj", "Beijing"), ("wtbs", "WTBs2025")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.2))
    for ax, pool, title in ((axes[0], "XWTBD", "(a) X-WTBD pool (target: WTBD)"), (axes[1], "XWTBL", "(b) X-WTBlade pool (target: WTBlade-Defect)")):
        y = np.arange(len(src))[::-1]
        keep = np.array([e[pool].get(f"{k}|kept", 0) for k, _ in src])
        twin = np.array([sum(v for kk, v in e[pool].items() if kk.startswith(k + "|twin") or kk == f"{k}|closure") for k, _ in src])
        drop = np.array([e[pool].get(f"{k}|no_boxes", 0) + e[pool].get(f"{k}|gan", 0) for k, _ in src])
        ax.barh(y, keep, 0.7, color=C[0], edgecolor=SURF, label="kept")
        ax.barh(y, twin, 0.7, left=keep, color=C[1], edgecolor=SURF, label="removed: twin or closure")
        ax.barh(y, drop, 0.7, left=keep + twin, color=C[3], edgecolor=SURF, label="removed: no box / GAN")
        for a, k, t, d, (key, _) in zip(y, keep, twin, drop, src):
            txt = "target, held out" if e[pool].get(f"{key}|eval") else f"{k:,} kept, {t:,} twins"
            ax.text(k + t + d + 150 if k else 150, a, txt, va="center", fontsize=5.6, color=INK)
        ax.set_yticks(y); ax.set_yticklabels([s[1] for s in src], fontsize=6.3); ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
        ax.set_xlim(0, 15500); ax.set_xlabel("images"); ax.set_title(title, fontsize=8, loc="left")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=6, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "fig20_pools")


def fig21_pool_audit():
    im = cv2.imread(os.path.join(A, "g27", "pool_XWTBL_audit_pixel.jpg"))
    h = im.shape[0]
    save_image(im[:min(h, 640)], "fig21_pool_audit")


def fig22_failed():
    """Pre-registered method tests with an interval (log entries named per row; test mAP50-95 points unless noted)."""
    rows = [("Multi-view fusion at test time, WTBlade", -0.8, -1.5, 0.0), ("Retrieval prior as input (H3)", 0.1, -2.0, 3.0),
            ("Reward fine-tuning, region AP, 4 seeds (H5)", 3.6, -1.6, 6.0), ("Generated minus copy-paste defects (H10)", -2.5, None, None),
            ("Matched copy-paste minus none (H10)", 3.7, 1.1, 8.1), ("Class-selective tolerance, corrosion only", 9.3, -0.3, 24.6),
            ("Label-free selection minus random (H6)", 5.2, 1.9, 10.5), ("Label-free selection minus FreeSel (H6)", 2.1, -1.2, 9.1),
            ("G27 copy-paste, mean of two datasets", -0.3, -4.1, 0.9), ("Flip TTA on RF-DETR, WTBD (G21)", -2.9, None, None),
            ("Query-consensus rescoring, WTBD (G20)", -4.6, None, None), ("Prototype rescoring, WTBD (G15-R)", 1.1, None, None),
            ("Learned re-ranker, WTBD (H18)", 2.5, None, None)][::-1]
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    for i, (lab, p, lo, hi) in enumerate(rows):
        if lo is not None:
            ax.plot([lo, hi], [i, i], color=INK2, lw=1)
        ax.plot(p, i, "o", color=C[0] if lo is not None else SURF, mec=C[0], ms=5)
        ax.text(41, i, f"{p:+.1f}" + (f" [{lo:+.1f}, {hi:+.1f}]" if lo is not None else ""), va="center", ha="right", fontsize=5.8, color=INK)
    ax.axvline(0, color=INK, lw=0.8); ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=6.3)
    ax.set_xlim(-15, 41.5); ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True); ax.set_xlabel("effect, points (hollow marker: no interval logged)")
    fig.tight_layout()
    save(fig, "fig22_failed_tests")


def fig23_timeline():
    """Reported score at each stage of the project, in order (log §2, §5, §7, §11, §12, G26, G27, this figure's table in the paper)."""
    s = json.load(open(EV))["models"]
    a0 = s["WTBD-group"]["runs"]["RF-DETR-M A0 seed 0"]
    rows = [("BladeYOLO paper, Wind Surface", 86.5, "pub"), ("Faithful reproduction (Run 3)", 52.6, "leak"), ("BladeYOLO-L, Wind Surface (Run 6)", 78.2, "leak"),
            ("YOLO12-L, merged leak-free v2", 35.8, "free"), ("BladeYOLO-L, merged leak-free v2", 40.0, "free"), ("BladeYOLO-v1, merged leak-free v3", 31.7, "free"),
            ("BladeYOLO-v2, WTBD official", 84.0, "leak"), ("BladeYOLO-v2, WTBD-group", 54.8, "free"), ("YOLO12-L, WTBD official", 70.4, "leak"), ("YOLO12-L, WTBD-group", 43.8, "free"),
            ("RF-DETR-M, WTBD official", 90.4, "leak"), ("RF-DETR-M, WTBD-group", 100 * a0["mAP50"], "free"),
            ("RF-DETR-M, WTBlade official", 87.2, "leak"), ("RF-DETR-M, WTBlade-group", 100 * s["WTBlade-group"]["runs"]["RF-DETR-M A0 seed 0"]["mAP50"], "free")]
    col = {"pub": C[3], "leak": C[1], "free": C[0]}
    lab = {"pub": "published", "leak": "split with train/test twins", "free": "leak-free split"}
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    y = np.arange(len(rows))[::-1]
    for a, (n, v, k) in zip(y, rows):
        ax.barh(a, v, 0.72, color=col[k], edgecolor=SURF)
        ax.text(v + 0.8, a, f"{v:.1f}", va="center", fontsize=6, color=INK)
    for k in ("pub", "leak", "free"):
        ax.barh([0], [0], color=col[k], label=lab[k])
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=6.2); ax.set_xlim(0, 100); ax.set_xlabel("test mAP50, %, in project order (Wind Surface runs: val = test)")
    ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, fontsize=6.5, loc="upper center", ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fig23_score_history")


def fig24_perclass():
    """Per-class test AP50 on the leak-free benchmarks (evidence.json), one light-to-dark hue."""
    e = json.load(open(EV))["models"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw=dict(width_ratios=[1, 1.15]))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#2a78d6", "#0f3a70"])
    for ax, ds, order in ((axes[0], "WTBD-group", ["YOLO12-L", "BladeYOLO-v2 without DINOv3", "BladeYOLO-v2 (ViT-S)", "RF-DETR-M A0 seed 0", "RF-DETR-M A0 seed 1", "RF-DETR-M A1 seed 0", "RF-DETR-M A1 seed 1"]),
                          (axes[1], "WTBlade-group", ["YOLO12-L", "BladeYOLO-v2 (ViT-B)", "RF-DETR-M A0 seed 0", "RF-DETR-M A0 seed 1", "RF-DETR-M A1 seed 0", "RF-DETR-M A1 seed 1"])):
        r = e[ds]
        m = np.array([[100 * x for x in r["runs"][o]["ap50"]] for o in order])
        ax.imshow(m, cmap=cmap, vmin=0, vmax=100, aspect="auto")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f"{m[i, j]:.0f}", ha="center", va="center", fontsize=6, color="white" if m[i, j] > 52 else INK)
        ax.set_xticks(range(len(r["classes"]))); ax.set_xticklabels(r["classes"], fontsize=5.6, rotation=30, ha="right")
        ax.set_yticks(range(len(order))); ax.set_yticklabels([o.replace("BladeYOLO-v2", "BYv2").replace(" without DINOv3", " no DINO").replace("RF-DETR-M ", "RF-DETR ").replace(" seed ", " s") for o in order], fontsize=5.6)
        ax.grid(False); ax.set_title(f"{ds} test AP50, %", fontsize=8, loc="left")
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.tight_layout()
    save(fig, "fig24_per_class")


def fig25_cp_check():
    """The check sheet made before the G27 runs (tools/g27_paste.py)."""
    im = cv2.imread(os.path.join(A, "g27", "cp_check.jpg"))
    save_image(im, "fig25_cp_check")


def fig26_training_examples():
    d = os.path.join(RUNS, "bladeyolo2_wtbd_group")
    ims = [cv2.imread(os.path.join(d, f)) for f in ("train_batch0.jpg", "val_batch0_labels.jpg", "val_batch0_pred.jpg")]
    h = 620
    ims = [cv2.resize(i, (int(i.shape[1] * h / i.shape[0]), h), interpolation=cv2.INTER_AREA) for i in ims]
    save_image(grid([ims]), "fig26_training_examples")


def fig27_diagnostics():
    d = RUNS
    fs = [("bladeyolo2_wtbd_group/confusion_matrix_normalized.png", "BladeYOLO-v2, WTBD-group val"), ("yolo12l_wtbd_group/test/confusion_matrix_normalized.png", "YOLO12-L, WTBD-group test"),
          ("yolo12l_wtblade_group/test/confusion_matrix_normalized.png", "YOLO12-L, WTBlade-group test"), ("bladeyolo2_wtbd_group/BoxPR_curve.png", "BladeYOLO-v2, WTBD-group val")]
    h = 620
    out = []
    for f, cap in fs:
        i = cv2.imread(os.path.join(d, f))
        i = cv2.resize(i, (int(i.shape[1] * h / i.shape[0]), h), interpolation=cv2.INTER_AREA)
        caption(i, cap, 0, 0.5)
        out.append(i)
    save_image(grid([out[:2], out[2:]]), "fig27_diagnostics")


def fig28_residual_twins():
    im = cv2.imread(os.path.join(A, "split_audit", "WTBD-group_test.jpg"))
    save_image(im[:320, :], "fig28_residual_twins_raw")
    t = im[:320, :].copy()
    t[160:, 652:] = 255
    save_image(t, "fig28_residual_twins")
    os.remove(os.path.join(OUT, "fig28_residual_twins_raw.jpg"))


FIGS = {k[3:]: v for k, v in list(globals().items()) if k.startswith("fig") and callable(v)}

if __name__ == "__main__":
    for k, f in FIGS.items():
        if len(sys.argv) < 2 or any(a in k for a in sys.argv[1:]):
            f()
