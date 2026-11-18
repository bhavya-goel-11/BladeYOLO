"""Figures for the IEEE version of the paper (paper/ieee/): G29 (RF-DETR + frozen DINOv3 prior) and the official-protocol
comparison with BladeYOLO. Reads runs/analysis/paper/{official,paired}.json and the run folders; the G29 scene-bootstrap
intervals are copied from runs/analysis/g29/score.md (tools/g29_score.py), whose JSON keeps only point estimates and P.
  .venv/bin/python tools/paper_figures3.py [name ...]   -> paper/figures/fig29..fig33
"""

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_figures import C, GRID, INK, INK2, SURF  # noqa: E402,F401  (shared palette and rcParams)
from paper_figures2 import (B, RUNS, caption, curve, grid, label_path, read_labels, save,  # noqa: E402
                            save_image, tile)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "runs", "analysis", "paper")
ARCH = os.path.join(ROOT, "runs", "analysis", "archive", "kaggle_a0s1")

# runs/analysis/g29/score.md: FP - A0, scene bootstrap, mean of seeds 0 and 1, mAP50-95 points
G29_P = [("WTBD-group", "test", 2.18, -0.8, 4.4, 0.916), ("WTBlade-group", "test", 0.72, -0.6, 1.9, 0.866),
         ("mean", "test", 1.45, -0.2, 2.7, 0.958), ("WTBD-group", "val", 0.43, -3.1, 3.3, 0.569),
         ("WTBlade-group", "val", -0.02, -2.8, 2.2, 0.465), ("mean", "val", 0.20, -2.0, 1.9, 0.526)]
G29_S = [("class-matched", 5.69, 0.8, 19.5), ("rand1", -0.05, -8.1, 5.2), ("rand2", 0.55, -6.5, 12.5),
         ("rand3", -0.69, -5.5, 3.6), ("family mean", 1.38, -1.8, 6.2)]
G29_PC = {"WTBD-group": {"craze": (5.2, 9.0), "corrosion": (26.1, 26.7), "surface_injure": (61.1, 66.9),
                         "thunderstrike": (25.9, 27.0), "crack": (28.3, 29.7), "hide_craze": (46.5, 46.9)},
          "WTBlade-group": {"burn": (11.1, 10.7), "crack": (5.4, 5.9), "deformity": (24.2, 24.0), "dirt": (26.2, 27.0),
                            "oil": (9.3, 11.2), "peeling": (29.6, 31.2), "rust": (10.0, 10.8)}}


def run_dir(name):
    return os.path.join(ARCH, name) if name.endswith("_s1") and "_fp_" not in name else os.path.join(RUNS, name)


def curve_at(path, key="val/ema_mAP_50_95"):
    import csv
    rows = [r for r in csv.DictReader(open(os.path.join(path, "metrics.csv"))) if r.get(key)]
    return [int(r["epoch"]) + 1 for r in rows], [100 * float(r[key]) for r in rows]


def forest(ax, rows, title, xlabel="FP − baseline, mAP50-95 points"):
    y = np.arange(len(rows))[::-1]
    for yi, (lab, d, lo, hi, col, mk) in zip(y, rows):
        ax.plot([lo, hi], [yi, yi], color=col, lw=1.4)
        ax.plot(d, yi, marker=mk, color=col, ms=5, mec=SURF, mew=0.6)
        ax.text(hi + 0.25, yi, f"{d:+.2f}", va="center", fontsize=6.3, color=INK)
    ax.axvline(0, color=INK2, lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.6)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=8, loc="left")


def fig29_g29_effects():
    """G29 paired effects: leak-free primary (test and val), 25% secondary, official protocols by leak subset."""
    paired = json.load(open(os.path.join(PAPER, "paired.json")))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.1, 1, 1.1]})
    rows = [(f"{d} {s}", v, lo, hi, C[0] if s == "test" else C[2], "D" if d == "mean" else "o") for d, s, v, lo, hi, _ in G29_P]
    forest(axes[0], rows, "(a) Leak-free, full labels (P)")
    rows = [(lab, v, lo, hi, C[1], "D" if lab == "family mean" else "o") for lab, v, lo, hi in G29_S]
    forest(axes[1], rows, "(b) 25% of WTBD-group (S), test")
    rows = []
    for key, short in (("WTBD-official | G29: + frozen DINOv3 (M 768 cosine)", "WTBD-off."),
                       ("Wind Surface Defect | G29: + frozen DINOv3 (M 640, 100 ep cosine)", "Wind Surf.")):
        for sub, col in (("all", C[0]), ("leaked", C[3]), ("no twin", C[1])):
            r = paired[key][sub]
            rows.append((f"{short} {sub}", 100 * r["delta"][1], 100 * r["lo"][1], 100 * r["hi"][1], col, "o"))
    forest(axes[2], rows, "(c) Official protocols, vs G26 twin")
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig29_g29_effects")


def fig30_g29_curves():
    """Validation mAP50-95 per epoch, FP against its twin of identical recipe."""
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.0))
    panels = [("(a) WTBD-group", "rfdetrm768cos_wtbd_group_full", "rfdetrm768cos_fp_wtbd_group_full", True),
              ("(b) WTBlade-group", "rfdetrm640cos_wtblade_group_full", "rfdetrm640cos_fp_wtblade_group_full", True),
              ("(c) WTBD-official", "rfdetrm768cos_wtbd_official", "rfdetrm768cos_fp_wtbd_official", False),
              ("(d) Wind Surface", "rfdetrm640cos_windsurface_official", "rfdetrm640cos_fp_windsurface_official", False)]
    for ax, (title, a0, fp, seeds) in zip(axes, panels):
        for suf, ls in (("", "-"), ("_s1", ":")) if seeds else (("", "-"),):
            for name, col, lab in ((a0 + suf, C[0], "RF-DETR-M"), (fp + suf, C[1], "+ frozen DINOv3")):
                path = run_dir(name)
                if not os.path.exists(os.path.join(path, "metrics.csv")):
                    continue
                x, y = curve_at(path)
                ax.plot(x, y, color=col, ls=ls, lw=1.0, label=lab if not suf else None)
        ax.set_title(title, fontsize=7.5, loc="left")
        ax.set_xlabel("epoch", fontsize=7)
        ax.tick_params(labelsize=6)
    axes[0].set_ylabel("val mAP50-95, %", fontsize=7)
    axes[0].legend(fontsize=5.6, loc="lower right")
    fig.tight_layout(w_pad=0.4)
    save(fig, "fig30_g29_curves")


def fig31_g29_perclass():
    """Per-class test AP50-95, A0 and FP, mean of two seeds."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.3), gridspec_kw={"width_ratios": [6, 7]}, sharey=True)
    for ax, (ds, pc) in zip(axes, G29_PC.items()):
        names = list(pc)
        x = np.arange(len(names))
        a0 = [pc[n][0] for n in names]
        fp = [pc[n][1] for n in names]
        ax.bar(x - 0.19, a0, 0.36, color=C[0], label="RF-DETR-M (A0)", edgecolor=SURF)
        ax.bar(x + 0.19, fp, 0.36, color=C[1], label="+ frozen DINOv3 (FP)", edgecolor=SURF)
        for xi, a, f in zip(x, a0, fp):
            ax.text(xi, max(a, f) + 1.2, f"{f - a:+.1f}", ha="center", fontsize=6.2, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=6.4)
        ax.set_title(f"({'ab'[list(G29_PC).index(ds)]}) {ds}, test", fontsize=8, loc="left")
    axes[0].set_ylabel("AP50-95, % (mean of seeds)")
    axes[0].legend(fontsize=6, loc="upper left")
    fig.tight_layout()
    save(fig, "fig31_g29_perclass")


def fig32_official():
    """Official protocols: the BladeYOLO paper, our reproductions and our detectors, mAP50 and mAP50-95."""
    off = json.load(open(os.path.join(PAPER, "official.json")))

    def ours(ds, run):
        return [100 * v for v in off[ds]["runs"][run]["all"]["coco"]]

    groups = [
        ("Wind Surface\n(val 714)", [
            ("BladeYOLO (paper)", (86.5, 57.4), INK2),
            ("faithful reimpl. (Run 3)", (52.6, 28.0), "#9b9a96"),
            ("BladeYOLO-L (Run 6)", (78.2, 46.9), "#c9c8c4"),
            ("RF-DETR-M (G26)", tuple(ours("Wind Surface Defect", "rfdetrm640cos_windsurface_official")), C[0]),
            ("+ frozen DINOv3 (G29)", tuple(ours("Wind Surface Defect", "rfdetrm640cos_fp_windsurface_official")), C[1])]),
        ("WTBlade-Defect\n(test 553)", [
            ("BladeYOLO (paper)", (85.8, 74.6), INK2),
            ("YOLOv12-L (as reported)", (83.3, 71.9), "#9b9a96"),
            ("RF-DETR-M (G26)", tuple(ours("WTBlade-Defect", "rfdetrm640cos_wtblade_defect_official")), C[0])]),
        ("WTBD-official\n(test 161)", [
            ("YOLO12-L (ours)", tuple(ours("WTBD-official", "yolo12l_wtbd_official")), "#c9c8c4"),
            ("BladeYOLO-v2 (ours)", tuple(ours("WTBD-official", "bladeyolo2_wtbd_official")), "#9b9a96"),
            ("RF-DETR-M (G26)", tuple(ours("WTBD-official", "rfdetrm768cos_wtbd_official")), C[0]),
            ("+ frozen DINOv3 (G29)", tuple(ours("WTBD-official", "rfdetrm768cos_fp_wtbd_official")), C[1])])]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), sharey=True)
    for m, ax in enumerate(axes):
        x0 = 0
        ticks, tl = [], []
        for g, items in groups:
            for j, (lab, v, col) in enumerate(items):
                reported = "paper" in lab or "reported" in lab
                ax.bar(x0 + j, v[m], 0.85, color=col, edgecolor=INK2 if reported else SURF, hatch="////" if reported else None,
                       lw=0.5)
                ax.text(x0 + j, v[m] + 1, f"{v[m]:.1f}", ha="center", fontsize=5.6, rotation=90, va="bottom")
                ticks.append(x0 + j)
                tl.append(lab)
            ax.text(x0 + (len(items) - 1) / 2, 101, g, ha="center", va="bottom", fontsize=6.0, color=INK, linespacing=1.0)
            if x0:
                ax.axvline(x0 - 1, color=GRID, lw=0.8)
            x0 += len(items) + 1
        ax.set_xticks(ticks)
        ax.set_xticklabels(tl, fontsize=5.6, rotation=60, ha="right", rotation_mode="anchor")
        ax.set_ylim(0, 114)
        ax.set_yticks(range(0, 101, 20))
        ax.set_title(f"({'ab'[m]}) {'mAP50' if m == 0 else 'mAP50-95'}, %", fontsize=8, loc="left")
    fig.tight_layout()
    save(fig, "fig32_official_vs_bladeyolo")


def fig33_g29_predictions():
    """Labels, A0 and FP (seed 0) on seeded random test images of both leak-free benchmarks, confidence >= 0.3."""
    rng = random.Random(29)
    sets = (("WTBD-group", "rfdetrm768cos_wtbd_group_full", "rfdetrm768cos_fp_wtbd_group_full"),
            ("WTBlade-group", "rfdetrm640cos_wtblade_group_full", "rfdetrm640cos_fp_wtblade_group_full"))
    per = []
    for ds, a0, fp in sets:
        root = os.path.join(B, "leakfree", ds)
        names = yaml.safe_load(open(os.path.join(root, "data.yaml")))["names"]
        names = list(names.values()) if isinstance(names, dict) else list(names)
        d = os.path.join(root, "test", "images")
        fs = [f for f in sorted(os.listdir(d)) if read_labels(label_path(os.path.join(d, f)))]
        rng.shuffle(fs)
        preds = {r: json.load(open(os.path.join(RUNS, r, "preds_test.json"))) for r in (a0, fp)}
        rows = []
        for f in fs[:3]:
            p = os.path.join(d, f)
            gt = tile(p, 240, names=names)
            caption(gt, f"{ds}: labels", 0, 0.4)
            row = [gt]
            for run, lab in ((a0, "A0"), (fp, "FP")):
                key = next(k for k in preds[run] if os.path.basename(k) == f)
                kept = [r for r in np.array(preds[run][key]).reshape(-1, 6).tolist() if r[4] >= 0.3]
                t = tile(p, 240, boxes=[(int(r[5]), r[0], r[1], r[2], r[3], f"{names[int(r[5])][:9]} {r[4]:.2f}") for r in kept])
                caption(t, f"{lab}: {len(kept)} box{'' if len(kept) == 1 else 'es'}", 0, 0.4)
                row.append(t)
            rows.append(row)
        per.append(rows)
    save_image(grid([per[0][i] + per[1][i] for i in range(3)]), "fig33_g29_predictions")


FIGS = {k[3:]: v for k, v in list(globals().items()) if k.startswith("fig") and callable(v)}

if __name__ == "__main__":
    for k, f in FIGS.items():
        if len(sys.argv) < 2 or any(a in k for a in sys.argv[1:]):
            f()
