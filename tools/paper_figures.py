"""Figures for the preprint (paper/README.md). Reads runs/analysis/paper/{official,paired}.json; the older results are
copied from research_log.md (section given per constant) because their analyses predate a JSON output.
Palette: categorical slots 1-4 of the dataviz reference palette, light mode (validated: tools of the dataviz skill,
--pairs all for slots 1-3); every bar carries a direct value label, so no identity is colour-only.
  .venv/bin/python tools/paper_figures.py   -> paper/figures/fig{1..6}_*.{pdf,png}
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "runs", "analysis", "paper")
OUT = os.path.join(ROOT, "paper", "figures")
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # blue, orange, aqua, yellow
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "legend.frameon": False, "pdf.fonttype": 42})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def bars(ax, x, vals, color, width, labels=True, fmt="{:.1f}"):
    b = ax.bar(x, vals, width, color=color, edgecolor=SURF, linewidth=1.0)
    if labels:
        for xi, v in zip(x, vals):
            ax.text(xi, v + 1, fmt.format(v), ha="center", va="bottom", fontsize=6.5, color=INK)
    return b


def fig1_leak_subsets():
    """Official-protocol score of the best RF-DETR run on leaked vs no-twin test images."""
    off = json.load(open(os.path.join(SRC, "official.json")))
    rows = [("WTBD-official", "rfdetrm768cos_wtbd_official"), ("Wind Surface", "rfdetrm640cos_windsurface_official"),
            ("WTBlade-Defect", "rfdetrm640cos_wtblade_defect_official")]
    keys = {"WTBD-official": "WTBD-official", "Wind Surface": "Wind Surface Defect", "WTBlade-Defect": "WTBlade-Defect"}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for m, ax in enumerate(axes):
        x = np.arange(len(rows))
        for j, (sub, col) in enumerate((("leaked", C[0]), ("no twin", C[1]))):
            v = [100 * off[keys[n]]["runs"][r][sub]["coco"][m] for n, r in rows]
            lo = [100 * off[keys[n]]["runs"][r][sub]["coco_ci"][m][0] for n, r in rows]
            hi = [100 * off[keys[n]]["runs"][r][sub]["coco_ci"][m][1] for n, r in rows]
            xx = x + (j - 0.5) * 0.36
            bars(ax, xx, v, col, 0.34, labels=False)
            ax.errorbar(xx, v, yerr=[np.subtract(v, lo), np.subtract(hi, v)], fmt="none", ecolor=INK2, lw=0.8,
                        capsize=2)
            for xi, vi, h in zip(xx, v, hi):
                ax.text(xi, h + 1.5, f"{vi:.0f}", ha="center", fontsize=6.5, color=INK)
        ax.set_xticks(x, [n.replace("-", "-\n", 1) if n != "Wind Surface" else "Wind\nSurface" for n, _ in rows])
        ax.set_title(["mAP50", "mAP50-95"][m], fontsize=8, color=INK)
        ax.set_ylim(0, 105)
    axes[0].set_ylabel("official test score (%)")
    h = [plt.Rectangle((0, 0), 1, 1, color=C[0]), plt.Rectangle((0, 0), 1, 1, color=C[1])]
    fig.legend(h, ["test image has a training twin", "no training twin"], loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.06))
    save(fig, "fig1_leak_subsets")


def fig2_tuning():
    """Paired Δ of the G26 tuned recipe on leaked vs no-twin images: tuning on a leaky benchmark buys memorisation."""
    p = json.load(open(os.path.join(SRC, "paired.json")))
    rows = [("WTBD-official\ncosine − constant", "WTBD-official | G26 recipe: cosine − constant LR (M 768)"),
            ("Wind Surface\n100 ep cos − 60 ep", "Wind Surface Defect | G26 recipe: 100 ep cosine − 60 ep constant"),
            ("WTBlade-Defect\n40 ep cos − 25 ep", "WTBlade-Defect | G26 recipe: 40 ep cosine − 25 ep constant")]
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    ax.axhline(0, color=INK2, lw=0.8)
    for i, (lab, k) in enumerate(rows):
        for j, (sub, col) in enumerate((("leaked", C[0]), ("no twin", C[1]))):
            d = p[k][sub]
            v, lo, hi = 100 * d["delta"][1], 100 * d["lo"][1], 100 * d["hi"][1]
            x = i + (j - 0.5) * 0.3
            ax.errorbar([x], [v], yerr=[[v - lo], [hi - v]], fmt="o", color=col, ms=5, lw=1.2, capsize=2,
                        mec=SURF, mew=1)
            ax.text(x + 0.06, v, f"{v:+.1f}", va="center", fontsize=6.5, color=INK)
    ax.set_xticks(range(len(rows)), [r[0] for r in rows])
    ax.set_ylabel("Δ mAP50-95, tuned − untuned (points)")
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=C[0]), plt.Line2D([], [], marker="o", ls="", color=C[1])],
              labels=["leaked test images", "no-twin test images"], loc="lower left", fontsize=7)
    save(fig, "fig2_tuning_memorisation")


def fig3_inflation():
    """Same model and recipe, only the WTBD split differs (research_log §11, BladeYOLO-v2, Ultralytics test mAP50)."""
    cls = ["crack", "craze", "hide_craze", "surface_injure", "thunderstrike", "corrosion", "all"]
    official = [99.1, 90.9, 80.0, 81.6, 99.5, 52.7, 84.0]
    group = [38.1, 7.6, 56.9, 74.7, 98.0, 53.8, 54.8]
    fig, ax = plt.subplots(figsize=(6.0, 2.4))
    x = np.arange(len(cls))
    bars(ax, x - 0.18, official, C[0], 0.34, fmt="{:.0f}")
    bars(ax, x + 0.18, group, C[1], 0.34, fmt="{:.0f}")
    ax.set_xticks(x, cls)
    ax.set_ylim(0, 110)
    ax.set_ylabel("test mAP50 (%)")
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=C[0]), plt.Rectangle((0, 0), 1, 1, color=C[1])],
              ["published random split", "leak-free group split"], loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, 1.18))
    save(fig, "fig3_wtbd_inflation")


def fig4_learning_curve():
    """WTBD-group test mAP50-95 (bootstrap evaluator) by labelled-scene fraction (research_log §24 B)."""
    f = [25, 50, 100]
    series = [("BladeYOLO-v2 (frozen DINOv3 ViT-S)", [7.6, 17.5, 24.2], C[0]),
              ("same, without DINOv3", [3.7, 8.8, 15.2], C[1]),
              ("YOLO12-L", [4.1, 9.9, 18.5], C[2])]
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for lab, v, col in series:
        ax.plot(f, v, color=col, lw=2, marker="o", ms=5, mec=SURF, mew=1)
        ax.text(103, v[-1], lab.split(" (")[0].replace("same, ", ""), va="center", fontsize=6.5, color=INK)
    ax.set_xticks(f, [f"{x}%" for x in f])
    ax.set_xlim(20, 105)
    ax.set_xlabel("labelled training scenes")
    ax.set_ylabel("leak-free test mAP50-95 (%)")
    ax.legend([plt.Line2D([], [], color=c, lw=2) for _, _, c in series], [s[0] for s in series], loc="upper left",
              fontsize=6.5)
    save(fig, "fig4_dinov3_learning_curve")


def fig5_subset_luck():
    """Eight random 25% scene draws, same recipe and seed, with and without DINOv3 (research_log H12)."""
    vitb = [11.2, 13.5, 11.8, 12.3, 9.0, 10.0, 12.4, 19.6]
    nod = [5.6, 6.7, 4.6, 6.3, 2.6, 5.6, 4.9, 7.2]
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for a, b in zip(nod, vitb):
        ax.plot([0, 1], [a, b], color=GRID, lw=1, zorder=1)
    ax.scatter([0] * 8, nod, color=C[1], s=28, edgecolor=SURF, lw=1, zorder=2)
    ax.scatter([1] * 8, vitb, color=C[0], s=28, edgecolor=SURF, lw=1, zorder=2)
    ax.set_xticks([0, 1], ["without DINOv3", "frozen DINOv3 ViT-B"])
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylabel("leak-free test mAP50-95 (%)")
    ax.text(1.08, 19.6, "best draw", fontsize=6.5, va="center", color=INK2)
    ax.text(1.08, 9.0, "worst draw", fontsize=6.5, va="center", color=INK2)
    save(fig, "fig5_subset_luck")


def fig6_error_anatomy():
    """Where each test defect ends up (research_log, error anatomy, 1 Nov)."""
    rows = ["WTBD · RF-DETR-M 25%", "WTBD · frozen ViT-B 25%", "WTBD · BladeYOLO-v2 100%",
            "WTBlade · RF-DETR-M 25%", "WTBlade · RF-DETR-M 8%", "WTBlade · frozen ViT-B 100%"]
    parts = {"found (conf ≥ 0.1)": [55, 41, 63, 55, 43, 58], "right box, ranked low": [37, 26, 17, 28, 33, 27],
             "other class only": [4, 15, 8, 3, 7, 6], "no box": [4, 18, 12, 14, 16, 9]}
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    left = np.zeros(len(rows))
    for (lab, v), col in zip(parts.items(), C):
        ax.barh(rows, v, left=left, color=col, edgecolor=SURF, linewidth=2, height=0.62, label=lab)
        for i, (l0, vi) in enumerate(zip(left, v)):
            if vi >= 7:
                ax.text(l0 + vi / 2, i, f"{vi}", ha="center", va="center", fontsize=6.5, color=INK)
        left += v
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("share of test defects (%)")
    ax.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.4, 1.16), fontsize=6.5)
    save(fig, "fig6_error_anatomy")


if __name__ == "__main__":
    fig1_leak_subsets()
    fig2_tuning()
    fig3_inflation()
    fig4_learning_curve()
    fig5_subset_luck()
    fig6_error_anatomy()
