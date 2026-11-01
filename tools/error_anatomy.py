"""Error anatomy of low-label detectors (research_log.md, after the box-extent diagnostic): for every test box, is
the right answer (a same-class box at IoU >= 0.5) among the detector's outputs, and if so, how is it ranked?

Per test box, outcome from the model's predictions (conf >= 0.001; optionally only each image's top-N by score):
  hit        a same-class box at IoU >= 0.5 with conf >= CONF
  ranked low a same-class box at IoU >= 0.5 exists, but every one has conf < CONF
  other cls  only other-class boxes at IoU >= 0.5
  no box     nothing at IoU >= 0.5 (a proposal / extent failure)
'candidate recall' = hit + ranked low. YOLO-family runs keep <= 300 boxes after NMS (about 50-80 per image), RF-DETR
keeps ~300 per image, so recall is also reported within each image's top-50 boxes by score.
For 'ranked low' boxes: median score and the rank of the true class among the classes predicted at that place.

  .venv/bin/python tools/error_anatomy.py  -> runs/analysis/extent/anatomy.md
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import bootstrap_ci as bc  # noqa: E402
import detection_metrics as dm  # noqa: E402

B = os.path.join(ROOT, "runs", "analysis", "bootstrap")
WTBD = (os.path.join(ROOT, "external", "WTBD-group"), ["craze", "corrosion", "surface_injure", "thunderstrike", "crack", "hide_craze"])
WTBLADE = (os.path.join(ROOT, "external", "WTBlade-group"), ["burn", "crack", "deformity", "dirt", "oil", "peeling", "rust"])
F25 = lambda pre, ds, suf="": [f"{pre}_{ds}_group_f25{suf}"] + [f"{pre}_rand{k}_{ds}_group_f25{suf}" for k in (1, 2, 3)]
GROUPS = [
    ("WTBD", WTBD, "RF-DETR-M 25% (4 subsets)", os.path.join(B, "seeds_f25", "preds"), F25("rfdetrm", "wtbd")),
    ("WTBD", WTBD, "ViT-B 25% (4 subsets)", os.path.join(B, "seeds_f25", "preds"), F25("bladeyolo2b", "wtbd", "_s0")),
    ("WTBD", WTBD, "BladeYOLO-v2 100%", os.path.join(B, "seeds_f25", "preds"), ["bladeyolo2_wtbd_group", "bladeyolo2_wtbd_group_s1"]),
    ("WTBlade", WTBLADE, "RF-DETR-M 25% (4 subsets)", os.path.join(B, "wtblade_f25", "test"), F25("rfdetrm", "wtblade")),
    ("WTBlade", WTBLADE, "RF-DETR-M 8% (4 subsets)", os.path.join(B, "wtblade_f08", "test"),
     [m.replace("f25", "f08") for m in F25("rfdetrm", "wtblade")]),
    ("WTBlade", WTBLADE, "ViT-B 100%", os.path.join(ROOT, "runs", "analysis", "transfer", "preds_WTBlade-group"), ["bladeyolo2b_wtblade_group"]),
]
CONF, TOPN = 0.1, 50


def load_preds(path, data):
    p = json.load(open(path))
    return {os.path.join(data, k) if not os.path.isabs(k) else k: np.array(v).reshape(-1, 6) for k, v in p.items()}


def anatomy(imgs, data, cache, runs, nc):
    rows = []  # (class, outcome 0..3, in top-N, low score, true-class rank)
    for r in runs:
        p = load_preds(os.path.join(cache, r + ".json"), data)
        for f, im in imgs.items():
            d = p.get(f, np.zeros((0, 6)))
            top = np.zeros(len(d), bool)
            top[np.argsort(-d[:, 4])[:TOPN]] = True
            for g in im["gt"]:
                c = int(g[0])
                iou = dm.iou_matrix(g[None, 1:], d[:, :4])[0] if len(d) else np.zeros(0)
                loc = iou >= 0.5
                same = loc & (d[:, 5] == c)
                intop = bool((same & top).any())
                if (same & (d[:, 4] >= CONF)).any():
                    rows.append((c, 0, intop, np.nan, 1))
                elif same.any():
                    s = d[same][:, 4].max()
                    rank = 1 + np.unique(d[loc & (d[:, 4] > s)][:, 5]).size
                    rows.append((c, 1, intop, s, rank))
                else:
                    rows.append((c, 2 if loc.any() else 3, False, np.nan, np.nan))
    return np.array(rows, float)


def line(name, a):
    n = len(a)
    if not n:
        return f"| {name} | 0 | | | | | | | | |"
    sh = [np.mean(a[:, 1] == k) for k in range(4)]
    low = a[a[:, 1] == 1]
    return (f"| {name} | {n} | {sh[0]:.2f} | {sh[1]:.2f} | {sh[2]:.2f} | {sh[3]:.2f} | **{sh[0] + sh[1]:.2f}** | "
            f"{np.mean(a[:, 2]):.2f} | {np.median(low[:, 3]) if len(low) else float('nan'):.3f} | "
            f"{np.median(low[:, 4]) if len(low) else float('nan'):.0f} |")


def main():
    head = ["| detector | test boxes | hit | ranked low | other cls | no box | candidate recall | recall in top-50 | "
            "median score, ranked low | median true-class rank |", "|---|---|---|---|---|---|---|---|---|---|"]
    lines = ["# Error anatomy: is the right box among the outputs?", "",
             f"hit = same class, IoU ≥ 0.5, conf ≥ {CONF}; ranked low = such a box exists only below {CONF}; "
             f"candidate recall = hit + ranked low (predictions at conf ≥ 0.001); top-50 = within each image's 50 "
             "highest-scoring boxes.", ""]
    per = []
    for ds, (data, names), name, cache, runs in GROUPS:
        imgs = bc.load_yolo_split(data, "test")
        a = anatomy(imgs, data, cache, runs, len(names))
        per.append((ds, names, name, a))
    for ds in ("WTBD", "WTBlade"):
        lines += [f"## {ds}, all classes", ""] + head + [line(n, a) for d, _, n, a in per if d == ds] + [""]
    for ds in ("WTBD", "WTBlade"):
        names = next(nm for d, nm, _, _ in per if d == ds)
        for c, cn in enumerate(names):
            lines += [f"### {ds}: {cn}", ""] + head + [line(n, a[a[:, 0] == c]) for d, _, n, a in per if d == ds] + [""]
    os.makedirs(os.path.join(ROOT, "runs", "analysis", "extent"), exist_ok=True)
    open(os.path.join(ROOT, "runs", "analysis", "extent", "anatomy.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[:22]))


if __name__ == "__main__":
    main()
