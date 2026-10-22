"""Transfer between the two leak-free blade benchmarks (Phase 28, round 2, item 5): does a detector trained on one
find the defects of the other?

WTBD (6 damage classes) and WTBlade (7 classes, two of them contaminants: dirt, oil) do not share a taxonomy, so
transfer is scored class-agnostically over damage only: every damage class -> "defect", contaminant labels and
predictions dropped. Two classes also map one to one and are scored class-aware: crack <-> crack, thunderstrike <->
burn (lightning damage). Each test set: the in-domain model against the transferred one, paired scene bootstrap.

  .venv/bin/python tools/transfer_eval.py  -> runs/analysis/transfer/report.md
"""

import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import bootstrap_ci as bc  # noqa: E402

RUNS = os.path.join(ROOT, "runs", "detect", "BladeYOLO_WindSurface")
OUT = os.path.join(ROOT, "runs", "analysis", "transfer")
SETS = {  # data, groups.csv, damage class ids, models: name -> (run, the model's damage class ids, its class -> mapped name)
    "WTBD-group test": dict(
        data=os.path.join(ROOT, "external", "WTBD-group"),
        groups=os.path.join(ROOT, "runs", "analysis", "bootstrap", "wtbd_group", "groups.csv"),
        damage=[0, 1, 2, 3, 4, 5], mapped={4: "crack", 3: "lightning"}),
    "WTBlade-group test": dict(
        data=os.path.join(ROOT, "external", "WTBlade-group"),
        groups=os.path.join(ROOT, "external", "WTBlade-group", "groups.csv"),
        damage=[0, 1, 2, 5, 6], mapped={1: "crack", 0: "lightning"}),
}
MODELS = {  # run -> (trained on, its damage class ids, mapped classes)
    "bladeyolo2_wtbd_group": ("WTBD-group test", [0, 1, 2, 3, 4, 5], {4: "crack", 3: "lightning"}),
    "bladeyolo2_wtbd_group_s1": ("WTBD-group test", [0, 1, 2, 3, 4, 5], {4: "crack", 3: "lightning"}),
    "bladeyolo2b_wtblade_group": ("WTBlade-group test", [0, 1, 2, 5, 6], {1: "crack", 0: "lightning"}),
}


def relabel(gt_or_pred, col, keep, to):
    """Keep rows whose class (column col) is in keep, mapped through `to` (dict) or to 0 (to=None)."""
    m = np.isin(gt_or_pred[:, col], keep)
    out = gt_or_pred[m].copy()
    out[:, col] = [to[int(c)] if to else 0 for c in out[:, col]] if len(out) else out[:, col]
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    lines = ["# Transfer between WTBD-group and WTBlade-group (100% training sets)", "",
             "Damage only (WTBlade dirt and oil dropped). In-domain models: WTBD ViT-S/16 (2 seeds), WTBlade ViT-B/16.", ""]
    for name, S in SETS.items():
        imgs = bc.load_yolo_split(S["data"], "test")
        files = sorted(imgs)
        preds = {r: bc.predict(os.path.join(RUNS, r), imgs, os.path.join(OUT, "preds_" + name.split()[0])) for r in MODELS}
        g = {}
        for row in csv.DictReader(open(S["groups"])):
            if row.get("subset", "test") == "test":
                g.setdefault(row["group"], []).append(os.path.join(os.path.abspath(S["data"]), "test", "images", row["image"]))
        groups = [v for v in g.values() if v]
        rng = np.random.default_rng(0)
        draws = [[f for gi in rng.integers(0, len(groups), len(groups)) for f in groups[gi]] for _ in range(1000)]
        lines += [f"## {name} ({len(files)} images, {len(groups)} scene groups)", "",
                  "| model (trained on) | agnostic AP50 | agnostic AP50-95 [95% CI] | crack AP50 | lightning AP50 |", "|---|---|---|---|---|"]
        boot = {}
        for r, (home, mdamage, mmap) in MODELS.items():
            res = []
            for mode in ("agnostic", "crack", "lightning"):
                if mode == "agnostic":
                    gi = {k: dict(v, gt=relabel(v["gt"], 0, S["damage"], None)) for k, v in imgs.items()}
                    pr = {k: relabel(v, 5, mdamage, None) for k, v in preds[r].items()}
                else:
                    tc = [c for c, m in S["mapped"].items() if m == mode]
                    mc = [c for c, m in mmap.items() if m == mode]
                    gi = {k: dict(v, gt=relabel(v["gt"], 0, tc, {tc[0]: 0})) for k, v in imgs.items()}
                    pr = {k: relabel(v, 5, mc, {mc[0]: 0}) for k, v in preds[r].items()}
                ms = bc.per_image_matches(gi, pr, 1)
                res.append(bc.score(ms, files, 1)[:2])
                if mode == "agnostic":
                    boot[r] = np.array([bc.score(ms, d, 1)[:2] for d in draws])
            b = boot[r][:, 1]
            lines.append(f"| {r} ({home.split()[0]}) | {res[0][0]:.3f} | {res[0][1]:.3f} [{np.percentile(b, 2.5):.3f}, {np.percentile(b, 97.5):.3f}] | "
                         f"{res[1][0]:.3f} | {res[2][0]:.3f} |")
            print(lines[-1], flush=True)
        home_runs = [r for r, m in MODELS.items() if m[0] == name]
        away_runs = [r for r, m in MODELS.items() if m[0] != name]
        d = np.mean([boot[r] for r in away_runs], 0) - np.mean([boot[r] for r in home_runs], 0)
        lines.append(f"\ntransferred − in-domain (agnostic): AP50 {np.mean(d[:, 0]):+.3f} [{np.percentile(d[:, 0], 2.5):+.3f}, {np.percentile(d[:, 0], 97.5):+.3f}], "
                     f"AP50-95 {np.mean(d[:, 1]):+.3f} [{np.percentile(d[:, 1], 2.5):+.3f}, {np.percentile(d[:, 1], 97.5):+.3f}]\n")
    open(os.path.join(OUT, "report.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
