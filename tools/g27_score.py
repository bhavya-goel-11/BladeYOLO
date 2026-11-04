"""G27 scorer (research_log.md, G27; written before any G27 run). Phase A on the two leak-free protocols.

Arms (RF-DETR-M; run names under runs/detect/BladeYOLO_WindSurface/, predictions preds_{val,test}.json from
rfdetr_v1.py predict):
  A0  G26 recipe                 A1  A0 + copy-paste      A2  A1 + D1 (SSL encoder)      A3  A1 + D2 (stage 1)

Decision (A-1, seed 0, VALIDATION only): every arm's paired scene-bootstrap difference to A0 on val mAP50-95.
  R = the arm with the highest mean val mAP50-95 over the two datasets among the arms that beat A0 on val on BOTH
  datasets (point estimates); R = A0 if none does. Each component's own val contrast is reported (CP: A1 - A0,
  D1: A2 - A1, D2: A3 - A1).
Primary test claim (A-2, seeds 0 + 1, TEST): R - A0 mAP50-95 averaged over the two seeds;
  passes if > 0 on each dataset AND the mean over the two datasets has P(> 0) >= 0.95 (scenes resampled
  independently per dataset, 2,000 replicates, paired within a dataset).
Other architectures (existing cached test predictions, plus YOLO12-L on WTBlade-group when present): point estimates
  and the paired R - model difference on test.

  .venv/bin/python tools/g27_score.py        -> runs/analysis/g27/score.md (+ .json); arms not yet present are listed
"""

import csv
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import bootstrap_ci as bc  # noqa: E402

RUNS = os.path.join(ROOT, "runs", "detect", "BladeYOLO_WindSurface")
OUT = os.path.join(ROOT, "runs", "analysis", "g27")
N_BOOT = 2000
ARMS = ["A0", "A1", "A2", "A3"]
DATASETS = {
    "WTBD-group": dict(
        data=os.path.join(ROOT, "external", "WTBD-group"), nc=6,
        groups={"test": os.path.join(ROOT, "runs", "analysis", "bootstrap", "wtbd_group", "groups.csv"),
                "val": os.path.join(ROOT, "runs", "analysis", "bootstrap", "h5_last_val", "groups.csv")},
        runs={"A0": "rfdetrm768cos_wtbd_group_full", "A1": "rfdetrm768cos_cp_wtbd_group_full",
              "A2": "rfdetrm768cos_cp_d1_wtbd_group_full", "A3": "rfdetrm768cos_cp_d2_wtbd_group_full"},
        others={"BladeYOLO-v2 (ViT-S)": "bladeyolo2_wtbd_group", "YOLO12-L": "yolo12l_wtbd_group",
                "BladeYOLO-v2 without DINOv3": "bladeyolo2nodino_wtbd_group"}),
    "WTBlade-group": dict(
        data=os.path.join(ROOT, "external", "WTBlade-group"), nc=7,
        groups={s: os.path.join(ROOT, "external", "WTBlade-group", "groups.csv") for s in ("test", "val")},
        runs={"A0": "rfdetrm640cos_wtblade_group_full", "A1": "rfdetrm640cos_cp_wtblade_group_full",
              "A2": "rfdetrm640cos_cp_d1_wtblade_group_full", "A3": "rfdetrm640cos_cp_d2_wtblade_group_full"},
        others={"BladeYOLO-v2 (ViT-B)": "bladeyolo2b_wtblade_group_last", "YOLO12-L": "yolo12l_wtblade_group"}),
}


def scenes(cfg, split, files):
    g = {}
    for r in csv.DictReader(open(cfg["groups"][split])):
        if r.get("subset", split) != split:
            continue
        g.setdefault(r["group"], []).append(os.path.abspath(os.path.join(cfg["data"], split, "images", r["image"])))
    groups = list(g.values())
    assert sorted(f for v in groups for f in v) == sorted(files), f"{cfg['data']} {split}: groups do not cover the split"
    return groups


def rf_preds(run, cfg, split, files):
    """rfdetr_v1.predict output (file-name keys) -> {abs path: (n, 6)}; None if absent."""
    p = os.path.join(RUNS, run, f"preds_{split}.json")
    if not os.path.exists(p):
        return None
    raw = json.load(open(p))
    out = {os.path.abspath(os.path.join(cfg["data"], split, "images", k)): np.array(v).reshape(-1, 6) for k, v in raw.items()}
    assert set(out) == set(files), f"{run} {split}: prediction files differ from the split"
    return out


def cached_preds(name, files, imgs=None):
    """A bootstrap cache file (abs-path keys) covering exactly the split; else, if the Ultralytics run's weights are
    local (e.g. the new yolo12l_wtblade_group), predict once into runs/analysis/bootstrap/g27/; else None."""
    for p in sorted(glob.glob(os.path.join(ROOT, "runs", "analysis", "bootstrap", "**", f"{name}.json"), recursive=True)):
        d = json.load(open(p))
        if set(d) == set(files):
            return {k: np.array(v).reshape(-1, 6) for k, v in d.items()}
    if imgs is not None and os.path.exists(os.path.join(RUNS, name, "weights", "best.pt")):
        return bc.predict(os.path.join(RUNS, name), imgs, os.path.join(ROOT, "runs", "analysis", "bootstrap", "g27"))
    return None


def draws(groups, seed):
    rng = np.random.default_rng(seed)
    return [[f for g in rng.integers(0, len(groups), len(groups)) for f in groups[g]] for _ in range(N_BOOT)]


def boot(m, files, dr, nc):
    """Point mAP50-95 and its bootstrap replicates."""
    return bc.score(m, files, nc)[1], np.array([bc.score(m, d, nc)[1] for d in dr])


def ci(x):
    return f"[{100 * np.percentile(x, 2.5):+.1f}, {100 * np.percentile(x, 97.5):+.1f}]"


def main():
    os.makedirs(OUT, exist_ok=True)
    res, lines = {}, ["# G27 phase A: leak-free protocols (tools/g27_score.py)", "",
                      f"Scene bootstrap, {N_BOOT} replicates, mAP50-95 in percent (bootstrap_ci evaluator).", ""]
    for name, cfg in DATASETS.items():
        r = res[name] = {}
        for split in ("val", "test"):
            imgs = bc.load_yolo_split(cfg["data"], split)
            files = sorted(imgs)
            dr = draws(scenes(cfg, split, files), 0)
            for arm in ARMS:
                for seed, suf in ((0, ""), (1, "_s1")):
                    p = rf_preds(cfg["runs"][arm] + suf, cfg, split, files)
                    if p is not None:
                        m = bc.per_image_matches(imgs, p, cfg["nc"])
                        r[(arm, seed, split)] = boot(m, files, dr, cfg["nc"])
            if split == "test":
                for lab, run in cfg["others"].items():
                    p = cached_preds(run, files, imgs)
                    if p is not None:
                        r[(lab, None, split)] = boot(bc.per_image_matches(imgs, p, cfg["nc"]), files, dr, cfg["nc"])
    # A-1 decision on val, seed 0
    lines += ["## A-1: validation, seed 0 (decides the recipe R)", "",
              "| arm | " + " | ".join(f"{d} val" for d in DATASETS) + " | " + " | ".join(f"{d} Δ vs A0 [95% CI]" for d in DATASETS) + " |",
              "|---|" + "---|" * (2 * len(DATASETS))]
    have = {a: all((a, 0, "val") in res[d] for d in DATASETS) for a in ARMS}
    beats = {}
    for a in ARMS:
        if not have[a]:
            lines.append(f"| {a} | pending |" + " |" * (2 * len(DATASETS) - 1))
            continue
        vals = [100 * res[d][(a, 0, "val")][0] for d in DATASETS]
        deltas = []
        for d in DATASETS:
            if have["A0"]:
                pa, ba = res[d][(a, 0, "val")]
                p0, b0 = res[d][("A0", 0, "val")]
                deltas.append((pa - p0, ba - b0))
        beats[a] = a != "A0" and len(deltas) == len(DATASETS) and all(x[0] > 0 for x in deltas)
        lines.append(f"| {a} | " + " | ".join(f"{v:.1f}" for v in vals) + " | " +
                     " | ".join(f"{100 * x[0]:+.1f} {ci(x[1])}" for x in deltas) + " |")
    lines.append("")
    for comp, a, b in (("CP", "A1", "A0"), ("D1", "A2", "A1"), ("D2", "A3", "A1")):
        if have[a] and have[b]:
            ds = [100 * (res[d][(a, 0, "val")][0] - res[d][(b, 0, "val")][0]) for d in DATASETS]
            lines.append(f"- component {comp} ({a} − {b}), val: " + ", ".join(f"{d} {x:+.1f}" for d, x in zip(DATASETS, ds)))
    decided = all(have.values())
    R = None
    if decided:
        ok = [a for a in ARMS if beats.get(a)]
        R = max(ok, key=lambda a: np.mean([res[d][(a, 0, "val")][0] for d in DATASETS])) if ok else "A0"
        lines += ["", f"**R = {R}** (arms beating A0 on val on both datasets: {ok or 'none'})."]
    else:
        lines += ["", f"R undecided: arms present {[a for a in ARMS if have[a]]}."]
    # A-2 primary test claim
    lines += ["", "## A-2: test, R − A0, mean of seeds 0 and 1 (primary claim)", ""]
    if R and R != "A0" and all((x, s, "test") in res[d] for d in DATASETS for x in (R, "A0") for s in (0, 1)):
        per = {}
        for d in DATASETS:
            pr = np.mean([res[d][(R, s, "test")][0] - res[d][("A0", s, "test")][0] for s in (0, 1)])
            br = np.mean([res[d][(R, s, "test")][1] - res[d][("A0", s, "test")][1] for s in (0, 1)], 0)
            per[d] = (pr, br)
            lines.append(f"- {d}: {100 * pr:+.2f} {ci(br)}, P(>0) {np.mean(br > 0):.3f}")
        pm = np.mean([v[0] for v in per.values()])
        bm = np.mean([v[1] for v in per.values()], 0)
        ok = all(v[0] > 0 for v in per.values()) and np.mean(bm > 0) >= 0.95
        lines += [f"- mean over datasets: {100 * pm:+.2f} {ci(bm)}, P(>0) {np.mean(bm > 0):.3f}",
                  "", f"**Primary claim {'passes' if ok else 'fails'}.**"]
    else:
        lines.append("Pending (needs R ≠ A0 and seeds 0 + 1 of R and A0 on both datasets)." if R != "A0" else
                     "R = A0: no recipe change to confirm.")
    # test table: every arm and other architectures
    lines += ["", "## Test, all arms and other architectures (point estimates; secondary)", "",
              "| model | " + " | ".join(DATASETS) + " |", "|---|" + "---|" * len(DATASETS)]
    keys = [(a, s) for a in ARMS for s in (0, 1)] + [(lab, None) for d in DATASETS for lab in DATASETS[d]["others"]]
    seen = []
    for k in keys:
        if k in seen:
            continue
        seen.append(k)
        cells = [f"{100 * res[d][(*k, 'test')][0]:.1f}" if (*k, "test") in res[d] else "–" for d in DATASETS]
        if any(c != "–" for c in cells):
            lab = f"{k[0]} seed {k[1]}" if k[1] is not None else k[0]
            lines.append(f"| {lab} | " + " | ".join(cells) + " |")
    json.dump({d: {"|".join(map(str, k)): float(v[0]) for k, v in r.items()} for d, r in res.items()} | {"R": R},
              open(os.path.join(OUT, "score.json"), "w"), indent=1)
    open(os.path.join(OUT, "score.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
