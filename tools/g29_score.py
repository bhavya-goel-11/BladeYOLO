"""G29 scorer (research_log.md, G29 pre-registration): RF-DETR-M + frozen DINOv3 ViT-B prior (FP) against its twins.

P, primary (leak-free, decides): FP − A0 test mAP50-95, scene bootstrap (2,000 replicates, paired within a dataset,
   scenes resampled independently per dataset), mean of seeds 0 and 1. Passes if the point estimate is > 0 on each
   dataset and the mean over the two datasets has P(> 0) >= 0.95 (the G27 A-2 bar).
   Twins (Kaggle T4, fp16): seed 0 = runs/detect/.../rfdetrm768cos_wtbd_group_full, rfdetrm640cos_wtblade_group_full;
   seed 1 = runs/analysis/archive/kaggle_a0s1/<same>_s1 (the T4 runs, not the A100 reruns).
   Reported: val in the same form; per class; WTBD-group without its 6 pixel-twin test images; FP against every other
   architecture on both tests; error anatomy (hit / ranked low / other class / no box).
S, secondary: the four WTBD-group 25% subsets, FP (rfdetrmfp*) − V1 seed 0 (rfdetrm*), family mean; reported as passing
   if mean > 0 with P(> 0) >= 0.95 and positive on >= 3 of 4.

  .venv/bin/python tools/g29_score.py      -> runs/analysis/g29/score.md (+ score.json)
"""

import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import bootstrap_ci as bc  # noqa: E402
import detection_metrics as dm  # noqa: E402
import g27_score as g  # noqa: E402

RUNS = g.RUNS
ARCHIVE = os.path.join(ROOT, "runs", "analysis", "archive", "kaggle_a0s1")
OUT = os.path.join(ROOT, "runs", "analysis", "g29")
FP = {"WTBD-group": "rfdetrm768cos_fp_wtbd_group_full", "WTBlade-group": "rfdetrm640cos_fp_wtblade_group_full"}
CLASSES = {"WTBD-group": ["craze", "corrosion", "surface_injure", "thunderstrike", "crack", "hide_craze"],
           "WTBlade-group": ["burn", "crack", "deformity", "dirt", "oil", "peeling", "rust"]}
F25 = ["", "_rand1", "_rand2", "_rand3"]


def preds(folder, cfg, split, files):
    p = os.path.join(folder, f"preds_{split}.json")
    raw = json.load(open(p))
    out = {os.path.abspath(os.path.join(cfg["data"], split, "images", k)): np.array(v).reshape(-1, 6) for k, v in raw.items()}
    assert set(out) == set(files), f"{folder} {split}: prediction files differ from the split"
    return out


def arm_folders(name):
    a0 = g.DATASETS[name]["runs"]["A0"]
    return {("FP", 0): os.path.join(RUNS, FP[name]), ("FP", 1): os.path.join(RUNS, FP[name] + "_s1"),
            ("A0", 0): os.path.join(RUNS, a0), ("A0", 1): os.path.join(ARCHIVE, a0 + "_s1")}


def anatomy(imgs, p, conf=0.1):
    """Share of test boxes: hit (same-class IoU >= 0.5, conf >= 0.1), ranked low (only below 0.1), other class, no box."""
    out = np.zeros(4)
    for f, im in imgs.items():
        d = p[f]
        for gt in im["gt"]:
            iou = dm.iou_matrix(gt[None, 1:], d[:, :4])[0] if len(d) else np.zeros(0)
            loc, same = iou >= 0.5, (iou >= 0.5) & (d[:, 5] == gt[0]) if len(d) else np.zeros(0, bool)
            out[0 if (same & (d[:, 4] >= conf)).any() else 1 if same.any() else 2 if loc.any() else 3] += 1
    return out / out.sum()


def pct(x):
    return f"{100 * x:.1f}"


def ci(x):
    return f"[{100 * np.percentile(x, 2.5):+.1f}, {100 * np.percentile(x, 97.5):+.1f}]"


def primary(lines, js):
    tw = list(csv.DictReader(open(os.path.join(ROOT, "runs", "analysis", "split_audit", "twins.csv"))))
    res = {}
    for name, cfg in g.DATASETS.items():
        folders = arm_folders(name)
        for split in ("test", "val"):
            imgs = bc.load_yolo_split(cfg["data"], split)
            files = sorted(imgs)
            dr = g.draws(g.scenes(cfg, split, files), 0)
            for k, folder in folders.items():
                p = preds(folder, cfg, split, files)
                m = bc.per_image_matches(imgs, p, cfg["nc"])
                pt = bc.score(m, files, cfg["nc"])
                res[name, k, split] = dict(point=pt[1], map50=pt[0], cls=pt[3], boot=np.array([bc.score(m, d, cfg["nc"])[1] for d in dr]))
                if split == "test":
                    drop = {os.path.abspath(os.path.join(cfg["data"], "test", "images", r["image"])) for r in tw
                            if r["split"] == name and r["subset"] == "test"}
                    res[name, k, split]["clean"] = bc.score(m, [f for f in files if f not in drop], cfg["nc"])[1]
                    res[name, k, split]["anatomy"] = anatomy(imgs, p)
            if split == "test":
                for lab, run in cfg["others"].items():
                    p = g.cached_preds(run, files)
                    if p is not None:
                        res[name, lab, split] = dict(point=bc.score(bc.per_image_matches(imgs, p, cfg["nc"]), files, cfg["nc"])[1])
            print(f"scored {name} {split}", flush=True)

    lines += ["## P: FP − A0, leak-free, mean of seeds 0 and 1 (primary)", "",
              "| dataset | split | A0 s0 / s1 | FP s0 / s1 | FP − A0 [95% CI] | P(> 0) |", "|---|---|---|---|---|---|"]
    per = {}
    for split in ("test", "val"):
        for name in g.DATASETS:
            d = [res[name, ("FP", s), split]["point"] - res[name, ("A0", s), split]["point"] for s in (0, 1)]
            b = np.mean([res[name, ("FP", s), split]["boot"] - res[name, ("A0", s), split]["boot"] for s in (0, 1)], 0)
            per[name, split] = (float(np.mean(d)), b)
            lines.append(f"| {name} | {split} | {pct(res[name, ('A0', 0), split]['point'])} / {pct(res[name, ('A0', 1), split]['point'])} | "
                         f"{pct(res[name, ('FP', 0), split]['point'])} / {pct(res[name, ('FP', 1), split]['point'])} | "
                         f"{100 * np.mean(d):+.2f} {ci(b)} | {np.mean(b > 0):.3f} |")
            js[f"{name}|{split}|diff"] = float(np.mean(d))
            js[f"{name}|{split}|P"] = float(np.mean(b > 0))
    for split in ("test", "val"):
        pm = np.mean([per[n, split][0] for n in g.DATASETS])
        bm = np.mean([per[n, split][1] for n in g.DATASETS], 0)
        lines.append(f"| mean | {split} | | | {100 * pm:+.2f} {ci(bm)} | {np.mean(bm > 0):.3f} |")
        js[f"mean|{split}|diff"], js[f"mean|{split}|P"] = float(pm), float(np.mean(bm > 0))
    ok = all(per[n, "test"][0] > 0 for n in g.DATASETS) and js["mean|test|P"] >= 0.95
    js["primary_passes"] = bool(ok)
    lines += ["", f"**Primary claim {'passes' if ok else 'fails'}** (> 0 on each dataset: "
              f"{[bool(per[n, 'test'][0] > 0) for n in g.DATASETS]}; mean P(> 0) {js['mean|test|P']:.3f}, bar 0.95).", ""]

    lines += ["### Per class, test AP50-95, mean of seeds (A0 → FP)", ""]
    for name in g.DATASETS:
        a = np.mean([res[name, ("A0", s), "test"]["cls"] for s in (0, 1)], 0)
        f = np.mean([res[name, ("FP", s), "test"]["cls"] for s in (0, 1)], 0)
        lines.append(f"- {name}: " + ", ".join(f"{c} {pct(x)} → {pct(y)}" for c, x, y in zip(CLASSES[name], a, f)))
    lines += ["", "### Reported with the claim", ""]
    for name in g.DATASETS:
        c = {k: np.mean([res[name, (k, s), "test"]["clean"] for s in (0, 1)]) for k in ("A0", "FP")}
        m50 = {k: np.mean([res[name, (k, s), "test"]["map50"] for s in (0, 1)]) for k in ("A0", "FP")}
        an = {k: np.mean([res[name, (k, s), "test"]["anatomy"] for s in (0, 1)], 0) for k in ("A0", "FP")}
        lines.append(f"- {name}: mAP50 {pct(m50['A0'])} → {pct(m50['FP'])}; without pixel twins mAP50-95 "
                     f"{pct(c['A0'])} → {pct(c['FP'])} ({100 * (c['FP'] - c['A0']):+.2f}); error anatomy hit / ranked low / "
                     f"other class / no box: A0 " + " / ".join(pct(x) for x in an["A0"]) + ", FP " + " / ".join(pct(x) for x in an["FP"]))
    lines += ["", "### FP against the other architectures, test mAP50-95 (point estimates)", "",
              "| model | " + " | ".join(g.DATASETS) + " |", "|---|" + "---|" * len(g.DATASETS)]
    rows = [("RF-DETR-M + frozen DINOv3 (FP), seeds 0 / 1", lambda n: f"{pct(res[n, ('FP', 0), 'test']['point'])} / {pct(res[n, ('FP', 1), 'test']['point'])}"),
            ("RF-DETR-M (A0, T4), seeds 0 / 1", lambda n: f"{pct(res[n, ('A0', 0), 'test']['point'])} / {pct(res[n, ('A0', 1), 'test']['point'])}")]
    for lab in dict.fromkeys(lab for n in g.DATASETS for lab in g.DATASETS[n]["others"]):
        rows.append((lab, lambda n, lab=lab: pct(res[n, lab, "test"]["point"]) if (n, lab, "test") in res else "–"))
    for lab, f in rows:
        lines.append(f"| {lab} | " + " | ".join(f(n) for n in g.DATASETS) + " |")
    beats = {n: min(res[n, ("FP", s), "test"]["point"] for s in (0, 1)) > max(
        res[n, lab, "test"]["point"] for lab in g.DATASETS[n]["others"] if (n, lab, "test") in res) for n in g.DATASETS}
    fp_mean = {n: np.mean([res[n, ("FP", s), "test"]["point"] for s in (0, 1)]) for n in g.DATASETS}
    other_max = {n: max(res[n, lab, "test"]["point"] for lab in g.DATASETS[n]["others"] if (n, lab, "test") in res) for n in g.DATASETS}
    lines += ["", "FP (mean of seeds) above every other architecture: " + ", ".join(
        f"{n} {pct(fp_mean[n])} vs best other {pct(other_max[n])} ({'yes' if fp_mean[n] > other_max[n] else 'no'}; "
        f"both seeds above: {'yes' if beats[n] else 'no'})" for n in g.DATASETS), ""]


def secondary(lines, js):
    cfg = g.DATASETS["WTBD-group"]
    lines += ["## S: 25% of WTBD-group, FP − V1 seed 0 (secondary)", "",
              "| subset | split | V1 | FP | FP − V1 [95% CI] | P(> 0) |", "|---|---|---|---|---|---|"]
    cache = {"test": os.path.join(ROOT, "runs", "analysis", "bootstrap", "seeds_f25", "preds"),
             "val": os.path.join(ROOT, "runs", "analysis", "bootstrap", "v1_val_preds")}
    for split in ("test", "val"):
        imgs = bc.load_yolo_split(cfg["data"], split)
        files = sorted(imgs)
        dr = g.draws(g.scenes(cfg, split, files), 0)
        diffs, boots = [], []
        for k in F25:
            base = json.load(open(os.path.join(cache[split], f"rfdetrm{k}_wtbd_group_f25.json")))
            base = {kk: np.array(v).reshape(-1, 6) for kk, v in base.items()}
            assert set(base) == set(files), (k, split)
            fp = preds(os.path.join(RUNS, f"rfdetrmfp{k}_wtbd_group_f25"), cfg, split, files)
            sc = {}
            for lab, p in (("V1", base), ("FP", fp)):
                m = bc.per_image_matches(imgs, p, cfg["nc"])
                sc[lab] = (bc.score(m, files, cfg["nc"])[1], np.array([bc.score(m, d, cfg["nc"])[1] for d in dr]))
            d, b = sc["FP"][0] - sc["V1"][0], sc["FP"][1] - sc["V1"][1]
            diffs.append(d)
            boots.append(b)
            lines.append(f"| {k.strip('_') or 'class-matched'} | {split} | {pct(sc['V1'][0])} | {pct(sc['FP'][0])} | "
                         f"{100 * d:+.2f} {ci(b)} | {np.mean(b > 0):.3f} |")
        bm = np.mean(boots, 0)
        js[f"f25|{split}|diff"], js[f"f25|{split}|P"] = float(np.mean(diffs)), float(np.mean(bm > 0))
        js[f"f25|{split}|positive"] = int(np.sum(np.array(diffs) > 0))
        lines.append(f"| family mean | {split} | | | {100 * np.mean(diffs):+.2f} {ci(bm)} | {np.mean(bm > 0):.3f} |")
        print(f"scored f25 {split}", flush=True)
    ok = js["f25|test|diff"] > 0 and js["f25|test|P"] >= 0.95 and js["f25|test|positive"] >= 3
    lines += ["", f"**S {'passes' if ok else 'does not pass'}** (mean {100 * js['f25|test|diff']:+.2f}, P(> 0) "
              f"{js['f25|test|P']:.3f}, positive on {js['f25|test|positive']}/4).", ""]


def main():
    os.makedirs(OUT, exist_ok=True)
    js = {}
    lines = ["# G29: RF-DETR-M + frozen DINOv3 ViT-B prior (tools/g29_score.py)", "",
             f"Scene bootstrap, {g.N_BOOT} replicates; mAP50-95 in percent (bootstrap_ci evaluator).", ""]
    primary(lines, js)
    secondary(lines, js)
    json.dump(js, open(os.path.join(OUT, "score.json"), "w"), indent=1)
    open(os.path.join(OUT, "score.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
