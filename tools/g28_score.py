"""G28 scorer (research_log.md, G28; written before any G28 run): leave-one-dataset-out, class-agnostic.

For each target T (WTBD-group, WTBlade-group), on T's test set:
  ID-T  RF-DETR-M trained on T's group train, every box -> "defect"
  X-T   RF-DETR-M trained on every other source, twin-free against all of T (tools/g27_pool.py, pools XWTBD / XWTBL)
  ref   G27 A0 (class-aware), its predictions scored with every class -> "defect" (reported, not decisive)
Scores (scene bootstrap, 2,000 replicates, paired within a target): standard AP50 and AP50-95 (one-to-one matching),
tolerant AP50 and region AP (tools/granularity_eval.py; region AP is pixel level, independent of how a defect is
split into boxes), and recall AR50 (share of labels matched at IoU 0.5 by any prediction; unaffected by X boxing
things T does not label).
  G_std = ID - X (standard AP50), G_reg = ID - X (region AP), convention share = 1 - G_reg / G_std.
Decision (point estimates, per target; CIs reported):
  alive              on BOTH targets: X region AP >= 0.80 x ID region AP and convention share >= 0.5
  already generalizes  X >= ID in standard AP50 (on that target)
  dead               otherwise

  .venv/bin/python tools/g28_score.py        -> runs/analysis/g28/score.md (+ .json); missing runs are listed
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import bootstrap_ci as bc  # noqa: E402
import g27_score as g27  # noqa: E402
import granularity_eval as ge  # noqa: E402

OUT = os.path.join(ROOT, "runs", "analysis", "g28")
REGION_RATIO, CONVENTION_SHARE = 0.80, 0.5
TARGETS = {
    "WTBD-group": dict(cfg=g27.DATASETS["WTBD-group"], ID="rfdetrm640ag_id_wtbd_group", X="rfdetrm640ag_x_wtbd_group",
                       ref="rfdetrm768cos_wtbd_group_full"),
    "WTBlade-group": dict(cfg=g27.DATASETS["WTBlade-group"], ID="rfdetrm640ag_id_wtblade_group",
                          X="rfdetrm640ag_x_wtblade_group", ref="rfdetrm640cos_wtblade_group_full"),
}
ARMS = ("ID", "X", "ref")


def agnostic(a):
    a = np.array(a, float).copy()
    a[:, 0 if a.shape[1] == 5 else 5] = 0
    return a


def metrics(pre, m, files):
    """Standard AP50 / AP50-95, tolerant AP50, region AP and AR50 of one arm over a multiset of images."""
    s = ge.scores(pre, files, 1)
    tp = sum(float(m[f][(0, 0.5)][0].sum()) for f in files)
    n = sum(m[f][(0, 0.5)][2] for f in files)
    return dict(std50=float(s["standard50"][0]), std=float(s["standard"][0]), tol50=float(s["tolerant50"][0]),
                region=float(s["region"][0]), ar50=tp / n if n else float("nan"))


def main():
    os.makedirs(OUT, exist_ok=True)
    lines = ["# G28: leave-one-dataset-out, class-agnostic (tools/g28_score.py)", "",
             f"Test sets, scene bootstrap {g27.N_BOOT} replicates (paired within a target). Values in percent.", ""]
    res, verdicts = {}, {}
    for tname, t in TARGETS.items():
        cfg = t["cfg"]
        imgs = bc.load_yolo_split(cfg["data"], "test")
        for v in imgs.values():
            v["gt"] = agnostic(v["gt"])
        files = sorted(imgs)
        preds = {}
        for arm in ARMS:
            p = g27.rf_preds(t[arm], cfg, "test", files)
            if p is not None:
                preds[arm] = {f: agnostic(x) for f, x in p.items()}
        missing = [t[a] for a in ("ID", "X") if a not in preds]
        lines += [f"## {tname}", ""]
        if missing:
            lines += ["Missing runs: " + ", ".join(f"`{m}`" for m in missing), ""]
        if not preds:
            continue
        taus = np.unique(np.quantile(np.concatenate([p[:, 4] for P in preds.values() for p in P.values() if len(p)]),
                                     np.linspace(0, 1, ge.N_TAU)))
        pre = {a: ge.per_image(imgs, P, 1, taus) for a, P in preds.items()}
        mt = {a: bc.per_image_matches(imgs, P, 1) for a, P in preds.items()}
        dr = g27.draws(g27.scenes(cfg, "test", files), 0)
        point = {a: metrics(pre[a], mt[a], files) for a in preds}
        reps = {a: [metrics(pre[a], mt[a], d) for d in dr] for a in preds}
        keys = ("std50", "std", "tol50", "region", "ar50")
        lines += ["| arm | run | AP50 | AP50-95 | tolerant AP50 | region AP | AR50 |", "|---|---|---|---|---|---|---|"]
        for a in preds:
            lines.append(f"| {a} | `{t[a]}` | " + " | ".join(
                f"{100 * point[a][k]:.1f} {g27.ci(np.array([r[k] for r in reps[a]]))}" for k in keys) + " |")
        r = res[tname] = dict(point=point)
        if "ID" in preds and "X" in preds:
            d = {k: np.array([i[k] - x[k] for i, x in zip(reps["ID"], reps["X"])]) for k in keys}
            g_std, g_reg = point["ID"]["std50"] - point["X"]["std50"], point["ID"]["region"] - point["X"]["region"]
            share = 1 - g_reg / g_std if g_std > 0 else float("nan")
            ratio = point["X"]["region"] / point["ID"]["region"]
            lines += ["", "| ID − X | point | 95% CI | P(> 0) |", "|---|---|---|---|"]
            lines += [f"| {k} | {100 * (point['ID'][k] - point['X'][k]):+.1f} | {g27.ci(d[k])} | {(d[k] > 0).mean():.3f} |"
                      for k in keys]
            lines += ["", f"- X / ID region AP: **{ratio:.2f}** (bar ≥ {REGION_RATIO})",
                      f"- convention share 1 − G_reg / G_std: **{share:.2f}** (bar ≥ {CONVENTION_SHARE}; "
                      f"G_std {100 * g_std:+.1f}, G_reg {100 * g_reg:+.1f})"]
            if g_std <= 0:
                verdicts[tname] = "already generalizes"
            elif ratio >= REGION_RATIO and share >= CONVENTION_SHARE:
                verdicts[tname] = "alive"
            else:
                verdicts[tname] = "dead"
            r.update(g_std=g_std, g_reg=g_reg, share=share, ratio=ratio, verdict=verdicts[tname])
            lines += [f"- **{tname}: {verdicts[tname]}**"]
        lines.append("")
    if len(verdicts) == len(TARGETS):
        v = set(verdicts.values())
        overall = ("alive" if v == {"alive"} else
                   "already generalizes" if "already generalizes" in v and "dead" not in v else "dead")
        lines += [f"## Decision: **{overall}** ({', '.join(f'{k}: {x}' for k, x in verdicts.items())})", ""]
    else:
        lines += ["## Decision: pending (both targets need ID and X)", ""]
    out = "\n".join(lines)
    open(os.path.join(OUT, "score.md"), "w").write(out)
    json.dump(res, open(os.path.join(OUT, "score.json"), "w"), indent=1)
    print(out)


if __name__ == "__main__":
    main()
