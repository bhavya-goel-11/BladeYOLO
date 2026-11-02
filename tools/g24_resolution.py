"""G24 (research_log.md): RF-DETR-M at resolution 768 vs the default 576 on the same subsets, as the
localisation-regime fix. WTBlade-group f08 (R1, R2) and WTBD-group f25 (R3 = relative gain, WTBlade > WTBD).

  R1  WTBlade test mAP50-95, 768 - 576 > 0, paired scene bootstrap P >= 0.95 (2,000), better on >= 3/4 models
  R2  WTBlade crack no-box rate (error anatomy) lower at 768 on >= 3/4 models
  R3  relative gain (delta / 576 family mean) larger on WTBlade than on WTBD, bootstrap P >= 0.90

  .venv/bin/python tools/g24_resolution.py  -> runs/analysis/g24/report.md (WTBD half skipped until imported)
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import bootstrap_ci as bc  # noqa: E402
import error_anatomy as ea  # noqa: E402
import proto_rescore as pr  # noqa: E402

B = os.path.join(ROOT, "runs", "analysis", "bootstrap")
OUT = os.path.join(ROOT, "runs", "analysis", "g24")
SUBS = ["", "_rand1", "_rand2", "_rand3"]
N_BOOT = 2000
SETS = {
    "WTBlade": dict(data=ea.WTBLADE, cache=os.path.join(B, "wtblade_f08", "test"), frac="f08", ds="wtblade"),
    "WTBD": dict(data=ea.WTBD, cache=os.path.join(B, "seeds_f25", "preds"), frac="f25", ds="wtbd"),
}


def name(pre, s, ds, frac):
    return f"{pre}{s}_{ds}_group_{frac}" if s else f"{pre}_{ds}_group_{frac}"


def load(path, data):
    return ea.load_preds(path, data) if os.path.exists(path) else None


def main():
    os.makedirs(OUT, exist_ok=True)
    lines = ["# G24: RF-DETR-M at 768 vs 576 on the same subsets (research_log.md)", ""]
    rel = {}
    for key, cfg in SETS.items():
        (data, names) = cfg["data"]
        if key == "WTBlade":
            pr.use_wtblade("f08")
        else:  # back to the WTBD defaults
            pr.NC, pr.CLASSES = 6, names
        pairs = [(name("rfdetrm", s, cfg["ds"], cfg["frac"]), name("rfdetrmr768", s, cfg["ds"], cfg["frac"])) for s in SUBS]
        preds = {r: load(os.path.join(cfg["cache"], r + ".json"), data) for p in pairs for r in p}
        if any(v is None for v in preds.values()):
            lines += [f"## {key}: 768 predictions not imported yet", ""]
            continue
        imgs = bc.load_yolo_split(data, "test")
        files = sorted(imgs)
        ms = {r: bc.per_image_matches(imgs, p, len(names)) for r, p in preds.items()}
        pt = {r: pr.metrics(ms[r], files) for r in preds}
        groups = pr.scenes("test", files) if key == "WTBlade" else bc_groups(files)
        rng = np.random.default_rng(0)
        bs_d, bs_rel = [], []
        for _ in range(N_BOOT):
            dr = [f for gi in rng.integers(0, len(groups), len(groups)) for f in groups[gi]]
            lo = np.mean([pr.metrics(ms[a], dr)[0] for a, _ in pairs])
            hi = np.mean([pr.metrics(ms[b], dr)[0] for _, b in pairs])
            bs_d.append(hi - lo)
            bs_rel.append((hi - lo) / lo)
        bs_d, bs_rel = np.array(bs_d), np.array(bs_rel)
        lo_m = np.mean([pt[a][0] for a, _ in pairs])
        hi_m = np.mean([pt[b][0] for _, b in pairs])
        rel[key] = ((hi_m - lo_m) / lo_m, bs_rel)
        wins = sum(pt[b][0] > pt[a][0] for a, b in pairs)
        lines += [f"## {key} test ({len(files)} images, {len(groups)} scenes)", "",
                  "| subset | 576 mAP50-95 | 768 mAP50-95 | Δ |", "|---|---|---|---|"]
        lines += [f"| {s or 'class-matched'} | {pt[a][0]:.4f} | {pt[b][0]:.4f} | {pt[b][0] - pt[a][0]:+.4f} |"
                  for s, (a, b) in zip(SUBS, pairs)]
        lines += [f"| **family mean** | **{lo_m:.4f}** | **{hi_m:.4f}** | **{hi_m - lo_m:+.4f}** |", "",
                  f"- Δ {hi_m - lo_m:+.4f} [{np.percentile(bs_d, 2.5):+.4f}, {np.percentile(bs_d, 97.5):+.4f}], "
                  f"P(>0) {(bs_d > 0).mean():.3f}, better on {wins}/4; relative {100 * (hi_m - lo_m) / lo_m:+.1f}%", ""]
        pc_lo = np.nanmean([pt[a][2] for a, _ in pairs], 0)
        pc_hi = np.nanmean([pt[b][2] for _, b in pairs], 0)
        lines += ["| class | 576 AP50-95 | 768 AP50-95 | Δ |", "|---|---|---|---|"]
        lines += [f"| {n} | {pc_lo[c]:.4f} | {pc_hi[c]:.4f} | {pc_hi[c] - pc_lo[c]:+.4f} |" for c, n in enumerate(names)]
        # error anatomy per model: no-box share for each class
        lines += ["", "| class | no-box rate 576 (per model) | no-box rate 768 (per model) | lower at 768 |", "|---|---|---|---|"]
        nobox = {}
        for r in preds:
            a = ea.anatomy(imgs, data, cfg["cache"], [r], len(names))
            nobox[r] = [np.mean(a[a[:, 0] == c, 1] == 3) for c in range(len(names))]
        for c, n in enumerate(names):
            lo_r = [nobox[a][c] for a, _ in pairs]
            hi_r = [nobox[b][c] for _, b in pairs]
            lines.append(f"| {n} | {' / '.join(f'{v:.2f}' for v in lo_r)} | {' / '.join(f'{v:.2f}' for v in hi_r)} | "
                         f"{sum(h < l for l, h in zip(lo_r, hi_r))}/4 |")
        lines.append("")
        if key == "WTBlade":
            r1 = hi_m - lo_m > 0 and (bs_d > 0).mean() >= 0.95 and wins >= 3
            ci = names.index("crack")
            r2 = sum(nobox[b][ci] < nobox[a][ci] for a, b in pairs) >= 3
            lines += [f"**R1 {'✓' if r1 else '✗'}  R2 {'✓' if r2 else '✗'}**", ""]
    if len(rel) == 2:
        d = rel["WTBlade"][1] - rel["WTBD"][1]
        r3 = rel["WTBlade"][0] > rel["WTBD"][0] and (d > 0).mean() >= 0.90
        lines += [f"**R3 {'✓' if r3 else '✗'}**: relative gain WTBlade {100 * rel['WTBlade'][0]:+.1f}% vs WTBD "
                  f"{100 * rel['WTBD'][0]:+.1f}%, P(WTBlade > WTBD) {(d > 0).mean():.3f} (independent scene draws)"]
    open(os.path.join(OUT, "report.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def bc_groups(files):
    """WTBD-group test scenes (runs/analysis/bootstrap/wtbd_group/groups.csv) as lists of files."""
    import csv

    g = {r["image"]: r["group"] for r in csv.DictReader(open(os.path.join(B, "wtbd_group", "groups.csv")))}
    out = {}
    for f in files:
        out.setdefault(g[os.path.basename(f)], []).append(f)
    return list(out.values())


if __name__ == "__main__":
    main()
