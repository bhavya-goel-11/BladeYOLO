"""G22 (research_log.md): does end-to-end fine-tuning move coverage from recall to ranking? The G16/G16-R proximity
law per model family, on WTBD-group and WTBlade-group, with a frozen-encoder RF-DETR control. Nothing here is tuned.

  x      max cosine (frozen DINOv3 ViT-B box embeddings, tools/proto_rescore.py caches) between a test box and the
         model's training-subset boxes of its class (NaN if the subset has none)
  y      highest same-class confidence at IoU >= 0.5 (0 if none);  det = y >= 0.05;  cand = y >= 0.001
  slope  coefficient of x in  outcome ~ x + log(1 + n)  with test-box and subset fixed effects (absorbed by alternating
         demeaning, identical to the dummy-variable OLS of tools/coverage_gate.py), cluster bootstrap over test scenes,
         the same draws for every family of a dataset

  .venv/bin/python tools/g22_coverage.py check    reproduce G16-R's WTBD RF-DETR slopes with this estimator
  .venv/bin/python tools/g22_coverage.py stageA   WTBlade, fine-tuned RF-DETR f25 (4) and f25+f08 (8) -> runs/analysis/g22/stageA.md
  .venv/bin/python tools/g22_coverage.py stageB   both datasets, fine-tuned vs frozen-encoder (rfdetrmfz_*) on the same
                                                  f25 subsets -> runs/analysis/g22/stageB.md
"""

import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import bootstrap_ci as bc  # noqa: E402
from coverage_gate import fe_slope, outcome, proximity  # noqa: E402

SPLITS = os.path.join(ROOT, "tools", "splits")
B = os.path.join(ROOT, "runs", "analysis", "bootstrap")
PR = os.path.join(ROOT, "runs", "analysis", "proto_rescore")
OUT = os.path.join(ROOT, "runs", "analysis", "g22")
N_BOOT, DET, CAND = 1000, 0.05, 0.001
SUBS = ["", "_rand1", "_rand2", "_rand3"]


def runs(pre, ds, frac, cache):
    """[(run, subset file, prediction cache)] for the class-matched + rand1-3 subsets."""
    out = []
    for s in SUBS:
        name = f"{pre}{s}_{ds}_group_{frac}" if s else f"{pre}_{ds}_group_{frac}"
        out.append((name, os.path.join(SPLITS, f"{ds}_group_train_{frac}{s}.txt"), cache))
    return out


DATASETS = {
    "WTBD": dict(data=os.path.join(ROOT, "data", "blade-defect-benchmark", "leakfree", "WTBD-group"), proto=PR,
                 groups=os.path.join(B, "wtbd_group", "groups.csv"),
                 fams={"fine-tuned f25": runs("rfdetrm", "wtbd", "f25", os.path.join(B, "seeds_f25", "preds")),
                       "frozen f25": runs("rfdetrmfz", "wtbd", "f25", os.path.join(B, "seeds_f25", "preds"))}),
    "WTBlade": dict(data=os.path.join(ROOT, "data", "blade-defect-benchmark", "leakfree", "WTBlade-group"), proto=os.path.join(PR, "wtblade"),
                    groups=os.path.join(ROOT, "data", "blade-defect-benchmark", "leakfree", "WTBlade-group", "groups.csv"),
                    fams={"fine-tuned f25": runs("rfdetrm", "wtblade", "f25", os.path.join(B, "wtblade_f25", "test")),
                          "fine-tuned f08": runs("rfdetrm", "wtblade", "f08", os.path.join(B, "wtblade_f08", "test")),
                          "frozen f25": runs("rfdetrmfz", "wtblade", "f25", os.path.join(B, "wtblade_f25", "test"))}),
}


# ---- estimator ----

def demean(v, box, sub, iters=200, tol=1e-10):
    """Residual of v (n,) or (n, k) after projecting out box and subset fixed effects (alternating projections)."""
    v = np.array(v, float).reshape(len(box), -1)
    nb, ns = np.bincount(box), np.bincount(sub)
    for _ in range(iters):
        prev = v.copy()
        for g, n in ((box, nb), (sub, ns)):
            m = np.stack([np.bincount(g, v[:, j], minlength=len(n)) for j in range(v.shape[1])], 1) / np.maximum(n, 1)[:, None]
            v = v - m[g]
        if np.abs(v - prev).max() < tol:
            break
    return v


def slope(y, X, box, sub):
    """Coefficients of X in y ~ X + box FE + subset FE (Frisch-Waugh-Lovell on the demeaned data)."""
    _, box = np.unique(box, return_inverse=True)
    _, sub = np.unique(sub, return_inverse=True)
    Z = demean(np.c_[y, X], box, sub)
    return np.linalg.lstsq(Z[:, 1:], Z[:, 0], rcond=None)[0]


# ---- data ----

def test_boxes(cfg):
    import proto_rescore as pr

    imgs = bc.load_yolo_split(cfg["data"], "test")
    grp = {r["image"]: int(r["group"]) for r in csv.DictReader(open(cfg["groups"]))}
    rows = []
    for f, im in sorted(imgs.items()):
        if not len(im["gt"]):
            continue
        fmap = np.load(os.path.join(cfg["proto"], "maps", "test", os.path.splitext(os.path.basename(f))[0] + ".npy"))
        for g, e in zip(im["gt"], pr.box_embed(fmap.astype(np.float32), im["gt"][:, 1:])):
            rows.append(dict(img=f, cls=int(g[0]), box=g, emb=e, group=grp[os.path.basename(f)]))
    return imgs, rows


def panel(cfg, fam, imgs, tb, bank, nc):
    """Rows over (test box, model): box index, subset id, scene group, log(1+n), x, y."""
    q = np.stack([b["emb"] for b in tb])
    qc = np.array([b["cls"] for b in tb])
    P = []
    for si, (run, sfile, cache) in enumerate(fam):
        names = [n.strip() for n in open(sfile) if n.strip()]
        e = np.concatenate([bank[n][1] for n in names if n in bank and len(bank[n][1])])
        c = np.concatenate([bank[n][0][:, 0] for n in names if n in bank and len(bank[n][1])]).astype(int)
        n_c = np.bincount(c, minlength=nc)
        x = proximity(q, qc, e, c, 1)
        pred = bc.predict(os.path.join(ROOT, "runs", "detect", "BladeYOLO_WindSurface", run), imgs, cache)
        y = np.array([outcome(b["box"][None], pred.get(b["img"], np.zeros((0, 6))))[0] for b in tb])
        P += [(i, si, b["group"], np.log1p(n_c[b["cls"]]), x[i], y[i]) for i, b in enumerate(tb)]
    return np.array(P, float)


def available(fam):
    return all(os.path.exists(os.path.join(cache, run + ".json")) for run, _, cache in fam)


def analyse(ds, fams, outcomes):
    """{(family, outcome): (estimate, bootstrap array)} on shared scene draws."""
    cfg = DATASETS[ds]
    import yaml

    nc = len(yaml.safe_load(open(os.path.join(cfg["data"], "data.yaml")))["names"])
    bank = np.load(os.path.join(cfg["proto"], "train_bank.npz"), allow_pickle=True)["bank"].item()
    imgs, tb = test_boxes(cfg)
    panels = {}
    for name in fams:
        fam = cfg["fams"][name] if not isinstance(name, tuple) else sum((cfg["fams"][n] for n in name), [])
        panels[name] = panel(cfg, fam, imgs, tb, bank, nc)
    groups = np.unique(panels[fams[0]][:, 2]).astype(int)
    rng = np.random.default_rng(0)
    draws = [groups[rng.integers(0, len(groups), len(groups))] for _ in range(N_BOOT)]
    res, info = {}, {}
    for name, P in panels.items():
        ok = np.isfinite(P[:, 4])
        P = P[ok]
        by_g = {g: np.where(P[:, 2] == g)[0] for g in groups}
        Ys = {"conf": P[:, 5], "det": (P[:, 5] >= DET).astype(float), "cand": (P[:, 5] >= CAND).astype(float)}
        X = P[:, [4, 3]]
        info[name] = dict(rows=len(P), boxes=len(np.unique(P[:, 0])), models=len(np.unique(P[:, 1])),
                          det=Ys["det"].mean(), cand=Ys["cand"].mean())
        for o in outcomes:
            est = slope(Ys[o], X, P[:, 0], P[:, 1])[0]
            bs = []
            for d in draws:
                idx = np.concatenate([by_g[g] for g in d])
                # a resampled scene's boxes become distinct boxes (as in coverage_gate's draws: box ids repeat, so
                # duplicates share one fixed effect there; we keep that convention)
                bs.append(slope(Ys[o][idx], X[idx], P[idx, 0], P[idx, 1])[0])
            res[(name, o)] = (est, np.array(bs))
    return res, info, len(tb), len(groups)


def fmt(est, b):
    return f"{est:+.3f} [{np.percentile(b, 2.5):+.3f}, {np.percentile(b, 97.5):+.3f}], P(>0) {(b > 0).mean():.3f}"


def check():
    """Point estimates must equal G16-R's dummy-variable OLS (RF-DETR class-aware +0.478, binarised -0.016)."""
    cfg = DATASETS["WTBD"]
    bank = np.load(os.path.join(cfg["proto"], "train_bank.npz"), allow_pickle=True)["bank"].item()
    imgs, tb = test_boxes(cfg)
    P = panel(cfg, cfg["fams"]["fine-tuned f25"], imgs, tb, bank, 6)
    P = P[np.isfinite(P[:, 4])]
    for lab, y in (("conf", P[:, 5]), ("det", (P[:, 5] >= DET).astype(float))):
        a = slope(y, P[:, [4, 3]], P[:, 0], P[:, 1])[0]
        b = fe_slope(y, P[:, [4, 3]], P[:, 0].astype(int), P[:, 1].astype(int))[0]
        print(f"WTBD RF-DETR {lab}: demeaned {a:+.4f}  dummy OLS {b:+.4f}")


def stage_a():
    os.makedirs(OUT, exist_ok=True)
    fams = ["fine-tuned f25", ("fine-tuned f25", "fine-tuned f08")]
    res, info, nb, ng = analyse("WTBlade", fams, ("conf", "det", "cand"))
    lines = ["# G22 stage A: proximity law on fine-tuned RF-DETR, WTBlade-group test (research_log.md)", "",
             f"{nb} test boxes, {ng} scene clusters, {N_BOOT} replicates; x = max DINOv3 ViT-B cosine to the subset's "
             "same-class boxes; slope of x with box and subset fixed effects and log(1+n).", "",
             "| models | rows | detected (≥ 0.05) | candidate (≥ 0.001) | class-aware conf | detected | candidate |",
             "|---|---|---|---|---|---|---|"]
    for f in fams:
        lab = f if isinstance(f, str) else "fine-tuned f25 + f08"
        i = info[f]
        lines.append(f"| {lab} ({i['models']}) | {i['rows']} | {i['det']:.3f} | {i['cand']:.3f} | "
                     + " | ".join(fmt(*res[(f, o)]) for o in ("conf", "det", "cand")) + " |")
    a1 = all(res[(f, "conf")][0] > 0 and (res[(f, "conf")][1] > 0).mean() >= 0.95 for f in fams)
    lines += ["", f"**A1 {'holds' if a1 else 'fails'}** (class-aware confidence slope > 0, P ≥ 0.95, on f25 and pooled)."]
    open(os.path.join(OUT, "stageA.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def stage_b():
    os.makedirs(OUT, exist_ok=True)
    lines = ["# G22 stage B: fine-tuned vs frozen-encoder RF-DETR, same f25 subsets (research_log.md)", ""]
    verdict = {}
    for ds in DATASETS:
        fams = ["fine-tuned f25", "frozen f25"]
        if not all(available(DATASETS[ds]["fams"][f]) for f in fams):
            lines += [f"## {ds}: frozen-encoder predictions not imported yet", ""]
            verdict[ds] = None
            continue
        res, info, nb, ng = analyse(ds, fams, ("conf", "det", "cand"))
        lines += [f"## {ds} ({nb} test boxes, {ng} scenes)", "",
                  "| family | detected (≥ 0.05) | class-aware conf | detected | candidate |", "|---|---|---|---|---|"]
        for f in fams:
            lines.append(f"| {f} | {info[f]['det']:.3f} | " + " | ".join(fmt(*res[(f, o)]) for o in ("conf", "det", "cand")) + " |")
        d_est = res[("fine-tuned f25", "det")][0] - res[("frozen f25", "det")][0]
        d_b = res[("fine-tuned f25", "det")][1] - res[("frozen f25", "det")][1]
        b1 = res[("frozen f25", "det")][0] > 0 and (res[("frozen f25", "det")][1] > 0).mean() >= 0.95
        b2 = d_est < 0 and (d_b < 0).mean() >= 0.95
        b3 = res[("fine-tuned f25", "conf")][0] > 0 and (res[("fine-tuned f25", "conf")][1] > 0).mean() >= 0.95
        verdict[ds] = b1 and b2 and b3
        lines += ["", f"- detected slope, fine-tuned − frozen: {d_est:+.3f} [{np.percentile(d_b, 2.5):+.3f}, "
                      f"{np.percentile(d_b, 97.5):+.3f}], P(<0) {(d_b < 0).mean():.3f}",
                  f"- B1 {'✓' if b1 else '✗'}  B2 {'✓' if b2 else '✗'}  B3 {'✓' if b3 else '✗'}", ""]
    if all(v is not None for v in verdict.values()):
        lines.append(f"**Item 1 {'replicates' if all(verdict.values()) else 'does not replicate'}** "
                     f"({', '.join(f'{d}: {v}' for d, v in verdict.items())}).")
    open(os.path.join(OUT, "stageB.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    {"check": check, "stageA": stage_a, "stageB": stage_b}[sys.argv[1]]()
