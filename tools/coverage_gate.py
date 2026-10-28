"""Gate G16 (Phase 31): is detection of a held-out defect governed by its proximity, in frozen DINOv3 space, to the
nearest labelled defect of its class (the coverage hypothesis H16)? Definitions and the bar are pre-registered in
research_log.md (G16); nothing here is tuned.

Inputs are cached only: test predictions of the models trained on known 25% subsets (tools/bootstrap_ci.py cache),
DINOv3 ViT-B maps of the test/val images and the training-box bank (tools/proto_rescore.py), and the 128-per-image
training patches of tools/select_scenes.py.
  x(b, S) = max cosine between test box b and the labelled boxes of b's class in subset S (undefined if none)
  y(b, m) = highest confidence of model m's predictions of b's class with IoU >= 0.5 to b (0 if none)
  1. y ~ x + log(1 + n) with test-box and subset fixed effects; cluster bootstrap over test scenes
  2. X(S) = mean over classes of mean x over the class's test boxes (x = 0 if undefined) vs subset quality
  3. T(S) = transductive coverage of the unlabelled test (val) images' patches vs subset quality

  .venv/bin/python tools/coverage_gate.py  -> runs/analysis/coverage_gate/{report.md, pairs.csv, subsets.csv}
"""

import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import bootstrap_ci as bc  # noqa: E402
import detection_metrics as dm  # noqa: E402

DATA = os.path.join(ROOT, "external", "WTBD-group")
SPLITS = os.path.join(ROOT, "tools", "splits")
RUNS = os.path.join(ROOT, "runs", "detect", "BladeYOLO_WindSurface")
ARCHIVE = os.path.join(ROOT, "runs", "analysis", "archive")
PREDS = os.path.join(ROOT, "runs", "analysis", "bootstrap", "seeds_f25", "preds")
PROTO = os.path.join(ROOT, "runs", "analysis", "proto_rescore")
PATCHES = os.path.join(ROOT, "runs", "analysis", "select_scenes", "wtbd_group_train_f25_patches_b.npy")
GROUPS = {"test": os.path.join(ROOT, "runs", "analysis", "bootstrap", "wtbd_group", "groups.csv"),
          "val": os.path.join(ROOT, "runs", "analysis", "bootstrap", "h5_last_val", "groups.csv")}
OUT = os.path.join(ROOT, "runs", "analysis", "coverage_gate")
NC, N_BOOT, K, PER_IMAGE = 6, 1000, 1024, 128
FAMILIES = {  # subset file suffix -> runs trained on it
    "ViT-B": {
        "": ["bladeyolo2b_wtbd_group_f25_s0", "bladeyolo2b_wtbd_group_f25_s1"],
        "_cover": ["bladeyolo2b_cover_wtbd_group_f25_s0", "bladeyolo2b_cover_wtbd_group_f25_s1"],
        "_cover_k1": ["bladeyolo2b_coverk1_wtbd_group_f25_s0"],
        "_cover_k2": ["bladeyolo2b_coverk2_wtbd_group_f25_s0"],
        **{f"_rand{k}": [f"bladeyolo2b_rand{k}_wtbd_group_f25_s0"] for k in range(1, 9)},
        **{f"_freesel{s}_s{k}": [f"bladeyolo2b_freesel{s}s{k}_wtbd_group_f25"] for s in (224, 640) for k in (0, 1)},
    },
    "no DINOv3": {
        "": ["bladeyolo2nodino_wtbd_group_f25_s0", "bladeyolo2nodino_wtbd_group_f25_s1"],
        **{f"_rand{k}": [f"bladeyolo2nodino_rand{k}_wtbd_group_f25"] for k in range(1, 9)},
    },
}


# ---- pure pieces (tests/test_coverage_gate.py) ----

def proximity(q, q_cls, bank, bank_cls, k=1):
    """(n,) mean of the top-k cosines of each query to bank rows of its own class; NaN where the class has none."""
    out = np.full(len(q), np.nan)
    for c in np.unique(q_cls):
        bc_ = bank[bank_cls == c]
        if len(bc_):
            s = q[q_cls == c] @ bc_.T
            kk = min(k, s.shape[1])
            out[q_cls == c] = np.sort(s, 1)[:, -kk:].mean(1)
    return out


def outcome(gt, pred, any_class=False, iou_thr=0.5):
    """(n,) highest confidence of predictions (x1 y1 x2 y2 conf cls) of each gt's class (any class if any_class)
    with IoU >= iou_thr to it; 0 if none. gt is (n, 5) [cls, x1, y1, x2, y2]."""
    y = np.zeros(len(gt))
    if not len(gt) or not len(pred):
        return y
    iou = dm.iou_matrix(gt[:, 1:], pred[:, :4])
    ok = iou >= iou_thr
    if not any_class:
        ok &= gt[:, :1] == pred[None, :, 5]
    conf = np.where(ok, pred[None, :, 4], 0.0)
    return conf.max(1)


def fe_slope(y, X, box, sub):
    """OLS coefficients of the columns of X in y ~ X + box fixed effects + subset fixed effects."""
    ub, bi = np.unique(box, return_inverse=True)
    us, si = np.unique(sub, return_inverse=True)
    D = np.zeros((len(y), len(ub) + len(us) - 1))
    D[np.arange(len(y)), bi] = 1
    m = si > 0
    D[np.where(m)[0], len(ub) + si[m] - 1] = 1
    A = np.c_[X, D]
    return np.linalg.lstsq(A, y, rcond=None)[0][:X.shape[1]]


def coverage(target_centroids, patches):
    """Mean over target centroids of the highest cosine to any of the patches (all L2-normalised)."""
    best = np.full(len(target_centroids), -1.0)
    for i in range(0, len(patches), 8192):
        best = np.maximum(best, (patches[i:i + 8192] @ target_centroids.T).max(0))
    return float(best.mean())


# ---- data ----

def results_csv(run):
    for d in (os.path.join(RUNS, run), os.path.join(ARCHIVE, f"run_{run}")):
        p = os.path.join(d, "results.csv")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(run)


def best_val(run):
    res = list(csv.DictReader(open(results_csv(run))))
    k = [c for c in res[0] if "mAP50-95" in c][0]
    return max(float(r[k]) for r in res)


def subset_names(suf):
    return [n.strip() for n in open(os.path.join(SPLITS, f"wtbd_group_train_f25{suf}.txt")) if n.strip()]


def split_boxes(split):
    """Every gt box of the split: image path, class, 640-px box, embedding, scene group."""
    import proto_rescore as pr

    imgs = bc.load_yolo_split(DATA, split)
    grp = {r["image"]: int(r["group"]) for r in csv.DictReader(open(GROUPS[split]))}
    rows = []
    for f, im in sorted(imgs.items()):
        if not len(im["gt"]):
            continue
        fmap = np.load(os.path.join(PROTO, "maps", split, os.path.splitext(os.path.basename(f))[0] + ".npy"))
        emb = pr.box_embed(fmap.astype(np.float32), im["gt"][:, 1:])
        for g, e in zip(im["gt"], emb):
            rows.append(dict(img=f, cls=int(g[0]), box=g, emb=e, group=grp[os.path.basename(f)]))
    return imgs, rows


def target_centroids(split, seed=0):
    from sklearn.cluster import KMeans

    d = os.path.join(PROTO, "maps", split)
    rng = np.random.default_rng(seed)
    P = []
    for f in sorted(os.listdir(d)):
        m = np.load(os.path.join(d, f)).astype(np.float32).reshape(-1, 768)
        m = m[rng.choice(len(m), PER_IMAGE, replace=False)]
        P.append(m / np.linalg.norm(m, axis=1, keepdims=True))
    C = KMeans(K, n_init=1, random_state=seed).fit(np.concatenate(P)).cluster_centers_
    return C / np.linalg.norm(C, axis=1, keepdims=True)


def main():
    import subset_properties as sp

    os.makedirs(OUT, exist_ok=True)
    bank = np.load(os.path.join(PROTO, "train_bank.npz"), allow_pickle=True)["bank"].item()
    train_files = sorted(os.listdir(os.path.join(DATA, "train", "images")))
    tidx = {f: i for i, f in enumerate(train_files)}
    F = np.load(PATCHES, mmap_mode="r")
    te, tboxes = split_boxes("test")
    va, vboxes = split_boxes("val")
    tfiles = sorted(te)
    print(f"test boxes {len(tboxes)}, val boxes {len(vboxes)}", flush=True)
    Ct, Cv = target_centroids("test"), target_centroids("val")
    print("target centroids done", flush=True)

    def sub_bank(names):
        e = [bank[n][1] for n in names if n in bank]
        c = [bank[n][0][:, 0] for n in names if n in bank]
        return np.concatenate(e), np.concatenate(c).astype(int)

    lines = ["# G16: coverage gate (pre-registered, research_log.md Phase 31)", ""]
    pairs, subrows = [], []
    for fam, subsets in FAMILIES.items():
        for suf, runs in subsets.items():
            names = subset_names(suf)
            be, bcls = sub_bank(names)
            n_c = np.bincount(bcls, minlength=NC)
            q = np.stack([b["emb"] for b in tboxes])
            qc = np.array([b["cls"] for b in tboxes])
            x1, x3 = proximity(q, qc, be, bcls, 1), proximity(q, qc, be, bcls, 3)
            ys, ya = [], []
            for r in runs:
                pred = bc.predict(os.path.join(RUNS, r), te, PREDS)
                ys.append(np.concatenate([outcome(b["box"][None], pred.get(b["img"], np.zeros((0, 6))))
                                          for b in tboxes]))
                ya.append(np.concatenate([outcome(b["box"][None], pred.get(b["img"], np.zeros((0, 6))), True)
                                          for b in tboxes]))
            y, yagn = np.mean(ys, 0), np.mean(ya, 0)
            for i, b in enumerate(tboxes):
                pairs.append(dict(family=fam, subset=suf or "_class-matched", box=i, group=b["group"], cls=b["cls"],
                                  x=x1[i], x3=x3[i], n=int(n_c[b["cls"]]), y=y[i], y_agn=yagn[i]))
            # subset level
            test = np.mean([bc.score(bc.per_image_matches(te, bc.predict(os.path.join(RUNS, r), te, PREDS), NC),
                                     tfiles, NC)[1] for r in runs])
            X = np.nanmean([np.nan_to_num(x1[qc == c]).mean() for c in range(NC)])
            row = dict(family=fam, subset=suf[1:] or "class-matched", test=test, X=X)
            if fam == "ViT-B":
                vq = np.stack([b["emb"] for b in vboxes])
                vc = np.array([b["cls"] for b in vboxes])
                vx = proximity(vq, vc, be, bcls, 1)
                sel = np.array(sorted(tidx[n] for n in names))
                Fs = np.asarray(F[sel], np.float32).reshape(-1, F.shape[-1])
                row.update(val=float(np.mean([best_val(r) for r in runs])),
                           Xval=np.mean([np.nan_to_num(vx[vc == c]).mean() for c in range(NC)]),
                           T=coverage(Ct, Fs), Tval=coverage(Cv, Fs))
            subrows.append(row)
            print(f"{fam} {row['subset']}: test {test:.3f} X {X:.4f}" + (f" T {row['T']:.4f}" if "T" in row else ""),
                  flush=True)

    for name, rows in (("pairs.csv", pairs), ("subsets.csv", subrows)):
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(os.path.join(OUT, name), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    # condition 1
    lines += ["## Condition 1: instance-level law (box + subset fixed effects, log(1+n) covariate)", "",
              "| family | outcome | proximity | coef of x | 95% CI | P(>0) | pairs |", "|---|---|---|---|---|---|---|"]
    c1 = {}
    for fam in FAMILIES:
        P = [p for p in pairs if p["family"] == fam]
        groups = sorted({p["group"] for p in P})
        by_group = {g: [i for i, p in enumerate(P) if p["group"] == g] for g in groups}
        for out_k, xk in (("y", "x"), ("y", "x3"), ("y_agn", "x")):
            ok = np.array([np.isfinite(p[xk]) for p in P])
            yv = np.array([p[out_k] for p in P])
            Xv = np.c_[[p[xk] for p in P], np.log1p([p["n"] for p in P])]
            bx, sb = np.array([p["box"] for p in P]), np.array([p["subset"] for p in P])
            est = fe_slope(yv[ok], Xv[ok], bx[ok], sb[ok])[0]
            rng = np.random.default_rng(0)
            boot = []
            for _ in range(N_BOOT):
                idx = np.concatenate([by_group[groups[g]] for g in rng.integers(0, len(groups), len(groups))])
                idx = idx[ok[idx]]
                boot.append(fe_slope(yv[idx], Xv[idx], bx[idx], sb[idx])[0])
            boot = np.array(boot)
            pp = float((boot > 0).mean())
            if (out_k, xk) == ("y", "x"):
                c1[fam] = pp >= 0.95 and est > 0
            lines.append(f"| {fam} | {'class-aware' if out_k == 'y' else 'class-agnostic'} | "
                         f"{'max' if xk == 'x' else 'top-3 mean'} | {est:+.3f} | [{np.percentile(boot, 2.5):+.3f}, "
                         f"{np.percentile(boot, 97.5):+.3f}] | {pp:.3f} | {int(ok.sum())} |")
        absent = [p for p in pairs if p["family"] == fam and not np.isfinite(p["x"])]
        if absent:
            lines.append(f"| {fam} | class absent from subset: mean y {np.mean([p['y'] for p in absent]):.3f} "
                         f"over {len(absent)} pairs | | | | | |")
    lines += ["", f"Condition 1 {'**passes**' if all(c1.values()) else 'fails'} "
                  f"({', '.join(f'{k}: {v}' for k, v in c1.items())}).", ""]

    # conditions 2 and 3
    def rho(a, b, n=20000):
        return sp.spearman_perm(np.asarray(a, float), np.asarray(b, float), n=n)

    vb = [r for r in subrows if r["family"] == "ViT-B"]
    nd = [r for r in subrows if r["family"] == "no DINOv3"]
    rnd = lambda rows: [r for r in rows if r["subset"].startswith("rand")]  # noqa: E731
    res = {}
    for stat, valstat in (("X", "Xval"), ("T", "Tval")):
        r_all, p_all = rho([r[stat] for r in vb], [r["test"] for r in vb])
        r_rb = rho([r[stat] for r in rnd(vb)], [r["test"] for r in rnd(vb)], 1)[0]
        r_val = rho([r[valstat] for r in vb], [r["val"] for r in vb], 1)[0]
        r_rn = rho([r["X"] for r in rnd(nd)], [r["test"] for r in rnd(nd)], 1)[0] if stat == "X" else np.nan
        ok = r_all >= 0.6 and p_all < 0.05 and r_rb >= 0.5 and r_val >= 0.4 and (stat == "T" or r_rn >= 0.5)
        res[stat] = ok
        lines.append(f"- **{stat}** ({'test-box proximity, uses labels: explanatory' if stat == 'X' else 'transductive patch coverage, label-free'}): "
                     f"ρ test (16 subsets) {r_all:+.2f}, p {p_all:.4f}; within 8 random draws {r_rb:+.2f} (ViT-B)"
                     + (f", {r_rn:+.2f} (no DINOv3)" if stat == "X" else "")
                     + f"; ρ val {r_val:+.2f} → condition {2 if stat == 'X' else 3} {'**passes**' if ok else 'fails'}")
    lines += ["", "| family | subset | test mAP50-95 | X | val mAP50-95 | X val | T | T val |",
              "|---|---|---|---|---|---|---|---|"]
    for r in sorted(subrows, key=lambda r: (r["family"], -r["test"])):
        lines.append(f"| {r['family']} | {r['subset']} | {r['test']:.3f} | {r['X']:.4f} | " + " | ".join(
            f"{r[k]:.4f}" if k in r else "-" for k in ("val", "Xval", "T", "Tval")) + " |")
    lines += ["", f"**Verdict:** H16 {'holds' if all(c1.values()) else 'does not hold'} at the instance level; "
                  f"subset luck {'is' if res['X'] else 'is not'} explained by test coverage; the transductive selector "
                  f"H16b {'runs' if res['T'] else 'does not run'}."]
    open(os.path.join(OUT, "report.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
