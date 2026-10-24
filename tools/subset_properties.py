"""H7 (Phase 28): does a label-free property of a labelled subset predict how well a detector trains on it?

The 16 distinct 189-image subsets of WTBD-group behind the H6 runs, each scored by its runs' test mAP50-95 (bootstrap
evaluator; mean when two runs share a subset) and best-epoch val mAP50-95. Properties P1-P9 and the decision rule are
pre-registered in research_log.md (H7); label-based statistics are reported only to explain, never to select.

  .venv/bin/python tools/subset_properties.py  -> runs/analysis/subset_properties/{table.csv, report.md}
"""

import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import bootstrap_ci as bc  # noqa: E402
import select_scenes as ss  # noqa: E402

DATA = os.path.join(ROOT, "external", "WTBD-group")
RUNS = os.path.join(ROOT, "runs", "detect", "BladeYOLO_WindSurface")
SPLITS = os.path.join(ROOT, "tools", "splits")
SEL = os.path.join(ROOT, "runs", "analysis", "select_scenes")
PREDS = os.path.join(ROOT, "runs", "analysis", "bootstrap", "seeds_f25", "preds")
OUT = os.path.join(ROOT, "runs", "analysis", "subset_properties")
NC = 6
SUBSETS = {  # subset file suffix -> runs trained on it
    "": ["bladeyolo2b_wtbd_group_f25_s0", "bladeyolo2b_wtbd_group_f25_s1"],
    "_cover": ["bladeyolo2b_cover_wtbd_group_f25_s0", "bladeyolo2b_cover_wtbd_group_f25_s1"],
    "_cover_k1": ["bladeyolo2b_coverk1_wtbd_group_f25_s0"],
    "_cover_k2": ["bladeyolo2b_coverk2_wtbd_group_f25_s0"],
    **{f"_rand{k}": [f"bladeyolo2b_rand{k}_wtbd_group_f25_s0"] for k in range(1, 9)},
    **{f"_freesel{s}_s{k}": [f"bladeyolo2b_freesel{s}s{k}_wtbd_group_f25"] for s in (224, 640) for k in (0, 1)},
}
LABEL_FREE = ["P1 scenes", "P2 coverage K1024", "P3 coverage K64", "P4 tail coverage", "P5 image diversity",
              "P6 MMD2 to pool", "P7 typicality", "P8 FreeSel pattern coverage", "P9 atypical-patch mass"]
LABEL_BASED = ["boxes", "min class boxes", "thunderstrike boxes", "crack boxes", "classes >= 10 boxes", "class entropy"]


def centroids(X, k, seed):
    from sklearn.cluster import KMeans

    km = KMeans(k, n_init=1, random_state=seed).fit(X)
    C = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
    return C, np.bincount(km.labels_, minlength=k)


def mmd2(A, B, gamma=4.0, n=4000, rng=None):
    """Biased MMD^2 with k(a, b) = exp(-gamma * (1 - cos)), on random samples of n rows."""
    A, B = A[rng.choice(len(A), min(n, len(A)), replace=False)], B[rng.choice(len(B), min(n, len(B)), replace=False)]
    k = lambda X, Y: np.exp(-gamma * (1 - X @ Y.T)).mean()  # noqa: E731
    return k(A, A) + k(B, B) - 2 * k(A, B)


def spearman_perm(x, y, n=20000, seed=0):
    from scipy.stats import rankdata

    rx, ry = rankdata(x), rankdata(y)
    r = np.corrcoef(rx, ry)[0, 1]
    rng = np.random.default_rng(seed)
    null = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(n)])
    return r, float((np.abs(null) >= abs(r) - 1e-12).mean())


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(os.listdir(os.path.join(DATA, "train", "images")))
    idx = {f: i for i, f in enumerate(files)}
    paths = [os.path.join(DATA, "train", "images", f) for f in files]
    F = np.load(os.path.join(SEL, "wtbd_group_train_f25_patches_b.npy")).astype(np.float32)  # (n, 128, D)
    X = F.reshape(-1, F.shape[-1])
    M = F.mean(1)
    M /= np.linalg.norm(M, axis=1, keepdims=True)
    groups = ss.scenes(paths, os.path.join(ROOT, ".dataset_cache"), 4)
    scene_of = np.empty(len(files), int)
    for g, mem in enumerate(groups):
        scene_of[mem] = g
    print(f"{len(files)} images, {len(groups)} scenes", flush=True)
    C1024 = [centroids(X, 1024, s) for s in (0, 1, 2)]
    C64, _ = centroids(X, 64, 0)
    tail = np.argsort(C1024[0][1])[:256]
    knn_sim = np.sort(M @ M.T, 1)[:, -11:-1].mean(1)  # 10 nearest other images
    patt = np.load(os.path.join(SEL, "freesel_patterns_224_s0.npy"))  # (n, 5, 384)
    pflat = patt.reshape(-1, patt.shape[-1])
    labels = []
    for f in files:
        lab = os.path.join(DATA, "train", "labels", os.path.splitext(f)[0] + ".txt")
        labels.append([int(l.split()[0]) for l in open(lab) if l.strip()] if os.path.exists(lab) else [])
    rng = np.random.default_rng(0)

    te = bc.load_yolo_split(DATA, "test")
    tfiles = sorted(te)
    rows = []
    for suf, runs in SUBSETS.items():
        sel = np.array(sorted(idx[n.strip()] for n in open(os.path.join(SPLITS, f"wtbd_group_train_f25{suf}.txt")) if n.strip()))
        Xs = F[sel].reshape(-1, F.shape[-1])
        cov = lambda C: (Xs @ C.T).max(0)  # noqa: E731
        c1024 = [cov(C) for C, _ in C1024]
        sims = M[sel] @ M[sel].T
        cnt = np.bincount([c for i in sel for c in labels[i]], minlength=NC)
        p = cnt / cnt.sum()
        props = {
            "P1 scenes": len(set(scene_of[sel])),
            "P2 coverage K1024": float(np.mean([c.mean() for c in c1024])),
            "P3 coverage K64": float(cov(C64).mean()),
            "P4 tail coverage": float(c1024[0][tail].mean()),
            "P5 image diversity": float((1 - sims[np.triu_indices(len(sel), 1)]).mean()),
            "P6 MMD2 to pool": float(mmd2(Xs, X, rng=rng)),
            "P7 typicality": float(knn_sim[sel].mean()),
            "P8 FreeSel pattern coverage": float((pflat @ patt[sel].reshape(-1, patt.shape[-1]).T).max(1).mean()),
            "P9 atypical-patch mass": float((1 - (Xs @ C1024[0][0].T).max(1)).mean()),
            "boxes": int(cnt.sum()), "min class boxes": int(cnt.min()), "thunderstrike boxes": int(cnt[3]),
            "crack boxes": int(cnt[4]), "classes >= 10 boxes": int((cnt >= 10).sum()),
            "class entropy": float(-(p[p > 0] * np.log(p[p > 0])).sum()),
        }
        test = []
        for r in runs:
            ms = bc.per_image_matches(te, bc.predict(os.path.join(RUNS, r), te, PREDS), NC)
            test.append(bc.score(ms, tfiles, NC)[1])
        val = []
        for r in runs:
            res = list(csv.DictReader(open(os.path.join(RUNS, r, "results.csv"))))
            k = [c for c in res[0] if "mAP50-95" in c][0]
            val.append(max(float(x[k]) for x in res))
        rows.append(dict(subset="class-matched" if suf == "" else suf[1:], images=len(sel), test=float(np.mean(test)),
                         val=float(np.mean(val)), **props))
        print(f"{rows[-1]['subset']}: test {rows[-1]['test']:.3f} val {rows[-1]['val']:.3f}", flush=True)
    with open(os.path.join(OUT, "table.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    test, val = np.array([r["test"] for r in rows]), np.array([r["val"] for r in rows])
    rnd = np.array([r["subset"].startswith("rand") for r in rows])
    lines = ["# H7: label-free subset properties as predictors of subset quality", "",
             f"{len(rows)} distinct subsets of WTBD-group train ({len(files)} images, {len(groups)} scenes). "
             "Spearman ρ; permutation p (20,000 permutations, two-sided). Rule (pre-registered): |ρ_test| ≥ 0.6 with "
             "p < 0.05/9, ρ_val same sign with |ρ| ≥ 0.4, ρ within the 8 random draws same sign with |ρ| ≥ 0.5.", "",
             "| property | ρ test | p | ρ val | ρ test (8 random) | candidate |", "|---|---|---|---|---|---|"]
    for name in LABEL_FREE + LABEL_BASED:
        x = np.array([r[name] for r in rows], float)
        rt, pt = spearman_perm(x, test)
        rv, _ = spearman_perm(x, val, n=1)
        rr, _ = spearman_perm(x[rnd], test[rnd], n=1)
        ok = (abs(rt) >= 0.6 and pt < 0.05 / 9 and np.sign(rv) == np.sign(rt) and abs(rv) >= 0.4
              and np.sign(rr) == np.sign(rt) and abs(rr) >= 0.5)
        tag = ("**yes**" if ok else "no") if name in LABEL_FREE else "(label-based)"
        lines.append(f"| {name} | {rt:+.2f} | {pt:.4f} | {rv:+.2f} | {rr:+.2f} | {tag} |")
    lines += ["", "| subset | test | val | " + " | ".join(n.split()[0] for n in LABEL_FREE) + " |",
              "|---|---|---|" + "---|" * len(LABEL_FREE)]
    for r in sorted(rows, key=lambda r: -r["test"]):
        lines.append(f"| {r['subset']} | {r['test']:.3f} | {r['val']:.3f} | "
                     + " | ".join(f"{r[n]:.4g}" for n in LABEL_FREE) + " |")
    open(os.path.join(OUT, "report.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
