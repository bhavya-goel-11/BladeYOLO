"""H20 (research_log.md): does a subset's box-annotation convention, measured against the held-out labels, predict
its score? 16 H7 subsets of WTBD-group, test/val mAP50-95 from runs/analysis/subset_properties/table.csv.
  G1 mean over classes of W1(log box area @640) between the subset's boxes and the reference's   (primary)
  G2 same for log aspect ratio;  G3 same for c-boxes per image containing c
  .venv/bin/python tools/h20_convention.py -> runs/analysis/h20/report.md
"""
import csv, os, sys
import numpy as np
from scipy.stats import wasserstein_distance
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import bootstrap_ci as bc  # noqa: E402
from subset_properties import spearman_perm  # noqa: E402
DATA = os.path.join(ROOT, "external", "WTBD-group")
TABLE = os.path.join(ROOT, "runs", "analysis", "subset_properties", "table.csv")
OUT = os.path.join(ROOT, "runs", "analysis", "h20")
NC = 6


def box_stats(imgs):
    """per class: (log area, log aspect, per-image counts)"""
    A, R, K = [[] for _ in range(NC)], [[] for _ in range(NC)], [[] for _ in range(NC)]
    for im in imgs.values():
        g = im["gt"]
        for c in range(NC):
            b = g[g[:, 0] == c]
            if len(b):
                w, h = b[:, 3] - b[:, 1], b[:, 4] - b[:, 2]
                A[c] += list(np.log(np.maximum(w * h, 1))); R[c] += list(np.log(np.maximum(w, 1) / np.maximum(h, 1)))
                K[c].append(len(b))
    return A, R, K


def dist(S, T):
    return [np.mean([wasserstein_distance(s[c], t[c]) for c in range(NC) if len(s[c]) and len(t[c])]) for s, t in zip(S, T)]


def main():
    os.makedirs(OUT, exist_ok=True)
    tr = bc.load_yolo_split(DATA, "train")
    ref = {s: box_stats(bc.load_yolo_split(DATA, s)) for s in ("test", "val")}
    rows = list(csv.DictReader(open(TABLE)))
    G = {}
    for r in rows:
        suf = "" if r["subset"] == "class-matched" else "_" + r["subset"]
        keep = set(open(os.path.join(ROOT, "tools", "splits", f"wtbd_group_train_f25{suf}.txt")).read().split())
        S = box_stats({f: im for f, im in tr.items() if os.path.basename(f) in keep})
        G[r["subset"]] = {k: dist(S, ref[k]) for k in ref}
    test = np.array([float(r["test"]) for r in rows]); val = np.array([float(r["val"]) for r in rows])
    rnd = np.array([r["subset"].startswith("rand") for r in rows])
    lines = ["# H20: annotation-convention distance vs subset score (16 H7 subsets, WTBD-group)", "",
             "| statistic | reference | ρ vs test | perm p | ρ vs val | ρ vs test, 8 random |", "|---|---|---|---|---|---|"]
    res = {}
    for gi, gname in enumerate(("G1 log area", "G2 log aspect", "G3 boxes/image")):
        for ref_s in ("test", "val"):
            x = np.array([G[r["subset"]][ref_s][gi] for r in rows])
            rt, pt = spearman_perm(x, test); rv, _ = spearman_perm(x, val, n=1); rr, _ = spearman_perm(x[rnd], test[rnd], n=1)
            res[(gi, ref_s)] = (rt, pt, rv, rr)
            lines.append(f"| {gname} | {ref_s} | {rt:+.2f} | {pt:.4f} | {rv:+.2f} | {rr:+.2f} |")
    rt, pt, _, rr = res[(0, "test")]; rv_val = res[(0, "val")][2]; rt_val = res[(0, "val")][0]
    ok = rt <= -0.6 and pt < 0.05 and rv_val <= -0.4 and rr <= -0.5 and rt_val <= -0.4
    lines += ["", f"(a) ρ(G1 vs test, test) {rt:+.2f}, p {pt:.4f}; (b) ρ(G1 vs val, val) {rv_val:+.2f}; "
                  f"(c) random draws {rr:+.2f}; (d) ρ(G1 vs val, test) {rt_val:+.2f}", "",
              f"**H20 {'passes' if ok else 'fails'}.**"]
    lines += ["", "| subset | test | val | G1 vs test | G1 vs val | G3 vs test |", "|---|---|---|---|---|---|"]
    lines += [f"| {r['subset']} | {float(r['test']):.3f} | {float(r['val']):.3f} | {G[r['subset']]['test'][0]:.3f} | "
              f"{G[r['subset']]['val'][0]:.3f} | {G[r['subset']]['test'][2]:.3f} |" for r in rows]
    open(os.path.join(OUT, "report.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
