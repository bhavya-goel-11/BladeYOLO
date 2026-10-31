"""H18-W scope check (research_log.md): does the G15-R prototype rescoring gain fall with the number of labelled boxes
of the class in the model's subset, pooled over WTBD-group (4 RF-DETR seed-0 models, from G15-R's report.json) and
WTBlade-group (4 models, recomputed here)? Criterion fixed before looking: Spearman rho(log n, fixed - original AP50-95)
< 0 with one-sided permutation P < 0.05.

  .venv/bin/python tools/h18w_scope.py  -> runs/analysis/proto_rescore/wtblade/scope.md
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import proto_rescore as pr  # noqa: E402


def counts(data, suf, nc):
    n = np.zeros(nc, int)
    for f in pr.subset_names(suf):
        lab = os.path.join(data, "train", "labels", os.path.splitext(f)[0] + ".txt")
        if os.path.exists(lab):
            for l in open(lab):
                if l.strip():
                    n[int(l.split()[0])] += 1
    return n


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    rows = []  # (dataset, model, class, n, delta)
    g15 = json.load(open(os.path.join(pr.OUT, "rfdetr", "report.json")))["test"]["per"]
    for m, v in g15.items():
        n = counts(pr.DATA, pr.RFDETR[m], pr.NC)
        rows += [("WTBD", m, pr.CLASSES[c], n[c], v["resc_pc"][c] - v["orig_pc"][c]) for c in range(pr.NC)]
    pr.use_wtblade()
    emb = pr.Embedder()
    bank = pr.train_bank(emb)
    ms, files, _ = pr.evaluate("test", pr.RFDETR, emb, bank)
    for m, (o, r) in ms.items():
        po, prs = pr.metrics(o, files)[2], pr.metrics(r, files)[2]
        n = counts(pr.DATA, pr.RFDETR[m], pr.NC)
        rows += [("WTBlade", m, pr.CLASSES[c], n[c], prs[c] - po[c]) for c in range(pr.NC)]
    rows = [r for r in rows if np.isfinite(r[4])]
    x, y = np.log1p([r[3] for r in rows]), np.array([r[4] for r in rows])
    rho = spearman(x, y)
    rng = np.random.default_rng(0)
    perm = np.array([spearman(x, rng.permutation(y)) for _ in range(10000)])
    p = float((perm <= rho).mean())
    lines = ["# H18-W scope check: G15-R gain vs labelled boxes of the class", "",
             f"{len(rows)} (model, class) points; Spearman rho(log n, fixed - original) = {rho:+.3f}, "
             f"one-sided permutation P = {p:.4f} -> **{'scope found' if rho < 0 and p < 0.05 else 'no scope'}**", ""]
    for ds in ("WTBD", "WTBlade"):
        sub = [r for r in rows if r[0] == ds]
        xs, ys = np.log1p([r[3] for r in sub]), np.array([r[4] for r in sub])
        lines.append(f"- {ds}: {len(sub)} points, rho {spearman(xs, ys):+.3f}, mean delta {ys.mean():+.4f}")
    lines += ["", "| n labelled boxes | points | mean delta AP50-95 | share > 0 |", "|---|---|---|---|"]
    for lo, hi in ((0, 30), (30, 60), (60, 120), (120, 250), (250, 10 ** 6)):
        sel = [r[4] for r in rows if lo <= r[3] < hi]
        if sel:
            lines.append(f"| {lo}–{hi if hi < 10 ** 6 else '∞'} | {len(sel)} | {np.mean(sel):+.4f} | "
                         f"{np.mean(np.array(sel) > 0):.2f} |")
    lines += ["", "| dataset | model | class | n | delta |", "|---|---|---|---|---|"]
    lines += [f"| {d} | {m} | {c} | {n} | {dl:+.4f} |" for d, m, c, n, dl in sorted(rows, key=lambda r: r[3])]
    open(os.path.join(pr.OUT, "scope.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
