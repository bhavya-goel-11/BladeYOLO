"""H19-W (research_log.md): does G15-R prototype rescoring help exactly the classes with fewer than 30 labelled boxes,
within one dataset? 4 RF-DETR Medium models on 8% WTBlade-group scene subsets (tools/splits/wtblade_group_train_f08*),
fixed rule frozen from WTBD (alpha 0.5, k 3, tau 0.05; bank = the model's own subset). A point is (model, class) with
n = labelled boxes of the class in that subset; low = 1 <= n < 30, high = n >= 30, n = 0 reported only.

Gates (all must hold): mean delta over low points > 0 with P(>0) >= 0.95; mean(low) - mean(high) > 0 with P >= 0.95;
>= 70% of low points with delta > 0. delta = fixed - original test AP50-95; paired scene bootstrap, 2,000 replicates.
Reported: the count-gated rule (rescore only classes with n < 30) vs original, all-class mAP50-95.

  .venv/bin/python tools/h19w_lowcount.py            -> runs/analysis/proto_rescore/wtblade/h19w_f08.md
  .venv/bin/python tools/h19w_lowcount.py --frac f25 (dry run on the H18-W models; same code path)
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import proto_rescore as pr  # noqa: E402
from h18w_scope import counts, spearman  # noqa: E402

LOW, N_BOOT, SHARE = 30, 2000, 0.7


def summarise(n, po, pr_):
    """n, po, pr_: (models, classes). Returns gate statistics and the count-gated rule's mAP difference."""
    d = pr_ - po
    low, high = (n >= 1) & (n < LOW), n >= LOW
    gated = np.where(low, pr_, po)
    return dict(low=float(np.nanmean(d[low])), high=float(np.nanmean(d[high])),
                contrast=float(np.nanmean(d[low]) - np.nanmean(d[high])),
                share=float(np.mean(d[low] > 0)),
                gated=float(np.mean(np.nanmean(gated, 1) - np.nanmean(po, 1))),
                fixed=float(np.mean(np.nanmean(pr_, 1) - np.nanmean(po, 1))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frac", default="f08")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    args = ap.parse_args()
    pr.use_wtblade(args.frac)
    emb = pr.Embedder()
    bank = pr.train_bank(emb)
    ms, files, _ = pr.evaluate("test", pr.RFDETR, emb, bank)
    models = list(ms)
    n = np.array([counts(pr.DATA, pr.RFDETR[m], pr.NC) for m in models])
    po = np.array([pr.metrics(ms[m][0], files)[2] for m in models])
    prs = np.array([pr.metrics(ms[m][1], files)[2] for m in models])
    pt = summarise(n, po, prs)
    groups = pr.scenes("test", files)
    rng = np.random.default_rng(0)
    bs = []
    for b in range(args.boot):
        d = [f for gi in rng.integers(0, len(groups), len(groups)) for f in groups[gi]]
        bs.append(summarise(n, np.array([pr.metrics(ms[m][0], d)[2] for m in models]),
                            np.array([pr.metrics(ms[m][1], d)[2] for m in models])))
        if (b + 1) % 200 == 0:
            print(f"  boot {b + 1}/{args.boot}", flush=True)
    ci = {k: (float(np.percentile([x[k] for x in bs], 2.5)), float(np.percentile([x[k] for x in bs], 97.5)),
              float(np.mean([x[k] > 0 for x in bs]))) for k in pt if k != "share"}
    g1 = pt["low"] > 0 and ci["low"][2] >= 0.95
    g2 = pt["contrast"] > 0 and ci["contrast"][2] >= 0.95
    g3 = pt["share"] >= SHARE
    d = prs - po
    sel = n >= 1
    lines = [f"# H19-W ({args.frac}): G15-R gain for classes below vs above {LOW} labelled boxes", "",
             f"{len(models)} models, {int(((n >= 1) & (n < LOW)).sum())} low points, {int((n >= LOW).sum())} high, "
             f"{int((n == 0).sum())} with n = 0; {len(groups)} test scenes, {args.boot} paired bootstrap replicates.", "",
             "| statistic | point | 95% CI | P(>0) |", "|---|---|---|---|"]
    for k, name in [("low", "mean Δ, low points"), ("high", "mean Δ, high points"), ("contrast", "low − high"),
                    ("fixed", "fixed rule − original, all-class mAP"), ("gated", "count-gated rule − original, all-class mAP")]:
        lines.append(f"| {name} | {pt[k]:+.4f} | [{ci[k][0]:+.4f}, {ci[k][1]:+.4f}] | {ci[k][2]:.3f} |")
    lines += ["", f"- share of low points with Δ > 0: {pt['share']:.2f}",
              f"- Spearman ρ(log n, Δ) over n ≥ 1: {spearman(np.log(n[sel]), d[sel]):+.3f}",
              f"- Gate 1 (low > 0, P ≥ 0.95): {'pass' if g1 else 'FAIL'}; gate 2 (low − high > 0, P ≥ 0.95): "
              f"{'pass' if g2 else 'FAIL'}; gate 3 (share ≥ {SHARE}): {'pass' if g3 else 'FAIL'}",
              f"- **{'CONFIRMED' if g1 and g2 and g3 else 'NOT CONFIRMED'}**", "",
              "| model | class | n | original | fixed | Δ |", "|---|---|---|---|---|---|"]
    rows = sorted(((n[i, c], models[i], c) for i in range(len(models)) for c in range(pr.NC)))
    lines += [f"| {m} | {pr.CLASSES[c]} | {k} | {po[models.index(m), c]:.4f} | {prs[models.index(m), c]:.4f} | "
              f"{d[models.index(m), c]:+.4f} |" for k, m, c in rows]
    out = os.path.join(pr.OUT, f"h19w_{args.frac}")
    open(out + ".md", "w").write("\n".join(lines) + "\n")
    json.dump(dict(point=pt, ci=ci, n=n.tolist(), orig=po.tolist(), fixed=prs.tolist(), models=models),
              open(out + ".json", "w"), indent=1)
    print("\n".join(lines[:16]))


if __name__ == "__main__":
    main()
