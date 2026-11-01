"""G20 (research_log.md): query-consensus rescoring, a label-free ranking signal for RF-DETR. Pre-registered before
computing (1 Nov).

Error anatomy showed that at 25% labels RF-DETR proposes the right box for 83-92% of test defects but scores 28-37% of
them below 0.1. Idea: a DETR's queries pile up on real defects (several queries, mostly suppressed to low scores by
one-to-one matching), while background false positives are isolated. So the score mass of the *other* queries at the
same place is evidence of a real object that needs no labels, no exemplars and no class prototypes:

    s'_p = s_p + lam * sum_{q != p, IoU(p, q) >= 0.5} s_q          (any class q; boxes unchanged)

lam is chosen on each dataset's val split from {0.25, 0.5, 1, 2} (family mean mAP50-95 of the 4 RF-DETR 25% models),
then applied once to test. G20 passes if on BOTH WTBD-group and WTBlade-group test: rescored - original family mean
mAP50-95 > 0 with paired scene bootstrap P(>0) >= 0.95 (2,000 replicates) and rescored > original on >= 3 of 4 models.
Reported: WTBlade 8% models with WTBlade's lam; per-class; the val curve.

  .venv/bin/python tools/g20_consensus.py  -> runs/analysis/g20/report.md
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import bootstrap_ci as bc  # noqa: E402
import detection_metrics as dm  # noqa: E402
import proto_rescore as pr  # noqa: E402

LAMS = (0.25, 0.5, 1.0, 2.0)
N_BOOT = 2000
OUT = os.path.join(ROOT, "runs", "analysis", "g20")


def consensus(p, lam):
    if len(p) < 2:
        return p
    iou = dm.iou_matrix(p[:, :4], p[:, :4])
    np.fill_diagonal(iou, 0)
    q = p.copy()
    q[:, 4] = p[:, 4] + lam * ((iou >= 0.5) * p[None, :, 4]).sum(1)
    return q


def setup(split):
    """imgs, files, {model: preds} for the current pr configuration."""
    imgs = bc.load_yolo_split(pr.DATA, split)
    cache = pr.PREDS[split] if split == "test" else pr.RFDETR_VAL_PREDS
    preds = {m: bc.predict(os.path.join(pr.RUNS, m), imgs, cache) for m in pr.RFDETR}
    return imgs, sorted(imgs), preds


def maps(imgs, files, preds, lam):
    out = {}
    for m, p in preds.items():
        new = {f: consensus(v, lam) for f, v in p.items()} if lam else p
        out[m] = bc.per_image_matches(imgs, new, pr.NC)
    return out


def boot(ms_o, ms_r, files, groups, seed=0):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(N_BOOT):
        draw = [f for gi in rng.integers(0, len(groups), len(groups)) for f in groups[gi]]
        d.append(np.mean([pr.metrics(ms_r[m], draw)[0] - pr.metrics(ms_o[m], draw)[0] for m in ms_o]))
    return np.array(d)


def dataset(name, lines, lam_fixed=None):
    if lam_fixed is None:
        imgs, files, preds = setup("val")
        base = maps(imgs, files, preds, 0)
        b0 = np.mean([pr.metrics(base[m], files)[0] for m in base])
        curve = {lam: np.mean([pr.metrics(v, files)[0] for v in maps(imgs, files, preds, lam).values()]) for lam in LAMS}
        lam = max(curve, key=curve.get)
        lines += [f"## {name}", "", f"val family mAP50-95: original {b0:.4f}; " +
                  ", ".join(f"λ {k}: {v:.4f}" for k, v in curve.items()) + f" → λ = {lam}", ""]
    else:
        lam = lam_fixed
        lines += [f"## {name} (λ = {lam} from WTBlade 25% val)", ""]
    imgs, files, preds = setup("test")
    ms_o, ms_r = maps(imgs, files, preds, 0), maps(imgs, files, preds, lam)
    per = {m: (pr.metrics(ms_o[m], files), pr.metrics(ms_r[m], files)) for m in ms_o}
    d = boot(ms_o, ms_r, files, pr.scenes("test", files))
    pt = np.mean([r[0] - o[0] for o, r in per.values()])
    wins = sum(r[0] > o[0] for o, r in per.values())
    lines += ["| model | original | consensus | Δ |", "|---|---|---|---|"]
    lines += [f"| {m} | {o[0]:.4f} | {r[0]:.4f} | {r[0] - o[0]:+.4f} |" for m, (o, r) in per.items()]
    po, prc = np.mean([o[2] for o, _ in per.values()], 0), np.mean([r[2] for _, r in per.values()], 0)
    lines += ["", f"family Δ {pt:+.4f} [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}], "
              f"P(>0) {np.mean(d > 0):.3f}; better on {wins}/{len(per)} models", "",
              "per class (family mean): " + ", ".join(f"{c} {a:.3f}→{b:.3f}" for c, a, b in zip(pr.CLASSES, po, prc)), ""]
    return lam, pt > 0 and np.mean(d > 0) >= 0.95 and wins >= 3


def main():
    os.makedirs(OUT, exist_ok=True)
    lines = ["# G20: query-consensus rescoring on RF-DETR", ""]
    _, ok_wtbd = dataset("WTBD-group, RF-DETR-M 25%", lines)
    pr.use_wtblade("f25")
    lam, ok_wtb = dataset("WTBlade-group, RF-DETR-M 25%", lines)
    pr.use_wtblade("f08")
    dataset("WTBlade-group, RF-DETR-M 8% (reported)", lines, lam_fixed=lam)
    lines += [f"**G20 {'passes' if ok_wtbd and ok_wtb else 'fails'}** (WTBD {'✓' if ok_wtbd else '✗'}, "
              f"WTBlade {'✓' if ok_wtb else '✗'})"]
    open(os.path.join(OUT, "report.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
