"""Numbers for the paper that no earlier analysis stored (CPU, cached predictions): per-class test AP of every model on
the leak-free benchmarks, class-agnostic PR curves of the G28 arms, and the composition of the G28 pools.
  .venv/bin/python tools/paper_evidence.py   -> runs/analysis/paper/evidence.json
"""

import csv
import json
import os
import sys
from collections import Counter

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import bootstrap_ci as bc  # noqa: E402
import g27_score as g27  # noqa: E402
import g28_score as g28  # noqa: E402

OUT = os.path.join(ROOT, "runs", "analysis", "paper", "evidence.json")
R_GRID = np.linspace(0, 1, 101)


def per_class(imgs, preds, nc):
    files = sorted(imgs)
    m50, m, c50, c = bc.score(bc.per_image_matches(imgs, preds, nc), files, nc)
    return dict(mAP50=m50, mAP=m, ap50=[float(x) for x in c50], ap=[float(x) for x in c])


def pr_curve(imgs, preds):
    """Class-agnostic precision envelope at IoU 0.5, its max F1, and the boxes per image kept at that operating point."""
    m = bc.per_image_matches(imgs, preds, 1)
    files = sorted(imgs)
    tp = np.concatenate([m[f][(0, 0.5)][0] for f in files]).astype(float)
    conf = np.concatenate([m[f][(0, 0.5)][1] for f in files])
    n = sum(m[f][(0, 0.5)][2] for f in files)
    order = np.argsort(-conf)
    tp, conf = tp[order], conf[order]
    tc = np.cumsum(tp)
    rec, prec = tc / n, tc / np.arange(1, len(tp) + 1)
    env = np.maximum.accumulate(prec[::-1])[::-1]
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    b = int(np.argmax(f1))
    grid = [float(env[np.searchsorted(rec, r)]) if r <= rec[-1] else 0.0 for r in R_GRID]
    return dict(precision=grid, max_recall=float(rec[-1]), best_f1=float(f1[b]), f1_conf=float(conf[b]),
                f1_recall=float(rec[b]), f1_precision=float(prec[b]), boxes_per_image_at_f1=(b + 1) / len(files),
                boxes_per_image=len(tp) / len(files))


def main():
    out = dict(models={}, g28_pr={}, pools={})
    for name, cfg in g27.DATASETS.items():
        imgs = bc.load_yolo_split(cfg["data"], "test")
        files = sorted(imgs)
        names = yaml.safe_load(open(os.path.join(cfg["data"], "data.yaml")))["names"]
        names = list(names.values()) if isinstance(names, dict) else names
        d = out["models"][name] = dict(classes=names, runs={})
        for arm, run in cfg["runs"].items():
            for s, suf in ((0, ""), (1, "_s1")):
                p = g27.rf_preds(run + suf, cfg, "test", files)
                if p is not None:
                    d["runs"][f"RF-DETR-M {arm} seed {s}"] = per_class(imgs, p, cfg["nc"])
        for lab, run in cfg["others"].items():
            p = g27.cached_preds(run, files, imgs)
            if p is not None:
                d["runs"][lab] = per_class(imgs, p, cfg["nc"])
        print(name, list(d["runs"]), flush=True)
    for tname, t in g28.TARGETS.items():
        cfg = t["cfg"]
        imgs = bc.load_yolo_split(cfg["data"], "test")
        for v in imgs.values():
            v["gt"] = g28.agnostic(v["gt"])
        files = sorted(imgs)
        out["g28_pr"][tname] = {a: pr_curve(imgs, {f: g28.agnostic(x) for f, x in g27.rf_preds(t[a], cfg, "test", files).items()})
                                for a in g28.ARMS}
        print(tname, {a: round(v["best_f1"], 3) for a, v in out["g28_pr"][tname].items()}, flush=True)
    for pool in ("XWTBD", "XWTBL"):
        rows = list(csv.DictReader(open(os.path.join(ROOT, "tools", "splits", f"g27_pool_{pool}.csv"))))
        c = Counter((r["source"], r["status"]) for r in rows)
        out["pools"][pool] = {f"{s}|{st}": n for (s, st), n in sorted(c.items())}
        out["pools"][pool + "_boxes_kept"] = sum(int(r["boxes"]) for r in rows if r["status"] == "kept")
        print(pool, out["pools"][pool], flush=True)
    out["r_grid"] = [float(x) for x in R_GRID]
    json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
