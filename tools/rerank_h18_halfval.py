"""Robustness check for H18: fit the calibrator on each half of val (the 2 scene folds of the val cross-fit) and on
the whole val, apply to test; report test family-mean all-class and rare AP vs the fixed rule (point estimates)."""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
import bootstrap_ci as bc, proto_rescore as pr, rerank_h18 as h

bank = np.load(os.path.join(pr.OUT, "train_bank.npz"), allow_pickle=True)["bank"].item()
banks = h.model_banks(bank)
vimgs, _, V = h.split_data("val", banks)
timgs, tpreds, T = h.split_data("test", banks)
groups = pr.scenes("val", sorted(vimgs))
fold = {f: gi % 2 for gi, g in enumerate(sorted(groups, key=lambda g: min(g))) for f in g}
Xv = np.concatenate([V[m][1] for m in pr.RFDETR]); tv = np.concatenate([V[m][2] for m in pr.RFDETR])
fv = np.concatenate([[fold[f] for f in V[m][0]] for m in pr.RFDETR])
files = sorted(timgs)
def fam(conf_fn):
    r = [pr.metrics(bc.per_image_matches(timgs, h.with_conf(tpreds[m], T[m][0], conf_fn(m)), 6), files) for m in pr.RFDETR]
    return np.mean([x[0] for x in r]), np.mean([x[1] for x in r])
base = fam(lambda m: T[m][3])
print(f"fixed rule: all {base[0]:.4f} rare {base[1]:.4f}")
for name, sel in (("val fold 0", fv == 0), ("val fold 1", fv == 1), ("all val", np.ones(len(fv), bool))):
    f = h.fit(Xv[sel], tv[sel])
    a = fam(lambda m: f(T[m][1]))
    print(f"H18 fitted on {name} ({sel.sum()} rows): all {a[0]:.4f} ({a[0]-base[0]:+.4f} vs fixed) rare {a[1]:.4f}")
