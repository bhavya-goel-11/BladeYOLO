"""H18 (research_log.md): a learned, coverage-aware re-ranker for RF-DETR's detections.

Per detection, features from the detector and from G15's prototype coverage signal (DINOv3 ViT-B at 640 px, the model's
own subset bank): z1 logit(conf), z2 log p(class) from the prototype softmax, z3 top-3 mean cosine to the class's
labelled boxes, z4 log(1 + labelled boxes of the class), z1*z3, z2*z4, a missing-class indicator, one-hot class.
Target: the fraction of IoU thresholds 0.50:0.95 at which the detection is a true positive (greedy matching in the
original confidence order). One logistic calibrator (fractional targets, L2, C = 1) is fitted on the 4 models' val
predictions and applied to their test predictions. Arms on test: original, fixed rule (G15-R), H18.

  .venv/bin/python tools/rerank_h18.py  -> runs/analysis/rerank_h18/{report.md, report.json}
  .venv/bin/python tools/rerank_h18.py replicate  -> runs/analysis/rerank_h18/replicate/ (H18-R: the calibrator fitted
      on the seed-0 models' val, applied unchanged to the seed-1 models' test and val)
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import bootstrap_ci as bc  # noqa: E402
import detection_metrics as dm  # noqa: E402
import proto_rescore as pr  # noqa: E402

OUT = os.path.join(ROOT, "runs", "analysis", "rerank_h18")
PREDS = {"test": pr.PREDS["test"], "val": pr.RFDETR_VAL_PREDS}
NC, N_BOOT = 6, 1000
EPS = 1e-6


# ---- pure pieces (tests/test_rerank_h18.py) ----

def targets(gt, pred, thrs=dm.IOU_THRS):
    """(n,) fraction of IoU thresholds at which each prediction (x1 y1 x2 y2 conf cls) is a true positive, matching
    greedily per class in descending confidence; gt is (m, 5) [cls, x1, y1, x2, y2]."""
    t = np.zeros(len(pred))
    for c in np.unique(pred[:, 5]).astype(int) if len(pred) else []:
        pi = np.where(pred[:, 5] == c)[0]
        pi = pi[np.argsort(-pred[pi, 4], kind="stable")]
        gc = gt[gt[:, 0] == c]
        if not len(gc):
            continue
        iou = dm.iou_matrix(pred[pi, :4], gc[:, 1:])
        for th in thrs:
            used = np.zeros(len(gc), bool)
            for r, i in enumerate(pi):
                cand = np.where((iou[r] >= th) & ~used)[0]
                if len(cand):
                    used[cand[np.argmax(iou[r, cand])]] = True
                    t[i] += 1
    return t / len(thrs)


def features(pred, emb, bank, bank_cls, nc=NC, k=pr.K, tau=pr.TAU):
    """(n, 7 + nc) feature matrix for predictions with box embeddings emb, against one model's subset bank."""
    n = len(pred)
    cls = pred[:, 5].astype(int)
    p = pr.class_probs(emb, bank, bank_cls, nc, k, tau)[np.arange(n), cls] if n else np.zeros(0)
    sim = emb @ bank.T if n else np.zeros((0, len(bank)))
    z3 = np.zeros(n)
    for c in np.unique(cls):
        sc = sim[cls == c][:, bank_cls == c]
        if sc.shape[1]:
            z3[cls == c] = np.sort(sc, 1)[:, -min(k, sc.shape[1]):].mean(1)
    miss = ~np.isfinite(p)
    z1 = np.log(np.clip(pred[:, 4], EPS, 1 - EPS) / (1 - np.clip(pred[:, 4], EPS, 1 - EPS)))
    z2 = np.where(miss, 0.0, np.log(np.clip(np.nan_to_num(p), EPS, 1)))
    z4 = np.log1p(np.bincount(bank_cls, minlength=nc))[cls]
    return np.c_[z1, z2, z3, z4, z1 * z3, z2 * z4, miss, np.eye(nc)[cls]]


def fit(X, t, C=1.0):
    """Logistic regression on fractional targets (each row twice, weighted t and 1 - t) with standardised features."""
    from sklearn.linear_model import LogisticRegression

    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    m = LogisticRegression(C=C, max_iter=2000).fit(np.r_[Z, Z], np.r_[np.ones(len(Z)), np.zeros(len(Z))],
                                                   sample_weight=np.r_[t, 1 - t])
    return lambda X_: m.predict_proba((X_ - mu) / sd)[:, 1]


# ---- data ----

RFDETR_S1 = {m.replace("rfdetrm_", "rfdetrm_s1_"): suf for m, suf in pr.RFDETR.items()}


def model_banks(bank, models=pr.RFDETR):
    out = {}
    for m, suf in models.items():
        names = [n.strip() for n in open(os.path.join(pr.SPLITS, f"wtbd_group_train_f25{suf}.txt")) if n.strip()]
        sel = [bank[n] for n in names if n in bank]
        out[m] = (np.concatenate([e for _, e in sel]), np.concatenate([g[:, 0] for g, _ in sel]).astype(int))
    return out


def split_data(split, banks):
    """imgs, preds {model: {file: pred}}, per model: (files per row, X, t, fixed-rule conf); models = banks' keys."""
    imgs = bc.load_yolo_split(pr.DATA, split)
    preds = {m: bc.predict(os.path.join(pr.RUNS, m), imgs, PREDS[split]) for m in banks}
    rows = {m: ([], [], [], []) for m in banks}
    for i, f in enumerate(sorted(imgs)):
        fmap = np.load(os.path.join(pr.OUT, "maps", split, os.path.splitext(os.path.basename(f))[0] + ".npy"))
        fmap = fmap.astype(np.float32)
        for m in banks:
            p = preds[m].get(f, np.zeros((0, 6)))
            if not len(p):
                continue
            e = pr.box_embed(fmap, p[:, :4])
            r = rows[m]
            r[0].extend([f] * len(p))
            r[1].append(features(p, e, *banks[m]))
            r[2].append(targets(imgs[f]["gt"], p))
            r[3].append(pr.rescore(p, e, *banks[m])[:, 4])
        if (i + 1) % 40 == 0:
            print(f"  {split} {i + 1}/{len(imgs)}", flush=True)
    return imgs, preds, {m: (np.array(r[0]), np.concatenate(r[1]), np.concatenate(r[2]), np.concatenate(r[3]))
                         for m, r in rows.items()}


def with_conf(preds, files, conf):
    """Copy of {file: pred} with confidences replaced row by row (rows in sorted-file, original order)."""
    out, j = {}, 0
    for f in sorted(preds):
        p = preds[f]
        if len(p):
            q = p.copy()
            q[:, 4] = conf[j:j + len(p)]
            assert (files[j:j + len(p)] == f).all()
            j += len(p)
            out[f] = q
        else:
            out[f] = p
    return out


def compare(arms, imgs, split):
    """arms {arm: {model: preds}} -> point metrics per arm and model, bootstrap of family-mean differences."""
    files = sorted(imgs)
    ms = {a: {m: bc.per_image_matches(imgs, p, NC) for m, p in v.items()} for a, v in arms.items()}
    point = {a: {m: pr.metrics(x, files) for m, x in v.items()} for a, v in ms.items()}
    groups = pr.scenes(split, files)
    rng = np.random.default_rng(0)
    boot = {a: [] for a in ms}
    for _ in range(N_BOOT):
        d = [f for g in rng.integers(0, len(groups), len(groups)) for f in groups[g]]
        for a in ms:
            boot[a].append(np.mean([np.r_[r[:2], r[2]] for r in (pr.metrics(x, d) for x in ms[a].values())], 0))
    return point, {a: np.array(b) for a, b in boot.items()}


def summarise(point, boot, models):
    """Family means per arm, and the H18-R quantities."""
    fam = {a: np.mean([np.r_[v[:2], v[2]] for v in point[a].values()], 0) for a in point}
    d_fix = boot["H18"][:, 0] - boot["fixed"][:, 0]
    d_org = boot["H18"][:, 0] - boot["original"][:, 0]
    wins = sum(point["H18"][m][0] > point["fixed"][m][0] for m in models)
    return fam, d_fix, d_org, wins


def main_replicate():
    """H18-R (research_log.md)."""
    out = os.path.join(OUT, "replicate")
    os.makedirs(out, exist_ok=True)
    bank = np.load(os.path.join(pr.OUT, "train_bank.npz"), allow_pickle=True)["bank"].item()
    _, _, V0 = split_data("val", model_banks(bank))
    f_all = fit(np.concatenate([V0[m][1] for m in pr.RFDETR]), np.concatenate([V0[m][2] for m in pr.RFDETR]))
    del V0
    b1 = model_banks(bank, RFDETR_S1)
    ci = lambda x: f"[{np.percentile(x, 2.5):+.4f}, {np.percentile(x, 97.5):+.4f}]"  # noqa: E731
    lines = ["# H18-R: frozen seed-0 calibrator on the seed-1 RF-DETR models (research_log.md)", ""]
    for split in ("test", "val"):
        imgs, preds, D = split_data(split, b1)
        arms = {"original": preds,
                "fixed": {m: with_conf(preds[m], D[m][0], D[m][3]) for m in b1},
                "H18": {m: with_conf(preds[m], D[m][0], f_all(D[m][1])) for m in b1}}
        point, boot = compare(arms, imgs, split)
        fam, d_fix, d_org, wins = summarise(point, boot, b1)
        lines += [f"## {split}", "", "| model | original | fixed rule | H18 |", "|---|---|---|---|"]
        lines += [f"| {m} | " + " | ".join(f"{point[a][m][0]:.3f}" for a in ("original", "fixed", "H18")) + " |"
                  for m in b1]
        lines += ["", "| family mean | mAP50-95 | rare AP50-95 | " + " | ".join(pr.CLASSES) + " |",
                  "|---|---|---|" + "---|" * NC] + [f"| {a} | " + " | ".join(f"{x:.3f}" for x in fam[a]) + " |"
                                                     for a in fam]
        e_fix, e_org = fam["H18"][0] - fam["fixed"][0], fam["H18"][0] - fam["original"][0]
        lines += ["", f"- H18 − fixed: {e_fix:+.4f} {ci(d_fix)}, P(>0) {(d_fix > 0).mean():.3f}",
                  f"- H18 − original: {e_org:+.4f} {ci(d_org)}, P(>0) {(d_org > 0).mean():.3f}",
                  f"- H18 beats the fixed rule on {wins}/4 models", ""]
        if split == "test":
            ok = [e_fix > 0 and (d_fix > 0).mean() >= 0.90, e_org > 0 and (d_org > 0).mean() >= 0.95, wins >= 3]
            lines += [f"Conditions: {ok} → **H18-R {'replicates' if all(ok) else 'does not replicate'}**", ""]
        del D, arms
    open(os.path.join(out, "report.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    os.makedirs(OUT, exist_ok=True)
    bank = np.load(os.path.join(pr.OUT, "train_bank.npz"), allow_pickle=True)["bank"].item()
    banks = model_banks(bank)
    vimgs, vpreds, V = split_data("val", banks)
    timgs, tpreds, T = split_data("test", banks)
    Xv = np.concatenate([V[m][1] for m in pr.RFDETR])
    tv = np.concatenate([V[m][2] for m in pr.RFDETR])
    print(f"val rows {len(Xv)}, mean target {tv.mean():.4f}", flush=True)
    f_all = fit(Xv, tv)

    arms_t = {"original": tpreds,
              "fixed": {m: with_conf(tpreds[m], T[m][0], T[m][3]) for m in pr.RFDETR},
              "H18": {m: with_conf(tpreds[m], T[m][0], f_all(T[m][1])) for m in pr.RFDETR}}
    # val sanity check: 2-fold cross-fitting over val scenes
    groups = pr.scenes("val", sorted(vimgs))
    fold = {f: gi % 2 for gi, g in enumerate(sorted(groups, key=lambda g: min(g))) for f in g}
    vconf = {}
    for k in (0, 1):
        tr = np.concatenate([np.array([fold[f] != k for f in V[m][0]]) for m in pr.RFDETR])
        fk = fit(Xv[tr], tv[tr])
        for m in pr.RFDETR:
            sel = np.array([fold[f] == k for f in V[m][0]])
            vconf.setdefault(m, np.zeros(len(sel)))[sel] = fk(V[m][1][sel])
    arms_v = {"original": vpreds,
              "fixed": {m: with_conf(vpreds[m], V[m][0], V[m][3]) for m in pr.RFDETR},
              "H18": {m: with_conf(vpreds[m], V[m][0], vconf[m]) for m in pr.RFDETR}}

    report, lines = {}, ["# H18: learned coverage-aware re-ranker on RF-DETR (research_log.md)", "",
                         f"Calibrator fitted on {len(Xv)} val detections of the 4 models; test untouched. Paired scene "
                         f"bootstrap, {N_BOOT} replicates; family = mean over the 4 models.", ""]
    ci = lambda x: f"[{np.percentile(x, 2.5):+.4f}, {np.percentile(x, 97.5):+.4f}]"  # noqa: E731
    for split, arms, imgs in (("test", arms_t, timgs), ("val (2-fold scene cross-fit)", arms_v, vimgs)):
        point, boot = compare(arms, imgs, split.split()[0])
        fam = {a: np.mean([np.r_[v[:2], v[2]] for v in point[a].values()], 0) for a in point}
        lines += [f"## {split}", "", "| model | original | fixed rule | H18 |", "|---|---|---|---|"]
        for m in pr.RFDETR:
            lines.append(f"| {m} | " + " | ".join(f"{point[a][m][0]:.3f}" for a in ("original", "fixed", "H18")) + " |")
        lines += ["", "| family mean | mAP50-95 | rare AP50-95 | " + " | ".join(pr.CLASSES) + " |",
                  "|---|---|---|" + "---|" * NC]
        for a in fam:
            lines.append(f"| {a} | " + " | ".join(f"{x:.3f}" for x in fam[a]) + " |")
        d_all = boot["H18"][:, 0] - boot["fixed"][:, 0]
        d_rare = boot["H18"][:, 1] - boot["original"][:, 1]
        e_all = fam["H18"][0] - fam["fixed"][0]
        e_rare = fam["H18"][1] - fam["original"][1]
        cls_drop = fam["H18"][2:] - fam["original"][2:]
        wins = sum(point["H18"][m][0] > point["fixed"][m][0] for m in pr.RFDETR)
        lines += ["", f"- all-class H18 − fixed: {e_all:+.4f} {ci(d_all)}, P(>0) {(d_all > 0).mean():.3f}",
                  f"- all-class H18 − original: {fam['H18'][0] - fam['original'][0]:+.4f} "
                  f"{ci(boot['H18'][:, 0] - boot['original'][:, 0])}",
                  f"- rare H18 − original: {e_rare:+.4f} {ci(d_rare)}, P(>0) {(d_rare > 0).mean():.3f}",
                  f"- per class H18 − original: " + ", ".join(f"{c} {x:+.3f}" for c, x in zip(pr.CLASSES, cls_drop)),
                  f"- H18 beats the fixed rule on {wins}/4 models", ""]
        ok = [e_all >= 0.01 and (d_all > 0).mean() >= 0.95, e_rare >= 0.03, cls_drop.min() >= -0.01, wins >= 3]
        report[split] = dict(fam={a: fam[a].tolist() for a in fam}, all_vs_fixed=e_all, p_all=(d_all > 0).mean(),
                             rare_vs_orig=e_rare, class_change=cls_drop.tolist(), wins=wins, conditions=ok)
        if split == "test":
            lines += [f"Conditions (all ≥ +0.01 vs fixed with P ≥ 0.95; rare ≥ +0.03 vs original; no class < −0.01; "
                      f"≥ 3/4 models): {ok} → **H18 {'passes' if all(ok) else 'fails'}**", ""]
    json.dump(report, open(os.path.join(OUT, "report.json"), "w"), indent=1, default=float)
    open(os.path.join(OUT, "report.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main_replicate() if sys.argv[1:] == ["replicate"] else main()
