"""G22-O (research_log.md): the proximity law with proximity measured in each RF-DETR model's own features (backbone
projector output, stride 16, what the decoder attends to) instead of frozen DINOv3 ViT-B.

  embed  (.venv-rfdetr)  per model: box embeddings (mean of the 36x36 tokens whose centres lie in the box, L2-normalised;
                         nearest token if none) of the test boxes and of the model's training-subset boxes
                         -> runs/analysis/g22/own/<run>.npz
  score  (.venv)         per dataset: slopes (confidence, detected) with x_own vs x_dino on the same models, G22
                         estimator and shared scene draws -> runs/analysis/g22/ownspace.md

  .venv-rfdetr/bin/python tools/g22_ownspace.py embed [run ...]
  .venv/bin/python tools/g22_ownspace.py score
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
RUNS = os.path.join(ROOT, "runs", "detect", "BladeYOLO_WindSurface")
OWN = os.path.join(ROOT, "runs", "analysis", "g22", "own")
SPLITS = os.path.join(ROOT, "tools", "splits")
WTBD = os.path.join(ROOT, "external", "WTBD-group")
WTBLADE = os.path.join(ROOT, "external", "WTBlade-group")
# run -> (dataset key, data dir, subset file); only models whose checkpoints are local
MODELS = {
    "rfdetrm_wtbd_group_f25": ("WTBD", WTBD, "wtbd_group_train_f25.txt"),
    "rfdetrm_rand3_wtbd_group_f25": ("WTBD", WTBD, "wtbd_group_train_f25_rand3.txt"),
    **{f"rfdetrm{s}_wtblade_group_f08": ("WTBlade", WTBLADE, f"wtblade_group_train_f08{s}.txt")
       for s in ("", "_rand1", "_rand2", "_rand3")},
}
NC = {"WTBD": 6, "WTBlade": 7}


def labels(data, split, name):
    """(n, 5) YOLO rows [cls, cx, cy, w, h] (normalised), in file order (= bootstrap_ci.load_yolo_split order)."""
    p = os.path.join(data, split, "labels", os.path.splitext(name)[0] + ".txt")
    rows = [list(map(float, l.split()[:5])) for l in open(p) if l.strip()] if os.path.exists(p) else []
    return np.array(rows).reshape(-1, 5)


def token_embed(fmap, yolo):
    """fmap (C, G, G) over the whole (square-resized) image; yolo (n, 5) normalised boxes -> (n, C) L2-normalised."""
    C, G, _ = fmap.shape
    centres = (np.arange(G) + 0.5) / G
    out = np.zeros((len(yolo), C), np.float32)
    for i, (_, cx, cy, w, h) in enumerate(yolo):
        xs = np.where((centres >= cx - w / 2) & (centres <= cx + w / 2))[0]
        ys = np.where((centres >= cy - h / 2) & (centres <= cy + h / 2))[0]
        if not len(xs):
            xs = [min(int(cx * G), G - 1)]
        if not len(ys):
            ys = [min(int(cy * G), G - 1)]
        e = fmap[:, ys][:, :, xs].reshape(C, -1).mean(1)
        out[i] = e / max(np.linalg.norm(e), 1e-12)
    return out


def embed(runs):
    from PIL import Image

    import rfdetr_v1 as v1

    os.makedirs(OWN, exist_ok=True)
    for run in runs:
        path = os.path.join(OWN, run + ".npz")
        if os.path.exists(path):
            continue
        ds, data, sfile = MODELS[run]
        rdir = os.path.join(RUNS, run)
        m = v1.model_class("medium")(pretrain_weights=v1.best_checkpoint(rdir), num_classes=NC[ds],
                                     **v1.model_kwargs(resolution=v1.run_resolution(rdir)))
        cap = {}
        hook = m.model.model.backbone[0].register_forward_hook(
            lambda mod, inp, out: cap.__setitem__("f", out[0][0].tensors[0].detach().float().numpy()))
        keep = set(open(os.path.join(SPLITS, sfile)).read().split())
        res = {"train": {}, "test": {}}
        for split, names in (("train", sorted(keep)), ("test", sorted(os.listdir(os.path.join(data, "test", "images"))))):
            for i, f in enumerate(names):
                y = labels(data, split, f)
                if not len(y):
                    continue
                m.predict(Image.open(os.path.join(data, split, "images", f)).convert("RGB"), threshold=0.5)
                res[split][f] = (y[:, 0].astype(int), token_embed(cap["f"], y))
                if (i + 1) % 100 == 0:
                    print(f"  {run} {split} {i + 1}/{len(names)}", flush=True)
        hook.remove()
        np.savez_compressed(path, res=np.array(res, dtype=object))
        print(f"{run}: train {len(res['train'])} / test {len(res['test'])} images -> {path}", flush=True)


def score():
    import g22_coverage as g
    from coverage_gate import outcome, proximity
    import bootstrap_ci as bc

    lines = ["# G22-O: proximity in the detector's own features vs DINOv3 ViT-B (research_log.md)", "",
             "Same models, same boxes, same scene draws; slope of x (max cosine to the model's subset boxes of the class) "
             "with box and subset fixed effects and log(1+n); 1,000 cluster-bootstrap replicates.", ""]
    for ds in ("WTBD", "WTBlade"):
        runs = [r for r, v in MODELS.items() if v[0] == ds and os.path.exists(os.path.join(OWN, r + ".npz"))]
        if not runs:
            continue
        cfg = g.DATASETS[ds]
        bank = np.load(os.path.join(cfg["proto"], "train_bank.npz"), allow_pickle=True)["bank"].item()
        imgs, tb = g.test_boxes(cfg)
        seen = {}
        for b in tb:  # index of each test box within its image (label-file order)
            k = os.path.basename(b["img"])
            b["k"] = seen.get(k, 0)
            seen[k] = b["k"] + 1
        q = np.stack([b["emb"] for b in tb])
        qc = np.array([b["cls"] for b in tb])
        P = {"own": [], "dino": []}
        for si, run in enumerate(runs):
            own = np.load(os.path.join(OWN, run + ".npz"), allow_pickle=True)["res"].item()
            names = [n.strip() for n in open(os.path.join(SPLITS, MODELS[run][2])) if n.strip()]
            e_d = np.concatenate([bank[n][1] for n in names if n in bank and len(bank[n][1])])
            c_d = np.concatenate([bank[n][0][:, 0] for n in names if n in bank and len(bank[n][1])]).astype(int)
            c_o = np.concatenate([own["train"][n][0] for n in names if n in own["train"]])
            e_o = np.concatenate([own["train"][n][1] for n in names if n in own["train"]])
            q_o = np.stack([own["test"][os.path.basename(b["img"])][1][b["k"]] for b in tb])
            assert all(own["test"][os.path.basename(b["img"])][0][b["k"]] == b["cls"] for b in tb)
            n_c = np.bincount(c_d, minlength=NC[ds])
            x = {"dino": proximity(q, qc, e_d, c_d, 1), "own": proximity(q_o, qc, e_o, c_o, 1)}
            cache = [c for r, _, c in sum(cfg["fams"].values(), []) if r == run][0]
            pred = bc.predict(os.path.join(RUNS, run), imgs, cache)
            y = np.array([outcome(b["box"][None], pred.get(b["img"], np.zeros((0, 6))))[0] for b in tb])
            for sp in P:
                P[sp] += [(i, si, b["group"], np.log1p(n_c[b["cls"]]), x[sp][i], y[i]) for i, b in enumerate(tb)]
        groups = np.unique([b["group"] for b in tb])
        rng = np.random.default_rng(0)
        draws = [groups[rng.integers(0, len(groups), len(groups))] for _ in range(g.N_BOOT)]
        lines += [f"## {ds}: {', '.join(runs)}", "", "| proximity space | ρ(x_own, x_dino) | class-aware conf | detected (≥ 0.05) |",
                  "|---|---|---|---|"]
        for sp in ("dino", "own"):
            A = np.array(P[sp], float)
            A = A[np.isfinite(A[:, 4])]
            by_g = {gg: np.where(A[:, 2] == gg)[0] for gg in groups}
            cells = []
            for yv in (A[:, 5], (A[:, 5] >= g.DET).astype(float)):
                est = g.slope(yv, A[:, [4, 3]], A[:, 0], A[:, 1])[0]
                bs = np.array([g.slope(yv[i], A[i][:, [4, 3]], A[i, 0], A[i, 1])[0]
                               for i in (np.concatenate([by_g[gg] for gg in d]) for d in draws)])
                cells.append(g.fmt(est, bs))
            from scipy.stats import spearmanr
            a, b = np.array(P["own"], float)[:, 4], np.array(P["dino"], float)[:, 4]
            ok = np.isfinite(a) & np.isfinite(b)
            lines.append(f"| {sp} | {spearmanr(a[ok], b[ok])[0]:+.2f} | {cells[0]} | {cells[1]} |")
        lines.append("")
    os.makedirs(os.path.dirname(OWN), exist_ok=True)
    open(os.path.join(os.path.dirname(OWN), "ownspace.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    if sys.argv[1] == "embed":
        embed(sys.argv[2:] or list(MODELS))
    else:
        score()
