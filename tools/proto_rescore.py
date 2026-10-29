"""Gate G15 (Phase 30): post-hoc rescoring of a trained detector's outputs with frozen DINOv3 prototypes of its own
labelled boxes.

Each detection is embedded as the mean of DINOv3 ViT-B last-layer patch tokens (stride 16 at 640 px) whose centres lie
in its box (the nearest patch if none), L2-normalised; likewise every labelled box of the model's training subset.
Class score q_c = mean of the top-K cosine similarities to class c's labelled boxes, p = softmax(q / TAU) over the
classes with labelled boxes. Rescored confidence = conf^(1 - ALPHA) * p(class of the detection)^ALPHA; the class is
never changed, and a class without labelled boxes keeps its confidence. ALPHA, K, TAU fixed in research_log.md.

  .venv/bin/python tools/proto_rescore.py [test|val]  -> runs/analysis/proto_rescore/{report.md, report.json}
  .venv/bin/python tools/proto_rescore.py rfdetr [test|val]  -> runs/analysis/proto_rescore/rfdetr/ (G15-R: the 4
      RF-DETR models of V1, cached predictions; same settings)
Each split is merged into report.json as it finishes (one split per process keeps memory within 7 GB).
"""

import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import bootstrap_ci as bc  # noqa: E402

DATA = os.path.join(ROOT, "external", "WTBD-group")
RUNS = os.path.join(ROOT, "runs", "detect", "BladeYOLO_WindSurface")
SPLITS = os.path.join(ROOT, "tools", "splits")
PREDS = {"test": os.path.join(ROOT, "runs", "analysis", "bootstrap", "seeds_f25", "preds"),
         "val": os.path.join(ROOT, "runs", "analysis", "proto_rescore", "val_preds")}
GROUPS = {"test": os.path.join(ROOT, "runs", "analysis", "bootstrap", "wtbd_group", "groups.csv"),
          "val": os.path.join(ROOT, "runs", "analysis", "bootstrap", "h5_last_val", "groups.csv")}
OUT = os.path.join(ROOT, "runs", "analysis", "proto_rescore")
CLASSES = ["craze", "corrosion", "surface_injure", "thunderstrike", "crack", "hide_craze"]
RARE = [3, 4]
ALPHA, K, TAU, GRID, STRIDE = 0.5, 3, 0.05, 40, 16
FAMILIES = {  # model -> training subset suffix
    "ViT-B": {**{f"bladeyolo2b_rand{k}_wtbd_group_f25_s0": f"_rand{k}" for k in range(1, 9)},
              "bladeyolo2b_wtbd_group_f25_s0": "", "bladeyolo2b_wtbd_group_f25_s1": ""},
    "no DINOv3": {**{f"bladeyolo2nodino_rand{k}_wtbd_group_f25": f"_rand{k}" for k in range(1, 9)},
                  "bladeyolo2nodino_wtbd_group_f25_s0": "", "bladeyolo2nodino_wtbd_group_f25_s1": ""},
}
VAL_MODELS = ["bladeyolo2b_wtbd_group_f25_s0", "bladeyolo2b_wtbd_group_f25_s1"]  # weights kept
RFDETR = {"rfdetrm_wtbd_group_f25": "", **{f"rfdetrm_rand{k}_wtbd_group_f25": f"_rand{k}" for k in (1, 2, 3)}}
RFDETR_VAL_PREDS = os.path.join(ROOT, "runs", "analysis", "bootstrap", "v1_val_preds")


class Embedder:
    def __init__(self):
        import torch

        from models.dino_weights import SIZES, build_dinov3

        torch.set_num_threads(os.cpu_count())
        self.torch, self.vit, self.blocks = torch, build_dinov3("b").eval(), SIZES["b"][3]
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __call__(self, path, cache=None):
        """(GRID, GRID, 768) float32 patch tokens of the image resized to 640 x 640 (cached as float16)."""
        import cv2

        if cache and os.path.exists(cache):
            return np.load(cache).astype(np.float32)
        torch = self.torch
        rgb = cv2.resize(cv2.imread(path), (640, 640))[:, :, ::-1].copy()
        x = (torch.from_numpy(rgb).permute(2, 0, 1)[None].float() / 255 - self.mean) / self.std
        with torch.no_grad():
            y = self.vit.get_intermediate_layers(x, n=self.blocks, reshape=False, norm=True)[-1]
        m = y[0].reshape(GRID, GRID, -1).numpy().astype(np.float32)
        if cache:
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            np.save(cache, m.astype(np.float16))
        return m


def box_embed(fmap, boxes):
    """(n, d) L2-normalised mean token of the patches whose centres lie in each [x1, y1, x2, y2] box (640 px)."""
    S = np.zeros((GRID + 1, GRID + 1, fmap.shape[2]), np.float64)
    S[1:, 1:] = fmap.cumsum(0).cumsum(1)
    b = np.asarray(boxes, float).reshape(-1, 4)
    lo = np.ceil((b[:, :2] - STRIDE / 2) / STRIDE).clip(0, GRID).astype(int)        # first patch (x, y)
    hi = (np.floor((b[:, 2:] - STRIDE / 2) / STRIDE) + 1).clip(0, GRID).astype(int)  # one past the last
    empty = (hi <= lo).any(1)
    c = ((b[:, :2] + b[:, 2:]) / 2 / STRIDE).astype(int).clip(0, GRID - 1)
    lo[empty], hi[empty] = c[empty], c[empty] + 1
    (x0, y0), (x1, y1) = lo.T, hi.T
    s = S[y1, x1] - S[y0, x1] - S[y1, x0] + S[y0, x0]
    e = s / ((y1 - y0) * (x1 - x0))[:, None]
    return (e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def class_probs(det, bank, bank_cls, nc=6, k=K, tau=TAU):
    """(n, nc) softmax over classes of the mean top-k cosine similarity to each class's labelled boxes; columns of
    classes without labelled boxes are NaN."""
    sim = det @ bank.T
    q = np.full((len(det), nc), -np.inf)
    for c in range(nc):
        sc = sim[:, bank_cls == c]
        if sc.shape[1]:
            kk = min(k, sc.shape[1])
            q[:, c] = np.sort(sc, 1)[:, -kk:].mean(1)
    have = np.isfinite(q[0]) if len(q) else np.zeros(nc, bool)
    z = np.where(have, q / tau, -np.inf)
    z -= z.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    p[:, ~have] = np.nan
    return p


def rescore(pred, emb, bank, bank_cls, alpha=ALPHA):
    """pred (n, 6) [x1, y1, x2, y2, conf, cls] -> same with the rescored confidence."""
    if not len(pred):
        return pred
    p = class_probs(emb, bank, bank_cls)[np.arange(len(pred)), pred[:, 5].astype(int)]
    out = pred.copy()
    ok = np.isfinite(p)
    out[ok, 4] = pred[ok, 4] ** (1 - alpha) * p[ok] ** alpha
    return out


def train_bank(emb):
    """{train image name: (boxes (n, 5) [cls, x1, y1, x2, y2], embeddings (n, d))} for every labelled train image."""
    path = os.path.join(OUT, "train_bank.npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return z["bank"].item()
    tr = bc.load_yolo_split(DATA, "train")
    bank = {}
    for i, (f, im) in enumerate(sorted(tr.items())):
        if len(im["gt"]):
            bank[os.path.basename(f)] = (im["gt"], box_embed(emb(f), im["gt"][:, 1:]))
        if (i + 1) % 50 == 0:
            print(f"  train {i + 1}/{len(tr)}", flush=True)
    np.savez(path, bank=np.array(bank, dtype=object))
    return bank


def scenes(split, files):
    g = {}
    for r in csv.DictReader(open(GROUPS[split])):
        g.setdefault(r["group"], []).append(os.path.join(DATA, split, "images", r["image"]))
    groups = [[os.path.abspath(f) for f in v] for v in g.values()]
    assert sorted(f for v in groups for f in v) == sorted(files), f"{split}: groups do not cover the split"
    return groups


def evaluate(split, models, emb, bank, rng_seed=0):
    """{model: (original matches, rescored matches)} and the scene draws, for one split."""
    imgs = bc.load_yolo_split(DATA, split)
    files = sorted(imgs)
    preds = {m: bc.predict(os.path.join(RUNS, m), imgs, PREDS[split]) for m in models}
    banks = {}
    for m, suf in models.items():
        names = [n.strip() for n in open(os.path.join(SPLITS, f"wtbd_group_train_f25{suf}.txt")) if n.strip()]
        sel = [bank[n] for n in names if n in bank]
        banks[m] = (np.concatenate([e for _, e in sel]), np.concatenate([g[:, 0] for g, _ in sel]).astype(int))
    new = {m: {} for m in models}
    for i, f in enumerate(files):
        fmap = emb(f, os.path.join(OUT, "maps", split, os.path.splitext(os.path.basename(f))[0] + ".npy"))
        for m in models:
            p = preds[m].get(f, np.zeros((0, 6)))
            new[m][f] = rescore(p, box_embed(fmap, p[:, :4]), *banks[m]) if len(p) else p
        if (i + 1) % 20 == 0:
            print(f"  {split} {i + 1}/{len(files)}", flush=True)
    ms = {m: (bc.per_image_matches(imgs, preds[m], 6), bc.per_image_matches(imgs, new[m], 6)) for m in models}
    rng = np.random.default_rng(rng_seed)
    groups = scenes(split, files)
    draws = [[f for gi in rng.integers(0, len(groups), len(groups)) for f in groups[gi]] for _ in range(1000)]
    return ms, files, draws


def metrics(match, files):
    """(all-class mAP50-95, rare-class mean AP50-95, per-class AP50-95)."""
    _, m, _, pc = bc.score(match, files, 6)
    return m, float(np.nanmean(pc[RARE])), pc


def compare(ms, files, draws):
    """Point estimates per model and the bootstrap of the family mean differences (all, rare)."""
    per = {m: (metrics(o, files), metrics(r, files)) for m, (o, r) in ms.items()}
    bs = np.array([[np.subtract(metrics(r, d)[:2], metrics(o, d)[:2]) for o, r in ms.values()] for d in draws])
    d = bs.mean(1)  # (draws, 2): family mean of rescored - original
    pt = np.mean([np.subtract(b[:2], a[:2]) for a, b in per.values()], 0)
    ci = {name: dict(diff=float(pt[j]), lo=float(np.percentile(d[:, j], 2.5)), hi=float(np.percentile(d[:, j], 97.5)),
                     p=float((d[:, j] > 0).mean())) for j, name in enumerate(("all", "rare"))}
    return per, ci


def main_rfdetr(splits):
    """G15-R (research_log.md): G15 unchanged on the 4 RF-DETR models, test and val, one family."""
    out = os.path.join(OUT, "rfdetr")
    os.makedirs(out, exist_ok=True)
    PREDS["val"] = RFDETR_VAL_PREDS
    emb = Embedder()
    bank = train_bank(emb)
    rpath = os.path.join(out, "report.json")
    report = json.load(open(rpath)) if os.path.exists(rpath) else {}
    for split in splits:
        ms, files, draws = evaluate(split, RFDETR, emb, bank)
        per, ci = compare(ms, files, draws)
        up = sum(per[m][1][1] > per[m][0][1] for m in RFDETR)
        report[split] = dict(ci=ci, improved_rare=up, models=len(RFDETR),
                             per={m: dict(orig=list(per[m][0][:2]), resc=list(per[m][1][:2]),
                                          orig_pc=per[m][0][2].tolist(), resc_pc=per[m][1][2].tolist()) for m in RFDETR})
        print(f"{split}: " + "; ".join(f"{k} {v['diff']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}] P {v['p']:.3f}"
                                       for k, v in ci.items()) + f"; rare up {up}/{len(RFDETR)}", flush=True)
        del ms
        json.dump(report, open(rpath, "w"), indent=1)
    lines = ["# G15-R: G15 prototype rescoring on the 4 RF-DETR models", "",
             f"ALPHA {ALPHA}, K {K}, TAU {TAU}; DINOv3 ViT-B at 640 px; paired scene bootstrap, 1,000 replicates.", ""]
    for key, r in report.items():
        lines += [f"## {key}", "", "| model | mAP50-95 | rescored | rare AP50-95 | rescored |", "|---|---|---|---|---|"]
        for m, v in r["per"].items():
            lines.append(f"| {m} | {v['orig'][0]:.3f} | {v['resc'][0]:.3f} | {v['orig'][1]:.3f} | {v['resc'][1]:.3f} |")
        pc_o = np.nanmean([v["orig_pc"] for v in r["per"].values()], 0)
        pc_r = np.nanmean([v["resc_pc"] for v in r["per"].values()], 0)
        lines += ["", "per class AP50-95 (family mean), original → rescored: "
                  + ", ".join(f"{c} {a:.3f} → {b:.3f}" for c, a, b in zip(CLASSES, pc_o, pc_r)), ""]
        lines += [f"- {k}: rescored − original {v['diff']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}], P(>0) {v['p']:.3f}"
                  for k, v in r["ci"].items()] + [f"- rare AP improved on {r['improved_rare']}/{r['models']} models", ""]
    t, v = report.get("test"), report.get("val")
    if t and v:
        passed = (t["ci"]["rare"]["diff"] >= 0.02 and t["ci"]["rare"]["p"] >= 0.95 and t["ci"]["all"]["diff"] >= -0.005
                  and t["improved_rare"] >= 3 and v["ci"]["rare"]["diff"] > 0)
        lines += [f"**G15-R {'passes' if passed else 'fails'}** (rule in research_log.md)."]
    open(os.path.join(out, "report.md"), "w").write("\n".join(lines) + "\n")
    print(lines[-1])


def main():
    os.makedirs(OUT, exist_ok=True)
    emb = Embedder()
    bank = train_bank(emb)
    print(f"bank: {sum(len(g) for g, _ in bank.values())} labelled boxes in {len(bank)} images", flush=True)
    lines = ["# G15: post-hoc DINOv3 prototype rescoring", "",
             f"ALPHA {ALPHA}, K {K}, TAU {TAU}; patch tokens of DINOv3 ViT-B at 640 px. Paired scene bootstrap, "
             "1,000 replicates; family = mean over its models.", ""]
    rpath = os.path.join(OUT, "report.json")
    report = json.load(open(rpath)) if os.path.exists(rpath) else {}
    for split in sys.argv[1:] or ("test", "val"):
        models = {m: s for fam in FAMILIES.values() for m, s in fam.items()} if split == "test" else \
                 {m: FAMILIES["ViT-B"][m] for m in VAL_MODELS}
        ms, files, draws = evaluate(split, models, emb, bank)
        fams = FAMILIES if split == "test" else {"ViT-B (class-matched, 2 seeds)": {m: "" for m in VAL_MODELS}}
        for fam, mods in fams.items():
            per, ci = compare({m: ms[m] for m in mods}, files, draws)
            up = sum(per[m][1][1] > per[m][0][1] for m in mods)
            report[f"{split} {fam}"] = dict(ci=ci, improved_rare=up, models=len(mods),
                                            per={m: dict(orig=list(per[m][0][:2]), resc=list(per[m][1][:2]),
                                                         orig_pc=per[m][0][2].tolist(), resc_pc=per[m][1][2].tolist())
                                                 for m in mods})
            print(f"{split} {fam}: " + "; ".join(f"{k} {v['diff']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}] P {v['p']:.3f}"
                                                 for k, v in ci.items()) + f"; rare up {up}/{len(mods)}", flush=True)
        del ms
        json.dump(report, open(rpath, "w"), indent=1)
    for key, r in report.items():
        lines += [f"## {key}", "", "| model | mAP50-95 | rescored | rare AP50-95 | rescored |", "|---|---|---|---|---|"]
        for m, v in r["per"].items():
            lines.append(f"| {m} | {v['orig'][0]:.3f} | {v['resc'][0]:.3f} | {v['orig'][1]:.3f} | {v['resc'][1]:.3f} |")
        pc_o = np.nanmean([v["orig_pc"] for v in r["per"].values()], 0)
        pc_r = np.nanmean([v["resc_pc"] for v in r["per"].values()], 0)
        lines += ["", "per class AP50-95 (family mean), original → rescored: "
                  + ", ".join(f"{c} {a:.3f} → {b:.3f}" for c, a, b in zip(CLASSES, pc_o, pc_r)), ""]
        lines += [f"- {k}: rescored − original {v['diff']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}], P(>0) {v['p']:.3f}"
                  for k, v in r["ci"].items()] + [f"- rare AP improved on {r['improved_rare']}/{r['models']} models", ""]
    t, v = report.get("test ViT-B"), report.get("val ViT-B (class-matched, 2 seeds)")
    if t and v:
        passed = (t["ci"]["rare"]["diff"] >= 0.02 and t["ci"]["rare"]["p"] >= 0.95 and t["ci"]["all"]["diff"] >= -0.005
                  and t["improved_rare"] >= 7 and v["ci"]["rare"]["diff"] > 0)
        lines += [f"**G15 {'passes' if passed else 'fails'}** (rule in research_log.md, Phase 30)."]
    open(os.path.join(OUT, "report.md"), "w").write("\n".join(lines) + "\n")
    print(lines[-1])


if __name__ == "__main__":
    if sys.argv[1:2] == ["rfdetr"]:
        main_rfdetr(sys.argv[2:] or ("test", "val"))
    else:
        main()
