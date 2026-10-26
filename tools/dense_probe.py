"""Gates G8 and G9 (Phase 29): which dense features separate texture-defined defects from blade surface?

A dense linear probe on frozen DINOv3 ViT-B features (last block, stride 16 at 640 px) brought to stride 4 (160 x 160):
  bilinear     bilinear upsampling (the control)
  anyup        learned, encoder-agnostic upsampling (AnyUp, ICLR 2026; torch.hub wimmerth/anyup, paper model)
  bilinear+hf  bilinear plus 7 hand-crafted high-frequency channels (|DoG| and gradient magnitude at sigma 1, 2, 4,
               local std 5 x 5) on the grey 640 px image, average-pooled to stride 4
Pixel targets: union of boxes, the smallest covering box decides the class (0 = background). Probe: multinomial
logistic regression (C = 1, standardised features), trained on the class-matched 25% subset with 256 background and
256 defect pixels sampled per image. Score: per-class pixel AP on test (and val), pooled over images, from per-image
score histograms (1,000 bins of log-probability), with a scene bootstrap. Decision rules in research_log.md (Phase 29).

  .venv/bin/python tools/dense_probe.py  -> runs/analysis/dense_probe/{report.md, report.json}
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
OUT = os.path.join(ROOT, "runs", "analysis", "dense_probe")
SUBSET = os.path.join(ROOT, "tools", "splits", "wtbd_group_train_f25.txt")
GROUPS = {"test": os.path.join(ROOT, "runs", "analysis", "bootstrap", "wtbd_group", "groups.csv"),
          "val": os.path.join(ROOT, "runs", "analysis", "bootstrap", "h5_last_val", "groups.csv")}
CLASSES = ["craze", "corrosion", "surface_injure", "thunderstrike", "crack", "hide_craze"]
WEAK = [0, 4, 5]
ARMS = ["bilinear", "anyup", "bilinear+hf"]
SIDE, STRIDE = 160, 4
N_BINS, LO = 1000, -20.0
PER_IMAGE = 256


def pixel_targets(gt):
    """(SIDE, SIDE) class + 1 at each cell centre (0 = background); the smallest covering box wins."""
    t = np.zeros((SIDE, SIDE), int)
    area = np.full((SIDE, SIDE), np.inf)
    c = np.arange(SIDE) * STRIDE + STRIDE / 2
    for cls, x1, y1, x2, y2 in gt:
        a = (x2 - x1) * (y2 - y1)
        m = (c[:, None] >= y1) & (c[:, None] <= y2) & (c[None] >= x1) & (c[None] <= x2) & (a < area)
        t[m], area[m] = int(cls) + 1, a
    return t


def hf_channels(grey):
    """(SIDE, SIDE, 7) high-frequency energy of a (640, 640) grey image in [0, 1]."""
    from scipy import ndimage

    ch = []
    for s in (1, 2, 4):
        ch.append(np.abs(ndimage.gaussian_filter(grey, s) - ndimage.gaussian_filter(grey, 2 * s)))
        ch.append(ndimage.gaussian_gradient_magnitude(grey, s))
    m = ndimage.uniform_filter(grey, 5)
    ch.append(np.sqrt(np.maximum(ndimage.uniform_filter(grey ** 2, 5) - m ** 2, 0)))
    x = np.stack(ch, -1)
    return x.reshape(SIDE, STRIDE, SIDE, STRIDE, 7).mean((1, 3))


def histograms(prob, target):
    """(6, 2, N_BINS) counts of defect (1) and other (0) pixels per log-probability bin, per class."""
    lp = np.clip(np.log(np.maximum(prob, 1e-30)), LO, 0)
    b = np.minimum(((lp - LO) / -LO * N_BINS).astype(int), N_BINS - 1)
    h = np.zeros((len(CLASSES), 2, N_BINS))
    for c in range(len(CLASSES)):
        pos = target == c + 1
        h[c, 1] = np.bincount(b[:, c][pos], minlength=N_BINS)
        h[c, 0] = np.bincount(b[:, c][~pos], minlength=N_BINS)
    return h


def ap_from_hist(h):
    """AP of one class from (2, N_BINS) counts, thresholds from the highest bin down; nan without positives."""
    tp, fp = np.cumsum(h[1][::-1]), np.cumsum(h[0][::-1])
    if tp[-1] == 0:
        return np.nan
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / tp[-1]
    return float(np.sum(prec * np.diff(np.r_[0, rec])))


class Features:
    def __init__(self):
        import torch

        from models.dino_weights import SIZES, build_dinov3

        torch.set_num_threads(os.cpu_count())
        self.torch = torch
        self.vit = build_dinov3("b").eval()
        self.blocks = SIZES["b"][3]
        self.up = torch.hub.load("wimmerth/anyup", "anyup", verbose=False).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __call__(self, path):
        """{arm: (SIDE * SIDE, d) float32} for one image."""
        import cv2

        torch = self.torch
        rgb = cv2.resize(cv2.imread(path), (640, 640))[:, :, ::-1].copy()
        x = (torch.from_numpy(rgb).permute(2, 0, 1)[None].float() / 255 - self.mean) / self.std
        with torch.no_grad():
            y = self.vit.get_intermediate_layers(x, n=self.blocks, reshape=False, norm=True)[-1]  # (1, 1600, 768)
            lr = y.transpose(1, 2).reshape(1, -1, 40, 40)
            bil = torch.nn.functional.interpolate(lr, size=(SIDE, SIDE), mode="bilinear", align_corners=False)
            any_ = self.up(x, lr, output_size=(SIDE, SIDE), q_chunk_size=256)
        flat = lambda t: t[0].permute(1, 2, 0).reshape(SIDE * SIDE, -1).numpy().astype(np.float32)  # noqa: E731
        bil = flat(bil)
        hf = hf_channels(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255).reshape(SIDE * SIDE, -1)
        return {"bilinear": bil, "anyup": flat(any_), "bilinear+hf": np.c_[bil, hf.astype(np.float32)]}


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    os.makedirs(OUT, exist_ok=True)
    feats = Features()
    rng = np.random.default_rng(0)
    tr = bc.load_yolo_split(DATA, "train")
    sub = {n.strip() for n in open(SUBSET) if n.strip()}
    train = sorted(p for p in tr if os.path.basename(p) in sub)
    X = {a: [] for a in ARMS}
    Y = []
    for i, p in enumerate(train):
        t = pixel_targets(tr[p]["gt"]).reshape(-1)
        fg, bg = np.where(t > 0)[0], np.where(t == 0)[0]
        idx = np.r_[rng.choice(fg, min(PER_IMAGE, len(fg)), replace=False) if len(fg) else [],
                    rng.choice(bg, min(PER_IMAGE, len(bg)), replace=False)].astype(int)
        F = feats(p)
        for a in ARMS:
            X[a].append(F[a][idx])
        Y.append(t[idx])
        if i % 20 == 0:
            print(f"  train {i}/{len(train)}", flush=True)
    Y = np.concatenate(Y)
    probes = {}
    for a in ARMS:
        sc = StandardScaler().fit(np.concatenate(X[a]))
        probes[a] = (sc, LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(np.concatenate(X[a])), Y))
        print(f"  probe {a} trained on {len(Y)} pixels", flush=True)
    del X

    report = {}
    lines = ["# Dense probe: which features separate blade defects? (gates G8, G9)", "",
             f"Probe trained on {len(train)} images of the class-matched 25% subset ({len(Y)} pixels). Per-class pixel AP "
             "on stride-4 cells (160 x 160 at 640 px), scene bootstrap (1,000 replicates).", ""]
    for split in ("test", "val"):
        imgs = bc.load_yolo_split(DATA, split)
        files = sorted(imgs)
        H = {a: {} for a in ARMS}
        for i, p in enumerate(files):
            t = pixel_targets(imgs[p]["gt"]).reshape(-1)
            F = feats(p)
            for a in ARMS:
                sc, lr = probes[a]
                H[a][p] = histograms(lr.predict_proba(sc.transform(F[a]))[:, 1:], t)
            if i % 20 == 0:
                print(f"  {split} {i}/{len(files)}", flush=True)
        g = {}
        for r in csv.DictReader(open(GROUPS[split])):
            g.setdefault(r["group"], []).append(os.path.join(DATA, split, "images", r["image"]))
        groups = [[f for f in v if f in H[ARMS[0]]] for v in g.values()]
        groups = [v for v in groups if v]
        aps = lambda a, fs: np.array([ap_from_hist(h) for h in sum(H[a][f] for f in fs)])  # noqa: E731
        point = {a: aps(a, files) for a in ARMS}
        rng_b = np.random.default_rng(0)
        boot = {a: [] for a in ARMS}
        for _ in range(1000):
            draw = [f for gi in rng_b.integers(0, len(groups), len(groups)) for f in groups[gi]]
            for a in ARMS:
                boot[a].append(aps(a, draw))
        boot = {a: np.array(v) for a, v in boot.items()}
        lines += [f"## {split} ({len(files)} images, {len(groups)} scene groups)", "",
                  "| arm | mean AP (6 classes) | weak (craze, crack, hide_craze) | " + " | ".join(CLASSES) + " |",
                  "|---|---|---|" + "---|" * len(CLASSES)]
        for a in ARMS:
            lines.append(f"| {a} | {np.nanmean(point[a]):.3f} | {np.nanmean(point[a][WEAK]):.3f} | "
                         + " | ".join(f"{v:.3f}" for v in point[a]) + " |")
        report[split] = {"point": {a: point[a].tolist() for a in ARMS}, "differences": {}}
        lines.append("")
        for a, name, cls in (("anyup", "G8: anyup − bilinear", list(range(6))), ("anyup", "G8 weak: anyup − bilinear", WEAK),
                             ("bilinear+hf", "G9: (bilinear+hf) − bilinear, weak", WEAK),
                             ("bilinear+hf", "(bilinear+hf) − bilinear, 6 classes", list(range(6)))):
            d = np.nanmean(boot[a][:, cls], 1) - np.nanmean(boot["bilinear"][:, cls], 1)
            v = float(np.nanmean(point[a][cls]) - np.nanmean(point["bilinear"][cls]))
            ci = [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]
            report[split]["differences"][name] = dict(value=v, ci=ci, p_gt_0=float(np.mean(d > 0)))
            lines.append(f"- {name}: {v:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}], P(>0) {np.mean(d > 0):.3f}")
        lines.append("")
    json.dump(report, open(os.path.join(OUT, "report.json"), "w"), indent=2)
    open(os.path.join(OUT, "report.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
