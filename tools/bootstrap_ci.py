"""Confidence intervals for test mAP by resampling scene groups, and paired intervals for differences between models.

Test images are grouped into scenes with the matching used for the splits (tools/matching.py: DINOv2 retrieval,
flip-aware ORB/RANSAC verification; same scene if cosine >= DINO_GROUP or >= MIN_INLIERS inliers), because images of
one scene are not independent samples. Each run predicts the test split once (conf 0.001, NMS IoU 0.7, the run's
training image size; cached). Every bootstrap replicate draws scene groups with replacement and scores all runs on the
same draw, so the interval of a difference accounts for the scenes both models share. mAP: 101-point COCO AP per class
at IoU 0.50 and 0.50:0.95, averaged over the classes present in the draw.

Usage:
    python tools/bootstrap_ci.py runs/.../bladeyolo2_wtbd_group runs/.../yolo12l_wtbd_group \
        --data external/WTBD-group --name wtbd_group
Writes runs/analysis/bootstrap/<name>/{report.md, report.json, groups.csv}.
"""

import argparse
import json
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import matching as mt  # noqa: E402
import detection_metrics as dm  # noqa: E402

IOU_THRS = dm.IOU_THRS


def load_yolo_split(data, split):
    """{image path: dict(gt=(n,5) [cls, x1, y1, x2, y2] in 640-input pixels, scale)} for a YOLO-layout dataset."""
    import cv2

    out = {}
    img_dir = os.path.join(data, split, "images")
    for f in sorted(os.listdir(img_dir)):
        p = os.path.join(img_dir, f)
        H, W = cv2.imread(p).shape[:2]
        s = 640 / max(W, H)
        lab = os.path.join(data, split, "labels", os.path.splitext(f)[0] + ".txt")
        rows = [list(map(float, l.split())) for l in open(lab) if l.strip()] if os.path.exists(lab) else []
        g = np.array([[c, (x - w / 2) * W * s, (y - h / 2) * H * s, (x + w / 2) * W * s, (y + h / 2) * H * s]
                      for c, x, y, w, h in rows]).reshape(-1, 5)
        out[os.path.abspath(p)] = dict(gt=g, scale=s)
    return out


def predict(run, imgs, cache_dir, batch=8):
    path = os.path.join(cache_dir, f"{os.path.basename(os.path.normpath(run))}.json")
    if os.path.exists(path):
        return {k: np.array(v).reshape(-1, 6) for k, v in json.load(open(path)).items()}
    sys.path.insert(0, ROOT)
    import models  # noqa: F401  (BladeYOLO modules for unpickling)
    from ultralytics import YOLO

    model = YOLO(os.path.join(run, "weights", "best.pt"))
    preds, files = {}, sorted(imgs)
    for i in range(0, len(files), batch):
        for f, r in zip(files[i:i + batch], model.predict(files[i:i + batch], imgsz=dm.run_imgsz(run), conf=0.001,
                                                          iou=0.7, max_det=300, verbose=False, device=dm.device())):
            b = r.boxes.cpu()
            preds[f] = np.concatenate([b.xyxy.numpy() * imgs[f]["scale"], b.conf.numpy()[:, None],
                                       b.cls.numpy()[:, None]], 1).round(3)
        print(f"  {os.path.basename(run)}: {min(i + batch, len(files))}/{len(files)}", flush=True)
    os.makedirs(cache_dir, exist_ok=True)
    json.dump({k: v.tolist() for k, v in preds.items()}, open(path, "w"))
    return preds


def scene_groups(paths, cache, workers):
    """Connected components of the same-scene graph among the given images."""
    E = mt.dinov2_embeddings(paths, os.path.join(cache, "dinov2_vits14.npz"))
    k = min(mt.KNN, len(paths) - 1)
    nn_idx, nn_sim = mt.knn(E, k)
    vcache = os.path.join(cache, "verify.json")
    known = json.load(open(vcache)) if os.path.exists(vcache) else {}
    key = lambda a, b: f"{min(a, b)}\t{max(a, b)}"  # noqa: E731
    pairs = {(min(a, int(b)), max(a, int(b))) for a in range(len(paths)) for b in nn_idx[a] if a != int(b)}
    todo = sorted(p for p in pairs if key(paths[p[0]], paths[p[1]]) not in known)
    if todo:
        res = mt.verify_pairs(paths, todo, workers)
        known.update({key(paths[a], paths[b]): list(v) for (a, b), v in res.items()})
        json.dump(known, open(vcache, "w"))
    dsu = mt.DSU(len(paths))
    for a in range(len(paths)):
        for b, s in zip(nn_idx[a], nn_sim[a]):
            b = int(b)
            if a != b and (s >= mt.DINO_GROUP or known[key(paths[a], paths[b])][0] >= mt.MIN_INLIERS):
                dsu.union(a, b)
    return dsu.groups()


def per_image_matches(imgs, preds, nc):
    """For every image, class and IoU threshold: (tp flags, confidences, number of GT boxes)."""
    out = {}
    for f, im in imgs.items():
        g, p = im["gt"], preds.get(f, np.zeros((0, 6)))
        d = {}
        for c in range(nc):
            gc, pc = g[g[:, 0] == c], p[p[:, 5] == c]
            pc = pc[np.argsort(-pc[:, 4])]
            iou = dm.iou_matrix(pc[:, :4], gc[:, 1:]) if len(pc) and len(gc) else np.zeros((len(pc), len(gc)))
            for t in IOU_THRS:
                used, tp = np.zeros(len(gc), bool), np.zeros(len(pc), bool)
                for i in range(len(pc)):
                    cand = np.where((iou[i] >= t) & ~used)[0]
                    if len(cand):
                        j = cand[np.argmax(iou[i, cand])]
                        used[j] = tp[i] = True
                d[(c, round(float(t), 2))] = (tp, pc[:, 4], len(gc))
        out[f] = d
    return out


def score(matches, files, nc):
    """(mAP50, mAP50-95, per-class AP50) over a multiset of images."""
    ap = np.full((nc, len(IOU_THRS)), np.nan)
    for c in range(nc):
        for k, t in enumerate(IOU_THRS):
            ms = [matches[f][(c, round(float(t), 2))] for f in files]
            n = sum(m[2] for m in ms)
            if n:
                ap[c, k] = dm.ap_from(np.concatenate([m[0] for m in ms]), np.concatenate([m[1] for m in ms]), n)
    return float(np.nanmean(ap[:, 0])), float(np.nanmean(ap.mean(1))), ap[:, 0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run folders (weights/best.pt); differences are taken against the first")
    ap.add_argument("--data", required=True, help="YOLO-layout dataset folder with data.yaml and test/")
    ap.add_argument("--split", default="test")
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=2000, help="bootstrap replicates")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default=os.path.join(ROOT, ".dataset_cache"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    out = os.path.join(ROOT, "runs", "analysis", "bootstrap", args.name)
    os.makedirs(out, exist_ok=True)
    names = yaml.safe_load(open(os.path.join(args.data, "data.yaml")))["names"]
    nc = len(names)
    imgs = load_yolo_split(os.path.abspath(args.data), args.split)
    files = sorted(imgs)
    groups = [[files[i] for i in g] for g in scene_groups(files, args.cache, args.workers)]
    with open(os.path.join(out, "groups.csv"), "w") as f:
        f.write("image,group\n")
        f.writelines(f"{os.path.basename(p)},{gi}\n" for gi, g in enumerate(groups) for p in g)
    scenes = {c: len({gi for gi, g in enumerate(groups) for p in g if (imgs[p]["gt"][:, 0] == c).any()})
              for c in range(nc)}
    print(f"{len(files)} images in {len(groups)} scene groups; scenes per class: "
          + ", ".join(f"{names[c]} {scenes[c]}" for c in range(nc)))

    runs = [os.path.abspath(r) for r in args.runs]
    labels = [os.path.basename(r) for r in runs]
    matches = [per_image_matches(imgs, predict(r, imgs, os.path.join(out, "preds")), nc) for r in runs]
    point = [score(m, files, nc) for m in matches]

    rng = np.random.default_rng(args.seed)
    boot = np.zeros((args.n, len(runs), 2))
    for b in range(args.n):
        draw = [p for gi in rng.integers(0, len(groups), len(groups)) for p in groups[gi]]
        for k, m in enumerate(matches):
            boot[b, k] = score(m, draw, nc)[:2]
        if (b + 1) % 200 == 0:
            print(f"  bootstrap {b + 1}/{args.n}", flush=True)

    ci = lambda x: [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]  # noqa: E731
    report = dict(data=args.data, split=args.split, images=len(files), scene_groups=len(groups),
                  scenes_per_class={names[c]: scenes[c] for c in range(nc)}, replicates=args.n, runs={}, differences={})
    for k, r in enumerate(labels):
        report["runs"][r] = dict(mAP50=point[k][0], mAP50_ci=ci(boot[:, k, 0]), mAP50_95=point[k][1],
                                 mAP50_95_ci=ci(boot[:, k, 1]),
                                 AP50_per_class={names[c]: float(point[k][2][c]) for c in range(nc)})
    for k in range(1, len(runs)):
        d = boot[:, 0] - boot[:, k]
        report["differences"][f"{labels[0]} - {labels[k]}"] = dict(
            mAP50=point[0][0] - point[k][0], mAP50_ci=ci(d[:, 0]), p_mAP50_le_0=float(np.mean(d[:, 0] <= 0)),
            mAP50_95=point[0][1] - point[k][1], mAP50_95_ci=ci(d[:, 1]), p_mAP50_95_le_0=float(np.mean(d[:, 1] <= 0)))
    json.dump(report, open(os.path.join(out, "report.json"), "w"), indent=2)

    f2 = lambda v, c: f"{v:.3f} [{c[0]:.3f}, {c[1]:.3f}]"  # noqa: E731
    md = [f"# Bootstrap over scene groups: {args.name}", "",
          f"{len(files)} {args.split} images in {len(groups)} scene groups, {args.n} replicates, 95% percentile "
          "intervals. Scenes per class: " + ", ".join(f"{names[c]} {scenes[c]}" for c in range(nc)) + ".", "",
          "| run | mAP50 [95% CI] | mAP50-95 [95% CI] |", "|---|---|---|"]
    md += [f"| {r} | {f2(v['mAP50'], v['mAP50_ci'])} | {f2(v['mAP50_95'], v['mAP50_95_ci'])} |"
           for r, v in report["runs"].items()]
    if report["differences"]:
        md += ["", "| difference (paired) | mAP50 [95% CI] | P(≤0) | mAP50-95 [95% CI] | P(≤0) |", "|---|---|---|---|---|"]
        md += [f"| {r} | {f2(v['mAP50'], v['mAP50_ci'])} | {v['p_mAP50_le_0']:.3f} | "
               f"{f2(v['mAP50_95'], v['mAP50_95_ci'])} | {v['p_mAP50_95_le_0']:.3f} |"
               for r, v in report["differences"].items()]
    md += ["", "| class | scenes | " + " | ".join(f"{r} AP50" for r in labels) + " |",
           "|---|---|" + "---|" * len(labels)]
    md += [f"| {names[c]} | {scenes[c]} | " + " | ".join(f"{point[k][2][c]:.3f}" for k in range(len(runs))) + " |"
           for c in range(nc)]
    open(os.path.join(out, "report.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
