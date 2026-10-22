"""Multi-view inference: fuse a detector's outputs across overlapping views of the same surface.

Inspection flights photograph each blade surface several times. For a target image b, every other image a of the same
split with a verified transform a -> b (ORB/RANSAC similarity, from the label audit, tools/view_label_audit.py) has its
detections moved into b. Per class, b's own detections and the moved ones are clustered greedily by IoU, highest score
first. A cluster's score is the **mean, over the views that see the cluster's location, of each view's best score in
the cluster** (a view that sees the spot and detects nothing there counts as 0): consistent detections keep their
score, detections only one of several views makes are diluted, and a defect b missed but other views found appears.
Partner views are weighted by w; the box is the score-weighted mean of the members, or b's own box when b has one
(geom "own"; similarity transforms ignore perspective, so moved boxes are approximate).

Partners are stratified as in the label audit: "other view" (the moved frame covers < SAME_VIEW of b: a different
photograph) or "same view" (augmented copies and re-crops, which a real flight would not produce). Settings are
chosen on val by mean mAP50-95 and applied once to test, scored with the paired scene bootstrap, on all test images
and on those with a partner.

With --views DIR (WTBlade-group, tools/build_wtblade_group.py) the transforms come from DIR/views_{val,test}.json,
registered between distinct source frames: there "same view" partners are neighbouring video frames, not copies.

  .venv/bin/python tools/multiview_fusion.py --runs runs/detect/BladeYOLO_WindSurface/bladeyolo2_wtbd_group_f25 ...
  .venv/bin/python tools/multiview_fusion.py --runs runs/.../bladeyolo2b_wtblade_group --data external/WTBlade-group \
      --views external/WTBlade-group --groups external/WTBlade-group/groups.csv --out runs/analysis/multiview_wtblade
"""

import argparse
import csv
import itertools
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import bootstrap_ci as bc  # noqa: E402
import detection_metrics as dm  # noqa: E402
import matching as mt  # noqa: E402

MIN_INLIERS = mt.MIN_INLIERS
IOU = (0.4, 0.55)
W = (0.5, 1.0)
GEOM = ("own", "mean")
PARTNERS = ("other", "all")


def affine(size_a, size_b, fl, M):
    """3x3 matrix taking a's pixel coordinates to b's (the label audit's move(): scale to 480 px, flip, M, unscale)."""
    sa, sb = 480 / max(size_a), 480 / max(size_b)
    ha, wa = round(size_a[0] * sa), round(size_a[1] * sa)
    F = np.eye(3)
    if fl in (1, -1):
        F[0] = [-1, 0, wa - 1]
    if fl in (0, -1):
        F[1] = [0, -1, ha - 1]
    return np.diag([1 / sb, 1 / sb, 1]) @ np.vstack([np.array(M, float), [0, 0, 1]]) @ F @ np.diag([sa, sa, 1])


def move_boxes(boxes, A):
    """(n,4) boxes through the 3x3 affine A -> axis-aligned boxes of the moved corners."""
    if not len(boxes):
        return np.zeros((0, 4))
    x1, y1, x2, y2 = boxes.T
    c = np.stack([np.c_[x1, y1], np.c_[x2, y1], np.c_[x2, y2], np.c_[x1, y2]], 1)  # (n,4,2)
    m = c @ A[:2, :2].T + A[:2, 2]
    return np.c_[m[..., 0].min(1), m[..., 1].min(1), m[..., 0].max(1), m[..., 1].max(1)]


def coverage(A, size_a, size_b):
    """Share of b covered by a's frame moved through A (approximated on a 64x64 grid of b)."""
    Hb, Wb = size_b
    ys, xs = np.mgrid[0:64, 0:64]
    p = np.c_[(xs.ravel() + 0.5) * Wb / 64, (ys.ravel() + 0.5) * Hb / 64, np.ones(64 * 64)]
    q = p @ np.linalg.inv(A).T
    return float(np.mean((q[:, 0] >= 0) & (q[:, 0] < size_a[1]) & (q[:, 1] >= 0) & (q[:, 1] < size_a[0])))


def view_graph(data, split, transforms, official):
    """{target path: [(source path, A source->target in 640-input pixels of each, stratum)]} within one split."""
    import cv2

    paths = []
    for s in sorted(os.listdir(official)):
        d = os.path.join(official, s, "images")
        if os.path.isdir(d):
            paths += [os.path.join(d, f) for f in sorted(os.listdir(d))]
    here = {f: os.path.abspath(os.path.join(data, split, "images", f)) for f in os.listdir(os.path.join(data, split, "images"))}
    size = {}

    def sz(p):
        if p not in size:
            size[p] = cv2.imread(p).shape[:2]
        return size[p]

    graph = {p: [] for p in here.values()}
    for key, v in json.load(open(transforms)).items():
        i, j = map(int, key.split())
        a, b = here.get(os.path.basename(paths[i])), here.get(os.path.basename(paths[j]))
        if a is None or b is None or a == b or v[0] < MIN_INLIERS or v[2] is None:
            continue
        A = affine(sz(paths[i]), sz(paths[j]), v[1], v[2])
        s = np.sqrt(abs(np.linalg.det(A[:2, :2])))
        if not 0.33 < s < 3:
            continue
        cov_ab, cov_ba = coverage(A, sz(a), sz(b)), coverage(np.linalg.inv(A), sz(b), sz(a))
        stratum = "same" if max(cov_ab, cov_ba) >= mt.SAME_VIEW else "other"
        ka, kb = 640 / max(sz(a)), 640 / max(sz(b))  # predictions and labels are in 640-input pixels
        A640 = np.diag([kb, kb, 1]) @ A @ np.diag([1 / ka, 1 / ka, 1])
        for src, dst, M_ in ((a, b, A640), (b, a, np.linalg.inv(A640))):
            if all(src != e[0] for e in graph[dst]):
                graph[dst].append((src, M_, stratum))
    return graph


def view_graph_json(data, split, views):
    """As view_graph, from tools/build_wtblade_group.py's views_<split>.json ({target: [[source, inliers, flip, M,
    coverage], ...]}, both directions registered separately)."""
    import cv2

    here = os.path.join(os.path.abspath(data), split, "images")
    graph = {os.path.join(here, f): [] for f in os.listdir(here)}
    for b, srcs in json.load(open(os.path.join(views, f"views_{split}.json"))).items():
        pb = os.path.join(here, b)
        sb = cv2.imread(pb).shape[:2]
        for a, inl, fl, M, cov in srcs:
            pa = os.path.join(here, a)
            sa = cv2.imread(pa).shape[:2]
            A = affine(sa, sb, fl, M)
            s = np.sqrt(abs(np.linalg.det(A[:2, :2])))
            if inl < MIN_INLIERS or not 0.33 < s < 3:
                continue
            ka, kb = 640 / max(sa), 640 / max(sb)
            graph[pb].append((pa, np.diag([kb, kb, 1]) @ A @ np.diag([1 / ka, 1 / ka, 1]), "same" if cov >= mt.SAME_VIEW else "other"))
    return graph


def fuse(preds, graph, iou, w, geom, partners, size=640.0, min_conf=0.001):
    """Fused predictions {image: (n,6) [x1,y1,x2,y2,score,cls]} for every image in preds."""
    out = {}
    for b, own in preds.items():
        own = own[own[:, 4] >= min_conf]
        srcs = [(b, np.eye(3), 1.0, own)]
        for a, A, st in graph.get(b, []):
            if partners == "other" and st != "other":
                continue
            d = preds.get(a, np.zeros((0, 6)))
            d = d[d[:, 4] >= min_conf]
            m = move_boxes(d[:, :4], A)
            cx, cy = (m[:, 0] + m[:, 2]) / 2, (m[:, 1] + m[:, 3]) / 2
            keep = (cx >= 0) & (cx < size) & (cy >= 0) & (cy < size)
            srcs.append((a, A, w, np.c_[np.clip(m[keep], 0, size), d[keep, 4:]]))
        if len(srcs) == 1:
            out[b] = own
            continue
        inv = [np.linalg.inv(A) for _, A, _, _ in srcs]
        res = []
        for c in np.unique(np.concatenate([s[3][:, 5] for s in srcs])) if any(len(s[3]) for s in srcs) else []:
            items = [(k, r) for k, s in enumerate(srcs) for r in s[3][s[3][:, 5] == c]]
            items.sort(key=lambda t: -t[1][4])
            clusters = []
            for k, r in items:
                best, bi = 0.0, -1
                for ci, cl in enumerate(clusters):
                    o = dm.iou_matrix(r[None, :4], cl["box"][None])[0, 0]
                    if o > best:
                        best, bi = o, ci
                if best >= iou:
                    cl = clusters[bi]
                    cl["m"].append((k, r))
                    cl["best"][k] = max(cl["best"].get(k, 0.0), r[4])
                else:
                    clusters.append(dict(box=r[:4].copy(), m=[(k, r)], best={k: r[4]}))
            for cl in clusters:
                cx, cy = (cl["box"][0] + cl["box"][2]) / 2, (cl["box"][1] + cl["box"][3]) / 2
                seen = []  # views whose own frame contains the cluster centre
                for k, I in enumerate(inv):
                    p = I @ [cx, cy, 1]
                    if 0 <= p[0] < size and 0 <= p[1] < size:
                        seen.append(k)
                wt = {k: srcs[k][2] for k in seen}
                score = sum(wt[k] * cl["best"].get(k, 0.0) for k in seen) / max(sum(wt.values()), 1e-9)
                mem = np.array([r for _, r in cl["m"]])
                if geom == "own" and any(k == 0 for k, _ in cl["m"]):
                    box = max((r for k, r in cl["m"] if k == 0), key=lambda r: r[4])[:4]
                else:
                    sw = mem[:, 4] * np.array([srcs[k][2] for k, _ in cl["m"]])
                    box = (mem[:, :4] * sw[:, None]).sum(0) / sw.sum()
                res.append([*box, score, c])
        r = np.array(res).reshape(-1, 6)
        out[b] = r[np.argsort(-r[:, 4])][:300]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--data", default=os.path.join(ROOT, "external", "WTBD-group"))
    ap.add_argument("--official", default=os.path.join(ROOT, "external", "WTBD-official"),
                    help="the dataset the transforms' indices refer to (the label audit ran on WTBD-official)")
    ap.add_argument("--transforms", default=os.path.join(ROOT, "runs", "analysis", "novel", "label_audit", "wtbd", "transforms_v3.json"))
    ap.add_argument("--groups", default=os.path.join(ROOT, "runs", "analysis", "bootstrap", "wtbd_group", "groups.csv"))
    ap.add_argument("--preds", default=os.path.join(ROOT, "runs", "analysis", "sam_refine"), help="has preds_val/, preds_test/")
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "analysis", "multiview"))
    ap.add_argument("--views", default=None, help="folder with views_{val,test}.json (WTBlade-group) instead of --transforms")
    args = ap.parse_args()
    import yaml

    nc = len(yaml.safe_load(open(os.path.join(args.data, "data.yaml")))["names"])
    os.makedirs(args.out, exist_ok=True)
    names = [os.path.basename(os.path.normpath(r)) for r in args.runs]
    S = {}
    for split in ("val", "test"):
        imgs = bc.load_yolo_split(args.data, split)
        preds = {n: bc.predict(r, imgs, os.path.join(args.preds, f"preds_{split}")) for n, r in zip(names, args.runs)}
        graph = view_graph_json(args.data, split, args.views) if args.views else view_graph(args.data, split, args.transforms, args.official)
        for st in ("other", "same"):
            k = sum(any(e[2] == st for e in v) for v in graph.values())
            print(f"{split}: {k}/{len(graph)} images have a '{st} view' partner")
        S[split] = dict(imgs=imgs, preds=preds, graph=graph, files=sorted(imgs))

    V = S["val"]
    ms = lambda sp, p: bc.per_image_matches(S[sp]["imgs"], p, nc)
    base = np.mean([bc.score(ms("val", V["preds"][n]), V["files"], nc)[:2] for n in names], 0)
    grid, best = [], {}
    for partners in PARTNERS:
        for iou, w, geom in itertools.product(IOU, W, GEOM):
            sc = np.mean([bc.score(ms("val", fuse(V["preds"][n], V["graph"], iou, w, geom, partners)), V["files"], nc)[:2]
                          for n in names], 0)
            grid.append((partners, iou, w, geom, *sc))
            print(f"val {partners:5s} iou {iou} w {w} geom {geom:4s}: {sc[0]:.3f} / {sc[1]:.3f} (original {base[0]:.3f} / {base[1]:.3f})", flush=True)
        best[partners] = max((g for g in grid if g[0] == partners), key=lambda g: g[5])
    with open(os.path.join(args.out, "val_grid.csv"), "w") as fh:
        csv.writer(fh).writerows([("partners", "iou", "w", "geom", "mAP50", "mAP50-95"), *grid])

    T = S["test"]
    g = {}
    for row in csv.DictReader(open(args.groups)):
        if row.get("subset", "test") != "test":
            continue
        g.setdefault(row["group"], []).append(os.path.join(os.path.abspath(args.data), "test", "images", row["image"]))
    groups = list(g.values())
    rng = np.random.default_rng(0)
    draws = [[f for gi in rng.integers(0, len(groups), len(groups)) for f in groups[gi]] for _ in range(1000)]
    lines = ["# Multi-view inference (settings from val)", "",
             "| images | partners | setting | run | original mAP50 / 50-95 | fused | difference mAP50 [95% CI] | difference mAP50-95 [95% CI] |",
             "|---|---|---|---|---|---|---|---|"]
    report = {}
    for partners, (_, iou, w, geom, *_v) in best.items():
        linked = {f for f in T["files"] if any(partners == "all" or e[2] == "other" for e in T["graph"][f])}
        for subset, keep in (("all", set(T["files"])), ("with a partner", linked)):
            files = [f for f in T["files"] if f in keep]
            sub_draws = [[f for f in dr if f in keep] for dr in draws]
            o_all, f_all, D = [], [], []
            for n in names + ["mean"]:
                if n == "mean":
                    o, f_, d = np.mean(o_all, 0), np.mean(f_all, 0), np.mean(D, 0)
                else:
                    mo, mf = ms("test", T["preds"][n]), ms("test", fuse(T["preds"][n], T["graph"], iou, w, geom, partners))
                    o, f_ = bc.score(mo, files, nc)[:2], bc.score(mf, files, nc)[:2]
                    o_all.append(o), f_all.append(f_)
                    d = np.array([np.subtract(bc.score(mf, dr, nc)[:2], bc.score(mo, dr, nc)[:2]) for dr in sub_draws])
                    D.append(d)
                ci = [f"{f_[j] - o[j]:+.3f} [{np.nanpercentile(d[:, j], 2.5):+.3f}, {np.nanpercentile(d[:, j], 97.5):+.3f}] "
                      f"P(>0) {np.mean(d[:, j] > 0):.2f}" for j in (0, 1)]
                lines.append(f"| {subset} ({len(files)}) | {partners} | iou {iou}, w {w}, {geom} | {n} | {o[0]:.3f} / {o[1]:.3f} | "
                             f"{f_[0]:.3f} / {f_[1]:.3f} | {ci[0]} | {ci[1]} |")
                report[f"{subset}/{partners}/{n}"] = dict(original=list(o), fused=list(f_))
                print(lines[-1], flush=True)
    open(os.path.join(args.out, "report.md"), "w").write("\n".join(lines) + "\n")
    json.dump(report, open(os.path.join(args.out, "report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
