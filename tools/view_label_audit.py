"""Cross-view label audit: are the same defects labelled the same way in different images of the same surface?

Pairs of images showing the same surface (augmented copies, overlapping crops, neighbouring frames) are found as in
tools/build_dataset.py: DINOv2 nearest neighbours, then flip-aware ORB matching with a RANSAC similarity transform
(>= MIN_INLIERS inliers). Every label of image a that the transform places inside image b (>= 80% of its area) is
compared with b's labels:
  agree           a b label of the same class with IoU >= 0.5
  class_conflict  a b label of another class with IoU >= 0.5, none of the same class
  split           no single match, but >= 50% of the projected box is covered by >= 2 same-class b labels, or it covers
                  >= 2 same-class b labels (the same defect cut into boxes differently)
  missing         nothing labelled there in b (no b label overlaps the projected box at IoU >= 0.1, < 20% covered)
  loose           some other partial overlap (shifted or resized box)
Only labels whose own box holds >= LOCAL_MIN of the match's inlier keypoints are audited: the defect itself must be
matched, not a burned-in caption, the sky or the tower (such matches can link different blades that look alike).
A pair counts once per direction. Pairs are reported in two strata:
  same view       the moved frame covers >= SAME_VIEW of the other image (augmented copies, re-crops): the similarity
                  transform is exact, so every difference is an annotation difference
  other view      partially overlapping or different viewpoints: perspective is not modelled, so projected boxes can be
                  off and "loose"/"missing" are upper bounds
Example sheets per stratum and outcome let the transform be checked by eye.

Usage:
    python tools/view_label_audit.py external/WTBD-official --name wtbd
    python tools/view_label_audit.py WTBlade-Defect --name wtblade
Writes runs/analysis/novel/label_audit/<name>/{report.md, report.json, pairs.csv, examples_<outcome>.jpg}.
"""

import argparse
import collections
import json
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import build_dataset as B  # noqa: E402
import error_analysis as ea  # noqa: E402

FLIPS = (None, 1, 0, -1)
OUTCOMES = ("agree", "class_conflict", "split", "loose", "missing")
STRATA = ("same_view", "other_view")
LOCAL_MIN = 8  # inlier keypoints inside a label box for it to be audited


def load(root):
    """[(image path, (H, W), labels (n,5) [cls, x1, y1, x2, y2] in pixels)] over every split folder."""
    items = []
    for split in sorted(os.listdir(root)):
        img_dir = os.path.join(root, split, "images")
        if not os.path.isdir(img_dir):
            continue
        for f in sorted(os.listdir(img_dir)):
            p = os.path.abspath(os.path.join(img_dir, f))
            H, W = cv2.imread(p).shape[:2]
            lab = os.path.join(root, split, "labels", os.path.splitext(f)[0] + ".txt")
            rows = [list(map(float, l.split())) for l in open(lab) if l.strip()] if os.path.exists(lab) else []
            g = np.array([[c, (x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H]
                          for c, x, y, w, h in rows]).reshape(-1, 5)
            items.append((p, (H, W), g))
    return items


def transform_job(args):
    """Best flip + similarity transform a -> b on 480 px greyscale.
    Returns (inliers, flip, M, inlier points in a's 480 px frame before flipping) or (0, None, None, [])."""
    a_path, b_path = args
    ga, gb = B._gray(a_path), B._gray(b_path)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    kb, db = B._orb(gb)
    best = (0, None, None, None)
    h, w = ga.shape
    for fl in FLIPS:
        ka, da = B._orb(ga if fl is None else cv2.flip(ga, fl))
        if da is None or db is None or len(da) < 8 or len(db) < 8:
            continue
        good = [m[0] for m in bf.knnMatch(da, db, k=2) if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
        if len(good) < 8:
            continue
        qa = ka[[g.queryIdx for g in good]]
        M, inl = cv2.estimateAffinePartial2D(qa, kb[[g.trainIdx for g in good]], method=cv2.RANSAC,
                                             ransacReprojThreshold=6.0)
        if M is not None and int(inl.sum()) > best[0]:
            pts = qa[inl.ravel() == 1].copy()  # undo the flip: points in a's own 480 px frame
            if fl in (1, -1):
                pts[:, 0] = w - 1 - pts[:, 0]
            if fl in (0, -1):
                pts[:, 1] = h - 1 - pts[:, 1]
            best = (int(inl.sum()), fl, M, pts)
    if best[2] is None:
        return 0, None, None, []
    return best[0], best[1], best[2].tolist(), np.round(best[3]).astype(int).tolist()


def local_support(box, size_a, pts):
    """Inlier keypoints inside a label box (expanded by 10% of its size, at least 8 px at 480): the defect itself matched."""
    s = 480 / max(size_a)
    x1, y1, x2, y2 = np.asarray(box, float) * s
    mx, my = max(0.1 * (x2 - x1), 8), max(0.1 * (y2 - y1), 8)
    p = np.asarray(pts, float).reshape(-1, 2)
    return int(((p[:, 0] >= x1 - mx) & (p[:, 0] <= x2 + mx) & (p[:, 1] >= y1 - my) & (p[:, 1] <= y2 + my)).sum())


def move(pts, size_a, size_b, fl, M):
    """Points (n,2) in a's pixels -> b's pixels: scale to the 480 px matching size, flip, transform, unscale."""
    sa, sb = 480 / max(size_a), 480 / max(size_b)
    ha, wa = round(size_a[0] * sa), round(size_a[1] * sa)
    c = np.asarray(pts, float) * sa
    if fl in (1, -1):
        c[:, 0] = wa - 1 - c[:, 0]
    if fl in (0, -1):
        c[:, 1] = ha - 1 - c[:, 1]
    return (np.c_[c, np.ones(len(c))] @ np.array(M).T) / sb


def project(boxes, size_a, size_b, fl, M):
    """Boxes (n,4) in a's pixels -> axis-aligned boxes in b's pixels."""
    out = []
    for x1, y1, x2, y2 in boxes:
        c = move([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], size_a, size_b, fl, M)
        out.append([c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max()])
    return np.array(out).reshape(-1, 4)


def view_coverage(size_a, size_b, fl, M):
    """Share of image b covered by image a after the transform (>= SAME_VIEW: the same view, e.g. an augmented copy)."""
    H, W = size_a
    c = move([[0, 0], [W, 0], [W, H], [0, H]], size_a, size_b, fl, M)
    c[:, 0], c[:, 1] = c[:, 0].clip(0, size_b[1]), c[:, 1].clip(0, size_b[0])
    return float(cv2.contourArea(c.astype(np.float32)) / (size_b[0] * size_b[1]))


def frac_inside(box, H, W):
    x1, y1, x2, y2 = box
    inter = max(0, min(x2, W) - max(x1, 0)) * max(0, min(y2, H) - max(y1, 0))
    return inter / max((x2 - x1) * (y2 - y1), 1e-9)


def classify(c, box, gb):
    """Outcome of one projected label against b's labels."""
    if not len(gb):
        return "missing"
    iou = ea.iou_matrix(box[None], gb[:, 1:])[0]
    same = gb[:, 0] == c
    if (iou[same] >= 0.5).any():
        return "agree"
    if (iou[~same] >= 0.5).any():
        return "class_conflict"
    gs = gb[same][:, 1:]
    if len(gs) >= 2:
        # covered by several same-class boxes, or covering several of them
        S = 400
        x1, y1, x2, y2 = box
        sx, sy = S / max(x2 - x1, 1e-6), S / max(y2 - y1, 1e-6)
        m = np.zeros((S, S), bool)
        hit = 0
        for b in gs:
            bx1, by1 = int(np.clip((b[0] - x1) * sx, 0, S)), int(np.clip((b[1] - y1) * sy, 0, S))
            bx2, by2 = int(np.clip((b[2] - x1) * sx, 0, S)), int(np.clip((b[3] - y1) * sy, 0, S))
            if bx2 > bx1 and by2 > by1:
                m[by1:by2, bx1:bx2] = True
                hit += 1
        inside = ((np.minimum(gs[:, 2], x2) - np.maximum(gs[:, 0], x1)).clip(0) *
                  (np.minimum(gs[:, 3], y2) - np.maximum(gs[:, 1], y1)).clip(0)) / \
            np.maximum((gs[:, 2] - gs[:, 0]) * (gs[:, 3] - gs[:, 1]), 1e-9)
        if (hit >= 2 and m.mean() >= 0.5) or (inside >= 0.5).sum() >= 2:
            return "split"
    cover = 0.0
    if len(gb):
        x1, y1, x2, y2 = box
        inter = ((np.minimum(gb[:, 3], y2) - np.maximum(gb[:, 1], y1)).clip(0) *
                 (np.minimum(gb[:, 2], x2) - np.maximum(gb[:, 0], x1)).clip(0)).sum()
        cover = inter / max((x2 - x1) * (y2 - y1), 1e-9)
    if iou.max() < 0.1 and cover < 0.2:
        return "missing"
    return "loose"


def sheet(examples, items, names, path, n=12, size=256):
    """Pairs side by side: a with the label (green), b with the projection (yellow) and b's labels (red)."""
    tiles = []
    for ia, ib, c, box_a, box_b in examples[:n]:
        row = []
        for idx, boxes, col in ((ia, [box_a], (0, 220, 0)), (ib, [box_b], (0, 220, 255))):
            im = cv2.imread(items[idx][0])
            if idx == ib:
                for g in items[ib][2]:
                    cv2.rectangle(im, tuple(map(int, g[1:3])), tuple(map(int, g[3:5])), (0, 0, 255), 3)
                    cv2.putText(im, names[int(g[0])][:10], (int(g[1]), int(g[2]) + 30), 0, 1.0, (0, 0, 255), 2)
            for b in boxes:
                cv2.rectangle(im, tuple(map(int, b[:2])), tuple(map(int, b[2:])), col, 4)
            cv2.putText(im, names[c][:12], (10, 40), 0, 1.3, col, 3)
            s = size / max(im.shape[:2])
            im = cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)))
            pad = np.full((size, size, 3), 255, np.uint8)
            pad[:im.shape[0], :im.shape[1]] = im
            row.append(pad)
        tiles.append(np.hstack(row + [np.full((size, 8, 3), 255, np.uint8)]))
    if not tiles:
        return
    cols = 3
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    cv2.imwrite(path, np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", help="YOLO-layout dataset root with <split>/images and <split>/labels folders")
    ap.add_argument("--name", required=True)
    ap.add_argument("--cache", default=os.path.join(ROOT, ".dataset_cache"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-inliers", type=int, default=B.MIN_INLIERS)
    args = ap.parse_args()

    out = os.path.join(ROOT, "runs", "analysis", "novel", "label_audit", args.name)
    os.makedirs(out, exist_ok=True)
    names = yaml.safe_load(open(os.path.join(args.data, "data.yaml")))["names"]
    items = load(args.data)
    paths = [p for p, _, _ in items]
    print(f"{len(items)} images, {sum(len(g) for _, _, g in items)} labels")
    E = B.dinov2_embeddings(paths, os.path.join(args.cache, "dinov2_vits14.npz"))
    nn_idx, _ = B.knn(E, min(B.KNN, len(paths) - 1))
    pairs = sorted({(min(a, int(b)), max(a, int(b))) for a in range(len(paths)) for b in nn_idx[a] if a != int(b)})

    tcache = os.path.join(out, "transforms_v2.json")
    T = json.load(open(tcache)) if os.path.exists(tcache) else {}
    jobs = sorted({(x, y) for a, b in pairs for x, y in ((a, b), (b, a)) if f"{x} {y}" not in T})
    print(f"{len(pairs)} candidate pairs, {len(jobs)} directed transforms to estimate")
    with Pool(args.workers) as pool:
        for k, r in enumerate(pool.imap(transform_job, [(paths[x], paths[y]) for x, y in jobs], chunksize=16)):
            T[f"{jobs[k][0]} {jobs[k][1]}"] = r
            if (k + 1) % 2000 == 0:
                print(f"  transforms {k + 1}/{len(jobs)}", flush=True)
                json.dump(T, open(tcache, "w"))
    json.dump(T, open(tcache, "w"))

    counts = {st: collections.Counter() for st in STRATA}
    per_class = {st: collections.defaultdict(collections.Counter) for st in STRATA}
    examples = {st: collections.defaultdict(list) for st in STRATA}
    n_pairs = collections.Counter()
    with open(os.path.join(out, "pairs.csv"), "w") as f:
        f.write("image_a,image_b,inliers,flip,view_coverage,stratum," + ",".join(OUTCOMES) + "\n")
        for a, b in pairs:
            for x, y in ((a, b), (b, a)):
                inl, fl, M, pts = T[f"{x} {y}"]
                if inl < args.min_inliers or M is None:
                    continue
                (pa, sa, ga), (pb, sb, gb) = items[x], items[y]
                cov = view_coverage(sa, sb, fl, M)
                st = "same_view" if cov >= B.SAME_VIEW else "other_view"
                n_pairs[st] += 1
                proj = project(ga[:, 1:], sa, sb, fl, M) if len(ga) else np.zeros((0, 4))
                row = collections.Counter()
                for (c, *_), box_a, box in zip(ga, ga[:, 1:], proj):
                    if frac_inside(box, *sb) < 0.8 or local_support(box_a, sa, pts) < LOCAL_MIN:
                        continue
                    o = classify(int(c), box, gb)
                    row[o] += 1
                    counts[st][o] += 1
                    per_class[st][names[int(c)]][o] += 1
                    examples[st][o].append((x, y, int(c), box_a, box))
                f.write(f"{os.path.basename(pa)},{os.path.basename(pb)},{inl},{fl},{cov:.3f},{st}," +
                        ",".join(str(row[o]) for o in OUTCOMES) + "\n")
    rng = np.random.default_rng(0)
    for st in STRATA:
        for o in OUTCOMES:
            ex = examples[st][o]
            sheet([ex[i] for i in rng.permutation(len(ex))], items, names, os.path.join(out, f"examples_{st}_{o}.jpg"))

    report = dict(data=args.data, images=len(items), candidate_pairs=len(pairs), strata={})
    md = [f"# Cross-view label audit: {args.name}", "",
          f"{len(items)} images, {len(pairs)} candidate pairs. Directed pairs with >= {args.min_inliers} inliers: "
          + ", ".join(f"{st} {n_pairs[st]}" for st in STRATA) + f" (same view = moved frame covers >= {B.SAME_VIEW:.0%} "
          "of the other image: exact alignment, differences are annotation differences).", ""]
    for st in STRATA:
        total = sum(counts[st].values())
        report["strata"][st] = dict(directed_pairs=n_pairs[st], projected_labels=total,
                                    outcomes={o: counts[st][o] for o in OUTCOMES},
                                    per_class={c: {o: v[o] for o in OUTCOMES} for c, v in sorted(per_class[st].items())})
        md += [f"## {st.replace('_', ' ')} ({total} projected labels)", "", "| class | " + " | ".join(OUTCOMES) +
               " | agreement |", "|---|" + "---|" * (len(OUTCOMES) + 1)]
        for c, v in sorted(per_class[st].items()):
            t = sum(v.values())
            md.append(f"| {c} | " + " | ".join(str(v[o]) for o in OUTCOMES) + f" | {v['agree'] / max(t, 1):.1%} |")
        md.append("| **all** | " + " | ".join(f"{counts[st][o]} ({counts[st][o] / max(total, 1):.1%})" for o in OUTCOMES)
                  + f" | **{counts[st]['agree'] / max(total, 1):.1%}** |")
        md.append("")
    json.dump(report, open(os.path.join(out, "report.json"), "w"), indent=2)
    open(os.path.join(out, "report.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
