"""Merge Wind Surface Defect + WTBlade-Defect into one deduplicated, group-split YOLO dataset.

Both sources contain augmented copies of the same photos (Wind Surface: `<name>_0`/`<name>_1` twins;
WTBlade: several Roboflow exports `<stem>_jpg.rf.<hash>` per source image, many with cutouts or
rotations). A random split therefore leaks near-identical images across splits. This script:

1. maps labels to the 5 Wind Surface classes and cleans boxes,
2. finds duplicates by name and by content (DINOv2 retrieval + ORB/RANSAC geometric verification,
   flip-aware), keeps one image per duplicate cluster,
3. groups everything that must not be separated (duplicates, overlapping views, tiles of one drone
   photo) and splits whole groups 70/15/15, stratified by class,
4. audits the result: no val/test image may match a train image.

Usage:
    python tools/build_dataset.py --ws WindSurface-Defect --wt WTBlade-Defect --out WindSurface-Defect-v3
"""

import argparse
import collections
import csv
import hashlib
import json
import os
import re
import shutil
from multiprocessing import Pool

import cv2
import numpy as np

# Blade-surface damage taxonomy as used in industrial blade inspection. Ids 0-4 are the Wind Surface Defect
# classes (corrosion, dirt, hide_craze, surface_eye, thunderstrike) under their industry names, so Wind Surface
# labels keep their ids and the paper-comparable five classes can be reported on their own.
CLASSES = ["leading_edge_erosion", "contamination", "crack", "pitting", "lightning_damage", "coating_damage"]
# WTBlade-Defect ids: burn, crack, deformity, dirt, oil, peeling, rust. Rust, oil and deformity are hub/nacelle
# hardware findings (bearing and bolt rust, oil leaks, seal deformation), not blade-surface damage: dropped.
WT_TO_WS = {0: 4, 1: 2, 2: None, 3: 1, 4: None, 5: 5, 6: None}
# Beijing wind-turbine (Roboflow beijing-university-9d61y/wind-turbine-ebl65), the collection Wind Surface Defect was
# cut from: corrosion, craze, hide_craze, surface_eye, surface_injure, surface_oil, thunderstrike.
BJ_TO_WS = {0: 0, 1: 2, 2: 2, 3: 3, 4: 5, 5: 1, 6: 4}
# WTBs2025 (provenance/licence unknown): oil leakage, paint cracks, localized damage, lightning strikes, surface stains,
# erosion, coating detachment, protective film damage, pinholes. Whole-blade photos downscaled to 640 px, so many
# defects are a few pixels wide: pinholes (median 1.5 px, one box per hole) and localized damage (5 px) are not
# learnable and conflict with our region-level pitting boxes. Images containing them are dropped entirely, as they
# would otherwise hold real, unlabelled blade defects. Oil leakage is a hub/bearing fault (as for WTBlade).
WTBS_TO_WS = {0: None, 1: 2, 2: None, 3: 4, 4: 1, 5: 0, 6: 5, 7: 5, 8: None}
SOURCES = {
    "ws": "Wind Surface Defect (Liu & Liu, Appl. Soft Comput. 2025)",
    "wt": "WTBlade-Defect / fengChe (Roboflow detr-swsa0/fengche-evxno, CC BY 4.0)",
    "bj": "Beijing wind-turbine (Roboflow beijing-university-9d61y/wind-turbine-ebl65, CC BY 4.0)",
    "wtbs": "WTBs2025 (provenance and licence unknown; filtered, see below)",
}
# Per-source collection rules.
#   class_map       source class id -> taxonomy id (None: box dropped; images stay, the class is not a blade defect)
#   drop_image_ids  source classes that are blade defects we cannot use: images containing them are dropped
#   min_box_px      images with any box whose shorter side is below this are dropped (unlearnable at 640 px)
#   skip_name       file-name pattern of images to exclude (synthetic, GAN-generated images)
#   twins           Wind Surface stores augmented twins as <name>_0 / <name>_1
SOURCE_RULES = {
    "ws": dict(class_map=None, twins=True),
    "wt": dict(class_map=WT_TO_WS),
    "bj": dict(class_map=BJ_TO_WS),
    "wtbs": dict(class_map=WTBS_TO_WS, drop_image_ids={2, 8}, min_box_px=16, skip_name=r"GAN"),
}

KNN = 10  # retrieval candidates per image
MIN_INLIERS = 25  # ORB/RANSAC inliers for a verified content match (random pairs: >=25 in ~1%, mostly true dups)
SAME_VIEW = 0.7  # verified match covering this much of the other image = duplicate, not just overlap
DINO_DUP = 0.97  # cosine above which two images are duplicates even without keypoints (copies, consecutive frames)
DINO_GROUP = 0.93  # cosine above which two images show the same scene (augmented view, neighbouring frame): same split
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}


# ----------------------------------------------------------------------------------------- items
def find_image(img_dir, stem):
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".PNG"):
        p = os.path.join(img_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


def read_boxes(label_path, class_map=None):
    boxes = []
    if os.path.exists(label_path):
        for line in open(label_path):
            v = line.split()
            if len(v) != 5:
                continue
            c = int(v[0])
            c = class_map.get(c) if class_map is not None else c
            if c is not None:
                boxes.append((c, *map(float, v[1:])))
    return boxes


def clean_boxes(boxes, img):
    """Clip to the image, drop degenerate boxes and boxes lying (almost) entirely on black padding."""
    H, W = img.shape[:2]
    out, dropped = [], 0
    for c, x, y, w, h in boxes:
        x1, y1, x2, y2 = max(0.0, x - w / 2), max(0.0, y - h / 2), min(1.0, x + w / 2), min(1.0, y + h / 2)
        px1, py1, px2, py2 = int(x1 * W), int(y1 * H), int(np.ceil(x2 * W)), int(np.ceil(y2 * H))
        if px2 - px1 < 2 or py2 - py1 < 2:
            dropped += 1
            continue
        if (img[py1:py2, px1:px2].max(axis=2) < 10).mean() > 0.9:
            dropped += 1
            continue
        row = (c, round((x1 + x2) / 2, 6), round((y1 + y2) / 2, 6), round(x2 - x1, 6), round(y2 - y1, 6))
        if row not in out:
            out.append(row)
    return out, dropped


def source_stem(stem):
    """Roboflow names exports `<source>_jpg.rf.<hash>`; every export of one source image shares <source>.
    Sources re-exported with format suffixes (`<tile>_png_jpg`, `<tile>_JPG_jpg`) keep the same <source>."""
    return re.sub(r"(_(png|jpe?g))+$", "", re.split(r"\.rf\.", stem)[0], flags=re.IGNORECASE).lstrip("_")


TILE_NAMES = (
    r"^(DJI_\d+)_\d+_\d+(?:_\d+)?$",  # DJI_<photo>_<row>_<col>[_<n>]
    r"^(\d+_\d+)_\d+_\d+_\d+_\d+_\d+_(\d+)_(\d+)$",  # <set>_<photo>_<x>_<y>_<w>_<h>_<?>_<W>_<H>
    r"^(\d+)_\d+_\d+_\d+_\d+_\d+_(\d+)_(\d+)$",  # <photo>_<x>_<y>_<w>_<h>_<?>_<W>_<H>
)


def photo_key(stem):
    """Parent photo of a tile, from tile names that encode it (overlapping crops of one large image)."""
    for pattern in TILE_NAMES:
        m = re.match(pattern, stem)
        if m:
            return "photo:" + "_".join(m.groups())
    return None


def dup_key(src, source):
    """Wind Surface and Beijing share one image collection: their tile names identify the same source image across
    datasets. Other names (plain numbers, Roboflow stems) are only meaningful within their own dataset."""
    return ("tile:" if photo_key(source) else f"{src}:") + source


def roboflow_split_dirs(root):
    """(images dir, labels dir) for every split of a Roboflow-style export (<split>/images or <split>/<split>/images)."""
    for split in sorted(os.listdir(root)):
        for img_dir in (f"{root}/{split}/{split}/images", f"{root}/{split}/images"):
            if os.path.isdir(img_dir):
                yield img_dir, os.path.join(os.path.dirname(img_dir), "labels")
                break


def collect(roots):
    """roots: {source: dataset root}; sources without a root are skipped."""
    items, stats = [], collections.Counter()
    for src, root in roots.items():
        if not root:
            continue
        rules = SOURCE_RULES[src]
        for img_dir, lab_dir in roboflow_split_dirs(root):
            for f in sorted(os.listdir(img_dir)):
                stem = os.path.splitext(f)[0]
                if rules.get("skip_name") and re.search(rules["skip_name"], stem):
                    stats[f"{src} images skipped (synthetic)"] += 1
                    continue
                raw = read_boxes(os.path.join(lab_dir, stem + ".txt"))
                if rules.get("drop_image_ids") and {b[0] for b in raw} & rules["drop_image_ids"]:
                    stats[f"{src} images dropped (unusable blade-defect class present)"] += 1
                    continue
                source = source_stem(re.sub(r"_[01]$", "", stem) if rules.get("twins") else stem)
                boxes = read_boxes(os.path.join(lab_dir, stem + ".txt"), rules["class_map"]) if rules["class_map"] else raw
                stats[f"{src} boxes dropped (class not in taxonomy)"] += len(raw) - len(boxes)
                items.append(dict(src=src, path=os.path.join(img_dir, f), stem=stem, dup_key=dup_key(src, source),
                                  group_key=photo_key(source), boxes=boxes, min_box_px=rules.get("min_box_px", 0)))
    return items, stats


def load_and_clean(item):
    img = cv2.imread(item["path"], cv2.IMREAD_COLOR)
    boxes, dropped = clean_boxes(item["boxes"], img)
    H, W = img.shape[:2]
    too_small = bool(item.get("min_box_px")) and any(min(w * W, h * H) < item["min_box_px"] for _, _, _, w, h in boxes)
    if too_small:  # drop the whole image: removing only the small boxes would leave visible defects unlabelled
        boxes = []
    small = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    return dict(boxes=boxes, dropped=dropped, md5=hashlib.md5(img.tobytes()).hexdigest(),
                black=float((small.max(axis=2) < 10).mean()), size=img.shape[1::-1], too_small=too_small)


# ------------------------------------------------------------------------------- content matching
def dinov2_embeddings(paths, cache):
    """L2-normalised DINOv2 ViT-S/14 [CLS, mean patch] embeddings, cached per file path."""
    known = {}
    if os.path.exists(cache):
        z = np.load(cache)
        known = dict(zip(z["paths"].tolist(), z["emb"]))
    todo = [p for p in paths if p not in known]
    if todo:
        import torch

        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        with torch.no_grad():
            for i in range(0, len(todo), 64):
                batch = todo[i:i + 64]
                x = torch.stack([
                    (torch.from_numpy(cv2.resize(cv2.imread(p)[:, :, ::-1], (224, 224), interpolation=cv2.INTER_CUBIC).copy())
                     .permute(2, 0, 1).float() / 255 - mean) / std
                    for p in batch
                ])
                f = model.forward_features(x)
                e = torch.nn.functional.normalize(torch.cat([f["x_norm_clstoken"], f["x_norm_patchtokens"].mean(1)], 1), dim=1)
                known.update(zip(batch, e.numpy().astype(np.float16)))
                print(f"  embedded {min(i + 64, len(todo))}/{len(todo)}", end="\r", flush=True)
        print()
        np.savez(cache, paths=np.array(list(known)), emb=np.stack(list(known.values())))
    return np.stack([known[p] for p in paths]).astype(np.float32)


def knn(E, k):
    idx, sim = np.zeros((len(E), k), int), np.zeros((len(E), k), np.float32)
    for i in range(0, len(E), 2048):
        s = E[i:i + 2048] @ E.T
        s[np.arange(len(s)), np.arange(i, i + len(s))] = -1
        j = np.argpartition(-s, k, axis=1)[:, :k]
        idx[i:i + 2048], sim[i:i + 2048] = j, np.take_along_axis(s, j, 1)
    return idx, sim


_ORB, _BF = None, None


def _orb(gray):
    global _ORB
    _ORB = _ORB or cv2.ORB_create(1500)
    k, d = _ORB.detectAndCompute(gray, None)
    return (np.float32([p.pt for p in k]) if k else np.zeros((0, 2), np.float32)), d


def _gray(path):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    s = 480 / max(g.shape)
    return cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def verify_job(args):
    """Match image a (and its 3 flips) against each candidate b. Returns [(b, inliers, coverage of b)]."""
    global _BF
    _BF = _BF or cv2.BFMatcher(cv2.NORM_HAMMING)
    a_path, cands = args
    ga = _gray(a_path)
    fa = [_orb(ga if fl is None else cv2.flip(ga, fl)) + (ga.shape,) for fl in (None, 1, 0, -1)]
    res = []
    for b, b_path in cands:
        gb = _gray(b_path)
        kb, db = _orb(gb)
        best = (0, 0.0)
        for ka, da, (h, w) in fa:
            if da is None or db is None or len(da) < 8 or len(db) < 8:
                continue
            good = [m[0] for m in _BF.knnMatch(da, db, k=2) if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
            if len(good) < 8:
                continue
            M, inl = cv2.estimateAffinePartial2D(ka[[g.queryIdx for g in good]], kb[[g.trainIdx for g in good]],
                                                 method=cv2.RANSAC, ransacReprojThreshold=6.0)
            if M is None or int(inl.sum()) <= best[0]:
                continue
            c = cv2.transform(np.float32([[0, 0], [w, 0], [w, h], [0, h]])[None], M)[0]
            c[:, 0], c[:, 1] = c[:, 0].clip(0, gb.shape[1]), c[:, 1].clip(0, gb.shape[0])
            best = (int(inl.sum()), float(cv2.contourArea(c) / (gb.shape[0] * gb.shape[1])))
        res.append((b, *best))
    return res


def verify_pairs(paths, pairs, workers):
    by_a = collections.defaultdict(list)
    for a, b in pairs:
        by_a[a].append((b, paths[b]))
    jobs = [(paths[a], c) for a, c in by_a.items()]
    out = {}
    with Pool(workers) as pool:
        for n, (a, res) in enumerate(zip(by_a, pool.imap(verify_job, jobs, chunksize=8))):
            for b, inl, cov in res:
                out[(a, b)] = (inl, cov)
            if n % 200 == 0:
                print(f"  verified {n}/{len(jobs)} anchors", end="\r", flush=True)
    print()
    return out


# ---------------------------------------------------------------------------------- union-find
class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)

    def groups(self):
        g = collections.defaultdict(list)
        for i in range(len(self.p)):
            g[self.find(i)].append(i)
        return list(g.values())


# ------------------------------------------------------------------------------------- splitting
def split_groups(counts, sizes, seed, trials=1000):
    """Assign whole groups to splits so every class and the image count are as close to SPLITS as possible.

    Greedy assignment (each group goes to the split furthest below its target for that group's classes),
    repeated over random group orders; the assignment with the smallest total deviation wins.
    """
    rng = np.random.default_rng(seed)
    names, ratio = list(SPLITS), np.array(list(SPLITS.values()))
    feats = np.concatenate([counts, sizes[:, None]], 1).astype(float)  # per-class boxes + image count
    total = feats.sum(0) + 1e-9
    best, best_score = None, np.inf
    for _ in range(trials):
        have = np.zeros((len(names), feats.shape[1]))
        assign = np.zeros(len(feats), int)
        for g in rng.permutation(len(feats)):
            w = feats[g] / total
            deficit = ((ratio[:, None] - have / total) * w).sum(1)
            assign[g] = s = int(np.argmax(deficit))
            have[s] += feats[g]
        score = np.abs(have / total - ratio[:, None]).sum()
        if score < best_score:
            best, best_score = assign.copy(), score
    return [names[k] for k in best]


# ------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ws", default="WindSurface-Defect")
    ap.add_argument("--wt", default="WTBlade-Defect")
    ap.add_argument("--bj", default="wind-turbine.v1i.yolov12", help="Beijing wind-turbine export ('' to skip)")
    ap.add_argument("--wtbs", default="WTBs2025", help="WTBs2025 export ('' to skip)")
    ap.add_argument("--out", default="WindSurface-Defect-v3")
    ap.add_argument("--cache", default=".dataset_cache")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    os.makedirs(args.cache, exist_ok=True)

    print("1/6 collecting and cleaning labels")
    items, stats = collect({"ws": args.ws, "wt": args.wt, "bj": args.bj, "wtbs": args.wtbs})
    with Pool(args.workers) as pool:
        for it, info in zip(items, pool.map(load_and_clean, items, chunksize=32)):
            it.update(info)
            stats[f"{it['src']} boxes dropped (degenerate/on padding)"] += info["dropped"]
            stats[f"{it['src']} images dropped (a box below {it['min_box_px']} px)"] += info["too_small"]
    stats["images in"] = len(items)
    for src in SOURCES:
        stats[f"{src} images in"] = sum(it["src"] == src for it in items)
    stats["images dropped (no target-class boxes)"] = sum(not it["boxes"] for it in items)
    items = [it for it in items if it["boxes"]]
    paths = [it["path"] for it in items]
    N = len(items)

    print("2/6 retrieval embeddings (DINOv2 ViT-S/14)")
    E = dinov2_embeddings(paths, os.path.join(args.cache, "dinov2_vits14.npz"))
    nn_idx, nn_sim = knn(E, KNN)

    print("3/6 geometric verification of candidate pairs (ORB + RANSAC, flip-aware)")
    # Cached by file path, so a rebuild with a different image set only verifies new pairs.
    vcache = os.path.join(args.cache, "verify.json")
    known = json.load(open(vcache)) if os.path.exists(vcache) else {}
    index = {p: i for i, p in enumerate(paths)}
    verified = {}
    for k, v in known.items():
        a, b = k.split("\t")
        if a in index and b in index:
            verified[(min(index[a], index[b]), max(index[a], index[b]))] = tuple(v)

    def verify(pairs):
        todo = sorted({(min(a, b), max(a, b)) for a, b in pairs} - verified.keys())
        if todo:
            new = verify_pairs(paths, todo, args.workers)
            verified.update(new)
            known.update({f"{paths[a]}\t{paths[b]}": v for (a, b), v in new.items()})
            json.dump(known, open(vcache, "w"))

    verify((a, int(b)) for a in range(N) for b in nn_idx[a])

    print("4/6 duplicates and split groups")
    dup_edges, grp = set(), DSU(N)
    by_key = collections.defaultdict(list)
    for i, it in enumerate(items):
        by_key[it["dup_key"]].append(i)
        by_key["md5:" + it["md5"]].append(i)
        if it["group_key"]:
            by_key[it["group_key"]].append(i)
    for k, idx in by_key.items():
        for x, a in enumerate(idx):
            grp.union(idx[0], a)
            if not k.startswith("photo:"):  # tiles of one photo are distinct images, only grouped
                dup_edges.update((a, b) for b in idx[x + 1:])
    n_match = 0
    for (a, b), (inl, cov) in verified.items():
        if inl >= MIN_INLIERS:
            grp.union(a, b)
            n_match += 1
            if cov >= SAME_VIEW:
                dup_edges.add((a, b))
    for a in range(N):
        for b, s in zip(nn_idx[a], nn_sim[a]):
            if s >= DINO_GROUP:
                grp.union(a, int(b))
            if s >= DINO_DUP:
                dup_edges.add((min(a, int(b)), max(a, int(b))))
    stats["verified content matches (same scene)"] = n_match
    stats["direct duplicate pairs (name, md5, same view, embedding)"] = len(dup_edges)

    # Remove duplicates without chaining: walk images best-first and keep one only if it is not a direct
    # duplicate of an image already kept. Consecutive video frames that are each near-identical to the
    # next therefore keep every frame that differs from the kept ones, instead of collapsing to one.
    adj = collections.defaultdict(set)
    for a, b in dup_edges:
        adj[a].add(b)
        adj[b].add(a)
    quality = sorted(range(N), key=lambda i: (items[i]["black"], -len(items[i]["boxes"]), items[i]["src"] != "ws", paths[i]))
    keep_set = set()
    for i in quality:
        if not adj[i] & keep_set:
            keep_set.add(i)
    keep = sorted(keep_set)
    stats["unique images kept"] = len(keep)
    stats["duplicates removed"] = N - len(keep)

    print("5/6 stratified group split + leakage audit")
    for rnd in range(1, 13):
        groups = [[i for i in g if i in keep_set] for g in grp.groups()]
        groups = [g for g in groups if g]
        counts = np.array([np.bincount([b[0] for i in g for b in items[i]["boxes"]], minlength=len(CLASSES)) for g in groups])
        assign = split_groups(counts, np.array([len(g) for g in groups]), args.seed)
        split_of = {i: assign[g] for g, members in enumerate(groups) for i in members}
        group_of = {i: g for g, members in enumerate(groups) for i in members}

        # Audit: verify every held-out image against its 10 most similar train images.
        train = np.array([i for i in keep if split_of[i] == "train"])
        held = np.array([i for i in keep if split_of[i] != "train"])
        sims = E[held] @ E[train].T
        top = train[np.argsort(-sims, axis=1)[:, :10]]
        pairs = [(int(a), int(b)) for a, bs in zip(held, top) for b in bs]
        verify(pairs)
        leaks = [(a, b) for a, b in pairs if verified[(min(a, b), max(a, b))][0] >= MIN_INLIERS]
        print(f"  round {rnd}: {len(groups)} groups, {len(leaks)} held-out/train matches")
        if not leaks:
            break
        for a, b in leaks:
            grp.union(a, b)
    stats["split groups"] = len(groups)
    stats["largest group (images)"] = max(len(g) for g in groups)
    stats["audit: held-out vs train pairs verified"] = len(pairs)
    stats["audit: verified matches into train"] = len(leaks)
    stats["audit: max embedding cosine held-out vs train"] = round(float(sims.max()), 4)

    print("6/6 writing dataset")
    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    for s in SPLITS:
        os.makedirs(f"{args.out}/images/{s}")
        os.makedirs(f"{args.out}/labels/{s}")
    rows, used = [], set()
    for i in keep:
        it, s = items[i], split_of[i]
        name = f"{it['src']}_{re.sub(r'[^A-Za-z0-9_-]+', '_', it['stem'])}"[:120]
        while name in used:
            name += "_"
        used.add(name)
        ext = os.path.splitext(it["path"])[1].lower()
        shutil.copy2(it["path"], f"{args.out}/images/{s}/{name}{ext}")
        with open(f"{args.out}/labels/{s}/{name}.txt", "w") as f:
            f.writelines(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in it["boxes"])
        rows.append(dict(file=f"images/{s}/{name}{ext}", split=s, group=group_of[i], source=it["src"],
                         original=os.path.relpath(it["path"]), width=it["size"][0], height=it["size"][1],
                         boxes=len(it["boxes"])))
    with open(f"{args.out}/manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(f"{args.out}/data.yaml", "w") as f:
        f.write("train: images/train\nval: images/val\ntest: images/test\n")
        f.write(f"nc: {len(CLASSES)}\nnames: {json.dumps(CLASSES)}\n")

    leak_paths = [(paths[a], paths[b], verified[(min(a, b), max(a, b))][0]) for a, b in leaks]
    write_report(args.out, stats, rows, counts_by_split(rows, args.out), leak_paths)
    print(json.dumps(stats, indent=2))
    if leaks:
        print(f"WARNING: {len(leaks)} held-out images still match train images; see {args.out}/README.md")


def counts_by_split(rows, out):
    c = {s: np.zeros(len(CLASSES), int) for s in SPLITS}
    groups = {s: [set() for _ in CLASSES] for s in SPLITS}  # independent split groups containing each class
    n = collections.Counter(r["split"] for r in rows)
    src = collections.Counter((r["split"], r["source"]) for r in rows)
    for r in rows:
        stem = os.path.splitext(os.path.basename(r["file"]))[0]
        for line in open(f"{out}/labels/{r['split']}/{stem}.txt"):
            k = int(line.split()[0])
            c[r["split"]][k] += 1
            groups[r["split"]][k].add(r["group"])
    return c, n, src, groups


def write_report(out, stats, rows, split_counts, leaks):
    c, n, src, groups = split_counts
    lines = [
        "# Wind Surface Defect v3 (merged, deduplicated, group-split, blade-surface taxonomy)",
        "",
        "Built by `tools/build_dataset.py` from:",
        "",
        *[f"- `{k}`: {v}" for k, v in SOURCES.items()],
        "",
        "Classes (industrial blade-inspection taxonomy):",
        "",
        "| id | class | Wind Surface (ws) | WTBlade (wt) | Beijing (bj) | WTBs2025 (wtbs) |",
        "|---|---|---|---|---|---|",
        "| 0 | leading_edge_erosion | corrosion | - | corrosion | erosion |",
        "| 1 | contamination | dirt | dirt | surface_oil | surface stains |",
        "| 2 | crack | hide_craze | crack | craze, hide_craze | paint cracks |",
        "| 3 | pitting | surface_eye | - | surface_eye | - |",
        "| 4 | lightning_damage | thunderstrike | burn | thunderstrike | lightning strikes |",
        "| 5 | coating_damage | - | peeling | surface_injure | coating detachment, protective film damage |",
        "",
        "WTBlade rust, oil and deformity and WTBs2025 oil leakage (hub/nacelle hardware, not blade surface) are dropped;",
        "images left without boxes are removed. WTBs2025 rules: GAN-generated images excluded; images containing",
        "pinholes or localized damage (1.5 px / 5 px median boxes) dropped entirely; images with any box under 16 px",
        "dropped entirely. Ids 0-4 are the original Wind Surface classes (paper-comparable subset).",
        "",
        "## Split",
        "",
        "| split | images | " + " | ".join(f"from {k}" for k in SOURCES) + " | " + " | ".join(CLASSES) + " |",
        "|---" * (2 + len(SOURCES) + len(CLASSES)) + "|",
    ]
    for s in SPLITS:
        lines.append(f"| {s} | {n[s]} | " + " | ".join(str(src[(s, k)]) for k in SOURCES) + " | "
                     + " | ".join(str(v) for v in c[s]) + " |")
    missing = [f"{CLASSES[k]} ({s})" for s in SPLITS for k in range(len(CLASSES)) if c[s][k] == 0]
    if missing:
        lines += ["", f"**Classes absent from a split: {', '.join(missing)}.** All boxes of such a class come from",
                  "images in a single split group (one photo/session), so it cannot appear on both sides without",
                  "leakage. Its AP is not measured on that split and does not enter that split's mAP."]
    lines += ["", "## Independence per class", "",
              "Boxes / independent split groups (a group is one photo, session or set of overlapping views). Few groups",
              "means few independent examples, whatever the box count.", "",
              "| class | " + " | ".join(SPLITS) + " |", "|---" * (1 + len(SPLITS)) + "|"]
    for k, name in enumerate(CLASSES):
        lines.append(f"| {name} | " + " | ".join(f"{c[s][k]} / {len(groups[s][k])}" for s in SPLITS) + " |")
    lines += ["", "## Build statistics", "", "| step | count |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in stats.items()]
    lines += ["", "`manifest.csv` maps every image to its source file and split group."]
    if leaks:
        lines += ["", "## Remaining matches into train", ""] + [f"- {a} ~ {b} ({i} inliers)" for a, b, i in leaks]
    open(f"{out}/README.md", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
