"""Audit every candidate dataset by content: how many legitimate examples each native class really has, and which
native classes of different datasets are the same defect concept.

Datasets are read with their own class names (no taxonomy mapping). Image matching is tools/build_dataset.py's (DINOv2
retrieval over all datasets together, flip-aware ORB/RANSAC verification, cached):
  duplicate   md5 / Roboflow source stem / embedding cosine >= DINO_DUP / verified match covering >= SAME_VIEW
  same scene  verified match (>= MIN_INLIERS) or embedding cosine >= DINO_GROUP or same parent photo (tile names)
Per dataset and class:
  raw images / boxes; unique images (one per within-dataset duplicate cluster); independent scenes (same-scene groups,
  the unit that matters for generalisation); learnable boxes (on unique images, shorter side >= 10 px and sqrt(area)
  >= 16 px at the 640 input) and the scenes containing one.
Class evidence:
  1. label agreement on shared images: an image present in two datasets (duplicate across datasets), boxes matched at
     IoU >= 0.5 -> which class each dataset gave the same defect (direct evidence of equivalence)
  2. defect-crop retrieval: DINOv2 embeddings of learnable box crops (15% context); for every crop its 10 nearest
     crops from other scenes: within its dataset (class purity = consistency/separability of the class) and in
     other datasets (which of their classes it resembles)
Usage:
    python tools/audit_datasets.py
Writes runs/analysis/datasets/{report.md, report.json}.
"""

import argparse
import collections
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import build_dataset as B  # noqa: E402

DATASETS = {  # key: (root, description)
    "ws": ("WindSurface-Defect", "Wind Surface Defect (Appl. Soft Comput. 2025)"),
    "wt": ("WTBlade-Defect", "WTBlade-Defect / fengChe (Roboflow, CC BY 4.0)"),
    "bj": ("wind-turbine.v1i.yolov12", "Beijing wind-turbine (Roboflow, CC BY 4.0)"),
    "bjc": ("wind_turbine_check.v1i.yolov12", "wind_turbine_check (Roboflow export, Beijing classes)"),
    "wtbs": ("WTBs2025", "WTBs2025 (Scientific Data 13:1310, 2026)"),
    "wtbd": ("external/WTBD-official", "WTBD (Scientific Data 13:396, 2026)"),
}
CROPS_PER_CLASS = 300


def class_names(root):
    import yaml

    n = yaml.safe_load(open(os.path.join(root, "data.yaml")))["names"]
    return [n[k] for k in sorted(n)] if isinstance(n, dict) else list(n)


def collect():
    items = []
    for key, (root, _) in DATASETS.items():
        names = class_names(root)
        for img_dir, lab_dir in B.roboflow_split_dirs(root):
            for f in sorted(os.listdir(img_dir)):
                stem = os.path.splitext(f)[0]
                lab = os.path.join(lab_dir, stem + ".txt")
                boxes = [b for b in B.read_boxes(lab) if 0 <= b[0] < len(names)] if os.path.exists(lab) else []
                src = B.source_stem(stem[:-2] if key == "ws" and stem[-2:] in ("_0", "_1") else stem)
                items.append(dict(ds=key, names=names, path=f"{img_dir}/{f}", stem=stem, boxes=boxes,
                                  synthetic=key == "wtbs" and ("GAN" in stem or stem.endswith("be")),
                                  dup_key=f"{key}:{src}", group_key=B.photo_key(src)))
    return items


def load(item):
    img = cv2.imread(item["path"], cv2.IMREAD_COLOR)
    boxes, _ = B.clean_boxes(item["boxes"], img)
    H, W = img.shape[:2]
    return dict(boxes=boxes, size=(W, H), md5=__import__("hashlib").md5(img.tobytes()).hexdigest())


def learnable(b, W, H):
    s = 640 / max(W, H)
    w, h = b[3] * W * s, b[4] * H * s
    return min(w, h) >= 10 and (w * h) ** 0.5 >= 16


def crop_embeddings(crops, cache, batch=32):
    """crops: list of (path, x1, y1, x2, y2) in pixels -> L2-normalised DINOv2 [CLS, mean patch] embeddings.
    CUDA when available; cached per crop and saved every 50 batches."""
    known = {}
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        known = dict(zip(z["keys"].tolist(), z["emb"]))
    key = lambda c: "%s|%d|%d|%d|%d" % c  # noqa: E731
    todo = [c for c in crops if key(c) not in known]
    if todo:
        import torch

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval().to(dev)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        def save():
            np.savez(cache, keys=np.array(list(known)), emb=np.stack(list(known.values())))

        print(f"  embedding {len(todo)} crops on {dev}")
        with torch.no_grad():
            for n, i in enumerate(range(0, len(todo), batch)):
                chunk = todo[i:i + batch]
                xs = []
                for p, x1, y1, x2, y2 in chunk:
                    im = cv2.imread(p)[y1:y2, x1:x2, ::-1]
                    xs.append((torch.from_numpy(cv2.resize(im, (224, 224), interpolation=cv2.INTER_CUBIC).copy())
                               .permute(2, 0, 1).float() / 255 - mean) / std)
                f = model.forward_features(torch.stack(xs).to(dev))
                e = torch.nn.functional.normalize(torch.cat([f["x_norm_clstoken"], f["x_norm_patchtokens"].mean(1)], 1), dim=1)
                known.update(zip([key(c) for c in chunk], e.cpu().numpy().astype(np.float16)))
                print(f"  crops embedded {min(i + batch, len(todo))}/{len(todo)}", end="\r", flush=True)
                if n % 50 == 49:
                    save()
        print()
        save()
    return np.stack([known[key(c)] for c in crops]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=os.path.join(ROOT, ".dataset_cache"))
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "analysis", "datasets"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.chdir(ROOT)  # relative paths, as tools/build_dataset.py caches them

    print("1/5 collecting")
    items = collect()
    from multiprocessing import Pool

    with Pool(args.workers) as pool:
        for it, info in zip(items, pool.map(load, items, chunksize=64)):
            it.update(info)
    paths = [it["path"] for it in items]
    N = len(items)
    print(f"  {N} images")

    print("2/5 image embeddings and matching")
    E = B.dinov2_embeddings(paths, os.path.join(args.cache, "dinov2_vits14.npz"))
    nn_idx, nn_sim = B.knn(E, B.KNN)
    vcache = os.path.join(args.cache, "verify.json")
    known = json.load(open(vcache)) if os.path.exists(vcache) else {}
    index = {p: i for i, p in enumerate(paths)}
    verified = {}
    for k, v in known.items():
        a, b = k.split("\t")
        if a in index and b in index:
            verified[(min(index[a], index[b]), max(index[a], index[b]))] = tuple(v)
    todo = sorted({(min(a, int(b)), max(a, int(b))) for a in range(N) for b in nn_idx[a]} - verified.keys())
    if todo:
        new = B.verify_pairs(paths, todo, args.workers)
        verified.update(new)
        known.update({f"{paths[a]}\t{paths[b]}": list(v) for (a, b), v in new.items()})
        json.dump(known, open(vcache, "w"))

    dup, grp = B.DSU(N), B.DSU(N)
    by_key = collections.defaultdict(list)
    for i, it in enumerate(items):
        by_key[it["dup_key"]].append(i)
        by_key["md5:" + it["md5"]].append(i)
        if it["group_key"]:
            by_key["g:" + it["group_key"]].append(i)
    for k, idx in by_key.items():
        for a in idx[1:]:
            grp.union(idx[0], a)
            if not k.startswith("g:"):
                dup.union(idx[0], a)
    for (a, b), (inl, cov) in verified.items():
        if inl >= B.MIN_INLIERS:
            grp.union(a, b)
            if cov >= B.SAME_VIEW:
                dup.union(a, b)
    for a in range(N):
        for b, s in zip(nn_idx[a], nn_sim[a]):
            if s >= B.DINO_GROUP:
                grp.union(a, int(b))
            if s >= B.DINO_DUP:
                dup.union(a, int(b))
    dup_of = [dup.find(i) for i in range(N)]
    grp_of = [grp.find(i) for i in range(N)]

    print("3/5 per-class statistics")
    ds_keys = list(DATASETS)
    rep = dict(datasets={}, overlap={}, agreement={}, retrieval={})
    unique = {}  # one representative per (dataset, duplicate cluster), the image with most boxes
    for i, it in enumerate(items):
        if it["synthetic"]:
            continue
        k = (it["ds"], dup_of[i])
        if k not in unique or len(it["boxes"]) > len(items[unique[k]]["boxes"]):
            unique[k] = i
    uniq = set(unique.values())
    for d in ds_keys:
        idx = [i for i, it in enumerate(items) if it["ds"] == d]
        names = items[idx[0]]["names"]
        rows = {}
        for c, name in enumerate(names):
            raw = [i for i in idx if any(b[0] == c for b in items[i]["boxes"])]
            u = [i for i in raw if i in uniq]
            lb = [(i, b) for i in u for b in items[i]["boxes"] if b[0] == c and learnable(b, *items[i]["size"])]
            sizes = [((b[3] * items[i]["size"][0]) * (b[4] * items[i]["size"][1])) ** 0.5 * 640 / max(items[i]["size"])
                     for i in u for b in items[i]["boxes"] if b[0] == c]
            rows[name] = dict(raw_images=len(raw), raw_boxes=sum(b[0] == c for i in raw for b in items[i]["boxes"]),
                              synthetic_images=sum(items[i]["synthetic"] for i in raw),
                              unique_images=len(u), scenes=len({grp_of[i] for i in u}),
                              learnable_boxes=len(lb), learnable_scenes=len({grp_of[i] for i, _ in lb}),
                              median_box_px=round(float(np.median(sizes)), 1) if sizes else None)
        n_u = sum(1 for i in idx if i in uniq)
        rep["datasets"][d] = dict(description=DATASETS[d][1], images=len(idx), unique_images=n_u,
                                  scenes=len({grp_of[i] for i in idx if i in uniq}), classes=rows)
    # cross-dataset overlap: images of d with a duplicate / same scene in d2
    ds_of = [it["ds"] for it in items]
    dup_members, grp_members = collections.defaultdict(set), collections.defaultdict(set)
    for i in range(N):
        dup_members[dup_of[i]].add(ds_of[i])
        grp_members[grp_of[i]].add(ds_of[i])
    for d in ds_keys:
        idx = [i for i in range(N) if ds_of[i] == d]
        rep["overlap"][d] = {d2: dict(duplicate=sum(d2 in dup_members[dup_of[i]] for i in idx),
                                      same_scene=sum(d2 in grp_members[grp_of[i]] for i in idx))
                             for d2 in ds_keys if d2 != d}

    print("4/5 label agreement on images shared across datasets")
    clusters = collections.defaultdict(list)
    for i in range(N):
        clusters[dup_of[i]].append(i)
    agree = collections.Counter()
    for members in clusters.values():
        by_ds = {}
        for i in members:
            by_ds.setdefault(ds_of[i], i)
        ds_here = sorted(by_ds)
        for x in range(len(ds_here)):
            for y in range(x + 1, len(ds_here)):
                a, b = by_ds[ds_here[x]], by_ds[ds_here[y]]
                if items[a]["size"][0] / items[a]["size"][1] != items[b]["size"][0] / items[b]["size"][1]:
                    continue  # different crops of a scene: boxes not comparable
                ba = np.array([[c, x - w / 2, y_ - h / 2, x + w / 2, y_ + h / 2] for c, x, y_, w, h in items[a]["boxes"]]).reshape(-1, 5)
                bb = np.array([[c, x - w / 2, y_ - h / 2, x + w / 2, y_ + h / 2] for c, x, y_, w, h in items[b]["boxes"]]).reshape(-1, 5)
                if not len(ba) or not len(bb):
                    continue
                iou = B_iou(ba[:, 1:], bb[:, 1:])
                for p in range(len(ba)):
                    q = int(iou[p].argmax())
                    if iou[p, q] >= 0.5:
                        na = items[a]["names"][int(ba[p, 0])]
                        nb = items[b]["names"][int(bb[q, 0])]
                        agree[(f"{ds_here[x]}:{na}", f"{ds_here[y]}:{nb}")] += 1
    rep["agreement"] = {f"{a} = {b}": n for (a, b), n in agree.most_common()}

    print("5/5 defect-crop retrieval")
    rng = np.random.default_rng(args.seed)
    crops, meta = [], []
    for d in ds_keys:
        idx = [i for i in range(N) if ds_of[i] == d and i in uniq]
        names = items[idx[0]]["names"]
        for c, name in enumerate(names):
            cand = [(i, b) for i in idx for b in items[i]["boxes"] if b[0] == c and learnable(b, *items[i]["size"])]
            if not cand:
                continue
            # spread the sample over scenes: one box per scene first, then more
            by_scene = collections.defaultdict(list)
            for i, b in cand:
                by_scene[grp_of[i]].append((i, b))
            order = [v for v in by_scene.values()]
            rng.shuffle(order)
            pick = []
            r = 0
            while len(pick) < CROPS_PER_CLASS and any(len(v) > r for v in order):
                pick += [v[r] for v in order if len(v) > r][:CROPS_PER_CLASS - len(pick)]
                r += 1
            for i, (_, x, y, w, h) in pick:
                W, H = items[i]["size"]
                x1, y1 = int(max(0, (x - w * 0.65) * W)), int(max(0, (y - h * 0.65) * H))
                x2, y2 = int(min(W, (x + w * 0.65) * W)), int(min(H, (y + h * 0.65) * H))
                if x2 - x1 >= 8 and y2 - y1 >= 8:
                    crops.append((paths[i], x1, y1, x2, y2))
                    meta.append((d, name, grp_of[i]))
    C = crop_embeddings(crops, os.path.join(args.cache, "crops_dinov2_vits14.npz"))
    lab = [f"{d}:{n}" for d, n, _ in meta]
    dsm = np.array([d for d, _, _ in meta])
    scene = np.array([g for _, _, g in meta])
    S = C @ C.T
    S[scene[:, None] == scene[None, :]] = -2  # never count another crop of the same scene
    within, across = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
    for i in range(len(C)):
        same = np.where(dsm == dsm[i])[0]
        other = np.where(dsm != dsm[i])[0]
        for j in same[np.argsort(-S[i, same])[:10]]:
            if S[i, j] > -2:
                within[lab[i]][lab[j]] += 1
        for j in other[np.argsort(-S[i, other])[:10]]:
            across[lab[i]][lab[j]] += 1
    for k in sorted(within):
        tot_w, tot_a = sum(within[k].values()), sum(across[k].values())
        rep["retrieval"][k] = dict(crops=lab.count(k), purity=round(within[k][k] / max(tot_w, 1), 3),
                                   within_confused_with={n: round(v / tot_w, 3) for n, v in within[k].most_common(4) if n != k},
                                   nearest_in_other_datasets={n: round(v / tot_a, 3) for n, v in across[k].most_common(5)},
                                   nearest_all={n: v / tot_a for n, v in across[k].items()})
    # class-equivalence evidence: shared-image label agreement + mutual crop retrieval (geometric mean of both
    # directions' neighbour fractions), for class pairs of different datasets
    agree_sym = collections.Counter()
    for (a, b), n in agree.items():
        agree_sym[tuple(sorted((a, b)))] += n
    keys = sorted(rep["retrieval"])
    pairs = []
    for x, a in enumerate(keys):
        for b in keys[x + 1:]:
            if a.split(":")[0] == b.split(":")[0]:
                continue
            ab = rep["retrieval"][a]["nearest_all"].get(b, 0.0)
            ba = rep["retrieval"][b]["nearest_all"].get(a, 0.0)
            n = agree_sym[(a, b)]
            if n or (ab * ba) ** 0.5 >= 0.03:
                pairs.append(dict(a=a, b=b, shared_label_agreement=n, retrieval_a_to_b=round(ab, 3),
                                  retrieval_b_to_a=round(ba, 3), retrieval_mutual=round((ab * ba) ** 0.5, 3)))
    rep["equivalence"] = sorted(pairs, key=lambda r: (-r["shared_label_agreement"], -r["retrieval_mutual"]))
    for r in rep["retrieval"].values():
        r.pop("nearest_all")
    json.dump(rep, open(os.path.join(args.out, "report.json"), "w"), indent=2)
    write_md(rep, os.path.join(args.out, "report.md"))
    print(f"written {args.out}/report.md")


def B_iou(a, b):
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), -1)
    area = lambda x: np.prod(x[:, 2:] - x[:, :2], -1)  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def write_md(rep, path):
    L = ["# Dataset audit (content-based)", ""]
    L += ["## Datasets", "", "| dataset | images | unique images | independent scenes |", "|---|---|---|---|"]
    L += [f"| {d}: {v['description']} | {v['images']} | {v['unique_images']} | {v['scenes']} |" for d, v in rep["datasets"].items()]
    L += ["", "## Native classes: legitimate examples", "",
          "| dataset | class | raw images | raw boxes | synthetic | unique images | scenes | learnable boxes | scenes with a learnable box | median box px@640 | crop purity |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for d, v in rep["datasets"].items():
        for c, r in v["classes"].items():
            pur = rep["retrieval"].get(f"{d}:{c}", {}).get("purity", "-")
            L.append(f"| {d} | {c} | {r['raw_images']} | {r['raw_boxes']} | {r['synthetic_images']} | {r['unique_images']} | "
                     f"{r['scenes']} | {r['learnable_boxes']} | **{r['learnable_scenes']}** | {r['median_box_px']} | {pur} |")
    L += ["", "## Overlap between datasets (images of the row dataset found in the column dataset: duplicate / same scene)", "",
          "| | " + " | ".join(rep["overlap"]) + " |", "|---" * (1 + len(rep["overlap"])) + "|"]
    for d, row in rep["overlap"].items():
        L.append(f"| {d} | " + " | ".join("-" if d2 == d else f"{row[d2]['duplicate']} / {row[d2]['same_scene']}" for d2 in rep["overlap"]) + " |")
    L += ["", "## Label agreement on the same defect in images shared by two datasets (boxes matched at IoU >= 0.5)", ""]
    L += [f"- {k}: {n}" for k, n in rep["agreement"].items() if n >= 3]
    L += ["", "## Class-equivalence evidence (classes of different datasets)", "",
          "Shared-label agreement: same defect, same image, labelled in both datasets. Retrieval: fraction of a class's "
          "nearest crops (in the other dataset's classes) that belong to the other class, both directions; mutual = "
          "geometric mean.", "",
          "| class a | class b | shared-label agreement | a->b | b->a | mutual |", "|---|---|---|---|---|---|"]
    L += [f"| {r['a']} | {r['b']} | {r['shared_label_agreement']} | {r['retrieval_a_to_b']} | {r['retrieval_b_to_a']} | "
          f"{r['retrieval_mutual']} |" for r in rep["equivalence"]]
    L += ["", "## Defect-crop retrieval (10 nearest crops from other scenes)", "",
          "| class | crops | purity (same class within dataset) | most confused with (within dataset) | nearest classes in other datasets |",
          "|---|---|---|---|---|"]
    for k, r in rep["retrieval"].items():
        L.append(f"| {k} | {r['crops']} | {r['purity']} | " + ", ".join(f"{n} {v}" for n, v in r["within_confused_with"].items())
                 + " | " + ", ".join(f"{n} {v}" for n, v in r["nearest_in_other_datasets"].items()) + " |")
    open(path, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
