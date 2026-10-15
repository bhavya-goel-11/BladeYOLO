"""Content matching for drone imagery: which images show the same surface.

DINOv2 ViT-S/14 embeddings retrieve candidate pairs (k nearest neighbours); ORB keypoints with a flip-aware RANSAC
similarity transform verify them. Used by the leak-free splits (tools/wtbd_groups.py), the scene groups of the
bootstrap and conformal tools, and the cross-view label audit.
"""

import collections
import os
from multiprocessing import Pool

import cv2
import numpy as np

KNN = 10  # retrieval candidates per image
MIN_INLIERS = 25  # ORB/RANSAC inliers for a verified content match (random pairs: >=25 in ~1%, mostly true dups)
SAME_VIEW = 0.7  # verified match covering this much of the other image = the same view, not just overlap
DINO_GROUP = 0.93  # cosine above which two images show the same scene (augmented view, neighbouring frame): same split
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}


def dinov2_embeddings(paths, cache, batch=32):
    """L2-normalised DINOv2 ViT-S/14 [CLS, mean patch] embeddings, cached per file path.

    Paths are cache keys (as given; keep them relative to the repo root to reuse .dataset_cache). Runs on CUDA when
    available; the cache is saved every 50 batches, so an interrupted run resumes.
    """
    known = {}
    if os.path.exists(cache):
        z = np.load(cache)
        known = dict(zip(z["paths"].tolist(), z["emb"]))
    todo = [p for p in paths if p not in known]
    if todo:
        import torch

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval().to(dev)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        def save():
            np.savez(cache, paths=np.array(list(known)), emb=np.stack(list(known.values())))

        print(f"  embedding {len(todo)} images on {dev}")
        with torch.no_grad():
            for n, i in enumerate(range(0, len(todo), batch)):
                chunk = todo[i:i + batch]
                x = torch.stack([
                    (torch.from_numpy(cv2.resize(cv2.imread(p)[:, :, ::-1], (224, 224), interpolation=cv2.INTER_CUBIC).copy())
                     .permute(2, 0, 1).float() / 255 - mean) / std
                    for p in chunk
                ]).to(dev)
                f = model.forward_features(x)
                e = torch.nn.functional.normalize(torch.cat([f["x_norm_clstoken"], f["x_norm_patchtokens"].mean(1)], 1), dim=1)
                known.update(zip(chunk, e.cpu().numpy().astype(np.float16)))
                print(f"  embedded {min(i + batch, len(todo))}/{len(todo)}", end="\r", flush=True)
                if n % 50 == 49:
                    save()
        print()
        save()
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
