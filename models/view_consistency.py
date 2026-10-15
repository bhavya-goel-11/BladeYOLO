"""Real-view consistency: teacher-student training with pseudo-labels moved between registered views (Phase 19-21).

An unlabelled pair (A, B) shows the same surface; T maps A's pixels to B's (tools/view_pool.py: flip + similarity
transform verified by ORB/RANSAC). The teacher (the trainer's EMA model) detects defects on A; boxes with confidence
>= tau are mapped into B and become B's labels for the student. B is masked to the area A covers, so defects outside
A's view are not trained as background, and a moved box is kept only if A (warped into B) and B show the same
structure inside it (gradient ZNCC >= LOCAL_ZNCC): a global transform fits the static scene, but a rotor turns
between video frames, so labels on moving blades must not be moved. Mode "real": B is another photograph of the
surface (pairs.csv rows with same_photo = 0). Mode "synthetic" (the standard teacher-student baseline): B is a random
augmentation of A with a known T, so the two modes differ only in where the second view comes from.

Enabled from train.py (--ssl real|synthetic, --ssl-pairs, --ssl-ratio, --ssl-tau, --ssl-start) through environment
variables, so DDP workers agree.
"""

import csv
import json
import math
import os
import random

import cv2
import numpy as np
import torch

LETTERBOX = 114
MIN_KEEP = 0.5  # share of a moved box that must stay inside B's view
LOCAL_ZNCC = 0.5  # gradient ZNCC of warped A and B inside a moved box for it to be kept


def letterbox_matrix(h, w, size):
    """Original pixels -> letterboxed (size x size, centred padding) pixels, and the image scale."""
    s = size / max(h, w)
    nh, nw = round(h * s), round(w * s)
    px, py = (size - nw) / 2, (size - nh) / 2
    return np.array([[s, 0, px], [0, s, py], [0, 0, 1]], np.float64), (nh, nw)


def letterbox(img, size):
    L, (nh, nw) = letterbox_matrix(*img.shape[:2], size)
    out = np.full((size, size, 3), LETTERBOX, np.uint8)
    y0, x0 = int(round(L[1, 2])), int(round(L[0, 2]))
    out[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, L


def audit_matrix(size_a, size_b, fl, M):
    """A's original pixels -> B's original pixels for a transform estimated on 480 px greyscale (flip, then M)."""
    sa, sb = 480 / max(size_a), 480 / max(size_b)
    ha, wa = round(size_a[0] * sa), round(size_a[1] * sa)
    S_a = np.diag([sa, sa, 1.0])
    F = np.eye(3)
    if fl in (1, -1):
        F = np.array([[-1, 0, wa - 1], [0, 1, 0], [0, 0, 1]], np.float64) @ F
    if fl in (0, -1):
        F = np.array([[1, 0, 0], [0, -1, ha - 1], [0, 0, 1]], np.float64) @ F
    M3 = np.vstack([np.array(M, np.float64), [0, 0, 1]])
    return np.diag([1 / sb, 1 / sb, 1.0]) @ M3 @ F @ S_a


def move_boxes(boxes, T):
    """(n, 4) xyxy through the 3x3 affine T; axis-aligned box of the moved corners."""
    if not len(boxes):
        return boxes.reshape(0, 4)
    x1, y1, x2, y2 = boxes.T
    c = np.stack([np.stack([x1, y1], 1), np.stack([x2, y1], 1), np.stack([x2, y2], 1), np.stack([x1, y2], 1)], 1)
    m = c @ T[:2, :2].T + T[:2, 2]
    return np.concatenate([m.min(1), m.max(1)], 1)


def synthetic_view(img, rng):
    """Random augmentation of img with its pixel map: flip, quarter turn, scale, shift, colour, blur."""
    h, w = img.shape[:2]
    T = np.eye(3)
    if rng.random() < 0.5:
        img, T = img[:, ::-1], np.array([[-1, 0, w - 1], [0, 1, 0], [0, 0, 1]], np.float64) @ T
    k = rng.randrange(4)
    for _ in range(k):  # rotate 90 degrees counter-clockwise: (x, y) -> (y, w - 1 - x)
        h, w = img.shape[:2]
        img, T = np.rot90(img), np.array([[0, 1, 0], [-1, 0, w - 1], [0, 0, 1]], np.float64) @ T
    h, w = img.shape[:2]
    s = rng.uniform(0.75, 1.25)
    A = np.array([[s, 0, (1 - s) * w / 2 + rng.uniform(-0.1, 0.1) * w], [0, s, (1 - s) * h / 2 + rng.uniform(-0.1, 0.1) * h]])
    img = cv2.warpAffine(np.ascontiguousarray(img), A, (w, h), borderValue=(LETTERBOX,) * 3)
    T = np.vstack([A, [0, 0, 1]]) @ T
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= rng.uniform(0.6, 1.4)
    hsv[..., 2] *= rng.uniform(0.6, 1.4)
    img = cv2.cvtColor(hsv.clip(0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (5, 5), 0)
    return img, T


class PairDataset(torch.utils.data.Dataset):
    """Letterboxed (A, B, T, B's valid-area mask): T maps A's letterboxed pixels to B's."""

    def __init__(self, mode, pairs_csv, size=640, seed=0):
        self.mode, self.size, self.seed = mode, size, seed
        base = os.path.dirname(os.path.abspath(pairs_csv))
        path = lambda p: p if os.path.isabs(p) or os.path.exists(p) else os.path.join(base, p)  # noqa: E731  exported pools
        rows = [r for r in csv.DictReader(open(pairs_csv))
                if r.get("same_photo", "0") == "0" and r["M"] not in ("", "null", "None")]
        if mode == "real":
            self.items = [(path(r["image_a"]), path(r["image_b"]), None if r["flip"] in ("", "None") else int(r["flip"]),
                           json.loads(r["M"])) for r in rows]
        else:  # synthetic: the same images as the real pairs, each paired with an augmentation of itself
            self.items = sorted({path(r["image_a"]) for r in rows} | {path(r["image_b"]) for r in rows})
        self.epoch = 0

    def sample_weights(self):
        """Per-item weights so that every view-A image is drawn equally often (video frames have many neighbours, so
        uniform pair sampling would be dominated by a few videos); uniform in synthetic mode (one item per image)."""
        if self.mode != "real":
            return torch.ones(len(self.items), dtype=torch.double)
        count = {}
        for a, *_ in self.items:
            count[a] = count.get(a, 0) + 1
        return torch.tensor([1.0 / count[a] for a, *_ in self.items], dtype=torch.double)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rng = random.Random(hash((self.seed, self.epoch, i)))
        if self.mode == "real":
            a_path, b_path, fl, M = self.items[i]
            a, b = cv2.imread(a_path), cv2.imread(b_path)
            T0 = audit_matrix(a.shape[:2], b.shape[:2], fl, M)
        else:
            a = cv2.imread(self.items[i])
            b, T0 = synthetic_view(a, rng)
        A, La = letterbox(a, self.size)
        B, Lb = letterbox(b, self.size)
        T = Lb @ T0 @ np.linalg.inv(La)
        # B's area seen by A: A's image region (inside its letterbox) moved into B, intersected with B's image region
        a_valid = np.zeros(A.shape[:2], np.uint8)
        y0, x0 = int(round(La[1, 2])), int(round(La[0, 2]))
        a_valid[y0:self.size - y0, x0:self.size - x0] = 255
        seen = cv2.warpAffine(a_valid, T[:2], (self.size, self.size), flags=cv2.INTER_NEAREST) > 0
        b_valid = np.zeros(B.shape[:2], bool)
        y0, x0 = int(round(Lb[1, 2])), int(round(Lb[0, 2]))
        b_valid[y0:self.size - y0, x0:self.size - x0] = True
        mask = seen & b_valid
        B = np.where(mask[..., None], B, LETTERBOX).astype(np.uint8)
        grad = lambda g: cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))  # noqa: E731
        gray = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32)  # noqa: E731
        ga = grad(cv2.warpAffine(gray(A), T[:2], (self.size, self.size), flags=cv2.INTER_LINEAR))  # A seen from B
        gb = grad(gray(B))
        to_t = lambda x: torch.from_numpy(np.ascontiguousarray(x[..., ::-1].transpose(2, 0, 1)))  # noqa: E731  BGR->RGB
        return (to_t(A), to_t(B), torch.from_numpy(T).float(), torch.from_numpy(mask),
                torch.from_numpy(np.stack([ga, gb]).astype(np.float16)))


def local_zncc(ga, gb, box):
    """ZNCC of two gradient-magnitude images inside an integer box (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = box
    a, b = ga[y1:y2, x1:x2].astype(np.float64).ravel(), gb[y1:y2, x1:x2].astype(np.float64).ravel()
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def transfer(det, T, mask, size, tau, grads=None):
    """Teacher detections on A (n, 6: xyxy, conf, cls; letterboxed px) -> B labels (m, 5: cls, xc, yc, w, h normalised).
    A moved box is kept if >= MIN_KEEP of it lies in B's valid area, its sides are >= 4 px and, given grads (gradient
    magnitudes of A warped into B and of B), the two agree inside it (ZNCC >= LOCAL_ZNCC)."""
    det = det[det[:, 4] >= tau]
    if not len(det):
        return np.zeros((0, 5), np.float32)
    moved = move_boxes(det[:, :4].astype(np.float64), T)
    out = []
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return np.zeros((0, 5), np.float32)
    vx1, vy1, vx2, vy2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    for (x1, y1, x2, y2), c in zip(moved, det[:, 5]):
        cx1, cy1, cx2, cy2 = max(x1, vx1), max(y1, vy1), min(x2, vx2), min(y2, vy2)
        if cx2 - cx1 < 4 or cy2 - cy1 < 4:
            continue
        inside = mask[int(cy1):int(math.ceil(cy2)), int(cx1):int(math.ceil(cx2))].mean()
        if (cx2 - cx1) * (cy2 - cy1) / max((x2 - x1) * (y2 - y1), 1e-9) * inside < MIN_KEEP:
            continue
        if grads is not None and local_zncc(grads[0], grads[1], (int(cx1), int(cy1), int(math.ceil(cx2)),
                                                                   int(math.ceil(cy2)))) < LOCAL_ZNCC:
            continue
        out.append([c, (cx1 + cx2) / 2 / size, (cy1 + cy2) / 2 / size, (cx2 - cx1) / size, (cy2 - cy1) / size])
    return np.array(out, np.float32).reshape(-1, 5)


def config():
    """SSL settings from the environment (set by train.py), or None when disabled."""
    mode = os.environ.get("BLADEYOLO_SSL", "")
    if mode not in ("real", "synthetic"):
        return None
    return dict(mode=mode, pairs=os.environ["BLADEYOLO_SSL_PAIRS"], ratio=float(os.environ.get("BLADEYOLO_SSL_RATIO", 0.25)),
                tau=float(os.environ.get("BLADEYOLO_SSL_TAU", 0.5)), start=int(os.environ.get("BLADEYOLO_SSL_START", 30)))
