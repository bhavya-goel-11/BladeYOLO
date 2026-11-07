"""Wind Surface Defect: for every val (= official test) image, its best pixel correlation with a training image the
official-protocol runs trained on (train minus the seeded 10% carve of rfdetr_v1.prepare_official), over the 8
flips/rotations, on 48x48 grey images. A pixel twin is corr >= 0.99 (tools/official_leak_eval.py).
  .venv/bin/python -I tools/ws_pixel_twins.py data/blade-defect-benchmark/raw/WindSurface-Defect   -> runs/analysis/paper/ws_pixel.npz
"""
import os, random, sys, numpy as np, cv2
D = sys.argv[1]
tr = sorted(os.listdir(f"{D}/train/images")); va = sorted(os.listdir(f"{D}/val/images"))
held = set(random.Random(0).sample(tr, round(0.1 * len(tr))))
def vec(p, s=48):
    g = cv2.resize(cv2.imread(p, cv2.IMREAD_GRAYSCALE), (s, s), interpolation=cv2.INTER_AREA).astype(np.float32)
    return g
def norm(a):
    a = a.reshape(len(a), -1); a = a - a.mean(1, keepdims=True); return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
T = np.stack([vec(f"{D}/train/images/{f}") for f in tr]); V = np.stack([vec(f"{D}/val/images/{f}") for f in va])
variants = [V, V[:, :, ::-1], V[:, ::-1, :], V[:, ::-1, ::-1], np.rot90(V, 1, (1, 2)), np.rot90(V, 3, (1, 2)), np.transpose(V, (0, 2, 1)), np.rot90(np.transpose(V,(0,2,1)),2,(1,2))]
Tn = norm(T); best = np.full((len(va), len(tr)), -1.0, np.float32)
for v in variants: best = np.maximum(best, norm(np.ascontiguousarray(v)) @ Tn.T)
trained = np.array([f not in held for f in tr])
b_all = best.max(1); b_tr = np.where(trained[None], best, -1).max(1)
for t in (0.98, 0.99, 0.995): print(t, (b_all >= t).sum(), (b_tr >= t).sum())
print("median best", np.median(b_all))
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "analysis", "paper")
os.makedirs(OUT, exist_ok=True)
np.savez(os.path.join(OUT, "ws_pixel.npz"), names=np.array(va), best_all=b_all, best_trained=b_tr, arg=np.where(trained[None], best, -1).argmax(1), train=np.array(tr))
