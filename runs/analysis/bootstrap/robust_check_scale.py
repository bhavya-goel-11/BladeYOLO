"""Phase 27: does the low-data gain grow with the frozen DINOv3's size? 25% of the labelled scenes.

ViT-S/16 (3 seeds), ViT-B/16 (2), ViT-L/16 (2), no DINOv3 (2); same protocol as robust_check_seeds.py (paired scene
bootstrap over the 58 test scenes, an arm is the mean of its seeds), all 6 classes and the 4 robust classes, plus
mAP50-95 per class.

With argv "curve": the learning curve instead, BladeYOLO-v2 (ViT-S) vs no DINOv3 vs YOLO12-L at 25 / 50 / 100%.
With argv "h3" (Phase 28): ViT-B + retrieval prior (train.py --retrieval) vs frozen ViT-B, 2 seeds each.
With argv "h6" (Phase 28): label-free scene selection (cover) vs the class-matched 25% subset and plain random draws.
"""
import csv, os, sys
import numpy as np
sys.path.insert(0, "tools")
import bootstrap_ci as bc

P = "bladeyolo2_wtbd_group"
N = "bladeyolo2nodino_wtbd_group"
SCALE = {"ViT-S": [P + "_f25", P + "_f25_s1", P + "_f25_s2"],
         "ViT-B": ["bladeyolo2b_wtbd_group_f25_s0", "bladeyolo2b_wtbd_group_f25_s1"],
         "ViT-L": ["bladeyolo2l_wtbd_group_f25_s0", "bladeyolo2l_wtbd_group_f25_s1"],
         "no DINOv3": [N + "_f25_s0", N + "_f25_s1"]}
SCALE_CMP = [("ViT-B", "ViT-S"), ("ViT-L", "ViT-S"), ("ViT-L", "ViT-B"), ("no DINOv3", "ViT-S")]
CURVE = {"ViT-S 25%": SCALE["ViT-S"], "no DINOv3 25%": SCALE["no DINOv3"], "YOLO12-L 25%": ["yolo12l_wtbd_group_f25"],
         "ViT-S 50%": [P + "_f50"], "no DINOv3 50%": [N + "_f50"], "YOLO12-L 50%": ["yolo12l_wtbd_group_f50"],
         "ViT-S 100%": [P, P + "_s1"], "no DINOv3 100%": [N], "YOLO12-L 100%": ["yolo12l_wtbd_group"]}
CURVE_CMP = [(f"no DINOv3 {f}", f"ViT-S {f}") for f in ("25%", "50%", "100%")] + \
            [(f"YOLO12-L {f}", f"no DINOv3 {f}") for f in ("25%", "50%", "100%")]
H3 = {"ViT-B": SCALE["ViT-B"], "ViT-B + retrieval": ["bladeyolo2b_retr_wtbd_group_f25_s0", "bladeyolo2b_retr_wtbd_group_f25_s1"]}
H3_CMP = [("ViT-B + retrieval", "ViT-B")]
H6 = {"class-matched": SCALE["ViT-B"], "cover": ["bladeyolo2b_cover_wtbd_group_f25_s0", "bladeyolo2b_cover_wtbd_group_f25_s1"],
      "plain random": ["bladeyolo2b_rand1_wtbd_group_f25_s0", "bladeyolo2b_rand2_wtbd_group_f25_s0"]}
H6_CMP = [("cover", "class-matched"), ("cover", "plain random"), ("plain random", "class-matched")]
CLS = ["craze", "corrosion", "surface_injure", "thunderstrike", "crack", "hide_craze"]


def main(arms, cmps):
    data = os.path.abspath("external/WTBD-group")
    g = {}
    for r in csv.DictReader(open("runs/analysis/bootstrap/wtbd_group/groups.csv")):
        g.setdefault(r["group"], []).append(os.path.join(data, "test", "images", r["image"]))
    groups = list(g.values())
    imgs = bc.load_yolo_split(data, "test")
    files = sorted(imgs)
    runs = list(dict.fromkeys(r for a in arms.values() for r in a))
    preds = {r: bc.predict(os.path.join("runs/detect/BladeYOLO_WindSurface", r), imgs, "runs/analysis/bootstrap/seeds_f25/preds") for r in runs}
    rng = np.random.default_rng(0)
    draws = [[f for gi in rng.integers(0, len(groups), len(groups)) for f in groups[gi]] for _ in range(1000)]
    for label, keep in (("all 6 classes", list(range(6))), ("4 robust classes", [1, 2, 4, 5])):
        sub = {k: dict(v, gt=v["gt"][np.isin(v["gt"][:, 0], keep)]) for k, v in imgs.items()}
        ms = {r: bc.per_image_matches(sub, {k: v[np.isin(v[:, 5], keep)] for k, v in preds[r].items()}, 6) for r in runs}
        full = {r: bc.score(ms[r], files, 6) for r in runs}
        arm = lambda a, fs: np.mean([bc.score(ms[r], fs, 6)[:2] for r in arms[a]], axis=0)
        pt = {a: np.mean([full[r][:2] for r in arms[a]], axis=0) for a in arms}
        bs = {a: np.array([arm(a, d) for d in draws]) for a in arms}
        print(f"== {label}")
        for a in arms:
            pcls = np.mean([full[r][3] for r in arms[a]], axis=0)
            print(f"  {a:>15s}: mAP50 {pt[a][0]:.3f}  mAP50-95 {pt[a][1]:.3f}  seeds "
                  + " ".join(f"{full[r][0]:.3f}/{full[r][1]:.3f}" for r in arms[a])
                  + "  per-class AP50-95 " + " ".join("-" if np.isnan(x) else f"{x:.3f}" for x in pcls))
        for a, b in cmps:
            for j, name in enumerate(("mAP50", "mAP50-95")):
                d = bs[a][:, j] - bs[b][:, j]
                print(f"  {a} - {b} {name}: {pt[a][j] - pt[b][j]:+.3f} "
                      f"[{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}] P(>0) {np.mean(d > 0):.3f}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if sys.argv[1:] else ""
    main(*{"curve": (CURVE, CURVE_CMP), "h3": (H3, H3_CMP), "h6": (H6, H6_CMP)}.get(mode, (SCALE, SCALE_CMP)))
