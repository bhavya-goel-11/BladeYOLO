"""H14 (Phase 30): detector-guided active learning against random extension, from several random starts.

  scenes  (local, once) scene of every training image (tools/select_scenes.py grouping) -> tools/splits/wtbd_group_train_scenes.csv
  start   random scene-level draws of --budget images (same rule as select_scenes.py --random)
          -> tools/splits/wtbd_group_train_f10_al<k>.txt
  extend  from a start subset and the detector trained on it, add whole scenes up to --budget images:
            unc  scenes ranked by mean image uncertainty, 1 - the highest detection confidence (>= 0.05; 1 if none)
            rnd  a random scene order (seed fixed per start)
          -> tools/splits/wtbd_group_train_f25_al<k>_{unc,rnd}.txt

  python tools/active_select.py scenes --data external/WTBD-group/data.yaml
  python tools/active_select.py start  --data external/WTBD-group/data.yaml --budget 75 --draws 3
  python tools/active_select.py extend --data D --start tools/splits/wtbd_group_train_f10_al1.txt \
      --weights runs/detect/BladeYOLO_WindSurface/bladeyolo2nodino_al1_wtbd_group_f10/weights/best.pt --budget 189
"""

import argparse
import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from synth_defects import train_dirs  # noqa: E402

SPLITS = os.path.join(ROOT, "tools", "splits")
SCENES = os.path.join(SPLITS, "wtbd_group_train_scenes.csv")
MIN_CONF = 0.05


def load_scenes():
    """[[image name, ...] per scene], in scene order."""
    g = {}
    for r in csv.DictReader(open(SCENES)):
        g.setdefault(int(r["scene"]), []).append(r["image"])
    return [sorted(g[k]) for k in sorted(g)]


def add_scenes(order, groups, chosen, budget):
    """Add scenes in order (skipping chosen ones) while the total stays within 5% over budget, until it is reached."""
    chosen, used = list(chosen), sum(len(groups[g]) for g in chosen)
    for g in order:
        if used >= budget:
            break
        if g not in chosen and used + len(groups[g]) <= budget * 1.05:
            chosen.append(g)
            used += len(groups[g])
    return chosen


def uncertainty(confs):
    """1 - the highest confidence >= MIN_CONF of an image's detections (1 when there is none)."""
    c = [x for x in confs if x >= MIN_CONF]
    return 1.0 - max(c) if c else 1.0


def rank_scenes(groups, candidates, image_unc):
    """Candidate scenes by mean image uncertainty, most uncertain first (ties: scene index)."""
    score = {g: np.mean([image_unc[n] for n in groups[g]]) for g in candidates}
    return sorted(candidates, key=lambda g: (-score[g], g)), score


def write(path, groups, chosen):
    names = sorted(n for g in chosen for n in groups[g])
    with open(path, "w") as f:
        f.write("\n".join(names) + "\n")
    print(f"{len(names)} images in {len(chosen)} scenes -> {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("scenes", "start", "extend"))
    p.add_argument("--data", required=True)
    p.add_argument("--budget", type=int, default=75)
    p.add_argument("--draws", type=int, default=3)
    p.add_argument("--start", default=None)
    p.add_argument("--weights", default=None)
    p.add_argument("--seed", type=int, default=10)
    args = p.parse_args()
    img_dir, _ = train_dirs(args.data)
    if args.mode == "scenes":
        import select_scenes as ss

        files = sorted(os.listdir(img_dir))
        groups = ss.scenes([os.path.join(img_dir, f) for f in files], os.path.join(ROOT, ".dataset_cache"), 4)
        with open(SCENES, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image", "scene"])
            w.writerows(sorted((files[i], g) for g, mem in enumerate(groups) for i in mem))
        print(f"{len(files)} images in {len(groups)} scenes -> {SCENES}")
        return
    groups = load_scenes()
    scene_of = {n: g for g, mem in enumerate(groups) for n in mem}
    if args.mode == "start":
        for k in range(1, args.draws + 1):
            order = np.random.default_rng(args.seed + k).permutation(len(groups)).tolist()
            write(os.path.join(SPLITS, f"wtbd_group_train_f10_al{k}.txt"), groups, add_scenes(order, groups, [], args.budget))
        return
    start = [n.strip() for n in open(args.start) if n.strip()]
    chosen = sorted({scene_of[n] for n in start})
    rest = [g for g in range(len(groups)) if g not in chosen]
    tag = os.path.basename(args.start).split("_f10_")[1][:-4]  # al<k>
    k = int(tag[2:])
    rnd = np.random.default_rng(args.seed + 100 + k).permutation(rest).tolist()
    write(os.path.join(SPLITS, f"wtbd_group_train_f25_{tag}_rnd.txt"), groups, add_scenes(rnd, groups, chosen, args.budget))

    import models  # noqa: F401  (BladeYOLO modules for unpickling)
    from ultralytics import YOLO

    model = YOLO(args.weights)
    pool = [n for g in rest for n in groups[g]]
    unc = {}
    for i in range(0, len(pool), 16):
        batch = pool[i:i + 16]
        for n, r in zip(batch, model.predict([os.path.join(img_dir, n) for n in batch], imgsz=640, conf=MIN_CONF,
                                             verbose=False)):
            unc[n] = uncertainty(r.boxes.conf.cpu().numpy().tolist())
    order, score = rank_scenes(groups, rest, unc)
    sel = add_scenes(order, groups, chosen, args.budget)
    write(os.path.join(SPLITS, f"wtbd_group_train_f25_{tag}_unc.txt"), groups, sel)
    added = [g for g in sel if g not in chosen]
    print(f"uncertainty: pool mean {np.mean(list(unc.values())):.3f}, added scenes mean "
          f"{np.mean([score[g] for g in added]):.3f}")


if __name__ == "__main__":
    main()
