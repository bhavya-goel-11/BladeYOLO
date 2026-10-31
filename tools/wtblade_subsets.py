"""25% scene subsets of WTBlade-group train for the cross-dataset replication (H18-W), drawn the way the WTBD ones were:
  class-matched  best of --trials random scene orders by L1 distance to the full set's class box shares, every class
                 present (tools/scene_fractions.py)
  rand1..randN   plain random scene draws of the same image budget, no class matching (tools/select_scenes.py)
Scenes are the WTBlade-group groups (external/WTBlade-group/groups.csv), so no scene straddles labelled and unlabelled.

Usage:
    python tools/wtblade_subsets.py
Writes tools/splits/wtblade_group_train_f25.txt, ..._rand<k>.txt and runs/analysis/wtblade/subsets_f25.txt (summary).
"""

import argparse
import csv
import os

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "external", "WTBlade-group")


def train_scenes(data):
    """files, (n, nc) box counts per file, list of index arrays per scene."""
    names = yaml.safe_load(open(os.path.join(data, "data.yaml")))["names"]
    files = sorted(os.listdir(os.path.join(data, "train", "images")))
    gid = {r["image"]: int(r["group"]) for r in csv.DictReader(open(os.path.join(data, "groups.csv"))) if r["subset"] == "train"}
    missing = set(files) - set(gid)
    assert not missing, f"{len(missing)} train images without a group, e.g. {sorted(missing)[:3]}"
    boxes = []
    for f in files:
        lab = os.path.join(data, "train", "labels", os.path.splitext(f)[0] + ".txt")
        cls = [int(l.split()[0]) for l in open(lab) if l.strip()] if os.path.exists(lab) else []
        boxes.append(np.bincount(cls, minlength=len(names)))
    g = np.array([gid[f] for f in files])
    groups = [np.flatnonzero(g == u) for u in np.unique(g)]
    return names, files, np.array(boxes), groups


def pack(order, groups, budget):
    """Scenes in order, skipping any that would overshoot the budget by > 5% (WTBlade has one 899-image video scene,
    42% of train, which therefore never fits a 25% budget)."""
    sel, used = [], 0
    for g in order:
        if used + len(groups[g]) <= budget * 1.05:
            sel.append(g)
            used += len(groups[g])
        if used >= budget:
            break
    return sel


def class_matched(boxes, groups, budget, trials, seed):
    gbox = np.array([boxes[g].sum(0) for g in groups])
    share = boxes.sum(0) / boxes.sum()
    rng = np.random.default_rng(seed)
    best, best_score = None, np.inf
    for _ in range(trials):
        sel = pack(rng.permutation(len(groups)), groups, budget)
        b = gbox[sel].sum(0)
        if (b == 0).any():
            continue
        score = np.abs(b / b.sum() - share).sum()
        if score < best_score:
            best, best_score = sel, score
    return best


def random_draws(groups, budget, n):
    rng = np.random.default_rng(0)
    return [pack(rng.permutation(len(groups)), groups, budget) for _ in range(n)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--fraction", type=float, default=0.25)
    ap.add_argument("--random", type=int, default=3)
    ap.add_argument("--trials", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefix", default="wtblade_group_train_f25")
    args = ap.parse_args()
    names, files, boxes, groups = train_scenes(args.data)
    budget = round(args.fraction * len(files))
    subsets = {"": class_matched(boxes, groups, budget, args.trials, args.seed)}
    subsets.update({f"_rand{k + 1}": s for k, s in enumerate(random_draws(groups, budget, args.random))})
    lines = [f"{len(files)} training images in {len(groups)} scenes; budget {budget} images; boxes "
             + ", ".join(f"{n} {int(c)}" for n, c in zip(names, boxes.sum(0)))]
    for suf, sel in subsets.items():
        idx = np.concatenate([groups[g] for g in sel])
        open(os.path.join(ROOT, "tools", "splits", f"{args.prefix}{suf}.txt"), "w").write(
            "\n".join(sorted(files[i] for i in idx)) + "\n")
        lines.append(f"{args.prefix}{suf}: {len(sel)} scenes, {len(idx)} images, boxes "
                     + ", ".join(f"{n} {int(c)}" for n, c in zip(names, boxes[idx].sum(0))))
    out = os.path.join(ROOT, "runs", "analysis", "wtblade")
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "subsets_f25.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
