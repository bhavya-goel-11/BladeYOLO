"""Box-extent diagnostic (research_log.md, box-extent direction): is the strict-IoU gap a label-convention ceiling or a
model weakness? Compares two IoU distributions on WTBD:

  annotators  WTBD's augmented copies were labelled independently (§14). For every same-view, axis-aligned directed copy
              pair (exact transform, tools/view_label_audit.py cache), each audited label of a is projected into b and
              its best same-class IoU with b's labels recorded: how tightly two annotations of one defect agree.
  models      for every WTBD-group test box, the best IoU of a same-class prediction with conf >= CONF.

Reported per class: share >= 0.5, and among those the share >= 0.75 and the median IoU. If annotators agree at
IoU >= 0.75 far more often than models hit it, the extent gap is the model's, not the labels'.

  .venv/bin/python tools/extent_ceiling.py  -> runs/analysis/extent/ceiling.md
"""

import json
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import bootstrap_ci as bc  # noqa: E402
import detection_metrics as dm  # noqa: E402
import matching as mt  # noqa: E402
import view_label_audit as va  # noqa: E402

OFFICIAL = os.path.join("external", "WTBD-official")  # relative, as the audit cache was built
GROUP = os.path.join(ROOT, "external", "WTBD-group")
AUDIT = os.path.join(ROOT, "runs", "analysis", "novel", "label_audit", "wtbd", "transforms_v3.json")
PREDS = os.path.join(ROOT, "runs", "analysis", "bootstrap", "seeds_f25", "preds")
OUT = os.path.join(ROOT, "runs", "analysis", "extent")
MODELS = {"RF-DETR-M 25% (4 subsets, s0)": ["rfdetrm_wtbd_group_f25"] + [f"rfdetrm_rand{k}_wtbd_group_f25" for k in (1, 2, 3)],
          "ViT-B 25% (4 subsets, s0)": ["bladeyolo2b_wtbd_group_f25_s0"] + [f"bladeyolo2b_rand{k}_wtbd_group_f25_s0" for k in (1, 2, 3)],
          "BladeYOLO-v2 100%": ["bladeyolo2_wtbd_group"]}
CONF = 0.1


def annotator_ious(test_only):
    os.chdir(ROOT)
    items = va.load(OFFICIAL)
    T = json.load(open(AUDIT))
    test_names = set(os.listdir(os.path.join(GROUP, "test", "images")))
    out = []  # (class, best same-class IoU)
    for key, (inl, fl, M, support) in T.items():
        if inl < mt.MIN_INLIERS or M is None:
            continue
        x, y = map(int, key.split())
        (pa, sa, ga), (pb, sb, gb) = items[x], items[y]
        if test_only and not (os.path.basename(pa) in test_names and os.path.basename(pb) in test_names):
            continue
        ang = np.degrees(np.arctan2(M[1][0], M[0][0])) % 90
        if va.view_coverage(sa, sb, fl, M) < mt.SAME_VIEW or min(ang, 90 - ang) > va.ROT_TOL or not len(ga):
            continue
        proj = va.project(ga[:, 1:], sa, sb, fl, M)
        for (c, *_), box, sup in zip(ga, proj, support):
            if va.frac_inside(box, *sb) < 0.8 or sup < va.LOCAL_MIN:
                continue
            same = gb[gb[:, 0] == c][:, 1:] if len(gb) else np.zeros((0, 4))
            out.append((int(c), float(dm.iou_matrix(box[None], same)[0].max()) if len(same) else 0.0))
    return out


def model_ious(runs):
    imgs = bc.load_yolo_split(GROUP, "test")
    out = []
    for r in runs:
        p = {k: np.array(v).reshape(-1, 6) for k, v in json.load(open(os.path.join(PREDS, r + ".json"))).items()}
        for f, im in imgs.items():
            d = p.get(f, np.zeros((0, 6)))
            d = d[d[:, 4] >= CONF]
            for g in im["gt"]:
                s = d[d[:, 5] == g[0]][:, :4]
                out.append((int(g[0]), float(dm.iou_matrix(g[None, 1:], s)[0].max()) if len(s) else 0.0))
    return out


def row(name, v):
    v = np.array([i for _, i in v])
    hit = v[v >= 0.5]
    return (f"| {name} | {len(v)} | {np.mean(v >= 0.5):.2f} | {np.mean(hit >= 0.75) if len(hit) else float('nan'):.2f} | "
            f"{np.median(hit) if len(hit) else float('nan'):.2f} | {np.mean(v >= 0.75):.2f} |")


def main():
    names = yaml.safe_load(open(os.path.join(GROUP, "data.yaml")))["names"]
    sets = {"annotators, all WTBD copies": annotator_ious(False), "annotators, test copies only": annotator_ious(True)}
    sets.update({k: model_ious(v) for k, v in MODELS.items()})
    head = ["| source | boxes | IoU ≥ 0.5 | of those ≥ 0.75 | median IoU of those | IoU ≥ 0.75 |", "|---|---|---|---|---|---|"]
    lines = ["# Box-extent ceiling: annotator agreement vs model localisation (WTBD)", "",
             f"Annotators: same-view axis-aligned copy pairs, best same-class IoU of a projected label in the other copy. "
             f"Models: best same-class IoU of a prediction with conf ≥ {CONF} per WTBD-group test box.", "", "## All classes", ""]
    lines += head + [row(k, v) for k, v in sets.items()]
    for c, n in enumerate(names):
        lines += ["", f"## {n}", ""] + head + [row(k, [x for x in v if x[0] == c]) for k, v in sets.items()]
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "ceiling.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
