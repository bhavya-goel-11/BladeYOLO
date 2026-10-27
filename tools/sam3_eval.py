"""H13 stage A (Phase 30): SAM 3 (facebook/sam3, gated) zero-shot with text prompts on the leak-free WTBD-group test.

The prompts are OWLv2's (tools/openvocab_eval.py PROMPTS: three fixed descriptions per class, written before any
result), so both open-vocabulary models get identical queries. Per image the vision encoder runs once; each prompt's
instances (boxes, scores >= 0.01, at most 100) are pooled per class, per-class NMS at IoU 0.5, top 300 per image.
  predict  (GPU, needs HF_TOKEN with access)  -> runs/analysis/sam3/preds_text.json (640-px boxes keyed by image name)
  score    (local) bootstrap evaluator against OWLv2 text and the trained ViT-B 25% detector -> runs/analysis/sam3/report.md

  python tools/sam3_eval.py predict --data /kaggle/input/.../WTBD-group/data.yaml
  .venv/bin/python tools/sam3_eval.py score
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

CLASSES = ["craze", "corrosion", "surface_injure", "thunderstrike", "crack", "hide_craze"]
PROMPTS = {  # identical to tools/openvocab_eval.py (kept here so predict needs no OWLv2 imports)
    "craze": ["crazing on a wind turbine blade", "a network of fine hairline cracks in the coating", "craze"],
    "corrosion": ["leading edge erosion on a wind turbine blade", "corroded blade surface", "corrosion"],
    "surface_injure": ["surface damage on a wind turbine blade", "a scratch or chipped paint on a blade", "surface injury"],
    "thunderstrike": ["lightning strike damage on a wind turbine blade", "a burn mark", "thunderstrike"],
    "crack": ["a crack in a wind turbine blade", "a structural crack", "crack"],
    "hide_craze": ["faint hidden crazing on a wind turbine blade", "subtle hairline cracks under the coating", "hidden craze"],
}
OUT = os.path.join(ROOT, "runs", "analysis", "sam3")
MODEL = "facebook/sam3"
TOP, PER_PROMPT, MIN_SCORE = 300, 100, 0.01


def test_dir(data):
    import yaml

    cfg = yaml.safe_load(open(data))
    root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(data)), cfg.get("path") or "."))
    for d in ("test/images", "test/test/images"):
        if os.path.isdir(os.path.join(root, d)):
            return os.path.join(root, d)
    raise FileNotFoundError(f"no test/images under {root}")


def pool(per_class, top=TOP):
    """[(boxes (n, 4), scores (n,)) per class] -> (m, 6) x1 y1 x2 y2 score cls after per-class NMS (IoU 0.5), top."""
    import torch
    import torchvision

    rows = []
    for c, (b, s) in enumerate(per_class):
        if len(s):
            bt, st = torch.as_tensor(b, dtype=torch.float32), torch.as_tensor(s, dtype=torch.float32)
            k = torchvision.ops.nms(bt, st, 0.5)
            rows.append(np.c_[bt[k].numpy(), st[k].numpy(), np.full(len(k), c)])
    if not rows:
        return np.zeros((0, 6))
    out = np.concatenate(rows)
    return out[np.argsort(-out[:, 4])][:top]


def predict(args):
    import torch
    from PIL import Image
    from transformers import Sam3Model, Sam3Processor

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = Sam3Processor.from_pretrained(MODEL)
    model = Sam3Model.from_pretrained(MODEL).to(dev).eval()
    prompts = [p for c in CLASSES for p in PROMPTS[c]]
    owner = [i for i, c in enumerate(CLASSES) for _ in PROMPTS[c]]
    text = proc(text=prompts, return_tensors="pt", padding=True).to(dev)
    d = test_dir(args.data)
    files = sorted(os.listdir(d))
    os.makedirs(OUT, exist_ok=True)
    out = {}
    for i, f in enumerate(files):
        img = Image.open(os.path.join(d, f)).convert("RGB")
        W, H = img.size
        s = 640 / max(W, H)  # boxes in 640-input pixels, as tools/bootstrap_ci.py
        inp = proc(images=img, return_tensors="pt").to(dev)
        per_class = [([], []) for _ in CLASSES]
        with torch.no_grad():
            vis = model.get_vision_features(pixel_values=inp.pixel_values)
            for j in range(len(prompts)):
                o = model(vision_embeds=vis, input_ids=text.input_ids[j:j + 1], attention_mask=text.attention_mask[j:j + 1])
                r = proc.post_process_instance_segmentation(o, threshold=MIN_SCORE, mask_threshold=0.5,
                                                            target_sizes=[(H, W)])[0]
                sc = r["scores"].float().cpu().numpy()
                k = np.argsort(-sc)[:PER_PROMPT]
                per_class[owner[j]][0].extend((r["boxes"].float().cpu().numpy()[k] * s).tolist())
                per_class[owner[j]][1].extend(sc[k].tolist())
        out[f] = pool([(np.array(b).reshape(-1, 4), np.array(sc)) for b, sc in per_class]).round(3).tolist()
        if i % 20 == 0:
            print(f"  test {i + 1}/{len(files)}: {len(out[f])} detections", flush=True)
    json.dump(out, open(os.path.join(OUT, "preds_text.json"), "w"))
    print(f"{len(out)} images -> {OUT}/preds_text.json")


def score(args):
    import bootstrap_ci as bc

    data = os.path.join(ROOT, "external", "WTBD-group")
    te = bc.load_yolo_split(data, "test")
    files = sorted(te)
    by_name = {os.path.basename(k): k for k in te}
    raw = json.load(open(os.path.join(OUT, "preds_text.json")))
    preds = {"SAM 3 text": {by_name[n]: np.array(v).reshape(-1, 6) for n, v in raw.items()},
             "OWLv2 text": {k: np.array(v).reshape(-1, 6) for k, v in
                            json.load(open(os.path.join(ROOT, "runs", "analysis", "openvocab", "preds_text.json"))).items()}}
    for s in (0, 1):
        preds[f"ViT-B 25% s{s}"] = bc.predict(os.path.join(ROOT, "runs", "detect", "BladeYOLO_WindSurface",
                                                           f"bladeyolo2b_wtbd_group_f25_s{s}"), te,
                                              os.path.join(ROOT, "runs", "analysis", "bootstrap", "seeds_f25", "preds"))
    g = {}
    for r in csv.DictReader(open(os.path.join(ROOT, "runs", "analysis", "bootstrap", "wtbd_group", "groups.csv"))):
        g.setdefault(r["group"], []).append(os.path.join(data, "test", "images", r["image"]))
    groups = [[os.path.abspath(f) for f in v] for v in g.values()]
    rng = np.random.default_rng(0)
    draws = [[f for gi in rng.integers(0, len(groups), len(groups)) for f in groups[gi]] for _ in range(1000)]
    lines = ["# H13 stage A: SAM 3 zero-shot (text) vs OWLv2 text vs trained ViT-B 25%, WTBD-group test", ""]
    rare_beats = []
    for mode, nc in (("class-aware", 6), ("class-agnostic", 1)):
        imgs = te if nc == 6 else {k: dict(v, gt=np.c_[np.zeros(len(v["gt"])), v["gt"][:, 1:]]) for k, v in te.items()}
        P = preds if nc == 6 else {a: {k: np.c_[v[:, :5], np.zeros(len(v))] for k, v in d.items()} for a, d in preds.items()}
        ms = {a: bc.per_image_matches(imgs, P[a], nc) for a in P}
        full = {a: bc.score(ms[a], files, nc) for a in P}
        bs = {a: np.array([bc.score(ms[a], d, nc)[1] for d in draws]) for a in P}
        lines += [f"## {mode}", "", "| model | mAP50 | mAP50-95 | 95% CI mAP50-95 |"
                  + (" per-class AP50-95 (" + ", ".join(CLASSES) + ") |" if nc == 6 else ""),
                  "|---|---|---|---|" + ("---|" if nc == 6 else "")]
        for a in P:
            lines.append(f"| {a} | {full[a][0]:.3f} | {full[a][1]:.3f} | [{np.percentile(bs[a], 2.5):.3f}, "
                         f"{np.percentile(bs[a], 97.5):.3f}] |"
                         + (" " + " ".join("-" if np.isnan(x) else f"{x:.3f}" for x in full[a][3]) + " |" if nc == 6 else ""))
        d = bs["SAM 3 text"] - bs["OWLv2 text"]
        t = bs["SAM 3 text"] - np.mean([bs[f"ViT-B 25% s{s}"] for s in (0, 1)], 0)
        lines += ["", f"- SAM 3 − OWLv2 mAP50-95: {full['SAM 3 text'][1] - full['OWLv2 text'][1]:+.3f} "
                  f"[{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}], P(>0) {np.mean(d > 0):.3f}",
                  f"- SAM 3 − trained ViT-B 25% (mean of 2 seeds): [{np.percentile(t, 2.5):+.3f}, {np.percentile(t, 97.5):+.3f}]", ""]
        if nc == 6:
            for c in (3, 4):
                rare_beats.append((CLASSES[c], full["SAM 3 text"][3][c], full["OWLv2 text"][3][c]))
    stage_b = [c for c, a, b in rare_beats if a > b]
    lines.append(f"Stage B condition (SAM 3 above OWLv2 on a rare class, AP50-95): "
                 + ", ".join(f"{c} {a:.3f} vs {b:.3f}" for c, a, b in rare_beats)
                 + f" -> {'met (' + ', '.join(stage_b) + ')' if stage_b else 'not met'}")
    open(os.path.join(OUT, "report.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("predict", "score"))
    p.add_argument("--data", default=os.path.join(ROOT, "external", "WTBD-group", "data.yaml"))
    args = p.parse_args()
    predict(args) if args.mode == "predict" else score(args)


if __name__ == "__main__":
    main()
