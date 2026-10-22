"""Open-vocabulary detection on the leak-free WTBD split, without training (Phase 28, round 2, item 4).

OWLv2 (google/owlv2-base-patch16-ensemble) predicts a box and a class embedding per patch (60x60 at 960 px); a class
is scored by the head's similarity between that embedding and a query. Two kinds of query:
  text   several fixed descriptions per class (PROMPTS, written before any result); class score = best description
  proto  the labelled 25% subset as exemplars, without training: for every labelled box, the class embedding of the
         patch whose predicted box overlaps it most; per class the mean (L2-normalised) is the query. A few-shot
         detector from the same labels the trained detectors get.
Both are scored with the bootstrap evaluator (tools/bootstrap_ci.py) against the trained ViT-B detector at 25%
(cached predictions of bladeyolo2b_wtbd_group_f25_s0/s1), class-aware and class-agnostic (every class -> "defect").

Usage: .venv/bin/python tools/openvocab_eval.py  -> runs/analysis/openvocab/{preds_*.json, report.md}
"""

import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import bootstrap_ci as bc  # noqa: E402
import detection_metrics as dm  # noqa: E402

MODEL = "google/owlv2-base-patch16-ensemble"
LOCAL = os.path.join(ROOT, "external", "owlv2")  # a local copy (config, tokenizer, weights) is used when present
CLASSES = ["craze", "corrosion", "surface_injure", "thunderstrike", "crack", "hide_craze"]
PROMPTS = {
    "craze": ["crazing on a wind turbine blade", "a network of fine hairline cracks in the coating", "craze"],
    "corrosion": ["leading edge erosion on a wind turbine blade", "corroded blade surface", "corrosion"],
    "surface_injure": ["surface damage on a wind turbine blade", "a scratch or chipped paint on a blade", "surface injury"],
    "thunderstrike": ["lightning strike damage on a wind turbine blade", "a burn mark", "thunderstrike"],
    "crack": ["a crack in a wind turbine blade", "a structural crack", "crack"],
    "hide_craze": ["faint hidden crazing on a wind turbine blade", "subtle hairline cracks under the coating", "hidden craze"],
}
OUT = os.path.join(ROOT, "runs", "analysis", "openvocab")
TOP = 300


class Owl:
    def __init__(self):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        torch.set_num_threads(os.cpu_count())
        src = LOCAL if os.path.exists(os.path.join(LOCAL, "model.safetensors")) else MODEL
        self.proc = Owlv2Processor.from_pretrained(src)
        self.model = Owlv2ForObjectDetection.from_pretrained(src).eval()

    @torch.no_grad()
    def embed(self, path):
        """Per patch: boxes (n, 4) xyxy in 640-input px, class embeddings (n, d), image features for the head."""
        from PIL import Image

        im = Image.open(path).convert("RGB")
        px = self.proc(images=im, return_tensors="pt")["pixel_values"]
        fmap = self.model.image_embedder(px)[0]
        b, h, w, d = fmap.shape
        feats = fmap.reshape(b, h * w, d)
        box = self.model.box_predictor(feats, fmap)[0]  # cxcywh in [0, 1] of the padded square
        # boxes are normalised to the padded square (side = the long side), predictions live at 640 / long side
        xyxy = torch.cat([box[:, :2] - box[:, 2:] / 2, box[:, :2] + box[:, 2:] / 2], 1) * 640
        _, cls_emb = self.model.class_predictor(feats)
        return xyxy.numpy(), cls_emb[0], feats

    @torch.no_grad()
    def text_queries(self):
        flat = [p for c in CLASSES for p in PROMPTS[c]]
        t = self.proc(text=[flat], return_tensors="pt")
        q = self.model.owlv2.get_text_features(input_ids=t["input_ids"], attention_mask=t["attention_mask"])
        owner = np.repeat(np.arange(len(CLASSES)), [len(PROMPTS[c]) for c in CLASSES])
        return q, owner

    @torch.no_grad()
    def scores(self, feats, queries, owner):
        """(n_patches, nc) class scores: sigmoid of the head's logits, best query per class."""
        logits, _ = self.model.class_predictor(feats, queries[None], torch.ones(1, len(queries), dtype=torch.bool))
        p = logits[0].sigmoid().numpy()
        return np.stack([p[:, owner == c].max(1) for c in range(len(CLASSES))], 1)


def detections(boxes, sc):
    """Top-TOP (box, class) pairs after per-class NMS -> (n, 6) x1 y1 x2 y2 score cls."""
    import torchvision

    n, nc = sc.shape
    b = torch.from_numpy(np.repeat(boxes, nc, 0)).float()
    s = torch.from_numpy(sc.reshape(-1)).float()
    c = torch.arange(nc).repeat(n)
    top = s.topk(min(3000, len(s))).indices
    keep = top[torchvision.ops.batched_nms(b[top], s[top], c[top], 0.5)][:TOP]
    return np.c_[b[keep].numpy(), s[keep].numpy(), c[keep].numpy()]


def main():
    os.makedirs(OUT, exist_ok=True)
    data = os.path.join(ROOT, "external", "WTBD-group")
    te = bc.load_yolo_split(data, "test")
    sub = {n.strip() for n in open(os.path.join(ROOT, "tools", "splits", "wtbd_group_train_f25.txt")) if n.strip()}
    tr = {k: v for k, v in bc.load_yolo_split(data, "train").items() if os.path.basename(k) in sub}
    files = sorted(te)
    cache = {a: os.path.join(OUT, f"preds_{a}.json") for a in ("text", "proto")}
    if not all(os.path.exists(p) for p in cache.values()):
        owl = Owl()
        protos = [[] for _ in CLASSES]
        for i, p in enumerate(sorted(tr)):  # exemplar embeddings from the labelled 25%
            boxes, emb, _ = owl.embed(p)
            for c, *g in tr[p]["gt"]:
                j = int(dm.iou_matrix(np.array([g]), boxes)[0].argmax())
                protos[int(c)].append(torch.nn.functional.normalize(emb[j], dim=0))
            if i % 25 == 0:
                print(f"  exemplars {i}/{len(tr)}", flush=True)
        pq = torch.stack([torch.nn.functional.normalize(torch.stack(v).mean(0), dim=0) for v in protos])
        tq, towner = owl.text_queries()
        out = {"text": {}, "proto": {}}
        for i, p in enumerate(files):
            boxes, _, feats = owl.embed(p)
            out["text"][p] = detections(boxes, owl.scores(feats, tq, towner)).round(3).tolist()
            out["proto"][p] = detections(boxes, owl.scores(feats, pq, np.arange(len(CLASSES)))).round(3).tolist()
            if i % 20 == 0:
                print(f"  test {i}/{len(files)}", flush=True)
        for a in out:
            json.dump(out[a], open(cache[a], "w"))
        print("exemplars per class: " + ", ".join(f"{c} {len(v)}" for c, v in zip(CLASSES, protos)))
    preds = {a: {k: np.array(v).reshape(-1, 6) for k, v in json.load(open(cache[a])).items()} for a in cache}
    for s in (0, 1):
        preds[f"ViT-B 25% s{s}"] = bc.predict(os.path.join(ROOT, "runs", "detect", "BladeYOLO_WindSurface", f"bladeyolo2b_wtbd_group_f25_s{s}"),
                                              te, os.path.join(ROOT, "runs", "analysis", "bootstrap", "seeds_f25", "preds"))

    import csv

    g = {}
    for r in csv.DictReader(open(os.path.join(ROOT, "runs", "analysis", "bootstrap", "wtbd_group", "groups.csv"))):
        g.setdefault(r["group"], []).append(os.path.join(data, "test", "images", r["image"]))
    groups = list(g.values())
    rng = np.random.default_rng(0)
    draws = [[f for gi in rng.integers(0, len(groups), len(groups)) for f in groups[gi]] for _ in range(1000)]
    lines = ["# Open-vocabulary detection without training (OWLv2) vs the trained ViT-B detector, WTBD-group test", ""]
    for mode, nc in (("class-aware", 6), ("class-agnostic", 1)):
        imgs = te if nc == 6 else {k: dict(v, gt=np.c_[np.zeros(len(v["gt"])), v["gt"][:, 1:]]) for k, v in te.items()}
        P = preds if nc == 6 else {a: {k: np.c_[v[:, :5], np.zeros(len(v))] for k, v in d.items()} for a, d in preds.items()}
        ms = {a: bc.per_image_matches(imgs, P[a], nc) for a in P}
        full = {a: bc.score(ms[a], files, nc) for a in P}
        bs = {a: np.array([bc.score(ms[a], d, nc)[:2] for d in draws]) for a in P}
        lines += [f"## {mode}", "", "| model | mAP50 | mAP50-95 | 95% CI mAP50-95 |" + (" per-class AP50 |" if nc == 6 else ""), "|---|---|---|---|" + ("---|" if nc == 6 else "")]
        for a in P:
            lines.append(f"| {a} | {full[a][0]:.3f} | {full[a][1]:.3f} | [{np.percentile(bs[a][:, 1], 2.5):.3f}, {np.percentile(bs[a][:, 1], 97.5):.3f}] |"
                         + (" " + " ".join("-" if np.isnan(x) else f"{x:.2f}" for x in full[a][2]) + " |" if nc == 6 else ""))
        trained = np.mean([bs[f"ViT-B 25% s{s}"] for s in (0, 1)], 0)
        for a in ("text", "proto"):
            d = bs[a] - trained
            lines.append(f"\n{a} − trained (mean of 2 seeds): mAP50 {np.mean(d[:, 0]):+.3f} [{np.percentile(d[:, 0], 2.5):+.3f}, {np.percentile(d[:, 0], 97.5):+.3f}], "
                         f"mAP50-95 {np.mean(d[:, 1]):+.3f} [{np.percentile(d[:, 1], 2.5):+.3f}, {np.percentile(d[:, 1], 97.5):+.3f}]")
        lines.append("")
    open(os.path.join(OUT, "report.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
