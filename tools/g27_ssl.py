"""G27 direction 1 (research_log.md): in-domain self-supervised adaptation of RF-DETR-M's DINOv2 encoder on the G27
pool images (tools/g27_pool.py; labels unused), written back into the COCO RF-DETR-M checkpoint so every fine-tune
starts from it via rfdetr_v1.py train --init.

Objective (masked feature distillation, data2vec-style, chosen because it needs no new heads at fine-tune time and
starts at zero drift): an EMA teacher sees the full crop; the student sees the same crop with ~50% of its 16-px patches
replaced by the encoder's own mask token (block masking) and regresses, at the masked patches only, the teacher's
average of its four output feature maps (stages 3/6/9/12, each layer-normalised), through a 1x1 linear head.
Only the encoder (backbone.0.encoder.encoder.*) changes; projector, decoder and heads stay COCO's. The EMA teacher's
weights are written out.

  Kaggle:  python tools/g27_ssl.py --images <g27-pool>/train/images --out runs/ssl/g27_d1 --steps 8000
  -> runs/ssl/g27_d1/rfdetrm_g27ssl.pth (RF-DETR checkpoint: {model, args}), log.csv, options.json
  CPU smoke test: tests/test_g27_ssl.py
"""

import argparse
import copy
import csv
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

ENC = "backbone.0.encoder.encoder."
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def coco_checkpoint(path=None):
    """Path of RF-DETR-M's COCO checkpoint (downloaded by rfdetr when path is None)."""
    if path:
        return path
    from rfdetr import RFDETRMedium

    m = RFDETRMedium()
    return m.model_config.pretrain_weights if os.path.isabs(m.model_config.pretrain_weights) else os.path.abspath(m.model_config.pretrain_weights)


def build_encoder(ckpt):
    """The windowed DINOv2 encoder of RF-DETR-M with the checkpoint's weights."""
    from rfdetr import RFDETRMedium

    nn_model = RFDETRMedium(pretrain_weights=ckpt).model.model
    return nn_model.backbone[0].encoder.encoder


def block_mask(n_side, ratio, rng, min_block=4, max_block=12):
    """(n_side**2,) bool: rectangular blocks of patches until `ratio` of the grid is covered (BEiT-style)."""
    m = np.zeros((n_side, n_side), bool)
    target = int(round(ratio * n_side * n_side))
    while m.sum() < target:
        h, w = rng.integers(min_block, max_block + 1, 2)
        y, x = rng.integers(0, n_side - h + 1), rng.integers(0, n_side - w + 1)
        m[y:y + h, x:x + w] = True
    flat = m.reshape(-1)
    extra = flat.sum() - target  # trim the overshoot so every image masks exactly `target` patches
    if extra > 0:
        on = np.flatnonzero(flat)
        flat[rng.choice(on, extra, replace=False)] = False
    return flat


def forward_masked(enc, x, mask=None):
    """Feature maps (B, C, h, w) of the 4 output stages; mask (B, h*w) bool replaces patches with the mask token
    before windowing (RF-DETR's backbone forward does not pass bool_masked_pos through, so the embedding call is
    wrapped for this one call)."""
    if mask is None:
        return list(enc(x)[0])
    emb = enc.embeddings
    orig = type(emb).forward
    emb.forward = lambda pv, bool_masked_pos=None: orig(emb, pv, bool_masked_pos=mask)
    try:
        return list(enc(x)[0])
    finally:
        del emb.forward


def target_of(maps):
    return torch.stack([F.layer_norm(m.permute(0, 2, 3, 1), m.shape[1:2]) for m in maps]).mean(0)  # (B, h, w, C)


class Crops(torch.utils.data.Dataset):
    def __init__(self, files, side, seed=0):
        self.files, self.side, self.seed = files, side, seed

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        from PIL import Image
        import torchvision.transforms.v2 as T

        im = Image.open(self.files[i]).convert("RGB")
        tf = T.Compose([T.RandomResizedCrop(self.side, scale=(0.3, 1.0), antialias=True), T.RandomHorizontalFlip(),
                        T.RandomVerticalFlip(), T.ColorJitter(0.2, 0.2, 0.1, 0.02), T.ToImage(),
                        T.ToDtype(torch.float32, scale=True), T.Normalize(MEAN, STD)])
        return tf(im)


def write_checkpoint(ckpt_in, enc, out):
    c = torch.load(ckpt_in, map_location="cpu", weights_only=False)
    model = c["model"]
    sd = {ENC + k: v.detach().cpu().clone() for k, v in enc.state_dict().items()}
    missing = [k for k in sd if k not in model]
    assert not missing, f"encoder keys not in the checkpoint: {missing[:3]}"
    model.update(sd)
    torch.save({"model": model, "args": c.get("args")}, out)
    return out


def train(images, out, steps, batch=16, side=576, lr=5e-5, wd=0.05, warmup=500, ratio=0.5, ema=(0.999, 0.9999),
          workers=4, seed=0, ckpt=None, log_every=50):
    os.makedirs(out, exist_ok=True)
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    ckpt = coco_checkpoint(ckpt)
    files = sorted(os.path.join(images, f) for f in os.listdir(images))
    opts = dict(images=os.path.abspath(images), n_images=len(files), steps=steps, batch=batch, side=side, lr=lr, wd=wd,
                warmup=warmup, mask_ratio=ratio, ema=list(ema), seed=seed, ckpt=ckpt)
    json.dump(opts, open(os.path.join(out, "options.json"), "w"), indent=1)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    student = build_encoder(ckpt).to(dev).train()
    teacher = copy.deepcopy(student).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    C = student.config.hidden_size
    head = torch.nn.Linear(C, C).to(dev)
    params = [{"params": [p for n, p in student.named_parameters() if p.ndim > 1], "weight_decay": wd},
              {"params": [p for n, p in student.named_parameters() if p.ndim <= 1], "weight_decay": 0.0},
              {"params": head.parameters(), "weight_decay": wd}]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    amp = dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    dl = torch.utils.data.DataLoader(Crops(files, side, seed), batch_size=batch, shuffle=True, num_workers=workers,
                                     drop_last=True, persistent_workers=workers > 0)
    n_side = side // student.config.patch_size
    rng = np.random.default_rng(seed)
    log = open(os.path.join(out, "log.csv"), "w", newline="")
    w = csv.writer(log)
    w.writerow(["step", "loss", "lr", "ema", "sec"])
    step, t0 = 0, time.time()
    while step < steps:
        for x in dl:
            if step >= steps:
                break
            x = x.to(dev, non_blocking=True)
            mask = torch.from_numpy(np.stack([block_mask(n_side, ratio, rng) for _ in range(len(x))])).to(dev)
            with torch.autocast(dev, dtype=torch.float16, enabled=amp):
                with torch.no_grad():
                    tgt = target_of(forward_masked(teacher, x))
                pred = head(forward_masked(student, x, mask)[-1].permute(0, 2, 3, 1))
            m = mask.view(len(x), n_side, n_side)
            loss = F.mse_loss(pred.float()[m], tgt.float()[m])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(list(student.parameters()) + list(head.parameters()), 3.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tau = ema[1] - (ema[1] - ema[0]) * 0.5 * (1 + math.cos(math.pi * step / steps))
            with torch.no_grad():
                for pt, ps in zip(teacher.parameters(), student.parameters()):
                    pt.mul_(tau).add_(ps.detach(), alpha=1 - tau)
            assert torch.isfinite(loss), f"non-finite loss at step {step}"
            if step % log_every == 0 or step == steps - 1:
                w.writerow([step, f"{loss.item():.5f}", f"{sched.get_last_lr()[0]:.3e}", f"{tau:.5f}", f"{time.time() - t0:.0f}"])
                log.flush()
                print(f"step {step}/{steps} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} tau {tau:.5f} "
                      f"{time.time() - t0:.0f}s", flush=True)
            step += 1
    log.close()
    path = write_checkpoint(ckpt, teacher, os.path.join(out, "rfdetrm_g27ssl.pth"))
    print(f"-> {path}")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--side", type=int, default=576)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default=None, help="RF-DETR-M COCO checkpoint (default: rfdetr's download)")
    a = ap.parse_args()
    train(a.images, a.out, a.steps, a.batch, a.side, a.lr, workers=a.workers, seed=a.seed, ckpt=a.ckpt)


if __name__ == "__main__":
    main()
