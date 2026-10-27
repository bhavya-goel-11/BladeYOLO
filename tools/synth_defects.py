"""H10 (Phase 29): few-shot generative synthesis of rare-class defects against a matched copy-paste control.

Both arms add the same N images to a labelled 25% subset: an image of the subset with one extra thunderstrike or crack
box, at a place fixed by a committed plan (same target images, classes, boxes and places in both arms). Only what fills
the box differs:
  paste    the real box the plan sampled its size from (same class, same subset), resized, feathered edge (CPU)
  synth    a Stable Diffusion inpainting model with a LoRA fitted on the subset's boxes of the two classes, inpainting
           the box, composited with the same feathered edge (GPU)
Only the subset's labels and images are used, so the test and val splits stay untouched.

  python tools/synth_defects.py plan    --data D --subset tools/splits/wtbd_group_train_f25_rand1.txt --plan tools/splits/h10_plan_rand1.csv
           (blade masks from SAM 2.1 at the labelled box centres; run once, locally; the plans are committed)
  python tools/synth_defects.py paste   --data D --subset ... --plan ... --out runs/synth/rand1
  python tools/synth_defects.py synth   --data D --subset ... --plan ... --out runs/synth/rand1   (fit, then inpaint)
-> runs/synth/rand1/<arm>/{images,labels}/, contact.jpg (before | after, first 16), and train_list.txt (the subset's
   names, then the new images) for train.py --train-subset.
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = (3, 4)  # thunderstrike, crack (WTBD official class ids)
PROMPT = {3: "a close-up photo of sks lightning strike damage on a wind turbine blade",
          4: "a close-up photo of sks crack on a wind turbine blade"}
SIDE = 512  # inpainting resolution
FIELDS = ["i", "image", "cls", "cx", "cy", "w", "h", "src_image", "src_cx", "src_cy", "src_w", "src_h"]


def train_dirs(data):
    """(images, labels) folders of the training split; the Kaggle zips unpack to train/train/images."""
    cfg = yaml.safe_load(open(data))
    root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(data)), cfg.get("path") or "."))
    for d in ("train/images", "train/train/images"):
        if os.path.isdir(os.path.join(root, d)):
            img = os.path.join(root, d)
            return img, os.path.join(os.path.dirname(img), "labels")
    raise FileNotFoundError(f"no train/images under {root}")


def read_labels(lab_dir, name):
    p = os.path.join(lab_dir, os.path.splitext(name)[0] + ".txt")
    rows = [list(map(float, l.split())) for l in open(p) if l.strip()] if os.path.exists(p) else []
    return np.array(rows, float).reshape(-1, 5)


def overlaps(box, labels, frac=0.1):
    """True if box (cx, cy, w, h) covers more than frac of an existing box, or an existing box covers more than frac
    of it (normalised coordinates)."""
    cx, cy, w, h = box
    for _, x, y, bw, bh in labels:
        iw = min(cx + w / 2, x + bw / 2) - max(cx - w / 2, x - bw / 2)
        ih = min(cy + h / 2, y + bh / 2) - max(cy - h / 2, y - bh / 2)
        if iw > 0 and ih > 0 and iw * ih > frac * min(w * h, bw * bh):
            return True
    return False


MASK_SIDE = 128  # blade masks are stored on a 128 x 128 grid of normalised image coordinates


def blade_masks(names, labels, img_dir, cache, weights):
    """{image: (128, 128) bool blade mask} for the subset's labelled images. Every labelled box lies on a blade, so SAM
    2.1 is prompted with a point at each box centre and the largest of its 3 masks is kept; the union is the blade.
    When SAM returns only the defect, the image has no free blade area and drops out as a target (it never marks sky
    as blade, which prompts beside the box did). Cached as PNGs in cache."""
    from ultralytics.models.sam import SAM2Predictor

    os.makedirs(cache, exist_ok=True)
    out, sam = {}, None
    for j, m in enumerate(n for n in names if len(labels[n])):
        p = os.path.join(cache, os.path.splitext(m)[0] + ".png")
        if not os.path.exists(p):
            if sam is None:
                sam = SAM2Predictor(overrides=dict(task="segment", mode="predict", imgsz=1024, model=weights, conf=0.0,
                                                   save=False, verbose=False, device="cpu"))
                sam.args.conf = -1.0  # keep every mask
            f = os.path.join(img_dir, m)
            H, W = cv2.imread(f).shape[:2]
            sam.set_image(f)
            blade = np.zeros((H, W), bool)
            for _, x, y, _, _ in labels[m]:
                k = sam(points=[[x * W, y * H]], labels=[1], multimask_output=True)[0].masks.data.cpu().numpy() > 0.5
                blade |= k[k.reshape(len(k), -1).sum(1).argmax()]
            sam.reset_image()
            cv2.imwrite(p, cv2.resize(blade.astype(np.uint8) * 255, (MASK_SIDE, MASK_SIDE), interpolation=cv2.INTER_AREA))
            print(f"  blade mask {j + 1}: {m} {blade.mean():.2f}", flush=True)
        out[m] = cv2.imread(p, cv2.IMREAD_GRAYSCALE) > 127
    return out


def on_blade(box, mask):
    """Fraction of box (cx, cy, w, h) on the blade mask."""
    x0, y0, x1, y1 = box_px(*box, MASK_SIDE, MASK_SIDE)
    return float(mask[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)].mean())


def make_plan(names, labels, n, frac_thunder, seed, masks=None, min_blade=0.8):
    """n rows: target image, class, box (size from a real box of the class, x0.8-1.2) not overlapping the image's
    boxes, and that real box as the paste source. With masks (blade_masks), targets are the images with a mask, the
    centre is a random blade pixel and at least min_blade of the box must be on the blade; without, the centre is
    near a real box centre of the subset."""
    rng = np.random.default_rng(seed)
    pool = {c: [(m, b) for m in names for b in labels[m] if int(b[0]) == c] for c in TARGET}
    centres = np.array([b[1:3] for m in names for b in labels[m]])
    blade_px = {m: (np.argwhere(k)[:, ::-1] + 0.5) / MASK_SIDE for m, k in (masks or {}).items() if k.any()}
    n3 = n if not pool[4] else 0 if not pool[3] else int(round(n * frac_thunder))
    seq = rng.permutation([3] * n3 + [4] * (n - n3))
    order, k, rows = rng.permutation(sorted(blade_px) if masks is not None else names), 0, []
    for i, c in enumerate(seq):
        for _ in range(20000):
            img = order[k % len(order)]
            k += 1
            src, sb = pool[c][rng.integers(len(pool[c]))]
            s = rng.uniform(0.8, 1.2)
            w, h = min(sb[3] * s, 0.95), min(sb[4] * s, 0.95)
            if masks is not None:
                cx, cy = blade_px[img][rng.integers(len(blade_px[img]))]
            else:
                cx, cy = centres[rng.integers(len(centres))] + rng.normal(0, 0.1, 2)
            cx, cy = np.clip(cx, w / 2, 1 - w / 2), np.clip(cy, h / 2, 1 - h / 2)
            if not overlaps((cx, cy, w, h), labels[img]) and (masks is None or on_blade((cx, cy, w, h), masks[img]) >= min_blade):
                break
        else:
            raise RuntimeError(f"no free place for row {i} (class {c})")
        rows.append(dict(i=i, image=img, cls=int(c), cx=round(float(cx), 5), cy=round(float(cy), 5), w=round(w, 5),
                         h=round(h, 5), src_image=src, src_cx=sb[1], src_cy=sb[2], src_w=sb[3], src_h=sb[4]))
    return rows


def box_px(cx, cy, w, h, H, W):
    x0, y0 = int(round((cx - w / 2) * W)), int(round((cy - h / 2) * H))
    x1, y1 = int(round((cx + w / 2) * W)), int(round((cy + h / 2) * H))
    return max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)


def feather(h, w, frac=0.08):
    """(h, w, 1) blending weight: 1 inside, ramping to 0 over the outer frac of the shorter side."""
    r = max(2.0, frac * min(h, w))
    dy, dx = np.minimum(np.arange(h), np.arange(h)[::-1]), np.minimum(np.arange(w), np.arange(w)[::-1])
    return np.clip((np.minimum.outer(dy, dx) + 1) / r, 0, 1)[..., None].astype(np.float32)


def blend(img, patch, box):
    """img with patch (already box-sized) composited into box with a feathered edge; outside box unchanged."""
    x0, y0, x1, y1 = box
    out = img.copy()
    a = feather(y1 - y0, x1 - x0)
    out[y0:y1, x0:x1] = (a * patch + (1 - a) * img[y0:y1, x0:x1]).round().clip(0, 255).astype(np.uint8)
    return out


def window(H, W, box, rng=None):
    """Square crop (x, y, side) holding the box, side max(512, 1.5 x box) within the image; random offset if rng."""
    x0, y0, x1, y1 = box
    s = int(min(max(SIDE, 1.5 * max(x1 - x0, y1 - y0)), H, W))
    org = []
    for a0, a1, L in ((x0, x1, W), (y0, y1, H)):
        lo, hi = max(0, a1 - s), min(a0, L - s)
        if lo > hi:  # box larger than the window: centre it
            lo = hi = int(np.clip((a0 + a1) / 2 - s / 2, 0, L - s))
        org.append(int(rng.integers(lo, hi + 1)) if rng is not None else (lo + hi) // 2)
    return org[0], org[1], s


def write_arm(out, arm, rows, names, img_dir, lab_dir, fill):
    """Write the arm's images and labels (the target's labels plus the new box), contact sheet and train list.
    fill(row, img, box) -> box-sized patch (BGR)."""
    d = os.path.join(out, arm)
    os.makedirs(os.path.join(d, "images"), exist_ok=True)
    os.makedirs(os.path.join(d, "labels"), exist_ok=True)
    rel, sheet = [], []
    for r in rows:
        img = cv2.imread(os.path.join(img_dir, r["image"]))
        H, W = img.shape[:2]
        box = box_px(float(r["cx"]), float(r["cy"]), float(r["w"]), float(r["h"]), H, W)
        new = blend(img, fill(r, img, box), box)
        stem = f"h10{arm}_{int(r['i']):03d}_{os.path.splitext(r['image'])[0]}"
        cv2.imwrite(os.path.join(d, "images", stem + ".jpg"), new, [cv2.IMWRITE_JPEG_QUALITY, 95])
        lab = read_labels(lab_dir, r["image"]).tolist() + [[int(r["cls"]), r["cx"], r["cy"], r["w"], r["h"]]]
        with open(os.path.join(d, "labels", stem + ".txt"), "w") as f:
            f.write("".join(f"{int(c)} {float(x):.6f} {float(y):.6f} {float(w):.6f} {float(h):.6f}\n" for c, x, y, w, h in lab))
        rel.append(os.path.relpath(os.path.join(d, "images", stem + ".jpg"), ROOT))
        if len(sheet) < 16:
            x, y, s = window(H, W, box)
            pair = [cv2.resize(im[y:y + s, x:x + s], (256, 256)) for im in (img, new)]
            sheet.append(np.concatenate(pair, 1))
        print(f"  {arm} {int(r['i']) + 1}/{len(rows)}", flush=True)
    if sheet:
        sheet += [np.zeros_like(sheet[0])] * (-len(sheet) % 4)
        cv2.imwrite(os.path.join(d, "contact.jpg"), np.concatenate([np.concatenate(sheet[i:i + 4], 1)
                                                                     for i in range(0, len(sheet), 4)], 0))
    with open(os.path.join(d, "train_list.txt"), "w") as f:
        f.write("\n".join(names + rel) + "\n")
    print(f"{arm}: {len(rel)} images -> {d}/train_list.txt ({len(names)} + {len(rel)} lines)")


def paste(args, rows, names, img_dir, lab_dir):
    def fill(r, img, box):
        src = cv2.imread(os.path.join(img_dir, r["src_image"]))
        sx0, sy0, sx1, sy1 = box_px(*(float(r[k]) for k in ("src_cx", "src_cy", "src_w", "src_h")), *src.shape[:2])
        return cv2.resize(src[sy0:sy1, sx0:sx1], (box[2] - box[0], box[3] - box[1]), interpolation=cv2.INTER_AREA)

    write_arm(args.out, "paste", rows, names, img_dir, lab_dir, fill)


# ---- generative arm (GPU; diffusers + peft) ----

def to_tensor_pair(img_rgb, box, rng):
    """Training sample: random window around the box at SIDE px, random flips and quarter turn. Returns image in
    [-1, 1] (3, S, S) and mask (1, S, S), 1 = inpaint (box dilated by 4 px)."""
    H, W = img_rgb.shape[:2]
    x, y, s = window(H, W, box, rng)
    crop = cv2.resize(img_rgb[y:y + s, x:x + s], (SIDE, SIDE), interpolation=cv2.INTER_AREA)
    f = SIDE / s
    m = np.zeros((SIDE, SIDE), np.float32)
    m[max(0, int((box[1] - y) * f) - 4):int((box[3] - y) * f) + 4, max(0, int((box[0] - x) * f) - 4):int((box[2] - x) * f) + 4] = 1
    if rng.random() < 0.5:
        crop, m = crop[:, ::-1], m[:, ::-1]
    if rng.random() < 0.5:
        crop, m = crop[::-1], m[::-1]
    q = int(rng.integers(4))
    crop, m = np.rot90(crop, q), np.rot90(m, q)
    return (np.ascontiguousarray(crop).transpose(2, 0, 1).astype(np.float32) / 127.5 - 1), np.ascontiguousarray(m)[None]


def train_lora(unet, vae, scheduler, emb, samples, steps, batch, lr, rank, seed, log_every=100):
    """LoRA on the UNet attention projections of an inpainting model (9 input channels: noisy latents, mask, masked
    image latents), epsilon objective, the mask being the defect box. samples: [(rgb image, box px, class row of emb)].
    Returns the per-step losses."""
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig

    dev = next(unet.parameters()).device
    cuda = dev.type == "cuda"
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    unet.add_adapter(LoraConfig(r=rank, lora_alpha=rank, init_lora_weights="gaussian",
                                target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
    params = [p for p in unet.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    sf = vae.config.scaling_factor
    losses = []
    unet.train()
    for step in range(steps):
        pick = [samples[j] for j in rng.integers(len(samples), size=batch)]
        pairs = [to_tensor_pair(im, box, rng) for im, box, _ in pick]
        x = torch.from_numpy(np.stack([p[0] for p in pairs])).to(dev, vae.dtype)
        m = torch.from_numpy(np.stack([p[1] for p in pairs])).to(dev, vae.dtype)
        with torch.no_grad():
            lat = vae.encode(x).latent_dist.sample() * sf
            mlat = vae.encode(x * (m < 0.5)).latent_dist.sample() * sf
        lat, mlat = lat.float(), mlat.float()
        mm = F.interpolate(m.float(), size=lat.shape[-2:])
        noise = torch.randn_like(lat)
        t = torch.randint(0, scheduler.config.num_train_timesteps, (batch,), device=dev)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=cuda):
            pred = unet(torch.cat([scheduler.add_noise(lat, noise, t), mm, mlat], 1), t,
                        encoder_hidden_states=emb[[c for _, _, c in pick]]).sample
        loss = F.mse_loss(pred.float(), noise)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        losses.append(loss.item())
        if (step + 1) % log_every == 0:
            print(f"  lora step {step + 1}/{steps} loss {np.mean(losses[-log_every:]):.4f}", flush=True)
    return losses


def lora_samples(names, labels, img_dir):
    out, cache = [], {}
    for m in names:
        for c, cx, cy, w, h in labels[m]:
            if int(c) in TARGET:
                if m not in cache:
                    cache[m] = cv2.cvtColor(cv2.imread(os.path.join(img_dir, m)), cv2.COLOR_BGR2RGB)
                H, W = cache[m].shape[:2]
                out.append((cache[m], box_px(cx, cy, w, h, H, W), TARGET.index(int(c))))
    return out


def fit(args, names, labels, img_dir):
    import torch
    from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionInpaintPipeline, UNet2DConditionModel
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft.utils import get_peft_model_state_dict
    from transformers import CLIPTextModel, CLIPTokenizer

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = CLIPTokenizer.from_pretrained(args.base, subfolder="tokenizer")
    te = CLIPTextModel.from_pretrained(args.base, subfolder="text_encoder").to(dev)
    with torch.no_grad():
        ids = tok([PROMPT[c] for c in TARGET], padding="max_length", max_length=tok.model_max_length,
                  truncation=True, return_tensors="pt").input_ids.to(dev)
        emb = te(ids)[0].float()
    del te
    vae = AutoencoderKL.from_pretrained(args.base, subfolder="vae", torch_dtype=torch.float16 if dev.type == "cuda"
                                        else torch.float32).to(dev)
    unet = UNet2DConditionModel.from_pretrained(args.base, subfolder="unet").to(dev)
    unet.enable_gradient_checkpointing()
    sched = DDPMScheduler.from_pretrained(args.base, subfolder="scheduler")
    samples = lora_samples(names, labels, img_dir)
    print(f"LoRA on {len(samples)} boxes: " + ", ".join(f"class {c}: {sum(s[2] == j for s in samples)}"
                                                       for j, c in enumerate(TARGET)), flush=True)
    losses = train_lora(unet, vae, sched, emb, samples, args.steps, args.lora_batch, args.lr, args.rank, args.seed)
    d = os.path.join(args.out, "lora")
    StableDiffusionInpaintPipeline.save_lora_weights(
        d, unet_lora_layers=convert_state_dict_to_diffusers(get_peft_model_state_dict(unet)), safe_serialization=True)
    np.savetxt(os.path.join(d, "losses.txt"), losses, fmt="%.5f")
    print(f"LoRA saved to {d}", flush=True)


def synth(args, rows, names, labels, img_dir, lab_dir):
    import torch
    from diffusers import StableDiffusionInpaintPipeline
    from PIL import Image

    if not os.path.exists(os.path.join(args.out, "lora", "pytorch_lora_weights.safetensors")):
        fit(args, names, labels, img_dir)
    import gc
    gc.collect()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.base, torch_dtype=torch.float16 if dev == "cuda" else torch.float32, safety_checker=None,
        requires_safety_checker=False).to(dev)
    pipe.load_lora_weights(os.path.join(args.out, "lora"))
    pipe.set_progress_bar_config(disable=True)

    def fill(r, img, box):
        H, W = img.shape[:2]
        x, y, s = window(H, W, box)
        f = SIDE / s
        crop = cv2.resize(cv2.cvtColor(img[y:y + s, x:x + s], cv2.COLOR_BGR2RGB), (SIDE, SIDE), interpolation=cv2.INTER_AREA)
        m = np.zeros((SIDE, SIDE), np.uint8)
        m[max(0, int((box[1] - y) * f) - 4):int((box[3] - y) * f) + 4, max(0, int((box[0] - x) * f) - 4):int((box[2] - x) * f) + 4] = 255
        g = torch.Generator(dev).manual_seed(args.seed + int(r["i"]))
        gen = pipe(prompt=PROMPT[int(r["cls"])], image=Image.fromarray(crop), mask_image=Image.fromarray(m),
                   height=SIDE, width=SIDE, num_inference_steps=args.infer_steps, guidance_scale=args.guidance,
                   generator=g).images[0]
        gen = cv2.resize(cv2.cvtColor(np.asarray(gen), cv2.COLOR_RGB2BGR), (s, s), interpolation=cv2.INTER_CUBIC)
        return gen[box[1] - y:box[3] - y, box[0] - x:box[2] - x]

    write_arm(args.out, "synth", rows, names, img_dir, lab_dir, fill)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("plan", "paste", "synth"))
    p.add_argument("--data", required=True)
    p.add_argument("--subset", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--out", default=None, help="output folder (paste, synth)")
    p.add_argument("--n", type=int, default=96, help="plan: images added")
    p.add_argument("--frac-thunder", type=float, default=2 / 3, help="plan: share of thunderstrike rows (rest crack)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--masks", default=os.path.join(ROOT, "runs", "synth", "blade_masks"), help="plan: blade mask cache")
    p.add_argument("--sam", default=os.path.join(ROOT, "runs", "analysis", "sam_refine", "sam2.1_t.pt"),
                   help="plan: SAM 2.1 weights for the blade masks")
    p.add_argument("--base", default="stable-diffusion-v1-5/stable-diffusion-inpainting")
    p.add_argument("--steps", type=int, default=1500, help="LoRA steps")
    p.add_argument("--lora-batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--infer-steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=7.5)
    args = p.parse_args()
    img_dir, lab_dir = train_dirs(args.data)
    names = [n.strip() for n in open(args.subset) if n.strip()]
    labels = {m: read_labels(lab_dir, m) for m in names}
    if args.mode == "plan":
        masks = blade_masks(names, labels, img_dir, args.masks, args.sam)
        rows = make_plan(names, labels, args.n, args.frac_thunder, args.seed, masks)
        with open(args.plan, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"{len(rows)} rows -> {args.plan}: " + ", ".join(f"class {c}: {sum(r['cls'] == c for r in rows)}" for c in TARGET))
        return
    rows = list(csv.DictReader(open(args.plan)))
    missing = {r["image"] for r in rows} - set(names)
    if missing:
        sys.exit(f"plan {args.plan} names images outside {args.subset}, e.g. {sorted(missing)[0]}")
    if args.mode == "paste":
        paste(args, rows, names, img_dir, lab_dir)
    else:
        synth(args, rows, names, labels, img_dir, lab_dir)


if __name__ == "__main__":
    main()
