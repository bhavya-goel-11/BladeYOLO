"""H10 synthesis (tools/synth_defects.py): the plan is deterministic and free of overlaps, the copy-paste arm changes
only the planned box and adds exactly one label, train.py accepts the resulting list, and one LoRA training step of a
tiny random inpainting UNet runs and changes only the LoRA weights.
"""

import os
import shutil
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import synth_defects as sd  # noqa: E402

OUT = os.path.join(ROOT, "runs", "tests", "synth_defects")


def fake_dataset():
    """6 images, 2 with thunderstrike, 1 with crack, in OUT/data (YOLO layout)."""
    d = os.path.join(OUT, "data")
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(os.path.join(d, "train", "images"))
    os.makedirs(os.path.join(d, "train", "labels"))
    rng = np.random.default_rng(0)
    labs = {"a.jpg": [(3, .3, .3, .2, .2)], "b.jpg": [(3, .7, .7, .2, .3), (1, .2, .8, .1, .1)],
            "c.jpg": [(4, .5, .5, .1, .4)], "d.jpg": [(2, .5, .5, .2, .2)], "e.jpg": [], "f.jpg": [(5, .4, .6, .3, .2)]}
    for n, rows in labs.items():
        f1, f2 = rng.uniform(2, 12, 2)  # smooth pattern, so JPEG re-encoding barely changes it
        yy, xx = np.mgrid[0:800, 0:800] / 800
        img = 127 + 120 * np.sin(2 * np.pi * (f1 * xx + f2 * yy))[..., None] * np.array([1, 0.6, -0.8])
        cv2.imwrite(os.path.join(d, "train", "images", n), img.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(os.path.join(d, "train", "labels", n[:-4] + ".txt"), "w") as f:
            f.write("".join(f"{c} {x} {y} {w} {h}\n" for c, x, y, w, h in rows))
    with open(os.path.join(d, "data.yaml"), "w") as f:
        f.write("path: .\ntrain: train/images\nval: train/images\nnc: 6\n"
                "names: [craze, corrosion, surface_injure, thunderstrike, crack, hide_craze]\n")
    return os.path.join(d, "data.yaml"), sorted(labs)


def test_plan_deterministic_and_free():
    data, names = fake_dataset()
    img_dir, lab_dir = sd.train_dirs(data)
    labels = {m: sd.read_labels(lab_dir, m) for m in names}
    rows = sd.make_plan(names, labels, 12, 2 / 3, 0)
    assert rows == sd.make_plan(names, labels, 12, 2 / 3, 0)
    assert sum(r["cls"] == 3 for r in rows) == 8 and sum(r["cls"] == 4 for r in rows) == 4
    for r in rows:
        assert not sd.overlaps((r["cx"], r["cy"], r["w"], r["h"]), labels[r["image"]])
        assert 0 <= r["cx"] - r["w"] / 2 and r["cx"] + r["w"] / 2 <= 1 + 1e-6
        assert any(int(b[0]) == r["cls"] for b in labels[r["src_image"]])
    print("ok test_plan_deterministic_and_free")


def test_paste_changes_only_the_box_and_train_accepts_list():
    import argparse
    import csv

    import train

    data, names = fake_dataset()
    img_dir, lab_dir = sd.train_dirs(data)
    labels = {m: sd.read_labels(lab_dir, m) for m in names}
    rows = sd.make_plan(names, labels, 4, 0.5, 1)
    plan = os.path.join(OUT, "plan.csv")
    with open(plan, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sd.FIELDS)
        w.writeheader()
        w.writerows(rows)
    rows = list(csv.DictReader(open(plan)))
    sd.paste(argparse.Namespace(out=os.path.join(OUT, "arm")), rows, names, img_dir, lab_dir)
    arm = os.path.join(OUT, "arm", "paste")
    r = rows[0]
    stem = f"h10paste_000_{r['image'][:-4]}"
    new, old = cv2.imread(os.path.join(arm, "images", stem + ".jpg")), cv2.imread(os.path.join(img_dir, r["image"]))
    x0, y0, x1, y1 = sd.box_px(*(float(r[k]) for k in ("cx", "cy", "w", "h")), *old.shape[:2])
    diff = np.abs(new.astype(int) - old.astype(int)).mean(2)
    outside = diff.copy()
    outside[y0:y1, x0:x1] = 0
    assert outside.mean() < 3 and diff[y0:y1, x0:x1].mean() > 20, (outside.mean(), diff[y0:y1, x0:x1].mean())
    assert len(sd.read_labels(os.path.join(arm, "labels"), stem + ".jpg")) == len(labels[r["image"]]) + 1
    lst = [l.strip() for l in open(os.path.join(arm, "train_list.txt"))]
    assert lst[:len(names)] == names and len(lst) == len(names) + 4
    out = train.resolve_data(data, os.path.join(arm, "train_list.txt"))
    import yaml
    imgs = [l.strip() for l in open(yaml.safe_load(open(out))["train"])]
    assert all(os.path.exists(p) for p in imgs) and sum("h10paste_" in p for p in imgs) == 4
    print("ok test_paste_changes_only_the_box_and_train_accepts_list")


def test_lora_step_tiny_unet():
    import torch
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel

    torch.manual_seed(0)
    unet = UNet2DConditionModel(sample_size=8, in_channels=9, out_channels=4, layers_per_block=1,
                                block_out_channels=(32, 64), down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
                                up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"), cross_attention_dim=32,
                                attention_head_dim=8, norm_num_groups=8)
    vae = AutoencoderKL(in_channels=3, out_channels=3, latent_channels=4, block_out_channels=(8, 16, 16, 16),
                        down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4,
                        norm_num_groups=8)
    before = {k: v.clone() for k, v in unet.state_dict().items()}
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (700, 900, 3), dtype=np.uint8)
    samples = [(img, (100, 200, 300, 260), 0), (img, (500, 100, 880, 690), 1)]
    old_side, sd.SIDE = sd.SIDE, 64  # 8x8 latents for speed
    try:
        losses = sd.train_lora(unet, vae, DDPMScheduler(), torch.randn(2, 77, 32), samples, steps=2, batch=2,
                               lr=1e-3, rank=4, seed=0, log_every=1)
    finally:
        sd.SIDE = old_side
    assert len(losses) == 2 and np.isfinite(losses).all()
    after = unet.state_dict()
    lora = [k for k in after if "lora" in k]
    assert lora and all(torch.equal(before[k], after[k]) for k in before if k in after)
    assert any(not torch.equal(after[k], torch.zeros_like(after[k])) for k in lora)
    from peft.utils import get_peft_model_state_dict
    from diffusers.utils import convert_state_dict_to_diffusers
    assert convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    print("ok test_lora_step_tiny_unet")


def test_window_holds_box():
    for H, W, box in ((800, 1000, (10, 20, 100, 300)), (1024, 1024, (0, 0, 1000, 900)), (788, 788, (700, 700, 788, 788))):
        rng = np.random.default_rng(0)
        for r in (None, rng, rng):
            x, y, s = sd.window(H, W, box, r)
            assert 0 <= x and x + s <= W and 0 <= y and y + s <= H
            if max(box[2] - box[0], box[3] - box[1]) <= s:
                assert x <= box[0] and box[2] <= x + s and y <= box[1] and box[3] <= y + s
    print("ok test_window_holds_box")


if __name__ == "__main__":
    test_window_holds_box()
    test_plan_deterministic_and_free()
    test_paste_changes_only_the_box_and_train_accepts_list()
    test_lora_step_tiny_unet()
