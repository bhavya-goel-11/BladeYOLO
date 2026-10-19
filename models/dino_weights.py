"""Frozen DINOv3 ViT-S/16, ViT-B/16 or ViT-L/16 for BladeYOLO-v2 (models/dino_fusion.py): source on the path,
weights from disk, in Meta's original .pth layout or the Hugging Face transformers layout (converted on load)."""

import glob
import json
import os
import sys
import zipfile

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DINOV3_REPO_ZIP = "https://github.com/facebookresearch/dinov3/archive/refs/heads/main.zip"
# size: (hub constructor, width, depth, blocks fused into P3/P4/P5 at the same relative depths 1/3, 2/3, 3/3)
SIZES = {
    "s": ("dinov3_vits16", 384, 12, (3, 7, 11)),
    "b": ("dinov3_vitb16", 768, 12, (3, 7, 11)),
    "l": ("dinov3_vitl16", 1024, 24, (7, 15, 23)),
}


def dino_size():
    """DINOv3 size for newly built models (BLADEYOLO_DINO, train.py --dino): s (default), b or l."""
    size = os.environ.get("BLADEYOLO_DINO", "s") or "s"
    if size not in SIZES:
        raise ValueError(f"BLADEYOLO_DINO={size!r}: expected one of {sorted(SIZES)}")
    return size


def weight_paths(size):
    """Candidate weight files, in order: DINOV3_WEIGHTS, the Kaggle model (beegee11/vit?16, the official Hugging Face
    release facebook/dinov3-vit?16-pretrain-lvd1689m; latest version first), then the local download
    (external/dinov3/vit?16/model.safetensors, from `hf download ... --local-dir`)."""
    v = f"vit{size}16"
    kaggle = glob.glob(f"/kaggle/input/models/beegee11/{v}/*/default/*/model.safetensors")
    kaggle.sort(key=lambda p: int(p.split("/")[-2]) if p.split("/")[-2].isdigit() else -1, reverse=True)
    return [os.environ.get("DINOV3_WEIGHTS", ""), *kaggle, os.path.join(ROOT_DIR, "external", "dinov3", v, "model.safetensors")]


def ensure_dinov3_importable():
    """Put the DINOv3 source on sys.path, downloading it into the torch hub cache if needed.

    The repo's hubconf pulls in segmentation/eval dependencies we do not need, so the backbone
    is imported from dinov3.hub.backbones directly. The path must stay on sys.path because
    checkpoints pickle the DINOv3 classes by reference (and DDP workers inherit sys.path).
    """
    try:
        import dinov3  # noqa: F401  (pip-installed or already on the path)

        return
    except ImportError:
        pass
    repo = os.path.join(torch.hub.get_dir(), "dinov3-main")
    if not os.path.isdir(repo):
        os.makedirs(torch.hub.get_dir(), exist_ok=True)
        zip_path = repo + ".zip"
        torch.hub.download_url_to_file(DINOV3_REPO_ZIP, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(torch.hub.get_dir())
        os.remove(zip_path)
    sys.path.insert(0, repo)


def hf_to_dinov3(hf):
    """Hugging Face DINOv3ViTModel state dict -> the dinov3 repo's layout. q/k/v are fused into qkv; Hugging Face has
    no key bias (key_bias=false), which is a zero block in qkv.bias. Buffers (rope periods, qkv bias mask) are not
    stored by Hugging Face and keep the values the dinov3 constructor gives them."""
    out = {"cls_token": hf["embeddings.cls_token"], "storage_tokens": hf["embeddings.register_tokens"],
           "mask_token": hf["embeddings.mask_token"].reshape(1, -1),
           "patch_embed.proj.weight": hf["embeddings.patch_embeddings.weight"],
           "patch_embed.proj.bias": hf["embeddings.patch_embeddings.bias"],
           "norm.weight": hf["norm.weight"], "norm.bias": hf["norm.bias"]}
    n = 1 + max(int(k.split(".")[1]) for k in hf if k.startswith("layer."))
    for i in range(n):
        h, o = f"layer.{i}.", f"blocks.{i}."
        q, k, v = (hf[h + f"attention.{x}_proj.weight"] for x in "qkv")
        out[o + "attn.qkv.weight"] = torch.cat([q, k, v])
        kb = hf.get(h + "attention.k_proj.bias", torch.zeros(k.shape[0], dtype=q.dtype))
        out[o + "attn.qkv.bias"] = torch.cat([hf[h + "attention.q_proj.bias"], kb, hf[h + "attention.v_proj.bias"]])
        for a, b in (("attention.o_proj", "attn.proj"), ("norm1", "norm1"), ("norm2", "norm2"),
                     ("mlp.up_proj", "mlp.fc1"), ("mlp.down_proj", "mlp.fc2")):
            out[o + b + ".weight"], out[o + b + ".bias"] = hf[h + a + ".weight"], hf[h + a + ".bias"]
        out[o + "ls1.gamma"], out[o + "ls2.gamma"] = hf[h + "layer_scale1.lambda1"], hf[h + "layer_scale2.lambda1"]
    unused = set(hf) - {k for k in hf if k.startswith(("embeddings.", "norm.")) or k.startswith("layer.")}
    if unused:
        raise RuntimeError(f"Unconverted Hugging Face DINOv3 tensors: {sorted(unused)}")
    return out


def hf_buffers(model):
    """Set the two buffers Hugging Face does not store as in Meta's release: the RoPE periods rounded to bfloat16 (the
    released checkpoints store them in bf16, so the model was trained with rounded periods) and a qkv bias mask that
    keeps the stored q and v biases and drops the (absent) k bias. Verified on ViT-S/16 (tests/test_dino_sizes.py)."""
    with torch.no_grad():
        model.rope_embed.periods.copy_(model.rope_embed.periods.bfloat16().float())
        for blk in model.blocks:
            m = blk.attn.qkv.bias_mask
            m.fill_(1.0)
            n = m.numel() // 3
            m[n:2 * n] = 0.0
    return model


def load_state(path, size):
    """State dict in the dinov3 repo's layout from a .pth or a Hugging Face model.safetensors (checked against its
    config.json when present, so a file of the wrong size is refused)."""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        cfg = os.path.join(os.path.dirname(path), "config.json")
        if os.path.exists(cfg):
            c = json.load(open(cfg))
            _, dim, depth, _ = SIZES[size]
            if (c.get("hidden_size"), c.get("num_hidden_layers")) != (dim, depth):
                raise RuntimeError(f"{path} is not DINOv3 ViT-{size.upper()}/16: {c.get('hidden_size')} wide, "
                                   f"{c.get('num_hidden_layers')} blocks")
        return hf_to_dinov3(load_file(path))
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    return {k.removeprefix("backbone."): v for k, v in state.items()}


def build_dinov3(size="s"):
    ensure_dinov3_importable()
    from dinov3.hub import backbones

    name = SIZES[size][0]
    model = getattr(backbones, name)(pretrained=False)
    paths = [p for p in weight_paths(size) if p]
    path = next((p for p in paths if os.path.exists(p)), None)
    if not path:
        raise FileNotFoundError(f"DINOv3 {name} weights not found in {paths}. Set DINOV3_WEIGHTS=/path/to/file "
                                f"(.pth or Hugging Face model.safetensors).")
    missing, unexpected = model.load_state_dict(load_state(path, size), strict=False)
    params = dict(model.named_parameters())
    if [k for k in missing if k in params] or unexpected:
        raise RuntimeError(f"DINOv3 weights do not match the model. Missing: {missing}. Unexpected: {unexpected}.")
    if path.endswith(".safetensors"):
        hf_buffers(model)
    print(f"[BladeYOLO] Loaded DINOv3 {name} weights from {path}")
    return model.float()


def build_dinov3_vits16():
    return build_dinov3("s")
