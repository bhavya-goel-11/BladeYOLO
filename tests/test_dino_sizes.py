"""DINOv3 sizes: the Hugging Face weights convert exactly to Meta's layout, and ViT-S/B/L build and fuse correctly.

Needs the real weights (gated): external/dinov3/vit{s,b,l}16/model.safetensors from `hf download
facebook/dinov3-vit?16-pretrain-lvd1689m --local-dir external/dinov3/vit?16`, and Meta's ViT-S .pth under
external/dinov3/official/. Tests whose files are missing are skipped.
"""

import glob
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from models import dino_weights as dw  # noqa: E402

HF = {s: os.path.join(ROOT, "external", "dinov3", f"vit{s}16", "model.safetensors") for s in "sbl"}
OFFICIAL_S = (glob.glob(os.path.join(ROOT, "external", "dinov3", "official", "**", "dinov3_vits16_pretrain_*.pth"),
                        recursive=True) or [""])[0]


def _skip(*paths):
    missing = [p for p in paths if not p or not os.path.exists(p)]
    if missing:
        print(f"skip: missing {missing}")
    return bool(missing)


def test_hf_conversion_matches_official_vits():
    """Every converted tensor equals Meta's original ViT-S/16 release, and both give the same features."""
    if _skip(HF["s"], OFFICIAL_S):
        return
    ref, conv = dw.load_state(OFFICIAL_S, "s"), dw.load_state(HF["s"], "s")
    dw.ensure_dinov3_importable()
    from dinov3.hub.backbones import dinov3_vits16

    params = dict(dinov3_vits16(pretrained=False).named_parameters())
    assert set(conv) == {k for k in ref if k in params}, set(conv) ^ {k for k in ref if k in params}
    for k, v in conv.items():
        assert v.shape == ref[k].shape and torch.equal(v.to(ref[k].dtype), ref[k]), k
    a, b = dinov3_vits16(pretrained=False), dinov3_vits16(pretrained=False)
    a.load_state_dict(ref, strict=False)
    b.load_state_dict(conv, strict=False)
    dw.hf_buffers(b)  # rope periods and qkv bias mask, as the loader sets them for .safetensors
    assert torch.equal(b.rope_embed.periods, ref["rope_embed.periods"].float())
    x = torch.rand(1, 3, 128, 160)
    with torch.no_grad():
        fa = a.eval().get_intermediate_layers(x, n=(3, 7, 11), reshape=True, norm=True)
        fb = b.eval().get_intermediate_layers(x, n=(3, 7, 11), reshape=True, norm=True)
    assert all(torch.equal(p, q) for p, q in zip(fa, fb)), max((p - q).abs().max().item() for p, q in zip(fa, fb))
    print(f"ok test_hf_conversion_matches_official_vits ({len(conv)} tensors identical)")


def test_wrong_size_refused():
    if _skip(HF["b"]):
        return
    try:
        dw.load_state(HF["b"], "s")
        raise AssertionError("ViT-B weights accepted as ViT-S")
    except RuntimeError:
        pass
    print("ok test_wrong_size_refused")


def test_sizes_build_and_fuse():
    """DinoConv builds the requested size; its three feature maps have that width and DinoFuse accepts them."""
    import models  # noqa: F401
    from models.dino_fusion import DinoConv, DinoFuse

    for size in "bl":
        if _skip(HF[size]):
            continue
        os.environ["BLADEYOLO_DINO"] = size
        os.environ.pop("DINOV3_WEIGHTS", None)
        try:
            conv, fuse = DinoConv(3, 64, 3, 2).eval(), DinoFuse(1, 512).eval()
        finally:
            os.environ.pop("BLADEYOLO_DINO")
        _, dim, depth, blocks = dw.SIZES[size]
        assert len(conv.dino.blocks) == depth and conv.dino_blocks == blocks and conv.dino_dim == dim
        with torch.no_grad():
            y = conv(torch.rand(1, 3, 128, 128))
            assert [tuple(t.shape) for t in y.dino] == [(1, dim, 8, 8)] * 3
            assert fuse([y, torch.rand(1, 512, 8, 8)]).shape == (1, 512, 8, 8)
        del conv, fuse
        print(f"ok test_sizes_build_and_fuse ViT-{size.upper()}/16 ({depth} blocks, width {dim}, blocks {blocks})")


if __name__ == "__main__":
    test_hf_conversion_matches_official_vits()
    test_wrong_size_refused()
    test_sizes_build_and_fuse()
