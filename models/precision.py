"""Mixed-precision safety: keep blocks in fp32 (FP32Block) and compute YOLOv12 area-attention scores in fp32."""

import torch


class FP32Block:
    """Mixin: the block computes in fp32 and keeps fp32 weights, whatever the surrounding precision.

    Under AMP autocast is disabled inside the block; when the model is converted with .half() (Ultralytics does this
    for validation and checkpoints) the block's tensors are converted back to fp32. Inputs are cast to fp32 and the
    output back to the input dtype, so the rest of the network is unaffected.
    """

    def _apply(self, fn, *args, **kwargs):
        super()._apply(fn, *args, **kwargs)
        if any(p.dtype in (torch.float16, torch.bfloat16) for p in self.parameters()):
            super()._apply(lambda t: t.float() if t.is_floating_point() else t)
        return self

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            return super().forward(x.float()).to(x.dtype)


def _area_attention_forward(self, x):
    """ultralytics AAttn.forward (8.4.174) with the attention scores and softmax computed in fp32.

    The stock code forms the scores as an fp16 matrix product, (q * scale)^T @ k: with activations of a few hundred and
    32-dim heads a score reaches ~4e5 > 65504 (fp16 max), becomes inf, and the softmax returns NaN. This was the first
    non-finite layer (layer 6, A2C2f) in all five NaN batches of the bladeyolo2_wtbd_group_tolerant run (research log,
    Phase 18). Softmax probabilities are <= 1, so the product with v stays in the input precision.
    """
    B, _, H, W = x.shape
    N = H * W
    qkv = self.qkv(x).flatten(2).transpose(1, 2)
    if self.area > 1:
        qkv = qkv.reshape(B * self.area, N // self.area, self.all_head_dim * 3)
        B, N, _ = qkv.shape
    q, k, v = (
        qkv.view(B, N, self.num_heads, self.head_dim * 3)
        .permute(0, 2, 3, 1)
        .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
    )
    with torch.autocast(device_type=x.device.type, enabled=False):
        attn = ((q.float() * (self.head_dim**-0.5)).transpose(-2, -1) @ k.float()).softmax(dim=-1).to(v.dtype)
    x = v @ attn.transpose(-2, -1)
    x = x.permute(0, 3, 1, 2)
    v = v.permute(0, 3, 1, 2)
    if self.area > 1:
        x = x.reshape(B // self.area, N * self.area, self.all_head_dim)
        v = v.reshape(B // self.area, N * self.area, self.all_head_dim)
        B, N, _ = x.shape
    x = x.reshape(B, H, W, self.all_head_dim).permute(0, 3, 1, 2).contiguous()
    v = v.reshape(B, H, W, self.all_head_dim).permute(0, 3, 1, 2).contiguous()
    x = x + self.pe(v)
    return self.proj(x)


def install_stable_area_attention():
    """Replace AAttn.forward, only for the Ultralytics version it was copied from (requirements.txt pins it)."""
    import ultralytics
    from ultralytics.nn.modules.block import AAttn
    from ultralytics.utils import LOGGER

    if ultralytics.__version__ != "8.4.174":
        LOGGER.warning(f"[BladeYOLO] fp32 area-attention scores not installed: written for ultralytics 8.4.174, "
                       f"found {ultralytics.__version__}")
        return
    AAttn._stock_forward = getattr(AAttn, "_stock_forward", AAttn.forward)
    AAttn.forward = _area_attention_forward
