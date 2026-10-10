"""2D selective state-space (Mamba) blocks for the neck.

SS2D follows VMamba: the feature map is scanned in four directions (row-major,
column-major and both reversed), each direction runs an input-dependent
selective scan, and the four results are merged back onto the grid. The scan
itself uses the Mamba-2 state-space-duality (SSD) chunked form, which expresses
the recurrence as batched matmuls, so it runs on any GPU in plain PyTorch
without compiling the mamba_ssm CUDA kernels.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv


def ssd_scan(x, a, B, C, chunk=64):
    """Selective scan h_t = exp(a_t) h_{t-1} + B_t x_t,  y_t = C_t h_t  (Mamba-2 SSD, chunked).

    x: (b, L, h, p) inputs already multiplied by dt
    a: (b, L, h)    log-decay dt * A (<= 0)
    B, C: (b, L, n) input/output projections shared across heads
    returns y: (b, L, h, p)
    """
    b, L, h, p = x.shape
    n = B.shape[-1]
    pad = (-L) % chunk
    if pad:  # zero padding at the end is causal-safe: it never influences earlier outputs
        x = F.pad(x, (0, 0, 0, 0, 0, pad))
        a = F.pad(a, (0, 0, 0, pad))
        B = F.pad(B, (0, 0, 0, pad))
        C = F.pad(C, (0, 0, 0, pad))
    nc = x.shape[1] // chunk
    x = x.view(b, nc, chunk, h, p)
    B = B.view(b, nc, chunk, n)
    C = C.view(b, nc, chunk, n)
    a_cum = a.view(b, nc, chunk, h).permute(0, 3, 1, 2).cumsum(-1)  # (b, h, c, l)

    # 1. Outputs from inputs inside the same chunk (masked "attention" form).
    causal = torch.ones(chunk, chunk, dtype=torch.bool, device=x.device).tril()
    seg = (a_cum[..., :, None] - a_cum[..., None, :]).masked_fill(~causal, -torch.inf)
    w = torch.exp(seg) * torch.einsum("bcln,bcsn->bcls", C, B).unsqueeze(1)  # (b, h, c, l, s)
    y = torch.einsum("bhcls,bcshp->bclhp", w, x)

    # 2. State left at the end of every chunk.
    decay_to_end = torch.exp(a_cum[..., -1:] - a_cum)  # (b, h, c, l)
    states = torch.einsum("bcln,bhcl,bclhp->bchpn", B, decay_to_end, x)

    # 3. Propagate states across chunks: state entering chunk z.
    chunk_cum = F.pad(a_cum[..., -1], (1, 0)).cumsum(-1)  # (b, h, c + 1)
    causal_c = torch.ones(nc + 1, nc + 1, dtype=torch.bool, device=x.device).tril()
    decay_chunk = torch.exp((chunk_cum[..., :, None] - chunk_cum[..., None, :]).masked_fill(~causal_c, -torch.inf))
    states = torch.cat([torch.zeros_like(states[:, :1]), states], dim=1)
    states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)[:, :-1]

    # 4. Contribution of the carried-in state to every position of the chunk.
    y = y + torch.einsum("bcln,bchpn,bhcl->bclhp", C, states, torch.exp(a_cum))
    return y.reshape(b, nc * chunk, h, p)[:, :L]


def _scan_orders(t):
    """(B, H, W, ...) -> (B, 4, H*W, ...): row-major, column-major, and both reversed."""
    Bn, H, W = t.shape[:3]
    rows = t.reshape(Bn, H * W, *t.shape[3:])
    cols = t.transpose(1, 2).reshape(Bn, H * W, *t.shape[3:])
    return torch.stack([rows, cols, rows.flip(1), cols.flip(1)], dim=1)


def _merge_orders(y, H, W):
    """Inverse of _scan_orders, summing the four directions: (B, 4, H*W, ...) -> (B, H, W, ...)."""
    Bn, rest = y.shape[0], y.shape[3:]
    rows = y[:, 0] + y[:, 2].flip(1)
    cols = y[:, 1] + y[:, 3].flip(1)
    return rows.view(Bn, H, W, *rest) + cols.view(Bn, W, H, *rest).transpose(1, 2)


class SS2D(nn.Module):
    """Four-direction selective scan over a channels-last feature map (B, H, W, C)."""

    K = 4  # scan directions

    def __init__(self, d_model, d_state=16, expand=1, headdim=64, chunk=64, dt_min=1e-3, dt_max=1e-1):
        super().__init__()
        d_inner = int(expand * d_model)
        assert d_inner % headdim == 0, f"d_inner={d_inner} must be divisible by headdim={headdim}"
        self.d_state, self.headdim, self.chunk = d_state, headdim, chunk
        self.nheads = d_inner // headdim

        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)  # x and gate z
        self.dwconv = nn.Conv2d(d_inner, d_inner, 3, padding=1, groups=d_inner)
        # Per-direction dt (one per head), B and C.
        self.x_proj = nn.Linear(d_inner, self.K * (self.nheads + 2 * d_state), bias=False)

        dt = torch.exp(torch.rand(self.K, self.nheads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus
        self.A_log = nn.Parameter(torch.log(torch.empty(self.K, self.nheads).uniform_(1, 16)))
        self.D = nn.Parameter(torch.ones(self.K, self.nheads))

        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x):
        Bn, H, W, _ = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = F.silu(self.dwconv(x.permute(0, 3, 1, 2))).permute(0, 2, 3, 1)  # (B, H, W, d_inner)

        proj = self.x_proj(x).view(Bn, H, W, self.K, -1)
        idx = torch.arange(self.K, device=x.device)
        proj = _scan_orders(proj)[:, idx, :, idx].transpose(0, 1)  # direction k uses its own projection
        dt, Bs, Cs = proj.split([self.nheads, self.d_state, self.d_state], dim=-1)

        # The recurrence runs in fp32: exp/cumsum of decays is not safe in fp16.
        with torch.autocast(device_type=x.device.type, enabled=False):
            dt = F.softplus(dt.float() + self.dt_bias[:, None])  # (B, K, L, h)
            a = dt * -torch.exp(self.A_log.float())[:, None]
            xs = _scan_orders(x.float()).view(Bn, self.K, H * W, self.nheads, self.headdim)
            y = ssd_scan(
                (xs * dt[..., None]).flatten(0, 1),
                a.flatten(0, 1),
                Bs.float().flatten(0, 1),
                Cs.float().flatten(0, 1),
                self.chunk,
            ).view_as(xs)
            y = y + xs * self.D[:, None, :, None].float()
            y = _merge_orders(y, H, W).flatten(-2)  # (B, H, W, d_inner)
            # Normalise before leaving fp32: the scan sums over the whole sequence and can exceed the fp16
            # range (65504); casting first turned it into inf -> NaN under AMP.
            n = self.out_norm
            y = F.layer_norm(y, n.normalized_shape, n.weight.float(), n.bias.float(), n.eps)

        y = y.to(z.dtype) * F.silu(z)
        return self.out_proj(y)


class VSSBlock(nn.Module):
    """Pre-norm residual SS2D followed by a depthwise-conv feed-forward."""

    def __init__(self, c, d_state=16, mlp_ratio=2.0):
        super().__init__()
        self.norm = nn.LayerNorm(c)
        self.ss2d = SS2D(c, d_state=d_state)
        h = int(c * mlp_ratio)
        self.ffn = nn.Sequential(Conv(c, h, 1), Conv(h, h, 3, g=h), Conv(h, c, 1, act=False))

    def forward(self, x):
        x = x + self.ss2d(self.norm(x.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)
        return x + self.ffn(x)


class C2fSS2D(nn.Module):
    """C2f-style CSP block whose inner blocks are VSSBlocks (global context via selective scan)."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(VSSBlock(self.c) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
