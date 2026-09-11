"""
CrossSS2D and LSBlock implementation for BladeYOLO.

Based on:
  - "BladeYOLO: Wind Turbine Blade Defect Detection with Limited Annotations
     and Weak-Saliency Awareness" (IEEE TGRS, 2026) — Section III-E, Fig. 3
  - VMamba: Visual State Space Model (Zhu et al., 2024)
  - Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao, 2023)

Architecture (Cross-SS2D):
  The target feature Fi determines the SSM state-transition dynamics (A, B, C, Δ),
  while the guidance feature Gi+1 modulates the output via directional gating.
  Both features are expanded into 4 directional scanning sequences (cross-scan)
  before processing through the S6 selective state space.
"""

import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Try to import the optimized CUDA selective scan from mamba_ssm.
# Falls back to a pure-PyTorch implementation if unavailable.
# ============================================================
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    HAS_CUDA_SELECTIVE_SCAN = True
except ImportError:
    HAS_CUDA_SELECTIVE_SCAN = False
    warnings.warn(
        "mamba_ssm not found. CrossSS2D will use a pure-PyTorch selective scan "
        "(functional but significantly slower). Install mamba_ssm for CUDA acceleration."
    )

__all__ = ['CrossSS2D', 'LSBlock']


# ============================================================
# Cross-Scan / Cross-Merge  (2D ↔ 4 directional 1D)
# Standard VMamba scanning pattern: row-major, col-major,
# reverse row-major, reverse col-major.
# ============================================================

def cross_scan_2d(x):
    """Expand a 2D feature map into 4 directional 1D sequences.

    Args:
        x: (B, C, H, W)

    Returns:
        (B, 4, C, H*W) — four directional sequences
    """
    B, C, H, W = x.shape
    x_hw = x.view(B, C, -1)                                          # row-major
    x_wh = x.transpose(2, 3).contiguous().view(B, C, -1)             # col-major
    return torch.stack([x_hw, x_wh, x_hw.flip(-1), x_wh.flip(-1)], dim=1)


def cross_merge_2d(ys, H, W):
    """Merge 4 directional 1D sequences back into a single 2D feature.

    Args:
        ys: (B, 4, C, L) where L = H*W
        H, W: original spatial dimensions

    Returns:
        (B, C, L) — merged feature (sum of four inverse-scanned directions)
    """
    B, _, C, L = ys.shape
    y0 = ys[:, 0]
    y1 = ys[:, 1].view(B, C, W, H).transpose(2, 3).contiguous().view(B, C, -1)
    y2 = ys[:, 2].flip(-1)
    y3 = ys[:, 3].flip(-1).view(B, C, W, H).transpose(2, 3).contiguous().view(B, C, -1)
    return y0 + y1 + y2 + y3


# ============================================================
# Pure-PyTorch Selective Scan  (reference / fallback)
# ============================================================

def selective_scan_pytorch(u, delta, A, B, C, D=None, z=None,
                           delta_bias=None, delta_softplus=False):
    """Reference selective scan in pure PyTorch.

    Args:
        u:     (B, D, L)   input sequence
        delta: (B, D, L)   timestep / dt
        A:     (D, N)      state matrix (negative)
        B:     (B, G, N, L) input matrix  (G groups, G | D)
        C:     (B, G, N, L) output matrix
        D:     (D,)         skip-connection weight (optional)
        z:     (B, D, L)   gating signal (optional)
        delta_bias:    (D,) added to delta before softplus (optional)
        delta_softplus: bool
    """
    B_batch, D_total, L = u.shape
    N = A.shape[1]
    G = B.shape[1]
    D_per_g = D_total // G

    if delta_bias is not None:
        delta = delta + delta_bias.unsqueeze(0).unsqueeze(-1)
    if delta_softplus:
        delta = F.softplus(delta)

    # Expand grouped B / C → full D channels
    if G < D_total:
        B_exp = B.unsqueeze(1).expand(-1, D_total, -1, -1) \
                 .reshape(B_batch, D_total, N, L)
        C_exp = C.unsqueeze(1).expand(-1, D_total, -1, -1) \
                 .reshape(B_batch, D_total, N, L)
    else:
        B_exp, C_exp = B.unsqueeze(1).expand(-1, D_total, -1, -1), C.unsqueeze(1).expand(-1, D_total, -1, -1)

    # Sequential scan
    h = torch.zeros(B_batch, D_total, N, device=u.device, dtype=u.dtype)
    ys = []
    for t in range(L):
        dt = delta[:, :, t].unsqueeze(-1)           # (B, D, 1)
        dA = torch.exp(dt * A.unsqueeze(0))          # (B, D, N)
        dBu = dt * B_exp[:, :, :, t] * u[:, :, t].unsqueeze(-1)
        h = dA * h + dBu
        y_t = (h * C_exp[:, :, :, t]).sum(-1)        # (B, D)
        ys.append(y_t)

    y = torch.stack(ys, dim=-1)                       # (B, D, L)

    if D is not None:
        y = y + u * D.unsqueeze(0).unsqueeze(-1)
    if z is not None:
        y = y * F.silu(z)
    return y


def _selective_scan(u, delta, A, B, C, D=None, z=None,
                    delta_bias=None, delta_softplus=False):
    """Dispatch to CUDA or PyTorch selective scan."""
    if HAS_CUDA_SELECTIVE_SCAN:
        return selective_scan_fn(
            u, delta, A, B, C,
            D=D, z=z, delta_bias=delta_bias,
            delta_softplus=delta_softplus,
            return_last_state=False,
        )
    return selective_scan_pytorch(
        u, delta, A, B, C,
        D=D, z=z, delta_bias=delta_bias,
        delta_softplus=delta_softplus,
    )


# ============================================================
# LSBlock  (Local Spatial Block)
# ============================================================

class LSBlock(nn.Module):
    """Lightweight local feature enhancement via depthwise-separable conv.

    Used alongside the SSM branch in CrossVSSBlock to capture
    short-range spatial patterns that the long-range selective scan
    may under-represent.
    """

    def __init__(self, in_features, hidden_features=None,
                 act_layer=nn.SiLU, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.dw = nn.Conv2d(in_features, in_features, 3,
                            padding=1, groups=in_features, bias=True)
        self.pw1 = nn.Conv2d(in_features, hidden_features, 1, bias=True)
        self.act = act_layer()
        self.pw2 = nn.Conv2d(hidden_features, in_features, 1, bias=True)
        self.drop = nn.Dropout(drop) if drop > 0.0 else nn.Identity()

    def forward(self, x):
        """x: (B, C, H, W) → (B, C, H, W)"""
        return x + self.drop(self.pw2(self.act(self.pw1(self.dw(x)))))


# ============================================================
# CrossSS2D  (Cross-Scale Selective Scan 2D)
# ============================================================

class CrossSS2D(nn.Module):
    """Cross-Scale Selective State Space 2D block.

    Implements Fig. 3 of BladeYOLO (IEEE TGRS 2026).

    *Target* feature  → SSM dynamics  (A, B, C, Δ)
    *Guidance* feature → output gating (per scan direction)

    Both are cross-scanned into 4 directional sequences before
    the S6 selective state space processes them.

    Args:
        guide_dim:      channels of guidance feature
        d_model:        channels of target feature (= output channels)
        d_state:        SSM state dimension N
        ssm_ratio:      inner-dimension expansion factor
        ssm_rank_ratio: (unused, kept for API compat)
        dt_rank:        rank of Δ projection ("auto" → ceil(d_model/16))
        act_layer:      activation function class
        d_conv:         depthwise-conv kernel size
        conv_bias:      depthwise-conv bias
        dropout:        output dropout rate
        forward_type:   (unused, kept for API compat)
    """

    def __init__(
        self,
        guide_dim: int,
        d_model: int,
        d_state: int = 16,
        ssm_ratio: float = 2.0,
        ssm_rank_ratio: float = 2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        d_conv: int = 3,
        conv_bias: bool = True,
        dropout: float = 0.0,
        forward_type: str = "v2",
        **kwargs,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        d_inner = int(d_model * ssm_ratio)
        self.d_inner = d_inner
        dt_rank_val = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)
        self.dt_rank = dt_rank_val
        self.K = 4  # number of cross-scan directions

        # ── input projections ──
        self.in_proj_x = nn.Linear(d_model, d_inner, bias=False)
        self.in_proj_guide = nn.Linear(guide_dim, d_inner, bias=False)

        # ── depthwise conv on target ──
        self.conv2d = nn.Conv2d(
            d_inner, d_inner, groups=d_inner,
            kernel_size=d_conv, padding=(d_conv - 1) // 2, bias=conv_bias,
        )
        self.act = act_layer()

        # ── per-direction SSM parameter projections ──
        # x_proj: d_inner → (dt_rank + 2·N)  ×K directions
        self.x_proj_weight = nn.Parameter(
            self._init_x_proj(self.K, dt_rank_val + d_state * 2, d_inner)
        )

        # dt projection: dt_rank → d_inner  ×K directions
        self.dt_projs_weight = nn.Parameter(
            self._init_dt_weight(self.K, dt_rank_val, d_inner)
        )
        self.dt_projs_bias = nn.Parameter(
            self._init_dt_bias(self.K, d_inner)
        )

        # ── state matrix A  (log-space, learnable) ──
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).expand(self.K * d_inner, -1).clone()
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # ── skip-connection D ──
        self.D = nn.Parameter(torch.ones(self.K * d_inner))
        self.D._no_weight_decay = True

        # ── output projection ──
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    # ── weight initialisers (static) ──────────────────────────

    @staticmethod
    def _init_x_proj(K, out_f, in_f):
        w = torch.empty(K, out_f, in_f)
        for k in range(K):
            nn.init.kaiming_uniform_(w[k], a=math.sqrt(5))
        return w

    @staticmethod
    def _init_dt_weight(K, dt_rank, d_inner, dt_scale=1.0):
        w = torch.empty(K, d_inner, dt_rank)
        std = dt_rank ** -0.5 * dt_scale
        for k in range(K):
            nn.init.uniform_(w[k], -std, std)
        return w

    @staticmethod
    def _init_dt_bias(K, d_inner,
                      dt_min=0.001, dt_max=0.1, dt_floor=1e-4):
        bias = torch.empty(K, d_inner)
        for k in range(K):
            dt = torch.exp(
                torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_floor)
            bias[k] = dt + torch.log(-torch.expm1(-dt))   # inv-softplus
        return bias

    # ── forward ───────────────────────────────────────────────

    def forward(self, x: torch.Tensor, guide_x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:       target feature  (B, C, H, W) — drives SSM dynamics
            guide_x: guidance feature (B, C, H, W) — modulates output

        Returns:
            (B, C, H, W) — semantically calibrated feature
        """
        B, C, H, W = x.shape
        L = H * W
        K, D, N = self.K, self.d_inner, self.d_state

        # 1. Project to inner dimension  (channel-last for nn.Linear)
        x_proj = self.in_proj_x(x.permute(0, 2, 3, 1))           # (B,H,W,D)
        z_guide = self.in_proj_guide(guide_x.permute(0, 2, 3, 1))  # (B,H,W,D)

        # 2. Depthwise conv + activation on target
        x_conv = self.act(self.conv2d(x_proj.permute(0, 3, 1, 2)))  # (B,D,H,W)

        # 3. Cross-scan both target and guide into 4 directions
        xs = cross_scan_2d(x_conv)                                   # (B,K,D,L)
        zs = cross_scan_2d(z_guide.permute(0, 3, 1, 2))             # (B,K,D,L)

        # 4. SSM parameters from target  (per-direction via einsum)
        #    x_proj_weight: (K, dt_rank+2N, D)
        x_dbl = torch.einsum("bkdl,krd->bkrl", xs, self.x_proj_weight)
        dts, Bs, Cs = x_dbl.split([self.dt_rank, N, N], dim=2)
        #    dts: (B,K,dt_rank,L)   Bs,Cs: (B,K,N,L)

        # 5. dt projection   dt_rank → D   (per direction)
        dts = torch.einsum("bkrl,kdr->bkdl", dts, self.dt_projs_weight)
        #    dts: (B,K,D,L)

        # 6 & 7. Apply selective scan per direction to strictly comply with standard mamba_ssm shapes
        A = -torch.exp(self.A_log.float()).view(K, D, N)
        D_skip = self.D.float().view(K, D)
        dt_bias = self.dt_projs_bias.float()  # (K, D)
        
        ys = []
        for k in range(K):
            y_k = _selective_scan(
                xs[:, k].contiguous(),           # u: (B, D, L)
                dts[:, k].contiguous(),          # delta: (B, D, L)
                A[k],                            # A: (D, N)
                Bs[:, k].contiguous(),           # B: (B, N, L)
                Cs[:, k].contiguous(),           # C: (B, N, L)
                D=D_skip[k],                     # D: (D,)
                z=zs[:, k].contiguous(),         # z: (B, D, L)
                delta_bias=dt_bias[k],           # dt_bias: (D,)
                delta_softplus=True
            )
            ys.append(y_k)
            
        ys = torch.stack(ys, dim=1)  # (B, K, D, L)

        # 8. Cross-merge → 2D
        y = cross_merge_2d(ys.reshape(B, K, D, L), H, W)  # (B, D, L)

        # 9. Output projection  (channel-last)
        y = y.view(B, D, H, W).permute(0, 2, 3, 1)        # (B,H,W,D)
        out = self.dropout(self.out_proj(y))                 # (B,H,W,C)
        return out.permute(0, 3, 1, 2)                       # (B,C,H,W)

