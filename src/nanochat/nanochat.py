"""Unified diffusion/attention interface inspired by nanochat.flash_attention.

This module implements the core operators from
"The Diffusion–Attention Connection" (Candanedo, 2026):

    d^->_{ij} = ||q_i||^2 - <q_i, k_j>
    d^<-_{ij} = ||k_j||^2 - <q_i, k_j>
    D^2_{ij}  = d^->_{ij} + d^<-_{ij}

and the induced row-stochastic operators:

    A^+       = softmax_j(-beta d^->)
    P^+       = softmax_j(-beta D^2)
    P^+_PoE   = normalize_j(A^+_{->} * A^+_{<-})

It mirrors the spirit of karpathy/nanochat/flash_attention.py:
- one small public wrapper namespace (`flash_diffusion`)
- training path (`flash_diffusion_func`)
- KV-cache inference path (`flash_diffusion_with_kvcache`)
- pure PyTorch implementation with no custom CUDA required

Unlike FlashAttention, this is not a fused kernel. It is a torch-native
reference implementation that preserves the same high-level shape contract:
    q, k, v: (B, T, H, D)
    y:       (B, Tq, Hq, D)
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional, Tuple

import torch


# =============================================================================
# Backend selection (kept intentionally lightweight)
# =============================================================================
_OVERRIDE_IMPL: Optional[str] = None  # 'torch' or None(auto)
USE_TORCH = True


# =============================================================================
# Utility helpers
# =============================================================================

def _resolve_impl() -> str:
    if _OVERRIDE_IMPL in (None, "torch"):
        return "torch"
    raise ValueError(f"Unknown flash_diffusion backend override: {_OVERRIDE_IMPL}")


class DiffusionOutput:
    def __init__(
        self,
        y: torch.Tensor,
        operator: Optional[torch.Tensor] = None,
        d_fwd: Optional[torch.Tensor] = None,
        d_bwd: Optional[torch.Tensor] = None,
        d2: Optional[torch.Tensor] = None,
        scores: Optional[torch.Tensor] = None,
    ):
        self.y = y
        self.operator = operator
        self.d_fwd = d_fwd
        self.d_bwd = d_bwd
        self.d2 = d2
        self.scores = scores


def _expand_kv_heads(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand KV heads to query heads for GQA/MQA-style layouts.

    Input/output layout is (B, T, H, D).
    """
    hq = q.size(2)
    hk = k.size(2)
    hv = v.size(2)
    if hk != hv:
        raise ValueError(f"k and v must have same head count, got {hk} and {hv}")
    if hq == hk:
        return q, k, v
    if hq % hk != 0:
        raise ValueError(f"q heads ({hq}) must be divisible by k/v heads ({hk})")
    rep = hq // hk
    k = k.repeat_interleave(rep, dim=2)
    v = v.repeat_interleave(rep, dim=2)
    return q, k, v


def _causal_window_mask(
    Tq: int,
    Tk: int,
    device: torch.device,
    *,
    causal: bool,
    window_size: Tuple[int, int],
) -> torch.Tensor:
    """Return valid-attention mask of shape (Tq, Tk).

    The convention matches nanochat.flash_attention.py:
    - causal=True means queries can only see keys up to their aligned position
    - window_size=(left, right), with -1 meaning unbounded on that side
    - for cache/chunk inference (Tq != Tk), query i is aligned to absolute key index Tk-Tq+i
    """
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)

    valid = torch.ones((Tq, Tk), dtype=torch.bool, device=device)
    if causal:
        valid = valid & (col_idx <= row_idx)

    left, right = window_size
    if left >= 0:
        valid = valid & ((row_idx - col_idx) <= left)
    if right >= 0:
        valid = valid & ((col_idx - row_idx) <= right)
    return valid


def _masked_row_softmax(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Row softmax over last dim with boolean validity mask.

    x: (..., Tq, Tk)
    valid: (Tq, Tk)
    """
    x = x.masked_fill(~valid, float("-inf"))
    all_masked = (~valid).all(dim=-1)
    if all_masked.any():
        x = x.clone()
        x[..., all_masked, :] = 0.0
    p = torch.softmax(x, dim=-1)
    if all_masked.any():
        p = p.clone()
        p[..., all_masked, :] = 0.0
    return p


# =============================================================================
# Core geometry
# =============================================================================

def qk_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Pairwise query-key scores.

    q, k: (B, T, H, D)
    returns: (B, H, Tq, Tk)
    """
    return torch.einsum("bthd,bshd->bhts", q, k)


def qk_bidivergence(
    q: torch.Tensor, k: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute forward/backward QK bidivergences and squared distance.

    Definitions:
        d_fwd[i,j] = ||q_i||^2 - <q_i, k_j>
        d_bwd[i,j] = ||k_j||^2 - <q_i, k_j>
        d2[i,j]    = d_fwd[i,j] + d_bwd[i,j] = ||q_i-k_j||^2

    Shapes:
        q, k:  (B, T, H, D)
        out:   (B, H, Tq, Tk)
    """
    scores = qk_scores(q, k)
    q_norm2 = (q.square().sum(dim=-1).transpose(1, 2))[..., :, None]
    k_norm2 = (k.square().sum(dim=-1).transpose(1, 2))[..., None, :]
    d_fwd = q_norm2 - scores
    d_bwd = k_norm2 - scores
    d2 = d_fwd + d_bwd
    return d_fwd, d_bwd, d2, scores


def attention_matrix_from_bidivergence(
    d_fwd: torch.Tensor,
    *,
    beta: float = 1.0,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    """A^+ = softmax_j(-beta d_fwd)."""
    Tq, Tk = d_fwd.size(-2), d_fwd.size(-1)
    valid = _causal_window_mask(
        Tq, Tk, d_fwd.device, causal=causal, window_size=window_size
    )
    return _masked_row_softmax(-beta * d_fwd, valid)


def diffusion_matrix_from_distance(
    d2: torch.Tensor,
    *,
    beta: float = 1.0,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    """P^+ = softmax_j(-beta D^2)."""
    Tq, Tk = d2.size(-2), d2.size(-1)
    valid = _causal_window_mask(
        Tq, Tk, d2.device, causal=causal, window_size=window_size
    )
    return _masked_row_softmax(-beta * d2, valid)


def poe_diffusion_matrix(
    d_fwd: torch.Tensor,
    d_bwd: torch.Tensor,
    *,
    beta: float = 1.0,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    """Product-of-experts diffusion operator from Appendix A / Eq. (28).

    Uses same-axis softmax for both experts and renormalizes the Hadamard product.
    """
    Tq, Tk = d_fwd.size(-2), d_fwd.size(-1)
    valid = _causal_window_mask(
        Tq, Tk, d_fwd.device, causal=causal, window_size=window_size
    )
    a_fwd = _masked_row_softmax(-beta * d_fwd, valid)
    a_bwd_same_axis = _masked_row_softmax(-beta * d_bwd, valid)
    prod = a_fwd * a_bwd_same_axis
    denom = prod.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(prod.dtype).tiny)
    return prod / denom


def magnetic_diffusion_matrix(
    d2: torch.Tensor,
    phase: torch.Tensor,
    *,
    beta: float = 1.0,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    r"""Complex magnetic diffusion operator \tilde{P}^+ = P^+ \odot exp(i phase)."""
    p = diffusion_matrix_from_distance(
        d2, beta=beta, causal=causal, window_size=window_size
    )
    return p.to(torch.complex64) * torch.exp(1j * phase.to(torch.float32))


def apply_markov_operator(operator: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply row-stochastic operator to values.

    operator: (B, H, Tq, Tk)
    v:        (B, Tk, H, D)
    returns:  (B, Tq, H, D)
    """
    v_h = v.transpose(1, 2)  # (B, H, Tk, D)
    y_h = torch.einsum("bhts,bhsd->bhtd", operator, v_h)
    return y_h.transpose(1, 2).contiguous()


# =============================================================================
# Public API
# =============================================================================

def flash_diffusion_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    mode: str = "dmap",
    beta: float = 1.0,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    return_operator: bool = False,
    return_geometry: bool = False,
    phase: Optional[torch.Tensor] = None,
):
    """Unified diffusion/attention forward pass.

    Args:
        q, k, v:
            Tensors of shape (B, T, H, D), matching nanochat.flash_attention.py.
            GQA/MQA is supported when q has more heads than k/v and the ratio is integral.
        mode:
            "attention"  -> A^+      = softmax_j(-beta d_fwd) == softmax_j(beta <q,k>)
            "dmap"       -> P^+      = softmax_j(-beta D^2)
            "poe"        -> PoE(A_fwd, A_bwd_same_axis)
            "magnetic"   -> complex P^+ * exp(i phase)
        beta:
            Inverse temperature / kernel scale from the paper.
        causal, window_size:
            Same masking semantics as nanochat.flash_attention.py.
        return_operator:
            If True, return (y, operator).
        return_geometry:
            If True, return DiffusionOutput with geometry tensors attached.
        phase:
            Complex phase for magnetic diffusion. Required when mode="magnetic".

    Returns:
        y of shape (B, Tq, Hq, D), or richer tuples/objects when requested.
    """
    _ = _resolve_impl()  # reserved for future backends

    q, k, v = _expand_kv_heads(q, k, v)
    d_fwd, d_bwd, d2, scores = qk_bidivergence(q, k)

    if mode == "attention":
        operator = attention_matrix_from_bidivergence(
            d_fwd, beta=beta, causal=causal, window_size=window_size
        )
    elif mode == "dmap":
        operator = diffusion_matrix_from_distance(
            d2, beta=beta, causal=causal, window_size=window_size
        )
    elif mode == "poe":
        operator = poe_diffusion_matrix(
            d_fwd, d_bwd, beta=beta, causal=causal, window_size=window_size
        )
    elif mode == "magnetic":
        if phase is None:
            raise ValueError("phase is required for mode='magnetic'")
        operator = magnetic_diffusion_matrix(
            d2, phase, beta=beta, causal=causal, window_size=window_size
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    y = apply_markov_operator(operator, v)

    if return_geometry:
        return DiffusionOutput(
            y=y,
            operator=operator,
            d_fwd=d_fwd,
            d_bwd=d_bwd,
            d2=d2,
            scores=scores,
        )
    if return_operator:
        return y, operator
    return y


def flash_diffusion_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    mode: str = "dmap",
    beta: float = 1.0,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    return_operator: bool = False,
    phase: Optional[torch.Tensor] = None,
):
    """KV-cache path mirroring nanochat.flash_attention.flash_attn_with_kvcache.

    Args:
        q: Queries, shape (B, T_new, Hq, D)
        k_cache, v_cache: preallocated caches, shape (B, T_max, Hkv, D)
        k, v: new keys/values to append, shape (B, T_new, Hkv, D)
        cache_seqlens: current cache positions, shape (B,) int tensor. Assumes uniform position across batch.
    """
    if cache_seqlens is None:
        raise ValueError("cache_seqlens is required")
    if (k is None) != (v is None):
        raise ValueError("k and v must either both be provided or both be None")

    B, T_new, _, _ = q.shape
    pos = int(cache_seqlens[0].item())

    if k is not None and v is not None:
        k_cache[:, pos:pos + T_new, :, :] = k
        v_cache[:, pos:pos + T_new, :, :] = v

    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]

    return flash_diffusion_func(
        q,
        k_full,
        v_full,
        mode=mode,
        beta=beta,
        causal=causal,
        window_size=window_size,
        return_operator=return_operator,
        phase=phase,
    )


# =============================================================================
# Sinkhorn / Schrödinger bridge helpers
# =============================================================================

def sinkhorn_from_logits(
    logits: torch.Tensor, n_iter: int = 20, eps: float = 1e-12
) -> torch.Tensor:
    """Compute a bistochastic matrix from logits via Sinkhorn iterations.

    logits: (..., N, N)
    returns: (..., N, N), approximately doubly stochastic
    """
    z = torch.exp(logits)
    for _ in range(n_iter):
        z = z / z.sum(dim=-2, keepdim=True).clamp_min(eps)  # column normalize
        z = z / z.sum(dim=-1, keepdim=True).clamp_min(eps)  # row normalize
    return z


def schrodinger_bridge_forward(
    reference_kernel: torch.Tensor,
    mu_plus: torch.Tensor,
    mu_minus: torch.Tensor,
    n_iter: int = 50,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Discrete one-step Schrödinger bridge by alternating scaling.

    reference_kernel: (..., N, N), strictly positive reference matrix P
    mu_plus:          (..., N), source marginal
    mu_minus:         (..., N), sink marginal

    Returns:
        forward_operator: (..., N, N), row-stochastic Π^+
        coupling:         (..., N, N), Π = diag(u+) P diag(u-)
        u_plus:           (..., N)
        u_minus:          (..., N)
    """
    p = reference_kernel.clamp_min(eps)
    u_minus = torch.ones_like(mu_minus)
    u_plus = torch.ones_like(mu_plus)

    for _ in range(n_iter):
        u_plus = mu_plus / torch.einsum("...ij,...j->...i", p, u_minus).clamp_min(eps)
        u_minus = mu_minus / torch.einsum("...ji,...j->...i", p, u_plus).clamp_min(eps)

    coupling = u_plus[..., :, None] * p * u_minus[..., None, :]
    forward = coupling / mu_plus[..., :, None].clamp_min(eps)
    return forward, coupling, u_plus, u_minus


# =============================================================================
# Export namespace in the same spirit as nanochat.flash_attention
# =============================================================================

flash_diffusion = SimpleNamespace(
    flash_diffusion_func=flash_diffusion_func,
    flash_diffusion_with_kvcache=flash_diffusion_with_kvcache,
    qk_bidivergence=qk_bidivergence,
    qk_scores=qk_scores,
    attention_matrix_from_bidivergence=attention_matrix_from_bidivergence,
    diffusion_matrix_from_distance=diffusion_matrix_from_distance,
    poe_diffusion_matrix=poe_diffusion_matrix,
    magnetic_diffusion_matrix=magnetic_diffusion_matrix,
    sinkhorn_from_logits=sinkhorn_from_logits,
    schrodinger_bridge_forward=schrodinger_bridge_forward,
)
