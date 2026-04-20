"""SDPA-backed Markov-operator attention wrapper for nanochat experiments.

This file is meant to replace / experiment alongside nanochat.flash_attention.

It exports three namespaces from the same module:

    from nanochat.flash_attention import flash_attn
    from nanochat.flash_attention import flash_diff
    from nanochat.flash_attention import flash_ntta

with row-stochastic operators

    A^+       : forward attention      = softmax_j(-beta d_fwd)
    P^+       : diffusion operator     = softmax_j(-beta d2)
    A^{<-+}   : backward attention     = softmax_j(-beta d_bwd)

where
    d_fwd[i,j] = ||q_i||^2 - <q_i, k_j>
    d_bwd[i,j] = ||k_j||^2 - <q_i, k_j>
    d2[i,j]    = d_fwd[i,j] + d_bwd[i,j] = ||q_i - k_j||^2

Because softmax is shift-invariant along each row, these can be implemented as:

    A^+       = softmax_j( beta      * <q_i, k_j> )
    A^{<-+}   = softmax_j( beta      * <q_i, k_j> - beta ||k_j||^2 )
    P^+       = softmax_j( 2 * beta  * <q_i, k_j> - beta ||k_j||^2 )

optionally with an extra key-side Doob potential psi_j added to the logits.

This version is built on torch.nn.functional.scaled_dot_product_attention, so on
supported CUDA backends it can dispatch to fused Flash / efficient attention kernels.
It is still safe on CPU / unsupported backends via PyTorch fallback.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
except Exception:  # pragma: no cover
    sdpa_kernel = None
    SDPBackend = None


# -----------------------------------------------------------------------------
# Backend helpers
# -----------------------------------------------------------------------------
_OVERRIDE_IMPL: Optional[str] = None  # 'auto' | 'flash' | 'efficient' | 'math'
USE_TORCH = True


def _resolve_impl() -> str:
    if _OVERRIDE_IMPL is None:
        return "auto"
    if _OVERRIDE_IMPL in {"auto", "flash", "efficient", "math"}:
        return _OVERRIDE_IMPL
    raise ValueError(f"Unknown backend override: {_OVERRIDE_IMPL}")


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def _sdpa_context():
    """Best-effort backend selection for SDPA."""
    impl = _resolve_impl()
    if sdpa_kernel is None or SDPBackend is None or impl == "auto":
        return _NullContext()

    if impl == "flash":
        backends = [SDPBackend.FLASH_ATTENTION]
    elif impl == "efficient":
        backends = [SDPBackend.EFFICIENT_ATTENTION]
    elif impl == "math":
        backends = [SDPBackend.MATH]
    else:
        return _NullContext()

    return sdpa_kernel(backends=backends)


# -----------------------------------------------------------------------------
# Shape / mask utilities
# -----------------------------------------------------------------------------
def _to_bhtd(x: torch.Tensor) -> torch.Tensor:
    """Convert (B, T, H, D) -> (B, H, T, D)."""
    if x.ndim != 4:
        raise ValueError(f"expected 4D tensor (B,T,H,D), got shape {tuple(x.shape)}")
    return x.transpose(1, 2).contiguous()


def _from_bhtd(x: torch.Tensor) -> torch.Tensor:
    """Convert (B, H, T, D) -> (B, T, H, D)."""
    if x.ndim != 4:
        raise ValueError(f"expected 4D tensor (B,H,T,D), got shape {tuple(x.shape)}")
    return x.transpose(1, 2).contiguous()


def _require_same_heads(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    hq = q.size(2)
    hk = k.size(2)
    hv = v.size(2)
    if not (hq == hk == hv):
        raise ValueError(
            "This experimental module assumes n_kv_head == n_head, so q, k, v "
            f"must have the same head count; got {hq}, {hk}, {hv}."
        )


def _normalize_doob_potential(
    doob_potential: Optional[torch.Tensor],
    *,
    B: int,
    H: int,
    Tk: int,
) -> Optional[torch.Tensor]:
    """Accept (B,H,Tk) or (B,Tk,H), return (B,H,1,Tk)."""
    if doob_potential is None:
        return None
    if doob_potential.ndim != 3:
        raise ValueError("doob_potential must have shape (B,H,Tk) or (B,Tk,H)")
    if doob_potential.shape == (B, H, Tk):
        psi = doob_potential
    elif doob_potential.shape == (B, Tk, H):
        psi = doob_potential.transpose(1, 2).contiguous()
    else:
        raise ValueError(
            f"Bad doob_potential shape {tuple(doob_potential.shape)}; expected "
            f"(B,H,Tk)=({B},{H},{Tk}) or (B,Tk,H)=({B},{Tk},{H})"
        )
    return psi[:, :, None, :]


def _causal_window_valid_mask(
    Tq: int,
    Tk: int,
    device: torch.device,
    *,
    causal: bool,
    window_size: Tuple[int, int],
) -> torch.Tensor:
    """Boolean validity mask of shape (Tq, Tk)."""
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


def _build_additive_bias(
    *,
    q_bhtd: torch.Tensor,
    k_bhtd: torch.Tensor,
    mode: str,
    beta: float,
    causal: bool,
    window_size: Tuple[int, int],
    doob_potential: Optional[torch.Tensor],
) -> tuple[Optional[torch.Tensor], float]:
    """
    Return (attn_mask, scale) for F.scaled_dot_product_attention.

    We implement:
        attn  : logits = beta      * qk
        diff  : logits = 2 * beta  * qk  - beta ||k||^2 + psi_j
        ntta  : logits = beta      * qk  - beta ||k||^2 + psi_j

    plus causal/window masking via additive -inf bias.
    """
    B, H, Tq, D = q_bhtd.shape
    _, _, Tk, _ = k_bhtd.shape

    if mode == "attn":
        scale = beta
        key_bias = None
    elif mode == "diff":
        scale = 2.0 * beta
        key_bias = -beta * k_bhtd.square().sum(dim=-1, keepdim=False)  # (B,H,Tk)
    elif mode == "ntta":
        scale = beta
        key_bias = -beta * k_bhtd.square().sum(dim=-1, keepdim=False)  # (B,H,Tk)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    psi = _normalize_doob_potential(doob_potential, B=B, H=H, Tk=Tk)

    attn_bias = None
    if key_bias is not None:
        attn_bias = key_bias[:, :, None, :]  # (B,H,1,Tk)

    if psi is not None:
        attn_bias = psi if attn_bias is None else (attn_bias + psi)

    valid = _causal_window_valid_mask(
        Tq, Tk, q_bhtd.device, causal=causal, window_size=window_size
    )
    if not bool(valid.all()):
        mask_bias = torch.zeros((1, 1, Tq, Tk), dtype=q_bhtd.dtype, device=q_bhtd.device)
        mask_bias = mask_bias.masked_fill(~valid.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn_bias = mask_bias if attn_bias is None else (attn_bias + mask_bias)

    return attn_bias, scale


# -----------------------------------------------------------------------------
# Optional geometry helpers
# -----------------------------------------------------------------------------
def qk_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Return <q_i, k_j> with shape (B,H,Tq,Tk) from inputs (B,T,H,D)."""
    q_bhtd = _to_bhtd(q)
    k_bhtd = _to_bhtd(k)
    return torch.einsum("bhtd,bhsd->bhts", q_bhtd, k_bhtd)


def qk_bidivergence(
    q: torch.Tensor, k: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (d_fwd, d_bwd, d2, scores) with shape (B,H,Tq,Tk)."""
    q_bhtd = _to_bhtd(q)
    k_bhtd = _to_bhtd(k)
    scores = torch.einsum("bhtd,bhsd->bhts", q_bhtd, k_bhtd)
    q_norm2 = q_bhtd.square().sum(dim=-1, keepdim=False)[..., :, None]
    k_norm2 = k_bhtd.square().sum(dim=-1, keepdim=False)[..., None, :]
    d_fwd = q_norm2 - scores
    d_bwd = k_norm2 - scores
    d2 = d_fwd + d_bwd
    return d_fwd, d_bwd, d2, scores


def _materialize_operator_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Softmax over last dim, for inspection only."""
    return torch.softmax(logits, dim=-1)


# -----------------------------------------------------------------------------
# Core forward path
# -----------------------------------------------------------------------------
def _markov_sdpa_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    mode: str,
    beta: float = 1.0,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    doob_potential: Optional[torch.Tensor] = None,
    return_operator: bool = False,
    return_geometry: bool = False,
):
    """
    Shared SDPA-backed forward path.

    q, k, v use nanochat-style layout (B, T, H, D).
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must all have shape (B,T,H,D)")
    _require_same_heads(q, k, v)

    q_bhtd = _to_bhtd(q)
    k_bhtd = _to_bhtd(k)
    v_bhtd = _to_bhtd(v)

    attn_bias, scale = _build_additive_bias(
        q_bhtd=q_bhtd,
        k_bhtd=k_bhtd,
        mode=mode,
        beta=beta,
        causal=causal,
        window_size=window_size,
        doob_potential=doob_potential,
    )

    with _sdpa_context():
        y_bhtd = F.scaled_dot_product_attention(
            q_bhtd,
            k_bhtd,
            v_bhtd,
            attn_mask=attn_bias,
            dropout_p=0.0,
            is_causal=False,   # causal/window is already baked into attn_bias
            scale=scale,
        )

    y = _from_bhtd(y_bhtd)

    if not (return_operator or return_geometry):
        return y

    d_fwd, d_bwd, d2, scores = qk_bidivergence(q, k)

    B, H, Tq, Tk = scores.shape
    psi = _normalize_doob_potential(doob_potential, B=B, H=H, Tk=Tk)

    if mode == "attn":
        logits = beta * scores
    elif mode == "diff":
        k_norm2 = _to_bhtd(k).square().sum(dim=-1, keepdim=False)[..., None, :]
        logits = 2.0 * beta * scores - beta * k_norm2
    elif mode == "ntta":
        k_norm2 = _to_bhtd(k).square().sum(dim=-1, keepdim=False)[..., None, :]
        logits = beta * scores - beta * k_norm2
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if psi is not None:
        logits = logits + psi

    valid = _causal_window_valid_mask(
        Tq, Tk, scores.device, causal=causal, window_size=window_size
    )
    if not bool(valid.all()):
        logits = logits.masked_fill(~valid.unsqueeze(0).unsqueeze(0), float("-inf"))

    operator = _materialize_operator_from_logits(logits)

    if return_geometry:
        return {
            "y": y,
            "operator": operator,
            "scores": scores,
            "d_fwd": d_fwd,
            "d_bwd": d_bwd,
            "d2": d2,
            "logits": logits,
        }

    return y, operator


def _markov_sdpa_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    mode: str,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    beta: float = 1.0,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    doob_potential: Optional[torch.Tensor] = None,
    return_operator: bool = False,
    return_geometry: bool = False,
):
    """KV-cache path mirroring nanochat.flash_attention.flash_attn_with_kvcache."""
    if cache_seqlens is None:
        raise ValueError("cache_seqlens is required")
    if (k is None) != (v is None):
        raise ValueError("k and v must either both be provided or both be None")

    pos = int(cache_seqlens[0].item())
    T_new = q.size(1)

    if k is not None and v is not None:
        k_cache[:, pos:pos + T_new, :, :] = k
        v_cache[:, pos:pos + T_new, :, :] = v

    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]

    return _markov_sdpa_func(
        q,
        k_full,
        v_full,
        mode=mode,
        beta=beta,
        causal=causal,
        window_size=window_size,
        doob_potential=doob_potential,
        return_operator=return_operator,
        return_geometry=return_geometry,
    )


# -----------------------------------------------------------------------------
# Public wrappers
# -----------------------------------------------------------------------------
def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    beta: float = 1.0,
    doob_potential: Optional[torch.Tensor] = None,
    return_operator: bool = False,
    return_geometry: bool = False,
):
    return _markov_sdpa_func(
        q, k, v,
        mode="attn",
        beta=beta,
        causal=causal,
        window_size=window_size,
        doob_potential=doob_potential,
        return_operator=return_operator,
        return_geometry=return_geometry,
    )


def flash_attn_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    beta: float = 1.0,
    doob_potential: Optional[torch.Tensor] = None,
    return_operator: bool = False,
    return_geometry: bool = False,
):
    return _markov_sdpa_with_kvcache(
        q,
        k_cache,
        v_cache,
        mode="attn",
        k=k,
        v=v,
        cache_seqlens=cache_seqlens,
        beta=beta,
        causal=causal,
        window_size=window_size,
        doob_potential=doob_potential,
        return_operator=return_operator,
        return_geometry=return_geometry,
    )


def flash_diff_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    beta: float = 1.0,
    doob_potential: Optional[torch.Tensor] = None,
    return_operator: bool = False,
    return_geometry: bool = False,
):
    return _markov_sdpa_func(
        q, k, v,
        mode="diff",
        beta=beta,
        causal=causal,
        window_size=window_size,
        doob_potential=doob_potential,
        return_operator=return_operator,
        return_geometry=return_geometry,
    )


def flash_diff_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    beta: float = 1.0,
    doob_potential: Optional[torch.Tensor] = None,
    return_operator: bool = False,
    return_geometry: bool = False,
):
    return _markov_sdpa_with_kvcache(
        q,
        k_cache,
        v_cache,
        mode="diff",
        k=k,
        v=v,
        cache_seqlens=cache_seqlens,
        beta=beta,
        causal=causal,
        window_size=window_size,
        doob_potential=doob_potential,
        return_operator=return_operator,
        return_geometry=return_geometry,
    )


def flash_ntta_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    beta: float = 1.0,
    doob_potential: Optional[torch.Tensor] = None,
    return_operator: bool = False,
    return_geometry: bool = False,
):
    return _markov_sdpa_func(
        q, k, v,
        mode="ntta",
        beta=beta,
        causal=causal,
        window_size=window_size,
        doob_potential=doob_potential,
        return_operator=return_operator,
        return_geometry=return_geometry,
    )


def flash_ntta_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    beta: float = 1.0,
    doob_potential: Optional[torch.Tensor] = None,
    return_operator: bool = False,
    return_geometry: bool = False,
):
    return _markov_sdpa_with_kvcache(
        q,
        k_cache,
        v_cache,
        mode="ntta",
        k=k,
        v=v,
        cache_seqlens=cache_seqlens,
        beta=beta,
        causal=causal,
        window_size=window_size,
        doob_potential=doob_potential,
        return_operator=return_operator,
        return_geometry=return_geometry,
    )


# -----------------------------------------------------------------------------
# Export namespaces in the same spirit as nanochat.flash_attention
# -----------------------------------------------------------------------------
flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)

flash_diff = SimpleNamespace(
    flash_attn_func=flash_diff_func,
    flash_attn_with_kvcache=flash_diff_with_kvcache,
)

flash_ntta = SimpleNamespace(
    flash_attn_func=flash_ntta_func,
    flash_attn_with_kvcache=flash_ntta_with_kvcache,
)
