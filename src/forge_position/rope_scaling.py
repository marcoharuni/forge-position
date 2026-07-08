"""Educational RoPE scaling utilities.

YaRN and LongRoPE-style functions here are clear approximations for learning
and experimentation. They are not drop-in reproductions of paper or vendor
implementations.
"""

from __future__ import annotations

import torch

from .rope import build_rope_cache


def linear_position_scaling(
    position_ids: torch.Tensor, scaling_factor: float
) -> torch.Tensor:
    """Position Interpolation: linearly down-scale positions by a factor."""

    if scaling_factor <= 0:
        raise ValueError("scaling_factor must be positive")
    return position_ids.to(torch.float32) / scaling_factor


def dynamic_ntk_base(
    base: float,
    seq_len: int,
    max_position_embeddings: int,
    scaling_factor: float,
    dim: int,
) -> float:
    """Compute the common dynamic NTK-aware RoPE base approximation."""

    if seq_len <= max_position_embeddings:
        return base
    if dim <= 2:
        raise ValueError("dim must be greater than 2 for dynamic NTK scaling")
    if scaling_factor <= 1:
        scaling_factor = 1.000001
    ratio = (scaling_factor * seq_len / max_position_embeddings) - (scaling_factor - 1)
    return base * (ratio ** (dim / (dim - 2)))


def ntk_scaled_rope_cache(
    seq_len: int,
    dim: int,
    base: float = 10000.0,
    max_position_embeddings: int = 2048,
    scaling_factor: float = 2.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a RoPE cache using a dynamic NTK-aware base approximation."""

    scaled_base = dynamic_ntk_base(base, seq_len, max_position_embeddings, scaling_factor, dim)
    return build_rope_cache(seq_len, dim, scaled_base, device=device, dtype=dtype)


def yarn_scaled_rope_cache(
    seq_len: int,
    dim: int,
    base: float = 10000.0,
    max_position_embeddings: int = 2048,
    scaling_factor: float = 2.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build an educational YaRN-like scaled RoPE cache.

    This blends unscaled low-frequency channels with linearly interpolated
    higher-frequency channels. It is intentionally documented as an educational
    approximation, not a reproduction of YaRN's full implementation.
    """

    if seq_len <= 0 or dim <= 0 or dim % 2 != 0:
        raise ValueError("seq_len must be positive and dim must be positive/even")
    work_dtype = dtype if dtype is not None and dtype.is_floating_point else torch.float32
    positions = torch.arange(seq_len, device=device, dtype=work_dtype)
    inv_freq = base ** (
        -torch.arange(0, dim, 2, device=device, dtype=work_dtype) / dim
    )
    ramp = torch.linspace(0.0, 1.0, dim // 2, device=device, dtype=work_dtype)
    scale = 1.0 + (scaling_factor - 1.0) * ramp
    if seq_len <= max_position_embeddings:
        scale = torch.ones_like(scale)
    freqs = torch.outer(positions, inv_freq / scale)
    cos, sin = freqs.cos(), freqs.sin()
    return (cos if dtype is None else cos.to(dtype), sin if dtype is None else sin.to(dtype))


def longrope_style_position_map(
    position_ids: torch.Tensor,
    short_factor: float | None = None,
    long_factor: float | None = None,
) -> torch.Tensor:
    """Map positions with a LongRoPE-inspired piecewise interpolation.

    Educational approximation, not a drop-in reproduction of the paper
    implementation. Earlier positions are compressed less than later positions.
    """

    short_factor = 1.0 if short_factor is None else short_factor
    long_factor = 4.0 if long_factor is None else long_factor
    if short_factor <= 0 or long_factor <= 0:
        raise ValueError("short_factor and long_factor must be positive")
    positions = position_ids.to(torch.float32)
    if positions.numel() == 0:
        return positions
    pivot = positions.max().clamp_min(1) * 0.25
    early = positions / short_factor
    late = pivot / short_factor + (positions - pivot).clamp_min(0) / long_factor
    return torch.where(positions <= pivot, early, late)
