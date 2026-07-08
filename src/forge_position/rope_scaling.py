"""Educational RoPE scaling utilities.

YaRN and LongRoPE-style functions here are clear approximations for learning
and experimentation. They are not drop-in reproductions of paper or vendor
implementations.
"""

from __future__ import annotations

import math

import torch
from torch import nn

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


class RotaryEmbedding(nn.Module):
    """RoPE cache module with an educational YaRN/NTK-by-parts scaling path.

    This class follows the shape convention used by several teaching
    implementations: query and key tensors are shaped ``[B, T, H, D]``. The
    rest of this repository's attention modules use ``[B, H, T, D]`` and call
    the functional helpers in ``rope.py``.

    When ``scaling_factor > 1``, the inverse-frequency schedule blends
    unscaled RoPE frequencies with linearly interpolated frequencies between
    two NTK-by-parts cut points. This is useful for studying YaRN-like behavior,
    but it is still educational reference code, not a guaranteed reproduction
    of any production implementation.
    """

    def __init__(
        self,
        head_dim: int,
        base: float = 10000.0,
        dtype: torch.dtype = torch.float32,
        initial_context_length: int = 4096,
        max_context_length: int = 131072,
        scaling_factor: float = 1.0,
        ntk_alpha: float = 1.0,
        ntk_beta: float = 32.0,
        device: torch.device | str | None = None,
        wrap_positions: bool = False,
    ) -> None:
        super().__init__()
        if head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError(f"head_dim must be a positive even integer, got {head_dim}")
        if base <= 0:
            raise ValueError("base must be positive")
        if initial_context_length <= 0 or max_context_length <= 0:
            raise ValueError("context lengths must be positive")
        if scaling_factor <= 0:
            raise ValueError("scaling_factor must be positive")
        if ntk_alpha <= 0 or ntk_beta <= 0 or ntk_alpha >= ntk_beta:
            raise ValueError("expected 0 < ntk_alpha < ntk_beta")

        self.head_dim = head_dim
        self.base = float(base)
        self.dtype = dtype
        self.initial_context_length = initial_context_length
        self.max_context_length = max_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta
        self.wrap_positions = wrap_positions

        cos, sin = self._compute_cos_sin(0, max_context_length, device=device)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _compute_concentration_and_inv_freq(
        self, device: torch.device | str | None = None
    ) -> tuple[float, torch.Tensor]:
        pair_indices = torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device)
        freqs = self.base ** (pair_indices / self.head_dim)

        if self.scaling_factor <= 1.0:
            return 1.0, 1.0 / freqs

        concentration = 0.1 * math.log(self.scaling_factor) + 1.0
        d_half = self.head_dim // 2
        low = (
            d_half
            * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi))
            / math.log(self.base)
        )
        high = (
            d_half
            * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi))
            / math.log(self.base)
        )

        interpolation = 1.0 / (self.scaling_factor * freqs)
        extrapolation = 1.0 / freqs

        if not (0.0 < low < high < d_half - 1):
            ramp = torch.linspace(0.0, 1.0, d_half, device=freqs.device, dtype=freqs.dtype)
        else:
            ramp = (
                torch.arange(d_half, dtype=freqs.dtype, device=freqs.device) - low
            ) / (high - low)
        mask = 1.0 - ramp.clamp(0.0, 1.0)
        inv_freqs = interpolation * (1.0 - mask) + extrapolation * mask
        return concentration, inv_freqs

    def _compute_cos_sin(
        self,
        start: int,
        num_tokens: int,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if start < 0 or num_tokens <= 0:
            raise ValueError("start must be non-negative and num_tokens must be positive")
        concentration, inv_freqs = self._compute_concentration_and_inv_freq(device)
        positions = torch.arange(
            start, start + num_tokens, dtype=torch.float32, device=inv_freqs.device
        )
        freqs = torch.outer(positions, inv_freqs)
        cos = freqs.cos() * concentration
        sin = freqs.sin() * concentration
        return cos.to(self.dtype), sin.to(self.dtype)

    def _select_positions(
        self,
        num_tokens: int,
        offset: int | torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(offset, int):
            idx = torch.arange(num_tokens, device=device, dtype=torch.long) + offset
        else:
            if offset.ndim == 0:
                idx = torch.arange(num_tokens, device=device, dtype=torch.long) + offset.to(
                    device=device, dtype=torch.long
                )
            elif offset.shape == (num_tokens,):
                idx = offset.to(device=device, dtype=torch.long)
            else:
                raise ValueError("offset must be an int, scalar tensor, or tensor shaped [T]")
        if self.wrap_positions:
            idx = idx % self.max_context_length
        elif torch.any(idx >= self.max_context_length).item() or torch.any(idx < 0).item():
            raise ValueError("requested positions exceed cached RoPE length")
        return idx

    def _rotate(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"x must have shape [B, T, H, D], got {tuple(x.shape)}")
        if x.shape[-1] != self.head_dim:
            raise ValueError(f"expected head_dim={self.head_dim}, got {x.shape[-1]}")
        cos = cos.unsqueeze(0).unsqueeze(2).to(device=x.device, dtype=x.dtype)
        sin = sin.unsqueeze(0).unsqueeze(2).to(device=x.device, dtype=x.dtype)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        offset: int | torch.Tensor = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate query and key tensors shaped ``[B, T, H, D]``."""

        if query.ndim != 4 or key.ndim != 4:
            raise ValueError("query and key must have shape [B, T, H, D]")
        if query.shape[:2] != key.shape[:2] or query.shape[-1] != key.shape[-1]:
            raise ValueError("query and key must match batch, sequence, and head_dim")
        num_tokens = query.shape[1]
        idx = self._select_positions(num_tokens, offset, query.device)
        cos = self.cos.index_select(0, idx)
        sin = self.sin.index_select(0, idx)
        return self._rotate(query, cos, sin), self._rotate(key, cos, sin)


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
