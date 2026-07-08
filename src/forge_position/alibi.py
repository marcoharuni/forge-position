"""Attention with Linear Biases (ALiBi)."""

from __future__ import annotations

import math

import torch


def get_alibi_slopes(n_heads: int) -> torch.Tensor:
    """Return ALiBi slopes for any positive number of heads."""

    if n_heads <= 0:
        raise ValueError("n_heads must be positive")

    def power_of_two_slopes(power: int) -> list[float]:
        start = 2.0 ** (-(2.0 ** -(math.log2(power) - 3)))
        ratio = start
        return [start * ratio**i for i in range(power)]

    if math.log2(n_heads).is_integer():
        slopes = power_of_two_slopes(n_heads)
    else:
        closest_power = 2 ** math.floor(math.log2(n_heads))
        slopes = power_of_two_slopes(closest_power)
        extra = power_of_two_slopes(2 * closest_power)[0::2]
        slopes.extend(extra[: n_heads - closest_power])
    return torch.tensor(slopes, dtype=torch.float32)


def build_alibi_bias(
    n_heads: int,
    query_len: int,
    key_len: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Build an ALiBi bias shaped ``[1, H, Tq, Tk]``.

    The most recent visible key receives zero bias; older keys receive
    increasingly negative bias proportional to distance.
    """

    if query_len <= 0 or key_len <= 0:
        raise ValueError("query_len and key_len must be positive")
    out_dtype = dtype if dtype is not None else torch.float32
    slopes = get_alibi_slopes(n_heads).to(device=device, dtype=out_dtype)
    key_pos = torch.arange(key_len, device=device, dtype=out_dtype)
    query_pos = torch.arange(
        key_len - query_len, key_len, device=device, dtype=out_dtype
    )
    distances = (query_pos[:, None] - key_pos[None, :]).clamp_min(0)
    return -slopes.view(1, n_heads, 1, 1) * distances.view(1, 1, query_len, key_len)


def apply_alibi(attn_scores: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Add ALiBi bias to attention scores."""

    if attn_scores.ndim != 4:
        raise ValueError("attn_scores must have shape [B, H, Tq, Tk]")
    return attn_scores + bias.to(device=attn_scores.device, dtype=attn_scores.dtype)
