"""Rotary Position Embedding (RoPE) reference implementation."""

from __future__ import annotations

import torch


def rotate_half(x: torch.Tensor, interleaved: bool = False) -> torch.Tensor:
    """Rotate pairs by 90 degrees in RoPE space.

    Non-interleaved style treats the first half and second half as paired
    coordinates: ``[x0, x1, y0, y1] -> [-y0, -y1, x0, x1]``.
    Interleaved style treats adjacent channels as pairs:
    ``[x0, y0, x1, y1] -> [-y0, x0, -y1, x1]``.
    """

    dim = x.shape[-1]
    if dim % 2 != 0:
        raise ValueError(f"RoPE rotation dimension must be even, got {dim}")
    if interleaved:
        pairs = x.reshape(*x.shape[:-1], dim // 2, 2)
        rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1)
        return rotated.reshape_as(x)
    x1, x2 = x[..., : dim // 2], x[..., dim // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def build_rope_cache(
    seq_len: int,
    dim: int,
    base: float = 10000.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build RoPE cosine and sine phase tables.

    Returns half-width phase tables shaped ``[seq_len, dim // 2]``. The apply
    functions expand them to interleaved or non-interleaved layouts as needed.
    """

    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if dim <= 0 or dim % 2 != 0:
        raise ValueError(f"dim must be a positive even integer, got {dim}")

    work_dtype = dtype if dtype is not None and dtype.is_floating_point else torch.float32
    positions = torch.arange(seq_len, device=device, dtype=work_dtype)
    inv_freq = base ** (
        -torch.arange(0, dim, 2, device=device, dtype=work_dtype) / dim
    )
    freqs = torch.outer(positions, inv_freq)
    cos, sin = freqs.cos(), freqs.sin()
    if dtype is not None:
        cos, sin = cos.to(dtype), sin.to(dtype)
    return cos, sin


def _select_cache(
    cache: torch.Tensor,
    *,
    position_ids: torch.Tensor | None,
    batch_size: int,
    seq_len: int,
    rotary_dim: int,
    interleaved: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Select and expand a RoPE cache to ``[B or 1, 1, T, rotary_dim]``."""

    if cache.ndim == 2:
        if position_ids is None:
            if cache.shape[0] < seq_len:
                raise ValueError(f"cache length {cache.shape[0]} is shorter than T={seq_len}")
            selected = cache[:seq_len].unsqueeze(0)
        else:
            if position_ids.shape != (batch_size, seq_len):
                raise ValueError(
                    f"position_ids must have shape [B, T], got {tuple(position_ids.shape)}"
                )
            if torch.any(position_ids >= cache.shape[0]).item():
                raise ValueError("position_ids refer past the RoPE cache length")
            selected = cache.to(device=device)[position_ids.to(device=device)]
    elif cache.ndim == 3:
        if cache.shape[:2] != (batch_size, seq_len):
            raise ValueError(
                f"3D cache must have shape [B, T, C], got {tuple(cache.shape)}"
            )
        selected = cache
    else:
        raise ValueError("cos/sin cache must have shape [T, C] or [B, T, C]")

    if selected.shape[-1] == rotary_dim // 2:
        if interleaved:
            selected = selected.repeat_interleave(2, dim=-1)
        else:
            selected = torch.cat((selected, selected), dim=-1)
    elif selected.shape[-1] != rotary_dim:
        raise ValueError(
            f"cache last dim must be rotary_dim/2 or rotary_dim, got {selected.shape[-1]}"
        )
    return selected.to(device=device, dtype=dtype).unsqueeze(1)


def apply_rope_single(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    interleaved: bool = False,
    rotary_dim: int | None = None,
) -> torch.Tensor:
    """Apply RoPE to one tensor shaped ``[B, H, T, D]``."""

    if x.ndim != 4:
        raise ValueError(f"x must have shape [B, H, T, D], got {tuple(x.shape)}")
    batch_size, _, seq_len, head_dim = x.shape
    rotary_dim = head_dim if rotary_dim is None else rotary_dim
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2 != 0:
        raise ValueError(
            f"rotary_dim must be an even integer in [1, {head_dim}], got {rotary_dim}"
        )

    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    cos_selected = _select_cache(
        cos,
        position_ids=position_ids,
        batch_size=batch_size,
        seq_len=seq_len,
        rotary_dim=rotary_dim,
        interleaved=interleaved,
        device=x.device,
        dtype=x.dtype,
    )
    sin_selected = _select_cache(
        sin,
        position_ids=position_ids,
        batch_size=batch_size,
        seq_len=seq_len,
        rotary_dim=rotary_dim,
        interleaved=interleaved,
        device=x.device,
        dtype=x.dtype,
    )
    rotated = (x_rot * cos_selected) + (rotate_half(x_rot, interleaved) * sin_selected)
    return torch.cat((rotated, x_pass), dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    interleaved: bool = False,
    rotary_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors shaped ``[B, H, T, D]``."""

    if q.shape != k.shape:
        raise ValueError(f"q and k must have identical shapes, got {q.shape} and {k.shape}")
    return (
        apply_rope_single(q, cos, sin, position_ids, interleaved, rotary_dim),
        apply_rope_single(k, cos, sin, position_ids, interleaved, rotary_dim),
    )
