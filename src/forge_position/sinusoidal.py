"""Sinusoidal positional encodings from the original Transformer."""

from __future__ import annotations

import torch
from torch import nn


def build_sinusoidal_table(
    max_seq_len: int,
    d_model: int,
    base: float = 10000.0,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Build a ``[max_seq_len, d_model]`` table with the exact sin/cos formula.

    Odd ``d_model`` values are handled by filling the final even channel with a
    sine term and omitting its missing cosine partner.
    """

    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    if d_model <= 0:
        raise ValueError("d_model must be positive")

    work_dtype = dtype if dtype is not None and dtype.is_floating_point else torch.float32
    positions = torch.arange(max_seq_len, device=device, dtype=work_dtype).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=work_dtype)
        * (-torch.log(torch.tensor(base, device=device, dtype=work_dtype)) / d_model)
    )
    table = torch.zeros(max_seq_len, d_model, device=device, dtype=work_dtype)
    angles = positions * div_term
    table[:, 0::2] = torch.sin(angles)
    if d_model > 1:
        table[:, 1::2] = torch.cos(angles[:, : table[:, 1::2].shape[1]])
    return table if dtype is None else table.to(dtype)


class SinusoidalPositionEncoding(nn.Module):
    """Add fixed sinusoidal encodings to ``[B, T, D]`` tensors."""

    def __init__(self, max_seq_len: int, d_model: int, base: float = 10000.0) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        table = build_sinusoidal_table(max_seq_len, d_model, base=base)
        self.register_buffer("table", table, persistent=False)

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return ``x`` plus deterministic positional encodings."""

        if x.ndim != 3:
            raise ValueError(f"x must have shape [B, T, D], got {tuple(x.shape)}")
        batch_size, seq_len, d_model = x.shape
        if d_model != self.d_model:
            raise ValueError(f"expected D={self.d_model}, got D={d_model}")

        table = self.table.to(device=x.device, dtype=x.dtype)
        if position_ids is None:
            if seq_len > self.max_seq_len:
                raise ValueError(
                    f"sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}"
                )
            pos = table[:seq_len].unsqueeze(0)
        else:
            if position_ids.shape != (batch_size, seq_len):
                raise ValueError(
                    "position_ids must have shape [B, T], "
                    f"got {tuple(position_ids.shape)}"
                )
            if torch.any(position_ids >= self.max_seq_len).item():
                raise ValueError(f"position_ids exceed max_seq_len={self.max_seq_len}")
            pos = table[position_ids.to(device=x.device)]
        return x + pos
