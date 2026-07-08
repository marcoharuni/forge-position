"""Learned absolute positional embeddings."""

from __future__ import annotations

import torch
from torch import nn


class LearnedAbsolutePositionEmbedding(nn.Module):
    """Add a learned absolute position vector to token representations.

    Args:
        max_seq_len: Maximum supported position index plus one.
        d_model: Model width. Must match the final dimension of the input.
    """

    def __init__(self, max_seq_len: int, d_model: int) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.embedding = nn.Embedding(max_seq_len, d_model)

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return ``x + position_embedding`` for ``x`` shaped ``[B, T, D]``."""

        if x.ndim != 3:
            raise ValueError(f"x must have shape [B, T, D], got {tuple(x.shape)}")
        batch_size, seq_len, d_model = x.shape
        if d_model != self.d_model:
            raise ValueError(f"expected D={self.d_model}, got D={d_model}")

        if position_ids is None:
            if seq_len > self.max_seq_len:
                raise ValueError(
                    f"sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}"
                )
            position_ids = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len)
        else:
            if position_ids.shape != (batch_size, seq_len):
                raise ValueError(
                    "position_ids must have shape [B, T], "
                    f"got {tuple(position_ids.shape)} for x={tuple(x.shape)}"
                )
            if torch.any(position_ids >= self.max_seq_len).item():
                raise ValueError(f"position_ids exceed max_seq_len={self.max_seq_len}")

        return x + self.embedding(position_ids.to(device=x.device))
