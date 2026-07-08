"""KV-cache helpers with explicit position handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class KVCache:
    """Cached key/value tensors for autoregressive decoding.

    Tensors are shaped ``[B, H, T, D]``. ``position_ids`` tracks the positions
    used when creating the cached keys, which is crucial for RoPE debugging.
    """

    key: torch.Tensor
    value: torch.Tensor
    position_ids: torch.Tensor | None = None

    @property
    def seq_len(self) -> int:
        """Number of cached tokens."""

        return self.key.shape[-2]


def append_kv(
    cache: KVCache | None,
    k: torch.Tensor,
    v: torch.Tensor,
    position_ids: torch.Tensor | None = None,
) -> KVCache:
    """Append key/value tensors along the sequence dimension."""

    if k.shape != v.shape:
        raise ValueError(f"k and v must have the same shape, got {k.shape} and {v.shape}")
    if k.ndim != 4:
        raise ValueError(f"k/v must have shape [B, H, T, D], got {tuple(k.shape)}")
    if position_ids is not None and position_ids.shape != (k.shape[0], k.shape[-2]):
        raise ValueError("position_ids must have shape [B, T]")

    if cache is None:
        return KVCache(k, v, position_ids)
    if cache.key.shape[:2] != k.shape[:2] or cache.key.shape[-1] != k.shape[-1]:
        raise ValueError("cache and new tensors must match [B, H, D]")
    key = torch.cat((cache.key, k), dim=-2)
    value = torch.cat((cache.value, v), dim=-2)
    if cache.position_ids is not None or position_ids is not None:
        if cache.position_ids is None or position_ids is None:
            raise ValueError("cannot mix cached and uncached position_ids")
        all_positions = torch.cat((cache.position_ids, position_ids), dim=-1)
    else:
        all_positions = None
    return KVCache(key, value, all_positions)


def build_position_ids_for_prefill(
    batch_size: int,
    seq_len: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return ``[[0, 1, ..., T-1]]`` repeated for each batch item."""

    if batch_size <= 0 or seq_len <= 0:
        raise ValueError("batch_size and seq_len must be positive")
    return torch.arange(seq_len, device=device, dtype=torch.long).expand(batch_size, seq_len)


def build_position_ids_for_decode(
    batch_size: int,
    past_len: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the next decode position ``past_len`` shaped ``[B, 1]``."""

    if batch_size <= 0 or past_len < 0:
        raise ValueError("batch_size must be positive and past_len must be non-negative")
    return torch.full((batch_size, 1), past_len, device=device, dtype=torch.long)


@torch.no_grad()
def verify_prefill_decode_equivalence(
    forward_fn: Callable[..., tuple[torch.Tensor, KVCache | None] | torch.Tensor],
    tokens: torch.Tensor,
) -> float:
    """Compare full-prefill logits with token-by-token cached decoding.

    ``forward_fn`` should accept ``tokens``, ``position_ids``, ``past_kv``, and
    ``use_cache`` keyword arguments, as ``TinyDecoderOnlyTransformer`` does.
    Returns the maximum absolute difference between logits.
    """

    batch_size, seq_len = tokens.shape
    prefill_pos = build_position_ids_for_prefill(batch_size, seq_len, tokens.device)
    full = forward_fn(tokens, position_ids=prefill_pos, use_cache=False)
    full_logits = full[0] if isinstance(full, tuple) else full

    cache = None
    pieces = []
    for idx in range(seq_len):
        pos = build_position_ids_for_decode(batch_size, idx, tokens.device)
        out = forward_fn(
            tokens[:, idx : idx + 1],
            position_ids=pos,
            past_kv=cache,
            use_cache=True,
        )
        logits, cache = out if isinstance(out, tuple) else (out, None)
        pieces.append(logits)
    decoded_logits = torch.cat(pieces, dim=1)
    return (full_logits - decoded_logits).abs().max().item()
