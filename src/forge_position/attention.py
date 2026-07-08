"""Position-aware causal self-attention."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .alibi import build_alibi_bias
from .kv_cache import KVCache, append_kv
from .rope import apply_rope, build_rope_cache
from .rope_scaling import ntk_scaled_rope_cache, yarn_scaled_rope_cache


class PositionAwareSelfAttention(nn.Module):
    """Small MHA module demonstrating where positional methods enter attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        position_method: str = "rope",
        max_seq_len: int = 2048,
        layer_idx: int = 0,
        rope_base: float = 10000.0,
        rotary_dim: int | None = None,
        rope_interleaved: bool = False,
        rope_scaling_factor: float = 2.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.position_method = position_method
        self.max_seq_len = max_seq_len
        self.layer_idx = layer_idx
        self.rope_base = rope_base
        self.rotary_dim = self.head_dim if rotary_dim is None else rotary_dim
        self.rope_interleaved = rope_interleaved
        self.rope_scaling_factor = rope_scaling_factor

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def _rope_cache(self, total_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        method = self.position_method
        if method == "rope_ntk_scaled":
            return ntk_scaled_rope_cache(
                total_len,
                self.rotary_dim,
                self.rope_base,
                self.max_seq_len,
                self.rope_scaling_factor,
                device=device,
                dtype=dtype,
            )
        if method == "rope_yarn_scaled":
            return yarn_scaled_rope_cache(
                total_len,
                self.rotary_dim,
                self.rope_base,
                self.max_seq_len,
                self.rope_scaling_factor,
                device=device,
                dtype=dtype,
            )
        return build_rope_cache(total_len, self.rotary_dim, self.rope_base, device=device, dtype=dtype)

    def _causal_mask(
        self, query_len: int, key_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        query_pos = torch.arange(key_len - query_len, key_len, device=device)
        key_pos = torch.arange(key_len, device=device)
        allowed = key_pos.unsqueeze(0) <= query_pos.unsqueeze(1)
        mask = torch.zeros(query_len, key_len, device=device, dtype=dtype)
        mask = mask.masked_fill(~allowed, torch.finfo(dtype).min)
        return mask.view(1, 1, query_len, key_len)

    def forward(
        self,
        x: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        past_kv: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        """Run causal self-attention over ``x`` shaped ``[B, T, D]``."""

        if x.ndim != 3:
            raise ValueError(f"x must have shape [B, T, D], got {tuple(x.shape)}")
        batch_size, seq_len, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = self._split_heads(q), self._split_heads(k), self._split_heads(v)

        past_len = 0 if past_kv is None else past_kv.seq_len
        if position_ids is None:
            position_ids = torch.arange(
                past_len, past_len + seq_len, device=x.device, dtype=torch.long
            ).expand(batch_size, seq_len)
        elif position_ids.shape != (batch_size, seq_len):
            raise ValueError("position_ids must have shape [B, T]")

        method = self.position_method
        if method in {"rope", "hybrid", "rope_linear_scaled", "rope_ntk_scaled", "rope_yarn_scaled"}:
            rope_position_ids = position_ids
            if method == "rope_linear_scaled":
                rope_position_ids = (position_ids.to(torch.float32) / self.rope_scaling_factor).round().long()
            cache_len = int(rope_position_ids.max().item()) + 1
            cache_len = max(cache_len, past_len + seq_len, 1)
            cos, sin = self._rope_cache(cache_len, x.device, q.dtype)
            q, k = apply_rope(
                q,
                k,
                cos,
                sin,
                position_ids=rope_position_ids,
                interleaved=self.rope_interleaved,
                rotary_dim=self.rotary_dim,
            )

        new_cache = append_kv(past_kv, k, v, position_ids) if use_cache else None
        if past_kv is not None:
            k_all = torch.cat((past_kv.key, k), dim=-2)
            v_all = torch.cat((past_kv.value, v), dim=-2)
        else:
            k_all, v_all = k, v

        key_len = k_all.shape[-2]
        attn_mask = self._causal_mask(seq_len, key_len, x.device, q.dtype)
        if method == "alibi":
            attn_mask = attn_mask + build_alibi_bias(
                self.n_heads, seq_len, key_len, device=x.device, dtype=q.dtype
            )

        y = F.scaled_dot_product_attention(
            q,
            k_all,
            v_all,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        return self.out_proj(self._merge_heads(y)), new_cache
