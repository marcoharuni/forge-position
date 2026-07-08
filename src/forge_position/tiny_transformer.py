"""A CPU-friendly decoder-only Transformer for positional encoding labs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .attention import PositionAwareSelfAttention
from .kv_cache import KVCache, build_position_ids_for_decode, build_position_ids_for_prefill
from .learned import LearnedAbsolutePositionEmbedding
from .nope import HybridPositionPattern
from .sinusoidal import SinusoidalPositionEncoding


@dataclass
class TinyTransformerConfig:
    """Configuration for ``TinyDecoderOnlyTransformer``."""

    vocab_size: int = 64
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    max_seq_len: int = 128
    position_method: str = "rope"
    dropout: float = 0.0


class TransformerBlock(nn.Module):
    """Pre-LayerNorm decoder block with GELU MLP."""

    def __init__(self, config: TinyTransformerConfig, layer_idx: int) -> None:
        super().__init__()
        method = config.position_method
        if method == "hybrid":
            method = HybridPositionPattern(config.n_layers).method_for_layer(layer_idx)
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = PositionAwareSelfAttention(
            config.d_model,
            config.n_heads,
            position_method=method,
            max_seq_len=config.max_seq_len,
            layer_idx=layer_idx,
        )
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        past_kv: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        """Run one decoder block."""

        attn_out, new_cache = self.attn(
            self.ln1(x),
            position_ids=position_ids,
            past_kv=past_kv,
            use_cache=use_cache,
        )
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_cache


class TinyDecoderOnlyTransformer(nn.Module):
    """Tiny causal LM used for examples and tests.

    The model uses LayerNorm because it is familiar and robust for educational
    CPU experiments.
    """

    valid_position_methods = {
        "learned",
        "sinusoidal",
        "rope",
        "alibi",
        "nope",
        "hybrid",
        "rope_linear_scaled",
        "rope_ntk_scaled",
        "rope_yarn_scaled",
    }

    def __init__(self, config: TinyTransformerConfig) -> None:
        super().__init__()
        if config.position_method not in self.valid_position_methods:
            raise ValueError(f"unknown position_method={config.position_method}")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.input_position: nn.Module | None
        if config.position_method == "learned":
            self.input_position = LearnedAbsolutePositionEmbedding(
                config.max_seq_len, config.d_model
            )
            attn_method = "nope"
        elif config.position_method == "sinusoidal":
            self.input_position = SinusoidalPositionEncoding(
                max(config.max_seq_len * 4, config.max_seq_len), config.d_model
            )
            attn_method = "nope"
        else:
            self.input_position = None
            attn_method = config.position_method

        block_config = TinyTransformerConfig(**{**config.__dict__, "position_method": attn_method})
        self.blocks = nn.ModuleList(
            [TransformerBlock(block_config, idx) for idx in range(config.n_layers)]
        )
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        targets: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_kv: list[KVCache] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | tuple[torch.Tensor, list[KVCache]]:
        """Return logits and either loss or a new cache list."""

        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B, T]")
        batch_size, seq_len = input_ids.shape
        if position_ids is None:
            past_len = 0 if past_kv is None else past_kv[0].seq_len
            if past_len == 0:
                position_ids = build_position_ids_for_prefill(
                    batch_size, seq_len, input_ids.device
                )
            else:
                position_ids = torch.arange(
                    past_len, past_len + seq_len, device=input_ids.device
                ).expand(batch_size, seq_len)

        x = self.token_embedding(input_ids)
        if self.input_position is not None:
            x = self.input_position(x, position_ids)

        new_caches: list[KVCache] = []
        for idx, block in enumerate(self.blocks):
            layer_past = None if past_kv is None else past_kv[idx]
            x, cache = block(
                x,
                position_ids=position_ids,
                past_kv=layer_past,
                use_cache=use_cache,
            )
            if use_cache and cache is not None:
                new_caches.append(cache)

        logits = self.lm_head(self.ln_f(x))
        if use_cache:
            return logits, new_caches

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                targets[:, 1:].contiguous().view(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 8) -> torch.Tensor:
        """Greedy CPU-friendly generation for a few tokens."""

        self.eval()
        out = input_ids
        cache: list[KVCache] | None = None
        for step in range(max_new_tokens):
            if cache is None:
                pos = build_position_ids_for_prefill(out.shape[0], out.shape[1], out.device)
                logits, cache = self(out, position_ids=pos, use_cache=True)
            else:
                token = out[:, -1:]
                pos = build_position_ids_for_decode(out.shape[0], cache[0].seq_len, out.device)
                logits, cache = self(token, position_ids=pos, past_kv=cache, use_cache=True)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            out = torch.cat((out, next_token), dim=1)
        return out
