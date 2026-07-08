"""No positional encoding (NoPE) helpers and hybrid layer patterns."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NoPEAttentionConfig:
    """Configuration for attention without explicit positional encoding."""

    enabled: bool = True
    causal_mask_supplies_position_signal: bool = True


class HybridPositionPattern:
    """Choose RoPE for selected layers and NoPE for the rest."""

    def __init__(self, n_layers: int, rope_layers: list[int] | None = None) -> None:
        if n_layers <= 0:
            raise ValueError("n_layers must be positive")
        self.n_layers = n_layers
        if rope_layers is None:
            rope_layers = [layer for layer in range(n_layers) if layer % 2 == 0]
        invalid = [layer for layer in rope_layers if layer < 0 or layer >= n_layers]
        if invalid:
            raise ValueError(f"rope_layers out of range: {invalid}")
        self.rope_layers = set(rope_layers)

    def method_for_layer(self, layer_idx: int) -> str:
        """Return ``'rope'`` or ``'nope'`` for a zero-based layer index."""

        if layer_idx < 0 or layer_idx >= self.n_layers:
            raise ValueError(f"layer_idx {layer_idx} out of range for {self.n_layers} layers")
        return "rope" if layer_idx in self.rope_layers else "nope"
