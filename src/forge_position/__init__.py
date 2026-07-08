"""forge-position: positional encoding laboratories for modern LLMs."""

from .alibi import apply_alibi, build_alibi_bias, get_alibi_slopes
from .attention import PositionAwareSelfAttention
from .kv_cache import (
    KVCache,
    append_kv,
    build_position_ids_for_decode,
    build_position_ids_for_prefill,
    verify_prefill_decode_equivalence,
)
from .learned import LearnedAbsolutePositionEmbedding
from .metrics import (
    attention_distance_statistics,
    exact_match,
    length_bucket_accuracy,
    lost_in_middle_score,
    passkey_accuracy,
    perplexity_from_loss,
)
from .nope import HybridPositionPattern, NoPEAttentionConfig
from .rope import apply_rope, apply_rope_single, build_rope_cache, rotate_half
from .rope_scaling import (
    dynamic_ntk_base,
    linear_position_scaling,
    longrope_style_position_map,
    ntk_scaled_rope_cache,
    yarn_scaled_rope_cache,
)
from .sinusoidal import SinusoidalPositionEncoding, build_sinusoidal_table
from .tiny_transformer import TinyDecoderOnlyTransformer, TinyTransformerConfig
from .visualization import (
    plot_alibi_bias,
    plot_length_extrapolation_results,
    plot_position_mapping,
    plot_rope_frequencies,
    plot_rope_rotation_circle,
    plot_sinusoidal_table,
)

__all__ = [
    "HybridPositionPattern",
    "KVCache",
    "LearnedAbsolutePositionEmbedding",
    "NoPEAttentionConfig",
    "PositionAwareSelfAttention",
    "SinusoidalPositionEncoding",
    "TinyDecoderOnlyTransformer",
    "TinyTransformerConfig",
    "append_kv",
    "apply_alibi",
    "apply_rope",
    "apply_rope_single",
    "attention_distance_statistics",
    "build_alibi_bias",
    "build_position_ids_for_decode",
    "build_position_ids_for_prefill",
    "build_rope_cache",
    "build_sinusoidal_table",
    "dynamic_ntk_base",
    "exact_match",
    "get_alibi_slopes",
    "length_bucket_accuracy",
    "linear_position_scaling",
    "longrope_style_position_map",
    "lost_in_middle_score",
    "ntk_scaled_rope_cache",
    "passkey_accuracy",
    "perplexity_from_loss",
    "plot_alibi_bias",
    "plot_length_extrapolation_results",
    "plot_position_mapping",
    "plot_rope_frequencies",
    "plot_rope_rotation_circle",
    "plot_sinusoidal_table",
    "rotate_half",
    "verify_prefill_decode_equivalence",
    "yarn_scaled_rope_cache",
]
