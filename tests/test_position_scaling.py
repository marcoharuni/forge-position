import torch

from forge_position import (
    RotaryEmbedding,
    linear_position_scaling,
    longrope_style_position_map,
    ntk_scaled_rope_cache,
    yarn_scaled_rope_cache,
)


def test_scaled_rope_caches_have_shapes_and_no_nans() -> None:
    cos, sin = ntk_scaled_rope_cache(64, 8, max_position_embeddings=16, scaling_factor=4)
    assert cos.shape == (64, 4)
    assert sin.shape == (64, 4)
    assert not torch.isnan(cos).any()
    cos_y, sin_y = yarn_scaled_rope_cache(64, 8, max_position_embeddings=16, scaling_factor=4)
    assert cos_y.shape == (64, 4)
    assert not torch.isnan(sin_y).any()


def test_scaled_position_ids_are_monotonic() -> None:
    ids = torch.arange(32)
    linear = linear_position_scaling(ids, 4)
    mapped = longrope_style_position_map(ids, short_factor=1, long_factor=4)
    assert torch.all(linear[1:] >= linear[:-1])
    assert torch.all(mapped[1:] >= mapped[:-1])


def test_rotary_embedding_class_shapes_and_finiteness() -> None:
    rotary = RotaryEmbedding(
        head_dim=8,
        base=10000.0,
        initial_context_length=32,
        max_context_length=64,
        scaling_factor=4.0,
    )
    q = torch.randn(2, 5, 3, 8)
    k = torch.randn(2, 5, 3, 8)
    q_out, k_out = rotary(q, k, offset=7)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape
    assert not torch.isnan(q_out).any()
    assert not torch.isnan(k_out).any()


def test_rotary_embedding_unscaled_preserves_norm() -> None:
    rotary = RotaryEmbedding(head_dim=8, max_context_length=16, scaling_factor=1.0)
    q = torch.randn(1, 4, 2, 8)
    k = torch.randn(1, 4, 2, 8)
    q_out, k_out = rotary(q, k)
    assert torch.allclose(q.norm(dim=-1), q_out.norm(dim=-1), atol=1e-5)
    assert torch.allclose(k.norm(dim=-1), k_out.norm(dim=-1), atol=1e-5)
