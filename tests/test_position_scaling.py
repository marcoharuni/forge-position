import torch

from forge_position import (
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
