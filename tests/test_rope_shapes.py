import torch

from forge_position import apply_rope, apply_rope_single, build_rope_cache


def test_rope_output_shapes_and_partial_rotary_dim() -> None:
    q = torch.randn(2, 3, 5, 8)
    k = torch.randn(2, 3, 5, 8)
    cos, sin = build_rope_cache(5, 4)
    q_out, k_out = apply_rope(q, k, cos, sin, rotary_dim=4)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape
    assert torch.allclose(q_out[..., 4:], q[..., 4:])


def test_rope_preserves_norm_on_rotated_dims() -> None:
    x = torch.randn(2, 4, 6, 8)
    cos, sin = build_rope_cache(6, 8)
    y = apply_rope_single(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)


def test_rope_position_ids_and_interleaved() -> None:
    x = torch.randn(2, 2, 3, 6)
    position_ids = torch.tensor([[0, 2, 4], [1, 3, 5]])
    cos, sin = build_rope_cache(6, 6)
    y = apply_rope_single(x, cos, sin, position_ids=position_ids, interleaved=True)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
