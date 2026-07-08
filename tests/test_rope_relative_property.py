import torch

from forge_position import apply_rope, build_rope_cache


def test_rope_consistent_shift_preserves_attention_scores() -> None:
    torch.manual_seed(0)
    q = torch.randn(1, 2, 5, 8)
    k = torch.randn(1, 2, 5, 8)
    cos, sin = build_rope_cache(16, 8)
    base_pos = torch.arange(5).unsqueeze(0)
    shifted_pos = base_pos + 7

    q1, k1 = apply_rope(q, k, cos, sin, position_ids=base_pos)
    q2, k2 = apply_rope(q, k, cos, sin, position_ids=shifted_pos)
    scores1 = q1 @ k1.transpose(-2, -1)
    scores2 = q2 @ k2.transpose(-2, -1)
    assert torch.allclose(scores1, scores2, atol=1e-5)
