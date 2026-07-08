import torch

from forge_position import apply_alibi, build_alibi_bias, get_alibi_slopes


def test_alibi_shape_for_non_power_of_two_heads() -> None:
    bias = build_alibi_bias(3, 4, 6)
    assert bias.shape == (1, 3, 4, 6)
    slopes = get_alibi_slopes(3)
    assert slopes.shape == (3,)
    assert torch.all(slopes > 0)


def test_alibi_recency_penalty_is_monotonic() -> None:
    bias = build_alibi_bias(2, 5, 5)
    last_query = bias[0, 0, -1]
    assert last_query[0] < last_query[1] < last_query[2] < last_query[3] < last_query[4]
    scores = torch.zeros(1, 2, 5, 5)
    shifted = apply_alibi(scores, bias)
    assert torch.allclose(shifted, bias)
