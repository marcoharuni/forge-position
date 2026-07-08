import torch

from forge_position import (
    append_kv,
    build_position_ids_for_decode,
    build_position_ids_for_prefill,
)


def test_prefill_and_decode_position_ids() -> None:
    prefill = build_position_ids_for_prefill(2, 4)
    decode = build_position_ids_for_decode(2, past_len=4)
    assert prefill.tolist() == [[0, 1, 2, 3], [0, 1, 2, 3]]
    assert decode.tolist() == [[4], [4]]


def test_append_kv_tracks_positions() -> None:
    k = torch.randn(1, 2, 3, 4)
    v = torch.randn(1, 2, 3, 4)
    pos = torch.tensor([[0, 1, 2]])
    cache = append_kv(None, k, v, pos)
    cache = append_kv(cache, k[:, :, :1], v[:, :, :1], torch.tensor([[3]]))
    assert cache.seq_len == 4
    assert cache.position_ids is not None
    assert cache.position_ids.tolist() == [[0, 1, 2, 3]]
