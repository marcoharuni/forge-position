"""Compare attention scores before and after RoPE."""

import torch

from forge_position import apply_rope, build_rope_cache


def main() -> None:
    torch.manual_seed(7)
    q = torch.randn(1, 1, 6, 8)
    k = torch.randn(1, 1, 6, 8)
    raw_scores = (q @ k.transpose(-2, -1))[0, 0]
    cos, sin = build_rope_cache(16, 8)
    positions = torch.arange(6).unsqueeze(0)
    q_rot, k_rot = apply_rope(q, k, cos, sin, position_ids=positions)
    rope_scores = (q_rot @ k_rot.transpose(-2, -1))[0, 0]

    shifted = positions + 5
    q_shift, k_shift = apply_rope(q, k, cos, sin, position_ids=shifted)
    shifted_scores = (q_shift @ k_shift.transpose(-2, -1))[0, 0]

    print("Raw score[4, 1]:", round(raw_scores[4, 1].item(), 4))
    print("RoPE score[4, 1]:", round(rope_scores[4, 1].item(), 4))
    print(
        "Max difference after shifting all positions together:",
        float((rope_scores - shifted_scores).abs().max()),
    )
    print("That near-zero shift difference is the RoPE relative-position sanity check.")


if __name__ == "__main__":
    main()
