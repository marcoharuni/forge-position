"""Demonstrate correct and incorrect KV-cache position IDs."""

import torch

from forge_position import (
    build_position_ids_for_decode,
    build_position_ids_for_prefill,
)


def main() -> None:
    batch_size = 1
    prompt_len = 5
    prefill = build_position_ids_for_prefill(batch_size, prompt_len)
    correct_decode = build_position_ids_for_decode(batch_size, past_len=prompt_len)
    reset_bug = torch.zeros_like(correct_decode)
    left_padding_bug = torch.tensor([[0, 0, 1, 2, 3]])

    print("Correct prefill position_ids:", prefill.tolist())
    print("Correct first decode position_id:", correct_decode.tolist())
    print("Reset-position bug:", reset_bug.tolist())
    print("Left-padding bug example:", left_padding_bug.tolist())
    print("Expected: decode continues at past_len, not zero.")
    print("Wrong behavior: RoPE phase for the next token reuses position 0.")


if __name__ == "__main__":
    main()
