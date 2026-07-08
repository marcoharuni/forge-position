"""Tiny CPU length-extrapolation lab on a synthetic next-token task."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from forge_position import (
    TinyDecoderOnlyTransformer,
    TinyTransformerConfig,
    plot_length_extrapolation_results,
)


def make_batch(batch_size: int, seq_len: int, vocab_size: int) -> torch.Tensor:
    starts = torch.randint(0, vocab_size, (batch_size, 1))
    offsets = torch.arange(seq_len).unsqueeze(0)
    return (starts + offsets) % vocab_size


def train_and_eval(method: str, train_len: int, eval_lengths: list[int]) -> dict[str, float]:
    torch.manual_seed(11)
    vocab_size = 32
    model = TinyDecoderOnlyTransformer(
        TinyTransformerConfig(
            vocab_size=vocab_size,
            d_model=32,
            n_heads=4,
            n_layers=2,
            max_seq_len=max([train_len, *eval_lengths]),
            position_method=method,
        )
    )
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    for _ in tqdm(range(8), desc=f"train {method}", leave=False):
        tokens = make_batch(8, train_len, vocab_size)
        logits, _ = model(tokens)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size), tokens[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    scores: dict[str, float] = {}
    with torch.no_grad():
        for length in eval_lengths:
            tokens = make_batch(16, length, vocab_size)
            logits, _ = model(tokens)
            pred = logits[:, :-1].argmax(dim=-1)
            acc = (pred == tokens[:, 1:]).float().mean().item()
            scores[str(length)] = acc
    return scores


def main() -> None:
    methods = ["learned", "sinusoidal", "rope", "alibi", "nope"]
    train_len = 16
    eval_lengths = [16, 24, 32]
    results = {method: train_and_eval(method, train_len, eval_lengths) for method in methods}
    out = Path("docs/length_extrapolation_results.json")
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    plot = plot_length_extrapolation_results(results)
    print("Synthetic toy task only; these are not benchmarks.")
    print(json.dumps(results, indent=2))
    print(f"Saved JSON to {out}")
    print(f"Saved plot to {plot}")


if __name__ == "__main__":
    main()
