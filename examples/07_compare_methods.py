"""Quick CPU-friendly comparison table for positional methods."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from forge_position import TinyDecoderOnlyTransformer, TinyTransformerConfig, plot_length_extrapolation_results


def make_batch(batch_size: int, seq_len: int, vocab_size: int) -> torch.Tensor:
    starts = torch.arange(batch_size).unsqueeze(1) % vocab_size
    return (starts + torch.arange(seq_len).unsqueeze(0)) % vocab_size


def run_method(method: str, train_length: int, eval_length: int) -> dict[str, float | int | str]:
    torch.manual_seed(23)
    vocab_size = 24
    model = TinyDecoderOnlyTransformer(
        TinyTransformerConfig(
            vocab_size=vocab_size,
            d_model=24,
            n_heads=4,
            n_layers=1,
            max_seq_len=max(train_length, eval_length),
            position_method=method,
        )
    )
    opt = torch.optim.AdamW(model.parameters(), lr=4e-3)
    for _ in range(6):
        tokens = make_batch(8, train_length, vocab_size)
        logits, _ = model(tokens)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size), tokens[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        tokens = make_batch(8, eval_length, vocab_size)
        logits, _ = model(tokens)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size), tokens[:, 1:].reshape(-1))
        acc = (logits[:, :-1].argmax(dim=-1) == tokens[:, 1:]).float().mean().item()
    return {
        "method": method,
        "train_length": train_length,
        "eval_length": eval_length,
        "loss": float(loss),
        "accuracy": acc,
    }


def main() -> None:
    methods = ["learned", "sinusoidal", "rope", "alibi", "nope"]
    rows = [run_method(method, train_length=16, eval_length=32) for method in methods]
    print("Toy comparison after six CPU training steps. Not a benchmark.")
    print("method        train_length  eval_length  loss    accuracy")
    for row in rows:
        print(
            f"{row['method']:<13} {row['train_length']:>12} {row['eval_length']:>12} "
            f"{row['loss']:.4f}  {row['accuracy']:.3f}"
        )
    out = Path("docs/compare_methods_results.json")
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    plot_data = {str(row["method"]): {str(row["eval_length"]): float(row["accuracy"])} for row in rows}
    plot = plot_length_extrapolation_results(plot_data, "docs/figures/compare_methods.png")
    print(f"Saved JSON to {out}")
    print(f"Saved plot to {plot}")


if __name__ == "__main__":
    main()
