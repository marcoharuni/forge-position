"""Small metrics for positional encoding experiments."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

import torch


def perplexity_from_loss(loss: float | torch.Tensor) -> float:
    """Convert cross-entropy loss to perplexity."""

    value = float(loss.detach().cpu()) if isinstance(loss, torch.Tensor) else float(loss)
    return math.exp(value)


def exact_match(predictions: Iterable[str], targets: Iterable[str]) -> float:
    """Return exact string match rate."""

    pairs = list(zip(predictions, targets))
    if not pairs:
        return 0.0
    return sum(pred == target for pred, target in pairs) / len(pairs)


def passkey_accuracy(predicted_keys: Iterable[str], true_keys: Iterable[str]) -> float:
    """Return passkey retrieval accuracy."""

    return exact_match(predicted_keys, true_keys)


def length_bucket_accuracy(
    predictions: Iterable[int],
    targets: Iterable[int],
    lengths: Iterable[int],
    bucket_size: int = 128,
) -> dict[str, float]:
    """Group token-level correctness by sequence-length bucket."""

    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for pred, target, length in zip(predictions, targets, lengths):
        bucket_start = (int(length) // bucket_size) * bucket_size
        key = f"{bucket_start}-{bucket_start + bucket_size - 1}"
        totals[key][0] += int(pred == target)
        totals[key][1] += 1
    return {key: correct / count for key, (correct, count) in totals.items() if count}


def attention_distance_statistics(attention_weights: torch.Tensor) -> dict[str, float]:
    """Summarize attention mass by relative distance for ``[B, H, T, T]`` weights."""

    if attention_weights.ndim != 4:
        raise ValueError("attention_weights must have shape [B, H, T, T]")
    _, _, query_len, key_len = attention_weights.shape
    q_pos = torch.arange(query_len, device=attention_weights.device)
    k_pos = torch.arange(key_len, device=attention_weights.device)
    distances = (q_pos[:, None] - k_pos[None, :]).clamp_min(0).to(attention_weights.dtype)
    mass = attention_weights.mean(dim=(0, 1))
    expected = (mass * distances).sum() / mass.sum().clamp_min(1e-12)
    recent_mass = mass[distances <= 4].sum() / mass.sum().clamp_min(1e-12)
    return {
        "expected_distance": float(expected.detach().cpu()),
        "recent_mass_distance_le_4": float(recent_mass.detach().cpu()),
    }


def lost_in_middle_score(
    correct: Iterable[bool],
    normalized_positions: Iterable[float],
) -> float:
    """Estimate middle degradation: edge accuracy minus middle accuracy."""

    edge_correct = []
    middle_correct = []
    for is_correct, pos in zip(correct, normalized_positions):
        if 0.33 <= pos <= 0.67:
            middle_correct.append(bool(is_correct))
        else:
            edge_correct.append(bool(is_correct))
    edge = sum(edge_correct) / len(edge_correct) if edge_correct else 0.0
    middle = sum(middle_correct) / len(middle_correct) if middle_correct else 0.0
    return edge - middle
