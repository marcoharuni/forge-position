"""Matplotlib visualization helpers for positional encoding labs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


DEFAULT_FIGURE_DIR = Path("docs/figures")


def _prepare_path(path: str | Path | None, filename: str) -> Path:
    out = Path(path) if path is not None else DEFAULT_FIGURE_DIR / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def plot_sinusoidal_table(table: torch.Tensor, path: str | Path | None = None) -> Path:
    """Save a heatmap of a sinusoidal table."""

    out = _prepare_path(path, "sinusoidal_table.png")
    plt.figure(figsize=(8, 4))
    plt.imshow(table.detach().cpu().numpy(), aspect="auto", cmap="viridis")
    plt.colorbar(label="value")
    plt.xlabel("channel")
    plt.ylabel("position")
    plt.title("Sinusoidal positional encoding")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def plot_rope_frequencies(cos: torch.Tensor, path: str | Path | None = None) -> Path:
    """Save representative RoPE cosine frequencies."""

    out = _prepare_path(path, "rope_frequencies.png")
    arr = cos.detach().cpu().numpy()
    plt.figure(figsize=(8, 4))
    for idx in np.linspace(0, arr.shape[1] - 1, min(6, arr.shape[1]), dtype=int):
        plt.plot(arr[:, idx], label=f"freq {idx}")
    plt.xlabel("position")
    plt.ylabel("cos phase")
    plt.title("RoPE frequency bands")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def plot_rope_rotation_circle(path: str | Path | None = None) -> Path:
    """Save a 2D RoPE rotation circle."""

    out = _prepare_path(path, "rope_rotation_circle.png")
    theta = np.linspace(0, 2 * np.pi, 256)
    plt.figure(figsize=(5, 5))
    plt.plot(np.cos(theta), np.sin(theta), color="black")
    for angle in [0.0, 0.7, 1.4, 2.1]:
        plt.arrow(0, 0, np.cos(angle), np.sin(angle), head_width=0.04, length_includes_head=True)
        plt.text(np.cos(angle) * 1.08, np.sin(angle) * 1.08, f"{angle:.1f}")
    plt.axis("equal")
    plt.xlabel("even channel")
    plt.ylabel("odd channel")
    plt.title("RoPE as rotation")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def plot_alibi_bias(bias: torch.Tensor, path: str | Path | None = None) -> Path:
    """Save ALiBi bias matrices for the first few heads."""

    out = _prepare_path(path, "alibi_bias.png")
    arr = bias.detach().cpu().numpy()[0]
    n_heads = min(4, arr.shape[0])
    fig, axes = plt.subplots(1, n_heads, figsize=(4 * n_heads, 3), squeeze=False)
    for idx in range(n_heads):
        ax = axes[0, idx]
        im = ax.imshow(arr[idx], aspect="auto", cmap="magma")
        ax.set_title(f"head {idx}")
        ax.set_xlabel("key")
        ax.set_ylabel("query")
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def plot_position_mapping(
    original: torch.Tensor, mapped: torch.Tensor, path: str | Path | None = None
) -> Path:
    """Save original-to-scaled position mapping."""

    out = _prepare_path(path, "position_mapping.png")
    plt.figure(figsize=(6, 4))
    plt.plot(original.detach().cpu().numpy(), mapped.detach().cpu().numpy())
    plt.xlabel("original position")
    plt.ylabel("mapped position")
    plt.title("Position scaling map")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def plot_length_extrapolation_results(
    results: dict[str, dict[str, float]], path: str | Path | None = None
) -> Path:
    """Save a small comparison plot from ``{method: {length: score}}`` results."""

    out = _prepare_path(path, "length_extrapolation_results.png")
    plt.figure(figsize=(7, 4))
    for method, values in results.items():
        xs = [int(length) for length in values.keys()]
        ys = [float(score) for score in values.values()]
        plt.plot(xs, ys, marker="o", label=method)
    plt.xlabel("evaluation length")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.05)
    plt.title("Length extrapolation toy task")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return out
