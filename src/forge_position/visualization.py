"""Matplotlib visualization helpers for positional encoding labs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import torch


DEFAULT_FIGURE_DIR = Path("docs/figures")


def _prepare_path(path: str | Path | None, filename: str) -> Path:
    out = Path(path) if path is not None else DEFAULT_FIGURE_DIR / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _clean_axes(ax: plt.Axes, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#d9482f",
    width: float = 3.0,
    mutation_scale: float = 18.0,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=width,
            color=color,
        )
    )


def _rounded_label(
    ax: plt.Axes,
    xy: tuple[float, float],
    text: str,
    *,
    color: str,
    width: float = 1.0,
    height: float = 0.34,
    fontsize: int = 12,
) -> None:
    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.04,rounding_size=0.06",
            facecolor=color,
            edgecolor="#202020",
            linewidth=1.0,
        )
    )
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize)


def _clock(
    ax: plt.Axes,
    center: tuple[float, float],
    radius: float,
    *,
    angle: float,
    label: str,
    color: str = "#d9482f",
) -> None:
    ax.add_patch(Circle(center, radius, fill=False, edgecolor="#7a858f", linewidth=1.6))
    end = (center[0] + radius * 0.75 * np.cos(angle), center[1] + radius * 0.75 * np.sin(angle))
    ax.plot(center[0], center[1], marker="o", color="#7a858f", markersize=6)
    _arrow(ax, center, end, color=color, width=2.4, mutation_scale=16)
    ax.text(center[0], center[1] - radius - 0.32, label, ha="center", va="top", fontsize=13)


def plot_attention_order_blindness(path: str | Path | None = None) -> Path:
    """Save an original diagram showing why attention needs position signals."""

    out = _prepare_path(path, "attention_order_blindness.png")
    fig, ax = plt.subplots(figsize=(10, 5))
    _clean_axes(ax, (0, 10), (0, 5))
    ax.text(5, 4.55, "A model blind to order", ha="center", fontsize=24, fontweight="bold")

    colors = {"dog": "#dcecff", "bites": "#dff0df", "man": "#f6e8d6"}
    for idx, token in enumerate(["dog", "bites", "man"]):
        _rounded_label(ax, (1.3 + idx * 0.52, 3.6 - idx * 0.5), token, color=colors[token])
    for idx, token in enumerate(["man", "bites", "dog"]):
        _rounded_label(ax, (1.3 + idx * 0.52, 1.7 - idx * 0.5), token, color=colors[token])

    for y in [3.2, 1.3]:
        _arrow(ax, (3.0, y), (4.0, y), color="#111111", width=2.0, mutation_scale=18)
        ax.add_patch(
            FancyBboxPatch(
                (4.1, y - 0.35),
                2.0,
                0.7,
                boxstyle="round,pad=0.08,rounding_size=0.12",
                facecolor="#f4f4f2",
                edgecolor="#111111",
                linewidth=1.2,
            )
        )
        ax.text(5.1, y, "ATTENTION", ha="center", va="center", fontsize=15, fontweight="bold")

    _arrow(ax, (6.1, 3.2), (8.1, 2.65), color="#111111", width=2.0, mutation_scale=18)
    _arrow(ax, (6.1, 1.3), (8.1, 1.85), color="#111111", width=2.0, mutation_scale=18)
    ax.text(
        8.1,
        2.25,
        "same bag of words\nwithout position",
        ha="center",
        va="center",
        fontsize=14,
        color="#c8321a",
        fontweight="bold",
    )
    ax.text(
        5,
        0.45,
        "Self-attention is permutation-equivariant: order must be injected by a positional signal.",
        ha="center",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_add_vs_rotate(path: str | Path | None = None) -> Path:
    """Save an original diagram contrasting additive embeddings with RoPE."""

    out = _prepare_path(path, "add_vs_rotate.png")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax in axes:
        _clean_axes(ax, (-1.4, 1.6), (-1.3, 1.4))

    axes[0].set_title("Add a position vector", fontsize=17, fontweight="bold")
    origin = (-1.0, -0.8)
    token = (0.15, -0.1)
    pos = (0.55, 0.75)
    result = (1.15, 0.65)
    _arrow(axes[0], origin, token, color="#5c84b1", width=3)
    _arrow(axes[0], token, result, color="#9db5d1", width=3)
    _arrow(axes[0], origin, result, color="#7b9cc3", width=3)
    _arrow(axes[0], origin, pos, color="#9db5d1", width=3)
    axes[0].text(-0.72, -0.2, "token", fontsize=11)
    axes[0].text(0.48, 0.93, "position", fontsize=11)
    axes[0].text(0.35, -0.85, "token + position", fontsize=11)

    axes[1].set_title("Rotate Q and K", fontsize=17, fontweight="bold")
    axes[1].add_patch(Circle((0, 0), 1.0, fill=False, edgecolor="#4f5963", linewidth=1.6))
    angle = 0.78
    _arrow(axes[1], (0, 0), (0.88, 0), color="#e8513a", width=3)
    _arrow(axes[1], (0, 0), (0.88 * np.cos(angle), 0.88 * np.sin(angle)), color="#e8513a", width=3)
    axes[1].add_patch(Arc((0, 0), 1.3, 1.3, theta1=0, theta2=np.degrees(angle), color="#e8513a", linewidth=2))
    axes[1].text(0.44, -0.18, "initial", fontsize=11)
    axes[1].text(0.12, 0.88, "rotated", fontsize=11)
    axes[1].text(0.78, 0.42, "angle grows\nwith position", fontsize=11)

    fig.text(0.5, 0.05, "Absolute embeddings add a vector; RoPE writes position as phase.", ha="center", fontsize=13)
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    plt.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_rope_position_angles(path: str | Path | None = None) -> Path:
    """Save an original diagram showing position as rotation angle."""

    out = _prepare_path(path, "rope_position_angles.png")
    fig, ax = plt.subplots(figsize=(10, 4))
    _clean_axes(ax, (-0.5, 10.5), (-1.9, 2.0))
    ax.text(5, 1.65, "Position is an angle", ha="center", fontsize=22, fontweight="bold")
    theta = 0.62
    xs = [1.2, 3.7, 6.2, 8.7]
    for pos, x in enumerate(xs):
        _clock(ax, (x, 0.25), 0.85, angle=pos * theta, label=f"position {pos}")
        ax.text(x, -1.3, f"{pos} x theta", ha="center", fontsize=10)
    ax.text(5, -1.65, "rotate each pair by m x theta, where m is the token position", ha="center", fontsize=12)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_rope_relative_distance(path: str | Path | None = None) -> Path:
    """Save an original diagram showing why RoPE dot products encode distance."""

    out = _prepare_path(path, "rope_relative_distance.png")
    fig, ax = plt.subplots(figsize=(7, 6))
    _clean_axes(ax, (-2.2, 2.2), (-2.3, 2.2))
    ax.set_title("Why rotation means relative distance", fontsize=18, fontweight="bold")
    ax.add_patch(Circle((0, 0), 1.45, fill=False, edgecolor="#6f7378", linewidth=2))
    q_angle = 0.85
    k_angle = 2.55
    q_end = (1.25 * np.cos(q_angle), 1.25 * np.sin(q_angle))
    k_end = (1.25 * np.cos(k_angle), 1.25 * np.sin(k_angle))
    _arrow(ax, (0, 0), q_end, color="#d9482f", width=3)
    _arrow(ax, (0, 0), k_end, color="#d9482f", width=3)
    ax.add_patch(Arc((0, 0), 1.0, 1.0, theta1=np.degrees(q_angle), theta2=np.degrees(k_angle), color="#d9482f", linewidth=2))
    ax.text(q_end[0] + 0.05, q_end[1], "query\nposition m", fontsize=10)
    ax.text(k_end[0] - 0.75, k_end[1], "key\nposition n", fontsize=10)
    ax.text(0, -0.05, "(m - n) x theta", ha="center", fontsize=11)
    ax.text(0, -1.8, "Shift both positions together: the angle gap stays the same.", ha="center", fontsize=12)
    ax.text(0, -2.08, "Dot product keeps the relative offset, not the absolute index.", ha="center", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_rope_multi_frequency_clock(path: str | Path | None = None) -> Path:
    """Save an original multi-frequency RoPE clock-hand diagram."""

    out = _prepare_path(path, "rope_multi_frequency_clock.png")
    fig, ax = plt.subplots(figsize=(8, 6))
    _clean_axes(ax, (-2.3, 2.8), (-2.1, 2.2))
    ax.set_title("A clock with many hands", fontsize=20, fontweight="bold")
    center = (0, 0)
    radius = 1.55
    ax.add_patch(Circle(center, radius, fill=False, edgecolor="#777777", linewidth=2))
    for tick in range(48):
        angle = 2 * np.pi * tick / 48
        inner = radius * (0.9 if tick % 4 else 0.84)
        outer = radius * 0.98
        ax.plot(
            [inner * np.cos(angle), outer * np.cos(angle)],
            [inner * np.sin(angle), outer * np.sin(angle)],
            color="#999999",
            linewidth=1.0,
        )
    hands = [
        ("fast", 1.25, "#e85035", 1.25),
        ("medium", 0.0, "#c49a32", 1.15),
        ("slow", -2.1, "#6699c2", 1.2),
    ]
    for label, angle, color, length in hands:
        _arrow(ax, center, (length * np.cos(angle), length * np.sin(angle)), color=color, width=4, mutation_scale=18)
        ax.text((length + 0.18) * np.cos(angle), (length + 0.18) * np.sin(angle), label.upper(), fontsize=11)
    ax.plot(0, 0, marker="o", color="#666666", markersize=12)
    ax.text(2.1, 0.8, "fast pairs capture\nnearby position", fontsize=11)
    ax.text(-2.2, -0.75, "slow pairs span\nlong distance", fontsize=11)
    ax.text(0, -1.9, "RoPE splits each head into 2D pairs rotating at different speeds.", ha="center", fontsize=12)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close(fig)
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
