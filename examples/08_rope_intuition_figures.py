"""Generate original RoPE intuition figures for the book companion repo."""

from forge_position import (
    plot_add_vs_rotate,
    plot_attention_order_blindness,
    plot_rope_multi_frequency_clock,
    plot_rope_position_angles,
    plot_rope_relative_distance,
)


def main() -> None:
    paths = [
        plot_attention_order_blindness(),
        plot_add_vs_rotate(),
        plot_rope_position_angles(),
        plot_rope_relative_distance(),
        plot_rope_multi_frequency_clock(),
    ]
    print("Generated original intuition figures:")
    for path in paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
