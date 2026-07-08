"""Visualize the original Transformer sinusoidal positional encoding."""

from forge_position import build_sinusoidal_table, plot_sinusoidal_table


def main() -> None:
    table = build_sinusoidal_table(max_seq_len=128, d_model=64)
    path = plot_sinusoidal_table(table)
    print("Sinusoidal encoding uses sin(pos / base^(2i/d)) on even channels")
    print("and cos(pos / base^(2i/d)) on odd channels.")
    print(f"Saved heatmap to {path}")


if __name__ == "__main__":
    main()
