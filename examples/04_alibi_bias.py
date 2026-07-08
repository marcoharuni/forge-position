"""Visualize ALiBi slopes and recency bias."""

from forge_position import build_alibi_bias, get_alibi_slopes, plot_alibi_bias


def main() -> None:
    slopes = get_alibi_slopes(6)
    bias = build_alibi_bias(n_heads=6, query_len=32, key_len=32)
    path = plot_alibi_bias(bias)
    print("ALiBi slopes:", [round(v, 6) for v in slopes.tolist()])
    print("ALiBi adds a negative distance-proportional bias to attention scores.")
    print(f"Saved bias plot to {path}")


if __name__ == "__main__":
    main()
