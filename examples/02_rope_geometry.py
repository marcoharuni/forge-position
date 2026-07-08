"""Show RoPE as a position-dependent 2D rotation."""

from forge_position import build_rope_cache, plot_rope_frequencies, plot_rope_rotation_circle


def main() -> None:
    cos, _ = build_rope_cache(seq_len=128, dim=32)
    circle_path = plot_rope_rotation_circle()
    freq_path = plot_rope_frequencies(cos)
    print("RoPE rotates each 2D feature pair by an angle position * frequency.")
    print("The dot product of two rotated vectors depends on their relative offset.")
    print("Key equation: R_m q dot R_n k = q dot R_(n-m) k")
    print(f"Saved rotation circle to {circle_path}")
    print(f"Saved frequency plot to {freq_path}")


if __name__ == "__main__":
    main()
