# Colab

Each notebook is intentionally small and Colab-friendly.

The notebooks use this setup pattern:

```bash
!curl -LsSf https://astral.sh/uv/install.sh | sh
!git clone https://github.com/marcoharuni/forge-position.git
%cd forge-position
!/root/.local/bin/uv sync
```

If you upload the repository as a zip instead of cloning it, run notebook cells
from the repository root so imports and generated figure paths resolve
correctly.

Notebooks:

- `notebooks/01_sinusoidal_vs_rope.ipynb`
- `notebooks/02_rope_geometry.ipynb`
- `notebooks/03_alibi_bias_visualization.ipynb`
- `notebooks/04_kv_cache_position_bugs.ipynb`
- `notebooks/05_length_extrapolation_lab.ipynb`

Each notebook includes an "Open in Colab" badge pointing at
`marcoharuni/forge-position` on GitHub.

The RoPE intuition figures are intentionally concentrated in
`02_rope_geometry.ipynb`. The other notebooks avoid repeating the same visuals
so each lab remains short and focused.
