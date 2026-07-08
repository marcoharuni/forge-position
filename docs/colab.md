# Colab

Each notebook is intentionally small and can be adapted for Colab.

Recommended setup cell:

```bash
!curl -LsSf https://astral.sh/uv/install.sh | sh
!uv sync
```

If you upload the repository as a zip or clone it in Colab, run notebook cells
from the repository root so imports resolve through the uv-managed environment.

Notebooks:

- `notebooks/01_sinusoidal_vs_rope.ipynb`
- `notebooks/02_rope_geometry.ipynb`
- `notebooks/03_alibi_bias_visualization.ipynb`
- `notebooks/04_kv_cache_position_bugs.ipynb`
- `notebooks/05_length_extrapolation_lab.ipynb`

Each notebook includes an "Open in Colab" badge pointing at
`marcoharuni/forge-position` on GitHub.
