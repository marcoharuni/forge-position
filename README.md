# forge-position

Companion repository for Marco Haruni's technical book:

**Positional Encoding in Modern LLMs**  
*From Sinusoids and RoPE to Long-Context Scaling*

`forge-position` is a clean PyTorch laboratory for learning positional encoding
from first principles. It includes absolute learned embeddings, sinusoidal
encodings, RoPE, partial RoPE, interleaved and non-interleaved RoPE, ALiBi,
NoPE, hybrid RoPE/NoPE attention, RoPE scaling utilities, KV-cache position
handling, failure-mode tests, examples, notebooks, and documentation.

It is not a benchmark suite, a production inference engine, or a claim that any
included method is state of the art. The goal is educational clarity.

## Installation

This project uses uv as the primary workflow.

```bash
uv sync
```

## Quickstart

```bash
uv run pytest
uv run python examples/01_sinusoidal_visualization.py
uv run python examples/02_rope_geometry.py
uv run python examples/05_kv_cache_positions.py
uv run python examples/07_compare_methods.py
```

Generated figures are saved under `docs/figures/`.

## Methods Included

- learned absolute positional embeddings
- sinusoidal positional encodings
- RoPE
- partial RoPE through `rotary_dim`
- interleaved and non-interleaved RoPE pairing styles
- ALiBi
- NoPE
- hybrid RoPE/NoPE layer patterns
- linear, NTK-aware, YaRN-like, and LongRoPE-like scaling helpers
- KV-cache position ID utilities

YaRN and LongRoPE-style helpers are marked as educational approximations, not
drop-in reproductions of paper or vendor implementations.

## Examples

```bash
uv run python examples/01_sinusoidal_visualization.py
uv run python examples/02_rope_geometry.py
uv run python examples/03_rope_attention_scores.py
uv run python examples/04_alibi_bias.py
uv run python examples/05_kv_cache_positions.py
uv run python examples/06_length_extrapolation.py
uv run python examples/07_compare_methods.py
uv run python examples/08_rope_intuition_figures.py
```

The synthetic training examples are deliberately small and CPU-friendly. Their
numbers are sanity-check outputs, not research benchmarks.

## Tests

```bash
uv run pytest
```

Tests cover shapes, deterministic sinusoidal values, RoPE norm preservation,
RoPE relative-shift behavior, `position_ids`, partial rotation, ALiBi bias
monotonicity, KV-cache decode offsets, scaled cache NaN checks, and tiny model
CPU forward/generation paths.

## Notebooks

Lightweight notebooks live in `notebooks/`. See `docs/colab.md` for Colab
setup notes.

## Reading With The Book

Use the source files as reference implementations while reading the chapters:

- `sinusoidal.py` for the original Transformer formula
- `rope.py` for Q/K rotation geometry
- `alibi.py` for attention-bias methods
- `kv_cache.py` and `docs/failure_modes.md` for real decode bugs
- `tiny_transformer.py` for small controlled experiments

## Research Honesty

The docs cite source papers and official docs checked during repository
construction. Recent long-context methods evolve quickly, so uncertain
2025-2026 methods are flagged for human verification in
`docs/research_notes.md`. No tests require internet access or a GPU.
