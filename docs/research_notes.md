# Research Notes

These notes support Marco Haruni's *Positional Encoding in Modern LLMs*. They
summarize implementation-relevant ideas, not every experimental result from the
literature.

## Sources Checked

- Vaswani et al., "Attention Is All You Need", arXiv:1706.03762.
- Dai et al., "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context", arXiv:1901.02860.
- Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer", arXiv:1910.10683.
- Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", arXiv:2104.09864.
- Press et al., "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation", arXiv:2108.12409.
- Haviv et al., "Transformer Language Models without Positional Encodings Still Learn Positional Information", arXiv:2203.16634.
- Chen et al., "Extending Context Window of Large Language Models via Positional Interpolation", arXiv:2306.15595.
- Peng et al., "YaRN: Efficient Context Window Extension of Large Language Models", arXiv:2309.00071.
- Ding et al., "LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens", arXiv:2402.13753.
- Yang et al., "Rope to Nope and Back Again: A New Hybrid Attention Strategy", arXiv:2501.18795.
- Shang et al., "LongRoPE2: Near-Lossless LLM Context Window Scaling", arXiv:2502.20082.
- Lin et al., "Forgetting Transformer: Softmax Attention with a Forget Gate", arXiv:2503.02130.
- PyTorch official `torch.nn.functional.scaled_dot_product_attention` documentation.
- Astral uv official project, lock, sync, and run documentation.

## Method Summaries

Learned absolute embeddings assign a trainable vector to each absolute index.
They are simple and fast, but cannot naturally use positions outside the learned
embedding table.

Sinusoidal encodings add fixed sine and cosine features to token embeddings.
The original Transformer used them because relative shifts can be represented
linearly in the encoding space.

Transformer-XL introduced recurrence across segments and a relative positional
encoding scheme so attention can reason about tokens beyond a fixed segment
without confusing positions from different segments.

T5 uses learned relative position bias buckets. The positional signal enters as
an attention-logit bias rather than as a vector added to token embeddings.

RoPE rotates query and key feature pairs by position-dependent phases. It
injects absolute phase while making the query-key dot product depend naturally
on relative offsets. Values should not be rotated.

ALiBi adds a per-head linear distance penalty directly to attention scores. It
has no positional embedding table and gives each head a different recency bias.

NoPE removes explicit positional encodings. In causal language models, the
causal mask and depth can still leak or induce positional information, but this
is a model behavior to study rather than a guarantee.

Hybrid RoPE/NoPE uses explicit RoPE in some layers and no positional encoding in
others. Recent work reports complementary attention patterns, but recipes are
still young and architecture-dependent.

Position Interpolation maps long-context positions back into the trained range
by dividing position indices by a scale factor. This avoids raw extrapolation
to unseen RoPE phases.

NTK-aware scaling changes the RoPE base as context length grows. Public model
implementations vary; this repository includes a common educational dynamic
base approximation.

YaRN and LongRoPE-style scaling are included as educational reference code. The
compact cache helpers are approximations; the `RotaryEmbedding` class adds a
more explicit YaRN/NTK-by-parts study path. Neither should be treated as a
drop-in reproduction of paper or vendor implementations.

Forgetting Transformer adds a forget gate to softmax attention and reports that
it can work without positional embeddings. It is documented here as related
research, not implemented as a full architecture.

## Comparison Table

| method | mechanism | strengths | weaknesses | extrapolation behavior | implementation risk |
| --- | --- | --- | --- | --- | --- |
| learned absolute | trainable position table added to embeddings | simple, familiar | fixed max position table | poor beyond table without resizing | low |
| sinusoidal | fixed sin/cos features | deterministic, no learned table | added signal can be weaker than attention-native methods | can evaluate beyond train length | low |
| Transformer-XL relative | relative terms plus segment recurrence | handles segment memory | more complex attention math | designed for longer dependencies | high |
| T5 relative bias | bucketed attention bias | compact and effective | bucket choices matter | graceful within bucket design | medium |
| RoPE | rotate Q/K by position phase | relative dot-product structure | phase mismatch bugs are common | raw extrapolation can fail | medium |
| partial RoPE | rotate only part of each head | preserves some unrotated capacity | rotary_dim is another knob | depends on model and task | medium |
| ALiBi | linear score penalty | no table, cheap long lengths | strong recency prior | often extrapolates well | low |
| NoPE | no explicit PE | simplest architecture | relies on implicit signals | uncertain, depth/task dependent | low |
| hybrid RoPE/NoPE | layer-wise mix | studies complementary behavior | young design space | promising but unsettled | medium |
| PI | divide positions by scale | stable long-context extension idea | compresses local resolution | extends RoPE range | medium |
| NTK-aware | change RoPE base | often practical | many variants | can extend usable phases | medium |
| YaRN | ramped/interpolated RoPE scaling | efficient extension reports | exact recipe is nuanced | strong with fine-tuning | high |
| LongRoPE | non-uniform interpolation/search | very long context reports | search and tuning heavy | strong in paper settings | high |
| LongRoPE2 | rescaling plus mixed context training | aims to preserve short context | 2025 method, implementation details matter | promising paper results | high |

## Needs Human Verification

The 2025-2026 landscape changes quickly. LongRoPE2, hybrid RoPE/NoPE recipes,
Forgetting Transformer variants, and RoPE limitation papers should be checked
against the latest paper revisions and official code before being cited as
production guidance. This repository deliberately avoids claiming production
parity for YaRN, LongRoPE, or LongRoPE2.

The term "iRoPE" was not added as an implemented method because no reliable,
canonical source was identified by that exact name during the search pass. If a
specific paper or implementation is intended, add it here before treating it as
part of the book's method taxonomy.
