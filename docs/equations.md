# Equations

## Sinusoidal Encoding

For position `pos`, channel pair index `i`, and width `d_model`:

```text
PE(pos, 2i)     = sin(pos / 10000^(2i / d_model))
PE(pos, 2i + 1) = cos(pos / 10000^(2i / d_model))
```

The encoding is added to token embeddings before the Transformer blocks.

## RoPE Rotation Matrix

For a two-dimensional feature pair and angle `theta_m = m * omega_i`:

```text
R(theta_m) = [[cos(theta_m), -sin(theta_m)],
              [sin(theta_m),  cos(theta_m)]]
```

RoPE applies this rotation to query and key pairs. Values are not rotated.

More generally, attention uses query-key dot products. For token positions `m`
and `n`, RoPE aims for the score to depend on relative distance:

```text
f_q(x_m, m)^T f_k(x_n, n) = g(x_m, x_n, m - n)
```

One block-diagonal construction applies a separate 2 x 2 rotation to each pair
inside a head:

```text
f_W(x_m, m) =
blockdiag(R(m theta_0), R(m theta_1), ..., R(m theta_{d/2-1})) W x_m
```

with

```text
theta_i = base^(-2i / d),  i = 0, ..., d/2 - 1
```

This repository defaults to `base = 10000`, matching the original RoPE
presentation and many open implementations. Some modern systems use different
bases, such as larger values for long-context settings; changing the base
changes every phase and should be treated as part of the experiment or
checkpoint configuration.

## RoPE Complex View

Treat a pair `(x_even, x_odd)` as a complex number `z`.

```text
RoPE(z, m, i) = z * exp(j * m * omega_i)
```

Then query-key products naturally include relative phase:

```text
<R_m q, R_n k> depends on n - m
```

This is why shifting every query and key position by the same offset preserves
the relative attention-score structure.

## ALiBi Bias

For head slope `s_h`, query position `i`, and key position `j`:

```text
bias[h, i, j] = -s_h * max(i - j, 0)
```

Older visible keys receive a more negative logit bias. Future keys are still
handled by the causal mask.

## Position Interpolation

For a scale factor `a > 1`:

```text
mapped_position = position / a
```

This maps a longer evaluation context into a shorter trained position range.

## NTK-Aware Scaling Idea

NTK-aware RoPE scaling changes the RoPE base as the sequence length grows. A
common dynamic approximation is:

```text
base' = base * ((a * L / L_train) - (a - 1))^(dim / (dim - 2))
```

where `L` is the requested sequence length and `a` is a scaling factor.

## YaRN and LongRoPE Caveat

The code in `rope_scaling.py` uses educational approximations for YaRN-like and
LongRoPE-like mappings. The papers include additional implementation details,
search procedures, ramps, or fine-tuning recipes. Do not treat these functions
as exact reproductions.

`rope_scaling.RotaryEmbedding` also includes a YaRN/NTK-by-parts style cache
module for study. Its high-level idea is:

```text
r(i) = L_train * theta_i / (2 pi)
```

where large `r(i)` corresponds to fast clocks that cycle many times over the
training window, and small `r(i)` corresponds to slow clocks. Two thresholds,
`alpha` and `beta`, define a transition region where unscaled frequencies and
linearly interpolated frequencies are blended. The implementation is designed
for clarity and tests, not production parity.
