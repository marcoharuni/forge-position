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
