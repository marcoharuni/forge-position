# Failure Modes

## Wrong `position_ids`

RoPE and learned absolute embeddings both depend on correct position IDs. During
decode, the next token should use `past_len`, not `0`.

## Applying RoPE to V

RoPE belongs on Q and K. Rotating V changes the content vectors that are mixed
after the softmax and breaks the intended derivation.

## Applying RoPE Before Q/K Projection

RoPE is normally applied after projecting hidden states into query and key head
spaces. Applying it before projection changes the operation and can mix phases
across heads.

## Interleaved vs Non-Interleaved Mismatch

Some implementations pair adjacent channels. Others pair the first half with
the second half. A checkpoint trained with one convention should not be served
with the other.

## Wrong `rotary_dim`

Partial RoPE rotates only a prefix of each head. Using the wrong prefix size can
silently degrade outputs while leaving tensor shapes valid.

## Left Padding With RoPE

Left-padded batches need position IDs that reflect real token positions. A
batch item with padding should not assign the first non-padding token a phase
that conflicts with the intended sequence layout.

## Decode Offset Bug

Prefill positions are usually `0..T-1`. The first decoded token must use `T`.
Resetting to `0` makes new keys reuse old phases.

## Changing RoPE Base at Inference

Changing the RoPE base changes every phase. Scaling methods should be selected
deliberately and documented with the checkpoint or experiment.

## Evaluating Beyond Training Length Without Scaling

Raw RoPE extrapolation can enter phase regions never trained. This may work on
some tasks and fail badly on others.

## Forgetting to Update Cache Positions

The cache is not just K and V. For RoPE debugging, track the position IDs used
to create cached keys.

## Mixing ALiBi Mask Signs

ALiBi is added to attention logits. Older keys should receive more negative
biases. If distant keys become more positive, the sign is reversed.

## Applying RoPE Twice

If cached keys are already rotated, do not rotate them again after retrieval.
Store either raw keys with positions or rotated keys, and be consistent.
