import torch

from forge_position import TinyDecoderOnlyTransformer, TinyTransformerConfig


def test_tiny_transformer_forward_logits_shape() -> None:
    model = TinyDecoderOnlyTransformer(
        TinyTransformerConfig(vocab_size=20, d_model=16, n_heads=4, n_layers=2, max_seq_len=16)
    )
    tokens = torch.randint(0, 20, (2, 6))
    logits, loss = model(tokens, targets=tokens)
    assert logits.shape == (2, 6, 20)
    assert loss is not None


def test_tiny_transformer_generate_runs_on_cpu() -> None:
    model = TinyDecoderOnlyTransformer(
        TinyTransformerConfig(
            vocab_size=20,
            d_model=16,
            n_heads=4,
            n_layers=2,
            max_seq_len=16,
            position_method="alibi",
        )
    )
    tokens = torch.randint(0, 20, (1, 4))
    out = model.generate(tokens, max_new_tokens=3)
    assert out.shape == (1, 7)
