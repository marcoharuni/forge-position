import torch

from forge_position import SinusoidalPositionEncoding, build_sinusoidal_table


def test_sinusoidal_shape_and_first_values() -> None:
    table = build_sinusoidal_table(8, 5)
    assert table.shape == (8, 5)
    assert torch.allclose(table[0, 0::2], torch.zeros(3))
    assert torch.allclose(table[0, 1::2], torch.ones(2))


def test_sinusoidal_is_deterministic_and_parameter_free() -> None:
    a = build_sinusoidal_table(4, 6)
    b = build_sinusoidal_table(4, 6)
    assert torch.allclose(a, b)
    module = SinusoidalPositionEncoding(16, 6)
    assert list(module.parameters()) == []
    x = torch.zeros(2, 4, 6)
    y = module(x)
    assert y.shape == x.shape
    assert torch.allclose(y[0], y[1])
