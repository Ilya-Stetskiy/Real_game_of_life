import torch

from NCA.model import NCA
from NCA.utils import build_initial_state


def test_build_initial_state_adds_zero_hidden() -> None:
    visible = torch.ones(2, 1, 4, 5)
    state = build_initial_state(visible, hidden_channels=1, hidden_init="zeros")
    assert state.shape == (2, 2, 4, 5)
    assert torch.allclose(state[:, 0], visible[:, 0])
    assert torch.count_nonzero(state[:, 1]) == 0


def test_model_forward_keeps_shape_and_mask_shape() -> None:
    model = NCA(state_channels=2, model_width=8, kernel_size=3, update_prob=0.5)
    x = torch.randn(3, 2, 6, 7)
    mask = model._sample_update_mask(x, stochastic=True)
    output = model(x, stochastic=True)

    assert mask.shape == (3, 1, 6, 7)
    assert output.shape == x.shape


def test_alive_mask_uses_visible_channel_and_gates_hidden() -> None:
    model = NCA(state_channels=2, model_width=8, kernel_size=3, update_prob=1.0, use_alive_mask=True)
    x = torch.zeros(1, 2, 5, 5)
    x[:, 1] = 3.0

    output = model(x, stochastic=False)
    assert torch.count_nonzero(output) == 0

    x[:, 0, 2, 2] = 1.0
    output = model(x, stochastic=False)
    assert output[:, 0].sum() > 0
