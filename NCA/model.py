from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class NCA(nn.Module):
    """Neural Cellular Automata with one shared local update rule."""

    def __init__(
        self,
        state_channels: int = 2,
        model_width: int = 64,
        kernel_size: int = 3,
        update_prob: float = 0.5,
        use_alive_mask: bool = False,
        alive_threshold: float = 0.1,
        primary_channel: int = 0,
    ) -> None:
        super().__init__()
        if state_channels < 1:
            raise ValueError("state_channels must be >= 1.")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        if not 0.0 < update_prob <= 1.0:
            raise ValueError("update_prob must be in (0, 1].")
        if not 0 <= primary_channel < state_channels:
            raise ValueError("primary_channel must index one of the state channels.")

        self.state_channels = state_channels
        self.model_width = model_width
        self.kernel_size = kernel_size
        self.update_prob = float(update_prob)
        self.use_alive_mask = bool(use_alive_mask)
        self.alive_threshold = float(alive_threshold)
        self.primary_channel = int(primary_channel)

        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(state_channels, model_width, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(model_width, state_channels, kernel_size=kernel_size, padding=padding)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def _sample_update_mask(self, x: torch.Tensor, stochastic: bool) -> torch.Tensor:
        if not stochastic:
            return torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]), dtype=x.dtype, device=x.device)
        probs = torch.full((x.shape[0], 1, x.shape[2], x.shape[3]), self.update_prob, dtype=x.dtype, device=x.device)
        return torch.bernoulli(probs)

    def _alive_mask(self, visible: torch.Tensor) -> torch.Tensor:
        if visible.ndim != 4 or visible.shape[1] != 1:
            raise ValueError("alive mask expects visible channel shaped [B, 1, H, W].")
        neighborhood = F.max_pool2d(visible, kernel_size=3, stride=1, padding=1)
        return (neighborhood > self.alive_threshold).to(visible.dtype)

    def forward(self, x: torch.Tensor, stochastic: bool = True) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input [B, F, H, W], got {x.shape}.")
        if x.shape[1] != self.state_channels:
            raise ValueError(
                f"Expected {self.state_channels} state channels, got {x.shape[1]}."
            )

        update_mask = self._sample_update_mask(x, stochastic=stochastic)
        pre_alive = self._alive_mask(x[:, self.primary_channel:self.primary_channel + 1]) if self.use_alive_mask else None
        if pre_alive is not None:
            update_mask = update_mask * pre_alive

        delta = self.conv2(torch.relu(self.conv1(x)))
        next_state = x + update_mask * delta

        if self.use_alive_mask:
            post_alive = self._alive_mask(next_state[:, self.primary_channel:self.primary_channel + 1])
            next_state = next_state * post_alive

        return next_state
