from __future__ import annotations

from pathlib import Path
import random
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preferred: Optional[str] = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_relative_path(path: str | Path, root: str | Path) -> Path:
    candidate = Path(path)
    root_path = Path(root)
    if candidate.is_absolute():
        try:
            candidate.relative_to(root_path)
        except ValueError as exc:
            raise ValueError(f"Path {candidate} must stay relative to {root_path}.") from exc
        return candidate
    return root_path / candidate


def build_initial_state(
    visible: torch.Tensor,
    hidden_channels: int = 1,
    hidden_init: str = "zeros",
) -> torch.Tensor:
    if visible.ndim != 4:
        raise ValueError(f"Expected visible input [B, C, H, W], got {visible.shape}.")
    if hidden_channels < 0:
        raise ValueError("hidden_channels must be >= 0.")
    if hidden_init != "zeros":
        raise ValueError("Only hidden_init='zeros' is supported in v1.")

    if hidden_channels == 0:
        return visible

    hidden = torch.zeros(
        (visible.shape[0], hidden_channels, visible.shape[2], visible.shape[3]),
        dtype=visible.dtype,
        device=visible.device,
    )
    return torch.cat([visible, hidden], dim=1)


def rollout_model(
    model: torch.nn.Module,
    initial_state: torch.Tensor,
    steps: int,
    stochastic: bool = True,
) -> torch.Tensor:
    states: List[torch.Tensor] = []
    current = initial_state
    for _ in range(int(steps)):
        current = model(current, stochastic=stochastic)
        states.append(current)
    if not states:
        raise ValueError("steps must be >= 1.")
    return torch.stack(states, dim=0)


def mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((prediction - target) ** 2)


def visible_to_probability(
    visible: torch.Tensor,
    supervised_loss: str = "mse",
) -> torch.Tensor:
    if supervised_loss == "mse":
        return torch.clamp(visible, 0.0, 1.0)
    if supervised_loss in {"bce", "bce_dice"}:
        return torch.sigmoid(visible)
    raise ValueError("supervised_loss must be one of 'mse', 'bce', 'bce_dice'.")


def population_mass_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    pred_mass = prediction.sum(dim=(-2, -1))
    target_mass = target.sum(dim=(-2, -1))
    return torch.mean(torch.abs(pred_mass - target_mass) / (target_mass.abs() + eps))


def compute_binary_mask_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-8,
) -> Dict[str, float]:
    pred_mask = prediction >= threshold
    target_mask = target >= threshold

    intersection = (pred_mask & target_mask).sum(dim=(-3, -2, -1)).float()
    pred_area = pred_mask.sum(dim=(-3, -2, -1)).float()
    target_area = target_mask.sum(dim=(-3, -2, -1)).float()
    union = (pred_mask | target_mask).sum(dim=(-3, -2, -1)).float()

    dice = (2.0 * intersection + eps) / (pred_area + target_area + eps)
    iou = (intersection + eps) / (union + eps)
    precision = (intersection + eps) / (pred_area + eps)
    recall = (intersection + eps) / (target_area + eps)

    return {
        "dice": float(dice.mean().item()),
        "iou": float(iou.mean().item()),
        "precision": float(precision.mean().item()),
        "recall": float(recall.mean().item()),
    }


def compute_rollout_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    visible_channel: int = 0,
    supervised_loss: str = "mse",
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    predictions: [S, B, F, H, W]
    targets: [B, S, C_obs, H, W]
    """
    target_steps = targets.permute(1, 0, 2, 3, 4)
    if not 0 <= visible_channel < target_steps.shape[2]:
        raise ValueError(f"visible_channel={visible_channel} is out of range for targets with {target_steps.shape[2]} channels.")
    visible_predictions = visible_to_probability(
        predictions[:, :, visible_channel:visible_channel + 1],
        supervised_loss=supervised_loss,
    )
    visible_targets = target_steps[:, :, visible_channel:visible_channel + 1]
    per_step_mse = torch.mean((visible_predictions - visible_targets) ** 2, dim=(1, 2, 3, 4))
    per_step_mass_error = []
    for step in range(visible_predictions.shape[0]):
        per_step_mass_error.append(population_mass_error(visible_predictions[step], visible_targets[step]))
    mass_curve = torch.stack(per_step_mass_error)
    binary_metrics = compute_binary_mask_metrics(
        visible_predictions.reshape(-1, 1, visible_predictions.shape[-2], visible_predictions.shape[-1]),
        visible_targets.reshape(-1, 1, visible_targets.shape[-2], visible_targets.shape[-1]),
        threshold=threshold,
    )

    return {
        "rollout_mse": float(per_step_mse.mean().item()),
        "one_step_mse": float(per_step_mse[0].item()),
        "population_mass_error": float(mass_curve.mean().item()),
        **binary_metrics,
    }


def compute_stochastic_metrics(
    sampled_predictions: torch.Tensor,
    targets: torch.Tensor,
    visible_channel: int = 0,
    eps: float = 1e-8,
    supervised_loss: str = "mse",
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    sampled_predictions: [K, S, B, F, H, W]
    targets: [B, S, C_obs, H, W]
    """
    if not 0 <= visible_channel < targets.shape[2]:
        raise ValueError(f"visible_channel={visible_channel} is out of range for targets with {targets.shape[2]} channels.")
    visible = visible_to_probability(
        sampled_predictions[:, :, :, visible_channel:visible_channel + 1],
        supervised_loss=supervised_loss,
    )
    target_steps = targets.permute(1, 0, 2, 3, 4)[:, :, visible_channel:visible_channel + 1].unsqueeze(0)
    mse_per_rollout = torch.mean((visible - target_steps) ** 2, dim=(1, 2, 3, 4, 5))
    ensemble_mean = visible.mean(dim=0)
    ensemble_mean_mse = torch.mean((ensemble_mean - target_steps[0]) ** 2)
    pixelwise_std = torch.std(visible, dim=0, unbiased=False)

    rollout_mass = visible.sum(dim=(-2, -1))
    mass_std = torch.std(rollout_mass, dim=0, unbiased=False)
    binary_metrics = compute_binary_mask_metrics(
        ensemble_mean.reshape(-1, 1, ensemble_mean.shape[-2], ensemble_mean.shape[-1]),
        target_steps[0].reshape(-1, 1, target_steps.shape[-2], target_steps.shape[-1]),
        threshold=threshold,
    )

    return {
        "expected_mse": float(mse_per_rollout.mean().item()),
        "ensemble_mean_mse": float(ensemble_mean_mse.item()),
        "pixelwise_std_mean": float(pixelwise_std.mean().item()),
        "mass_std": float(mass_std.mean().item()),
        "population_mass_error": float(
            population_mass_error(ensemble_mean, target_steps[0], eps=eps).item()
        ),
        **binary_metrics,
    }


def save_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    config: Dict,
    metrics: Optional[Dict[str, float]] = None,
) -> None:
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": None if optimizer is None else optimizer.state_dict(),
        "epoch": epoch,
        "config": config,
        "metrics": metrics or {},
    }
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Optional[str | torch.device] = None,
) -> Dict:
    checkpoint = torch.load(checkpoint_path, map_location=map_location or "cpu")
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def detach_metrics(metrics: Dict[str, torch.Tensor | float]) -> Dict[str, float]:
    detached: Dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            detached[key] = float(value.detach().cpu().item())
        else:
            detached[key] = float(value)
    return detached
