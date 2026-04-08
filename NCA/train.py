from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from .dataset import VisibleChannelNormalizer, build_dataloaders
from .model import NCA
from .utils import (
    build_initial_state,
    compute_rollout_metrics,
    compute_stochastic_metrics,
    detach_metrics,
    get_device,
    load_checkpoint,
    rollout_model,
    save_checkpoint,
    set_seed,
    to_device,
)


def compute_rollout_loss(
    predicted_states: torch.Tensor,
    targets_visible: torch.Tensor,
    target_mask: torch.Tensor,
    loss_mode: str = "hybrid",
    lambda_intermediate: float = 0.5,
    lambda_hidden_l2: float = 1e-4,
    visible_channel: int = 0,
) -> Dict[str, torch.Tensor]:
    if loss_mode not in {"final_only", "intermediate", "hybrid"}:
        raise ValueError("loss_mode must be one of 'final_only', 'intermediate', 'hybrid'.")

    steps, batch_size = predicted_states.shape[:2]
    if targets_visible.shape[1] != steps:
        raise ValueError("predicted_states and targets_visible must share the same rollout length.")

    visible_predictions = predicted_states[:, :, visible_channel:visible_channel + 1]
    target_steps = targets_visible.permute(1, 0, 2, 3, 4)
    mask_steps = target_mask.permute(1, 0)

    per_step_losses: List[torch.Tensor] = []
    for step in range(steps):
        valid = mask_steps[step]
        if not torch.any(valid):
            continue
        diff = visible_predictions[step, valid] - target_steps[step, valid]
        per_step_losses.append(torch.mean(diff ** 2))

    if not per_step_losses:
        raise ValueError("No supervised rollout steps available for loss computation.")

    final_step_indices = target_mask.sum(dim=1) - 1
    batch_indices = torch.arange(batch_size, device=predicted_states.device)
    final_visible = visible_predictions[final_step_indices, batch_indices]
    final_targets = targets_visible[batch_indices, final_step_indices]
    final_loss = torch.mean((final_visible - final_targets) ** 2)

    if steps > 1:
        intermediate_terms = []
        for sample_index in range(batch_size):
            sample_last = int(final_step_indices[sample_index].item())
            if sample_last <= 0:
                continue
            diff = visible_predictions[:sample_last, sample_index] - target_steps[:sample_last, sample_index]
            intermediate_terms.append(torch.mean(diff ** 2))
        intermediate_loss = torch.stack(intermediate_terms).mean() if intermediate_terms else torch.zeros_like(final_loss)
    else:
        intermediate_loss = torch.zeros_like(final_loss)

    if predicted_states.shape[2] > 1:
        hidden_penalty = torch.mean(predicted_states[:, :, 1:] ** 2)
    else:
        hidden_penalty = torch.zeros_like(final_loss)

    if loss_mode == "final_only":
        total_loss = final_loss + lambda_hidden_l2 * hidden_penalty
    elif loss_mode == "intermediate":
        total_loss = intermediate_loss + lambda_hidden_l2 * hidden_penalty
    else:
        total_loss = final_loss + lambda_intermediate * intermediate_loss + lambda_hidden_l2 * hidden_penalty

    return {
        "loss": total_loss,
        "final_loss": final_loss,
        "intermediate_loss": intermediate_loss,
        "hidden_penalty": hidden_penalty,
    }


def run_rollout_batch(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    stochastic: bool = True,
    hidden_channels: int = 1,
) -> torch.Tensor:
    horizons = batch["horizons"]
    max_steps = int(horizons.max().item())
    initial_state = build_initial_state(batch["input_visible"], hidden_channels=hidden_channels, hidden_init="zeros")
    return rollout_model(model, initial_state, steps=max_steps, stochastic=stochastic)


def train_epoch(
    model: nn.Module,
    loader: Iterable[Dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_mode: str = "hybrid",
    lambda_intermediate: float = 0.5,
    lambda_hidden_l2: float = 1e-4,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    model.train()
    accumulators = {"loss": [], "final_loss": [], "intermediate_loss": [], "hidden_penalty": []}

    for batch in loader:
        batch = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        predictions = run_rollout_batch(model, batch, stochastic=True)
        losses = compute_rollout_loss(
            predictions,
            batch["targets_visible"],
            batch["target_mask"],
            loss_mode=loss_mode,
            lambda_intermediate=lambda_intermediate,
            lambda_hidden_l2=lambda_hidden_l2,
        )
        losses["loss"].backward()
        clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        for key in accumulators:
            accumulators[key].append(float(losses[key].detach().cpu().item()))

    return {key: float(np.mean(values)) for key, values in accumulators.items()}


@torch.no_grad()
def deterministic_eval(
    model: nn.Module,
    loader: Iterable[Dict[str, torch.Tensor]],
    device: torch.device,
    hidden_channels: int = 1,
) -> Dict[str, float]:
    model.eval()
    collected: Dict[str, List[float]] = {"one_step_mse": [], "rollout_mse": [], "population_mass_error": []}

    for batch in loader:
        batch = to_device(batch, device)
        predictions = run_rollout_batch(model, batch, stochastic=False, hidden_channels=hidden_channels)
        metrics = compute_rollout_metrics(predictions, batch["targets_visible"])
        for key in collected:
            collected[key].append(metrics[key])

    return {key: float(np.mean(values)) for key, values in collected.items()}


@torch.no_grad()
def stochastic_eval(
    model: nn.Module,
    loader: Iterable[Dict[str, torch.Tensor]],
    device: torch.device,
    num_rollouts: int = 8,
    hidden_channels: int = 1,
) -> Dict[str, float]:
    model.eval()
    collected: Dict[str, List[float]] = {
        "expected_mse": [],
        "ensemble_mean_mse": [],
        "pixelwise_std_mean": [],
        "mass_std": [],
        "population_mass_error": [],
    }

    for batch in loader:
        batch = to_device(batch, device)
        predictions = []
        for _ in range(num_rollouts):
            predictions.append(run_rollout_batch(model, batch, stochastic=True, hidden_channels=hidden_channels))
        sampled = torch.stack(predictions, dim=0)
        metrics = compute_stochastic_metrics(sampled, batch["targets_visible"])
        for key in collected:
            collected[key].append(metrics[key])

    return {key: float(np.mean(values)) for key, values in collected.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a minimalist Neural Cellular Automata baseline.")
    parser.add_argument("--data-root", type=str, default="NCA/data")
    parser.add_argument("--pattern", type=str, default="**/*.npy")
    parser.add_argument("--out-dir", type=str, default="NCA/runs/default")
    parser.add_argument("--split-mode", type=str, default="within_file", choices=["within_file", "by_file"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-steps", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--eval-steps", type=int, default=16)
    parser.add_argument("--num-rollouts", type=int, default=8)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--model-width", type=int, default=64)
    parser.add_argument("--update-prob", type=float, default=0.5)
    parser.add_argument("--loss-mode", type=str, default="hybrid", choices=["final_only", "intermediate", "hybrid"])
    parser.add_argument("--lambda-intermediate", type=float, default=0.5)
    parser.add_argument("--lambda-hidden-l2", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--use-alive-mask", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loaders, normalizer = build_dataloaders(
        data_root=args.data_root,
        pattern=args.pattern,
        split_mode=args.split_mode,
        train_steps=(args.min_steps, args.max_steps),
        eval_steps={"one_step": 1, "rollout": args.eval_steps, "stochastic": args.eval_steps},
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    model = NCA(
        state_channels=2,
        model_width=args.model_width,
        kernel_size=args.kernel_size,
        update_prob=args.update_prob,
        use_alive_mask=args.use_alive_mask,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model=model, optimizer=optimizer, map_location=device)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1

    history: List[Dict[str, float]] = []
    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_epoch(
            model=model,
            loader=loaders["train"],
            optimizer=optimizer,
            device=device,
            loss_mode=args.loss_mode,
            lambda_intermediate=args.lambda_intermediate,
            lambda_hidden_l2=args.lambda_hidden_l2,
            grad_clip=args.grad_clip,
        )
        det_one_step = deterministic_eval(model, loaders["val_one_step"], device=device) if "val_one_step" in loaders else {}
        det_rollout = deterministic_eval(model, loaders["val_rollout"], device=device) if "val_rollout" in loaders else {}
        stochastic_metrics = (
            stochastic_eval(
                model,
                loaders["val_rollout"],
                device=device,
                num_rollouts=args.num_rollouts,
            )
            if "val_rollout" in loaders
            else {}
        )

        epoch_metrics = {
            "epoch": float(epoch),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_one_step_{key}": value for key, value in det_one_step.items()},
            **{f"val_rollout_{key}": value for key, value in det_rollout.items()},
            **{f"val_stochastic_{key}": value for key, value in stochastic_metrics.items()},
        }
        history.append(epoch_metrics)

        checkpoint_path = out_dir / "checkpoint_latest.pt"
        save_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            config={
                "args": vars(args),
                "normalizer": normalizer.state_dict(),
            },
            metrics=epoch_metrics,
        )
        print(epoch_metrics)

    np.save(out_dir / "history.npy", np.asarray(history, dtype=object))


if __name__ == "__main__":
    main()
