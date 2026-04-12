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
    data_channels: int = 1,
    loss_channels: str = "primary",
    primary_channel: int = 0,
    supervised_loss: str = "mse",
    bce_pos_weight: float = 1.0,
    lambda_dice: float = 0.5,
) -> Dict[str, torch.Tensor]:
    if loss_mode not in {"final_only", "intermediate", "hybrid"}:
        raise ValueError("loss_mode must be one of 'final_only', 'intermediate', 'hybrid'.")
    if loss_channels not in {"primary", "all_observed"}:
        raise ValueError("loss_channels must be one of 'primary', 'all_observed'.")
    if supervised_loss not in {"mse", "bce", "bce_dice"}:
        raise ValueError("supervised_loss must be one of 'mse', 'bce', 'bce_dice'.")
    if data_channels < 1:
        raise ValueError("data_channels must be >= 1.")
    if not 0 <= primary_channel < data_channels:
        raise ValueError("primary_channel must index one of the observed channels.")

    steps, batch_size = predicted_states.shape[:2]
    if targets_visible.shape[1] != steps:
        raise ValueError("predicted_states and targets_visible must share the same rollout length.")
    if targets_visible.shape[2] != data_channels:
        raise ValueError(
            f"Expected targets_visible to have data_channels={data_channels}, got {targets_visible.shape[2]}."
        )

    if loss_channels == "primary":
        supervised_predictions = predicted_states[:, :, primary_channel:primary_channel + 1]
        supervised_targets = targets_visible[:, :, primary_channel:primary_channel + 1]
    else:
        supervised_predictions = predicted_states[:, :, :data_channels]
        supervised_targets = targets_visible

    target_steps = supervised_targets.permute(1, 0, 2, 3, 4)
    mask_steps = target_mask.permute(1, 0)
    pos_weight = torch.as_tensor(float(bce_pos_weight), device=predicted_states.device, dtype=predicted_states.dtype)

    def _supervised_term(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if supervised_loss == "mse":
            return torch.mean((pred - target) ** 2)

        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            pred,
            target,
            pos_weight=pos_weight,
        )
        if supervised_loss == "bce":
            return bce

        probs = torch.sigmoid(pred)
        intersection = (probs * target).sum(dim=(-3, -2, -1))
        denom = probs.sum(dim=(-3, -2, -1)) + target.sum(dim=(-3, -2, -1))
        dice_loss = 1.0 - ((2.0 * intersection + 1e-8) / (denom + 1e-8)).mean()
        return bce + lambda_dice * dice_loss

    per_step_losses: List[torch.Tensor] = []
    for step in range(steps):
        valid = mask_steps[step]
        if not torch.any(valid):
            continue
        per_step_losses.append(_supervised_term(supervised_predictions[step, valid], target_steps[step, valid]))

    if not per_step_losses:
        raise ValueError("No supervised rollout steps available for loss computation.")

    final_step_indices = target_mask.sum(dim=1) - 1
    batch_indices = torch.arange(batch_size, device=predicted_states.device)
    final_visible = supervised_predictions[final_step_indices, batch_indices]
    final_targets = supervised_targets[batch_indices, final_step_indices]
    final_loss = _supervised_term(final_visible, final_targets)

    if steps > 1:
        intermediate_terms = []
        for sample_index in range(batch_size):
            sample_last = int(final_step_indices[sample_index].item())
            if sample_last <= 0:
                continue
            intermediate_terms.append(
                _supervised_term(
                    supervised_predictions[:sample_last, sample_index],
                    target_steps[:sample_last, sample_index],
                )
            )
        intermediate_loss = torch.stack(intermediate_terms).mean() if intermediate_terms else torch.zeros_like(final_loss)
    else:
        intermediate_loss = torch.zeros_like(final_loss)

    if predicted_states.shape[2] > data_channels:
        hidden_penalty = torch.mean(predicted_states[:, :, data_channels:] ** 2)
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
    data_channels: int = 1,
    hidden_channels: int = 1,
    loss_channels: str = "primary",
    primary_channel: int = 0,
    supervised_loss: str = "mse",
    bce_pos_weight: float = 1.0,
    lambda_dice: float = 0.5,
    show_progress: bool = False,
    progress_desc: str = "train",
) -> Dict[str, float]:
    model.train()
    accumulators = {"loss": [], "final_loss": [], "intermediate_loss": [], "hidden_penalty": []}

    iterator = tqdm(loader, desc=progress_desc, leave=False) if show_progress else loader
    for batch in iterator:
        batch = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        predictions = run_rollout_batch(model, batch, stochastic=True, hidden_channels=hidden_channels)
        losses = compute_rollout_loss(
            predictions,
            batch["targets_visible"],
            batch["target_mask"],
            loss_mode=loss_mode,
            lambda_intermediate=lambda_intermediate,
            lambda_hidden_l2=lambda_hidden_l2,
            data_channels=data_channels,
            loss_channels=loss_channels,
            primary_channel=primary_channel,
            supervised_loss=supervised_loss,
            bce_pos_weight=bce_pos_weight,
            lambda_dice=lambda_dice,
        )
        losses["loss"].backward()
        clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        for key in accumulators:
            accumulators[key].append(float(losses[key].detach().cpu().item()))

        if show_progress:
            iterator.set_postfix(
                loss=f"{accumulators['loss'][-1]:.4e}",
                final=f"{accumulators['final_loss'][-1]:.4e}",
            )

    return {key: float(np.mean(values)) for key, values in accumulators.items()}


@torch.no_grad()
def deterministic_eval(
    model: nn.Module,
    loader: Iterable[Dict[str, torch.Tensor]],
    device: torch.device,
    data_channels: int = 1,
    hidden_channels: int = 1,
    primary_channel: int = 0,
    supervised_loss: str = "mse",
    threshold: float = 0.5,
    show_progress: bool = False,
    progress_desc: str = "det_eval",
) -> Dict[str, float]:
    model.eval()
    collected: Dict[str, List[float]] = {
        "one_step_mse": [],
        "rollout_mse": [],
        "population_mass_error": [],
        "dice": [],
        "iou": [],
        "precision": [],
        "recall": [],
    }

    iterator = tqdm(loader, desc=progress_desc, leave=False) if show_progress else loader
    for batch in iterator:
        batch = to_device(batch, device)
        predictions = run_rollout_batch(model, batch, stochastic=False, hidden_channels=hidden_channels)
        metrics = compute_rollout_metrics(
            predictions,
            batch["targets_visible"],
            visible_channel=primary_channel,
            supervised_loss=supervised_loss,
            threshold=threshold,
        )
        for key in collected:
            collected[key].append(metrics[key])

        if show_progress:
            iterator.set_postfix(rollout_mse=f"{metrics['rollout_mse']:.4e}")

    return {key: float(np.mean(values)) for key, values in collected.items()}


@torch.no_grad()
def stochastic_eval(
    model: nn.Module,
    loader: Iterable[Dict[str, torch.Tensor]],
    device: torch.device,
    num_rollouts: int = 8,
    data_channels: int = 1,
    hidden_channels: int = 1,
    primary_channel: int = 0,
    supervised_loss: str = "mse",
    threshold: float = 0.5,
    show_progress: bool = False,
    progress_desc: str = "stoch_eval",
) -> Dict[str, float]:
    model.eval()
    collected: Dict[str, List[float]] = {
        "expected_mse": [],
        "ensemble_mean_mse": [],
        "pixelwise_std_mean": [],
        "mass_std": [],
        "population_mass_error": [],
        "dice": [],
        "iou": [],
        "precision": [],
        "recall": [],
    }

    iterator = tqdm(loader, desc=progress_desc, leave=False) if show_progress else loader
    for batch in iterator:
        batch = to_device(batch, device)
        predictions = []
        for _ in range(num_rollouts):
            predictions.append(run_rollout_batch(model, batch, stochastic=True, hidden_channels=hidden_channels))
        sampled = torch.stack(predictions, dim=0)
        metrics = compute_stochastic_metrics(
            sampled,
            batch["targets_visible"],
            visible_channel=primary_channel,
            supervised_loss=supervised_loss,
            threshold=threshold,
        )
        for key in collected:
            collected[key].append(metrics[key])

        if show_progress:
            iterator.set_postfix(expected_mse=f"{metrics['expected_mse']:.4e}")

    return {key: float(np.mean(values)) for key, values in collected.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a minimalist Neural Cellular Automata baseline.")
    parser.add_argument("--data-root", type=str, default="NCA/data")
    parser.add_argument("--pattern", type=str, default="**/*.npy")
    parser.add_argument("--out-dir", type=str, default="NCA/runs/default")
    parser.add_argument("--split-mode", type=str, default="by_group", choices=["within_file", "by_file", "by_group"])
    parser.add_argument("--split-ratios", type=float, nargs=3, default=(0.8, 0.1, 0.1))
    parser.add_argument("--group-regex", type=str, default=r"pos(\d+)")
    parser.add_argument("--group-regex-group", type=int, default=1)
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
    parser.add_argument("--data-channels", type=int, default=1)
    parser.add_argument("--hidden-channels", type=int, default=1)
    parser.add_argument("--primary-channel", type=int, default=0)
    parser.add_argument("--loss-channels", type=str, default="primary", choices=["primary", "all_observed"])
    parser.add_argument("--supervised-loss", type=str, default="mse", choices=["mse", "bce", "bce_dice"])
    parser.add_argument("--bce-pos-weight", type=float, default=1.0)
    parser.add_argument("--lambda-dice", type=float, default=0.5)
    parser.add_argument("--eval-threshold", type=float, default=0.5)
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
        split_ratios=args.split_ratios,
        train_steps=(args.min_steps, args.max_steps),
        eval_steps={"one_step": 1, "rollout": args.eval_steps, "stochastic": args.eval_steps},
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        data_channels=args.data_channels,
        primary_channel=args.primary_channel,
        group_regex=args.group_regex,
        group_regex_group=args.group_regex_group,
    )

    model = NCA(
        state_channels=args.data_channels + args.hidden_channels,
        model_width=args.model_width,
        kernel_size=args.kernel_size,
        update_prob=args.update_prob,
        use_alive_mask=args.use_alive_mask,
        primary_channel=args.primary_channel,
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
            data_channels=args.data_channels,
            hidden_channels=args.hidden_channels,
            loss_channels=args.loss_channels,
            primary_channel=args.primary_channel,
            supervised_loss=args.supervised_loss,
            bce_pos_weight=args.bce_pos_weight,
            lambda_dice=args.lambda_dice,
        )
        det_one_step = (
            deterministic_eval(
                model,
                loaders["val_one_step"],
                device=device,
                data_channels=args.data_channels,
                hidden_channels=args.hidden_channels,
                primary_channel=args.primary_channel,
                supervised_loss=args.supervised_loss,
                threshold=args.eval_threshold,
            )
            if "val_one_step" in loaders
            else {}
        )
        det_rollout = (
            deterministic_eval(
                model,
                loaders["val_rollout"],
                device=device,
                data_channels=args.data_channels,
                hidden_channels=args.hidden_channels,
                primary_channel=args.primary_channel,
                supervised_loss=args.supervised_loss,
                threshold=args.eval_threshold,
            )
            if "val_rollout" in loaders
            else {}
        )
        stochastic_metrics = (
            stochastic_eval(
                model,
                loaders["val_rollout"],
                device=device,
                num_rollouts=args.num_rollouts,
                data_channels=args.data_channels,
                hidden_channels=args.hidden_channels,
                primary_channel=args.primary_channel,
                supervised_loss=args.supervised_loss,
                threshold=args.eval_threshold,
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
