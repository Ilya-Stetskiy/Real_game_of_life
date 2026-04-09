from pathlib import Path

import numpy as np
import torch

from NCA.dataset import build_dataloaders
from NCA.model import NCA
from NCA.train import compute_rollout_loss, deterministic_eval, stochastic_eval, train_epoch
from NCA.utils import compute_binary_mask_metrics, compute_rollout_metrics, compute_stochastic_metrics, load_checkpoint, save_checkpoint
from NCA.visualize import plot_triptych, save_rollout_animation


def _make_synthetic_rollout_data(root: Path, file_count: int = 4, timesteps: int = 18, size: int = 6) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for file_id in range(file_count):
        state = np.zeros((timesteps, size, size), dtype=np.float32)
        state[0, size // 2, size // 2] = 1.0 + file_id * 0.05
        for step in range(1, timesteps):
            prev = state[step - 1]
            neighborhood = (
                prev
                + np.roll(prev, 1, axis=0)
                + np.roll(prev, -1, axis=0)
                + np.roll(prev, 1, axis=1)
                + np.roll(prev, -1, axis=1)
            ) / 5.0
            state[step] = np.clip(0.85 * prev + 0.25 * neighborhood, 0.0, 1.0)
        np.save(root / f"synthetic_{file_id}.npy", state[..., None])


def test_compute_rollout_loss_modes() -> None:
    predicted = torch.zeros(3, 2, 2, 2, 2)
    targets = torch.zeros(2, 3, 1, 2, 2)
    target_mask = torch.ones(2, 3, dtype=torch.bool)

    predicted[:, :, 0] = 1.0
    predicted[:, :, 1] = 2.0
    targets[:, 0] = 0.0
    targets[:, 1] = 0.5
    targets[:, 2] = 1.5

    losses = compute_rollout_loss(
        predicted,
        targets,
        target_mask,
        loss_mode="hybrid",
        lambda_intermediate=0.5,
        lambda_hidden_l2=0.1,
    )

    assert torch.isclose(losses["final_loss"], torch.tensor(0.25))
    assert torch.isclose(losses["intermediate_loss"], torch.tensor(0.625))
    assert torch.isclose(losses["hidden_penalty"], torch.tensor(4.0))
    assert torch.isclose(losses["loss"], torch.tensor(0.25 + 0.5 * 0.625 + 0.1 * 4.0))


def test_compute_rollout_loss_all_observed_channels() -> None:
    predicted = torch.zeros(2, 1, 4, 2, 2)
    targets = torch.zeros(1, 2, 2, 2, 2)
    target_mask = torch.ones(1, 2, dtype=torch.bool)

    predicted[:, :, 0] = 1.0
    predicted[:, :, 1] = 3.0
    predicted[:, :, 2:] = 5.0
    targets[:, :, 0] = 0.0
    targets[:, :, 1] = 1.0

    losses = compute_rollout_loss(
        predicted,
        targets,
        target_mask,
        loss_mode="final_only",
        lambda_hidden_l2=0.1,
        data_channels=2,
        loss_channels="all_observed",
        primary_channel=0,
    )

    assert torch.isclose(losses["final_loss"], torch.tensor(2.5))
    assert torch.isclose(losses["hidden_penalty"], torch.tensor(25.0))
    assert torch.isclose(losses["loss"], torch.tensor(5.0))


def test_compute_rollout_loss_bce_dice_prefers_confident_logits() -> None:
    target = torch.tensor([[[[[1.0, 0.0], [0.0, 1.0]]]]])
    mask = torch.ones(1, 1, dtype=torch.bool)
    good = torch.tensor([[[[[6.0, -6.0], [-6.0, 6.0]]]]])
    bad = torch.zeros_like(good)

    good_losses = compute_rollout_loss(
        good,
        target,
        mask,
        supervised_loss="bce_dice",
    )
    bad_losses = compute_rollout_loss(
        bad,
        target,
        mask,
        supervised_loss="bce_dice",
    )

    assert good_losses["final_loss"] < bad_losses["final_loss"]


def test_binary_mask_metrics_match_simple_case() -> None:
    prediction = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    target = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
    metrics = compute_binary_mask_metrics(prediction, target, threshold=0.5)

    assert np.isclose(metrics["dice"], 2.0 / 3.0)
    assert np.isclose(metrics["iou"], 0.5)
    assert np.isclose(metrics["precision"], 0.5)
    assert np.isclose(metrics["recall"], 1.0)


def test_metrics_functions_return_expected_keys() -> None:
    predictions = torch.ones(2, 1, 2, 3, 3)
    targets = torch.ones(1, 2, 1, 3, 3)
    sampled = torch.stack([predictions, predictions * 0.5], dim=0)

    rollout_metrics = compute_rollout_metrics(predictions, targets)
    stochastic_metrics = compute_stochastic_metrics(sampled, targets)

    assert {"one_step_mse", "rollout_mse", "population_mass_error", "dice", "iou"} <= rollout_metrics.keys()
    assert {"expected_mse", "ensemble_mean_mse", "pixelwise_std_mean", "mass_std", "dice", "iou"} <= stochastic_metrics.keys()


def test_end_to_end_cpu_smoke(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _make_synthetic_rollout_data(data_root)

    loaders, normalizer = build_dataloaders(
        data_root=data_root,
        pattern="*.npy",
        split_mode="within_file",
        split_ratios=(0.6, 0.2, 0.2),
        train_steps=(2, 4),
        eval_steps={"one_step": 1, "rollout": 3, "stochastic": 3},
        batch_size=2,
        eval_batch_size=2,
        seed=3,
    )

    model = NCA(state_channels=2, model_width=16, kernel_size=3, update_prob=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    device = torch.device("cpu")

    train_metrics = train_epoch(model, loaders["train"], optimizer, device=device)
    det_metrics = deterministic_eval(model, loaders["val_rollout"], device=device)
    stoch_metrics = stochastic_eval(model, loaders["val_rollout"], device=device, num_rollouts=3)

    assert train_metrics["loss"] >= 0.0
    assert det_metrics["rollout_mse"] >= 0.0
    assert stoch_metrics["expected_mse"] >= 0.0
    assert 0.0 <= det_metrics["dice"] <= 1.0
    assert 0.0 <= stoch_metrics["iou"] <= 1.0

    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=0,
        config={"normalizer": normalizer.state_dict()},
        metrics=det_metrics,
    )
    restored = NCA(state_channels=2, model_width=16, kernel_size=3, update_prob=0.5)
    checkpoint = load_checkpoint(checkpoint_path, model=restored, optimizer=None, map_location="cpu")
    assert checkpoint["epoch"] == 0

    batch = next(iter(loaders["val_rollout"]))
    prediction = model(torch.cat([batch["input_visible"], torch.zeros_like(batch["input_visible"])], dim=1), stochastic=False)
    triptych_path = plot_triptych(
        batch["input_visible"][0],
        batch["targets_visible"][0, 0],
        prediction[0, 0:1],
        tmp_path / "triptych.png",
    )
    animation_path = save_rollout_animation(
        torch.stack([batch["input_visible"], batch["targets_visible"][:, 0]], dim=0).permute(0, 1, 2, 3, 4),
        tmp_path / "rollout.gif",
    )

    assert triptych_path.exists()
    assert animation_path.exists()


def test_end_to_end_multichannel_smoke(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    for file_id in range(3):
        frames = np.zeros((12, 5, 5, 2), dtype=np.float32)
        frames[0, 2, 2, 0] = 1.0 + file_id * 0.1
        frames[..., 1] = 0.25 + file_id * 0.05
        for step in range(1, frames.shape[0]):
            prev = frames[step - 1, ..., 0]
            frames[step, ..., 0] = np.clip(0.8 * prev + 0.1, 0.0, 1.0)
            frames[step, ..., 1] = frames[step - 1, ..., 1] + 0.02
        np.save(data_root / f"multi_{file_id}.npy", frames)

    loaders, _ = build_dataloaders(
        data_root=data_root,
        pattern="*.npy",
        split_mode="within_file",
        split_ratios=(0.6, 0.2, 0.2),
        train_steps=(2, 3),
        eval_steps={"one_step": 1, "rollout": 2, "stochastic": 2},
        batch_size=2,
        eval_batch_size=2,
        seed=9,
        data_channels=2,
        primary_channel=0,
    )

    model = NCA(state_channels=4, model_width=16, kernel_size=3, update_prob=0.5, primary_channel=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    device = torch.device("cpu")

    train_metrics = train_epoch(
        model,
        loaders["train"],
        optimizer,
        device=device,
        data_channels=2,
        hidden_channels=2,
        loss_channels="all_observed",
        primary_channel=0,
        supervised_loss="bce_dice",
        bce_pos_weight=2.0,
    )
    det_metrics = deterministic_eval(
        model,
        loaders["val_rollout"],
        device=device,
        data_channels=2,
        hidden_channels=2,
        primary_channel=0,
        supervised_loss="bce_dice",
    )
    stoch_metrics = stochastic_eval(
        model,
        loaders["val_rollout"],
        device=device,
        num_rollouts=2,
        data_channels=2,
        hidden_channels=2,
        primary_channel=0,
        supervised_loss="bce_dice",
    )

    assert train_metrics["loss"] >= 0.0
    assert det_metrics["rollout_mse"] >= 0.0
    assert stoch_metrics["expected_mse"] >= 0.0
    assert 0.0 <= det_metrics["dice"] <= 1.0
    assert 0.0 <= stoch_metrics["iou"] <= 1.0
