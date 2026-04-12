from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from NCA.dataset import NCADataset, VisibleChannelNormalizer, nca_collate_fn
from NCA.model import NCA
from NCA.train import deterministic_eval, stochastic_eval, train_epoch
from NCA.utils import build_initial_state, get_device, rollout_model, save_checkpoint, set_seed, to_device
from NCA.visualize import plot_metric_curves, plot_triptych, plot_uncertainty_heatmap, save_rollout_animation


CONFIG = {
    "project_root": PROJECT_ROOT,
    "data_root": PROJECT_ROOT / "NCA" / "data_v2",
    "run_dir": PROJECT_ROOT / "NCA" / "runs" / "data_v2",
    "train_steps": (8, 16),
    "eval_steps": {
        "one_step": 1,
        "rollout": 16,
        "stochastic": 16,
    },
    "batch_size": 8,
    "epochs": 5,
    "lr": 1e-3,
    "kernel_size": 3,
    "model_width": 64,
    "update_prob": 0.5,
    "data_channels": 1,
    "hidden_channels": 8,
    "primary_channel": 0,
    "loss_channels": "primary",
    "supervised_loss": "bce_dice",
    "bce_pos_weight": 0.25,
    "lambda_dice": 1.5,
    "eval_threshold": 0.30,
    "num_rollouts": 8,
    "loss_mode": "hybrid",
    "lambda_intermediate": 0.5,
    "lambda_hidden_l2": 1e-3,
    "use_alive_mask": False,
    "seed": 0,
    "num_workers": 0,
    "device": None,
}


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def inspect_data(data_root: Path) -> Dict[str, object]:
    result: Dict[str, object] = {"splits": {}}
    for split in ("train", "val", "test"):
        files = sorted((data_root / split).glob("*.npy"))
        split_info: Dict[str, object] = {"files": len(files)}
        if files:
            sample = np.load(files[0], mmap_mode="r")
            split_info.update(
                {
                    "sample_file": files[0].name,
                    "shape": list(sample.shape),
                    "dtype": str(sample.dtype),
                    "min": float(sample.min()),
                    "max": float(sample.max()),
                    "mean": float(sample.mean()),
                    "nonzero_fraction": float(np.count_nonzero(sample) / sample.size),
                }
            )
        result["splits"][split] = split_info
    return result


def make_dataset(files: List[Path], min_steps: int, max_steps: int, augment: bool, normalizer, config):
    return NCADataset(
        data_root=config["data_root"],
        files=files,
        split="train",
        split_mode="by_file",
        split_ratios=(1.0, 0.0, 0.0),
        min_steps=min_steps,
        max_steps=max_steps,
        augment=augment,
        normalizer=normalizer,
        seed=config["seed"],
        cache_arrays=True,
        data_channels=config["data_channels"],
    )


def make_loaders(config) -> tuple[Dict[str, DataLoader], VisibleChannelNormalizer, Dict[str, object]]:
    data_root = config["data_root"]
    split_files = {
        split: sorted((data_root / split).glob("*.npy"))
        for split in ("train", "val", "test")
    }
    missing = [split for split, files in split_files.items() if not files]
    if missing:
        raise FileNotFoundError(f"Empty data_v2 split folders: {missing}")

    normalizer = VisibleChannelNormalizer(
        data_channels=config["data_channels"],
        primary_channel=config["primary_channel"],
    ).fit(split_files["train"])

    datasets = {
        "train": make_dataset(
            split_files["train"],
            min_steps=config["train_steps"][0],
            max_steps=config["train_steps"][1],
            augment=True,
            normalizer=normalizer,
            config=config,
        ),
        "val_one_step": make_dataset(
            split_files["val"],
            min_steps=config["eval_steps"]["one_step"],
            max_steps=config["eval_steps"]["one_step"],
            augment=False,
            normalizer=normalizer,
            config=config,
        ),
        "val_rollout": make_dataset(
            split_files["val"],
            min_steps=config["eval_steps"]["rollout"],
            max_steps=config["eval_steps"]["rollout"],
            augment=False,
            normalizer=normalizer,
            config=config,
        ),
        "test_rollout": make_dataset(
            split_files["test"],
            min_steps=config["eval_steps"]["rollout"],
            max_steps=config["eval_steps"]["rollout"],
            augment=False,
            normalizer=normalizer,
            config=config,
        ),
    }

    loaders = {
        name: DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=(name == "train"),
            num_workers=config["num_workers"],
            collate_fn=nca_collate_fn,
        )
        for name, dataset in datasets.items()
    }
    dataset_info = {
        name: {
            "files": len(split_files["train" if name == "train" else name.split("_")[0]]),
            "samples": len(dataset),
        }
        for name, dataset in datasets.items()
    }
    return loaders, normalizer, dataset_info


@torch.no_grad()
def make_visuals(model, loader, device, config) -> Dict[str, str]:
    run_dir = config["run_dir"]
    model.eval()
    batch = next(iter(loader))
    batch = to_device(batch, device)
    rollout = rollout_model(
        model,
        build_initial_state(batch["input_visible"], hidden_channels=config["hidden_channels"]),
        steps=int(batch["horizons"].max().item()),
        stochastic=False,
    )
    final_indices = batch["horizons"] - 1
    prediction = torch.stack([rollout[int(step.item()), row] for row, step in enumerate(final_indices)], dim=0)
    target = torch.stack(
        [batch["targets_visible"][row, int(step.item())] for row, step in enumerate(final_indices)],
        dim=0,
    )
    if config["supervised_loss"] in {"bce", "bce_dice"}:
        prediction_visible = torch.sigmoid(prediction[:, : config["data_channels"]])
    else:
        prediction_visible = prediction[:, : config["data_channels"]].clamp(0.0, 1.0)

    sampled = []
    for _ in range(config["num_rollouts"]):
        sampled.append(
            rollout_model(
                model,
                build_initial_state(batch["input_visible"], hidden_channels=config["hidden_channels"]),
                steps=int(batch["horizons"].max().item()),
                stochastic=True,
            )
        )
    sampled_predictions = torch.stack(sampled, dim=0)

    paths = {
        "triptych": plot_triptych(
            batch["input_visible"][0],
            target[0],
            prediction_visible[0],
            run_dir / "triptych.png",
            title="data_v2 primary channel",
            channel_index=config["primary_channel"],
        ),
        "uncertainty": plot_uncertainty_heatmap(
            sampled_predictions,
            run_dir / "uncertainty.png",
            visible_channel=config["primary_channel"],
        ),
        "det_rollout_gif": save_rollout_animation(
            torch.sigmoid(rollout[:, :1, : config["data_channels"]])
            if config["supervised_loss"] in {"bce", "bce_dice"}
            else rollout[:, :1, : config["data_channels"]].clamp(0.0, 1.0),
            run_dir / "det_rollout.gif",
            channel_index=config["primary_channel"],
        ),
    }
    return {key: str(path) for key, path in paths.items()}


def write_history_csv(history: List[Dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in history for key in row})
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    config = CONFIG.copy()
    set_seed(config["seed"])
    run_dir = config["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)

    data_info = inspect_data(config["data_root"])
    loaders, normalizer, loader_info = make_loaders(config)
    device = get_device(config["device"])

    model = NCA(
        state_channels=config["data_channels"] + config["hidden_channels"],
        model_width=config["model_width"],
        kernel_size=config["kernel_size"],
        update_prob=config["update_prob"],
        use_alive_mask=config["use_alive_mask"],
        primary_channel=config["primary_channel"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    history: List[Dict[str, float]] = []
    for epoch in range(config["epochs"]):
        train_metrics = train_epoch(
            model=model,
            loader=loaders["train"],
            optimizer=optimizer,
            device=device,
            loss_mode=config["loss_mode"],
            lambda_intermediate=config["lambda_intermediate"],
            lambda_hidden_l2=config["lambda_hidden_l2"],
            data_channels=config["data_channels"],
            hidden_channels=config["hidden_channels"],
            loss_channels=config["loss_channels"],
            primary_channel=config["primary_channel"],
            supervised_loss=config["supervised_loss"],
            bce_pos_weight=config["bce_pos_weight"],
            lambda_dice=config["lambda_dice"],
            show_progress=True,
            progress_desc=f"data_v2 train {epoch + 1}/{config['epochs']}",
        )
        val_one_step = deterministic_eval(
            model,
            loaders["val_one_step"],
            device=device,
            data_channels=config["data_channels"],
            hidden_channels=config["hidden_channels"],
            primary_channel=config["primary_channel"],
            supervised_loss=config["supervised_loss"],
            threshold=config["eval_threshold"],
            show_progress=True,
            progress_desc=f"data_v2 val one-step {epoch + 1}/{config['epochs']}",
        )
        val_rollout = deterministic_eval(
            model,
            loaders["val_rollout"],
            device=device,
            data_channels=config["data_channels"],
            hidden_channels=config["hidden_channels"],
            primary_channel=config["primary_channel"],
            supervised_loss=config["supervised_loss"],
            threshold=config["eval_threshold"],
            show_progress=True,
            progress_desc=f"data_v2 val rollout {epoch + 1}/{config['epochs']}",
        )
        epoch_metrics = {
            "epoch": float(epoch),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_one_step_{key}": value for key, value in val_one_step.items()},
            **{f"val_rollout_{key}": value for key, value in val_rollout.items()},
        }
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics, ensure_ascii=False, sort_keys=True))

    val_stochastic = stochastic_eval(
        model,
        loaders["val_rollout"],
        device=device,
        num_rollouts=config["num_rollouts"],
        data_channels=config["data_channels"],
        hidden_channels=config["hidden_channels"],
        primary_channel=config["primary_channel"],
        supervised_loss=config["supervised_loss"],
        threshold=config["eval_threshold"],
        show_progress=True,
        progress_desc="data_v2 val stochastic",
    )
    test_rollout = deterministic_eval(
        model,
        loaders["test_rollout"],
        device=device,
        data_channels=config["data_channels"],
        hidden_channels=config["hidden_channels"],
        primary_channel=config["primary_channel"],
        supervised_loss=config["supervised_loss"],
        threshold=config["eval_threshold"],
        show_progress=True,
        progress_desc="data_v2 test rollout",
    )
    test_stochastic = stochastic_eval(
        model,
        loaders["test_rollout"],
        device=device,
        num_rollouts=config["num_rollouts"],
        data_channels=config["data_channels"],
        hidden_channels=config["hidden_channels"],
        primary_channel=config["primary_channel"],
        supervised_loss=config["supervised_loss"],
        threshold=config["eval_threshold"],
        show_progress=True,
        progress_desc="data_v2 test stochastic",
    )

    save_checkpoint(
        checkpoint_path=run_dir / "checkpoint_latest.pt",
        model=model,
        optimizer=optimizer,
        epoch=config["epochs"] - 1,
        config={
            "config": _jsonable(config),
            "normalizer": normalizer.state_dict(),
            "data_info": data_info,
            "loader_info": loader_info,
        },
        metrics=history[-1],
    )
    np.save(run_dir / "history.npy", np.asarray(history, dtype=object))
    write_history_csv(history, run_dir / "history.csv")
    plot_metric_curves(
        history,
        run_dir / "loss_curve.png",
        keys=["train_loss", "val_rollout_iou", "val_rollout_dice"],
    )
    visuals = make_visuals(model, loaders["test_rollout"], device, config)

    results = {
        "config": _jsonable(config),
        "device": str(device),
        "data_info": data_info,
        "loader_info": loader_info,
        "history": history,
        "final_val_stochastic": val_stochastic,
        "final_test_rollout": test_rollout,
        "final_test_stochastic": test_stochastic,
        "visuals": visuals,
    }
    with (run_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
