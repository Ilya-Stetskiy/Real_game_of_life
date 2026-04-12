from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from NCA.dataset import build_dataloaders
from NCA.model import NCA
from NCA.train import deterministic_eval, train_epoch
from NCA.utils import build_initial_state, load_checkpoint, rollout_model, save_checkpoint, set_seed

CONFIG = {
    "data_root": PROJECT_ROOT / "NCA" / "data",
    "pattern": "*.npy",
    "run_dir": PROJECT_ROOT / "NCA" / "runs" / "sweep_hidden_l2",
    "split_mode": "by_file",
    "split_ratios": (0.6, 0.2, 0.2),
    "train_steps": (8, 16),
    "eval_steps": {"one_step": 1, "rollout": 16, "stochastic": 16},
    "batch_size": 8,
    "epochs": 5,
    "lr": 1e-3,
    "kernel_size": 3,
    "model_width": 64,
    "update_prob": 0.5,
    "data_channels": 2,
    "hidden_channels": 8,
    "primary_channel": 0,
    "loss_channels": "primary",
    "supervised_loss": "bce_dice",
    "bce_pos_weight": 0.25,
    "lambda_dice": 0.5,
    "eval_threshold": 0.30,
    "loss_mode": "hybrid",
    "lambda_intermediate": 0.5,
    "grad_clip": 1.0,
    "use_alive_mask": False,
    "seed": 0,
    "device": "cpu",
}
LAMBDA_HIDDEN_L2 = [1e-4, 3e-4, 1e-3, 3e-3]


def hidden_stats(model: NCA, loader, device: torch.device) -> dict:
    batch = next(iter(loader))
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    state0 = build_initial_state(batch["input_visible"], hidden_channels=CONFIG["hidden_channels"], hidden_init="zeros")
    steps = int(batch["horizons"].max().item())
    with torch.no_grad():
        rollout = rollout_model(model, state0, steps=steps, stochastic=False)
    hidden = rollout[-1, :, CONFIG["data_channels"]:]
    temporal = rollout[:, :, CONFIG["data_channels"]:]
    step_mean_abs = temporal.abs().mean(dim=(1, 2, 3, 4))
    return {
        "hidden_final_mean_abs": float(hidden.abs().mean().item()),
        "hidden_final_std": float(hidden.std(unbiased=False).item()),
        "hidden_final_max_abs": float(hidden.abs().max().item()),
        "hidden_step0_mean_abs": float(step_mean_abs[0].item()),
        "hidden_step_final_mean_abs": float(step_mean_abs[-1].item()),
        "hidden_temporal_delta": float((step_mean_abs[-1] - step_mean_abs[0]).item()),
    }


def main() -> None:
    run_dir = CONFIG["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(CONFIG["device"])
    loaders, normalizer = build_dataloaders(
        data_root=CONFIG["data_root"], pattern=CONFIG["pattern"], split_mode=CONFIG["split_mode"], split_ratios=CONFIG["split_ratios"],
        train_steps=CONFIG["train_steps"], eval_steps=CONFIG["eval_steps"], batch_size=CONFIG["batch_size"], eval_batch_size=CONFIG["batch_size"],
        seed=CONFIG["seed"], num_workers=0, data_channels=CONFIG["data_channels"], primary_channel=CONFIG["primary_channel"],
    )
    rows = []
    best = None
    for hidden_l2 in LAMBDA_HIDDEN_L2:
        print(f"=== train lambda_hidden_l2={hidden_l2:g} ===", flush=True)
        set_seed(CONFIG["seed"])
        model = NCA(
            state_channels=CONFIG["data_channels"] + CONFIG["hidden_channels"], model_width=CONFIG["model_width"], kernel_size=CONFIG["kernel_size"],
            update_prob=CONFIG["update_prob"], use_alive_mask=CONFIG["use_alive_mask"], primary_channel=CONFIG["primary_channel"],
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
        history = []
        for epoch in range(CONFIG["epochs"]):
            train_metrics = train_epoch(
                model=model, loader=loaders["train"], optimizer=optimizer, device=device,
                loss_mode=CONFIG["loss_mode"], lambda_intermediate=CONFIG["lambda_intermediate"], lambda_hidden_l2=hidden_l2,
                grad_clip=CONFIG["grad_clip"], data_channels=CONFIG["data_channels"], hidden_channels=CONFIG["hidden_channels"],
                loss_channels=CONFIG["loss_channels"], primary_channel=CONFIG["primary_channel"], supervised_loss=CONFIG["supervised_loss"],
                bce_pos_weight=CONFIG["bce_pos_weight"], lambda_dice=CONFIG["lambda_dice"], show_progress=False,
            )
            history.append({"epoch": epoch, **train_metrics})
            print(f"hidden_l2={hidden_l2:g} epoch={epoch} train_loss={train_metrics['loss']:.6f} hidden_penalty={train_metrics['hidden_penalty']:.6f}", flush=True)
        metrics = deterministic_eval(
            model=model, loader=loaders["val_rollout"], device=device, data_channels=CONFIG["data_channels"],
            hidden_channels=CONFIG["hidden_channels"], primary_channel=CONFIG["primary_channel"], supervised_loss=CONFIG["supervised_loss"],
            threshold=CONFIG["eval_threshold"], show_progress=False,
        )
        hstats = hidden_stats(model, loaders["val_rollout"], device)
        row = {
            "lambda_hidden_l2": hidden_l2,
            **{f"val_rollout_{k}": v for k, v in metrics.items()},
            **hstats,
            "train_loss": history[-1]["loss"],
            "train_final_loss": history[-1]["final_loss"],
            "train_intermediate_loss": history[-1]["intermediate_loss"],
            "train_hidden_penalty": history[-1]["hidden_penalty"],
        }
        rows.append(row)
        if best is None or row["val_rollout_iou"] > best["val_rollout_iou"]:
            best = row
        model_dir = run_dir / f"hidden_l2_{hidden_l2:g}"
        model_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            model_dir / "checkpoint_latest.pt", model=model, optimizer=optimizer, epoch=CONFIG["epochs"] - 1,
            config={"config": CONFIG, "lambda_hidden_l2": hidden_l2, "normalizer": normalizer.state_dict()}, metrics=row,
        )
        np.save(model_dir / "train_history.npy", np.asarray(history, dtype=object))
        print(
            f"eval hidden_l2={hidden_l2:g} iou={row['val_rollout_iou']:.4f} dice={row['val_rollout_dice']:.4f} "
            f"precision={row['val_rollout_precision']:.4f} recall={row['val_rollout_recall']:.4f} hidden_abs={row['hidden_final_mean_abs']:.4f}",
            flush=True,
        )

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with (run_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {"rows": rows, "best_by_iou": best}
    (run_dir / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("=== best_by_iou ===", flush=True)
    print(json.dumps(best, indent=2), flush=True)


if __name__ == "__main__":
    main()
