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
from NCA.utils import save_checkpoint, set_seed

BASE = {
    "data_root": PROJECT_ROOT / "NCA" / "data",
    "pattern": "*.npy",
    "run_dir": PROJECT_ROOT / "NCA" / "runs" / "sweep_lambda_dice",
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
    "eval_threshold": 0.30,
    "loss_mode": "hybrid",
    "lambda_intermediate": 0.5,
    "lambda_hidden_l2": 1e-3,
    "grad_clip": 1.0,
    "use_alive_mask": False,
    "seed": 0,
    "device": "cpu",
}
LAMBDA_DICE = [0.0, 0.25, 0.5, 1.0, 1.5]


def main() -> None:
    run_dir = BASE["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(BASE["device"])
    loaders, normalizer = build_dataloaders(
        data_root=BASE["data_root"], pattern=BASE["pattern"], split_mode=BASE["split_mode"], split_ratios=BASE["split_ratios"],
        train_steps=BASE["train_steps"], eval_steps=BASE["eval_steps"], batch_size=BASE["batch_size"], eval_batch_size=BASE["batch_size"],
        seed=BASE["seed"], num_workers=0, data_channels=BASE["data_channels"], primary_channel=BASE["primary_channel"],
    )
    rows = []
    best = None
    for lambda_dice in LAMBDA_DICE:
        print(f"=== train lambda_dice={lambda_dice:g} ===", flush=True)
        set_seed(BASE["seed"])
        model = NCA(
            state_channels=BASE["data_channels"] + BASE["hidden_channels"], model_width=BASE["model_width"], kernel_size=BASE["kernel_size"],
            update_prob=BASE["update_prob"], use_alive_mask=BASE["use_alive_mask"], primary_channel=BASE["primary_channel"],
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=BASE["lr"])
        history = []
        for epoch in range(BASE["epochs"]):
            train_metrics = train_epoch(
                model=model, loader=loaders["train"], optimizer=optimizer, device=device,
                loss_mode=BASE["loss_mode"], lambda_intermediate=BASE["lambda_intermediate"], lambda_hidden_l2=BASE["lambda_hidden_l2"],
                grad_clip=BASE["grad_clip"], data_channels=BASE["data_channels"], hidden_channels=BASE["hidden_channels"],
                loss_channels=BASE["loss_channels"], primary_channel=BASE["primary_channel"], supervised_loss=BASE["supervised_loss"],
                bce_pos_weight=BASE["bce_pos_weight"], lambda_dice=lambda_dice, show_progress=False,
            )
            history.append({"epoch": epoch, **train_metrics})
            print(f"lambda_dice={lambda_dice:g} epoch={epoch} train_loss={train_metrics['loss']:.6f} final={train_metrics['final_loss']:.6f}", flush=True)
        metrics = deterministic_eval(
            model=model, loader=loaders["val_rollout"], device=device, data_channels=BASE["data_channels"], hidden_channels=BASE["hidden_channels"],
            primary_channel=BASE["primary_channel"], supervised_loss=BASE["supervised_loss"], threshold=BASE["eval_threshold"], show_progress=False,
        )
        row = {
            "lambda_dice": lambda_dice,
            **{f"val_rollout_{k}": v for k, v in metrics.items()},
            "train_loss": history[-1]["loss"],
            "train_final_loss": history[-1]["final_loss"],
            "train_intermediate_loss": history[-1]["intermediate_loss"],
            "train_hidden_penalty": history[-1]["hidden_penalty"],
        }
        rows.append(row)
        if best is None or row["val_rollout_iou"] > best["val_rollout_iou"]:
            best = row
        model_dir = run_dir / f"lambda_dice_{lambda_dice:g}"
        model_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            model_dir / "checkpoint_latest.pt", model=model, optimizer=optimizer, epoch=BASE["epochs"] - 1,
            config={"config": BASE, "lambda_dice": lambda_dice, "normalizer": normalizer.state_dict()}, metrics=row,
        )
        np.save(model_dir / "train_history.npy", np.asarray(history, dtype=object))
        print(f"eval lambda_dice={lambda_dice:g} iou={row['val_rollout_iou']:.4f} dice={row['val_rollout_dice']:.4f} precision={row['val_rollout_precision']:.4f} recall={row['val_rollout_recall']:.4f}", flush=True)
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
