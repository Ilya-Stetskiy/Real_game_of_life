from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .dataset_cache import DEFAULT_CACHE_PATH, load_graph_cache
from .gnn_model import CellInteractionGNN
from .rollout import RolloutGraphConfig, differentiable_prediction_to_next_graph
from .train_one_step import (
    apply_node_feature_normalization,
    fit_node_feature_normalization,
    infer_division_horizons,
    infer_shape_dim,
    jsonable,
    resolve_device,
)


@dataclass(frozen=True)
class RolloutBPTTConfig:
    cache_path: Path = DEFAULT_CACHE_PATH
    out_dir: Path = Path(__file__).resolve().parent / "runs" / "rollout_bptt"
    epochs: int = 20
    rollout_steps: int = 3
    hidden_dim: int = 128
    layers: int = 4
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    seed: int = 17
    device: str = "auto"
    lambda_pos: float = 1.0
    lambda_shape: float = 0.25
    match_tolerance: float = 1e-4
    normalize_node_features: bool = True
    normalization_epsilon: float = 1e-6
    init_from: Path | None = None


def match_next_graph_indices(
    current_graph: Any,
    next_graph: Any,
    current_indices: Tensor,
    *,
    tolerance: float = 1e-4,
) -> Tensor:
    """Map rollout nodes to their single-track GT descendants in the next frame."""

    current_indices = current_indices.detach().long().cpu()
    result = torch.full_like(current_indices, -1)
    if current_indices.numel() == 0 or int(next_graph.num_nodes) == 0:
        return result

    current_pos = current_graph.pos_xy.detach().float().cpu()
    next_pos = next_graph.pos_xy.detach().float().cpu()
    target_delta = current_graph.target_delta_pos.detach().float().cpu()
    valid_regression = current_graph.valid_regression_mask.detach().bool().cpu()
    used_next: set[int] = set()

    for rollout_index, current_index_value in enumerate(current_indices.tolist()):
        current_index = int(current_index_value)
        if current_index < 0 or current_index >= current_pos.size(0) or not bool(valid_regression[current_index]):
            continue
        expected = current_pos[current_index] + target_delta[current_index]
        distances = torch.linalg.norm(next_pos - expected, dim=1)
        for next_index in torch.argsort(distances).tolist():
            if int(next_index) in used_next:
                continue
            if float(distances[next_index].item()) <= float(tolerance):
                result[rollout_index] = int(next_index)
                used_next.add(int(next_index))
            break
    return result


def rollout_window_loss(
    model: nn.Module,
    sequence: Sequence[Any],
    *,
    start_index: int,
    rollout_steps: int,
    rollout_config: RolloutGraphConfig,
    node_feature_normalization: dict[str, Any] | None = None,
    lambda_pos: float = 1.0,
    lambda_shape: float = 0.25,
    match_tolerance: float = 1e-4,
) -> tuple[Tensor, dict[str, float | int]]:
    """Run a differentiable rollout window and compare predicted states to GT frames."""

    if rollout_steps < 1:
        raise ValueError("rollout_steps must be >= 1")
    if start_index < 0 or start_index + rollout_steps >= len(sequence):
        raise ValueError("Requested rollout window exceeds sequence bounds.")

    rollout_graph = sequence[start_index]
    gt_graph = sequence[start_index]
    gt_indices = torch.arange(int(rollout_graph.num_nodes), dtype=torch.long)
    step_losses: list[Tensor] = []
    pos_losses: list[Tensor] = []
    shape_losses: list[Tensor] = []
    matched_nodes = 0

    for offset in range(1, rollout_steps + 1):
        output = model(rollout_graph)
        rollout_graph = differentiable_prediction_to_next_graph(
            rollout_graph,
            output,
            config=rollout_config,
            node_feature_normalization=node_feature_normalization,
        )
        next_gt_graph = sequence[start_index + offset]
        gt_indices = match_next_graph_indices(
            gt_graph,
            next_gt_graph,
            gt_indices,
            tolerance=match_tolerance,
        ).to(device=rollout_graph.x.device)
        pos_loss, shape_loss, count = aligned_state_loss(
            rollout_graph,
            next_gt_graph,
            gt_indices,
            node_feature_normalization=node_feature_normalization,
        )
        if count > 0:
            pos_losses.append(pos_loss)
            shape_losses.append(shape_loss)
            step_losses.append(float(lambda_pos) * pos_loss + float(lambda_shape) * shape_loss)
            matched_nodes += count
        gt_graph = next_gt_graph

    if not step_losses:
        zero = rollout_graph.x.sum() * 0.0
        return zero, {"steps": 0, "matched_nodes": 0, "loss_pos": 0.0, "loss_shape": 0.0}

    loss = torch.stack(step_losses).mean()
    return loss, {
        "steps": len(step_losses),
        "matched_nodes": matched_nodes,
        "loss_pos": float(torch.stack(pos_losses).mean().detach().item()),
        "loss_shape": float(torch.stack(shape_losses).mean().detach().item()),
    }


def aligned_state_loss(
    predicted_graph: Any,
    gt_graph: Any,
    gt_indices: Tensor,
    *,
    node_feature_normalization: dict[str, Any] | None = None,
) -> tuple[Tensor, Tensor, int]:
    """Compare rollout state to aligned GT nodes in physical coordinates."""

    valid = gt_indices >= 0
    count = int(valid.sum().item())
    if count == 0:
        zero = predicted_graph.x.sum() * 0.0
        return zero, zero, 0

    gt_indices = gt_indices[valid].to(device=gt_graph.x.device)
    pred_pos = predicted_graph.pos_xy[valid]
    gt_pos = gt_graph.pos_xy[gt_indices].to(device=pred_pos.device, dtype=pred_pos.dtype)
    pos_loss = torch.mean((pred_pos - gt_pos) ** 2)

    feature_columns = tuple(str(column) for column in predicted_graph.node_feature_columns)
    shape_columns = tuple(str(column) for column in getattr(predicted_graph, "shape_target_columns", []))
    shape_indices = [feature_columns.index(column) for column in shape_columns if column in feature_columns]
    if not shape_indices:
        shape_loss = predicted_graph.x.sum() * 0.0
    else:
        pred_physical = _denormalize_x(predicted_graph.x, node_feature_normalization)
        gt_physical = _denormalize_x(gt_graph.x, node_feature_normalization).to(
            device=pred_physical.device,
            dtype=pred_physical.dtype,
        )
        pred_shape = pred_physical[valid][:, shape_indices]
        gt_shape = gt_physical[gt_indices][:, shape_indices]
        shape_loss = torch.mean((pred_shape - gt_shape) ** 2)
    return pos_loss, shape_loss, count


def train_from_cache(config: RolloutBPTTConfig) -> dict[str, Any]:
    if config.rollout_steps < 1:
        raise ValueError("rollout_steps must be >= 1")
    set_seed(config.seed)
    cache = load_graph_cache(config.cache_path)
    graphs = cache["graphs"]
    splits = cache["splits"]
    if not graphs:
        raise ValueError("Graph cache is empty.")

    split_graphs = {name: [graphs[index] for index in indices] for name, indices in splits.items()}
    train_graphs = split_graphs.get("train", [])
    if not train_graphs:
        raise ValueError("Train split is empty.")

    normalization = None
    if config.normalize_node_features:
        normalization = fit_node_feature_normalization(train_graphs, epsilon=config.normalization_epsilon)
        apply_node_feature_normalization(graphs, normalization)

    sequences = {name: group_graphs_by_sequence(items) for name, items in split_graphs.items()}
    device = resolve_device(config.device)
    model = CellInteractionGNN(
        node_dim=int(graphs[0].x.size(-1)),
        edge_dim=int(graphs[0].edge_attr.size(-1)),
        shape_dim=infer_shape_dim(graphs),
        hidden_dim=config.hidden_dim,
        num_message_passing_layers=config.layers,
        dropout=config.dropout,
        num_division_horizons=len(infer_division_horizons(graphs, (3, 5, 10))),
    ).to(device)
    if config.init_from is not None:
        checkpoint = torch.load(config.init_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    rollout_config = rollout_config_from_cache(cache)

    config.out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_metric = float("inf")
    best_epoch = 0
    for epoch in range(1, config.epochs + 1):
        train_metrics = run_rollout_epoch(
            model,
            sequences.get("train", []),
            device=device,
            optimizer=optimizer,
            config=config,
            rollout_config=rollout_config,
            node_feature_normalization=normalization,
        )
        history.append({"epoch": epoch, "phase": "train", **train_metrics})
        val_sequences = sequences.get("val", [])
        if val_sequences:
            val_metrics = run_rollout_epoch(
                model,
                val_sequences,
                device=device,
                optimizer=None,
                config=config,
                rollout_config=rollout_config,
                node_feature_normalization=normalization,
            )
            history.append({"epoch": epoch, "phase": "val", **val_metrics})
            selection_metric = val_metrics["loss_total"]
        else:
            selection_metric = train_metrics["loss_total"]

        if selection_metric < best_metric:
            best_metric = float(selection_metric)
            best_epoch = epoch
            save_checkpoint(
                config.out_dir / "best.pt",
                model=model,
                config=config,
                cache=cache,
                history=history,
                epoch=epoch,
                best_metric=best_metric,
                normalization=normalization,
            )
        save_checkpoint(
            config.out_dir / "last.pt",
            model=model,
            config=config,
            cache=cache,
            history=history,
            epoch=epoch,
            best_metric=best_metric,
            normalization=normalization,
        )
        (config.out_dir / "history.json").write_text(
            json.dumps(jsonable(history), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"epoch={epoch} train_loss={train_metrics['loss_total']:.6f} "
            f"matched={int(train_metrics['matched_nodes'])}"
        )

    summary = {
        "model_type": "rollout_bptt",
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "config": jsonable(asdict(config)),
        "dataset_config": cache.get("dataset_config", {}),
        "node_feature_normalization": jsonable(normalization),
    }
    (config.out_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"model": model, "history": history, "summary": summary}


def run_rollout_epoch(
    model: nn.Module,
    sequences: Sequence[Sequence[Any]],
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    config: RolloutBPTTConfig,
    rollout_config: RolloutGraphConfig,
    node_feature_normalization: dict[str, Any] | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    pos_sum = 0.0
    shape_sum = 0.0
    matched_nodes = 0
    windows = 0
    start_time = time.perf_counter()
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for sequence in sequences:
            sequence = [graph.to(device) for graph in sequence]
            for start_index in range(max(0, len(sequence) - config.rollout_steps)):
                loss, stats = rollout_window_loss(
                    model,
                    sequence,
                    start_index=start_index,
                    rollout_steps=config.rollout_steps,
                    rollout_config=rollout_config,
                    node_feature_normalization=node_feature_normalization,
                    lambda_pos=config.lambda_pos,
                    lambda_shape=config.lambda_shape,
                    match_tolerance=config.match_tolerance,
                )
                if int(stats["matched_nodes"]) == 0:
                    continue
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if config.grad_clip_norm > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                    optimizer.step()
                loss_sum += float(loss.detach().item())
                pos_sum += float(stats["loss_pos"])
                shape_sum += float(stats["loss_shape"])
                matched_nodes += int(stats["matched_nodes"])
                windows += 1

    if windows == 0:
        raise ValueError("No valid rollout windows with matched GT descendants.")
    seconds = max(time.perf_counter() - start_time, 1e-9)
    return {
        "loss_total": loss_sum / windows,
        "loss_pos": pos_sum / windows,
        "loss_shape": shape_sum / windows,
        "windows": float(windows),
        "matched_nodes": float(matched_nodes),
        "epoch_seconds": float(seconds),
    }


def group_graphs_by_sequence(graphs: Sequence[Any]) -> list[list[Any]]:
    grouped: dict[str, list[Any]] = {}
    for graph in graphs:
        grouped.setdefault(str(getattr(graph, "sequence_uid", "unknown")), []).append(graph)
    return [
        sorted(items, key=lambda graph: int(getattr(graph, "frame", 0)))
        for _, items in sorted(grouped.items())
        if items
    ]


def rollout_config_from_cache(cache: dict[str, Any]) -> RolloutGraphConfig:
    dataset_config = cache.get("dataset_config", {})
    return RolloutGraphConfig(
        position_features=tuple(dataset_config.get("position_cols", ("x", "y"))),
        edge_radius=dataset_config.get("edge_radius"),
        edge_k_nearest=int(dataset_config.get("edge_k_nearest", 0)),
        bidirectional_edges=bool(dataset_config.get("bidirectional_edges", True)),
        keep_edge_topology=False,
        drop_disappeared=False,
        enable_division_births=False,
        nan_fill_value=float(dataset_config.get("nan_fill_value", 0.0)),
    )


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    config: RolloutBPTTConfig,
    cache: dict[str, Any],
    history: list[dict[str, Any]],
    epoch: int,
    best_metric: float,
    normalization: dict[str, Any] | None,
) -> None:
    torch.save(
        {
            "model_type": "rollout_bptt",
            "model_state": model.state_dict(),
            "train_config": jsonable(asdict(config)),
            "dataset_config": cache.get("dataset_config", {}),
            "node_feature_normalization": normalization,
            "history": history,
            "epoch": epoch,
            "best_metric": best_metric,
        },
        path,
    )


def _denormalize_x(x: Tensor, stats: dict[str, Any] | None) -> Tensor:
    if stats is None:
        return x
    mean = torch.as_tensor(stats["mean"], device=x.device, dtype=x.dtype)
    std = torch.as_tensor(stats["std"], device=x.device, dtype=x.dtype)
    return x * std + mean


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> RolloutBPTTConfig:
    defaults = RolloutBPTTConfig()
    parser = argparse.ArgumentParser(description="Train GNN through differentiable autoregressive rollout windows.")
    parser.add_argument("--cache", type=Path, default=defaults.cache_path)
    parser.add_argument("--out-dir", type=Path, default=defaults.out_dir)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--rollout-steps", type=int, default=defaults.rollout_steps)
    parser.add_argument("--hidden-dim", type=int, default=defaults.hidden_dim)
    parser.add_argument("--layers", type=int, default=defaults.layers)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--grad-clip-norm", type=float, default=defaults.grad_clip_norm)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--lambda-pos", type=float, default=defaults.lambda_pos)
    parser.add_argument("--lambda-shape", type=float, default=defaults.lambda_shape)
    parser.add_argument("--match-tolerance", type=float, default=defaults.match_tolerance)
    parser.add_argument("--init-from", type=Path, default=defaults.init_from)
    parser.add_argument("--no-normalize-node-features", action="store_true")
    args = parser.parse_args()
    return RolloutBPTTConfig(
        cache_path=args.cache,
        out_dir=args.out_dir,
        epochs=args.epochs,
        rollout_steps=args.rollout_steps,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        seed=args.seed,
        device=args.device,
        lambda_pos=args.lambda_pos,
        lambda_shape=args.lambda_shape,
        match_tolerance=args.match_tolerance,
        normalize_node_features=not args.no_normalize_node_features,
        init_from=args.init_from,
    )


def main() -> None:
    train_from_cache(parse_args())


if __name__ == "__main__":
    main()
