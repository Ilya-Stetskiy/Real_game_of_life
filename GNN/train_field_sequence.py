from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from .dataset_cache import DEFAULT_CACHE_PATH, load_graph_cache
from .gnn_model import cell_dynamics_loss
from .hybrid_field_gnn_model import FieldConditionedCellGNN
from .spatial_field import FieldGeometry
from .train_one_step import (
    apply_node_feature_normalization,
    accumulate_regression_sums,
    add_metric_prefix_alias,
    collect_event_scores,
    division_horizon_tensors,
    division_horizon_metric_name,
    epoch_event_metrics,
    fit_node_feature_normalization,
    infer_division_horizons,
    infer_shape_dim,
    jsonable,
    load_training_state,
    prepare_field_geometry_config,
    require_field_pos_xy,
    resolve_device,
    save_checkpoint,
    set_seed,
    target_pos_weight,
    target_pos_weights_for_division_horizons,
)


@dataclass(frozen=True)
class SequenceTrainConfig:
    cache_path: Path = DEFAULT_CACHE_PATH
    out_dir: Path = Path(__file__).resolve().parent / "runs" / "field_sequence"
    epochs: int = 20
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
    lambda_division: float = 1.0
    lambda_death: float = 0.25
    lambda_division_horizon: float = 1.0
    division_horizons: tuple[int, ...] = (3, 5, 10)
    max_pos_weight: float = 100.0
    scheduler_patience: int = 20
    scheduler_factor: float = 0.5
    min_learning_rate: float = 1e-6
    early_stopping_patience: int = 0
    min_delta: float = 0.0
    checkpoint_every: int = 0
    resume_from: Path | None = None
    normalize_node_features: bool = True
    normalization_epsilon: float = 1e-6
    field_channels: int = 16
    field_height: int = 128
    field_width: int = 128
    field_patch_radius: int = 1
    field_context_dim: int = 64
    field_update_hidden_channels: int = 64
    field_cell_size: float = 4.0
    field_origin_x: float = 0.0
    field_origin_y: float = 0.0
    bptt_window: int = 3
    field_auto_geometry: bool = True
    min_field_coverage: float = 0.95
    strict_field_coverage: bool = False


def train_from_cache(config: SequenceTrainConfig) -> dict[str, Any]:
    if config.bptt_window < 1:
        raise ValueError("bptt_window must be >= 1")
    set_seed(config.seed)
    cache = load_graph_cache(config.cache_path)
    graphs = cache["graphs"]
    splits = cache["splits"]
    if not graphs:
        raise ValueError("Graph cache is empty.")

    split_graphs = {
        name: [graphs[index] for index in indices]
        for name, indices in splits.items()
    }
    train_graphs = split_graphs.get("train", [])
    val_graphs = split_graphs.get("val", [])
    test_graphs = split_graphs.get("test", [])
    if not train_graphs:
        raise ValueError("Train split is empty.")
    require_field_pos_xy(graphs)
    config = prepare_field_geometry_config(
        config,
        train_graphs=train_graphs,
        split_graphs=split_graphs,
    )

    node_feature_normalization = None
    if config.normalize_node_features:
        node_feature_normalization = fit_node_feature_normalization(train_graphs, epsilon=config.normalization_epsilon)
        apply_node_feature_normalization(graphs, node_feature_normalization)

    train_sequences = group_graphs_by_sequence(train_graphs)
    val_sequences = group_graphs_by_sequence(val_graphs)
    test_sequences = group_graphs_by_sequence(test_graphs)
    if not train_sequences:
        raise ValueError("Train split does not contain any non-empty sequences.")

    device = resolve_device(config.device)
    node_dim = int(graphs[0].x.size(-1))
    edge_dim = int(graphs[0].edge_attr.size(-1))
    shape_dim = infer_shape_dim(graphs)
    division_horizons = infer_division_horizons(graphs, config.division_horizons)
    model = build_sequence_model(
        config,
        node_dim=node_dim,
        edge_dim=edge_dim,
        shape_dim=shape_dim,
        num_division_horizons=len(division_horizons),
    ).to(device)

    pos_weight_division = target_pos_weight(train_graphs, "target_division", "valid_event_mask", config.max_pos_weight).to(device)
    pos_weight_death = target_pos_weight(train_graphs, "target_death", "valid_event_mask", config.max_pos_weight).to(device)
    pos_weight_division_horizon = target_pos_weights_for_division_horizons(
        train_graphs,
        division_horizons,
        config.max_pos_weight,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.min_learning_rate,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    config.out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_metric = float("inf")
    best_epoch = 0
    start_epoch = 1
    if config.resume_from is not None:
        start_epoch, best_epoch, best_metric, history = load_training_state(
            config.resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )

    epochs_without_improvement = 0
    epochs_completed = start_epoch - 1
    for epoch in range(start_epoch, config.epochs + 1):
        train_metrics = run_sequence_epoch(
            model,
            train_sequences,
            device=device,
            optimizer=optimizer,
            config=config,
            division_horizons=division_horizons,
            pos_weight_division=pos_weight_division,
            pos_weight_death=pos_weight_death,
            pos_weight_division_horizon=pos_weight_division_horizon,
        )
        train_row = {"epoch": epoch, "phase": "train", **train_metrics}
        history.append(train_row)

        if val_sequences:
            val_metrics = run_sequence_epoch(
                model,
                val_sequences,
                device=device,
                optimizer=None,
                config=config,
                division_horizons=division_horizons,
                pos_weight_division=pos_weight_division,
                pos_weight_death=pos_weight_death,
                pos_weight_division_horizon=pos_weight_division_horizon,
            )
            val_row = {"epoch": epoch, "phase": "val", **val_metrics}
            history.append(val_row)
            selection_metric = val_metrics["loss_total"]
        else:
            selection_metric = train_metrics["loss_total"]

        scheduler.step(selection_metric)
        current_lr = float(optimizer.param_groups[0]["lr"])
        train_row["learning_rate"] = current_lr
        if val_sequences:
            val_row["learning_rate"] = current_lr

        if selection_metric < best_metric - config.min_delta:
            best_metric = float(selection_metric)
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                config.out_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                config,
                cache,
                epoch,
                history,
                best_metric,
                node_feature_normalization,
            )
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            config.out_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            config,
            cache,
            epoch,
            history,
            best_metric,
            node_feature_normalization,
        )
        if config.checkpoint_every > 0 and epoch % config.checkpoint_every == 0:
            save_checkpoint(
                config.out_dir / f"epoch_{epoch:04d}.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                config,
                cache,
                epoch,
                history,
                best_metric,
                node_feature_normalization,
            )
        write_history(history, config.out_dir)
        print_epoch(epoch, train_metrics, history[-1] if history[-1]["phase"] == "val" else None)
        epochs_completed = epoch

        if config.early_stopping_patience > 0 and epochs_without_improvement >= config.early_stopping_patience:
            break

    test_metrics = None
    if test_sequences:
        best_path = config.out_dir / "best.pt"
        if best_path.exists():
            best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(best_checkpoint["model_state"])
        test_metrics = run_sequence_epoch(
            model,
            test_sequences,
            device=device,
            optimizer=None,
            config=config,
            division_horizons=division_horizons,
            pos_weight_division=pos_weight_division,
            pos_weight_death=pos_weight_death,
            pos_weight_division_horizon=pos_weight_division_horizon,
        )
        history.append({"epoch": epochs_completed, "phase": "test", **test_metrics})
        write_history(history, config.out_dir)

    summary = {
        "model_type": "field_sequence",
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "epochs_completed": epochs_completed,
        "test_metrics": test_metrics,
        "node_dim": node_dim,
        "edge_dim": edge_dim,
        "shape_dim": shape_dim,
        "division_horizons": division_horizons,
        "train_sequences": len(train_sequences),
        "val_sequences": len(val_sequences),
        "test_sequences": len(test_sequences),
        "train_graphs": len(train_graphs),
        "val_graphs": len(val_graphs),
        "test_graphs": len(test_graphs),
        "config": jsonable(asdict(config)),
        "cache_summary": cache.get("summary", {}),
        "node_feature_normalization": jsonable(node_feature_normalization),
    }
    (config.out_dir / "run_summary.json").write_text(
        json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"history": history, "summary": summary, "model": model}


def build_sequence_model(
    config: SequenceTrainConfig,
    *,
    node_dim: int,
    edge_dim: int,
    shape_dim: int,
    num_division_horizons: int,
) -> FieldConditionedCellGNN:
    return FieldConditionedCellGNN(
        node_dim=node_dim,
        edge_dim=edge_dim,
        shape_dim=shape_dim,
        field_channels=config.field_channels,
        field_patch_radius=config.field_patch_radius,
        field_context_dim=config.field_context_dim,
        hidden_dim=config.hidden_dim,
        num_message_passing_layers=config.layers,
        dropout=config.dropout,
        num_division_horizons=num_division_horizons,
        field_update_hidden_channels=config.field_update_hidden_channels,
        field_geometry=FieldGeometry(
            origin_xy=(config.field_origin_x, config.field_origin_y),
            cell_size=config.field_cell_size,
        ),
    )


def group_graphs_by_sequence(graphs: list[Any]) -> list[list[Any]]:
    grouped: dict[str, list[Any]] = {}
    for graph in graphs:
        sequence_uid = str(getattr(graph, "sequence_uid", "unknown"))
        grouped.setdefault(sequence_uid, []).append(graph)
    return [
        sorted(items, key=lambda graph: int(getattr(graph, "frame", 0)))
        for _, items in sorted(grouped.items())
        if items
    ]


def run_sequence_epoch(
    model: FieldConditionedCellGNN,
    sequences: list[list[Any]],
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    config: SequenceTrainConfig,
    division_horizons: tuple[int, ...],
    pos_weight_division: torch.Tensor,
    pos_weight_death: torch.Tensor,
    pos_weight_division_horizon: torch.Tensor,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    loss_sums: dict[str, float] = {}
    loss_counts: dict[str, int] = {}
    total_nodes = 0
    total_graphs = 0
    start_time = time.perf_counter()
    regression_sums = {
        "pos_sq_error": 0.0,
        "pos_count": 0,
        "shape_sq_error": 0.0,
        "shape_count": 0,
        "polarization_theta_sq_error": 0.0,
        "polarization_theta_count": 0,
        "polarization_aspect_sq_error": 0.0,
        "polarization_aspect_count": 0,
    }
    event_scores: dict[str, list[torch.Tensor]] = {
        "division": [],
        "death": [],
    }
    event_targets: dict[str, list[torch.Tensor]] = {
        "division": [],
        "death": [],
    }
    for horizon in division_horizons:
        key = division_horizon_metric_name(horizon)
        event_scores[key] = []
        event_targets[key] = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for sequence in sequences:
            field = model.initial_field(
                batch_size=1,
                height=config.field_height,
                width=config.field_width,
                device=device,
                dtype=torch.float32,
            )
            accumulated_loss = None
            accumulated_steps = 0
            for step_index, graph in enumerate(sequence, start=1):
                graph = graph.to(device)
                output = model(graph, field)
                target_division_horizon, valid_division_horizon_mask = division_horizon_tensors(graph, division_horizons)
                loss, stats = cell_dynamics_loss(
                    output.cell_output,
                    target_delta_pos=graph.target_delta_pos,
                    target_delta_shape=graph.target_delta_shape,
                    target_division=graph.target_division,
                    target_death=graph.target_death,
                    valid_regression_mask=graph.valid_regression_mask,
                    valid_shape_mask=graph.valid_shape_mask,
                    valid_event_mask=graph.valid_event_mask,
                    target_division_horizon=target_division_horizon,
                    valid_division_horizon_mask=valid_division_horizon_mask,
                    pos_weight_division=pos_weight_division,
                    pos_weight_death=pos_weight_death,
                    pos_weight_division_horizon=pos_weight_division_horizon,
                    lambda_pos=config.lambda_pos,
                    lambda_shape=config.lambda_shape,
                    lambda_division=config.lambda_division,
                    lambda_death=config.lambda_death,
                    lambda_division_horizon=config.lambda_division_horizon,
                )
                if training and loss.requires_grad:
                    accumulated_loss = loss if accumulated_loss is None else accumulated_loss + loss
                    accumulated_steps += 1

                batch_counts = batch_loss_counts(graph, valid_division_horizon_mask)
                for key, value in stats.items():
                    if key == "loss_total":
                        continue
                    count = int(batch_counts.get(key, 0))
                    loss_sums.setdefault(key, 0.0)
                    loss_counts.setdefault(key, 0)
                    if count > 0:
                        loss_sums[key] += float(value.item()) * count
                        loss_counts[key] += count
                total_nodes += int(graph.num_nodes)
                total_graphs += 1
                with torch.no_grad():
                    accumulate_regression_sums(output.cell_output, graph, regression_sums)
                    collect_event_scores(output.cell_output, graph, event_scores, event_targets, division_horizons)
                field = output.field_next

                should_flush = (
                    training
                    and accumulated_loss is not None
                    and (accumulated_steps >= config.bptt_window or step_index == len(sequence))
                )
                if should_flush:
                    optimizer.zero_grad(set_to_none=True)
                    (accumulated_loss / float(accumulated_steps)).backward()
                    if config.grad_clip_norm > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                    optimizer.step()
                    accumulated_loss = None
                    accumulated_steps = 0
                    field = field.detach()
                elif not training and step_index % config.bptt_window == 0:
                    field = field.detach()

    if total_nodes == 0:
        return {}
    metrics = {
        key: (loss_sums.get(key, 0.0) / loss_counts[key] if loss_counts.get(key, 0) > 0 else 0.0)
        for key in sorted(loss_counts)
    }
    metrics["loss_total"] = (
        config.lambda_pos * metrics.get("loss_pos", 0.0)
        + config.lambda_shape * metrics.get("loss_shape", 0.0)
        + config.lambda_division * metrics.get("loss_division", 0.0)
        + config.lambda_death * metrics.get("loss_death", 0.0)
        + config.lambda_division_horizon * metrics.get("loss_division_horizon", 0.0)
    )
    if regression_sums["pos_count"] > 0:
        metrics["pos_rmse"] = float((regression_sums["pos_sq_error"] / regression_sums["pos_count"]) ** 0.5)
    if regression_sums["shape_count"] > 0:
        metrics["shape_rmse"] = float((regression_sums["shape_sq_error"] / regression_sums["shape_count"]) ** 0.5)
    metrics.update(epoch_event_metrics(event_scores, event_targets))
    add_metric_prefix_alias(metrics, source_prefix="death", alias_prefix="disappearance")
    seconds = max(time.perf_counter() - start_time, 1e-9)
    metrics["epoch_seconds"] = float(seconds)
    metrics["nodes_per_second"] = float(total_nodes / seconds)
    metrics["graphs_per_second"] = float(total_graphs / seconds)
    metrics["sequences"] = float(len(sequences))
    return metrics


def batch_loss_counts(graph: Any, valid_division_horizon_mask: torch.Tensor | None) -> dict[str, int]:
    valid_event = graph.valid_event_mask.bool() if hasattr(graph, "valid_event_mask") else None
    return {
        "loss_pos": int(graph.valid_regression_mask.bool().sum().item()) if hasattr(graph, "valid_regression_mask") else 0,
        "loss_shape": int(graph.valid_shape_mask.bool().sum().item()) if hasattr(graph, "valid_shape_mask") else 0,
        "loss_division": int(valid_event.sum().item()) if valid_event is not None else 0,
        "loss_death": int(valid_event.sum().item()) if valid_event is not None else 0,
        "loss_division_horizon": int(valid_division_horizon_mask.bool().sum().item()) if valid_division_horizon_mask is not None else 0,
    }


def write_history(history: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "history.json").write_text(json.dumps(jsonable(history), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not history:
        return
    fieldnames = sorted({key for row in history for key in row})
    with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def print_epoch(epoch: int, train_metrics: dict[str, float], val_row: dict[str, Any] | None) -> None:
    text = f"epoch={epoch} train_loss={train_metrics.get('loss_total', float('nan')):.5f}"
    if val_row is not None:
        text += f" val_loss={val_row.get('loss_total', float('nan')):.5f}"
    print(text)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    defaults = SequenceTrainConfig()
    parser = argparse.ArgumentParser(description="Train field-conditioned GNN with latent field carried across sequence frames.")
    parser.add_argument("--cache", type=Path, default=defaults.cache_path)
    parser.add_argument("--out-dir", type=Path, default=defaults.out_dir)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--hidden-dim", type=int, default=defaults.hidden_dim)
    parser.add_argument("--layers", type=int, default=defaults.layers)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--grad-clip-norm", type=float, default=defaults.grad_clip_norm)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--lambda-pos", type=float, default=defaults.lambda_pos)
    parser.add_argument("--lambda-shape", type=float, default=defaults.lambda_shape)
    parser.add_argument("--lambda-division", type=float, default=defaults.lambda_division)
    parser.add_argument("--lambda-death", type=float, default=defaults.lambda_death)
    parser.add_argument("--lambda-division-horizon", type=float, default=defaults.lambda_division_horizon)
    parser.add_argument("--division-horizons", type=int, nargs="*", default=list(defaults.division_horizons))
    parser.add_argument("--max-pos-weight", type=float, default=defaults.max_pos_weight)
    parser.add_argument("--scheduler-patience", type=int, default=defaults.scheduler_patience)
    parser.add_argument("--scheduler-factor", type=float, default=defaults.scheduler_factor)
    parser.add_argument("--min-learning-rate", type=float, default=defaults.min_learning_rate)
    parser.add_argument("--early-stopping-patience", type=int, default=defaults.early_stopping_patience)
    parser.add_argument("--min-delta", type=float, default=defaults.min_delta)
    parser.add_argument("--checkpoint-every", type=int, default=defaults.checkpoint_every)
    parser.add_argument("--resume-from", type=Path, default=defaults.resume_from)
    parser.add_argument("--no-normalize-node-features", dest="normalize_node_features", action="store_false", default=True)
    parser.add_argument("--normalization-epsilon", type=float, default=defaults.normalization_epsilon)
    parser.add_argument("--field-channels", type=int, default=defaults.field_channels)
    parser.add_argument("--field-height", type=int, default=defaults.field_height)
    parser.add_argument("--field-width", type=int, default=defaults.field_width)
    parser.add_argument("--field-patch-radius", type=int, default=defaults.field_patch_radius)
    parser.add_argument("--field-context-dim", type=int, default=defaults.field_context_dim)
    parser.add_argument("--field-update-hidden-channels", type=int, default=defaults.field_update_hidden_channels)
    parser.add_argument("--field-cell-size", type=float, default=defaults.field_cell_size)
    parser.add_argument("--field-origin-x", type=float, default=defaults.field_origin_x)
    parser.add_argument("--field-origin-y", type=float, default=defaults.field_origin_y)
    parser.add_argument("--bptt-window", type=int, default=defaults.bptt_window)
    parser.add_argument("--no-field-auto-geometry", dest="field_auto_geometry", action="store_false", default=True)
    parser.add_argument("--min-field-coverage", type=float, default=defaults.min_field_coverage)
    parser.add_argument("--strict-field-coverage", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    config = SequenceTrainConfig(
        cache_path=args.cache,
        out_dir=args.out_dir,
        epochs=args.epochs,
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
        lambda_division=args.lambda_division,
        lambda_death=args.lambda_death,
        lambda_division_horizon=args.lambda_division_horizon,
        division_horizons=tuple(args.division_horizons),
        max_pos_weight=args.max_pos_weight,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        min_learning_rate=args.min_learning_rate,
        early_stopping_patience=args.early_stopping_patience,
        min_delta=args.min_delta,
        checkpoint_every=args.checkpoint_every,
        resume_from=args.resume_from,
        normalize_node_features=args.normalize_node_features,
        normalization_epsilon=args.normalization_epsilon,
        field_channels=args.field_channels,
        field_height=args.field_height,
        field_width=args.field_width,
        field_patch_radius=args.field_patch_radius,
        field_context_dim=args.field_context_dim,
        field_update_hidden_channels=args.field_update_hidden_channels,
        field_cell_size=args.field_cell_size,
        field_origin_x=args.field_origin_x,
        field_origin_y=args.field_origin_y,
        bptt_window=args.bptt_window,
        field_auto_geometry=args.field_auto_geometry,
        min_field_coverage=args.min_field_coverage,
        strict_field_coverage=args.strict_field_coverage,
    )
    result = train_from_cache(config)
    print(json.dumps(jsonable(result["summary"]), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
