from __future__ import annotations

import argparse
import csv
import json
import random
import time
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch_geometric.loader import DataLoader

from .dataset_cache import DEFAULT_CACHE_PATH, build_and_save_graph_cache, load_graph_cache
from .gnn_model import CellInteractionGNN, cell_dynamics_loss
from .hybrid_field_gnn_model import FieldConditionedCellGNN, FieldConditionedGNNOutput
from .spatial_field import FieldGeometry, field_coverage


DEFAULT_TOP_KS = (10, 20, 50, 100)
FIELD_CACHE_REBUILD_HINT = (
    "field models require physical coordinates data.pos_xy; rebuild graph cache, e.g. "
    "python -m Real_game_of_life.GNN.dataset_cache --out Real_game_of_life/GNN/cache/frame_graphs_dynamic_v2.pt"
)


@dataclass(frozen=True)
class TrainConfig:
    cache_path: Path = DEFAULT_CACHE_PATH
    out_dir: Path = Path(__file__).resolve().parent / "runs" / "one_step_baseline"
    epochs: int = 20
    batch_size: int = 16
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
    num_workers: int = 0
    scheduler_patience: int = 20
    scheduler_factor: float = 0.5
    min_learning_rate: float = 1e-6
    early_stopping_patience: int = 0
    min_delta: float = 0.0
    checkpoint_every: int = 0
    resume_from: Path | None = None
    amp: bool = False
    normalize_node_features: bool = True
    normalization_epsilon: float = 1e-6
    model_type: str = "gnn"
    field_channels: int = 16
    field_height: int = 128
    field_width: int = 128
    field_patch_radius: int = 1
    field_context_dim: int = 64
    field_update_hidden_channels: int = 64
    field_cell_size: float = 4.0
    field_origin_x: float = 0.0
    field_origin_y: float = 0.0
    field_auto_geometry: bool = True
    min_field_coverage: float = 0.95
    strict_field_coverage: bool = False


def train_from_cache(config: TrainConfig) -> dict[str, Any]:
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
    if config.model_type == "field_gnn":
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

    device = resolve_device(config.device)
    node_dim = int(graphs[0].x.size(-1))
    edge_dim = int(graphs[0].edge_attr.size(-1))
    shape_dim = infer_shape_dim(graphs)
    division_horizons = infer_division_horizons(graphs, config.division_horizons)
    model = build_model(
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
    use_amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    loaders = {
        "train": make_loader(train_graphs, config, shuffle=True),
        "val": make_loader(val_graphs, config, shuffle=False) if val_graphs else None,
        "test": make_loader(test_graphs, config, shuffle=False) if test_graphs else None,
    }

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
        train_metrics = run_epoch(
            model,
            loaders["train"],
            device=device,
            optimizer=optimizer,
            config=config,
            pos_weight_division=pos_weight_division,
            pos_weight_death=pos_weight_death,
            pos_weight_division_horizon=pos_weight_division_horizon,
            division_horizons=division_horizons,
            scaler=scaler,
        )
        train_row = {"epoch": epoch, "phase": "train", **train_metrics}
        history.append(train_row)

        if loaders["val"] is not None:
            val_metrics = run_epoch(
                model,
                loaders["val"],
                device=device,
                optimizer=None,
                config=config,
                pos_weight_division=pos_weight_division,
                pos_weight_death=pos_weight_death,
                pos_weight_division_horizon=pos_weight_division_horizon,
                division_horizons=division_horizons,
                scaler=None,
            )
            val_row = {"epoch": epoch, "phase": "val", **val_metrics}
            history.append(val_row)
            selection_metric = val_metrics["loss_total"]
        else:
            selection_metric = train_metrics["loss_total"]

        scheduler.step(selection_metric)
        current_lr = float(optimizer.param_groups[0]["lr"])
        train_row["learning_rate"] = current_lr
        if loaders["val"] is not None:
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
            print(
                "early_stopping "
                f"epoch={epoch} best_epoch={best_epoch} best_metric={best_metric:.6f}"
            )
            break

    test_metrics = None
    if loaders["test"] is not None:
        best_path = config.out_dir / "best.pt"
        if best_path.exists():
            best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(best_checkpoint["model_state"])
        test_metrics = run_epoch(
            model,
            loaders["test"],
            device=device,
            optimizer=None,
            config=config,
            pos_weight_division=pos_weight_division,
            pos_weight_death=pos_weight_death,
            pos_weight_division_horizon=pos_weight_division_horizon,
            division_horizons=division_horizons,
            scaler=None,
        )
        history.append({"epoch": epochs_completed, "phase": "test", **test_metrics})
        write_history(history, config.out_dir)

    run_summary = {
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "epochs_completed": epochs_completed,
        "test_metrics": test_metrics,
        "node_dim": node_dim,
        "edge_dim": edge_dim,
        "shape_dim": shape_dim,
        "division_horizons": division_horizons,
        "model_type": config.model_type,
        "train_graphs": len(train_graphs),
        "val_graphs": len(val_graphs),
        "test_graphs": len(test_graphs),
        "config": jsonable(asdict(config)),
        "cache_summary": cache.get("summary", {}),
        "node_feature_normalization": jsonable(node_feature_normalization),
    }
    (config.out_dir / "run_summary.json").write_text(
        json.dumps(jsonable(run_summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"history": history, "summary": run_summary, "model": model}


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    config: TrainConfig,
    pos_weight_division: torch.Tensor,
    pos_weight_death: torch.Tensor,
    pos_weight_division_horizon: torch.Tensor,
    division_horizons: tuple[int, ...],
    scaler: torch.amp.GradScaler | None,
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
        for batch in loader:
            batch = batch.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=bool(scaler and scaler.is_enabled())):
                output = forward_model_for_batch(model, batch, config=config, device=device)
                target_division_horizon, valid_division_horizon_mask = division_horizon_tensors(batch, division_horizons)
                loss, stats = cell_dynamics_loss(
                    output,
                    target_delta_pos=batch.target_delta_pos,
                    target_delta_shape=batch.target_delta_shape,
                    target_division=batch.target_division,
                    target_death=batch.target_death,
                    valid_regression_mask=batch.valid_regression_mask,
                    valid_shape_mask=batch.valid_shape_mask,
                    valid_event_mask=batch.valid_event_mask,
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

            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    if config.grad_clip_norm > 0:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if config.grad_clip_norm > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                    optimizer.step()

            weight = int(batch.num_nodes)
            total_nodes += weight
            total_graphs += int(getattr(batch, "num_graphs", 1))
            batch_counts = batch_loss_counts(batch, valid_division_horizon_mask)
            for key, value in stats.items():
                if key == "loss_total":
                    continue
                count = int(batch_counts.get(key, 0))
                loss_sums.setdefault(key, 0.0)
                loss_counts.setdefault(key, 0)
                if count > 0:
                    loss_sums[key] += float(value.item()) * count
                    loss_counts[key] += count

            with torch.no_grad():
                accumulate_regression_sums(output, batch, regression_sums)
                collect_event_scores(output, batch, event_scores, event_targets, division_horizons)

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
    return metrics


def build_model(
    config: TrainConfig,
    *,
    node_dim: int,
    edge_dim: int,
    shape_dim: int,
    num_division_horizons: int,
) -> nn.Module:
    if config.model_type == "gnn":
        return CellInteractionGNN(
            node_dim=node_dim,
            edge_dim=edge_dim,
            shape_dim=shape_dim,
            hidden_dim=config.hidden_dim,
            num_message_passing_layers=config.layers,
            dropout=config.dropout,
            num_division_horizons=num_division_horizons,
        )
    if config.model_type == "field_gnn":
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
            train_field_update=False,
        )
    raise ValueError(f"Unsupported model_type: {config.model_type!r}")


def require_field_pos_xy(graphs: list[Any]) -> None:
    missing = [index for index, graph in enumerate(graphs) if not hasattr(graph, "pos_xy") or graph.pos_xy is None]
    if missing:
        example = ", ".join(str(index) for index in missing[:5])
        raise ValueError(f"{FIELD_CACHE_REBUILD_HINT}. Missing pos_xy in graph indices: {example}")


def prepare_field_geometry_config(
    config: TrainConfig,
    *,
    train_graphs: list[Any],
    split_graphs: dict[str, list[Any]],
) -> TrainConfig:
    if config.field_auto_geometry:
        pos_xy = _cat_pos_xy(train_graphs)
        if pos_xy.numel() > 0:
            minimum = pos_xy.min(dim=0).values
            maximum = pos_xy.max(dim=0).values
            span = (maximum - minimum).clamp_min(1e-6)
            cell_size = max(
                float(span[0].item()) / max(config.field_width - 1, 1),
                float(span[1].item()) / max(config.field_height - 1, 1),
                1e-6,
            )
            config = replace(
                config,
                field_origin_x=float(minimum[0].item()),
                field_origin_y=float(minimum[1].item()),
                field_cell_size=cell_size,
            )
    report_field_coverage(
        split_graphs,
        field_height=config.field_height,
        field_width=config.field_width,
        geometry=FieldGeometry(
            origin_xy=(config.field_origin_x, config.field_origin_y),
            cell_size=config.field_cell_size,
        ),
        min_coverage=config.min_field_coverage,
        strict=config.strict_field_coverage,
    )
    return config


def report_field_coverage(
    split_graphs: dict[str, list[Any]],
    *,
    field_height: int,
    field_width: int,
    geometry: FieldGeometry,
    min_coverage: float,
    strict: bool,
) -> dict[str, float]:
    coverages: dict[str, float] = {}
    for name, graphs in split_graphs.items():
        if not graphs:
            continue
        coverage = field_coverage(
            _cat_pos_xy(graphs),
            field_height=field_height,
            field_width=field_width,
            geometry=geometry,
        )
        coverages[name] = coverage
        message = (
            f"field coverage {name}={coverage:.3f} "
            f"(min={min_coverage:.3f}, origin={geometry.origin_xy}, cell_size={geometry.cell_size:.6g}, "
            f"size={field_width}x{field_height})"
        )
        print(message)
        if coverage < min_coverage:
            if strict:
                raise ValueError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
    return coverages


def _cat_pos_xy(graphs: list[Any]) -> torch.Tensor:
    tensors = [graph.pos_xy.detach().float().cpu() for graph in graphs if hasattr(graph, "pos_xy") and graph.pos_xy is not None]
    if not tensors:
        return torch.empty((0, 2), dtype=torch.float32)
    return torch.cat(tensors, dim=0)


def forward_model_for_batch(
    model: nn.Module,
    batch: Any,
    *,
    config: TrainConfig,
    device: torch.device,
):
    if config.model_type == "field_gnn":
        if not isinstance(model, FieldConditionedCellGNN):
            raise TypeError("model_type='field_gnn' requires FieldConditionedCellGNN")
        field = model.initial_field(
            batch_size=int(getattr(batch, "num_graphs", 1)),
            height=config.field_height,
            width=config.field_width,
            device=device,
            dtype=torch.float32,
        )
        output = model(batch, field)
        if not isinstance(output, FieldConditionedGNNOutput):
            raise TypeError("FieldConditionedCellGNN returned an unexpected output type")
        return output.cell_output
    return model(batch)


def fit_node_feature_normalization(graphs: list[Any], *, epsilon: float) -> dict[str, torch.Tensor]:
    x = torch.cat([graph.x.detach().cpu().float() for graph in graphs], dim=0)
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False).clamp_min(float(epsilon))
    return {"mean": mean, "std": std}


def apply_node_feature_normalization(graphs: list[Any], stats: dict[str, torch.Tensor]) -> None:
    mean = stats["mean"].float()
    std = stats["std"].float()
    for graph in graphs:
        graph.x = (graph.x.float() - mean) / std


def batch_loss_counts(batch: Any, valid_division_horizon_mask: torch.Tensor | None) -> dict[str, int]:
    valid_event = batch.valid_event_mask.bool() if hasattr(batch, "valid_event_mask") else None
    return {
        "loss_pos": int(batch.valid_regression_mask.bool().sum().item()) if hasattr(batch, "valid_regression_mask") else 0,
        "loss_shape": int(batch.valid_shape_mask.bool().sum().item()) if hasattr(batch, "valid_shape_mask") else 0,
        "loss_division": int(valid_event.sum().item()) if valid_event is not None else 0,
        "loss_death": int(valid_event.sum().item()) if valid_event is not None else 0,
        "loss_division_horizon": int(valid_division_horizon_mask.bool().sum().item()) if valid_division_horizon_mask is not None else 0,
    }


def add_metric_prefix_alias(metrics: dict[str, float], *, source_prefix: str, alias_prefix: str) -> None:
    source = f"{source_prefix}_"
    for key, value in list(metrics.items()):
        if key.startswith(source):
            metrics[f"{alias_prefix}_{key[len(source):]}"] = value


def accumulate_regression_sums(output, batch, sums: dict[str, float | int]) -> None:
    valid_reg = batch.valid_regression_mask.bool()
    if int(valid_reg.sum()) > 0:
        error = output.delta_pos[valid_reg] - batch.target_delta_pos[valid_reg]
        sums["pos_sq_error"] = float(sums["pos_sq_error"]) + float((error ** 2).sum().item())
        sums["pos_count"] = int(sums["pos_count"]) + int(error.numel())
    valid_shape = batch.valid_shape_mask.bool()
    if int(valid_shape.sum()) > 0:
        error = output.delta_shape[valid_shape] - batch.target_delta_shape[valid_shape]
        sums["shape_sq_error"] = float(sums["shape_sq_error"]) + float((error ** 2).sum().item())
        sums["shape_count"] = int(sums["shape_count"]) + int(error.numel())


def regression_batch_metrics(output, batch) -> dict[str, float]:
    metrics: dict[str, float] = {}
    valid_reg = batch.valid_regression_mask.bool()
    if int(valid_reg.sum()) > 0:
        pos_mse = ((output.delta_pos[valid_reg] - batch.target_delta_pos[valid_reg]) ** 2).mean()
        metrics["pos_rmse"] = float(torch.sqrt(pos_mse).item())
    valid_shape = batch.valid_shape_mask.bool()
    if int(valid_shape.sum()) > 0:
        shape_mse = ((output.delta_shape[valid_shape] - batch.target_delta_shape[valid_shape]) ** 2).mean()
        metrics["shape_rmse"] = float(torch.sqrt(shape_mse).item())
    return metrics


def collect_event_scores(
    output,
    batch,
    scores: dict[str, list[torch.Tensor]],
    targets: dict[str, list[torch.Tensor]],
    division_horizons: tuple[int, ...],
) -> None:
    valid_event = batch.valid_event_mask.bool()
    if int(valid_event.sum()) > 0:
        scores["division"].append(torch.sigmoid(output.division_logits[valid_event]).detach().cpu())
        targets["division"].append(batch.target_division[valid_event].detach().float().cpu())
        scores["death"].append(torch.sigmoid(output.death_logits[valid_event]).detach().cpu())
        targets["death"].append(batch.target_death[valid_event].detach().float().cpu())

    if output.division_horizon_logits is None:
        return
    for index, horizon in enumerate(division_horizons):
        target_attr = f"target_division_within_{horizon}"
        mask_attr = f"valid_division_within_{horizon}"
        if not hasattr(batch, target_attr) or not hasattr(batch, mask_attr):
            continue
        mask = getattr(batch, mask_attr).bool()
        if int(mask.sum()) == 0:
            continue
        key = division_horizon_metric_name(horizon)
        scores[key].append(torch.sigmoid(output.division_horizon_logits[:, index][mask]).detach().cpu())
        targets[key].append(getattr(batch, target_attr)[mask].detach().float().cpu())


def epoch_event_metrics(
    scores: dict[str, list[torch.Tensor]],
    targets: dict[str, list[torch.Tensor]],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name in sorted(scores):
        if not scores[name]:
            continue
        score = torch.cat(scores[name])
        target = torch.cat(targets[name]).bool()
        metrics.update(binary_classification_metrics(name, score, target))
    return metrics


def binary_classification_metrics(prefix: str, score: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = score >= 0.5
    tp = int((pred & target).sum().item())
    fp = int((pred & ~target).sum().item())
    fn = int((~pred & target).sum().item())
    tn = int((~pred & ~target).sum().item())
    total = tp + fp + fn + tn

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0
    positive_rate = (tp + fn) / total if total > 0 else 0.0
    predicted_positive_rate = (tp + fp) / total if total > 0 else 0.0
    average_precision = binary_average_precision(score, target)

    metrics = {
        f"{prefix}_acc": float(accuracy),
        f"{prefix}_precision": float(precision),
        f"{prefix}_recall": float(recall),
        f"{prefix}_f1": float(f1),
        f"{prefix}_ap": float(average_precision),
        f"{prefix}_positive_rate": float(positive_rate),
        f"{prefix}_predicted_positive_rate": float(predicted_positive_rate),
        f"{prefix}_tp": float(tp),
        f"{prefix}_fp": float(fp),
        f"{prefix}_fn": float(fn),
        f"{prefix}_tn": float(tn),
    }
    metrics.update(top_k_metrics(prefix, score, target, DEFAULT_TOP_KS))
    return metrics


def binary_average_precision(score: torch.Tensor, target: torch.Tensor) -> float:
    target = target.bool()
    positives = int(target.sum().item())
    if positives == 0:
        return 0.0
    order = torch.argsort(score, descending=True)
    sorted_target = target[order].float()
    true_positives = torch.cumsum(sorted_target, dim=0)
    ranks = torch.arange(1, sorted_target.numel() + 1, dtype=torch.float32)
    precision_at_rank = true_positives / ranks
    ap = (precision_at_rank * sorted_target).sum() / positives
    return float(ap.item())


def top_k_metrics(
    prefix: str,
    score: torch.Tensor,
    target: torch.Tensor,
    top_ks: Iterable[int] = DEFAULT_TOP_KS,
) -> dict[str, float]:
    score = score.detach().flatten().float().cpu()
    target = target.detach().flatten().bool().cpu()
    total = int(target.numel())
    positives = int(target.sum().item())
    metrics: dict[str, float] = {}
    if total == 0:
        return metrics

    order = torch.argsort(score, descending=True)
    for requested_k in top_ks:
        requested_k = int(requested_k)
        k = min(requested_k, total)
        if k <= 0:
            continue
        top_target = target[order[:k]]
        hits = int(top_target.sum().item())
        metrics[f"{prefix}_top{requested_k}_effective_k"] = float(k)
        metrics[f"{prefix}_top{requested_k}_hits"] = float(hits)
        metrics[f"{prefix}_top{requested_k}_precision"] = float(hits / k)
        metrics[f"{prefix}_top{requested_k}_recall"] = float(hits / positives) if positives > 0 else 0.0
    return metrics


def make_loader(graphs: list[Any], config: TrainConfig, *, shuffle: bool) -> DataLoader:
    return DataLoader(
        graphs,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
    )


def infer_shape_dim(graphs: list[Any]) -> int:
    for graph in graphs:
        target = getattr(graph, "target_delta_shape", None)
        if target is not None and target.dim() == 2 and target.size(-1) > 0:
            return int(target.size(-1))
    raise ValueError("Graphs do not contain non-empty target_delta_shape.")


def infer_division_horizons(graphs: list[Any], requested_horizons: tuple[int, ...]) -> tuple[int, ...]:
    available: list[int] = []
    for horizon in requested_horizons:
        target_attr = f"target_division_within_{horizon}"
        mask_attr = f"valid_division_within_{horizon}"
        if any(hasattr(graph, target_attr) and hasattr(graph, mask_attr) for graph in graphs):
            available.append(int(horizon))
    return tuple(available)


def division_horizon_tensors(batch: Any, horizons: tuple[int, ...]) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not horizons:
        return None, None
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for horizon in horizons:
        target_attr = f"target_division_within_{horizon}"
        mask_attr = f"valid_division_within_{horizon}"
        if not hasattr(batch, target_attr) or not hasattr(batch, mask_attr):
            raise ValueError(f"Batch is missing division horizon target or mask for horizon={horizon}")
        targets.append(getattr(batch, target_attr).float())
        masks.append(getattr(batch, mask_attr).bool())
    return torch.stack(targets, dim=1), torch.stack(masks, dim=1)


def division_horizon_metric_name(horizon: int) -> str:
    return f"division_h{int(horizon)}"


def target_pos_weight(graphs: list[Any], target_attr: str, mask_attr: str, max_pos_weight: float) -> torch.Tensor:
    positives = 0.0
    total = 0.0
    for graph in graphs:
        target = getattr(graph, target_attr)
        mask = getattr(graph, mask_attr).bool()
        positives += float(target[mask].sum().item())
        total += float(mask.sum().item())
    negatives = max(0.0, total - positives)
    if positives <= 0:
        weight = 1.0
    else:
        weight = negatives / positives
    return torch.tensor(min(float(weight), float(max_pos_weight)), dtype=torch.float32)


def target_pos_weights_for_division_horizons(
    graphs: list[Any],
    horizons: tuple[int, ...],
    max_pos_weight: float,
) -> torch.Tensor:
    weights = [
        target_pos_weight(
            graphs,
            f"target_division_within_{horizon}",
            f"valid_division_within_{horizon}",
            max_pos_weight,
        )
        for horizon in horizons
    ]
    if not weights:
        return torch.empty(0, dtype=torch.float32)
    return torch.stack(weights)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    scaler: torch.amp.GradScaler,
    config: TrainConfig,
    cache: dict[str, Any],
    epoch: int,
    history: list[dict[str, Any]],
    best_metric: float,
    node_feature_normalization: dict[str, torch.Tensor] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_metric": best_metric,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "train_config": jsonable(asdict(config)),
            "cache_summary": cache.get("summary", {}),
            "dataset_config": cache.get("dataset_config", {}),
            "split_config": cache.get("split_config", {}),
            "history": history,
            "node_feature_normalization": jsonable(node_feature_normalization),
        },
        path,
    )


def load_training_state(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> tuple[int, int, float, list[dict[str, Any]]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    if "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    if "scaler_state" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state"])
    epoch = int(checkpoint.get("epoch", 0))
    best_metric = float(checkpoint.get("best_metric", float("inf")))
    history = list(checkpoint.get("history", []))
    best_epoch = 0
    for row in history:
        if row.get("phase") == "val" and float(row.get("loss_total", float("inf"))) == best_metric:
            best_epoch = int(row.get("epoch", 0))
            break
    if best_epoch == 0:
        best_epoch = epoch
    return epoch + 1, best_epoch, best_metric, history


def write_history(history: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "history.json"
    csv_path = out_dir / "history.csv"
    json_path.write_text(json.dumps(jsonable(history), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not history:
        return
    fieldnames = sorted({key for row in history for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def print_epoch(epoch: int, train_metrics: dict[str, float], val_row: dict[str, Any] | None) -> None:
    train_loss = train_metrics.get("loss_total", float("nan"))
    text = f"epoch={epoch} train_loss={train_loss:.5f}"
    if "pos_rmse" in train_metrics:
        text += f" train_pos_rmse={train_metrics['pos_rmse']:.5f}"
    if val_row is not None:
        text += f" val_loss={val_row.get('loss_total', float('nan')):.5f}"
    print(text)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    return value


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one-step cell interaction GNN from a graph cache.")
    parser.add_argument("--cache", type=Path, default=TrainConfig().cache_path)
    parser.add_argument("--out-dir", type=Path, default=TrainConfig().out_dir)
    parser.add_argument("--epochs", type=int, default=TrainConfig().epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig().batch_size)
    parser.add_argument("--hidden-dim", type=int, default=TrainConfig().hidden_dim)
    parser.add_argument("--layers", type=int, default=TrainConfig().layers)
    parser.add_argument("--dropout", type=float, default=TrainConfig().dropout)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig().learning_rate)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig().weight_decay)
    parser.add_argument("--device", default=TrainConfig().device)
    parser.add_argument("--seed", type=int, default=TrainConfig().seed)
    parser.add_argument("--lambda-pos", type=float, default=TrainConfig().lambda_pos)
    parser.add_argument("--lambda-shape", type=float, default=TrainConfig().lambda_shape)
    parser.add_argument("--lambda-division", type=float, default=TrainConfig().lambda_division)
    parser.add_argument("--lambda-death", type=float, default=TrainConfig().lambda_death)
    parser.add_argument("--lambda-division-horizon", type=float, default=TrainConfig().lambda_division_horizon)
    parser.add_argument("--division-horizons", type=int, nargs="*", default=list(TrainConfig().division_horizons))
    parser.add_argument("--max-pos-weight", type=float, default=TrainConfig().max_pos_weight)
    parser.add_argument("--grad-clip-norm", type=float, default=TrainConfig().grad_clip_norm)
    parser.add_argument("--num-workers", type=int, default=TrainConfig().num_workers)
    parser.add_argument("--scheduler-patience", type=int, default=TrainConfig().scheduler_patience)
    parser.add_argument("--scheduler-factor", type=float, default=TrainConfig().scheduler_factor)
    parser.add_argument("--min-learning-rate", type=float, default=TrainConfig().min_learning_rate)
    parser.add_argument("--early-stopping-patience", type=int, default=TrainConfig().early_stopping_patience)
    parser.add_argument("--min-delta", type=float, default=TrainConfig().min_delta)
    parser.add_argument("--checkpoint-every", type=int, default=TrainConfig().checkpoint_every)
    parser.add_argument("--resume-from", type=Path, default=TrainConfig().resume_from)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-normalize-node-features", dest="normalize_node_features", action="store_false", default=True)
    parser.add_argument("--normalization-epsilon", type=float, default=TrainConfig().normalization_epsilon)
    parser.add_argument("--model-type", choices=("gnn", "field_gnn"), default=TrainConfig().model_type)
    parser.add_argument("--field-channels", type=int, default=TrainConfig().field_channels)
    parser.add_argument("--field-height", type=int, default=TrainConfig().field_height)
    parser.add_argument("--field-width", type=int, default=TrainConfig().field_width)
    parser.add_argument("--field-patch-radius", type=int, default=TrainConfig().field_patch_radius)
    parser.add_argument("--field-context-dim", type=int, default=TrainConfig().field_context_dim)
    parser.add_argument("--field-update-hidden-channels", type=int, default=TrainConfig().field_update_hidden_channels)
    parser.add_argument("--field-cell-size", type=float, default=TrainConfig().field_cell_size)
    parser.add_argument("--field-origin-x", type=float, default=TrainConfig().field_origin_x)
    parser.add_argument("--field-origin-y", type=float, default=TrainConfig().field_origin_y)
    parser.add_argument("--no-field-auto-geometry", dest="field_auto_geometry", action="store_false", default=True)
    parser.add_argument("--min-field-coverage", type=float, default=TrainConfig().min_field_coverage)
    parser.add_argument("--strict-field-coverage", action="store_true")
    parser.add_argument("--build-cache-if-missing", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.build_cache_if_missing and not args.cache.exists():
        build_and_save_graph_cache(args.cache)

    config = TrainConfig(
        cache_path=args.cache,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
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
        num_workers=args.num_workers,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        min_learning_rate=args.min_learning_rate,
        early_stopping_patience=args.early_stopping_patience,
        min_delta=args.min_delta,
        checkpoint_every=args.checkpoint_every,
        resume_from=args.resume_from,
        amp=args.amp,
        normalize_node_features=args.normalize_node_features,
        normalization_epsilon=args.normalization_epsilon,
        model_type=args.model_type,
        field_channels=args.field_channels,
        field_height=args.field_height,
        field_width=args.field_width,
        field_patch_radius=args.field_patch_radius,
        field_context_dim=args.field_context_dim,
        field_update_hidden_channels=args.field_update_hidden_channels,
        field_cell_size=args.field_cell_size,
        field_origin_x=args.field_origin_x,
        field_origin_y=args.field_origin_y,
        field_auto_geometry=args.field_auto_geometry,
        min_field_coverage=args.min_field_coverage,
        strict_field_coverage=args.strict_field_coverage,
    )
    result = train_from_cache(config)
    print(json.dumps(jsonable(result["summary"]), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
