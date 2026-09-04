from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from .dataset_cache import DEFAULT_CACHE_PATH, load_graph_cache
from .gnn_model import CellInteractionGNN
from .graph_dataset import POLARIZATION_ANGLE_COLUMN, POLARIZATION_MAGNITUDE_COLUMN, wrap_nematic_delta
from .rollout import differentiable_prediction_to_next_graph
from .train_one_step import apply_node_feature_normalization, infer_division_horizons, infer_shape_dim, jsonable, resolve_device
from .train_rollout_bptt import group_graphs_by_sequence, match_next_graph_indices, rollout_config_from_cache


def evaluate_model_rollout(
    model: nn.Module,
    sequences: Sequence[Sequence[Any]],
    *,
    horizons: Sequence[int],
    rollout_config: Any,
    device: torch.device,
    node_feature_normalization: dict[str, Any] | None = None,
    match_tolerance: float = 1e-4,
) -> list[dict[str, float | int]]:
    """Evaluate autoregressive rollout state errors at requested horizons."""

    horizons = tuple(sorted({int(value) for value in horizons if int(value) > 0}))
    if not horizons:
        raise ValueError("At least one positive horizon is required.")
    max_horizon = max(horizons)
    accumulators = {
        horizon: {
            "position_errors": [],
            "position_sq_errors": [],
            "shape_sq_errors": [],
            "valid_shapes": 0,
            "polarization_theta_sq_errors": [],
            "polarization_aspect_sq_errors": [],
            "matched_nodes": 0,
        }
        for horizon in horizons
    }

    model.eval()
    with torch.no_grad():
        for sequence in sequences:
            sequence = [graph.to(device) for graph in sequence]
            for start_index in range(max(0, len(sequence) - 1)):
                rollout_graph = sequence[start_index]
                gt_graph = sequence[start_index]
                gt_indices = torch.arange(int(rollout_graph.num_nodes), dtype=torch.long)
                for offset in range(1, min(max_horizon, len(sequence) - start_index - 1) + 1):
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
                    if offset in accumulators:
                        _accumulate_horizon_metrics(
                            accumulators[offset],
                            rollout_graph,
                            next_gt_graph,
                            gt_indices,
                            node_feature_normalization=node_feature_normalization,
                        )
                    gt_graph = next_gt_graph

    return [_finalize_horizon_metrics(horizon, accumulators[horizon]) for horizon in horizons]


def compare_checkpoints(
    *,
    cache_path: Path,
    checkpoints: Sequence[Path],
    labels: Sequence[str] | None,
    horizons: Sequence[int],
    device_name: str,
    match_tolerance: float,
) -> list[dict[str, Any]]:
    cache = load_graph_cache(cache_path)
    graphs = cache["graphs"]
    test_graphs = [graphs[index] for index in cache["splits"].get("test", [])]
    if not test_graphs:
        raise ValueError("Test split is empty.")
    test_sequences = group_graphs_by_sequence(test_graphs)
    rollout_config = rollout_config_from_cache(cache)
    device = resolve_device(device_name)
    labels = list(labels) if labels is not None else [path.stem for path in checkpoints]
    if len(labels) != len(checkpoints):
        raise ValueError("labels length must match checkpoints length.")

    rows: list[dict[str, Any]] = []
    for label, checkpoint_path in zip(labels, checkpoints):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        normalization = normalization_tensors(checkpoint.get("node_feature_normalization"))
        model_graphs = _clone_graphs(test_graphs)
        if normalization is not None:
            apply_node_feature_normalization(model_graphs, normalization)
        model_sequences = group_graphs_by_sequence(model_graphs)
        model = _build_model_for_checkpoint(checkpoint, model_graphs).to(device)
        load_result = model.load_state_dict(checkpoint["model_state"], strict=False)
        if load_result.missing_keys or load_result.unexpected_keys:
            print(
                f"{label}: checkpoint architecture mismatch (loaded non-strict) -- "
                f"missing={load_result.missing_keys} unexpected={load_result.unexpected_keys}. "
                "Heads absent from the checkpoint keep randomly-initialized weights; "
                "metrics for those heads (e.g. polarization_*_rmse on pre-polarization checkpoints) are meaningless."
            )
        metrics = evaluate_model_rollout(
            model,
            model_sequences,
            horizons=horizons,
            rollout_config=rollout_config,
            device=device,
            node_feature_normalization=normalization,
            match_tolerance=match_tolerance,
        )
        rows.extend({"model": label, "checkpoint": str(checkpoint_path), **row} for row in metrics)
    return rows


def _accumulate_horizon_metrics(
    accumulator: dict[str, Any],
    predicted_graph: Any,
    gt_graph: Any,
    gt_indices: torch.Tensor,
    *,
    node_feature_normalization: dict[str, Any] | None,
) -> None:
    valid = gt_indices >= 0
    count = int(valid.sum().item())
    if count == 0:
        return
    gt_indices = gt_indices[valid].to(device=gt_graph.x.device)
    pred_pos = predicted_graph.pos_xy[valid]
    gt_pos = gt_graph.pos_xy[gt_indices].to(device=pred_pos.device, dtype=pred_pos.dtype)
    position_delta = pred_pos - gt_pos
    position_errors = torch.linalg.norm(position_delta, dim=1)
    accumulator["position_errors"].extend(position_errors.detach().cpu().tolist())
    accumulator["position_sq_errors"].extend((position_delta ** 2).sum(dim=1).detach().cpu().tolist())
    accumulator["matched_nodes"] += count

    feature_columns = tuple(str(column) for column in predicted_graph.node_feature_columns)
    pred_physical = _denormalize_x(predicted_graph.x, node_feature_normalization)
    gt_physical = _denormalize_x(gt_graph.x, node_feature_normalization).to(
        device=pred_physical.device,
        dtype=pred_physical.dtype,
    )

    polarization_columns = tuple(str(column) for column in getattr(predicted_graph, "polarization_columns", []))
    if polarization_columns:
        if POLARIZATION_ANGLE_COLUMN in polarization_columns and POLARIZATION_ANGLE_COLUMN in feature_columns:
            angle_index = feature_columns.index(POLARIZATION_ANGLE_COLUMN)
            pred_angle = pred_physical[valid][:, angle_index]
            gt_angle = gt_physical[gt_indices][:, angle_index]
            angle_error = wrap_nematic_delta((pred_angle - gt_angle).detach().cpu().numpy())
            accumulator["polarization_theta_sq_errors"].extend((angle_error ** 2).tolist())
        if POLARIZATION_MAGNITUDE_COLUMN in polarization_columns and POLARIZATION_MAGNITUDE_COLUMN in feature_columns:
            aspect_index = feature_columns.index(POLARIZATION_MAGNITUDE_COLUMN)
            pred_aspect = pred_physical[valid][:, aspect_index]
            gt_aspect = gt_physical[gt_indices][:, aspect_index]
            accumulator["polarization_aspect_sq_errors"].extend(
                ((pred_aspect - gt_aspect) ** 2).detach().cpu().tolist()
            )

    shape_columns = tuple(str(column) for column in getattr(predicted_graph, "shape_target_columns", []))
    shape_indices = [feature_columns.index(column) for column in shape_columns if column in feature_columns]
    if not shape_indices:
        return
    pred_shape = pred_physical[valid][:, shape_indices]
    gt_shape = gt_physical[gt_indices][:, shape_indices]
    accumulator["shape_sq_errors"].extend(((pred_shape - gt_shape) ** 2).detach().cpu().flatten().tolist())
    valid_shape = (
        torch.isfinite(pred_shape).all(dim=1)
        & (pred_shape.min(dim=1).values >= 0.05)
        & (pred_shape.max(dim=1).values <= 3.0)
    )
    accumulator["valid_shapes"] += int(valid_shape.sum().item())


def _finalize_horizon_metrics(horizon: int, accumulator: dict[str, Any]) -> dict[str, float | int]:
    matched = int(accumulator["matched_nodes"])
    position_errors = np.asarray(accumulator["position_errors"], dtype=float)
    position_sq_errors = np.asarray(accumulator["position_sq_errors"], dtype=float)
    shape_sq_errors = np.asarray(accumulator["shape_sq_errors"], dtype=float)
    polarization_theta_sq_errors = np.asarray(accumulator["polarization_theta_sq_errors"], dtype=float)
    polarization_aspect_sq_errors = np.asarray(accumulator["polarization_aspect_sq_errors"], dtype=float)
    return {
        "horizon": int(horizon),
        "matched_nodes": matched,
        "position_mean": float(position_errors.mean()) if position_errors.size else float("nan"),
        "position_median": float(np.median(position_errors)) if position_errors.size else float("nan"),
        "position_rmse": float(np.sqrt(position_sq_errors.mean())) if position_sq_errors.size else float("nan"),
        "shape_rmse": float(np.sqrt(shape_sq_errors.mean())) if shape_sq_errors.size else float("nan"),
        "valid_shape_fraction": float(accumulator["valid_shapes"] / matched) if matched else float("nan"),
        "polarization_theta_rmse": (
            float(np.sqrt(polarization_theta_sq_errors.mean())) if polarization_theta_sq_errors.size else float("nan")
        ),
        "polarization_aspect_rmse": (
            float(np.sqrt(polarization_aspect_sq_errors.mean())) if polarization_aspect_sq_errors.size else float("nan")
        ),
    }


def _build_model_for_checkpoint(checkpoint: dict[str, Any], graphs: Sequence[Any]) -> CellInteractionGNN:
    config = checkpoint.get("train_config", {})
    return CellInteractionGNN(
        node_dim=int(graphs[0].x.size(-1)),
        edge_dim=int(graphs[0].edge_attr.size(-1)),
        shape_dim=infer_shape_dim(list(graphs)),
        hidden_dim=int(config["hidden_dim"]),
        num_message_passing_layers=int(config["layers"]),
        dropout=float(config.get("dropout", 0.0)),
        num_division_horizons=len(infer_division_horizons(list(graphs), (3, 5, 10))),
    )


def _clone_graphs(graphs: Sequence[Any]) -> list[Any]:
    return [graph.clone() for graph in graphs]


def normalization_tensors(stats: dict[str, Any] | None) -> dict[str, torch.Tensor] | None:
    if stats is None:
        return None
    return {
        "mean": torch.as_tensor(stats["mean"], dtype=torch.float32),
        "std": torch.as_tensor(stats["std"], dtype=torch.float32),
    }


def _denormalize_x(x: torch.Tensor, stats: dict[str, Any] | None) -> torch.Tensor:
    if stats is None:
        return x
    mean = torch.as_tensor(stats["mean"], device=x.device, dtype=x.dtype)
    std = torch.as_tensor(stats["std"], device=x.device, dtype=x.dtype)
    return x * std + mean


def _parse_csv_ints(text: str) -> tuple[int, ...]:
    return tuple(int(value.strip()) for value in text.split(",") if value.strip())


def _parse_csv_strings(text: str | None) -> list[str] | None:
    if text is None:
        return None
    return [value.strip() for value in text.split(",") if value.strip()]


def write_reports(rows: Sequence[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(jsonable(list(rows)), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_path = output_path.with_suffix(".csv")
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare autoregressive rollout metrics for GNN checkpoints.")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", default=None, help="Comma-separated labels matching checkpoints.")
    parser.add_argument("--horizons", default="1,3,5,10,20")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--match-tolerance", type=float, default=1e-4)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = compare_checkpoints(
        cache_path=args.cache,
        checkpoints=args.checkpoints,
        labels=_parse_csv_strings(args.labels),
        horizons=_parse_csv_ints(args.horizons),
        device_name=args.device,
        match_tolerance=args.match_tolerance,
    )
    write_reports(rows, args.out)
    for row in rows:
        print(
            f"{row['model']} h={row['horizon']} matched={row['matched_nodes']} "
            f"pos_mean={row['position_mean']:.4f} pos_median={row['position_median']:.4f} "
            f"shape_rmse={row['shape_rmse']:.4f} valid_shape={row['valid_shape_fraction']:.4f} "
            f"polarization_theta_rmse={row['polarization_theta_rmse']:.4f} "
            f"polarization_aspect_rmse={row['polarization_aspect_rmse']:.4f}"
        )


if __name__ == "__main__":
    main()
