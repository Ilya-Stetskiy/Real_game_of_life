from __future__ import annotations

import copy
import re
import warnings
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor
from torch_geometric.data import Data

from .gnn_model import CellGNNOutput
from .graph_dataset import POLARIZATION_ANGLE_COLUMN, POLARIZATION_ANGLE_PERIOD
from .hybrid_field_gnn_model import FieldConditionedGNNOutput


def _wrap_nematic_delta(x: Tensor, period: float = POLARIZATION_ANGLE_PERIOD) -> Tensor:
    """Torch equivalent of graph_dataset.wrap_nematic_delta; keeps gradients flowing."""
    half = period / 2.0
    return torch.remainder(x + half, period) - half


@dataclass(frozen=True)
class RolloutGraphConfig:
    """Rules for converting one-step predictions back into a model-ready graph."""

    position_features: tuple[str, str] = ("x", "y")
    edge_radius: float | None = None
    edge_k_nearest: int = 0
    bidirectional_edges: bool = True
    keep_edge_topology: bool = False
    drop_disappeared: bool = False
    disappearance_threshold: float = 0.5
    enable_division_births: bool = False
    division_birth_threshold: float = 0.9
    division_birth_offset: tuple[float, float] = (2.0, 0.0)
    nan_fill_value: float = 0.0


@dataclass
class FieldRolloutStep:
    """Next graph and next latent field produced by one field-conditioned step."""

    graph: Data
    field: Tensor


def prediction_to_next_graph(
    graph: Data,
    output: CellGNNOutput,
    *,
    config: RolloutGraphConfig | None = None,
    node_feature_normalization: dict[str, Any] | None = None,
) -> Data:
    """Build a next-step PyG graph from the current graph and model output."""

    cfg = config or RolloutGraphConfig()
    output_device = graph.x.device
    feature_columns = _feature_columns(graph, "node_feature_columns")
    edge_feature_columns = _feature_columns(graph, "edge_feature_columns")
    physical_x = _denormalize_x(graph.x.detach().float().cpu(), node_feature_normalization)
    previous_x = physical_x.clone()

    delta_pos = output.delta_pos.detach().float().cpu()
    if delta_pos.shape != (physical_x.size(0), 2):
        raise ValueError(f"delta_pos must have shape {(physical_x.size(0), 2)}, got {tuple(delta_pos.shape)}")
    for offset, column in enumerate(cfg.position_features):
        index = _feature_index(feature_columns, column)
        physical_x[:, index] = physical_x[:, index] + delta_pos[:, offset]

    delta_shape = output.delta_shape.detach().float().cpu()
    shape_columns = _shape_feature_columns(graph, feature_columns, delta_shape.size(-1))
    if delta_shape.shape != (physical_x.size(0), len(shape_columns)):
        raise ValueError(
            f"delta_shape must have shape {(physical_x.size(0), len(shape_columns))}, got {tuple(delta_shape.shape)}"
        )
    for offset, column in enumerate(shape_columns):
        index = _feature_index(feature_columns, column)
        physical_x[:, index] = physical_x[:, index] + delta_shape[:, offset]

    _apply_polarization_delta(graph, output, physical_x, feature_columns)

    _shift_temporal_features(previous_x, physical_x, feature_columns, cfg.nan_fill_value)

    keep_mask = _node_keep_mask(output, physical_x.size(0), cfg)
    if not cfg.enable_division_births and bool((torch.sigmoid(output.division_logits.detach().float().cpu())[keep_mask] > cfg.division_birth_threshold).any()):
        warnings.warn(
            "division logits are evaluated but do not create daughter nodes; set enable_division_births=True to opt in",
            RuntimeWarning,
            stacklevel=2,
        )
    physical_x = physical_x[keep_mask]
    previous_node_count = graph.x.size(0)
    source_node_ids = _next_node_ids(graph, keep_mask)
    if cfg.enable_division_births:
        physical_x, source_node_ids = _append_division_births(
            physical_x,
            output,
            keep_mask,
            source_node_ids,
            feature_columns=feature_columns,
            position_features=cfg.position_features,
            threshold=cfg.division_birth_threshold,
            offset=cfg.division_birth_offset,
            frame=getattr(graph, "frame", None),
        )
    normalized_x = _normalize_x(physical_x, node_feature_normalization).to(device=output_device, dtype=graph.x.dtype)

    next_graph = Data(x=normalized_x)
    next_graph.node_feature_columns = list(feature_columns)
    next_graph.edge_feature_columns = list(edge_feature_columns)
    next_graph.node_ids = source_node_ids
    next_graph.sequence_uid = getattr(graph, "sequence_uid", None)
    if hasattr(graph, "frame") and graph.frame is not None:
        next_graph.frame = int(graph.frame) + 1
    position_indices = tuple(_feature_index(feature_columns, column) for column in cfg.position_features)
    next_graph.pos_xy = physical_x[:, list(position_indices)].to(device=output_device, dtype=torch.float32)

    if (not cfg.enable_division_births) and (cfg.keep_edge_topology or (cfg.edge_radius is None and cfg.edge_k_nearest < 1)):
        edge_index = _filtered_edge_index(graph.edge_index.detach().cpu(), keep_mask, previous_node_count)
    else:
        edge_index = _build_spatial_edge_index(
            physical_x[:, list(position_indices)],
            radius=cfg.edge_radius,
            k_nearest=cfg.edge_k_nearest,
            bidirectional=cfg.bidirectional_edges,
        )
    next_graph.edge_index = edge_index.to(device=output_device, dtype=torch.long)
    next_graph.edge_attr = _edge_attr_from_positions(
        physical_x,
        edge_index,
        feature_columns=feature_columns,
        edge_feature_columns=edge_feature_columns,
        position_features=cfg.position_features,
        nan_fill_value=cfg.nan_fill_value,
    ).to(device=output_device, dtype=graph.edge_attr.dtype)

    _attach_placeholder_targets(next_graph, graph)
    return next_graph


def differentiable_prediction_to_next_graph(
    graph: Data,
    output: CellGNNOutput,
    *,
    config: RolloutGraphConfig | None = None,
    node_feature_normalization: dict[str, Any] | None = None,
) -> Data:
    """Build a next-step graph while preserving gradients through predicted features.

    Spatial radius/kNN topology is rebuilt from predicted positions, but the
    discrete neighbor selection itself is not differentiable. Gradients do flow
    through next node features, positions, and edge attributes after selection.
    """

    cfg = config or RolloutGraphConfig()
    if cfg.drop_disappeared:
        raise ValueError("Differentiable rollout does not support drop_disappeared=True.")
    if cfg.enable_division_births:
        raise ValueError("Differentiable rollout does not support enable_division_births=True.")

    feature_columns = _feature_columns(graph, "node_feature_columns")
    edge_feature_columns = _feature_columns(graph, "edge_feature_columns")
    physical_x = _denormalize_x_on_device(graph.x.float(), node_feature_normalization)
    previous_x = physical_x.clone()

    delta_pos = output.delta_pos.float()
    if delta_pos.shape != (physical_x.size(0), 2):
        raise ValueError(f"delta_pos must have shape {(physical_x.size(0), 2)}, got {tuple(delta_pos.shape)}")
    for offset, column in enumerate(cfg.position_features):
        index = _feature_index(feature_columns, column)
        physical_x[:, index] = physical_x[:, index] + delta_pos[:, offset]

    delta_shape = output.delta_shape.float()
    shape_columns = _shape_feature_columns(graph, feature_columns, delta_shape.size(-1))
    if delta_shape.shape != (physical_x.size(0), len(shape_columns)):
        raise ValueError(
            f"delta_shape must have shape {(physical_x.size(0), len(shape_columns))}, got {tuple(delta_shape.shape)}"
        )
    for offset, column in enumerate(shape_columns):
        index = _feature_index(feature_columns, column)
        physical_x[:, index] = physical_x[:, index] + delta_shape[:, offset]

    _apply_polarization_delta(graph, output, physical_x, feature_columns, detach=False)

    _shift_temporal_features(previous_x, physical_x, feature_columns, cfg.nan_fill_value)
    normalized_x = _normalize_x_on_device(physical_x, node_feature_normalization).to(dtype=graph.x.dtype)

    next_graph = Data(x=normalized_x)
    next_graph.node_feature_columns = list(feature_columns)
    next_graph.edge_feature_columns = list(edge_feature_columns)
    next_graph.node_ids = list(getattr(graph, "node_ids", []))
    next_graph.sequence_uid = getattr(graph, "sequence_uid", None)
    if hasattr(graph, "shape_target_columns"):
        next_graph.shape_target_columns = list(graph.shape_target_columns)
    if hasattr(graph, "frame") and graph.frame is not None:
        next_graph.frame = int(graph.frame) + 1

    position_indices = tuple(_feature_index(feature_columns, column) for column in cfg.position_features)
    next_graph.pos_xy = physical_x[:, list(position_indices)]
    edge_index = _build_spatial_edge_index(
        next_graph.pos_xy,
        radius=cfg.edge_radius,
        k_nearest=cfg.edge_k_nearest,
        bidirectional=cfg.bidirectional_edges,
    ).to(device=graph.x.device, dtype=torch.long)
    next_graph.edge_index = edge_index
    next_graph.edge_attr = _edge_attr_from_positions(
        physical_x,
        edge_index,
        feature_columns=feature_columns,
        edge_feature_columns=edge_feature_columns,
        position_features=cfg.position_features,
        nan_fill_value=cfg.nan_fill_value,
    ).to(device=graph.x.device, dtype=graph.edge_attr.dtype)

    _attach_placeholder_targets(next_graph, graph)
    return next_graph


def field_prediction_to_next_graph(
    graph: Data,
    output: FieldConditionedGNNOutput,
    *,
    config: RolloutGraphConfig | None = None,
    node_feature_normalization: dict[str, Any] | None = None,
    detach_field: bool = True,
) -> FieldRolloutStep:
    """Build the next graph and carry the next latent field from a hybrid prediction."""

    next_graph = prediction_to_next_graph(
        graph,
        output.cell_output,
        config=config,
        node_feature_normalization=node_feature_normalization,
    )
    field = output.field_next.detach() if detach_field else output.field_next
    return FieldRolloutStep(graph=next_graph, field=field)


def _feature_columns(graph: Data, attr: str) -> tuple[str, ...]:
    if not hasattr(graph, attr):
        raise ValueError(f"Input graph must contain {attr}.")
    columns = tuple(str(column) for column in getattr(graph, attr))
    if not columns:
        raise ValueError(f"Input graph {attr} is empty.")
    return columns


def _feature_index(columns: Sequence[str], column: str) -> int:
    try:
        return columns.index(column)
    except ValueError as exc:
        raise ValueError(f"Required feature column is missing: {column!r}") from exc


def _shape_feature_columns(graph: Data, feature_columns: Sequence[str], output_shape_dim: int) -> tuple[str, ...]:
    if hasattr(graph, "shape_target_columns"):
        columns = tuple(str(column) for column in graph.shape_target_columns)
    else:
        columns = tuple(column for column in feature_columns if column.startswith("shape_r_norm_"))
    columns = tuple(column for column in columns if column in feature_columns)
    if len(columns) != output_shape_dim:
        raise ValueError(f"Could not map delta_shape dim={output_shape_dim} to shape feature columns={columns}.")
    return columns


def _polarization_feature_columns(graph: Data, feature_columns: Sequence[str]) -> tuple[str, ...]:
    if not hasattr(graph, "polarization_columns"):
        return ()
    columns = tuple(str(column) for column in graph.polarization_columns)
    return tuple(column for column in columns if column in feature_columns)


def _apply_polarization_delta(
    graph: Data,
    output: CellGNNOutput,
    physical_x: Tensor,
    feature_columns: Sequence[str],
    *,
    detach: bool = True,
) -> None:
    delta_polarization = getattr(output, "delta_polarization", None)
    if delta_polarization is None:
        return
    polarization_columns = _polarization_feature_columns(graph, feature_columns)
    if not polarization_columns:
        return
    delta_polarization = delta_polarization.float()
    if detach:
        delta_polarization = delta_polarization.detach().cpu()
    if delta_polarization.shape != (physical_x.size(0), len(polarization_columns)):
        raise ValueError(
            f"delta_polarization must have shape {(physical_x.size(0), len(polarization_columns))}, "
            f"got {tuple(delta_polarization.shape)}"
        )
    for offset, column in enumerate(polarization_columns):
        index = _feature_index(feature_columns, column)
        physical_x[:, index] = physical_x[:, index] + delta_polarization[:, offset]
    if POLARIZATION_ANGLE_COLUMN in polarization_columns:
        angle_index = _feature_index(feature_columns, POLARIZATION_ANGLE_COLUMN)
        physical_x[:, angle_index] = _wrap_nematic_delta(physical_x[:, angle_index])


def _stats_tensors(stats: dict[str, Any] | None, size: int) -> tuple[Tensor, Tensor] | None:
    if stats is None:
        return None
    mean = torch.as_tensor(stats["mean"], dtype=torch.float32).flatten()
    std = torch.as_tensor(stats["std"], dtype=torch.float32).flatten()
    if mean.numel() != size or std.numel() != size:
        raise ValueError(f"Normalization stats size mismatch: expected {size}, got mean={mean.numel()}, std={std.numel()}.")
    return mean, std


def _denormalize_x(x: Tensor, stats: dict[str, Any] | None) -> Tensor:
    tensors = _stats_tensors(stats, x.size(-1))
    if tensors is None:
        return x.clone()
    mean, std = tensors
    return x * std + mean


def _normalize_x(x: Tensor, stats: dict[str, Any] | None) -> Tensor:
    tensors = _stats_tensors(stats, x.size(-1))
    if tensors is None:
        return x
    mean, std = tensors
    return (x - mean) / std


def _stats_tensors_on_device(stats: dict[str, Any] | None, x: Tensor) -> tuple[Tensor, Tensor] | None:
    tensors = _stats_tensors(stats, x.size(-1))
    if tensors is None:
        return None
    mean, std = tensors
    return mean.to(device=x.device, dtype=x.dtype), std.to(device=x.device, dtype=x.dtype)


def _denormalize_x_on_device(x: Tensor, stats: dict[str, Any] | None) -> Tensor:
    tensors = _stats_tensors_on_device(stats, x)
    if tensors is None:
        return x.clone()
    mean, std = tensors
    return x * std + mean


def _normalize_x_on_device(x: Tensor, stats: dict[str, Any] | None) -> Tensor:
    tensors = _stats_tensors_on_device(stats, x)
    if tensors is None:
        return x
    mean, std = tensors
    return (x - mean) / std


def _shift_temporal_features(previous_x: Tensor, next_x: Tensor, feature_columns: Sequence[str], nan_fill_value: float) -> None:
    lags = sorted({
        int(match.group(1))
        for column in feature_columns
        if (match := re.match(r"temporal_lag(\d+)_has_ancestor$", column))
    })
    if not lags:
        return

    previous = previous_x.clone()
    updated = next_x.clone()
    for lag in reversed(lags):
        has_column = f"temporal_lag{lag}_has_ancestor"
        gap_column = f"temporal_lag{lag}_frame_gap"
        has_index = feature_columns.index(has_column) if has_column in feature_columns else None
        gap_index = feature_columns.index(gap_column) if gap_column in feature_columns else None

        if lag == 1:
            if has_index is not None:
                next_x[:, has_index] = 1.0
            if gap_index is not None:
                next_x[:, gap_index] = 1.0
        else:
            prev_has = f"temporal_lag{lag - 1}_has_ancestor"
            prev_gap = f"temporal_lag{lag - 1}_frame_gap"
            if has_index is not None:
                next_x[:, has_index] = previous[:, feature_columns.index(prev_has)] if prev_has in feature_columns else 0.0
            if gap_index is not None:
                if prev_gap in feature_columns:
                    next_x[:, gap_index] = previous[:, feature_columns.index(prev_gap)] + 1.0
                else:
                    next_x[:, gap_index] = nan_fill_value

        for column in _temporal_base_columns(feature_columns, lag):
            temporal_column = f"temporal_lag{lag}_{column}"
            delta_column = f"temporal_lag{lag}_delta_{column}"
            target_index = feature_columns.index(temporal_column)
            if lag == 1:
                ancestor_values = previous[:, _feature_index(feature_columns, column)]
            else:
                previous_temporal = f"temporal_lag{lag - 1}_{column}"
                ancestor_values = (
                    previous[:, feature_columns.index(previous_temporal)]
                    if previous_temporal in feature_columns
                    else torch.full((next_x.size(0),), float(nan_fill_value))
                )
            next_x[:, target_index] = ancestor_values
            if delta_column in feature_columns and column in feature_columns:
                next_x[:, feature_columns.index(delta_column)] = updated[:, _feature_index(feature_columns, column)] - ancestor_values


def _temporal_base_columns(feature_columns: Sequence[str], lag: int) -> tuple[str, ...]:
    prefix = f"temporal_lag{lag}_"
    columns: list[str] = []
    for column in feature_columns:
        if not column.startswith(prefix):
            continue
        suffix = column[len(prefix):]
        if suffix in {"has_ancestor", "frame_gap"} or suffix.startswith("delta_"):
            continue
        columns.append(suffix)
    return tuple(columns)


def _node_keep_mask(output: CellGNNOutput, num_nodes: int, cfg: RolloutGraphConfig) -> Tensor:
    if not cfg.drop_disappeared:
        return torch.ones(num_nodes, dtype=torch.bool)
    probabilities = torch.sigmoid(output.death_logits.detach().float().cpu())
    if probabilities.numel() != num_nodes:
        raise ValueError(f"death_logits must have length {num_nodes}, got {probabilities.numel()}.")
    return probabilities < float(cfg.disappearance_threshold)


def _next_node_ids(graph: Data, keep_mask: Tensor) -> list[Any]:
    node_ids = list(getattr(graph, "node_ids", list(range(int(keep_mask.numel())))))
    return [copy.deepcopy(node_id) for node_id, keep in zip(node_ids, keep_mask.tolist()) if keep]


def _append_division_births(
    physical_x: Tensor,
    output: CellGNNOutput,
    keep_mask: Tensor,
    node_ids: list[Any],
    *,
    feature_columns: Sequence[str],
    position_features: tuple[str, str],
    threshold: float,
    offset: tuple[float, float],
    frame: Any,
) -> tuple[Tensor, list[Any]]:
    if physical_x.numel() == 0:
        return physical_x, node_ids
    probabilities = torch.sigmoid(output.division_logits.detach().float().cpu())
    birth_mask = probabilities[keep_mask] > float(threshold)
    if int(birth_mask.sum().item()) == 0:
        return physical_x, node_ids

    x_index = _feature_index(feature_columns, position_features[0])
    y_index = _feature_index(feature_columns, position_features[1])
    daughters = physical_x[birth_mask].clone()
    daughters[:, x_index] = daughters[:, x_index] + float(offset[0])
    daughters[:, y_index] = daughters[:, y_index] + float(offset[1])
    daughter_ids = [
        f"{node_id}_daughter_f{frame if frame is not None else 'next'}"
        for node_id, is_birth in zip(node_ids, birth_mask.tolist())
        if is_birth
    ]
    return torch.cat([physical_x, daughters], dim=0), [*node_ids, *daughter_ids]


def _filtered_edge_index(edge_index: Tensor, keep_mask: Tensor, previous_node_count: int) -> Tensor:
    if edge_index.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)
    if keep_mask.numel() != previous_node_count:
        raise ValueError("keep_mask length does not match previous node count.")
    old_to_new = torch.full((previous_node_count,), -1, dtype=torch.long)
    old_to_new[keep_mask] = torch.arange(int(keep_mask.sum()), dtype=torch.long)
    edge_index = edge_index.long()
    valid = keep_mask[edge_index[0]] & keep_mask[edge_index[1]]
    return old_to_new[edge_index[:, valid]]


def _build_spatial_edge_index(positions: Tensor, *, radius: float | None, k_nearest: int, bidirectional: bool) -> Tensor:
    num_nodes = int(positions.size(0))
    if num_nodes == 0:
        return torch.empty((2, 0), dtype=torch.long)
    distances = torch.cdist(positions.float(), positions.float())
    pairs: set[tuple[int, int]] = set()
    for source in range(num_nodes):
        candidates: set[int] = set()
        if radius is not None:
            within = torch.nonzero((distances[source] <= float(radius)) & (distances[source] > 0), as_tuple=False).flatten()
            candidates.update(int(item) for item in within.tolist())
        if k_nearest > 0 and num_nodes > 1:
            order = torch.argsort(distances[source])
            nearest = [int(item) for item in order.tolist() if int(item) != source][:k_nearest]
            candidates.update(nearest)
        for target in candidates:
            pairs.add((source, target))
            if bidirectional:
                pairs.add((target, source))
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(sorted(pairs), dtype=torch.long).t().contiguous()


def _edge_attr_from_positions(
    x: Tensor,
    edge_index: Tensor,
    *,
    feature_columns: Sequence[str],
    edge_feature_columns: Sequence[str],
    position_features: tuple[str, str],
    nan_fill_value: float,
) -> Tensor:
    if edge_index.numel() == 0:
        return torch.empty((0, len(edge_feature_columns)), dtype=torch.float32)
    x_index = _feature_index(feature_columns, position_features[0])
    y_index = _feature_index(feature_columns, position_features[1])
    source = x[edge_index[0]][:, [x_index, y_index]]
    target = x[edge_index[1]][:, [x_index, y_index]]
    delta = target - source
    distance = torch.linalg.norm(delta, dim=1)
    safe_distance = distance.clamp_min(1e-12)
    values = {
        "dx": delta[:, 0],
        "dy": delta[:, 1],
        "distance": distance,
        "unit_dx": delta[:, 0] / safe_distance,
        "unit_dy": delta[:, 1] / safe_distance,
    }
    columns = [values.get(column, torch.full_like(distance, float(nan_fill_value))) for column in edge_feature_columns]
    return torch.stack(columns, dim=1)


def _attach_placeholder_targets(next_graph: Data, source_graph: Data) -> None:
    num_nodes = int(next_graph.x.size(0))
    device = next_graph.x.device
    if hasattr(source_graph, "target_delta_pos"):
        next_graph.target_delta_pos = torch.zeros((num_nodes, 2), dtype=torch.float32, device=device)
    if hasattr(source_graph, "valid_regression_mask"):
        next_graph.valid_regression_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    if hasattr(source_graph, "target_delta_shape"):
        shape_dim = int(source_graph.target_delta_shape.size(-1))
        next_graph.target_delta_shape = torch.zeros((num_nodes, shape_dim), dtype=torch.float32, device=device)
    if hasattr(source_graph, "valid_shape_mask"):
        next_graph.valid_shape_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    if hasattr(source_graph, "shape_target_columns"):
        next_graph.shape_target_columns = list(source_graph.shape_target_columns)
    if hasattr(source_graph, "target_delta_polarization"):
        polarization_dim = int(source_graph.target_delta_polarization.size(-1))
        next_graph.target_delta_polarization = torch.zeros((num_nodes, polarization_dim), dtype=torch.float32, device=device)
    if hasattr(source_graph, "valid_polarization_mask"):
        next_graph.valid_polarization_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    if hasattr(source_graph, "polarization_columns"):
        next_graph.polarization_columns = list(source_graph.polarization_columns)
    for attr in ("target_division", "target_death", "valid_event_mask"):
        if hasattr(source_graph, attr):
            dtype = torch.bool if attr.startswith("valid_") else torch.float32
            setattr(next_graph, attr, torch.zeros(num_nodes, dtype=dtype, device=device))
    for attr in dir(source_graph):
        if attr.startswith("target_division_within_") and hasattr(source_graph, attr):
            setattr(next_graph, attr, torch.zeros(num_nodes, dtype=torch.float32, device=device))
        if attr.startswith("valid_division_within_") and hasattr(source_graph, attr):
            setattr(next_graph, attr, torch.zeros(num_nodes, dtype=torch.bool, device=device))
