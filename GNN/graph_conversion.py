from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


EdgeBuildMode = Literal["auto", "provided", "radius", "knn", "radius_or_knn", "none"]


NODE_ID_CANDIDATES = ("spot_uid", "spot_id", "cell_uid", "cell_id", "node_id")
EDGE_ENDPOINT_CANDIDATES = (
    ("source_id", "target_id"),
    ("source", "target"),
    ("SPOT_SOURCE_ID", "SPOT_TARGET_ID"),
    ("spot_source_id", "spot_target_id"),
)
DEFAULT_EDGE_FEATURE_COLUMNS = ("dx", "dy", "distance", "unit_dx", "unit_dy")
DEFAULT_NODE_EXCLUDE_COLUMNS = {
    "sequence_uid",
    "sequence_name",
    "dataset",
    "source_split",
    "xml_path",
    "spot_uid",
    "spot_id",
    "cell_uid",
    "cell_id",
    "node_id",
    "frame",
    "t",
    "next_id",
    "prev_id",
    "next_ids",
    "prev_ids",
    "n_next",
    "n_prev",
    "has_next",
    "has_prev",
    "is_split",
    "is_merge",
    "dx",
    "dy",
    "speed",
    "parents_id",
    "childs_id",
    "cell_start",
    "cell_end",
    "cell_end_reason",
    "cell_end_frame",
    "cell_end_t",
    "cell_lifetime_frames",
    "cell_n_spots",
    "frames_to_cell_end",
    "frames_from_cell_start",
    "cell_phase",
    "is_observed_division",
    "is_observed_division_cell",
    "future_observed_division",
}
DEFAULT_NODE_EXCLUDE_PREFIXES = (
    "target_",
    "eligible_",
    "division_within_",
)
DEFAULT_NODE_EXCLUDE_SUFFIXES = (
    "_uid",
    "_id",
    "_ids",
)
DEFAULT_EDGE_EXCLUDE_COLUMNS = {
    "edge_id",
    "source_id",
    "target_id",
    "source",
    "target",
    "SPOT_SOURCE_ID",
    "SPOT_TARGET_ID",
    "spot_source_id",
    "spot_target_id",
    "source_index",
    "target_index",
}


@dataclass(frozen=True)
class GraphBuildConfig:
    """Configuration for converting tabular cell observations into a graph."""

    node_id_col: Optional[str] = None
    edge_source_col: Optional[str] = None
    edge_target_col: Optional[str] = None
    group_cols: Optional[tuple[str, ...]] = None
    position_cols: tuple[str, str] = ("x", "y")
    node_feature_columns: Optional[tuple[str, ...]] = None
    edge_feature_columns: Optional[tuple[str, ...]] = None
    edge_build_mode: EdgeBuildMode = "auto"
    radius: Optional[float] = 80.0
    k_nearest: int = 0
    bidirectional: bool = True
    include_self_loops: bool = False
    nan_fill_value: float = 0.0
    drop_invalid_edges: bool = True


@dataclass(frozen=True)
class GraphTables:
    """Tabular representation recovered from a graph."""

    nodes: pd.DataFrame
    edges: pd.DataFrame
    node_id_col: str
    edge_source_col: str
    edge_target_col: str
    node_feature_columns: tuple[str, ...]
    edge_feature_columns: tuple[str, ...]


@dataclass(frozen=True)
class CellGraph:
    """Numpy graph representation that can be exported to PyTorch Geometric."""

    x: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    node_ids: np.ndarray
    nodes: pd.DataFrame
    edges: pd.DataFrame
    node_id_col: str
    edge_source_col: str
    edge_target_col: str
    node_feature_columns: tuple[str, ...]
    edge_feature_columns: tuple[str, ...]

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def node_dim(self) -> int:
        return int(self.x.shape[1])

    @property
    def edge_dim(self) -> int:
        return int(self.edge_attr.shape[1])

    def to_tables(self) -> GraphTables:
        return graph_to_data(self)

    def to_pyg(self):
        """Export to torch_geometric.data.Data when torch/PyG are installed."""

        try:
            import torch
            from torch_geometric.data import Data
        except Exception as exc:  # pragma: no cover - depends on local env.
            raise ImportError("torch and torch_geometric are required for to_pyg().") from exc

        data = Data(
            x=torch.as_tensor(self.x, dtype=torch.float32),
            edge_index=torch.as_tensor(self.edge_index, dtype=torch.long),
            edge_attr=torch.as_tensor(self.edge_attr, dtype=torch.float32),
        )
        data.node_ids = self.node_ids.tolist()
        data.node_feature_columns = list(self.node_feature_columns)
        data.edge_feature_columns = list(self.edge_feature_columns)

        if {"target_dx", "target_dy"}.issubset(self.nodes.columns):
            data.target_delta_pos = torch.as_tensor(
                _numeric_matrix(self.nodes, ("target_dx", "target_dy"), 0.0),
                dtype=torch.float32,
            )
            data.valid_regression_mask = torch.as_tensor(
                self.nodes[["target_dx", "target_dy"]].notna().all(axis=1).to_numpy(),
                dtype=torch.bool,
            )
        if "target_division" in self.nodes.columns:
            data.target_division = torch.as_tensor(
                pd.to_numeric(self.nodes["target_division"], errors="coerce").fillna(0).to_numpy(),
                dtype=torch.float32,
            )
        if "target_death" in self.nodes.columns:
            data.target_death = torch.as_tensor(
                pd.to_numeric(self.nodes["target_death"], errors="coerce").fillna(0).to_numpy(),
                dtype=torch.float32,
            )
        return data


def data_to_graph(
    nodes: pd.DataFrame,
    edges: Optional[pd.DataFrame] = None,
    *,
    config: Optional[GraphBuildConfig] = None,
    **overrides,
) -> CellGraph:
    """Convert node/edge tables into a CellGraph.

    If edges are not provided, spatial edges are generated inside each
    sequence/frame group using radius and/or k-nearest-neighbor rules.
    """

    cfg = _merge_config(config, overrides)
    node_table, node_id_col = _prepare_nodes(nodes, cfg.node_id_col)
    cfg = replace(cfg, node_id_col=node_id_col)
    node_ids = node_table[node_id_col].to_numpy()
    node_id_to_index = {node_id: index for index, node_id in enumerate(node_ids)}

    mode = _resolve_edge_build_mode(cfg.edge_build_mode, edges)
    if mode == "provided":
        if edges is None:
            raise ValueError("edges must be provided when edge_build_mode='provided'.")
        edge_table = _build_edges_from_table(node_table, edges, node_id_to_index, cfg)
    elif mode == "none":
        edge_table = _empty_edge_table()
    else:
        edge_table = _build_spatial_edges(node_table, cfg)

    node_feature_columns = _resolve_feature_columns(
        node_table,
        cfg.node_feature_columns,
        infer_node_feature_columns,
        "node_feature_columns",
    )
    edge_feature_columns = _resolve_feature_columns(
        edge_table,
        cfg.edge_feature_columns,
        infer_edge_feature_columns,
        "edge_feature_columns",
    )

    x = _numeric_matrix(node_table, node_feature_columns, cfg.nan_fill_value)
    edge_attr = _numeric_matrix(edge_table, edge_feature_columns, cfg.nan_fill_value)
    edge_index = _edge_index_matrix(edge_table)

    return CellGraph(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_ids=node_ids.copy(),
        nodes=node_table.copy(),
        edges=edge_table.copy(),
        node_id_col=node_id_col,
        edge_source_col="source_id",
        edge_target_col="target_id",
        node_feature_columns=node_feature_columns,
        edge_feature_columns=edge_feature_columns,
    )


def graph_to_data(graph: CellGraph) -> GraphTables:
    """Recover node and edge tables from a CellGraph."""

    return GraphTables(
        nodes=graph.nodes.copy(),
        edges=graph.edges.copy(),
        node_id_col=graph.node_id_col,
        edge_source_col=graph.edge_source_col,
        edge_target_col=graph.edge_target_col,
        node_feature_columns=tuple(graph.node_feature_columns),
        edge_feature_columns=tuple(graph.edge_feature_columns),
    )


def infer_node_feature_columns(nodes: pd.DataFrame) -> tuple[str, ...]:
    columns: list[str] = []
    for column in nodes.columns:
        if column in DEFAULT_NODE_EXCLUDE_COLUMNS:
            continue
        if column.endswith(DEFAULT_NODE_EXCLUDE_SUFFIXES):
            continue
        if column.startswith(DEFAULT_NODE_EXCLUDE_PREFIXES):
            continue
        if _is_numeric_like(nodes[column]):
            columns.append(column)
    return tuple(columns)


def infer_edge_feature_columns(edges: pd.DataFrame) -> tuple[str, ...]:
    columns: list[str] = []
    for column in DEFAULT_EDGE_FEATURE_COLUMNS:
        if column in edges.columns and _is_numeric_like(edges[column]):
            columns.append(column)
    for column in edges.columns:
        if column in columns or column in DEFAULT_EDGE_EXCLUDE_COLUMNS:
            continue
        if _is_numeric_like(edges[column]):
            columns.append(column)
    return tuple(columns)


def assert_graph_equivalent(left: CellGraph, right: CellGraph) -> None:
    """Raise AssertionError if two CellGraph objects differ logically."""

    if left.node_id_col != right.node_id_col:
        raise AssertionError(f"node_id_col differs: {left.node_id_col} != {right.node_id_col}")
    if left.node_feature_columns != right.node_feature_columns:
        raise AssertionError("node feature schema differs")
    if left.edge_feature_columns != right.edge_feature_columns:
        raise AssertionError("edge feature schema differs")
    np.testing.assert_array_equal(left.node_ids, right.node_ids)
    np.testing.assert_allclose(left.x, right.x, equal_nan=True)
    np.testing.assert_array_equal(left.edge_index, right.edge_index)
    np.testing.assert_allclose(left.edge_attr, right.edge_attr, equal_nan=True)
    pd.testing.assert_frame_equal(left.nodes, right.nodes, check_dtype=False)
    pd.testing.assert_frame_equal(left.edges, right.edges, check_dtype=False)


def _merge_config(config: Optional[GraphBuildConfig], overrides: dict) -> GraphBuildConfig:
    cfg = config or GraphBuildConfig()
    unknown = set(overrides) - set(GraphBuildConfig.__dataclass_fields__)
    if unknown:
        raise TypeError(f"Unknown GraphBuildConfig field(s): {sorted(unknown)}")
    return replace(cfg, **overrides) if overrides else cfg


def _prepare_nodes(nodes: pd.DataFrame, node_id_col: Optional[str]) -> tuple[pd.DataFrame, str]:
    if not isinstance(nodes, pd.DataFrame):
        raise TypeError("nodes must be a pandas DataFrame.")
    node_table = nodes.copy().reset_index(drop=True)

    resolved = node_id_col or next((column for column in NODE_ID_CANDIDATES if column in node_table.columns), None)
    if resolved is None:
        resolved = "node_id"
        if resolved in node_table.columns:
            raise ValueError("Could not infer node id column and 'node_id' already exists.")
        node_table.insert(0, resolved, np.arange(len(node_table), dtype=np.int64))
    if resolved not in node_table.columns:
        raise ValueError(f"node_id_col={resolved!r} is not present in nodes.")
    if node_table[resolved].duplicated().any():
        duplicated = node_table.loc[node_table[resolved].duplicated(), resolved].head(5).tolist()
        raise ValueError(f"Node ids must be unique. Duplicates include: {duplicated}")
    return node_table, resolved


def _resolve_edge_build_mode(mode: EdgeBuildMode, edges: Optional[pd.DataFrame]) -> str:
    if mode == "auto":
        return "provided" if edges is not None else "radius"
    return mode


def _build_edges_from_table(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    node_id_to_index: dict,
    cfg: GraphBuildConfig,
) -> pd.DataFrame:
    if not isinstance(edges, pd.DataFrame):
        raise TypeError("edges must be a pandas DataFrame.")
    source_col, target_col = _resolve_edge_endpoint_columns(edges, cfg.edge_source_col, cfg.edge_target_col)
    rows: list[dict] = []
    positions = _node_positions(nodes, cfg.position_cols)

    for edge in edges.reset_index(drop=True).itertuples(index=False):
        row = edge._asdict()
        source_id = row[source_col]
        target_id = row[target_col]
        if source_id not in node_id_to_index or target_id not in node_id_to_index:
            if cfg.drop_invalid_edges:
                continue
            raise ValueError(f"Edge references unknown node: {source_id!r} -> {target_id!r}")

        source_index = int(node_id_to_index[source_id])
        target_index = int(node_id_to_index[target_id])
        payload = dict(row)
        payload.update(
            {
                "source_id": source_id,
                "target_id": target_id,
                "source_index": source_index,
                "target_index": target_index,
            }
        )
        payload.update(_geometry_features(positions, source_index, target_index))
        rows.append(payload)

    return _finalize_edge_table(rows)


def _build_spatial_edges(nodes: pd.DataFrame, cfg: GraphBuildConfig) -> pd.DataFrame:
    x_col, y_col = cfg.position_cols
    missing = [column for column in (x_col, y_col) if column not in nodes.columns]
    if missing:
        raise ValueError(f"Spatial edge construction requires position columns: {missing}")
    if cfg.radius is None and cfg.k_nearest < 1 and cfg.edge_build_mode in {"radius", "knn", "radius_or_knn", "auto"}:
        raise ValueError("Set radius and/or k_nearest for spatial edge construction.")

    group_cols = _resolve_group_columns(nodes, cfg.group_cols)
    positions = _node_positions(nodes, cfg.position_cols)
    node_ids = nodes[cfg.node_id_col].to_numpy()
    edge_pairs: set[tuple[int, int]] = set()

    groups = [(None, nodes)] if not group_cols else nodes.groupby(list(group_cols), sort=False, dropna=False)
    for _, group in groups:
        indices = group.index.to_numpy(dtype=np.int64)
        if len(indices) == 0:
            continue
        local_positions = positions[indices]
        distances = _pairwise_distances(local_positions)
        for local_source, source_index in enumerate(indices):
            candidates = _selected_neighbor_indices(distances[local_source], cfg)
            for local_target in candidates:
                target_index = int(indices[local_target])
                if source_index == target_index and not cfg.include_self_loops:
                    continue
                edge_pairs.add((int(source_index), target_index))
                if cfg.bidirectional and source_index != target_index:
                    edge_pairs.add((target_index, int(source_index)))

    rows: list[dict] = []
    for source_index, target_index in sorted(edge_pairs):
        payload = {
            "source_id": node_ids[source_index],
            "target_id": node_ids[target_index],
            "source_index": int(source_index),
            "target_index": int(target_index),
        }
        for column in group_cols:
            payload[column] = nodes.at[source_index, column]
        payload.update(_geometry_features(positions, source_index, target_index))
        rows.append(payload)

    return _finalize_edge_table(rows)


def _selected_neighbor_indices(distances: np.ndarray, cfg: GraphBuildConfig) -> list[int]:
    finite = np.isfinite(distances)
    if not cfg.include_self_loops:
        finite &= distances > 0
    selected: set[int] = set()

    mode = cfg.edge_build_mode
    if mode == "auto":
        mode = "radius"

    if mode in {"radius", "radius_or_knn"} and cfg.radius is not None:
        selected.update(np.flatnonzero(finite & (distances <= float(cfg.radius))).astype(int).tolist())

    if mode in {"knn", "radius_or_knn"} or (mode == "radius" and cfg.radius is None):
        if cfg.k_nearest > 0:
            candidates = np.flatnonzero(finite)
            order = candidates[np.argsort(distances[candidates], kind="mergesort")]
            selected.update(order[: cfg.k_nearest].astype(int).tolist())

    return sorted(selected)


def _resolve_group_columns(nodes: pd.DataFrame, group_cols: Optional[tuple[str, ...]]) -> tuple[str, ...]:
    if group_cols is not None:
        missing = [column for column in group_cols if column not in nodes.columns]
        if missing:
            raise ValueError(f"group_cols missing from nodes: {missing}")
        return tuple(group_cols)
    if {"sequence_uid", "frame"}.issubset(nodes.columns):
        return ("sequence_uid", "frame")
    if "frame" in nodes.columns:
        return ("frame",)
    return ()


def _resolve_edge_endpoint_columns(
    edges: pd.DataFrame,
    source_col: Optional[str],
    target_col: Optional[str],
) -> tuple[str, str]:
    if source_col is not None or target_col is not None:
        if source_col is None or target_col is None:
            raise ValueError("edge_source_col and edge_target_col must be set together.")
        missing = [column for column in (source_col, target_col) if column not in edges.columns]
        if missing:
            raise ValueError(f"Edge endpoint columns missing from edges: {missing}")
        return source_col, target_col
    for candidate_source, candidate_target in EDGE_ENDPOINT_CANDIDATES:
        if candidate_source in edges.columns and candidate_target in edges.columns:
            return candidate_source, candidate_target
    raise ValueError(
        "Could not infer edge endpoint columns. Expected one of "
        f"{EDGE_ENDPOINT_CANDIDATES} or pass edge_source_col/edge_target_col."
    )


def _resolve_feature_columns(
    table: pd.DataFrame,
    configured: Optional[tuple[str, ...]],
    infer_fn,
    name: str,
) -> tuple[str, ...]:
    if configured is None:
        return tuple(infer_fn(table))
    missing = [column for column in configured if column not in table.columns]
    if missing:
        raise ValueError(f"{name} missing from table: {missing}")
    return tuple(configured)


def _numeric_matrix(table: pd.DataFrame, columns: Sequence[str], nan_fill_value: float) -> np.ndarray:
    if not columns:
        return np.empty((len(table), 0), dtype=np.float32)
    numeric = table.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(float(nan_fill_value))
    return numeric.to_numpy(dtype=np.float32, copy=True)


def _edge_index_matrix(edges: pd.DataFrame) -> np.ndarray:
    if edges.empty:
        return np.empty((2, 0), dtype=np.int64)
    return edges[["source_index", "target_index"]].to_numpy(dtype=np.int64).T.copy()


def _node_positions(nodes: pd.DataFrame, position_cols: tuple[str, str]) -> np.ndarray:
    if all(column in nodes.columns for column in position_cols):
        return nodes.loc[:, list(position_cols)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    return np.full((len(nodes), 2), np.nan, dtype=np.float64)


def _pairwise_distances(points: np.ndarray) -> np.ndarray:
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def _geometry_features(positions: np.ndarray, source_index: int, target_index: int) -> dict[str, float]:
    source = positions[source_index]
    target = positions[target_index]
    dx = float(target[0] - source[0])
    dy = float(target[1] - source[1])
    distance = float(np.hypot(dx, dy))
    if np.isfinite(distance) and distance > 0:
        unit_dx = dx / distance
        unit_dy = dy / distance
    else:
        unit_dx = 0.0
        unit_dy = 0.0
    return {
        "dx": dx,
        "dy": dy,
        "distance": distance,
        "unit_dx": unit_dx,
        "unit_dy": unit_dy,
    }


def _finalize_edge_table(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return _empty_edge_table()
    edge_table = pd.DataFrame(rows).reset_index(drop=True)
    if "edge_id" not in edge_table.columns:
        edge_table.insert(0, "edge_id", np.arange(len(edge_table), dtype=np.int64))
    first_columns = ["edge_id", "source_id", "target_id", "source_index", "target_index"]
    ordered = [column for column in first_columns if column in edge_table.columns]
    ordered.extend(column for column in edge_table.columns if column not in ordered)
    return edge_table.loc[:, ordered]


def _empty_edge_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "edge_id": pd.Series(dtype=np.int64),
            "source_id": pd.Series(dtype=object),
            "target_id": pd.Series(dtype=object),
            "source_index": pd.Series(dtype=np.int64),
            "target_index": pd.Series(dtype=np.int64),
            "dx": pd.Series(dtype=float),
            "dy": pd.Series(dtype=float),
            "distance": pd.Series(dtype=float),
            "unit_dx": pd.Series(dtype=float),
            "unit_dy": pd.Series(dtype=float),
        }
    )


def _is_numeric_like(series: pd.Series) -> bool:
    return is_numeric_dtype(series) or is_bool_dtype(series)
