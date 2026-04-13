from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .graph_conversion import CellGraph, GraphBuildConfig, data_to_graph


def _default_processed_spots_path() -> Path:
    database_root = Path(__file__).resolve().parents[1] / "HeLa_Database"
    candidates = (
        database_root / "shape_division_analysis_dynamic" / "spot_shape_division_dataset.parquet",
        database_root / "HeLa клетки" / "shape_division_analysis_dynamic" / "spot_shape_division_dataset.parquet",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DEFAULT_PROCESSED_SPOTS = _default_processed_spots_path()

DEFAULT_SCALAR_FEATURES = (
    "x",
    "y",
    "AREA",
    "PERIMETER",
    "CIRCULARITY",
    "SOLIDITY",
    "RADIUS",
    "ELLIPSE_MAJOR",
    "ELLIPSE_MINOR",
    "ELLIPSE_ASPECTRATIO",
    "ELLIPSE_THETA",
    "MEAN_INTENSITY_CH1",
    "contour_point_count",
    "shape_missing_fraction",
    "shape_area_ratio",
    "shape_reconstruction_area_ratio",
    "shape_contour_centroid_dx",
    "shape_contour_centroid_dy",
    "shape_mean_radius",
    "shape_radius_std",
    "shape_radius_cv",
    "n_neighbors",
    "density",
    "Fx",
    "Fy",
)
DEFAULT_EDGE_FEATURES = ("dx", "dy", "distance", "unit_dx", "unit_dy")
DEFAULT_TEMPORAL_FEATURES = (
    "x",
    "y",
    "AREA",
    "SOLIDITY",
    "shape_mean_radius",
    "shape_radius_cv",
    "n_neighbors",
    "density",
)


@dataclass(frozen=True)
class FrameGraphDatasetConfig:
    """Configuration for processed spot table -> one-frame graph dataset."""

    source_path: Path = DEFAULT_PROCESSED_SPOTS
    sequence_col: str = "sequence_uid"
    frame_col: str = "frame"
    node_id_col: str = "spot_id"
    position_cols: tuple[str, str] = ("x", "y")
    horizons: tuple[int, ...] = (3, 5, 10)
    edge_radius: Optional[float] = 80.0
    edge_k_nearest: int = 0
    bidirectional_edges: bool = True
    include_shape_radii: bool = True
    shape_feature_prefix: str = "shape_r_norm_"
    node_feature_columns: Optional[tuple[str, ...]] = None
    extra_node_feature_columns: tuple[str, ...] = ()
    edge_feature_columns: tuple[str, ...] = DEFAULT_EDGE_FEATURES
    temporal_lags: tuple[int, ...] = ()
    temporal_feature_columns: Optional[tuple[str, ...]] = None
    include_temporal_deltas: bool = True
    min_nodes_per_graph: int = 1
    nan_fill_value: float = 0.0


def load_processed_spots(path: str | Path | None = None) -> pd.DataFrame:
    """Read the processed spot parquet used for GNN graph construction."""

    source = Path(path) if path is not None else DEFAULT_PROCESSED_SPOTS
    if not source.exists():
        raise FileNotFoundError(f"Processed spot table not found: {source}")
    return pd.read_parquet(source)


def default_node_feature_columns(
    spots: pd.DataFrame,
    cfg: FrameGraphDatasetConfig | None = None,
) -> tuple[str, ...]:
    """Return safe current-frame node features present in the processed table."""

    cfg = cfg or FrameGraphDatasetConfig()
    if cfg.node_feature_columns is not None:
        missing = [column for column in cfg.node_feature_columns if column not in spots.columns]
        if missing:
            raise ValueError(f"node_feature_columns missing from spots: {missing}")
        return tuple(cfg.node_feature_columns)

    columns: list[str] = [column for column in DEFAULT_SCALAR_FEATURES if column in spots.columns]
    if cfg.include_shape_radii:
        columns.extend(sorted(column for column in spots.columns if column.startswith(cfg.shape_feature_prefix)))
    columns.extend(column for column in cfg.extra_node_feature_columns if column in spots.columns and column not in columns)
    columns.extend(column for column in temporal_feature_columns(spots, cfg) if column in spots.columns and column not in columns)
    return tuple(columns)


def temporal_feature_columns(spots: pd.DataFrame, cfg: FrameGraphDatasetConfig | None = None) -> tuple[str, ...]:
    cfg = cfg or FrameGraphDatasetConfig()
    columns: list[str] = []
    for lag in cfg.temporal_lags:
        prefix = f"temporal_lag{int(lag)}"
        columns.append(f"{prefix}_has_ancestor")
        columns.append(f"{prefix}_frame_gap")
        base_columns = temporal_base_feature_columns(spots, cfg)
        for column in base_columns:
            columns.append(f"{prefix}_{column}")
            if cfg.include_temporal_deltas:
                columns.append(f"{prefix}_delta_{column}")
    return tuple(columns)


def temporal_base_feature_columns(spots: pd.DataFrame, cfg: FrameGraphDatasetConfig | None = None) -> tuple[str, ...]:
    cfg = cfg or FrameGraphDatasetConfig()
    candidates = cfg.temporal_feature_columns or DEFAULT_TEMPORAL_FEATURES
    return tuple(column for column in candidates if column in spots.columns)


def default_shape_target_columns(
    spots: pd.DataFrame,
    cfg: FrameGraphDatasetConfig | None = None,
) -> tuple[str, ...]:
    cfg = cfg or FrameGraphDatasetConfig()
    return tuple(sorted(column for column in spots.columns if column.startswith(cfg.shape_feature_prefix)))


def add_one_step_targets(
    spots: pd.DataFrame,
    cfg: FrameGraphDatasetConfig | None = None,
    *,
    shape_target_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Attach one-step supervision columns derived from TrackMate next links.

    Regression targets are valid only for nodes with exactly one known next spot.
    Split nodes get event labels, while their position/shape regression target is
    masked out to avoid forcing two daughter cells into one arbitrary target.
    """

    cfg = cfg or FrameGraphDatasetConfig()
    required = [cfg.sequence_col, cfg.frame_col, cfg.node_id_col, *cfg.position_cols, "n_next"]
    missing = [column for column in required if column not in spots.columns]
    if missing:
        raise ValueError(f"Cannot build one-step targets, missing columns: {missing}")

    shape_columns = tuple(shape_target_columns) if shape_target_columns is not None else default_shape_target_columns(spots, cfg)
    out = spots.copy().reset_index(drop=True)
    out[cfg.node_id_col] = pd.to_numeric(out[cfg.node_id_col], errors="raise").astype(np.int64)
    out[cfg.frame_col] = pd.to_numeric(out[cfg.frame_col], errors="raise").astype(np.int64)
    out["n_next"] = pd.to_numeric(out["n_next"], errors="coerce").fillna(0).astype(np.int64)

    max_frame_by_sequence = out.groupby(cfg.sequence_col, dropna=False)[cfg.frame_col].transform("max")
    out["valid_event_mask"] = out[cfg.frame_col] < max_frame_by_sequence
    out["target_has_next"] = out["n_next"] > 0
    out["target_single_next"] = out["n_next"].eq(1)
    out["target_division"] = out["n_next"] > 1
    out["target_death"] = out["n_next"].eq(0) & out["valid_event_mask"]
    out["target_child_count"] = out["n_next"].clip(lower=0)
    out["target_child_ids"] = out["next_ids"].fillna("").astype(str) if "next_ids" in out.columns else ""

    out["target_dx"] = np.nan
    out["target_dy"] = np.nan
    out["target_centroid_dx"] = np.nan
    out["target_centroid_dy"] = np.nan
    out["valid_regression_mask"] = False
    out["valid_shape_mask"] = False

    for column in shape_columns:
        out[f"target_delta_{column}"] = np.nan

    row_index_col = "_gnn_row_index"
    next_id_col = "_gnn_next_id"
    out[row_index_col] = np.arange(len(out), dtype=np.int64)
    out[next_id_col] = pd.to_numeric(out.get("next_id", np.nan), errors="coerce")

    single_mask = out["n_next"].eq(1) & out[next_id_col].notna()
    if single_mask.any():
        left_columns = [
            row_index_col,
            cfg.sequence_col,
            next_id_col,
            *cfg.position_cols,
            *shape_columns,
        ]
        left = out.loc[single_mask, left_columns].copy()
        left[next_id_col] = left[next_id_col].astype(np.int64)

        right_renames = {
            cfg.node_id_col: next_id_col,
            cfg.position_cols[0]: "_next_x",
            cfg.position_cols[1]: "_next_y",
        }
        right_renames.update({column: f"_next_{column}" for column in shape_columns})
        right = out[[cfg.sequence_col, cfg.node_id_col, *cfg.position_cols, *shape_columns]].rename(columns=right_renames)

        joined = left.merge(right, on=[cfg.sequence_col, next_id_col], how="left", sort=False)
        valid_next = joined["_next_x"].notna() & joined["_next_y"].notna()
        valid_rows = joined.loc[valid_next, row_index_col].astype(np.int64).to_numpy()
        if len(valid_rows):
            out.loc[valid_rows, "target_dx"] = (
                joined.loc[valid_next, "_next_x"].to_numpy(dtype=float)
                - joined.loc[valid_next, cfg.position_cols[0]].to_numpy(dtype=float)
            )
            out.loc[valid_rows, "target_dy"] = (
                joined.loc[valid_next, "_next_y"].to_numpy(dtype=float)
                - joined.loc[valid_next, cfg.position_cols[1]].to_numpy(dtype=float)
            )
            out.loc[valid_rows, "target_centroid_dx"] = out.loc[valid_rows, "target_dx"].to_numpy(dtype=float)
            out.loc[valid_rows, "target_centroid_dy"] = out.loc[valid_rows, "target_dy"].to_numpy(dtype=float)
            out.loc[valid_rows, "valid_regression_mask"] = True

        if shape_columns:
            shape_valid = valid_next.copy()
            for column in shape_columns:
                current_values = pd.to_numeric(joined[column], errors="coerce")
                next_values = pd.to_numeric(joined[f"_next_{column}"], errors="coerce")
                column_valid = valid_next & current_values.notna() & next_values.notna()
                shape_valid &= column_valid
                rows = joined.loc[column_valid, row_index_col].astype(np.int64).to_numpy()
                if len(rows):
                    out.loc[rows, f"target_delta_{column}"] = (
                        next_values.loc[column_valid].to_numpy(dtype=float)
                        - current_values.loc[column_valid].to_numpy(dtype=float)
                    )
            if shape_valid.any():
                rows = joined.loc[shape_valid, row_index_col].astype(np.int64).to_numpy()
                out.loc[rows, "valid_shape_mask"] = True

    index_by_spot = {
        (row[cfg.sequence_col], int(row[cfg.node_id_col])): index
        for index, row in out[[cfg.sequence_col, cfg.node_id_col]].iterrows()
    }

    split_or_multi_mask = out["n_next"] > 1
    for index, row in out.loc[split_or_multi_mask].iterrows():
        sequence = row[cfg.sequence_col]
        child_ids = _next_child_ids(row)
        child_indices = [
            index_by_spot[(sequence, child_id)]
            for child_id in child_ids
            if (sequence, child_id) in index_by_spot
        ]

        if child_indices:
            child_positions = out.loc[child_indices, list(cfg.position_cols)].apply(pd.to_numeric, errors="coerce")
            centroid = child_positions.mean(axis=0)
            out.at[index, "target_centroid_dx"] = float(centroid.iloc[0] - row[cfg.position_cols[0]])
            out.at[index, "target_centroid_dy"] = float(centroid.iloc[1] - row[cfg.position_cols[1]])

    return out.drop(columns=[row_index_col, next_id_col])


def build_frame_graphs(
    spots: pd.DataFrame,
    cfg: FrameGraphDatasetConfig | None = None,
    *,
    add_targets: bool = True,
) -> list[CellGraph]:
    """Build one CellGraph per sequence/frame from a processed spot table."""

    cfg = cfg or FrameGraphDatasetConfig()
    prepared = add_one_step_targets(spots, cfg) if add_targets else spots.copy().reset_index(drop=True)
    prepared = add_temporal_features(prepared, cfg)
    node_features = default_node_feature_columns(prepared, cfg)

    graphs: list[CellGraph] = []
    graph_cfg = GraphBuildConfig(
        node_id_col=cfg.node_id_col,
        group_cols=(cfg.sequence_col, cfg.frame_col),
        position_cols=cfg.position_cols,
        node_feature_columns=node_features,
        edge_feature_columns=cfg.edge_feature_columns,
        edge_build_mode="radius_or_knn" if cfg.edge_k_nearest > 0 and cfg.edge_radius is not None else "radius",
        radius=cfg.edge_radius,
        k_nearest=cfg.edge_k_nearest,
        bidirectional=cfg.bidirectional_edges,
        nan_fill_value=cfg.nan_fill_value,
    )

    grouped = prepared.groupby([cfg.sequence_col, cfg.frame_col], sort=True, dropna=False)
    for (_, _), frame_df in grouped:
        frame_df = frame_df.reset_index(drop=True)
        if len(frame_df) < cfg.min_nodes_per_graph:
            continue
        graphs.append(data_to_graph(frame_df, config=graph_cfg))
    return graphs


def build_pyg_frame_graphs(
    spots: pd.DataFrame,
    cfg: FrameGraphDatasetConfig | None = None,
    *,
    add_targets: bool = True,
):
    """Build PyG Data objects per sequence/frame."""

    return [cell_graph_to_pyg_training_data(graph, cfg) for graph in build_frame_graphs(spots, cfg, add_targets=add_targets)]


def add_temporal_features(spots: pd.DataFrame, cfg: FrameGraphDatasetConfig | None = None) -> pd.DataFrame:
    """Attach past-trajectory features by following TrackMate parent links.

    The features use only ancestor spots from earlier frames. If a spot has no
    known ancestor at a requested lag, the lag values are left as NaN and later
    filled by the graph converter's nan_fill_value.
    """

    cfg = cfg or FrameGraphDatasetConfig()
    lags = tuple(sorted({int(lag) for lag in cfg.temporal_lags if int(lag) > 0}))
    if not lags:
        return spots

    required = [cfg.sequence_col, cfg.frame_col, cfg.node_id_col]
    missing = [column for column in required if column not in spots.columns]
    if missing:
        raise ValueError(f"Cannot build temporal features, missing columns: {missing}")

    base_columns = temporal_base_feature_columns(spots, cfg)
    out = spots.copy().reset_index(drop=True)
    out[cfg.node_id_col] = pd.to_numeric(out[cfg.node_id_col], errors="raise").astype(np.int64)
    out[cfg.frame_col] = pd.to_numeric(out[cfg.frame_col], errors="raise").astype(np.int64)

    row_by_key = {
        (row[cfg.sequence_col], int(row[cfg.node_id_col])): int(index)
        for index, row in out[[cfg.sequence_col, cfg.node_id_col]].iterrows()
    }
    parent_by_key = infer_parent_links(out, cfg)

    ancestor_for_lag: dict[int, list[int | None]] = {lag: [] for lag in lags}
    for _, row in out[[cfg.sequence_col, cfg.node_id_col]].iterrows():
        sequence = row[cfg.sequence_col]
        current_key = (sequence, int(row[cfg.node_id_col]))
        current_ancestor = current_key
        for lag in range(1, max(lags) + 1):
            current_ancestor = parent_by_key.get(current_ancestor)
            if lag in ancestor_for_lag:
                ancestor_for_lag[lag].append(row_by_key.get(current_ancestor) if current_ancestor is not None else None)

    for lag in lags:
        prefix = f"temporal_lag{lag}"
        ancestor_indices = ancestor_for_lag[lag]
        has_ancestor = np.asarray([index is not None for index in ancestor_indices], dtype=bool)
        out[f"{prefix}_has_ancestor"] = has_ancestor
        out[f"{prefix}_frame_gap"] = np.nan
        for column in base_columns:
            out[f"{prefix}_{column}"] = np.nan
            if cfg.include_temporal_deltas:
                out[f"{prefix}_delta_{column}"] = np.nan

        valid_rows = np.flatnonzero(has_ancestor)
        if len(valid_rows) == 0:
            continue
        ancestor_rows = np.asarray([ancestor_indices[index] for index in valid_rows], dtype=np.int64)
        out.loc[valid_rows, f"{prefix}_frame_gap"] = (
            out.loc[valid_rows, cfg.frame_col].to_numpy(dtype=float)
            - out.loc[ancestor_rows, cfg.frame_col].to_numpy(dtype=float)
        )
        for column in base_columns:
            current_values = pd.to_numeric(out.loc[valid_rows, column], errors="coerce").to_numpy(dtype=float)
            ancestor_values = pd.to_numeric(out.loc[ancestor_rows, column], errors="coerce").to_numpy(dtype=float)
            out.loc[valid_rows, f"{prefix}_{column}"] = ancestor_values
            if cfg.include_temporal_deltas:
                out.loc[valid_rows, f"{prefix}_delta_{column}"] = current_values - ancestor_values
    return out


def infer_parent_links(spots: pd.DataFrame, cfg: FrameGraphDatasetConfig) -> dict[tuple[object, int], tuple[object, int]]:
    parent_by_key: dict[tuple[object, int], tuple[object, int]] = {}
    if "next_ids" not in spots.columns and "next_id" not in spots.columns:
        return parent_by_key

    for _, row in spots.iterrows():
        sequence = row[cfg.sequence_col]
        parent_key = (sequence, int(row[cfg.node_id_col]))
        for child_id in _next_child_ids(row):
            child_key = (sequence, int(child_id))
            parent_by_key.setdefault(child_key, parent_key)
    return parent_by_key


def cell_graph_to_pyg_training_data(graph: CellGraph, cfg: FrameGraphDatasetConfig | None = None):
    """Convert a CellGraph to PyG Data and attach one-step/horizon targets."""

    cfg = cfg or FrameGraphDatasetConfig()
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on local env.
        raise ImportError("torch is required for PyG training data export.") from exc

    data = graph.to_pyg()
    nodes = graph.nodes

    if nodes.empty:
        data.sequence_uid = None
        data.frame = None
        return data

    data.sequence_uid = str(nodes[cfg.sequence_col].iloc[0]) if cfg.sequence_col in nodes else None
    data.frame = int(nodes[cfg.frame_col].iloc[0]) if cfg.frame_col in nodes else None

    for column, dtype in (
        ("valid_event_mask", torch.bool),
        ("valid_regression_mask", torch.bool),
        ("valid_shape_mask", torch.bool),
        ("target_has_next", torch.float32),
        ("target_single_next", torch.float32),
        ("target_division", torch.float32),
        ("target_death", torch.float32),
        ("target_child_count", torch.float32),
    ):
        if column in nodes:
            values = _node_column_tensor(nodes, column, dtype=dtype)
            setattr(data, column, values)

    shape_target_columns = sorted(column for column in nodes.columns if column.startswith("target_delta_shape_r_norm_"))
    if shape_target_columns:
        data.target_delta_shape = torch.as_tensor(
            _numeric_matrix(nodes, shape_target_columns, cfg.nan_fill_value),
            dtype=torch.float32,
        )
        data.shape_target_columns = [column.replace("target_delta_", "", 1) for column in shape_target_columns]

    for horizon in cfg.horizons:
        target_column = f"division_within_{horizon}_frames"
        eligible_column = f"eligible_within_{horizon}_frames"
        if target_column in nodes:
            setattr(data, f"target_division_within_{horizon}", _node_column_tensor(nodes, target_column, dtype=torch.float32))
        if eligible_column in nodes:
            setattr(data, f"valid_division_within_{horizon}", _node_column_tensor(nodes, eligible_column, dtype=torch.bool))
    return data


def build_graphs_from_processed(
    path: str | Path | None = None,
    cfg: FrameGraphDatasetConfig | None = None,
) -> list[CellGraph]:
    """Convenience entry point for the repository's processed HeLa data."""

    base_cfg = cfg or FrameGraphDatasetConfig()
    source = Path(path) if path is not None else base_cfg.source_path
    spots = load_processed_spots(source)
    return build_frame_graphs(spots, replace(base_cfg, source_path=source))


def _next_child_ids(row: pd.Series) -> list[int]:
    if int(row.get("n_next", 0)) == 1 and not pd.isna(row.get("next_id", np.nan)):
        return [int(row["next_id"])]
    text = row.get("next_ids", "")
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return []
    ids: list[int] = []
    for part in str(text).split("|"):
        if not part:
            continue
        try:
            ids.append(int(float(part)))
        except ValueError:
            continue
    return ids


def _node_column_tensor(nodes: pd.DataFrame, column: str, *, dtype):
    import torch

    if dtype is torch.bool:
        return torch.as_tensor(nodes[column].fillna(False).astype(bool).to_numpy(), dtype=torch.bool)
    return torch.as_tensor(
        pd.to_numeric(nodes[column], errors="coerce").fillna(0.0).to_numpy(),
        dtype=dtype,
    )


def _numeric_matrix(table: pd.DataFrame, columns: Sequence[str], nan_fill_value: float) -> np.ndarray:
    if not columns:
        return np.empty((len(table), 0), dtype=np.float32)
    numeric = table.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(float(nan_fill_value))
    return numeric.to_numpy(dtype=np.float32, copy=True)
