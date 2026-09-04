from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from Real_game_of_life.GNN.graph_dataset import (
    FrameGraphDatasetConfig,
    POLARIZATION_ANGLE_COLUMN,
    POLARIZATION_MAGNITUDE_COLUMN,
    add_one_step_targets,
    add_temporal_features,
    build_frame_graphs,
    cell_graph_to_pyg_training_data,
    default_node_feature_columns,
    load_processed_spots,
    wrap_nematic_delta,
)


def _sample_spots() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_uid": ["seq_a", "seq_a", "seq_a", "seq_a", "seq_a", "seq_a"],
            "frame": [0, 0, 1, 1, 1, 1],
            "spot_id": [1, 2, 3, 4, 5, 6],
            "x": [0.0, 10.0, 1.0, 9.0, 11.0, 30.0],
            "y": [0.0, 0.0, 0.0, -1.0, 1.0, 0.0],
            "AREA": [100.0, 120.0, 105.0, 60.0, 62.0, 150.0],
            "SOLIDITY": [0.90, 0.80, 0.91, 0.70, 0.72, 0.95],
            "n_neighbors": [1, 1, 2, 2, 2, 0],
            "density": [0.1, 0.1, 0.2, 0.2, 0.2, 0.0],
            "Fx": [0.0, 0.0, 0.1, -0.1, 0.0, 0.0],
            "Fy": [0.0, 0.0, 0.0, 0.0, 0.1, 0.0],
            "shape_r_norm_000": [1.0, 2.0, 1.1, 1.8, 2.2, 3.0],
            "shape_r_norm_001": [1.5, 2.5, 1.6, 2.3, 2.7, 3.5],
            "next_id": [3.0, np.nan, np.nan, np.nan, np.nan, np.nan],
            "next_ids": ["3", "4|5", "", "", "", ""],
            "n_next": [1, 2, 0, 0, 0, 0],
            "division_within_3_frames": [False, True, False, False, False, False],
            "eligible_within_3_frames": [True, True, False, False, False, False],
            "division_within_5_frames": [False, True, False, False, False, False],
            "eligible_within_5_frames": [True, True, False, False, False, False],
            "division_within_10_frames": [False, True, False, False, False, False],
            "eligible_within_10_frames": [True, True, False, False, False, False],
        }
    )


def _cfg() -> FrameGraphDatasetConfig:
    return FrameGraphDatasetConfig(
        node_feature_columns=("x", "y", "AREA", "SOLIDITY", "shape_r_norm_000", "shape_r_norm_001"),
        edge_radius=3.0,
        horizons=(3, 5, 10),
    )


def test_add_one_step_targets_masks_split_and_last_frame_nodes() -> None:
    prepared = add_one_step_targets(_sample_spots(), _cfg())
    by_id = prepared.set_index("spot_id")

    assert bool(by_id.at[1, "valid_regression_mask"]) is True
    assert by_id.at[1, "target_dx"] == 1.0
    assert by_id.at[1, "target_dy"] == 0.0
    assert bool(by_id.at[1, "valid_shape_mask"]) is True
    assert by_id.at[1, "target_delta_shape_r_norm_000"] == pytest.approx(0.1)
    assert by_id.at[1, "target_delta_shape_r_norm_001"] == pytest.approx(0.1)

    assert bool(by_id.at[2, "target_division"]) is True
    assert by_id.at[2, "target_child_count"] == 2
    assert bool(by_id.at[2, "valid_regression_mask"]) is False
    assert by_id.at[2, "target_centroid_dx"] == pytest.approx(0.0)
    assert by_id.at[2, "target_centroid_dy"] == pytest.approx(0.0)

    assert bool(by_id.at[3, "valid_event_mask"]) is False
    assert bool(by_id.at[3, "target_death"]) is False


def test_add_one_step_targets_rejects_duplicate_spot_ids_within_sequence() -> None:
    spots = _sample_spots()
    spots.loc[1, "spot_id"] = spots.loc[0, "spot_id"]

    with pytest.raises(ValueError, match="unique within"):
        add_one_step_targets(spots, _cfg())


def test_add_temporal_features_follow_single_and_split_parent_links() -> None:
    cfg = FrameGraphDatasetConfig(
        edge_radius=3.0,
        horizons=(3, 5, 10),
        temporal_lags=(1,),
        temporal_feature_columns=("x", "AREA"),
    )
    prepared = add_one_step_targets(_sample_spots(), cfg)
    temporal = add_temporal_features(prepared, cfg).set_index("spot_id")

    assert bool(temporal.at[1, "temporal_lag1_has_ancestor"]) is False
    assert bool(temporal.at[3, "temporal_lag1_has_ancestor"]) is True
    assert temporal.at[3, "temporal_lag1_frame_gap"] == 1.0
    assert temporal.at[3, "temporal_lag1_x"] == 0.0
    assert temporal.at[3, "temporal_lag1_delta_x"] == 1.0
    assert temporal.at[3, "temporal_lag1_AREA"] == 100.0
    assert temporal.at[3, "temporal_lag1_delta_AREA"] == 5.0

    assert bool(temporal.at[4, "temporal_lag1_has_ancestor"]) is True
    assert temporal.at[4, "temporal_lag1_x"] == 10.0
    assert temporal.at[4, "temporal_lag1_delta_x"] == -1.0
    assert bool(temporal.at[5, "temporal_lag1_has_ancestor"]) is True
    assert temporal.at[5, "temporal_lag1_delta_x"] == 1.0


def test_temporal_features_enter_auto_node_feature_set_without_targets() -> None:
    cfg = FrameGraphDatasetConfig(
        node_feature_columns=None,
        edge_radius=3.0,
        horizons=(3, 5, 10),
        temporal_lags=(1,),
        temporal_feature_columns=("x", "AREA"),
    )
    graphs = build_frame_graphs(_sample_spots(), cfg)
    feature_names = graphs[0].node_feature_columns

    assert "temporal_lag1_has_ancestor" in feature_names
    assert "temporal_lag1_x" in feature_names
    assert "temporal_lag1_delta_x" in feature_names
    assert "temporal_lag1_AREA" in feature_names
    assert "temporal_lag1_delta_AREA" in feature_names

    second = cell_graph_to_pyg_training_data(graphs[1], cfg)
    x_index = feature_names.index("temporal_lag1_x")
    delta_x_index = feature_names.index("temporal_lag1_delta_x")
    assert second.x[:, x_index].tolist()[:3] == [0.0, 10.0, 10.0]
    assert second.x[:, delta_x_index].tolist()[:3] == [1.0, -1.0, 1.0]


def test_feature_guard_rejects_target_and_linkage_columns() -> None:
    spots = _sample_spots()
    with pytest.raises(ValueError, match="leak"):
        default_node_feature_columns(
            spots,
            FrameGraphDatasetConfig(node_feature_columns=("x", "target_division")),
        )
    with pytest.raises(ValueError, match="leak"):
        add_temporal_features(
            spots,
            FrameGraphDatasetConfig(temporal_lags=(1,), temporal_feature_columns=("x", "n_next")),
        )


def test_build_frame_graphs_creates_one_graph_per_frame_without_cross_frame_edges() -> None:
    graphs = build_frame_graphs(_sample_spots(), _cfg())

    assert len(graphs) == 2
    first = graphs[0]
    second = graphs[1]
    assert first.num_nodes == 2
    assert second.num_nodes == 4

    first_pairs = {
        (first.nodes.at[source, "spot_id"], first.nodes.at[target, "spot_id"])
        for source, target in first.edge_index.T
    }
    assert first_pairs == set()

    second_pairs = {
        (second.nodes.at[source, "spot_id"], second.nodes.at[target, "spot_id"])
        for source, target in second.edge_index.T
    }
    assert second_pairs == {(4, 5), (5, 4)}
    assert all(source in {4, 5} and target in {4, 5} for source, target in second_pairs)


def test_pyg_training_data_contains_one_step_and_horizon_targets() -> None:
    graph = build_frame_graphs(_sample_spots(), _cfg())[0]
    data = cell_graph_to_pyg_training_data(graph, _cfg())

    assert data.sequence_uid == "seq_a"
    assert data.frame == 0
    assert tuple(data.pos_xy.shape) == (2, 2)
    assert data.pos_xy.tolist() == [[0.0, 0.0], [10.0, 0.0]]
    assert tuple(data.target_delta_pos.shape) == (2, 2)
    assert tuple(data.target_delta_shape.shape) == (2, 2)
    assert data.valid_regression_mask.tolist() == [True, False]
    assert data.valid_shape_mask.tolist() == [True, False]
    assert data.target_division.tolist() == [0.0, 1.0]
    assert data.target_death.tolist() == [0.0, 0.0]
    assert data.valid_event_mask.dtype == torch.bool
    assert data.target_division_within_3.tolist() == [0.0, 1.0]
    assert data.valid_division_within_3.tolist() == [True, True]


def test_default_node_features_do_not_include_future_leakage_columns() -> None:
    prepared = add_one_step_targets(_sample_spots(), _cfg())
    features = default_node_feature_columns(prepared, FrameGraphDatasetConfig())

    assert "target_dx" not in features
    assert "target_division" not in features
    assert "division_within_3_frames" not in features
    assert "next_id" not in features
    assert "shape_r_norm_000" in features


def test_real_processed_spots_smoke_builds_pyg_graph() -> None:
    candidates = (
        Path(r"D:/Proga/Game_of_life/Real_game_of_life/HeLa_Database/shape_division_analysis_dynamic/spot_shape_division_dataset.parquet"),
        Path(r"D:/Proga/Game_of_life/Real_game_of_life/HeLa_Database/HeLa клетки/shape_division_analysis_dynamic/spot_shape_division_dataset.parquet"),
    )
    path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    if not path.exists():
        pytest.skip("processed HeLa parquet is not available")

    try:
        spots = load_processed_spots(path)
    except ImportError as exc:
        pytest.skip(f"parquet support unavailable: {exc}")

    sequence = spots["sequence_uid"].iloc[0]
    subset = spots[spots["sequence_uid"].eq(sequence) & spots["frame"].isin([0, 1])].copy()
    graphs = build_frame_graphs(
        subset,
        FrameGraphDatasetConfig(
            source_path=path,
            edge_radius=40.0,
            node_feature_columns=("x", "y", "AREA", "SOLIDITY", "shape_r_norm_000", "shape_r_norm_001"),
            horizons=(3, 5, 10),
        ),
    )

    assert graphs
    data = cell_graph_to_pyg_training_data(graphs[0], FrameGraphDatasetConfig(horizons=(3, 5, 10)))
    assert data.x.ndim == 2
    assert data.edge_index.shape[0] == 2
    assert data.edge_attr.shape[0] == data.edge_index.shape[1]
    assert hasattr(data, "target_delta_pos")
    assert hasattr(data, "target_division_within_3")


def test_wrap_nematic_delta_picks_shortest_rotation_across_the_pi_boundary() -> None:
    assert wrap_nematic_delta(0.3) == pytest.approx(0.3)
    assert wrap_nematic_delta(-0.3) == pytest.approx(-0.3)
    # theta going from 3.10 to -1.55 is a small rotation once the pi-periodicity
    # (axis, not vector) is accounted for, not the raw ~-4.65 jump.
    raw_diff = -1.55 - 3.10
    wrapped = wrap_nematic_delta(raw_diff)
    assert -np.pi / 2 <= wrapped < np.pi / 2
    assert wrapped == pytest.approx(-1.5084073464102072)
    # the interval is half-open [-half, half): +half wraps down to -half.
    assert wrap_nematic_delta(np.pi / 2) == pytest.approx(-np.pi / 2)
    assert wrap_nematic_delta(-np.pi / 2) == pytest.approx(-np.pi / 2)
    np.testing.assert_allclose(wrap_nematic_delta(np.array([0.1, -0.1 - np.pi])), [0.1, -0.1])


def _sample_spots_with_polarization() -> pd.DataFrame:
    spots = _sample_spots()
    spots[POLARIZATION_ANGLE_COLUMN] = [3.10, 0.2, -1.55, 0.5, 1.0, -0.4]
    spots[POLARIZATION_MAGNITUDE_COLUMN] = [1.5, 2.0, 1.8, 2.2, 1.1, 3.0]
    return spots


def test_add_one_step_targets_computes_wrapped_polarization_deltas() -> None:
    prepared = add_one_step_targets(_sample_spots_with_polarization(), _cfg())
    by_id = prepared.set_index("spot_id")

    # spot 1 (frame 0) -> spot 3 (frame 1): theta 3.10 -> -1.55, aspect 1.5 -> 1.8
    assert bool(by_id.at[1, "valid_polarization_mask"]) is True
    assert by_id.at[1, "target_delta_ELLIPSE_THETA"] == pytest.approx(-1.5084073464102072)
    assert by_id.at[1, "target_delta_ELLIPSE_ASPECTRATIO"] == pytest.approx(0.3)

    # spot 3 (frame 1) has no next spot -> no polarization regression target.
    assert bool(by_id.at[3, "valid_polarization_mask"]) is False

    # spot 2 splits into two daughters -> not a valid single-next regression target.
    assert bool(by_id.at[2, "valid_polarization_mask"]) is False


def test_add_one_step_targets_skips_polarization_when_columns_absent() -> None:
    prepared = add_one_step_targets(_sample_spots(), _cfg())
    assert "target_delta_ELLIPSE_THETA" not in prepared.columns
    assert "target_delta_ELLIPSE_ASPECTRATIO" not in prepared.columns
    assert "valid_polarization_mask" not in prepared.columns


def test_cell_graph_to_pyg_training_data_attaches_polarization_targets() -> None:
    cfg = FrameGraphDatasetConfig(
        node_feature_columns=("x", "y", "AREA", "SOLIDITY", POLARIZATION_ANGLE_COLUMN, POLARIZATION_MAGNITUDE_COLUMN),
        edge_radius=3.0,
        horizons=(3, 5, 10),
    )
    graphs = build_frame_graphs(_sample_spots_with_polarization(), cfg)
    data = cell_graph_to_pyg_training_data(graphs[0], cfg)

    assert data.polarization_columns == [POLARIZATION_ANGLE_COLUMN, POLARIZATION_MAGNITUDE_COLUMN]
    assert data.target_delta_polarization.shape == (2, 2)
    assert bool(data.valid_polarization_mask[0]) is True
    assert float(data.target_delta_polarization[0, 0]) == pytest.approx(-1.5084073464102072, abs=1e-5)
    assert float(data.target_delta_polarization[0, 1]) == pytest.approx(0.3, abs=1e-5)
