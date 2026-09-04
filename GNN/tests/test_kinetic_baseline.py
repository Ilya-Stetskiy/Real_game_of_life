from __future__ import annotations

import math

import pytest
import torch
from torch_geometric.data import Data

from Real_game_of_life.GNN.kinetic_baseline import (
    KineticPersistentRandomWalkModel,
    calibrate_persistent_random_walk,
)


def _synthetic_graph(*, vx: list[float], vy: list[float], next_dx: list[float], next_dy: list[float], has_ancestor: list[float]) -> Data:
    n = len(vx)
    feature_columns = ["x", "y", "temporal_lag1_has_ancestor", "temporal_lag1_delta_x", "temporal_lag1_delta_y"]
    x = torch.zeros((n, len(feature_columns)), dtype=torch.float32)
    x[:, feature_columns.index("temporal_lag1_has_ancestor")] = torch.tensor(has_ancestor)
    x[:, feature_columns.index("temporal_lag1_delta_x")] = torch.tensor(vx)
    x[:, feature_columns.index("temporal_lag1_delta_y")] = torch.tensor(vy)
    data = Data(x=x, edge_index=torch.zeros((2, 0), dtype=torch.long), edge_attr=torch.zeros((0, 1)))
    data.node_feature_columns = feature_columns
    data.target_delta_pos = torch.stack([torch.tensor(next_dx), torch.tensor(next_dy)], dim=1)
    data.valid_regression_mask = torch.ones(n, dtype=torch.bool)
    return data


def test_calibrate_persistent_random_walk_recovers_known_phi() -> None:
    # v_{t+1} = 0.5 * v_t exactly -> calibration should recover phi=0.5.
    graph = _synthetic_graph(
        vx=[2.0, 4.0, -1.0],
        vy=[1.0, 0.0, 3.0],
        next_dx=[1.0, 2.0, -0.5],
        next_dy=[0.5, 0.0, 1.5],
        has_ancestor=[1.0, 1.0, 1.0],
    )

    calibration = calibrate_persistent_random_walk([graph], node_feature_columns=graph.node_feature_columns)

    assert calibration["phi"] == pytest.approx(0.5)
    assert calibration["tau_frames"] == pytest.approx(-1.0 / math.log(0.5))
    assert calibration["n_pairs"] == 3


def test_calibrate_persistent_random_walk_ignores_nodes_without_ancestor() -> None:
    graph = _synthetic_graph(
        vx=[2.0, 999.0],
        vy=[1.0, 999.0],
        next_dx=[1.0, 0.0],
        next_dy=[0.5, 0.0],
        has_ancestor=[1.0, 0.0],
    )

    calibration = calibrate_persistent_random_walk([graph], node_feature_columns=graph.node_feature_columns)

    assert calibration["n_pairs"] == 1
    assert calibration["phi"] == pytest.approx(0.5)


def test_calibrate_persistent_random_walk_requires_temporal_lag_features() -> None:
    graph = _synthetic_graph(vx=[1.0], vy=[1.0], next_dx=[1.0], next_dy=[1.0], has_ancestor=[1.0])
    graph.node_feature_columns = ["x", "y"]

    try:
        calibrate_persistent_random_walk([graph], node_feature_columns=graph.node_feature_columns)
    except ValueError as exc:
        assert "temporal_lag1_delta" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_kinetic_model_scales_velocity_by_phi() -> None:
    graph = _synthetic_graph(
        vx=[2.0, -4.0],
        vy=[1.0, 3.0],
        next_dx=[0.0, 0.0],
        next_dy=[0.0, 0.0],
        has_ancestor=[1.0, 1.0],
    )
    model = KineticPersistentRandomWalkModel(phi=0.25, shape_dim=3, node_feature_columns=graph.node_feature_columns)

    output = model(graph)

    assert torch.allclose(output.delta_pos, 0.25 * torch.tensor([[2.0, 1.0], [-4.0, 3.0]]))
    assert output.delta_shape.shape == (2, 3)
    assert torch.all(output.delta_shape == 0.0)
    assert output.delta_polarization is None
