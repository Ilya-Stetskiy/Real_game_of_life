from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from Real_game_of_life.GNN.evaluate_rollout import evaluate_model_rollout, normalization_tensors
from Real_game_of_life.GNN.gnn_model import CellGNNOutput
from Real_game_of_life.GNN.graph_dataset import FrameGraphDatasetConfig, build_frame_graphs, cell_graph_to_pyg_training_data
from Real_game_of_life.GNN.rollout import RolloutGraphConfig
from Real_game_of_life.GNN.tests.test_train_rollout_bptt import _ConstantDeltaModel, _track_graphs


def test_evaluate_model_rollout_reports_horizon_metrics() -> None:
    graphs = _track_graphs()
    model = _ConstantDeltaModel()

    result = evaluate_model_rollout(
        model,
        [graphs],
        horizons=(1, 2),
        rollout_config=RolloutGraphConfig(edge_radius=10.0),
        device=torch.device("cpu"),
    )

    by_horizon = {row["horizon"]: row for row in result}
    assert set(by_horizon) == {1, 2}
    assert by_horizon[1]["matched_nodes"] == 2
    assert by_horizon[2]["matched_nodes"] == 1
    assert by_horizon[1]["position_mean"] > 0
    assert by_horizon[2]["position_mean"] > by_horizon[1]["position_mean"]
    assert 0.0 <= by_horizon[1]["valid_shape_fraction"] <= 1.0
    # no polarization_columns on these graphs -> the metric key is present but empty (NaN).
    assert "polarization_theta_rmse" in by_horizon[1]
    assert "polarization_aspect_rmse" in by_horizon[1]


def _track_graphs_with_polarization():
    spots = pd.DataFrame(
        {
            "sequence_uid": ["seq_a", "seq_a", "seq_a"],
            "frame": [0, 1, 2],
            "spot_id": [1, 2, 3],
            "x": [0.0, 1.0, 3.0],
            "y": [0.0, 0.0, 0.0],
            "shape_r_norm_000": [1.0, 1.1, 1.3],
            "ELLIPSE_THETA": [0.2, 0.3, 0.5],
            "ELLIPSE_ASPECTRATIO": [1.5, 1.6, 1.8],
            "next_id": [2.0, 3.0, np.nan],
            "next_ids": ["2", "3", ""],
            "n_next": [1, 1, 0],
        }
    )
    cfg = FrameGraphDatasetConfig(
        node_feature_columns=("x", "y", "shape_r_norm_000", "ELLIPSE_THETA", "ELLIPSE_ASPECTRATIO"),
        edge_radius=10.0,
        horizons=(),
    )
    return [cell_graph_to_pyg_training_data(graph, cfg) for graph in build_frame_graphs(spots, cfg)]


class _ConstantPolarizationDeltaModel(_ConstantDeltaModel):
    def __init__(self) -> None:
        super().__init__()
        self.dtheta = nn.Parameter(torch.tensor(0.05))
        self.daspect = nn.Parameter(torch.tensor(0.05))

    def forward(self, graph) -> CellGNNOutput:
        output = super().forward(graph)
        num_nodes = int(graph.num_nodes)
        output.delta_polarization = torch.stack(
            [self.dtheta.expand(num_nodes), self.daspect.expand(num_nodes)], dim=1
        )
        return output


def test_evaluate_model_rollout_reports_polarization_rmse() -> None:
    graphs = _track_graphs_with_polarization()
    model = _ConstantPolarizationDeltaModel()

    result = evaluate_model_rollout(
        model,
        [graphs],
        horizons=(1, 2),
        rollout_config=RolloutGraphConfig(edge_radius=10.0),
        device=torch.device("cpu"),
    )

    by_horizon = {row["horizon"]: row for row in result}
    assert by_horizon[1]["polarization_theta_rmse"] > 0
    assert by_horizon[1]["polarization_aspect_rmse"] > 0
    assert by_horizon[2]["polarization_theta_rmse"] >= 0
    assert by_horizon[2]["polarization_aspect_rmse"] >= 0


def test_normalization_tensors_accepts_json_lists() -> None:
    stats = normalization_tensors({"mean": [1.0, 2.0], "std": [3.0, 4.0]})

    assert torch.equal(stats["mean"], torch.tensor([1.0, 2.0]))
    assert torch.equal(stats["std"], torch.tensor([3.0, 4.0]))
