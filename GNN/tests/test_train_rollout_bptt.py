from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from Real_game_of_life.GNN.gnn_model import CellGNNOutput
from Real_game_of_life.GNN.graph_dataset import FrameGraphDatasetConfig, build_frame_graphs, cell_graph_to_pyg_training_data
from Real_game_of_life.GNN.rollout import RolloutGraphConfig
from Real_game_of_life.GNN.train_rollout_bptt import match_next_graph_indices, rollout_window_loss


def _track_graphs():
    spots = pd.DataFrame(
        {
            "sequence_uid": ["seq_a", "seq_a", "seq_a"],
            "frame": [0, 1, 2],
            "spot_id": [1, 2, 3],
            "x": [0.0, 1.0, 3.0],
            "y": [0.0, 0.0, 0.0],
            "shape_r_norm_000": [1.0, 1.1, 1.3],
            "next_id": [2.0, 3.0, np.nan],
            "next_ids": ["2", "3", ""],
            "n_next": [1, 1, 0],
        }
    )
    cfg = FrameGraphDatasetConfig(
        node_feature_columns=("x", "y", "shape_r_norm_000"),
        edge_radius=10.0,
        horizons=(),
    )
    return [cell_graph_to_pyg_training_data(graph, cfg) for graph in build_frame_graphs(spots, cfg)]


class _ConstantDeltaModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dx = nn.Parameter(torch.tensor(0.5))
        self.shape_delta = nn.Parameter(torch.tensor(0.05))

    def forward(self, graph) -> CellGNNOutput:
        num_nodes = int(graph.num_nodes)
        delta_pos = torch.stack(
            [
                self.dx.expand(num_nodes),
                torch.zeros(num_nodes, device=self.dx.device),
            ],
            dim=1,
        )
        return CellGNNOutput(
            delta_pos=delta_pos,
            delta_shape=self.shape_delta.expand(num_nodes, 1),
            division_logits=torch.zeros(num_nodes, device=self.dx.device),
            death_logits=torch.zeros(num_nodes, device=self.dx.device),
            node_embeddings=torch.zeros((num_nodes, 1), device=self.dx.device),
        )


def test_match_next_graph_indices_follows_exact_target_position() -> None:
    graphs = _track_graphs()

    next_indices = match_next_graph_indices(
        graphs[0],
        graphs[1],
        torch.tensor([0], dtype=torch.long),
    )

    assert next_indices.tolist() == [0]


def test_rollout_window_loss_backpropagates_through_two_predicted_steps() -> None:
    graphs = _track_graphs()
    model = _ConstantDeltaModel()

    loss, stats = rollout_window_loss(
        model,
        graphs,
        start_index=0,
        rollout_steps=2,
        rollout_config=RolloutGraphConfig(edge_radius=10.0),
        lambda_pos=1.0,
        lambda_shape=0.25,
    )
    loss.backward()

    assert loss.requires_grad
    assert stats["steps"] == 2
    assert stats["matched_nodes"] == 2
    assert model.dx.grad is not None
    assert model.shape_delta.grad is not None
    assert torch.isfinite(model.dx.grad)
    assert torch.isfinite(model.shape_delta.grad)
