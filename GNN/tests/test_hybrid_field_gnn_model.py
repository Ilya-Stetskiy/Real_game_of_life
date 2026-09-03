from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from Real_game_of_life.GNN.graph_dataset import FrameGraphDatasetConfig, build_frame_graphs, cell_graph_to_pyg_training_data
from Real_game_of_life.GNN.hybrid_field_gnn_model import FieldConditionedCellGNN
from Real_game_of_life.GNN.tests.test_graph_dataset import _sample_spots


def _frame_graphs() -> list[Data]:
    cfg = FrameGraphDatasetConfig(
        node_feature_columns=("x", "y", "AREA", "SOLIDITY", "shape_r_norm_000", "shape_r_norm_001"),
        edge_radius=12.0,
        horizons=(3, 5, 10),
    )
    return [cell_graph_to_pyg_training_data(graph, cfg) for graph in build_frame_graphs(_sample_spots(), cfg)]


def _model(graph: Data) -> FieldConditionedCellGNN:
    return FieldConditionedCellGNN(
        node_dim=graph.x.size(-1),
        edge_dim=graph.edge_attr.size(-1),
        shape_dim=graph.target_delta_shape.size(-1),
        field_channels=4,
        field_patch_radius=1,
        field_context_dim=8,
        hidden_dim=16,
        num_message_passing_layers=1,
        dropout=0.0,
        num_trajectory_hypotheses=2,
        num_division_horizons=3,
        field_update_hidden_channels=8,
    )


def test_field_conditioned_gnn_predicts_cells_and_next_field() -> None:
    graph = _frame_graphs()[0]
    model = _model(graph)
    field = model.initial_field(batch_size=1, height=16, width=16)

    output = model(graph, field)

    assert tuple(output.cell_output.delta_pos.shape) == (graph.num_nodes, 2)
    assert tuple(output.cell_output.delta_shape.shape) == tuple(graph.target_delta_shape.shape)
    assert tuple(output.cell_output.division_logits.shape) == (graph.num_nodes,)
    assert tuple(output.cell_output.death_logits.shape) == (graph.num_nodes,)
    assert tuple(output.cell_output.node_embeddings.shape) == (graph.num_nodes, 16)
    assert tuple(output.cell_output.trajectory_hypotheses.shape) == (graph.num_nodes, 2, 2)
    assert tuple(output.cell_output.division_horizon_logits.shape) == (graph.num_nodes, 3)
    assert tuple(output.field_context.shape) == (graph.num_nodes, 8)
    assert tuple(output.field_next.shape) == tuple(field.shape)


def test_field_conditioned_gnn_uses_batch_index_for_batched_fields() -> None:
    graphs = _frame_graphs()
    batch = next(iter(DataLoader(graphs, batch_size=2, shuffle=False)))
    model = _model(graphs[0])
    field = model.initial_field(batch_size=2, height=40, width=40)

    output = model(batch, field)

    assert tuple(output.cell_output.delta_pos.shape) == (batch.num_nodes, 2)
    assert tuple(output.field_next.shape) == tuple(field.shape)
    assert tuple(output.field_context.shape) == (batch.num_nodes, 8)


def test_field_conditioned_gnn_requires_physical_coordinates() -> None:
    graph = _frame_graphs()[0]
    model = _model(graph)
    field = model.initial_field(batch_size=1, height=16, width=16)
    del graph.pos_xy

    with pytest.raises(ValueError, match="pos_xy"):
        model(graph, field)
