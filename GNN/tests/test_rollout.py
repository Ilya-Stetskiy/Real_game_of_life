from __future__ import annotations

import pytest
import torch

from Real_game_of_life.GNN.gnn_model import CellGNNOutput, CellInteractionGNN
from Real_game_of_life.GNN.graph_dataset import FrameGraphDatasetConfig, build_frame_graphs, cell_graph_to_pyg_training_data
from Real_game_of_life.GNN.hybrid_field_gnn_model import FieldConditionedCellGNN
from Real_game_of_life.GNN.rollout import (
    RolloutGraphConfig,
    differentiable_prediction_to_next_graph,
    field_prediction_to_next_graph,
    prediction_to_next_graph,
)
from Real_game_of_life.GNN.tests.test_graph_dataset import _sample_spots
from Real_game_of_life.GNN.train_one_step import fit_node_feature_normalization


def _output(num_nodes: int, shape_dim: int, *, dx: float = 0.0, dy: float = 0.0) -> CellGNNOutput:
    delta_pos = torch.zeros((num_nodes, 2), dtype=torch.float32)
    delta_pos[:, 0] = dx
    delta_pos[:, 1] = dy
    delta_shape = torch.full((num_nodes, shape_dim), 0.25, dtype=torch.float32)
    return CellGNNOutput(
        delta_pos=delta_pos,
        delta_shape=delta_shape,
        division_logits=torch.zeros(num_nodes, dtype=torch.float32),
        death_logits=torch.full((num_nodes,), -10.0, dtype=torch.float32),
        node_embeddings=torch.zeros((num_nodes, 4), dtype=torch.float32),
    )


def _output_to(output: CellGNNOutput, device: torch.device | str) -> CellGNNOutput:
    return CellGNNOutput(
        delta_pos=output.delta_pos.to(device),
        delta_shape=output.delta_shape.to(device),
        division_logits=output.division_logits.to(device),
        death_logits=output.death_logits.to(device),
        node_embeddings=output.node_embeddings.to(device),
        trajectory_hypotheses=output.trajectory_hypotheses.to(device) if output.trajectory_hypotheses is not None else None,
        division_horizon_logits=output.division_horizon_logits.to(device) if output.division_horizon_logits is not None else None,
    )


def _frame_graph(edge_radius: float = 12.0):
    cfg = FrameGraphDatasetConfig(
        node_feature_columns=("x", "y", "AREA", "SOLIDITY", "shape_r_norm_000", "shape_r_norm_001"),
        edge_radius=edge_radius,
        horizons=(3, 5, 10),
    )
    return cell_graph_to_pyg_training_data(build_frame_graphs(_sample_spots(), cfg)[0], cfg)


def test_prediction_to_next_graph_updates_features_and_rebuilds_edges() -> None:
    graph = _frame_graph(edge_radius=12.0)
    next_graph = prediction_to_next_graph(
        graph,
        _output(graph.num_nodes, graph.target_delta_shape.size(1), dx=1.0, dy=2.0),
        config=RolloutGraphConfig(edge_radius=20.0),
    )

    assert next_graph.node_feature_columns == graph.node_feature_columns
    assert next_graph.edge_feature_columns == graph.edge_feature_columns
    assert next_graph.frame == graph.frame + 1
    assert next_graph.x[:, 0].tolist() == [1.0, 11.0]
    assert next_graph.x[:, 1].tolist() == [2.0, 2.0]
    assert next_graph.pos_xy.tolist() == [[1.0, 2.0], [11.0, 2.0]]
    assert next_graph.x[:, 4].tolist() == [1.25, 2.25]
    assert next_graph.valid_regression_mask.tolist() == [False, False]
    assert next_graph.edge_index.shape[0] == 2
    assert next_graph.edge_attr.shape[0] == next_graph.edge_index.shape[1]
    assert torch.allclose(next_graph.edge_attr[:, 2], torch.full((next_graph.edge_attr.size(0),), 10.0))


def test_prediction_to_next_graph_preserves_cpu_device() -> None:
    graph = _frame_graph(edge_radius=12.0)

    next_graph = prediction_to_next_graph(
        graph,
        _output(graph.num_nodes, graph.target_delta_shape.size(1)),
        config=RolloutGraphConfig(edge_radius=20.0),
    )

    assert next_graph.x.device == graph.x.device
    assert next_graph.edge_index.device == graph.x.device
    assert next_graph.edge_attr.device == graph.x.device
    assert next_graph.target_delta_pos.device == graph.x.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_prediction_to_next_graph_preserves_cuda_device() -> None:
    graph = _frame_graph(edge_radius=12.0).to("cuda")
    output = _output_to(_output(graph.num_nodes, graph.target_delta_shape.size(1)), "cuda")

    next_graph = prediction_to_next_graph(
        graph,
        output,
        config=RolloutGraphConfig(edge_radius=20.0),
    )

    assert next_graph.x.device.type == "cuda"
    assert next_graph.edge_index.device.type == "cuda"
    assert next_graph.edge_attr.device.type == "cuda"
    assert next_graph.target_delta_pos.device.type == "cuda"


def test_prediction_to_next_graph_respects_train_normalization_stats() -> None:
    graph = _frame_graph(edge_radius=12.0)
    stats = fit_node_feature_normalization([graph], epsilon=1e-6)
    normalized = graph.clone()
    normalized.x = (graph.x.float() - stats["mean"]) / stats["std"]

    output = _output(normalized.num_nodes, normalized.target_delta_shape.size(1), dx=1.0, dy=0.0)
    output.delta_pos[1, 0] = 0.0
    next_graph = prediction_to_next_graph(
        normalized,
        output,
        config=RolloutGraphConfig(keep_edge_topology=True),
        node_feature_normalization=stats,
    )
    physical_next_x = next_graph.x.float() * stats["std"] + stats["mean"]

    assert torch.allclose(physical_next_x[:, 0], torch.tensor([1.0, 10.0]))
    assert torch.allclose(physical_next_x[:, 4], graph.x[:, 4] + 0.25)
    assert torch.allclose(next_graph.edge_attr[:, 2], torch.full((next_graph.edge_attr.size(0),), 9.0))


def test_differentiable_prediction_to_next_graph_preserves_gradients_through_features_and_edges() -> None:
    graph = _frame_graph(edge_radius=12.0)
    delta_pos = torch.zeros((graph.num_nodes, 2), dtype=torch.float32, requires_grad=True)
    delta_shape = torch.zeros_like(graph.target_delta_shape, requires_grad=True)
    output = _output(graph.num_nodes, graph.target_delta_shape.size(1))
    output.delta_pos = delta_pos
    output.delta_shape = delta_shape

    next_graph = differentiable_prediction_to_next_graph(
        graph,
        output,
        config=RolloutGraphConfig(edge_radius=20.0),
    )
    loss = next_graph.x.sum() + next_graph.edge_attr[:, 2].sum()
    loss.backward()

    assert next_graph.x.requires_grad
    assert next_graph.pos_xy.requires_grad
    assert next_graph.edge_attr.requires_grad
    assert delta_pos.grad is not None
    assert delta_shape.grad is not None
    assert torch.isfinite(delta_pos.grad).all()
    assert torch.isfinite(delta_shape.grad).all()


def test_differentiable_prediction_to_next_graph_rebuilds_dynamic_radius_edges() -> None:
    graph = _frame_graph(edge_radius=12.0)
    delta_pos = torch.tensor([[0.0, 0.0], [20.0, 0.0]], requires_grad=True)
    output = _output(graph.num_nodes, graph.target_delta_shape.size(1))
    output.delta_pos = delta_pos

    next_graph = differentiable_prediction_to_next_graph(
        graph,
        output,
        config=RolloutGraphConfig(edge_radius=12.0),
    )

    assert next_graph.edge_index.shape == (2, 0)
    assert next_graph.edge_attr.shape == (0, graph.edge_attr.size(1))


def test_prediction_to_next_graph_shifts_temporal_lag_features() -> None:
    cfg = FrameGraphDatasetConfig(
        node_feature_columns=None,
        edge_radius=3.0,
        horizons=(3, 5, 10),
        temporal_lags=(1, 2),
        temporal_feature_columns=("x", "AREA"),
    )
    graph = cell_graph_to_pyg_training_data(build_frame_graphs(_sample_spots(), cfg)[1], cfg)
    columns = list(graph.node_feature_columns)
    x_index = columns.index("x")
    lag1_x_index = columns.index("temporal_lag1_x")
    lag1_delta_x_index = columns.index("temporal_lag1_delta_x")
    lag2_x_index = columns.index("temporal_lag2_x")

    next_graph = prediction_to_next_graph(
        graph,
        _output(graph.num_nodes, graph.target_delta_shape.size(1), dx=1.0, dy=0.0),
        config=RolloutGraphConfig(keep_edge_topology=True),
    )

    assert torch.allclose(next_graph.x[:, x_index], graph.x[:, x_index] + 1.0)
    assert torch.allclose(next_graph.x[:, lag1_x_index], graph.x[:, x_index])
    assert torch.allclose(next_graph.x[:, lag1_delta_x_index], torch.ones(graph.num_nodes))
    assert torch.allclose(next_graph.x[:, lag2_x_index], graph.x[:, lag1_x_index])


def test_prediction_to_next_graph_can_drop_disappeared_nodes() -> None:
    graph = _frame_graph(edge_radius=12.0)
    output = _output(graph.num_nodes, graph.target_delta_shape.size(1))
    output.death_logits = torch.tensor([10.0, -10.0])

    next_graph = prediction_to_next_graph(
        graph,
        output,
        config=RolloutGraphConfig(keep_edge_topology=True, drop_disappeared=True, disappearance_threshold=0.5),
    )

    assert next_graph.num_nodes == 1
    assert next_graph.node_ids == [graph.node_ids[1]]
    assert next_graph.edge_index.shape == (2, 0)


def test_prediction_to_next_graph_can_create_division_births_when_enabled() -> None:
    graph = _frame_graph(edge_radius=12.0)
    output = _output(graph.num_nodes, graph.target_delta_shape.size(1))
    output.division_logits = torch.tensor([10.0, -10.0])

    with pytest.warns(RuntimeWarning, match="do not create daughter nodes"):
        without_births = prediction_to_next_graph(
            graph,
            output,
            config=RolloutGraphConfig(edge_radius=20.0, enable_division_births=False),
        )
    with_births = prediction_to_next_graph(
        graph,
        output,
        config=RolloutGraphConfig(edge_radius=20.0, enable_division_births=True, division_birth_offset=(2.0, 0.0)),
    )

    assert without_births.num_nodes == graph.num_nodes
    assert with_births.num_nodes == graph.num_nodes + 1
    assert with_births.pos_xy[-1].tolist() == [graph.pos_xy[0, 0].item() + 2.0, graph.pos_xy[0, 1].item()]
    assert with_births.edge_index.shape[0] == 2
    assert with_births.edge_attr.shape[0] == with_births.edge_index.shape[1]


def test_model_prediction_can_be_reused_as_next_model_input() -> None:
    graph = _frame_graph(edge_radius=12.0)
    model = CellInteractionGNN(
        node_dim=graph.x.size(1),
        edge_dim=graph.edge_attr.size(1),
        shape_dim=graph.target_delta_shape.size(1),
        hidden_dim=16,
        num_message_passing_layers=1,
        num_division_horizons=3,
    )
    model.eval()

    with torch.no_grad():
        output = model(graph)
        next_graph = prediction_to_next_graph(
            graph,
            output,
            config=RolloutGraphConfig(edge_radius=12.0),
        )
        next_output = model(next_graph)

    assert next_graph.x.shape[1] == graph.x.shape[1]
    assert next_graph.edge_attr.shape[1] == graph.edge_attr.shape[1]
    assert next_output.delta_pos.shape == output.delta_pos.shape


def test_field_prediction_to_next_graph_carries_next_field_and_pos_xy() -> None:
    graph = _frame_graph(edge_radius=12.0)
    model = FieldConditionedCellGNN(
        node_dim=graph.x.size(1),
        edge_dim=graph.edge_attr.size(1),
        shape_dim=graph.target_delta_shape.size(1),
        field_channels=4,
        field_patch_radius=1,
        field_context_dim=8,
        hidden_dim=16,
        num_message_passing_layers=1,
        dropout=0.0,
        num_division_horizons=3,
        field_update_hidden_channels=8,
    )
    field = model.initial_field(batch_size=1, height=24, width=24)

    with torch.no_grad():
        output = model(graph, field)
        step = field_prediction_to_next_graph(
            graph,
            output,
            config=RolloutGraphConfig(edge_radius=12.0),
        )
        next_output = model(step.graph, step.field)

    assert tuple(step.field.shape) == tuple(field.shape)
    assert tuple(step.graph.pos_xy.shape) == (graph.num_nodes, 2)
    assert next_output.cell_output.delta_pos.shape == output.cell_output.delta_pos.shape
