from __future__ import annotations

from pathlib import Path

import torch

from Real_game_of_life.GNN.dataset_cache import SplitConfig, build_graph_cache, save_graph_cache
from Real_game_of_life.GNN.graph_dataset import FrameGraphDatasetConfig
from Real_game_of_life.GNN.tests.test_graph_dataset import _sample_spots
from Real_game_of_life.GNN.train_field_sequence import (
    SequenceTrainConfig,
    build_sequence_model,
    group_graphs_by_sequence,
    run_sequence_epoch,
    train_from_cache,
)
from Real_game_of_life.GNN.train_one_step import (
    infer_division_horizons,
    infer_shape_dim,
    target_pos_weight,
    target_pos_weights_for_division_horizons,
)


def test_group_graphs_by_sequence_sorts_frames() -> None:
    cfg = FrameGraphDatasetConfig(
        node_feature_columns=("x", "y", "AREA", "SOLIDITY", "shape_r_norm_000", "shape_r_norm_001"),
        edge_radius=3.0,
        horizons=(3, 5, 10),
    )
    cache = build_graph_cache(
        spots=_sample_spots(),
        dataset_config=cfg,
        split_config=SplitConfig(mode="none"),
    )
    graphs = list(reversed(cache["graphs"]))

    sequences = group_graphs_by_sequence(graphs)

    assert len(sequences) == 1
    assert [graph.frame for graph in sequences[0]] == [0, 1]


def test_train_field_sequence_runs_smoke_epoch(tmp_path: Path) -> None:
    cfg = FrameGraphDatasetConfig(
        node_feature_columns=("x", "y", "AREA", "SOLIDITY", "shape_r_norm_000", "shape_r_norm_001"),
        edge_radius=3.0,
        horizons=(3, 5, 10),
    )
    cache = build_graph_cache(
        spots=_sample_spots(),
        dataset_config=cfg,
        split_config=SplitConfig(mode="none"),
    )
    cache_path = tmp_path / "cache.pt"
    save_graph_cache(cache, cache_path)

    result = train_from_cache(
        SequenceTrainConfig(
            cache_path=cache_path,
            out_dir=tmp_path / "field_sequence",
            epochs=1,
            hidden_dim=16,
            layers=1,
            dropout=0.0,
            learning_rate=1e-3,
            device="cpu",
            field_channels=4,
            field_height=40,
            field_width=40,
            field_context_dim=8,
            field_update_hidden_channels=8,
            bptt_window=3,
        )
    )

    assert result["summary"]["model_type"] == "field_sequence"
    assert result["summary"]["train_sequences"] == 1
    assert result["summary"]["config"]["bptt_window"] == 3
    assert result["history"][0]["sequences"] == 1.0
    assert "pos_rmse" in result["history"][0]
    assert "shape_rmse" in result["history"][0]
    assert "division_ap" in result["history"][0]
    assert "division_h3_ap" in result["history"][0]
    assert (tmp_path / "field_sequence" / "best.pt").exists()
    assert (tmp_path / "field_sequence" / "last.pt").exists()
    assert (tmp_path / "field_sequence" / "history.csv").exists()


def test_sequence_field_update_gets_gradients_with_bptt_window() -> None:
    cfg = FrameGraphDatasetConfig(
        node_feature_columns=("x", "y", "AREA", "SOLIDITY", "shape_r_norm_000", "shape_r_norm_001"),
        edge_radius=3.0,
        horizons=(3, 5, 10),
    )
    cache = build_graph_cache(
        spots=_sample_spots(),
        dataset_config=cfg,
        split_config=SplitConfig(mode="none"),
    )
    graphs = cache["graphs"]
    sequence = [graphs[0], graphs[1], graphs[0].clone()]
    for index, graph in enumerate(sequence):
        graph.frame = index

    config = SequenceTrainConfig(
        hidden_dim=16,
        layers=1,
        dropout=0.0,
        field_channels=4,
        field_height=40,
        field_width=40,
        field_context_dim=8,
        field_update_hidden_channels=8,
        bptt_window=3,
        field_auto_geometry=False,
        min_field_coverage=0.0,
    )
    horizons = infer_division_horizons(graphs, config.division_horizons)
    model = build_sequence_model(
        config,
        node_dim=graphs[0].x.size(-1),
        edge_dim=graphs[0].edge_attr.size(-1),
        shape_dim=infer_shape_dim(graphs),
        num_division_horizons=len(horizons),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    run_sequence_epoch(
        model,
        [sequence],
        device=torch.device("cpu"),
        optimizer=optimizer,
        config=config,
        division_horizons=horizons,
        pos_weight_division=target_pos_weight(graphs, "target_division", "valid_event_mask", 100.0),
        pos_weight_death=target_pos_weight(graphs, "target_death", "valid_event_mask", 100.0),
        pos_weight_division_horizon=target_pos_weights_for_division_horizons(graphs, horizons, 100.0),
    )

    assert any(parameter.grad is not None for parameter in model.field_update.parameters())
