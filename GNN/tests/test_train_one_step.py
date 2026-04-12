from __future__ import annotations

from pathlib import Path

import torch

from Real_game_of_life.GNN.dataset_cache import SplitConfig, build_graph_cache, save_graph_cache
from Real_game_of_life.GNN.graph_dataset import FrameGraphDatasetConfig
from Real_game_of_life.GNN.tests.test_graph_dataset import _sample_spots
from Real_game_of_life.GNN.train_one_step import (
    TrainConfig,
    binary_average_precision,
    binary_classification_metrics,
    division_horizon_tensors,
    infer_division_horizons,
    train_from_cache,
)


def test_binary_classification_metrics_report_rare_event_counts() -> None:
    scores = torch.tensor([0.9, 0.7, 0.4, 0.1])
    target = torch.tensor([1, 0, 1, 0], dtype=torch.bool)

    metrics = binary_classification_metrics("division", scores, target)

    assert metrics["division_tp"] == 1.0
    assert metrics["division_fp"] == 1.0
    assert metrics["division_fn"] == 1.0
    assert metrics["division_tn"] == 1.0
    assert metrics["division_precision"] == 0.5
    assert metrics["division_recall"] == 0.5
    assert metrics["division_f1"] == 0.5
    assert metrics["division_ap"] == binary_average_precision(scores, target)


def test_train_from_cache_runs_smoke_epoch(tmp_path: Path) -> None:
    dataset_config = FrameGraphDatasetConfig(
        node_feature_columns=("x", "y", "AREA", "SOLIDITY", "shape_r_norm_000", "shape_r_norm_001"),
        edge_radius=3.0,
        horizons=(3, 5, 10),
    )
    cache = build_graph_cache(
        spots=_sample_spots(),
        dataset_config=dataset_config,
        split_config=SplitConfig(mode="none"),
    )
    cache_path = tmp_path / "cache.pt"
    save_graph_cache(cache, cache_path)

    result = train_from_cache(
        TrainConfig(
            cache_path=cache_path,
            out_dir=tmp_path / "run",
            epochs=2,
            batch_size=2,
            hidden_dim=16,
            layers=1,
            dropout=0.0,
            learning_rate=1e-3,
            device="cpu",
        )
    )

    assert result["summary"]["best_epoch"] in {1, 2}
    assert (tmp_path / "run" / "best.pt").exists()
    assert (tmp_path / "run" / "last.pt").exists()
    assert (tmp_path / "run" / "history.csv").exists()
    assert len(result["history"]) == 2
    assert "division_precision" in result["history"][0]
    assert "division_recall" in result["history"][0]
    assert "division_ap" in result["history"][0]
    assert result["summary"]["division_horizons"] == (3, 5, 10)
    assert "loss_division_horizon" in result["history"][0]
    assert "division_h3_ap" in result["history"][0]
    assert "division_h5_recall" in result["history"][0]


def test_division_horizon_tensors_stack_targets_and_masks() -> None:
    dataset_config = FrameGraphDatasetConfig(
        node_feature_columns=("x", "y", "AREA", "SOLIDITY", "shape_r_norm_000", "shape_r_norm_001"),
        edge_radius=3.0,
        horizons=(3, 5, 10),
    )
    cache = build_graph_cache(
        spots=_sample_spots(),
        dataset_config=dataset_config,
        split_config=SplitConfig(mode="none"),
    )
    graph = cache["graphs"][0]

    assert infer_division_horizons(cache["graphs"], (3, 5, 10, 20)) == (3, 5, 10)
    target, mask = division_horizon_tensors(graph, (3, 5, 10))

    assert tuple(target.shape) == (2, 3)
    assert tuple(mask.shape) == (2, 3)
    assert target.tolist() == [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    assert mask.tolist() == [[True, True, True], [True, True, True]]
