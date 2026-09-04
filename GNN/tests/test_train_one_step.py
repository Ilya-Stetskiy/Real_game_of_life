from __future__ import annotations

from pathlib import Path

import pytest
import torch

from Real_game_of_life.GNN.dataset_cache import SplitConfig, build_graph_cache, save_graph_cache
from Real_game_of_life.GNN.graph_dataset import FrameGraphDatasetConfig
from Real_game_of_life.GNN.spatial_field import FieldGeometry
from Real_game_of_life.GNN.tests.test_graph_dataset import _sample_spots, _sample_spots_with_polarization
from Real_game_of_life.GNN.train_one_step import (
    TrainConfig,
    binary_average_precision,
    binary_classification_metrics,
    batch_loss_counts,
    division_horizon_tensors,
    fit_node_feature_normalization,
    build_model,
    infer_division_horizons,
    report_field_coverage,
    top_k_metrics,
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
    assert metrics["division_top10_effective_k"] == 4.0
    assert metrics["division_top10_hits"] == 2.0
    assert metrics["division_top10_precision"] == 0.5
    assert metrics["division_top10_recall"] == 1.0


def test_top_k_metrics_report_hits_precision_and_recall() -> None:
    scores = torch.tensor([0.95, 0.80, 0.70, 0.10, 0.05])
    target = torch.tensor([0, 1, 0, 1, 0], dtype=torch.bool)

    metrics = top_k_metrics("division_h10", scores, target, top_ks=(1, 3, 10))

    assert metrics["division_h10_top1_hits"] == 0.0
    assert metrics["division_h10_top1_precision"] == 0.0
    assert metrics["division_h10_top1_recall"] == 0.0
    assert metrics["division_h10_top3_hits"] == 1.0
    assert metrics["division_h10_top3_precision"] == 1 / 3
    assert metrics["division_h10_top3_recall"] == 0.5
    assert metrics["division_h10_top10_effective_k"] == 5.0
    assert metrics["division_h10_top10_hits"] == 2.0
    assert metrics["division_h10_top10_recall"] == 1.0


def test_batch_loss_counts_use_valid_target_masks() -> None:
    class Batch:
        valid_regression_mask = torch.tensor([True, False, False])
        valid_shape_mask = torch.tensor([True, True, False])
        valid_polarization_mask = torch.tensor([True, False, False])
        valid_event_mask = torch.tensor([True, True, True])

    horizon_mask = torch.tensor([[True, False], [False, False], [True, True]])
    counts = batch_loss_counts(Batch(), horizon_mask)

    assert counts["loss_pos"] == 1
    assert counts["loss_shape"] == 2
    assert counts["loss_polarization"] == 1
    assert counts["loss_division"] == 3
    assert counts["loss_death"] == 3
    assert counts["loss_division_horizon"] == 3


def test_node_feature_normalization_uses_train_statistics() -> None:
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
    stats = fit_node_feature_normalization(cache["graphs"], epsilon=1e-6)
    x = torch.cat([graph.x.float() for graph in cache["graphs"]], dim=0)

    assert torch.allclose(stats["mean"], x.mean(dim=0))
    assert torch.all(stats["std"] > 0)


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


def test_train_from_cache_reports_polarization_rmse(tmp_path: Path) -> None:
    dataset_config = FrameGraphDatasetConfig(
        node_feature_columns=(
            "x",
            "y",
            "AREA",
            "SOLIDITY",
            "shape_r_norm_000",
            "shape_r_norm_001",
            "ELLIPSE_THETA",
            "ELLIPSE_ASPECTRATIO",
        ),
        edge_radius=3.0,
        horizons=(3, 5, 10),
    )
    cache = build_graph_cache(
        spots=_sample_spots_with_polarization(),
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

    train_row = result["history"][0]
    assert "polarization_theta_rmse" in train_row
    assert "polarization_aspect_rmse" in train_row
    assert torch.isfinite(torch.tensor(train_row["polarization_theta_rmse"]))
    assert torch.isfinite(torch.tensor(train_row["polarization_aspect_rmse"]))


def test_train_from_cache_runs_field_gnn_smoke_epoch(tmp_path: Path) -> None:
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
            out_dir=tmp_path / "field_run",
            epochs=1,
            batch_size=2,
            hidden_dim=16,
            layers=1,
            dropout=0.0,
            learning_rate=1e-3,
            device="cpu",
            model_type="field_gnn",
            field_channels=4,
            field_height=40,
            field_width=40,
            field_context_dim=8,
            field_update_hidden_channels=8,
        )
    )

    assert result["summary"]["model_type"] == "field_gnn"
    assert (tmp_path / "field_run" / "best.pt").exists()
    assert "loss_total" in result["history"][0]


def test_field_gnn_requires_pos_xy_but_gnn_only_does_not(tmp_path: Path) -> None:
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
    for graph in cache["graphs"]:
        del graph.pos_xy
    cache_path = tmp_path / "old_cache.pt"
    save_graph_cache(cache, cache_path)

    gnn_result = train_from_cache(
        TrainConfig(
            cache_path=cache_path,
            out_dir=tmp_path / "gnn_run",
            epochs=1,
            batch_size=2,
            hidden_dim=16,
            layers=1,
            dropout=0.0,
            device="cpu",
        )
    )
    assert gnn_result["summary"]["model_type"] == "gnn"

    with pytest.raises(ValueError, match="field models require physical coordinates data.pos_xy"):
        train_from_cache(
            TrainConfig(
                cache_path=cache_path,
                out_dir=tmp_path / "field_run",
                epochs=1,
                batch_size=2,
                hidden_dim=16,
                layers=1,
                dropout=0.0,
                device="cpu",
                model_type="field_gnn",
            )
        )


def test_one_step_field_gnn_freezes_recurrent_field_update() -> None:
    model = build_model(
        TrainConfig(model_type="field_gnn", hidden_dim=16, layers=1, field_channels=4, field_context_dim=8),
        node_dim=6,
        edge_dim=5,
        shape_dim=2,
        num_division_horizons=3,
    )

    assert all(not parameter.requires_grad for parameter in model.field_writer.parameters())
    assert all(not parameter.requires_grad for parameter in model.field_update.parameters())


def test_field_coverage_warning_and_strict_error() -> None:
    class Graph:
        pos_xy = torch.tensor([[0.0, 0.0], [510.0, 510.0]])

    with pytest.warns(RuntimeWarning, match="field coverage"):
        coverages = report_field_coverage(
            {"train": [Graph()]},
            field_height=128,
            field_width=128,
            geometry=FieldGeometry(origin_xy=(0.0, 0.0), cell_size=1.0),
            min_coverage=0.95,
            strict=False,
        )
    assert coverages["train"] == 0.5

    with pytest.raises(ValueError, match="field coverage"):
        report_field_coverage(
            {"train": [Graph()]},
            field_height=128,
            field_width=128,
            geometry=FieldGeometry(origin_xy=(0.0, 0.0), cell_size=1.0),
            min_coverage=0.95,
            strict=True,
        )


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
    report_field_coverage,
