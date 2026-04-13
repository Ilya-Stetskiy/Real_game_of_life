from __future__ import annotations

from pathlib import Path

from Real_game_of_life.GNN.dataset_cache import SplitConfig, build_graph_cache, save_graph_cache
from Real_game_of_life.GNN.graph_dataset import FrameGraphDatasetConfig
from Real_game_of_life.GNN.tabular_baseline import (
    TabularBaselineConfig,
    run_tabular_baseline,
    target_to_attrs,
)
from Real_game_of_life.GNN.tests.test_graph_dataset import _sample_spots


def test_target_to_attrs_maps_horizon_targets() -> None:
    assert target_to_attrs("division_h10") == (
        "target_division_within_10",
        "valid_division_within_10",
    )
    assert target_to_attrs("death") == ("target_death", "valid_event_mask")


def test_run_tabular_baseline_writes_summary_and_report(tmp_path: Path) -> None:
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
    cache["splits"] = {"train": [0], "val": [0], "test": [0]}
    cache_path = tmp_path / "cache.pt"
    save_graph_cache(cache, cache_path)

    result = run_tabular_baseline(
        TabularBaselineConfig(
            cache_path=cache_path,
            out_dir=tmp_path / "baseline",
            target="division_h10",
            random_forest_trees=5,
        )
    )

    assert result["split_counts"]["test"]["positives"] == 1
    assert "dummy_prior" in result["models"]
    assert "logistic_regression" in result["models"]
    assert "random_forest" in result["models"]
    assert "logistic_regression_top10_recall" in result["models"]["logistic_regression"]["test"]
    assert (tmp_path / "baseline" / "tabular_baseline_summary.json").exists()
    assert (tmp_path / "baseline" / "tabular_baseline_report.md").exists()
