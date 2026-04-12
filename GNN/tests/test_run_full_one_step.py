from __future__ import annotations

from pathlib import Path

from Real_game_of_life.GNN.dataset_cache import SplitConfig, build_graph_cache, save_graph_cache
from Real_game_of_life.GNN.graph_dataset import FrameGraphDatasetConfig
from Real_game_of_life.GNN.run_full_one_step import run_full_training
from Real_game_of_life.GNN.tests.test_graph_dataset import _sample_spots


def test_run_full_training_writes_server_artifacts(tmp_path: Path) -> None:
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

    summary = run_full_training(
        preset="smoke",
        cache_path=cache_path,
        out_dir=tmp_path / "full_run",
        build_cache_if_missing=False,
        train_overrides={
            "epochs": 1,
            "batch_size": 2,
            "hidden_dim": 16,
            "layers": 1,
            "dropout": 0.0,
            "device": "cpu",
        },
    )

    out_dir = tmp_path / "full_run"
    assert summary["training_summary"]["epochs_completed"] == 1
    assert (out_dir / "best.pt").exists()
    assert (out_dir / "last.pt").exists()
    assert (out_dir / "history.csv").exists()
    assert (out_dir / "environment.json").exists()
    assert (out_dir / "effective_config.json").exists()
    assert (out_dir / "full_run_summary.json").exists()
    assert (out_dir / "final_report.md").exists()
