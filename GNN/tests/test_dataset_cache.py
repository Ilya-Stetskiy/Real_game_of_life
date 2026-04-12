from __future__ import annotations

from pathlib import Path

import torch

from Real_game_of_life.GNN.dataset_cache import (
    SplitConfig,
    build_graph_cache,
    build_splits,
    graph_group_key,
    load_graph_cache,
    save_graph_cache,
    split_group_keys,
)
from Real_game_of_life.GNN.graph_dataset import FrameGraphDatasetConfig
from Real_game_of_life.GNN.tests.test_graph_dataset import _sample_spots


def test_split_group_keys_keeps_all_groups_and_nonempty_test_for_many_groups() -> None:
    train, val, test = split_group_keys(
        [f"g{i}" for i in range(10)],
        train_fraction=0.7,
        val_fraction=0.15,
        test_fraction=0.15,
    )

    assert set(train).isdisjoint(val)
    assert set(train).isdisjoint(test)
    assert set(val).isdisjoint(test)
    assert sorted([*train, *val, *test]) == [f"g{i}" for i in range(10)]
    assert train
    assert val
    assert test


def test_graph_group_key_uses_position_when_available() -> None:
    class Graph:
        sequence_uid = "seq0007_HeLa-S3_nuc_pos12_q3_5c"

    assert graph_group_key(Graph(), "by_position") == "pos12"
    assert graph_group_key(Graph(), "by_sequence") == Graph.sequence_uid
    assert graph_group_key(Graph(), "none") == "__all__"


def test_build_graph_cache_saves_loads_and_summarizes(tmp_path: Path) -> None:
    cfg = FrameGraphDatasetConfig(
        node_feature_columns=("x", "y", "AREA", "SOLIDITY", "shape_r_norm_000", "shape_r_norm_001"),
        edge_radius=3.0,
        horizons=(3, 5, 10),
    )
    cache = build_graph_cache(
        spots=_sample_spots(),
        dataset_config=cfg,
        split_config=SplitConfig(mode="none"),
        max_graphs=None,
    )

    assert len(cache["graphs"]) == 2
    assert cache["splits"] == {"train": [0, 1], "val": [], "test": []}
    assert cache["summary"]["graphs"] == 2
    assert cache["summary"]["nodes"] == 6
    assert cache["summary"]["target_valid_regression"] == 1
    assert cache["summary"]["target_division"] == 1


def test_save_load_cache_roundtrip(tmp_path: Path) -> None:
    from Real_game_of_life.GNN.graph_dataset import build_pyg_frame_graphs
    from Real_game_of_life.GNN.dataset_cache import summarize_graphs

    cfg = FrameGraphDatasetConfig(
        node_feature_columns=("x", "y", "AREA", "SOLIDITY", "shape_r_norm_000", "shape_r_norm_001"),
        edge_radius=3.0,
        horizons=(3, 5, 10),
    )
    graphs = build_pyg_frame_graphs(_sample_spots(), cfg)
    splits = build_splits(graphs, SplitConfig(mode="none"))
    cache = {
        "version": 1,
        "graphs": graphs,
        "splits": splits,
        "summary": summarize_graphs(graphs, cfg, splits),
        "dataset_config": {},
        "split_config": {},
    }

    path = tmp_path / "graphs.pt"
    save_graph_cache(cache, path)
    loaded = load_graph_cache(path)

    assert path.exists()
    assert path.with_suffix(".pt.summary.json").exists()
    assert len(loaded["graphs"]) == len(graphs)
    assert loaded["summary"]["graphs"] == len(graphs)
    assert torch.equal(loaded["graphs"][0].x, graphs[0].x)
