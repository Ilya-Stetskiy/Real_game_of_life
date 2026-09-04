from __future__ import annotations

from Real_game_of_life.GNN.robustness_sweep import aggregate_rows, summarize_values


def test_summarize_values_reports_spread_statistics() -> None:
    summary = summarize_values([1.0, 2.0, 3.0, 4.0])

    assert summary["n"] == 4
    assert summary["mean"] == 2.5
    assert summary["median"] == 2.5
    assert summary["min"] == 1.0
    assert summary["max"] == 4.0
    assert summary["std"] > 0


def test_summarize_values_handles_empty_input() -> None:
    summary = summarize_values([])

    assert summary["n"] == 0
    assert summary["mean"] is None


def test_aggregate_rows_groups_by_key_and_skips_missing_metrics() -> None:
    rows = [
        {"split_mode": "a", "seed": 1, "pos_rmse": 1.0, "shape_rmse": None},
        {"split_mode": "a", "seed": 2, "pos_rmse": 3.0, "shape_rmse": 0.5},
        {"split_mode": "b", "seed": 1, "pos_rmse": 10.0, "shape_rmse": 0.2},
    ]

    by_split = aggregate_rows(rows, group_keys=("split_mode",), metric_keys=("pos_rmse", "shape_rmse"))
    by_split_map = {row["split_mode"]: row for row in by_split}

    assert by_split_map["a"]["n_runs"] == 2
    assert by_split_map["a"]["pos_rmse"]["mean"] == 2.0
    assert by_split_map["a"]["shape_rmse"]["n"] == 1
    assert by_split_map["b"]["pos_rmse"]["mean"] == 10.0

    overall = aggregate_rows(rows, group_keys=(), metric_keys=("pos_rmse",))
    assert overall[0]["n_runs"] == 3
    assert overall[0]["pos_rmse"]["n"] == 3
