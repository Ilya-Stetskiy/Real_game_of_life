from __future__ import annotations

import torch

from Real_game_of_life.GNN.evaluate_rollout import evaluate_model_rollout, normalization_tensors
from Real_game_of_life.GNN.rollout import RolloutGraphConfig
from Real_game_of_life.GNN.tests.test_train_rollout_bptt import _ConstantDeltaModel, _track_graphs


def test_evaluate_model_rollout_reports_horizon_metrics() -> None:
    graphs = _track_graphs()
    model = _ConstantDeltaModel()

    result = evaluate_model_rollout(
        model,
        [graphs],
        horizons=(1, 2),
        rollout_config=RolloutGraphConfig(edge_radius=10.0),
        device=torch.device("cpu"),
    )

    by_horizon = {row["horizon"]: row for row in result}
    assert set(by_horizon) == {1, 2}
    assert by_horizon[1]["matched_nodes"] == 2
    assert by_horizon[2]["matched_nodes"] == 1
    assert by_horizon[1]["position_mean"] > 0
    assert by_horizon[2]["position_mean"] > by_horizon[1]["position_mean"]
    assert 0.0 <= by_horizon[1]["valid_shape_fraction"] <= 1.0


def test_normalization_tensors_accepts_json_lists() -> None:
    stats = normalization_tensors({"mean": [1.0, 2.0], "std": [3.0, 4.0]})

    assert torch.equal(stats["mean"], torch.tensor([1.0, 2.0]))
    assert torch.equal(stats["std"], torch.tensor([3.0, 4.0]))
