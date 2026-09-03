from __future__ import annotations

import pandas as pd

from Real_game_of_life.HeLa_Database.cell_division_prediction_model import add_temporal_features


def test_temporal_features_do_not_cross_sequence_for_reused_cell_uid() -> None:
    spots = pd.DataFrame(
        {
            "dataset": ["hela", "hela", "hela", "hela"],
            "sequence_uid": ["seq_a", "seq_a", "seq_b", "seq_b"],
            "cell_uid": [1, 1, 1, 1],
            "frame": [0, 1, 0, 1],
            "spot_id": [10, 11, 20, 21],
            "AREA": [100.0, 110.0, 200.0, 220.0],
            "ELLIPSE_MAJOR": [2.0, 2.0, 2.0, 2.0],
            "ELLIPSE_MINOR": [1.0, 1.0, 1.0, 1.0],
        }
    )

    temporal = add_temporal_features(spots, (1,))
    by_key = temporal.set_index(["sequence_uid", "frame"])

    assert pd.isna(by_key.at[("seq_b", 0), "AREA_lag1"])
    assert by_key.at[("seq_b", 1), "AREA_lag1"] == 200.0
    assert by_key.at[("seq_b", 1), "AREA_delta1"] == 20.0
