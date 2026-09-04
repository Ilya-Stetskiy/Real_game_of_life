from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

from Real_game_of_life.GNN.cpm_baseline import (
    CPMBaselineModel,
    CPMParameters,
    _clamp_aspect_ratio,
    _ellipse_from_mask,
    _ellipse_from_second_moments,
    _ellipse_pixel_offsets,
    _find_free_pixel,
    _parse_final_frame,
    _seed_pixel_grid,
    _usable_pixel_values,
    _write_cfg,
    _write_pgm,
    calibrate_cpm_volume,
    calibrated_parameters,
)
from Real_game_of_life.GNN.graph_dataset import wrap_nematic_delta


def _synthetic_graph(*, x: list[float], y: list[float], area: list[float]) -> Data:
    n = len(x)
    feature_columns = ["x", "y", "AREA"]
    data_x = torch.zeros((n, len(feature_columns)), dtype=torch.float32)
    data_x[:, feature_columns.index("x")] = torch.tensor(x)
    data_x[:, feature_columns.index("y")] = torch.tensor(y)
    data_x[:, feature_columns.index("AREA")] = torch.tensor(area)
    data = Data(x=data_x, edge_index=torch.zeros((2, 0), dtype=torch.long), edge_attr=torch.zeros((0, 1)))
    data.node_feature_columns = feature_columns
    data.valid_regression_mask = torch.ones(n, dtype=torch.bool)
    data.pos_xy = torch.stack([torch.tensor(x), torch.tensor(y)], dim=1)
    return data


# --------------------------------------------------------------------------------------
# Pure geometry -- no engine dependency.
# --------------------------------------------------------------------------------------


def test_ellipse_round_trip_axis_aligned() -> None:
    offsets = _ellipse_pixel_offsets(semi_major=20.0, semi_minor=8.0, theta=0.0)
    mask = np.zeros((60, 60), dtype=bool)
    for dx, dy in offsets:
        mask[30 + dy, 30 + dx] = True

    theta, aspect = _ellipse_from_mask(mask)

    assert theta == pytest.approx(0.0, abs=0.1)
    assert aspect == pytest.approx(20.0 / 8.0, rel=0.1)


def test_ellipse_round_trip_rotated() -> None:
    true_theta = 0.7
    offsets = _ellipse_pixel_offsets(semi_major=18.0, semi_minor=6.0, theta=true_theta)
    mask = np.zeros((60, 60), dtype=bool)
    for dx, dy in offsets:
        mask[30 + dy, 30 + dx] = True

    theta, aspect = _ellipse_from_mask(mask)

    assert float(wrap_nematic_delta(theta - true_theta)) == pytest.approx(0.0, abs=0.15)
    assert aspect == pytest.approx(3.0, rel=0.15)


def test_ellipse_from_mask_degenerate_mask_returns_default() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[5, 5] = True

    theta, aspect = _ellipse_from_mask(mask)

    assert theta == 0.0
    assert aspect == 1.0


def test_ellipse_from_second_moments_matches_cpm_hela_convention() -> None:
    # cpm-hela-model's own SquareCellGrid::computeSecondMoments() uses the plain
    # image-moment convention (no swap/sign flip, unlike a physics-inertia-tensor
    # convention) -- empirically confirmed by seeding a known (theta=0.7, aspect=3.0)
    # ellipse through the real engine and reading back MU20/MU02/MU11 = (51.199,
    # 39.9659, 35.7946), which this formula recovers as theta=0.7076, aspect=2.958
    # (see plan Часть D шаг 2 verification). Regression guard against re-deriving this
    # wrong (e.g. by copying CC3D's swapped convention) without re-checking.
    theta, aspect = _ellipse_from_second_moments(mu20=51.199, mu02=39.9659, mu11=35.7946)

    assert theta == pytest.approx(0.7, abs=0.02)
    assert aspect == pytest.approx(3.0, abs=0.05)


def test_ellipse_from_second_moments_near_degenerate_shape_gives_extreme_aspect() -> None:
    # A near-line-shaped pixel mask drives lambda_minor toward its numerical floor,
    # producing an astronomically large (but finite) aspect ratio -- this is exactly
    # the value _clamp_aspect_ratio exists to bound before it corrupts a rollout.
    _theta, aspect = _ellipse_from_second_moments(mu20=1000.0, mu02=1e-12, mu11=0.0)

    assert aspect > 1e5


def test_clamp_aspect_ratio_bounds_extreme_values() -> None:
    clamped = _clamp_aspect_ratio(np.array([0.5, 3.0, 1e8, -5.0]))

    assert clamped[0] == pytest.approx(1.0)  # below 1 clamped up
    assert clamped[1] == pytest.approx(3.0)  # plausible value untouched
    assert clamped[2] < 1e6  # astronomical value clamped down
    assert clamped[3] == pytest.approx(1.0)


def test_usable_pixel_values_skips_reserved_and_respects_ceiling() -> None:
    values = _usable_pixel_values(5)

    assert len(values) == 5
    assert 0 not in values and 255 not in values
    assert set(values).isdisjoint({6, 17, 56, 68})

    with pytest.raises(ValueError, match="8-bit"):
        _usable_pixel_values(10_000)


# --------------------------------------------------------------------------------------
# PGM/CFG file generation and CSV parsing -- pure I/O, no subprocess.
# --------------------------------------------------------------------------------------


def test_seed_pixel_grid_places_expected_pixel_values(tmp_path: Path) -> None:
    positions = np.array([[10.0, 10.0], [50.0, 10.0]])
    thetas = np.array([0.0, 0.0])
    aspects = np.array([1.0, 1.0])
    areas = np.array([80.0, 80.0])
    values = _usable_pixel_values(2)

    grid, origin = _seed_pixel_grid(
        positions=positions, thetas=thetas, aspects=aspects, areas=areas, pixel_values=values, padding=20.0
    )

    assert set(np.unique(grid).tolist()) == {0, *values}
    # Each cell's centroid pixel should carry its own assigned value.
    cx0, cy0 = positions[0] - origin
    cx1, cy1 = positions[1] - origin
    assert grid[int(round(cy0)), int(round(cx0))] == values[0]
    assert grid[int(round(cy1)), int(round(cx1))] == values[1]


def test_seed_pixel_grid_handles_duplicate_centroids_without_dropping_a_cell() -> None:
    # Two real cells landing on the exact same rounded pixel (near-duplicate centroids,
    # or after several MCS of adhesion pulling them together across rollout steps) must
    # not make one of them vanish from the lattice -- that produced NaN position deltas
    # (see plan Часть D verification: matched a real dataset run before this fix).
    positions = np.array([[30.0, 30.0], [30.0, 30.0], [30.4, 30.4]])
    thetas = np.zeros(3)
    aspects = np.ones(3)
    areas = np.array([20.0, 20.0, 20.0])  # small ellipses so they don't naturally overlap-cover
    values = _usable_pixel_values(3)

    grid, _ = _seed_pixel_grid(
        positions=positions, thetas=thetas, aspects=aspects, areas=areas, pixel_values=values, padding=20.0
    )

    present = set(np.unique(grid).tolist())
    assert set(values).issubset(present)


def test_find_free_pixel_returns_nearest_unclaimed() -> None:
    claimed = {(5, 5)}
    row, col = _find_free_pixel(claimed, 5, 5, height=20, width=20)

    assert (row, col) not in claimed
    assert max(abs(row - 5), abs(col - 5)) == 1  # nearest ring


def test_write_pgm_round_trip_header(tmp_path: Path) -> None:
    grid = np.zeros((10, 20), dtype=np.uint8)
    grid[5, 5] = 42
    path = tmp_path / "frame.pgm"

    _write_pgm(path, width=20, height=10, pixels=grid)
    raw = path.read_bytes()

    assert raw.startswith(b"P5\n")
    assert b"20 10" in raw
    # header ends right before the pixel bytes: last header line is "255\n"
    header_end = raw.index(b"255\n") + len(b"255\n")
    pixel_bytes = raw[header_end:]
    assert len(pixel_bytes) == 200
    assert pixel_bytes[5 * 20 + 5] == 42


def test_write_cfg_embeds_parameters(tmp_path: Path) -> None:
    params = CPMParameters(target_volume=333.0, boltz_temp=12.5, contact_energy_cell=99.0)
    path = tmp_path / "frame.cfg"

    _write_cfg(path, params, image_name="frame", max_mcs=7, pixel_values=[1, 2, 3])
    text = path.read_text(encoding="utf-8")

    assert "SIM_PARAM,BOLTZ_TEMP,12.5" in text
    assert "VOLUME,333" in text
    assert "MAP_TEMPLATE,1,2" in text
    assert "MAP_TEMPLATE,2,2" in text
    assert "MAP_TEMPLATE,3,2" in text


def test_parse_final_frame_keeps_only_last_mcs_and_seeded_ids(tmp_path: Path) -> None:
    # SUPER_ID 0 is always the medium/space "cell" (see _parse_final_frame docstring),
    # never one of our seeded cells -- our cells are always SUPER_ID 1..num_cells,
    # regardless of which pixel byte values we chose to paint them with.
    csv_path = tmp_path / "sim_positions.csv"
    csv_path.write_text(
        "MCS,SUPER_ID,X,Y,MU20,MU02,MU11\n"
        "0,0,5.0,5.0,1.0,1.0,0.0\n"
        "0,1,10.0,10.0,5.0,5.0,0.0\n"
        "1,0,5.5,5.5,1.0,1.0,0.0\n"
        "1,1,11.0,10.0,5.0,5.0,0.0\n"
        "1,2,20.0,20.0,3.0,3.0,0.0\n",
        encoding="utf-8",
    )

    final = _parse_final_frame(csv_path, num_cells=2)

    assert list(final.index) == [1, 2]
    assert final.loc[1, "X"] == pytest.approx(11.0)


def test_parse_final_frame_reindexes_missing_cell_to_nan_row(tmp_path: Path) -> None:
    # If a cell's SUPER_ID never appears at the final MCS (shouldn't normally happen,
    # since the engine protects a cell's last pixel from removal, but this keeps the
    # failure mode visible/debuggable as NaN rather than silently misaligning rows).
    csv_path = tmp_path / "sim_positions.csv"
    csv_path.write_text(
        "MCS,SUPER_ID,X,Y,MU20,MU02,MU11\n"
        "0,1,10.0,10.0,5.0,5.0,0.0\n"
        "0,2,20.0,20.0,3.0,3.0,0.0\n",
        encoding="utf-8",
    )

    final = _parse_final_frame(csv_path, num_cells=3)

    assert list(final.index) == [1, 2, 3]
    assert bool(pd.isna(final.loc[3, "X"]))


# --------------------------------------------------------------------------------------
# Calibration -- pure numpy, no simulation.
# --------------------------------------------------------------------------------------


def test_calibrate_cpm_volume_computes_mean() -> None:
    graph = _synthetic_graph(x=[0.0, 10.0], y=[0.0, 10.0], area=[100.0, 200.0])

    stats = calibrate_cpm_volume([graph], node_feature_columns=graph.node_feature_columns)

    assert stats["target_volume"] == pytest.approx(150.0)
    assert stats["n_cells"] == 2


def test_calibrate_cpm_volume_requires_area() -> None:
    graph = _synthetic_graph(x=[0.0], y=[0.0], area=[100.0])
    graph.node_feature_columns = ["x", "y"]

    with pytest.raises(ValueError, match="AREA"):
        calibrate_cpm_volume([graph], node_feature_columns=graph.node_feature_columns)


def test_calibrated_parameters_only_overrides_target_volume() -> None:
    graph = _synthetic_graph(x=[0.0], y=[0.0], area=[121.0])
    base = CPMParameters(mcs_per_frame=7, contact_energy_cell=3.5)

    params, stats = calibrated_parameters(
        [graph], node_feature_columns=graph.node_feature_columns, base=base
    )

    assert params.target_volume == pytest.approx(121.0)
    assert params.mcs_per_frame == 7
    assert params.contact_energy_cell == pytest.approx(3.5)
    assert stats["n_cells"] == 1


def test_cpm_parameters_json_round_trip(tmp_path: Path) -> None:
    params = CPMParameters(target_volume=175.0, mcs_per_frame=30)
    path = tmp_path / "params.json"

    params.to_json(path)
    loaded = CPMParameters.from_json(path)

    assert loaded == params


# --------------------------------------------------------------------------------------
# Real engine subprocess integration -- skipped unless a compiled cpm_hela binary is
# available (set CPM_HELA_BINARY, or place one at Real_game_of_life/GNN/bin/cpm_hela).
# --------------------------------------------------------------------------------------


def _find_binary() -> Path | None:
    env_path = os.environ.get("CPM_HELA_BINARY")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    default = Path(__file__).resolve().parents[1] / "bin" / "cpm_hela"
    return default if default.exists() else None


def test_cpm_model_forward_runs_a_real_simulation() -> None:
    binary = _find_binary()
    if binary is None:
        pytest.skip("no compiled cpm_hela binary available (set CPM_HELA_BINARY)")

    graph = _synthetic_graph(x=[60.0, 110.0], y=[60.0, 60.0], area=[120.0, 120.0])
    params = CPMParameters(mcs_per_frame=5, target_volume=120.0)
    model = CPMBaselineModel(
        params=params, shape_dim=0, node_feature_columns=graph.node_feature_columns, binary_path=binary
    )

    output = model(graph)

    assert output.delta_pos.shape == (2, 2)
    assert torch.isfinite(output.delta_pos).all()
