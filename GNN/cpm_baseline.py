from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from .dataset_cache import SplitConfig, build_graph_cache, save_graph_cache
from .evaluate_rollout import evaluate_model_rollout, write_reports
from .gnn_model import CellGNNOutput
from .graph_dataset import (
    DEFAULT_PROCESSED_SPOTS,
    POLARIZATION_ANGLE_COLUMN,
    POLARIZATION_ANGLE_PERIOD,
    POLARIZATION_COLUMNS,
    POLARIZATION_MAGNITUDE_COLUMN,
    FrameGraphDatasetConfig,
    load_processed_spots,
    wrap_nematic_delta,
)
from .kinetic_baseline import merge_rollout_reports  # re-exported: reuse, don't duplicate
from .train_one_step import infer_shape_dim, jsonable, resolve_device
from .train_rollout_bptt import group_graphs_by_sequence, rollout_config_from_cache

__all__ = [
    "CPMParameters",
    "CPMBaselineConfig",
    "calibrate_cpm_volume",
    "CPMBaselineModel",
    "run_cpm_baseline",
    "merge_rollout_reports",
]

POSITION_X_COLUMN = "x"
POSITION_Y_COLUMN = "y"
AREA_COLUMN = "AREA"

# CPM engine cell types, matching cpm-hela-model's default.cfg convention exactly:
# type 0 = boundary (static, never touched), type 1 = medium/empty space, type 2 = HeLa.
BOUNDARY_TYPE = 0
MEDIUM_TYPE = 1
CELL_TYPE = 2

# The engine's PGM loader assigns a distinct SuperCell per distinct pixel BYTE value
# (see initializeGrid() in Main.cpp), so at most 254 real cells (byte values 1..254,
# reserving 0=medium and 255=boundary) can be seeded into one simulation call. It also
# hardcodes a handful of skipped values in its MAP_TEMPLATE,<start>,<type>,<count> range
# expansion; we sidestep that entirely by emitting one explicit MAP_TEMPLATE line per
# cell, but still avoid those specific values for parity/paranoia.
_RESERVED_PIXEL_VALUES = frozenset({0, 255, 6, 17, 56, 68})
_MAX_CELLS_PER_CALL = 256 - len(_RESERVED_PIXEL_VALUES) - 2  # minus 0 and 255 already counted

# A near-degenerate (almost line-like) pixel mask -- e.g. a cell reduced to a handful of
# nearly-collinear pixels by the Metropolis dynamics -- drives lambda_minor in
# _ellipse_from_second_moments toward its 1e-9 floor, which can blow the resulting
# aspect ratio up to an astronomically large, physically meaningless value. Since aspect
# deltas accumulate additively, unclamped, across rollout steps (see
# rollout._apply_polarization_delta -- unlike theta, aspect has no wraparound to bound
# it), one such glitch permanently corrupts that cell's seeded shape on every later
# step. The real dataset's own ELLIPSE_ASPECTRATIO tops out under 6 (see plan Часть A),
# so this ceiling is a generous numerical safety net, not a biological constraint.
_MAX_PLAUSIBLE_ASPECT_RATIO = 20.0


# --------------------------------------------------------------------------------------
# Parameters -- the one file a user edits to change an experiment (--params-json).
# Field defaults are the labmate's own calibrated cpm-hela-model values (default.cfg /
# https://github.com/Anqeliccom/cpm-hela-model), not guesses: MCS_HOUR_EST, LAMBDA,
# BOLTZ_TEMP, the HeLa J (contact energy) vector, and TEMPLATE volume all come straight
# from that file. Only target_volume gets overridden per-experiment from this dataset's
# own AREA statistics (see calibrate_cpm_volume) -- everything else stays at the
# labmate's fit unless the user deliberately changes it here.
# --------------------------------------------------------------------------------------


@dataclass
class CPMParameters:
    """Physical parameters for the labmate's cpm-hela-model CPM engine.

    contact_energy_* are the adhesion (J) terms -- NOT identifiable from centroid tracks
    alone, so they are left at the labmate's own calibrated defaults rather than fit
    here (see calibrate_cpm_volume for the one thing this module does calibrate).
    mcs_per_frame has no calibrated value: the source parquet has no real seconds/frame
    (see graph_dataset.py), so there is no way to convert the labmate's MCS_HOUR_EST
    into "MCS per our frame" -- it is a free parameter, tune/measure empirically.
    """

    lattice_padding_px: float = 40.0
    mcs_per_frame: int = 50
    boltz_temp: float = 37.0
    lambda_volume: float = 0.002683
    contact_energy_boundary: float = 1_000_000.0
    contact_energy_medium: float = 14.0
    contact_energy_cell: float = 35.0
    target_volume: float = 507.0

    @classmethod
    def from_json(cls, path: str | Path) -> "CPMParameters":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in data.items() if key in known})

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class CPMBaselineConfig:
    binary_path: Path = Path("cpm_hela")
    out_dir: Path = Path(__file__).resolve().parent / "runs" / "cpm_baseline"
    horizons: tuple[int, ...] = (1, 3, 5, 10, 20)
    max_graphs: int | None = None
    device: str = "auto"  # the engine itself is a separate CPU subprocess; unused here
    match_tolerance: float = 1e-4
    subprocess_timeout_seconds: float = 120.0
    keep_workdirs: bool = False


# --------------------------------------------------------------------------------------
# Calibration that does NOT require running a simulation (pure numpy, always testable).
# --------------------------------------------------------------------------------------


def calibrate_cpm_volume(
    graphs: Sequence[Any],
    *,
    node_feature_columns: Sequence[str],
) -> dict[str, float]:
    """Estimate target_volume directly from tracked cell AREA (mean over train split).

    No search involved. cpm-hela-model has no surface/perimeter energy term (only
    volume + adhesion), so unlike a CompuCell3D-style engine there is nothing analogous
    to calibrate for shape besides this.
    """

    if AREA_COLUMN not in node_feature_columns:
        raise ValueError(
            "calibrate_cpm_volume requires an 'AREA' node feature "
            "(it's in DEFAULT_SCALAR_FEATURES; check FrameGraphDatasetConfig.node_feature_columns)."
        )
    area_index = node_feature_columns.index(AREA_COLUMN)

    areas: list[float] = []
    for graph in graphs:
        valid = getattr(graph, "valid_regression_mask", None)
        values_area = graph.x[:, area_index]
        if valid is not None:
            values_area = values_area[valid.bool()]
        areas.extend(float(v) for v in values_area.tolist())

    if not areas:
        raise ValueError("No valid cells found to calibrate target_volume.")

    areas_arr = np.asarray(areas, dtype=np.float64)
    return {
        "target_volume": float(np.mean(areas_arr)),
        "target_volume_std": float(np.std(areas_arr)),
        "n_cells": int(areas_arr.size),
    }


def calibrated_parameters(
    graphs: Sequence[Any],
    *,
    node_feature_columns: Sequence[str],
    base: Optional[CPMParameters] = None,
) -> tuple[CPMParameters, dict[str, float]]:
    """Fill in target_volume from data; leaves adhesion/lambda/boltz_temp at `base`'s
    (the labmate's calibrated) values -- see CPMParameters docstring."""

    base = base or CPMParameters()
    stats = calibrate_cpm_volume(graphs, node_feature_columns=node_feature_columns)
    params = CPMParameters(**{**asdict(base), "target_volume": stats["target_volume"]})
    return params, stats


# --------------------------------------------------------------------------------------
# Pure-numpy ellipse geometry and second-moment math -- no engine dependency, always
# unit-testable. The (mu20, mu02, mu11) convention here matches cpm-hela-model's own
# SquareCellGrid::computeSecondMoments() exactly (same raw-image-moment definition it
# already used internally for divideCellShortAxis) -- empirically confirmed by seeding
# a known (theta=0.7, aspect=3.0) ellipse and reading back theta=0.708, aspect=2.96
# (see plan Часть D шаг 2 verification); no swap/sign correction needed, unlike a
# physics-moment-of-inertia convention (e.g. CompuCell3D's cell.iXX/iYY/iXY).
# --------------------------------------------------------------------------------------


def _ellipse_pixel_offsets(semi_major: float, semi_minor: float, theta: float) -> np.ndarray:
    """Integer (dx, dy) pixel offsets covering a filled ellipse of the given semi-axes,
    rotated by `theta` radians, centered at (0, 0). Used to rasterize a real cell's
    ELLIPSE_THETA/ASPECTRATIO/AREA onto the CPM lattice for seeding.
    """

    semi_major = max(float(semi_major), 0.5)
    semi_minor = max(float(semi_minor), 0.5)
    radius = int(math.ceil(max(semi_major, semi_minor))) + 1
    ys, xs = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    # Rotate pixel coordinates into the ellipse's own (major, minor) frame.
    u = xs * cos_t + ys * sin_t
    v = -xs * sin_t + ys * cos_t
    inside = (u / semi_major) ** 2 + (v / semi_minor) ** 2 <= 1.0
    offsets = np.stack([xs[inside], ys[inside]], axis=1)
    return offsets.astype(np.int64)


def _ellipse_from_second_moments(mu20: float, mu02: float, mu11: float) -> tuple[float, float]:
    """Nematic orientation (radians, wrapped to period pi) and aspect ratio (>=1) from
    the image-moment convention second-moment matrix [[mu20, mu11], [mu11, mu02]].
    Closed-form eigendecomposition of the symmetric 2x2 matrix -- avoids
    np.linalg.eigh, which segfaults in at least one local numpy/MKL build once torch
    has been imported (torch's bundled MKL runtime conflicts with numpy's); a 2x2
    symmetric matrix has a trivial closed form anyway.
    """

    trace = mu20 + mu02
    diff = mu20 - mu02
    discriminant = math.sqrt(diff * diff + 4.0 * mu11 * mu11)
    lambda_major = max((trace + discriminant) / 2.0, 1e-9)
    lambda_minor = max((trace - discriminant) / 2.0, 1e-9)

    theta = 0.5 * math.atan2(2.0 * mu11, diff)
    theta = wrap_nematic_delta(theta, POLARIZATION_ANGLE_PERIOD)
    aspect = float(math.sqrt(lambda_major / lambda_minor))
    return float(theta), aspect


def _clamp_aspect_ratio(aspect: np.ndarray) -> np.ndarray:
    """Clamp a computed aspect ratio to [1.0, _MAX_PLAUSIBLE_ASPECT_RATIO] -- see that
    constant's docstring for why this guard exists."""

    return np.clip(aspect, 1.0, _MAX_PLAUSIBLE_ASPECT_RATIO)


def _ellipse_from_mask(mask: np.ndarray) -> tuple[float, float]:
    """Fit a nematic orientation and aspect ratio to a boolean 2D pixel mask via its
    second-moment (covariance) matrix. Returns (0.0, 1.0) for a degenerate/too-small
    mask. Used for seeding-side sanity checks/tests."""

    ys, xs = np.nonzero(mask)
    if xs.size < 3:
        return 0.0, 1.0
    xs = xs.astype(np.float64) - xs.mean()
    ys = ys.astype(np.float64) - ys.mean()
    mu20 = float(np.mean(xs * xs))
    mu02 = float(np.mean(ys * ys))
    mu11 = float(np.mean(xs * ys))
    return _ellipse_from_second_moments(mu20, mu02, mu11)


def _usable_pixel_values(n: int) -> list[int]:
    """The first `n` byte values in [1, 254] usable as distinct per-cell SuperCell
    labels in the seed PGM (see _RESERVED_PIXEL_VALUES). Raises ValueError if the
    engine's 8-bit-label ceiling can't fit `n` cells in one simulation call."""

    values = [v for v in range(1, 255) if v not in _RESERVED_PIXEL_VALUES]
    if n > len(values):
        raise ValueError(
            f"CPM engine can seed at most {len(values)} cells per call (8-bit pixel labels); "
            f"got {n} nodes in this graph. Reduce max_graphs/crop the frame, or split the rollout."
        )
    return values[:n]


# --------------------------------------------------------------------------------------
# PGM/CFG generation and CSV parsing for cpm-hela-model's file-based CLI protocol.
# --------------------------------------------------------------------------------------


def _write_pgm(path: Path, width: int, height: int, pixels: np.ndarray) -> None:
    """Write a binary (P5) PGM -- cpm-hela-model's Main.cpp reads raw bytes per pixel
    despite some stale "P2" comments in its own source; verified against the shipped
    default.pgm, whose actual magic bytes are "P5\\n...".
    """

    assert pixels.shape == (height, width)
    header = f"P5\n# generated by cpm_baseline.py\n{width} {height}\n255\n".encode("ascii")
    path.write_bytes(header + pixels.astype(np.uint8).tobytes())


def _write_cfg(path: Path, params: CPMParameters, *, image_name: str, max_mcs: int, pixel_values: Sequence[int]) -> None:
    map_lines = "\n".join(f"MAP_TEMPLATE,{value},2" for value in pixel_values)
    text = f"""#Simulation parameters
SIM_PARAM,MCS_HOUR_EST,{max(int(max_mcs), 1)}
SIM_PARAM,MAX_HOURS,1
SIM_PARAM,PIXEL_SCALE,2
SIM_PARAM,FPS,60
SIM_PARAM,BOLTZ_TEMP,{params.boltz_temp}
SIM_PARAM,LAMBDA,{params.lambda_volume}
SIM_PARAM,AUTO_QUIT,1
SIM_PARAM,IMAGE,{image_name}
#
CELL_TYPE,0
J,0.0:0.0:{params.contact_energy_boundary}
DO_DIVIDE,0
IS_STATIC,1
IGNORE_VOLUME,1
COLOUR,0
END_TYPE
#
CELL_TYPE,1
J,0.0:0.0:{params.contact_energy_medium}
DO_DIVIDE,0
IS_STATIC,0
IGNORE_VOLUME,1
COLOUR,1
END_TYPE
#
CELL_TYPE,2
J,{params.contact_energy_boundary}:{params.contact_energy_medium}:{params.contact_energy_cell}
DO_DIVIDE,0
IS_STATIC,0
IGNORE_VOLUME,0
COLOUR,2
END_TYPE
#
TEMPLATE,0
TYPE,0
SPECIAL,1
END_TEMPLATE
#
TEMPLATE,1
TYPE,1
SPECIAL,2
END_TEMPLATE
#
TEMPLATE,2
TYPE,2
VOLUME,{int(round(params.target_volume))}
END_TEMPLATE
#
MAP_TEMPLATE,255,0
MAP_TEMPLATE,0,1
{map_lines}
#
COLOUR_SCHEME,0
R,255,255
G,255,255
B,255,255
END_COLOUR
#
COLOUR_SCHEME,1
R,0,0
G,0,0
B,0,0
END_COLOUR
#
COLOUR_SCHEME,2
R,170,230
G,180,240
B,40,90
END_COLOUR
"""
    path.write_text(text, encoding="utf-8")


def _seed_pixel_grid(
    *,
    positions: np.ndarray,
    thetas: np.ndarray,
    aspects: np.ndarray,
    areas: np.ndarray,
    pixel_values: Sequence[int],
    padding: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize each real cell as a filled ellipse onto a uint8 grid, one distinct
    pixel value per cell. Returns (grid[height, width], origin[x, y]) where origin is
    the real-coordinate offset subtracted before rasterizing (add it back to convert
    lattice pixel coordinates back to real coordinates).
    """

    positions = np.nan_to_num(positions, nan=0.0)
    thetas = np.nan_to_num(thetas, nan=0.0)
    aspects = np.nan_to_num(aspects, nan=1.0)
    areas = np.nan_to_num(areas, nan=4.0)

    pad = float(padding)
    min_xy = positions.min(axis=0) - pad
    max_xy = positions.max(axis=0) + pad
    width = max(int(math.ceil(max_xy[0] - min_xy[0])), 32)
    height = max(int(math.ceil(max_xy[1] - min_xy[1])), 32)

    grid = np.zeros((height, width), dtype=np.uint8)
    offsets_cache: dict[tuple[float, float, float], np.ndarray] = {}
    # Paint largest cells first: with real (possibly close/overlapping) centroids,
    # painting order otherwise lets a later, smaller cell get fully swallowed by an
    # earlier, larger one's footprint -- see the guaranteed centroid-pixel pass below
    # for the (still possible with any order) case where a cell ends up with zero
    # pixels regardless.
    order = np.argsort(-areas)
    for local_index in order:
        value = pixel_values[local_index]
        area = float(max(areas[local_index], 4.0))
        aspect = float(max(aspects[local_index], 1.0))
        semi_minor = math.sqrt(area / (math.pi * aspect))
        semi_major = semi_minor * aspect
        key = (round(semi_major, 3), round(semi_minor, 3), round(float(thetas[local_index]), 3))
        if key not in offsets_cache:
            offsets_cache[key] = _ellipse_pixel_offsets(semi_major, semi_minor, float(thetas[local_index]))
        offsets = offsets_cache[key]
        cx = positions[local_index, 0] - min_xy[0]
        cy = positions[local_index, 1] - min_xy[1]
        px = np.clip(np.round(cx + offsets[:, 0]).astype(np.int64), 0, width - 1)
        py = np.clip(np.round(cy + offsets[:, 1]).astype(np.int64), 0, height - 1)
        grid[py, px] = value

    # Guarantee every cell keeps at least its own centroid pixel, even if a neighbor's
    # ellipse fully overwrote its footprint above -- otherwise that cell vanishes from
    # the simulation entirely (computeCentroids/computeSecondMoments never see its id),
    # producing NaN deltas that then poison every later rollout step's aggregate metric.
    # Two real cells can round to the *same* integer pixel (near-duplicate centroids,
    # or after several MCS of adhesion pulling them together across rollout steps), so
    # this also has to guard against guaranteed-pixels colliding with EACH OTHER, not
    # just with ellipse-body pixels: track claimed guaranteed-pixels and nudge a
    # colliding cell to the nearest free one instead of silently overwriting.
    claimed: set[tuple[int, int]] = set()
    for local_index, value in enumerate(pixel_values):
        cx = int(round(positions[local_index, 0] - min_xy[0]))
        cy = int(round(positions[local_index, 1] - min_xy[1]))
        cx = min(max(cx, 0), width - 1)
        cy = min(max(cy, 0), height - 1)
        if (cy, cx) in claimed:
            cy, cx = _find_free_pixel(claimed, cy, cx, height=height, width=width)
        grid[cy, cx] = value
        claimed.add((cy, cx))

    return grid, min_xy


def _find_free_pixel(
    claimed: set[tuple[int, int]], cy: int, cx: int, *, height: int, width: int, max_radius: int = 12
) -> tuple[int, int]:
    """Nearest (row, col) to (cy, cx), in expanding-ring order, not already in
    `claimed`. Falls back to (cy, cx) itself if nothing is free within max_radius
    (only possible for implausibly dense seeding)."""

    for radius in range(1, max_radius + 1):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if max(abs(dy), abs(dx)) != radius:
                    continue
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < height and 0 <= nx < width and (ny, nx) not in claimed:
                    return ny, nx
    return cy, cx


def _parse_final_frame(csv_path: Path, num_cells: int) -> pd.DataFrame:
    """Read sim_positions.csv and return only the rows for our seeded cells at the
    final logged MCS (cpm-hela-model always logs every MCS, so we discard the rest --
    we only need the state after the full mcs_per_frame advance).

    IMPORTANT: SUPER_ID in this CSV is NOT the pixel byte value we painted into the PGM
    -- cpm-hela-model's initializeGrid() assigns SuperCell ids sequentially while
    iterating templateColourMap (a std::map, so always in ASCENDING pixel-value order,
    regardless of MAP_TEMPLATE line order in the .cfg), starting from the space/medium
    entry (pixel value 0, which always sorts first). So with our .cfg always declaring
    exactly one medium entry (0) below all N cell values and one boundary entry (255)
    above all of them, medium gets SUPER_ID 0 and the cell that used the k-th smallest
    (0-indexed) of our pixel_values always gets SUPER_ID k+1 -- independent of which
    actual byte value we chose for it. Boundary cells are IS_STATIC and never appear in
    this log at all. _seed_pixel_grid/_write_cfg always assign pixel_values in ascending
    order per node-local-index, so SUPER_ID (1..num_cells) maps directly, in order, back
    onto node-local-index (0..num_cells-1) -- see the reindex below.
    """

    table = pd.read_csv(csv_path)
    seeded_ids = list(range(1, num_cells + 1))
    table = table[table["SUPER_ID"].isin(seeded_ids)]
    if table.empty:
        raise RuntimeError(f"No rows for seeded cells found in {csv_path}.")
    final_mcs = table["MCS"].max()
    return table[table["MCS"] == final_mcs].set_index("SUPER_ID").reindex(seeded_ids)


# --------------------------------------------------------------------------------------
# The CPM rollout model. Stateless across forward() calls, unlike an in-process CPM
# library would need to be: each call shells out to a fresh cpm_hela subprocess, seeded
# entirely from the graph passed in, so there is no cross-call state/reset-detection to
# get wrong (evaluate_model_rollout calls model(rollout_graph) once per horizon step,
# and each of those maps to one independent "seed -> run mcs_per_frame MCS -> read
# final centroids/moments" subprocess invocation).
# --------------------------------------------------------------------------------------


class CPMBaselineModel(nn.Module):
    def __init__(
        self,
        *,
        params: CPMParameters,
        shape_dim: int,
        node_feature_columns: Sequence[str],
        binary_path: str | Path,
        polarization_columns: Sequence[str] = POLARIZATION_COLUMNS,
        subprocess_timeout_seconds: float = 120.0,
        keep_workdirs: bool = False,
    ) -> None:
        super().__init__()
        self.params = params
        self.shape_dim = int(shape_dim)
        self.node_feature_columns = tuple(node_feature_columns)
        self.polarization_columns = tuple(polarization_columns)
        self.binary_path = Path(binary_path)
        self.subprocess_timeout_seconds = float(subprocess_timeout_seconds)
        self.keep_workdirs = bool(keep_workdirs)

        self._x_index = self.node_feature_columns.index(POSITION_X_COLUMN)
        self._y_index = self.node_feature_columns.index(POSITION_Y_COLUMN)
        self._theta_index = (
            self.node_feature_columns.index(POLARIZATION_ANGLE_COLUMN)
            if POLARIZATION_ANGLE_COLUMN in self.node_feature_columns
            else None
        )
        self._aspect_index = (
            self.node_feature_columns.index(POLARIZATION_MAGNITUDE_COLUMN)
            if POLARIZATION_MAGNITUDE_COLUMN in self.node_feature_columns
            else None
        )
        self._area_index = self.node_feature_columns.index(AREA_COLUMN) if AREA_COLUMN in self.node_feature_columns else None

    def _run_one_step(
        self, *, positions: np.ndarray, thetas: np.ndarray, aspects: np.ndarray, areas: np.ndarray
    ) -> pd.DataFrame:
        num_nodes = positions.shape[0]
        pixel_values = _usable_pixel_values(num_nodes)
        grid, origin = _seed_pixel_grid(
            positions=positions,
            thetas=thetas,
            aspects=aspects,
            areas=areas,
            pixel_values=pixel_values,
            padding=self.params.lattice_padding_px,
        )

        workdir = Path(tempfile.mkdtemp(prefix="cpm_baseline_"))
        try:
            _write_pgm(workdir / "frame.pgm", grid.shape[1], grid.shape[0], grid)
            _write_cfg(
                workdir / "frame.cfg",
                self.params,
                image_name="frame",
                max_mcs=self.params.mcs_per_frame,
                pixel_values=pixel_values,
            )
            if not self.binary_path.exists():
                raise FileNotFoundError(
                    f"CPM engine binary not found at {self.binary_path}. Build cpm-hela-model "
                    "(cmake -DSSH_HEADLESS=ON && make) and pass CPMBaselineConfig(binary_path=...)."
                )
            subprocess.run(
                [str(self.binary_path.resolve()), "-h", "-f", "frame"],
                cwd=workdir,
                check=True,
                timeout=self.subprocess_timeout_seconds,
                capture_output=True,
            )
            final_frame = _parse_final_frame(workdir / "sim_positions.csv", num_nodes)
        finally:
            if not self.keep_workdirs:
                shutil.rmtree(workdir, ignore_errors=True)

        # final_frame is already ordered SUPER_ID=1..num_nodes, which (see
        # _parse_final_frame's docstring) corresponds 1:1, in order, to node-local-index
        # 0..num_nodes-1 -- no further reindexing by pixel value needed/correct here.
        final_frame = final_frame.copy()
        final_frame["X"] = final_frame["X"] + origin[0]
        final_frame["Y"] = final_frame["Y"] + origin[1]
        return final_frame

    def forward(self, graph: Any) -> CellGNNOutput:
        num_nodes = int(graph.num_nodes)
        device = graph.x.device
        physical_x = graph.x.detach().cpu().numpy()
        positions = physical_x[:, [self._x_index, self._y_index]]
        thetas = physical_x[:, self._theta_index] if self._theta_index is not None else np.zeros(num_nodes)
        aspects = physical_x[:, self._aspect_index] if self._aspect_index is not None else np.ones(num_nodes)
        areas = physical_x[:, self._area_index] if self._area_index is not None else np.full(num_nodes, self.params.target_volume)

        final_frame = self._run_one_step(positions=positions, thetas=thetas, aspects=aspects, areas=areas)

        new_positions = final_frame[["X", "Y"]].to_numpy(dtype=np.float64)
        delta_pos = torch.as_tensor(new_positions - positions, dtype=torch.float32, device=device)

        delta_polarization: Optional[torch.Tensor] = None
        if self._theta_index is not None and self._aspect_index is not None:
            new_theta = np.zeros(num_nodes, dtype=np.float64)
            new_aspect = np.ones(num_nodes, dtype=np.float64)
            for local_index, (_, row) in enumerate(final_frame.iterrows()):
                new_theta[local_index], new_aspect[local_index] = _ellipse_from_second_moments(
                    row["MU20"], row["MU02"], row["MU11"]
                )
            new_aspect = _clamp_aspect_ratio(new_aspect)
            dtheta = wrap_nematic_delta(new_theta - thetas, POLARIZATION_ANGLE_PERIOD)
            daspect = new_aspect - aspects
            delta_polarization = torch.as_tensor(
                np.stack([dtheta, daspect], axis=1), dtype=torch.float32, device=device
            )

        return CellGNNOutput(
            delta_pos=delta_pos,
            delta_shape=torch.zeros((num_nodes, self.shape_dim), dtype=torch.float32, device=device),
            division_logits=torch.zeros(num_nodes, dtype=torch.float32, device=device),
            death_logits=torch.zeros(num_nodes, dtype=torch.float32, device=device),
            node_embeddings=torch.zeros((num_nodes, 1), dtype=torch.float32, device=device),
            delta_polarization=delta_polarization,
        )


# --------------------------------------------------------------------------------------
# Orchestration -- mirrors kinetic_baseline.run_kinetic_baseline's shape.
# --------------------------------------------------------------------------------------


def run_cpm_baseline(
    config: CPMBaselineConfig,
    params: Optional[CPMParameters] = None,
    *,
    dataset_config: Optional[FrameGraphDatasetConfig] = None,
    split_config: Optional[SplitConfig] = None,
) -> dict[str, Any]:
    """Calibrate target_volume on train, evaluate the CPM baseline on test at the given
    horizons. Rows share evaluate_rollout.py's schema (model="cpm"), written
    independently of any GNN/kinetic-baseline cache -- combine with
    merge_rollout_reports() afterwards, same as kinetic_baseline.py does.
    """

    dataset_config = dataset_config or FrameGraphDatasetConfig()
    spots = load_processed_spots(dataset_config.source_path)
    cache = build_graph_cache(
        spots=spots,
        dataset_config=dataset_config,
        split_config=split_config or SplitConfig(),
        max_graphs=config.max_graphs,
    )
    config.out_dir.mkdir(parents=True, exist_ok=True)
    save_graph_cache(cache, config.out_dir / "cpm_baseline_cache.pt")
    graphs = cache["graphs"]
    splits = cache["splits"]
    train_graphs = [graphs[index] for index in splits.get("train", [])]
    test_graphs = [graphs[index] for index in splits.get("test", [])]
    if not train_graphs:
        raise ValueError("Train split is empty; cannot calibrate the CPM baseline.")
    if not test_graphs:
        raise ValueError("Test split is empty; cannot evaluate the CPM baseline.")

    node_feature_columns = tuple(str(column) for column in graphs[0].node_feature_columns)
    calibrated, volume_stats = calibrated_parameters(train_graphs, node_feature_columns=node_feature_columns, base=params)

    try:
        shape_dim = infer_shape_dim(graphs)
    except ValueError:
        shape_dim = 0

    device = resolve_device(config.device)
    model = CPMBaselineModel(
        params=calibrated,
        shape_dim=shape_dim,
        node_feature_columns=node_feature_columns,
        binary_path=config.binary_path,
        subprocess_timeout_seconds=config.subprocess_timeout_seconds,
        keep_workdirs=config.keep_workdirs,
    ).to(device)

    test_sequences = group_graphs_by_sequence(test_graphs)
    rollout_config = rollout_config_from_cache(cache)
    metrics = evaluate_model_rollout(
        model,
        test_sequences,
        horizons=config.horizons,
        rollout_config=rollout_config,
        device=device,
        node_feature_normalization=None,
        match_tolerance=config.match_tolerance,
    )
    rows = [{"model": "cpm", "checkpoint": None, **row} for row in metrics]

    results = {
        "config": jsonable(asdict(config)),
        "params": jsonable(asdict(calibrated)),
        "volume_calibration": jsonable(volume_stats),
        "cache_summary": cache.get("summary", {}),
        "rows": rows,
    }
    write_json(config.out_dir / "cpm_baseline_summary.json", results)
    write_reports(rows, config.out_dir / "cpm_baseline_rollout_eval.json")
    write_report(config.out_dir / "cpm_baseline_report.md", results)
    write_calibration_handoff(config.out_dir / "cpm_calibration_handoff.json", calibrated, volume_stats)
    return results


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_calibration_handoff(path: Path, params: CPMParameters, volume_stats: dict[str, float]) -> None:
    """The artifact meant for the labmate: the data-derived target_volume plus whatever
    engine parameters were used, kept separate from the evaluation results file."""

    write_json(
        path,
        {
            "target_volume": params.target_volume,
            "volume_source_stats": volume_stats,
            "mcs_per_frame": params.mcs_per_frame,
            "boltz_temp": params.boltz_temp,
            "lambda_volume": params.lambda_volume,
            "contact_energy_medium": params.contact_energy_medium,
            "contact_energy_cell": params.contact_energy_cell,
            "note": (
                "boltz_temp/lambda_volume/contact_energy_* default to your own cpm-hela-model "
                "default.cfg calibration (not refit here) -- only target_volume is re-derived "
                "from this dataset's own AREA statistics."
            ),
        },
    )


def write_report(path: Path, results: dict[str, Any]) -> None:
    params = results["params"]
    lines = [
        "# CPM baseline (cpm-hela-model) vs GNN",
        "",
        "## Parameters used",
        "",
        f"- target_volume (px^2, from train AREA): `{params['target_volume']:.2f}`",
        f"- mcs_per_frame: `{params['mcs_per_frame']}`",
        f"- boltz_temp / lambda_volume: `{params['boltz_temp']}` / `{params['lambda_volume']}`",
        f"- contact_energy_medium / contact_energy_cell: `{params['contact_energy_medium']}` / "
        f"`{params['contact_energy_cell']}` (labmate's default.cfg calibration, not refit here)",
        "",
        "## horizon -> RMSE",
        "",
        "| model | horizon | matched_nodes | position_rmse | polarization_theta_rmse | polarization_aspect_rmse |",
        "|---|---|---|---|---|---|",
    ]
    for row in sorted(results["rows"], key=lambda r: int(r["horizon"])):
        theta = row.get("polarization_theta_rmse")
        aspect = row.get("polarization_aspect_rmse")
        theta_text = f"{theta:.4f}" if theta is not None and not math.isnan(theta) else "n/a"
        aspect_text = f"{aspect:.4f}" if aspect is not None and not math.isnan(aspect) else "n/a"
        lines.append(
            f"| {row['model']} | {row['horizon']} | {row['matched_nodes']} | {row['position_rmse']:.4f} | "
            f"{theta_text} | {aspect_text} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a multi-cell Cellular Potts Model rollout baseline (cpm-hela-model engine), "
            "in the same JSON schema as evaluate_rollout.py / kinetic_baseline.py."
        )
    )
    parser.add_argument("--binary", type=Path, required=True, help="Path to the compiled cpm_hela executable.")
    parser.add_argument("--out-dir", type=Path, default=CPMBaselineConfig().out_dir)
    parser.add_argument("--horizons", type=int, nargs="+", default=list(CPMBaselineConfig().horizons))
    parser.add_argument("--max-graphs", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--device", default=CPMBaselineConfig().device)
    parser.add_argument("--match-tolerance", type=float, default=CPMBaselineConfig().match_tolerance)
    parser.add_argument("--subprocess-timeout", type=float, default=CPMBaselineConfig().subprocess_timeout_seconds)
    parser.add_argument("--keep-workdirs", action="store_true")
    parser.add_argument("--seed", type=int, default=SplitConfig().seed)
    parser.add_argument("--split-mode", default=SplitConfig().mode)
    parser.add_argument("--source", type=Path, default=DEFAULT_PROCESSED_SPOTS)
    parser.add_argument("--params-json", type=Path, default=None, help="Load starting CPMParameters from this file.")
    parser.add_argument(
        "--calibrate-only",
        action="store_true",
        help="Only build the cache, calibrate target_volume, dump params.json, and exit.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    config = CPMBaselineConfig(
        binary_path=args.binary,
        out_dir=args.out_dir,
        horizons=tuple(args.horizons),
        max_graphs=args.max_graphs,
        device=args.device,
        match_tolerance=args.match_tolerance,
        subprocess_timeout_seconds=args.subprocess_timeout,
        keep_workdirs=args.keep_workdirs,
    )
    params = CPMParameters.from_json(args.params_json) if args.params_json else None
    dataset_config = FrameGraphDatasetConfig(source_path=args.source)
    split_config = SplitConfig(mode=args.split_mode, seed=args.seed)

    if args.calibrate_only:
        spots = load_processed_spots(dataset_config.source_path)
        cache = build_graph_cache(
            spots=spots, dataset_config=dataset_config, split_config=split_config, max_graphs=config.max_graphs
        )
        graphs = cache["graphs"]
        train_graphs = [graphs[index] for index in cache["splits"].get("train", [])]
        node_feature_columns = tuple(str(column) for column in graphs[0].node_feature_columns)
        calibrated, stats = calibrated_parameters(train_graphs, node_feature_columns=node_feature_columns, base=params)
        config.out_dir.mkdir(parents=True, exist_ok=True)
        calibrated.to_json(config.out_dir / "params.json")
        print(json.dumps({"params": asdict(calibrated), "volume_calibration": stats}, indent=2))
        return 0

    results = run_cpm_baseline(config, params, dataset_config=dataset_config, split_config=split_config)
    for row in results["rows"]:
        print(
            f"{row['model']} h={row['horizon']} matched={row['matched_nodes']} "
            f"position_rmse={row['position_rmse']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
