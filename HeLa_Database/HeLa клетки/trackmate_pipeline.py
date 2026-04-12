# =========================================================
# TrackMate → ML Pipeline (HeLa / Cell Tracking)
# =========================================================

from __future__ import annotations

import math
import argparse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.spatial import cKDTree


# =========================================================
# CONFIG
# =========================================================

@dataclass
class Config:
    neighbor_radius: float = 40.0
    k_nearest: int = 5
    decay: float = 30.0
    n_sectors: int = 8
    shape_samples: int = 64
    shape_align_to_ellipse: bool = True

    min_area: float = 50.0
    max_area: float = 1000.0
    min_track_length: int = 3


# =========================================================
# XML PARSING
# =========================================================

def _safe(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def _safe_int(x, default=-1):
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def _ids_to_text(ids):
    return "|".join(str(int(item)) for item in ids)


def parse_contour_points(text):
    if not text:
        return np.empty((0, 2), dtype=np.float32)

    try:
        values = [float(item) for item in text.split()]
    except ValueError:
        return np.empty((0, 2), dtype=np.float32)

    if len(values) < 6 or len(values) % 2:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(values, dtype=np.float32).reshape(-1, 2)


def contour_points_to_text(points):
    if points.size == 0:
        return ""
    return "|".join(f"{x:.6g}:{y:.6g}" for x, y in points)


def contour_text_to_points(text):
    if not isinstance(text, str) or not text:
        return np.empty((0, 2), dtype=np.float32)

    points = []
    for pair in text.split("|"):
        if not pair:
            continue
        try:
            x, y = pair.split(":", 1)
            points.append((float(x), float(y)))
        except ValueError:
            return np.empty((0, 2), dtype=np.float32)
    if len(points) < 3:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def parse_spots(root):
    rows = []

    for frame in root.iter("SpotsInFrame"):
        f = int(frame.attrib["frame"])

        for spot in frame.iter("Spot"):
            contour_points = parse_contour_points(spot.text)
            row = {
                "spot_id": int(spot.attrib["ID"]),
                "frame": f,
                "t": _safe(spot.attrib.get("POSITION_T")),
                "x": _safe(spot.attrib.get("POSITION_X")),
                "y": _safe(spot.attrib.get("POSITION_Y")),
                "AREA": _safe(spot.attrib.get("AREA")),
                "PERIMETER": _safe(spot.attrib.get("PERIMETER")),
                "CIRCULARITY": _safe(spot.attrib.get("CIRCULARITY")),
                "SOLIDITY": _safe(spot.attrib.get("SOLIDITY")),
                "RADIUS": _safe(spot.attrib.get("RADIUS")),
                "ELLIPSE_MAJOR": _safe(spot.attrib.get("ELLIPSE_MAJOR")),
                "ELLIPSE_MINOR": _safe(spot.attrib.get("ELLIPSE_MINOR")),
                "ELLIPSE_ASPECTRATIO": _safe(spot.attrib.get("ELLIPSE_ASPECTRATIO")),
                "ELLIPSE_THETA": _safe(spot.attrib.get("ELLIPSE_THETA")),
                "MEAN_INTENSITY_CH1": _safe(spot.attrib.get("MEAN_INTENSITY_CH1")),
                "contour_xy_local": contour_points_to_text(contour_points),
                "contour_point_count": int(len(contour_points)),
            }
            rows.append(row)

    return pd.DataFrame(rows)


def parse_edges(root):
    edges = []

    for track in root.iter("Track"):
        track_id = _safe_int(track.attrib.get("TRACK_ID"))
        track_index = _safe_int(track.attrib.get("TRACK_INDEX"))

        for edge in track.findall("Edge"):
            edges.append({
                "source": int(edge.attrib["SPOT_SOURCE_ID"]),
                "target": int(edge.attrib["SPOT_TARGET_ID"]),
                "track_id": track_id,
                "track_index": track_index,
                "edge_speed": _safe(edge.attrib.get("SPEED")),
                "edge_displacement": _safe(edge.attrib.get("DISPLACEMENT")),
                "edge_time": _safe(edge.attrib.get("EDGE_TIME")),
            })

    if not edges:
        for edge in root.iter("Edge"):
            edges.append({
                "source": int(edge.attrib["SPOT_SOURCE_ID"]),
                "target": int(edge.attrib["SPOT_TARGET_ID"]),
                "track_id": _safe_int(edge.attrib.get("TRACK_ID")),
                "track_index": _safe_int(edge.attrib.get("TRACK_INDEX")),
                "edge_speed": _safe(edge.attrib.get("SPEED")),
                "edge_displacement": _safe(edge.attrib.get("DISPLACEMENT")),
                "edge_time": _safe(edge.attrib.get("EDGE_TIME")),
            })

    return pd.DataFrame(edges)


# =========================================================
# GRAPH / TRACK FEATURES
# =========================================================

def build_tracks(spots, edges):
    spots = spots.copy()

    if edges.empty:
        spots["next_id"] = np.nan
        spots["prev_id"] = np.nan
        spots["next_ids"] = ""
        spots["prev_ids"] = ""
        spots["n_next"] = 0
        spots["n_prev"] = 0
        spots["has_next"] = False
        spots["has_prev"] = False
        spots["is_split"] = False
        spots["is_merge"] = False
        spots["dx"] = np.nan
        spots["dy"] = np.nan
        spots["speed"] = np.nan
        return spots

    outgoing = edges.groupby("source")["target"].agg(lambda values: tuple(int(v) for v in values))
    incoming = edges.groupby("target")["source"].agg(lambda values: tuple(int(v) for v in values))

    next_map = outgoing.to_dict()
    prev_map = incoming.to_dict()

    def _spot_ids_to_text(ids):
        if not isinstance(ids, tuple):
            return ""
        return _ids_to_text(ids)

    def _single_id(ids):
        if isinstance(ids, tuple) and len(ids) == 1:
            return float(ids[0])
        return np.nan

    spots["next_ids"] = spots["spot_id"].map(next_map).map(_spot_ids_to_text)
    spots["prev_ids"] = spots["spot_id"].map(prev_map).map(_spot_ids_to_text)
    spots["n_next"] = spots["spot_id"].map(next_map).map(lambda ids: len(ids) if isinstance(ids, tuple) else 0)
    spots["n_prev"] = spots["spot_id"].map(prev_map).map(lambda ids: len(ids) if isinstance(ids, tuple) else 0)
    spots["next_id"] = spots["spot_id"].map(next_map).map(_single_id)
    spots["prev_id"] = spots["spot_id"].map(prev_map).map(_single_id)

    spots["has_next"] = spots["n_next"] > 0
    spots["has_prev"] = spots["n_prev"] > 0
    spots["is_split"] = spots["n_next"] > 1
    spots["is_merge"] = spots["n_prev"] > 1

    # Single-target movement can stay on the spot table. Split movements are
    # represented exactly in the edge-level supervised table.
    lookup = spots.set_index("spot_id")[["x", "y", "t"]].to_dict("index")

    dx, dy, speed = [], [], []

    for row in spots.itertuples():
        if pd.isna(row.next_id):
            dx.append(np.nan)
            dy.append(np.nan)
            speed.append(np.nan)
            continue

        nxt = lookup.get(int(row.next_id))
        if nxt is None:
            dx.append(np.nan)
            dy.append(np.nan)
            speed.append(np.nan)
            continue

        dx_i = nxt["x"] - row.x
        dy_i = nxt["y"] - row.y
        dt = nxt["t"] - row.t

        dx.append(dx_i)
        dy.append(dy_i)
        speed.append(np.sqrt(dx_i**2 + dy_i**2) / dt if dt else np.nan)

    spots["dx"] = dx
    spots["dy"] = dy
    spots["speed"] = speed

    return spots


# =========================================================
# CELL LIFECYCLES
# =========================================================

def build_cell_lifecycles(spots, edges):
    spots = spots.copy()
    lifecycle_columns = {
        "cell_id": pd.Series(dtype="Int64"),
        "generation": pd.Series(dtype="Int64"),
        "cell_start": pd.Series(dtype=bool),
        "cell_end": pd.Series(dtype=bool),
        "cell_end_reason": pd.Series(dtype=object),
        "parents_id": pd.Series(dtype=object),
        "childs_id": pd.Series(dtype=object),
    }

    if spots.empty:
        for column, values in lifecycle_columns.items():
            spots[column] = values
        return spots, pd.DataFrame(columns=[
            "cell_id",
            "generation",
            "parents_id",
            "childs_id",
            "start_spot_id",
            "end_spot_id",
            "start_frame",
            "end_frame",
            "start_t",
            "end_t",
            "lifetime_frames",
            "n_spots",
            "end_reason",
        ])

    spot_ids = set(spots["spot_id"].astype(int))
    spot_lookup = {
        int(row.spot_id): row._asdict()
        for row in spots.itertuples(index=False)
    }
    sorted_spot_ids = sorted(
        spot_ids,
        key=lambda spot_id: (
            int(spot_lookup[spot_id]["frame"]),
            float(spot_lookup[spot_id]["t"]) if not pd.isna(spot_lookup[spot_id]["t"]) else float("inf"),
            spot_id,
        ),
    )

    outgoing = {
        source: tuple(group.sort_values(["target"])["target"].astype(int))
        for source, group in edges.groupby("source")
    } if not edges.empty else {}
    incoming = {
        target: tuple(group.sort_values(["source"])["source"].astype(int))
        for target, group in edges.groupby("target")
    } if not edges.empty else {}

    spot_to_cell: dict[int, int] = {}
    cell_rows: list[dict[str, object]] = []
    queue: list[tuple[int, tuple[int, ...]]] = []

    def enqueue(start_spot_id, parent_cell_ids):
        if start_spot_id in spot_ids and start_spot_id not in spot_to_cell:
            queue.append((int(start_spot_id), tuple(sorted(set(int(parent) for parent in parent_cell_ids)))))

    for spot_id in sorted_spot_ids:
        if not incoming.get(spot_id):
            enqueue(spot_id, ())

    while queue or len(spot_to_cell) < len(spot_ids):
        if not queue:
            for spot_id in sorted_spot_ids:
                if spot_id not in spot_to_cell:
                    parent_cell_ids = [
                        spot_to_cell[parent_spot]
                        for parent_spot in incoming.get(spot_id, ())
                        if parent_spot in spot_to_cell
                    ]
                    enqueue(spot_id, parent_cell_ids)
                    break

        start_spot_id, parent_cell_ids = queue.pop(0)
        if start_spot_id in spot_to_cell:
            continue

        cell_id = len(cell_rows)
        if parent_cell_ids:
            parent_generations = [int(cell_rows[parent]["generation"]) for parent in parent_cell_ids]
            generation = max(parent_generations) + 1
        else:
            generation = 0

        chain: list[int] = []
        child_start_ids: list[int] = []
        end_reason = "death"
        current = start_spot_id

        while True:
            if current in spot_to_cell:
                end_reason = "merge"
                break

            spot_to_cell[current] = cell_id
            chain.append(current)

            targets = [target for target in outgoing.get(current, ()) if target in spot_ids]
            if not targets:
                end_reason = "death"
                break

            if len(targets) == 1 and len(incoming.get(targets[0], ())) == 1:
                current = targets[0]
                continue

            child_start_ids = targets
            end_reason = "division" if len(targets) > 1 else "merge"
            break

        if not chain:
            continue

        first = spot_lookup[chain[0]]
        last = spot_lookup[chain[-1]]
        cell_rows.append({
            "cell_id": cell_id,
            "generation": generation,
            "parents_id": _ids_to_text(parent_cell_ids),
            "childs_id": "",
            "start_spot_id": chain[0],
            "end_spot_id": chain[-1],
            "start_frame": int(first["frame"]),
            "end_frame": int(last["frame"]),
            "start_t": first["t"],
            "end_t": last["t"],
            "lifetime_frames": int(last["frame"]) - int(first["frame"]) + 1,
            "n_spots": len(chain),
            "end_reason": end_reason,
        })

        for child_start_id in child_start_ids:
            parent_ids = [
                spot_to_cell[parent_spot]
                for parent_spot in incoming.get(child_start_id, ())
                if parent_spot in spot_to_cell
            ]
            enqueue(child_start_id, parent_ids)

    children_by_parent: dict[int, set[int]] = {row["cell_id"]: set() for row in cell_rows}
    for row in cell_rows:
        if not row["parents_id"]:
            continue
        for parent in str(row["parents_id"]).split("|"):
            if parent:
                children_by_parent.setdefault(int(parent), set()).add(int(row["cell_id"]))

    for row in cell_rows:
        row["childs_id"] = _ids_to_text(sorted(children_by_parent.get(int(row["cell_id"]), ())))

    cells = pd.DataFrame(cell_rows)
    cell_lookup = cells.set_index("cell_id").to_dict("index") if not cells.empty else {}

    spots["cell_id"] = spots["spot_id"].map(lambda spot_id: spot_to_cell.get(int(spot_id))).astype("Int64")
    spots["generation"] = spots["cell_id"].map(lambda cell_id: cell_lookup[int(cell_id)]["generation"]).astype("Int64")
    spots["cell_start"] = spots["spot_id"].isin(set(cells["start_spot_id"].astype(int)))
    spots["cell_end"] = spots["spot_id"].isin(set(cells["end_spot_id"].astype(int)))
    spots["cell_end_reason"] = spots["cell_id"].map(lambda cell_id: cell_lookup[int(cell_id)]["end_reason"])
    spots["parents_id"] = spots["cell_id"].map(lambda cell_id: cell_lookup[int(cell_id)]["parents_id"])
    spots["childs_id"] = spots["cell_id"].map(lambda cell_id: cell_lookup[int(cell_id)]["childs_id"])

    return spots, cells


# =========================================================
# SHAPE FEATURES
# =========================================================

def polygon_area(points):
    if len(points) < 3:
        return np.nan
    x = points[:, 0]
    y = points[:, 1]
    return float(abs(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def polygon_perimeter(points):
    if len(points) < 2:
        return np.nan
    closed = np.vstack([points, points[0]])
    return float(np.linalg.norm(np.diff(closed, axis=0), axis=1).sum())


def polygon_centroid(points):
    if len(points) < 3:
        return (np.nan, np.nan)

    x = points[:, 0]
    y = points[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    signed_area = 0.5 * np.sum(cross)
    if abs(signed_area) < 1e-9:
        return (float(np.mean(x)), float(np.mean(y)))
    cx = np.sum((x + np.roll(x, -1)) * cross) / (6.0 * signed_area)
    cy = np.sum((y + np.roll(y, -1)) * cross) / (6.0 * signed_area)
    return (float(cx), float(cy))


def _fill_missing_radii(radii):
    result = radii.copy()
    missing = np.isnan(result)
    if not missing.any():
        return result
    valid_idx = np.flatnonzero(~missing)
    if len(valid_idx) == 0:
        return result

    all_idx = np.arange(len(result))
    extended_idx = np.r_[valid_idx - len(result), valid_idx, valid_idx + len(result)]
    extended_values = np.r_[result[valid_idx], result[valid_idx], result[valid_idx]]
    result[missing] = np.interp(all_idx[missing], extended_idx, extended_values)
    return result


def sample_contour_radii(points, samples, angle_offset=0.0):
    if len(points) < 3 or samples <= 0:
        return np.full(max(samples, 0), np.nan, dtype=np.float32), 0

    points = np.asarray(points, dtype=np.float64)
    start = points
    end = np.roll(points, -1, axis=0)
    segment = end - start

    angles = angle_offset + np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
    directions = np.column_stack([np.cos(angles), np.sin(angles)])

    denom = directions[:, 0, None] * segment[None, :, 1] - directions[:, 1, None] * segment[None, :, 0]
    valid = np.abs(denom) > 1e-9
    cross_start_segment = start[:, 0] * segment[:, 1] - start[:, 1] * segment[:, 0]
    cross_start_direction = (
        start[None, :, 0] * directions[:, 1, None]
        - start[None, :, 1] * directions[:, 0, None]
    )

    t = np.divide(
        cross_start_segment[None, :],
        denom,
        out=np.full_like(denom, np.nan, dtype=np.float64),
        where=valid,
    )
    u = np.divide(
        cross_start_direction,
        denom,
        out=np.full_like(denom, np.nan, dtype=np.float64),
        where=valid,
    )
    hits = (t >= -1e-7) & (u >= -1e-7) & (u <= 1.0 + 1e-7)
    t = np.where(hits, np.maximum(t, 0.0), np.nan)

    radii = np.full(samples, np.nan, dtype=np.float64)
    valid_rays = 0
    for index in range(samples):
        ray_hits = t[index]
        ray_hits = ray_hits[~np.isnan(ray_hits)]
        if len(ray_hits):
            radii[index] = float(np.min(ray_hits))
            valid_rays += 1

    return _fill_missing_radii(radii).astype(np.float32), valid_rays


def add_shape_features(spots, cfg: Config):
    if cfg.shape_samples < 1:
        raise ValueError("shape_samples must be positive.")

    spots = spots.copy()
    radius_columns = [f"shape_r_{index:03d}" for index in range(cfg.shape_samples)]
    norm_columns = [f"shape_r_norm_{index:03d}" for index in range(cfg.shape_samples)]
    meta_columns = [
        "shape_samples",
        "shape_aligned_to_ellipse",
        "shape_angle_offset",
        "shape_valid_rays",
        "shape_missing_fraction",
        "shape_contour_area",
        "shape_contour_perimeter",
        "shape_area_ratio",
        "shape_reconstruction_area",
        "shape_reconstruction_area_ratio",
        "shape_contour_centroid_dx",
        "shape_contour_centroid_dy",
        "shape_mean_radius",
        "shape_radius_std",
        "shape_radius_cv",
    ]
    if spots.empty:
        for column in meta_columns + radius_columns + norm_columns:
            spots[column] = []
        return spots

    rows = []

    for row in spots.itertuples(index=False):
        points = contour_text_to_points(getattr(row, "contour_xy_local", ""))
        orientation = getattr(row, "ELLIPSE_THETA", np.nan)
        if pd.isna(orientation) or not cfg.shape_align_to_ellipse:
            orientation = 0.0
        area = float(getattr(row, "AREA", np.nan))

        radii, valid_rays = sample_contour_radii(points, cfg.shape_samples, float(orientation))
        norm_scale = math.sqrt(area / math.pi) if area and area > 0 else np.nan
        if np.isfinite(norm_scale) and norm_scale > 0:
            norm_radii = radii / norm_scale
        else:
            norm_radii = np.full_like(radii, np.nan)

        contour_area = polygon_area(points)
        contour_perimeter = polygon_perimeter(points)
        centroid_x, centroid_y = polygon_centroid(points)
        reconstructed_area = (
            float(0.5 * (2.0 * math.pi / cfg.shape_samples) * np.nansum(radii.astype(float) ** 2))
            if valid_rays
            else np.nan
        )
        mean_radius = float(np.nanmean(radii)) if np.isfinite(radii).any() else np.nan
        std_radius = float(np.nanstd(radii)) if np.isfinite(radii).any() else np.nan
        radius_cv = std_radius / mean_radius if mean_radius and mean_radius > 0 else np.nan

        payload = {
            "shape_samples": cfg.shape_samples,
            "shape_aligned_to_ellipse": bool(cfg.shape_align_to_ellipse),
            "shape_angle_offset": float(orientation),
            "shape_valid_rays": int(valid_rays),
            "shape_missing_fraction": float(1.0 - valid_rays / cfg.shape_samples),
            "shape_contour_area": contour_area,
            "shape_contour_perimeter": contour_perimeter,
            "shape_area_ratio": contour_area / area if area and area > 0 and np.isfinite(contour_area) else np.nan,
            "shape_reconstruction_area": reconstructed_area,
            "shape_reconstruction_area_ratio": (
                reconstructed_area / area if area and area > 0 and np.isfinite(reconstructed_area) else np.nan
            ),
            "shape_contour_centroid_dx": centroid_x,
            "shape_contour_centroid_dy": centroid_y,
            "shape_mean_radius": mean_radius,
            "shape_radius_std": std_radius,
            "shape_radius_cv": radius_cv,
        }
        payload.update({column: float(value) for column, value in zip(radius_columns, radii)})
        payload.update({column: float(value) for column, value in zip(norm_columns, norm_radii)})
        rows.append(payload)

    shape_df = pd.DataFrame(rows, index=spots.index)
    return pd.concat([spots, shape_df], axis=1)


# =========================================================
# NEIGHBOR FEATURES
# =========================================================

def add_neighbors(df, cfg: Config):
    if df.empty:
        df = df.copy()
        df["n_neighbors"] = []
        df["density"] = []
        df["Fx"] = []
        df["Fy"] = []
        return df

    out = []

    for frame, g in df.groupby("frame"):
        pts = g[["x", "y"]].values
        tree = cKDTree(pts)

        neighbors = tree.query_ball_point(pts, cfg.neighbor_radius)

        n_neighbors = []
        density = []
        Fx, Fy = [], []

        for i, nbrs in enumerate(neighbors):
            nbrs = [j for j in nbrs if j != i]
            n_neighbors.append(len(nbrs))
            density.append(len(nbrs) / (math.pi * cfg.neighbor_radius**2))

            fx, fy = 0, 0
            for j in nbrs:
                vec = pts[j] - pts[i]
                dist = np.linalg.norm(vec)
                if dist == 0:
                    continue
                w = math.exp(-dist / cfg.decay)
                fx += w * vec[0] / dist
                fy += w * vec[1] / dist

            Fx.append(fx)
            Fy.append(fy)

        g = g.copy()
        g["n_neighbors"] = n_neighbors
        g["density"] = density
        g["Fx"] = Fx
        g["Fy"] = Fy

        out.append(g)

    return pd.concat(out)


# =========================================================
# SUPERVISED DATASET
# =========================================================

def build_supervised(df):
    return build_supervised_from_edges(df, None)


def build_supervised_from_edges(df, edges):
    df = df.copy()

    if edges is None:
        next_df = df.add_suffix("_next")
        next_df = next_df.rename(columns={"spot_id_next": "next_id"})

        merged = df.merge(next_df, on="next_id", how="left")

        has_target = merged["next_id"].notna() & merged["x_next"].notna() & merged["y_next"].notna()
        merged["target_dx"] = merged["x_next"] - merged["x"]
        merged["target_dy"] = merged["y_next"] - merged["y"]
        merged["target_move"] = has_target.astype(int)
        merged.loc[~has_target, ["target_dx", "target_dy"]] = np.nan

        return merged

    edges = filter_edges_to_spots(edges, df)
    if edges.empty:
        supervised = df.copy()
        supervised["target_id"] = np.nan
        supervised["track_id"] = np.nan
        supervised["track_index"] = np.nan
        supervised["edge_speed"] = np.nan
        supervised["edge_displacement"] = np.nan
        supervised["edge_time"] = np.nan
        supervised["target_dx"] = np.nan
        supervised["target_dy"] = np.nan
        supervised["target_move"] = 0
        return supervised

    source_edges = edges.rename(columns={"source": "spot_id", "target": "target_id"})
    next_df = df.add_suffix("_next")
    next_df = next_df.rename(columns={"spot_id_next": "target_id"})

    positives = source_edges.merge(df, on="spot_id", how="inner")
    positives = positives.merge(next_df, on="target_id", how="inner")

    positives["target_dx"] = positives["x_next"] - positives["x"]
    positives["target_dy"] = positives["y_next"] - positives["y"]
    positives["target_move"] = 1

    no_next = df[~df["spot_id"].isin(edges["source"])].copy()
    missing_columns = [
        "target_id",
        "track_id",
        "track_index",
        "edge_speed",
        "edge_displacement",
        "edge_time",
        *[column for column in next_df.columns if column != "target_id"],
        "target_dx",
        "target_dy",
    ]
    missing_values = {column: np.nan for column in missing_columns if column not in no_next.columns}
    missing_values["target_move"] = 0
    no_next = pd.concat([no_next, pd.DataFrame(missing_values, index=no_next.index)], axis=1)

    no_next = no_next.dropna(axis=1, how="all")
    return pd.concat([positives, no_next], ignore_index=True, sort=False)


# =========================================================
# FILTER
# =========================================================

def filter_data(df, cfg: Config):
    df = df.copy()

    if "AREA" in df.columns:
        df = df[df["AREA"] > cfg.min_area]
        df = df[df["AREA"] < cfg.max_area]

    return df


def filter_edges_to_spots(edges, spots):
    if edges.empty:
        return edges.copy()

    valid_ids = set(spots["spot_id"].astype(int))
    return edges[
        edges["source"].isin(valid_ids) & edges["target"].isin(valid_ids)
    ].copy().reset_index(drop=True)


# =========================================================
# MAIN PIPELINE
# =========================================================

def run_pipeline(xml_path: str, out_dir: str, cfg: Config):
    print("Loading XML...")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    print("Parsing spots...")
    spots = parse_spots(root)

    print("Parsing edges...")
    edges = parse_edges(root)

    print("Filtering...")
    spots = filter_data(spots, cfg)
    edges = filter_edges_to_spots(edges, spots)

    print("Computing shape features...")
    spots = add_shape_features(spots, cfg)

    print("Building tracks...")
    spots = build_tracks(spots, edges)

    print("Building cell lifecycles...")
    spots, cells = build_cell_lifecycles(spots, edges)

    print("Computing neighbors...")
    spots = add_neighbors(spots, cfg)

    print("Building supervised dataset...")
    supervised = build_supervised_from_edges(spots, edges)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Saving...")
    spots.to_parquet(out / "spots.parquet")
    cells.to_parquet(out / "cells.parquet")
    supervised.to_parquet(out / "supervised.parquet")

    print("Done.")


# =========================================================
# CLI
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--xml", required=True)
    parser.add_argument("--out", default="output")
    parser.add_argument("--shape-samples", type=int, default=64)
    parser.add_argument("--no-shape-align", action="store_true")

    args = parser.parse_args()

    cfg = Config(
        shape_samples=args.shape_samples,
        shape_align_to_ellipse=not args.no_shape_align,
    )

    run_pipeline(args.xml, args.out, cfg)
