from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from trackmate_pipeline import (
    Config,
    add_neighbors,
    add_shape_features,
    build_cell_lifecycles,
    build_supervised_from_edges,
    build_tracks,
    filter_data,
    filter_edges_to_spots,
    neighbor_scale_label,
    normalized_neighbor_scales,
    parse_float_tuple,
    parse_edges,
    parse_spots,
)


MULTISCALE_EXACT_COLUMNS = {
    "distance_to_colony_edge",
    "is_boundary_cell",
}

MULTISCALE_PREFIXES = (
    "n_neighbors_r",
    "density_r",
    "Fx_r",
    "Fy_r",
    "F_norm_r",
    "min_dist_r",
    "mean_dist_r",
    "free_space_r",
    "occupancy_area_r",
    "density_grad_x_r",
    "density_grad_y_r",
    "sector_r",
    "ring_count_",
    "ring_density_",
)


def multiscale_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in df.columns
        if column in MULTISCALE_EXACT_COLUMNS
        or any(column.startswith(prefix) for prefix in MULTISCALE_PREFIXES)
    ]


def discover_xml_files(source_root: Path, pattern: str) -> list[Path]:
    if source_root.is_file():
        if source_root.suffix.lower() != ".xml":
            raise ValueError(f"Expected an XML file, got {source_root}.")
        return [source_root]

    files = sorted(path for path in source_root.rglob(pattern) if path.is_file())
    if not files:
        raise FileNotFoundError(f"No XML files found under {source_root} with pattern {pattern!r}.")
    return files


def sequence_name_from_xml(path: Path) -> str:
    name = path.stem
    if name.endswith("_trackmate"):
        return name[:-10]
    return name


def infer_dataset_and_split(path: Path) -> tuple[str, str]:
    parts = path.parts
    if "DynamicNuclearNet" in parts:
        split = next((part for part in parts if part in {"train", "val", "test"}), "unknown")
        return "DynamicNuclearNet", split
    if "6139958" in parts:
        return "6139958", "all"
    if "H2BmCherry_timelapse_60h" in parts:
        return "H2BmCherry_timelapse_60h", "all"
    return path.parent.name, "all"


def split_pipe_ids(value: object) -> list[int]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value)
    if not text:
        return []
    return [int(part) for part in text.split("|") if part]


def ids_to_uids(sequence_uid: str, value: object) -> str:
    return "|".join(f"{sequence_uid}:cell{cell_id}" for cell_id in split_pipe_ids(value))


def to_jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def build_cell_statistics(
    cells: pd.DataFrame,
    spots: pd.DataFrame,
    edges: pd.DataFrame,
    cfg: Config,
    sequence_uid: str,
    sequence_name: str,
    dataset: str,
    source_split: str,
    xml_path: Path,
    source_root: Path,
) -> pd.DataFrame:
    if cells.empty:
        return cells.copy()

    result = cells.copy()
    result.insert(0, "sequence_uid", sequence_uid)
    result.insert(1, "sequence_name", sequence_name)
    result.insert(2, "dataset", dataset)
    result.insert(3, "source_split", source_split)
    try:
        result.insert(4, "xml_path", str(xml_path.relative_to(source_root)))
    except ValueError:
        result.insert(4, "xml_path", str(xml_path))

    result["cell_uid"] = result["cell_id"].map(lambda cell_id: f"{sequence_uid}:cell{int(cell_id)}")
    result["parents_uid"] = result["parents_id"].map(lambda value: ids_to_uids(sequence_uid, value))
    result["childs_uid"] = result["childs_id"].map(lambda value: ids_to_uids(sequence_uid, value))
    result["parent_count"] = result["parents_id"].map(lambda value: len(split_pipe_ids(value)))
    result["child_count"] = result["childs_id"].map(lambda value: len(split_pipe_ids(value)))
    result["is_division"] = result["end_reason"].eq("division")
    result["is_death"] = result["end_reason"].eq("death")

    spot_stats = spots.groupby("cell_id").agg(
        spot_observations=("spot_id", "count"),
        area_mean=("AREA", "mean"),
        area_median=("AREA", "median"),
        area_min=("AREA", "min"),
        area_max=("AREA", "max"),
        circularity_mean=("CIRCULARITY", "mean"),
        solidity_mean=("SOLIDITY", "mean"),
        intensity_mean=("MEAN_INTENSITY_CH1", "mean"),
        speed_mean=("speed", "mean"),
        speed_max=("speed", "max"),
        neighbor_count_mean=("n_neighbors", "mean"),
        local_density_mean=("density", "mean"),
        force_x_mean=("Fx", "mean"),
        force_y_mean=("Fy", "mean"),
        shape_mean_radius_mean=("shape_mean_radius", "mean"),
        shape_radius_cv_mean=("shape_radius_cv", "mean"),
        shape_missing_fraction_mean=("shape_missing_fraction", "mean"),
        shape_area_ratio_mean=("shape_area_ratio", "mean"),
        shape_reconstruction_area_ratio_mean=("shape_reconstruction_area_ratio", "mean"),
    ).reset_index()
    result = result.merge(spot_stats, on="cell_id", how="left")

    shape_norm_columns = [column for column in spots.columns if column.startswith("shape_r_norm_")]
    if shape_norm_columns:
        shape_means = spots.groupby("cell_id")[shape_norm_columns].mean().reset_index()
        shape_means = shape_means.rename(
            columns={
                column: column.replace("shape_r_norm_", "shape_r_norm_mean_")
                for column in shape_norm_columns
            }
        )
        result = result.merge(shape_means, on="cell_id", how="left")

    multiscale_columns = multiscale_feature_columns(spots)
    if multiscale_columns:
        extra_stats = spots.groupby("cell_id").agg({
            column: ["mean", "median", "max"]
            for column in multiscale_columns
        })
        extra_stats.columns = [
            "_".join(str(part) for part in column if part)
            for column in extra_stats.columns.to_flat_index()
        ]
        extra_stats = extra_stats.reset_index()
        result = result.merge(extra_stats, on="cell_id", how="left")

    spot_by_id = spots.set_index("spot_id")
    result["start_x"] = result["start_spot_id"].map(lambda spot_id: spot_by_id.at[spot_id, "x"])
    result["start_y"] = result["start_spot_id"].map(lambda spot_id: spot_by_id.at[spot_id, "y"])
    result["end_x"] = result["end_spot_id"].map(lambda spot_id: spot_by_id.at[spot_id, "x"])
    result["end_y"] = result["end_spot_id"].map(lambda spot_id: spot_by_id.at[spot_id, "y"])
    result["net_displacement"] = np.hypot(result["end_x"] - result["start_x"], result["end_y"] - result["start_y"])

    if not edges.empty:
        spot_to_cell = spots.set_index("spot_id")["cell_id"].dropna().astype(int).to_dict()
        edge_stats_source = edges.copy()
        edge_stats_source["source_cell_id"] = edge_stats_source["source"].map(spot_to_cell)
        edge_stats_source["target_cell_id"] = edge_stats_source["target"].map(spot_to_cell)
        internal_edges = edge_stats_source[
            edge_stats_source["source_cell_id"].notna()
            & edge_stats_source["target_cell_id"].notna()
            & edge_stats_source["source_cell_id"].eq(edge_stats_source["target_cell_id"])
        ]
        edge_stats = internal_edges.groupby("source_cell_id").agg(
            internal_edge_count=("target", "count"),
            path_length=("edge_displacement", "sum"),
            edge_speed_mean=("edge_speed", "mean"),
            edge_speed_max=("edge_speed", "max"),
        ).reset_index().rename(columns={"source_cell_id": "cell_id"})
        edge_stats["cell_id"] = edge_stats["cell_id"].astype(int)
        result = result.merge(edge_stats, on="cell_id", how="left")
    else:
        result["internal_edge_count"] = 0
        result["path_length"] = 0.0
        result["edge_speed_mean"] = np.nan
        result["edge_speed_max"] = np.nan

    result["internal_edge_count"] = result["internal_edge_count"].fillna(0).astype(int)
    result["path_length"] = result["path_length"].fillna(0.0)
    result["straightness"] = np.divide(
        result["net_displacement"],
        result["path_length"],
        out=np.zeros(len(result), dtype=float),
        where=result["path_length"].to_numpy() > 0,
    )
    result["valid_for_ml"] = result["n_spots"] >= cfg.min_track_length
    return result


def build_lineage_edges(cells: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if cells.empty:
        return pd.DataFrame(columns=[
            "sequence_uid",
            "sequence_name",
            "dataset",
            "source_split",
            "parent_cell_id",
            "child_cell_id",
            "parent_cell_uid",
            "child_cell_uid",
            "parent_generation",
            "child_generation",
        ])

    by_cell_id = cells.set_index("cell_id").to_dict("index")
    for row in cells.itertuples(index=False):
        for child_id in split_pipe_ids(row.childs_id):
            child = by_cell_id.get(child_id)
            if child is None:
                continue
            rows.append({
                "sequence_uid": row.sequence_uid,
                "sequence_name": row.sequence_name,
                "dataset": row.dataset,
                "source_split": row.source_split,
                "parent_cell_id": int(row.cell_id),
                "child_cell_id": int(child_id),
                "parent_cell_uid": row.cell_uid,
                "child_cell_uid": child["cell_uid"],
                "parent_generation": int(row.generation),
                "child_generation": int(child["generation"]),
            })
    return pd.DataFrame(rows)


def summarize_sequence(
    sequence_uid: str,
    sequence_name: str,
    dataset: str,
    source_split: str,
    xml_path: Path,
    source_root: Path,
    raw_spots: pd.DataFrame,
    raw_edges: pd.DataFrame,
    spots: pd.DataFrame,
    edges: pd.DataFrame,
    cells: pd.DataFrame,
    supervised: pd.DataFrame,
    cfg: Config,
) -> dict[str, object]:
    try:
        relative_xml = str(xml_path.relative_to(source_root))
    except ValueError:
        relative_xml = str(xml_path)

    missing_target_moves = 0
    if not supervised.empty and "target_dx" in supervised.columns:
        missing_target_moves = int(supervised.loc[supervised["target_move"].eq(1), "target_dx"].isna().sum())

    summary = {
        "sequence_uid": sequence_uid,
        "sequence_name": sequence_name,
        "dataset": dataset,
        "source_split": source_split,
        "xml_path": relative_xml,
        "raw_spots": len(raw_spots),
        "raw_edges": len(raw_edges),
        "filtered_spots": len(spots),
        "filtered_edges": len(edges),
        "removed_spots": len(raw_spots) - len(spots),
        "removed_edges": len(raw_edges) - len(edges),
        "cells": len(cells),
        "generation_max": int(cells["generation"].max()) if not cells.empty else 0,
        "generation_mean": float(cells["generation"].mean()) if not cells.empty else 0.0,
        "division_cells": int(cells["end_reason"].eq("division").sum()) if not cells.empty else 0,
        "death_cells": int(cells["end_reason"].eq("death").sum()) if not cells.empty else 0,
        "valid_for_ml_cells": int((cells["n_spots"] >= cfg.min_track_length).sum()) if not cells.empty else 0,
        "mean_lifetime_frames": float(cells["lifetime_frames"].mean()) if not cells.empty else 0.0,
        "median_lifetime_frames": float(cells["lifetime_frames"].median()) if not cells.empty else 0.0,
        "mean_area": float(spots["AREA"].mean()) if not spots.empty else 0.0,
        "mean_speed": float(spots["speed"].mean()) if not spots.empty else 0.0,
        "mean_neighbors": float(spots["n_neighbors"].mean()) if "n_neighbors" in spots and not spots.empty else 0.0,
        "mean_shape_missing_fraction": (
            float(spots["shape_missing_fraction"].mean())
            if "shape_missing_fraction" in spots and not spots.empty
            else 0.0
        ),
        "mean_shape_area_ratio": (
            float(spots["shape_area_ratio"].mean())
            if "shape_area_ratio" in spots and not spots.empty
            else 0.0
        ),
        "shape_valid_spots": (
            int((spots["shape_valid_rays"] > 0).sum())
            if "shape_valid_rays" in spots and not spots.empty
            else 0
        ),
        "spots_without_cell": int(spots["cell_id"].isna().sum()) if "cell_id" in spots else 0,
        "move_rows_with_missing_target": missing_target_moves,
    }

    if not spots.empty:
        for radius in normalized_neighbor_scales(cfg)[0]:
            label = neighbor_scale_label(radius)
            neighbors_column = f"n_neighbors_r{label}"
            density_column = f"density_r{label}"
            if neighbors_column in spots:
                summary[f"mean_neighbors_r{label}"] = float(spots[neighbors_column].mean())
            if density_column in spots:
                summary[f"mean_density_r{label}"] = float(spots[density_column].mean())
        if "distance_to_colony_edge" in spots:
            summary["mean_distance_to_colony_edge"] = float(spots["distance_to_colony_edge"].mean())
            summary["boundary_cell_fraction"] = float(spots["is_boundary_cell"].mean()) if "is_boundary_cell" in spots else 0.0

    return summary


def grouped_summary(cells: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    if cells.empty:
        return pd.DataFrame(columns=group_columns)

    result = cells.groupby(group_columns, dropna=False).agg(
        cells=("cell_id", "count"),
        valid_for_ml_cells=("valid_for_ml", "sum"),
        division_cells=("is_division", "sum"),
        death_cells=("is_death", "sum"),
        mean_lifetime_frames=("lifetime_frames", "mean"),
        median_lifetime_frames=("lifetime_frames", "median"),
        mean_spots_per_cell=("n_spots", "mean"),
        mean_area=("area_mean", "mean"),
        mean_speed=("edge_speed_mean", "mean"),
        mean_path_length=("path_length", "mean"),
        mean_net_displacement=("net_displacement", "mean"),
        mean_straightness=("straightness", "mean"),
        mean_shape_radius=("shape_mean_radius_mean", "mean"),
        mean_shape_radius_cv=("shape_radius_cv_mean", "mean"),
        mean_shape_missing_fraction=("shape_missing_fraction_mean", "mean"),
        mean_shape_area_ratio=("shape_area_ratio_mean", "mean"),
    ).reset_index()

    extra_columns = [
        column
        for column in cells.columns
        if column.endswith("_mean")
        and (
            column.startswith("distance_to_colony_edge_")
            or column.startswith("is_boundary_cell_")
            or any(column.startswith(prefix) for prefix in MULTISCALE_PREFIXES)
        )
    ]
    if extra_columns:
        extra = cells.groupby(group_columns, dropna=False)[extra_columns].mean().reset_index()
        extra = extra.rename(columns={column: f"group_{column}" for column in extra_columns})
        result = result.merge(extra, on=group_columns, how="left")
    return result


def event_summary(cells: pd.DataFrame) -> pd.DataFrame:
    if cells.empty:
        return pd.DataFrame(columns=["dataset", "source_split", "sequence_name", "generation", "end_frame", "end_reason", "events"])
    return cells.groupby(
        ["dataset", "source_split", "sequence_name", "generation", "end_frame", "end_reason"],
        dropna=False,
    ).size().reset_index(name="events")


def process_xml(xml_path: Path, source_root: Path, sequence_index: int, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    root = ET.parse(xml_path).getroot()
    sequence_name = sequence_name_from_xml(xml_path)
    dataset, source_split = infer_dataset_and_split(xml_path)
    sequence_uid = f"seq{sequence_index:04d}_{sequence_name}"

    raw_spots = parse_spots(root)
    raw_edges = parse_edges(root)
    spots = filter_data(raw_spots, cfg)
    edges = filter_edges_to_spots(raw_edges, spots)
    spots = add_shape_features(spots, cfg)
    spots = build_tracks(spots, edges)
    spots, cells = build_cell_lifecycles(spots, edges)
    spots = add_neighbors(spots, cfg)
    supervised = build_supervised_from_edges(spots, edges)

    cell_stats = build_cell_statistics(
        cells=cells,
        spots=spots,
        edges=edges,
        cfg=cfg,
        sequence_uid=sequence_uid,
        sequence_name=sequence_name,
        dataset=dataset,
        source_split=source_split,
        xml_path=xml_path,
        source_root=source_root,
    )
    sequence_stats = summarize_sequence(
        sequence_uid=sequence_uid,
        sequence_name=sequence_name,
        dataset=dataset,
        source_split=source_split,
        xml_path=xml_path,
        source_root=source_root,
        raw_spots=raw_spots,
        raw_edges=raw_edges,
        spots=spots,
        edges=edges,
        cells=cells,
        supervised=supervised,
        cfg=cfg,
    )
    lineage = build_lineage_edges(cell_stats)
    return cell_stats, lineage, sequence_stats


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_dataset_summary(sequences: pd.DataFrame, cells: pd.DataFrame, lineage: pd.DataFrame, cfg: Config) -> dict[str, object]:
    generation_counts = Counter()
    end_reason_counts = Counter()
    if not cells.empty:
        generation_counts.update({str(int(key)): int(value) for key, value in cells["generation"].value_counts().sort_index().items()})
        end_reason_counts.update({str(key): int(value) for key, value in cells["end_reason"].value_counts().items()})

    return {
        "config": {
            "neighbor_radii": list(cfg.neighbor_radii),
            "neighbor_decays": list(cfg.neighbor_decays),
            "k_nearest": cfg.k_nearest,
            "n_sectors": cfg.n_sectors,
            "shape_samples": cfg.shape_samples,
            "shape_align_to_ellipse": cfg.shape_align_to_ellipse,
            "min_area": cfg.min_area,
            "max_area": cfg.max_area,
            "min_track_length": cfg.min_track_length,
        },
        "xml_files": int(len(sequences)),
        "raw_spots": int(sequences["raw_spots"].sum()) if not sequences.empty else 0,
        "raw_edges": int(sequences["raw_edges"].sum()) if not sequences.empty else 0,
        "filtered_spots": int(sequences["filtered_spots"].sum()) if not sequences.empty else 0,
        "filtered_edges": int(sequences["filtered_edges"].sum()) if not sequences.empty else 0,
        "cells": int(len(cells)),
        "lineage_edges": int(len(lineage)),
        "division_cells": int(cells["is_division"].sum()) if not cells.empty else 0,
        "death_cells": int(cells["is_death"].sum()) if not cells.empty else 0,
        "max_generation": int(cells["generation"].max()) if not cells.empty else 0,
        "generation_counts": dict(generation_counts),
        "end_reason_counts": dict(end_reason_counts),
        "move_rows_with_missing_target": int(sequences["move_rows_with_missing_target"].sum()) if not sequences.empty else 0,
        "spots_without_cell": int(sequences["spots_without_cell"].sum()) if not sequences.empty else 0,
        "shape_valid_spots": int(sequences["shape_valid_spots"].sum()) if not sequences.empty else 0,
        "mean_shape_missing_fraction": float(sequences["mean_shape_missing_fraction"].mean()) if not sequences.empty else 0.0,
        "mean_shape_area_ratio": float(sequences["mean_shape_area_ratio"].mean()) if not sequences.empty else 0.0,
    }


def run_statistics(args: argparse.Namespace) -> int:
    source_root = Path(args.source_root).resolve()
    relative_root = source_root.parent if source_root.is_file() else source_root
    out_dir = Path(args.out_dir).resolve()
    neighbor_radii = parse_float_tuple(args.neighbor_radii, "neighbor-radii")
    neighbor_decays = parse_float_tuple(args.neighbor_decays, "neighbor-decays")
    cfg = Config(
        neighbor_radii=neighbor_radii,
        neighbor_decays=neighbor_decays,
        k_nearest=args.k_nearest,
        n_sectors=args.n_sectors,
        shape_samples=args.shape_samples,
        shape_align_to_ellipse=not args.no_shape_align,
        min_area=args.min_area,
        max_area=args.max_area,
        min_track_length=args.min_track_length,
    )

    xml_files = discover_xml_files(source_root, args.pattern)
    print(f"Found {len(xml_files)} XML files.")

    cell_tables: list[pd.DataFrame] = []
    lineage_tables: list[pd.DataFrame] = []
    sequence_rows: list[dict[str, object]] = []
    failed_rows: list[dict[str, object]] = []

    for index, xml_path in enumerate(xml_files, start=1):
        try:
            cells, lineage, sequence_stats = process_xml(xml_path, relative_root, index, cfg)
        except Exception as exc:
            print(f"[{index}/{len(xml_files)}] failed {xml_path}: {exc}")
            failed_rows.append({"xml_path": str(xml_path), "error": str(exc)})
            if not args.keep_going:
                raise
            continue

        cell_tables.append(cells)
        lineage_tables.append(lineage)
        sequence_rows.append(sequence_stats)
        print(
            f"[{index}/{len(xml_files)}] {sequence_stats['sequence_name']}: "
            f"cells={sequence_stats['cells']} divisions={sequence_stats['division_cells']} "
            f"max_generation={sequence_stats['generation_max']}"
        )

    cells_all = pd.concat(cell_tables, ignore_index=True, sort=False) if cell_tables else pd.DataFrame()
    lineage_all = pd.concat(lineage_tables, ignore_index=True, sort=False) if lineage_tables else pd.DataFrame()
    sequence_summary = pd.DataFrame(sequence_rows)
    failed = pd.DataFrame(failed_rows)

    generation_summary = grouped_summary(cells_all, ["dataset", "source_split", "generation"])
    sequence_generation_summary = grouped_summary(cells_all, ["dataset", "source_split", "sequence_name", "generation"])
    events = event_summary(cells_all)
    dataset_summary = build_dataset_summary(sequence_summary, cells_all, lineage_all, cfg)

    write_csv(sequence_summary, out_dir / "sequence_summary.csv")
    write_csv(cells_all, out_dir / "cell_lifecycle_statistics.csv")
    write_csv(generation_summary, out_dir / "generation_summary.csv")
    write_csv(sequence_generation_summary, out_dir / "sequence_generation_summary.csv")
    write_csv(events, out_dir / "event_summary.csv")
    write_csv(lineage_all, out_dir / "lineage_edges.csv")
    if not failed.empty:
        write_csv(failed, out_dir / "failed_xml.csv")
    write_json(dataset_summary, out_dir / "dataset_summary.json")

    print(f"Saved statistics to {out_dir}")
    return 0


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build lineage and lifecycle statistics from TrackMate XML files.")
    parser.add_argument("--source-root", default=".", help="XML file or folder where XML files are searched recursively.")
    parser.add_argument("--pattern", default="*.xml", help="Recursive glob pattern used when source-root is a folder.")
    parser.add_argument("--out-dir", default="statistics_output", help="Output folder for CSV and JSON statistics.")
    parser.add_argument("--neighbor-radii", type=str, default="20,40,80,160")
    parser.add_argument("--neighbor-decays", type=str, default="15,30,60,120")
    parser.add_argument("--k-nearest", type=int, default=5)
    parser.add_argument("--n-sectors", type=int, default=8)
    parser.add_argument("--shape-samples", type=int, default=64)
    parser.add_argument("--no-shape-align", action="store_true")
    parser.add_argument("--min-area", type=float, default=50.0)
    parser.add_argument("--max-area", type=float, default=1000.0)
    parser.add_argument("--min-track-length", type=int, default=3)
    parser.add_argument("--keep-going", action="store_true", help="Continue if one XML fails and write failed_xml.csv.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run_statistics(parse_args()))
