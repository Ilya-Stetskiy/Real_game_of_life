from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from trackmate_pipeline import (
    Config,
    add_neighbors,
    add_shape_features,
    build_cell_lifecycles,
    build_tracks,
    filter_data,
    filter_edges_to_spots,
    parse_float_tuple,
    parse_edges,
    parse_spots,
)
from trackmate_statistics import discover_xml_files, infer_dataset_and_split, sequence_name_from_xml, to_jsonable


SHAPE_SCALAR_FEATURES = [
    "CIRCULARITY",
    "SOLIDITY",
    "ELLIPSE_ASPECTRATIO",
    "shape_radius_cv",
    "shape_area_ratio",
    "shape_reconstruction_area_ratio",
    "shape_contour_centroid_dx",
    "shape_contour_centroid_dy",
]

SIZE_FEATURES = [
    "AREA",
    "PERIMETER",
    "RADIUS",
    "ELLIPSE_MAJOR",
    "ELLIPSE_MINOR",
    "shape_mean_radius",
]


def parse_horizons(value: str) -> list[int]:
    horizons = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not horizons or any(item < 1 for item in horizons):
        raise ValueError("Horizons must be positive integers, for example: 3,5,10.")
    return horizons


def write_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def safe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def build_spot_table(xml_path: Path, source_root: Path, sequence_index: int, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
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

    if cells.empty or spots.empty:
        return spots, cells, {
            "sequence_uid": sequence_uid,
            "sequence_name": sequence_name,
            "dataset": dataset,
            "source_split": source_split,
            "xml_path": safe_relative(xml_path, source_root),
            "raw_spots": len(raw_spots),
            "filtered_spots": len(spots),
            "cells": len(cells),
            "division_cells": 0,
            "shape_valid_spots": 0,
        }

    cell_context = cells[[
        "cell_id",
        "start_frame",
        "end_frame",
        "start_t",
        "end_t",
        "lifetime_frames",
        "n_spots",
    ]].rename(columns={
        "start_frame": "cell_start_frame",
        "end_frame": "cell_end_frame",
        "start_t": "cell_start_t",
        "end_t": "cell_end_t",
        "lifetime_frames": "cell_lifetime_frames",
        "n_spots": "cell_n_spots",
    })

    spots = spots.merge(cell_context, on="cell_id", how="left")
    spots.insert(0, "sequence_uid", sequence_uid)
    spots.insert(1, "sequence_name", sequence_name)
    spots.insert(2, "dataset", dataset)
    spots.insert(3, "source_split", source_split)
    spots.insert(4, "xml_path", safe_relative(xml_path, source_root))
    spots["spot_uid"] = spots["sequence_uid"] + ":spot" + spots["spot_id"].astype(str)
    spots["cell_uid"] = spots["sequence_uid"] + ":cell" + spots["cell_id"].fillna(-1).astype(int).astype(str)
    spots["cell_generation"] = spots["generation"]
    spots["is_observed_division_cell"] = spots["cell_end_reason"].eq("division")
    spots["frames_to_cell_end"] = spots["cell_end_frame"] - spots["frame"]
    spots["frames_from_cell_start"] = spots["frame"] - spots["cell_start_frame"]
    spots["cell_phase"] = np.divide(
        spots["frames_from_cell_start"],
        spots["cell_lifetime_frames"],
        out=np.zeros(len(spots), dtype=float),
        where=spots["cell_lifetime_frames"].to_numpy() > 0,
    )

    cells = cells.copy()
    cells.insert(0, "sequence_uid", sequence_uid)
    cells.insert(1, "sequence_name", sequence_name)
    cells.insert(2, "dataset", dataset)
    cells.insert(3, "source_split", source_split)
    cells.insert(4, "xml_path", safe_relative(xml_path, source_root))
    cells["cell_uid"] = cells["sequence_uid"] + ":cell" + cells["cell_id"].astype(str)
    cells["is_observed_division"] = cells["end_reason"].eq("division")

    sequence_summary = {
        "sequence_uid": sequence_uid,
        "sequence_name": sequence_name,
        "dataset": dataset,
        "source_split": source_split,
        "xml_path": safe_relative(xml_path, source_root),
        "raw_spots": len(raw_spots),
        "filtered_spots": len(spots),
        "raw_edges": len(raw_edges),
        "filtered_edges": len(edges),
        "cells": len(cells),
        "division_cells": int(cells["is_observed_division"].sum()),
        "shape_valid_spots": int((spots["shape_valid_rays"] > 0).sum()),
        "mean_shape_missing_fraction": float(spots["shape_missing_fraction"].mean()) if len(spots) else 0.0,
    }
    return spots, cells, sequence_summary


def add_future_labels(spots: pd.DataFrame, horizons: list[int], lead_frames: int) -> pd.DataFrame:
    spots = spots.copy()
    spots["future_observed_division"] = spots["is_observed_division_cell"] & (spots["frames_to_cell_end"] >= lead_frames)
    spots["eligible_future_eventual"] = spots["frames_to_cell_end"] >= lead_frames

    for horizon in horizons:
        target = f"division_within_{horizon}_frames"
        eligible = f"eligible_within_{horizon}_frames"
        spots[target] = (
            spots["is_observed_division_cell"]
            & (spots["frames_to_cell_end"] >= lead_frames)
            & (spots["frames_to_cell_end"] <= horizon)
        )
        spots[eligible] = (
            (spots["frames_to_cell_end"] >= horizon)
            | (
                spots["is_observed_division_cell"]
                & (spots["frames_to_cell_end"] >= lead_frames)
                & (spots["frames_to_cell_end"] <= horizon)
            )
        )

    return spots


def cohen_d(pos: pd.Series, neg: pd.Series) -> float:
    pos = pd.to_numeric(pos, errors="coerce").dropna()
    neg = pd.to_numeric(neg, errors="coerce").dropna()
    if len(pos) < 2 or len(neg) < 2:
        return np.nan
    pooled = math.sqrt(((len(pos) - 1) * pos.var(ddof=1) + (len(neg) - 1) * neg.var(ddof=1)) / (len(pos) + len(neg) - 2))
    if pooled == 0 or not np.isfinite(pooled):
        return np.nan
    return float((pos.mean() - neg.mean()) / pooled)


def feature_effects(df: pd.DataFrame, target: str, features: list[str]) -> pd.DataFrame:
    rows = []
    if df.empty or target not in df:
        return pd.DataFrame()

    y = df[target].astype(bool)
    for feature in features:
        if feature not in df:
            continue
        values = pd.to_numeric(df[feature], errors="coerce")
        pos = values[y].dropna()
        neg = values[~y].dropna()
        if len(pos) < 2 or len(neg) < 2:
            continue
        rows.append({
            "target": target,
            "feature": feature,
            "positive_count": int(len(pos)),
            "negative_count": int(len(neg)),
            "positive_mean": float(pos.mean()),
            "negative_mean": float(neg.mean()),
            "positive_median": float(pos.median()),
            "negative_median": float(neg.median()),
            "mean_diff": float(pos.mean() - neg.mean()),
            "cohen_d": cohen_d(pos, neg),
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["abs_cohen_d"] = result["cohen_d"].abs()
        result = result.sort_values("abs_cohen_d", ascending=False)
    return result


def angular_shape_difference(df: pd.DataFrame, target: str, prefix: str) -> pd.DataFrame:
    columns = [column for column in df.columns if column.startswith(prefix)]
    if not columns or target not in df:
        return pd.DataFrame()

    columns = sorted(columns)
    y = df[target].astype(bool)
    pos = df.loc[y, columns].apply(pd.to_numeric, errors="coerce")
    neg = df.loc[~y, columns].apply(pd.to_numeric, errors="coerce")
    if len(pos) == 0 or len(neg) == 0:
        return pd.DataFrame()

    rows = []
    samples = len(columns)
    for index, column in enumerate(columns):
        pos_mean = float(pos[column].mean())
        neg_mean = float(neg[column].mean())
        rows.append({
            "target": target,
            "shape_column": column,
            "angle_deg": float(index * 360.0 / samples),
            "positive_mean": pos_mean,
            "negative_mean": neg_mean,
            "mean_diff": pos_mean - neg_mean,
            "abs_mean_diff": abs(pos_mean - neg_mean),
        })
    return pd.DataFrame(rows).sort_values("abs_mean_diff", ascending=False)


def usable_features(df: pd.DataFrame, features: list[str]) -> list[str]:
    usable = []
    for feature in features:
        if feature not in df:
            continue
        values = pd.to_numeric(df[feature], errors="coerce")
        finite = values[np.isfinite(values)]
        if len(finite) >= 3 and finite.nunique() > 1:
            usable.append(feature)
    return usable


def grouped_logistic_score(df: pd.DataFrame, target: str, features: list[str], group_column: str) -> dict[str, object]:
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score, roc_auc_score
        from sklearn.model_selection import GroupKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        return {"target": target, "status": "skipped", "reason": f"sklearn unavailable: {exc}"}

    features = usable_features(df, features)
    needed = [target, group_column, *features]
    data = df[needed].copy()
    data = data.dropna(subset=[target, group_column])
    data = data[data[features].notna().any(axis=1)] if features else data.iloc[0:0]

    if not features or data.empty:
        return {"target": target, "status": "skipped", "reason": "no usable features"}

    y = data[target].astype(int).to_numpy()
    groups = data[group_column].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if len(np.unique(y)) < 2:
        return {"target": target, "status": "skipped", "reason": "single target class", "rows": int(len(data))}
    if len(unique_groups) < 3:
        return {"target": target, "status": "skipped", "reason": "not enough groups", "rows": int(len(data))}

    n_splits = min(5, len(unique_groups))
    splitter = GroupKFold(n_splits=n_splits)
    aucs = []
    aps = []
    valid_folds = 0

    for train_index, test_index in splitter.split(data[features], y, groups):
        y_train = y[train_index]
        y_test = y[test_index]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear"),
        )
        model.fit(data.iloc[train_index][features], y_train)
        scores = model.predict_proba(data.iloc[test_index][features])[:, 1]
        aucs.append(float(roc_auc_score(y_test, scores)))
        aps.append(float(average_precision_score(y_test, scores)))
        valid_folds += 1

    if not aucs:
        return {"target": target, "status": "skipped", "reason": "no fold had both target classes", "rows": int(len(data))}

    return {
        "target": target,
        "status": "ok",
        "rows": int(len(data)),
        "positives": int(y.sum()),
        "positive_rate": float(y.mean()),
        "groups": int(len(unique_groups)),
        "folds": int(valid_folds),
        "features": int(len(features)),
        "roc_auc_mean": float(np.mean(aucs)),
        "roc_auc_std": float(np.std(aucs)),
        "average_precision_mean": float(np.mean(aps)),
        "average_precision_std": float(np.std(aps)),
    }


def build_cell_preterminal_table(spots: pd.DataFrame, lead_frames: int, min_preterminal_spots: int) -> pd.DataFrame:
    eligible = spots[spots["frames_to_cell_end"] >= lead_frames].copy()
    if eligible.empty:
        return pd.DataFrame()

    shape_norm_columns = sorted(column for column in eligible.columns if column.startswith("shape_r_norm_"))
    aggregate_features = [feature for feature in [*SHAPE_SCALAR_FEATURES, *SIZE_FEATURES, *shape_norm_columns] if feature in eligible]
    grouped = eligible.groupby([
        "sequence_uid",
        "sequence_name",
        "dataset",
        "source_split",
        "cell_id",
        "cell_uid",
        "cell_generation",
        "cell_end_reason",
        "cell_lifetime_frames",
        "cell_n_spots",
    ], dropna=False)[aggregate_features].mean().reset_index()
    counts = eligible.groupby("cell_uid").size().rename("preterminal_spots").reset_index()
    grouped = grouped.merge(counts, on="cell_uid", how="left")
    grouped = grouped[grouped["preterminal_spots"] >= min_preterminal_spots]
    grouped["is_observed_division"] = grouped["cell_end_reason"].eq("division")
    return grouped


def run_analysis(args: argparse.Namespace) -> int:
    source_root = Path(args.source_root).resolve()
    relative_root = source_root.parent if source_root.is_file() else source_root
    out_dir = Path(args.out_dir).resolve()
    horizons = parse_horizons(args.horizons)
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
    spot_tables: list[pd.DataFrame] = []
    cell_tables: list[pd.DataFrame] = []
    sequence_rows: list[dict[str, object]] = []
    failed_rows: list[dict[str, object]] = []

    print(f"Found {len(xml_files)} XML files.")
    for index, xml_path in enumerate(xml_files, start=1):
        try:
            spots, cells, sequence_summary = build_spot_table(xml_path, relative_root, index, cfg)
        except Exception as exc:
            print(f"[{index}/{len(xml_files)}] failed {xml_path}: {exc}")
            failed_rows.append({"xml_path": str(xml_path), "error": str(exc)})
            if not args.keep_going:
                raise
            continue

        spots = add_future_labels(spots, horizons, args.lead_frames)
        spot_tables.append(spots)
        cell_tables.append(cells)
        sequence_rows.append(sequence_summary)
        print(
            f"[{index}/{len(xml_files)}] {sequence_summary['sequence_name']}: "
            f"spots={sequence_summary['filtered_spots']} cells={sequence_summary['cells']} "
            f"divisions={sequence_summary['division_cells']}"
        )

    spots_all = pd.concat(spot_tables, ignore_index=True, sort=False) if spot_tables else pd.DataFrame()
    cells_all = pd.concat(cell_tables, ignore_index=True, sort=False) if cell_tables else pd.DataFrame()
    sequence_summary = pd.DataFrame(sequence_rows)
    failed = pd.DataFrame(failed_rows)
    cell_preterminal = build_cell_preterminal_table(spots_all, args.lead_frames, args.min_preterminal_spots)

    shape_norm_columns = sorted(column for column in spots_all.columns if column.startswith("shape_r_norm_"))
    scalar_features = [feature for feature in SHAPE_SCALAR_FEATURES if feature in spots_all]
    size_features = [feature for feature in SIZE_FEATURES if feature in spots_all]
    spot_shape_features = [*scalar_features, *shape_norm_columns]
    spot_shape_size_features = [*scalar_features, *size_features, *shape_norm_columns]
    cell_shape_columns = sorted(column for column in cell_preterminal.columns if column.startswith("shape_r_norm_"))
    cell_shape_features = [feature for feature in SHAPE_SCALAR_FEATURES if feature in cell_preterminal] + cell_shape_columns
    cell_shape_size_features = [
        feature for feature in [*SHAPE_SCALAR_FEATURES, *SIZE_FEATURES] if feature in cell_preterminal
    ] + cell_shape_columns

    effect_tables = []
    angular_tables = []
    model_rows = []

    if not cell_preterminal.empty:
        effect_tables.append(feature_effects(cell_preterminal, "is_observed_division", cell_shape_size_features))
        angular_tables.append(angular_shape_difference(cell_preterminal, "is_observed_division", "shape_r_norm_"))
        for feature_set_name, features in (
            ("cell_preterminal_shape", cell_shape_features),
            ("cell_preterminal_shape_plus_size", cell_shape_size_features),
        ):
            row = grouped_logistic_score(cell_preterminal, "is_observed_division", features, "sequence_uid")
            row["task"] = feature_set_name
            model_rows.append(row)

    for horizon in horizons:
        eligible_column = f"eligible_within_{horizon}_frames"
        target_column = f"division_within_{horizon}_frames"
        horizon_spots = spots_all[spots_all[eligible_column].fillna(False)].copy() if eligible_column in spots_all else pd.DataFrame()
        if horizon_spots.empty:
            continue
        effects = feature_effects(horizon_spots, target_column, spot_shape_size_features)
        if not effects.empty:
            effects.insert(0, "horizon_frames", horizon)
            effect_tables.append(effects)
        angles = angular_shape_difference(horizon_spots, target_column, "shape_r_norm_")
        if not angles.empty:
            angles.insert(0, "horizon_frames", horizon)
            angular_tables.append(angles)
        for feature_set_name, features in (
            (f"spot_division_within_{horizon}_shape", spot_shape_features),
            (f"spot_division_within_{horizon}_shape_plus_size", spot_shape_size_features),
        ):
            row = grouped_logistic_score(horizon_spots, target_column, features, "sequence_uid")
            row["task"] = feature_set_name
            row["horizon_frames"] = horizon
            model_rows.append(row)

    effects_all = pd.concat(effect_tables, ignore_index=True, sort=False) if effect_tables else pd.DataFrame()
    angular_all = pd.concat(angular_tables, ignore_index=True, sort=False) if angular_tables else pd.DataFrame()
    model_scores = pd.DataFrame(model_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    spots_all.to_parquet(out_dir / "spot_shape_division_dataset.parquet", index=False)
    cells_all.to_csv(out_dir / "cells_with_division_labels.csv", index=False)
    cell_preterminal.to_csv(out_dir / "cell_preterminal_shape_features.csv", index=False)
    sequence_summary.to_csv(out_dir / "sequence_summary.csv", index=False)
    effects_all.to_csv(out_dir / "shape_division_effects.csv", index=False)
    angular_all.to_csv(out_dir / "angular_shape_difference.csv", index=False)
    model_scores.to_csv(out_dir / "shape_division_model_scores.csv", index=False)
    if not failed.empty:
        failed.to_csv(out_dir / "failed_xml.csv", index=False)

    summary = {
        "config": {
            "neighbor_radii": list(cfg.neighbor_radii),
            "neighbor_decays": list(cfg.neighbor_decays),
            "n_sectors": cfg.n_sectors,
            "shape_samples": cfg.shape_samples,
            "shape_align_to_ellipse": cfg.shape_align_to_ellipse,
            "lead_frames": args.lead_frames,
            "horizons": horizons,
            "min_area": cfg.min_area,
            "max_area": cfg.max_area,
            "min_preterminal_spots": args.min_preterminal_spots,
        },
        "xml_files": int(len(xml_files)),
        "processed_xml_files": int(len(sequence_summary)),
        "failed_xml_files": int(len(failed)),
        "spots": int(len(spots_all)),
        "cells": int(len(cells_all)),
        "preterminal_cells": int(len(cell_preterminal)),
        "division_cells": int(cells_all["is_observed_division"].sum()) if not cells_all.empty else 0,
        "shape_valid_spots": int((spots_all["shape_valid_rays"] > 0).sum()) if not spots_all.empty else 0,
        "mean_shape_missing_fraction": (
            float(spots_all["shape_missing_fraction"].mean()) if not spots_all.empty else 0.0
        ),
        "top_effects": effects_all.head(20).to_dict("records") if not effects_all.empty else [],
        "model_scores": model_scores.to_dict("records") if not model_scores.empty else [],
    }
    write_json(summary, out_dir / "shape_division_analysis_summary.json")
    print(f"Saved shape/division analysis to {out_dir}")
    return 0


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze whether cell shape is associated with future observed division.")
    parser.add_argument("--source-root", default=".", help="XML file or folder where XML files are searched recursively.")
    parser.add_argument("--pattern", default="*.xml", help="Recursive glob pattern used when source-root is a folder.")
    parser.add_argument("--out-dir", default="shape_division_analysis_output")
    parser.add_argument("--horizons", default="3,5,10", help="Comma-separated future division horizons in frames.")
    parser.add_argument("--lead-frames", type=int, default=1, help="Exclude terminal division/death frame from future labels.")
    parser.add_argument("--min-preterminal-spots", type=int, default=2)
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
    raise SystemExit(run_analysis(parse_args()))
