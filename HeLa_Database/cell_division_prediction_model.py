from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd


DEFAULT_SOURCE = Path(__file__).resolve().parent / "shape_division_analysis_dynamic" / "spot_shape_division_dataset.parquet"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "division_prediction_model"

IDENTITY_COLUMNS = [
    "sequence_uid",
    "sequence_name",
    "dataset",
    "source_split",
    "spot_uid",
    "spot_id",
    "cell_uid",
    "cell_id",
    "frame",
    "t",
]

SHAPE_SCALAR_FEATURES = [
    "CIRCULARITY",
    "SOLIDITY",
    "ELLIPSE_ASPECTRATIO",
    "ELLIPSE_ECCENTRICITY",
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

MOTION_CONTEXT_FEATURES = [
    "x",
    "y",
    "dx",
    "dy",
    "speed",
    "n_neighbors",
    "density",
    "Fx",
    "Fy",
    "frames_from_cell_start",
    "generation",
]


@dataclass(frozen=True)
class DivisionPredictionConfig:
    source: Path = DEFAULT_SOURCE
    out_dir: Path = DEFAULT_OUT_DIR
    horizons: tuple[int, ...] = (3, 5, 10)
    lags: tuple[int, ...] = (1, 3, 5)
    seed: int = 17
    folds: int = 5
    random_forest_trees: int = 500
    random_forest_min_samples_leaf: int = 20
    max_logistic_iter: int = 200
    models: tuple[str, ...] = ("logistic_regression",)


def parse_int_tuple(value: str) -> tuple[int, ...]:
    items = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    if not items or any(item < 1 for item in items):
        raise ValueError("Expected positive comma-separated integers.")
    return items


def read_source(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Source dataset does not exist: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported source format: {path.suffix}. Use .parquet or .csv.")


def add_eccentricity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    major = pd.to_numeric(df.get("ELLIPSE_MAJOR"), errors="coerce").to_numpy(dtype=float)
    minor = pd.to_numeric(df.get("ELLIPSE_MINOR"), errors="coerce").to_numpy(dtype=float)
    ratio = np.divide(minor, major, out=np.full(len(df), np.nan, dtype=float), where=major > 0)
    df["ELLIPSE_ECCENTRICITY"] = np.sqrt(np.clip(1.0 - ratio * ratio, 0.0, 1.0))
    return df


def numeric_existing_columns(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    result = []
    for column in columns:
        if column not in df:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if values.notna().sum() >= 3 and values.nunique(dropna=True) > 1:
            result.append(column)
    return result


def shape_vector_columns(df: pd.DataFrame) -> list[str]:
    return sorted(column for column in df.columns if column.startswith("shape_r_norm_"))


def add_temporal_features(df: pd.DataFrame, lags: tuple[int, ...]) -> pd.DataFrame:
    df = add_eccentricity(df)
    sort_columns = [column for column in ["sequence_uid", "cell_uid", "frame", "spot_id"] if column in df]
    df = df.sort_values(sort_columns).copy()

    current_features = numeric_existing_columns(
        df,
        [*SHAPE_SCALAR_FEATURES, *SIZE_FEATURES, *MOTION_CONTEXT_FEATURES, *shape_vector_columns(df)],
    )
    for column in current_features:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    grouped = df.groupby(temporal_group_columns(df), sort=False, dropna=False)
    dynamic_base = numeric_existing_columns(
        df,
        [
            *SHAPE_SCALAR_FEATURES,
            *SIZE_FEATURES,
            "x",
            "y",
            "dx",
            "dy",
            "speed",
            "n_neighbors",
            "density",
            "Fx",
            "Fy",
        ],
    )

    for lag in lags:
        new_columns = {}
        for column in dynamic_base:
            lag_values = grouped[column].shift(lag)
            new_columns[f"{column}_lag{lag}"] = lag_values
            new_columns[f"{column}_delta{lag}"] = df[column] - lag_values

        if {"x", "y"}.issubset(df.columns):
            dx = df["x"] - grouped["x"].shift(lag)
            dy = df["y"] - grouped["y"].shift(lag)
            new_columns[f"net_displacement_lag{lag}"] = np.sqrt(dx * dx + dy * dy)
            new_columns[f"net_speed_lag{lag}"] = new_columns[f"net_displacement_lag{lag}"] / float(lag)
        df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)

    if {"dx", "dy", "Fx", "Fy"}.issubset(df.columns):
        force_norm = np.sqrt(df["Fx"] * df["Fx"] + df["Fy"] * df["Fy"])
        velocity_norm = np.sqrt(df["dx"] * df["dx"] + df["dy"] * df["dy"])
        dot = df["dx"] * df["Fx"] + df["dy"] * df["Fy"]
        df["motion_pressure_alignment"] = np.divide(
            dot,
            (force_norm * velocity_norm).to_numpy(dtype=float),
            out=np.zeros(len(df), dtype=float),
            where=(force_norm.to_numpy(dtype=float) > 0) & (velocity_norm.to_numpy(dtype=float) > 0),
        )

    rolling_columns = {}
    for column in ["AREA", "ELLIPSE_ECCENTRICITY", "ELLIPSE_ASPECTRATIO", "CIRCULARITY", "SOLIDITY", "speed"]:
        if column not in df:
            continue
        rolling_columns[f"{column}_rolling3_mean"] = grouped[column].transform(lambda x: x.rolling(3, min_periods=1).mean())
        rolling_columns[f"{column}_rolling3_std"] = grouped[column].transform(lambda x: x.rolling(3, min_periods=2).std())
    if rolling_columns:
        df = pd.concat([df, pd.DataFrame(rolling_columns, index=df.index)], axis=1)

    return df


def temporal_group_columns(df: pd.DataFrame) -> list[str]:
    columns = [column for column in ("dataset", "sequence_uid", "cell_uid") if column in df]
    if "cell_uid" not in columns:
        raise ValueError("Cannot build temporal features without cell_uid column.")
    return columns


def candidate_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded_prefixes = ("division_within_", "eligible_within_", "future_observed_division", "is_observed_division")
    excluded = {
        *IDENTITY_COLUMNS,
        "xml_path",
        "contour_xy_local",
        "next_ids",
        "prev_ids",
        "next_id",
        "prev_id",
        "cell_start",
        "cell_end",
        "cell_end_reason",
        "parents_id",
        "childs_id",
        "cell_start_frame",
        "cell_end_frame",
        "cell_start_t",
        "cell_end_t",
        "cell_lifetime_frames",
        "cell_n_spots",
        "cell_phase",
        "frames_to_cell_end",
        "eligible_future_eventual",
        "has_next",
        "has_prev",
        "is_split",
        "is_merge",
        "n_next",
        "n_prev",
    }
    features = []
    for column in df.columns:
        if column in excluded or (column.startswith("shape_r_") and not column.startswith("shape_r_norm_")) or any(column.startswith(prefix) for prefix in excluded_prefixes):
            continue
        if not pd.api.types.is_numeric_dtype(df[column]):
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if values.notna().sum() >= 3 and values.nunique(dropna=True) > 1:
            features.append(column)
    return features


def build_model_specs(config: DivisionPredictionConfig) -> dict[str, Any]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "logistic_regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            SGDClassifier(
                loss="log_loss",
                max_iter=config.max_logistic_iter,
                tol=1e-3,
                class_weight="balanced",
                n_jobs=-1,
                random_state=config.seed,
            ),
        ),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=config.random_forest_trees,
                min_samples_leaf=config.random_forest_min_samples_leaf,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=config.seed,
            ),
        ),
    }


def valid_group_folds(df: pd.DataFrame, target: str, group_column: str, folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import GroupKFold
    from sklearn.base import clone

    groups = df[group_column].astype(str).to_numpy()
    y = df[target].astype(int).to_numpy()
    n_splits = min(folds, len(np.unique(groups)))
    if n_splits < 2:
        return []

    result = []
    splitter = GroupKFold(n_splits=n_splits)
    for train_index, test_index in splitter.split(df, y, groups):
        if len(np.unique(y[train_index])) < 2 or len(np.unique(y[test_index])) < 2:
            continue
        result.append((train_index, test_index))
    return result


def topk_metrics(y_true: np.ndarray, score: np.ndarray, k: int) -> dict[str, float]:
    if y_true.size == 0:
        return {f"top{k}_hits": 0.0, f"top{k}_precision": 0.0, f"top{k}_recall": 0.0}
    k = min(k, y_true.size)
    order = np.argsort(-score)[:k]
    hits = float(y_true[order].sum())
    positives = float(y_true.sum())
    return {
        f"top{k}_hits": hits,
        f"top{k}_precision": hits / float(k) if k else 0.0,
        f"top{k}_recall": hits / positives if positives else 0.0,
    }


def score_predictions(y_true: np.ndarray, score: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

    result: dict[str, float] = {
        "rows": float(y_true.size),
        "positives": float(y_true.sum()),
        "positive_rate": float(y_true.mean()) if y_true.size else 0.0,
    }
    if len(np.unique(y_true)) > 1:
        result["roc_auc"] = float(roc_auc_score(y_true, score))
        result["average_precision"] = float(average_precision_score(y_true, score))
    else:
        result["roc_auc"] = math.nan
        result["average_precision"] = math.nan

    predicted = score >= 0.5
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true.astype(bool),
        predicted,
        average="binary",
        zero_division=0,
    )
    result.update({"precision_at_0_5": float(precision), "recall_at_0_5": float(recall), "f1_at_0_5": float(f1)})
    for k in (10, 50, 100):
        result.update(topk_metrics(y_true, score, k))
    return result


def evaluate_models(
    data: pd.DataFrame,
    target: str,
    features: list[str],
    config: DivisionPredictionConfig,
) -> tuple[list[dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    from sklearn.base import clone

    folds = valid_group_folds(data, target, "sequence_uid", config.folds)
    if not folds:
        return (
            [{"target": target, "status": "skipped", "reason": "no valid group fold contains both classes"}],
            pd.DataFrame(),
            {},
        )

    rows = []
    prediction_parts = []
    best_name = ""
    best_ap = -np.inf
    x_all = data[features]
    y_all = data[target].astype(int).to_numpy()

    for model_name, model_template in build_model_specs(config).items():
        if model_name not in config.models:
            continue
        oof = np.full(len(data), np.nan, dtype=float)
        fold_rows = []
        for fold_index, (train_index, test_index) in enumerate(folds, start=1):
            model = clone(model_template)
            model.fit(x_all.iloc[train_index], y_all[train_index])
            scores = model.predict_proba(x_all.iloc[test_index])[:, 1]
            oof[test_index] = scores
            fold_score = score_predictions(y_all[test_index], scores)
            fold_score.update({"target": target, "model": model_name, "fold": fold_index, "status": "fold"})
            fold_rows.append(fold_score)

        valid = np.isfinite(oof)
        aggregate = score_predictions(y_all[valid], oof[valid])
        aggregate.update(
            {
                "target": target,
                "model": model_name,
                "status": "ok",
                "fold": np.nan,
                "folds": len(folds),
                "features": len(features),
                "validated_rows": int(valid.sum()),
            }
        )
        rows.append(aggregate)
        rows.extend(fold_rows)

        prediction = data[[column for column in IDENTITY_COLUMNS if column in data]].copy()
        prediction["target"] = target
        prediction["model"] = model_name
        prediction["y_true"] = y_all
        prediction["score"] = oof
        prediction_parts.append(prediction)

        ap = aggregate.get("average_precision", math.nan)
        if np.isfinite(ap) and ap > best_ap:
            best_ap = float(ap)
            best_name = model_name

    if not best_name:
        best_name = "logistic_regression"

    final_model = build_model_specs(config)[best_name]
    final_model.fit(x_all, y_all)
    best = {"target": target, "model_name": best_name, "model": final_model, "features": features}
    predictions = pd.concat(prediction_parts, ignore_index=True, sort=False)
    return rows, predictions, best


def extract_feature_importance(model_info: dict[str, Any]) -> pd.DataFrame:
    if not model_info:
        return pd.DataFrame()
    model_name = model_info["model_name"]
    model = model_info["model"]
    features = model_info["features"]

    if model_name == "logistic_regression":
        classifier = model.named_steps["sgdclassifier"]
        values = classifier.coef_[0]
    elif model_name == "random_forest":
        classifier = model.named_steps["randomforestclassifier"]
        values = classifier.feature_importances_
    else:
        return pd.DataFrame()

    return pd.DataFrame(
        {
            "target": model_info["target"],
            "model": model_name,
            "feature": features,
            "importance": values,
            "abs_importance": np.abs(values),
        }
    ).sort_values("abs_importance", ascending=False)


def split_ids(value: Any) -> list[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    result = []
    for item in str(value).replace(",", "|").split("|"):
        item = item.strip()
        if not item:
            continue
        try:
            result.append(int(float(item)))
        except ValueError:
            continue
    return result


def division_event_geometry(df: pd.DataFrame) -> pd.DataFrame:
    required = {"sequence_uid", "spot_id", "next_ids", "x", "y", "frame"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    index = df.set_index(["sequence_uid", "spot_id"], drop=False)
    rows = []
    split_rows = df[df.get("n_next", 0).fillna(0) >= 2]
    for _, parent in split_rows.iterrows():
        child_ids = split_ids(parent.get("next_ids"))
        children = []
        for child_id in child_ids[:2]:
            key = (parent["sequence_uid"], child_id)
            if key in index.index:
                children.append(index.loc[key])
        if len(children) < 2:
            continue

        c1, c2 = children[0], children[1]
        vx = float(c2["x"] - c1["x"])
        vy = float(c2["y"] - c1["y"])
        midpoint_x = float((c1["x"] + c2["x"]) / 2.0)
        midpoint_y = float((c1["y"] + c2["y"]) / 2.0)
        rows.append(
            {
                "sequence_uid": parent["sequence_uid"],
                "sequence_name": parent.get("sequence_name"),
                "parent_spot_uid": parent.get("spot_uid"),
                "parent_cell_uid": parent.get("cell_uid"),
                "frame": int(parent["frame"]),
                "child_1_spot_id": int(c1["spot_id"]),
                "child_2_spot_id": int(c2["spot_id"]),
                "daughter_separation": float(math.sqrt(vx * vx + vy * vy)),
                "daughter_axis_angle_deg": float((math.degrees(math.atan2(vy, vx)) + 360.0) % 180.0),
                "parent_to_daughter_midpoint": float(
                    math.sqrt((midpoint_x - parent["x"]) ** 2 + (midpoint_y - parent["y"]) ** 2)
                ),
                "parent_area": float(parent.get("AREA", np.nan)),
                "parent_eccentricity": float(parent.get("ELLIPSE_ECCENTRICITY", np.nan)),
                "parent_aspect_ratio": float(parent.get("ELLIPSE_ASPECTRATIO", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def summarize_division_geometry(geometry: pd.DataFrame) -> dict[str, Any]:
    if geometry.empty:
        return {"events": 0}
    summary: dict[str, Any] = {"events": int(len(geometry))}
    for column in [
        "daughter_separation",
        "daughter_axis_angle_deg",
        "parent_to_daughter_midpoint",
        "parent_area",
        "parent_eccentricity",
        "parent_aspect_ratio",
    ]:
        values = pd.to_numeric(geometry[column], errors="coerce").dropna()
        if values.empty:
            continue
        summary[column] = {"mean": float(values.mean()), "median": float(values.median()), "std": float(values.std(ddof=0))}
    return summary


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items() if key != "model"}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_report(path: Path, summary: dict[str, Any], scores: pd.DataFrame, importance: pd.DataFrame) -> None:
    lines = [
        "# Cell division prediction model",
        "",
        "## Dataset",
        "",
        f"- Source: `{summary['config']['source']}`",
        f"- Rows: `{summary['rows']}`",
        f"- Sequences: `{summary['sequences']}`",
        f"- Cells: `{summary['cells']}`",
        f"- Division geometry events: `{summary['division_geometry']['events']}`",
        "",
        "## Model results",
        "",
    ]

    aggregate = scores[(scores["status"] == "ok") & (scores["fold"].isna())] if "fold" in scores else scores
    for _, row in aggregate.sort_values(["target", "average_precision"], ascending=[True, False]).iterrows():
        lines.extend(
            [
                f"### {row['target']} / {row['model']}",
                "",
                f"- Rows: `{int(row['rows'])}`",
                f"- Positives: `{int(row['positives'])}`",
                f"- Positive rate: `{float(row['positive_rate']):.6g}`",
                f"- ROC-AUC: `{float(row['roc_auc']):.6g}`",
                f"- Average precision: `{float(row['average_precision']):.6g}`",
                f"- Top-50 hits: `{float(row.get('top50_hits', 0.0)):.6g}`",
                f"- Top-50 recall: `{float(row.get('top50_recall', 0.0)):.6g}`",
                "",
            ]
        )

    if not importance.empty:
        lines.extend(["## Top features", ""])
        for _, row in importance.head(25).iterrows():
            lines.append(f"- `{row['target']}` / `{row['model']}`: `{row['feature']}` = `{float(row['importance']):.6g}`")
        lines.append("")

    lines.extend(
        [
            "## Notes",
            "",
            "- The model excludes known leakage columns such as `frames_to_cell_end`, `cell_lifetime_frames`, terminal split flags and future labels.",
            "- Validation uses `GroupKFold` by `sequence_uid`, so frames from the same microscopy sequence do not leak across train/test folds.",
            "- Division is a rare event here; average precision and top-k hits are more informative than accuracy.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config: DivisionPredictionConfig) -> dict[str, Any]:
    df = read_source(config.source)
    df = add_temporal_features(df, config.lags)
    features = candidate_feature_columns(df)
    config.out_dir.mkdir(parents=True, exist_ok=True)

    all_score_rows: list[dict[str, Any]] = []
    all_predictions = []
    model_bundle: dict[str, Any] = {"config": asdict(config), "models": {}}
    importance_tables = []

    for horizon in config.horizons:
        target = f"division_within_{horizon}_frames"
        eligible = f"eligible_within_{horizon}_frames"
        if target not in df or eligible not in df:
            all_score_rows.append({"target": target, "status": "skipped", "reason": "missing target or eligibility column"})
            continue

        data = df[df[eligible].fillna(False)].copy()
        data = data.dropna(subset=["sequence_uid", target])
        if data.empty or data[target].nunique(dropna=True) < 2:
            all_score_rows.append({"target": target, "status": "skipped", "reason": "not enough target classes"})
            continue

        score_rows, predictions, best = evaluate_models(data, target, features, config)
        all_score_rows.extend(score_rows)
        if not predictions.empty:
            all_predictions.append(predictions)
        if best:
            model_bundle["models"][target] = {"model_name": best["model_name"], "features": best["features"], "model": best["model"]}
            importance_tables.append(extract_feature_importance(best))

    scores = pd.DataFrame(all_score_rows)
    predictions = pd.concat(all_predictions, ignore_index=True, sort=False) if all_predictions else pd.DataFrame()
    importance = pd.concat(importance_tables, ignore_index=True, sort=False) if importance_tables else pd.DataFrame()
    geometry = division_event_geometry(df)

    scores.to_csv(config.out_dir / "division_model_scores.csv", index=False)
    predictions.to_csv(config.out_dir / "division_oof_predictions.csv", index=False)
    importance.to_csv(config.out_dir / "division_feature_importance.csv", index=False)
    geometry.to_csv(config.out_dir / "division_event_geometry.csv", index=False)
    joblib.dump(model_bundle, config.out_dir / "division_prediction_models.joblib")

    summary = {
        "config": asdict(config),
        "rows": int(len(df)),
        "sequences": int(df["sequence_uid"].nunique()) if "sequence_uid" in df else 0,
        "cells": int(df["cell_uid"].nunique()) if "cell_uid" in df else 0,
        "features": int(len(features)),
        "targets": list(model_bundle["models"].keys()),
        "division_geometry": summarize_division_geometry(geometry),
        "best_models": {
            target: {"model_name": value["model_name"], "features": len(value["features"])}
            for target, value in model_bundle["models"].items()
        },
    }
    write_json(config.out_dir / "division_prediction_summary.json", summary)
    write_report(config.out_dir / "division_prediction_report.md", summary, scores, importance)
    return summary


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train leakage-aware tabular models for future cell division prediction.")
    parser.add_argument("--source", type=Path, default=DivisionPredictionConfig().source)
    parser.add_argument("--out-dir", type=Path, default=DivisionPredictionConfig().out_dir)
    parser.add_argument("--horizons", default="3,5,10", help="Comma-separated future division horizons in frames.")
    parser.add_argument("--lags", default="1,3,5", help="Comma-separated temporal lags in frames.")
    parser.add_argument("--seed", type=int, default=DivisionPredictionConfig().seed)
    parser.add_argument("--folds", type=int, default=DivisionPredictionConfig().folds)
    parser.add_argument("--random-forest-trees", type=int, default=DivisionPredictionConfig().random_forest_trees)
    parser.add_argument(
        "--random-forest-min-samples-leaf",
        type=int,
        default=DivisionPredictionConfig().random_forest_min_samples_leaf,
    )
    parser.add_argument("--max-logistic-iter", type=int, default=DivisionPredictionConfig().max_logistic_iter)
    parser.add_argument("--models", default="logistic_regression", help="Comma-separated models: logistic_regression,random_forest.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    config = DivisionPredictionConfig(
        source=args.source,
        out_dir=args.out_dir,
        horizons=parse_int_tuple(args.horizons),
        lags=parse_int_tuple(args.lags),
        seed=args.seed,
        folds=args.folds,
        random_forest_trees=args.random_forest_trees,
        random_forest_min_samples_leaf=args.random_forest_min_samples_leaf,
        max_logistic_iter=args.max_logistic_iter,
        models=tuple(item.strip() for item in args.models.split(",") if item.strip()),
    )
    summary = run(config)
    print(f"Saved division prediction model to {config.out_dir}")
    print(json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
