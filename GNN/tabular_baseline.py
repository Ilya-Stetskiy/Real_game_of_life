from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .dataset_cache import DEFAULT_CACHE_PATH, load_graph_cache
from .train_one_step import binary_classification_metrics, jsonable


@dataclass(frozen=True)
class TabularBaselineConfig:
    cache_path: Path = DEFAULT_CACHE_PATH
    out_dir: Path = Path(__file__).resolve().parent / "runs" / "tabular_baseline"
    target: str = "division_h10"
    seed: int = 17
    max_iter: int = 1000
    random_forest_trees: int = 300
    random_forest_min_samples_leaf: int = 10


def run_tabular_baseline(config: TabularBaselineConfig) -> dict[str, Any]:
    cache = load_graph_cache(config.cache_path)
    graphs = cache["graphs"]
    splits = cache["splits"]
    target_attr, mask_attr = target_to_attrs(config.target)

    split_arrays = {
        name: graph_split_to_arrays(graphs, indices, target_attr, mask_attr)
        for name, indices in splits.items()
    }
    train = split_arrays["train"]
    val = split_arrays["val"]
    test = split_arrays["test"]
    if train["x"].shape[0] == 0:
        raise ValueError(f"Train split has no valid examples for target={config.target!r}.")

    models = {
        "dummy_prior": fit_dummy_prior(train["y"]),
        "logistic_regression": fit_logistic_regression(train["x"], train["y"], config),
        "random_forest": fit_random_forest(train["x"], train["y"], config),
    }

    results: dict[str, Any] = {
        "config": jsonable(asdict(config)),
        "target": config.target,
        "target_attr": target_attr,
        "mask_attr": mask_attr,
        "split_counts": split_counts(split_arrays),
        "cache_summary": cache.get("summary", {}),
        "models": {},
    }
    for model_name, model in models.items():
        model_result = {}
        for split_name, arrays in split_arrays.items():
            scores = predict_scores(model, arrays["x"])
            model_result[split_name] = metrics_from_numpy(model_name, scores, arrays["y"])
        results["models"][model_name] = model_result

    config.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.out_dir / "tabular_baseline_summary.json", results)
    write_report(config.out_dir / "tabular_baseline_report.md", results)
    return results


def target_to_attrs(target: str) -> tuple[str, str]:
    normalized = target.strip().lower()
    if normalized == "division":
        return "target_division", "valid_event_mask"
    if normalized == "death":
        return "target_death", "valid_event_mask"
    if normalized.startswith("division_h"):
        horizon = int(normalized.replace("division_h", "", 1))
        return f"target_division_within_{horizon}", f"valid_division_within_{horizon}"
    raise ValueError("target must be one of: division, death, division_h3, division_h5, division_h10")


def graph_split_to_arrays(graphs: list[Any], indices: list[int], target_attr: str, mask_attr: str) -> dict[str, np.ndarray]:
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for index in indices:
        graph = graphs[index]
        if not hasattr(graph, target_attr) or not hasattr(graph, mask_attr):
            continue
        mask = getattr(graph, mask_attr).detach().cpu().bool().numpy()
        if not mask.any():
            continue
        x_parts.append(graph.x.detach().cpu().float().numpy()[mask])
        y_parts.append(getattr(graph, target_attr).detach().cpu().float().numpy()[mask])
    if not x_parts:
        node_dim = int(graphs[0].x.size(-1)) if graphs else 0
        return {"x": np.zeros((0, node_dim), dtype=np.float32), "y": np.zeros((0,), dtype=np.float32)}
    return {
        "x": np.concatenate(x_parts, axis=0).astype(np.float32, copy=False),
        "y": np.concatenate(y_parts, axis=0).astype(np.float32, copy=False),
    }


def split_counts(split_arrays: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, int]]:
    return {
        name: {
            "examples": int(arrays["y"].shape[0]),
            "positives": int(arrays["y"].sum()),
            "negatives": int(arrays["y"].shape[0] - arrays["y"].sum()),
        }
        for name, arrays in split_arrays.items()
    }


def fit_dummy_prior(y: np.ndarray) -> dict[str, Any]:
    positive_rate = float(y.mean()) if y.size else 0.0
    return {"kind": "dummy_prior", "positive_rate": positive_rate}


def fit_logistic_regression(x: np.ndarray, y: np.ndarray, config: TabularBaselineConfig) -> Any:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if int(y.sum()) == 0 or int(y.sum()) == y.size:
        return fit_dummy_prior(y)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=config.max_iter,
            class_weight="balanced",
            random_state=config.seed,
            solver="lbfgs",
        ),
    )
    return model.fit(x, y.astype(int))


def fit_random_forest(x: np.ndarray, y: np.ndarray, config: TabularBaselineConfig) -> Any:
    from sklearn.ensemble import RandomForestClassifier

    if int(y.sum()) == 0 or int(y.sum()) == y.size:
        return fit_dummy_prior(y)
    model = RandomForestClassifier(
        n_estimators=config.random_forest_trees,
        min_samples_leaf=config.random_forest_min_samples_leaf,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=config.seed,
    )
    return model.fit(x, y.astype(int))


def predict_scores(model: Any, x: np.ndarray) -> np.ndarray:
    if isinstance(model, dict) and model.get("kind") == "dummy_prior":
        return np.full((x.shape[0],), float(model["positive_rate"]), dtype=np.float32)
    if x.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    proba = model.predict_proba(x)
    if proba.shape[1] == 1:
        return np.zeros((x.shape[0],), dtype=np.float32)
    return proba[:, 1].astype(np.float32, copy=False)


def metrics_from_numpy(prefix: str, score: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return binary_classification_metrics(
        prefix,
        torch.as_tensor(score, dtype=torch.float32),
        torch.as_tensor(target, dtype=torch.bool),
    )


def write_report(path: Path, results: dict[str, Any]) -> None:
    lines = [
        "# Tabular baseline",
        "",
        f"- Target: `{results['target']}`",
        f"- Target attr: `{results['target_attr']}`",
        f"- Mask attr: `{results['mask_attr']}`",
        f"- Split counts: `{results['split_counts']}`",
        "",
        "## Test metrics",
        "",
    ]
    for model_name, model_results in results["models"].items():
        metrics = model_results.get("test", {})
        lines.extend(
            [
                f"### {model_name}",
                "",
                metric_line(f"{model_name}_ap", metrics),
                metric_line(f"{model_name}_precision", metrics),
                metric_line(f"{model_name}_recall", metrics),
                metric_line(f"{model_name}_f1", metrics),
                metric_line(f"{model_name}_top10_hits", metrics),
                metric_line(f"{model_name}_top10_recall", metrics),
                metric_line(f"{model_name}_top50_hits", metrics),
                metric_line(f"{model_name}_top50_recall", metrics),
                metric_line(f"{model_name}_top100_hits", metrics),
                metric_line(f"{model_name}_top100_recall", metrics),
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def metric_line(name: str, metrics: dict[str, Any]) -> str:
    value = metrics.get(name)
    if value is None:
        text = "n/a"
    else:
        text = f"{float(value):.6g}"
    return f"- `{name}`: `{text}`"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tabular baselines on graph-cache node features.")
    parser.add_argument("--cache", type=Path, default=TabularBaselineConfig().cache_path)
    parser.add_argument("--out-dir", type=Path, default=TabularBaselineConfig().out_dir)
    parser.add_argument("--target", default=TabularBaselineConfig().target)
    parser.add_argument("--seed", type=int, default=TabularBaselineConfig().seed)
    parser.add_argument("--max-iter", type=int, default=TabularBaselineConfig().max_iter)
    parser.add_argument("--random-forest-trees", type=int, default=TabularBaselineConfig().random_forest_trees)
    parser.add_argument(
        "--random-forest-min-samples-leaf",
        type=int,
        default=TabularBaselineConfig().random_forest_min_samples_leaf,
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_tabular_baseline(
        TabularBaselineConfig(
            cache_path=args.cache,
            out_dir=args.out_dir,
            target=args.target,
            seed=args.seed,
            max_iter=args.max_iter,
            random_forest_trees=args.random_forest_trees,
            random_forest_min_samples_leaf=args.random_forest_min_samples_leaf,
        )
    )
    print(json.dumps(jsonable(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
