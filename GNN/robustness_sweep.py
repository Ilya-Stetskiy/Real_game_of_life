from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from .dataset_cache import SplitConfig, build_and_save_graph_cache
from .evaluate_rollout import _build_model_for_checkpoint, normalization_tensors
from .graph_dataset import DEFAULT_PROCESSED_SPOTS, FrameGraphDatasetConfig, load_processed_spots
from .train_one_step import TrainConfig, apply_node_feature_normalization, jsonable, resolve_device, train_from_cache


METRIC_KEYS = (
    "pos_rmse",
    "polarization_theta_rmse",
    "polarization_aspect_rmse",
    "shape_rmse",
    "division_ap",
    "death_ap",
    "division_h10_ap",
)


@dataclass(frozen=True)
class RobustnessSweepConfig:
    out_dir: Path = Path(__file__).resolve().parent / "runs" / "robustness"
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
    split_modes: tuple[str, ...] = ("by_position_event_balanced", "by_sequence")
    epochs: int = 20
    batch_size: int = 16
    hidden_dim: int = 128
    layers: int = 4
    dropout: float = 0.1
    learning_rate: float = 1e-3
    device: str = "auto"
    max_graphs: int | None = None
    keep_caches: bool = False
    per_sequence_diagnostic: bool = True


def run_robustness_sweep(
    config: RobustnessSweepConfig,
    *,
    dataset_config: FrameGraphDatasetConfig | None = None,
) -> dict[str, Any]:
    """Train the one-step GNN across multiple seeds/split-modes and aggregate test metrics.

    Answers task 1: are pos/polarization/division metrics stable across the choice of
    seed and split, or an artifact of one particular split? Each (split_mode, seed)
    combination gets its own cache (built via dataset_cache.build_and_save_graph_cache,
    same as the CLI) and its own train_one_step.train_from_cache run; test-split metrics
    are collected into rows and aggregated with mean/std/median/IQR/min/max.
    """

    dataset_config = dataset_config or FrameGraphDatasetConfig()
    spots = load_processed_spots(dataset_config.source_path)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = config.out_dir / "caches"
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    first_run: dict[str, Any] | None = None
    for split_mode in config.split_modes:
        for seed in config.seeds:
            cache_path = cache_dir / f"cache_{split_mode}_seed{seed}.pt"
            build_and_save_graph_cache(
                cache_path,
                spots=spots,
                dataset_config=dataset_config,
                split_config=SplitConfig(mode=split_mode, seed=seed),
                max_graphs=config.max_graphs,
            )
            run_out_dir = config.out_dir / "runs" / f"{split_mode}_seed{seed}"
            result = train_from_cache(
                TrainConfig(
                    cache_path=cache_path,
                    out_dir=run_out_dir,
                    epochs=config.epochs,
                    batch_size=config.batch_size,
                    hidden_dim=config.hidden_dim,
                    layers=config.layers,
                    dropout=config.dropout,
                    learning_rate=config.learning_rate,
                    seed=seed,
                    device=config.device,
                )
            )
            test_metrics = result["summary"].get("test_metrics") or {}
            row = {
                "split_mode": split_mode,
                "seed": seed,
                "train_graphs": result["summary"]["train_graphs"],
                "val_graphs": result["summary"]["val_graphs"],
                "test_graphs": result["summary"]["test_graphs"],
                **{key: test_metrics.get(key) for key in METRIC_KEYS},
            }
            rows.append(row)
            if first_run is None:
                first_run = {"cache_path": cache_path, "checkpoint_path": run_out_dir / "best.pt"}
            if not config.keep_caches:
                cache_path.unlink(missing_ok=True)
                cache_path.with_suffix(cache_path.suffix + ".summary.json").unlink(missing_ok=True)

    per_sequence_rows: list[dict[str, Any]] = []
    if config.per_sequence_diagnostic and first_run is not None and config.keep_caches:
        per_sequence_rows = per_sequence_position_rmse(
            cache_path=first_run["cache_path"],
            checkpoint_path=first_run["checkpoint_path"],
            device_name=config.device,
        )

    results = {
        "config": jsonable(asdict(config)),
        "rows": rows,
        "aggregates_by_split_mode": aggregate_rows(rows, group_keys=("split_mode",), metric_keys=METRIC_KEYS),
        "aggregates_overall": aggregate_rows(rows, group_keys=(), metric_keys=METRIC_KEYS),
        "per_sequence_diagnostic": per_sequence_rows,
    }
    write_json(config.out_dir / "robustness_summary.json", results)
    write_report(config.out_dir / "robustness_report.md", results)
    return results


def aggregate_rows(
    rows: Sequence[dict[str, Any]],
    *,
    group_keys: Sequence[str],
    metric_keys: Sequence[str],
) -> list[dict[str, Any]]:
    """Group rows by group_keys and summarize each metric_key with mean/std/median/IQR/min/max/n."""

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[group_key] for group_key in group_keys)
        groups.setdefault(key, []).append(row)

    summaries: list[dict[str, Any]] = []
    for key, group_rows in groups.items():
        summary: dict[str, Any] = dict(zip(group_keys, key))
        summary["n_runs"] = len(group_rows)
        for metric_key in metric_keys:
            values = [row[metric_key] for row in group_rows if row.get(metric_key) is not None]
            summary[metric_key] = summarize_values(values)
        summaries.append(summary)
    return summaries


def summarize_values(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"n": 0, "mean": None, "std": None, "median": None, "iqr": None, "min": None, "max": None}
    ordered = sorted(values)
    quantiles = statistics.quantiles(ordered, n=4, method="inclusive") if len(ordered) >= 2 else [ordered[0]] * 3
    return {
        "n": len(ordered),
        "mean": statistics.fmean(ordered),
        "std": statistics.pstdev(ordered) if len(ordered) > 1 else 0.0,
        "median": statistics.median(ordered),
        "iqr": float(quantiles[2] - quantiles[0]),
        "min": ordered[0],
        "max": ordered[-1],
    }


def per_sequence_position_rmse(
    *,
    cache_path: Path,
    checkpoint_path: Path,
    device_name: str,
) -> list[dict[str, Any]]:
    """Diagnostic: pos_rmse on the test split broken down by sequence_uid (one video).

    Answers "does one or two videos dominate the aggregate error?" for one representative
    sweep run, without paying for a full leave-one-sequence-out sweep over all 57 videos.
    """

    from .dataset_cache import load_graph_cache

    if not checkpoint_path.exists():
        return []
    cache = load_graph_cache(cache_path)
    device = resolve_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    test_graphs = [cache["graphs"][index] for index in cache["splits"].get("test", [])]
    if not test_graphs:
        return []

    normalization = normalization_tensors(checkpoint.get("node_feature_normalization"))
    graphs = [graph.clone() for graph in test_graphs]
    if normalization is not None:
        apply_node_feature_normalization(graphs, normalization)
    model = _build_model_for_checkpoint(checkpoint, graphs).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    sums: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for graph in graphs:
            graph = graph.to(device)
            output = model(graph)
            valid = graph.valid_regression_mask.bool()
            if int(valid.sum()) == 0:
                continue
            error = output.delta_pos[valid] - graph.target_delta_pos[valid]
            sq_error = float((error ** 2).sum().item())
            count = int(error.numel())
            sequence_uid = str(getattr(graph, "sequence_uid", "unknown"))
            bucket = sums.setdefault(sequence_uid, {"sq_error": 0.0, "count": 0, "nodes": 0})
            bucket["sq_error"] += sq_error
            bucket["count"] += count
            bucket["nodes"] += int(valid.sum().item())

    rows = [
        {
            "sequence_uid": sequence_uid,
            "matched_nodes": bucket["nodes"],
            "pos_rmse": (bucket["sq_error"] / bucket["count"]) ** 0.5 if bucket["count"] else None,
        }
        for sequence_uid, bucket in sums.items()
    ]
    rows.sort(key=lambda row: (row["pos_rmse"] is None, -(row["pos_rmse"] or 0.0)))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_report(path: Path, results: dict[str, Any]) -> None:
    lines = ["# Robustness sweep: seeds x split-modes", ""]
    lines.append(f"Total runs: {len(results['rows'])}")
    lines.append("")
    lines.append("## Overall (across all split modes and seeds)")
    lines.append("")
    lines.extend(_aggregate_table(results["aggregates_overall"], group_keys=()))
    lines.append("")
    lines.append("## By split mode")
    lines.append("")
    lines.extend(_aggregate_table(results["aggregates_by_split_mode"], group_keys=("split_mode",)))
    if results.get("per_sequence_diagnostic"):
        lines.append("")
        lines.append("## Per-sequence pos_rmse on one representative run's test split")
        lines.append("")
        lines.append("| sequence_uid | matched_nodes | pos_rmse |")
        lines.append("|---|---|---|")
        for row in results["per_sequence_diagnostic"]:
            pos_rmse = row["pos_rmse"]
            pos_rmse_text = f"{pos_rmse:.4f}" if pos_rmse is not None else "n/a"
            lines.append(f"| {row['sequence_uid']} | {row['matched_nodes']} | {pos_rmse_text} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _aggregate_table(summaries: list[dict[str, Any]], *, group_keys: Sequence[str]) -> list[str]:
    lines: list[str] = []
    for summary in summaries:
        header = ", ".join(f"{key}={summary[key]}" for key in group_keys) if group_keys else "all runs"
        lines.append(f"### {header} (n_runs={summary['n_runs']})")
        lines.append("")
        lines.append("| metric | mean | std | median | iqr | min | max |")
        lines.append("|---|---|---|---|---|---|---|")
        for metric_key in METRIC_KEYS:
            stats = summary.get(metric_key) or {}
            if not stats.get("n"):
                continue
            lines.append(
                f"| {metric_key} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['median']:.4f} | "
                f"{stats['iqr']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} |"
            )
        lines.append("")
    return lines


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep train_one_step across seeds and split-modes to assess robustness.")
    parser.add_argument("--out-dir", type=Path, default=RobustnessSweepConfig().out_dir)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(RobustnessSweepConfig().seeds))
    parser.add_argument(
        "--split-modes",
        nargs="+",
        default=list(RobustnessSweepConfig().split_modes),
        choices=("by_position", "by_position_event_balanced", "by_sequence", "none"),
    )
    parser.add_argument("--epochs", type=int, default=RobustnessSweepConfig().epochs)
    parser.add_argument("--batch-size", type=int, default=RobustnessSweepConfig().batch_size)
    parser.add_argument("--hidden-dim", type=int, default=RobustnessSweepConfig().hidden_dim)
    parser.add_argument("--layers", type=int, default=RobustnessSweepConfig().layers)
    parser.add_argument("--dropout", type=float, default=RobustnessSweepConfig().dropout)
    parser.add_argument("--learning-rate", type=float, default=RobustnessSweepConfig().learning_rate)
    parser.add_argument("--device", default=RobustnessSweepConfig().device)
    parser.add_argument("--max-graphs", type=int, default=None, help="Optional smoke-test limit per cache.")
    parser.add_argument("--keep-caches", action="store_true", help="Keep per-run caches (needed for --per-sequence-diagnostic).")
    parser.add_argument("--no-per-sequence-diagnostic", dest="per_sequence_diagnostic", action="store_false", default=True)
    parser.add_argument("--source", type=Path, default=DEFAULT_PROCESSED_SPOTS)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    config = RobustnessSweepConfig(
        out_dir=args.out_dir,
        seeds=tuple(args.seeds),
        split_modes=tuple(args.split_modes),
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        device=args.device,
        max_graphs=args.max_graphs,
        keep_caches=args.keep_caches,
        per_sequence_diagnostic=args.per_sequence_diagnostic,
    )
    dataset_config = FrameGraphDatasetConfig(source_path=args.source)
    results = run_robustness_sweep(config, dataset_config=dataset_config)
    print(json.dumps(jsonable(results["aggregates_overall"]), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
