from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import torch

from .dataset_cache import DEFAULT_CACHE_PATH, SplitConfig, build_and_save_graph_cache, load_graph_cache
from .graph_dataset import DEFAULT_PROCESSED_SPOTS, FrameGraphDatasetConfig
from .train_one_step import TrainConfig, jsonable, train_from_cache


PRESET_OVERRIDES: dict[str, dict[str, Any]] = {
    "smoke": {
        "epochs": 2,
        "batch_size": 16,
        "hidden_dim": 32,
        "layers": 1,
        "dropout": 0.0,
        "learning_rate": 1e-3,
        "checkpoint_every": 0,
        "early_stopping_patience": 0,
        "num_workers": 0,
        "amp": False,
    },
    "baseline": {
        "epochs": 80,
        "batch_size": 32,
        "hidden_dim": 128,
        "layers": 4,
        "dropout": 0.1,
        "learning_rate": 1e-3,
        "checkpoint_every": 10,
        "early_stopping_patience": 25,
        "num_workers": 2,
        "amp": False,
    },
    "server": {
        "epochs": 300,
        "batch_size": 64,
        "hidden_dim": 256,
        "layers": 5,
        "dropout": 0.15,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "grad_clip_norm": 1.0,
        "lambda_pos": 1.0,
        "lambda_shape": 0.25,
        "lambda_division": 2.0,
        "lambda_death": 0.5,
        "max_pos_weight": 100.0,
        "scheduler_patience": 20,
        "scheduler_factor": 0.5,
        "min_learning_rate": 1e-6,
        "early_stopping_patience": 60,
        "min_delta": 1e-5,
        "checkpoint_every": 25,
        "num_workers": 4,
        "amp": True,
    },
    "large": {
        "epochs": 800,
        "batch_size": 96,
        "hidden_dim": 384,
        "layers": 6,
        "dropout": 0.15,
        "learning_rate": 2e-4,
        "weight_decay": 1e-4,
        "grad_clip_norm": 1.0,
        "lambda_pos": 1.0,
        "lambda_shape": 0.25,
        "lambda_division": 2.0,
        "lambda_death": 0.5,
        "max_pos_weight": 100.0,
        "scheduler_patience": 30,
        "scheduler_factor": 0.5,
        "min_learning_rate": 5e-7,
        "early_stopping_patience": 120,
        "min_delta": 1e-5,
        "checkpoint_every": 25,
        "num_workers": 6,
        "amp": True,
    },
}


def run_full_training(
    *,
    preset: str = "server",
    cache_path: str | Path = DEFAULT_CACHE_PATH,
    out_dir: str | Path | None = None,
    source_path: str | Path | None = None,
    rebuild_cache: bool = False,
    build_cache_if_missing: bool = True,
    max_graphs: int | None = None,
    edge_radius: float = 40.0,
    split_mode: str = "by_position",
    seed: int = 17,
    train_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if preset not in PRESET_OVERRIDES:
        raise ValueError(f"Unknown preset: {preset!r}. Available presets: {sorted(PRESET_OVERRIDES)}")

    cache_path = Path(cache_path)
    out_dir = Path(out_dir) if out_dir is not None else default_run_dir(preset)
    out_dir.mkdir(parents=True, exist_ok=True)

    environment = collect_environment()
    write_json(out_dir / "environment.json", environment)

    cache_exists = cache_path.exists()
    if rebuild_cache or (build_cache_if_missing and not cache_exists):
        dataset_config = FrameGraphDatasetConfig(
            source_path=Path(source_path) if source_path is not None else DEFAULT_PROCESSED_SPOTS,
            edge_radius=edge_radius,
        )
        split_config = SplitConfig(mode=split_mode, seed=seed)  # type: ignore[arg-type]
        cache = build_and_save_graph_cache(
            cache_path,
            dataset_config=dataset_config,
            split_config=split_config,
            max_graphs=max_graphs,
        )
        cache_action = "rebuilt" if rebuild_cache else "built"
    elif cache_exists:
        cache = load_graph_cache(cache_path)
        cache_action = "loaded"
    else:
        raise FileNotFoundError(
            f"Cache does not exist: {cache_path}. Use build_cache_if_missing=True or --build-cache-if-missing."
        )

    cache_summary = cache.get("summary", {})
    write_json(out_dir / "cache_summary.json", cache_summary)
    del cache

    train_config = make_train_config(
        preset=preset,
        cache_path=cache_path,
        out_dir=out_dir,
        seed=seed,
        overrides=train_overrides or {},
    )
    effective_config = {
        "preset": preset,
        "cache_action": cache_action,
        "cache_path": str(cache_path),
        "source_path": str(source_path or DEFAULT_PROCESSED_SPOTS),
        "rebuild_cache": rebuild_cache,
        "build_cache_if_missing": build_cache_if_missing,
        "max_graphs": max_graphs,
        "edge_radius": edge_radius,
        "split_mode": split_mode,
        "seed": seed,
        "train_config": jsonable(asdict(train_config)),
    }
    write_json(out_dir / "effective_config.json", effective_config)

    start = time.perf_counter()
    result = train_from_cache(train_config)
    wall_seconds = time.perf_counter() - start

    full_summary = {
        "preset": preset,
        "wall_seconds": wall_seconds,
        "environment": environment,
        "cache_action": cache_action,
        "cache_summary": cache_summary,
        "training_summary": result["summary"],
    }
    write_json(out_dir / "full_run_summary.json", full_summary)
    write_final_report(out_dir / "final_report.md", full_summary)
    return full_summary


def make_train_config(
    *,
    preset: str,
    cache_path: Path,
    out_dir: Path,
    seed: int,
    overrides: dict[str, Any],
) -> TrainConfig:
    config = TrainConfig(cache_path=cache_path, out_dir=out_dir, seed=seed)
    values = dict(PRESET_OVERRIDES[preset])
    values.update({key: value for key, value in overrides.items() if value is not None})
    for key, value in values.items():
        config = replace(config, **{key: value})
    return config


def collect_environment() -> dict[str, Any]:
    try:
        import torch_geometric

        torch_geometric_version = torch_geometric.__version__
    except Exception:
        torch_geometric_version = None

    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 3),
                }
            )

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_geometric_version": torch_geometric_version,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": cuda_devices,
    }


def write_final_report(path: Path, summary: dict[str, Any]) -> None:
    train = summary["training_summary"]
    test = train.get("test_metrics") or {}
    cache = summary.get("cache_summary") or {}
    env = summary.get("environment") or {}

    lines = [
        "# One-step GNN full run",
        "",
        f"- Preset: `{summary['preset']}`",
        f"- Wall time: `{format_seconds(float(summary['wall_seconds']))}`",
        f"- Best epoch: `{train.get('best_epoch')}`",
        f"- Best validation loss: `{format_float(train.get('best_metric'))}`",
        f"- Epochs completed: `{train.get('epochs_completed')}`",
        f"- Device CUDA available: `{env.get('cuda_available')}`",
        "",
        "## Data",
        "",
        f"- Graphs: `{cache.get('graphs')}`",
        f"- Nodes: `{cache.get('nodes')}`",
        f"- Edges: `{cache.get('edges')}`",
        f"- Splits: `{cache.get('splits')}`",
        f"- Node dim: `{train.get('node_dim')}`",
        f"- Edge dim: `{train.get('edge_dim')}`",
        "",
        "## Test metrics",
        "",
        metric_line("loss_total", test),
        metric_line("pos_rmse", test),
        metric_line("shape_rmse", test),
        metric_line("division_precision", test),
        metric_line("division_recall", test),
        metric_line("division_f1", test),
        metric_line("division_ap", test),
        metric_line("division_tp", test),
        metric_line("division_fp", test),
        metric_line("division_fn", test),
        metric_line("death_precision", test),
        metric_line("death_recall", test),
        metric_line("death_f1", test),
        metric_line("death_ap", test),
        metric_line("death_tp", test),
        metric_line("death_fp", test),
        metric_line("death_fn", test),
        "",
        "## Artifacts",
        "",
        "- `best.pt` - checkpoint selected by validation loss",
        "- `last.pt` - last checkpoint",
        "- `history.csv` and `history.json` - epoch metrics",
        "- `run_summary.json` - training summary",
        "- `full_run_summary.json` - full-cycle summary with environment and cache info",
        "- `environment.json` - Python, PyTorch, CUDA and PyG versions",
        "",
        "Note: one-step division is a rare-event target. Use precision, recall and AP before accuracy.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def metric_line(name: str, metrics: dict[str, Any]) -> str:
    return f"- `{name}`: `{format_float(metrics.get(name))}`"


def format_float(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def format_seconds(value: float) -> str:
    if value < 60:
        return f"{value:.1f}s"
    if value < 3600:
        return f"{value / 60:.1f}m"
    return f"{value / 3600:.2f}h"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def default_run_dir(preset: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(__file__).resolve().parent / "runs" / f"one_step_{preset}_{timestamp}"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full one-step GNN training cycle.")
    parser.add_argument("--preset", choices=sorted(PRESET_OVERRIDES), default="server")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--source", type=Path, default=DEFAULT_PROCESSED_SPOTS)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--build-cache-if-missing", dest="build_cache_if_missing", action="store_true", default=True)
    parser.add_argument("--no-build-cache-if-missing", dest="build_cache_if_missing", action="store_false")
    parser.add_argument("--max-graphs", type=int, default=None)
    parser.add_argument("--edge-radius", type=float, default=40.0)
    parser.add_argument("--split-mode", choices=["by_position", "by_sequence", "none"], default="by_position")
    parser.add_argument("--seed", type=int, default=17)

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--lambda-pos", type=float, default=None)
    parser.add_argument("--lambda-shape", type=float, default=None)
    parser.add_argument("--lambda-division", type=float, default=None)
    parser.add_argument("--lambda-death", type=float, default=None)
    parser.add_argument("--max-pos-weight", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--scheduler-patience", type=int, default=None)
    parser.add_argument("--scheduler-factor", type=float, default=None)
    parser.add_argument("--min-learning-rate", type=float, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--min-delta", type=float, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true", default=None)
    amp_group.add_argument("--no-amp", dest="amp", action="store_false")
    return parser.parse_args(argv)


def train_overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "layers": args.layers,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "grad_clip_norm": args.grad_clip_norm,
        "device": args.device,
        "lambda_pos": args.lambda_pos,
        "lambda_shape": args.lambda_shape,
        "lambda_division": args.lambda_division,
        "lambda_death": args.lambda_death,
        "max_pos_weight": args.max_pos_weight,
        "num_workers": args.num_workers,
        "scheduler_patience": args.scheduler_patience,
        "scheduler_factor": args.scheduler_factor,
        "min_learning_rate": args.min_learning_rate,
        "early_stopping_patience": args.early_stopping_patience,
        "min_delta": args.min_delta,
        "checkpoint_every": args.checkpoint_every,
        "resume_from": args.resume_from,
        "amp": args.amp,
    }


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_full_training(
        preset=args.preset,
        cache_path=args.cache,
        out_dir=args.out_dir,
        source_path=args.source,
        rebuild_cache=args.rebuild_cache,
        build_cache_if_missing=args.build_cache_if_missing,
        max_graphs=args.max_graphs,
        edge_radius=args.edge_radius,
        split_mode=args.split_mode,
        seed=args.seed,
        train_overrides=train_overrides_from_args(args),
    )
    print(json.dumps(jsonable(summary["training_summary"]), ensure_ascii=False, indent=2))
    print(f"full_run_dir={summary['training_summary']['config']['out_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
