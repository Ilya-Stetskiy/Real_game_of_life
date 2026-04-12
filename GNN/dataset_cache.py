from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np

from .graph_dataset import (
    FrameGraphDatasetConfig,
    build_frame_graphs,
    cell_graph_to_pyg_training_data,
    default_node_feature_columns,
    load_processed_spots,
)


CACHE_VERSION = 1
DEFAULT_CACHE_PATH = Path(__file__).resolve().parent / "cache" / "frame_graphs_dynamic.pt"
SplitMode = Literal["by_position", "by_sequence", "none"]


@dataclass(frozen=True)
class SplitConfig:
    mode: SplitMode = "by_position"
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 17


def build_graph_cache(
    *,
    source_path: str | Path | None = None,
    spots: Any | None = None,
    dataset_config: FrameGraphDatasetConfig | None = None,
    split_config: SplitConfig | None = None,
    max_graphs: int | None = None,
) -> dict[str, Any]:
    """Build a serializable PyG graph cache from processed HeLa spot data."""

    cfg = dataset_config or FrameGraphDatasetConfig()
    if source_path is not None:
        cfg = replace(cfg, source_path=Path(source_path))

    spot_table = load_processed_spots(cfg.source_path) if spots is None else spots
    node_features = default_node_feature_columns(spot_table, cfg)
    cfg = replace(cfg, node_feature_columns=node_features)

    cell_graphs = build_frame_graphs(spot_table, cfg)
    if max_graphs is not None:
        cell_graphs = cell_graphs[: int(max_graphs)]
    graphs = [cell_graph_to_pyg_training_data(graph, cfg) for graph in cell_graphs]
    splits = build_splits(graphs, split_config or SplitConfig())
    summary = summarize_graphs(graphs, cfg, splits)

    return {
        "version": CACHE_VERSION,
        "graphs": graphs,
        "splits": splits,
        "summary": summary,
        "dataset_config": _jsonable_dataclass(cfg),
        "split_config": _jsonable_dataclass(split_config or SplitConfig()),
    }


def save_graph_cache(cache: dict[str, Any], output_path: str | Path) -> Path:
    """Save cache with torch.save and a sidecar summary JSON."""

    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on local env.
        raise ImportError("torch is required to save graph caches.") from exc

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)
    summary_path = path.with_suffix(path.suffix + ".summary.json")
    summary_path.write_text(
        json.dumps(_to_jsonable(cache["summary"]), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def load_graph_cache(path: str | Path) -> dict[str, Any]:
    """Load a graph cache saved by save_graph_cache."""

    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on local env.
        raise ImportError("torch is required to load graph caches.") from exc

    return torch.load(Path(path), map_location="cpu", weights_only=False)


def build_and_save_graph_cache(
    output_path: str | Path = DEFAULT_CACHE_PATH,
    *,
    source_path: str | Path | None = None,
    spots: Any | None = None,
    dataset_config: FrameGraphDatasetConfig | None = None,
    split_config: SplitConfig | None = None,
    max_graphs: int | None = None,
) -> dict[str, Any]:
    cache = build_graph_cache(
        source_path=source_path,
        spots=spots,
        dataset_config=dataset_config,
        split_config=split_config,
        max_graphs=max_graphs,
    )
    save_graph_cache(cache, output_path)
    return cache


def build_splits(graphs: Sequence[Any], split_config: SplitConfig | None = None) -> dict[str, list[int]]:
    cfg = split_config or SplitConfig()
    indices_by_group: dict[str, list[int]] = {}
    for index, graph in enumerate(graphs):
        group = graph_group_key(graph, cfg.mode)
        indices_by_group.setdefault(group, []).append(index)

    if cfg.mode == "none":
        return {"train": list(range(len(graphs))), "val": [], "test": []}

    group_keys = sorted(indices_by_group)
    rng = np.random.default_rng(cfg.seed)
    if group_keys:
        group_keys = list(rng.permutation(group_keys))

    train_groups, val_groups, test_groups = split_group_keys(
        group_keys,
        train_fraction=cfg.train_fraction,
        val_fraction=cfg.val_fraction,
        test_fraction=cfg.test_fraction,
    )
    return {
        "train": _indices_for_groups(indices_by_group, train_groups),
        "val": _indices_for_groups(indices_by_group, val_groups),
        "test": _indices_for_groups(indices_by_group, test_groups),
    }


def split_group_keys(
    group_keys: Sequence[str],
    *,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> tuple[list[str], list[str], list[str]]:
    total_fraction = train_fraction + val_fraction + test_fraction
    if total_fraction <= 0:
        raise ValueError("Split fractions must sum to a positive value.")
    train_fraction /= total_fraction
    val_fraction /= total_fraction

    n_groups = len(group_keys)
    if n_groups == 0:
        return [], [], []
    if n_groups == 1:
        return list(group_keys), [], []
    if n_groups == 2:
        return [group_keys[0]], [], [group_keys[1]]

    n_train = max(1, int(round(n_groups * train_fraction)))
    n_val = max(1, int(round(n_groups * val_fraction)))
    if n_train + n_val >= n_groups:
        n_train = max(1, n_groups - 2)
        n_val = 1

    train_groups = list(group_keys[:n_train])
    val_groups = list(group_keys[n_train : n_train + n_val])
    test_groups = list(group_keys[n_train + n_val :])
    if not test_groups:
        test_groups = [val_groups.pop()]
    return train_groups, val_groups, test_groups


def graph_group_key(graph: Any, mode: SplitMode) -> str:
    sequence_uid = str(getattr(graph, "sequence_uid", "unknown"))
    if mode == "by_sequence":
        return sequence_uid
    if mode == "none":
        return "__all__"
    match = re.search(r"pos(\d+)", sequence_uid)
    if match:
        return f"pos{match.group(1)}"
    return sequence_uid


def summarize_graphs(
    graphs: Sequence[Any],
    cfg: FrameGraphDatasetConfig,
    splits: dict[str, list[int]],
) -> dict[str, Any]:
    node_counts = [int(graph.num_nodes) for graph in graphs]
    edge_counts = [int(graph.num_edges) for graph in graphs]
    sequences = sorted({str(getattr(graph, "sequence_uid", "unknown")) for graph in graphs})
    groups = sorted({graph_group_key(graph, "by_position") for graph in graphs})

    summary: dict[str, Any] = {
        "cache_version": CACHE_VERSION,
        "graphs": len(graphs),
        "nodes": int(sum(node_counts)),
        "edges": int(sum(edge_counts)),
        "sequences": len(sequences),
        "split_groups": len(groups),
        "node_dim": int(graphs[0].x.size(-1)) if graphs else 0,
        "edge_dim": int(graphs[0].edge_attr.size(-1)) if graphs else 0,
        "node_features": list(cfg.node_feature_columns or ()),
        "edge_features": list(cfg.edge_feature_columns),
        "splits": {name: len(indices) for name, indices in splits.items()},
        "target_valid_regression": _sum_graph_attr(graphs, "valid_regression_mask"),
        "target_division": _sum_graph_attr(graphs, "target_division"),
        "target_death": _sum_graph_attr(graphs, "target_death"),
    }
    for horizon in cfg.horizons:
        summary[f"target_division_within_{horizon}"] = _sum_graph_attr(graphs, f"target_division_within_{horizon}")
        summary[f"valid_division_within_{horizon}"] = _sum_graph_attr(graphs, f"valid_division_within_{horizon}")
    return summary


def _indices_for_groups(indices_by_group: dict[str, list[int]], groups: Sequence[str]) -> list[int]:
    indices: list[int] = []
    for group in groups:
        indices.extend(indices_by_group[group])
    return sorted(indices)


def _sum_graph_attr(graphs: Sequence[Any], attr: str) -> int:
    total = 0
    for graph in graphs:
        value = getattr(graph, attr, None)
        if value is None:
            continue
        total += int(value.sum().item())
    return total


def _jsonable_dataclass(value) -> dict[str, Any]:
    return _to_jsonable(asdict(value))


def _to_jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    return value


def parse_feature_columns(value: str | None) -> tuple[str, ...] | None:
    if value is None or not value.strip():
        return None
    return tuple(part.strip() for part in value.split(",") if part.strip())


def parse_horizons(value: str) -> tuple[int, ...]:
    horizons = tuple(sorted({int(part.strip()) for part in value.split(",") if part.strip()}))
    if not horizons or any(horizon < 1 for horizon in horizons):
        raise ValueError("horizons must contain positive integers.")
    return horizons


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cached PyG frame graphs from processed HeLa spot data.")
    parser.add_argument("--source", type=Path, default=FrameGraphDatasetConfig().source_path)
    parser.add_argument("--out", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--edge-radius", type=float, default=FrameGraphDatasetConfig().edge_radius)
    parser.add_argument("--edge-k-nearest", type=int, default=FrameGraphDatasetConfig().edge_k_nearest)
    parser.add_argument("--horizons", default="3,5,10")
    parser.add_argument("--node-features", default=None, help="Comma-separated feature columns. Omit to use defaults.")
    parser.add_argument("--split-mode", choices=("by_position", "by_sequence", "none"), default="by_position")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-graphs", type=int, default=None, help="Optional smoke-test limit.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_config = FrameGraphDatasetConfig(
        source_path=args.source,
        edge_radius=args.edge_radius,
        edge_k_nearest=args.edge_k_nearest,
        horizons=parse_horizons(args.horizons),
        node_feature_columns=parse_feature_columns(args.node_features),
    )
    split_config = SplitConfig(
        mode=args.split_mode,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    cache = build_and_save_graph_cache(
        args.out,
        source_path=args.source,
        dataset_config=dataset_config,
        split_config=split_config,
        max_graphs=args.max_graphs,
    )
    print(json.dumps(_to_jsonable(cache["summary"]), ensure_ascii=False, indent=2))
    print(f"Saved graph cache to {args.out}")
    print(f"Saved summary to {args.out.with_suffix(args.out.suffix + '.summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
