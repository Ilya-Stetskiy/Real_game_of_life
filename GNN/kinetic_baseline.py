from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch import nn

from .dataset_cache import SplitConfig, build_graph_cache, save_graph_cache
from .evaluate_rollout import evaluate_model_rollout, write_reports
from .gnn_model import CellGNNOutput
from .graph_dataset import DEFAULT_PROCESSED_SPOTS, FrameGraphDatasetConfig, load_processed_spots
from .train_one_step import infer_shape_dim, jsonable, resolve_device
from .train_rollout_bptt import group_graphs_by_sequence, rollout_config_from_cache

TEMPORAL_VELOCITY_X_COLUMN = "temporal_lag1_delta_x"
TEMPORAL_VELOCITY_Y_COLUMN = "temporal_lag1_delta_y"
TEMPORAL_HAS_ANCESTOR_COLUMN = "temporal_lag1_has_ancestor"


@dataclass(frozen=True)
class KineticBaselineConfig:
    out_dir: Path = Path(__file__).resolve().parent / "runs" / "kinetic_baseline"
    horizons: tuple[int, ...] = (1, 3, 5, 10, 20)
    max_graphs: int | None = None
    device: str = "auto"
    match_tolerance: float = 1e-4


def calibrate_persistent_random_walk(
    graphs: Sequence[Any],
    *,
    node_feature_columns: Sequence[str],
) -> dict[str, float]:
    """Fit an isotropic AR(1) velocity-persistence coefficient from ground-truth tracks.

    For every node with a known incoming velocity v_t (temporal_lag1_delta_{x,y}, i.e.
    the displacement that produced the current frame) and a known outgoing velocity
    v_{t+1} (target_delta_pos, the displacement to the next frame), phi minimizes
    sum ||v_{t+1} - phi * v_t||^2, i.e. phi = <v_t, v_{t+1}> / <v_t, v_t> pooled over
    both x/y components and all cells/frames in the given (train-split) graphs. This is
    the calibration for a discrete Ornstein-Uhlenbeck / persistent-random-walk model:
    v_{t+1} = phi * v_t + noise. tau = -1 / ln(phi) is the persistence time in frames.
    """

    if TEMPORAL_VELOCITY_X_COLUMN not in node_feature_columns or TEMPORAL_VELOCITY_Y_COLUMN not in node_feature_columns:
        raise ValueError(
            "calibrate_persistent_random_walk requires temporal_lag1_delta_x/y node features; "
            "build the cache with FrameGraphDatasetConfig(temporal_lags=(1,))."
        )
    vx_index = node_feature_columns.index(TEMPORAL_VELOCITY_X_COLUMN)
    vy_index = node_feature_columns.index(TEMPORAL_VELOCITY_Y_COLUMN)
    has_ancestor_index = (
        node_feature_columns.index(TEMPORAL_HAS_ANCESTOR_COLUMN)
        if TEMPORAL_HAS_ANCESTOR_COLUMN in node_feature_columns
        else None
    )

    dot_v_next = 0.0
    dot_v_v = 0.0
    speed_sq_sum = 0.0
    n_pairs = 0
    for graph in graphs:
        valid_next = getattr(graph, "valid_regression_mask", None)
        if valid_next is None or not hasattr(graph, "target_delta_pos"):
            continue
        valid_next = valid_next.bool()
        has_ancestor = (
            graph.x[:, has_ancestor_index] > 0.5
            if has_ancestor_index is not None
            else torch.ones(graph.num_nodes, dtype=torch.bool)
        )
        valid = valid_next & has_ancestor
        if not bool(valid.any()):
            continue
        v_in = graph.x[valid][:, [vx_index, vy_index]].double()
        v_out = graph.target_delta_pos[valid].double()
        dot_v_next += float((v_in * v_out).sum().item())
        dot_v_v += float((v_in * v_in).sum().item())
        speed_sq_sum += float((v_in ** 2).sum().item())
        n_pairs += int(v_in.size(0))

    if dot_v_v <= 0.0:
        raise ValueError("No valid (v_t, v_{t+1}) pairs found to calibrate the kinetic baseline.")

    phi = dot_v_next / dot_v_v
    phi_clamped = min(max(phi, 1e-6), 1.0 - 1e-6)
    tau = -1.0 / math.log(phi_clamped)
    return {
        "phi": phi,
        "phi_clamped": phi_clamped,
        "tau_frames": tau,
        "n_pairs": n_pairs,
        "mean_speed_component": (speed_sq_sum / (2 * n_pairs)) ** 0.5 if n_pairs else float("nan"),
    }


class KineticPersistentRandomWalkModel(nn.Module):
    """Physical, parameter-free position baseline: decaying-velocity extrapolation.

    Reads the most recent frame-to-frame displacement from temporal_lag1_delta_{x,y}
    and predicts the next step's displacement as phi * v -- iterating this one-step
    rule autoregressively (via the same rollout machinery used for the GNN) reproduces
    the closed-form expected displacement of a discrete persistent-random-walk / OU
    velocity process: E[x_{t+h} - x_t] = v_t * phi * (1 - phi^h) / (1 - phi).
    Shape and polarization are left unpredicted (delta_shape=0, delta_polarization=None
    -> persistence), since this baseline targets bare position kinematics only.
    """

    def __init__(self, *, phi: float, shape_dim: int, node_feature_columns: Sequence[str]) -> None:
        super().__init__()
        self.phi = float(phi)
        self.shape_dim = int(shape_dim)
        self.node_feature_columns = tuple(node_feature_columns)
        self._vx_index = self.node_feature_columns.index(TEMPORAL_VELOCITY_X_COLUMN)
        self._vy_index = self.node_feature_columns.index(TEMPORAL_VELOCITY_Y_COLUMN)

    def forward(self, graph: Any) -> CellGNNOutput:
        num_nodes = int(graph.num_nodes)
        velocity = graph.x[:, [self._vx_index, self._vy_index]].float()
        delta_pos = self.phi * velocity
        device = graph.x.device
        return CellGNNOutput(
            delta_pos=delta_pos,
            delta_shape=torch.zeros((num_nodes, self.shape_dim), dtype=torch.float32, device=device),
            division_logits=torch.zeros(num_nodes, dtype=torch.float32, device=device),
            death_logits=torch.zeros(num_nodes, dtype=torch.float32, device=device),
            node_embeddings=torch.zeros((num_nodes, 1), dtype=torch.float32, device=device),
        )


def _kinetic_dataset_config(base: FrameGraphDatasetConfig | None) -> FrameGraphDatasetConfig:
    from dataclasses import replace

    base = base or FrameGraphDatasetConfig()
    if 1 not in base.temporal_lags:
        base = replace(base, temporal_lags=tuple(sorted({1, *base.temporal_lags})))
    return base


def run_kinetic_baseline(
    config: KineticBaselineConfig,
    *,
    dataset_config: FrameGraphDatasetConfig | None = None,
    split_config: SplitConfig | None = None,
) -> dict[str, Any]:
    """Calibrate the kinetic baseline on train and evaluate it on test at the given horizons.

    Rows are written in the same schema as evaluate_rollout.py's rollout_eval_*.json
    (model, checkpoint, horizon, matched_nodes, position_mean/median/rmse, shape_rmse,
    valid_shape_fraction, polarization_theta_rmse, polarization_aspect_rmse), with
    model="kinetic_prw" -- use merge_rollout_reports() to combine this file with an
    evaluate_rollout.py comparison (which needs its own, differently-featured cache
    matching the GNN checkpoints it evaluates) into one horizon -> RMSE table.
    """

    dataset_config = _kinetic_dataset_config(dataset_config)
    spots = load_processed_spots(dataset_config.source_path)
    cache = build_graph_cache(
        spots=spots,
        dataset_config=dataset_config,
        split_config=split_config or SplitConfig(),
        max_graphs=config.max_graphs,
    )
    config.out_dir.mkdir(parents=True, exist_ok=True)
    save_graph_cache(cache, config.out_dir / "kinetic_baseline_cache.pt")
    graphs = cache["graphs"]
    splits = cache["splits"]
    train_graphs = [graphs[index] for index in splits.get("train", [])]
    test_graphs = [graphs[index] for index in splits.get("test", [])]
    if not train_graphs:
        raise ValueError("Train split is empty; cannot calibrate the kinetic baseline.")
    if not test_graphs:
        raise ValueError("Test split is empty; cannot evaluate the kinetic baseline.")

    node_feature_columns = tuple(str(column) for column in graphs[0].node_feature_columns)
    calibration = calibrate_persistent_random_walk(train_graphs, node_feature_columns=node_feature_columns)

    try:
        shape_dim = infer_shape_dim(graphs)
    except ValueError:
        shape_dim = 0

    device = resolve_device(config.device)
    model = KineticPersistentRandomWalkModel(
        phi=calibration["phi_clamped"],
        shape_dim=shape_dim,
        node_feature_columns=node_feature_columns,
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
    rows = [{"model": "kinetic_prw", "checkpoint": None, **row} for row in metrics]

    results = {
        "config": jsonable(asdict(config)),
        "calibration": jsonable(calibration),
        "cache_summary": cache.get("summary", {}),
        "rows": rows,
    }
    write_json(config.out_dir / "kinetic_baseline_summary.json", results)
    write_reports(rows, config.out_dir / "kinetic_baseline_rollout_eval.json")
    write_report(config.out_dir / "kinetic_baseline_report.md", results)
    return results


def merge_rollout_reports(*paths: str | Path) -> list[dict[str, Any]]:
    """Concatenate rollout_eval_*.json-schema row lists (e.g. evaluate_rollout.py's output
    and this module's kinetic_baseline_rollout_eval.json) into one horizon -> RMSE table."""

    rows: list[dict[str, Any]] = []
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        rows.extend(data if isinstance(data, list) else data.get("rows", []))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_report(path: Path, results: dict[str, Any]) -> None:
    calibration = results["calibration"]
    lines = [
        "# Kinetic baseline (persistent random walk) vs GNN",
        "",
        "## Calibration (train split)",
        "",
        f"- phi (velocity persistence per frame): `{calibration['phi']:.4f}` "
        f"(clamped to `{calibration['phi_clamped']:.4f}` for the expected-trajectory formula)",
        f"- tau (persistence time, frames): `{calibration['tau_frames']:.2f}`",
        f"- mean speed component (px/frame, per axis): `{calibration['mean_speed_component']:.4f}`",
        f"- calibration pairs: `{calibration['n_pairs']}`",
        "",
        "## horizon -> RMSE by model",
        "",
        "| model | horizon | matched_nodes | position_rmse | polarization_theta_rmse | polarization_aspect_rmse |",
        "|---|---|---|---|---|---|",
    ]
    for row in sorted(results["rows"], key=lambda r: (str(r["model"]), int(r["horizon"]))):
        theta = row.get("polarization_theta_rmse")
        aspect = row.get("polarization_aspect_rmse")
        lines.append(
            f"| {row['model']} | {row['horizon']} | {row['matched_nodes']} | {row['position_rmse']:.4f} | "
            f"{theta:.4f} | {aspect:.4f} |"
            if theta is not None and aspect is not None
            else f"| {row['model']} | {row['horizon']} | {row['matched_nodes']} | {row['position_rmse']:.4f} | n/a | n/a |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate a physical persistent-random-walk position baseline (no learned parameters) and "
            "evaluate it by rollout horizon, in the same JSON schema as evaluate_rollout.py."
        )
    )
    parser.add_argument("--out-dir", type=Path, default=KineticBaselineConfig().out_dir)
    parser.add_argument("--horizons", type=int, nargs="+", default=list(KineticBaselineConfig().horizons))
    parser.add_argument("--max-graphs", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--device", default=KineticBaselineConfig().device)
    parser.add_argument("--match-tolerance", type=float, default=KineticBaselineConfig().match_tolerance)
    parser.add_argument("--seed", type=int, default=SplitConfig().seed)
    parser.add_argument("--split-mode", default=SplitConfig().mode)
    parser.add_argument("--source", type=Path, default=DEFAULT_PROCESSED_SPOTS)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    config = KineticBaselineConfig(
        out_dir=args.out_dir,
        horizons=tuple(args.horizons),
        max_graphs=args.max_graphs,
        device=args.device,
        match_tolerance=args.match_tolerance,
    )
    dataset_config = FrameGraphDatasetConfig(source_path=args.source)
    split_config = SplitConfig(mode=args.split_mode, seed=args.seed)
    results = run_kinetic_baseline(config, dataset_config=dataset_config, split_config=split_config)
    for row in results["rows"]:
        print(
            f"{row['model']} h={row['horizon']} matched={row['matched_nodes']} "
            f"position_rmse={row['position_rmse']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
