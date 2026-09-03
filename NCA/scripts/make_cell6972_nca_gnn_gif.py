from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from PIL import Image


def find_project_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "NCA" / "data_v2").exists():
            return candidate
        nested = candidate / "Real_game_of_life"
        if (nested / "NCA" / "data_v2").exists():
            return nested
    raise FileNotFoundError("Не найдена папка NCA/data_v2 относительно текущего каталога.")


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from GNN.graph_dataset import FrameGraphDatasetConfig, build_frame_graphs, load_processed_spots


def parse_local_contour(value):
    if not isinstance(value, str) or not value:
        return None
    points = []
    for pair in value.split("|"):
        if ":" not in pair:
            continue
        x_text, y_text = pair.split(":", 1)
        try:
            points.append((float(x_text), float(y_text)))
        except ValueError:
            continue
    return np.asarray(points, dtype=float) if len(points) >= 3 else None


def choose_sample(data_root: Path, split: str, file_index: int) -> Path:
    split_files = sorted((data_root / split).glob("*.npy"))
    if not split_files:
        raise FileNotFoundError(f"В split={split!r} не найдено .npy файлов.")
    if file_index >= len(split_files):
        raise IndexError(f"file_index={file_index} вне диапазона: найдено файлов {len(split_files)}")
    return split_files[file_index]


def matching_sequence_spots(sample_path: Path, cfg: FrameGraphDatasetConfig) -> tuple[str, pd.DataFrame]:
    spots = load_processed_spots(cfg.source_path)
    sample_name = sample_path.stem
    sequence_mask = spots[cfg.sequence_col].astype(str).str.endswith(sample_name)
    if "sequence_name" in spots.columns:
        sequence_mask |= spots["sequence_name"].astype(str).eq(sample_name)

    sequence_ids = sorted(spots.loc[sequence_mask, cfg.sequence_col].astype(str).unique())
    if not sequence_ids:
        raise ValueError(f"Не нашлась GNN-последовательность для {sample_name!r}.")

    sequence_uid = sequence_ids[0]
    sequence_spots = spots.loc[spots[cfg.sequence_col].astype(str).eq(sequence_uid)].copy()
    return sequence_uid, sequence_spots


def track_ids_by_frame(
    sequence_spots: pd.DataFrame,
    cfg: FrameGraphDatasetConfig,
    start_spot_id: int,
    max_steps: int,
) -> dict[int, int]:
    rows_by_id = {
        int(row[cfg.node_id_col]): row
        for _, row in sequence_spots.iterrows()
        if pd.notna(row.get(cfg.node_id_col))
    }

    out: dict[int, int] = {}
    current_id = int(start_spot_id)
    for _ in range(max_steps + 1):
        row = rows_by_id.get(current_id)
        if row is None:
            break
        out[int(row[cfg.frame_col])] = current_id
        next_id = row.get("next_id", np.nan)
        if pd.isna(next_id):
            break
        current_id = int(float(next_id))
    return out


def build_hop_region(graph, center_idx: int, hops: int):
    adjacency = {index: set() for index in range(graph.num_nodes)}
    for source, target in graph.edge_index.T:
        source = int(source)
        target = int(target)
        adjacency[source].add(target)
        adjacency[target].add(source)

    hop_by_node = {center_idx: 0}
    frontier = {center_idx}
    for hop in range(1, hops + 1):
        next_frontier = set()
        for node in frontier:
            next_frontier.update(adjacency[node])
        next_frontier -= set(hop_by_node)
        for node in sorted(next_frontier):
            hop_by_node[node] = hop
        frontier = next_frontier

    selected = sorted(hop_by_node, key=lambda index: (hop_by_node[index], index))
    selected_set = set(selected)
    selected_edges = []
    for source, target in graph.edge_index.T:
        source = int(source)
        target = int(target)
        if source in selected_set and target in selected_set:
            selected_edges.append((source, target))
    return selected, selected_edges, hop_by_node


def draw_combined_frame(
    *,
    frame: np.ndarray,
    sample_path: Path,
    frame_index: int,
    graph,
    nodes: pd.DataFrame,
    pos: np.ndarray,
    cfg: FrameGraphDatasetConfig,
    center_spot_id: int,
    grid_size: float,
    combined_gnn_hops: int,
    nca_steps_to_show: tuple[int, ...],
    nca_kernel_size: int,
    nca_conv_layers_per_step: int,
    out_path: Path,
) -> None:
    node_ids_numeric = pd.to_numeric(nodes[cfg.node_id_col], errors="coerce")
    center_matches = np.flatnonzero(node_ids_numeric.to_numpy() == center_spot_id)
    if len(center_matches) == 0:
        raise ValueError(f"spot_id={center_spot_id} не найден в frame={frame_index}")
    center_idx = int(center_matches[0])

    pos_grid = pos / grid_size
    center_grid = pos_grid[center_idx].copy()
    frame_height, frame_width = frame.shape
    one_step_radius = (nca_kernel_size // 2) * nca_conv_layers_per_step

    selected, selected_edges, hop_by_node = build_hop_region(graph, center_idx, combined_gnn_hops)

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.set_facecolor("#ffffff")
    ax.set_xticks(np.arange(-0.5, frame_width + 0.5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, frame_height + 0.5, 1), minor=True)
    ax.grid(which="minor", color="#b8b8b8", linewidth=0.55, alpha=0.95, zorder=0)
    for grid_line_x in np.arange(-0.5, frame_width + 0.5, 5):
        ax.axvline(grid_line_x, color="#777777", linewidth=0.85, alpha=0.65, zorder=0)
    for grid_line_y in np.arange(-0.5, frame_height + 0.5, 5):
        ax.axhline(grid_line_y, color="#777777", linewidth=0.85, alpha=0.65, zorder=0)
    ax.tick_params(which="minor", bottom=False, left=False)

    nca_colors = {1: "#fdae61", 2: "#2b83ba", 4: "#8073ac"}
    for steps in nca_steps_to_show:
        radius = one_step_radius * int(steps)
        width = 2 * radius + 1
        ax.add_patch(
            plt.Rectangle(
                (center_grid[0] - radius - 0.5, center_grid[1] - radius - 0.5),
                width,
                width,
                fill=False,
                edgecolor=nca_colors.get(steps, "#555555"),
                linewidth=1.7,
                alpha=0.95,
                zorder=5,
                label=f"NCA one update: {width:g}x{width:g}",
            )
        )

    if selected_edges:
        segments = [(pos_grid[source], pos_grid[target]) for source, target in selected_edges]
        ax.add_collection(
            LineCollection(segments, colors="#00a6ff", linewidths=1.05, alpha=0.48, zorder=3, label="GNN edges")
        )

    if cfg.edge_radius is not None:
        gnn_radius_grid = float(cfg.edge_radius) / grid_size
        ax.add_patch(
            plt.Circle(
                center_grid,
                gnn_radius_grid,
                fill=False,
                color="#00a6ff",
                linestyle="--",
                linewidth=1.2,
                alpha=0.75,
                zorder=4,
                label=f"GNN edge radius={gnn_radius_grid:.1f} grid",
            )
        )

    hop_colors = {0: "#d7191c", 1: "#fdae61", 2: "#2b83ba", 3: "#abdda4", 4: "#8073ac"}
    area_values = pd.to_numeric(nodes["AREA"], errors="coerce") if "AREA" in nodes.columns else pd.Series(np.nan, index=nodes.index)
    radius_values = (
        pd.to_numeric(nodes["RADIUS"], errors="coerce") if "RADIUS" in nodes.columns else pd.Series(np.nan, index=nodes.index)
    )

    for index in selected:
        is_center = index == center_idx
        hop = hop_by_node[index]
        color = hop_colors.get(hop, "#999999")
        contour = parse_local_contour(nodes.loc[index, "contour_xy_local"]) if "contour_xy_local" in nodes.columns else None
        if contour is not None:
            contour_grid = contour / grid_size + pos_grid[index]
            ax.add_patch(
                Polygon(
                    contour_grid,
                    closed=True,
                    facecolor="#e4572e" if is_center else "#f2c94c",
                    edgecolor="#8b0000" if is_center else "#8a5a00",
                    linewidth=2.1 if is_center else 0.9,
                    alpha=0.52 if is_center else 0.24,
                    zorder=6 if is_center else 2,
                )
            )

        area_raw = area_values.iloc[index]
        radius_raw = radius_values.iloc[index]
        area_grid = area_raw / (grid_size**2) if pd.notna(area_raw) else np.nan
        marker_size = 125 if is_center else 46
        if pd.notna(area_grid) and math.isfinite(float(area_grid)):
            marker_size += min(90, 8 * math.sqrt(max(float(area_grid), 0.0)))
        ax.scatter(
            pos_grid[index, 0],
            pos_grid[index, 1],
            s=marker_size,
            color=color,
            edgecolors="white",
            linewidths=0.8,
            zorder=7 if is_center else 6,
            label="центр" if is_center else None,
        )
        if is_center or hop <= 1:
            spot_id = int(node_ids_numeric.iloc[index]) if pd.notna(node_ids_numeric.iloc[index]) else nodes.loc[index, cfg.node_id_col]
            label = [str(spot_id)]
            if pd.notna(area_grid):
                label.append(f"A={area_grid:.1f}")
            if pd.notna(radius_raw):
                label.append(f"R={radius_raw / grid_size:.1f}")
            ax.annotate(
                "\n".join(label),
                pos_grid[index] + np.array([0.35, 0.35]),
                fontsize=7,
                color="#222222",
                alpha=0.9,
                zorder=8,
            )

    ax.scatter(center_grid[0], center_grid[1], s=180, facecolors="none", edgecolors="#8b0000", linewidths=2.2, zorder=9)
    ax.set_title(
        f"NCA + GNN вокруг spot_id={center_spot_id} | frame={frame_index} | {sample_path.stem}",
        fontsize=11,
    )
    ax.set_xlabel("NCA grid x")
    ax.set_ylabel("NCA grid y")
    ax.set_xlim(-0.5, frame_width - 0.5)
    ax.set_ylim(frame_height - 0.5, -0.5)
    ax.set_aspect("equal", adjustable="box")

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper right", fontsize=8, framealpha=0.94)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate NCA+GNN animation around a tracked cell.")
    parser.add_argument("--spot-id", type=int, default=6972)
    parser.add_argument("--split", default="train")
    parser.add_argument("--file-index", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "NCA" / "runs" / "cell_6972_nca_gnn.gif")
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()
    if not args.out.is_absolute():
        args.out = (Path.cwd() / args.out).resolve()

    data_root = PROJECT_ROOT / "NCA" / "data_v2"
    sample_path = choose_sample(data_root, args.split, args.file_index)
    sample = np.load(sample_path, mmap_mode="r")
    if sample.ndim != 4:
        raise ValueError(f"{sample_path} должен иметь форму [T, H, W, C], получено {sample.shape}")

    cfg = FrameGraphDatasetConfig()
    sequence_uid, sequence_spots = matching_sequence_spots(sample_path, cfg)
    frame_graphs = build_frame_graphs(sequence_spots, cfg, add_targets=False)
    graphs_by_frame = {
        int(item.nodes[cfg.frame_col].iloc[0]): item
        for item in frame_graphs
        if not item.nodes.empty
    }

    qc = pd.read_csv(data_root / "qc.csv")
    qc_match = qc[qc["name"].astype(str).eq(sample_path.stem)]
    grid_size = float(qc_match.iloc[0]["grid_size"]) if not qc_match.empty and "grid_size" in qc_match.columns else 8.0

    track_by_frame = track_ids_by_frame(sequence_spots, cfg, args.spot_id, args.frames + args.start_frame + 5)
    if not track_by_frame:
        raise ValueError(f"Не удалось построить трек от spot_id={args.spot_id}")

    out_dir = args.out.parent
    frames_dir = out_dir / f"{args.out.stem}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    png_paths = []
    max_frame = min(sample.shape[0] - 1, args.start_frame + args.frames - 1)
    for frame_index in range(args.start_frame, max_frame + 1):
        center_spot_id = track_by_frame.get(frame_index)
        if center_spot_id is None or frame_index not in graphs_by_frame:
            continue
        graph = graphs_by_frame[frame_index]
        nodes = graph.nodes.reset_index(drop=True)
        pos = nodes.loc[:, list(cfg.position_cols)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        frame = np.asarray(sample[frame_index, :, :, 0])
        png_path = frames_dir / f"frame_{frame_index:04d}_spot_{center_spot_id}.png"
        draw_combined_frame(
            frame=frame,
            sample_path=sample_path,
            frame_index=frame_index,
            graph=graph,
            nodes=nodes,
            pos=pos,
            cfg=cfg,
            center_spot_id=center_spot_id,
            grid_size=grid_size,
            combined_gnn_hops=args.hops,
            nca_steps_to_show=(1,),
            nca_kernel_size=3,
            nca_conv_layers_per_step=2,
            out_path=png_path,
        )
        png_paths.append(png_path)
        print(f"saved {png_path.relative_to(PROJECT_ROOT)}")

    if not png_paths:
        raise RuntimeError("Не удалось сохранить ни одного кадра для GIF.")

    images = [Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE) for path in png_paths]
    images[0].save(
        args.out,
        save_all=True,
        append_images=images[1:],
        duration=int(round(1000.0 / args.fps)),
        loop=0,
        optimize=False,
    )
    print(f"GIF: {args.out}")
    print(f"frames: {len(png_paths)}")
    print(f"sequence_uid: {sequence_uid}")
    print(f"sample: {sample_path.relative_to(PROJECT_ROOT)}")

    if not args.keep_frames:
        for path in png_paths:
            path.unlink()
        try:
            frames_dir.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    main()
