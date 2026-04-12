from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image


DNN_NAME_RE = re.compile(r"HeLa-S3_nuc_pos(?P<pos>\d+)_q(?P<q>\d+)_5c$")
DEFAULT_CHANNELS = ("density",)
TRACKMATE_CHANNELS = (
    "spot_count",
    "mean_area",
    "flow_x",
    "flow_y",
    "birth_count",
    "death_count",
    "split_count",
)
ALL_CHANNELS = ("density", "occupancy", *TRACKMATE_CHANNELS)


@dataclass(frozen=True)
class SequenceRecord:
    dataset: str
    name: str
    source_split: str
    group: str
    raw_path: Path | None
    mask_path: Path
    spots_path: Path | None = None
    edges_path: Path | None = None
    tracks_path: Path | None = None
    metadata_path: Path | None = None
    xml_path: Path | None = None
    pos: int | None = None
    q: int | None = None


def split_counts(total: int, ratios: Sequence[float]) -> tuple[int, int, int]:
    if len(ratios) != 3:
        raise ValueError("split ratios must contain train, val, test.")
    if total < 0 or any(ratio < 0 for ratio in ratios):
        raise ValueError("split ratios must be non-negative.")
    ratio_sum = sum(ratios)
    if ratio_sum <= 0:
        raise ValueError("split ratios must sum to a positive value.")

    normalized = [ratio / ratio_sum for ratio in ratios]
    counts = [int(math.floor(total * ratio)) for ratio in normalized]
    for index in range(total - sum(counts)):
        counts[index % 3] += 1
    return counts[0], counts[1], counts[2]


def parse_channels(value: str) -> list[str]:
    aliases = {
        "density_only": DEFAULT_CHANNELS,
        "trackmate": TRACKMATE_CHANNELS,
        "all": ALL_CHANNELS,
    }
    channels: list[str] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        expanded = aliases.get(item, (item,))
        channels.extend(expanded)

    seen: set[str] = set()
    result: list[str] = []
    for channel in channels:
        if channel not in ALL_CHANNELS:
            raise ValueError(f"Unsupported channel {channel!r}. Expected one of {ALL_CHANNELS}.")
        if channel not in seen:
            seen.add(channel)
            result.append(channel)
    if not result:
        raise ValueError("At least one output channel is required.")
    return result


def read_key_value_metadata(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}

    metadata: dict[str, str] = {}
    for line in path.read_text(errors="ignore").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def image_info(path: Path | None) -> dict[str, int | tuple[int, int]] | None:
    if path is None or not path.exists():
        return None
    with Image.open(path) as image:
        return {
            "width": int(image.size[0]),
            "height": int(image.size[1]),
            "frames": int(getattr(image, "n_frames", 1)),
        }


def aligned_frame_indices(mask_frames: int, target_frames: int | None, mode: str) -> list[int]:
    if target_frames is None or target_frames == mask_frames:
        return list(range(mask_frames))

    if mode == "strict":
        raise ValueError(f"Frame count mismatch: mask has {mask_frames}, raw has {target_frames}.")
    if mode == "truncate":
        return list(range(min(mask_frames, target_frames)))
    if mode == "sample":
        if target_frames <= 1:
            return [0]
        return [int(round(value)) for value in np.linspace(0, mask_frames - 1, target_frames)]
    raise ValueError(f"Unsupported frame alignment mode: {mode}")


def iter_mask_frames(mask_path: Path, target_frames: int | None, frame_align: str) -> Iterable[np.ndarray]:
    with Image.open(mask_path) as image:
        mask_frames = int(getattr(image, "n_frames", 1))
        for frame_index in aligned_frame_indices(mask_frames, target_frames, frame_align):
            image.seek(frame_index)
            yield np.asarray(image).squeeze()


def grid_fill_ratio(frame: np.ndarray, grid_size: int) -> np.ndarray:
    if grid_size <= 0:
        raise ValueError("grid_size must be positive.")

    binary = np.not_equal(np.asarray(frame).squeeze(), 0)
    if binary.ndim != 2:
        raise ValueError(f"Expected a 2D mask frame, got shape {binary.shape}.")

    height, width = binary.shape
    row_starts = np.arange(0, height, grid_size)
    col_starts = np.arange(0, width, grid_size)
    row_sizes = np.diff(np.r_[row_starts, height]).astype(np.float32)
    col_sizes = np.diff(np.r_[col_starts, width]).astype(np.float32)

    counts = np.add.reduceat(binary.astype(np.float32), row_starts, axis=0)
    counts = np.add.reduceat(counts, col_starts, axis=1)
    return counts / (row_sizes[:, None] * col_sizes[None, :])


def compute_density_stack(
    mask_path: Path,
    grid_size: int,
    target_frames: int | None,
    frame_align: str,
) -> np.ndarray:
    frames = [grid_fill_ratio(frame, grid_size) for frame in iter_mask_frames(mask_path, target_frames, frame_align)]
    if not frames:
        raise ValueError(f"No frames found in {mask_path}.")
    return np.stack(frames, axis=0).astype(np.float32)


def discover_dynamic_nuclear_net(source_root: Path) -> list[SequenceRecord]:
    records: list[SequenceRecord] = []
    for split in ("train", "val", "test"):
        split_dir = source_root / split
        if not split_dir.exists():
            continue
        for raw_path in sorted(split_dir.glob("HeLa-S3_nuc_pos*_q*_5c.tif")):
            name = raw_path.stem
            match = DNN_NAME_RE.match(name)
            pos = int(match.group("pos")) if match else None
            q = int(match.group("q")) if match else None
            out_dir = split_dir / "out" / name
            records.append(
                SequenceRecord(
                    dataset="DynamicNuclearNet",
                    name=name,
                    source_split=split,
                    group=f"pos{pos}" if pos is not None else name,
                    raw_path=raw_path,
                    mask_path=out_dir / f"{name}_mask.tif",
                    spots_path=out_dir / f"{name}_spots.csv",
                    edges_path=out_dir / f"{name}_edges.csv",
                    tracks_path=out_dir / f"{name}_tracks.csv",
                    metadata_path=raw_path.with_suffix(".txt"),
                    xml_path=out_dir / f"{name}_trackmate.xml",
                    pos=pos,
                    q=q,
                )
            )
    return records


def discover_generic_pairs(source_root: Path) -> list[SequenceRecord]:
    image_candidates: dict[str, list[Path]] = defaultdict(list)
    mask_candidates: dict[str, list[Path]] = defaultdict(list)

    for path in source_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".tif", ".tiff"}:
            continue
        stem = path.stem
        stem_lower = stem.lower()
        if stem_lower.endswith("_mask"):
            mask_candidates[stem[:-5].lower()].append(path)
        elif stem_lower.startswith("lblimg_"):
            mask_candidates[stem[7:].lower()].append(path)
        else:
            image_candidates[stem_lower].append(path)

    records: list[SequenceRecord] = []
    for key, raw_paths in sorted(image_candidates.items()):
        masks = mask_candidates.get(key, [])
        if len(raw_paths) != 1 or len(masks) != 1:
            continue
        raw_path = raw_paths[0]
        mask_path = masks[0]
        parent = raw_path.parent
        csv_root_candidates = [parent / "result_csv", parent / "result_scv", parent]
        spots = edges = tracks = None
        for csv_root in csv_root_candidates:
            if not csv_root.exists():
                continue
            spots_matches = sorted(csv_root.glob("*_spots.csv"))
            edges_matches = sorted(csv_root.glob("*_edges.csv"))
            tracks_matches = sorted(csv_root.glob("*_tracks.csv"))
            spots = spots or (spots_matches[0] if len(spots_matches) == 1 else None)
            edges = edges or (edges_matches[0] if len(edges_matches) == 1 else None)
            tracks = tracks or (tracks_matches[0] if len(tracks_matches) == 1 else None)

        records.append(
            SequenceRecord(
                dataset=source_root.name,
                name=raw_path.stem,
                source_split="all",
                group=raw_path.stem,
                raw_path=raw_path,
                mask_path=mask_path,
                spots_path=spots,
                edges_path=edges,
                tracks_path=tracks,
                metadata_path=None,
                xml_path=next(iter(parent.glob("*.xml")), None),
            )
        )
    return records


def validate_record(record: SequenceRecord) -> None:
    missing = [name for name, path in (("mask", record.mask_path), ("raw", record.raw_path)) if path and not path.exists()]
    if missing:
        raise FileNotFoundError(f"{record.name}: missing {', '.join(missing)} file.")


def assign_output_splits(
    records: Sequence[SequenceRecord],
    policy: str,
    split_ratios: Sequence[float],
    seed: int,
) -> dict[str, str]:
    if policy == "source":
        return {record.name: record.source_split for record in records}
    if policy == "flat":
        return {record.name: "" for record in records}
    if policy != "by_group":
        raise ValueError(f"Unsupported split policy: {policy}")

    groups = sorted({record.group for record in records})
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(groups, dtype=object)
    rng.shuffle(shuffled)

    train_count, val_count, test_count = split_counts(len(groups), split_ratios)
    group_to_split: dict[str, str] = {}
    for group in shuffled[:train_count]:
        group_to_split[str(group)] = "train"
    for group in shuffled[train_count:train_count + val_count]:
        group_to_split[str(group)] = "val"
    for group in shuffled[train_count + val_count:train_count + val_count + test_count]:
        group_to_split[str(group)] = "test"
    return {record.name: group_to_split[record.group] for record in records}


def _float_value(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _string_value(row: dict[str, str], key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    value = value.strip()
    return value or None


def load_spots(spots_path: Path | None) -> list[dict[str, float | int | str]]:
    if spots_path is None or not spots_path.exists():
        return []

    spots: list[dict[str, float | int | str]] = []
    with spots_path.open(newline="", errors="ignore") as handle:
        for row in csv.DictReader(handle):
            spot_id = _string_value(row, "ID")
            track_id = _string_value(row, "TRACK_ID")
            x = _float_value(row, "POSITION_X")
            y = _float_value(row, "POSITION_Y")
            frame = _float_value(row, "FRAME")
            if spot_id is None or x is None or y is None or frame is None:
                continue
            radius = _float_value(row, "RADIUS")
            area = _float_value(row, "AREA")
            if area is None and radius is not None:
                area = math.pi * radius * radius
            spots.append(
                {
                    "id": spot_id,
                    "track_id": track_id or "",
                    "x": float(x),
                    "y": float(y),
                    "frame": int(round(frame)),
                    "area": float(area) if area is not None else 0.0,
                }
            )
    return spots


def load_split_source_ids(edges_path: Path | None) -> set[str]:
    if edges_path is None or not edges_path.exists():
        return set()

    outgoing: Counter[str] = Counter()
    with edges_path.open(newline="", errors="ignore") as handle:
        for row in csv.DictReader(handle):
            source_id = _string_value(row, "SPOT_SOURCE_ID")
            target_id = _string_value(row, "SPOT_TARGET_ID")
            if source_id is None or target_id is None:
                continue
            outgoing[source_id] += 1
    return {source_id for source_id, count in outgoing.items() if count > 1}


def spot_cell(spot: dict[str, float | int | str], grid_size: int, rows: int, cols: int) -> tuple[int, int]:
    row = int(float(spot["y"]) // grid_size)
    col = int(float(spot["x"]) // grid_size)
    return max(0, min(rows - 1, row)), max(0, min(cols - 1, col))


def compute_trackmate_channels(
    record: SequenceRecord,
    channels: Sequence[str],
    frames: int,
    rows: int,
    cols: int,
    grid_size: int,
) -> dict[str, np.ndarray]:
    requested = [channel for channel in channels if channel in TRACKMATE_CHANNELS]
    if not requested:
        return {}

    output = {
        channel: np.zeros((frames, rows, cols), dtype=np.float32)
        for channel in requested
    }
    spots = load_spots(record.spots_path)
    if not spots:
        return output

    by_id = {str(spot["id"]): spot for spot in spots}
    valid_spots = [spot for spot in spots if 0 <= int(spot["frame"]) < frames]

    if "spot_count" in output or "mean_area" in output:
        area_sums = np.zeros((frames, rows, cols), dtype=np.float32)
        area_counts = np.zeros((frames, rows, cols), dtype=np.float32)
        for spot in valid_spots:
            frame = int(spot["frame"])
            row, col = spot_cell(spot, grid_size, rows, cols)
            if "spot_count" in output:
                output["spot_count"][frame, row, col] += 1.0
            area = float(spot["area"])
            if area > 0:
                area_sums[frame, row, col] += area
                area_counts[frame, row, col] += 1.0
        if "mean_area" in output:
            np.divide(area_sums, area_counts, out=output["mean_area"], where=area_counts > 0)

    by_track: dict[str, list[dict[str, float | int | str]]] = defaultdict(list)
    for spot in valid_spots:
        track_id = str(spot["track_id"])
        if track_id and track_id.lower() != "nan":
            by_track[track_id].append(spot)

    if {"flow_x", "flow_y"} & set(output):
        flow_counts = np.zeros((frames, rows, cols), dtype=np.float32)
        for track_spots in by_track.values():
            ordered = sorted(track_spots, key=lambda item: (int(item["frame"]), str(item["id"])))
            for current, nxt in zip(ordered, ordered[1:]):
                frame = int(current["frame"])
                if frame < 0 or frame >= frames:
                    continue
                row, col = spot_cell(current, grid_size, rows, cols)
                dx = (float(nxt["x"]) - float(current["x"])) / float(grid_size)
                dy = (float(nxt["y"]) - float(current["y"])) / float(grid_size)
                if "flow_x" in output:
                    output["flow_x"][frame, row, col] += dx
                if "flow_y" in output:
                    output["flow_y"][frame, row, col] += dy
                flow_counts[frame, row, col] += 1.0
        for channel in ("flow_x", "flow_y"):
            if channel in output:
                np.divide(output[channel], flow_counts, out=output[channel], where=flow_counts > 0)

    if "birth_count" in output or "death_count" in output:
        for track_spots in by_track.values():
            ordered = sorted(track_spots, key=lambda item: (int(item["frame"]), str(item["id"])))
            if "birth_count" in output and ordered:
                spot = ordered[0]
                frame = int(spot["frame"])
                row, col = spot_cell(spot, grid_size, rows, cols)
                output["birth_count"][frame, row, col] += 1.0
            if "death_count" in output and ordered:
                spot = ordered[-1]
                frame = int(spot["frame"])
                row, col = spot_cell(spot, grid_size, rows, cols)
                output["death_count"][frame, row, col] += 1.0

    if "split_count" in output:
        for source_id in load_split_source_ids(record.edges_path):
            spot = by_id.get(source_id)
            if spot is None:
                continue
            frame = int(spot["frame"])
            if not 0 <= frame < frames:
                continue
            row, col = spot_cell(spot, grid_size, rows, cols)
            output["split_count"][frame, row, col] += 1.0

    return output


def build_channel_stack(
    record: SequenceRecord,
    channels: Sequence[str],
    grid_size: int,
    occupancy_threshold: float,
    frame_align: str,
) -> tuple[np.ndarray, dict[str, int | tuple[int, int] | None]]:
    raw_info = image_info(record.raw_path)
    target_frames = int(raw_info["frames"]) if raw_info is not None else None
    density = compute_density_stack(record.mask_path, grid_size, target_frames, frame_align)
    frames, rows, cols = density.shape

    channel_arrays: dict[str, np.ndarray] = {}
    if "density" in channels:
        channel_arrays["density"] = density
    if "occupancy" in channels:
        channel_arrays["occupancy"] = (density > occupancy_threshold).astype(np.float32)

    channel_arrays.update(compute_trackmate_channels(record, channels, frames, rows, cols, grid_size))
    stacked = np.stack([channel_arrays[channel] for channel in channels], axis=-1).astype(np.float32)
    return stacked, raw_info


def channel_stats(array: np.ndarray, channels: Sequence[str], occupancy_threshold: float) -> dict[str, float | int]:
    stats: dict[str, float | int] = {
        "frames": int(array.shape[0]),
        "rows": int(array.shape[1]),
        "cols": int(array.shape[2]),
        "channels": int(array.shape[3]),
    }
    for index, channel in enumerate(channels):
        values = array[..., index]
        prefix = f"{channel}_"
        stats[prefix + "min"] = float(np.min(values))
        stats[prefix + "max"] = float(np.max(values))
        stats[prefix + "mean"] = float(np.mean(values))
        stats[prefix + "zero_frac"] = float(np.mean(values == 0))
        if channel in {"density", "occupancy"}:
            stats[prefix + "nonzero_frac"] = float(np.mean(values > 0))
            if occupancy_threshold <= 0:
                stats[prefix + "positive_threshold_frac"] = float(np.mean(values > 0))
            else:
                stats[prefix + "positive_threshold_frac"] = float(np.mean(values >= occupancy_threshold))
            stats[prefix + "eq_one_frac"] = float(np.mean(values == 1))
            stats[prefix + "empty_frames"] = int(np.sum(values.reshape(values.shape[0], -1).sum(axis=1) == 0))
    if array.shape[0] > 1:
        delta = np.abs(np.diff(array[..., 0], axis=0))
        stats["primary_mean_abs_delta"] = float(np.mean(delta))
        stats["primary_changed_frac"] = float(np.mean(delta > 0))
    else:
        stats["primary_mean_abs_delta"] = 0.0
        stats["primary_changed_frac"] = 0.0
    return stats


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_manifest_jsonl(path: Path, records: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_qc_csv(path: Path, records: Sequence[dict[str, object]]) -> None:
    if not records:
        return
    fieldnames: list[str] = []
    for record in records:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def summarize(records: Sequence[dict[str, object]], channels: Sequence[str]) -> dict[str, object]:
    split_counts_payload = Counter(str(record.get("output_split", "")) for record in records)
    q_counts = Counter(record.get("q") for record in records if record.get("q") is not None)
    pos_counts = Counter(record.get("pos") for record in records if record.get("pos") is not None)
    missing_pos_q: list[list[int]] = []
    if pos_counts and q_counts:
        for pos in range(24):
            for q in range(4):
                if not any(record.get("pos") == pos and record.get("q") == q for record in records):
                    missing_pos_q.append([pos, q])

    density_means = [
        float(record["density_mean"])
        for record in records
        if "density_mean" in record
    ]
    density_zero = [
        float(record["density_zero_frac"])
        for record in records
        if "density_zero_frac" in record
    ]
    return {
        "files": len(records),
        "channels": list(channels),
        "split_counts": dict(sorted(split_counts_payload.items())),
        "pos_counts": {str(key): value for key, value in sorted(pos_counts.items())},
        "q_counts": {str(key): value for key, value in sorted(q_counts.items())},
        "missing_pos_q_assuming_24x4": missing_pos_q,
        "density_mean_avg": float(np.mean(density_means)) if density_means else None,
        "density_zero_frac_avg": float(np.mean(density_zero)) if density_zero else None,
    }


def output_path_for(out_root: Path, output_split: str, name: str) -> Path:
    if output_split:
        return out_root / output_split / f"{name}.npy"
    return out_root / f"{name}.npy"


def build_maps(args: argparse.Namespace) -> int:
    source_root = Path(args.source_root)
    out_root = Path(args.out_root)
    channels = parse_channels(args.channels)

    if args.dataset == "dnn":
        records = discover_dynamic_nuclear_net(source_root)
    elif args.dataset == "pairs":
        records = discover_generic_pairs(source_root)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    if not records:
        raise FileNotFoundError(f"No input records found under {source_root}.")

    output_splits = assign_output_splits(records, args.split_policy, args.split_ratios, args.seed)
    print(f"Found {len(records)} records. channels={channels}. split_policy={args.split_policy}")

    if args.dry_run:
        for split, count in sorted(Counter(output_splits.values()).items()):
            print(f"{split or 'flat'}: {count}")
        return 0

    manifest_records: list[dict[str, object]] = []
    for index, record in enumerate(records, start=1):
        validate_record(record)
        output_split = output_splits[record.name]
        out_path = output_path_for(out_root, output_split, record.name)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and not args.overwrite:
            array = np.load(out_path, mmap_mode="r")
            if array.shape[-1] != len(channels):
                raise ValueError(
                    f"{out_path} has {array.shape[-1]} channels, but current channel config expects "
                    f"{len(channels)}. Use --overwrite or a different --out-root."
                )
            raw_info = image_info(record.raw_path)
            print(f"[{index}/{len(records)}] reuse existing {out_path} shape={array.shape}")
        else:
            array, raw_info = build_channel_stack(
                record=record,
                channels=channels,
                grid_size=args.grid_size,
                occupancy_threshold=args.occupancy_threshold,
                frame_align=args.frame_align,
            )
            np.save(out_path, array)
            print(f"[{index}/{len(records)}] wrote {out_path} shape={array.shape}")
        stats = channel_stats(array, channels, args.occupancy_threshold)
        row: dict[str, object] = {
            "dataset": record.dataset,
            "name": record.name,
            "source_split": record.source_split,
            "output_split": output_split,
            "group": record.group,
            "pos": record.pos,
            "q": record.q,
            "output_path": str(out_path),
            "raw_path": str(record.raw_path) if record.raw_path else None,
            "mask_path": str(record.mask_path),
            "spots_path": str(record.spots_path) if record.spots_path else None,
            "edges_path": str(record.edges_path) if record.edges_path else None,
            "tracks_path": str(record.tracks_path) if record.tracks_path else None,
            "metadata": read_key_value_metadata(record.metadata_path),
            "raw_info": raw_info,
            "grid_size": args.grid_size,
            "channel_names": list(channels),
            **stats,
        }
        manifest_records.append(row)

    write_manifest_jsonl(out_root / "manifest.jsonl", manifest_records)
    write_qc_csv(out_root / "qc.csv", manifest_records)
    write_json(out_root / "summary.json", summarize(manifest_records, channels))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build NCA-ready grid maps from HeLa TIFF masks and TrackMate CSV.")
    parser.add_argument("--source-root", required=True, help="DynamicNuclearNet tif_data root or a generic folder.")
    parser.add_argument("--out-root", required=True, help="Output folder for .npy, manifest.jsonl, qc.csv, summary.json.")
    parser.add_argument("--dataset", choices=["dnn", "pairs"], default="dnn")
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--channels", default="density", help="Comma list: density,occupancy,trackmate,all.")
    parser.add_argument("--occupancy-threshold", type=float, default=0.0)
    parser.add_argument("--split-policy", choices=["source", "by_group", "flat"], default="source")
    parser.add_argument("--split-ratios", type=float, nargs=3, default=(0.6, 0.2, 0.2))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame-align", choices=["strict", "truncate", "sample"], default="strict")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    return build_maps(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
