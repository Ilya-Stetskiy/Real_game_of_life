from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def discover_npy_files(data_root: Path | str, pattern: str = "**/*.npy") -> List[Path]:
    """Return sorted .npy files under the given relative root."""
    root = Path(data_root)
    files = sorted(path for path in root.glob(pattern) if path.is_file())
    if not files:
        raise FileNotFoundError(f"No .npy files found under {root} with pattern {pattern!r}.")
    return files


def _split_counts(total: int, split_ratios: Sequence[float]) -> Tuple[int, int, int]:
    if len(split_ratios) != 3:
        raise ValueError("split_ratios must have exactly 3 values: train, val, test.")
    if any(r < 0 for r in split_ratios):
        raise ValueError("split ratios must be non-negative.")
    ratio_sum = sum(split_ratios)
    if ratio_sum <= 0:
        raise ValueError("split ratios must sum to a positive value.")

    normalized = [ratio / ratio_sum for ratio in split_ratios]
    counts = [int(math.floor(total * ratio)) for ratio in normalized]
    remainder = total - sum(counts)
    for index in range(remainder):
        counts[index % 3] += 1
    return counts[0], counts[1], counts[2]


@dataclass(frozen=True)
class FileMetadata:
    path: Path
    file_id: int
    timesteps: int
    height: int
    width: int
    channels: int


@dataclass(frozen=True)
class SampleIndex:
    file_id: int
    t0: int


class VisibleChannelNormalizer:
    """Keep density raw and normalize optional visible auxiliary channels."""

    def __init__(self, means: Optional[np.ndarray] = None, stds: Optional[np.ndarray] = None):
        self.means = means
        self.stds = stds

    @property
    def enabled(self) -> bool:
        return self.means is not None and self.stds is not None and self.means.size > 0

    def fit(self, files: Sequence[Path]) -> "VisibleChannelNormalizer":
        if not files:
            self.means = np.empty(0, dtype=np.float32)
            self.stds = np.empty(0, dtype=np.float32)
            return self

        accum_sum = None
        accum_sq = None
        count = 0
        extra_channels = None

        for path in files:
            array = np.load(path, mmap_mode="r")
            validate_array_shape(array, source=str(path))
            if array.shape[-1] <= 1:
                continue

            extras = np.asarray(array[..., 1:], dtype=np.float64)
            extra_channels = extras.shape[-1]
            flattened = extras.reshape(-1, extra_channels)
            sum_values = flattened.sum(axis=0)
            sq_values = np.square(flattened).sum(axis=0)

            if accum_sum is None:
                accum_sum = sum_values
                accum_sq = sq_values
            else:
                accum_sum += sum_values
                accum_sq += sq_values
            count += flattened.shape[0]

        if accum_sum is None or count == 0 or extra_channels is None:
            self.means = np.empty(0, dtype=np.float32)
            self.stds = np.empty(0, dtype=np.float32)
            return self

        means = accum_sum / count
        variances = np.maximum(accum_sq / count - np.square(means), 1e-8)
        stds = np.sqrt(variances)
        self.means = means.astype(np.float32)
        self.stds = stds.astype(np.float32)
        return self

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return tensor

        if tensor.shape[-3] <= 1:
            return tensor

        means = torch.as_tensor(self.means, dtype=tensor.dtype, device=tensor.device)
        stds = torch.as_tensor(self.stds, dtype=tensor.dtype, device=tensor.device)
        if tensor.ndim == 3:
            tensor[1:] = (tensor[1:] - means[:, None, None]) / stds[:, None, None]
        elif tensor.ndim == 4:
            tensor[:, 1:] = (tensor[:, 1:] - means[None, :, None, None]) / stds[None, :, None, None]
        else:
            raise ValueError(f"Unsupported tensor rank for normalization: {tensor.ndim}")
        return tensor

    def state_dict(self) -> Dict[str, Optional[List[float]]]:
        return {
            "means": None if self.means is None else self.means.tolist(),
            "stds": None if self.stds is None else self.stds.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Optional[List[float]]]) -> "VisibleChannelNormalizer":
        means = state.get("means")
        stds = state.get("stds")
        return cls(
            means=None if means is None else np.asarray(means, dtype=np.float32),
            stds=None if stds is None else np.asarray(stds, dtype=np.float32),
        )


def validate_array_shape(array: np.ndarray, source: str = "<array>") -> None:
    if array.ndim != 4:
        raise ValueError(f"{source} must have shape [T, H, W, F], got {array.shape}.")
    if array.shape[0] < 2:
        raise ValueError(f"{source} must contain at least 2 timesteps, got {array.shape[0]}.")
    if array.shape[-1] < 1:
        raise ValueError(f"{source} must contain at least 1 feature channel, got {array.shape[-1]}.")


def fit_normalizer_for_train_split(
    files: Sequence[Path],
    split_ratios: Sequence[float],
    split_mode: str,
    seed: int = 0,
) -> VisibleChannelNormalizer:
    """Fit visible auxiliary-channel stats using train-only data."""
    normalizer = VisibleChannelNormalizer()
    if not files:
        return normalizer

    if split_mode == "by_file":
        rng = np.random.default_rng(seed)
        file_ids = np.arange(len(files))
        rng.shuffle(file_ids)
        train_count, _, _ = _split_counts(len(file_ids), split_ratios)
        train_ids = set(file_ids[:train_count].tolist())
        return normalizer.fit([path for idx, path in enumerate(files) if idx in train_ids])

    accum_sum = None
    accum_sq = None
    count = 0

    for path in files:
        array = np.load(path, mmap_mode="r")
        validate_array_shape(array, source=str(path))
        if array.shape[-1] <= 1:
            continue

        train_count, _, _ = _split_counts(array.shape[0], split_ratios)
        if train_count <= 0:
            continue

        extras = np.asarray(array[:train_count, ..., 1:], dtype=np.float64)
        flattened = extras.reshape(-1, extras.shape[-1])
        sum_values = flattened.sum(axis=0)
        sq_values = np.square(flattened).sum(axis=0)

        if accum_sum is None:
            accum_sum = sum_values
            accum_sq = sq_values
        else:
            accum_sum += sum_values
            accum_sq += sq_values
        count += flattened.shape[0]

    if accum_sum is None or count == 0:
        return normalizer.fit([])

    means = accum_sum / count
    variances = np.maximum(accum_sq / count - np.square(means), 1e-8)
    normalizer.means = means.astype(np.float32)
    normalizer.stds = np.sqrt(variances).astype(np.float32)
    return normalizer


class NCADataset(Dataset):
    """Dataset of rollout snippets sampled from one or more trajectories."""

    def __init__(
        self,
        data_root: Path | str,
        pattern: str = "**/*.npy",
        split: str = "train",
        split_ratios: Sequence[float] = (0.8, 0.1, 0.1),
        split_mode: str = "within_file",
        min_steps: int = 4,
        max_steps: int = 16,
        augment: bool = False,
        normalizer: Optional[VisibleChannelNormalizer] = None,
        seed: int = 0,
        files: Optional[Sequence[Path]] = None,
        cache_arrays: bool = True,
    ) -> None:
        super().__init__()
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of 'train', 'val', 'test'.")
        if split_mode not in {"within_file", "by_file"}:
            raise ValueError("split_mode must be 'within_file' or 'by_file'.")
        if min_steps < 1 or max_steps < min_steps:
            raise ValueError("Expected 1 <= min_steps <= max_steps.")

        self.data_root = Path(data_root)
        self.pattern = pattern
        self.split = split
        self.split_ratios = tuple(float(r) for r in split_ratios)
        self.split_mode = split_mode
        self.min_steps = int(min_steps)
        self.max_steps = int(max_steps)
        self.augment = bool(augment)
        self.normalizer = normalizer or VisibleChannelNormalizer()
        self.seed = int(seed)
        self.cache_arrays = bool(cache_arrays)

        if files is None:
            file_paths = discover_npy_files(self.data_root, self.pattern)
        else:
            file_paths = [Path(path) for path in files]
            if not file_paths:
                raise ValueError("files cannot be empty.")

        self.file_paths = file_paths
        self.metadata = self._inspect_files(file_paths)
        self.indices = self._build_indices()
        self._array_cache: Dict[int, np.ndarray] = {}

        if not self.indices:
            raise ValueError(
                f"No valid samples found for split={self.split!r}, split_mode={self.split_mode!r}, "
                f"max_steps={self.max_steps}."
            )

    def _inspect_files(self, file_paths: Sequence[Path]) -> List[FileMetadata]:
        metadata: List[FileMetadata] = []
        reference_hwf: Optional[Tuple[int, int, int]] = None

        for file_id, path in enumerate(file_paths):
            array = np.load(path, mmap_mode="r")
            validate_array_shape(array, source=str(path))
            timesteps, height, width, channels = array.shape
            if reference_hwf is None:
                reference_hwf = (height, width, channels)
            elif reference_hwf != (height, width, channels):
                raise ValueError(
                    "All files must share the same [H, W, F] shape for batching, "
                    f"expected {reference_hwf}, got {(height, width, channels)} from {path}."
                )

            metadata.append(
                FileMetadata(
                    path=path,
                    file_id=file_id,
                    timesteps=timesteps,
                    height=height,
                    width=width,
                    channels=channels,
                )
            )
        return metadata

    @property
    def visible_channels(self) -> int:
        return self.metadata[0].channels

    def train_files(self) -> List[Path]:
        if self.split_mode == "by_file":
            selected_ids = {index.file_id for index in self.indices}
            return [meta.path for meta in self.metadata if meta.file_id in selected_ids]
        return [meta.path for meta in self.metadata]

    def _build_indices(self) -> List[SampleIndex]:
        if self.split_mode == "within_file":
            return self._build_within_file_indices()
        return self._build_by_file_indices()

    def _build_within_file_indices(self) -> List[SampleIndex]:
        split_lookup = {"train": 0, "val": 1, "test": 2}
        requested = split_lookup[self.split]
        indices: List[SampleIndex] = []

        for meta in self.metadata:
            train_count, val_count, test_count = _split_counts(meta.timesteps, self.split_ratios)
            boundaries = [0, train_count, train_count + val_count, train_count + val_count + test_count]
            split_start = boundaries[requested]
            split_end = boundaries[requested + 1]
            last_start = split_end - self.max_steps - 1

            if last_start < split_start:
                continue

            for t0 in range(split_start, last_start + 1):
                indices.append(SampleIndex(file_id=meta.file_id, t0=t0))
        return indices

    def _build_by_file_indices(self) -> List[SampleIndex]:
        rng = np.random.default_rng(self.seed)
        file_ids = np.arange(len(self.metadata))
        rng.shuffle(file_ids)
        train_count, val_count, test_count = _split_counts(len(file_ids), self.split_ratios)
        split_ids = {
            "train": set(file_ids[:train_count].tolist()),
            "val": set(file_ids[train_count:train_count + val_count].tolist()),
            "test": set(file_ids[train_count + val_count:train_count + val_count + test_count].tolist()),
        }

        indices: List[SampleIndex] = []
        for meta in self.metadata:
            if meta.file_id not in split_ids[self.split]:
                continue
            max_start = meta.timesteps - self.max_steps - 1
            if max_start < 0:
                continue
            for t0 in range(max_start + 1):
                indices.append(SampleIndex(file_id=meta.file_id, t0=t0))
        return indices

    def _load_array(self, file_id: int) -> np.ndarray:
        if file_id in self._array_cache:
            return self._array_cache[file_id]

        array = np.load(self.metadata[file_id].path).astype(np.float32, copy=False)
        if self.cache_arrays:
            self._array_cache[file_id] = array
        return array

    def _sample_horizon(self, index: int) -> int:
        return int(np.random.randint(self.min_steps, self.max_steps + 1))

    def _sample_augmentation(self, index: int) -> Tuple[int, bool]:
        rotation = int(np.random.randint(0, 4))
        flip = bool(np.random.randint(0, 2))
        return rotation, flip

    def _apply_augmentation(self, tensor: torch.Tensor, rotation: int, flip: bool) -> torch.Tensor:
        if rotation:
            tensor = torch.rot90(tensor, k=rotation, dims=(-2, -1))
        if flip:
            tensor = torch.flip(tensor, dims=(-1,))
        return tensor

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | int]:
        sample = self.indices[index]
        horizon = self._sample_horizon(index)
        array = self._load_array(sample.file_id)

        visible = torch.from_numpy(array[sample.t0].copy()).permute(2, 0, 1).contiguous().float()
        targets = (
            torch.from_numpy(array[sample.t0 + 1:sample.t0 + horizon + 1].copy())
            .permute(0, 3, 1, 2)
            .contiguous()
            .float()
        )

        visible = self.normalizer.transform(visible)
        targets = self.normalizer.transform(targets)

        if self.augment:
            rotation, flip = self._sample_augmentation(index)
            visible = self._apply_augmentation(visible, rotation, flip)
            targets = self._apply_augmentation(targets, rotation, flip)

        return {
            "input_visible": visible,
            "targets_visible": targets,
            "horizon": horizon,
            "file_id": sample.file_id,
            "t0": sample.t0,
        }


def nca_collate_fn(batch: Sequence[Dict[str, torch.Tensor | int]]) -> Dict[str, torch.Tensor]:
    input_visible = torch.stack([item["input_visible"] for item in batch], dim=0)
    horizons = torch.as_tensor([item["horizon"] for item in batch], dtype=torch.long)
    file_ids = torch.as_tensor([item["file_id"] for item in batch], dtype=torch.long)
    t0 = torch.as_tensor([item["t0"] for item in batch], dtype=torch.long)

    max_horizon = int(horizons.max().item())
    batch_size = len(batch)
    channels, height, width = input_visible.shape[1:]
    targets = torch.zeros((batch_size, max_horizon, channels, height, width), dtype=input_visible.dtype)
    target_mask = torch.zeros((batch_size, max_horizon), dtype=torch.bool)

    for row, item in enumerate(batch):
        current = item["targets_visible"]
        current_horizon = current.shape[0]
        targets[row, :current_horizon] = current
        target_mask[row, :current_horizon] = True

    return {
        "input_visible": input_visible,
        "targets_visible": targets,
        "target_mask": target_mask,
        "horizons": horizons,
        "file_ids": file_ids,
        "t0": t0,
    }


def build_dataloaders(
    data_root: Path | str,
    pattern: str = "**/*.npy",
    split_mode: str = "within_file",
    split_ratios: Sequence[float] = (0.8, 0.1, 0.1),
    train_steps: Tuple[int, int] = (4, 16),
    eval_steps: Optional[Dict[str, int]] = None,
    batch_size: int = 16,
    eval_batch_size: Optional[int] = None,
    seed: int = 0,
    num_workers: int = 0,
    cache_arrays: bool = True,
) -> Tuple[Dict[str, DataLoader], VisibleChannelNormalizer]:
    """Build train/val/test loaders for training and deterministic/stochastic eval."""
    root = Path(data_root)
    files = discover_npy_files(root, pattern)
    normalizer = fit_normalizer_for_train_split(
        files=files,
        split_ratios=split_ratios,
        split_mode=split_mode,
        seed=seed,
    )

    eval_steps = eval_steps or {"one_step": 1, "rollout": train_steps[1], "stochastic": train_steps[1]}
    eval_batch_size = eval_batch_size or batch_size

    dataset_configs = {
        "train": dict(
            split="train",
            min_steps=train_steps[0],
            max_steps=train_steps[1],
            augment=True,
            seed=seed,
            batch_size=batch_size,
            shuffle=True,
        ),
        "val_one_step": dict(
            split="val",
            min_steps=eval_steps["one_step"],
            max_steps=eval_steps["one_step"],
            augment=False,
            seed=seed + 1,
            batch_size=eval_batch_size,
            shuffle=False,
        ),
        "val_rollout": dict(
            split="val",
            min_steps=eval_steps["rollout"],
            max_steps=eval_steps["rollout"],
            augment=False,
            seed=seed + 2,
            batch_size=eval_batch_size,
            shuffle=False,
        ),
        "test_rollout": dict(
            split="test",
            min_steps=eval_steps["stochastic"],
            max_steps=eval_steps["stochastic"],
            augment=False,
            seed=seed + 3,
            batch_size=eval_batch_size,
            shuffle=False,
        ),
    }

    loaders: Dict[str, DataLoader] = {}
    for name, config in dataset_configs.items():
        try:
            dataset = NCADataset(
                data_root=root,
                pattern=pattern,
                split=config["split"],
                split_ratios=split_ratios,
                split_mode=split_mode,
                min_steps=config["min_steps"],
                max_steps=config["max_steps"],
                augment=config["augment"],
                normalizer=normalizer,
                seed=config["seed"],
                files=files,
                cache_arrays=cache_arrays,
            )
        except ValueError:
            if name == "train":
                raise
            continue

        loaders[name] = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=config["shuffle"],
            num_workers=num_workers,
            collate_fn=nca_collate_fn,
        )
    return loaders, normalizer
