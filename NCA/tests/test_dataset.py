from pathlib import Path

import numpy as np
import torch

from NCA.dataset import NCADataset, nca_collate_fn


def _make_density_series(timesteps: int, height: int, width: int, offset: float) -> np.ndarray:
    base = np.arange(height * width, dtype=np.float32).reshape(height, width) / (height * width)
    frames = []
    for step in range(timesteps):
        frames.append(base + offset + step * 0.1)
    return np.stack(frames, axis=0)[..., None]


def _write_dataset(root: Path, file_count: int = 4, timesteps: int = 10, height: int = 5, width: int = 6) -> None:
    for file_id in range(file_count):
        array = _make_density_series(timesteps, height, width, offset=file_id * 0.25)
        np.save(root / f"traj_{file_id}.npy", array.astype(np.float32))


def test_dataset_shapes_and_collate(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    dataset = NCADataset(
        data_root=tmp_path,
        split="train",
        split_mode="within_file",
        min_steps=2,
        max_steps=4,
        augment=False,
        seed=7,
    )

    item = dataset[0]
    assert item["input_visible"].shape == (1, 5, 6)
    assert item["targets_visible"].shape[1:] == (1, 5, 6)
    assert 2 <= item["horizon"] <= 4

    collated = nca_collate_fn([dataset[0], dataset[1]])
    assert collated["input_visible"].shape == (2, 1, 5, 6)
    assert collated["targets_visible"].shape[0] == 2
    assert collated["target_mask"].dtype == torch.bool


def test_by_file_split_has_disjoint_file_ids(tmp_path: Path) -> None:
    _write_dataset(tmp_path, file_count=6, timesteps=12)
    train = NCADataset(
        data_root=tmp_path,
        split="train",
        split_mode="by_file",
        split_ratios=(0.5, 0.25, 0.25),
        min_steps=2,
        max_steps=3,
        seed=11,
    )
    val = NCADataset(
        data_root=tmp_path,
        split="val",
        split_mode="by_file",
        split_ratios=(0.5, 0.25, 0.25),
        min_steps=2,
        max_steps=3,
        seed=11,
    )

    train_ids = {sample.file_id for sample in train.indices}
    val_ids = {sample.file_id for sample in val.indices}
    assert train_ids
    assert val_ids
    assert train_ids.isdisjoint(val_ids)


def test_within_file_split_is_time_disjoint(tmp_path: Path) -> None:
    _write_dataset(tmp_path, file_count=1, timesteps=20)
    train = NCADataset(
        data_root=tmp_path,
        split="train",
        split_mode="within_file",
        split_ratios=(0.5, 0.25, 0.25),
        min_steps=2,
        max_steps=3,
        seed=5,
    )
    val = NCADataset(
        data_root=tmp_path,
        split="val",
        split_mode="within_file",
        split_ratios=(0.5, 0.25, 0.25),
        min_steps=2,
        max_steps=3,
        seed=5,
    )

    train_times = {sample.t0 for sample in train.indices}
    val_times = {sample.t0 for sample in val.indices}
    assert train_times.isdisjoint(val_times)


def test_augmentation_keeps_input_target_alignment(tmp_path: Path) -> None:
    _write_dataset(tmp_path, file_count=2, timesteps=8, height=4, width=4)
    dataset = NCADataset(
        data_root=tmp_path,
        split="train",
        split_mode="within_file",
        min_steps=2,
        max_steps=2,
        augment=True,
        seed=13,
    )

    item = dataset[0]
    delta = item["targets_visible"][0] - item["input_visible"]
    expected = torch.full_like(delta, 0.1)
    assert torch.allclose(delta, expected)
