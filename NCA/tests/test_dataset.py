from pathlib import Path

import numpy as np
import torch

from NCA.dataset import NCADataset, build_dataloaders, nca_collate_fn


def _make_density_series(timesteps: int, height: int, width: int, offset: float) -> np.ndarray:
    base = np.arange(height * width, dtype=np.float32).reshape(height, width) / (height * width)
    frames = []
    for step in range(timesteps):
        frames.append(base + offset + step * 0.1)
    return np.stack(frames, axis=0)[..., None]


def _make_multichannel_series(
    timesteps: int,
    height: int,
    width: int,
    offset: float,
    channels: int,
) -> np.ndarray:
    base = _make_density_series(timesteps, height, width, offset=offset)
    features = [base[..., 0]]
    for channel in range(1, channels):
        features.append(base[..., 0] * (channel + 1) + channel * 0.05)
    return np.stack(features, axis=-1).astype(np.float32)


def _write_dataset(
    root: Path,
    file_count: int = 4,
    timesteps: int = 10,
    height: int = 5,
    width: int = 6,
    channels: int = 1,
) -> None:
    for file_id in range(file_count):
        array = _make_multichannel_series(timesteps, height, width, offset=file_id * 0.25, channels=channels)
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


def test_by_group_split_keeps_related_files_together(tmp_path: Path) -> None:
    for pos in range(4):
        for q in range(2):
            array = _make_multichannel_series(12, 4, 4, offset=pos + q * 0.1, channels=1)
            np.save(tmp_path / f"sample_pos{pos}_q{q}.npy", array)

    train = NCADataset(
        data_root=tmp_path,
        split="train",
        split_mode="by_group",
        split_ratios=(0.5, 0.25, 0.25),
        min_steps=2,
        max_steps=3,
        seed=11,
    )
    val = NCADataset(
        data_root=tmp_path,
        split="val",
        split_mode="by_group",
        split_ratios=(0.5, 0.25, 0.25),
        min_steps=2,
        max_steps=3,
        seed=11,
    )
    test = NCADataset(
        data_root=tmp_path,
        split="test",
        split_mode="by_group",
        split_ratios=(0.5, 0.25, 0.25),
        min_steps=2,
        max_steps=3,
        seed=11,
    )

    def groups(dataset: NCADataset) -> set[str]:
        file_ids = {sample.file_id for sample in dataset.indices}
        return {meta.group_key for meta in dataset.metadata if meta.file_id in file_ids}

    train_groups = groups(train)
    val_groups = groups(val)
    test_groups = groups(test)

    assert train_groups
    assert val_groups
    assert test_groups
    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)


def test_build_dataloaders_by_group_uses_consistent_split_seed(tmp_path: Path) -> None:
    for pos in range(6):
        for q in range(2):
            array = _make_multichannel_series(12, 4, 4, offset=pos + q * 0.1, channels=1)
            np.save(tmp_path / f"sample_pos{pos}_q{q}.npy", array)

    loaders, _ = build_dataloaders(
        data_root=tmp_path,
        pattern="*.npy",
        split_mode="by_group",
        split_ratios=(0.5, 0.25, 0.25),
        train_steps=(2, 3),
        eval_steps={"one_step": 1, "rollout": 2, "stochastic": 2},
        batch_size=2,
        seed=19,
    )

    def groups(loader_name: str) -> set[str]:
        dataset = loaders[loader_name].dataset
        file_ids = {sample.file_id for sample in dataset.indices}
        return {meta.group_key for meta in dataset.metadata if meta.file_id in file_ids}

    train_groups = groups("train")
    val_groups = groups("val_rollout")
    test_groups = groups("test_rollout")

    assert train_groups
    assert val_groups
    assert test_groups
    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)


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


def test_dataset_supports_multiple_observed_channels(tmp_path: Path) -> None:
    _write_dataset(tmp_path, file_count=2, timesteps=8, height=4, width=4, channels=2)
    dataset = NCADataset(
        data_root=tmp_path,
        split="train",
        split_mode="within_file",
        min_steps=2,
        max_steps=3,
        augment=False,
        seed=17,
        data_channels=2,
    )

    item = dataset[0]
    assert item["input_visible"].shape == (2, 4, 4)
    assert item["targets_visible"].shape[1:] == (2, 4, 4)

    collated = nca_collate_fn([dataset[0], dataset[1]])
    assert collated["input_visible"].shape == (2, 2, 4, 4)
