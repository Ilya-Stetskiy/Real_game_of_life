from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np


def _import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    return plt, animation


def to_numpy_visible(array) -> np.ndarray:
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    array = np.asarray(array)
    if array.ndim == 4:
        return array[0]
    if array.ndim == 3:
        return array[0]
    if array.ndim == 2:
        return array
    raise ValueError(f"Expected array rank 2, 3, or 4, got {array.ndim}.")


def select_visible_channel(array, channel_index: int = 0) -> np.ndarray:
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    array = np.asarray(array)
    if array.ndim == 4:
        return array[channel_index]
    if array.ndim == 3:
        return array[channel_index]
    if array.ndim == 2:
        if channel_index != 0:
            raise ValueError("channel_index must be 0 for rank-2 arrays.")
        return array
    raise ValueError(f"Expected array rank 2, 3, or 4, got {array.ndim}.")


def plot_triptych(
    input_visible,
    target_visible,
    prediction_visible,
    output_path: str | Path,
    title: Optional[str] = None,
    channel_index: int = 0,
) -> Path:
    plt, _ = _import_matplotlib()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figures = [
        ("Input", select_visible_channel(input_visible, channel_index=channel_index)),
        ("Target", select_visible_channel(target_visible, channel_index=channel_index)),
        ("Prediction", select_visible_channel(prediction_visible, channel_index=channel_index)),
    ]
    vmin = min(frame.min() for _, frame in figures)
    vmax = max(frame.max() for _, frame in figures)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis, (name, frame) in zip(axes, figures):
        image = axis.imshow(frame, cmap="viridis", vmin=vmin, vmax=vmax)
        axis.set_title(name)
        axis.axis("off")
    if title:
        fig.suptitle(title)
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_metric_curves(
    metric_history: Sequence[dict],
    output_path: str | Path,
    keys: Sequence[str],
) -> Path:
    plt, _ = _import_matplotlib()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(8, 4))
    epochs = [item.get("epoch", idx) for idx, item in enumerate(metric_history)]
    for key in keys:
        axis.plot(epochs, [item[key] for item in metric_history], label=key)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Metric")
    axis.legend()
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_uncertainty_heatmap(
    sampled_predictions,
    output_path: str | Path,
    visible_channel: int = 0,
    step_index: int = -1,
) -> Path:
    plt, _ = _import_matplotlib()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(sampled_predictions, "detach"):
        sampled_predictions = sampled_predictions.detach().cpu().numpy()
    sampled_predictions = np.asarray(sampled_predictions)
    frame = sampled_predictions[:, step_index, 0, visible_channel]
    std_map = frame.std(axis=0)

    fig, axis = plt.subplots(figsize=(4, 4))
    image = axis.imshow(std_map, cmap="magma")
    axis.set_title("Pixelwise std")
    axis.axis("off")
    fig.colorbar(image, ax=axis, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_rollout_animation(
    rollout_visible,
    output_path: str | Path,
    fps: int = 4,
    title: str = "NCA rollout",
    channel_index: int = 0,
) -> Path:
    plt, animation = _import_matplotlib()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(rollout_visible, "detach"):
        rollout_visible = rollout_visible.detach().cpu().numpy()
    rollout_visible = np.asarray(rollout_visible)
    if rollout_visible.ndim == 5:
        rollout_visible = rollout_visible[:, 0, channel_index]
    elif rollout_visible.ndim == 4:
        rollout_visible = rollout_visible[:, channel_index]

    fig, axis = plt.subplots(figsize=(4, 4))
    image = axis.imshow(rollout_visible[0], cmap="viridis", animated=True)
    axis.set_title(title)
    axis.axis("off")

    def update(frame_index: int):
        image.set_array(rollout_visible[frame_index])
        axis.set_title(f"{title} | step {frame_index + 1}")
        return [image]

    anim = animation.FuncAnimation(fig, update, frames=len(rollout_visible), interval=1000 / fps, blit=True)
    if output_path.suffix.lower() == ".gif":
        anim.save(output_path, writer="pillow", fps=fps)
    else:
        anim.save(output_path, fps=fps)
    plt.close(fig)
    return output_path
