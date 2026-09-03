from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class FieldGeometry:
    """Mapping between physical xy coordinates and latent field grid cells."""

    origin_xy: tuple[float, float] = (0.0, 0.0)
    cell_size: float = 1.0

    def __post_init__(self) -> None:
        if self.cell_size <= 0:
            raise ValueError("cell_size must be positive")


class LatentFieldRead(nn.Module):
    """Read a square latent-field patch around each cell coordinate."""

    def __init__(self, *, patch_radius: int, geometry: FieldGeometry | None = None) -> None:
        super().__init__()
        if patch_radius < 0:
            raise ValueError("patch_radius must be >= 0")
        self.patch_radius = int(patch_radius)
        self.geometry = geometry or FieldGeometry()

        offsets = torch.arange(-self.patch_radius, self.patch_radius + 1, dtype=torch.float32)
        yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")
        self.register_buffer("_offsets_xy", torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1), persistent=False)

    @property
    def patch_size(self) -> int:
        return 2 * self.patch_radius + 1

    def forward(self, field: Tensor, pos_xy: Tensor, batch_index: Tensor | None = None) -> Tensor:
        if field.dim() != 4:
            raise ValueError(f"Expected field shape [B, C, H, W], got {tuple(field.shape)}")
        if pos_xy.dim() != 2 or pos_xy.size(-1) != 2:
            raise ValueError(f"Expected pos_xy shape [N, 2], got {tuple(pos_xy.shape)}")

        batch_size, channels, height, width = field.shape
        if batch_index is None:
            if batch_size != 1:
                raise ValueError("batch_index is required when field batch size is greater than 1")
            batch_index = torch.zeros(pos_xy.size(0), dtype=torch.long, device=pos_xy.device)
        else:
            batch_index = batch_index.to(device=pos_xy.device, dtype=torch.long)
        if batch_index.numel() != pos_xy.size(0):
            raise ValueError("batch_index length must match pos_xy rows")

        grid_xy = physical_to_grid(pos_xy.float(), geometry=self.geometry)
        sample_xy = grid_xy[:, None, :] + self._offsets_xy.to(pos_xy.device)[None, :, :]
        normalized = grid_to_normalized(sample_xy, height=height, width=width)

        patches = field.new_zeros((pos_xy.size(0), channels, self.patch_size, self.patch_size))
        # TODO: vectorize grouped grid_sample for large CUDA batches; this loop is
        # acceptable for current graph batches but not optimized for throughput.
        for batch_id in batch_index.unique(sorted=True).tolist():
            cell_mask = batch_index == int(batch_id)
            if int(batch_id) < 0 or int(batch_id) >= batch_size:
                raise ValueError(f"batch_index contains out-of-range value: {batch_id}")
            grid = normalized[cell_mask].view(1, -1, 1, 2)
            sampled = F.grid_sample(
                field[int(batch_id):int(batch_id) + 1],
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            patches[cell_mask] = sampled.view(channels, int(cell_mask.sum()), self.patch_size, self.patch_size).permute(1, 0, 2, 3)
        return patches


class CellToFieldSplat(nn.Module):
    """Project cell features to latent field channels and bilinearly splat them."""

    def __init__(self, *, input_dim: int, field_channels: int, geometry: FieldGeometry | None = None) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be >= 1")
        if field_channels < 1:
            raise ValueError("field_channels must be >= 1")
        self.input_dim = int(input_dim)
        self.field_channels = int(field_channels)
        self.geometry = geometry or FieldGeometry()
        self.projection = nn.Linear(input_dim, field_channels)

    def forward(
        self,
        cell_features: Tensor,
        pos_xy: Tensor,
        *,
        field_shape: tuple[int, int, int, int],
        batch_index: Tensor | None = None,
    ) -> Tensor:
        if cell_features.dim() != 2 or cell_features.size(-1) != self.input_dim:
            raise ValueError(f"Expected cell_features shape [N, {self.input_dim}], got {tuple(cell_features.shape)}")
        if pos_xy.dim() != 2 or pos_xy.size(-1) != 2:
            raise ValueError(f"Expected pos_xy shape [N, 2], got {tuple(pos_xy.shape)}")
        if pos_xy.size(0) != cell_features.size(0):
            raise ValueError("pos_xy rows must match cell_features rows")

        batch_size, channels, height, width = field_shape
        if channels != self.field_channels:
            raise ValueError(f"field_shape channels must be {self.field_channels}, got {channels}")
        if batch_index is None:
            if batch_size != 1:
                raise ValueError("batch_index is required when field batch size is greater than 1")
            batch_index = torch.zeros(cell_features.size(0), dtype=torch.long, device=cell_features.device)
        else:
            batch_index = batch_index.to(device=cell_features.device, dtype=torch.long)

        writes = self.projection(cell_features.float())
        grid_xy = physical_to_grid(pos_xy.to(device=cell_features.device, dtype=torch.float32), geometry=self.geometry)
        return bilinear_splat(writes, grid_xy, batch_index, batch_size=batch_size, height=height, width=width)


class LatentFieldUpdate(nn.Module):
    """Local residual update for latent spatial field state."""

    def __init__(self, *, field_channels: int, write_channels: int, hidden_channels: int = 64, kernel_size: int = 3) -> None:
        super().__init__()
        if field_channels < 1:
            raise ValueError("field_channels must be >= 1")
        if write_channels < 1:
            raise ValueError("write_channels must be >= 1")
        if hidden_channels < 1:
            raise ValueError("hidden_channels must be >= 1")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

        padding = kernel_size // 2
        self.field_channels = int(field_channels)
        self.write_channels = int(write_channels)
        self.net = nn.Sequential(
            nn.Conv2d(field_channels + write_channels, hidden_channels, kernel_size=kernel_size, padding=padding),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, field_channels * 2, kernel_size=kernel_size, padding=padding),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, field: Tensor, write_map: Tensor) -> Tensor:
        if field.dim() != 4:
            raise ValueError(f"Expected field shape [B, C, H, W], got {tuple(field.shape)}")
        if write_map.dim() != 4:
            raise ValueError(f"Expected write_map shape [B, Cw, H, W], got {tuple(write_map.shape)}")
        if field.size(0) != write_map.size(0) or field.shape[-2:] != write_map.shape[-2:]:
            raise ValueError("field and write_map must share batch, height, and width")
        if field.size(1) != self.field_channels or write_map.size(1) != self.write_channels:
            raise ValueError("field/write_map channel count does not match module configuration")

        delta, gate_logits = self.net(torch.cat([field, write_map], dim=1)).chunk(2, dim=1)
        return field + torch.sigmoid(gate_logits) * delta


def physical_to_grid(pos_xy: Tensor, *, geometry: FieldGeometry) -> Tensor:
    origin = pos_xy.new_tensor(geometry.origin_xy)
    return (pos_xy - origin) / float(geometry.cell_size)


def field_coverage(
    pos_xy: Tensor,
    *,
    field_height: int,
    field_width: int,
    geometry: FieldGeometry,
) -> float:
    """Return the fraction of coordinates that fall inside the field grid."""

    if pos_xy.numel() == 0:
        return 1.0
    if pos_xy.dim() != 2 or pos_xy.size(-1) != 2:
        raise ValueError(f"Expected pos_xy shape [N, 2], got {tuple(pos_xy.shape)}")
    grid_xy = physical_to_grid(pos_xy.detach().float().cpu(), geometry=geometry)
    inside = (
        (grid_xy[:, 0] >= 0)
        & (grid_xy[:, 0] < field_width)
        & (grid_xy[:, 1] >= 0)
        & (grid_xy[:, 1] < field_height)
    )
    return float(inside.float().mean().item())


def grid_to_normalized(grid_xy: Tensor, *, height: int, width: int) -> Tensor:
    if height < 1 or width < 1:
        raise ValueError("height and width must be positive")
    x_denominator = max(width - 1, 1)
    y_denominator = max(height - 1, 1)
    normalized_x = 2.0 * grid_xy[..., 0] / float(x_denominator) - 1.0
    normalized_y = 2.0 * grid_xy[..., 1] / float(y_denominator) - 1.0
    return torch.stack([normalized_x, normalized_y], dim=-1)


def bilinear_splat(
    values: Tensor,
    grid_xy: Tensor,
    batch_index: Tensor,
    *,
    batch_size: int,
    height: int,
    width: int,
) -> Tensor:
    if values.dim() != 2:
        raise ValueError(f"Expected values shape [N, C], got {tuple(values.shape)}")
    if grid_xy.shape != (values.size(0), 2):
        raise ValueError("grid_xy must have shape [N, 2]")
    if batch_index.numel() != values.size(0):
        raise ValueError("batch_index length must match values rows")

    channels = values.size(1)
    out = values.new_zeros((batch_size * height * width, channels))
    x = grid_xy[:, 0]
    y = grid_xy[:, 1]
    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    dx = x - x0.to(x.dtype)
    dy = y - y0.to(y.dtype)

    for ox, oy, weight in (
        (0, 0, (1.0 - dx) * (1.0 - dy)),
        (1, 0, dx * (1.0 - dy)),
        (0, 1, (1.0 - dx) * dy),
        (1, 1, dx * dy),
    ):
        xi = x0 + ox
        yi = y0 + oy
        valid = (batch_index >= 0) & (batch_index < batch_size) & (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
        flat_index = (batch_index[valid] * height + yi[valid]) * width + xi[valid]
        out.index_add_(0, flat_index, values[valid] * weight[valid].unsqueeze(-1))

    return out.view(batch_size, height, width, channels).permute(0, 3, 1, 2).contiguous()
