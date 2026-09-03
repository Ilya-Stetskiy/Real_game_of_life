from __future__ import annotations

import torch
import pytest

from Real_game_of_life.GNN.spatial_field import (
    CellToFieldSplat,
    FieldGeometry,
    LatentFieldRead,
    LatentFieldUpdate,
    bilinear_splat,
    field_coverage,
)


def test_latent_field_read_returns_local_patch_at_physical_coordinates() -> None:
    field = torch.arange(25, dtype=torch.float32).view(1, 1, 5, 5)
    reader = LatentFieldRead(patch_radius=1, geometry=FieldGeometry(origin_xy=(0.0, 0.0), cell_size=1.0))

    patches = reader(field, torch.tensor([[2.0, 2.0]]))

    assert tuple(patches.shape) == (1, 1, 3, 3)
    assert patches[0, 0].tolist() == [
        [6.0, 7.0, 8.0],
        [11.0, 12.0, 13.0],
        [16.0, 17.0, 18.0],
    ]


def test_bilinear_splat_distributes_values_to_neighboring_cells() -> None:
    values = torch.tensor([[2.0]])
    grid_xy = torch.tensor([[1.5, 1.5]])
    batch_index = torch.tensor([0])

    splatted = bilinear_splat(values, grid_xy, batch_index, batch_size=1, height=4, width=4)

    assert tuple(splatted.shape) == (1, 1, 4, 4)
    assert splatted[0, 0, 1, 1].item() == 0.5
    assert splatted[0, 0, 1, 2].item() == 0.5
    assert splatted[0, 0, 2, 1].item() == 0.5
    assert splatted[0, 0, 2, 2].item() == 0.5
    assert splatted.sum().item() == 2.0


def test_cell_to_field_splat_projects_cell_features() -> None:
    splat = CellToFieldSplat(input_dim=2, field_channels=1)
    with torch.no_grad():
        splat.projection.weight[:] = torch.tensor([[1.0, 2.0]])
        splat.projection.bias.zero_()

    write_map = splat(
        torch.tensor([[1.0, 3.0]]),
        torch.tensor([[2.0, 1.0]]),
        field_shape=(1, 1, 4, 4),
    )

    assert tuple(write_map.shape) == (1, 1, 4, 4)
    assert write_map[0, 0, 1, 2].item() == 7.0
    assert write_map.sum().item() == 7.0


def test_latent_field_update_is_initially_noop_and_preserves_shape() -> None:
    update = LatentFieldUpdate(field_channels=2, write_channels=3, hidden_channels=4)
    field = torch.randn(1, 2, 5, 6)
    write_map = torch.randn(1, 3, 5, 6)

    next_field = update(field, write_map)

    assert tuple(next_field.shape) == tuple(field.shape)
    assert torch.allclose(next_field, field)


def test_field_coverage_accounts_for_cell_size_and_origin() -> None:
    coords = torch.tensor([[0.0, 0.0], [127.0, 127.0], [510.0, 510.0]])

    low = field_coverage(
        coords,
        field_height=128,
        field_width=128,
        geometry=FieldGeometry(origin_xy=(0.0, 0.0), cell_size=1.0),
    )
    high = field_coverage(
        coords,
        field_height=128,
        field_width=128,
        geometry=FieldGeometry(origin_xy=(0.0, 0.0), cell_size=4.0),
    )
    shifted = field_coverage(
        torch.tensor([[100.0, 100.0], [104.0, 104.0]]),
        field_height=2,
        field_width=2,
        geometry=FieldGeometry(origin_xy=(100.0, 100.0), cell_size=4.0),
    )

    assert low == pytest.approx(2 / 3)
    assert high == 1.0
    assert shifted == 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_bilinear_splat_cuda_smoke() -> None:
    values = torch.tensor([[2.0]], device="cuda")
    grid_xy = torch.tensor([[1.5, 1.5]], device="cuda")
    batch_index = torch.tensor([0], device="cuda")

    splatted = bilinear_splat(values, grid_xy, batch_index, batch_size=1, height=4, width=4)

    assert splatted.device.type == "cuda"
    assert torch.allclose(splatted.sum(), torch.tensor(2.0, device="cuda"))
