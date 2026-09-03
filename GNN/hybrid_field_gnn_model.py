from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from .gnn_layers import MLP
from .gnn_model import CellGNNOutput, CellInteractionGNN
from .spatial_field import CellToFieldSplat, FieldGeometry, LatentFieldRead, LatentFieldUpdate


@dataclass
class FieldConditionedGNNOutput:
    """Cell predictions plus the next latent spatial field state."""

    cell_output: CellGNNOutput
    field_next: Tensor
    field_context: Tensor


class FieldConditionedCellGNN(nn.Module):
    """GNN cell dynamics model conditioned on a latent spatial field."""

    def __init__(
        self,
        *,
        node_dim: int,
        edge_dim: int,
        shape_dim: int,
        field_channels: int,
        field_patch_radius: int = 1,
        field_context_dim: Optional[int] = None,
        hidden_dim: int = 128,
        num_message_passing_layers: int = 4,
        dropout: float = 0.1,
        use_temporal_state: bool = False,
        num_trajectory_hypotheses: int = 1,
        num_division_horizons: int = 0,
        field_update_hidden_channels: int = 64,
        field_geometry: FieldGeometry | None = None,
        train_field_update: bool = True,
    ) -> None:
        super().__init__()
        if field_channels < 1:
            raise ValueError("field_channels must be >= 1")
        if num_trajectory_hypotheses < 1:
            raise ValueError("num_trajectory_hypotheses must be >= 1")
        if num_division_horizons < 0:
            raise ValueError("num_division_horizons must be >= 0")

        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        self.shape_dim = int(shape_dim)
        self.hidden_dim = int(hidden_dim)
        self.field_channels = int(field_channels)
        self.field_patch_radius = int(field_patch_radius)
        self.field_context_dim = int(field_context_dim or hidden_dim)
        self.num_trajectory_hypotheses = int(num_trajectory_hypotheses)
        self.num_division_horizons = int(num_division_horizons)

        self.gnn_backbone = CellInteractionGNN(
            node_dim=node_dim,
            edge_dim=edge_dim,
            shape_dim=shape_dim,
            hidden_dim=hidden_dim,
            num_message_passing_layers=num_message_passing_layers,
            dropout=dropout,
            use_temporal_state=use_temporal_state,
            num_trajectory_hypotheses=num_trajectory_hypotheses,
            num_division_horizons=num_division_horizons,
        )
        self.field_reader = LatentFieldRead(patch_radius=field_patch_radius, geometry=field_geometry)
        patch_dim = field_channels * self.field_reader.patch_size * self.field_reader.patch_size
        self.field_context_encoder = MLP(
            patch_dim,
            self.field_context_dim,
            self.field_context_dim,
            num_layers=2,
            dropout=dropout,
            layer_norm=True,
        )
        self.decoder_trunk = MLP(
            hidden_dim + self.field_context_dim,
            hidden_dim,
            hidden_dim,
            num_layers=2,
            dropout=dropout,
            layer_norm=True,
        )
        self.delta_pos_head = nn.Linear(hidden_dim, 2)
        self.delta_shape_head = nn.Linear(hidden_dim, shape_dim)
        self.division_head = nn.Linear(hidden_dim, 1)
        self.death_head = nn.Linear(hidden_dim, 1)
        self.division_horizon_head = (
            nn.Linear(hidden_dim, num_division_horizons)
            if num_division_horizons > 0
            else None
        )
        self.trajectory_head = (
            nn.Linear(hidden_dim, num_trajectory_hypotheses * 2)
            if num_trajectory_hypotheses > 1
            else None
        )
        self.field_writer = CellToFieldSplat(
            input_dim=hidden_dim,
            field_channels=field_channels,
            geometry=field_geometry,
        )
        self.field_update = LatentFieldUpdate(
            field_channels=field_channels,
            write_channels=field_channels,
            hidden_channels=field_update_hidden_channels,
        )
        self.train_field_update = bool(train_field_update)
        if not self.train_field_update:
            for module in (self.field_writer, self.field_update):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)

    def forward(self, data: Data, field: Tensor, h_prev: Optional[Tensor] = None) -> FieldConditionedGNNOutput:
        if not hasattr(data, "pos_xy") or data.pos_xy is None:
            raise ValueError("Data object must contain data.pos_xy with physical cell coordinates")
        if field.dim() != 4:
            raise ValueError(f"Expected field shape [B, C, H, W], got {tuple(field.shape)}")
        if field.size(1) != self.field_channels:
            raise ValueError(f"Expected field_channels={self.field_channels}, got {field.size(1)}")

        gnn_output = self.gnn_backbone(data, h_prev=h_prev)
        pos_xy = data.pos_xy.to(device=field.device, dtype=torch.float32)
        batch_index = getattr(data, "batch", None)
        if batch_index is not None:
            batch_index = batch_index.to(device=field.device)

        patches = self.field_reader(field, pos_xy, batch_index=batch_index)
        field_context = self.field_context_encoder(patches.flatten(start_dim=1))
        z = self.decoder_trunk(torch.cat([gnn_output.node_embeddings, field_context], dim=-1))

        trajectory_hypotheses = None
        if self.trajectory_head is not None:
            raw = self.trajectory_head(z)
            trajectory_hypotheses = raw.view(raw.size(0), self.num_trajectory_hypotheses, 2)

        cell_output = CellGNNOutput(
            delta_pos=self.delta_pos_head(z),
            delta_shape=self.delta_shape_head(z),
            division_logits=self.division_head(z).squeeze(-1),
            death_logits=self.death_head(z).squeeze(-1),
            node_embeddings=gnn_output.node_embeddings,
            trajectory_hypotheses=trajectory_hypotheses,
            division_horizon_logits=(
                self.division_horizon_head(z)
                if self.division_horizon_head is not None
                else None
            ),
        )

        write_map = self.field_writer(
            gnn_output.node_embeddings,
            pos_xy,
            field_shape=tuple(field.shape),
            batch_index=batch_index,
        )
        field_next = self.field_update(field, write_map)
        return FieldConditionedGNNOutput(
            cell_output=cell_output,
            field_next=field_next,
            field_context=field_context,
        )

    def initial_field(
        self,
        *,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        if batch_size < 1 or height < 1 or width < 1:
            raise ValueError("batch_size, height, and width must be positive")
        return torch.zeros((batch_size, self.field_channels, height, width), device=device, dtype=dtype)
