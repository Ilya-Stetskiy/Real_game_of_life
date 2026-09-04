from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from .gnn_layers import EdgeAwareAttentionConv, MLP, TemporalGRUCell


@dataclass
class CellGNNOutput:
    delta_pos: Tensor
    delta_shape: Tensor
    division_logits: Tensor
    death_logits: Tensor
    node_embeddings: Tensor
    trajectory_hypotheses: Optional[Tensor] = None
    division_horizon_logits: Optional[Tensor] = None
    delta_polarization: Optional[Tensor] = None


class CellInteractionGNN(nn.Module):
    """Edge-aware GNN for one-step cell dynamics prediction."""

    def __init__(
        self,
        *,
        node_dim: int,
        edge_dim: int,
        shape_dim: int,
        hidden_dim: int = 128,
        num_message_passing_layers: int = 4,
        dropout: float = 0.1,
        use_temporal_state: bool = False,
        num_trajectory_hypotheses: int = 1,
        num_division_horizons: int = 0,
    ) -> None:
        super().__init__()
        if num_message_passing_layers < 1:
            raise ValueError("num_message_passing_layers must be >= 1")
        if num_trajectory_hypotheses < 1:
            raise ValueError("num_trajectory_hypotheses must be >= 1")
        if num_division_horizons < 0:
            raise ValueError("num_division_horizons must be >= 0")
        if shape_dim < 1:
            raise ValueError("shape_dim must be >= 1")

        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        self.shape_dim = int(shape_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_message_passing_layers = int(num_message_passing_layers)
        self.use_temporal_state = bool(use_temporal_state)
        self.num_trajectory_hypotheses = int(num_trajectory_hypotheses)
        self.num_division_horizons = int(num_division_horizons)

        self.node_encoder = MLP(node_dim, hidden_dim, hidden_dim, num_layers=2, dropout=dropout, layer_norm=True)
        self.edge_encoder = MLP(edge_dim, hidden_dim, hidden_dim, num_layers=2, dropout=dropout, layer_norm=True)
        self.layers = nn.ModuleList(
            EdgeAwareAttentionConv(hidden_dim=hidden_dim, edge_dim=hidden_dim, dropout=dropout)
            for _ in range(num_message_passing_layers)
        )
        self.temporal_cell = TemporalGRUCell(hidden_dim) if use_temporal_state else None
        self.decoder_trunk = MLP(hidden_dim, hidden_dim, hidden_dim, num_layers=2, dropout=dropout, layer_norm=True)

        self.delta_pos_head = nn.Linear(hidden_dim, 2)
        self.delta_shape_head = nn.Linear(hidden_dim, shape_dim)
        self.delta_polarization_head = nn.Linear(hidden_dim, 2)
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

    def forward(self, data: Data, h_prev: Optional[Tensor] = None) -> CellGNNOutput:
        self._validate_data(data)

        h = self.node_encoder(data.x.float())
        edge_attr = self.edge_encoder(data.edge_attr.float())
        for layer in self.layers:
            h = layer(h, data.edge_index, edge_attr)

        if self.temporal_cell is not None:
            h = self.temporal_cell(h, h_prev)

        z = self.decoder_trunk(h)
        hypotheses = None
        if self.trajectory_head is not None:
            raw = self.trajectory_head(z)
            hypotheses = raw.view(raw.size(0), self.num_trajectory_hypotheses, 2)

        return CellGNNOutput(
            delta_pos=self.delta_pos_head(z),
            delta_shape=self.delta_shape_head(z),
            delta_polarization=self.delta_polarization_head(z),
            division_logits=self.division_head(z).squeeze(-1),
            death_logits=self.death_head(z).squeeze(-1),
            node_embeddings=h,
            trajectory_hypotheses=hypotheses,
            division_horizon_logits=(
                self.division_horizon_head(z)
                if self.division_horizon_head is not None
                else None
            ),
        )

    def _validate_data(self, data: Data) -> None:
        if not hasattr(data, "x") or data.x is None:
            raise ValueError("Data object must contain data.x")
        if not hasattr(data, "edge_index") or data.edge_index is None:
            raise ValueError("Data object must contain data.edge_index")
        if not hasattr(data, "edge_attr") or data.edge_attr is None:
            raise ValueError("Data object must contain data.edge_attr")
        if data.x.dim() != 2:
            raise ValueError(f"Expected data.x shape [N, F], got {tuple(data.x.shape)}")
        if data.edge_index.dim() != 2 or data.edge_index.size(0) != 2:
            raise ValueError(f"Expected data.edge_index shape [2, E], got {tuple(data.edge_index.shape)}")
        if data.edge_index.dtype != torch.long:
            raise ValueError("data.edge_index must have dtype torch.long")
        if data.edge_attr.dim() != 2:
            raise ValueError(f"Expected data.edge_attr shape [E, Fe], got {tuple(data.edge_attr.shape)}")
        if data.edge_attr.size(0) != data.edge_index.size(1):
            raise ValueError("data.edge_attr rows must match data.edge_index columns")
        if data.x.size(-1) != self.node_dim:
            raise ValueError(f"Expected node_dim={self.node_dim}, got {data.x.size(-1)}")
        if data.edge_attr.size(-1) != self.edge_dim:
            raise ValueError(f"Expected edge_dim={self.edge_dim}, got {data.edge_attr.size(-1)}")


def best_of_k_position_loss(
    hypotheses: Tensor,
    target_delta_pos: Tensor,
    valid_mask: Optional[Tensor] = None,
) -> Tensor:
    if hypotheses.dim() != 3 or hypotheses.size(-1) != 2:
        raise ValueError("hypotheses must have shape [N, K, 2]")
    if target_delta_pos.shape[-1] != 2:
        raise ValueError("target_delta_pos must have shape [N, 2]")

    min_sq = ((hypotheses - target_delta_pos.unsqueeze(1)) ** 2).sum(dim=-1).min(dim=1).values
    if valid_mask is not None:
        valid_mask = valid_mask.bool()
        if int(valid_mask.sum()) == 0:
            return min_sq.new_tensor(0.0)
        min_sq = min_sq[valid_mask]
    return min_sq.mean()


def masked_regression_loss(pred: Tensor, target: Tensor, valid_mask: Optional[Tensor] = None) -> Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"pred and target shape mismatch: {pred.shape} vs {target.shape}")
    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1) if loss.dim() > 1 else loss
    if valid_mask is not None:
        valid_mask = valid_mask.bool()
        if int(valid_mask.sum()) == 0:
            return loss.new_tensor(0.0)
        loss = loss[valid_mask]
    return loss.mean()


def masked_bce_with_logits(
    logits: Tensor,
    target: Tensor,
    valid_mask: Optional[Tensor] = None,
    pos_weight: Optional[Tensor] = None,
) -> Tensor:
    if logits.shape != target.shape:
        raise ValueError(f"logits and target shape mismatch: {logits.shape} vs {target.shape}")
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits,
        target.float(),
        pos_weight=pos_weight,
        reduction="none",
    )
    if valid_mask is not None:
        valid_mask = valid_mask.bool()
        if int(valid_mask.sum()) == 0:
            return loss.new_tensor(0.0)
        loss = loss[valid_mask]
    return loss.mean()


def cell_dynamics_loss(
    output: CellGNNOutput,
    *,
    target_delta_pos: Tensor,
    target_delta_shape: Tensor,
    target_division: Tensor,
    target_death: Tensor,
    target_delta_polarization: Optional[Tensor] = None,
    valid_regression_mask: Optional[Tensor] = None,
    valid_shape_mask: Optional[Tensor] = None,
    valid_polarization_mask: Optional[Tensor] = None,
    valid_event_mask: Optional[Tensor] = None,
    target_division_horizon: Optional[Tensor] = None,
    valid_division_horizon_mask: Optional[Tensor] = None,
    pos_weight_division: Optional[Tensor] = None,
    pos_weight_death: Optional[Tensor] = None,
    pos_weight_division_horizon: Optional[Tensor] = None,
    lambda_pos: float = 1.0,
    lambda_shape: float = 1.0,
    lambda_polarization: float = 1.0,
    lambda_division: float = 1.0,
    lambda_death: float = 1.0,
    lambda_division_horizon: float = 0.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    if output.trajectory_hypotheses is not None:
        pos_loss = best_of_k_position_loss(
            output.trajectory_hypotheses,
            target_delta_pos,
            valid_mask=valid_regression_mask,
        )
    else:
        pos_loss = masked_regression_loss(output.delta_pos, target_delta_pos, valid_regression_mask)

    shape_loss = masked_regression_loss(output.delta_shape, target_delta_shape, valid_shape_mask)
    if output.delta_polarization is not None and target_delta_polarization is not None:
        polarization_loss = masked_regression_loss(
            output.delta_polarization, target_delta_polarization, valid_polarization_mask
        )
    else:
        polarization_loss = output.delta_shape.new_tensor(0.0)
    division_loss = masked_bce_with_logits(
        output.division_logits,
        target_division.float(),
        valid_event_mask,
        pos_weight_division,
    )
    death_loss = masked_bce_with_logits(
        output.death_logits,
        target_death.float(),
        valid_event_mask,
        pos_weight_death,
    )
    if output.division_horizon_logits is not None and target_division_horizon is not None:
        division_horizon_loss = masked_bce_with_logits(
            output.division_horizon_logits,
            target_division_horizon.float(),
            valid_division_horizon_mask,
            pos_weight_division_horizon,
        )
    else:
        division_horizon_loss = output.death_logits.new_tensor(0.0)

    total = (
        lambda_pos * pos_loss
        + lambda_shape * shape_loss
        + lambda_polarization * polarization_loss
        + lambda_division * division_loss
        + lambda_death * death_loss
        + lambda_division_horizon * division_horizon_loss
    )
    return total, {
        "loss_total": total.detach(),
        "loss_pos": pos_loss.detach(),
        "loss_shape": shape_loss.detach(),
        "loss_polarization": polarization_loss.detach(),
        "loss_division": division_loss.detach(),
        "loss_death": death_loss.detach(),
        "loss_division_horizon": division_horizon_loss.detach(),
    }

