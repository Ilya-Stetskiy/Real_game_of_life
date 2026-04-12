from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from gnn_layers import EdgeAwareAttentionConv, MLP, TemporalGRUCell


@dataclass
class CellGNNOutput:
    delta_pos: Tensor
    delta_shape: Tensor
    division_logits: Tensor
    death_logits: Tensor
    node_embeddings: Tensor
    trajectory_hypotheses: Optional[Tensor] = None


class CellInteractionGNN(nn.Module):
    """
    Edge-aware GNN for one-step cell dynamics prediction.

    Inputs expected in a PyG Data object:
    - data.x:         [N, node_dim]
    - data.edge_index:[2, E]
    - data.edge_attr: [E, edge_dim]

    Outputs:
    - delta_pos:              [N, 2]
    - delta_shape:            [N, shape_dim]
    - division_logits:        [N]
    - death_logits:           [N]
    - trajectory_hypotheses:  [N, K, 2] or None

    Design choices matched to the project discussion:
    - node features = cell state + aggregated environment
    - edge features = pairwise geometry / visibility / contact
    - message passing is edge-aware and attention-based
    - residual updates make motion and shape learning more stable
    - optional K trajectory heads support Best-of-K training
    """

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
    ) -> None:
        super().__init__()
        if num_message_passing_layers < 1:
            raise ValueError("num_message_passing_layers must be >= 1")
        if num_trajectory_hypotheses < 1:
            raise ValueError("num_trajectory_hypotheses must be >= 1")

        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        self.shape_dim = int(shape_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_message_passing_layers = int(num_message_passing_layers)
        self.use_temporal_state = bool(use_temporal_state)
        self.num_trajectory_hypotheses = int(num_trajectory_hypotheses)

        self.node_encoder = MLP(
            in_dim=node_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            num_layers=2,
            dropout=dropout,
            layer_norm=True,
        )
        self.edge_encoder = MLP(
            in_dim=edge_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            num_layers=2,
            dropout=dropout,
            layer_norm=True,
        )

        self.layers = nn.ModuleList(
            EdgeAwareAttentionConv(hidden_dim=hidden_dim, edge_dim=hidden_dim, dropout=dropout)
            for _ in range(num_message_passing_layers)
        )
        self.temporal_cell = TemporalGRUCell(hidden_dim) if use_temporal_state else None

        # Shared decoder trunk.
        self.decoder_trunk = MLP(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            num_layers=2,
            dropout=dropout,
            layer_norm=True,
        )

        self.delta_pos_head = nn.Linear(hidden_dim, 2)
        self.delta_shape_head = nn.Linear(hidden_dim, shape_dim)
        self.division_head = nn.Linear(hidden_dim, 1)
        self.death_head = nn.Linear(hidden_dim, 1)

        if num_trajectory_hypotheses > 1:
            self.trajectory_head = nn.Linear(hidden_dim, num_trajectory_hypotheses * 2)
        else:
            self.trajectory_head = None

    def forward(self, data: Data, h_prev: Optional[Tensor] = None) -> CellGNNOutput:
        self._validate_data(data)

        x = data.x.float()
        edge_attr = data.edge_attr.float()
        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr)

        for layer in self.layers:
            h = layer(h, data.edge_index, e)

        if self.temporal_cell is not None:
            h = self.temporal_cell(h, h_prev)

        z = self.decoder_trunk(h)

        delta_pos = self.delta_pos_head(z)
        delta_shape = self.delta_shape_head(z)
        division_logits = self.division_head(z).squeeze(-1)
        death_logits = self.death_head(z).squeeze(-1)

        hypotheses = None
        if self.trajectory_head is not None:
            raw = self.trajectory_head(z)
            hypotheses = raw.view(raw.size(0), self.num_trajectory_hypotheses, 2)

        return CellGNNOutput(
            delta_pos=delta_pos,
            delta_shape=delta_shape,
            division_logits=division_logits,
            death_logits=death_logits,
            node_embeddings=h,
            trajectory_hypotheses=hypotheses,
        )

    def _validate_data(self, data: Data) -> None:
        if not hasattr(data, "x") or data.x is None:
            raise ValueError("Data object must contain node features in data.x")
        if not hasattr(data, "edge_index") or data.edge_index is None:
            raise ValueError("Data object must contain edge_index")
        if not hasattr(data, "edge_attr") or data.edge_attr is None:
            raise ValueError("Data object must contain edge_attr")

        if data.x.dim() != 2:
            raise ValueError(f"Expected data.x shape [N, F], got {tuple(data.x.shape)}")
        if data.edge_attr.dim() != 2:
            raise ValueError(
                f"Expected data.edge_attr shape [E, Fe], got {tuple(data.edge_attr.shape)}"
            )
        if data.x.size(-1) != self.node_dim:
            raise ValueError(
                f"Expected node_dim={self.node_dim}, got {data.x.size(-1)}"
            )
        if data.edge_attr.size(-1) != self.edge_dim:
            raise ValueError(
                f"Expected edge_dim={self.edge_dim}, got {data.edge_attr.size(-1)}"
            )


def best_of_k_position_loss(
    hypotheses: Tensor,
    target_delta_pos: Tensor,
    valid_mask: Optional[Tensor] = None,
) -> Tensor:
    """
    hypotheses: [N, K, 2]
    target_delta_pos: [N, 2]
    valid_mask: [N] bool, optional
    """
    if hypotheses.dim() != 3 or hypotheses.size(-1) != 2:
        raise ValueError("hypotheses must have shape [N, K, 2]")
    if target_delta_pos.shape[-1] != 2:
        raise ValueError("target_delta_pos must have shape [N, 2]")

    diff = hypotheses - target_delta_pos.unsqueeze(1)
    sq = (diff ** 2).sum(dim=-1)  # [N, K]
    min_sq = sq.min(dim=1).values

    if valid_mask is not None:
        valid_mask = valid_mask.bool()
        if valid_mask.sum() == 0:
            return min_sq.new_tensor(0.0)
        min_sq = min_sq[valid_mask]
    return min_sq.mean()


def masked_regression_loss(
    pred: Tensor,
    target: Tensor,
    valid_mask: Optional[Tensor] = None,
    reduction: str = "mean",
) -> Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"pred and target shape mismatch: {pred.shape} vs {target.shape}")
    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1) if loss.dim() > 1 else loss

    if valid_mask is not None:
        valid_mask = valid_mask.bool()
        if valid_mask.sum() == 0:
            return loss.new_tensor(0.0)
        loss = loss[valid_mask]

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"Unsupported reduction: {reduction}")


def cell_dynamics_loss(
    output: CellGNNOutput,
    *,
    target_delta_pos: Tensor,
    target_delta_shape: Tensor,
    target_division: Tensor,
    target_death: Tensor,
    valid_regression_mask: Optional[Tensor] = None,
    pos_weight_division: Optional[Tensor] = None,
    pos_weight_death: Optional[Tensor] = None,
    lambda_pos: float = 1.0,
    lambda_shape: float = 1.0,
    lambda_division: float = 1.0,
    lambda_death: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_division)
    division_loss = bce(output.division_logits, target_division.float())

    bce_death = nn.BCEWithLogitsLoss(pos_weight=pos_weight_death)
    death_loss = bce_death(output.death_logits, target_death.float())

    if output.trajectory_hypotheses is not None:
        pos_loss = best_of_k_position_loss(
            output.trajectory_hypotheses,
            target_delta_pos,
            valid_mask=valid_regression_mask,
        )
    else:
        pos_loss = masked_regression_loss(
            output.delta_pos,
            target_delta_pos,
            valid_mask=valid_regression_mask,
        )

    shape_loss = masked_regression_loss(
        output.delta_shape,
        target_delta_shape,
        valid_mask=valid_regression_mask,
    )

    total = (
        lambda_pos * pos_loss
        + lambda_shape * shape_loss
        + lambda_division * division_loss
        + lambda_death * death_loss
    )
    stats = {
        "loss_total": total.detach(),
        "loss_pos": pos_loss.detach(),
        "loss_shape": shape_loss.detach(),
        "loss_division": division_loss.detach(),
        "loss_death": death_loss.detach(),
    }
    return total, stats
