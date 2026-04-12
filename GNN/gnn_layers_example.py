from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        *,
        num_layers: int = 2,
        dropout: float = 0.0,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        layers: list[nn.Module] = []
        dims = [in_dim]
        if num_layers == 1:
            dims.append(out_dim)
        else:
            dims.extend([hidden_dim] * (num_layers - 1))
            dims.append(out_dim)

        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            is_last = idx == len(dims) - 2
            if not is_last:
                if layer_norm:
                    layers.append(nn.LayerNorm(dims[idx + 1]))
                layers.append(nn.SiLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class EdgeAwareAttentionConv(MessagePassing):
    """
    Edge-aware message passing layer for cell-interaction graphs.

    Message for edge j -> i depends on:
    - current destination embedding h_i
    - current source embedding h_j
    - encoded edge attributes e_ij

    The layer also learns attention weights alpha_ij so the model can decide
    which neighbors matter more for motion / division / death.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        *,
        dropout: float = 0.0,
        aggr: str = "add",
    ) -> None:
        super().__init__(aggr=aggr, node_dim=0)
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.dropout = float(dropout)

        msg_in = hidden_dim * 2 + edge_dim
        self.msg_mlp = MLP(msg_in, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)
        self.att_mlp = MLP(msg_in, hidden_dim, 1, num_layers=2, dropout=dropout)
        self.update_mlp = MLP(hidden_dim * 2, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        if x.dim() != 2:
            raise ValueError(f"Expected x to have shape [N, F], got {tuple(x.shape)}")
        if edge_attr.dim() != 2:
            raise ValueError(
                f"Expected edge_attr to have shape [E, Fe], got {tuple(edge_attr.shape)}"
            )
        propagated = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)
        updated = self.update_mlp(torch.cat([x, propagated], dim=-1))
        return self.norm(x + updated)

    def message(
        self,
        x_i: Tensor,
        x_j: Tensor,
        edge_attr: Tensor,
        index: Tensor,
        ptr: Optional[Tensor],
        size_i: Optional[int],
    ) -> Tensor:
        pair = torch.cat([x_i, x_j, edge_attr], dim=-1)
        logits = self.att_mlp(pair).squeeze(-1)
        alpha = softmax(logits, index=index, ptr=ptr, num_nodes=size_i)
        if self.training and self.dropout > 0:
            alpha = torch.dropout(alpha, p=self.dropout, train=True)
        msg = self.msg_mlp(pair)
        return msg * alpha.unsqueeze(-1)


class TemporalGRUCell(nn.Module):
    """Optional temporal state updater for future rollout support."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, x: Tensor, h_prev: Optional[Tensor] = None) -> Tensor:
        if h_prev is None:
            h_prev = torch.zeros_like(x)
        return self.gru(x, h_prev)
