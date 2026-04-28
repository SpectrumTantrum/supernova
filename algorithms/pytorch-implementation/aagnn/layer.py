"""Abnormality-aware GNN layer (paper §3.1, Eqs. 1-4).

Reference: Zhou et al., "Subtractive Aggregation for Attributed Network Anomaly
Detection", CIKM 2021.

A single graph layer that, for each node i, computes
    z_i  = W x_i                                    (Eq. 1, no bias)
    h_i  = sigma(z_i - Aggregate({z_j : j in N_i^k}))
where Aggregate is either the mean (Eq. 2) or the attention-weighted sum
(Eqs. 3-4). The linear projection has bias=False to avoid the Deep SVDD
trivial-solution collapse documented in Ruff et al. 2018 (AAGNN ref [21]).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


_AGGREGATORS = {"mean", "attention"}
_ACTIVATIONS = {"relu", "leaky_relu", "tanh"}


def pack_neighbors(
    neigh_lists: list[list[int]],
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack variable-length neighbour lists into (n, K) index + mask tensors.

    Returns:
        N_idx: (n, K) int64 — neighbour indices, padded with 0 (a placeholder
            that is masked out and so never contributes to the aggregate).
        N_mask: (n, K) float — 1 where the slot is a real neighbour, 0 for
            padding. For nodes with no neighbours the entire row is zero.
    """
    n = len(neigh_lists)
    K = max((len(nbrs) for nbrs in neigh_lists), default=0)
    if K == 0:
        # Every node is isolated — caller still needs well-shaped tensors.
        return (
            torch.zeros((n, 1), dtype=torch.int64, device=device),
            torch.zeros((n, 1), dtype=torch.float32, device=device),
        )
    N_idx = torch.zeros((n, K), dtype=torch.int64, device=device)
    N_mask = torch.zeros((n, K), dtype=torch.float32, device=device)
    for i, nbrs in enumerate(neigh_lists):
        if not nbrs:
            continue
        idx = torch.as_tensor(nbrs, dtype=torch.int64, device=device)
        N_idx[i, : idx.numel()] = idx
        N_mask[i, : idx.numel()] = 1.0
    return N_idx, N_mask


class AbnormalityAwareLayer(nn.Module):
    """Subtractive-aggregation graph layer (paper §3.1).

    Args:
        in_dim: input feature dimension (f).
        out_dim: output / projection dimension (d).
        aggregator: "mean" (Eq. 2) or "attention" (Eqs. 3-4).
        activation: output non-linearity, one of {"relu", "leaky_relu", "tanh"}.
        leaky_relu_slope: negative slope used for the LeakyReLU inside the
            attention-score computation AND for the output activation when
            ``activation="leaky_relu"``.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        aggregator: str = "mean",
        activation: str = "relu",
        leaky_relu_slope: float = 0.2,
    ) -> None:
        super().__init__()
        if aggregator not in _AGGREGATORS:
            raise ValueError(
                f"aggregator must be one of {sorted(_AGGREGATORS)}, got {aggregator!r}"
            )
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {sorted(_ACTIVATIONS)}, got {activation!r}"
            )

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.aggregator = aggregator
        self.activation = activation
        self.leaky_relu_slope = leaky_relu_slope

        # bias=False is non-negotiable: a bias term would let the optimiser
        # collapse every node to the hypersphere centre and yield a trivial
        # zero-loss solution (Ruff et al. 2018, AAGNN ref [21]).
        self.W = nn.Linear(in_dim, out_dim, bias=False)

        if aggregator == "attention":
            # 1-D attention vector per Eq. 3. Initialise via a Glorot-style
            # uniform: viewing as a (1, 2*out_dim) fan-in/fan-out gives the
            # same bound nn.init.xavier_uniform_ would produce.
            bound = (6.0 / (1 + 2 * out_dim)) ** 0.5
            attn = torch.empty(2 * out_dim)
            nn.init.uniform_(attn, -bound, bound)
            self.attn = nn.Parameter(attn)
        else:
            self.register_parameter("attn", None)

    def _activate(self, t: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            return F.relu(t)
        if self.activation == "leaky_relu":
            return F.leaky_relu(t, negative_slope=self.leaky_relu_slope)
        return torch.tanh(t)

    def forward(
        self,
        X: torch.Tensor,
        neigh_lists: list[list[int]],
    ) -> torch.Tensor:
        """Compute h_i = sigma(z_i - Aggregate(z_j : j in N_i^k)).

        Args:
            X: (n, in_dim) node feature matrix.
            neigh_lists: length-n list of neighbour-index lists (no self).

        Returns:
            (n, out_dim) tensor.
        """
        if X.dim() != 2 or X.shape[1] != self.in_dim:
            raise ValueError(
                f"X must have shape (n, {self.in_dim}), got {tuple(X.shape)}"
            )
        if len(neigh_lists) != X.shape[0]:
            raise ValueError(
                f"len(neigh_lists)={len(neigh_lists)} != n={X.shape[0]}"
            )

        Z = self.W(X)  # (n, out_dim)
        N_idx, N_mask = pack_neighbors(neigh_lists, device=Z.device)
        Zn = Z[N_idx]  # (n, K, out_dim) — gather neighbour z's

        if self.aggregator == "mean":
            num = (Zn * N_mask.unsqueeze(-1)).sum(dim=1)  # (n, out_dim)
            # clamp keeps the divisor finite for empty neighbourhoods; the
            # numerator is already zero in that case so the row is zero.
            denom = N_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            agg = num / denom
        else:  # attention
            K = Zn.shape[1]
            Zi = Z.unsqueeze(1).expand(-1, K, -1)             # (n, K, out_dim)
            cat = torch.cat([Zi, Zn], dim=-1)                  # (n, K, 2*out_dim)
            e = F.leaky_relu(cat @ self.attn, negative_slope=self.leaky_relu_slope)
            e = e.masked_fill(N_mask == 0, float("-inf"))
            alpha = F.softmax(e, dim=-1)                       # (n, K)
            # Rows with no real neighbours had all -inf scores; softmax
            # produces NaN there. Replace those rows with zeros so the
            # aggregate is the zero vector and h_i = sigma(z_i).
            has_nbr = N_mask.sum(dim=-1, keepdim=True) > 0
            alpha = torch.where(has_nbr, alpha, torch.zeros_like(alpha))
            agg = (alpha.unsqueeze(-1) * Zn).sum(dim=1)        # (n, out_dim)

        return self._activate(Z - agg)
