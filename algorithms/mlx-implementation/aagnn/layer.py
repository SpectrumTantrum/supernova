"""MLX abnormality-aware GNN layer (paper §3.1, Eqs. 1-4)."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

_AGGREGATORS = {"mean", "attention"}
_ACTIVATIONS = {"relu", "leaky_relu", "tanh"}


def _activate(t: mx.array, activation: str) -> mx.array:
    if activation == "relu":
        return nn.relu(t)
    if activation == "leaky_relu":
        return nn.leaky_relu(t, negative_slope=0.2)
    return mx.tanh(t)


class AbnormalityAwareLayer(nn.Module):
    """Subtractive-aggregation graph layer for AAGNN."""

    def __init__(self, in_dim: int, out_dim: int, aggregator: str = "mean", activation: str = "relu") -> None:
        super().__init__()
        if aggregator not in _AGGREGATORS:
            raise ValueError(f"aggregator must be one of {sorted(_AGGREGATORS)}, got {aggregator!r}")
        if activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(_ACTIVATIONS)}, got {activation!r}")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.aggregator = aggregator
        self.activation = activation
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        if aggregator == "attention":
            bound = (6.0 / (1 + 2 * out_dim)) ** 0.5
            self.attn = mx.random.uniform(low=-bound, high=bound, shape=(2 * out_dim,))
        else:
            self.attn = None

    def __call__(self, X: mx.array, neigh_lists: list[list[int]]) -> mx.array:
        if X.ndim != 2 or X.shape[1] != self.in_dim:
            raise ValueError(f"X must have shape (n, {self.in_dim}), got {X.shape}")
        Z = self.W(X)
        rows = []
        for i, nbrs in enumerate(neigh_lists):
            if not nbrs:
                agg = mx.zeros((self.out_dim,), dtype=Z.dtype)
            else:
                Zn = Z[mx.array(nbrs, dtype=mx.int32)]
                if self.aggregator == "mean":
                    agg = mx.mean(Zn, axis=0)
                else:
                    Zi = mx.broadcast_to(Z[i], Zn.shape)
                    e = nn.leaky_relu(mx.matmul(mx.concatenate([Zi, Zn], axis=-1), self.attn), negative_slope=0.2)
                    alpha = mx.softmax(e, axis=-1)
                    agg = mx.sum(alpha[:, None] * Zn, axis=0)
            rows.append(_activate(Z[i] - agg, self.activation))
        return mx.stack(rows, axis=0)
