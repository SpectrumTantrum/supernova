"""MLX GCN encoder for MHGL (paper §3 Eq. 3.2, §4.2)."""

from __future__ import annotations

import math
import mlx.core as mx
import mlx.nn as nn


def _activate(t: mx.array, activation: str) -> mx.array:
    if activation == "relu":
        return nn.relu(t)
    if activation == "leaky_relu":
        return nn.leaky_relu(t, negative_slope=0.2)
    return mx.tanh(t)


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, *, bias: bool = True) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        bound = math.sqrt(6.0 / (in_dim + out_dim))
        self.weight = mx.random.uniform(low=-bound, high=bound, shape=(in_dim, out_dim))
        self.bias = mx.zeros((out_dim,)) if bias else None

    def __call__(self, H: mx.array, A_hat: mx.array) -> mx.array:
        out = (A_hat @ H) @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class GCNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: tuple[int, ...] = (256, 128, 64, 32), *, activation: str = "relu", bias: bool = True) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must be non-empty")
        if activation not in {"relu", "leaky_relu", "tanh"}:
            raise ValueError(f"unsupported activation {activation!r}")
        self.in_dim = in_dim
        self.hidden_dims = tuple(hidden_dims)
        self.activation = activation
        dims = (in_dim,) + self.hidden_dims
        self.layers = [GCNLayer(dims[i], dims[i + 1], bias=bias) for i in range(len(self.hidden_dims))]

    @property
    def out_dim(self) -> int:
        return self.hidden_dims[-1]

    def __call__(self, X: mx.array, A_hat: mx.array) -> mx.array:
        H = X
        for layer in self.layers:
            H = _activate(layer(H, A_hat), self.activation)
        return H
