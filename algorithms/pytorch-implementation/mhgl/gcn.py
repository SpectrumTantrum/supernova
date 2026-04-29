"""GCN encoder for MHGL (paper §3 Eq. 3.2, §4.2).

Reference: Zhou et al., "Unseen Anomaly Detection on Networks via
Multi-Hpersphere Learning", SIAM SDM 2022. The paper says "we employ GCN [19]
in our implementation" where [19] is Kipf & Welling 2017 (ICLR), and §4.2
specifies four GCN layers with widths 256, 128, 64, 32 and ReLU activation.

Vanilla GCN propagation:

    H^{l+1} = sigma( A_hat @ H^l @ W^l + b^l )

where A_hat = D^{-1/2}(A + I)D^{-1/2} is precomputed in data.build_normalized_adj.

Note on bias: AAGNN sets bias=False on its single linear projection because the
single-hypersphere Deep SVDD objective (Ruff et al. 2018) admits a trivial
collapse to a constant if any bias is present. MHGL is multi-hypersphere with
an additional inverse-distance repulsion term (Eq. 3.6 second term) that blows
up if all representations collapse to a single point — making the trivial
solution non-degenerate. The paper therefore uses the Kipf default (bias=True);
we follow it. Do not "fix" bias=True to bias=False to match AAGNN.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """Single GCN layer: H' = sigma(A_hat @ H @ W + b)  (paper Eq. 3.2)."""

    def __init__(self, in_dim: int, out_dim: int, *, bias: bool = True) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter("bias", None)
        # Glorot uniform — matches Kipf & Welling's reference implementation.
        bound = math.sqrt(6.0 / (in_dim + out_dim))
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, H: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        if H.dim() != 2 or H.shape[1] != self.in_dim:
            raise ValueError(
                f"H must have shape (n, {self.in_dim}), got {tuple(H.shape)}"
            )
        # (A_hat @ H) @ W keeps the sparse op on a (n, in_dim) tensor; the dense
        # matmul afterwards is (n, in_dim) @ (in_dim, out_dim).
        out = torch.sparse.mm(A_hat, H) @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class GCNEncoder(nn.Module):
    """Stack of GCN layers per paper §4.2.

    Default widths (256, 128, 64, 32) replicate the paper. ``example.py`` uses
    a smaller stack to keep the smoke test fast on CPU; override via the
    ``hidden_dims`` argument.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: tuple[int, ...] = (256, 128, 64, 32),
        *,
        activation: str = "relu",
        bias: bool = True,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must be non-empty")
        if activation not in {"relu", "leaky_relu", "tanh"}:
            raise ValueError(f"unsupported activation {activation!r}")

        self.in_dim = in_dim
        self.hidden_dims = tuple(hidden_dims)
        self.activation = activation
        dims = (in_dim,) + self.hidden_dims
        self.layers = nn.ModuleList(
            [GCNLayer(dims[i], dims[i + 1], bias=bias) for i in range(len(self.hidden_dims))]
        )

    @property
    def out_dim(self) -> int:
        return self.hidden_dims[-1]

    def _activate(self, t: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            return F.relu(t)
        if self.activation == "leaky_relu":
            return F.leaky_relu(t, negative_slope=0.2)
        return torch.tanh(t)

    def forward(self, X: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        H = X
        # Paper §4.2 reads as ReLU after every layer (no special-casing the last).
        for layer in self.layers:
            H = self._activate(layer(H, A_hat))
        return H
