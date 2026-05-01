"""Flow Predictor — paper §3.2 (MLX)."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class FlowPredictor(nn.Module):
    """Transformer encoder + per-flow softmax head."""

    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        n_heads: int = 8,
        dim_ff: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")
        self.transformer = nn.TransformerEncoder(
            num_layers=n_layers,
            dims=d_model,
            num_heads=n_heads,
            mlp_dims=dim_ff,
            dropout=dropout,
            activation=nn.relu,
            norm_first=False,
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, 1),
        )

    def __call__(self, e: mx.array, dest_padding_mask: mx.array | None = None) -> mx.array:
        h = self.transformer(e, None)
        scores = mx.squeeze(self.head(h), axis=-1)
        if dest_padding_mask is not None:
            scores = mx.where(dest_padding_mask, mx.array(-mx.inf, dtype=scores.dtype), scores)
        return mx.softmax(scores, axis=-1)


def flow_cross_entropy(pred_probs: mx.array, true_proportions: mx.array, eps: float = 1e-12) -> mx.array:
    """Paper Eq. 3 — multinomial cross-entropy."""
    log_p = mx.log(mx.maximum(pred_probs, eps))
    return -mx.mean(mx.sum(true_proportions * log_p, axis=-1))


def common_part_of_commuters(pred_counts: mx.array, true_counts: mx.array, eps: float = 1e-12) -> float:
    """Paper Eq. 4."""
    num = 2.0 * mx.sum(mx.minimum(pred_counts, true_counts))
    den = mx.sum(pred_counts) + mx.sum(true_counts)
    return float(num / mx.maximum(den, eps))
