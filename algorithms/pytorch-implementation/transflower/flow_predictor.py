"""Flow Predictor — paper §3.2.

For each origin o_i the encoder produces a set of N candidate flow embeddings
e_{o_i, d_j}. The flow predictor is:

    1. N-layer Transformer encoder with multi-head self-attention (eq. 2).
       The "sequence" is the set of candidate destinations from one origin —
       attention runs over destinations, not over time.
    2. Per-flow prediction head (FFN → scalar score s(o_i, d_j)).
    3. Softmax over the destination axis → probabilities P_{i,j}.

The cross-entropy loss (eq. 3) is exposed as `flow_cross_entropy`.
"""

from __future__ import annotations

import torch
import torch.nn as nn


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

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="relu",
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, 1),
        )

    def forward(
        self,
        e: torch.Tensor,
        dest_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        e: (B, N, d_model) per-flow embeddings from the geo-spatial encoder.
        dest_padding_mask: (B, N) bool, True at padded slots — these positions
            are excluded from attention and forced to probability 0.

        Returns (B, N) probabilities — each row sums to 1 over its non-padded
        entries.
        """
        h = self.transformer(e, src_key_padding_mask=dest_padding_mask)   # (B, N, d_model)
        scores = self.head(h).squeeze(-1)                                  # (B, N)
        if dest_padding_mask is not None:
            scores = scores.masked_fill(dest_padding_mask, float("-inf"))
        return torch.softmax(scores, dim=-1)


def flow_cross_entropy(
    pred_probs: torch.Tensor,
    true_proportions: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Paper Eq. 3 — multinomial cross-entropy.

        H = -sum_j (f_{ij} / O_i) · ln P_{i,j}    averaged over the origin batch.

    pred_probs: (B, N) softmax outputs (each row sums to 1).
    true_proportions: (B, N) observed flow proportions f_{ij}/O_i.
    """
    log_p = torch.log(pred_probs.clamp(min=eps))
    return -(true_proportions * log_p).sum(dim=-1).mean()


def common_part_of_commuters(
    pred_counts: torch.Tensor,
    true_counts: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    """Paper Eq. 4.

        CPC = 2 · sum min(v_p, v_r) / (sum v_p + sum v_r)

    Both inputs are flow-volume tensors of any matching shape (e.g. (B, N) or
    (N, N)). Returns a scalar in [0, 1] — 1 means perfect overlap.
    """
    num = 2.0 * torch.minimum(pred_counts, true_counts).sum()
    den = pred_counts.sum() + true_counts.sum()
    return float(num / den.clamp(min=eps))
