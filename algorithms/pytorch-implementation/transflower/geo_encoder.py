"""Geo-Spatial Encoder — paper §3.1.

Two sub-modules:

    GeographicFeatureEncoder (§3.1.1)
        Linear over the 41-dim concat [x_o (20); x_d (20); r (1)] producing
        the 256-dim individual flow embedding x_{o_i, d_j}.

    RelativeLocationEncoder (§3.1.2, eq. 1)
        Multi-scale Space2Vec with 3 base vectors at 2π/3 angles. Two variants:
            - 'rle'        : two branches with different base orientations,
                             each branch = MultiScale → FFN, then merged via FFN.
            - 'rle_prime'  : single branch — MultiScale → FFN.
        Output is loc_{o_i, d_j}.

GeoSpatialEncoder concatenates them: e_{o_i, d_j} = [x_{o,d} ; loc_{o,d}].
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class MultiScaleSpace2Vec(nn.Module):
    """Multi-scale relative-location encoding (paper §3.1.2 eq. 1).

    For S scales, 3 base vectors a_j at 2π/3 angles, and divisors
    λ_min · g^(s−1) where g = (λ_max/λ_min)^(1/(S−1)):

        PE_{s,j}(rl) = [cos(<rl, a_j> / div_s), sin(<rl, a_j> / div_s)]

    Output dim = 3 · S · 2 = 6S.
    """

    def __init__(
        self,
        n_scales: int = 16,
        lambda_min: float = 1.0,
        lambda_max: float = 20013.0,
        angle_offset: float = 0.0,
    ) -> None:
        super().__init__()
        if lambda_max <= lambda_min:
            raise ValueError(f"lambda_max ({lambda_max}) must exceed lambda_min ({lambda_min}).")
        self.n_scales = n_scales

        # 3 base vectors at 2π/3 angles, optionally rotated by angle_offset.
        base_angles = torch.tensor(
            [angle_offset, angle_offset + 2 * math.pi / 3, angle_offset + 4 * math.pi / 3],
            dtype=torch.float32,
        )
        base = torch.stack([torch.cos(base_angles), torch.sin(base_angles)], dim=-1)  # (3, 2)
        self.register_buffer("base", base)

        if n_scales > 1:
            g = (lambda_max / lambda_min) ** (1.0 / (n_scales - 1))
        else:
            g = 1.0
        divisors = lambda_min * (g ** torch.arange(n_scales, dtype=torch.float32))  # (S,)
        self.register_buffer("divisors", divisors)

    @property
    def out_dim(self) -> int:
        return 3 * self.n_scales * 2

    def forward(self, rl: torch.Tensor) -> torch.Tensor:
        """rl: (..., 2). Returns (..., 6·S)."""
        proj = rl @ self.base.T                              # (..., 3)
        scaled = proj.unsqueeze(-1) / self.divisors          # (..., 3, S)
        cos = torch.cos(scaled)
        sin = torch.sin(scaled)
        out = torch.stack([cos, sin], dim=-1)                # (..., 3, S, 2)
        return out.flatten(start_dim=-3)                     # (..., 6S)


class RelativeLocationEncoder(nn.Module):
    """Paper §3.1.2 Fig. 1b (RLE) and Fig. 1c (RLE')."""

    def __init__(
        self,
        variant: str = "rle",
        n_scales: int = 16,
        lambda_min: float = 1.0,
        lambda_max: float = 20013.0,
        hidden: int = 256,
        d_out: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if variant not in ("rle", "rle_prime"):
            raise ValueError(f"variant must be 'rle' or 'rle_prime', got {variant!r}")
        self.variant = variant

        self.branch1 = MultiScaleSpace2Vec(n_scales, lambda_min, lambda_max, angle_offset=0.0)
        self.ffn1 = nn.Sequential(
            nn.Linear(self.branch1.out_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )

        if variant == "rle":
            # Second branch rotated by π/3 — distinct base-vector orientation.
            self.branch2 = MultiScaleSpace2Vec(n_scales, lambda_min, lambda_max, angle_offset=math.pi / 3)
            self.ffn2 = nn.Sequential(
                nn.Linear(self.branch2.out_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
            )
            self.merge = nn.Sequential(
                nn.Linear(2 * hidden, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, d_out),
            )
        else:
            self.branch2 = None
            self.ffn2 = None
            self.merge = nn.Linear(hidden, d_out)

    def forward(self, rl: torch.Tensor) -> torch.Tensor:
        z1 = self.ffn1(self.branch1(rl))
        if self.variant == "rle":
            z2 = self.ffn2(self.branch2(rl))
            return self.merge(torch.cat([z1, z2], dim=-1))
        return self.merge(z1)


class GeographicFeatureEncoder(nn.Module):
    """Paper §3.1.1 — linear over [x_o ; x_d ; r] → x_{o,d}.

    Input normalisation (`log1p` on count features, `r / r_scale` on distance)
    is applied so that count features ∈ [0, ~5] and distance ∈ [0, ~1] enter
    the linear layer at comparable scales. The paper does not specify
    normalisation explicitly — with their 5M-flow dataset the linear layer
    learns the per-feature scaling internally; with smaller datasets the
    network needs the help. Disable via `normalise_inputs=False` if your
    features are already pre-scaled.
    """

    def __init__(
        self,
        d_feature: int = 20,
        d_out: int = 256,
        r_scale: float = 1.0,
        normalise_inputs: bool = True,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(2 * d_feature + 1, d_out)
        self.r_scale = float(r_scale) if r_scale > 0 else 1.0
        self.normalise_inputs = normalise_inputs

    def forward(
        self,
        x_o: torch.Tensor,
        x_d: torch.Tensor,
        r: torch.Tensor,
    ) -> torch.Tensor:
        """x_o, x_d: (..., D);  r: (...) or (..., 1).  Returns (..., d_out)."""
        if r.dim() == x_o.dim() - 1:
            r = r.unsqueeze(-1)
        if self.normalise_inputs:
            x_o = torch.log1p(x_o.clamp(min=0.0))
            x_d = torch.log1p(x_d.clamp(min=0.0))
            r = r / self.r_scale
        return self.proj(torch.cat([x_o, x_d, r], dim=-1))


class GeoSpatialEncoder(nn.Module):
    """Paper §3.1 wrapper — produces e_{o,d} = [x_{o,d} ; loc_{o,d}]."""

    def __init__(
        self,
        d_feature: int = 20,
        d_geo: int = 256,
        d_loc: int = 256,
        rle_variant: str = "rle",
        n_scales: int = 16,
        lambda_min: float = 1.0,
        lambda_max: float = 20013.0,
        dropout: float = 0.1,
        normalise_inputs: bool = True,
    ) -> None:
        super().__init__()
        # The geographic encoder uses lambda_max as the distance scale so the
        # haversine `r` enters the linear layer in [0, ~1] instead of meters.
        self.geo = GeographicFeatureEncoder(
            d_feature=d_feature, d_out=d_geo,
            r_scale=lambda_max, normalise_inputs=normalise_inputs,
        )
        self.rle = RelativeLocationEncoder(
            variant=rle_variant, n_scales=n_scales,
            lambda_min=lambda_min, lambda_max=lambda_max,
            hidden=d_loc, d_out=d_loc, dropout=dropout,
        )
        self.d_out = d_geo + d_loc

    def forward(
        self,
        x_o: torch.Tensor,
        x_d: torch.Tensor,
        r: torch.Tensor,
        rl: torch.Tensor,
    ) -> torch.Tensor:
        """All inputs are expected in shape (B, N, ·)."""
        x = self.geo(x_o, x_d, r)
        loc = self.rle(rl)
        return torch.cat([x, loc], dim=-1)
