"""Geo-Spatial Encoder — paper §3.1 (MLX)."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


class MultiScaleSpace2Vec(nn.Module):
    """Multi-scale relative-location encoding (paper §3.1.2 eq. 1)."""

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

        base_angles = mx.array(
            [angle_offset, angle_offset + 2 * math.pi / 3, angle_offset + 4 * math.pi / 3],
            dtype=mx.float32,
        )
        self.base = mx.stack([mx.cos(base_angles), mx.sin(base_angles)], axis=-1)

        if n_scales > 1:
            g = (lambda_max / lambda_min) ** (1.0 / (n_scales - 1))
        else:
            g = 1.0
        self.divisors = lambda_min * (g ** mx.arange(n_scales, dtype=mx.float32))

    @property
    def out_dim(self) -> int:
        return 3 * self.n_scales * 2

    def __call__(self, rl: mx.array) -> mx.array:
        """rl: (..., 2). Returns (..., 6*S)."""
        proj = rl @ mx.transpose(self.base)
        scaled = mx.expand_dims(proj, -1) / self.divisors
        out = mx.stack([mx.cos(scaled), mx.sin(scaled)], axis=-1)
        return mx.reshape(out, (*out.shape[:-3], self.out_dim))


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

    def __call__(self, rl: mx.array) -> mx.array:
        z1 = self.ffn1(self.branch1(rl))
        if self.variant == "rle":
            assert self.branch2 is not None and self.ffn2 is not None
            z2 = self.ffn2(self.branch2(rl))
            return self.merge(mx.concatenate([z1, z2], axis=-1))
        return self.merge(z1)


class GeographicFeatureEncoder(nn.Module):
    """Paper §3.1.1 — linear over [x_o ; x_d ; r] -> x_{o,d}."""

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

    def __call__(self, x_o: mx.array, x_d: mx.array, r: mx.array) -> mx.array:
        """x_o, x_d: (..., D); r: (...) or (..., 1). Returns (..., d_out)."""
        if r.ndim == x_o.ndim - 1:
            r = mx.expand_dims(r, -1)
        if self.normalise_inputs:
            x_o = mx.log1p(mx.maximum(x_o, 0.0))
            x_d = mx.log1p(mx.maximum(x_d, 0.0))
            r = r / self.r_scale
        return self.proj(mx.concatenate([x_o, x_d, r], axis=-1))


class GeoSpatialEncoder(nn.Module):
    """Paper §3.1 wrapper — produces e_{o,d} = [x_{o,d}; loc_{o,d}]."""

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
        self.geo = GeographicFeatureEncoder(
            d_feature=d_feature,
            d_out=d_geo,
            r_scale=lambda_max,
            normalise_inputs=normalise_inputs,
        )
        self.rle = RelativeLocationEncoder(
            variant=rle_variant,
            n_scales=n_scales,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            hidden=d_loc,
            d_out=d_loc,
            dropout=dropout,
        )
        self.d_out = d_geo + d_loc

    def __call__(self, x_o: mx.array, x_d: mx.array, r: mx.array, rl: mx.array) -> mx.array:
        x = self.geo(x_o, x_d, r)
        loc = self.rle(rl)
        return mx.concatenate([x, loc], axis=-1)
