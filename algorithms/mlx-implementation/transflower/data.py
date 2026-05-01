"""Data structures and synthetic-city generator for TransFlower.

Reference: Luo et al., "TransFlower" arXiv:2402.15398v1 (2024).
    §2 Problem Formulation — Region (Def. 1), CommutingFlow (Def. 2).
    §3.1.1 Geographic Feature Encoder — 19-dim place attributes + population.

Defines:
    - Region, Flow dataclasses
    - haversine_meters (great-circle distance)
    - SyntheticCity (planted-cluster generator with anisotropic gravity flows)
    - prepare_region_tensors (pre-computes feature/distance/relative-location
      tensors used by the model)
    - build_flow_proportions / build_flow_counts (per-origin row-normalised
      and raw flow tensors used for cross-entropy and CPC respectively)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import mlx.core as mx


# --- Paper §3.1.1 constants -----------------------------------------------------

# 19 OSM-derived place attributes (4 food + 4 retail + 3 education
# + 3 health + 5 transport categories), per paper §3.1.1.
NUM_PLACE_CATEGORIES = 19
# Per-region feature vector = 19 attribute counts + 1 population scalar.
D_FEATURE = NUM_PLACE_CATEGORIES + 1


# --- Records --------------------------------------------------------------------

@dataclass(frozen=True)
class Region:
    """Paper Def. 1.

    `place_features` is the 19-dim OSM attribute count vector (food, retail,
    education, health, transport sub-totals). `population` is concatenated to
    form the 20-dim feature vector consumed by the geographic encoder.
    """
    region_id: int
    lat: float
    lon: float
    place_features: np.ndarray   # shape (19,), float32
    population: float

    def feature_vector(self) -> np.ndarray:
        return np.concatenate([self.place_features, [self.population]]).astype(np.float32)


@dataclass(frozen=True)
class Flow:
    """Paper Def. 2: 3-tuple (o, d, v) where v is the commuter count."""
    o_id: int
    d_id: int
    count: float


# --- Geometry -------------------------------------------------------------------

def haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle surface distance in meters."""
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# --- Synthetic clustered city ---------------------------------------------------

@dataclass
class SyntheticCity:
    """Miniature city with planted land-use clusters and gravity-with-anisotropy flows.

    Each region is assigned to one of K latent clusters whose place-feature
    profile and attractiveness drive ground-truth flows via:

        f(o,d) ∝ pop_o · attractiveness_d · exp(-r_{o,d}/sigma) · (1 + a·cos(angle - pref_o))

    The cosine term injects directional bias keyed to the origin's cluster, so
    a multi-scale anisotropic encoder (the RLE) has structure to learn that a
    distance-only model cannot capture.

    Returns cluster IDs alongside regions and flows; the smoke test uses these
    to verify same-cluster destinations end up with higher predicted probability.
    """

    n_regions: int = 64
    n_clusters: int = 4
    seed: int = 0
    box_km: float = 20.0             # half-width of the urban area
    sigma_km: float = 6.0            # gravity decay scale
    anisotropy_strength: float = 0.7
    flows_per_unit_pop: int = 30     # multinomial sample size scaling
    center_lat: float = 37.7749      # San Francisco (just a plausible anchor)
    center_lon: float = -122.4194

    # (cluster_name, dominant 19-dim feature indices, attractiveness_d, anisotropy preference)
    CLUSTERS = [
        ("residential", [4, 5, 6, 7],                     0.5, math.pi / 2),
        ("business",    [10, 11, 12, 14, 15, 16],         1.8, -math.pi / 2),
        ("university",  [8, 9],                            1.2, 0.0),
        ("retail",      [0, 1, 2, 3, 4, 5, 6, 7],         1.0, math.pi),
    ]

    def generate(self) -> tuple[list[Region], list[Flow], dict[int, int], dict[str, float]]:
        rng = np.random.default_rng(self.seed)

        # Cluster centers placed on a smaller circle inside the box.
        radius_km = self.box_km * 0.35
        ring_angles = np.linspace(0, 2 * math.pi, self.n_clusters, endpoint=False) + math.pi / 4
        cluster_xy_km = np.stack(
            [radius_km * np.cos(ring_angles), radius_km * np.sin(ring_angles)], axis=1,
        )

        # Local equirectangular conversion: 1 deg lat ~ 111 km; 1 deg lon ~ 111·cos(lat).
        deg_per_km_lat = 1.0 / 111.0
        deg_per_km_lon = 1.0 / (111.0 * math.cos(math.radians(self.center_lat)))

        regions: list[Region] = []
        region_to_cluster: dict[int, int] = {}
        for rid in range(self.n_regions):
            c = rid % self.n_clusters
            cx_km, cy_km = cluster_xy_km[c]
            x_km = cx_km + rng.normal(0, 1.5)
            y_km = cy_km + rng.normal(0, 1.5)
            lat = self.center_lat + y_km * deg_per_km_lat
            lon = self.center_lon + x_km * deg_per_km_lon

            # Cluster-weighted Poisson place features.
            _, dom_idxs, _, _ = self.CLUSTERS[c]
            rate = np.full(NUM_PLACE_CATEGORIES, 2.0)
            rate[dom_idxs] += 10.0
            feats = rng.poisson(rate).astype(np.float32)

            # Population: residential heavy, business light.
            pop_mu = {"residential": 3.0, "business": 1.4, "university": 2.2, "retail": 2.0}[
                self.CLUSTERS[c][0]
            ]
            pop = float(rng.lognormal(pop_mu, 0.3))

            regions.append(Region(rid, float(lat), float(lon), feats, pop))
            region_to_cluster[rid] = c

        # Pairwise distances and flow generation.
        N = len(regions)
        dist_m = np.zeros((N, N), dtype=np.float64)
        for i, ri in enumerate(regions):
            for j, rj in enumerate(regions):
                if i == j:
                    continue
                dist_m[i, j] = haversine_meters(ri.lat, ri.lon, rj.lat, rj.lon)

        sigma_m = self.sigma_km * 1000.0
        flows: list[Flow] = []
        for i, ri in enumerate(regions):
            ci = region_to_cluster[i]
            _, _, _, pref = self.CLUSTERS[ci]

            scores = np.zeros(N)
            for j, rj in enumerate(regions):
                if i == j:
                    continue
                cj = region_to_cluster[j]
                _, _, attr_d, _ = self.CLUSTERS[cj]
                # Angle from i to j in the local plane.
                dy = rj.lat - ri.lat
                dx = (rj.lon - ri.lon) * math.cos(math.radians(ri.lat))
                angle = math.atan2(dy, dx)
                anis = 1.0 + self.anisotropy_strength * math.cos(angle - pref)
                scores[j] = (
                    ri.population * attr_d
                    * math.exp(-dist_m[i, j] / sigma_m)
                    * max(anis, 0.05)
                )

            total = max(int(round(ri.population * self.flows_per_unit_pop)), 1)
            tot_score = float(scores.sum())
            if tot_score <= 0:
                continue
            probs = scores / tot_score
            counts = rng.multinomial(total, probs)
            for j, c in enumerate(counts):
                if c > 0:
                    flows.append(Flow(i, j, float(c)))

        meta = {
            "lambda_min_m": 1.0,
            "lambda_max_m": float(np.max(dist_m)),
            "n_clusters": self.n_clusters,
        }
        return regions, flows, region_to_cluster, meta


# --- Tensor preparation ---------------------------------------------------------

def prepare_region_tensors(
    regions: list[Region], device: str = "cpu",
) -> tuple[mx.array, mx.array, mx.array]:
    """Pre-compute feature, distance, and relative-location tensors.

    Returns:
        feats   : (N, D_FEATURE) float32 — per-region [place_features ; pop]
        dist_m  : (N, N) float32 — pairwise haversine meters
        rl_m    : (N, N, 2) float32 — rl[i, j] = (loc_i − loc_j) in local
                  east/north meters (origin minus destination, paper §3.1.2)
    """
    N = len(regions)
    _ = device
    feats = mx.array(np.stack([r.feature_vector() for r in regions]), dtype=mx.float32)

    lats = np.array([r.lat for r in regions], dtype=np.float64)
    lons = np.array([r.lon for r in regions], dtype=np.float64)
    mean_lat = float(lats.mean())
    mean_lon = float(lons.mean())

    # Local equirectangular projection (meters).
    m_per_deg_lat = 111_000.0
    m_per_deg_lon = 111_000.0 * math.cos(math.radians(mean_lat))
    locs_m = np.stack(
        [(lons - mean_lon) * m_per_deg_lon,   # east
         (lats - mean_lat) * m_per_deg_lat],  # north
        axis=-1,
    ).astype(np.float32)
    rl_m = locs_m[:, None, :] - locs_m[None, :, :]   # (N, N, 2): rl[i,j] = loc_i - loc_j

    dist_np = np.zeros((N, N), dtype=np.float32)
    for i, ri in enumerate(regions):
        for j, rj in enumerate(regions):
            if i == j:
                continue
            dist_np[i, j] = haversine_meters(ri.lat, ri.lon, rj.lat, rj.lon)

    return feats, mx.array(dist_np, dtype=mx.float32), mx.array(rl_m, dtype=mx.float32)


def build_flow_counts(
    flows: Iterable[Flow],
    n_regions: int,
    device: str = "cpu",
    region_id_to_idx: dict[int, int] | None = None,
) -> mx.array:
    """Per-origin raw flow count matrix F[i, j] = f_{ij} (paper Def. 2)."""
    _ = device
    F_np = np.zeros((n_regions, n_regions), dtype=np.float32)
    for fl in flows:
        if region_id_to_idx is None:
            o_idx, d_idx = fl.o_id, fl.d_id
        else:
            try:
                o_idx = region_id_to_idx[fl.o_id]
                d_idx = region_id_to_idx[fl.d_id]
            except KeyError as exc:
                raise ValueError(f"flow endpoint region_id {exc.args[0]!r} is not in regions") from exc
        if not (0 <= o_idx < n_regions and 0 <= d_idx < n_regions):
            raise ValueError(
                f"flow endpoint indices out of range [0, {n_regions}): "
                f"({o_idx}, {d_idx})"
            )
        F_np[o_idx, d_idx] += fl.count
    return mx.array(F_np, dtype=mx.float32)


def build_flow_proportions(F: mx.array, eps: float = 1e-12) -> mx.array:
    """Row-normalise to f_{ij}/O_i (paper Eq. 3 input proportions).

    Origins with zero observed outflow get a uniform row so the cross-entropy
    over them is well-defined and contributes a constant.
    """
    row_sums = mx.expand_dims(mx.sum(F, axis=-1), -1)
    safe = mx.where(row_sums > 0, row_sums, mx.ones_like(row_sums))
    P = F / mx.maximum(safe, eps)
    n_regions = F.shape[-1]
    uniform = mx.ones_like(P) / n_regions
    return mx.where(row_sums <= 0, uniform, P)


def split_flows(
    flows: list[Flow], val_frac: float = 0.2, seed: int = 1234,
) -> tuple[list[Flow], list[Flow]]:
    """Random 80/20 split of flow records — paper §4.1.3 setup."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(flows))
    rng.shuffle(idx)
    cut = int(round(len(flows) * (1.0 - val_frac)))
    train = [flows[i] for i in idx[:cut]]
    val = [flows[i] for i in idx[cut:]]
    return train, val
