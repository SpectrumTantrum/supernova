"""Data structures and synthetic-data generator for Geo-Tile2Vec.

Reference: Luo et al., "Geo-Tile2Vec", ACM TSAS 2023, §2.1, §3.2.1.

Defines:
    - TileId, POI, Trajectory, MobilityEvent
    - latlon_to_tile (closed-form Slippy Map projection)
    - build_mobility_events (50 m POI snap, drops trajectories with no nearby POI)
    - build_skipgram_pairs (D-event time/category co-occurrence at threshold T)
    - SyntheticCity (planted-cluster generator that drives example.py)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterable, NamedTuple

import numpy as np

# --- Paper §2.1 / §4.1 constants -------------------------------------------------

DEFAULT_TILE_LEVEL = 18                # ≈120 m × 120 m at mid-latitudes
DEFAULT_POI_SNAP_METERS = 50.0         # §2.1 Def. 2.3
DEFAULT_TIME_THRESHOLD_MIN = 15        # §4.1
NUM_POI_CATEGORIES = 16                # Table A.1
NUM_TIME_BUCKETS = 24                  # §3.2.1, hourly bucketing

# 16 POI categories from Table A.1
POI_CATEGORIES = [
    "enterprise", "medical", "infrastructure", "recreational",
    "residential", "educational", "cultural", "scenic_spot",
    "institutional", "life_service", "restaurant", "shopping",
    "sports", "hotel", "car_related", "banking_finance",
]
CATEGORY_TO_IDX = {c: i for i, c in enumerate(POI_CATEGORIES)}

EVENT_TYPE_O = 0
EVENT_TYPE_D = 1


# --- Geometry --------------------------------------------------------------------

class TileId(NamedTuple):
    z: int
    x: int
    y: int


def latlon_to_tile(lat: float, lon: float, z: int = DEFAULT_TILE_LEVEL) -> TileId:
    """Slippy Map (web Mercator) lat/lon → tile id at zoom z.

    Closed-form, no pyproj. See OSM wiki "Slippy map tilenames".
    """
    lat_rad = math.radians(lat)
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    # asinh(tan(φ)) == ln(tan(φ) + sec(φ)); asinh is more numerically stable.
    return TileId(z, x, y)


def haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters."""
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# --- Records ---------------------------------------------------------------------

@dataclass(frozen=True)
class POI:
    poi_id: int
    lat: float
    lon: float
    name: str
    category: str           # one of POI_CATEGORIES
    poi_class: int = 0      # finer subclass (paper has 416 classes); we just use idx

    @property
    def category_idx(self) -> int:
        return CATEGORY_TO_IDX[self.category]


@dataclass(frozen=True)
class Trajectory:
    traj_id: int
    o_lat: float
    o_lon: float
    o_time_min: int          # minutes since midnight (in [0, 1440))
    d_lat: float
    d_lon: float
    d_time_min: int


@dataclass(frozen=True)
class MobilityEvent:
    """Paper Def. 2.3.

    `event_idx` is the integer id used as the Skip-Gram vocabulary index.
    """
    event_idx: int
    tile: TileId
    event_type: int          # EVENT_TYPE_O or EVENT_TYPE_D
    time_min: int
    poi: POI

    @property
    def hour_bucket(self) -> int:
        return (self.time_min // 60) % NUM_TIME_BUCKETS


@dataclass
class StreetViewShot:
    """A single shooting point holding 4 directional images.

    `image_features` is filled in lazily by the Stage-2 pipeline once the
    ResNet-18 has been run; here we just keep the raw image arrays.
    """
    shot_id: int
    tile: TileId
    images: list[np.ndarray] = field(default_factory=list)   # 4 × HxWx3 uint8


# --- Mobility-event construction (paper §2.1 / §3.2.1) --------------------------

def _nearest_poi_within(
    lat: float, lon: float, pois: list[POI], max_meters: float,
) -> POI | None:
    best, best_d = None, math.inf
    for p in pois:
        d = haversine_meters(lat, lon, p.lat, p.lon)
        if d <= max_meters and d < best_d:
            best, best_d = p, d
    return best


def build_mobility_events(
    trajectories: Iterable[Trajectory],
    pois: list[POI],
    *,
    snap_meters: float = DEFAULT_POI_SNAP_METERS,
    tile_level: int = DEFAULT_TILE_LEVEL,
) -> tuple[list[MobilityEvent], list[MobilityEvent]]:
    """Snap each trajectory's O and D to the nearest POI within `snap_meters`.

    Returns (o_events, d_events) aligned by index — `o_events[i]` and
    `d_events[i]` come from the same trajectory.
    """
    o_events: list[MobilityEvent] = []
    d_events: list[MobilityEvent] = []
    for tr in trajectories:
        o_poi = _nearest_poi_within(tr.o_lat, tr.o_lon, pois, snap_meters)
        d_poi = _nearest_poi_within(tr.d_lat, tr.d_lon, pois, snap_meters)
        if o_poi is None or d_poi is None:
            continue
        i = len(o_events)
        o_events.append(MobilityEvent(
            event_idx=i,
            tile=latlon_to_tile(tr.o_lat, tr.o_lon, tile_level),
            event_type=EVENT_TYPE_O,
            time_min=tr.o_time_min,
            poi=o_poi,
        ))
        d_events.append(MobilityEvent(
            event_idx=i,
            tile=latlon_to_tile(tr.d_lat, tr.d_lon, tile_level),
            event_type=EVENT_TYPE_D,
            time_min=tr.d_time_min,
            poi=d_poi,
        ))
    return o_events, d_events


def build_skipgram_pairs(
    o_events: list[MobilityEvent],
    d_events: list[MobilityEvent],
    *,
    time_threshold_min: int = DEFAULT_TIME_THRESHOLD_MIN,
) -> list[tuple[int, int]]:
    """Co-occurrence pairs of O-event indices.

    Paper §3.2.1: two O events co-occur iff their D events (a) end within
    `time_threshold_min` of each other and (b) share the same drop-off POI
    category. Returns symmetric (a, b) pairs with a != b.
    """
    by_cat: dict[int, list[int]] = {}
    for i, d in enumerate(d_events):
        by_cat.setdefault(d.poi.category_idx, []).append(i)

    pairs: list[tuple[int, int]] = []
    for cat, idxs in by_cat.items():
        # Sort by D time so we can use a sliding window.
        idxs.sort(key=lambda i: d_events[i].time_min)
        L = len(idxs)
        # Note: window restart per category is correct (j is local to the loop).
        for i in range(L):
            t_i = d_events[idxs[i]].time_min
            # Walk back to find left edge of window.
            j = i
            while j > 0 and d_events[idxs[j - 1]].time_min >= t_i - time_threshold_min:
                j -= 1
            for k in range(j, L):
                t_k = d_events[idxs[k]].time_min
                if t_k > t_i + time_threshold_min:
                    break
                if k == i:
                    continue
                a, b = o_events[idxs[i]].event_idx, o_events[idxs[k]].event_idx
                pairs.append((a, b))
    return pairs


# --- Synthetic data generator (drives example.py) -------------------------------

@dataclass
class SyntheticCity:
    """A miniature city with planted land-use clusters.

    Each tile belongs to one of 4 latent clusters with a distinctive POI-category
    distribution and OD time-of-day pattern. The cosine-similarity smoke test
    verifies that learned embeddings recover this latent structure.

    Coordinates are placed on a small lat/lon grid so distinct tiles really land
    in distinct level-18 Mercator cells.
    """

    n_tiles_per_side: int = 14            # 14*14 = 196 tiles
    trajectories_per_tile: int = 60
    pois_per_tile: int = 20
    images_per_tile: int = 1              # 1 shot × 4 images per tile
    image_hw: int = 64                    # tiny RGB squares for the smoke test
    seed: int = 0
    center_lat: float = 39.9042           # Beijing center
    center_lon: float = 116.4074
    # Spacing must be > one level-18 tile width (~120 m). 0.005° lat ≈ 555 m
    # at this latitude, so neighbors are clearly separated.
    tile_spacing_deg: float = 0.005

    # Cluster definitions: (name, top categories with weights, OD hour mode).
    CLUSTERS = [
        ("residential",
         {"residential": 0.6, "life_service": 0.2, "shopping": 0.1, "restaurant": 0.1},
         (8, 19)),         # leave home morning, return evening
        ("commercial",
         {"enterprise": 0.5, "shopping": 0.2, "restaurant": 0.2, "banking_finance": 0.1},
         (10, 14)),        # mid-day activity
        ("scenic",
         {"scenic_spot": 0.5, "recreational": 0.3, "cultural": 0.1, "hotel": 0.1},
         (11, 17)),        # leisure mid-day to afternoon
        ("educational",
         {"educational": 0.6, "institutional": 0.2, "sports": 0.1, "life_service": 0.1},
         (8, 13)),         # school start / mid-day
    ]

    def generate(self) -> tuple[
        list[POI], list[Trajectory], list[StreetViewShot], dict[TileId, int]
    ]:
        rng = random.Random(self.seed)
        np_rng = np.random.default_rng(self.seed)

        pois: list[POI] = []
        trajectories: list[Trajectory] = []
        shots: list[StreetViewShot] = []
        tile_to_cluster: dict[TileId, int] = {}
        next_poi_id = 0
        next_traj_id = 0
        next_shot_id = 0

        for gx in range(self.n_tiles_per_side):
            for gy in range(self.n_tiles_per_side):
                cluster_idx = (gx + gy) % len(self.CLUSTERS)  # checkerboard-ish
                # Add some noise so clusters aren't perfectly periodic.
                if rng.random() < 0.15:
                    cluster_idx = rng.randrange(len(self.CLUSTERS))
                _, cat_weights, (mode_o, mode_d) = self.CLUSTERS[cluster_idx]

                # Tile center in lat/lon.
                lat = self.center_lat + (gx - self.n_tiles_per_side / 2) * self.tile_spacing_deg
                lon = self.center_lon + (gy - self.n_tiles_per_side / 2) * self.tile_spacing_deg
                tid = latlon_to_tile(lat, lon)
                tile_to_cluster[tid] = cluster_idx

                # POIs inside the tile (jittered around the center within ~50 m).
                tile_pois: list[POI] = []
                cats, weights = zip(*cat_weights.items())
                for _ in range(self.pois_per_tile):
                    cat = rng.choices(cats, weights=weights, k=1)[0]
                    p = POI(
                        poi_id=next_poi_id,
                        lat=lat + np_rng.normal(0, 0.0002),   # ~22 m
                        lon=lon + np_rng.normal(0, 0.0002),
                        name=f"poi_{next_poi_id}",
                        category=cat,
                    )
                    tile_pois.append(p)
                    pois.append(p)
                    next_poi_id += 1

                # Trajectories originating in this tile.
                for _ in range(self.trajectories_per_tile):
                    # Origin: jitter near tile center.
                    o_lat = lat + np_rng.normal(0, 0.0002)
                    o_lon = lon + np_rng.normal(0, 0.0002)
                    o_time = int(np_rng.normal(mode_o * 60, 30)) % (24 * 60)

                    # Destination: another tile of the SAME cluster, so similar
                    # tiles see similar D-categories — that's what makes the
                    # Skip-Gram learnable.
                    dest_gx, dest_gy = self._sample_dest(rng, cluster_idx)
                    d_lat = self.center_lat + (dest_gx - self.n_tiles_per_side / 2) * self.tile_spacing_deg
                    d_lon = self.center_lon + (dest_gy - self.n_tiles_per_side / 2) * self.tile_spacing_deg
                    d_lat += np_rng.normal(0, 0.0002)
                    d_lon += np_rng.normal(0, 0.0002)
                    d_time = (o_time + int(abs(np_rng.normal(15, 5)))) % (24 * 60)

                    trajectories.append(Trajectory(
                        traj_id=next_traj_id,
                        o_lat=o_lat, o_lon=o_lon, o_time_min=o_time,
                        d_lat=d_lat, d_lon=d_lon, d_time_min=d_time,
                    ))
                    next_traj_id += 1

                # Synthetic street-view images: cluster-distinctive color tints.
                # Real Places365 weights would have hooks for actual scenes; for
                # the smoke test we just need cluster-correlated pixel statistics.
                tint = np.array([
                    [200, 100, 100],   # residential = warm
                    [100, 100, 200],   # commercial  = cool
                    [100, 200, 100],   # scenic      = green
                    [200, 200, 100],   # educational = yellow
                ])[cluster_idx]
                for _ in range(self.images_per_tile):
                    imgs = []
                    for _ in range(4):       # 4 directions
                        base = np_rng.integers(0, 60, size=(self.image_hw, self.image_hw, 3), dtype=np.uint8)
                        img = np.clip(base + tint, 0, 255).astype(np.uint8)
                        imgs.append(img)
                    shots.append(StreetViewShot(shot_id=next_shot_id, tile=tid, images=imgs))
                    next_shot_id += 1

        return pois, trajectories, shots, tile_to_cluster

    def _sample_dest(self, rng: random.Random, cluster_idx: int) -> tuple[int, int]:
        # Resample tile coordinates until we land on the same cluster.
        for _ in range(20):
            gx = rng.randrange(self.n_tiles_per_side)
            gy = rng.randrange(self.n_tiles_per_side)
            if (gx + gy) % len(self.CLUSTERS) == cluster_idx:
                return gx, gy
        return rng.randrange(self.n_tiles_per_side), rng.randrange(self.n_tiles_per_side)
