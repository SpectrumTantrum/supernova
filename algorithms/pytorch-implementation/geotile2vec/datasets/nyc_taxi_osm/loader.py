"""Convert raw Citi Bike trip CSVs + Manhattan OSM POIs into POI / Trajectory lists.

Reads files produced by `fetch.py` from `./raw/` and emits records compatible
with the parent `data.py`. POIs outside `MANHATTAN_BBOX` are dropped, and
trajectories whose start OR end falls outside the bbox are dropped — the 50 m
POI snap in `build_mobility_events` would discard them anyway.
"""

from __future__ import annotations

import json
import pathlib
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

# Parent algorithm folder has no __init__.py; jump up two levels:
#   parents[0] = nyc_taxi_osm/  parents[1] = datasets/  parents[2] = geotile2vec/
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from data import POI, POI_CATEGORIES, Trajectory  # noqa: E402

RAW = Path(__file__).resolve().parent / "raw"

# Manhattan bbox (south, west, north, east) — must match fetch.py.
MANHATTAN_BBOX = (40.700, -74.020, 40.880, -73.910)

# Map raw OSM tag values to the 16 paper categories. Anything not listed is
# dropped on load. This is intentionally conservative — the paper's category
# scheme has 16 buckets and we only emit values that map unambiguously.
OSM_TAG_TO_CATEGORY: dict[tuple[str, str], str] = {
    # amenity
    ("amenity", "restaurant"):       "restaurant",
    ("amenity", "cafe"):             "restaurant",
    ("amenity", "bar"):              "restaurant",
    ("amenity", "pub"):              "restaurant",
    ("amenity", "fast_food"):        "restaurant",
    ("amenity", "food_court"):       "restaurant",
    ("amenity", "school"):           "educational",
    ("amenity", "university"):       "educational",
    ("amenity", "college"):          "educational",
    ("amenity", "kindergarten"):     "educational",
    ("amenity", "library"):          "educational",
    ("amenity", "bank"):             "banking_finance",
    ("amenity", "atm"):              "banking_finance",
    ("amenity", "bureau_de_change"): "banking_finance",
    ("amenity", "clinic"):           "medical",
    ("amenity", "hospital"):         "medical",
    ("amenity", "pharmacy"):         "medical",
    ("amenity", "doctors"):          "medical",
    ("amenity", "dentist"):          "medical",
    ("amenity", "veterinary"):       "medical",
    ("amenity", "fuel"):             "car_related",
    ("amenity", "parking"):          "car_related",
    ("amenity", "car_rental"):       "car_related",
    ("amenity", "car_wash"):         "car_related",
    ("amenity", "charging_station"): "car_related",
    ("amenity", "theatre"):          "cultural",
    ("amenity", "cinema"):           "cultural",
    ("amenity", "arts_centre"):      "cultural",
    ("amenity", "community_centre"): "cultural",
    ("amenity", "place_of_worship"): "institutional",
    ("amenity", "townhall"):         "institutional",
    ("amenity", "courthouse"):       "institutional",
    ("amenity", "police"):           "institutional",
    ("amenity", "fire_station"):     "institutional",
    ("amenity", "post_office"):      "institutional",
    ("amenity", "marketplace"):      "shopping",
    # shop — almost all map to shopping; a handful go elsewhere.
    ("shop", "supermarket"):         "shopping",
    ("shop", "mall"):                "shopping",
    ("shop", "department_store"):    "shopping",
    ("shop", "convenience"):         "shopping",
    ("shop", "clothes"):             "shopping",
    ("shop", "shoes"):               "shopping",
    ("shop", "electronics"):         "shopping",
    ("shop", "books"):               "shopping",
    ("shop", "bakery"):              "shopping",
    ("shop", "butcher"):             "shopping",
    ("shop", "greengrocer"):         "shopping",
    ("shop", "florist"):             "shopping",
    ("shop", "hardware"):            "shopping",
    ("shop", "jewelry"):             "shopping",
    ("shop", "gift"):                "shopping",
    ("shop", "car"):                 "car_related",
    ("shop", "car_repair"):          "car_related",
    # leisure
    ("leisure", "park"):             "recreational",
    ("leisure", "garden"):           "recreational",
    ("leisure", "playground"):       "recreational",
    ("leisure", "fitness_centre"):   "sports",
    ("leisure", "sports_centre"):    "sports",
    ("leisure", "stadium"):          "sports",
    ("leisure", "pitch"):            "sports",
    ("leisure", "swimming_pool"):    "sports",
    # tourism
    ("tourism", "hotel"):            "hotel",
    ("tourism", "hostel"):           "hotel",
    ("tourism", "motel"):            "hotel",
    ("tourism", "guest_house"):      "hotel",
    ("tourism", "museum"):           "cultural",
    ("tourism", "gallery"):          "cultural",
    ("tourism", "attraction"):       "scenic_spot",
    ("tourism", "viewpoint"):        "scenic_spot",
    ("tourism", "artwork"):          "scenic_spot",
}

CITIBIKE_COLS = [
    "starttime", "stoptime",
    "start station latitude", "start station longitude",
    "end station latitude", "end station longitude",
]


@dataclass
class LoadedData:
    pois: list[POI]
    trajectories: list[Trajectory]
    bbox: tuple[float, float, float, float]


def _in_bbox(lat: float, lon: float, bbox: tuple[float, float, float, float]) -> bool:
    s, w, n, e = bbox
    return s <= lat <= n and w <= lon <= e


def load_pois(path: Path | None = None) -> list[POI]:
    path = path or (RAW / "osm_manhattan.json")
    if not path.exists():
        raise FileNotFoundError(f"Missing {path} — run `python fetch.py` first.")
    with open(path) as f:
        payload = json.load(f)

    pois: list[POI] = []
    next_id = 0
    for el in payload.get("elements", []):
        if el.get("type") != "node":
            continue
        lat, lon = el.get("lat"), el.get("lon")
        if lat is None or lon is None:
            continue
        if not _in_bbox(lat, lon, MANHATTAN_BBOX):
            continue
        tags = el.get("tags", {}) or {}
        category: str | None = None
        for key in ("amenity", "shop", "leisure", "tourism"):
            val = tags.get(key)
            if val and (key, val) in OSM_TAG_TO_CATEGORY:
                category = OSM_TAG_TO_CATEGORY[(key, val)]
                break
        if category is None:
            continue
        assert category in POI_CATEGORIES, f"Bad category mapping: {category}"
        name = tags.get("name") or f"{category}_{next_id}"
        pois.append(POI(
            poi_id=next_id, lat=float(lat), lon=float(lon),
            name=name, category=category,
        ))
        next_id += 1
    return pois


def load_trajectories(
    path: Path | None = None,
    *,
    max_rows: int | None = None,
) -> list[Trajectory]:
    if path is None:
        path = RAW / "citibike_201506.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No Citi Bike CSV at {path} — run `python fetch.py [--small]` first."
        )

    df = pd.read_csv(
        path, usecols=CITIBIKE_COLS,
        nrows=max_rows * 4 if max_rows else None,   # over-read, then drop nans + bbox
    )
    df = df.dropna()

    s, w, n, e = MANHATTAN_BBOX
    in_box = (
        df["start station latitude"].between(s, n)
        & df["start station longitude"].between(w, e)
        & df["end station latitude"].between(s, n)
        & df["end station longitude"].between(w, e)
    )
    df = df[in_box]

    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=0).reset_index(drop=True)

    # Citi Bike timestamps come in two formats across years; pandas handles
    # both with format=None + errors='coerce'.
    starttime = pd.to_datetime(df["starttime"], errors="coerce")
    stoptime = pd.to_datetime(df["stoptime"], errors="coerce")
    valid = starttime.notna() & stoptime.notna()
    df = df[valid.to_numpy()]
    starttime = starttime[valid.to_numpy()]
    stoptime = stoptime[valid.to_numpy()]

    o_min = (starttime.dt.hour * 60 + starttime.dt.minute).astype(int).to_numpy()
    d_min = (stoptime.dt.hour * 60 + stoptime.dt.minute).astype(int).to_numpy()
    o_lat = df["start station latitude"].astype(float).to_numpy()
    o_lon = df["start station longitude"].astype(float).to_numpy()
    d_lat = df["end station latitude"].astype(float).to_numpy()
    d_lon = df["end station longitude"].astype(float).to_numpy()

    return [
        Trajectory(
            traj_id=i,
            o_lat=float(o_lat[i]), o_lon=float(o_lon[i]), o_time_min=int(o_min[i]),
            d_lat=float(d_lat[i]), d_lon=float(d_lon[i]), d_time_min=int(d_min[i]),
        )
        for i in range(len(df))
    ]


def load_all(
    *,
    max_trajectories: int | None = None,
    max_pois: int | None = None,
) -> LoadedData:
    pois = load_pois()
    if max_pois is not None and len(pois) > max_pois:
        sampled = random.Random(0).sample(pois, max_pois)
        # Reassign poi_ids contiguously so they remain a dense [0, n) range.
        pois = [replace(p, poi_id=i) for i, p in enumerate(sampled)]
    trajectories = load_trajectories(max_rows=max_trajectories)
    return LoadedData(pois=pois, trajectories=trajectories, bbox=MANHATTAN_BBOX)


def main() -> int:
    print("Loading POIs from raw/osm_manhattan.json …")
    pois = load_pois()
    print(f"  n_pois = {len(pois):,}")
    cat_counts: dict[str, int] = {}
    for p in pois:
        cat_counts[p.category] = cat_counts.get(p.category, 0) + 1
    print("  POI categories (top 10):")
    for cat, n in sorted(cat_counts.items(), key=lambda kv: -kv[1])[:10]:
        print(f"    {cat:<20s} {n:>6d}")

    print("\nLoading trajectories …")
    trajectories = load_trajectories(max_rows=50_000)
    print(f"  n_trajectories = {len(trajectories):,}  (sampled to <=50k)")
    if trajectories:
        times = [t.o_time_min for t in trajectories]
        print(f"  pickup hours: min={min(times)//60:02d}:{min(times)%60:02d}  "
              f"max={max(times)//60:02d}:{max(times)%60:02d}  "
              f"mean={sum(times)/len(times)/60:.2f}h")
        sample = trajectories[0]
        print(f"  sample trajectory: {sample}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
