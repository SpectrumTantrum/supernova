"""Build TransFlower (Region, Flow) records from LEHD LODES8 MA CSVs.

Pipeline
--------
1.  Block-level WAC + crosswalk are aggregated to Census Tract (paper §4.1.3
    operates at tract resolution): tract GEOID = first 11 chars of the
    15-digit block GEOID (state + county + tract).

2.  Per-tract place_features = sum of NAICS supersector job counts
    CNS02..CNS20 (19 columns, dropping CNS01 agriculture which is mostly
    zero in MA). Centroid lat/lon = mean of constituent block centroids.
    `population` = total workplace jobs C000 — the only per-tract scalar
    available in WAC; serves as the mass term for outflow scaling in the
    flow predictor.

3.  OD flows: collapse w_geocode and h_geocode to tract, sum S000, drop
    self-flows (intra-tract commutes, which dominate counts and are
    uninformative for cross-region modelling).

Vintages: any of 2021, 2020, 2019 (whichever was downloaded). The xwalk
is a single file with no year suffix.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Iterable

import numpy as np
import pandas as pd

# Add the transflower/ parent to sys.path so we can import its data.py.
# parents[0] = lodes_ma, parents[1] = datasets, parents[2] = transflower.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from data import Flow, NUM_PLACE_CATEGORIES, Region  # noqa: E402


RAW_DIR = pathlib.Path(__file__).resolve().parent / "raw"

# 19 of the 20 NAICS supersector columns in WAC (CNS01 agriculture dropped:
# very sparse statewide, dominated by zeros).
NAICS_COLS = [f"CNS{i:02d}" for i in range(2, 21)]
assert len(NAICS_COLS) == NUM_PLACE_CATEGORIES, (
    f"expected {NUM_PLACE_CATEGORIES} NAICS cols, got {len(NAICS_COLS)}"
)


def _find_csv(prefix: str) -> pathlib.Path:
    """Return the most recent matching CSV in raw/ (e.g. ma_wac_*.csv.gz)."""
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"raw/ does not exist: {RAW_DIR}. Run fetch.py first.")
    matches = sorted(RAW_DIR.glob(f"{prefix}*.csv.gz"))
    if not matches:
        raise FileNotFoundError(
            f"no {prefix}*.csv.gz in {RAW_DIR}. Run fetch.py first."
        )
    return matches[-1]


def _to_tract(geoid15: pd.Series) -> pd.Series:
    """15-digit block GEOID -> 11-digit tract GEOID (first 11 chars)."""
    return geoid15.astype(str).str.zfill(15).str[:11]


def load_regions(
    county_prefix: str | None = None,
) -> tuple[list[Region], dict[str, int]]:
    """Build per-tract Region records.

    Args:
        county_prefix: optional 5-digit state+county FIPS to filter on
            (e.g. "25025" for Suffolk County, MA). Tract GEOIDs starting
            with this prefix are kept; others are dropped.

    Returns:
        regions: list[Region] indexed by enumeration order (region_id == idx).
        geoid_to_id: dict mapping the 11-digit tract GEOID string to that
            integer region_id. Needed by load_flows() to join OD records.
    """
    wac_path = _find_csv("ma_wac_")
    xwalk_path = _find_csv("ma_xwalk")

    # Read WAC. w_geocode as string (otherwise pandas drops the leading 25).
    wac_cols = ["w_geocode", "C000"] + NAICS_COLS
    wac = pd.read_csv(
        wac_path,
        usecols=wac_cols,
        dtype={"w_geocode": str, **{c: "float32" for c in ["C000"] + NAICS_COLS}},
    )
    wac["tract"] = _to_tract(wac["w_geocode"])

    if county_prefix is not None:
        wac = wac[wac["tract"].str.startswith(county_prefix)]

    # Aggregate WAC -> tract.
    agg_cols = ["C000"] + NAICS_COLS
    tract_wac = wac.groupby("tract", sort=True)[agg_cols].sum()

    # Read xwalk (block centroids + tract GEOID column "trct").
    xwalk = pd.read_csv(
        xwalk_path,
        usecols=["tabblk2020", "trct", "blklatdd", "blklondd"],
        dtype={"tabblk2020": str, "trct": str},
    )
    xwalk["tract"] = xwalk["trct"].astype(str).str.zfill(11)
    if county_prefix is not None:
        xwalk = xwalk[xwalk["tract"].str.startswith(county_prefix)]
    # Mean centroid per tract.
    tract_geo = xwalk.groupby("tract")[["blklatdd", "blklondd"]].mean()

    joined = tract_wac.join(tract_geo, how="inner").dropna(
        subset=["blklatdd", "blklondd"]
    )

    regions: list[Region] = []
    geoid_to_id: dict[str, int] = {}
    for rid, (geoid, row) in enumerate(joined.iterrows()):
        feats = np.array([row[c] for c in NAICS_COLS], dtype=np.float32)
        regions.append(
            Region(
                region_id=rid,
                lat=float(row["blklatdd"]),
                lon=float(row["blklondd"]),
                place_features=feats,
                population=float(row["C000"]),
            )
        )
        geoid_to_id[geoid] = rid

    return regions, geoid_to_id


def load_flows(
    geoid_to_id: dict[str, int],
    drop_self_flows: bool = True,
) -> list[Flow]:
    """Build Flow records from LODES OD main.

    Flows whose origin or destination tract isn't in `geoid_to_id` are
    dropped (e.g. when caller filtered to a single county; cross-county
    commutes lose one endpoint and are excluded).
    """
    try:
        od_path = _find_csv("ma_od_")
    except FileNotFoundError:
        print("  load_flows: no OD CSV found (fetch.py --small skips it). "
              "Returning [].", file=sys.stderr)
        return []

    od = pd.read_csv(
        od_path,
        usecols=["w_geocode", "h_geocode", "S000"],
        dtype={"w_geocode": str, "h_geocode": str, "S000": "float32"},
    )
    od["w_tract"] = _to_tract(od["w_geocode"])
    od["h_tract"] = _to_tract(od["h_geocode"])

    if drop_self_flows:
        od = od[od["w_tract"] != od["h_tract"]]

    # Filter to known tracts before groupby (much smaller than the raw OD).
    keep = od["w_tract"].isin(geoid_to_id) & od["h_tract"].isin(geoid_to_id)
    od = od[keep]

    # h_geocode = home (origin), w_geocode = workplace (destination).
    grouped = od.groupby(["h_tract", "w_tract"], sort=False)["S000"].sum()

    flows: list[Flow] = []
    for (h_tract, w_tract), count in grouped.items():
        flows.append(
            Flow(
                o_id=geoid_to_id[h_tract],
                d_id=geoid_to_id[w_tract],
                count=float(count),
            )
        )
    return flows


def main() -> int:
    print("Loading LODES MA tract regions (full state) …")
    regions, geoid_to_id = load_regions()
    print(f"  n_regions: {len(regions)}")
    if regions:
        r0 = regions[0]
        print(f"  sample region_id=0  geoid={list(geoid_to_id)[0]}")
        print(f"    lat,lon          = ({r0.lat:.4f}, {r0.lon:.4f})")
        print(f"    population (C000)= {r0.population:.1f}")
        print(f"    place_features   = {r0.place_features.tolist()}")
        print(f"    feature_vector dim = {r0.feature_vector().shape}")

    print("\nLoading LODES MA flows …")
    flows = load_flows(geoid_to_id)
    print(f"  n_flows: {len(flows):,}")
    if flows:
        f0 = flows[0]
        print(f"  sample flow: o={f0.o_id}  d={f0.d_id}  count={f0.count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
