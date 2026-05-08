"""Idempotent downloader for Citi Bike 2015-06 trips + Manhattan OSM POIs.

Sources & licences:
  - Citi Bike system data (NYC bike-share):
      https://citibikenyc.com/system-data
      Files at https://s3.amazonaws.com/tripdata/<bundle>.zip
      Released by Lyft / Citi Bike under the Citi Bike Data License Agreement
      (https://ride.citibikenyc.com/data-sharing-policy) — permits public
      research and non-commercial reuse with attribution.
      We use Citi Bike June 2015 because its CSV schema includes raw start/end
      station latitude / longitude, plus pickup + drop-off datetimes — a
      direct match for Geo-Tile2Vec's `Trajectory` contract.
      (Spec originally called for NYC TLC yellow-taxi 2015-06; TLC has since
      retroactively reformatted all trip-data parquets to LocationID-only,
      removing raw lat/lon. Citi Bike fills the same role for the same era
      and same NYC region.)
  - OpenStreetMap POIs via the Overpass API:
      https://overpass-api.de/api/interpreter
      OSM data is © OpenStreetMap contributors, licensed under the
      Open Database License (ODbL) — https://www.openstreetmap.org/copyright

Run:
    python fetch.py            # full June 2015 CSV (~660 MB unzipped)
    python fetch.py --small    # cap extracted CSV at ~50 MB (~200k rows)

The zip download is always ~276 MB regardless of `--small` — zip metadata
sits at the end of the file, so the whole archive must be fetched before
any inner file can be opened. `--small` only reduces the CSV that lands on
disk in `raw/`. Re-running is a no-op if `raw/` is already populated.
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import requests

RAW = Path(__file__).resolve().parent / "raw"

CITIBIKE_ZIP_URL = "https://s3.amazonaws.com/tripdata/2015-citibike-tripdata.zip"
CITIBIKE_INNER_CSV = "2015-citibike-tripdata/6_June/201506-citibike-tripdata_1.csv"
CITIBIKE_OUT = "citibike_201506.csv"

# Manhattan bounding box (south, west, north, east).
MANHATTAN_BBOX = (40.700, -74.020, 40.880, -73.910)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
USER_AGENT = "supernova-geotile2vec/1.0 (https://github.com/SpectrumTantrum/supernova)"

# Overpass QL: nodes with amenity/shop/leisure/tourism in the bbox.
# `out tags center` keeps only what we need (no full geometry).
OVERPASS_QUERY = """
[out:json][timeout:90];
(
  node["amenity"]({s},{w},{n},{e});
  node["shop"]({s},{w},{n},{e});
  node["leisure"]({s},{w},{n},{e});
  node["tourism"]({s},{w},{n},{e});
);
out tags center;
""".strip()


def _head_or_die(url: str) -> int:
    try:
        r = requests.head(url, allow_redirects=True, timeout=30,
                          headers={"User-Agent": USER_AGENT})
    except requests.RequestException as e:
        raise RuntimeError(f"HEAD failed for {url}: {e}") from e
    if not r.ok:
        raise RuntimeError(f"HEAD {url} returned HTTP {r.status_code}")
    return int(r.headers.get("Content-Length", "0"))


def fetch_citibike(small: bool) -> Path:
    dest = RAW / CITIBIKE_OUT
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"  skipping citibike: {dest.name} already present ({dest.stat().st_size:,} bytes)")
        return dest

    expected = _head_or_die(CITIBIKE_ZIP_URL)
    print(f"  downloading {CITIBIKE_ZIP_URL} ({expected:,} bytes) …")

    # The full 2015 zip is ~276 MB; we only need one inner CSV (~660 MB raw,
    # ~150 MB compressed inside the zip). Stream into RAM and unpack just
    # that file — avoids a 600+ MB intermediate file on disk.
    buf = io.BytesIO()
    with requests.get(CITIBIKE_ZIP_URL, stream=True, timeout=300,
                      headers={"User-Agent": USER_AGENT}) as r:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=1 << 20):
            if chunk:
                buf.write(chunk)
    if expected and buf.tell() != expected:
        raise RuntimeError(f"Short read: got {buf.tell()} bytes, expected {expected}")
    buf.seek(0)

    with zipfile.ZipFile(buf) as z:
        names = z.namelist()
        if CITIBIKE_INNER_CSV not in names:
            raise RuntimeError(
                f"Inner CSV {CITIBIKE_INNER_CSV!r} missing from zip; got {names[:5]}…"
            )
        with z.open(CITIBIKE_INNER_CSV) as src:
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            cap = 50 * 1024 * 1024 if small else None   # ~50 MB ≈ 200k trips
            with open(tmp, "wb") as f:
                written = 0
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    written += len(chunk)
                    if cap is not None and written >= cap:
                        break
            if cap is not None:
                # Trim trailing partial row (chunk boundary rarely aligns to '\n').
                with open(tmp, "rb") as rf:
                    data = rf.read()
                last_nl = data.rfind(b"\n")
                if last_nl != -1 and last_nl < len(data) - 1:
                    with open(tmp, "wb") as wf:
                        wf.write(data[: last_nl + 1])
            tmp.rename(dest)
            print(f"  wrote {dest.name} ({dest.stat().st_size:,} bytes)")
    return dest


def fetch_osm() -> Path:
    dest = RAW / "osm_manhattan.json"
    if dest.exists() and dest.stat().st_size > 10_000:
        print(f"  skipping OSM: {dest.name} already present ({dest.stat().st_size:,} bytes)")
        return dest
    s, w, n, e = MANHATTAN_BBOX
    query = OVERPASS_QUERY.format(s=s, w=w, n=n, e=e)
    print(f"  querying Overpass for amenity/shop/leisure/tourism in {MANHATTAN_BBOX} …")
    r = requests.post(
        OVERPASS_URL,
        data={"data": query},
        timeout=180,
        headers={"User-Agent": USER_AGENT},
    )
    if not r.ok:
        raise RuntimeError(f"Overpass returned HTTP {r.status_code}: {r.text[:300]}")
    payload = r.content
    if len(payload) < 10_000:
        raise RuntimeError(f"Overpass response suspiciously small: {len(payload)} bytes")
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_bytes(payload)
    tmp.rename(dest)
    print(f"  wrote {dest.name} ({len(payload):,} bytes)")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true",
                    help="Cap extracted CSV at ~50 MB on disk (zip download is always ~276 MB).")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    print(f"Raw data dir: {RAW}")
    fetch_citibike(small=args.small)
    fetch_osm()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
