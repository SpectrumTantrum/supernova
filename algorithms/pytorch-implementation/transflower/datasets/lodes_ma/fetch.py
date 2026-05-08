"""Download LEHD LODES8 Massachusetts data for TransFlower.

Sources (US Census Bureau, public domain):
    - LODES8 OD main:
        https://lehd.ces.census.gov/data/lodes/LODES8/ma/od/ma_od_main_JT00_<year>.csv.gz
      Block-to-block primary-job commute counts (column S000) for all MA
      workers whose home and workplace are both inside Massachusetts.
    - LODES8 WAC (workplace area characteristics):
        https://lehd.ces.census.gov/data/lodes/LODES8/ma/wac/ma_wac_S000_JT00_<year>.csv.gz
      Per-block job counts split by NAICS supersector (CNS01..CNS20).
    - LODES8 geographic crosswalk:
        https://lehd.ces.census.gov/data/lodes/LODES8/ma/ma_xwalk.csv.gz
      Block-level metadata: centroid lat/lon, parent tract GEOID.

License: U.S. federal works are not subject to copyright (17 U.S.C. § 105);
the LEHD program publishes LODES under that public-domain status. See
https://lehd.ces.census.gov/data/ for the data-use notice.

The script is idempotent — already-downloaded files are skipped. A HEAD
pre-flight is run on every URL so failures surface before any 50 MB
download starts.

Usage:
    python fetch.py            # all three CSVs
    python fetch.py --small    # WAC + xwalk only (skips ~50 MB OD download)
    python fetch.py --year 2020
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import requests


BASE = "https://lehd.ces.census.gov/data/lodes/LODES8/ma"
RAW_DIR = pathlib.Path(__file__).resolve().parent / "raw"

# Vintages to try, newest first. LODES is republished annually; 2021 is the
# most recent vintage in the LODES8 release as of this commit.
DEFAULT_YEARS = (2021, 2020, 2019)


def _url(kind: str, year: int) -> str:
    if kind == "od":
        return f"{BASE}/od/ma_od_main_JT00_{year}.csv.gz"
    if kind == "wac":
        return f"{BASE}/wac/ma_wac_S000_JT00_{year}.csv.gz"
    if kind == "xwalk":
        # The xwalk is not year-stamped; one file per state.
        return f"{BASE}/ma_xwalk.csv.gz"
    raise ValueError(kind)


def _local(kind: str, year: int) -> pathlib.Path:
    if kind == "xwalk":
        return RAW_DIR / "ma_xwalk.csv.gz"
    return RAW_DIR / f"ma_{kind}_{year}.csv.gz"


def _head_ok(url: str) -> bool:
    try:
        r = requests.head(url, allow_redirects=True, timeout=30)
    except requests.RequestException as exc:
        print(f"  HEAD {url}\n    -> network error: {exc}", file=sys.stderr)
        return False
    if r.status_code != 200:
        print(f"  HEAD {url}\n    -> HTTP {r.status_code}", file=sys.stderr)
        return False
    return True


def _download(url: str, dest: pathlib.Path) -> None:
    print(f"  GET {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        tmp = dest.with_suffix(dest.suffix + ".part")
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100.0 * done / total
                    print(f"\r    {done / 1e6:6.1f} / {total / 1e6:6.1f} MB ({pct:5.1f}%)",
                          end="", flush=True)
        print()
        tmp.rename(dest)


def fetch_one(kind: str, year: int) -> pathlib.Path | None:
    """Returns local path on success, None if every candidate URL 404s."""
    # xwalk has no year suffix; OD/WAC fall back to older vintages on 404.
    if kind == "xwalk":
        candidates = [0]
    else:
        candidates = [year] + [y for y in DEFAULT_YEARS if y != year]

    for y in candidates:
        url = _url(kind, y)
        dest = _local(kind, y)
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  skip (cached): {dest.name}")
            return dest
        if not _head_ok(url):
            continue
        _download(url, dest)
        return dest

    print(f"  FAIL: no working vintage for kind={kind!r}; tried {candidates}",
          file=sys.stderr)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=DEFAULT_YEARS[0])
    ap.add_argument("--small", action="store_true",
                    help="Skip the OD CSV (still ~5 MB WAC + ~30 MB xwalk).")
    args = ap.parse_args()

    RAW_DIR.mkdir(exist_ok=True)
    print(f"raw dir: {RAW_DIR}")
    print(f"vintage: {args.year}  (fallbacks: {DEFAULT_YEARS})")

    kinds = ["wac", "xwalk"] if args.small else ["wac", "xwalk", "od"]
    results: dict[str, pathlib.Path | None] = {}
    for kind in kinds:
        print(f"\n[{kind}]")
        results[kind] = fetch_one(kind, args.year)

    if any(v is None for v in results.values()):
        print("\nOne or more downloads failed.", file=sys.stderr)
        return 1
    print("\nAll downloads OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
