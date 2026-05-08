"""Download the DBLPv7 and ACMv9 cross-network citation datasets.

Source: Shen et al., "Adversarial Deep Network Embedding for Cross-network
Node Classification", AAAI 2020. The paper authors distribute the .mat
files in their reference codebase at:

    https://github.com/shenxiaocam/ACDNE/tree/master/ACDNE_codes/input

Files (verified live 2026-05-08):
    dblpv7.mat  ~559 KB  — DBLP citation network sliced to 2004–2008
    acmv9.mat  ~1.1 MB  — ACM citation network sliced to ≥2009

License: the upstream GitHub repository does not include a LICENSE file.
The data is published alongside an academic paper for research-reproduction
purposes; treat as research-only use and cite Shen et al. AAAI 2020 in any
downstream work.

Both .mat files are MATLAB v5 format (loadmat-compatible, no h5py needed).

Usage:
    python fetch.py             # download both files into ./raw/
    python fetch.py --small     # no-op flag (both files together are ~1.7 MB)

Idempotent — re-running is a no-op once files are present and match the
expected byte-length advertised by the GitHub Content-Length header.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import requests


URLS = {
    "dblpv7.mat": "https://raw.githubusercontent.com/shenxiaocam/ACDNE/master/ACDNE_codes/input/dblpv7.mat",
    "acmv9.mat": "https://raw.githubusercontent.com/shenxiaocam/ACDNE/master/ACDNE_codes/input/acmv9.mat",
}

RAW_DIR = pathlib.Path(__file__).resolve().parent / "raw"


def _remote_size(url: str) -> int | None:
    """HEAD-probe the upstream Content-Length, or None if unavailable."""
    resp = requests.head(url, allow_redirects=True, timeout=20)
    resp.raise_for_status()
    cl = resp.headers.get("Content-Length")
    return int(cl) if cl is not None else None


def _download(url: str, dest: pathlib.Path) -> None:
    """Stream ``url`` to ``dest`` with a temp-file rename so partials don't stick."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=64 * 1024):
                if chunk:
                    fh.write(chunk)
    tmp.replace(dest)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--small",
        action="store_true",
        help="No-op: both files together are ~1.7 MB; flag kept for parity "
             "with sibling dataset adapters.",
    )
    args = ap.parse_args()
    _ = args.small  # parsed for parity; files are already small.

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Target directory: {RAW_DIR}")

    for name, url in URLS.items():
        dest = RAW_DIR / name
        expected = _remote_size(url)
        if dest.exists():
            actual = dest.stat().st_size
            if expected is None or actual == expected:
                print(f"  {name}: present ({actual} bytes) — skipping.")
                continue
            print(f"  {name}: size mismatch (have {actual}, expected {expected}) — re-downloading.")
        size_str = f"{expected} bytes" if expected else "unknown size"
        print(f"  {name}: downloading from {url} ({size_str})…")
        _download(url, dest)
        print(f"  {name}: wrote {dest.stat().st_size} bytes -> {dest}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
