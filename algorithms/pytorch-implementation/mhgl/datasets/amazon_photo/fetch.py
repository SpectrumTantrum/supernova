"""Fetch the Amazon Photo benchmark NPZ for MHGL.

Source
------
Shchur, Mumme, Bojchevski, Günnemann, "Pitfalls of Graph Neural Network
Evaluation", *Relational Representation Learning Workshop, NeurIPS 2018*.
The companion benchmark repo (shchur/gnn-benchmark, MIT-licensed) hosts the
preprocessed Amazon co-purchase Photo subgraph as a single NumPy archive:

    7,487 nodes  ·  ~119 k directed edges  ·  745 features  ·  8 classes

License: MIT (see https://github.com/shchur/gnn-benchmark/blob/master/LICENSE).

Network behaviour
-----------------
- Pre-flight HEAD on each URL; on 404 fall through to the next mirror.
- Idempotent: skips the download if ``raw/amazon_electronics_photo.npz``
  already exists with non-zero size.
- ``--small`` is a no-op (the archive is ~11 MB).
- Exits non-zero if every mirror fails — never fabricates a placeholder.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import requests


URLS: tuple[str, ...] = (
    "https://github.com/shchur/gnn-benchmark/raw/master/data/npz/amazon_electronics_photo.npz",
    "https://github.com/shchur/gnn-benchmark/raw/main/data/npz/amazon_electronics_photo.npz",
)

RAW_DIR = pathlib.Path(__file__).resolve().parent / "raw"
TARGET = RAW_DIR / "amazon_electronics_photo.npz"


def _head_ok(url: str, timeout: float = 15.0) -> bool:
    try:
        r = requests.head(url, allow_redirects=True, timeout=timeout)
        return r.status_code == 200
    except requests.RequestException:
        return False


def _download(url: str, dst: pathlib.Path, timeout: float = 60.0) -> None:
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        tmp = dst.with_suffix(dst.suffix + ".part")
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                if chunk:
                    f.write(chunk)
        tmp.replace(dst)


def fetch() -> pathlib.Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if TARGET.exists() and TARGET.stat().st_size > 0:
        print(f"[fetch] Already present: {TARGET} ({TARGET.stat().st_size:,} bytes)")
        return TARGET

    last_err: str | None = None
    for url in URLS:
        print(f"[fetch] HEAD {url}")
        if not _head_ok(url):
            last_err = f"HEAD failed for {url}"
            print(f"[fetch]   -> miss; trying next mirror")
            continue
        print(f"[fetch] GET  {url}")
        try:
            _download(url, TARGET)
        except requests.RequestException as e:
            last_err = f"GET failed for {url}: {e}"
            print(f"[fetch]   -> {last_err}")
            continue
        print(f"[fetch] Wrote {TARGET} ({TARGET.stat().st_size:,} bytes)")
        return TARGET

    raise RuntimeError(
        f"All mirrors for amazon_electronics_photo.npz failed. Last error: {last_err}. "
        f"You can manually drop the file at {TARGET}."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--small", action="store_true",
                    help="No-op (archive is ~11 MB); accepted for cross-dataset CLI parity.")
    ap.parse_args()
    fetch()
    return 0


if __name__ == "__main__":
    sys.exit(main())
