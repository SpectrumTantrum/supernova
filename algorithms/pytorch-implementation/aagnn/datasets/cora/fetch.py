"""Download the Cora citation network for AAGNN anomaly detection.

Source / provenance
-------------------
Cora is a citation network of 2,708 ML papers, 5,429 citation links, and
1,433 binary bag-of-words features per node. Released by the LINQS group at
UC Santa Cruz under a research-use licence — see
    https://linqs.org/datasets/#cora
The canonical reference is

    Sen, Namata, Bilgic, Getoor, Galligher & Eliassi-Rad (2008).
    "Collective Classification in Network Data." AI Magazine 29(3).

Mirrors are tried in order; the first that returns HTTP 200 wins:
    1. https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz   (primary)
    2. https://github.com/pyg-team/pyg-datasets/raw/master/cora.tgz  (community)
    3. https://raw.githubusercontent.com/kimiyoung/planetoid/master/data/ind.cora.x
       (Planetoid split; different format — last-resort, only if the first two
       fail. The loader currently expects the LINQS .content/.cites layout and
       will raise if only the Planetoid pickle is present.)

Raw archive lands in ``raw/cora.tgz`` and is extracted into ``raw/cora/``.
This script is idempotent — re-running with everything already on disk is a
no-op. The ``--small`` flag is accepted for cross-adapter symmetry but Cora
is already only ~170 KB so it does nothing.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import tarfile

import requests


HERE = pathlib.Path(__file__).resolve().parent
RAW_DIR = HERE / "raw"
ARCHIVE = RAW_DIR / "cora.tgz"
EXTRACT_DIR = RAW_DIR / "cora"

USER_AGENT = "Mozilla/5.0 (supernova/aagnn cora fetcher)"

MIRRORS: list[str] = [
    "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz",
    "https://github.com/pyg-team/pyg-datasets/raw/master/cora.tgz",
    # Planetoid is a different layout; loader will not consume it directly.
    "https://raw.githubusercontent.com/kimiyoung/planetoid/master/data/ind.cora.x",
]


def _head_ok(url: str) -> bool:
    """Pre-flight HEAD; some servers reject bare HEAD so we treat 405 as OK too."""
    try:
        r = requests.head(
            url,
            headers={"User-Agent": USER_AGENT},
            allow_redirects=True,
            timeout=15,
        )
    except requests.RequestException as exc:
        print(f"  HEAD {url}: {exc}")
        return False
    print(f"  HEAD {url}: {r.status_code}")
    return r.status_code in (200, 405)


def _download(url: str, dest: pathlib.Path) -> bool:
    try:
        with requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            stream=True,
            timeout=60,
            allow_redirects=True,
        ) as r:
            if r.status_code != 200:
                print(f"  GET {url}: {r.status_code}")
                return False
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".part")
            total = 0
            with tmp.open("wb") as fh:
                for chunk in r.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        fh.write(chunk)
                        total += len(chunk)
            tmp.replace(dest)
            print(f"  saved {dest} ({total:,} bytes)")
            return True
    except requests.RequestException as exc:
        print(f"  GET {url}: {exc}")
        return False


def _extracted() -> bool:
    return (EXTRACT_DIR / "cora.content").exists() and (EXTRACT_DIR / "cora.cites").exists()


def _extract(archive: pathlib.Path) -> None:
    EXTRACT_DIR.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        # `filter="data"` is the safe default in Python 3.14+; works in 3.12+ too.
        tar.extractall(RAW_DIR, filter="data")
    if not _extracted():
        raise RuntimeError(
            f"Extraction of {archive} did not produce cora.content + cora.cites under {EXTRACT_DIR}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--small",
        action="store_true",
        help="No-op for Cora (~170 KB). Accepted for cross-adapter parity.",
    )
    args = ap.parse_args()

    if args.small:
        print("Cora is small enough; --small ignored.")

    if _extracted():
        print(f"Cora already extracted at {EXTRACT_DIR} — nothing to do.")
        return 0

    if not ARCHIVE.exists():
        print("Downloading Cora…")
        # Try each mirror in order: HEAD pre-flight skips obviously-dead URLs
        # quickly, but a passing HEAD doesn't guarantee GET, so a GET failure
        # also falls through to the next mirror.
        for url in MIRRORS:
            if not _head_ok(url):
                continue
            if _download(url, ARCHIVE):
                break
        else:
            print("All mirrors failed.", file=sys.stderr)
            return 1
    else:
        print(f"Archive already on disk: {ARCHIVE}")

    print(f"Extracting {ARCHIVE} → {EXTRACT_DIR}…")
    _extract(ARCHIVE)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
