"""End-to-end Geo-Tile2Vec demo on real Citi Bike + Manhattan OSM data.

Run:
    python fetch.py --small      # one-time: pull data into raw/
    python run_example.py        # full e2e (slow on a laptop)
    python run_example.py --smoke  # tiny subsample, fast verification

Smoke-test contract:
    1. embeddings() returns shape (n_tiles, 300).
    2. tile-embedding variance is non-zero (data actually trained).
    3. (Best-effort) Welch's t-test on cosine similarity between same-category
       and different-category tile pairs has positive gap. Real-world bike-
       share data is noisier than the synthetic city, so this is reported as
       a diagnostic but is NOT required for passing.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
import torch
from scipy import stats

# Same import shim as loader.py — geotile2vec/ has no __init__.py.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from data import latlon_to_tile  # noqa: E402
from model import GeoTile2Vec, GeoTile2VecConfig  # noqa: E402

from loader import load_all  # noqa: E402


def _tile_to_dominant_category(pois, tile_level: int) -> dict:
    """Map each tile to the most common POI category falling inside it."""
    bag: dict = defaultdict(Counter)
    for p in pois:
        tid = latlon_to_tile(p.lat, p.lon, tile_level)
        bag[tid][p.category] += 1
    return {tid: c.most_common(1)[0][0] for tid, c in bag.items()}


def cosine_sim_gap_test(V: torch.Tensor, tile_order, tile_to_label) -> dict:
    """Mirror of example.py's smoke-test, but using POI-category as cluster."""
    V_np = V.detach().cpu().numpy()
    norms = np.linalg.norm(V_np, axis=1, keepdims=True) + 1e-12
    V_unit = V_np / norms

    same: list[float] = []
    diff: list[float] = []
    for i, j in combinations(range(len(tile_order)), 2):
        c_i = tile_to_label.get(tile_order[i])
        c_j = tile_to_label.get(tile_order[j])
        if c_i is None or c_j is None:
            continue
        sim = float(V_unit[i] @ V_unit[j])
        (same if c_i == c_j else diff).append(sim)

    if not same or not diff:
        return {"n_same_pairs": len(same), "n_diff_pairs": len(diff)}
    t_stat, p_value = stats.ttest_ind(same, diff, equal_var=False)
    return {
        "n_same_pairs": len(same),
        "n_diff_pairs": len(diff),
        "mean_same": float(np.mean(same)),
        "mean_diff": float(np.mean(diff)),
        "gap": float(np.mean(same) - np.mean(diff)),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="Heavy subsample (~5k trips, full POI set, 3 epochs) for fast verification.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("=" * 64)
    print("Geo-Tile2Vec — Citi Bike + OSM real-data smoke test")
    print("=" * 64)

    if args.smoke:
        # 200 random POIs across all of Manhattan leaves big snap gaps; keep
        # the full POI set (~17k) so most trip endpoints land near something.
        max_trips, max_pois, epochs, steps = 5_000, None, 3, 50
    else:
        max_trips, max_pois, epochs, steps = 50_000, None, 5, 200

    print(f"\n[1/4] Loading real data (max_trips={max_trips}, max_pois={max_pois}) …")
    data = load_all(max_trajectories=max_trips, max_pois=max_pois)
    print(f"  POIs:         {len(data.pois):,}")
    print(f"  Trajectories: {len(data.trajectories):,}")
    print(f"  bbox (S,W,N,E): {data.bbox}")

    if not data.pois or not data.trajectories:
        print("ERROR: empty POI or trajectory list — re-run fetch.py.", file=sys.stderr)
        return 2

    cfg = GeoTile2VecConfig(seed=args.seed, verbose=True)
    cfg.skipgram_epochs = epochs
    cfg.triplet1_epochs = epochs
    cfg.triplet1_steps = steps
    # Citi Bike stations sit at curbside, but OSM POIs are often mid-block.
    # 80 m gives Stage 1 something to snap to without bleeding across blocks.
    cfg.poi_snap_meters = 80.0

    model = GeoTile2Vec(cfg)
    print("\n[2/4] Training Stage 1 (no street-view) …")
    model.fit(data.pois, data.trajectories, None)

    V, tile_order = model.embeddings()
    print(f"\n[3/4] Embeddings: shape={tuple(V.shape)}, dtype={V.dtype}")
    assert V.shape[1] == cfg.d_event == 300, f"expected 300-d embeddings, got {V.shape}"
    assert V.shape[0] == len(tile_order) > 0, "no tiles produced"
    var = float(V.detach().cpu().numpy().var())
    print(f"  embedding variance: {var:.6f}")
    assert var > 1e-8, "embeddings are degenerate (all-zero / constant)"

    print("\n[4/4] Diagnostic Welch's t-test (POI-category as cluster proxy):")
    tile_to_cat = _tile_to_dominant_category(data.pois, cfg.tile_level)
    res = cosine_sim_gap_test(V, tile_order, tile_to_cat)
    if "mean_same" not in res:
        print(f"  n_same={res['n_same_pairs']}, n_diff={res['n_diff_pairs']} — not enough labels.")
    else:
        print(f"  same-cat pairs:  n={res['n_same_pairs']:,}, mean cos = {res['mean_same']:+.4f}")
        print(f"  diff-cat pairs:  n={res['n_diff_pairs']:,}, mean cos = {res['mean_diff']:+.4f}")
        print(f"  gap            : {res['gap']:+.4f}")
        print(f"  t = {res['t_stat']:+.3f}   p = {res['p_value']:.2e}")
        if res['gap'] > 0:
            print("  -> same-category tiles closer than different-category (expected).")
        else:
            print("  -> gap non-positive; real data is noisy, this is informational only.")

    print("\nPASS — embeddings have correct shape and non-zero variance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
