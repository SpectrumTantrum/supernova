"""End-to-end Geo-Tile2Vec demo on a synthetic clustered city.

Run:
    cd algorithms/geotile2vec
    python example.py            # tries Stage 2; falls back if offline
    python example.py --no-sv    # skip Stage 2 entirely

Smoke-test contract:
    1. All loss curves trend down.
    2. Welch's t-test on cosine similarities between (a) same-cluster tile pairs
       and (b) different-cluster tile pairs returns p < 0.05 with positive gap.
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations

import numpy as np
import mlx.core as mx
from scipy import stats

from data import SyntheticCity
from model import GeoTile2Vec, GeoTile2VecConfig


def cosine_sim_gap_test(V: mx.array, tile_order, tile_to_cluster):
    """Compute mean cosine similarity for same-cluster vs different-cluster
    tile pairs and run a Welch's t-test on the two distributions.
    """
    V_np = np.asarray(V)
    norms = np.linalg.norm(V_np, axis=1, keepdims=True) + 1e-12
    V_unit = V_np / norms

    same: list[float] = []
    diff: list[float] = []
    for i, j in combinations(range(len(tile_order)), 2):
        sim = float(V_unit[i] @ V_unit[j])
        c_i = tile_to_cluster.get(tile_order[i])
        c_j = tile_to_cluster.get(tile_order[j])
        if c_i is None or c_j is None:
            continue
        (same if c_i == c_j else diff).append(sim)

    if not same or not diff:
        raise RuntimeError("Could not collect both same- and different-cluster pairs.")

    mean_same, mean_diff = float(np.mean(same)), float(np.mean(diff))
    t_stat, p_value = stats.ttest_ind(same, diff, equal_var=False)
    return {
        "n_same_pairs": len(same),
        "n_diff_pairs": len(diff),
        "mean_same": mean_same,
        "mean_diff": mean_diff,
        "gap": mean_same - mean_diff,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
    }


def _loss_decreased(losses: list[float]) -> bool:
    return bool(losses) and losses[-1] < losses[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-sv", action="store_true",
                    help="Skip Stage 2 (street-view) entirely.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("=" * 64)
    print("Geo-Tile2Vec - synthetic city smoke test")
    print("=" * 64)

    print("\n[1/4] Generating synthetic clustered city…")
    city = SyntheticCity(seed=args.seed)
    pois, trajectories, shots, tile_to_cluster = city.generate()
    print(f"  POIs:         {len(pois):,}")
    print(f"  Trajectories: {len(trajectories):,}")
    print(f"  Shots:        {len(shots):,}")
    print(f"  Tiles:        {len(tile_to_cluster):,}")

    cfg = GeoTile2VecConfig(seed=args.seed, verbose=True)
    # Smaller training to keep the smoke test fast.
    cfg.skipgram_epochs = 3
    cfg.triplet1_epochs = 3
    cfg.triplet1_steps = 100
    cfg.triplet2_epochs = 3
    cfg.triplet2_steps = 100

    model = GeoTile2Vec(cfg)

    use_shots = None if args.no_sv else shots
    if use_shots is not None:
        # Try Stage 2; if real Places365 extraction is unavailable, fall back to Stage 1 only.
        try:
            print("\n[2/4] Training Stage 1 + Stage 2…")
            model.fit(pois, trajectories, use_shots)
        except RuntimeError as e:
            if "Places365" in str(e):
                print(f"\n  ! Stage 2 unavailable ({e})\n  ! Falling back to Stage 1 only.\n")
                model = GeoTile2Vec(cfg)
                model.fit(pois, trajectories, None)
            else:
                raise
    else:
        print("\n[2/4] Training Stage 1 only (--no-sv)…")
        model.fit(pois, trajectories, None)

    V, tile_order = model.embeddings()

    print("\n[3/4] Loss curves:")
    print(f"  Skip-Gram   : {[f'{l:.3f}' for l in model.history.skipgram_losses]}")
    print(f"  Triplet S1  : {[f'{l:.3f}' for l in model.history.triplet1_losses]}")
    if model.history.triplet2_losses:
        print(f"  Triplet S2  : {[f'{l:.3f}' for l in model.history.triplet2_losses]}")
    ok_skipgram = _loss_decreased(model.history.skipgram_losses)
    ok_triplet1 = _loss_decreased(model.history.triplet1_losses)
    ok_triplet2 = (
        True if not model.history.triplet2_losses
        else _loss_decreased(model.history.triplet2_losses)
    )

    print("\n[4/4] Cluster-similarity sanity check (Welch's t-test):")
    result = cosine_sim_gap_test(V, tile_order, tile_to_cluster)
    print(f"  same-cluster pairs:  n={result['n_same_pairs']:,}, mean cos = {result['mean_same']:+.4f}")
    print(f"  diff-cluster pairs:  n={result['n_diff_pairs']:,}, mean cos = {result['mean_diff']:+.4f}")
    print(f"  gap (same - diff)  : {result['gap']:+.4f}")
    print(f"  t = {result['t_stat']:+.3f}   p = {result['p_value']:.2e}")

    ok_gap = result["gap"] > 0
    ok_p = result["p_value"] < 0.05
    if ok_gap and ok_p and ok_skipgram and ok_triplet1 and ok_triplet2:
        print("\nPASS - same-cluster tiles are significantly closer than different-cluster tiles.")
        return 0
    print("\nFAIL - smoke-test contract not met:")
    print(f"    Skip-Gram loss decreased : {ok_skipgram}")
    print(f"    Stage-1 triplet decreased: {ok_triplet1}")
    print(f"    Stage-2 triplet decreased: {ok_triplet2}")
    print(f"    cluster gap positive     : {ok_gap}")
    print(f"    Welch p < 0.05           : {ok_p}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
