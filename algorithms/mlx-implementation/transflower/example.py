"""End-to-end TransFlower smoke test on a synthetic clustered city.

Run:
    cd algorithms/mlx-implementation/transflower
    python example.py

Smoke-test contract — `example.py` exits 0 only when ALL three hold:

    1.  Training cross-entropy decreases over epochs (last < first).
    2.  Welch's t-test on per-origin predicted probabilities — same-cluster
        destinations receive significantly higher P than different-cluster
        destinations (gap > 0 and p < 0.05).
    3.  Common Part of Commuters (CPC, paper Eq. 4) on the *total* flow set
        clears a loose threshold of 0.30. (The 20 % held-out split drives
        early stopping during training but is too noisy to use as the final
        contract on a synthetic dataset of this size.) The paper reports
        0.6–0.77 on real LODES data; the synthetic test asks only that the
        model recovers more structure than chance.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import mlx.core as mx
from scipy import stats

from data import SyntheticCity, split_flows
from model import TransFlower, TransFlowerConfig


def cluster_probability_t_test(
    P: mx.array,
    region_to_cluster: dict[int, int],
) -> dict[str, float]:
    """Pool predicted P_{i,j} into same-cluster vs. different-cluster groups
    (excluding self-loops) and run Welch's two-sample t-test.
    """
    P_np = np.array(P)
    N = P_np.shape[0]
    same: list[float] = []
    diff: list[float] = []
    for i in range(N):
        ci = region_to_cluster[i]
        for j in range(N):
            if i == j:
                continue
            (same if region_to_cluster[j] == ci else diff).append(float(P_np[i, j]))

    mean_same = float(np.mean(same))
    mean_diff = float(np.mean(diff))
    t_stat, p_value = stats.ttest_ind(same, diff, equal_var=False)
    return {
        "n_same": len(same),
        "n_diff": len(diff),
        "mean_same": mean_same,
        "mean_diff": mean_diff,
        "gap": mean_same - mean_diff,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--variant", choices=["rle", "rle_prime"], default="rle")
    args = ap.parse_args()

    print("=" * 64)
    print("TransFlower — synthetic clustered-city smoke test")
    print("=" * 64)

    print("\n[1/4] Generating synthetic city…")
    city = SyntheticCity(seed=args.seed)
    regions, flows, region_to_cluster, meta = city.generate()
    print(f"  regions:  {len(regions)}")
    print(f"  clusters: {meta['n_clusters']}")
    print(f"  flows:    {len(flows):,}")
    print(f"  λ_min:    {meta['lambda_min_m']:.1f} m")
    print(f"  λ_max:    {meta['lambda_max_m']:.1f} m")

    train_flows, val_flows = split_flows(flows, val_frac=0.2, seed=args.seed)
    print(f"  train:    {len(train_flows):,} flows")
    print(f"  val:      {len(val_flows):,} flows")

    cfg = TransFlowerConfig(
        rle_variant=args.variant,
        lambda_min=meta["lambda_min_m"],
        lambda_max=meta["lambda_max_m"],
        epochs=args.epochs,
        seed=args.seed,
        verbose=True,
    )
    model = TransFlower(cfg)

    print("\n[2/4] Training…")
    model.fit(regions, train_flows, val_flows)

    losses = model.history.train_loss
    cpcs = model.history.val_cpc
    print("\n[3/4] Loss / CPC trajectories:")
    print(f"  loss:    [{', '.join(f'{l:.3f}' for l in losses[:3])} … "
          f"{', '.join(f'{l:.3f}' for l in losses[-3:])}]")
    if cpcs:
        print(f"  val CPC: [{', '.join(f'{c:.3f}' for c in cpcs[:3])} … "
              f"{', '.join(f'{c:.3f}' for c in cpcs[-3:])}]")

    print("\n[4/4] Smoke-test contract checks:")

    P = model.predict_distributions(regions)
    # CPC against the *total* observed flow set: with ~2k flows over 64×64
    # destinations the per-flow multinomial noise dominates a 20 % held-out
    # split (CPC ceiling ~0.5 even for a perfect predictor). Total-flow CPC
    # is the cleaner measure of structural recovery for the smoke test;
    # held-out CPC (above) is what drives early stopping during training.
    test_cpc = model.cpc(regions, flows)
    test = cluster_probability_t_test(P, region_to_cluster)
    loss_decreased = losses[-1] < losses[0]

    print(f"  loss(first → last): {losses[0]:.4f} → {losses[-1]:.4f}  "
          f"({'DOWN' if loss_decreased else 'UP'})")
    print(f"  CPC on total flows: {test_cpc:.4f}")
    print(f"  cluster gap (same − diff): {test['gap']:+.4f}  "
          f"(mean_same={test['mean_same']:.4f}, mean_diff={test['mean_diff']:.4f})")
    print(f"  Welch's t-test:  t={test['t_stat']:+.3f}  p={test['p_value']:.2e}")
    print(f"    n_same={test['n_same']:,}  n_diff={test['n_diff']:,}")

    ok_loss = loss_decreased
    ok_cluster = (test["gap"] > 0) and (test["p_value"] < 0.05)
    ok_cpc = test_cpc > 0.30

    if ok_loss and ok_cluster and ok_cpc:
        print("\n✓ PASS — loss decreased, cluster structure recovered, CPC > 0.30.")
        return 0

    print("\n✗ FAIL — smoke-test contract not met:")
    print(f"    loss decreased         : {ok_loss}")
    print(f"    cluster gap & p < 0.05 : {ok_cluster}")
    print(f"    CPC > 0.30             : {ok_cpc}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
