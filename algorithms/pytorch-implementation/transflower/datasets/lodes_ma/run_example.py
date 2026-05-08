"""End-to-end TransFlower smoke test on Suffolk County, MA (LODES8).

Mirrors algorithms/pytorch-implementation/transflower/example.py — but uses
real LODES OD + WAC data instead of the synthetic clustered city.

Smoke-test contract:
    1.  Training cross-entropy decreases over epochs (last < first).
    2.  predict_distributions(regions) returns (N, N) where each row sums
        to 1 (softmax property).
    3.  Soft assertion: CPC ≥ 0.20 on a 20% held-out flow split. Real
        commute data is noisier than synthetic; the paper reports 0.6–0.77
        on multi-year averaged LODES, but a single-year Suffolk subset with
        a small model trained for 20 epochs lands meaningfully lower.

Run:
    cd algorithms/pytorch-implementation/transflower/datasets/lodes_ma
    python fetch.py
    python run_example.py
    python run_example.py --smoke   # 2 epochs, ≤50 regions
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from data import Flow, Region, split_flows  # noqa: E402
from model import TransFlower, TransFlowerConfig  # noqa: E402

from loader import load_flows, load_regions  # noqa: E402

SUFFOLK_PREFIX = "25025"  # state(25) + county(025) — Suffolk County, MA


def _subset_regions_and_flows(
    regions: list[Region], flows: list[Flow], n_max: int,
) -> tuple[list[Region], list[Flow]]:
    """First n_max regions, with region_ids reindexed to 0..n-1, plus flows
    whose endpoints both survive the trim."""
    head = regions[:n_max]
    remap = {r.region_id: i for i, r in enumerate(head)}
    sub_regions = [
        Region(i, r.lat, r.lon, r.place_features, r.population)
        for i, r in enumerate(head)
    ]
    sub_flows = [
        Flow(remap[fl.o_id], remap[fl.d_id], fl.count)
        for fl in flows
        if fl.o_id in remap and fl.d_id in remap
    ]
    return sub_regions, sub_flows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--smoke", action="store_true",
                    help="Fast settings: 2 epochs, ≤50 regions.")
    ap.add_argument("--variant", choices=["rle", "rle_prime"], default="rle")
    args = ap.parse_args()

    print("=" * 64)
    print("TransFlower — LODES8 MA (Suffolk County) smoke test")
    print("=" * 64)

    print("\n[1/4] Loading Suffolk County tracts from LODES …")
    regions, geoid_to_id = load_regions(county_prefix=SUFFOLK_PREFIX)
    flows = load_flows(geoid_to_id)
    print(f"  regions:  {len(regions)}")
    print(f"  flows:    {len(flows):,}")

    if not regions or not flows:
        print("\nNo data. Did you run `python fetch.py` (without --small)?",
              file=sys.stderr)
        return 1

    if args.smoke:
        regions, flows = _subset_regions_and_flows(regions, flows, n_max=50)
        epochs = 2
        print(f"  [smoke] subset to {len(regions)} regions, "
              f"{len(flows):,} flows, epochs=2")
    else:
        epochs = args.epochs

    train_flows, val_flows = split_flows(flows, val_frac=0.2, seed=args.seed)
    print(f"  train:    {len(train_flows):,} flows")
    print(f"  val:      {len(val_flows):,} flows")

    # lambda_max: rough study-area diameter (Suffolk is ~20 km across; pad).
    cfg = TransFlowerConfig(
        rle_variant=args.variant,
        lambda_min=1.0,
        lambda_max=50_000.0,
        epochs=epochs,
        seed=args.seed,
        verbose=True,
    )
    model = TransFlower(cfg)

    print("\n[2/4] Training …")
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
    N = len(regions)
    assert P.shape == (N, N), f"expected ({N},{N}), got {tuple(P.shape)}"
    row_sums = P.sum(dim=-1)
    rows_normalised = bool(torch.allclose(
        row_sums, torch.ones_like(row_sums), atol=1e-4
    ))

    val_cpc = model.cpc(regions, val_flows) if val_flows else float("nan")
    test_cpc = model.cpc(regions, flows)
    loss_decreased = losses[-1] < losses[0]

    print(f"  loss(first → last): {losses[0]:.4f} → {losses[-1]:.4f}  "
          f"({'DOWN' if loss_decreased else 'UP'})")
    print(f"  P shape           : {tuple(P.shape)}  (rows sum to 1: {rows_normalised})")
    print(f"  CPC on val flows  : {val_cpc:.4f}")
    print(f"  CPC on total flows: {test_cpc:.4f}")

    ok_loss = loss_decreased
    ok_shape = rows_normalised
    # Soft target on real LODES (paper hits 0.6+ on multi-year averages;
    # single-year Suffolk subset with a small model lands lower).
    cpc_target = 0.20

    if ok_loss and ok_shape:
        print("\n✓ PASS — loss decreased, P is row-normalised softmax.")
        if not args.smoke and val_cpc < cpc_target:
            print(f"  (warn) val CPC {val_cpc:.3f} < {cpc_target}; "
                  f"longer training likely helps.")
        return 0

    print("\n✗ FAIL — smoke-test contract not met:")
    print(f"    loss decreased         : {ok_loss}")
    print(f"    P rows sum to 1        : {ok_shape}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
