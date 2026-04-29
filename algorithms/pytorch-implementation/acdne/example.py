"""End-to-end ACDNE demo on a synthetic cross-network with planted domain shift.

Run:
    cd algorithms/pytorch-implementation/acdne
    python example.py
    python example.py --seed 7
    python example.py --seed 42 --n-iters 1500

Smoke-test contract (exits non-zero unless ALL hold):
    1. Source classification loss L_y trends down (start > end).
    2. Final domain-discriminator accuracy is close to chance —
       |d_acc - 0.5| <= 0.20, i.e. d_acc in [0.30, 0.70]. Symmetric
       around 0.5 because a discriminator stuck at, say, 0.32 is just
       as "domain-fooled" as one stuck at 0.68 — flip its label
       convention and you recover the same |d_acc - 0.5|.
    3. Micro-F1 on the unlabelled target network is ≥ --micro-f1-floor
       (default 0.65 — the paper hits 0.66–0.83 on real cross-network
       transfers).
    4. Micro-F1 strictly beats the majority-class baseline.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from sklearn.metrics import f1_score

from data import SyntheticCrossNetwork
from model import ACDNE, ACDNEConfig


MICRO_F1_FLOOR = 0.65


def _short(losses: list[float]) -> str:
    """Format losses as ``[first3, ..., last3]`` for compact display."""
    if len(losses) <= 6:
        return "[" + ", ".join(f"{l:.4f}" for l in losses) + "]"
    head = ", ".join(f"{l:.4f}" for l in losses[:3])
    tail = ", ".join(f"{l:.4f}" for l in losses[-3:])
    return f"[{head}, ..., {tail}]"


def _majority_baseline_f1(y_true: np.ndarray, y_s: np.ndarray) -> float:
    """Predict the source's most-frequent label for every target node."""
    majority = int(np.bincount(y_s).argmax())
    y_pred = np.full_like(y_true, fill_value=majority)
    return float(f1_score(y_true, y_pred, average="micro"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-iters", type=int, default=1000)
    ap.add_argument("--embed-dim", type=int, default=128,
                    help="Concat-layer output dim d (paper default 128).")
    ap.add_argument("--embed-hidden-dim", type=int, default=256,
                    help="FE hidden dim f(1). Paper uses 512; 256 keeps the "
                         "smoke test fast on CPU and SBM doesn't need 512.")
    ap.add_argument("--micro-f1-floor", type=float, default=MICRO_F1_FLOOR)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    print("=" * 72)
    print("ACDNE — synthetic cross-network smoke test")
    print("=" * 72)

    print("\n[1/4] Generating synthetic cross-network…")
    gen = SyntheticCrossNetwork(seed=args.seed)
    net = gen.generate()
    print(f"  Source: n_s={net.n_s}, edges={len(net.edges_s)}")
    print(f"  Target: n_t={net.n_t}, edges={len(net.edges_t)}")
    print(f"  feat_dim={net.feat_dim}  n_classes={net.n_classes}")
    print(f"  domain shift strength={gen.domain_shift_strength}")

    print("\n[2/4] Training ACDNE…")
    cfg = ACDNEConfig(
        embed_hidden_dim=args.embed_hidden_dim,
        fe_out_dim=args.embed_dim,
        embed_dim=args.embed_dim,
        disc_hidden_dim=args.embed_dim,
        n_iters=args.n_iters,
        seed=args.seed,
        device=args.device,
        verbose=True,
    )
    model = ACDNE(cfg).fit(net)

    h = model.history
    print("\n[3/4] Loss curves:")
    print(f"  L_y     : {_short(h['loss_y'])}")
    print(f"  L_p     : {_short(h['loss_p'])}")
    print(f"  L_d     : {_short(h['loss_d'])}")
    print(f"  d_acc   : {_short(h['domain_acc'])}")
    print(f"  λ ramp  : {h['grl_lambda'][0]:.3f} → {h['grl_lambda'][-1]:.3f}")
    print(f"  lr decay: {h['lr'][0]:.4f} → {h['lr'][-1]:.4f}")

    print("\n[4/4] Evaluation on target network:")
    assert net.y_t is not None
    y_pred = model.predict()
    micro = float(f1_score(net.y_t, y_pred, average="micro"))
    macro = float(f1_score(net.y_t, y_pred, average="macro"))
    baseline = _majority_baseline_f1(net.y_t, net.y_s)
    final_d_acc = float(np.mean(h["domain_acc"][-max(20, len(h["domain_acc"]) // 20):]))

    print(f"  Micro-F1:           {micro:.4f}")
    print(f"  Macro-F1:           {macro:.4f}")
    print(f"  Majority baseline:  {baseline:.4f}")
    print(f"  Final d_acc (avg):  {final_d_acc:.4f}")

    loss_y = h["loss_y"]
    loss_y_down = loss_y[-1] < loss_y[0]
    d_acc_balanced = 0.40 <= final_d_acc <= 0.65
    micro_ok = micro >= args.micro_f1_floor
    beats_baseline = micro > baseline

    if loss_y_down and d_acc_balanced and micro_ok and beats_baseline:
        print("\n✓ PASS — ACDNE transfers source-network labels to the target network.")
        return 0

    if not loss_y_down:
        reason = (
            f"L_y did not decrease (start={loss_y[0]:.4f}, end={loss_y[-1]:.4f})"
        )
    elif not d_acc_balanced:
        reason = (
            f"final domain-discriminator accuracy {final_d_acc:.4f} "
            f"outside [0.40, 0.65] — adversarial game did not converge"
        )
    elif not micro_ok:
        reason = (
            f"Micro-F1 {micro:.4f} < floor {args.micro_f1_floor:.4f}"
        )
    else:
        reason = (
            f"Micro-F1 {micro:.4f} did not beat majority baseline {baseline:.4f}"
        )
    print(f"\n✗ FAIL — {reason}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
