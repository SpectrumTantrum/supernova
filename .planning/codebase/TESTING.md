# Testing Patterns

**Analysis Date:** 2026-05-10

## Test Framework

There is **no traditional test framework** in this repo.

- No `tests/` directory anywhere.
- No `pytest`, `unittest`, `nose`, `hypothesis`, `tox`, or `pytest-cov`
  dependency in any `requirements.txt`.
- No `pytest.ini`, `pyproject.toml`, `setup.cfg`, `tox.ini`, or
  `conftest.py` anywhere in the tree.
- No coverage tooling (`.coveragerc`, `coverage.xml`, codecov config) and
  no enforced coverage target.
- No CI workflow files (`.github/workflows/`, `.gitlab-ci.yml`,
  `circleci/`).

This is **policy, not oversight.** `AGENTS.md` states explicitly: *"No
`tests/` directory — `example.py` is the smoke test."* The executor MUST
NOT propose adding pytest, fixtures, parametrized tests, mocks, or a CI
runner.

## The Smoke-Test Contract

Every algorithm ships a top-level `example.py` (and an MLX twin under
`algorithms/mlx-implementation/<algo>/example.py`) that:

1. Generates synthetic data offline (via `data.py`'s `Synthetic*`
   generator).
2. Trains the algorithm end-to-end on that data.
3. Computes one or more **measurable properties** of the trained model.
4. Returns `0` if and only if every gating property holds; otherwise
   returns `1` and prints the specific failing condition.

Per `AGENTS.md`: *"`example.py` exits non-zero unless the algorithm
satisfies a measurable property (Welch's t-test on cluster similarity,
classification accuracy threshold, reconstruction loss bound, etc.)."*

### Run Commands

```bash
# Synthetic smoke test (always offline, ~minutes on CPU):
cd algorithms/pytorch-implementation/<name> && python example.py

# Same algorithm under the MLX backend (Apple Silicon):
cd algorithms/mlx-implementation/<name> && python example.py

# Real-data adapter (requires `python fetch.py` once first):
cd algorithms/pytorch-implementation/<name>/datasets/<dataset> && \
    python fetch.py && python run_example.py
```

Exit code is the regression signal:

```bash
python example.py; echo "exit=$?"   # exit=0 means PASS, exit=1 means FAIL
```

There is no shared CI runner, no `make test`, no `pytest --collect`. To
regression-check a change across all algorithms, run each `example.py`
manually and confirm `exit=$? == 0`.

## Smoke-Test Structure

Every `example.py` follows the same `[1/4]` … `[4/4]` four-section
narrative:

```python
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ...
    args = ap.parse_args()

    print("=" * 64)
    print("<Algorithm> — synthetic <data> smoke test")
    print("=" * 64)

    print("\n[1/4] Generating synthetic …")
    # Synthetic*Network / SyntheticCity instantiated with args.seed
    ...

    print("\n[2/4] Training …")
    # cfg = <Algorithm>Config(... seed=args.seed, verbose=True)
    # model = <Algorithm>(cfg).fit(net)
    ...

    print("\n[3/4] Loss curves:")
    # print _short(model.history["train_losses"]) etc.
    ...

    print("\n[4/4] Evaluation:")
    # compute the measurable properties; check each gate
    if all_gates_passed:
        print("\n✓ PASS — <one-line success summary>.")
        return 0
    print(f"\n✗ FAIL — {first_failing_reason}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
```

Anchor implementations:
- `algorithms/pytorch-implementation/aagnn/example.py:41-128`
- `algorithms/pytorch-implementation/mhgl/example.py:48-181`
- `algorithms/pytorch-implementation/acdne/example.py:53-142`
- `algorithms/pytorch-implementation/transflower/example.py:66-144`
- `algorithms/pytorch-implementation/geotile2vec/example.py:66-144`

## Pass-Criteria Taxonomy

The codebase uses five families of measurable properties. The executor
should reach for an existing pattern when adding a new algorithm rather
than inventing a sixth.

### 1. Loss-Decrease Check

The simplest gate — start-loss > end-loss on the recorded history list.

```python
def _loss_decreased(losses: list[float]) -> bool:
    return bool(losses) and losses[-1] < losses[0]
```

Anchor: `algorithms/pytorch-implementation/geotile2vec/example.py:62-63`.

Inline equivalents:
- `algorithms/pytorch-implementation/aagnn/example.py:104-105`
  (`train_loss_down`, `val_loss_down`).
- `algorithms/pytorch-implementation/mhgl/example.py:148-149`
  (`loss_down`).
- `algorithms/pytorch-implementation/transflower/example.py:122`
  (`loss_decreased`).
- `algorithms/pytorch-implementation/acdne/example.py:113-114`
  (`loss_y_down` — only the supervised classification loss; `L_d` is
  adversarial and may oscillate).

### 2. Welch's t-Test on Score Distributions

The dominant statistical gate — used whenever the algorithm produces a
per-node / per-pair score and there is a known group label
(anomaly / normal, same-cluster / different-cluster). Two pass conditions
are checked together: `p < 0.05` AND positive mean-gap.

```python
from scipy import stats

t_stat, p_value = stats.ttest_ind(
    scores[anom_mask], scores[norm_mask], equal_var=False,
)
gap_ok = mean_anom > mean_norm
p_ok = float(p_value) < 0.05
```

Anchors:
- `algorithms/pytorch-implementation/aagnn/example.py:91-93, 107-108`
  — anomaly vs normal scores.
- `algorithms/pytorch-implementation/mhgl/example.py:119-123, 152-157`
  — both `(anomaly vs normal)` AND `(unseen-only vs normal)`. The
  unseen-only gate is what distinguishes MHGL from AAGNN; if you add a
  new unseen-anomaly algorithm, add this gate too.
- `algorithms/pytorch-implementation/geotile2vec/example.py:28-59`
  (`cosine_sim_gap_test`) — same-cluster vs different-cluster cosine
  similarities of tile pairs.
- `algorithms/pytorch-implementation/transflower/example.py:34-63`
  (`cluster_probability_t_test`) — same-cluster vs different-cluster
  predicted destination probabilities.

`equal_var=False` (Welch, not Student) is universal — synthetic-cluster
score distributions have unequal variances by construction.

### 3. Classification-Floor Threshold

Used when the algorithm's score has a standard supervised metric (ROC-AUC
for anomaly detection, Micro-F1 for cross-network classification). The
floor is loose for the synthetic SBM; the paper's reported numbers are
reproduced as comments alongside.

```python
AUC_FLOOR = 0.75       # smoke-test pass threshold (paper reports 0.82-0.85 on real data)
...
auc_T = float(roc_auc_score(net.labels[T], scores[T]))
auc_ok = auc_T >= args.auc_floor
```

Anchors:
- `algorithms/pytorch-implementation/aagnn/example.py:29, 84-85, 106`
  (`AUC_FLOOR=0.75`).
- `algorithms/pytorch-implementation/mhgl/example.py:32, 110, 150`
  (`AUC_FLOOR=0.75`).
- `algorithms/pytorch-implementation/acdne/example.py:34, 103, 117`
  (`MICRO_F1_FLOOR=0.65`).

The floor is overridable from the CLI (`--auc-floor`,
`--micro-f1-floor`) so the same `example.py` can be relaxed for noisier
runs without editing source.

### 4. "Beats Majority Baseline" Gate

A second classification gate that ensures the model is doing more than
predicting the most-common class. Live in
`algorithms/pytorch-implementation/acdne/example.py:46-50, 105, 118`:

```python
def _majority_baseline_f1(y_true: np.ndarray, y_s: np.ndarray) -> float:
    """Predict the source's most-frequent label for every target node."""
    majority = int(np.bincount(y_s).argmax())
    y_pred = np.full_like(y_true, fill_value=majority)
    return float(f1_score(y_true, y_pred, average="micro"))
...
beats_baseline = micro > baseline
```

Use this for any classification algorithm with class imbalance.

### 5. Adversarial-Game-Converged Gate (Domain-Discriminator Balance)

ACDNE-specific: the smoke test enforces that the GRL adversarial game
has reached its Nash equilibrium by checking that the final
domain-discriminator accuracy is symmetric around chance.

```python
final_d_acc = float(np.mean(h["domain_acc"][-max(20, len(h["domain_acc"]) // 20):]))
d_acc_tol = 0.15
d_acc_balanced = abs(final_d_acc - 0.5) <= d_acc_tol
```

Anchor: `algorithms/pytorch-implementation/acdne/example.py:13-15, 106-116`.
The `|d_acc - 0.5|` symmetry matters because a discriminator stuck at
0.32 is just as "domain-fooled" as one stuck at 0.68 (flipping the
label convention recovers the same gap).

### 6. CPC (Common Part of Commuters)

TransFlower-specific: a flow-prediction-quality metric (paper Eq. 4)
gated as `> 0.30` on the total observed flow set:

```python
test_cpc = model.cpc(regions, flows)
ok_cpc = test_cpc > 0.30
```

Anchor: `algorithms/pytorch-implementation/transflower/example.py:120, 134`.
Note the deliberate choice of total flows over held-out — explained in
the docstring at lines 14-19 (the held-out fold drives early-stopping
during training but is too noisy as a final gate on a synthetic dataset
of this size).

## Combining Gates

Every smoke test combines its gates with `and`, then prints the
**first** failing condition (`if … elif … elif …` cascade). This keeps
failure messages diagnostic and short.

Example — `algorithms/pytorch-implementation/aagnn/example.py:110-127`:

```python
if auc_ok and gap_ok and p_ok and train_loss_down and val_loss_down:
    print("\n✓ PASS — AAGNN distinguishes injected anomalies on the synthetic SBM.")
    return 0

if not train_loss_down:
    reason = (...)
elif not val_loss_down:
    reason = (...)
elif not auc_ok:
    reason = f"ROC-AUC {auc_T:.4f} < floor {args.auc_floor:.4f}"
elif not gap_ok:
    reason = (...)
else:
    reason = f"Welch's p-value {p_value:.2e} >= 0.05"
print(f"\n✗ FAIL — {reason}")
return 1
```

Order the checks from "most likely to fail when training is broken"
(loss didn't decrease) to "most likely to fail when training is
near-correct" (statistical gate just missed).

## Synthetic-Data Generators

Every algorithm ships at least one synthetic generator inside `data.py`:

| Algorithm    | Generator class                                     | Anchor                                                                 |
|--------------|-----------------------------------------------------|------------------------------------------------------------------------|
| Geo-Tile2Vec | `SyntheticCity`                                     | `algorithms/pytorch-implementation/geotile2vec/data.py:226-361`        |
| AAGNN        | `SyntheticAttributedNetwork`                        | `algorithms/pytorch-implementation/aagnn/data.py` (SBM + clique / swap) |
| MHGL         | `SyntheticAttributedNetwork` (seen + unseen anomalies) | `algorithms/pytorch-implementation/mhgl/data.py`                       |
| ACDNE        | `SyntheticCrossNetwork`                             | `algorithms/pytorch-implementation/acdne/data.py`                      |
| TransFlower  | `SyntheticCity` (gravity + anisotropy)              | `algorithms/pytorch-implementation/transflower/data.py:78-111`         |

Generator interface:

```python
@dataclass
class Synthetic<Thing>:
    seed: int = 0
    # other knobs with sensible defaults

    def generate(self) -> tuple[<records>, ...]:
        ...
```

The generator is **always seeded** so `example.py --seed N` produces
identical datasets across runs and across the PyTorch / MLX
implementations. The generated dataset includes whatever ground-truth
labels are needed by the smoke test (cluster IDs, anomaly masks,
labelled-vs-unlabelled split masks).

The MLX twin under `algorithms/mlx-implementation/<algo>/data.py`
mirrors the PyTorch generator function-for-function, so the same
`example.py` script structure produces equivalent loss curves up to
backend numerical differences.

## Real-Data Adapters

Each algorithm whose paper used a public dataset ships a
**self-contained adapter** under `algorithms/pytorch-implementation/<algo>/datasets/<name>/`:

| Algorithm    | Dataset                                          | Path                                                                                |
|--------------|--------------------------------------------------|-------------------------------------------------------------------------------------|
| AAGNN        | Cora citation network                            | `algorithms/pytorch-implementation/aagnn/datasets/cora/`                            |
| ACDNE        | DBLPv7 → ACMv9 cross-network                     | `algorithms/pytorch-implementation/acdne/datasets/dblp_acm/`                        |
| MHGL         | Amazon Photo                                     | `algorithms/pytorch-implementation/mhgl/datasets/amazon_photo/` (raw `.npz` only)   |
| TransFlower  | LEHD LODES8 Massachusetts                        | `algorithms/pytorch-implementation/transflower/datasets/lodes_ma/`                  |

### Adapter Layout

```
<algo>/datasets/<name>/
├── fetch.py            # Idempotent download from public mirror(s) → raw/
├── loader.py           # Parse raw/ into the algorithm's record type
├── run_example.py      # Smoke test against real data (mirrors parent example.py)
├── requirements.txt    # Adapter-only deps (requests, pandas, scipy.io)
└── raw/                # Downloaded files; gitignored except for tiny static fixtures
```

### Parent-Module Shim

Adapters live two levels deep, but they import the parent algorithm's
modules directly. The sanctioned pattern is a `sys.path.insert` shim
followed by `# noqa: E402`:

```python
import pathlib, sys

# parents[0] = <name>, parents[1] = datasets, parents[2] = <algo>.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from data import AttributedNetwork  # noqa: E402
from model import AAGNN, AAGNNConfig  # noqa: E402
```

Anchors:
- `algorithms/pytorch-implementation/aagnn/datasets/cora/loader.py:23-24`
- `algorithms/pytorch-implementation/aagnn/datasets/cora/run_example.py:28-32`
- `algorithms/pytorch-implementation/transflower/datasets/lodes_ma/loader.py:35-36`
- `algorithms/pytorch-implementation/transflower/datasets/lodes_ma/run_example.py:30-34`
- `algorithms/pytorch-implementation/acdne/datasets/dblp_acm/loader.py:32-33`

### Relaxed Real-Data Thresholds

`run_example.py` smoke tests intentionally relax the synthetic gates
because real data is noisier and higher-dimensional:

- `algorithms/pytorch-implementation/aagnn/datasets/cora/run_example.py:35`:
  `SOFT_AUC_FLOOR = 0.65` (vs synthetic `AUC_FLOOR = 0.75`) — and AUC is
  *informational*, not gating; the only hard gates are
  `Welch's p < 0.05` AND `mean_anom > mean_norm`.
- `algorithms/pytorch-implementation/transflower/datasets/lodes_ma/run_example.py:140-148`:
  CPC target `0.20` (vs synthetic `0.30`); printed as a warning, not an
  exit gate. The hard gates are `loss_decreased AND P_rows_sum_to_1`.

When adding a new real-data adapter, expect to relax (or downgrade to a
warning) the synthetic-tier statistical floors; the README in the
parent algorithm's folder should document the gap between the paper's
reported metric and what the adapter actually delivers.

### Idempotent `fetch.py`

Every `fetch.py` is rerunnable:

- HEAD pre-flight on each candidate URL before any GET
  (`algorithms/pytorch-implementation/aagnn/datasets/cora/fetch.py:53-66`,
   `algorithms/pytorch-implementation/transflower/datasets/lodes_ma/fetch.py:63-72`).
- Already-downloaded files are skipped silently
  (`algorithms/pytorch-implementation/aagnn/datasets/cora/fetch.py:124-126`).
- Multiple mirrors tried in order; first 200 wins
  (`algorithms/pytorch-implementation/aagnn/datasets/cora/fetch.py:45-50, 133-140`).
- A `--small` flag is accepted across adapters (no-op on tiny datasets,
  skips the heavy file on LODES) so the fast smoke path is uniform.

## Fast-Mode Toggles

Smoke tests accept a fast-mode flag for CI / dev iteration:

- `--smoke` — short epoch budget, smaller subset of the dataset:
  `algorithms/pytorch-implementation/aagnn/datasets/cora/run_example.py:51-52`
  (20 epochs vs 200);
  `algorithms/pytorch-implementation/transflower/datasets/lodes_ma/run_example.py:62-63, 82-86`
  (2 epochs, ≤50 regions).
- `--no-sv` — Geo-Tile2Vec only, skip Stage 2 (street-view) entirely so
  no Places365 download is attempted:
  `algorithms/pytorch-implementation/geotile2vec/example.py:68-69`.
- `--epochs N`, `--seed N`, `--hidden-dim N`, `--n-iters N` — every
  smoke test exposes the major hyperparameters as CLI knobs.

The `--seed` flag is the standard reproducibility lever; passing
multiple seeds (`for s in 0 1 7 42; do python example.py --seed $s; done`)
is the manual stability-check protocol.

## Inference Sanity Checks

Some smoke tests include shape / normalisation assertions in addition
to the statistical gates:

```python
P = model.predict_distributions(regions)
assert P.shape == (N, N), f"expected ({N},{N}), got {tuple(P.shape)}"
row_sums = P.sum(dim=-1)
rows_normalised = bool(torch.allclose(
    row_sums, torch.ones_like(row_sums), atol=1e-4
))
```

Anchor: `algorithms/pytorch-implementation/transflower/datasets/lodes_ma/run_example.py:119-125`.
Use these whenever the model output has a contractual shape or a
softmax/probability normalisation that should be verifiable by
construction.

## Mocking & Fixtures

- **No mocking framework used.** Synthetic generators replace the need
  for mocks; algorithms always train on a real (synthetic) dataset
  end-to-end.
- **No fixture files** committed to git (the `raw/` directories under
  `datasets/<name>/` are gitignored apart from very small static
  fixtures that ship with the adapter).
- **Offline fallback for pretrained weights:** Geo-Tile2Vec's Stage 2
  catches `torch.hub.load_state_dict_from_url` failures and re-raises
  with the cache path so the user can side-load weights manually
  (`algorithms/pytorch-implementation/geotile2vec/stage2_streetview.py:80-94`).
  `example.py` then catches that `RuntimeError` and falls back to
  Stage 1 only (`geotile2vec/example.py:97-110`). This is the canonical
  "external dependency missing" pattern.

## Coverage

- **Not measured.** No coverage targets, no enforced threshold, no
  coverage report committed.
- The implicit coverage is whatever paths `example.py` exercises:
  config defaults → `fit()` → every output method (`score`, `predict`,
  `embeddings`, `history`) → save/load round-trip is NOT exercised by
  default (a manual `model.save("x.pt"); <Algo>.load("x.pt")` round-trip
  is added if a phase modifies the I/O code).

## Common Loss-Curve Formatter

A near-identical `_short(losses)` helper appears in every numeric
`example.py` for compact loss-curve display:

```python
def _short(losses: list[float]) -> str:
    """Format losses as ``[first3, ..., last3]`` for compact display."""
    if len(losses) <= 6:
        return "[" + ", ".join(f"{l:.4f}" for l in losses) + "]"
    head = ", ".join(f"{l:.4f}" for l in losses[:3])
    tail = ", ".join(f"{l:.4f}" for l in losses[-3:])
    return f"[{head}, ..., {tail}]"
```

Anchors:
- `algorithms/pytorch-implementation/aagnn/example.py:32-38`
- `algorithms/pytorch-implementation/acdne/example.py:37-43`
- `algorithms/pytorch-implementation/mhgl/example.py:35-41`

This duplication is **deliberate** under the `AGENTS.md` "no speculative
abstractions" / "no shared utility modules across algorithms" policy. Do
not extract this into a shared helper. If you add a sixth algorithm with
a similar logging need, copy the function in.

## Regression Workflow

To verify a change has not broken any algorithm:

```bash
for algo in geotile2vec aagnn acdne mhgl transflower; do
    echo "=== pytorch / $algo ==="
    (cd algorithms/pytorch-implementation/$algo && python example.py) \
        || echo "FAILED: $algo"
done

# Same loop under algorithms/mlx-implementation/ on Apple Silicon.
```

Each `example.py` is self-contained (offline, deterministic given
`--seed`), so `exit=0` for all five is a sufficient regression signal
for code-only changes that don't touch external mirrors. Real-data
adapters require an extra `fetch.py` step and so are not part of the
default regression loop — run them only when their own `loader.py` /
`fetch.py` / `run_example.py` change.

---

*Testing analysis: 2026-05-10*
