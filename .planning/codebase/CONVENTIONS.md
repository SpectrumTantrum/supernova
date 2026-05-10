# Coding Conventions

**Analysis Date:** 2026-05-10

These conventions are derived from the explicit policy in `AGENTS.md` (the
`CLAUDE.md` / `GEMINI.md` symlinks point at the same file) and from the
uniform patterns observed across all five PyTorch and MLX algorithm
implementations under `algorithms/pytorch-implementation/` and
`algorithms/mlx-implementation/`.

## Naming Patterns

**Files:**
- snake_case throughout. No CamelCase or kebab-case files anywhere.
- Pipeline-stage modules use the literal `<algo>_<stage>.py` pattern
  mandated by `AGENTS.md`. Examples:
  - `algorithms/pytorch-implementation/geotile2vec/stage1_mobility.py`
  - `algorithms/pytorch-implementation/geotile2vec/stage2_streetview.py`
  - `algorithms/pytorch-implementation/transflower/flow_predictor.py`
  - `algorithms/pytorch-implementation/transflower/geo_encoder.py`
- Single-responsibility modules at the algorithm root: `data.py`, `model.py`,
  `train.py`, `layer.py` / `layers.py`, `gcn.py`, `pde.py`, `example.py`.
- Algorithm folders are deliberately flat (no `src/` or deep package
  hierarchies). Each holds 5–8 source files.

**Functions:**
- snake_case for module-level functions: `build_mobility_events`,
  `compute_pseudo_labels`, `train_aagnn`, `anomaly_scores`,
  `build_normalized_adj`, `ppmi_matrix`, `latlon_to_tile`,
  `haversine_meters`.
- Leading underscore for private helpers — never exported, often called only
  from one place: `_forward_no_grad` (`algorithms/pytorch-implementation/aagnn/train.py`),
  `_short` (in every `example.py`), `_semi_hard_triplet_loss`
  (`algorithms/pytorch-implementation/geotile2vec/stage1_mobility.py`),
  `_load_state` (`algorithms/pytorch-implementation/geotile2vec/stage2_streetview.py`),
  `_nearest_poi_within` (`algorithms/pytorch-implementation/geotile2vec/data.py`).

**Variables:**
- snake_case for locals and instance attributes (`tile_to_cluster`,
  `train_losses`, `n_iters`, `pseudo_label_pct`).
- Cached private state on `self` uses leading underscore:
  `self._encoder`, `self._A_hat`, `self._centres`, `self._scores`,
  `self._cached_e_t` (see `algorithms/pytorch-implementation/mhgl/model.py`,
  `algorithms/pytorch-implementation/acdne/model.py`).
- Math-paper symbols are kept short to mirror the source: `X`, `H`, `c`,
  `R`, `D`, `T`, `K`, `N`, `f`, `n`, `d`. Comments anchor them to paper
  equations.

**Types / Classes:**
- PascalCase for both `nn.Module` subclasses and `@dataclass` records:
  - `nn.Module`: `AbnormalityAwareLayer`, `GCNEncoder`, `EmbeddingModule`,
    `NodeClassifier`, `DomainDiscriminator`, `GeoSpatialEncoder`,
    `FlowPredictor`, `MobilityEventModel`, `Places365PretrainedResNet18`,
    `GradientReversal`, `FeatureExtractor`.
  - Dataclasses: `AAGNNConfig`, `MHGLConfig`, `ACDNEConfig`,
    `TransFlowerConfig`, `GeoTile2VecConfig`, `TrainingHistory`,
    `AttributedNetwork`, `CrossNetwork`, `Region`, `Flow`, `POI`,
    `Trajectory`, `MobilityEvent`, `StreetViewShot`, `SyntheticCity`,
    `SyntheticAttributedNetwork`, `SyntheticCrossNetwork`, `Pattern`.

**Constants:**
- UPPER_SNAKE for module-level constants. Always grouped near the top of
  the file under a `# --- Paper §X.Y constants ---` divider:
  - `algorithms/pytorch-implementation/geotile2vec/data.py`:
    `DEFAULT_TILE_LEVEL`, `DEFAULT_POI_SNAP_METERS`,
    `DEFAULT_TIME_THRESHOLD_MIN`, `NUM_POI_CATEGORIES`,
    `NUM_TIME_BUCKETS`, `POI_CATEGORIES`, `EVENT_TYPE_O`, `EVENT_TYPE_D`.
  - `algorithms/pytorch-implementation/transflower/data.py`:
    `NUM_PLACE_CATEGORIES`, `D_FEATURE`.
  - `algorithms/pytorch-implementation/aagnn/example.py`: `AUC_FLOOR`.
  - `algorithms/pytorch-implementation/acdne/example.py`: `MICRO_F1_FLOOR`.
  - `algorithms/pytorch-implementation/geotile2vec/stage2_streetview.py`:
    `PLACES365_RESNET18_URL`.

## Module Docstrings

Every module begins with a triple-quoted docstring that:

1. States what the module does in one sentence.
2. Cites the paper section and/or equation it implements.
3. (For orchestrators / pipelines) lists the responsibilities of each
   sibling module being wired in.

Anchor pattern — `algorithms/pytorch-implementation/aagnn/layer.py` lines 1-12:

```python
"""Abnormality-aware GNN layer (paper §3.1, Eqs. 1-4).

Reference: Zhou et al., "Subtractive Aggregation for Attributed Network
Anomaly Detection", CIKM 2021.

A single graph layer that, for each node i, computes
    z_i  = W x_i                                    (Eq. 1, no bias)
    h_i  = sigma(z_i - Aggregate({z_j : j in N_i^k}))
...
"""
```

Other canonical examples:
- `algorithms/pytorch-implementation/geotile2vec/data.py:1-11` — paper §2.1 / §3.2.1.
- `algorithms/pytorch-implementation/mhgl/train.py:1-30` — paper §3.2 Eqs. 3.5-3.8, Algorithm 2.
- `algorithms/pytorch-implementation/transflower/model.py:1-11` — Reference + typical-usage block.
- `algorithms/pytorch-implementation/acdne/data.py:1-18` — what it defines, paper section, what it omits.

## Type Hints

- Type hints throughout, every public function. `from __future__ import
  annotations` at the top of every module enables PEP 604 unions
  (`X | None`) and forward references without quoting.
- `dataclass` for records; `@dataclass(frozen=True)` for immutable record
  types with `__post_init__` validation:
  - `algorithms/pytorch-implementation/aagnn/data.py:30-69`
    (`AttributedNetwork` validates X dtype, edge canonical form `i < j`,
    duplicate-edge rejection, label shape).
  - `algorithms/pytorch-implementation/acdne/data.py:38-101`
    (`CrossNetwork` validates feat-dim parity, edge canonical form, label
    range).
  - `algorithms/pytorch-implementation/transflower/data.py:38-61` (`Region`,
    `Flow`).
- Plain `@dataclass` (mutable) for hyperparameter bundles and training
  history: `AAGNNConfig`, `MHGLConfig`, `ACDNEConfig`,
  `TransFlowerConfig`, `GeoTile2VecConfig`, `TrainingHistory`.
- `NamedTuple` is used for tiny coordinate triples
  (`TileId(z, x, y)` in `algorithms/pytorch-implementation/geotile2vec/data.py:45-48`).
- Keyword-only arguments via the `*` separator are standard for trainer
  and loss functions:
  - `algorithms/pytorch-implementation/aagnn/train.py:48` (`compute_pseudo_labels(..., *, pseudo_label_pct, ...)`).
  - `algorithms/pytorch-implementation/mhgl/train.py:80, 230` (`compute_high_confidence`, `train_mhgl`).
  - `algorithms/pytorch-implementation/geotile2vec/data.py:147` (`build_mobility_events`).
  - `algorithms/pytorch-implementation/geotile2vec/stage1_mobility.py:121, 202` (`train_skipgram`, `train_triplet_metric`).

## Imports

**Always at the top:**

```python
"""Module docstring …"""

from __future__ import annotations
```

**Order (3 groups, blank line between):**

1. Standard library (`argparse`, `sys`, `pathlib`, `random`, `math`,
   `os`, `dataclasses`, `typing`).
2. Third-party (`numpy`, `scipy`, `torch`, `torch.nn`, `torch.nn.functional`,
   `sklearn`, `mlx.core`, `pandas`, `requests`, `PIL`, `torchvision`).
3. Local sibling modules (`from data import ...`, `from model import ...`).

Canonical example — `algorithms/pytorch-implementation/aagnn/example.py:14-26`:

```python
from __future__ import annotations

import argparse
import sys

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

from data import SyntheticAttributedNetwork
from model import AAGNN, AAGNNConfig
```

**Path aliases:** none. Each algorithm directory is its own importable
folder; siblings are referenced by bare `from data import ...` after the
caller `cd`s into the algorithm directory (or after a `sys.path.insert`
shim in dataset adapters — see below).

## Code Style

**Comments — minimal, only when the WHY is non-obvious:**

`AGENTS.md` explicitly forbids decorative comments. Real-world examples of
load-bearing comments observed in the codebase:

- `algorithms/pytorch-implementation/aagnn/layer.py:93-95` — explains WHY
  `bias=False` is non-negotiable (Deep SVDD trivial-solution collapse).
- `algorithms/pytorch-implementation/aagnn/train.py:148-149` — explains
  WHY weight-decay is added explicitly to the loss instead of via the
  optimiser (PyTorch / MLX numerical equivalence).
- `algorithms/pytorch-implementation/mhgl/train.py:282-284` — explains the
  validation-loss scope ("is the loss surface stable" signal, not a
  generalisation metric).

Any comment that explains WHAT the line does (rather than why) is out of
style and should be removed.

**Section dividers** for long files:

```python
# --- Records ---------------------------------------------------------------------
# --- Geometry --------------------------------------------------------------------
# --- Mobility-event construction (paper §2.1 / §3.2.1) --------------------------
```

Anchor: `algorithms/pytorch-implementation/geotile2vec/data.py:43, 74, 130`.

**Method dividers inside classes** use a thinner rule:

```python
# ------------------------------------------------------------------ Fit
# ------------------------------------------------------------------ Outputs
# ------------------------------------------------------------------ I/O
```

Anchor: `algorithms/pytorch-implementation/aagnn/model.py:73, 125, 172`,
`algorithms/pytorch-implementation/mhgl/model.py:94, 220, 277`.

## Linting / Formatting

- **`ruff`** is the (only) linter — `.ruff_cache/` exists at the repo root,
  contents `0.12.0` and `CACHEDIR.TAG`. No committed `pyproject.toml`,
  `ruff.toml`, or `setup.cfg`; ruff runs against its built-in defaults.
- No `prettier`, `black`, `mypy`, `isort`, or `pre-commit` config files
  exist anywhere in the repo.
- The single sanctioned escape hatch is `# noqa: E402` after a
  `sys.path.insert` shim in dataset adapters — see
  `algorithms/pytorch-implementation/aagnn/datasets/cora/loader.py:23-24`,
  `algorithms/pytorch-implementation/transflower/datasets/lodes_ma/loader.py:35-36`,
  `algorithms/pytorch-implementation/acdne/datasets/dblp_acm/loader.py:32-33`.

## Error Handling

Three idiomatic raise patterns appear consistently. The executor MUST
match these.

**1. `RuntimeError("Call fit() first.")` — uninitialized-state guard.**

Universal across every orchestrator, on every output method:

- `algorithms/pytorch-implementation/geotile2vec/model.py:209, 220`
  (`embeddings()`, `save()`).
- `algorithms/pytorch-implementation/aagnn/model.py:134, 153, 169, 177`
  (`score()`, `predict()`, `split_indices()`, `save()`).
- `algorithms/pytorch-implementation/acdne/model.py:164, 175, 184, 196, 229`
  (every output / inference method).
- `algorithms/pytorch-implementation/mhgl/model.py:225, 246, 264, 273, 282`.
- `algorithms/pytorch-implementation/transflower/model.py:255` (`_require_fitted`).

**2. `ValueError(f"... got {x!r}")` — bad arguments.**

Use the `!r` repr-format so the actual value is unambiguous in the
message. Examples:

- `algorithms/pytorch-implementation/aagnn/layer.py:79-85` —
  `f"aggregator must be one of {sorted(_AGGREGATORS)}, got {aggregator!r}"`.
- `algorithms/pytorch-implementation/aagnn/train.py:124` —
  `f"optimizer must be 'adam' or 'sgd', got {optimizer!r}"`.
- `algorithms/pytorch-implementation/acdne/model.py:171` —
  `f"which must be 'source'|'target'|'both', got {which!r}"`.
- `algorithms/pytorch-implementation/geotile2vec/data.py:193` —
  `f"time_threshold_min must be >= 0, got {time_threshold_min}"`.

Numeric values may use `{x}` (without `!r`) when the context already
makes the type obvious (`got {n}`, `got {pseudo_label_pct}`).

**3. `RuntimeError` with offline-fallback cache-path hint — for downloaded
weights.**

This is the AGENTS.md-mandated pattern for `torch.hub` /
`torch.hub.load_state_dict_from_url` failures. Anchor at
`algorithms/pytorch-implementation/geotile2vec/stage2_streetview.py:80-94`:

```python
try:
    ckpt = torch.hub.load_state_dict_from_url(
        PLACES365_RESNET18_URL, map_location="cpu", weights_only=False,
    )
except Exception as e:
    cache = os.path.join(
        os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints",
        "resnet18_places365.pth.tar",
    )
    raise RuntimeError(
        "Failed to download Places365 ResNet-18 weights from "
        f"{PLACES365_RESNET18_URL}. Pre-download to {cache} and retry. "
        f"Original error: {e}"
    ) from e
```

The message MUST name (a) the URL it tried, (b) the local cache path the
user can side-load to, and (c) `from e` to chain the original exception.

**4. `FileNotFoundError("... Run fetch.py first.")` — for dataset adapters.**

Real-data loaders raise this when `raw/` is missing. Examples:

- `algorithms/pytorch-implementation/aagnn/datasets/cora/loader.py:42-47`:
  `f"Cora raw files not found under {RAW_DIR}. Run \`python fetch.py\` first."`.
- `algorithms/pytorch-implementation/acdne/datasets/dblp_acm/loader.py:50-52`:
  `f"Missing {path}. Run fetch.py first to download the .mat files."`.
- `algorithms/pytorch-implementation/transflower/datasets/lodes_ma/loader.py:51-58`.

## Logging

- **No `logging` module use anywhere.** All progress output is
  `print()`-based.
- Every orchestrator has a `verbose: bool = True` field on its `*Config`
  dataclass; print sites are gated on `cfg.verbose` (see
  `algorithms/pytorch-implementation/mhgl/model.py:186-191`,
  `algorithms/pytorch-implementation/transflower/model.py:153-154`,
  `algorithms/pytorch-implementation/aagnn/train.py:167-171`).
- Print-line prefix convention: `"[ALGORITHM] message"` or
  `"  [stage/substep] message"`:
  - `[GeoTile2Vec] Stage 1: building mobility events…`
    (`geotile2vec/model.py:119`).
  - `[AAGNN] |R|=… |D|=… |T|=…` (`aagnn/model.py:103-106`).
  - `[ACDNE] n_s=… n_t=…` (`acdne/model.py:116-120`).
  - `[MHGL] |labelled_normal|=…` (`mhgl/model.py:188-191`).
  - `  [stage1/skipgram] epoch …` (`geotile2vec/stage1_mobility.py:155`).
  - `  [transflower] epoch …` (`transflower/model.py:154`).

## Function & Module Design

**Function size:** small enough that the docstring can list every
paper-equation it implements. When the loop body grows, split it into
helpers (e.g. `_forward_no_grad` in
`algorithms/pytorch-implementation/aagnn/train.py:29-41` and
`algorithms/pytorch-implementation/mhgl/train.py:43-55`).

**Parameters:**
- All optional knobs are keyword-only (`*` separator) in trainer and loss
  functions.
- Defaults match the paper exactly; deviations are commented with the
  paper-cited value (e.g. `hidden_dims: tuple[int, ...] = (256, 128, 64, 32)
  # paper §4.2` in `algorithms/pytorch-implementation/mhgl/model.py:48`).
- `seed: int = 0` is universally the last keyword argument before
  device / verbose (`AAGNNConfig`, `MHGLConfig`, `ACDNEConfig`).

**Return values:**
- Tuples for paired arrays whose elements correspond positionally
  (`(o_events, d_events)`, `(R_idx, D_idx, T_idx, c)`).
- Dicts for heterogeneous training history
  (`{"train_losses": [...], "val_losses": [...]}`).
- Dataclasses for richer records.

**Module exports:**
- No `__init__.py` files in algorithm folders. Sibling modules are imported
  directly by name after `cd`-ing into the algorithm directory.
- No barrel files, no `__all__` declarations.
- The orchestrator `model.py` is the public API surface (`fit`, `score` /
  `predict` / `embeddings`, `save`, `load`, `history` / `patterns`,
  plus the `*Config` dataclass).

## Reproducibility

Every `fit()` re-seeds all RNGs that touch the training path, ordered
identically across orchestrators:

```python
random.seed(cfg.seed)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
```

Anchor: `algorithms/pytorch-implementation/aagnn/model.py:83-85`,
`algorithms/pytorch-implementation/acdne/model.py:84-86`,
`algorithms/pytorch-implementation/mhgl/model.py:114-116`. The
TransFlower variant omits `random.seed` because it doesn't use the stdlib
`random` module (`transflower/model.py:89-90`).

## Save / Load

Every orchestrator exposes `save(path: str)` and a classmethod
`load(path: str)` round-trip via `torch.save` / `torch.load(...,
map_location="cpu", weights_only=False)`. The on-disk dict uniformly
contains:

- `"config"`: `self.config.__dict__`
- one or more `*_state_dict` keys for `nn.Module`s
- the cached numpy outputs (`scores`, `cached_e_t`, etc.)
- `feat_dim`, `n` for input-shape validation on `load`

Canonical anchors:
`algorithms/pytorch-implementation/aagnn/model.py:174-218`,
`algorithms/pytorch-implementation/mhgl/model.py:279-339`,
`algorithms/pytorch-implementation/acdne/model.py:226-275`.

`weights_only=False` is intentional (loading non-tensor metadata such as
the config dict) and `tarfile.extractall(..., filter="data")` is used in
fetch scripts (`algorithms/pytorch-implementation/aagnn/datasets/cora/fetch.py:103-105`)
— both are deliberate, security-aware choices. Do not "fix" them.

## Dependencies

- One `requirements.txt` per algorithm. No top-level `pyproject.toml`,
  no monorepo lockfile.
- PyTorch versions are loose pins (`torch>=2.0`, `numpy>=1.24`,
  `scipy>=1.11`, `scikit-learn>=1.3`).
- MLX algorithms swap the framework line only:
  `algorithms/mlx-implementation/aagnn/requirements.txt` =
  `mlx>=0.31` + the same numpy / scipy / sklearn pins.
- Per-dataset adapters add their own `requirements.txt` for the fetch /
  parse path (`requests`, `pandas`, `scipy.io`).

## Prohibitions (from `AGENTS.md`)

The executor MUST NOT introduce any of the following — they are
explicitly out of scope:

- **No proprietary datasets** from source papers committed to the repo.
  Real datasets live behind `fetch.py` scripts that download from public
  mirrors (LINQS Cora, ArnetMiner DBLP/ACM, LEHD LODES). Paper PDFs go in
  `.research/` (gitignored).
- **No baseline-model comparisons.** Each algorithm folder ships only the
  target algorithm; AAGNN does not include Radar / DOMINANT / AnomalyDAE,
  ACDNE does not include MMD / DANN baselines, etc.
- **No downstream-evaluation tasks** (e.g. XGBoost classifiers on
  embeddings). The algorithm's output (embedding matrix, anomaly score
  vector, predicted label vector) is what users plug into their own
  evaluation.
- **No `tests/` directory.** `example.py` IS the test (see TESTING.md).
- **No backward-compat shims**, **no feature flags**, **no speculative
  abstractions**, **no shared utility modules** across algorithms.
  `_short(losses)` is reimplemented in `aagnn/example.py`,
  `acdne/example.py`, and `mhgl/example.py` rather than extracted — this
  is intentional under the policy.

---

*Convention analysis: 2026-05-10*
