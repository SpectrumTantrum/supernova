<!-- refreshed: 2026-05-10 -->
# Architecture

**Analysis Date:** 2026-05-10

## System Overview

```text
┌───────────────────────────────────────────────────────────────────────────┐
│                       supernova (portfolio repo root)                      │
│  AGENTS.md (canonical)  ├─ symlinks ─┤  CLAUDE.md, GEMINI.md               │
└────────────────────────────────┬──────────────────────────────────────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              ▼                                     ▼
┌────────────────────────────┐        ┌────────────────────────────┐
│  algorithms/               │        │  algorithms/               │
│    pytorch-implementation/ │        │    mlx-implementation/     │
│  (5 reference algorithms)  │        │  (5 mirror algorithms)     │
└─────────────┬──────────────┘        └────────────┬───────────────┘
              │                                    │
              ▼                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│        Per-algorithm flat module (5–8 source files, no shared core)        │
│                                                                            │
│   data.py   →   <algo>_<stage>.py / layer(s).py / gcn.py / pde.py …       │
│                                  │                                         │
│                                  ▼                                         │
│                              model.py  ◄──── orchestrator (config + fit)   │
│                                  │                                         │
│                                  ▼                                         │
│                              example.py ──── synthetic-data smoke test     │
│                                                                            │
│                              datasets/<name>/   (optional real-data        │
│                                fetch.py + loader.py + run_example.py)      │
└───────────────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| Repo manifest | Canonical agent instructions, layout convention, algorithm registry | `AGENTS.md` |
| Algorithm root (PyTorch) | Five independently runnable PyTorch implementations | `algorithms/pytorch-implementation/<name>/` |
| Algorithm root (MLX) | Five MLX mirror implementations of the same papers | `algorithms/mlx-implementation/<name>/` |
| Data + records + synthetic generator | Dataclasses, helpers, `Synthetic*` generator that drives `example.py` offline | `algorithms/<framework>/<name>/data.py` |
| Stage / layer module(s) | One module per pipeline stage cited in the paper | e.g. `stage1_mobility.py`, `stage2_streetview.py`, `layer.py`, `gcn.py`, `pde.py`, `flow_predictor.py`, `geo_encoder.py`, `train.py` |
| Top-level orchestrator | Dataclass `*Config`, `Model` class with `fit/predict/embed/score/save/load` | `algorithms/<framework>/<name>/model.py` |
| Smoke test | CLI entry point; runs the full pipeline on synthetic data and exits non-zero unless a measurable property holds | `algorithms/<framework>/<name>/example.py` |
| Real-data adapter (optional) | Downloads a public dataset, adapts it to the parent algorithm's record types, and ships its own runnable | `algorithms/pytorch-implementation/<name>/datasets/<dataset>/{fetch,loader,run_example}.py` |
| Paywalled paper PDFs | Reference material, kept off git | `research/` (and `.research/` per AGENTS.md convention — both gitignored) |

## Pattern Overview

**Overall:** Parallel independent algorithm modules — no shared core library.

**Key Characteristics:**
- Each algorithm folder is a self-contained mini-project: own `requirements.txt`, own entry point, own synthetic data, own README. Nothing imports across algorithms.
- Folders are deliberately **flat** (5–8 source files). No `src/`, no nested packages, no `__init__.py` exports.
- A two-framework split (PyTorch reference + MLX mirror) is achieved by duplicating the algorithm directory under `algorithms/mlx-implementation/<name>/` rather than by abstracting a backend layer.
- Per-algorithm internals follow a uniform pipeline pattern: `data` → stage modules → `model` orchestrator → `example` smoke test.
- Real-data support is an opt-in `datasets/<name>/` subfolder that uses a `sys.path` shim to import the parent `data.py` / `model.py` rather than reorganising the algorithm into a package.

## Layers

**Repository layer:**
- Purpose: Hosts independent algorithm projects + agent docs.
- Location: `/`
- Contains: `AGENTS.md`, `algorithms/`, `research/`, `CHANGELOG.md`, `.gitignore`.
- Depends on: nothing.
- Used by: humans, AI agents (via `AGENTS.md` / `CLAUDE.md` / `GEMINI.md`).

**Framework partition layer:**
- Purpose: Splits the portfolio by ML backend.
- Location: `algorithms/pytorch-implementation/`, `algorithms/mlx-implementation/`
- Contains: One subdirectory per algorithm.
- Depends on: nothing — folders are siblings, not parents.
- Used by: nothing — purely organisational.

**Algorithm layer (per `<framework>/<name>/`):**
- Purpose: A single paper, end-to-end.
- Location: e.g. `algorithms/pytorch-implementation/aagnn/`
- Contains: `data.py`, stage/layer modules, `model.py`, `example.py`, `README.md`, `requirements.txt`, optional `datasets/`.
- Depends on: external packages from its own `requirements.txt`; never on sibling algorithms.
- Used by: end users running `python example.py` from inside the folder.

**Pipeline-stage layer (inside an algorithm):**
- Purpose: One module per identifiable stage in the source paper (e.g. Stage 1 mobility, Stage 2 street-view; or encoder + predictor; or layer + train + pde).
- Location: e.g. `algorithms/pytorch-implementation/geotile2vec/stage1_mobility.py`, `stage2_streetview.py`; `algorithms/pytorch-implementation/transflower/geo_encoder.py`, `flow_predictor.py`; `algorithms/pytorch-implementation/mhgl/gcn.py`, `pde.py`, `train.py`.
- Contains: `torch.nn.Module` subclasses (or `mlx.nn.Module` in the MLX tree), pure-functional training helpers, and per-stage loss functions.
- Depends on: `data.py` records, third-party tensors.
- Used by: `model.py` only.

**Orchestrator layer:**
- Purpose: Wire the stage modules into a single trainer/predictor with paper-faithful default hyperparameters.
- Location: `algorithms/<framework>/<name>/model.py`
- Contains: `<Algo>Config` dataclass (every paper hyperparameter, citing section), `<Algo>` class (`fit`, `predict`/`score`/`embeddings`/`embed`, `save`, `load`), optional `TrainingHistory` dataclass.
- Depends on: `data.py` and the stage modules in the same folder.
- Used by: `example.py` and `datasets/<name>/run_example.py`.

**Smoke-test layer:**
- Purpose: One-command verification that the algorithm runs and meets a measurable property.
- Location: `algorithms/<framework>/<name>/example.py`
- Contains: synthetic-data construction, model `fit`, post-hoc statistical check (Welch's t-test, ROC-AUC, CPC, loss-decrease assertion), `sys.exit(0|1)`.
- Depends on: `data.py` (`Synthetic*` generator) and `model.py`.
- Used by: humans, CI.

**Real-data adapter layer (optional):**
- Purpose: Download a public dataset and adapt it to the parent algorithm's record types.
- Location: `algorithms/pytorch-implementation/<name>/datasets/<dataset>/`
- Contains: `fetch.py` (mirrors + idempotent download), `loader.py` (parses raw → parent dataclass via `sys.path` shim two levels up), `run_example.py` (real-data analogue of the parent `example.py`), `requirements.txt`, and a `raw/` cache directory.
- Depends on: parent algorithm's `data.py` / `model.py` via `sys.path.insert(0, parents[2])` shim.
- Used by: humans running the real-data smoke test.

## Data Flow

### Primary Synthetic-Data Path (every algorithm)

1. `example.py` parses CLI args and instantiates the algorithm's `Synthetic*` generator (`algorithms/<framework>/<name>/example.py`).
2. The generator produces typed records — `POI` / `Trajectory` / `StreetViewShot` for Geo-Tile2Vec, `Region` / `Flow` for TransFlower, `AttributedNetwork` for AAGNN/MHGL, `CrossNetwork` for ACDNE — all defined in `data.py`.
3. `example.py` constructs `<Algo>Config(...)` (often shrinking epochs/steps for speed) and `<Algo>(cfg)`.
4. `model.fit(records...)` runs each pipeline stage in order: build derived data → instantiate stage `nn.Module`s → run their training loops → cache outputs as private attributes (`self._V`, `self._layer`, `self._encoder`, …).
5. `example.py` calls the algorithm's read-side method (`embeddings()`, `score()`, `predict_distributions()`, `embed("target")`) and runs a statistical check.
6. `example.py` returns 0 (PASS) or 1 (FAIL) via `sys.exit`.

### Real-Data Path (algorithms with a `datasets/` subfolder)

1. User runs `python fetch.py` inside `algorithms/pytorch-implementation/<name>/datasets/<dataset>/`. Fetch tries an ordered list of mirrors, downloads to `raw/`, and is idempotent.
2. User runs `python run_example.py` in the same folder. The runner uses `sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))` to import the parent `data.py` and `model.py`.
3. `loader.py` parses the cached raw files and returns the parent algorithm's dataclass (`AttributedNetwork`, `CrossNetwork`, etc.), optionally applying the paper's anomaly-injection protocol (e.g. `load_cora_with_anomalies` for AAGNN).
4. `run_example.py` invokes the same `model.fit/score/predict` API the synthetic path uses, applies a relaxed statistical check tuned for the real distribution, and exits 0/1.

**State Management:**
- All training state lives on the orchestrator instance (`self._layer`, `self._V`, `self._encoder`, `self._predictor`, `self._c`, `self._scores`, `self.history`). The orchestrator is stateless until `fit()` runs; every read-side method raises `RuntimeError("Call fit() first.")` otherwise.
- Persistence is via `Model.save(path)` (a single `torch.save` / MLX equivalent dict containing `config.__dict__`, layer state dicts, cached outputs) and the matching `Model.load(path)` classmethod.

## Key Abstractions

**Frozen dataclass record:**
- Purpose: Strongly-typed, hashable input records to the algorithm.
- Examples: `POI`, `Trajectory`, `MobilityEvent`, `StreetViewShot`, `TileId` (NamedTuple) in `algorithms/pytorch-implementation/geotile2vec/data.py`; `AttributedNetwork` in `algorithms/pytorch-implementation/aagnn/data.py`; `Region` / `Flow` in `algorithms/pytorch-implementation/transflower/data.py`.
- Pattern: `@dataclass(frozen=True)` with `__post_init__` shape/dtype validation on numpy arrays (see `AttributedNetwork.__post_init__` at `algorithms/pytorch-implementation/aagnn/data.py:45`).

**Synthetic-data generator dataclass:**
- Purpose: Self-contained, seeded generator that yields the algorithm's input records — the contract that lets `example.py` run offline.
- Examples: `SyntheticCity` in `algorithms/pytorch-implementation/geotile2vec/data.py:227`; `SyntheticAttributedNetwork` in `algorithms/pytorch-implementation/aagnn/data.py`; `SyntheticCrossNetwork` in `algorithms/pytorch-implementation/acdne/data.py`.
- Pattern: `@dataclass` with knobs as fields and a `generate()` method returning `(records..., ground_truth)`.

**Orchestrator config dataclass:**
- Purpose: A single, fully-typed bag of paper hyperparameters with defaults pinned to the source paper's section/table.
- Examples: `GeoTile2VecConfig` (`algorithms/pytorch-implementation/geotile2vec/model.py:46`), `AAGNNConfig` (`algorithms/pytorch-implementation/aagnn/model.py:35`), `TransFlowerConfig` (`algorithms/pytorch-implementation/transflower/model.py:33`), `MHGLConfig` (`algorithms/pytorch-implementation/mhgl/model.py:43`), `ACDNEConfig` (`algorithms/pytorch-implementation/acdne/model.py:35`).
- Pattern: `@dataclass`; every field carries an inline comment citing the paper section/equation.

**Orchestrator class:**
- Purpose: End-to-end trainer/predictor with `fit/<read-side>/save/load`.
- Examples: `GeoTile2Vec`, `AAGNN`, `MHGL`, `ACDNE`, `TransFlower`.
- Pattern: `class <Algo>: def __init__(self, config: <Algo>Config | None = None)` → store `cfg`, init `Optional` private state to `None`. `fit()` returns `self` for chainable calls. `save/load` round-trips through `torch.save` (or MLX equivalent) of `config.__dict__` + layer state dicts + cached outputs.

**`torch.nn.Module` subclass per stage:**
- Purpose: Encapsulate one paper-defined neural component.
- Examples: `MobilityEventModel` (`stage1_mobility.py`), `Places365PretrainedResNet18` (`stage2_streetview.py`), `AbnormalityAwareLayer` (`aagnn/layer.py`), `GCNEncoder` (`mhgl/gcn.py`), `EmbeddingModule` / `NodeClassifier` / `DomainDiscriminator` (`acdne/layers.py`), `GeoSpatialEncoder` (`transflower/geo_encoder.py`), `FlowPredictor` (`transflower/flow_predictor.py`).
- Pattern: Inherits `torch.nn.Module` (or `mlx.nn.Module` in the MLX tree). Module docstring cites paper section and equations.

**Pure-functional training helpers:**
- Purpose: Stateless functions invoked by `model.fit()`; keep `nn.Module`s pure.
- Examples: `train_skipgram`, `train_triplet_metric`, `average_to_tiles` (geotile2vec); `compute_pseudo_labels`, `train_aagnn`, `anomaly_scores` (aagnn); `compute_centres`, `compute_high_confidence`, `train_mhgl`, `fit_pde` (mhgl); `train_acdne` (acdne); `flow_cross_entropy`, `common_part_of_commuters` (transflower).
- Pattern: free functions taking modules + data + hyperparams, returning loss histories or scored outputs.

## Entry Points

**Synthetic smoke test:**
- Location: `algorithms/<framework>/<name>/example.py`
- Triggers: `cd algorithms/<framework>/<name> && python example.py`.
- Responsibilities: build synthetic data, fit the orchestrator with a shrunk training budget, run a statistical check, exit 0/1.

**Real-data smoke test (opt-in):**
- Location: `algorithms/pytorch-implementation/<name>/datasets/<dataset>/run_example.py`
- Triggers: first `python fetch.py`, then `python run_example.py` in the same folder.
- Responsibilities: import the parent algorithm via a `sys.path` shim, load real data via `loader.py`, fit the orchestrator, run a relaxed real-world statistical check, exit 0/1.

**Library import:**
- Location: `algorithms/<framework>/<name>/model.py`
- Triggers: `from model import <Algo>, <Algo>Config` (run from inside the folder; folders are not packages).
- Responsibilities: expose the orchestrator class for downstream notebook/script use.

## Architectural Constraints

- **Flat folders, no packages:** No `__init__.py` is intended in algorithm folders, no `src/` layer, no cross-algorithm imports. Imports inside an algorithm are bare module names (`from data import …`, `from model import …`) and only resolve when CWD is the algorithm folder or its `datasets/<dataset>/` subfolder (which fixes resolution with a `sys.path.insert`).
- **No shared utilities:** Identical helpers (e.g. `haversine_meters`) are intentionally duplicated across algorithms rather than extracted to a shared package; preserves independent runnability.
- **Framework duplication, not abstraction:** PyTorch and MLX implementations are siblings under `algorithms/`, not branches of a backend abstraction. Each MLX module is a hand-port of its PyTorch sibling; APIs match by convention (same `<Algo>Config` fields, same `fit/score/embed`).
- **Paper fidelity over generality:** Hyperparameter defaults match the source paper; module/file docstrings cite the paper section. No backward-compat shims, no feature flags, no speculative abstractions (per `AGENTS.md`).
- **No tests directory:** `example.py` is the single smoke test. There is no `pytest`/`unittest` harness.
- **Offline-fallback for pretrained weights:** Vision backbones load via `torch.hub.load_state_dict_from_url` (PyTorch) or a hand-converted `.npz` cache (MLX, see `algorithms/mlx-implementation/geotile2vec/places365_backbone.py`), with a `RuntimeError` that names the expected cache path for side-loading.

## Anti-Patterns

### Cross-algorithm imports

**What happens:** Tempting to factor common helpers (e.g. `haversine_meters`, dataset-loading boilerplate) into a shared package above `algorithms/`.
**Why it's wrong:** Breaks the "each algorithm independently runnable" invariant in `AGENTS.md`. Adds hidden coupling that complicates per-algorithm `requirements.txt` and prevents copying a single algorithm folder out of the repo.
**Do this instead:** Duplicate the helper inside each algorithm's `data.py` (see `haversine_meters` repeated in `algorithms/pytorch-implementation/geotile2vec/data.py:64` and `algorithms/pytorch-implementation/transflower/data.py`).

### Adding `tests/` directory or `pytest` fixtures

**What happens:** A `tests/` folder appears alongside `data.py` / `model.py`.
**Why it's wrong:** `AGENTS.md` explicitly forbids it — `example.py` is the smoke test, and the smoke-test contract (PASS iff a measurable property holds) is the gate.
**Do this instead:** Tighten the assertion at the bottom of `example.py` and let it gate via `sys.exit`. For real datasets, add a `datasets/<name>/run_example.py`.

### Speculative `src/` layout or `__init__.py` exports

**What happens:** A maintainer reorganises an algorithm into `<algo>/<algo>/{data,model,...}.py` with a top-level `__init__.py`.
**Why it's wrong:** The `sys.path` shim used by `datasets/<dataset>/` adapters (e.g. `algorithms/pytorch-implementation/aagnn/datasets/cora/loader.py:23`) assumes the parent `data.py` / `model.py` sits two directories up. Repackaging breaks every adapter and contradicts the flat-layout convention.
**Do this instead:** Keep new files at the algorithm root. If a new pipeline stage is needed, add `algo_<stage>.py` next to `model.py`.

### Embedding downstream evaluation tasks

**What happens:** Adding XGBoost classifiers, baseline-model comparisons, or paper-table reproduction inside an algorithm folder.
**Why it's wrong:** `AGENTS.md` declares these out of scope. The deliverable is the algorithm's output (embedding / scores / distribution); downstream evaluation is the consumer's job.
**Do this instead:** Extend `example.py` only with assertions that confirm the algorithm satisfies a measurable property (Welch's t-test, ROC-AUC threshold, loss-decrease check).

## Error Handling

**Strategy:** Fail-fast with descriptive `RuntimeError` / `ValueError`; never silently degrade.

**Patterns:**
- Read-side methods that need a trained model raise `RuntimeError("Call fit() first.")` if private state is `None` (see `GeoTile2Vec.embeddings`, `AAGNN.score`, etc.).
- Dataclass `__post_init__` raises `ValueError` on shape/dtype mismatches (see `AttributedNetwork.__post_init__` at `algorithms/pytorch-implementation/aagnn/data.py:45`).
- Pretrained-weight loaders raise `RuntimeError` naming the expected cache path so users can side-load (see `algorithms/mlx-implementation/geotile2vec/places365_backbone.py` docstring).
- `example.py` may catch a specific `RuntimeError` from a Stage-2 weight download and gracefully fall back to Stage 1 only (see `algorithms/pytorch-implementation/geotile2vec/example.py:101`).
- `datasets/<dataset>/loader.py` raises `FileNotFoundError` with the exact `python fetch.py` instruction when raw files are missing (see `algorithms/pytorch-implementation/aagnn/datasets/cora/loader.py:42`).

## Cross-Cutting Concerns

**Logging:** Plain `print()` statements gated by `cfg.verbose` (no `logging` module). Each orchestrator prints stage banners (`"[GeoTile2Vec] Stage 1: …"`, `"[AAGNN] |R|=…"`).
**Validation:** Performed in dataclass `__post_init__` for input records, and via `if cfg.<knob> < 0: raise ValueError` guards inside data-prep helpers (see `build_skipgram_pairs` in `algorithms/pytorch-implementation/geotile2vec/data.py:192`).
**Authentication:** Not applicable — no networked services. Dataset `fetch.py` scripts are unauthenticated HTTP `GET` against public mirrors with `User-Agent` headers (see `algorithms/pytorch-implementation/aagnn/datasets/cora/fetch.py:43`).
**Reproducibility:** Each `<Algo>Config` carries a `seed: int = 0`, and `<Algo>.fit()` seeds `random`, `numpy.random`, and `torch.manual_seed` (or `mx.random.seed`) before any stochastic step (see `AAGNN.fit` lines around `algorithms/pytorch-implementation/aagnn/model.py:83`).
**Device handling:** Each `<Algo>Config` carries a `device: str = "cpu"`. The orchestrator passes it to tensor moves and submodule constructors; MLX siblings ignore the field (MLX is unified-memory).

---

*Architecture analysis: 2026-05-10*
