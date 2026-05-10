# Codebase Concerns

**Analysis Date:** 2026-05-10

This document surfaces in-flight work, fragile areas, and known concerns
in the `supernova` repository as of branch `add-acdne-dblp-acm-dataset`
(HEAD `a8cb198`). Wherever the original task brief diverged from the tree,
this document follows the tree.

---

## In-Flight Modifications (uncommitted)

`git status` shows 11 modified files plus 2 untracked entries spanning both
the PyTorch and MLX implementations. They are not random touch-ups — they
are three coherent sweeps and one new module.

### Sweep 1 — explicit L2 in the loss (paper-literal Eq. 5 / Eq. 3.6)

**Files:**
- `algorithms/pytorch-implementation/aagnn/train.py`
- `algorithms/mlx-implementation/aagnn/train.py`
- `algorithms/pytorch-implementation/mhgl/train.py`
- `algorithms/mlx-implementation/mhgl/train.py`

**What changed:** the L2 / Frobenius regulariser was previously delegated to
the optimiser's `weight_decay` argument (Adam/SGD L2-on-grad). It is now
added to the loss tensor explicitly as `+ 0.5 * weight_decay * sum(p*p)`,
and `weight_decay` is removed from the optimiser constructor.

**Why:** the docstrings call out "bit-for-bit numerical equivalence" between
the PyTorch and MLX backends. PyTorch Adam's `weight_decay` is L2-on-grad
(coupled-decay); the MLX port hand-rolled an explicit L2-on-loss summand.
The two formulations are mathematically equivalent under SGD but differ for
adaptive optimisers (Adam scales the gradient before stepping, so the
coupled vs. decoupled distinction matters). The fix unifies both backends
on the paper-literal formulation.

**Risk for downstream phases:** any prior README hyperparameter table
reporting `weight_decay=X` now refers to a slightly different update rule.
Re-tuning may be needed if loss curves are compared across versions. The
`weight_decay=0` path is unaffected.

### Sweep 2 — MHGL contraction is now pooled-pair-mean

**Files:**
- `algorithms/pytorch-implementation/mhgl/train.py`
- `algorithms/mlx-implementation/mhgl/train.py`

**What changed:** Eq. 3.6's contraction term was `mean_i [ mean_{j in D_i}
||h_j - c_i||^2 ]` (per-pattern mean, then mean-of-means). It is now
`(1/(p*n)) sum_i sum_j ||h_j - c_i||^2` — a pooled mean over every (i, j)
pair. Large patterns now contribute proportionally instead of being
down-weighted by their own bucket size.

**Risk:** loss curves and absolute loss magnitudes change — tutorial
hyperparameters (`sigma`, `lr`, `weight_decay`) tuned against the
mean-of-means form may need re-checking. Both backends moved together so
PyTorch ↔ MLX parity is preserved.

### Sweep 3 — MLX TransFlower correctness fixes

**Files:**
- `algorithms/mlx-implementation/transflower/model.py`
- `algorithms/mlx-implementation/transflower/flow_predictor.py`

**What changed:**
1. `model.py` adds a custom `RMSpropMomentum(optim.Optimizer)` subclass
   because `mlx.optimizers.RMSprop` has **no momentum buffer**. The
   previously-shipped MLX TransFlower silently ignored
   `cfg.momentum` (paper §4.1.3 prescribes momentum = 0.9). This is a
   correctness regression in the v0.4 MLX release that this sweep fixes.
2. `flow_predictor.py` now plumbs an additive `(B, 1, 1, N)` `-inf` mask
   into the transformer for padded destination keys. Previously padded
   keys leaked into attention.

**Risk:** any score / CPC numbers reported from the prior MLX TransFlower
were optimised with momentum=0 and leaky attention. They are not
comparable to the paper or to the PyTorch path until a re-run.

### Sweep 4 — MHGL exposes abnormal-pattern PDE output

**Files:**
- `algorithms/pytorch-implementation/mhgl/model.py`
- `algorithms/mlx-implementation/mhgl/model.py`

**What changed:** Algorithm 2 line 2 calls PDE on labelled abnormals
*and* labelled normals; only the latter previously survived. PDE is now
also run on labelled abnormals and exposed via a new
`MHGL.abnormal_patterns()` accessor. Eq. 3.6's repulsion term still uses
raw labelled-anomaly indices (so the loss math is unchanged).

**Risk:** the new field is added to `save()` payloads. `load()` falls
back to `[]` for pre-sweep checkpoints in MLX (`ckpt.get(...)`) but the
PyTorch `load()` uses `ckpt.get("abnormal_patterns", [])` too — both
backwards-compatible. Watch for stale checkpoints in user environments
once this lands.

### New file — native MLX Places365 backbone

**Files:**
- `algorithms/mlx-implementation/geotile2vec/places365_backbone.py` (new, 252 LOC)
- `algorithms/mlx-implementation/geotile2vec/stage2_streetview.py` (rewritten, +73 / -22)
- `algorithms/mlx-implementation/geotile2vec/requirements.txt` (adds `pillow>=10.0`)

**What changed:** the MLX Geo-Tile2Vec port previously refused to extract
real Places365 features (`allow_synthetic` was the only working path). It
now ships a native MLX ResNet-18, lazily imports torch *once* to convert
the CSAIL checkpoint, and caches the converted weights at
`~/.cache/supernova/mlx-places365/resnet18_places365_mlx.npz`.

**Risk:**
- Torch becomes a **build-time** dependency for the MLX path on any
  machine that has not yet primed the cache. The error message is good
  (URL + cache path) but users on torch-free machines must side-load the
  npz from elsewhere.
- The conversion code (`places365_backbone.py:155-202`) remaps
  PyTorch's `layerN.idx.{conv1|bn1|conv2|bn2|downsample.{0,1}}` keys to
  MLX's `b0/b1/ds_conv/ds_bn` topology by string parsing. Any future
  ResNet-variant addition risks drift unless it goes through `_map_pt_key`.
- New requirement `pillow>=10.0` only listed for MLX. PyTorch path
  already pulls it via `torchvision`.

### `.gitignore` sweep — second paper cache

**File:** `.gitignore`

**Change:** adds a new line `research/` next to the existing `.research/`.
Both directories exist on disk with identical sets of six PDFs (AAGNN,
ACDNE, DECL, Geo-Tile2Vec, MHGL, TransFlower). Likely a rename in
progress. Not committed.

**Concern:** two on-disk paper caches will drift if one is updated and the
other isn't. Pick one; delete the other; commit before more papers land.

### Untracked half-finished dataset adapter

**Files:**
- `algorithms/pytorch-implementation/mhgl/datasets/amazon_photo/raw/amazon_electronics_photo.npz` (17 MB)
- `algorithms/pytorch-implementation/mhgl/datasets/amazon_photo/__pycache__/loader.cpython-313.pyc`

**Concern:** the raw npz and a compiled `loader.pyc` exist, but **none of
the source files** (`fetch.py`, `loader.py`, `run_example.py`,
`requirements.txt`) are present. The other three real-data adapters
(`aagnn/datasets/cora/`, `transflower/datasets/lodes_ma/`,
`acdne/datasets/dblp_acm/`) all ship the same four-file template. This
adapter is half-built and will silently fail until the source files are
restored.

---

## Real-Data Dependencies — failure modes

The repo ships four real-dataset adapters (one stubbed). All download at
runtime; all use the `fetch.py` / `loader.py` / `run_example.py` triple.
Footprints are smaller than the brief implied — there is **no Citi Bike
adapter** in this tree.

| Adapter | Source | Footprint | Failure mode |
|---------|--------|-----------|--------------|
| `aagnn/datasets/cora/` | LINQS UCSC + 2 mirrors | ~170 KB | 3 mirrors tried; all-down → script exits with `All mirrors failed.` Planetoid mirror has wrong format and loader will reject it (`fetch.py:43-49`). |
| `acdne/datasets/dblp_acm/` | github.com/shenxiaocam/ACDNE | ~1.7 MB | Single mirror; size-mismatch triggers re-download (`fetch.py:87-90`). No HTTP fallback. |
| `transflower/datasets/lodes_ma/` | lehd.ces.census.gov | ~19 MB on disk (16 MB OD + 1.2 MB WAC + 1.9 MB xwalk, all gzipped) | HEAD pre-flight; `--small` flag skips the OD CSV. `loader.py:144-149` returns `[]` (no flows) when OD missing — silent fallthrough rather than error. |
| `mhgl/datasets/amazon_photo/` | (unknown — source files missing) | 17 MB npz on disk | Adapter is half-built; importing it will fail. |

**Pretrained model fetch:**
- Places365 ResNet-18 (~45 MB per `geotile2vec/README.md:51`).
- PyTorch path: `algorithms/pytorch-implementation/geotile2vec/stage2_streetview.py:80-94` — tries `torch.hub`, on failure raises with the cache path `~/.cache/torch/hub/checkpoints/resnet18_places365.pth.tar`.
- MLX path: `algorithms/mlx-implementation/geotile2vec/places365_backbone.py:205-217` — lazy-imports torch, downloads, converts, caches at `~/.cache/supernova/mlx-places365/resnet18_places365_mlx.npz`.
- Both depend on `places2.csail.mit.edu` staying live. No second mirror.

**Failure surface:**
- All adapters fail loudly, except LODES OD missing (returns `[]`).
- All caches are user-home, so CI re-runs hit the network unless a layer
  caches `~/.cache`.
- No checksum verification on any download — corruption surfaces only via
  the loader's parsing error.

---

## Test Coverage Gaps

**No CI-style suite.** AGENTS.md (`Code style` section) explicitly forbids
a `tests/` directory: "`example.py` is the smoke test." There is no
GitHub Actions workflow, no pre-commit hook, no `pytest` runner.

Coverage today:
- Each algorithm has one `example.py` smoke test against synthetic data.
- Three real-data adapters add a `run_example.py`, but each runs end-to-end
  on the real dataset — minutes long, network-dependent.
- No unit tests for: `data.py` adjacency-matrix correctness, sparse-matrix
  shapes, anomaly-injection invariants, mixup pseudo-label sampling, the
  PT→MLX weight-conversion key remap, RMSpropMomentum step semantics.

**Risk:** the four in-flight sweeps above each change a numerical contract
(loss formulation, optimiser semantics, attention masking). Without unit
tests, regressions surface only when someone re-runs an `example.py` and
notices a different AUC. The recent ACDNE work documents exactly this
class of regression — paper convergence on real data needed 3000 iters
(`acdne/datasets/dblp_acm/run_example.py:7-14`) where synthetic converged
in 1000.

**Specific gaps that matter most for the in-flight sweeps:**
- No test that the explicit-L2 loss matches the optimiser-`weight_decay`
  loss within numerical precision (the stated "bit-for-bit" claim is
  uninstrumented).
- No test that MLX `RMSpropMomentum` matches PyTorch's RMSprop step.
- No test that the PT→MLX ResNet-18 weight remap produces matching
  forward-pass outputs.

---

## Anomaly-Injection Protocols — overlapping vocabulary, distinct logic

**Two different protocols** live in sibling files with overlapping
language (`anomaly`, `inject`, SBM, `clique_size`):

- AAGNN (`algorithms/pytorch-implementation/aagnn/data.py:175-214`):
  uniform SBM + planted **structural cliques** + planted **contextual
  feature swaps** (Ding et al. 2019 / Song et al. 2007 protocol). All
  anomalies tagged `label=1` (binary).
- MHGL (`algorithms/pytorch-implementation/mhgl/data.py:158-299`):
  rare-class community protocol — anomalies are **whole small communities**
  with their own feature centroid (paper §4.1). Tags include
  `ANOM_SEEN=1` / `ANOM_UNSEEN=2` so unseen anomalies can be excluded
  from V_train.

**Concern:** these are *not* duplicates and any future refactor that tries
to unify them into a shared helper will break MHGL's seen/unseen split or
AAGNN's clique invariant. Keep them separate. If a shared helper is ever
written, the contract should be "build SBM only" and let each algorithm
inject anomalies on top.

The PyTorch and MLX copies of each protocol are bit-for-bit identical
(verified via `diff` on `aagnn/data.py`); the MLX MHGL `data.py` differs
only in that `build_normalized_adj` returns dense `np.ndarray` instead of
`torch.sparse_coo_tensor`.

---

## MLX vs. PyTorch Parity Gaps

| Surface | PyTorch | MLX | Gap |
|---------|---------|-----|-----|
| Algorithm count | 5 (Geo-Tile2Vec, TransFlower, AAGNN, MHGL, ACDNE) | 5 (same) | none |
| Real-data adapters | 4 (cora, dblp_acm, lodes_ma, amazon_photo-WIP) | **0** | MLX has no `datasets/` directories under any algorithm |
| Geo-Tile2Vec real backbone | `torch.hub` direct load | Native MLX + lazy-torch one-shot conversion | Newly closed by `places365_backbone.py` |
| TransFlower optimiser | `torch.optim.RMSprop(momentum=0.9)` | **In-flight fix:** `RMSpropMomentum` subclass | Was silently momentum=0 prior to sweep |
| TransFlower attention mask | `nn.Transformer` handles padded keys | **In-flight fix:** explicit `(B,1,1,N) -inf` mask | Was leaking padded keys |
| MHGL `build_normalized_adj` | sparse COO | dense | Memory blow-up risk on large MLX runs (n^2 dense matrix) |

**Concern:** users following AGENTS.md verbatim ("Verify: `cd
algorithms/<framework>/<name> && python example.py`") get correct synthetic
results on both backends, but real-data parity requires running the
PyTorch adapter and there is no MLX equivalent. Any "MLX implementation
of paper X" claim should be qualified as "synthetic-only."

---

## Pretrained Backbone Cache — single point of failure

`places2.csail.mit.edu` hosts the only Places365 ResNet-18 checkpoint
referenced anywhere in the repo (PyTorch: `stage2_streetview.py:18`; MLX:
`places365_backbone.py:37`). No mirror, no checksum.

If CSAIL takes the URL down:
- PyTorch path: clean `RuntimeError` naming the cache path users can
  side-load.
- MLX path: same, plus instructions to convert on a torch-equipped
  machine and copy the resulting npz.

**Improvement path:** mirror the checkpoint to a project-controlled bucket
or release asset, add SHA-256 verification, prefer the project mirror with
CSAIL as fallback.

---

## Convergence / Hyperparameter Sensitivity

ACDNE on real DBLP→ACM exposes a steep iter cliff:

- Synthetic SBM (`acdne/example.py:56`): default `n_iters=1000`.
- Real DBLPv7→ACMv9 (`acdne/datasets/dblp_acm/run_example.py:42`):
  default `FULL_ITERS=3000`, smoke run `SMOKE_ITERS=2000`.
- Paper authors' upstream code uses ~5,600 iters (30 epochs × 187
  batches), per the docstring at `run_example.py:10-16`.

**Concern:** the synthetic smoke test cannot catch convergence regressions
that only show up on the BoW-sparse real-data setting. The `run_example.py`
budget is documented but easy to under-set if a user copies the synthetic
defaults.

Other algorithms have not surfaced equivalent cliffs yet, but the same
risk applies to MHGL (mixup pseudo-label count `augmentation_alpha`
defaults to 2 per paper §4.2) and AAGNN (no `n_epochs` documented for
real-data settings — only synthetic AUC ≥ 0.75 floor in `example.py`).

---

## Source Code Markers

`grep -rn "TODO\|FIXME\|XXX\|HACK" algorithms/ --include="*.py"` returns
**zero matches**. Either the codebase is genuinely free of in-line debt
markers or contributors avoid them stylistically. Either way, debt is not
being tracked in the source — it lives in commit messages, READMEs, and
this document.

**Recommendation:** when sweeps land that defer something for later (e.g.
"momentum buffer should eventually become a config flag"), use a
`# TODO(supernova): …` marker so future greps surface it.

---

## Paper PDF Cache — gitignored, single source

`.research/` and `research/` (the latter newly gitignored in the
uncommitted `.gitignore` diff) each hold the same six paper PDFs:

- aagnn-subtractive-aggregation-network-anomaly-cikm-2021.pdf
- acdne-adversarial-deep-network-embedding-aaai-2020.pdf
- decl-denoising-aware-contrastive-learning-time-series-ijcai-2024.pdf (no algorithm yet)
- geo-tile2vec-multimodal-urban-analytics-tsas-2023.pdf
- mhgl-unseen-anomaly-detection-multi-hypersphere-learning-sdm-2022.pdf
- transflower-explainable-transformer-commuting-flow-arxiv-2024.pdf

These are paywalled and intentionally not committed. If a developer wipes
their machine, paper context evaporates — and module docstrings only cite
"paper §3.2" / "Eq. 5" without restating the equations.

**Mitigation:** module docstrings already cite section / equation numbers
(per AGENTS.md `Implementation` rule) so a fresh PDF download lets a
reviewer reattach. The DECL paper is present but no algorithm is wired up
yet — likely the next addition.

---

## Fragile Areas (summary index)

| Area | Files | Why fragile |
|------|-------|-------------|
| MLX TransFlower optimiser | `mlx-implementation/transflower/model.py:25-71` | New custom Optimizer subclass; no test of step semantics vs. PyTorch RMSprop |
| MLX Places365 weight remap | `mlx-implementation/geotile2vec/places365_backbone.py:155-202` | String-parses PyTorch state-dict keys; ResNet variant changes break silently |
| MHGL `load()` checkpoint compat | `*/mhgl/model.py` (`abnormal_patterns` fallback) | Pre-sweep checkpoints will load with empty abnormal patterns — silent semantic change |
| LODES OD missing | `transflower/datasets/lodes_ma/loader.py:144-149` | Returns `[]` instead of raising; downstream training silently runs on no flows |
| Anomaly-injection vocabulary overlap | `aagnn/data.py`, `mhgl/data.py` | Future refactor risks unifying two distinct protocols |
| `.research/` ↔ `research/` drift | repo root | Two paper caches in flux; rename incomplete |
| Half-built `amazon_photo` adapter | `pytorch-implementation/mhgl/datasets/amazon_photo/` | Raw data + .pyc present, source files missing |

---

## Scaling Limits

- **MHGL MLX dense `A_hat`** (`mlx-implementation/mhgl/data.py`
  `build_normalized_adj`): O(n²) memory. Synthetic n=360 is fine; real
  Amazon-Photo n≈7,650 is ~225 MB float32 — borderline on Apple Silicon.
  Real Amazon-Computer (n≈13,752) → ~756 MB. Larger graphs need a sparse
  port.
- **Geo-Tile2Vec PCA** (`stage2_streetview.py:127-137`): IncrementalPCA
  with `batch_size=max(n_components, 64)` — fine for thousands of shots,
  may need true streaming for hundreds of thousands.
- **TransFlower transformer**: O(N²) attention over destinations per
  origin — paper §3.1 settles at d_model=512, manageable.

---

## Missing Critical Features

- **No CI workflow.** AGENTS.md doesn't mention CI; there is no
  `.github/workflows/` directory.
- **No checksum verification on downloaded datasets or backbones.**
- **No MLX dataset adapters.** Real-data verification is PyTorch-only.
- **No DECL implementation** (PDF in `research/` but no algorithm folder).

---

*Concerns audit: 2026-05-10*
