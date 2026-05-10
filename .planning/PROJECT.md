# supernova

## What This Is

A from-scratch portfolio of geospatial and urban-analytics ML algorithms drawn from published research papers. Each algorithm under `algorithms/<framework>/<name>/` is independently runnable, paper-faithful, and produces measurable results via its own `example.py` smoke test. Built for personal exploration of urban-ML research and as a portfolio piece grounding those algorithms in Hong Kong open data.

## Core Value

Each algorithm runs end-to-end on real Hong Kong data with a measurable smoke-test result — paper fidelity, real local relevance, and a single demo that ties the five together.

## Requirements

### Validated

<!-- Shipped and confirmed valuable. Inferred from codebase map. -->

- ✓ 5 PyTorch implementations of urban-ML algorithms — `algorithms/pytorch-implementation/{geotile2vec,transflower,aagnn,mhgl,acdne}/` — existing
- ✓ 5 MLX counterpart implementations (partial parity) — `algorithms/mlx-implementation/{geotile2vec,transflower,aagnn,mhgl,acdne}/` — existing
- ✓ Per-algorithm synthetic data generators in `data.py` — existing
- ✓ Smoke-test contract: `example.py` exits non-zero unless a measurable property is satisfied (Welch's t-test, classification threshold, reconstruction loss bound) — existing
- ✓ Real-data adapters for 4 PyTorch algorithms — existing (PRs #1–#5):
  - MHGL (PR #1, merged)
  - TransFlower / LODES MA (PR #2, merged)
  - AAGNN / Cora (PR #3)
  - ACDNE / DBLPv7→ACMv9 (PR #4)
  - Geo-Tile2Vec / Citi Bike + OSM (PR #5)

### Active

<!-- Current scope. Building toward these. -->

- [ ] **MLX feature parity (loose)** — every PyTorch capability across the 5 algorithms has an MLX equivalent that runs to completion and hits the existing smoke-test thresholds. Internal numerics may differ from PyTorch.
- [ ] **HK real-data adapters for all 5 algorithms** — added alongside existing US/abstract adapters under each algorithm's `datasets/` directory. Each adapter ships a `fetch.py`, `loader.py`, and a smoke test on real HK data.
  - [ ] Geo-Tile2Vec on HK urban tiles (e.g. HKSAR open data + OSM HK) — geo-spatial native fit
  - [ ] TransFlower on HK origin-destination flow (e.g. MTR/bus OD, Travel Characteristics Survey) — geo-spatial native fit
  - [ ] AAGNN on an HK-relevant attributed graph with anomaly injection — abstract algorithm needs an HK graph to bind to
  - [ ] MHGL on an HK-relevant attributed graph for rare-class anomaly detection — same
  - [ ] ACDNE on a cross-network HK transfer setting (e.g. two HK academic / business / civic networks) — same
- [ ] **Lightweight web demo** — single Streamlit/Gradio-class app showing each of the 5 algorithms running on HK data with simple visualizations. Not a deployed SaaS; runs locally as the integrated artifact.

### Out of Scope

<!-- Explicit boundaries. Includes reasoning to prevent re-adding. -->

- Replacing existing US/abstract real-data adapters (Cora, DBLP→ACM, LODES MA, Citi Bike) — HK adapters are added alongside; user picks via flag. *Why:* preserves the work in PRs #1–#5 and keeps cross-region comparison possible.
- Strict numerical equivalence between MLX and PyTorch (bit-identical outputs given same input + seed) — loose feature parity only. *Why:* paper-level behavioral parity is the goal; chasing bit-for-bit equality across two frameworks is a tar pit and not the value driver.
- Heavyweight production deployment of the web demo (containerization, auth, multi-user state, hosting) — local demo only. *Why:* personal project; demo is for showcase, not for users.
- Comparative baselines from the source papers — out of scope per `AGENTS.md`. *Why:* repo focuses on faithful target-algorithm implementation, not benchmark reproduction.
- Downstream evaluation tasks (e.g. XGBoost classifiers consuming the embeddings) — out of scope per `AGENTS.md`. *Why:* algorithm output embeddings are what users plug into their own evaluation; we don't take that opinion.
- Adding a 6th algorithm in this milestone. *Why:* milestone is depth (parity + HK + demo), not breadth.
- CI / formal pytest test suite. *Why:* `example.py` smoke tests are the contract per `AGENTS.md`; introducing CI is a separate decision.

## Context

- **Brownfield project.** Codebase map exists at `.planning/codebase/` (7 docs, written 2026-05-10). `STACK.md`, `ARCHITECTURE.md`, `STRUCTURE.md`, `CONVENTIONS.md`, `TESTING.md`, `INTEGRATIONS.md`, `CONCERNS.md` are the source of truth for current state.
- **PyTorch real-data work already shipped.** All 5 PyTorch algorithms have a real-data adapter committed (some merged, some open as PRs). HK is the next layer on top.
- **MLX implementations have in-flight work.** 11 uncommitted modifications cluster into four sweeps (explicit-L2, MHGL pooled-pair-mean contraction, MLX TransFlower correctness, MHGL abnormal-pattern PDE) plus a new untracked `algorithms/mlx-implementation/geotile2vec/places365_backbone.py`. These should land before or be folded into the parity milestone.
- **No web/UI infrastructure exists** in the repo today. The demo is greenfield within `supernova` and should respect the flat per-algorithm folder convention (likely a top-level `demo/` or `web/` directory rather than touching algorithm folders).
- **Two non-geo algorithms (AAGNN, MHGL) and one cross-network algorithm (ACDNE) need HK data semantics defined** — research will surface candidate HK graph datasets for these. The geo-spatial pair (Geo-Tile2Vec, TransFlower) maps cleanly to HK transit/POI data; the others need creativity.
- **MLX is Apple Silicon only.** The web demo's MLX path will not run on non-Apple-Silicon machines; PyTorch is the universal fallback.
- **Personal-interest pace.** No external deadline; quality and learning value over velocity.
- **Paper PDFs live in `.research/`** (gitignored) — preserve them locally for context recovery.

## Constraints

- **Tech stack**: PyTorch (universal) + MLX (Apple Silicon) — both must keep working; no framework consolidation.
- **Repo layout**: Flat algorithm folders under `algorithms/<framework>/<name>/`, 5–8 source files each — no shared `src/`, no deep package hierarchies (per `AGENTS.md`).
- **Code style**: Minimal comments (only when WHY is non-obvious), type hints throughout, `dataclass` for records, paper-section docstring citations at the top of each module (per `AGENTS.md`).
- **Testing**: `example.py` is the smoke test — no `tests/` directory, no pytest. Real-data adapters use a separate smoke script under their `datasets/` directory. Smoke tests must be deterministic and exit non-zero on regression.
- **Dataset hygiene**: HK datasets must be from open / publicly available sources. No proprietary or paywalled data committed to the repo. Large fetched archives stay in `datasets/raw/` (gitignored).
- **MLX runtime**: Apple Silicon required for MLX paths. The demo must degrade gracefully on non-Apple hardware (PyTorch-only mode).
- **Paper fidelity**: Hyperparameters and architecture decisions must cite source-paper sections in code comments / docstrings. HK datasets may need iteration count / capacity tuning; document deviations in adapter READMEs.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Hong Kong as the real-data context for all 5 algorithms | Personal relevance — user lives/works in HK and wants the algorithms to mean something locally | — Pending |
| HK adapters added alongside existing US/abstract adapters (not replacing) | Preserves shipped work in PRs #1–#5 and keeps cross-region comparison available | — Pending |
| MLX parity defined as "loose feature parity" — every PyTorch capability has an MLX equivalent that runs to completion | Bit-identical numerics is a tar pit; behavioral parity hits the same smoke-test thresholds and is the actual value | — Pending |
| Lightweight web demo (Streamlit/Gradio class) as the integrated artifact | Anchors the milestone with a single user-visible deliverable; lightweight respects the no-deployment scope | — Pending |
| Research HK open data sources before locking requirements | All 5 algorithms need HK data; non-geo algorithms (AAGNN, MHGL, ACDNE) require creative source mapping that benefits from a research pass | — Pending |
| Run the project as Vertical MVP slices | Each algorithm × HK adapter × MLX parity is a natural end-to-end slice; integration via the demo at the end | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-05-10 after initialization*
