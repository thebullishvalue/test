# CHANGELOG
### Sanket — Wave-Regime Composite Index Terminal

All notable changes to the **Sanket** platform are documented here. Sanket is part of the **Pragyam Product Family** by [@thebullishvalue](https://github.com/thebullishvalue).

Format: `[version] · date — release title`

---

## [v3.5.0] · 2026-06-02
### Breadth Intelligence — Market & Sector Advance/Decline as a Three-Axis Edge

**"The Tape Joins the Engine"**

A feature release that folds advance/decline **breadth** into the ranking + intelligence stack along **three orthogonal axes**, from a single shared engine. Breadth is derived from the universe close panel the screener already holds (`get_universe_data`), so there is **zero new data dependency** — the same advances/declines the market shows are now read by the engine. A universe-wide breadth value is a *timing/regime* signal (one number per date), not a stock-discriminating one, so it deliberately enters in three different places rather than as a naïve cross-sectional factor (which would be inert — identical for every stock on a date → zero cross-sectional IC).

#### Features
- **Breadth Engine** (`breadth_engine.py`, new): ports the Hemrek "Relative Breadth" oscillator — EMA(10)-smoothed `A/(A+D)` blended with six Fibonacci-period SMAs (~[0,1], 0.40/0.50 oversold/overbought bands) — and adds `Breadth_Momentum` (3-bar) plus **sector-relative breadth** (`Sector_Rel_Breadth = sector_breadth − universe_breadth`, de-meaned so it's orthogonal to the market level). Built once per run from `data_dict`; attached identically in the live-screener and calibration-harvest paths so train and apply features match bar-for-bar. Self-tested.
- **Path A · Market-breadth regime tilt** (`priority_engine.py :: _breadth_tilt`, in `compute_priority`): a **bounded per-side multiplier** on final priority — `long ×(1+α·b)`, `short ×(1−α·b)`, where `b∈[-1,1]` blends breadth level + momentum and `α=0.20`. Breadth is uniform within a date, so this rescales **long-vs-short exposure** without reordering either side (within-side rank is invariant to the tilt — verified). Tilt is bounded to `[0.80, 1.20]` so it can never dominate the calibrated factors. Not IC-calibratable (zero within-date variance) → fixed, not searched.
- **Path B · Breadth confidence feature** (Layer 2): new `breadth_align = dir·(Universe_Breadth − 0.45)` in `CONF_FEATURES` / `signal_conf_features` — a long fired into an advancing tape scores positive, a short into a strong tape negative. A legitimate *temporal* feature (market-wide is fine for a per-signal classifier). Flows through `calibrate_signal_confidence` automatically (persisted in `feature_names`; name-aligned at predict, so old models still score).
- **Path C · F8 sector-relative breadth factor** (cross-sectional): `beta_F8_breadth_long/short` added to the inner score, the IC tuner kernel (`_PrecomputedDataset.M` → 7 factors, `_evaluate_ic`), and the Optuna search space. Because F8 is de-meaned against the market it varies across the cross-section (stocks in out-participating sectors rank up) and **can earn real IC** — unlike F7, it's searched by default (`enable_f8=True`), with the gate retained so it can be pinned to 0 if validation shows no edge. Sector map built from NSE sectoral-index membership (India universes only; fail-fast + cached, degrades to Path-C-off elsewhere so the screener is never stalled by breadth).

#### Design discipline
- **No double-counting**: Path C uses *sector minus universe* breadth, keeping it orthogonal to Path A's market-level tilt; Path A and Path B both condition on the same tape but on different stages (exposure scaling vs per-signal confidence), and Path B must prove marginal AUC under the existing probation gate. The self-regulating mechanisms (Optuna IC search for F8, Layer-2 calibration for `breadth_align`) shrink breadth toward zero automatically if it has no edge — so "default-on" is safe by construction.
- **Pine parity unaffected**: breadth is a market-wide / cross-sectional concept with no single-symbol equivalent, so — like Layer 3 — it is Python-only and `wrci.pine` carries the version stamp only. 1:1 indicator parity is preserved.

#### Versioning
- **Unification to `v3.5.0`** across `sanket.py`, `ui/theme.py`, `logger.py`, `breadth_engine.py`, `wrci.pine`, `count.pine` (stamp), `README.md`, and `LICENSE`.

#### Documentation
- **README**: new `breadth_engine.py` in the project structure; line counts refreshed — `sanket.py` (6,910), `priority_engine.py` (1,041), `intelligence.py` (807), `breadth_engine.py` (292).

---

## [v3.4.1] · 2026-06-02
### WaveTrend Parity Patch — `ci` Denominator Guard

**"1:1, To The Letter"**

A parity patch that brings `wrci.pine` into exact textual agreement with the `sanket.py` engine. A full line-by-line audit of the indicator against `run_full_analysis` / `compute_signal_sets` confirmed both sides are already in 1:1 functional parity across every engine — WaveTrend, Liquidity, AT Filter (Ehlers AutoTune), Conviction, Pulse, the Hemrek Count (HCI) trend gate, and all three signal sets (A · Momentum, B · Crossover, C · Threshold) with identical gates, parameters, and defaults. The audit surfaced a single source-level divergence: the WaveTrend channel-index (`ci`) zero-deviation guard was floored differently on each side. Output is unchanged on every real-data bar; the patch only closes a divergence that could appear on a pathologically flat series.

#### Fixes
- **WaveTrend `ci` guard aligned** (`wrci.pine §3A`): `(ap − esa) / (0.015 * math.max(d, 1e-9))` → `(ap − esa) / math.max(0.015 * d, 1e-6)`, matching `sanket.py :: run_full_analysis` exactly. The Pine form floored the raw deviation `d` (effective denominator floor ≈ 1.5e-11); the Python form floors the whole denominator at 1e-6. On real bars `0.015·d` dominates both guards, so screener and chart values are identical — the change only removes a ~6× divergence in the regime where `0.015·d < 1e-6` (near-flat / synthetic series). No engine, signal-logic, gate, parameter, or API change otherwise.

#### Versioning
- **Unification to `v3.4.1`** across `sanket.py`, `ui/theme.py`, `logger.py`, `wrci.pine`, `count.pine`, `README.md`, and `LICENSE`.

#### Documentation
- **README**: version strings refreshed to `v3.4.1`; project-structure line counts corrected to the current files — `sanket.py` (6,876), `priority_engine.py` (976), and `wrci.pine` (529, up from the stale 461 figure that predated the HCI / AT-Filter trace additions already present in the indicator).

---

## [v3.4.0] · 2026-06-02
### Meta Intelligence — The Final Intelligence Layer

**"The Final Layer"**

A feature release that upgrades **Layer 3** from a user-set threshold on a single confidence scalar into a calibrated, walk-forward-validated **meta intelligence** model. Layer 3 now *fuses* the two informationally-orthogonal views the earlier layers keep separate — the **cross-sectional Priority rank** (`compute_priority`) and the **per-signal Intel confidence** (Layers 1/2, per-symbol) — into a single `Meta_Score`, a 0–3 tier, and a human reason. Like the rest of the stack it is **probation-gated**: it may reorder/filter only when it has demonstrated out-of-sample edge, otherwise it stays advisory. `wrci.pine` carries the version stamp only — Layer 3 is a Python-only post-ranking layer with no Pine equivalent, so 1:1 indicator parity is unaffected.

#### Features
- **Layer 3 · Meta Intelligence** (`priority_engine.py`): `META_FEATURES` (rank percentile, confidence, their interaction, is-calibrated), `meta_conf_features`, `predict_meta_intel`, `set/get_active_meta_model`, and `compute_meta` → adds `Meta_Score` (0–1), `Meta_Tier` (0–3, fixed bands), `Meta_Source` (`meta`/`fallback`), `Meta_Active`, `Meta_Reason`, `Meta_Spread`. With no active model it falls back to `rank × confidence` (advisory).
- **Meta Intelligence calibrator** (`intelligence.py :: calibrate_meta`): materializes cross-sectional Priority on the harvested panel via a per-date `compute_priority` pass (the panel carries Intel confidence but not rank), fits a logistic on the same magnitude-aware directional-return-past-deadband label used by Layer 2, and reports out-of-sample diagnostics. New `_spearman_ir` helper computes the cross-sectional rank-IR (direction-signed, matching the Priority Engine's IC methodology).
- **Probation gate**: the model is `active` (allowed to reorder + Hide) **only if its OOS rank-IR beat naked Priority's** rank-IR and is positive; otherwise advisory (annotates, never hides). Same discipline that gates F7 and the Layer-2 filter.
- **Abstention**: when today's cross-section shows no spread in the Meta score, the screen falls back to the raw Priority order and labels it as such.
- **UI surfaces**: a new **`Meta`** column in the Action Dashboard, Priority Ranking, and Signal Strength tables (tier-banded fused score, ◆ calibrated / ◇ fallback); an **Intelligence-tab Layer-3 panel** reporting Meta-IR vs naked-Priority-IR, edge delta, AUC, and active/advisory status.

#### Behavior Changes
- **Layer-3 filter is now the Meta Filter**: the sidebar Off / Dim / Hide control and threshold act on the fused `Meta_Score` instead of `Intel_Confidence`. Today's fired signals filter by the Meta score; aged signals fall back to fire-bar Intel. An **advisory** meta model dims but **never hides** (probation guard); the threshold auto-seed prefers the meta AUC, then the Layer-2 AUC, then 0.45.
- **Profile artifacts**: each profile now persists a third model, `meta_intel`, alongside `weights` and `signal_conf`. Threaded through `save_profile` and every activation/import site via `set_active_meta_model`.
- **Calibration runner**: `run_priority_optimization` learns and logs the meta model (meta-IR vs priority-IR + active flag) right after Layer 2; `run_screener_analysis` applies `compute_meta` after `compute_signal_confidence`.

#### Versioning
- **Unification to `v3.4.0`** across `sanket.py`, `ui/theme.py`, `logger.py`, `wrci.pine` (stamp only), `README.md`, and `LICENSE`. (The prior `v3.3.0` changelog entry had shipped without the code version strings being bumped; this pass brings every component even.)

#### Documentation
- **README**: new **"The Intelligence Stack (Layers 1–3)"** section documenting all three layers, the probation/abstention discipline, and the Meta Filter; result-tabs and per-row output updated for the Intel + Meta columns; architecture blurbs and project-structure line counts refreshed.
- **LICENSE**: restriction §5 IP enumeration extended to the multi-layer intelligence stack (signal-confidence + meta intelligence calibration).
- Line counts: `sanket.py` (6,835), `priority_engine.py` (970), `intelligence.py` (793), `wrci.pine` (461), `logger.py` (226).

---

## [v3.3.0] · 2026-05-30
### Liquidity Engine, Inline Self-Tuning & Data-Source Hardening

**"Flow & Folded Intelligence"**

A feature release that adds a microstructure Liquidity engine and its per-set kinematic gates, folds the Self-Tuning calibration into the screener run (no separate mode), and hardens the F&O / index data sources. `wrci.pine` and `sanket.py` kept in 1:1 parity throughout.

#### Features
- **Liquidity Engine (microstructure flow)**: new ±100 oscillator (volume-weighted intrabar spread vs. multi-bar price impact → clipped z-score → sigmoid), with `liq_vel` (velocity) and `liq_accel` (acceleration). Added to both `wrci.pine` (§3B) and `sanket.py` (`run_full_analysis` → `Liquidity_Osc` / `Liq_Vel` / `Liq_Accel`), with a zero-volume divide guard on the Python side.
- **Inline, one-pass Self-Tuning**: the standalone "Intelligence (Self-Tuning)" analysis mode is **removed**; harvest + Optuna calibration now run inline on the **Single Date / Pulse** screener via `_ensure_intel_weights()`. Reuses a profile already calibrated **today** for the `(universe, index, timeframe)`; otherwise harvests a lookback (~2y daily / ~3y weekly) and calibrates, then ranks the screen with the tuned weights. Sidebar **Self-Tuning Intelligence** expander (below the Model Passport) carries trials / split / **Force recalibrate this run**.
- **Intelligence result tab**: Single-Date results gain an **Intelligence** tab (Train/Val IR, stability, factor-importance fANOVA chart, active-weights table) and a **Priority Rank** sub-tab listing the full universe by tuned priority (bull/bear aware).
- **NseKit F&O source**: F&O constituents now fetch via NseKit's official `underlying-information` endpoint as the primary source (survives datacenter-IP blocking), ahead of the legacy `equity-stockIndices` paths.

#### Behavior Changes
- **Per-set kinematic liquidity gates** (parity across `wrci.pine` + `sanket.py`): Set A & B require liquidity **level** (`Liquidity_Osc` same-signed); Set C requires liquidity **velocity** (`Liq_Vel`); Set D requires liquidity **level + acceleration** (`Liquidity_Osc` & `Liq_Accel`). Net effect: fewer, flow-confirmed signals.
- **Set C Δ-polarity gate**: Threshold now also requires `Conviction Δ` / `Pulse Δ` polarity (it previously omitted it), matching Sets A/B/D.
- **WT2 signal line = configurable MA, ALMA(20) default** (was SMA-4), plumbed through the sidebar/screener (`wt2_len`, `wt2_type`).
- **Profile key now includes timeframe**: profiles are keyed per `(universe, index, timeframe)` so daily and weekly weights no longer collide.

#### Fixes
- **F&O list correctness**: legacy `SECURITIES IN F&O` paths now skip the leading index-aggregate row (off-by-one phantom ticker) and de-duplicate; the NIFTY-500 fallback is flagged as a superset rather than reported as a clean F&O fetch.
- **Index-constituent resilience**: archive-CSV fallback tries both `archives.nseindia.com` and `nsearchives.nseindia.com`.
- **Model Passport refresh**: after a fresh inline calibration the Passport updated only on the next interaction; a guarded post-results `st.rerun()` now refreshes it in the same run (results are already persisted, so no recompute and no re-tune).
- **Streamlit deprecation**: all `use_container_width=True` replaced with `width='stretch'`.

#### Documentation
- **README**: added the Liquidity Engine core component + Micro Phase; corrected WT2 to configurable ALMA; reframed Intelligence as inline (no separate mode) with the new first-run workflow; renumbered Analysis Modes (Intelligence mode removed); fixed the profile key to `(universe, index, timeframe)`; Sets A–D table now lists each set's liquidity gate.
- **Docstrings/comments**: `compute_signal_sets` docstring documents the per-set liquidity gates; stale "Intelligence mode" references in the bulk-range comment and the legacy-profile import warning updated.
- Line counts: `sanket.py` (6,117), `wrci.pine` (523), `intelligence.py` (411), `priority_engine.py` (381).

---

## [v3.2.1] · 2026-05-21
### Set A Δ-Polarity Gate & Signal Engine Refactor

**"Symmetric Conviction"**

Behavior-changing tightening of the Momentum signal gate plus a focused refactor of `sanket.py` to extract long-lived inline blocks and surface the sidebar return as a typed dataclass. Pine indicator (`wrci.pine`) updated in lockstep to preserve 1:1 parity.

#### Behavior Change — Set A
- **Δ-polarity gate added to Set A** (Momentum): WT1/WT2 crossings now require `Conviction Δ` and `Pulse Δ` to be the same sign as the trade direction (long: both > 0; short: both < 0). Brings Set A in line with the gate already used by Sets B and D. Net effect: fewer but better-confirmed Set A signals; historical Set A counts will drop. The opposite-side Set B veto (long A blocked when B-short fires, and vice versa) is retained on top of the new gate.
- **`wrci.pine` synchronized**: Pine `momentum_long` / `momentum_short` now carry the same Δ gate, preserving 1:1 mathematical parity between the Python screener and the TradingView indicator.

#### Refactor (No Behavior Change)
- **`compute_signal_sets` helper extracted**: Sets A/B/C/D logic and the zone `Condition` column moved out of `run_full_analysis` into a dedicated function with a docstring explaining each set's predicate, gating, and the load-bearing `np.select` ordering for the zone label.
- **`SidebarState` dataclass**: `render_sidebar()` now returns a typed dataclass instead of a 16-element positional tuple. The data flow from sidebar to `main()` is name-keyed; new inputs no longer require updating a tuple unpack.
- **Shared HTML-builder palette helpers**: `_side_palette`, `_signed_color`, `_delta_arrow`, and `_GREEN` / `_RED` constants dedupe the green/red and arrow ternaries that were repeated across `_build_confluence_table_html`, `_build_signal_table_html`, `_build_narrative_table_html`, and `_build_signal_strength_table_html`.
- **Redundant imports removed**: Four local `import html as html_module` lines deleted — the top-level `import html` covers all `html.escape` callsites. Dead `from io import BytesIO` removed (all callers go through `io.BytesIO()`).

#### Documentation
- **README Signal Hierarchy table rewritten**: Sets B, C, D descriptions now accurately reflect the regime-filter crossover, signal-line-validated zone entry, and regime-zero-cross triggers respectively (previous text described unrelated logic).
- **README search-space breakdown corrected**: Now correctly enumerates 12 betas + 4 gammas (reversion + divergence, each side) + 5 tier multipliers = 21 dimensions.
- **README profile JSON example fixed**: Uses real field names (`val_score`/`train_score`/`sensitivity`/`tier_A_mult`) and the actual `" · "`-joined composite key format.
- **README line counts refreshed**: `sanket.py` (5,928) and `wrci.pine` (412).
- **`ui/components.py` Signal Types Reference rewritten**: Sets A and B descriptions match the actual triggers; Set D card added (CSS class `.signal-type.squeeze` already existed in `theme.css`, but the HTML card was missing).
- **Mislabeled comments fixed**: Two `# Set C: Momentum` comments at `sanket.py:2402-2408` and `sanket.py:5147` corrected to `# Set A: Momentum` (these labelled the legacy `L_`/`S_` alias columns, which read `long_cond` / `short_cond` — Set A's columns, not Set C's).

---

## [v3.2.0] · 2026-05-09
### System Hardening, Fidelity & UI Polish

**"Precision Instrument"**

Comprehensive institutional-grade hardening pass across the full stack — data correctness, multi-session isolation, calibration reliability, and terminal UI smoothness. No surface-level UX changes; all improvements are under-the-hood quality and fidelity improvements.

#### System Architecture
- **Per-session weight isolation**: Active weights now stored in `st.session_state["active_weights"]` per session, eliminating cross-user bleed when the app is deployed for multiple users simultaneously
- **Smart data registry**: TTL-aware fetch cache keyed by universe + date — avoids redundant OHLCV fetches across mode switches (15 min during market hours, 90 min outside)
- **Registry DataFrame copies**: Registry stores `.copy()` of each DataFrame, preventing downstream mutations from silently corrupting cached data
- **Reproducible calibration**: Optuna TPE sampler seeded with `seed=42` — calibration results are now reproducible across identical inputs
- **`HOLD_HORIZONS` constant**: Fibonacci-spaced horizons `[2, 3, 5, 8, 13]` extracted to `priority_engine.py` as a single source of truth, imported by `sanket.py` and `intelligence.py` — previously hardcoded in five separate places

#### Correctness Fixes
- **L/S Ratio division guard**: Long/Short signal ratio now emits `NaN` instead of `Inf` when short signal count is zero
- **Divergence order scaled to timeframe**: `argrelextrema` order parameter set to `2` for weekly and `3` for daily data — was fixed at `3` regardless of timeframe, causing missed weekly divergences
- **Regime detector warmup**: HMM state estimator now runs a 20-bar warm-up period before recording signal history, preventing false regime transitions at the start of the analysis window
- **ymax NaN/Inf guard**: Bar chart y-axis maximum now guarded against `NaN`/`Inf` values that caused silent chart rendering failures on short date ranges
- **Confluence score clipped**: `Confluence_Score` clamped to `[0.0, 1.0]` on both calibrated-priority and fallback paths — previously could exceed 1.0 on wide-spread cross-sections
- **% Change Since sentinel**: Percentage-change-since-analysis field now uses `None` sentinel instead of `0.0` when the analysis date equals the latest available date — eliminates spurious 0.00% displays

#### Calibration Improvements
- **Overfit detection split**: Separated `low_ir` (Val IR ≤ 0) and `overfit` (Train IR >> Val IR) flags — both are now detected independently with distinct user messages
- **Quality Check card**: Fourth metric card added to Calibration Diagnostics showing `No Edge` / `Overfit` / `Quality OK` status with semantic color coding
- **Small universe warning**: Calibration warns when average symbols per date falls below 20 — IC-based ranking is statistically unreliable on sparse cross-sections
- **Exception logging**: All `except: pass` patterns replaced with typed exception logging via `console.detail()` — silent failures are now surfaced in the terminal

#### Display & Data Quality
- **Run stats header**: Results header now shows total universe size, symbols fetched, analyzed, and failed — provides context that was previously invisible
- **Column display rename**: Result table columns use human-readable names (`Priority_Long_pct` → `Long Priority %ile`, `F1_PriceMom` → `Price Momentum`, etc.)
- **Widget state persistence**: All sidebar widgets keyed with `sb_*` session-state keys — widget selections persist across reruns without unexpected resets
- **Timeframe-aware passport**: Calibration profile keys now include timeframe — daily and weekly calibrations are stored and loaded independently under the same universe

#### UI Smoothness (No Visual Changes)
- **Skeleton shimmer suppressed**: `[data-testid="stSkeleton"] { display: none !important }` eliminates the native Streamlit shimmer that briefly appears between a button click and the first progress bar render
- **CSS loading cached**: `@st.cache_resource` on `_load_theme_css()` — the 4,300-line `theme.css` is read from disk once per process; subsequent reruns pay zero I/O cost
- **Equal-height metric cards**: `:has(.metric-card)` CSS block extends the flex chain through `element-container` and `stMarkdownContainer` — metric card rows are now equal height on all pages (Single Date, Pulse Narrative, Calibration Diagnostics, Historical Range, Intelligence, Correlation) regardless of content length or screen size

---

## [v3.1.0] · 2026-05-07
### Documentation & Version Unification

**"Uniform Signal"**

Version alignment pass bringing all system components — main application, UI theme, logger, and Pine Script indicator — to a single canonical version string. Accompanied by a full documentation rewrite.

#### Changes
- **Version unification**: `sanket.py`, `ui/theme.py`, and `logger.py` all bumped to `v3.1.0`, matching the existing `wrci.pine` indicator version
- **README rewrite**: Complete documentation overhaul — architecture deep-dive, engine internals, factor math, signal hierarchy, intelligence calibration workflow, deployment guide, profile structure reference
- **CHANGELOG rewrite**: Full version history reconstructed from v2.0.0 to present with accurate release notes
- **LICENSE update**: Product version updated to `v3.1.0`

---

## [v3.0.0] · 2026-05-06
### Production Calibration Release

**"The Asymmetric Engine"**

First production-grade release of the asymmetric Priority Engine with Bayesian self-tuning. Separated long and short factor betas, introduced per-universe profile persistence, and shipped the Intelligence Center UI.

#### Features
- **Asymmetric Priority Engine**: Separate `beta_*_long` and `beta_*_short` weights for all six factors — the system no longer assumes long/short symmetry
- **Intelligence Center**: Full Optuna TPE calibration workflow surfaced in UI — trial count control, live progress, val IC display, parameter importance chart
- **Model Passport**: Sidebar panel for viewing, exporting, importing, and deleting calibration profiles per universe
- **fANOVA Sensitivity Analysis**: Post-calibration parameter importance ranking using Optuna's fANOVA implementation
- **Per-Universe Profile Auto-Loading**: `load_profile_for()` auto-selects the matching profile when the user switches universe
- **F&O Sample Profile**: Seed calibration (`profiles/fno.json`) with pre-optimized weights for NSE F&O universe (val IC: 0.1556)
- **Legacy Profile Migration**: `_maybe_migrate_legacy_profile()` handles v1 → v2 key format upgrade

#### Internals
- **`_PrecomputedDataset`**: Pre-computes weight-invariant arrays before Optuna trials begin — ~50× speedup over per-trial recalculation
- **L2 regularization**: Added to `_evaluate_ic()` to prevent weight inflation in low-signal regimes
- **Change-point penalty**: Regime shift detection now applies a damping multiplier at structural breaks

---

## [v2.2.0] · 2026-05-05
### Obsidian Quant Transformation & Mathematical Parity

**"The Obsidian Quant Transformation"**

Final synchronization pass between the Python screener and the TradingView indicator. Achieved 1:1 mathematical parity across all calculations. Applied the Obsidian Quant design system terminal-wide.

#### Features
- **1:1 Mathematical Parity**: Unified HMA, WMA, and Linear Regression endpoint calculations across `sanket.py` and `wrci.pine` — screener signals now match chart signals exactly
- **Pulse Engine (v3)**: Implemented abnormal acceleration detection — 3-bar velocity modulated by 20-bar volatility Z-Score, with volume factor (`tanh(volZ/2)`) and price-action factor
- **Obsidian Quant UI**: Applied the full Obsidian design language — `#1a1a1a` background, JetBrains Mono data font, Syne display font, glass-morphism metric cards, staggered entrance animations
- **Pulse Narrative Matrix**: 4×4 state grid (SURGE/FIRM/SOFT/CRUSH × LEAD/DEEP/LIGHT/HOLLOW) for per-bar signal interpretation
- **Global Macro ETFs**: Expanded asset coverage to include global bond ETFs and treasury instruments
- **Fractal MTF Anchoring**: Integrated daily and weekly macro-regime context for tactical signal filtering

#### Fixes & Optimizations
- **VWAP Accuracy**: Refactored to ratio-of-sums instead of discrete averaging — eliminates accumulation drift
- **Signal Gating**: Hardened mutual exclusivity rules for Sets A–D — no double-firing on same bar
- **Anti-Clustering**: Enhanced pattern matching with anti-clustering logic to prevent redundant analog matches
- **Colorama Compatibility**: Fixed ANSI color rendering on Windows 10+ terminals via explicit `colorama.init()`

---

## [v2.1.0] · 2026-04-30
### WRCI Foundation

**"Wave-Regime Composite Index Core"**

Initial release of the WRCI engine and its companion Pine Script indicator. Established the WaveTrend core, Conviction engine, and signal hierarchy framework.

#### Features
- **WRCI Core Engine**: WaveTrend oscillator (WT1/WT2) with normalized HLC3 applied price
- **Conviction Engine**: Three-component composite score — Trend Strength (HMA slope / ATR), Momentum Quality (WT separation), Participation (Volume Z-Score × price direction)
- **Signal Sets A–D**: Defined non-redundant signal classification framework — Momentum, Contrarian, Threshold, Squeeze
- **Squeeze Engine**: Bollinger Band / Keltner Channel compression detection for volatility breakout identification
- **Pine Script v6 Indicator**: First release of `wrci.pine` companion TradingView indicator
- **Analog Pattern Matcher**: Cosine similarity-based historical pattern matching engine (first iteration)
- **Overbought/Oversold Logic**: Zone definitions at `±60` (Threshold) and `±80` (Extreme)

---

## [v2.0.0] · 2026-04-15
### Pragyam Family Rebirth

**"Modular Architecture & Web Terminal"**

Architectural rebuild from standalone script to the Pragyam modular product framework. Launched the Streamlit web terminal.

#### Changes
- **Pragyam Architecture**: Decomposed monolithic script into `sanket.py`, `priority_engine.py`, `intelligence.py`, `logger.py`, and `ui/` module
- **Streamlit Terminal**: Replaced CLI with interactive Streamlit web UI — session state management, sidebar routing, mode switching
- **Multi-Universe Scraping**: Introduced constituent scraping for NSE F&O (nsepython), NASDAQ, and S&P 500 (Wikipedia)
- **Obsidian Quant Design System**: Initial `theme.css` and `components.py` — dark terminal aesthetic with Plotly chart theming
- **Structured Logging**: Replaced `print()` statements with `ConsoleOutput` class — ANSI-colored, phase-timed, run-ID-tagged terminal output

---

*Full technical specifications: [README.md](README.md)*
*Author: [@thebullishvalue](https://github.com/thebullishvalue) · Pragyam Family*
