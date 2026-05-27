# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [v2.6.0] — 2026-04-06

### Google Sheets Infrastructure Simplification

Migrated from Google service account OAuth to the Google Visualization API (`gviz/tq`) with environment variable configuration. No changes to the sentiment engine, math primitives, or UI behavior.

#### Changed
- **Data ingestion endpoint** — switched from `/export?format=csv` with OAuth service account to `/gviz/tq?tqx=out:csv` with no authentication required
- **Configuration model** — replaced `st.secrets` TOML-based secrets with two environment variables: `ARTHAGATI_SHEET_ID` and `ARTHAGATI_SHEET_GID`
- **Timeout resilience** — increased request timeout from 30s to 60s with 3-attempt exponential backoff (2s, 4s, 8s)
- **Deployment simplicity** — no Google Cloud project, no service account JSON, no OAuth scopes needed

#### Removed
- `google-auth` dependency from `requirements.txt` — no longer needed for gviz endpoint
- `_SHEET_SCOPES` constant and OAuth import chain (`google.auth.transport.requests`, `google.oauth2.service_account`)
- Service account credential resolution logic from `_fetch_sheet_csv()`
- `.streamlit/secrets.toml` deployment pattern (replaced by environment variables)

#### Fixed
- Stale progress bar text: "service account auth" → "gviz API"
- VISION.md data ingestion diagram and Q&A section updated to reflect gviz architecture

---

## [v2.5.0] — 2026-04-05

### Production Readiness & Code Cleanup

Production-focused release. Dead code elimination, API surface reduction, and cross-file version synchronization. No behavioral changes to the sentiment engine or UI.

#### Removed
- Dead function `ornstein_uhlenbeck_estimate()` (42 lines) — zero traceable callsites; OU estimation is performed inline via vectorized expanding AR(1) within `calculate_historical_mood()`
- Unused `kalman_gains` return value from `kalman_filter_1d()` — only `filtered_state` and `estimate_variances` are consumed by the smoothing layer
- Stale `ornstein_uhlenbeck_estimate` entry from mathematical primitives documentation table

#### Changed
- `kalman_filter_1d()` signature modernized with PEP 604 type hints: `np.ndarray | pd.Series`, `float | None`, returns `tuple[np.ndarray, np.ndarray]`
- Mathematical primitives count updated: 12 → 11 functions across source code and documentation
- `COMPANY` constant in `arthagati.py` updated to `@thebullishvalue` (branding alignment)
- Version numbers synchronized across all files: `arthagati.py`, `README.md`, `requirements.txt`, `VISION.md` (VISION.md had lagged at v2.2.1 since v2.3.0)

#### Fixed
- Cross-file version consistency: all version identifiers now point to a single source of truth (`VERSION` in `arthagati.py`)

---

## [v2.4.0]

### Adversarial Audit Resolution

Major correctness release. Seven mathematical fixes and nine algorithmic improvements identified through adversarial audit. The sentiment engine now produces mathematically sound scores with no look-ahead bias, correct variance estimation, and stable regime detection.

#### Fixed — Correctness
- **OU Residual Sum of Squares** — Replaced incorrect algebraic expanding RSS formula with per-observation residuals `e²_i = (y_i − a_i − b_i·x_i)²` accumulated via expanding mean; sigma and half-life diagnostics are now correct under time-varying AR(1) coefficients
- **Backward Information Leakage** — Removed `bfill()` from data imputation; only `ffill()` applied, early NaN values remain NaN and are handled by `np.isfinite()` guards in all math primitives
- **DFA Segment Guard** — Increased minimum segment count from 1 to 4 per Peng et al. (1994), preventing degenerate single-segment Hurst estimates
- **MSF Regime Trend Artifact** — Replaced unbounded `cumsum()` with windowed `rolling(MSF_WINDOW).sum()` preventing directional count drift that created false regime signals
- **Rolling Entropy Off-by-One** — Fixed `sliding_window_view` scope and result index alignment
- **Sigmoid Overflow** — Added input clipping (`±500`) before `np.exp()` for extreme z-scores
- **rolling_mean_fast NaN Semantics** — Returns `NaN` instead of `0.0` for all-NaN windows

#### Changed — Algorithm Improvements
- **O(N log N) Adaptive Percentiles** — Replaced O(N²) inner loop with sorted-insert + `np.searchsorted` binary search (Greenwald & Khanna 2001 streaming quantile approach)
- **Kalman Warm-Up Bootstrap** — First 50 observations bootstrapped from first stable window per Harvey (1990), preventing poorly calibrated Kalman gains
- **Freedman-Diaconis Entropy Bins** — Adaptive bin selection via `2·IQR·n^{-1/3}` instead of capped `sqrt(N)`
- **Ledoit-Wolf Covariance Shrinkage** — Mahalanobis distance uses analytical OAS shrinkage (Chen et al. 2010) instead of ad-hoc diagonal regularization
- **Walk-Forward Weight Blending** — Checkpoint weights exponentially blended (α ≈ 0.29, HL = 2 checkpoints) eliminating discontinuous jumps at segment boundaries
- **Confidence Band Soft-Clip** — `tanh(x/100)·100` replaces hard `np.clip(±100)` preserving band width at score extremes
- **Least-Squares Trajectory Detrend** — Replaced endpoint anchoring with least-squares linear detrend (minimizes residual variance on V-shaped and reversal trajectories)
- **Backtest Train/Test Split** — 70/30 chronological split with separate in-sample and out-of-sample Pearson/Spearman correlations

---

## [v2.3.0]

### Walk-Forward Correlations & Bias Corrections

Eliminated look-ahead bias from the correlation engine and applied first-order bias corrections to statistical estimators.

#### Fixed
- **Look-Ahead Bias** — Layers 1–2 restructured to use expanding-window walk-forward correlations at quarterly checkpoints instead of full-sample
- **Percentile Semantics** — Symmetric [−1,+1] adjustments for PE and EY anchors, fixing asymmetric bearish/bullish capacity
- **Hurst Estimator Bias** — Replaced R/S with DFA-1 (Peng et al. 1994, Weron 2002) for robustness on short series
- **OU AR(1) Bias** — Kendall-Marriott-Pope first-order correction applied to expanding AR(1) coefficient
- **Dynamic Y-Axis** — Mood chart now scales to actual data bounds with 8% padding instead of fixed ±100

---

## [v2.2.1]

### UI Rendering & Memory Optimizations

#### Changed
- **WebGL Chart Rendering** — Regime transition markers migrated from individual SVG shapes (`add_vline`) to interleaved WebGL traces (`go.Scattergl`), eliminating DOM bloat on MAX timeframe

#### Fixed
- **Cache Memory Bloat** — Applied `max_entries=5` to all heavy `@st.cache_data` decorators, capping server RAM when users rapidly toggle predictor configurations

---

## [v2.2.0]

### Performance & Vectorization Architecture Rewrite

Execution time reduced by 99%+ through C-level NumPy vectorization of all mathematical primitives.

#### Added
- **C-Level Vectorization Engine** — All explicit Python loops replaced with NumPy `cumsum`, `sliding_window_view`, and array striding
- **O(N) Moving Averages & Variances** — Replaced Pandas `.rolling()`/`.expanding()` with exact NumPy cumulative sums
- **Pure-NumPy Ranking** — Custom vectorized tie-averaging rank algorithm replacing Pandas `.rank()` in weighted Spearman

#### Changed
- **Kalman Filter** — Exponential fading memory factor (Sorenson & Sacks) for non-stationary regime discounting
- **OU Estimation** — O(N²) expanding-window loop converted to single-pass O(N) vectorized algorithm
- **Trajectory Similarity** — 20-day cosine similarity migrated from explicit iteration to matrix striding multiplications
- **Regime Detection** — Fully vectorized Hurst × Entropy quadrant classification

#### Fixed
- **Memory Blowout** — 2D NumPy broadcasting in adaptive percentiles created O(N²) memory (40GB+ allocations); rewritten with O(N) 1D slice lookback reducing engine time from ~120s to <2s

---

## [v2.1.0]

### Diagnostics & Forward Returns

Extended the sentiment engine with forward-looking projections and historical validation.

#### Added
- 90-day OU forward mean-reversion projection on mood chart
- ±1.96σ Kalman confidence bands around smoothed mood score
- Forward return outcomes (30/60/90-day) on similar historical period cards
- Backtest scatter plot: mood score at T vs NIFTY return at T+30
- Data staleness warnings when Google Sheet is more than 3 days old

---

## [v2.0.0]

### Physics-Informed Mathematics

Complete overhaul of the sentiment engine from static correlations to stochastic process modeling.

#### Added
- **Ornstein-Uhlenbeck Normalization** — Mood modeled as mean-reverting diffusion `dx = θ(μ − x)dt + σdW` instead of global z-score
- **Kalman Smoothing** — 1D adaptive state estimation replacing fixed-window EMA
- **Mahalanobis Distance** — Covariance-aware historical period matching replacing Manhattan distance
- **Inverse-Variance MSF Weighting** — Markowitz minimum-variance signal allocation replacing fixed 30/25/25/20 weights
- **Adaptive Percentiles** — Decay-weighted empirical CDF replacing expanding rank percentiles
- **Decay-Spearman Correlations** — Recency-weighted rank correlation replacing full-sample Pearson
- **Shannon Entropy Weighting** — Noisy variable suppression via information-theoretic penalty
- **Predictor Quality Assessment** — Ranked variable scoring by |correlation| × (1 − entropy)
- **Staging → Commit Config** — Apply-button pattern preventing continuous recomputation
- **EY Auto-Derivation** — `1/PE × 100` when Earnings Yield absent from sheet
- **Yield Term Spreads** — `IN10Y − IN02Y` and `US10Y − US02Y` derived as orthogonal predictors

#### Removed
- Fixed Pearson correlations (replaced by decay-Spearman)
- Expanding rank percentiles (replaced by adaptive ECDF)
- Fixed MSF weights (replaced by inverse-variance)
- Manhattan distance similar periods (replaced by Mahalanobis)
- Global z-score normalization (replaced by OU)
- Simple moving average smoothing (replaced by Kalman)

---

## [v1.2.0]

### Initial Release

Baseline sentiment engine with Pearson correlations, expanding percentiles, and fixed-weight MSF oscillator.

---

*© 2026 Arthagati · @thebullishvalue*
