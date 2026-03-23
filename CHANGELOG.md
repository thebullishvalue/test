# Changelog

All notable changes to PRAGYAM will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [3.5.0] - 2026-03-23

### Fixed
- **[C1] Sortino ratio formula** — all 3 implementations (`calculate_advanced_metrics`, `calculate_trigger_based_metrics`, `_compute_backtest_metrics`) now use canonical RMS of downside returns: `sqrt(mean(min(r,0)²))` instead of incorrect `std(downside)`
- **[C2] SIP Time-Weighted Return** — TWR now isolates market return from capital injection by computing `(V_t - CF_t - V_{t-1}) / V_{t-1}` per period, eliminating cash-flow leakage into performance
- **[C3] Tier-level Sharpe ratios** — replaced fabricated linear decay formula (`sharpe * (1 - t * 0.05)`) with actual per-tier Sharpe computed from real tier-level returns across all buy days
- **[C4] Spectral analysis matrix** — now builds T×N return time-series matrix from per-strategy OOS returns instead of single-day cross-sectional indicator matrix; applies to both trigger-based and standard walk-forward
- **[C5] Trigger thresholds configurable** — thresholds now overridable via `PRAGYAM_SIP_TRIGGER`, `PRAGYAM_SWING_BUY`, `PRAGYAM_SWING_SELL` env vars; added `compute_adaptive_thresholds()` for percentile-based calibration
- **[H1] Softmax temperature** — introduced `SOFTMAX_TEMPERATURE = 5.0` parameter and removed the `+2` additive shift that killed weight differentiation in `calculate_strategy_weights`
- **[H2] Missing symbol handling** — `compute_portfolio_return` now uses `how='left'` merge; missing/halted symbols get 0 return instead of being silently dropped (redistributing weight)
- **[H4] Non-annualized Sharpe/Sortino** in `strategy_selection.py` `MasterPortfolio.get_metrics()` — now annualized by `sqrt(N)` factor; Sortino uses RMS downside
- **[H5] Ledoit-Wolf estimator** — replaced broken formula (dead variables `mu`, `delta`) with Oracle Approximating Shrinkage (OAS) estimator per Chen, Wiesel, Eldar & Hero (2010)
- **[H6] Zero returns on held days** — non-trigger days now compute actual returns from last held portfolio instead of reporting 0; portfolio prices updated daily for correct chaining
- **[M1] Correlation regime weight** — enabled from 0.0 → 0.10 in composite regime scoring; redistributed from momentum (0.30→0.25) and velocity (0.15→0.10)
- **[M2] Order-dependent clustering** — replaced greedy sequential clustering with Union-Find algorithm for order-independent correlation clustering in `rmt_core.py`
- **[M3] NSE holiday resampling** — `resample_data` now filters out partial weeks (< 2 trading days) to avoid noisy single-day bars from Friday closures
- **[M4] Robust trend estimation** — replaced `np.polyfit` with Theil-Sen slopes in `_analyze_momentum_regime` and `_analyze_trend_quality` for outlier resilience
- **[M5] Division-by-zero guards** — standardized all epsilon constants to canonical `_EPS = 1e-10` in `rmt_core.py`

### Added
- **[A1] Canonical `compute_risk_metrics()`** in `backtest_engine.py` — single source of truth for Sharpe, Sortino, Calmar, max drawdown, win rate, volatility, and CAGR; all 4 duplicate implementations now delegate to it
- **[A2] Strategy interface contracts** — `BaseStrategy.__init_subclass__` auto-wraps `generate_portfolio()` with `_validate_portfolio()` runtime validation (column presence, non-negative price/value, duplicate symbol removal); `PORTFOLIO_COLUMNS` tuple defines the contract
- **Transaction cost model** — `TRANSACTION_COST_BPS = 20` (NSE round-trip: brokerage + STT + GST + stamp duty); applied as one-way cost on rebalance days in all walk-forward loops
- `compute_adaptive_thresholds()` in `strategy_selection.py` — percentile-based threshold calibration from historical breadth distribution
- `scipy.stats.theilslopes` import for robust trend estimation

### Changed
- **[A1]** `calculate_advanced_metrics()`, `calculate_trigger_based_metrics()`, `_compute_backtest_metrics()` in `app.py` and `MasterPortfolio.get_metrics()` in `strategy_selection.py` all delegate core ratio math to `compute_risk_metrics()`; higher-order metrics (Kelly, Omega, tail ratio) remain in `calculate_advanced_metrics()` only
- `compute_portfolio_return` signature extended with `is_rebalance` parameter for transaction cost deduction
- Walk-forward loops track `last_curated_port` and `last_strategy_ports` state for held-position return computation
- Regime scoring weights rebalanced: momentum 0.25, trend 0.25, breadth 0.15, velocity 0.10, correlation 0.10, extremes 0.10, volatility 0.05

---

## [3.4.0] - 2026-03-23

### Added
- **Charts v2.0 visual redesign** — complete rewrite of all 15 chart functions in `charts.py` with institutional-grade aesthetics
  - `_axis()` helper for consistent axis configuration across all charts
  - `_hex_to_rgba()` utility for alpha-channel color conversions
  - Design tokens: `_GRID`, `_ZERO`, `_TICK`, `_LABEL`, `_FONT` for single-source styling
  - Zone shading bands in rolling metrics (replaces reference lines)
  - Quantile-based y-range clipping for outlier resilience
  - Stacked area charts via `stackgroup` with desaturated palette
  - Text labels on risk-return scatter with 12-char truncation
  - CML (Capital Market Line) overlay on scatter plots
- **Tab rendering architecture** — from-scratch rendering functions for tabs 3–5
  - `_render_risk_intelligence()` — correlation heatmaps, spectral analysis
  - `_render_strategy_analysis()` — scatter, radar, tier heatmap, conviction heatmap
  - `_render_backtest_data()` — walk-forward data and metrics

### Changed
- `charts.py` version bumped to v2.0 (Hemrek Capital Design System v2.0)
- COLORS palette desaturated for cleaner stacking: gold, emerald, cyan, amber, violet, rose, lime, orange
- Equity-drawdown chart: 74/26 row split, gold line 1.8px, fill alpha 0.07, HWM at 15% white opacity
- Risk-return scatter: markers+text mode, opacity 0.95, white border 2px, diamond optimal marker
- Factor radar: fill alpha 0.20, line width 2, angular tick 11pt, radial tick 9pt
- Weight evolution: stacked area with 0.65 alpha fills, 0.3px borders, bottom legend
- Heatmaps: `xgap`/`ygap` breathing room, adaptive colorscales (positive-only vs mixed)
- All chart backgrounds now transparent (CSS `.stPlotlyChart` card handles container)

### Removed
- `UNIFIED_CHARTS_AVAILABLE` dual-path chart system — eliminated all 6 branch points and ~150 lines of inline fallback chart code from `app.py`
- `_chart_layout_base()` from `app.py` — replaced by `get_chart_layout()` from `charts.py`
- `plotly.express` and `plotly.subplots` imports from `app.py` (no longer needed)
- ~500 lines of inline tab content from `main()` (replaced by dedicated rendering functions)

---

## [3.3.0] - 2026-03-22

### Added
- **Random Matrix Theory Engine** (`rmt_core.py`) — standalone spectral analysis module
  - Marchenko-Pastur distribution (PDF, edge computation, noise threshold)
  - Eigenvalue-based correlation matrix denoising (clipping + trace preservation)
  - Ledoit-Wolf shrinkage estimator (oracle-approximating)
  - Spectral diagnostics: absorption ratio, effective rank (ENB), Herfindahl eigenvalue index, condition number
  - Strategy redundancy detection via cleaned correlations
  - Greedy diversified selection algorithm (spectral independence filter)
  - Minimum-variance and risk-parity portfolio optimization using cleaned covariance
  - Rolling spectral analysis and spectral turnover metrics
  - Diversification ratio computation: DR = (weighted avg vol) / (portfolio vol)
- **Spectral Analysis UI section** in app.py — eigenvalue histogram with MP overlay, cleaned vs raw correlation comparison, rolling absorption ratio chart, factor loading heatmap, spectral risk dashboard (2x2 panel)
- **5 new visualization functions** in charts.py — `create_eigenvalue_histogram`, `create_cleaned_vs_raw_correlation`, `create_absorption_ratio_chart`, `create_factor_loading_heatmap`, `create_spectral_risk_dashboard`
- Walk-forward spectral tracking — absorption ratio, effective rank, condition number, and largest eigenvalue recorded every 5th step
- Spectral summary metrics in walk-forward results (mean AR, AR volatility, mean effective rank)
- Cross-strategy spectral metrics computed at end of each backtest run (effective strategy count, noise fraction, strategy clusters, signal eigenvalues)

### Changed
- `PrincipalComponentStrategy` now computes real eigenvectors via eigendecomposition instead of hardcoded PC loadings (falls back to hardcoded when cross-section too small)
- `select_top_strategies()` in backtest_engine.py applies RMT redundancy filter — selects spectrally independent strategies via greedy diversified selection (cleaned correlation < 0.7 threshold)
- `calculate_strategy_weights()` in app.py supports `method` parameter: `'softmax_sharpe'` (default, backward compatible), `'rmt_min_variance'`, `'rmt_risk_parity'`, `'equal'`
- `_analyze_correlation_regime()` in MarketRegimeDetectorV2 uses absorption ratio from eigendecomposition instead of indicator-agreement heuristic; original heuristic preserved as `_fallback_correlation_regime()`
- `curate_final_portfolio()` computes diversification ratio using RMT-cleaned covariance after portfolio assembly
- Added `rmt_core` to `known-first-party` in pyproject.toml Ruff isort config
- charts.py version bumped to 1.1.0

### Design Notes
- All RMT integrations use `try/except` for graceful degradation — system runs identically without rmt_core.py
- No new dependencies — uses only numpy and scipy (already in requirements)
- With T/N = 50/95 ~ 0.53, MP upper edge lambda+ ~ 3.88; any eigenvalue below this is indistinguishable from noise

---

## [3.2.0] - 2026-03-16

### Added
- Strategy registry with auto-discovery (`STRATEGY_REGISTRY`, `discover_strategies()` in strategies.py)
- `style.css` — extracted ~360 lines of inline CSS into standalone Hemrek Capital Design System file
- `pyproject.toml` with Ruff linter/formatter and mypy configuration
- `__all__` exports to all modules (strategies.py, backtest_engine.py, backdata.py; fixed stale entry in charts.py)
- `.gitignore` for Python, IDE, OS, and application artifacts

### Changed
- Named loggers across all modules (replaced bare `logging.info/error/warning`)
- Removed `logging.basicConfig` calls that overrode root logger config
- Migrated deprecated Streamlit `use_container_width` → `width='stretch'` across all modules
- Replaced redundant `elif not is_buy_day` with `else` in trigger backtest
- Fixed Sortino ratio formula in `backtest_engine.py` to use proper RMS of downside deviations
- Fixed week numbering collision across year boundaries in SIP backtest
- Widened return clipping from ±50% to ±100% in `PerformanceMetrics`
- Fixed rolling downside calculation in charts.py (was modifying data in-place)
- Removed magic-number metric fallbacks in `strategy_selection.py` (Sortino `*1.5`, Calmar `*10`)
- Eliminated double `calculate_advanced_metrics()` call in `_calculate_performance_on_window`
- Vectorized row-wise `.apply()` calls in CL2Strategy and CL3Strategy with `np.select`/`np.where`
- Replaced 97-line manual strategy instantiation block with `discover_strategies()` auto-registry
- Replaced 25-line manual strategy import block with single `from strategies import discover_strategies`

### Removed
- Dead code: `fix_csv_export` in strategies.py, `get_axis_config` in charts.py
- Unused imports: `time`, `ABC`/`abstractmethod`, `scipy.stats`, `StandardScaler`, `io` (from strategies.py)
- Stale `pragati.py` comment in strategies.py
- `matplotlib` and `openpyxl` from requirements.txt (not imported anywhere)
- `time.sleep(0.5)` UI delay in app.py
- Inline CSS block from app.py (~360 lines, moved to style.css)

### Fixed
- Bare `except:` clause → `except Exception:` in app.py conviction analysis
- `from datetime import timezone` moved from function body to top-level imports
- References to non-existent `pragati.py` and `advanced_strategy_selector.py` in README
- Version mismatch: README badge, requirements.txt header, and CHANGELOG now all say v3.2.0
- Stale `get_axis_config` entry in charts.py `__all__` (function was removed)

---

## [3.1.0] - 2026-02-04

### Added
- **Strategy Selection Framework** (`strategy_selection.py`)
  - Fetches REL_BREADTH data from Google Sheets (400 rows lookback)
  - SIP Mode: Accumulates portfolio on every trigger (REL_BREADTH < 0.42)
  - Swing Mode: Buy-sell cycles (Buy < 0.42, Sell >= 0.50)
  - MasterPortfolio class tracks cumulative holdings across SIP entries
  - Dispersion-weighted ranking (no fixed formula weights)
- Chart annotations restored ("Growth of ₹1 Investment", "Underwater Curve")

### Changed
- Performance tab redesigned with clean metric layout using st.metric()
- Strategy Deep Dive tab simplified with minimal headers
- Selection scoring uses rank-based adaptive weights
- Equity chart y-axis starts from sensible minimum (not zero)

### Removed
- Fixed threshold selection formulas (0.30×Sharpe + 0.25×Sortino...)
- Verbose section headers and info-boxes

### Fixed
- Plotly `titlefont` deprecation error
- `use_container_width` deprecation warnings
- Equity curve appearing flat due to y-axis starting at zero

---

## [3.0.0] - 2026-01-30

### Added
- Advanced Strategy Selector with TOPSIS multi-criteria optimization
- Bayesian shrinkage estimation for small sample periods
- Risk parity portfolio construction with SLSQP optimization
- Hidden Markov Model (HMM) regime detection integration
- Bootstrap confidence intervals for Sharpe ratio
- Lo (2002) standard error adjustment for serial correlation
- Pareto frontier identification for strategy selection
- Maximum diversification selection algorithm
- Strategy clustering via hierarchical clustering

### Changed
- Complete rewrite of strategy selection logic
- Enhanced backtest engine with walk-forward validation
- Improved regime detection with multi-factor approach

---

## [2.5.0] - 2025-12-15

### Added
- 80+ quantitative strategies implementation
- Tier-based position sizing system
- Regime-aware allocation framework
- Strategy correlation monitoring

### Changed
- Migrated to Streamlit 1.28+ API
- Improved data caching mechanisms
- Enhanced error handling

---

## [2.0.0] - 2025-10-01

### Added
- Walk-forward backtesting engine
- Multi-strategy portfolio construction
- Performance attribution analytics
- Interactive visualizations with Plotly

### Changed
- Complete architecture redesign
- Modular strategy framework

---

## [1.0.0] - 2025-07-01

### Added
- Initial release
- Basic momentum strategies
- Simple backtesting framework
- Streamlit dashboard

---

## Legend

- **Added**: New features
- **Changed**: Changes in existing functionality
- **Deprecated**: Features to be removed in future versions
- **Removed**: Features removed in this version
- **Fixed**: Bug fixes
- **Security**: Vulnerability fixes
