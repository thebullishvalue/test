# PRAGYAM (प्रज्ञम्)

<div align="center">

![Version](https://img.shields.io/badge/version-3.6.0-gold)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-Proprietary-red)
![Status](https://img.shields.io/badge/status-Production-green)

**Institutional-Grade Portfolio Intelligence System**

*Walk-forward portfolio curation with regime-aware strategy allocation*

[Features](#features) • [Installation](#installation) • [Quick Start](#quick-start) • [Documentation](#documentation) • [Architecture](#architecture)

---

<img src="https://img.shields.io/badge/Hemrek_Capital-FFC300?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0iIzBGMEYwRiIgZD0iTTEyIDJMMiA3bDEwIDUgMTAtNS0xMC01ek0yIDE3bDEwIDUgMTAtNS0xMC01LTEwIDV6Ii8+PC9zdmc+" alt="Hemrek Capital"/>

</div>

---

## Overview

**Pragyam** (Sanskrit: प्रज्ञम् - "Wisdom/Intelligence") is a hedge fund-grade portfolio intelligence platform designed for systematic equity investing in Indian markets. It combines 96 quantitative strategies with RMT-based spectral analysis, regime detection, and dynamic allocation to deliver institutional-quality portfolio construction.

### Key Differentiators

| Feature | Description |
|---------|-------------|
| **Multi-Strategy Engine** | 96 unique alpha-generating strategies spanning momentum, mean-reversion, volatility, and factor-based approaches |
| **Spectral Signal-Noise Separation** | Random Matrix Theory (Marchenko-Pastur) identifies which correlations are real signal vs statistical noise — the system knows its own noise floor |
| **Strategy Factor Decomposition** | RMT-based dimensionality reduction projects 60+ correlated strategies onto their true independent factors above the Marchenko-Pastur threshold |
| **Hierarchical Risk Parity** | Default allocation via Lopez de Prado (2016) dendrogram-based HRP — avoids covariance inversion, robust to estimation error |
| **Regime-Aware Allocation** | Real-time market regime detection adjusts strategy weights based on momentum, trend, breadth, volatility, and spectral correlation structure |
| **SPRT Regime Triggers** | Optional Sequential Probability Ratio Test (Wald, 1945) for evidence-accumulating regime change detection — replaces fixed-threshold triggers |
| **Conformal Prediction Intervals** | Distribution-free 90% coverage guarantee on next-period strategy returns (Vovk et al., 2005) |
| **Hedge Fund Analytics** | Institutional metrics including Sharpe, Sortino, Calmar, Omega, CVaR, absorption ratio, effective rank, and diversification ratio |
| **Tier-Based Construction** | Position sizing via conviction tiers with turnover-proportional transaction cost modeling |

---

## Features

### 📊 Portfolio Intelligence
- **Walk-Forward Backtesting**: Out-of-sample validation with embargo gaps, turnover-proportional transaction costs, and Modified Dietz TWR
- **Strategy Factor Decomposition**: RMT projects 60+ strategies onto true independent factors — reveals how many bets you actually have
- **Conformal Prediction Intervals**: Distribution-free 90% coverage bounds on next-period returns (no normality assumption)
- **Strategy Attribution**: Decompose returns by strategy, tier, and time period
- **Correlation Analysis**: Inter-strategy correlation monitoring with RMT-cleaned matrices
- **Weight Evolution**: Track how strategy allocations change through market regimes
- **Spectral Analysis**: Eigenvalue distribution vs Marchenko-Pastur overlay, absorption ratio tracking, factor loading decomposition

### 🎯 Strategy Universe
- **Momentum Strategies**: RSI, MACD, Rate of Change, Acceleration
- **Mean-Reversion**: Bollinger Bands, Z-Score, Kalman Filter
- **Volatility**: ATR-based, Regime-switching, Breakout detection
- **Factor-Based**: Value-momentum blends, Quality scores, Size tilts
- **Advanced**: HMM-based, Wavelet denoising, Copula blending

### 📈 Risk Analytics
- **Core Metrics**: CAGR, Volatility, Maximum Drawdown, Win Rate
- **Risk-Adjusted**: Sharpe, Sortino, Calmar, Omega, Information Ratio
- **Tail Risk**: VaR (95/99), CVaR, Expected Shortfall, Tail Ratio
- **Statistical**: Bootstrap CI, Lo(2002) Sharpe SE, Jarque-Bera normality

### 🔄 Regime Detection
- **Momentum Regime**: RSI breadth, momentum persistence scoring
- **Trend Regime**: 200-DMA positioning, trend quality metrics (Theil-Sen robust estimation)
- **Volatility Regime**: Bollinger Band Width (with near-zero guard), ATR percentile ranking
- **Breadth Regime**: Advance-decline analysis, sector rotation signals (separated from momentum)
- **Correlation Regime**: Spectral absorption ratio from eigendecomposition — detects systemic herding vs healthy dispersion
- **SPRT Triggers**: Optional evidence-accumulating regime detection via Sequential Probability Ratio Test (Wald, 1945)

---

## Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/hemrek/pragyam.git
cd pragyam

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
yfinance>=0.2.28
scipy>=1.11.0
scikit-learn>=1.3.0
```

---

## Quick Start

### Running Locally

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

### Streamlit Cloud Deployment

1. Push to GitHub repository
2. Connect to [Streamlit Cloud](https://share.streamlit.io)
3. Deploy with `app.py` as the main file

### Basic Usage

1. **Select Analysis Date**: Choose the portfolio construction date
2. **Choose Mode**: SIP (accumulation) or Swing (trading)
3. **Set Lookback**: Historical period for strategy evaluation
4. **Run Analysis**: Generate curated portfolio with full analytics

---

## Architecture

```
pragyam/
├── app.py                      # Main Streamlit application & portfolio curation
├── rmt_core.py                 # Random Matrix Theory spectral engine (Marchenko-Pastur)
├── strategies.py               # 96 strategy implementations + auto-discovery registry
├── charts.py                   # Unified Plotly visualization components (incl. spectral charts)
├── strategy_selection.py       # Trigger-based strategy evaluation (REL_BREADTH)
├── backtest_engine.py          # Walk-forward backtesting framework
├── backdata.py                 # Data fetching & indicator computation
├── style.css                   # Hemrek Capital Design System (dark theme)
├── symbols.txt                 # Universe of tradeable symbols (NSE tickers)
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Ruff, mypy, and package configuration
├── CHANGELOG.md                # Version history
└── docs/
    ├── STRATEGY_GUIDE.md       # Strategy documentation
    ├── MATHEMATICAL_FRAMEWORK.md  # Quantitative methods (incl. RMT)
    └── PROCESS_ARCHITECTURE.md # System architecture details
```

### Data Flow

```
┌───────────────────────────────────────────────────────────────────────────┐
│                          PRAGYAM DATA FLOW                                │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐            │
│  │  Yahoo   │───▶│ Indicator│───▶│  Regime  │───▶│ Strategy │            │
│  │ Finance  │    │  Engine  │    │ Detector │    │ Universe │            │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘            │
│                                  ▲    ▲               │                   │
│                           SPRT ──┘    │               ▼                   │
│                         Triggers      │         ┌──────────┐              │
│                                       │         │  Factor  │              │
│                                       │         │Reduction │              │
│                                       │         │  (RMT)   │              │
│                                       │         └────┬─────┘              │
│                                       │              ▼                    │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐            │
│  │Dashboard │◀───│ Backtest │◀───│ Portfolio│◀───│Redundancy│            │
│  │Conformal │    │  Engine  │    │  Builder │    │  Filter  │            │
│  │Intervals │    │(Embargo) │    │  (HRP)   │    │  (RMT)   │            │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘            │
│       ▲                                               ▲                   │
│       │          ┌──────────────────────────┐         │                   │
│       └──────────│  RMT SPECTRAL ENGINE     │─────────┘                   │
│                  │  Marchenko-Pastur,       │                             │
│                  │  Eigenvalue denoising,   │                             │
│                  │  HRP, Conformal PI,      │                             │
│                  │  Factor decomposition    │                             │
│                  └──────────────────────────┘                             │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## Configuration

### Symbol Universe

Edit `symbols.txt` to customize the stock universe:

```
RELIANCE.NS
TCS.NS
HDFCBANK.NS
INFY.NS
...
```

### Trigger Configuration

Adjust trigger thresholds for buy/sell signals in `app.py` via `TRIGGER_CONFIG`:

```python
# strategy_selection.py exposes these defaults
SIP_TRIGGER = 0.42          # Accumulate when REL_BREADTH < threshold
SWING_BUY_TRIGGER = 0.42    # Enter when breadth drops below
SWING_SELL_TRIGGER = 0.50   # Exit when breadth rises above
```

---

## Performance Metrics

### Metric Definitions

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Sharpe Ratio** | (R - Rf) / σ | Risk-adjusted return; >1 good, >2 excellent |
| **Sortino Ratio** | (R - Rf) / σd | Downside-adjusted; ignores upside volatility |
| **Calmar Ratio** | CAGR / \|MaxDD\| | Drawdown efficiency; >1 indicates resilience |
| **Omega Ratio** | Σ(gains) / Σ(losses) | Full distribution analysis; >1 positive expectancy |
| **Tail Ratio** | P95 / \|P5\| | Skewness measure; >1 indicates positive skew |

### Statistical Tests

- **Lo (2002) Sharpe SE**: Autocorrelation-adjusted standard error
- **Bootstrap CI**: 1000-sample confidence intervals
- **Jarque-Bera**: Normality test for return distribution

---

## API Reference

### Core Components

```python
# Market regime detection
from app import MarketRegimeDetectorV2

detector = MarketRegimeDetectorV2()
regime, mix, confidence, details = detector.detect_regime(data)

# Strategy evaluation with trigger-based selection
from strategy_selection import StrategySelectionEngine

engine = StrategySelectionEngine(strategies, data, breadth_data)
results = engine.evaluate()

# Generate indicator snapshots
from backdata import generate_historical_data

snapshots = generate_historical_data(
    symbols_to_process=["RELIANCE.NS", "TCS.NS"],
    start_date=start,
    end_date=end,
)
```

### Spectral Analysis (RMT)

```python
from rmt_core import (
    compute_spectral_diagnostics,
    detect_redundant_strategies,
    rmt_minimum_variance_weights,
    rmt_risk_parity_weights,
    hrp_weights,
    reduce_strategy_space,
    conformal_strategy_intervals,
)

# Full spectral analysis of a returns matrix
diagnostics = compute_spectral_diagnostics(returns_matrix)  # T x N
print(f"Signal eigenvalues: {diagnostics.mp_dist.n_signal}")
print(f"Noise eigenvalues:  {diagnostics.mp_dist.n_noise}")
print(f"Absorption ratio:   {diagnostics.absorption_ratio:.3f}")
print(f"Effective rank:     {diagnostics.effective_rank:.1f}")

# RMT-cleaned correlation matrix (noise eigenvalues clipped)
cleaned_corr = diagnostics.cleaned_corr

# Hierarchical Risk Parity weights (default allocation method)
weights = hrp_weights(returns_matrix, strategy_names)

# Strategy dimensionality reduction
factors = reduce_strategy_space(strategy_returns_dict)
print(f"True factors: {factors['n_factors']} / {len(strategy_returns_dict)} strategies")

# Conformal prediction intervals (90% coverage)
intervals = conformal_strategy_intervals(strategy_returns_dict, alpha=0.10)
for name, (lower, point, upper) in intervals.items():
    print(f"{name}: [{lower:.4f}, {upper:.4f}]")
```

### SPRT Triggers

```python
from strategy_selection import SPRTRegimeTrigger, get_sprt_trigger_dates

# Evidence-accumulating regime detection
buy_dates, sell_dates = get_sprt_trigger_dates(breadth_df, alpha=0.05, beta=0.10)
```

---

## Support

### Troubleshooting

| Issue | Solution |
|-------|----------|
| Yahoo Finance rate limiting | Add delays between requests or use cached data |
| Memory errors on large universes | Reduce symbol count or increase system memory |
| Slow backtest execution | Enable caching with `@st.cache_data` decorators |

### Contact

- **Technical Support**: tech@hemrekcapital.com
- **Documentation**: docs.hemrekcapital.com/pragyam
- **Issues**: GitHub Issues (for licensed users)

---

## License

**Proprietary Software** - © 2024-2026 Hemrek Capital

This software is licensed exclusively to authorized users. Redistribution, modification, or commercial use without explicit written permission is prohibited.

---

## Changelog

### v3.6.0 (March 2026)
- Quantitative hardening: 12 mathematical/architectural fixes (CRITICAL-1 through MEDIUM-3)
- Hierarchical Risk Parity (Lopez de Prado, 2016) as default allocation method
- SPRT regime triggers (Wald, 1945) — evidence-accumulating alternative to fixed thresholds
- Conformal prediction intervals (Vovk et al., 2005) — distribution-free 90% coverage
- Strategy factor decomposition — RMT spectral projection reveals true independent bets
- Walk-forward embargo (Lopez de Prado, 2018) prevents indicator serial correlation leakage
- Turnover-proportional transaction costs with proper `prev_portfolio` tracking
- Continuous Kelly criterion, adaptive softmax temperature, iterative MP sigma estimation

### v3.5.0 (March 2026)
- Adversarial audit: 15 metric/formula fixes (Sortino RMS, SIP TWR, tier Sharpe, spectral matrix)
- Unified metric pipeline — single canonical `compute_risk_metrics()` replaces 4 duplicate implementations
- Strategy interface contracts — runtime portfolio validation via `BaseStrategy.__init_subclass__`
- Transaction cost model (20 bps NSE round-trip) applied to all walk-forward loops
- Held-position returns on non-trigger days (no more zero-return assumption)
- Theil-Sen robust trend estimation, configurable trigger thresholds, NSE holiday-safe resampling

### v3.4.0 (March 2026)
- Charts v2.0 visual redesign — complete rewrite of all chart functions with institutional-grade aesthetics
- Tab rendering architecture overhaul — dedicated rendering functions for Risk, Strategy, and Backtest tabs
- Eliminated dual-path chart system (`UNIFIED_CHARTS_AVAILABLE`) and ~600 lines of inline fallback code
- Transparent chart backgrounds (CSS card handles container styling)

### v3.3.0 (March 2026)
- Random Matrix Theory (Marchenko-Pastur) integration across the full pipeline
- Spectral engine (`rmt_core.py`) — eigenvalue denoising, absorption ratio, effective rank
- RMT-optimized portfolio weights (min-variance, risk-parity using cleaned covariance)
- Redundancy-aware strategy selection (spectral independence filter)
- Spectral Analysis dashboard with 5 new visualization types
- Correlation regime detection via eigendecomposition

### v3.2.0 (March 2026)
- Code quality audit and refactoring
- Fixed deprecated Streamlit APIs
- Removed dead code and unused imports
- Named loggers across all modules

### v3.1.0 (February 2026)
- Strategy selection framework with REL_BREADTH triggers
- Unified chart styling (Hemrek design system)
- Trigger-based backtesting (SIP & Swing modes)

### v3.0.0 (January 2026)
- Advanced strategy selector with TOPSIS optimization
- Bayesian shrinkage estimation
- Risk parity portfolio construction
- HMM regime detection integration

### v2.0.0 (December 2025)
- 80+ strategy implementations
- Walk-forward backtesting engine
- Regime-aware allocation system

---

<div align="center">

**Built with ❤️ by Hemrek Capital**

*"Wisdom in Every Trade"*

</div>
