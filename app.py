# -*- coding: utf-8 -*-
"""
AARAMBH (आरंभ) v2.2 — Fair Value Breadth
A Hemrek Capital Product

Walk-forward valuation analysis for market reversals.
Out-of-sample ensemble fair value modeling, OU mean-reversion physics,
and Kalman-filtered breadth conviction scoring.

Architecture:
    1. Mathematical Primitives   — OU estimation, Kalman filter, Hurst exponent
    2. FairValueEngine           — Walk-forward regression + all downstream analytics
    3. Data Utilities            — Loading, cleaning, chart theming
    4. UI Rendering              — Landing page, dashboard sections, footer
    5. Main Application          — Sidebar config, engine orchestration, tab layout
"""

from __future__ import annotations

import html
import logging
import re
import time
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ── Optional Dependencies ────────────────────────────────────────────────────

try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller, kpss

    _HAS_STATSMODELS = True
except ImportError:
    sm = None
    _HAS_STATSMODELS = False

try:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import ElasticNetCV, HuberRegressor, RidgeCV
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

VERSION = "v2.2.0"
PRODUCT_NAME = "Aarambh"
COMPANY = "Hemrek Capital"

# Engine defaults
LOOKBACK_WINDOWS = (5, 10, 20, 50, 100)
MIN_TRAIN_SIZE = 20
MAX_TRAIN_SIZE = 500
REFIT_INTERVAL = 5
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
HUBER_EPSILON = 1.35
HUBER_MAX_ITER = 500
OU_PROJECTION_DAYS = 90
MIN_DATA_POINTS = 80

# Signal thresholds (conviction score → signal mapping)
CONVICTION_STRONG = 60
CONVICTION_MODERATE = 40
CONVICTION_WEAK = 20

# Z-score zone boundaries
Z_EXTREME = 2.0
Z_THRESHOLD = 1.0

# Staleness
STALENESS_DAYS = 3

# Timeframe filter mapping (trading days)
TIMEFRAME_TRADING_DAYS = {"1M": 21, "6M": 126, "1Y": 252, "2Y": 504}

# Default predictors for NIFTY50 use case
DEFAULT_PREDICTORS = (
    "AD_RATIO", "COUNT", "REL_AD_RATIO", "REL_BREADTH",
    "IN10Y", "IN02Y", "IN30Y", "INIRYY", "REPO",
    "US02Y", "US10Y", "US30Y", "NIFTY50_DY", "NIFTY50_PB",
)

# Default Google Sheets URL
DEFAULT_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1po7z42n3dYIQGAvn0D1-a4pmyxpnGPQ13TrNi3DB5_c/"
    "edit?gid=1938234952#gid=1938234952"
)

# Chart theme
CHART_BG = "#1A1A1A"
CHART_GRID = "#2A2A2A"
CHART_ZEROLINE = "#3A3A3A"
CHART_FONT_COLOR = "#EAEAEA"

# Signal colors
COLOR_GREEN = "#10b981"
COLOR_RED = "#ef4444"
COLOR_GOLD = "#FFC300"
COLOR_CYAN = "#06b6d4"
COLOR_AMBER = "#f59e0b"
COLOR_MUTED = "#888888"


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & CSS
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AARAMBH | Fair Value Breadth",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    :root {
        --primary-color: #FFC300;
        --primary-rgb: 255, 195, 0;
        --background-color: #0F0F0F;
        --secondary-background-color: #1A1A1A;
        --bg-card: #1A1A1A;
        --bg-elevated: #2A2A2A;
        --text-primary: #EAEAEA;
        --text-secondary: #EAEAEA;
        --text-muted: #888888;
        --border-color: #2A2A2A;
        --border-light: #3A3A3A;
        --success-green: #10b981;
        --danger-red: #ef4444;
        --warning-amber: #f59e0b;
        --info-cyan: #06b6d4;
        --purple: #8b5cf6;
        --neutral: #888888;
    }

    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    .main, [data-testid="stSidebar"] { background-color: var(--background-color); color: var(--text-primary); }
    .stApp > header { background-color: transparent; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    .block-container { padding-top: 3.5rem; max-width: 90%; padding-left: 2rem; padding-right: 2rem; }

    [data-testid="collapsedControl"] {
        display: flex !important; visibility: visible !important; opacity: 1 !important;
        background-color: var(--secondary-background-color) !important;
        border: 2px solid var(--primary-color) !important; border-radius: 8px !important;
        padding: 10px !important; margin: 12px !important;
        box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.4) !important;
        z-index: 999999 !important; position: fixed !important;
        top: 14px !important; left: 14px !important;
        width: 40px !important; height: 40px !important;
        align-items: center !important; justify-content: center !important;
    }
    [data-testid="collapsedControl"]:hover {
        background-color: rgba(var(--primary-rgb), 0.2) !important;
        box-shadow: 0 0 20px rgba(var(--primary-rgb), 0.6) !important; transform: scale(1.05);
    }
    [data-testid="collapsedControl"] svg { stroke: var(--primary-color) !important; width: 20px !important; height: 20px !important; }
    [data-testid="stSidebar"] button[kind="header"] { background-color: transparent !important; border: none !important; }
    [data-testid="stSidebar"] button[kind="header"] svg { stroke: var(--primary-color) !important; }
    button[kind="header"] { z-index: 999999 !important; }

    .premium-header {
        background: var(--secondary-background-color); padding: 1.25rem 2rem; border-radius: 16px;
        margin-bottom: 1.5rem; box-shadow: 0 0 20px rgba(var(--primary-rgb), 0.1);
        border: 1px solid var(--border-color); position: relative; overflow: hidden; margin-top: 1rem;
    }
    .premium-header::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(circle at 20% 50%, rgba(var(--primary-rgb),0.08) 0%, transparent 50%);
        pointer-events: none;
    }
    .premium-header h1 { margin: 0; font-size: 2rem; font-weight: 700; color: var(--text-primary); letter-spacing: -0.50px; position: relative; }
    .premium-header .tagline { color: var(--text-muted); font-size: 0.9rem; margin-top: 0.25rem; font-weight: 400; position: relative; }

    .metric-card {
        background-color: var(--bg-card); padding: 1.25rem; border-radius: 12px;
        border: 1px solid var(--border-color); box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.08);
        margin-bottom: 0.5rem; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative; overflow: hidden; min-height: 160px;
        display: flex; flex-direction: column; justify-content: center;
    }
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(0,0,0,0.3); border-color: var(--border-light); }
    .metric-card h4 { color: var(--text-muted); font-size: 0.75rem; margin-bottom: 0.5rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; min-height: 30px; display: flex; align-items: center; }
    .metric-card h3 { color: var(--text-primary); font-size: 1.1rem; font-weight: 700; margin-bottom: 0.5rem; }
    .metric-card p { color: var(--text-muted); font-size: 0.85rem; line-height: 1.5; margin: 0; }
    .metric-card h2 { color: var(--text-primary); font-size: 1.75rem; font-weight: 700; margin: 0; line-height: 1; }
    .metric-card .sub-metric { font-size: 0.75rem; color: var(--text-muted); margin-top: 0.5rem; font-weight: 500; }
    .metric-card.primary h2 { color: var(--primary-color); }
    .metric-card.success h2 { color: var(--success-green); }
    .metric-card.danger h2 { color: var(--danger-red); }
    .metric-card.info h2 { color: var(--info-cyan); }
    .metric-card.warning h2 { color: var(--warning-amber); }
    .metric-card.purple h2 { color: var(--purple); }
    .metric-card.neutral h2 { color: var(--neutral); }

    .signal-card { background: var(--bg-card); border-radius: 16px; border: 2px solid var(--border-color); padding: 1.5rem; position: relative; overflow: hidden; }
    .signal-card.overvalued { border-color: var(--danger-red); box-shadow: 0 0 30px rgba(239, 68, 68, 0.15); }
    .signal-card.undervalued { border-color: var(--success-green); box-shadow: 0 0 30px rgba(16, 185, 129, 0.15); }
    .signal-card.fair { border-color: var(--primary-color); box-shadow: 0 0 30px rgba(255, 195, 0, 0.15); }
    .signal-card .label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1.5px; color: var(--text-muted); font-weight: 600; margin-bottom: 0.5rem; }
    .signal-card .value { font-size: 2.5rem; font-weight: 700; line-height: 1; }
    .signal-card .subtext { font-size: 0.85rem; color: var(--text-secondary); margin-top: 0.75rem; }
    .signal-card.overvalued .value { color: var(--danger-red); }
    .signal-card.undervalued .value { color: var(--success-green); }
    .signal-card.fair .value { color: var(--primary-color); }

    .guide-box { background: rgba(var(--primary-rgb), 0.05); border-left: 3px solid var(--primary-color); padding: 1rem; border-radius: 8px; margin: 1rem 0; color: var(--text-secondary); font-size: 0.9rem; }
    .guide-box.success { background: rgba(16, 185, 129, 0.05); border-left-color: var(--success-green); }
    .guide-box.danger { background: rgba(239, 68, 68, 0.05); border-left-color: var(--danger-red); }

    .info-box { background: var(--secondary-background-color); border: 1px solid var(--border-color); padding: 1.25rem; border-radius: 12px; margin: 0.5rem 0; box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.08); }
    .info-box h4 { color: var(--primary-color); margin: 0 0 0.5rem 0; font-size: 1rem; font-weight: 700; }
    .info-box p { color: var(--text-muted); margin: 0; font-size: 0.9rem; line-height: 1.6; }

    .conviction-meter { background: var(--bg-elevated); border-radius: 12px; padding: 1rem; margin: 0.5rem 0; }
    .conviction-bar { height: 8px; border-radius: 4px; background: var(--border-color); overflow: hidden; }
    .conviction-fill { height: 100%; border-radius: 4px; transition: width 0.5s ease; }

    .section-divider { height: 1px; background: linear-gradient(90deg, transparent 0%, var(--border-color) 50%, transparent 100%); margin: 1.5rem 0; }

    .status-badge { display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.4rem 0.8rem; border-radius: 20px; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; }
    .status-badge.buy { background: rgba(16, 185, 129, 0.15); color: var(--success-green); border: 1px solid rgba(16, 185, 129, 0.3); }
    .status-badge.sell { background: rgba(239, 68, 68, 0.15); color: var(--danger-red); border: 1px solid rgba(239, 68, 68, 0.3); }
    .status-badge.neutral { background: rgba(136, 136, 136, 0.15); color: var(--neutral); border: 1px solid rgba(136, 136, 136, 0.3); }

    .stButton>button { border: 2px solid var(--primary-color); background: transparent; color: var(--primary-color); font-weight: 700; border-radius: 12px; padding: 0.75rem 2rem; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); text-transform: uppercase; letter-spacing: 0.5px; }
    .stButton>button:hover { box-shadow: 0 0 25px rgba(var(--primary-rgb), 0.6); background: var(--primary-color); color: #1A1A1A; transform: translateY(-2px); }

    .stTabs [data-baseweb="tab-list"] { gap: 24px; background: transparent; }
    .stTabs [data-baseweb="tab"] { color: var(--text-muted); border-bottom: 2px solid transparent; background: transparent; font-weight: 600; }
    .stTabs [aria-selected="true"] { color: var(--primary-color); border-bottom: 2px solid var(--primary-color); background: transparent !important; }

    .stPlotlyChart { border-radius: 12px; background-color: var(--secondary-background-color); padding: 10px; border: 1px solid var(--border-color); box-shadow: 0 0 25px rgba(var(--primary-rgb), 0.1); }
    .stDataFrame { border-radius: 12px; background-color: var(--secondary-background-color); border: 1px solid var(--border-color); }
    .sidebar-title { font-size: 0.75rem; font-weight: 700; color: var(--primary-color); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.75rem; }
    [data-testid="stSidebar"] { background: var(--secondary-background-color); border-right: 1px solid var(--border-color); }
    .stTextInput > div > div > input { background: var(--bg-elevated) !important; border: 1px solid var(--border-color) !important; border-radius: 8px !important; color: var(--text-primary) !important; }
    .stTextInput > div > div > input:focus { border-color: var(--primary-color) !important; box-shadow: 0 0 0 2px rgba(var(--primary-rgb), 0.2) !important; }
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--background-color); }
    ::-webkit-scrollbar-thumb { background: var(--border-color); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MATHEMATICAL PRIMITIVES
# ══════════════════════════════════════════════════════════════════════════════

def ornstein_uhlenbeck_estimate(
    series: np.ndarray,
    dt: float = 1.0,
) -> tuple[float, float, float]:
    """Estimate OU process parameters via AR(1) regression.

    Model: dx = θ(μ − x)dt + σdW

    Returns:
        (theta, mu, sigma) — mean-reversion speed, equilibrium level, volatility.
    """
    x = np.asarray(series, dtype=np.float64)
    x = x[np.isfinite(x)]

    if len(x) < 20:
        if len(x) > 1:
            return 0.05, 0.0, max(float(np.std(x)), 1e-6)
        return 0.05, 0.0, 1.0

    x_lag = x[:-1]
    x_curr = x[1:]
    n = len(x_lag)

    sx = np.sum(x_lag)
    sy = np.sum(x_curr)
    sxx = np.dot(x_lag, x_lag)
    sxy = np.dot(x_lag, x_curr)

    denom = n * sxx - sx ** 2
    if abs(denom) < 1e-12:
        return 0.05, float(np.mean(x)), max(float(np.std(x)), 1e-6)

    a = (n * sxy - sx * sy) / denom
    b = (sy * sxx - sx * sxy) / denom
    a = np.clip(a, 1e-6, 1.0 - 1e-6)

    # Jackknife bias correction (Tang & Chen, 2009): the OLS estimator of the
    # AR(1) coefficient has downward bias of order O(1/n), systematically
    # underestimating theta and overestimating the half-life.
    a_bias = (1 + 3 * a) / n
    a = np.clip(a - a_bias, 1e-6, 1.0 - 1e-6)

    theta = -np.log(a) / dt
    mu = b / (1 - a)

    residuals = x_curr - a * x_lag - b
    sigma_sq = np.var(residuals)
    if a > 0.98:
        sigma = max(float(np.std(residuals)) * np.sqrt(2 * max(theta, 1e-4)), 1e-6)
    else:
        sigma = np.sqrt(max(sigma_sq * 2 * theta / (1 - a ** 2), 1e-12))

    return max(float(theta), 1e-4), float(mu), max(float(sigma), 1e-6)


def kalman_filter_1d(
    observations: np.ndarray,
    process_var: float | None = None,
    measurement_var: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1D Kalman filter for smoothing noisy time series.

    Returns:
        (filtered_state, kalman_gains, estimate_variances)
    """
    obs = np.asarray(observations, dtype=np.float64)
    n = len(obs)
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    # Initialize with default guesses, but allow adaptation if None
    dynamic_pv = process_var is None
    dynamic_mv = measurement_var is None
    p_var = process_var if process_var is not None else 1e-3
    m_var = measurement_var if measurement_var is not None else 1.0

    state = float(obs[0]) if np.isfinite(obs[0]) else 0.0
    estimate_var = m_var

    filtered = np.zeros(n)
    gains = np.zeros(n)
    variances = np.zeros(n)
    filtered[0] = state
    variances[0] = estimate_var

    for i in range(1, n):
        # Dynamic variance adaptation via simple EWMA of innovations
        if i > 5 and (dynamic_pv or dynamic_mv):
            innovation = obs[i-1] - filtered[i-1]
            if dynamic_mv:
                m_var = np.clip(0.95 * m_var + 0.05 * (innovation ** 2), 1e-6, 1e4)
            if dynamic_pv:
                state_diff = filtered[i-1] - filtered[i-2]
                p_var = np.clip(0.95 * p_var + 0.05 * (state_diff ** 2), 1e-6, 1e2)

        pred_var = estimate_var + p_var
        if np.isfinite(obs[i]):
            K = pred_var / (pred_var + m_var)
            state = state + K * (obs[i] - state)
            estimate_var = (1 - K) * pred_var
            gains[i] = K
        else:
            estimate_var = pred_var
        filtered[i] = state
        variances[i] = estimate_var

    return filtered, gains, variances


def hurst_rs(series: np.ndarray, max_lag: int | None = None) -> float:
    """Hurst exponent via Rescaled Range (R/S) analysis.

    Returns:
        H < 0.5 → mean-reverting, H ≈ 0.5 → random walk, H > 0.5 → trending.
    """
    x = np.asarray(series, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 20:
        return 0.5

    if max_lag is None:
        max_lag = min(n // 2, 100)

    lags: list[int] = []
    rs_values: list[float] = []

    for lag in range(10, max_lag + 1, max(1, max_lag // 20)):
        rs_list: list[float] = []
        for start in range(0, n - lag, lag):
            segment = x[start : start + lag]
            if len(segment) < 10:
                continue
            mean_seg = np.mean(segment)
            dev = np.cumsum(segment - mean_seg)
            R = float(np.max(dev) - np.min(dev))
            S = float(np.std(segment, ddof=1))
            if S > 1e-10:
                rs_list.append(R / S)

        if rs_list:
            lags.append(lag)
            rs_values.append(float(np.mean(rs_list)))

    if len(lags) < 3:
        return 0.5

    slope, _, _, _, _ = stats.linregress(np.log(lags), np.log(rs_values))
    return float(np.clip(slope, 0.01, 0.99))


# ══════════════════════════════════════════════════════════════════════════════
# FAIR VALUE BREADTH ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class FairValueEngine:
    """Walk-forward fair value engine with multi-lookback breadth analytics.

    Pipeline:
        1. Expanding-window ensemble regression (Ridge + Huber + OLS)
        2. Multi-lookback z-score computation and zone classification
        3. Breadth aggregation and raw conviction scoring
        4. Kalman filtering of conviction with confidence bands
        5. OU estimation on residuals for half-life and forward projection
        6. Hurst exponent for mean-reversion validation
        7. Swing-based divergence detection
        8. Forward change analysis for signal performance
    """

    def __init__(self) -> None:
        self.ts_data: pd.DataFrame = pd.DataFrame()
        self.lookback_data: dict = {}
        self.model_stats: dict = {}
        self.ou_params: dict = {}
        self.ou_projection: np.ndarray = np.array([])
        self.pivots: dict = {}
        self.residual_stats: dict = {}
        self.hurst: float = 0.5
        self.latest_feature_impacts: dict = {}

    # ── Public API ────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        progress_callback=None,
    ) -> FairValueEngine:
        """Run the full walk-forward pipeline."""
        start_time = time.time()

        self.feature_names = feature_names or [f"X{i}" for i in range(X.shape[1])]
        self.n_samples = len(y)
        self.y = y.copy()

        self._walk_forward_regression(X, y, progress_callback)

        if progress_callback:
            progress_callback(1.0, "Computing multi-lookback signals...")

        self.residuals = y - self.predictions

        self._compute_model_stats()
        self._compute_multi_lookback_signals()
        self._compute_breadth_metrics()
        self._compute_kalman_conviction()
        self._find_pivots()
        self._compute_divergences()
        self._compute_forward_changes()
        self._compute_ou_diagnostics()
        self._compute_hurst()

        elapsed = time.time() - start_time
        logging.info(
            "v2.0 engine [%d obs, %d features] in %.1fs",
            self.n_samples, len(self.feature_names), elapsed,
        )

        if progress_callback:
            progress_callback(1.0, "Done.")

        return self

    def get_current_signal(self) -> dict:
        """Derive the current composite signal from the latest observation."""
        if self.ts_data.empty:
            return {
                "signal": "HOLD", "strength": "NEUTRAL", "confidence": "N/A",
                "conviction_score": 0, "conviction_upper": 0, "conviction_lower": 0,
                "regime": "NEUTRAL", "oversold_breadth": 0, "overbought_breadth": 0,
                "residual": 0, "fair_value": 0, "actual": 0, "avg_z": 0,
                "model_spread": 0, "has_bullish_div": False, "has_bearish_div": False,
                "ou_half_life": 0, "adf_pvalue": 1.0, "kpss_pvalue": 0.0, "hurst": 0.5,
            }
        current = self.ts_data.iloc[-1]
        conviction = current["ConvictionScore"]

        if conviction < -CONVICTION_STRONG:
            signal, strength = "BUY", "STRONG"
        elif conviction < -CONVICTION_MODERATE:
            signal, strength = "BUY", "MODERATE"
        elif conviction < -CONVICTION_WEAK:
            signal, strength = "BUY", "WEAK"
        elif conviction > CONVICTION_STRONG:
            signal, strength = "SELL", "STRONG"
        elif conviction > CONVICTION_MODERATE:
            signal, strength = "SELL", "MODERATE"
        elif conviction > CONVICTION_WEAK:
            signal, strength = "SELL", "WEAK"
        else:
            signal, strength = "HOLD", "NEUTRAL"

        oversold_breadth = current["OversoldBreadth"]
        overbought_breadth = current["OverboughtBreadth"]

        if signal == "BUY":
            confidence = "HIGH" if oversold_breadth >= 80 else "MEDIUM" if oversold_breadth >= 60 else "LOW"
        elif signal == "SELL":
            confidence = "HIGH" if overbought_breadth >= 80 else "MEDIUM" if overbought_breadth >= 60 else "LOW"
        else:
            confidence = "N/A"

        return {
            "signal": signal,
            "strength": strength,
            "confidence": confidence,
            "conviction_score": conviction,
            "conviction_upper": current["ConvictionUpper"],
            "conviction_lower": current["ConvictionLower"],
            "regime": current["Regime"],
            "oversold_breadth": oversold_breadth,
            "overbought_breadth": overbought_breadth,
            "residual": current["Residual"],
            "fair_value": current["FairValue"],
            "actual": current["Actual"],
            "avg_z": current["AvgZ"],
            "model_spread": current["ModelSpread"],
            "has_bullish_div": current["BullishDiv"],
            "has_bearish_div": current["BearishDiv"],
            "ou_half_life": self.ou_params["half_life"],
            "adf_pvalue": self.ou_params.get("adf_pvalue", 1.0),
            "kpss_pvalue": self.ou_params.get("kpss_pvalue", 0.0),
            "hurst": self.hurst,
        }

    def get_model_stats(self) -> dict:
        return self.model_stats

    def get_regime_stats(self) -> dict:
        ts = self.ts_data
        regime_counts = ts["Regime"].value_counts()
        return {
            "strongly_oversold": regime_counts.get("STRONGLY OVERSOLD", 0),
            "oversold": regime_counts.get("OVERSOLD", 0),
            "neutral": regime_counts.get("NEUTRAL", 0),
            "overbought": regime_counts.get("OVERBOUGHT", 0),
            "strongly_overbought": regime_counts.get("STRONGLY OVERBOUGHT", 0),
            "current_regime": ts["Regime"].iloc[-1],
            "total_buy_signals": ts["BuySignalBreadth"].sum(),
            "total_sell_signals": ts["SellSignalBreadth"].sum(),
            "total_bull_div": ts["BullishDiv"].sum(),
            "total_bear_div": ts["BearishDiv"].sum(),
            "total_pivot_tops": ts["IsPivotTop"].sum(),
            "total_pivot_bottoms": ts["IsPivotBottom"].sum(),
        }

    def get_signal_performance(self) -> dict:
        """Forward change analysis for BUY/SELL signals at each horizon."""
        ts = self.ts_data
        results = {}
        for period in (5, 10, 20):
            buy_changes: list[float] = []
            sell_changes: list[float] = []
            for i in range(len(ts) - period):
                score = ts["ConvictionScore"].iloc[i]
                fwd = ts[f"FwdChg_{period}"].iloc[i]
                if pd.isna(fwd):
                    continue
                if score < -CONVICTION_MODERATE:
                    buy_changes.append(fwd)
                if score > CONVICTION_MODERATE:
                    sell_changes.append(-fwd)
            results[period] = {
                "buy_avg": float(np.mean(buy_changes)) if buy_changes else 0.0,
                "buy_hit": float(np.mean([c > 0 for c in buy_changes])) if buy_changes else 0.0,
                "buy_count": len(buy_changes),
                "sell_avg": float(np.mean(sell_changes)) if sell_changes else 0.0,
                "sell_hit": float(np.mean([c > 0 for c in sell_changes])) if sell_changes else 0.0,
                "sell_count": len(sell_changes),
            }
        return results

    # ── Private: Walk-Forward Regression ──────────────────────────────────

    def _walk_forward_regression(
        self, X: np.ndarray, y: np.ndarray, progress_callback,
    ) -> None:
        """Expanding-window ensemble regression with periodic refitting."""
        n = self.n_samples
        self.predictions = np.full(n, np.nan)
        self.model_spread = np.zeros(n)

        # Pre-fill initial minimum train size
        for t in range(MIN_TRAIN_SIZE):
            self.predictions[t] = float(np.mean(y[:t])) if t > 0 else float(y[0])
            self.model_spread[t] = 0.0

        # Precompute global exponential decay weights to avoid repeated np.exp() calculations
        decay_rate = np.log(2) / 252.0
        global_weights = np.exp(-decay_rate * np.arange(MAX_TRAIN_SIZE - 1, -1, -1))

        # Dynamically scale refit interval to prevent over-computation on large datasets (Bounds: 5 to 21 days)
        dynamic_refit = int(np.clip(n // 150, 1, 5))
        
        chunks = []
        for t_start in range(MIN_TRAIN_SIZE, n, dynamic_refit):
            t_end = min(t_start + dynamic_refit, n)
            chunks.append((t_start, t_end))

        last_models: dict = {"ridge": None, "huber": None, "ols": None, "elasticnet": None, "pca_wls": None}
        valid_cols = np.ones(X.shape[1], dtype=bool)

        import os
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Using ThreadPoolExecutor because scikit-learn & numpy release the GIL during heavy math
        # This yields significant speedups without ProcessPool/Streamlit serialization overhead
        max_workers = min(len(chunks), (os.cpu_count() or 1) * 2)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {
                executor.submit(self._process_wf_chunk, start, end, X, y, global_weights): (start, end)
                for start, end in chunks
            }
            
            completed = 0
            for future in as_completed(future_to_chunk):
                start, end = future_to_chunk[future]
                completed += 1
                if progress_callback and completed % max(1, len(chunks) // 20) == 0:
                    progress_callback(completed / len(chunks), f"Walking forward... ({completed}/{len(chunks)} blocks)")

                try:
                    t_start, t_end, preds, spreads, models, v_cols = future.result()
                    self.predictions[t_start:t_end] = preds
                    self.model_spread[t_start:t_end] = spreads
                    
                    # Keep the models from the very last chunk for feature impact extraction
                    if t_end == n:
                        last_models = models
                        valid_cols = v_cols
                except Exception as e:
                    logging.error("Chunk [%d:%d] failed: %s", start, end, e)
                
        # Extract driving features from the final fitted ensemble matrix
        self._compute_feature_impacts(last_models, valid_cols)

    @staticmethod
    def _process_wf_chunk(
        t_start: int,
        t_end: int,
        X: np.ndarray,
        y: np.ndarray,
        global_weights: np.ndarray,
    ) -> tuple[int, int, np.ndarray, np.ndarray, dict, np.ndarray]:
        """Process a single walk-forward block (fit at start, predict up to end)."""
        start_idx = max(0, t_start - MAX_TRAIN_SIZE)
        models, scaler, valid_cols = FairValueEngine._fit_ensemble(
            X[start_idx:t_start], y[start_idx:t_start], t_start, global_weights
        )

        X_chunk = X[t_start:t_end]
        if len(X_chunk) == 0:
            return t_start, t_end, np.array([]), np.array([]), models, valid_cols

        preds_matrix = FairValueEngine._predict_ensemble(
            X_chunk, models, scaler, valid_cols, t_start
        )

        if preds_matrix:
            preds_stacked = np.vstack(preds_matrix)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                preds = np.nanmean(preds_stacked, axis=0)
                spreads = np.maximum(np.nanstd(preds_stacked, axis=0), 1e-6) if len(preds_matrix) > 1 else np.full(len(preds), 1e-6)
                
                # Fallback for any NaN projections
                nans = np.isnan(preds)
                if np.any(nans):
                    preds[nans] = float(np.mean(y[start_idx:t_start]))
                    spreads[nans] = 1e-6
        else:
            fallback = float(np.mean(y[start_idx:t_start]))
            preds = np.full(t_end - t_start, fallback)
            spreads = np.full(t_end - t_start, 1e-6)

        return t_start, t_end, preds, spreads, models, valid_cols

    @staticmethod
    def _fit_ensemble(
        X_train: np.ndarray, y_train: np.ndarray, t: int, global_weights: np.ndarray,
    ) -> tuple[dict, StandardScaler | None, np.ndarray]:
        """Fit Ridge + Huber + WLS ensemble on training data with exponential decay weighting."""
        models: dict = {"ridge": None, "huber": None, "ols": None, "elasticnet": None, "pca_wls": None}
        scaler = None

        # Filter out zero-variance predictors to prevent collinearity/singular matrix errors
        valid_cols = np.std(X_train, axis=0) > 1e-8
        if not np.any(valid_cols):  # Fallback if somehow everything is flat
            valid_cols = np.ones(X_train.shape[1], dtype=bool)
            
        X_train_clean = X_train[:, valid_cols]

        n_samples = len(y_train)
        # Slice precomputed weights directly from memory
        weights = global_weights[-n_samples:]

        if _HAS_SKLEARN:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train_clean)

            try:
                ridge = RidgeCV(alphas=list(RIDGE_ALPHAS), cv=None)
                ridge.fit(X_scaled, y_train, sample_weight=weights)
                models["ridge"] = ridge
            except Exception as e:
                logging.warning("Ridge fit failed at t=%d: %s", t, e)

            try:
                huber = HuberRegressor(epsilon=HUBER_EPSILON, max_iter=HUBER_MAX_ITER, tol=1e-3)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    huber.fit(X_scaled, y_train, sample_weight=weights)
                models["huber"] = huber
            except Exception as e:
                logging.warning("Huber fit failed at t=%d: %s", t, e)

            try:
                # ElasticNet combines L1 (feature selection) and L2 (shrinkage).
                # Testing ratios from 10% L1 to 100% L1 (Pure Lasso).
                # Revert to single-threaded ElasticNet to prevent thread explosion during outer parallelization
                enet = ElasticNetCV(
                    l1_ratio=[0.5, 0.9, 1.0],
                    n_alphas=15,
                    cv=2,
                    max_iter=10000,
                    tol=1e-2,
                    selection="random",
                    n_jobs=1
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    enet.fit(X_scaled, y_train, sample_weight=weights)
                models["elasticnet"] = enet
            except Exception as e:
                logging.warning("ElasticNet fit failed at t=%d: %s", t, e)

            try:
                pca = PCA(n_components=0.95, svd_solver="full")
                X_pca = pca.fit_transform(X_scaled)
                models["pca_wls"] = pca
                
                # Swap statsmodels for sklearn LinearRegression (bypasses heavy p-value/t-stat computation)
                ols = LinearRegression()
                ols.fit(X_pca, y_train, sample_weight=weights)
                models["ols"] = ols
            except Exception as e:
                logging.warning("PCA/OLS fit failed at t=%d: %s", t, e)

        return models, scaler, valid_cols

    @staticmethod
    def _predict_ensemble(
        X_pred: np.ndarray,
        models: dict,
        scaler: StandardScaler | None,
        valid_cols: np.ndarray,
        t_start: int,
    ) -> list[np.ndarray]:
        """Collect vectorized predictions from all available ensemble members."""
        preds_list: list[np.ndarray] = []

        def _add_safe_pred(arr: np.ndarray) -> None:
            arr_clean = np.where(np.isfinite(arr) & (np.abs(arr) < 1e10), arr, np.nan)
            if not np.all(np.isnan(arr_clean)):
                preds_list.append(arr_clean)

        X_pred_clean = X_pred[:, valid_cols]

        if _HAS_SKLEARN and scaler is not None:
            try:
                X_scaled = scaler.transform(X_pred_clean)
                if models["ridge"] is not None:
                    try:
                        _add_safe_pred(models["ridge"].predict(X_scaled))
                    except Exception as e:
                        logging.warning("Ridge predict failed at t=%d: %s", t_start, e)
                if models["huber"] is not None:
                    try:
                        _add_safe_pred(models["huber"].predict(X_scaled))
                    except Exception as e:
                        logging.warning("Huber predict failed at t=%d: %s", t_start, e)
                if models.get("elasticnet") is not None:
                    try:
                        _add_safe_pred(models["elasticnet"].predict(X_scaled))
                    except Exception as e:
                        logging.warning("ElasticNet predict failed at t=%d: %s", t_start, e)
                if models.get("ols") is not None and models.get("pca_wls") is not None:
                    try:
                        X_pca_pred = models["pca_wls"].transform(X_scaled)
                        _add_safe_pred(models["ols"].predict(X_pca_pred))
                    except Exception as e:
                        logging.warning("OLS predict failed at t=%d: %s", t_start, e)
            except Exception as e:
                logging.warning("Prediction cascade failed at t=%d: %s", t_start, e)

        return preds_list

    def _compute_feature_impacts(self, models: dict, valid_cols: np.ndarray) -> None:
        """Map PCA+WLS coefficients back to original features to determine current driving factors."""
        features = np.array(self.feature_names)[valid_cols]
        wls = models.get("ols")
        pca = models.get("pca_wls")

        if wls is not None and pca is not None:
            try:
                wls_weights = wls.coef_
                # Matrix multiplication: (Components) x (Feature contributions to components)
                # Yields the effective weight of each original feature in the WLS model
                feature_weights = np.dot(wls_weights, pca.components_)
                
                # Convert to relative percentage of absolute impact
                abs_weights = np.abs(feature_weights)
                total_impact = np.sum(abs_weights)
                if total_impact > 1e-10:
                    pct_impacts = (abs_weights / total_impact) * 100
                    impacts = {f: float(imp) for f, imp in zip(features, pct_impacts)}
                    self.latest_feature_impacts = dict(sorted(impacts.items(), key=lambda x: x[1], reverse=True))
                    return
            except Exception as e:
                logging.warning("Failed to compute feature impacts: %s", e)
                
        self.latest_feature_impacts = {}

    # ── Private: Analytics Pipeline ───────────────────────────────────────

    def _compute_model_stats(self) -> None:
        """OOS model fit statistics (only walk-forward portion)."""
        oos_mask = np.arange(self.n_samples) >= MIN_TRAIN_SIZE
        y_oos = self.y[oos_mask]
        pred_oos = self.predictions[oos_mask]

        valid = np.isfinite(pred_oos)
        y_v, p_v = y_oos[valid], pred_oos[valid]

        if len(y_v) > 2 and _HAS_SKLEARN:
            r2 = r2_score(y_v, p_v)
            rmse = float(np.sqrt(mean_squared_error(y_v, p_v)))
            mae = float(mean_absolute_error(y_v, p_v))
        else:
            ss_res = float(np.sum((y_v - p_v) ** 2))
            ss_tot = float(np.sum((y_v - np.mean(y_v)) ** 2))
            r2 = 1 - ss_res / max(ss_tot, 1e-10)
            rmse = float(np.sqrt(np.mean((y_v - p_v) ** 2)))
            mae = float(np.mean(np.abs(y_v - p_v)))

        # R² vs random walk (Welch & Goyal, 2008): measures whether the model
        # beats a naive "no change" forecast on persistent series like PE.
        if len(y_v) > 2:
            rw_forecast = np.empty_like(y_v)
            rw_forecast[0] = y_v[0]
            rw_forecast[1:] = y_v[:-1]
            ss_res = float(np.sum((y_v - p_v) ** 2))
            ss_rw = float(np.sum((y_v - rw_forecast) ** 2))
            r2_vs_rw = 1 - ss_res / max(ss_rw, 1e-10)
        else:
            r2_vs_rw = 0.0

        self.model_stats = {
            "r2_oos": r2,
            "r2_vs_rw": r2_vs_rw,
            "rmse_oos": rmse,
            "mae_oos": mae,
            "n_obs": len(y_v),
            "n_features": len(self.feature_names),
            "avg_model_spread": float(np.mean(self.model_spread[oos_mask])),
        }

    def _compute_multi_lookback_signals(self) -> None:
        """Z-scores and zone classifications for each lookback window."""
        r = self.residuals
        n = len(r)
        self.lookback_data = {}
        series = pd.Series(r)

        def _process_lookback(lb: int) -> tuple[int, dict] | None:
            if n < lb:
                return None
            min_periods = max(lb // 2, 5)

            # Distribution-free z-equivalents via empirical quantile ranks
            # Vectorized using cythonized rolling rank equivalent to np.mean(window <= r[t])
            s_r = pd.Series(r)
            rank_pct = s_r.rolling(window=lb, min_periods=min_periods).rank(method="max", pct=True).values
            roll_count = s_r.rolling(window=lb, min_periods=min_periods).count().values

            z_scores = np.full(n, np.nan)
            valid = (roll_count >= min_periods) & ~np.isnan(rank_pct)
            
            if np.any(valid):
                c_lower = 0.5 / roll_count[valid]
                c_upper = 1.0 - c_lower
                clipped_pct = np.clip(rank_pct[valid], c_lower, c_upper)
                z_scores[valid] = stats.norm.ppf(clipped_pct)

            zones = self._classify_zones(z_scores, n)
            buy_signals, sell_signals = self._detect_crossover_signals(z_scores, n)

            return lb, {
                "z_scores": z_scores,
                "zones": zones,
                "buy_signals": buy_signals,
                "sell_signals": sell_signals,
            }

        # Execute lookback calculations sequentially (avoiding unnecessary GIL overhead)
        for lb in LOOKBACK_WINDOWS:
            res = _process_lookback(lb)
            if res is not None:
                self.lookback_data[res[0]] = res[1]

        self.ts_data = pd.DataFrame({
            "Actual": self.y,
            "FairValue": self.predictions,
            "Residual": self.residuals,
            "ModelSpread": self.model_spread,
        })
        for lb, data in self.lookback_data.items():
            self.ts_data[f"Z_{lb}"] = data["z_scores"]
            self.ts_data[f"Zone_{lb}"] = data["zones"]
            self.ts_data[f"Buy_{lb}"] = data["buy_signals"]
            self.ts_data[f"Sell_{lb}"] = data["sell_signals"]

        # Free memory by clearing lookback data after assignment to ts_data
        self.lookback_data.clear()

    @staticmethod
    def _classify_zones(z_scores: np.ndarray, n: int) -> np.ndarray:
        """Map z-scores to valuation zone labels."""
        condlist = [
            z_scores > Z_EXTREME,
            z_scores > Z_THRESHOLD,
            z_scores > -Z_THRESHOLD,
            z_scores > -Z_EXTREME
        ]
        choicelist = [
            "Extreme Over",
            "Overvalued",
            "Fair Value",
            "Undervalued"
        ]
        zones = np.select(condlist, choicelist, default="Extreme Under")
        np.putmask(zones, np.isnan(z_scores), "N/A")
        return zones

    @staticmethod
    def _detect_crossover_signals(z_scores: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Detect z-score threshold crossovers as discrete signals."""
        z_curr = z_scores[1:]
        z_prev = z_scores[:-1]
        
        valid = np.isfinite(z_curr) & np.isfinite(z_prev)
        
        buy_cond = valid & (z_curr < -Z_THRESHOLD) & (z_prev >= -Z_THRESHOLD)
        sell_cond = valid & (z_curr > Z_THRESHOLD) & (z_prev <= Z_THRESHOLD)
        
        buy_signals = np.zeros(n, dtype=bool)
        sell_signals = np.zeros(n, dtype=bool)
        
        buy_signals[1:] = buy_cond
        sell_signals[1:] = sell_cond
        
        return buy_signals, sell_signals

    def _compute_breadth_metrics(self) -> None:
        """Aggregate zone/signal counts across lookback windows."""
        n = len(self.ts_data)
        valid_lookbacks = [lb for lb in LOOKBACK_WINDOWS if f"Z_{lb}" in self.ts_data.columns]
        num_lb = max(len(valid_lookbacks), 1)

        oversold = np.zeros(n)
        overbought = np.zeros(n)
        extreme_os = np.zeros(n)
        extreme_ob = np.zeros(n)
        buy_count = np.zeros(n)
        sell_count = np.zeros(n)

        z_scores_list = []

        for lb in valid_lookbacks:
            zones = self.ts_data[f"Zone_{lb}"].values
            z = self.ts_data[f"Z_{lb}"].values
            
            extreme_os += (zones == "Extreme Under")
            oversold += (zones == "Extreme Under") | (zones == "Undervalued")
            extreme_ob += (zones == "Extreme Over")
            overbought += (zones == "Extreme Over") | (zones == "Overvalued")
            
            buy_count += self.ts_data[f"Buy_{lb}"].values
            sell_count += self.ts_data[f"Sell_{lb}"].values
            
            z_scores_list.append(z)

        if z_scores_list:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                avg_z = np.nan_to_num(np.nanmean(np.vstack(z_scores_list), axis=0), nan=0.0)
        else:
            avg_z = np.zeros(n)

        self.ts_data["OversoldBreadth"] = oversold / num_lb * 100
        self.ts_data["OverboughtBreadth"] = overbought / num_lb * 100
        self.ts_data["ExtremeOversold"] = extreme_os / num_lb * 100
        self.ts_data["ExtremeOverbought"] = extreme_ob / num_lb * 100
        self.ts_data["BuySignalBreadth"] = buy_count
        self.ts_data["SellSignalBreadth"] = sell_count
        self.ts_data["AvgZ"] = avg_z

        self.ts_data["ConvictionRaw"] = (
            self.ts_data["OverboughtBreadth"]
            - self.ts_data["OversoldBreadth"]
            + (self.ts_data["ExtremeOverbought"] - self.ts_data["ExtremeOversold"]) * 0.5
        )

    def _compute_kalman_conviction(self) -> None:
        """Kalman-filter raw conviction → smooth score with confidence bands."""
        raw = self.ts_data["ConvictionRaw"].values
        filtered, _gains, variances = kalman_filter_1d(raw)
        kalman_std = np.sqrt(np.maximum(variances, 0))

        self.ts_data["ConvictionScore"] = np.clip(filtered, -100, 100)
        self.ts_data["ConvictionUpper"] = np.clip(filtered + 1.96 * kalman_std, -100, 100)
        self.ts_data["ConvictionLower"] = np.clip(filtered - 1.96 * kalman_std, -100, 100)

        regimes = []
        for score in self.ts_data["ConvictionScore"]:
            if score < -CONVICTION_STRONG:
                regimes.append("STRONGLY OVERSOLD")
            elif score < -CONVICTION_WEAK:
                regimes.append("OVERSOLD")
            elif score > CONVICTION_STRONG:
                regimes.append("STRONGLY OVERBOUGHT")
            elif score > CONVICTION_WEAK:
                regimes.append("OVERBOUGHT")
            else:
                regimes.append("NEUTRAL")
        self.ts_data["Regime"] = regimes

    def _compute_divergences(self) -> None:
        """Swing-based divergence detection between target and residual."""
        n = len(self.ts_data)
        bull_div = np.zeros(n, dtype=bool)
        bear_div = np.zeros(n, dtype=bool)

        order = 5
        if n < order * 3:
            self.ts_data["BullishDiv"] = bull_div
            self.ts_data["BearishDiv"] = bear_div
            return

        price = np.asarray(self.y)
        residual = np.asarray(self.residuals)
        
        last_low_idx = -1
        last_high_idx = -1
        
        # Precompute expanding standard deviation of residuals to avoid O(n^2) loop computation
        min_periods = min(20, max(2, len(residual) // 3))
        expanding_std = pd.Series(residual).expanding(min_periods=min_periods).std().bfill().values

        for i in range(order * 2, n):
            window_price = price[i - 2 * order : i + 1]
            
            if np.argmin(window_price) == order:
                curr_low = i - order
                if last_low_idx != -1 and price[curr_low] < price[last_low_idx] and residual[curr_low] > residual[last_low_idx]:
                    if residual[curr_low] < -expanding_std[curr_low] * 0.5:
                        bull_div[i] = True  # Flag divergence at time of confirmation (no lookahead)
                last_low_idx = curr_low
                
            if np.argmax(window_price) == order:
                curr_high = i - order
                if last_high_idx != -1 and price[curr_high] > price[last_high_idx] and residual[curr_high] < residual[last_high_idx]:
                    if residual[curr_high] > expanding_std[curr_high] * 0.5:
                        bear_div[i] = True  # Flag divergence at time of confirmation (no lookahead)
                last_high_idx = curr_high

        self.ts_data["BullishDiv"] = bull_div
        self.ts_data["BearishDiv"] = bear_div

    def _find_pivots(self, order: int = 5) -> None:
        """Identify pivot highs/lows in the residual series."""
        r = np.asarray(self.residuals)
        n = len(r)

        conf_tops = []
        conf_bottoms = []
        top_vals = []
        bottom_vals = []

        for i in range(order * 2, n):
            window = r[i - 2 * order : i + 1]
            if np.argmax(window) == order:
                conf_tops.append(i)
                top_vals.append(r[i - order])
            if np.argmin(window) == order:
                conf_bottoms.append(i)
                bottom_vals.append(r[i - order])

        self.pivots = {
            "tops": conf_tops,
            "bottoms": conf_bottoms,
            "avg_top": float(np.mean(top_vals)) if top_vals else float(np.mean(r) + np.std(r)),
            "avg_bottom": float(np.mean(bottom_vals)) if bottom_vals else float(np.mean(r) - np.std(r)),
        }

        self.ts_data["IsPivotTop"] = False
        self.ts_data["IsPivotBottom"] = False
        if conf_tops:
            self.ts_data.loc[conf_tops, "IsPivotTop"] = True
        if conf_bottoms:
            self.ts_data.loc[conf_bottoms, "IsPivotBottom"] = True

        self.residual_stats = {
            "mean": float(np.mean(r)),
            "std": float(np.std(r)),
            "current": float(r[-1]),
            "current_zscore": float((r[-1] - np.mean(r)) / max(np.std(r), 1e-8)),
            "percentile": float(stats.percentileofscore(r, r[-1])),
            "min": float(np.min(r)),
            "max": float(np.max(r)),
        }

    def _compute_forward_changes(self) -> None:
        """Forward % change in target variable at multiple horizons."""
        n = len(self.ts_data)
        y_arr = np.asarray(self.y)
        for period in (5, 10, 20):
            fwd = np.full(n, np.nan)
            y_curr = y_arr[:-period]
            y_fwd = y_arr[period:]
            
            # Vectorized protection against division by zero
            valid = np.abs(y_curr) > 1e-10
            fwd[:-period][valid] = (y_fwd[valid] - y_curr[valid]) / np.abs(y_curr[valid]) * 100
            
            self.ts_data[f"FwdChg_{period}"] = fwd

    def _compute_ou_diagnostics(self) -> None:
        """OU parameter estimation and forward projection on OOS residuals."""
        r = self.residuals
        oos_r = r[MIN_TRAIN_SIZE:]

        if len(oos_r) > 30:
            theta, mu, sigma = ornstein_uhlenbeck_estimate(oos_r)
            try:
                adf_pvalue = float(adfuller(oos_r, autolag="AIC")[1])
            except Exception:
                adf_pvalue = 1.0

            # KPSS (Kwiatkowski et al., 1992): complements ADF.
            # ADF H₀ = unit root; KPSS H₀ = stationarity.
            # Concordance: ADF rejects + KPSS fails to reject → confident stationarity.
            # Conflict: both reject → fractional integration / structural breaks.
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    kpss_pvalue = float(kpss(oos_r, regression="c", nlags="auto")[1])
            except Exception:
                kpss_pvalue = 0.0
                
            # Regime-Conditional Volatility Adjustment
            # Measure recent localized panic vs historical ambient variance
            recent_vol = max(float(np.std(oos_r[-20:])), 1e-6) if len(oos_r) >= 20 else sigma
            hist_vol = max(float(np.std(oos_r)), 1e-6)
            
            # Volatility multiplier: high panic compresses time (faster reversion)
            # Clipped between 0.5 (extreme lethargy) and 3.0 (extreme panic) to maintain stability
            vol_multiplier = float(np.clip(recent_vol / hist_vol, 0.5, 3.0))
            dynamic_theta = theta * vol_multiplier
        else:
            theta, mu, sigma = 0.05, 0.0, max(float(np.std(r)), 1e-6)
            adf_pvalue = 1.0
            kpss_pvalue = 0.0
            vol_multiplier = 1.0
            dynamic_theta = theta

        self.ou_params = {
            "theta": theta,
            "dynamic_theta": dynamic_theta,
            "mu": mu,
            "sigma": sigma,
            "half_life_base": np.log(2) / max(theta, 1e-4),
            "half_life": np.log(2) / max(dynamic_theta, 1e-4),
            "stationary_std": sigma / np.sqrt(2 * max(theta, 1e-4)),
            "adf_pvalue": adf_pvalue,
            "kpss_pvalue": kpss_pvalue,
            "vol_multiplier": vol_multiplier,
        }

        current_r = float(r[-1])
        proj_days = np.arange(1, OU_PROJECTION_DAYS + 1)
        # Project using the dynamic theta so the charted curve reflects current market speed
        self.ou_projection = mu + (current_r - mu) * np.exp(-dynamic_theta * proj_days)

    def _compute_hurst(self) -> None:
        """Hurst exponent on OOS residuals."""
        oos_r = self.residuals[MIN_TRAIN_SIZE:]
        self.hurst = hurst_rs(oos_r) if len(oos_r) > 30 else 0.5


# ══════════════════════════════════════════════════════════════════════════════
# DATA UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, max_entries=5)
def load_google_sheet(sheet_url: str) -> tuple[pd.DataFrame | None, str | None]:
    """Extract sheet ID and GID from a Google Sheets URL, fetch as CSV."""
    try:
        sheet_id_match = re.search(r"/d/([a-zA-Z0-9-_]+)", sheet_url)
        if not sheet_id_match:
            return None, "Invalid Google Sheets URL — could not extract sheet ID."
        sheet_id = sheet_id_match.group(1)
        gid_match = re.search(r"gid=(\d+)", sheet_url)
        gid = gid_match.group(1) if gid_match else "0"
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
        df = pd.read_csv(csv_url)
        return df, None
    except Exception as e:
        return None, str(e)


def clean_data(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    date_col: str | None = None,
) -> pd.DataFrame:
    """Select, coerce, and clean numeric columns; optionally parse dates."""
    features = [f for f in features if f != target]
    cols = [target] + features
    if date_col and date_col in df.columns:
        cols.append(date_col)

    # FIX: Intersect requested columns with available columns to prevent 
    # fatal KeyErrors if the remote Google Sheet schema changes while state persists.
    valid_cols = [c for c in cols if c in df.columns]
    if target not in valid_cols:
        return pd.DataFrame()  # Graceful degradation

    data = df[valid_cols].copy()
    
    # Handle dates first to ensure chronological sorting before imputation
    if date_col and date_col in data.columns:
        try:
            data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
            data = data.dropna(subset=[date_col]).sort_values(date_col)
        except Exception:
            pass

    for col in [target] + features:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Forward-fill to preserve dt=1 (propagate last known state), then back-fill leading NaNs
    numeric_cols = [target] + features
    data[numeric_cols] = data[numeric_cols].ffill().bfill()
    
    # Only drop rows if an entire column failed to fill (e.g., completely empty data)
    data = data.dropna(subset=numeric_cols)

    return data.reset_index(drop=True)


def apply_chart_theme(fig: go.Figure) -> None:
    """Apply the Hemrek dark theme to any Plotly figure (mutates in place)."""
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor=CHART_BG,
        paper_bgcolor=CHART_BG,
        font=dict(family="Inter", color=CHART_FONT_COLOR),
        margin=dict(t=40, l=20, r=20, b=20),
        hoverlabel=dict(bgcolor=CHART_BG, font_size=12),
    )
    fig.update_xaxes(gridcolor=CHART_GRID, zerolinecolor=CHART_ZEROLINE)
    fig.update_yaxes(gridcolor=CHART_GRID, zerolinecolor=CHART_ZEROLINE)


# ══════════════════════════════════════════════════════════════════════════════
# UI RENDERING
# ══════════════════════════════════════════════════════════════════════════════

def _render_header() -> None:
    st.markdown("""
    <div class="premium-header">
        <h1>AARAMBH : Fair Value Breadth</h1>
        <div class="tagline">Walk-Forward Valuation · OU Mean-Reversion · Kalman Conviction | Quantitative Reversal Analysis</div>
    </div>
    """, unsafe_allow_html=True)


def _render_landing_page() -> None:
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class='metric-card purple' style='min-height: 280px; justify-content: flex-start;'>
            <h3 style='color: var(--purple); margin-bottom: 0.5rem;'>🎯 Walk-Forward Fair Value</h3>
            <p style='color: var(--text-muted); font-size: 0.9rem; line-height: 1.6;'>
                v2.0 uses expanding-window regression: at each time T, the model only sees data up to T.
                No look-ahead bias. The R² you see is out-of-sample.
            </p>
            <br>
            <p style='color: var(--text-secondary); font-size: 0.85rem;'>
                <strong>Ensemble:</strong> Ridge + Huber + OLS<br>
                <strong>Validation:</strong> Walk-forward OOS<br>
                <strong>Uncertainty:</strong> Model disagreement
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='metric-card info' style='min-height: 280px; justify-content: flex-start;'>
            <h3 style='color: var(--info-cyan); margin-bottom: 0.5rem;'>📉 OU Mean-Reversion</h3>
            <p style='color: var(--text-muted); font-size: 0.9rem; line-height: 1.6;'>
                Residuals are modeled as Ornstein-Uhlenbeck: dx = θ(μ−x)dt + σdW.
                The half-life tells you when the gap is expected to close.
            </p>
            <br>
            <p style='color: var(--text-secondary); font-size: 0.85rem;'>
                <strong>Output:</strong> Half-life in days<br>
                <strong>Projection:</strong> 90-day forward path<br>
                <strong>Validation:</strong> Hurst exponent
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='metric-card primary' style='min-height: 280px; justify-content: flex-start;'>
            <h3 style='color: var(--primary-color); margin-bottom: 0.5rem;'>📊 Kalman Conviction</h3>
            <p style='color: var(--text-muted); font-size: 0.9rem; line-height: 1.6;'>
                Breadth conviction is Kalman-filtered. Noisy raw scores become smooth signals
                with 95% confidence bands. Wide band = uncertain.
            </p>
            <br>
            <p style='color: var(--text-secondary); font-size: 0.85rem;'>
                <strong>Lookbacks:</strong> 5D, 10D, 20D, 50D, 100D<br>
                <strong>Smoothing:</strong> Adaptive Kalman<br>
                <strong>Range:</strong> -100 to +100
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
        <h4>🚀 Getting Started</h4>
        <p>Use the <strong>Sidebar</strong> to load data (CSV/Excel or Google Sheet).
        Select a <strong>Target</strong> and <strong>Predictors</strong>, then click <strong>Apply</strong> to run the walk-forward engine.</p>
    </div>
    """, unsafe_allow_html=True)


def _render_footer() -> None:
    utc_now = datetime.now(timezone.utc)
    ist_now = utc_now + timedelta(hours=5, minutes=30)
    current_time_ist = ist_now.strftime("%Y-%m-%d %H:%M:%S IST")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.caption(f"© {ist_now.year} {PRODUCT_NAME} | {COMPANY} | {VERSION} | {current_time_ist}")


def _render_metric_card(label: str, value: str, sub: str, color_class: str = "neutral") -> None:
    esc = html.escape
    st.markdown(
        f'<div class="metric-card {esc(color_class)}">'
        f"<h4>{esc(label)}</h4><h2>{esc(value)}</h2>"
        f'<div class="sub-metric">{esc(sub)}</div></div>',
        unsafe_allow_html=True,
    )


# ── Tab: Regime Analysis ─────────────────────────────────────────────────

def _render_tab_regime(engine, ts_filtered, x_axis, x_title, signal, model_stats, regime_stats, ts) -> None:
    st.markdown("##### Kalman Conviction Score")
    st.markdown(
        '<p style="color: #888;">Negative = Oversold bias | Positive = Overbought bias · '
        "Shaded = 95% Kalman confidence band</p>",
        unsafe_allow_html=True,
    )

    fig_conv = go.Figure()

    if "ConvictionUpper" in ts_filtered.columns:
        fig_conv.add_trace(go.Scatter(
            x=x_axis, y=ts_filtered["ConvictionUpper"],
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig_conv.add_trace(go.Scatter(
            x=x_axis, y=ts_filtered["ConvictionLower"],
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
            fill="tonexty", fillcolor="rgba(255,195,0,0.08)", name="95% Band",
        ))

    fig_conv.add_trace(go.Scatter(
        x=x_axis, y=ts_filtered["ConvictionScore"].clip(lower=0),
        fill="tozeroy", fillcolor="rgba(239,68,68,0.15)", line=dict(width=0), showlegend=False,
    ))
    fig_conv.add_trace(go.Scatter(
        x=x_axis, y=ts_filtered["ConvictionScore"].clip(upper=0),
        fill="tozeroy", fillcolor="rgba(16,185,129,0.15)", line=dict(width=0), showlegend=False,
    ))
    fig_conv.add_trace(go.Scatter(
        x=x_axis, y=ts_filtered["ConvictionScore"], mode="lines", name="Conviction (Kalman)",
        line=dict(color=COLOR_GOLD, width=2),
    ))

    if "ConvictionRaw" in ts_filtered.columns:
        fig_conv.add_trace(go.Scatter(
            x=x_axis, y=ts_filtered["ConvictionRaw"], mode="lines", name="Raw Conviction",
            line=dict(color="#555", width=1, dash="dot"), opacity=0.5,
        ))

    fig_conv.add_hline(y=40, line_dash="dash", line_color="rgba(239,68,68,0.5)")
    fig_conv.add_hline(y=-40, line_dash="dash", line_color="rgba(16,185,129,0.5)")
    fig_conv.add_hline(y=0, line_color="rgba(255,255,255,0.3)")
    fig_conv.update_layout(
        title="Conviction Score (Kalman-Filtered)", height=400,
        xaxis_title=x_title, yaxis_title="Score", yaxis=dict(range=[-100, 100]),
    )
    apply_chart_theme(fig_conv)
    st.plotly_chart(fig_conv, width="stretch")

    st.markdown("---")
    st.markdown("##### Base Conviction Score")
    st.markdown(
        '<p style="color: #888;">Negative = Oversold bias | Positive = Overbought bias</p>',
        unsafe_allow_html=True,
    )

    if "ConvictionRaw" in ts_filtered.columns:
        fig_raw = go.Figure()
        fig_raw.add_trace(go.Scatter(
            x=x_axis, y=ts_filtered["ConvictionRaw"].clip(lower=0),
            fill="tozeroy", fillcolor="rgba(239,68,68,0.15)", line=dict(width=0), showlegend=False,
        ))
        fig_raw.add_trace(go.Scatter(
            x=x_axis, y=ts_filtered["ConvictionRaw"].clip(upper=0),
            fill="tozeroy", fillcolor="rgba(16,185,129,0.15)", line=dict(width=0), showlegend=False,
        ))
        conv_colors = [
            COLOR_GREEN if c < -40 else COLOR_RED if c > 40 else COLOR_MUTED
            for c in ts_filtered["ConvictionRaw"]
        ]
        fig_raw.add_trace(go.Scatter(
            x=x_axis, y=ts_filtered["ConvictionRaw"], mode="lines+markers", name="Raw Conviction",
            line=dict(color=COLOR_GOLD, width=2), marker=dict(size=4, color=conv_colors),
        ))
        fig_raw.add_hline(y=40, line_dash="dash", line_color="rgba(239,68,68,0.5)")
        fig_raw.add_hline(y=-40, line_dash="dash", line_color="rgba(16,185,129,0.5)")
        fig_raw.add_hline(y=0, line_color="rgba(255,255,255,0.3)")
        fig_raw.update_layout(
            title="Base Conviction Score", height=400,
            xaxis_title=x_title, yaxis_title="Score", yaxis=dict(range=[-100, 100]),
        )
        apply_chart_theme(fig_raw)
        st.plotly_chart(fig_raw, width="stretch")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Regime Distribution")
        total = len(ts)
        regime_data = {
            "Regime": [
                "🟢 Strongly Oversold", "🔵 Oversold", "⚪ Neutral",
                "🟠 Overbought", "🔴 Strongly Overbought",
            ],
            "Count": [
                regime_stats["strongly_oversold"], regime_stats["oversold"],
                regime_stats["neutral"], regime_stats["overbought"],
                regime_stats["strongly_overbought"],
            ],
            "Pct": [
                f"{regime_stats['strongly_oversold'] / total * 100:.1f}%",
                f"{regime_stats['oversold'] / total * 100:.1f}%",
                f"{regime_stats['neutral'] / total * 100:.1f}%",
                f"{regime_stats['overbought'] / total * 100:.1f}%",
                f"{regime_stats['strongly_overbought'] / total * 100:.1f}%",
            ],
        }
        st.dataframe(pd.DataFrame(regime_data), width="stretch", hide_index=True)

    with col2:
        st.markdown("##### Current Regime & Diagnostics")
        curr_regime = signal["regime"]
        box_class = "success" if "OVERSOLD" in curr_regime else "danger" if "OVERBOUGHT" in curr_regime else ""
        if "OVERSOLD" in curr_regime:
            regime_desc = "Multiple timeframes showing oversold conditions — historically a buying opportunity."
        elif "OVERBOUGHT" in curr_regime:
            regime_desc = "Multiple timeframes showing overbought conditions — historically a selling opportunity."
        else:
            regime_desc = "No strong directional bias across timeframes."

        st.markdown(
            f'<div class="guide-box {html.escape(box_class)}">'
            f"<strong>Current: {html.escape(curr_regime)}</strong><br><br>{html.escape(regime_desc)}</div>",
            unsafe_allow_html=True,
        )

        st.markdown("##### Model Diagnostics")
        h_label = (
            "Mean-Reverting ✅" if signal["hurst"] < 0.40
            else "Trending ⚠️" if signal["hurst"] > 0.60
            else "Random Walk"
        )
        r2_rw = model_stats.get("r2_vs_rw", 0.0)
        st.markdown(
            f"OOS R²: **{model_stats['r2_oos']:.4f}** | "
            f"R² vs RW: **{r2_rw:+.4f}** | "
            f"RMSE: **{model_stats['rmse_oos']:.4f}** | "
            f"OU Half-Life: **{signal['ou_half_life']:.0f} days** | "
            f"Hurst: **{signal['hurst']:.2f}** ({h_label}) | "
            f"Model Spread: **{model_stats['avg_model_spread']:.3f}**"
        )
    
    st.markdown(
        f'<div style="margin-top: 1.5rem; padding: 1.25rem; border-left: 3px solid {COLOR_CYAN}; background: rgba(6, 182, 212, 0.05); border-radius: 8px;">'
        f'<h4 style="color: {COLOR_CYAN}; font-size: 0.95rem; margin-top: 0; margin-bottom: 0.5rem; font-weight: 700;">🧠 Time-Weighted Fair Value</h4>'
        f'<p style="color: var(--text-muted); font-size: 0.85rem; margin: 0; line-height: 1.6;">'
        f'The underlying mathematical engine utilizes <b>Exponential Decay Weighting</b> (252-day half-life) for its regression ensemble. '
        f'This mathematically forces the models to prioritize the current market regime and safely '
        f'discount distant historical relationships.</p></div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("##### Current Fair Value Drivers")
    st.markdown('<p style="color: #888;">Relative % contribution of each macro predictor to the current WLS Fair Value.</p>', unsafe_allow_html=True)
    
    if hasattr(engine, "latest_feature_impacts") and engine.latest_feature_impacts:
        impacts = engine.latest_feature_impacts
        # Reverse for horizontal bar chart (top impacts at the top)
        labels = list(impacts.keys())[::-1]
        vals = list(impacts.values())[::-1]
        
        fig_imp = go.Figure(go.Bar(
            x=vals, y=labels, orientation="h",
            marker=dict(color=COLOR_CYAN, opacity=0.8)
        ))
        fig_imp.update_layout(
            height=max(300, len(labels) * 35),
            xaxis_title="% Relative Contribution to Model Variance",
        )
        apply_chart_theme(fig_imp)
        st.plotly_chart(fig_imp, width="stretch")
    else:
        st.info("Feature impact data is not available for the current ensemble configuration.")


# ── Tab: Signal Dashboard ────────────────────────────────────────────────

def _render_tab_signal(
    engine, ts_filtered, x_axis, x_title, signal, active_target, ts,
) -> None:
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("##### Current Signal Analysis")
        signal_class = (
            "undervalued" if signal["signal"] == "BUY"
            else "overvalued" if signal["signal"] == "SELL"
            else "fair"
        )
        signal_emoji = "🟢" if signal["signal"] == "BUY" else "🔴" if signal["signal"] == "SELL" else "🟡"

        st.markdown(f"""
        <div class="signal-card {html.escape(signal_class)}">
            <div class="label">WALK-FORWARD SIGNAL</div>
            <div class="value">{signal_emoji} {html.escape(signal['signal'])}</div>
            <div class="subtext">{html.escape(signal['strength'])} Strength • {html.escape(signal['confidence'])} Confidence •
            OU t½ = {signal['ou_half_life']:.0f}d</div>
        </div>
        """, unsafe_allow_html=True)

        conv_pct = (signal["conviction_score"] + 100) / 2
        conv_color = (
            COLOR_GREEN if signal["conviction_score"] < -20
            else COLOR_RED if signal["conviction_score"] > 20
            else COLOR_GOLD
        )

        st.markdown(f"""
        <div class="conviction-meter">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="color: {COLOR_GREEN}; font-size: 0.75rem;">OVERSOLD</span>
                <span style="color: #888; font-size: 0.75rem;">Conviction: {signal['conviction_score']:+.0f} [{signal['conviction_lower']:+.0f}, {signal['conviction_upper']:+.0f}]</span>
                <span style="color: {COLOR_RED}; font-size: 0.75rem;">OVERBOUGHT</span>
            </div>
            <div class="conviction-bar">
                <div class="conviction-fill" style="width: {conv_pct}%; background: {conv_color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if signal["has_bullish_div"]:
            st.markdown(
                '<span class="status-badge buy">🔔 BULLISH DIVERGENCE (Swing-Based)</span>',
                unsafe_allow_html=True,
            )
        if signal["has_bearish_div"]:
            st.markdown(
                '<span class="status-badge sell">🔔 BEARISH DIVERGENCE (Swing-Based)</span>',
                unsafe_allow_html=True,
            )

        if signal["model_spread"] > 1.0:
            st.markdown(f"""
            <div style="background: rgba(245,158,11,0.1); border: 1px solid {COLOR_AMBER}; border-radius: 8px; padding: 0.5rem 1rem; margin-top: 0.5rem;">
                <span style="color: {COLOR_AMBER}; font-size: 0.8rem;">⚠️ High model disagreement ({signal['model_spread']:.2f}) — fair value estimate is uncertain. Signal confidence may be lower than indicated.</span>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        st.markdown("##### Lookback Breakdown")
        for lb in LOOKBACK_WINDOWS:
            if f"Z_{lb}" not in ts.columns:
                continue
            z = ts[f"Z_{lb}"].iloc[-1]
            zone = ts[f"Zone_{lb}"].iloc[-1]
            zone_color = COLOR_GREEN if "Under" in zone else COLOR_RED if "Over" in zone else COLOR_MUTED
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; padding: 0.5rem; border-bottom: 1px solid #2A2A2A;">
                <span style="color: #888;">{lb}-Day</span>
                <span style="color: {zone_color}; font-weight: 600;">{zone} ({z:+.2f}σ)</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### Actual vs Walk-Forward Fair Value")

    fig = make_subplots(rows=2, cols=1, row_heights=[0.6, 0.4], shared_xaxes=True, vertical_spacing=0.05)

    fig.add_trace(go.Scatter(
        x=x_axis, y=ts_filtered["Actual"], mode="lines", name="Actual",
        line=dict(color=COLOR_GOLD, width=2),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x_axis, y=ts_filtered["FairValue"], mode="lines", name="Fair Value (OOS)",
        line=dict(color=COLOR_CYAN, width=2, dash="dash"),
    ), row=1, col=1)

    if "ModelSpread" in ts_filtered.columns:
        upper = ts_filtered["FairValue"] + ts_filtered["ModelSpread"]
        lower = ts_filtered["FairValue"] - ts_filtered["ModelSpread"]
        fig.add_trace(go.Scatter(
            x=x_axis, y=upper, mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=x_axis, y=lower, mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(6,182,212,0.08)",
            name="Model Uncertainty", hoverinfo="skip",
        ), row=1, col=1)

    bar_colors = [COLOR_GREEN if r < 0 else COLOR_RED for r in ts_filtered["Residual"]]
    fig.add_trace(go.Bar(
        x=x_axis, y=ts_filtered["Residual"], name="Residual (OOS)",
        marker_color=bar_colors, showlegend=False,
    ), row=2, col=1)
    fig.add_hline(y=0, line_color=COLOR_GOLD, line_width=1, row=2, col=1)

    if hasattr(engine, "ou_projection") and pd.api.types.is_datetime64_any_dtype(ts["Date"]):
        last_date = ts["Date"].iloc[-1]
        proj_dates = pd.bdate_range(start=last_date, periods=OU_PROJECTION_DAYS + 1)[1:]
        fig.add_trace(go.Scatter(
            x=proj_dates, y=engine.ou_projection,
            mode="lines", name="OU Projection",
            line=dict(color=COLOR_GOLD, width=1.5, dash="dot"), opacity=0.5,
        ), row=2, col=1)

    fig.update_layout(height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02))
    fig.update_yaxes(title_text=active_target, row=1, col=1)
    fig.update_yaxes(title_text="Residual", row=2, col=1)
    apply_chart_theme(fig)
    st.plotly_chart(fig, width="stretch")


# ── Tab: Zone Trends ─────────────────────────────────────────────────────

def _render_tab_zones(ts_filtered, x_axis, x_title) -> None:
    st.markdown("##### Overbought / Oversold Breadth Over Time")
    st.markdown(
        '<p style="color: #888;">% of lookback periods in oversold/overbought zones</p>',
        unsafe_allow_html=True,
    )

    fig_zones = go.Figure()
    fig_zones.add_trace(go.Scatter(
        x=x_axis, y=ts_filtered["OversoldBreadth"],
        fill="tozeroy", fillcolor="rgba(16,185,129,0.2)",
        line=dict(color=COLOR_GREEN, width=2), name="Oversold %",
    ))
    fig_zones.add_trace(go.Scatter(
        x=x_axis, y=ts_filtered["OverboughtBreadth"],
        fill="tozeroy", fillcolor="rgba(239,68,68,0.2)",
        line=dict(color=COLOR_RED, width=2), name="Overbought %",
    ))
    fig_zones.add_hline(y=60, line_dash="dash", line_color="rgba(255,195,0,0.3)")
    fig_zones.update_layout(
        title="Zone Breadth", height=400,
        xaxis_title=x_title, yaxis_title="% of Lookbacks", yaxis=dict(range=[0, 100]),
    )
    apply_chart_theme(fig_zones)
    st.plotly_chart(fig_zones, width="stretch")

    st.markdown("---")
    st.markdown("##### Average Z-Score Across Lookbacks")

    fig_z = go.Figure()
    z_colors = [COLOR_GREEN if z < -1 else COLOR_RED if z > 1 else COLOR_MUTED for z in ts_filtered["AvgZ"]]
    fig_z.add_trace(go.Bar(x=x_axis, y=ts_filtered["AvgZ"], marker_color=z_colors, name="Avg Z"))
    fig_z.add_hline(y=0, line_color=COLOR_GOLD, line_width=1)
    fig_z.add_hline(y=2, line_dash="dash", line_color="rgba(239,68,68,0.5)")
    fig_z.add_hline(y=-2, line_dash="dash", line_color="rgba(16,185,129,0.5)")
    fig_z.update_layout(
        title="Multi-Lookback Average Z-Score", height=350,
        xaxis_title=x_title, yaxis_title="Z-Score",
    )
    apply_chart_theme(fig_z)
    st.plotly_chart(fig_z, width="stretch")


# ── Tab: Signal Trends ───────────────────────────────────────────────────

def _render_tab_signals(engine, ts_filtered, x_axis, x_title) -> None:
    st.markdown("##### Buy/Sell Signal Count by Period")

    fig_signals = go.Figure()
    fig_signals.add_trace(go.Bar(
        x=x_axis, y=ts_filtered["BuySignalBreadth"], name="Buy Signals",
        marker=dict(color=COLOR_GREEN),
    ))
    fig_signals.add_trace(go.Bar(
        x=x_axis, y=-ts_filtered["SellSignalBreadth"], name="Sell Signals",
        marker=dict(color=COLOR_RED),
    ))
    fig_signals.update_layout(
        title="Signal Count by Period", height=350,
        xaxis_title=x_title, yaxis_title="Signal Count", barmode="relative",
    )
    apply_chart_theme(fig_signals)
    st.plotly_chart(fig_signals, width="stretch")

    st.markdown("---")
    st.markdown("##### Signal Statistics")

    perf = engine.get_signal_performance()
    perf_rows = []
    for period in (5, 10, 20):
        p = perf[period]
        perf_rows.append({
            "Holding Period": f"{period} Days",
            "Buy Hit Rate": f"{p['buy_hit'] * 100:.1f}%" if p["buy_count"] > 0 else "N/A",
            "Buy Avg Fwd Chg": f"{p['buy_avg']:.2f}%" if p["buy_count"] > 0 else "N/A",
            "Buy Count": p["buy_count"],
            "Sell Hit Rate": f"{p['sell_hit'] * 100:.1f}%" if p["sell_count"] > 0 else "N/A",
            "Sell Avg Fwd Chg": f"{p['sell_avg']:.2f}%" if p["sell_count"] > 0 else "N/A",
            "Sell Count": p["sell_count"],
        })
    st.dataframe(pd.DataFrame(perf_rows), width="stretch", hide_index=True)


# ── Tab: Data Table ──────────────────────────────────────────────────────

def _render_tab_data(ts_filtered, ts, active_target) -> None:
    st.markdown(f"##### Time Series Data ({len(ts_filtered)} observations)")

    display_cols = [
        "Date", "Actual", "FairValue", "Residual", "ModelSpread", "AvgZ",
        "OversoldBreadth", "OverboughtBreadth", "ConvictionScore", "Regime",
        "BullishDiv", "BearishDiv",
    ]
    display_cols = [c for c in display_cols if c in ts_filtered.columns]

    display_df = ts_filtered[display_cols].copy()
    rounding = {
        "AvgZ": 3, "ModelSpread": 3, "FairValue": 2,
        "Residual": 1, "ConvictionScore": 1, "OversoldBreadth": 1, "OverboughtBreadth": 1,
    }
    for col, decimals in rounding.items():
        if col in display_df.columns:
            display_df[col] = display_df[col].round(decimals)

    if "BullishDiv" in display_df.columns:
        display_df["BullishDiv"] = display_df["BullishDiv"].apply(lambda x: "🟢" if x else "")
    if "BearishDiv" in display_df.columns:
        display_df["BearishDiv"] = display_df["BearishDiv"].apply(lambda x: "🔴" if x else "")

    st.dataframe(display_df, width="stretch", hide_index=True, height=500)

    csv_data = ts.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Full CSV", csv_data,
        f"aarambh_{active_target}_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv",
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Sidebar: Data Source ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; margin-bottom: 1rem;">
            <div style="font-size: 1.75rem; font-weight: 800; color: #FFC300;">AARAMBH</div>
            <div style="color: #888888; font-size: 0.75rem; margin-top: 0.25rem;">आरंभ | Fair Value Breadth</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-title">📁 Data Source</div>', unsafe_allow_html=True)
        data_source = st.radio(
            "Source", ["📤 Upload", "📊 Google Sheets"],
            horizontal=True, label_visibility="collapsed",
        )

        df = None

        if data_source == "📤 Upload":
            uploaded_file = st.file_uploader("CSV/Excel", type=["csv", "xlsx"], label_visibility="collapsed")
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
                except Exception as e:
                    st.error(f"Error: {e}")
                    return
        else:
            sheet_url = st.text_input("Sheet URL", value=DEFAULT_SHEET_URL, label_visibility="collapsed")
            
            # Auto-load default dataset on initial startup to skip manual interaction
            if "data" not in st.session_state and sheet_url == DEFAULT_SHEET_URL:
                with st.spinner("Auto-loading default data..."):
                    df, error = load_google_sheet(sheet_url)
                    if not error:
                        st.session_state["data"] = df
                        st.toast("Default data auto-loaded!", icon="⚡")

            if st.button("🔄 LOAD DATA", type="primary"):
                with st.spinner("Loading..."):
                    df, error = load_google_sheet(sheet_url)
                    if error:
                        st.error(f"Failed: {error}")
                        return
                    # Immediately free large session state properties to prevent memory spike
                    st.session_state.pop("data", None)
                    st.session_state.pop("engine", None)
                    st.session_state.pop("engine_cache", None)
                    st.session_state["data"] = df
                    st.toast("Data loaded successfully!", icon="✅")
            if "data" in st.session_state:
                df = st.session_state["data"]

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Landing page when no data loaded ──────────────────────────────────
    if df is None:
        _render_header()
        _render_landing_page()
        _render_footer()
        return

    # ── Sidebar: Model Configuration ──────────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = df.columns.tolist()

    if len(numeric_cols) < 2:
        st.error("Need 2+ numeric columns.")
        return

    with st.sidebar:
        st.markdown('<div class="sidebar-title">🧠 Model Configuration</div>', unsafe_allow_html=True)

        default_target = "NIFTY50_PE" if "NIFTY50_PE" in numeric_cols else numeric_cols[0]
        active_target_state = st.session_state.get("active_target", default_target)
        if active_target_state not in numeric_cols:
            active_target_state = numeric_cols[0]

        target_col = st.selectbox(
            "Target Variable", numeric_cols,
            index=numeric_cols.index(active_target_state),
        )

        date_candidates = [c for c in all_cols if "date" in c.lower()]
        default_date = date_candidates[0] if date_candidates else "None"
        active_date_state = st.session_state.get("active_date_col", default_date)
        if active_date_state not in ["None"] + all_cols:
            active_date_state = "None"

        date_col = st.selectbox(
            "Date Column", ["None"] + all_cols,
            index=(["None"] + all_cols).index(active_date_state),
        )

        available = [c for c in numeric_cols if c != target_col]
        valid_defaults = [p for p in DEFAULT_PREDICTORS if p in available]

        if "active_features" not in st.session_state:
            st.session_state["active_features"] = tuple(valid_defaults or available[:3])

        with st.expander("Predictor Columns", expanded=False):
            st.caption("Select predictors, then click Apply to recompute.")

            staging_features = st.multiselect(
                "Predictor Columns", options=available,
                default=[f for f in st.session_state["active_features"] if f in available],
                label_visibility="collapsed",
                help="These columns are used as predictors for walk-forward fair value regression.",
            )

            if not staging_features:
                st.warning("⚠️ Select at least one predictor.")
                staging_features = [f for f in st.session_state["active_features"] if f in available]

            staging_set = set(staging_features)
            active_set = set(st.session_state["active_features"])
            has_pred_changes = staging_set != active_set
            has_other_changes = (target_col != active_target_state) or (date_col != active_date_state)
            has_changes = has_pred_changes or has_other_changes

            if has_pred_changes:
                added = staging_set - active_set
                removed = active_set - staging_set
                parts = []
                if added:
                    parts.append(f"+{len(added)} added")
                if removed:
                    parts.append(f"−{len(removed)} removed")
                st.caption(f"Pending: {', '.join(parts)}")
            elif has_other_changes:
                st.caption("Pending: Target/Date changes")

            apply_clicked = st.button(
                "✅ Apply Configuration" if has_changes else "No changes",
                width="stretch",
                disabled=not has_changes,
                type="primary" if has_changes else "secondary",
            )

            if apply_clicked and has_changes:
                st.session_state["active_target"] = target_col
                st.session_state["active_features"] = tuple(staging_features)
                st.session_state["active_date_col"] = date_col
                st.session_state.pop("engine", None)
                st.session_state.pop("engine_cache", None)
                st.rerun()

            active_count = len(st.session_state["active_features"])
            total_count = len(available)
            if active_count != total_count:
                st.info(f"Active: {active_count}/{total_count} predictors")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class='info-box'>
            <p style='font-size: 0.8rem; margin: 0; color: var(--text-muted); line-height: 1.5;'>
                <strong>Version:</strong> {VERSION}<br>
                <strong>Engine:</strong> Walk-Forward · OU · Kalman<br>
                <strong>Lookbacks:</strong> {', '.join(f'{lb}D' for lb in LOOKBACK_WINDOWS)}
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── Resolve active configuration ──────────────────────────────────────
    active_target = st.session_state.get("active_target", target_col)
    active_features = list(st.session_state.get("active_features", staging_features))
    active_date = st.session_state.get("active_date_col", date_col)

    # ── Header ────────────────────────────────────────────────────────────
    _render_header()

    # ── Data staleness warning ────────────────────────────────────────────
    if active_date != "None" and active_date in df.columns:
        try:
            dates = pd.to_datetime(df[active_date], errors="coerce").dropna()
            if len(dates) > 0:
                latest_date = dates.max().to_pydatetime()
                if latest_date.tzinfo is not None:
                    latest_date = latest_date.replace(tzinfo=None)
                now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
                data_age = (now_utc - latest_date).days
                if data_age > STALENESS_DAYS:
                    st.markdown(f"""
                    <div style="background: rgba(239,68,68,0.1); border: 1px solid {COLOR_RED}; border-radius: 10px;
                                padding: 0.75rem 1.25rem; margin-bottom: 1rem; display: flex; align-items: center; gap: 12px;">
                        <span style="font-size: 1.4rem;">⚠️</span>
                        <div>
                            <span style="color: {COLOR_RED}; font-weight: 700;">Stale Data</span>
                            <span style="color: #888; font-size: 0.85rem;"> — Last data point is <b>{latest_date.strftime('%d %b %Y')}</b> ({data_age} days ago). Update your data source.</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        except Exception:
            pass

    # ── Clean & Fit Engine ────────────────────────────────────────────────
    data = clean_data(df, active_target, active_features, active_date if active_date != "None" else None)

    if len(data) < MIN_DATA_POINTS:
        st.error(f"Need {MIN_DATA_POINTS}+ data points for walk-forward analysis.")
        return

    X = data[active_features].values
    y = data[active_target].values
    
    # Factor in the latest date or value to ensure in-place sheet updates trigger a re-run
    if active_date != "None" and active_date in data.columns:
        latest_sig = str(data[active_date].max())
    else:
        latest_sig = str(np.sum(y))  # Fallback if no date col exists
    cache_key = f"{active_target}|{'|'.join(sorted(active_features))}|{len(data)}|{latest_sig}"

    if st.session_state.get("engine_cache") != cache_key:
        with st.spinner("Preparing walk-forward engine..."):
            # Remove old engine from memory before fitting a new one
            if "engine" in st.session_state:
                del st.session_state["engine"]
            progress_bar = st.progress(0, text="Initializing engine...")
            engine = FairValueEngine()
            engine.fit(X, y, feature_names=active_features, progress_callback=progress_bar.progress)
            st.session_state["engine"] = engine
            st.session_state["engine_cache"] = cache_key
            progress_bar.empty()

    engine: FairValueEngine = st.session_state["engine"]
    signal = engine.get_current_signal()
    model_stats = engine.get_model_stats()
    regime_stats = engine.get_regime_stats()
    ts = engine.ts_data.copy()

    if active_date != "None" and active_date in data.columns:
        ts["Date"] = pd.to_datetime(data[active_date].values)
    else:
        ts["Date"] = np.arange(len(ts))

    # ── Metric Cards ──────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 2])

    with c1:
        color = "success" if signal["oversold_breadth"] > 60 else "neutral"
        _render_metric_card("Oversold", f'{signal["oversold_breadth"]:.0f}%', "Lookbacks in Zone", color)
    with c2:
        color = "danger" if signal["overbought_breadth"] > 60 else "neutral"
        _render_metric_card("Overbought", f'{signal["overbought_breadth"]:.0f}%', "Lookbacks in Zone", color)
    with c3:
        color = "success" if signal["conviction_score"] < -40 else "danger" if signal["conviction_score"] > 40 else "neutral"
        _render_metric_card("Conviction", f'{signal["conviction_score"]:+.0f}', "Kalman-Filtered", color)
    with c4:
        color = "success" if signal["signal"] == "BUY" else "danger" if signal["signal"] == "SELL" else "primary"
        _render_metric_card("Signal", signal["signal"], signal["strength"], color)
    with c5:
        color = "success" if "OVERSOLD" in signal["regime"] else "danger" if "OVERBOUGHT" in signal["regime"] else "neutral"
        _render_metric_card("Regime", signal["regime"], "Current State", color)

    # ── Diagnostics Row ───────────────────────────────────────────────────
    d1, d2, d3, d4, d5, d6 = st.columns(6)
    with d1:
        _render_metric_card("OU Half-Life", f'{signal["ou_half_life"]:.0f}d', "Bias-Corrected", "primary")
    with d2:
        h = signal["hurst"]
        h_label = "Trending" if h > 0.60 else "Random" if h > 0.40 else "Mean-Reverting"
        h_class = "danger" if h > 0.60 else "neutral" if h > 0.40 else "success"
        _render_metric_card("Residual Hurst", f"{h:.2f}", h_label, h_class)
    with d3:
        r2 = model_stats["r2_oos"]
        r2_class = "success" if r2 > 0.7 else "warning" if r2 > 0.4 else "danger"
        _render_metric_card("OOS R²", f"{r2:.3f}", "Walk-Forward", r2_class)
    with d4:
        # R² vs Random Walk (Welch & Goyal, 2008): the honest benchmark for persistent series.
        r2_rw = model_stats.get("r2_vs_rw", 0.0)
        rw_class = "success" if r2_rw > 0.05 else "warning" if r2_rw > -0.05 else "danger"
        _render_metric_card("R² vs RW", f"{r2_rw:.3f}", "Vs Random Walk", rw_class)
    with d5:
        spread = model_stats["avg_model_spread"]
        sp_class = "success" if spread < 0.5 else "warning" if spread < 1.5 else "danger"
        _render_metric_card("Model Spread", f"{spread:.2f}", "Ensemble Std Dev", sp_class)
    with d6:
        # Stationarity: ADF (H₀=unit root) + KPSS (H₀=stationary)
        adf_p = signal.get("adf_pvalue", 1.0)
        kpss_p = signal.get("kpss_pvalue", 0.0)
        adf_ok = adf_p < 0.05
        kpss_ok = kpss_p > 0.05
        if adf_ok and kpss_ok:
            stat_label, stat_class = "Stationary", "success"
        elif not adf_ok and not kpss_ok:
            stat_label, stat_class = "Non-Stationary", "danger"
        else:
            stat_label, stat_class = "Inconclusive", "warning"
        _render_metric_card("Stationarity", stat_label, f"ADF {adf_p:.2f} | KPSS {kpss_p:.2f}", stat_class)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Timeframe Filter ──────────────────────────────────────────────────
    tf_col1, tf_col2 = st.columns([1, 6])
    with tf_col1:
        st.markdown("##### ⏱️ View Period")
    with tf_col2:
        selected_tf = st.radio(
            "Timeframe", ["1M", "6M", "1Y", "2Y", "ALL"],
            index=2, horizontal=True, label_visibility="collapsed",
        )

    ts_filtered = ts.copy()
    if selected_tf != "ALL":
        if active_date != "None" and pd.api.types.is_datetime64_any_dtype(ts["Date"]):
            max_date = ts["Date"].max()
            offsets = {
                "1M": pd.DateOffset(months=1), "6M": pd.DateOffset(months=6),
                "1Y": pd.DateOffset(years=1), "2Y": pd.DateOffset(years=2),
            }
            cutoff = max_date - offsets.get(selected_tf, pd.DateOffset(years=1))
            ts_filtered = ts[ts["Date"] >= cutoff]
        else:
            n_days = TIMEFRAME_TRADING_DAYS.get(selected_tf, 252)
            ts_filtered = ts.iloc[max(0, len(ts) - n_days) :]

    x_axis = ts_filtered["Date"]
    x_title = "Date" if active_date != "None" else "Index"

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab_regime, tab_signal, tab_zones, tab_signals, tab_data = st.tabs([
        "**🎯 Regime Analysis**",
        "**📊 Signal Dashboard**",
        "**📈 Zone Trends**",
        "**📉 Signal Trends**",
        "**📋 Data Table**",
    ])

    with tab_regime:
        _render_tab_regime(engine, ts_filtered, x_axis, x_title, signal, model_stats, regime_stats, ts)

    with tab_signal:
        _render_tab_signal(engine, ts_filtered, x_axis, x_title, signal, active_target, ts)

    with tab_zones:
        _render_tab_zones(ts_filtered, x_axis, x_title)

    with tab_signals:
        _render_tab_signals(engine, ts_filtered, x_axis, x_title)

    with tab_data:
        _render_tab_data(ts_filtered, ts, active_target)

    _render_footer()


if __name__ == "__main__":
    main()
