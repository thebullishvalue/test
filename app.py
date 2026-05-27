# -*- coding: utf-8 -*-
"""
ARTHAGATI (अर्थगति) - Market Sentiment Analysis | An @thebullishvalue Product
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Quantitative market mood analysis with MSF-enhanced indicators.
TradingView-style charting with institutional-grade analytics.
"""

import logging
import os
import sys
import time
from datetime import datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytz
import requests
import streamlit as st
from plotly.subplots import make_subplots

# ── Make ui/ importable when running `streamlit run arthagati.py`
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="ARTHAGATI | Market Sentiment Analysis",
    layout="wide",
    page_icon="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0iI0Q0QTg1MyIgZD0iTTEyIDJsMi40NCA3LjUySDIyTDE1Ljc4IDE0LjA0bDIuNDQgNy41Mkw1MTIgMTcuMDhsLTYuMjIgNC41MiAyLjQ0LTcuNTJMMiAyOS41MmgmNy41NnoiLz48L3N2Zz4=",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════════
# UI MODULE — Obsidian Quant design system
# ══════════════════════════════════════════════════════════════════════════════

from ui.theme import (
    inject_css,
    progress_bar,
    VERSION,
    PRODUCT_NAME,
    COMPANY,
    C_AMBER,
    C_AMBER_BRIGHT,
    C_CYAN,
    C_EMERALD,
    C_ROSE,
    C_MUTED,
    C_BG_CARD,
    C_TEXT,
    PLOTLY_BASE,
    PLOTLY_GRID,
)
from ui.components import (
    render_header,
    render_section_header,
    render_metric_card,
    render_info_box,
    render_warning_box,
    sidebar_title,
    sidebar_masthead,
    sidebar_passport,
    section_gap,
    section_divider,
    render_footer,
    get_icon,
)
from ui.tabs.tab_landing import render_landing_page
from ui.tabs.tab_historical_mood import render as render_historical_mood
from ui.tabs.tab_similar_periods import render as render_similar_periods
from ui.tabs.tab_correlation import render as render_correlation_analysis

# Legacy aliases kept inside arthagati.py for engine code paths
C_PRIMARY = C_AMBER
C_GREEN   = C_EMERALD
C_RED     = C_ROSE
C_AMBER_LEGACY = C_AMBER
C_BG_GRID = PLOTLY_GRID

# ══════════════════════════════════════════════════════════════════════════════
# DATA SOURCE
# ══════════════════════════════════════════════════════════════════════════════

# Google Sheets coordinates are read from environment variables.
# Set these in your deployment environment or .env file:
#   ARTHAGATI_SHEET_ID  = "<spreadsheet-id>"
#   ARTHAGATI_SHEET_GID = "<worksheet-gid>"
#
# The sheet must be accessible via the Google Visualization API (public with link).
# No service account authentication is needed — the gviz endpoint works without auth.

SHEET_ID  = os.environ.get("ARTHAGATI_SHEET_ID", "")
SHEET_GID = os.environ.get("ARTHAGATI_SHEET_GID", "")

EXPECTED_COLUMNS = [
    'DATE', 'NIFTY',
    'AD_RATIO', 'REL_AD_RATIO', 'REL_BREADTH', 'BREADTH', 'COUNT',
    'NIFTY50_PE', 'NIFTY50_EY', 'NIFTY50_DY', 'NIFTY50_PB',
    'IN10Y', 'IN02Y', 'IN30Y', 'INIRYY',
    'REPO', 'CRR',
    'US02Y', 'US10Y', 'US30Y', 'US_FED',
    'PE_DEV', 'EY_DEV',
]

DEPENDENT_VARS = [
    'AD_RATIO', 'REL_AD_RATIO', 'REL_BREADTH', 'BREADTH', 'COUNT',
    'IN10Y', 'IN02Y', 'IN30Y', 'INIRYY',
    'REPO', 'CRR',
    'US02Y', 'US10Y', 'US30Y', 'US_FED',
    'NIFTY50_DY', 'NIFTY50_PB',
    'PE_DEV', 'EY_DEV',
    'IN_TERM_SPREAD', 'US_TERM_SPREAD',  # derived yield-curve slopes
]

# Columns that are anchors or index keys, never predictors
NON_PREDICTOR_COLS: frozenset[str] = frozenset({'DATE', 'NIFTY', 'NIFTY50_PE', 'NIFTY50_EY'})

# Timeframe labels → calendar-day window (None = use all data / special handling)
TIMEFRAMES: dict[str, int | None] = {
    '1W':  7,
    '1M':  30,
    '3M':  90,
    '6M':  180,
    'YTD': None,   # computed at runtime from Jan 1
    '1Y':  365,
    '2Y':  730,
    '5Y':  1825,
    'MAX': None,   # all available rows
}

# Note: Colour palette + Plotly base are now sourced from ui.theme — see
# the imports above. C_PRIMARY/C_GREEN/C_RED/C_CYAN/C_MUTED/C_BG_CARD/C_BG_GRID
# remain in scope for engine code paths.

# ══════════════════════════════════════════════════════════════════════════════
# MODEL HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

# Correlation engine
CORR_HALF_LIFE  = 504    # ~2 trading years; exponential recency weight for Spearman
PCT_HALF_LIFE   = 252    # ~1 trading year;  recency weight for adaptive ECDF
MOOD_SCALE      = 30.0   # maps OU-normalised signal → mood score
KALMAN_CI_Z     = 1.96   # Kalman confidence band (≈95%)
KALMAN_HALF_LIFE = 126   # Kalman fading memory half-life (trading days, independent of PCT)
DATA_TTL        = 3600   # Streamlit cache TTL for the Google Sheets fetch (seconds)

# Walk-forward correlation rebalancing (eliminates look-ahead bias)
CORR_MIN_WARMUP       = 252   # minimum observations before first correlation checkpoint
CORR_REBALANCE_PERIOD = 63    # expanding-window rebalance interval (≈quarterly)

# MSF Spread indicator
MSF_WINDOW      = 20     # rolling window for all MSF components
MSF_ROC_LEN     = 14     # NIFTY rate-of-change period
MSF_ZSCORE_CLIP = 3.0    # Z-score clipping threshold
MSF_SCALE       = 10.0   # output scaling factor

# Similar-period finder
SIMILAR_W_MAHA  = 0.55   # Mahalanobis distance weight
SIMILAR_W_TRAJ  = 0.35   # trajectory cosine-similarity weight
SIMILAR_W_RECV  = 0.10   # recency decay weight
TRAJ_WINDOW     = 20     # trajectory comparison window (trading days)
BACKTEST_HORIZON = 30    # default forward-return horizon (trading days)

# Chart display
OU_PROJ_DAYS    = 90     # OU mean-reversion projection horizon (calendar days)

# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN LOOK-UP TABLES
# ══════════════════════════════════════════════════════════════════════════════

# Maps regime label → (hex colour, CSS card class)
REGIME_STYLES: dict[str, tuple[str, str]] = {
    'Trending':       (C_EMERALD,      'success'),
    'Volatile Trend': (C_AMBER,        'warning'),
    'Mean-Reverting': (C_CYAN,         'info'),
    'Choppy':         (C_ROSE,         'danger'),
    'Unknown':        (C_MUTED,        'neutral'),
}

# PLOTLY_BASE imported from ui.theme (transparent paper/plot + JetBrains Mono).

# ══════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — injected from ui/theme.css (Obsidian Quant)
# ══════════════════════════════════════════════════════════════════════════════

inject_css()


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

# Legacy alias — engine code below calls _progress_bar(); route to the new
# Obsidian Quant progress-card renderer from ui.theme.
_progress_bar = progress_bar


def sigmoid(x, scale=1.0):
    """Sigmoid normalization to [-1, 1] range — overflow-safe."""
    z = np.clip(np.asarray(x, dtype=np.float64) / max(scale, 1e-12), -500, 500)
    return 2.0 / (1.0 + np.exp(-z)) - 1.0

def rolling_mean_fast(series, window):
    """O(N) rolling mean using numpy cumsums — NaN-aware (NaN values excluded from both sum and count)."""
    a = series.values if hasattr(series, 'values') else np.asarray(series, dtype=np.float64)
    n = len(a)
    if n == 0:
        return series

    valid = np.isfinite(a)
    a_clean = np.where(valid, a, 0.0)

    cs = np.cumsum(a_clean)
    cs_valid = np.cumsum(valid.astype(np.float64))

    cs_shifted = np.zeros(n, dtype=np.float64)
    cs_shifted[window:] = cs[:-window]
    cv_shifted = np.zeros(n, dtype=np.float64)
    cv_shifted[window:] = cs_valid[:-window]

    sums = cs - cs_shifted
    counts = cs_valid - cv_shifted

    # np.maximum prevents 0/0 division evaluation before np.where masks it
    means = np.where(counts > 0, sums / np.maximum(counts, 1.0), np.nan)
    return pd.Series(means, index=series.index) if hasattr(series, 'index') else means

def zscore_clipped(series, window, clip=3.0):
    """Z-score with rolling window and clipping — NaN-aware O(N) numpy cumsums."""
    a = series.values if hasattr(series, 'values') else np.asarray(series, dtype=np.float64)
    n = len(a)
    if n == 0:
        return series

    valid = np.isfinite(a)
    a_clean = np.where(valid, a, 0.0)

    cs = np.cumsum(a_clean)
    cs2 = np.cumsum(a_clean ** 2)
    cs_valid = np.cumsum(valid.astype(np.float64))

    cs_shifted = np.zeros(n, dtype=np.float64)
    cs_shifted[window:] = cs[:-window]
    cs2_shifted = np.zeros(n, dtype=np.float64)
    cs2_shifted[window:] = cs2[:-window]
    cv_shifted = np.zeros(n, dtype=np.float64)
    cv_shifted[window:] = cs_valid[:-window]

    sums = cs - cs_shifted
    sums2 = cs2 - cs2_shifted
    counts = cs_valid - cv_shifted

    means = np.where(counts > 0, sums / np.maximum(counts, 1.0), 0.0)
    var = np.where(counts > 1, (sums2 - (sums ** 2) / np.maximum(counts, 1)) / np.maximum(counts - 1, 1), 0.0)
    stds = np.sqrt(np.maximum(var, 0))

    with np.errstate(divide='ignore', invalid='ignore'):
        z = np.where(stds > 1e-12, (a_clean - means) / stds, 0.0)

    z = np.where(~valid, 0.0, z)
    z = np.clip(z, -clip, clip)
    return pd.Series(z, index=series.index) if hasattr(series, 'index') else z

# ══════════════════════════════════════════════════════════════════════════════
# v2.0 MATHEMATICAL PRIMITIVES
# ══════════════════════════════════════════════════════════════════════════════
#
# Design principle: every function has exactly ONE callsite and ONE job.
#
#   Function                        → Used in                   → Purpose
#   ─────────────────────────────────────────────────────────────────────────
#   exponential_decay_weights       → correlations              → recency weighting
#   weighted_spearman               → correlations              → robust rank correlation
#   shannon_entropy                 → variable weighting        → penalize noisy variables
#   adaptive_percentile             → mood scoring              → decay-weighted CDF
#   kalman_filter_1d                → mood smoothing            → adaptive noise filtering
#   rolling_hurst                   → diagnostics (output only) → trending vs reverting
#   rolling_entropy                 → diagnostics (output only) → market disorder
#   mahalanobis_distance_batch      → similar periods           → covariance-aware matching
#   cosine_similarity               → similar periods           → trajectory shape matching
#   detect_regime_transitions       → regime diagnostics        → quadrant classification
# ══════════════════════════════════════════════════════════════════════════════

def exponential_decay_weights(n, half_life):
    """
    Generate exponential decay weights for n observations.
    w_i = exp(-λ * i), λ = ln(2) / half_life.
    Most recent observation has weight 1.0, oldest decays toward 0.
    
    Used in: calculate_anchor_correlations (Layer 1)
    """
    if n <= 0:
        return np.array([])
    lam = np.log(2) / max(half_life, 1)
    indices = np.arange(n - 1, -1, -1, dtype=np.float64)
    weights = np.exp(-lam * indices)
    return weights / weights.sum()

def weighted_spearman(x, y, weights):
    """
    Exponential-decay-weighted Spearman rank correlation.
    Computes weighted Pearson on ranks — preserves rank-robustness
    while adding recency weighting.
    
    Why Spearman over Pearson:
      - Robust to outliers (rank-based, one extreme day doesn't dominate)
      - Captures monotonic nonlinear relationships (PE compression, yield inversions)
      - Invariant to marginal distribution shape
    
    Used in: calculate_anchor_correlations (Layer 1)
    """
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return 0.0
    x, y, w = x[valid], y[valid], weights[valid]
    
    def _rank(arr):
        sorter = np.argsort(arr)
        inv = np.empty(sorter.size, dtype=np.intp)
        inv[sorter] = np.arange(sorter.size, dtype=np.intp)
        
        arr_sorted = arr[sorter]
        obs = np.r_[True, arr_sorted[1:] != arr_sorted[:-1]]
        
        tie_indices = np.nonzero(obs)[0]
        if len(tie_indices) == len(arr):
            # Fast path: No ties, return standard ordinal rank
            return inv.astype(np.float64) + 1.0
            
        # Exact average-tie rank computation (fully C-vectorised)
        dense = np.cumsum(obs) - 1
        tie_counts = np.diff(np.r_[tie_indices, len(arr)])
        avg_ranks = tie_indices + (tie_counts + 1) / 2.0
        
        ranks_sorted = avg_ranks[dense]
        return ranks_sorted[inv]
    
    rx, ry = _rank(x), _rank(y)
    w_sum = w.sum()
    if w_sum == 0:
        return 0.0
    w_norm = w / w_sum
    
    mean_rx = np.sum(w_norm * rx)
    mean_ry = np.sum(w_norm * ry)
    
    cov_xy = np.sum(w_norm * (rx - mean_rx) * (ry - mean_ry))
    var_x = np.sum(w_norm * (rx - mean_rx) ** 2)
    var_y = np.sum(w_norm * (ry - mean_ry) ** 2)
    
    denom = np.sqrt(var_x * var_y)
    if denom < 1e-12:
        return 0.0
    return np.clip(cov_xy / denom, -1.0, 1.0)

def shannon_entropy(values, n_bins=20):
    """
    Shannon entropy H = -Σ p_i * log₂(p_i), normalized to [0, 1],
    with Miller-Madow bias correction: H_corrected = H_naive + (k-1)/(2·n·ln2)
    where k = number of non-empty bins, n = sample size.

    Beirlant et al. (1997) show the naive histogram plug-in estimator is biased
    by O(k/n); the Miller (1955) correction removes the first-order term.

    Bin count selection: Freedman-Diaconis rule (bin_width = 2·IQR·n^{-1/3}),
    floored at 5 and capped at n_bins to avoid overfitting on small samples.

    Used in: calculate_historical_mood → _build_weights (Layer 2)
    """
    clean = values[np.isfinite(values)]
    n_obs = len(clean)
    if n_obs < 5:
        return 0.5
    # Freedman-Diaconis bin-width: 2 × IQR × n^{-1/3}
    q75, q25 = np.percentile(clean, [75, 25])
    iqr = q75 - q25
    data_range = clean.max() - clean.min()
    if iqr > 1e-12 and data_range > 1e-12:
        fd_width = 2.0 * iqr * (n_obs ** (-1.0 / 3.0))
        fd_bins = int(np.ceil(data_range / fd_width))
    else:
        fd_bins = int(np.sqrt(n_obs))
    adaptive_bins = max(5, min(n_bins, fd_bins))
    counts, _ = np.histogram(clean, bins=adaptive_bins)
    probs = counts / counts.sum()
    non_empty = probs[probs > 0]
    if len(non_empty) <= 1:
        return 0.0
    h_naive = -np.sum(non_empty * np.log2(non_empty))
    # Miller-Madow first-order bias correction
    k = len(non_empty)
    h_corrected = h_naive + (k - 1) / (2.0 * n_obs * np.log(2))
    h_max = np.log2(adaptive_bins)
    return np.clip(h_corrected / h_max, 0.0, 1.0) if h_max > 0 else 0.0

def adaptive_percentile(series, half_life=252):
    """
    Exponential-decay-weighted empirical CDF — O(N log N) via sorted-insert.

    For each time t, the percentile of x_t is:
        P(t) = Σ_{i≤t} w_i · 𝟙(x_i ≤ x_t) / Σ_{i≤t} w_i
    where w_i = exp(-λ·(t-i)), λ = ln(2)/half_life.

    Implementation: maintain a sorted array of observed values with their
    insertion times. At each step, binary-search for x_t's rank position,
    compute the cumulative weighted CDF using vectorised decay on the
    sorted array. The search is O(log N), but list insertion is O(N).
    Total time complexity: O(N²).

    Greenwald & Khanna (2001) motivates the streaming quantile approach;
    here the sorted-insert + searchsorted is exact.

    Used in: calculate_historical_mood (Layer 3)
    """
    from bisect import bisect_right

    values = np.asarray(series, dtype=np.float64)
    n = len(values)
    if n == 0:
        return np.array([])

    lam = np.log(2) / max(half_life, 1)
    valid = np.isfinite(values)

    if not np.any(valid):
        return np.full(n, 0.5)

    result = np.full(n, np.nan)

    # Maintain parallel sorted arrays: sorted_vals (for bisect) and
    # sorted_times (insertion time for each value, same order).
    sorted_vals = []    # sorted by value
    sorted_times = []   # insertion time corresponding to each sorted value
    total_weight = 0.0  # running sum of all weights (decayed each step)
    decay_factor = np.exp(-lam)  # multiplicative decay per step

    for t in range(n):
        # Decay all existing weights by one step (equivalent to aging everything)
        total_weight *= decay_factor

        if not valid[t]:
            continue

        v = values[t]
        w_new = 1.0  # current observation always has weight 1.0 (most recent)

        # Insert into sorted order
        pos = bisect_right(sorted_vals, v)
        sorted_vals.insert(pos, v)
        sorted_times.insert(pos, t)
        total_weight += w_new

        if total_weight < 1e-12:
            continue

        # Compute weighted CDF: sum of weights for all values ≤ v
        # All values in sorted_vals[:pos+1] have value ≤ v (side='right')
        # Their weights are exp(-λ·(t - insertion_time))
        times_leq = np.array(sorted_times[:pos + 1], dtype=np.float64)
        w_leq = np.exp(-lam * (t - times_leq))
        result[t] = np.sum(w_leq) / total_weight

    # Convert lists to arrays for the final cleanup
    return pd.Series(result).ffill().fillna(0.5).values

def kalman_filter_1d(
    observations: np.ndarray | pd.Series,
    process_var: float | None = None,
    measurement_var: float | None = None,
    half_life: int = KALMAN_HALF_LIFE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    1D Fading Memory Kalman Filter (Sorenson & Sacks).

    Uses an exponential fading factor to discount past data,
    preventing filter divergence in non-stationary regimes.

    Returns:
        filtered_state: Smoothed state estimates for each observation.
        estimate_variances: Posterior variance estimates (used for confidence bands).
    """
    obs = np.asarray(observations, dtype=np.float64)
    n = len(obs)
    if n == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Causal noise estimation flags
    auto_process = process_var is None
    auto_measure = measurement_var is None
    
    s_obs = pd.Series(obs)
    
    # O(N) Causal variance estimations with burn-in bootstrap.
    # Harvey (1990): early expanding variance estimates are unreliable;
    # bootstrap the first BURN_IN observations from the first stable window.
    _BURN_IN = min(50, n // 4) if n > 20 else 1
    if auto_measure:
        m_vars = s_obs.expanding().var().fillna(1.0).values * 0.5
        m_vars = np.maximum(m_vars, 1e-8)
        if _BURN_IN > 1 and n > _BURN_IN:
            m_vars[:_BURN_IN] = m_vars[_BURN_IN]
    else:
        m_vars = np.full(n, measurement_var)

    if auto_process:
        p_vars = s_obs.diff().expanding().var().fillna(1e-3).values * 0.1
        p_vars = np.maximum(p_vars, 1e-8)
        if _BURN_IN > 1 and n > _BURN_IN:
            p_vars[:_BURN_IN] = p_vars[_BURN_IN]
    else:
        p_vars = np.full(n, process_var)
        
    state = obs[0] if np.isfinite(obs[0]) else 0.0
    estimate_var = m_vars[0]
    
    filtered = np.zeros(n)
    variances = np.zeros(n)
    filtered[0] = state
    variances[0] = estimate_var

    # Sorenson & Sacks Fading Memory parameter
    lam = np.log(2) / max(half_life, 1)
    alpha_sq = np.exp(2 * lam)  # Fading factor > 1

    for i in range(1, n):
        # Fading memory predict step
        pred_var = alpha_sq * estimate_var + p_vars[i]

        if np.isfinite(obs[i]):
            # Update step
            K = pred_var / (pred_var + m_vars[i])
            state = state + K * (obs[i] - state)
            estimate_var = (1 - K) * pred_var
        else:
            estimate_var = pred_var

        filtered[i] = state
        variances[i] = estimate_var

    return filtered, variances

def _hurst_dfa(series, max_lag=None):
    """
    Hurst exponent via Detrended Fluctuation Analysis (DFA-1).
    H > 0.5 → persistent (trending), H < 0.5 → anti-persistent (mean-reverting).

    DFA is more robust than R/S for short series and correctly distinguishes
    long-range dependence from short-range ARMA effects.
    Reference: Peng et al. (1994), "Mosaic organization of DNA nucleotides."
               Weron (2002) shows DFA outperforms R/S for n < 256.

    Internal helper for rolling_hurst.
    """
    ts = np.asarray(series, dtype=np.float64)
    ts = ts[np.isfinite(ts)]
    n = len(ts)
    if n < 20:
        return 0.5

    # Integrated profile: cumulative deviation from mean
    profile = np.cumsum(ts - ts.mean())

    min_scale = 10
    if max_lag is None:
        max_lag = min(n // 4, 200)
    if max_lag <= min_scale:
        return 0.5

    scales = np.unique(np.logspace(
        np.log10(min_scale), np.log10(max_lag), num=20,
    ).astype(int))
    scales = scales[(scales >= min_scale) & (scales <= max_lag)]

    if len(scales) < 3:
        return 0.5

    flucts = []
    for s in scales:
        n_seg = n // s
        if n_seg < 4:
            continue
        # Non-overlapping segments
        segments = profile[:n_seg * s].reshape(n_seg, s)

        # Vectorised linear detrend across all segments
        x = np.arange(s, dtype=np.float64)
        x_mean = x.mean()
        x_var = np.sum((x - x_mean) ** 2)
        if x_var < 1e-12:
            continue

        seg_means = segments.mean(axis=1, keepdims=True)
        slopes = np.sum((segments - seg_means) * (x - x_mean), axis=1) / x_var
        intercepts = seg_means.ravel() - slopes * x_mean

        trends = intercepts[:, None] + slopes[:, None] * x[None, :]
        residuals = segments - trends

        fluct = np.sqrt(np.mean(residuals ** 2))
        if fluct > 1e-12:
            flucts.append((s, fluct))

    if len(flucts) < 3:
        return 0.5

    log_s = np.log(np.array([f[0] for f in flucts], dtype=np.float64))
    log_f = np.log(np.array([f[1] for f in flucts], dtype=np.float64))

    valid = np.isfinite(log_s) & np.isfinite(log_f)
    if valid.sum() < 3:
        return 0.5
    log_s, log_f = log_s[valid], log_f[valid]
    mean_x, mean_y = log_s.mean(), log_f.mean()
    var_x = np.sum((log_s - mean_x) ** 2)
    H = np.sum((log_s - mean_x) * (log_f - mean_y)) / var_x if var_x > 1e-12 else 0.5
    return np.clip(H, 0.01, 0.99)

def rolling_hurst(series, window=90, step=5):
    """
    Rolling Hurst exponent via DFA. Computed every `step` points, forward-filled.
    Uses a sentinel to distinguish "not yet computed" from a legitimate H=0.5 estimate.
    Used in: calculate_historical_mood → diagnostics output
    """
    values = np.asarray(series, dtype=np.float64)
    n = len(values)
    _SENTINEL = -1.0  # impossible Hurst value — marks "not yet computed"
    result = np.full(n, _SENTINEL)
    for i in range(window, n, step):
        result[i] = _hurst_dfa(values[i - window:i])
    # Forward-fill only sentinel gaps (preserves legitimate H=0.5 estimates)
    for i in range(1, n):
        if result[i] == _SENTINEL and result[i - 1] != _SENTINEL:
            result[i] = result[i - 1]
    # Replace any remaining sentinels (before first computation) with 0.5
    result[result == _SENTINEL] = 0.5
    return result

def rolling_entropy(series, window=60, n_bins=15):
    """
    Rolling Shannon entropy of a series. Normalized to [0, 1].
    Used in: calculate_historical_mood → diagnostics output
    """
    from numpy.lib.stride_tricks import sliding_window_view
    
    values = series.values if hasattr(series, 'values') else np.asarray(series, dtype=np.float64)
    n = len(values)
    result = np.full(n, 0.5)
    if n < 5:
        return result
        
    if n >= window:
        # sliding_window_view on full array: windows[i] = values[i:i+window]
        # result[i+window-1] = entropy of values[i:i+window] (aligned to window end)
        windows = sliding_window_view(values, window)
        result[window - 1:window - 1 + len(windows)] = [shannon_entropy(w, n_bins) for w in windows]

    for i in range(5, min(window - 1, n)):
        result[i] = shannon_entropy(values[:i + 1], n_bins)
        
    return result

def _ledoit_wolf_shrinkage(S, n):
    """
    Ledoit & Wolf (2004) analytical shrinkage estimator.
    Σ* = δ·F + (1−δ)·S  where F = (tr(S)/p)·I  (scaled identity target).
    Optimal δ minimises E[‖Σ*−Σ‖²_F] under standard asymptotics.
    Returns the shrunk covariance matrix — always well-conditioned.
    """
    p = S.shape[0]
    if p == 0 or n < 2:
        return S
    trace_S = np.trace(S)
    mu = trace_S / p                       # target = μ·I
    delta_mat = S - mu * np.eye(p)
    sum_sq = np.sum(delta_mat ** 2)        # ‖S − μI‖²_F
    # Optimal shrinkage intensity (OAS closed-form, Chen et al. 2010)
    rho_num = ((1.0 - 2.0 / p) * sum_sq + trace_S ** 2)
    rho_den = ((n + 1.0 - 2.0 / p) * (sum_sq + trace_S ** 2 / p))
    rho = np.clip(rho_num / max(rho_den, 1e-12), 0.0, 1.0)
    return (1.0 - rho) * S + rho * mu * np.eye(p)

def mahalanobis_distance_batch(features, center, cov_matrix):
    """
    Mahalanobis distance: d_M = √((x−μ)ᵀ Σ⁻¹ (x−μ))
    Uses Ledoit-Wolf analytical shrinkage (2004) for a well-conditioned
    covariance inverse, replacing ad-hoc diagonal regularization.
    Used in: find_similar_periods
    """
    diff = features - center
    n_samples = features.shape[0]
    shrunk_cov = _ledoit_wolf_shrinkage(cov_matrix, n_samples)
    try:
        cov_inv = np.linalg.inv(shrunk_cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(shrunk_cov)
    left = diff @ cov_inv
    d_sq = np.maximum(np.sum(left * diff, axis=1), 0)
    return np.sqrt(d_sq)

def cosine_similarity(a, b):
    """
    Cosine similarity — measures trajectory shape match irrespective of magnitude.
    Used in: find_similar_periods → trajectory matching
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

def detect_regime_transitions(hurst_values, entropy_values, window=10):
    """
    Detect regime transitions using Hurst exponent + entropy jointly.
    
    The idea: market operates in one of 4 quadrants:
      High H, Low S  → Trending/Ordered   (momentum works, strong directional move)
      High H, High S → Trending/Disordered (volatile trend, large swings in one direction)
      Low H, Low S   → Mean-reverting/Ordered (range-bound, predictable oscillation)
      Low H, High S  → Mean-reverting/Disordered (choppy chaos, hardest to trade)
    
    A regime TRANSITION is when the market crosses quadrant boundaries.
    Specifically, the most important transitions are:
      Trending→Choppy : H drops below 0.5 while S rises → trend exhaustion
      Choppy→Trending : H rises above 0.5 while S drops → new trend emerging
    
    We smooth both signals and detect crossover events.
    
    Returns: array of regime labels + transition indices
    """
    h = np.asarray(hurst_values, dtype=np.float64)
    s = np.asarray(entropy_values, dtype=np.float64)
    n = len(h)
    
    if n < window * 2:
        return np.full(n, 'Unknown', dtype=object), []
    
    # Smooth both signals to avoid noise-triggered transitions
    h_smooth = pd.Series(h).rolling(window=window, min_periods=1).mean().values
    s_smooth = pd.Series(s).rolling(window=window, min_periods=1).mean().values
    
    # Median thresholds (adaptive to the data, not hardcoded)
    h_threshold = 0.5   # Theoretical random walk boundary
    s_median = np.median(s_smooth[s_smooth > 0]) if np.any(s_smooth > 0) else 0.5
    
    # Classify each point into regime quadrant
    regimes = np.full(n, 'Unknown', dtype=object)
    for i in range(n):
        trending = h_smooth[i] > h_threshold
        ordered = s_smooth[i] < s_median
        
        if trending and ordered:
            regimes[i] = 'Trending'         # Best for momentum
        elif trending and not ordered:
            regimes[i] = 'Volatile Trend'   # Momentum with risk
        elif not trending and ordered:
            regimes[i] = 'Mean-Reverting'   # Best for contrarian
        else:
            regimes[i] = 'Choppy'           # Hardest to trade
    
    # Detect transition points (regime[i] != regime[i-1])
    transitions = []
    for i in range(1, n):
        if regimes[i] != regimes[i - 1]:
            prev = regimes[i - 1]
            curr = regimes[i]
            
            # Classify transition significance
            # Major: Trending↔Choppy (complete character flip)
            # Minor: adjacent quadrant shifts
            major_pairs = {
                ('Trending', 'Choppy'), ('Choppy', 'Trending'),
                ('Trending', 'Mean-Reverting'), ('Mean-Reverting', 'Trending'),
            }
            is_major = (prev, curr) in major_pairs
            
            transitions.append({
                'index': i,
                'from': prev,
                'to': curr,
                'major': is_major,
                'hurst': h_smooth[i],
                'entropy': s_smooth[i],
            })
    
    return regimes, transitions

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_sheet_csv(max_retries: int = 3) -> str:
    """
    Fetch the Google Sheet as CSV via the Google Visualization API.

    Uses the /gviz/tq?tqx=out:csv endpoint — no OAuth/service account needed.
    The sheet must be set to "Anyone with the link can view" in sharing settings.

    Retries with exponential backoff on transient network failures.
    """
    if not SHEET_ID or not SHEET_GID:
        raise RuntimeError(
            "ARTHAGATI_SHEET_ID and ARTHAGATI_SHEET_GID environment variables are not set.\n"
            '  export ARTHAGATI_SHEET_ID="1po7z42n3dYIQGAvn0D1-a4pmyxpnGPQ13TrNi3DB5_c"\n'
            '  export ARTHAGATI_SHEET_GID="1938234952"'
        )

    url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&gid={SHEET_GID}"

    last_exception = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            return resp.text
        except requests.exceptions.Timeout as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt * 2
                logging.warning(
                    f"Google Sheets request timed out (attempt {attempt + 1}/{max_retries}). "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                logging.error(f"Google Sheets request failed after {max_retries} attempts: {e}")
        except requests.exceptions.RequestException as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt * 2
                logging.warning(
                    f"Google Sheets request failed (attempt {attempt + 1}/{max_retries}). "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                logging.error(f"Google Sheets request failed after {max_retries} attempts: {e}")

    raise RuntimeError(f"Failed to load sheet data after {max_retries} attempts: {last_exception}")


@st.cache_data(ttl=DATA_TTL, show_spinner=False)
def load_data() -> pd.DataFrame | None:
    """
    Fetch and parse market data from the private Google Sheet.

    Returns a clean DataFrame with:
      - All columns present in the sheet (none fabricated from EXPECTED_COLUMNS)
      - DATE parsed, all other columns coerced to float
      - Derived columns: IN_TERM_SPREAD, US_TERM_SPREAD, NIFTY50_EY (if absent)
      - Rows with NIFTY ≤ 0 or unparseable DATE dropped
    """
    start_time = time.time()
    try:
        csv_text = _fetch_sheet_csv()
        df = pd.read_csv(StringIO(csv_text), dtype=str)

        # Normalise column names: strip whitespace, drop unnamed padding columns
        df.columns = [c.strip() for c in df.columns]
        df = df[[c for c in df.columns if not c.startswith('Unnamed')]]

        # Hard requirements — nothing works without these two
        if 'DATE' not in df.columns or 'NIFTY' not in df.columns:
            raise ValueError("Required columns DATE and NIFTY not found in the sheet.")

        # Warn about any known-schema columns absent in the sheet, but do NOT fabricate them.
        # The predictor dropdown will only show columns that genuinely exist in the data.
        missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
        if missing:
            logging.warning(
                "Schema drift — %d expected column(s) absent from sheet: %s. "
                "Predictor set will be built from columns that are actually present.",
                len(missing), missing,
            )

        df['DATE'] = pd.to_datetime(df['DATE'], format='%m/%d/%Y', errors='coerce')

        non_date_cols = [c for c in df.columns if c != 'DATE']
        df[non_date_cols] = df[non_date_cols].apply(pd.to_numeric, errors='coerce')
        # Forward-fill only: persistent data (rates, yields) carries forward.
        # No back-fill — it would leak future values into early observations.
        # NaN-only columns and series starts remain NaN; all downstream math
        # primitives have np.isfinite() guards that handle missing data correctly.
        df[non_date_cols] = df[non_date_cols].ffill()

        df = df[df['NIFTY'] > 0].dropna(subset=['DATE']).copy()
        if df.empty:
            raise ValueError("No valid rows after filtering on NIFTY > 0 and a parseable DATE.")

        # Preserve column order: DATE and NIFTY first, then everything else
        core = ['DATE', 'NIFTY']
        df = df[core + [c for c in df.columns if c not in core]].sort_values('DATE').reset_index(drop=True)

        # Derive NIFTY50_EY from PE if the sheet omits it or populates it as a constant.
        # EY = 1/PE × 100.
        if 'NIFTY50_PE' in df.columns and df['NIFTY50_PE'].gt(0).any():
            if 'NIFTY50_EY' not in df.columns or df['NIFTY50_EY'].nunique() <= 1:
                df['NIFTY50_EY'] = (1.0 / df['NIFTY50_PE'].replace(0, np.nan) * 100).fillna(0)
                logging.info("NIFTY50_EY absent or constant — derived from PE (EY = 1/PE × 100).")

        # Derive yield-curve term spreads (10Y − 2Y).
        # Positive = normal curve (expansion). Negative = inverted (recession signal).
        if 'IN10Y' in df.columns and 'IN02Y' in df.columns:
            df['IN_TERM_SPREAD'] = df['IN10Y'] - df['IN02Y']
        else:
            df['IN_TERM_SPREAD'] = 0.0
        if 'US10Y' in df.columns and 'US02Y' in df.columns:
            df['US_TERM_SPREAD'] = df['US10Y'] - df['US02Y']
        else:
            df['US_TERM_SPREAD'] = 0.0

        elapsed = time.time() - start_time
        date_range = f"{df['DATE'].iloc[0].strftime('%Y-%m-%d')} → {df['DATE'].iloc[-1].strftime('%Y-%m-%d')}"
        logging.info(
            "Data loaded — %d rows × %d columns | %s | %.2fs",
            len(df), len(df.columns), date_range, elapsed,
        )
        return df

    except Exception as exc:
        logging.error("Data load failed — pipeline halted. Cause: %s", exc)
        st.error(f"Failed to load sheet data: {exc}")
        return None

# ══════════════════════════════════════════════════════════════════════════════
# MOOD SCORE CALCULATION
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(max_entries=5, show_spinner=False)
def calculate_anchor_correlations(df, anchor, dependent_vars=None):
    """
    Layer 1: Exponential-decay-weighted Spearman rank correlations.
    
    Half-life ~504 days (~2 trading years). This means:
    - Data from 2 years ago has half the weight of today
    - Data from 4 years ago has 1/4 the weight
    - Very old regimes fade naturally without being discarded
    """
    if dependent_vars is None:
        dependent_vars = DEPENDENT_VARS
    cols_to_check = [col for col in dependent_vars if col in df.columns]
    
    if anchor not in df.columns or not cols_to_check:
        return pd.DataFrame(columns=['variable', 'correlation', 'strength', 'type'])
    
    analysis_df = df[[anchor] + cols_to_check].select_dtypes(include=[np.number])
    if anchor not in analysis_df.columns:
        return pd.DataFrame(columns=['variable', 'correlation', 'strength', 'type'])
    
    anchor_vals = analysis_df[anchor].values
    n = len(anchor_vals)
    half_life = min(CORR_HALF_LIFE, n // 2) if n > 20 else max(n // 2, 5)
    weights = exponential_decay_weights(n, half_life)
    
    correlations = []
    for var in cols_to_check:
        if var == anchor or var not in analysis_df.columns:
            continue
        corr = weighted_spearman(anchor_vals, analysis_df[var].values, weights)
        if not np.isfinite(corr):
            corr = 0.0
        abs_corr = abs(corr)
        strength = ('Strong' if abs_corr >= 0.7 else
                   'Moderate' if abs_corr >= 0.5 else
                   'Weak' if abs_corr >= 0.3 else 'Very weak')
        correlations.append({
            'variable': var, 'correlation': corr,
            'strength': strength, 'type': 'positive' if corr > 0 else 'negative'
        })
    
    return pd.DataFrame(correlations)

@st.cache_data(max_entries=5, show_spinner=False)
def calculate_historical_mood(df, dependent_vars=None):
    """
    v2.3 Mood Score Engine — 5-layer architecture with walk-forward weights.

    Fixes vs v2.2:
      - Layers 1+2 now use EXPANDING-WINDOW correlations and entropy at periodic
        checkpoints (CORR_REBALANCE_PERIOD), eliminating look-ahead bias.
      - Layer 3 percentile semantics corrected: adjustments are symmetric [-1,+1]
        around zero (was [0,+1], creating asymmetric bearish/bullish capacity).
      - Layer 4 OU bias correction applied (Kendall-Marriott-Pope on AR(1) coef).
      - Layer 5 Kalman uses its own half-life (KALMAN_HALF_LIFE), decoupled from PCT.

    Diagnostics (output-only, do NOT modify the score):
      Hurst exponent (DFA), market entropy, OU half-life
    """
    if dependent_vars is None:
        dependent_vars = DEPENDENT_VARS
    start_time = time.time()

    if 'DATE' not in df.columns or 'NIFTY50_PE' not in df.columns or 'NIFTY50_EY' not in df.columns:
        logging.error(
            "Mood engine aborted — required anchor columns missing. "
            "Sheet must contain DATE, NIFTY50_PE, and NIFTY50_EY."
        )
        return pd.DataFrame(columns=['DATE', 'Mood_Score', 'Mood', 'Smoothed_Mood_Score', 'Mood_Volatility'])

    n = len(df)
    vars_to_check = [col for col in dependent_vars
                     if col in df.columns and col not in NON_PREDICTOR_COLS]

    # ── Layer 3 (computed first): Adaptive Percentiles ────────────────
    # These are already expanding-window (no look-ahead).
    pct_hl = min(PCT_HALF_LIFE, n // 2) if n > 20 else max(n // 2, 5)

    pe_percentiles = adaptive_percentile(df['NIFTY50_PE'].values, half_life=pct_hl)
    ey_percentiles = adaptive_percentile(df['NIFTY50_EY'].values, half_life=pct_hl)

    var_percentiles = {}
    for var in vars_to_check:
        var_percentiles[var] = adaptive_percentile(df[var].values, half_life=pct_hl)

    # ── Layers 1+2: Walk-Forward Correlations & Entropy ───────────────
    # At each checkpoint, compute expanding Spearman correlations and expanding
    # entropy using ONLY data available up to that point — no look-ahead.
    anchor_pe = df['NIFTY50_PE'].values
    anchor_ey = df['NIFTY50_EY'].values

    min_warmup = min(CORR_MIN_WARMUP, n // 2) if n > 50 else max(n // 3, 10)
    rebal = max(min(CORR_REBALANCE_PERIOD, max((n - min_warmup) // 3, 1)), 1)

    checkpoints = list(range(min_warmup, n, rebal))
    if not checkpoints or checkpoints[-1] != n - 1:
        checkpoints.append(n - 1)

    # Pre-compute returns for expanding entropy
    var_returns_all = {}
    for var in vars_to_check:
        vals = df[var].values
        rets = np.empty(len(vals))
        rets[0] = np.nan
        with np.errstate(divide='ignore', invalid='ignore'):
            rets[1:] = np.where(np.abs(vals[:-1]) > 1e-12, np.diff(vals) / np.abs(vals[:-1]), 0.0)
        rets = np.where(np.isfinite(rets), rets, np.nan)
        var_returns_all[var] = rets

    # Accumulate adjustments and strengths segment-by-segment
    pe_base = 1.0 - 2.0 * pe_percentiles     # High PE → low score (bearish)
    ey_base = 2.0 * ey_percentiles - 1.0      # High EY → high score (bullish)

    pe_adjustments = np.zeros(n)
    ey_adjustments = np.zeros(n)
    pe_strength_arr = np.zeros(n)
    ey_strength_arr = np.zeros(n)

    # Exponential weight blending across checkpoints to smooth discontinuities.
    # At each checkpoint, new weights are blended with previous:
    #   w_eff = α·w_new + (1−α)·w_prev,  α = 1 − exp(−ln(2)/blend_hl)
    # First checkpoint uses α=1 (no prior to blend with).
    _BLEND_HL = 2.0  # in checkpoint units (≈2 rebalance periods to fully converge)
    _blend_alpha = 1.0 - np.exp(-np.log(2) / max(_BLEND_HL, 0.5))
    prev_pe_w: dict[str, float] = {}
    prev_ey_w: dict[str, float] = {}
    prev_pe_corrs: dict[str, float] = {}
    prev_ey_corrs: dict[str, float] = {}

    for cp_idx, cp in enumerate(checkpoints):
        seg_start = checkpoints[cp_idx - 1] + 1 if cp_idx > 0 else 0
        seg_end = cp + 1

        cp_n = cp + 1
        cp_half_life = min(CORR_HALF_LIFE, cp_n // 2) if cp_n > 20 else max(cp_n // 2, 5)
        cp_weights = exponential_decay_weights(cp_n, cp_half_life)

        # Expanding correlations and entropy at this checkpoint
        cp_pe_corrs = {}
        cp_ey_corrs = {}
        cp_entropies = {}

        for var in vars_to_check:
            var_vals = df[var].values[:cp_n]

            # Expanding entropy on returns available up to checkpoint
            rets_cp = var_returns_all[var][1:cp_n]
            clean_rets = rets_cp[np.isfinite(rets_cp)]
            cp_entropies[var] = shannon_entropy(clean_rets) if len(clean_rets) > 10 else 0.5

            # Expanding Spearman with PE and EY
            pe_c = weighted_spearman(anchor_pe[:cp_n], var_vals, cp_weights)
            ey_c = weighted_spearman(anchor_ey[:cp_n], var_vals, cp_weights)
            cp_pe_corrs[var] = pe_c if np.isfinite(pe_c) else 0.0
            cp_ey_corrs[var] = ey_c if np.isfinite(ey_c) else 0.0

        # Build raw weights: |corr| × (1 − entropy)
        pe_raw_w, ey_raw_w = {}, {}
        for var in vars_to_check:
            entropy_pen = 1.0 - cp_entropies.get(var, 0.5)
            pe_raw_w[var] = abs(cp_pe_corrs[var]) * max(entropy_pen, 0.1)
            ey_raw_w[var] = abs(cp_ey_corrs[var]) * max(entropy_pen, 0.1)

        pe_total = max(sum(pe_raw_w.values()), 1e-10)
        ey_total = max(sum(ey_raw_w.values()), 1e-10)
        pe_w_new = {k: v / pe_total for k, v in pe_raw_w.items()}
        ey_w_new = {k: v / ey_total for k, v in ey_raw_w.items()}

        # Blend with previous checkpoint weights (first checkpoint: α=1, use raw)
        if prev_pe_w:
            pe_w = {v: _blend_alpha * pe_w_new.get(v, 0.0) + (1.0 - _blend_alpha) * prev_pe_w.get(v, 0.0) for v in vars_to_check}
            ey_w = {v: _blend_alpha * ey_w_new.get(v, 0.0) + (1.0 - _blend_alpha) * prev_ey_w.get(v, 0.0) for v in vars_to_check}
        else:
            pe_w = pe_w_new
            ey_w = ey_w_new
        prev_pe_w = dict(pe_w)
        prev_ey_w = dict(ey_w)

        # Also blend correlations for sign stability
        if prev_pe_corrs:
            blended_pe_corrs = {v: _blend_alpha * cp_pe_corrs.get(v, 0.0) + (1.0 - _blend_alpha) * prev_pe_corrs.get(v, 0.0) for v in vars_to_check}
            blended_ey_corrs = {v: _blend_alpha * cp_ey_corrs.get(v, 0.0) + (1.0 - _blend_alpha) * prev_ey_corrs.get(v, 0.0) for v in vars_to_check}
        else:
            blended_pe_corrs = dict(cp_pe_corrs)
            blended_ey_corrs = dict(cp_ey_corrs)
        prev_pe_corrs = dict(blended_pe_corrs)
        prev_ey_corrs = dict(blended_ey_corrs)

        pe_str = sum(abs(blended_pe_corrs[v]) for v in vars_to_check)
        ey_str = sum(abs(blended_ey_corrs[v]) for v in vars_to_check)

        # Compute adjustments for this segment using blended correlations/weights
        seg_pe = np.zeros(seg_end - seg_start)
        seg_ey = np.zeros(seg_end - seg_start)

        for var in vars_to_check:
            vpct = var_percentiles[var][seg_start:seg_end]

            # FIXED percentile semantics (L1):
            # PE: positive corr + high var_pct → high PE → bearish → push score DOWN
            #     Adjustment = sign × weight × (1 − 2·pct) maps [0,1] → [+1,−1]
            pe_sign = 1.0 if blended_pe_corrs[var] >= 0 else -1.0
            seg_pe += pe_sign * pe_w[var] * (1.0 - 2.0 * vpct)

            # EY: positive corr + high var_pct → high EY → bullish → push score UP
            #     Adjustment = sign × weight × (2·pct − 1) maps [0,1] → [−1,+1]
            ey_sign = 1.0 if blended_ey_corrs[var] >= 0 else -1.0
            seg_ey += ey_sign * ey_w[var] * (2.0 * vpct - 1.0)

        pe_adjustments[seg_start:seg_end] = seg_pe
        ey_adjustments[seg_start:seg_end] = seg_ey
        pe_strength_arr[seg_start:seg_end] = pe_str
        ey_strength_arr[seg_start:seg_end] = ey_str

    pe_scores = np.clip(0.5 * pe_base + 0.5 * pe_adjustments, -1, 1)
    ey_scores = np.clip(0.5 * ey_base + 0.5 * ey_adjustments, -1, 1)

    total_strength = pe_strength_arr + ey_strength_arr
    total_strength = np.where(total_strength > 0, total_strength, 1.0)
    raw_mood = (pe_strength_arr / total_strength) * pe_scores + (ey_strength_arr / total_strength) * ey_scores

    # ── Layer 4: OU Normalization ───────────────────────────────────────
    # Expanding z-score to get rough scale
    counts = np.arange(1, n + 1)
    cum_sum = np.cumsum(raw_mood)
    expanding_mean = cum_sum / counts

    cum_sq_sum = np.cumsum(raw_mood ** 2)
    var_expanding = (cum_sq_sum - (cum_sum ** 2) / counts) / np.maximum(counts - 1, 1)
    expanding_std = np.maximum(np.sqrt(np.maximum(var_expanding, 0)), 1e-6)
    expanding_std[0] = 1.0

    rough_scaled = (raw_mood - expanding_mean) / expanding_std

    # Vectorised Expanding OU Estimation with bias correction.
    #
    # H2 Fix: The previous algebraic expanding RSS (cumsum(y²) + n·a² + ...)
    # is only correct when (a, b) are constant; with per-step expanding estimates
    # that change at every index, the cross-terms are inconsistent.
    #
    # Correct approach: compute the per-observation residual e²_i = (y_i − a_i − b_i·x_i)²
    # using the current expanding (a, b) at each step, then EMA-smooth these squared
    # residuals for a stable variance estimate.
    ou_thetas = np.full(n, 0.05)
    ou_mus = np.zeros(n)
    ou_sigmas = np.ones(n)

    x_ou = rough_scaled[:-1]
    y_ou = rough_scaled[1:]
    n_points = np.arange(1, n)

    sum_x = np.cumsum(x_ou)
    sum_y = np.cumsum(y_ou)
    sum_x2 = np.cumsum(x_ou ** 2)
    sum_xy = np.cumsum(x_ou * y_ou)

    mean_x_ou = sum_x / n_points
    mean_y_ou = sum_y / n_points

    var_x_ou = sum_x2 - (sum_x ** 2) / n_points
    cov_xy_ou = sum_xy - (sum_x * sum_y) / n_points

    var_x_safe = np.where(var_x_ou < 1e-12, 1e-12, var_x_ou)
    b_hat = cov_xy_ou / var_x_safe

    # Kendall-Marriott-Pope first-order bias correction (vectorised)
    b = b_hat + (1.0 + 3.0 * b_hat) / np.maximum(n_points, 1)
    b = np.clip(b, 1e-6, 1.0 - 1e-6)

    a_ou = mean_y_ou - b * mean_x_ou

    theta_vals = np.clip(-np.log(b), 1e-4, 10.0)
    mu_vals = a_ou / (1.0 - b)

    # Per-observation residuals using the current expanding (a, b) at each step.
    # e²_i = (y_i − a_i − b_i·x_i)² — each residual uses the correct parameters.
    per_residual_sq = (y_ou - a_ou - b * x_ou) ** 2
    # Expanding mean of squared residuals (correct RSS regardless of how a,b vary)
    var_eps = np.maximum(np.cumsum(per_residual_sq) / n_points, 0)

    denom_ou = np.maximum(1.0 - b ** 2, 1e-12)
    sigma_sq = np.where((1.0 - b ** 2) > 1e-12, 2.0 * theta_vals * var_eps / denom_ou, var_eps)
    sigma_vals = np.sqrt(np.maximum(sigma_sq, 1e-12))

    valid_idx = n_points >= 50
    ou_thetas[1:][valid_idx] = theta_vals[valid_idx]
    ou_mus[1:][valid_idx] = mu_vals[valid_idx]
    ou_sigmas[1:][valid_idx] = sigma_vals[valid_idx]

    t_std = np.maximum(ou_sigmas / np.sqrt(2.0 * np.maximum(ou_thetas, 1e-4)), 1e-6)
    mood_scores = np.clip((rough_scaled - ou_mus) / t_std * MOOD_SCALE, -100, 100)

    theta, mu, sigma_ou = ou_thetas[-1], ou_mus[-1], ou_sigmas[-1]
    ou_half_life = np.log(2) / max(theta, 1e-4)

    # ── Layer 5: Kalman Smoothing ───────────────────────────────────────
    smoothed_mood_scores, kalman_variances = kalman_filter_1d(mood_scores)

    # Confidence band: ±KALMAN_CI_Z × √variance (~95% interval)
    kalman_std = np.sqrt(np.maximum(kalman_variances, 0))
    confidence_upper = smoothed_mood_scores + KALMAN_CI_Z * kalman_std
    confidence_lower = smoothed_mood_scores - KALMAN_CI_Z * kalman_std

    # Traditional volatility (backward compatible)
    mood_volatility = pd.Series(mood_scores).rolling(window=30, min_periods=1).std().fillna(0)

    # ── Classification (fixed thresholds — see VISION.md §6 for why) ───
    moods = np.where(mood_scores > 60, 'Very Bullish',
            np.where(mood_scores > 20, 'Bullish',
            np.where(mood_scores > -20, 'Neutral',
            np.where(mood_scores > -60, 'Bearish', 'Very Bearish'))))

    # ── Diagnostics (output-only — do NOT modify scores) ───────────────
    nifty_returns = df['NIFTY'].pct_change().fillna(0).values
    hurst_vals = rolling_hurst(df['NIFTY'].values, window=90, step=5)
    entropy_vals = rolling_entropy(nifty_returns, window=60, n_bins=15)

    # ── Regime Detection ────────────────────────────────────────────────
    regime_labels, regime_transitions = detect_regime_transitions(hurst_vals, entropy_vals)

    result_df = pd.DataFrame({
        'DATE': df['DATE'].values,
        'Mood_Score': mood_scores,
        'Mood': moods,
        'Smoothed_Mood_Score': smoothed_mood_scores,
        'Mood_Volatility': mood_volatility.values,
        'NIFTY': df['NIFTY'].values,
        'AD_RATIO': df['AD_RATIO'].values if 'AD_RATIO' in df.columns else np.ones(n),
        # v2.0 diagnostics
        'Hurst': hurst_vals,
        'Market_Entropy': entropy_vals,
        'OU_Half_Life': ou_half_life,
        'OU_Theta': theta,
        'OU_Mu': mu,
        # v2.1 additions
        'OU_Sigma': sigma_ou,
        # Soft-clip: tanh preserves band *width* near the extremes so users
        # still see how uncertain the reading is, unlike a hard clip at ±100
        # which would make the band appear artificially narrow.
        'Confidence_Upper': np.tanh(confidence_upper / 100.0) * 100.0,
        'Confidence_Lower': np.tanh(confidence_lower / 100.0) * 100.0,
        'Regime': regime_labels,
    })

    logging.info(
        "Mood engine complete — %d rows in %.2fs | "
        "OU: θ=%.3f  μ=%.2f  t½=%.0fd | "
        "Diagnostics: Hurst=%.2f  Entropy=%.2f  Regime=%s | "
        "Walk-forward checkpoints: %d",
        n, time.time() - start_time,
        theta, mu, ou_half_life,
        hurst_vals[-1], entropy_vals[-1], regime_labels[-1],
        len(checkpoints),
    )
    return result_df

# ══════════════════════════════════════════════════════════════════════════════
# MSF-ENHANCED SPREAD INDICATOR
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(max_entries=5, show_spinner=False)
def calculate_msf_spread(df, mood_col='Mood_Score', nifty_col='NIFTY', breadth_col='AD_RATIO'):
    """
    v2.0 MSF-Enhanced Spread Indicator.
    
    4 components (same purpose as v1.x — momentum/structure alignment detection):
      1. Momentum  — NIFTY ROC z-score (price velocity)
      2. Structure — Mood trend divergence + acceleration (mood curvature)
      3. Regime    — Adaptive-threshold directional count (market character)
      4. Flow      — Breadth divergence from mean (participation width)
    
    v2.0 changes:
      - Regime threshold adapts to local volatility (was: fixed 0.0033)
      - Inverse-variance weighting (was: fixed 30/25/25/20)
        Stable components get more weight — minimum-variance portfolio of signals.
    """
    start_time = time.time()
    result = pd.DataFrame(index=df.index)
    n = len(df)
    
    mood = df[mood_col].values if mood_col in df.columns else np.zeros(n)
    nifty = df[nifty_col].values if nifty_col in df.columns else mood
    breadth = df[breadth_col].values if breadth_col in df.columns else np.ones(n)
    
    mood_series = pd.Series(mood, index=df.index)
    nifty_series = pd.Series(nifty, index=df.index)
    breadth_series = pd.Series(breadth, index=df.index)
    
    if n == 0:
        logging.error("MSF Spread aborted — received an empty DataFrame; no rows to process.")
        return result
    
    # ── Component 1: Momentum (NIFTY ROC z-score) ──────────────────────
    roc_raw = nifty_series.pct_change(MSF_ROC_LEN)
    roc_z = zscore_clipped(roc_raw, MSF_WINDOW, MSF_ZSCORE_CLIP)
    momentum_norm = sigmoid(roc_z, 1.5)

    # ── Component 2: Structure (Mood trend divergence + acceleration) ──
    trend_fast = rolling_mean_fast(mood_series, 5)
    trend_slow = rolling_mean_fast(mood_series, MSF_WINDOW)
    trend_diff_z = zscore_clipped(trend_fast - trend_slow, MSF_WINDOW, MSF_ZSCORE_CLIP)
    mood_accel_raw = mood_series.diff(5).diff(5)
    mood_accel_z = zscore_clipped(mood_accel_raw, MSF_WINDOW, MSF_ZSCORE_CLIP)
    structure_z = (trend_diff_z + mood_accel_z) / np.sqrt(2.0)
    structure_norm = sigmoid(structure_z, 1.5)

    # ── Component 3: Regime (Adaptive threshold) ────────────────────────
    # v1.x: fixed 0.0033 threshold. v2.0: scales with local volatility.
    # A move is "directional" only if it exceeds half a local std.
    pct_vals = nifty_series.pct_change().fillna(0).values
    
    cs_pct = np.cumsum(pct_vals)
    cs2_pct = np.cumsum(pct_vals**2)
    cs_pct_shift = np.zeros(n, dtype=np.float64)
    cs_pct_shift[MSF_WINDOW:] = cs_pct[:-MSF_WINDOW]
    cs2_pct_shift = np.zeros(n, dtype=np.float64)
    cs2_pct_shift[MSF_WINDOW:] = cs2_pct[:-MSF_WINDOW]
    
    sums_pct = cs_pct - cs_pct_shift
    sums2_pct = cs2_pct - cs2_pct_shift
    counts_pct = np.minimum(np.arange(1, n + 1), MSF_WINDOW)
    
    var_pct = (sums2_pct - (sums_pct**2) / counts_pct) / np.maximum(counts_pct - 1, 1)
    rolling_vol = np.sqrt(np.maximum(var_pct, 0))
    
    rolling_vol[:4] = 0.003  # min_periods=5 fallback
    rolling_vol = np.where(rolling_vol < 1e-12, 0.003, rolling_vol)
    adaptive_threshold = np.clip(rolling_vol * 0.5, 0.001, None)

    regime_signals = np.where(pct_vals > adaptive_threshold, 1,
                     np.where(pct_vals < -adaptive_threshold, -1, 0))
    # Windowed sum (not cumsum) — prevents unbounded growth that creates
    # trend artifacts when cumsum drifts far from its rolling mean.
    regime_count = pd.Series(regime_signals, index=df.index).rolling(MSF_WINDOW, min_periods=1).sum()
    regime_raw = regime_count - rolling_mean_fast(regime_count, MSF_WINDOW)
    regime_z = zscore_clipped(regime_raw, MSF_WINDOW, MSF_ZSCORE_CLIP)
    regime_norm = sigmoid(regime_z, 1.5)

    # ── Component 4: Breadth Flow ───────────────────────────────────────
    breadth_ma = rolling_mean_fast(breadth_series, MSF_WINDOW)
    # Guard against near-zero denominators (not just exact zero)
    breadth_ma_safe = breadth_ma.where(breadth_ma.abs() > 1e-6, 1.0)
    breadth_ratio = breadth_series / breadth_ma_safe
    breadth_z = zscore_clipped(breadth_ratio - 1, MSF_WINDOW, MSF_ZSCORE_CLIP)
    flow_norm = sigmoid(breadth_z, 1.5)
    
    # ── Inverse-Variance Weighting ──────────────────────────────────────
    # Markowitz for signals: stable (low variance) components get more weight.
    components = {
        'momentum': momentum_norm,
        'structure': structure_norm,
        'regime': regime_norm,
        'flow': flow_norm,
    }
    
    tail_window = min(60, n)
    inv_vars = {}
    for name, comp in components.items():
        comp_vals = comp.values if hasattr(comp, 'values') else np.asarray(comp)
        tail = comp_vals[-tail_window:]
        tail_clean = tail[np.isfinite(tail)]
        var = np.var(tail_clean) if len(tail_clean) > 5 else 1.0
        inv_vars[name] = 1.0 / max(var, 1e-6)
    
    total_inv_var = sum(inv_vars.values())
    weights = {k: v / total_inv_var for k, v in inv_vars.items()}
    
    msf_raw = sum(weights[name] * comp for name, comp in components.items())
    msf_spread = msf_raw * MSF_SCALE

    result['msf_spread'] = msf_spread
    result['momentum']   = momentum_norm  * MSF_SCALE
    result['structure']  = structure_norm * MSF_SCALE
    result['regime']     = regime_norm    * MSF_SCALE
    result['flow']       = flow_norm      * MSF_SCALE
    
    weight_str = '  '.join(f"{k}={v:.0%}" for k, v in weights.items())
    logging.info(
        "MSF Spread complete — %.2fs | Inverse-variance weights: %s",
        time.time() - start_time, weight_str,
    )
    return result

# ══════════════════════════════════════════════════════════════════════════════
# SIMILAR PERIODS FINDER
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(max_entries=5, show_spinner=False)
def find_similar_periods(df, top_n=10, recency_weight=0.1):
    """
    v2.0 Similar Period Finder.
    
    3-part scoring:
      1. Mahalanobis distance (55%) — covariance-aware state matching
         Features: mood, volatility, NIFTY momentum, Hurst, entropy
      2. Trajectory cosine similarity (35%) — detrended mood path shape
      3. Exponential recency decay (10%) — prefer recent analogs
    """
    if df.empty or 'Mood_Score' not in df.columns:
        return []
    
    latest = df.iloc[-1]
    n = len(df)
    
    historical = df.iloc[:-30].copy() if n > 30 else df.iloc[:-1].copy()
    if historical.empty or len(historical) < 5:
        return []
    
    # ── Build Feature Vectors ───────────────────────────────────────────
    nifty_roc = df['NIFTY'].pct_change(MSF_ROC_LEN).fillna(0).values
    
    feature_cols = ['Mood_Score', 'Mood_Volatility']
    current_features = [latest['Mood_Score'], latest['Mood_Volatility']]
    hist_arrays = [historical['Mood_Score'].values, historical['Mood_Volatility'].values]
    
    # NIFTY momentum
    feature_cols.append('NIFTY_ROC')
    current_features.append(nifty_roc[-1] if len(nifty_roc) > 0 else 0.0)
    h_roc = nifty_roc[:len(historical)]
    if len(h_roc) < len(historical):
        h_roc = np.pad(h_roc, (len(historical) - len(h_roc), 0), constant_values=0)
    hist_arrays.append(h_roc[:len(historical)])
    
    # Hurst (if available from v2.0 engine)
    if 'Hurst' in df.columns:
        feature_cols.append('Hurst')
        current_features.append(latest['Hurst'])
        hist_arrays.append(historical['Hurst'].values)
    
    # Market Entropy (if available)
    if 'Market_Entropy' in df.columns:
        feature_cols.append('Market_Entropy')
        current_features.append(latest['Market_Entropy'])
        hist_arrays.append(historical['Market_Entropy'].values)
    
    current_vec = np.array(current_features, dtype=np.float64)
    hist_matrix = np.column_stack(hist_arrays)
    
    # Clean NaN/Inf → column medians
    for col in range(hist_matrix.shape[1]):
        col_data = hist_matrix[:, col]
        valid = np.isfinite(col_data)
        median_val = np.median(col_data[valid]) if valid.any() else 0.0
        hist_matrix[~valid, col] = median_val
    current_vec = np.where(np.isfinite(current_vec), current_vec, 0.0)
    
    # ── Part 1: Mahalanobis Distance (55%) ──────────────────────────────
    cov_matrix = np.cov(hist_matrix, rowvar=False)
    if cov_matrix.ndim < 2:
        cov_matrix = np.array([[max(float(cov_matrix), 1e-6)]])
    
    maha_dist = mahalanobis_distance_batch(hist_matrix, current_vec, cov_matrix)
    max_dist = maha_dist.max() if maha_dist.max() > 0 else 1.0
    maha_sim = 1.0 - (maha_dist / max_dist)
    
    # ── Part 2: Trajectory Cosine Similarity (SIMILAR_W_TRAJ) ──────────
    traj_sim = np.zeros(len(historical))

    if n > TRAJ_WINDOW:
        # Least-squares linear detrend (minimises residual variance, unlike endpoint
        # anchoring which distorts on V-shaped or reversal trajectories).
        _traj_x = np.arange(TRAJ_WINDOW, dtype=np.float64)
        _traj_xm = _traj_x - _traj_x.mean()
        _traj_xvar = np.sum(_traj_xm ** 2)

        def _ls_detrend(traj):
            if _traj_xvar < 1e-12:
                return traj - traj.mean()
            slope = np.sum(_traj_xm * (traj - traj.mean())) / _traj_xvar
            return traj - (traj.mean() + slope * _traj_xm)

        current_traj = df['Mood_Score'].values[-TRAJ_WINDOW:]
        ct_detrended = _ls_detrend(current_traj)

        for j, idx in enumerate(historical.index):
            pos = df.index.get_loc(idx)
            if pos >= TRAJ_WINDOW:
                hist_traj = df['Mood_Score'].values[pos - TRAJ_WINDOW:pos]
                ht_detrended = _ls_detrend(hist_traj)
                traj_sim[j] = (cosine_similarity(ct_detrended, ht_detrended) + 1) / 2

    # ── Part 3: Exponential Recency Decay (SIMILAR_W_RECV) ──────────────
    days_since = (latest['DATE'] - historical['DATE']).dt.days.values.astype(float)
    recency = np.exp(-np.log(2) * days_since / 365.0) * recency_weight
    recency_norm = recency / max(recency.max(), 1e-6)

    # ── Combined ────────────────────────────────────────────────────────
    combined = SIMILAR_W_MAHA * maha_sim + SIMILAR_W_TRAJ * traj_sim + SIMILAR_W_RECV * recency_norm
    
    historical = historical.copy()
    historical['similarity'] = combined
    top_similar = historical.nlargest(top_n, 'similarity')
    
    results = []
    nifty_vals = df['NIFTY'].values
    for _, row in top_similar.iterrows():
        idx_pos = df.index.get_loc(row.name)
        nifty_at = row['NIFTY'] if 'NIFTY' in row and row['NIFTY'] > 0 else None
        
        # Forward returns: what happened to NIFTY 30/60/90 days after this analog?
        fwd_returns = {}
        for horizon in [30, 60, 90]:
            fwd_idx = idx_pos + horizon
            if fwd_idx < len(nifty_vals) and nifty_at and nifty_at > 0:
                fwd_returns[horizon] = (nifty_vals[fwd_idx] / nifty_at - 1) * 100
            else:
                fwd_returns[horizon] = None
        
        results.append({
            'date': row['DATE'].strftime('%Y-%m-%d'),
            'similarity': row['similarity'],
            'mood_score': row['Mood_Score'],
            'mood': row['Mood'],
            'mood_volatility': row['Mood_Volatility'],
            'nifty': nifty_at or 0,
            'fwd_30d': fwd_returns.get(30),
            'fwd_60d': fwd_returns.get(60),
            'fwd_90d': fwd_returns.get(90),
        })
    
    return results


# ══════════════════════════════════════════════════════════════════════════════
# APPLICATION ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════

def _render_sidebar_masthead() -> None:
    """Top-of-sidebar product brand block."""
    sidebar_masthead(
        product="ARTHAGATI",
        sanskrit="अर्थगति",
        subtitle="Market Sentiment",
    )
    section_divider()


def _render_sidebar_passport() -> None:
    """Bottom-of-sidebar terminal spec card."""
    section_divider()
    sidebar_passport(
        version=VERSION,
        engine="OU · Kalman · Spearman",
        data_label=COMPANY,
    )


def main():
    # ── Session state ──────────────────────────────────────────────────────
    st.session_state.setdefault("analysis_started", False)
    st.session_state.setdefault("active_predictors", None)
    analysis_started = st.session_state["analysis_started"]

    # ── Landing state: masthead + Run Analysis button only ────────────────
    if not analysis_started:
        with st.sidebar:
            _render_sidebar_masthead()
            sidebar_title("Start", icon="play-circle")
            if st.button("Run Analysis", use_container_width=True, type="primary"):
                st.session_state["analysis_started"] = True
                st.rerun()
            _render_sidebar_passport()

        # ─ Main pane landing
        n_predictors = len(DEPENDENT_VARS)
        render_landing_page(version=VERSION, n_predictors=n_predictors)
        return

    # ── Analysis state: load data and populate predictor options first ─────
    _prog = st.empty()
    _progress_bar(_prog, 5, "Fetching market data", "Google Sheets · gviz API · CSV decode")
    raw_df = load_data()

    if raw_df is None:
        _prog.empty()
        st.stop()

    available_predictors = [
        col for col in raw_df.columns
        if col not in NON_PREDICTOR_COLS and pd.api.types.is_numeric_dtype(raw_df[col])
    ]
    current_preds = st.session_state.get("active_predictors")
    if not current_preds:
        st.session_state["active_predictors"] = tuple(available_predictors)
    else:
        valid = tuple(p for p in current_preds if p in available_predictors)
        st.session_state["active_predictors"] = valid if valid else tuple(available_predictors)

    # ── Sidebar — view mode + controls + model config ─────────────────────
    with st.sidebar:
        _render_sidebar_masthead()

        sidebar_title("View Mode", icon="grid")
        view_mode = st.radio(
            "View Mode",
            ["Historical Mood", "Similar Periods", "Correlation Analysis"],
            label_visibility="collapsed",
        )
        section_divider()

        sidebar_title("Controls", icon="settings")
        if st.button("Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        section_divider()

        sidebar_title("Model Configuration", icon="cpu")
        with st.expander("Predictor Columns", expanded=False):
            st.caption("Select predictors, then click Apply to recompute.")
            staging_predictors = st.multiselect(
                "Predictor Columns",
                options=available_predictors,
                default=list(st.session_state["active_predictors"]),
                label_visibility="collapsed",
                help="These columns are used as dependent variables for PE & EY correlation-weighted mood scoring.",
            )
            if not staging_predictors:
                st.warning("Select at least one predictor.")
                staging_predictors = list(st.session_state["active_predictors"])

            staging_set = set(staging_predictors)
            active_set = set(st.session_state["active_predictors"])
            has_changes = staging_set != active_set
            if has_changes:
                added = staging_set - active_set
                removed = active_set - staging_set
                changes = []
                if added:
                    changes.append(f"+{len(added)} added")
                if removed:
                    changes.append(f"−{len(removed)} removed")
                st.caption(f"Pending: {', '.join(changes)}")

            apply_clicked = st.button(
                "Apply Configuration" if has_changes else "No changes",
                use_container_width=True,
                disabled=not has_changes,
                type="primary" if has_changes else "secondary",
            )
            if apply_clicked and has_changes:
                st.session_state["active_predictors"] = tuple(staging_predictors)
                st.cache_data.clear()
                st.rerun()

            active_count = len(st.session_state["active_predictors"])
            total_count = len(available_predictors)
            if active_count != total_count:
                st.info(f"Active: {active_count}/{total_count} predictors")

        _render_sidebar_passport()

    # ── Main pane masthead (above the metric strip) ───────────────────────
    render_header(
        title="Arthagati",
        tagline="Market Sentiment Analysis  ·  OU · Kalman · Decay-Spearman",
    )

    # ── Stale-data warning ────────────────────────────────────────────────
    latest_date = raw_df["DATE"].max()
    ist_tz = pytz.timezone("Asia/Kolkata")
    today_ist = datetime.now(ist_tz).date()
    data_age_days = (pd.Timestamp(today_ist) - latest_date).days
    if data_age_days > 3:
        render_warning_box(
            title="Stale Data",
            content=(
                f"Last data point is {latest_date.strftime('%d %b %Y')} "
                f"({data_age_days} days ago). Scores reflect the last available data, "
                "not current market state. Update your Google Sheet."
            ),
        )

    # ── Run engine ────────────────────────────────────────────────────────
    _progress_bar(_prog, 40, "Computing correlations", "Decay-weighted Spearman · PE & EY anchors")
    selected_preds = st.session_state.get("active_predictors", tuple(available_predictors))

    _progress_bar(_prog, 65, "Running sentiment engine", "OU normalisation · Kalman smoothing · 5-layer pipeline")
    mood_df = calculate_historical_mood(raw_df, dependent_vars=selected_preds)
    if mood_df.empty:
        _prog.empty()
        st.error("Failed to calculate mood scores.")
        st.stop()

    _progress_bar(_prog, 88, "Computing MSF spread", "Momentum · Structure · Regime · Flow · inverse-variance weights")
    msf_df = calculate_msf_spread(mood_df)
    mood_df["MSF_Spread"] = msf_df["msf_spread"].values if not msf_df.empty else 0

    _progress_bar(_prog, 100, "Ready", "All systems nominal")
    time.sleep(0.25)
    _prog.empty()

    # ── Top metric strip ──────────────────────────────────────────────────
    latest      = mood_df.iloc[-1]
    mood_score  = latest["Mood_Score"]
    msf_spread  = latest["MSF_Spread"]

    if mood_score > 60:
        mood_class = "success"
    elif mood_score > 20:
        mood_class = "warning"
    elif mood_score < -60:
        mood_class = "danger"
    elif mood_score < -20:
        mood_class = "info"
    else:
        mood_class = "neutral"

    if msf_spread > 4:
        msf_class, msf_label = "danger", "Overbought"
    elif msf_spread > 2:
        msf_class, msf_label = "warning", "Bullish"
    elif msf_spread < -4:
        msf_class, msf_label = "success", "Oversold"
    elif msf_spread < -2:
        msf_class, msf_label = "info", "Bearish"
    else:
        msf_class, msf_label = "neutral", "Neutral"

    section_gap()
    col1, col2, col3, col4 = st.columns(4, gap="small")
    with col1:
        render_metric_card(
            label="Mood Score",
            value=f"{mood_score:.2f}",
            subtext=str(latest.get("Mood", "—")),
            color_class=mood_class,
            icon="activity",
        )
    with col2:
        render_metric_card(
            label="MSF Spread",
            value=f"{msf_spread:+.2f}",
            subtext=msf_label,
            color_class=msf_class,
            icon="chart",
        )
    with col3:
        render_metric_card(
            label="NIFTY 50",
            value=f"{latest['NIFTY']:,.0f}",
            subtext="Index level",
            color_class="warning",
            icon="trending-up",
        )
    with col4:
        render_metric_card(
            label="Analysis Date",
            value=latest["DATE"].strftime("%d %b"),
            subtext=latest["DATE"].strftime("%Y"),
            color_class="neutral",
            icon="globe",
        )

    # ── Diagnostics strip ─────────────────────────────────────────────────
    current_regime = latest.get("Regime", "Unknown")
    _reg_color, reg_class = REGIME_STYLES.get(current_regime, (C_MUTED, "neutral"))

    d1, d2, d3, d4 = st.columns(4, gap="small")
    with d1:
        render_metric_card(
            label="Market Regime",
            value=str(current_regime),
            subtext="Hurst + Entropy",
            color_class=reg_class,
            icon="compass",
        )
    with d2:
        ou_hl = latest.get("OU_Half_Life", 0)
        render_metric_card(
            label="OU Half-Life",
            value=f"{ou_hl:.0f}d",
            subtext="Expected reversion time",
            color_class="warning",
            icon="cpu",
        )
    with d3:
        h_val = latest.get("Hurst", 0.5)
        h_label = "Trending" if h_val > 0.55 else "Random" if h_val > 0.45 else "Reverting"
        h_class = "success" if h_val > 0.55 else "neutral" if h_val > 0.45 else "info"
        render_metric_card(
            label="Hurst Exponent",
            value=f"{h_val:.2f}",
            subtext=h_label,
            color_class=h_class,
            icon="trending-up",
        )
    with d4:
        s_val = latest.get("Market_Entropy", 0.5)
        s_label = "Disordered" if s_val > 0.6 else "Ordered" if s_val < 0.4 else "Mixed"
        s_class = "danger" if s_val > 0.6 else "success" if s_val < 0.4 else "neutral"
        render_metric_card(
            label="Market Entropy",
            value=f"{s_val:.2f}",
            subtext=s_label,
            color_class=s_class,
            icon="zap",
        )

    section_gap()

    # ── View dispatch ─────────────────────────────────────────────────────
    if view_mode == "Historical Mood":
        render_historical_mood(
            mood_df, msf_df,
            timeframes=TIMEFRAMES,
            regime_styles=REGIME_STYLES,
            mood_scale=MOOD_SCALE,
            ou_proj_days=OU_PROJ_DAYS,
        )
    elif view_mode == "Similar Periods":
        render_similar_periods(
            mood_df,
            find_similar_periods=find_similar_periods,
            backtest_horizon=BACKTEST_HORIZON,
        )
    else:
        render_correlation_analysis(
            raw_df,
            active_preds=st.session_state.get("active_predictors", tuple(available_predictors)),
            non_predictor_cols=NON_PREDICTOR_COLS,
            calculate_anchor_correlations=calculate_anchor_correlations,
            shannon_entropy=shannon_entropy,
        )

    # ── Footer ────────────────────────────────────────────────────────────
    utc_now = datetime.now(pytz.UTC)
    ist_now = utc_now.astimezone(pytz.timezone("Asia/Kolkata"))
    current_time_ist = ist_now.strftime("%Y-%m-%d %H:%M:%S IST")
    render_footer(PRODUCT_NAME, COMPANY, VERSION, current_time_ist)


# ══════════════════════════════════════════════════════════════════════════════
# RUN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

main()
