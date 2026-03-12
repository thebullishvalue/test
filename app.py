# -*- coding: utf-8 -*-
"""
AARAMBH (आरंभ) v2.0 - Fair Value Breadth | A Hemrek Capital Product
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Walk-Forward Valuation · OU Mean-Reversion · Kalman-Filtered Conviction
Multi-timeframe breadth analysis for market reversals.

v2.0 Changes from v1.1:
  1. Walk-forward (expanding window) regression — no look-ahead bias
  2. OU estimation on residuals — half-life, forward projection
  3. Kalman smoothing on conviction — adaptive noise filtering + confidence bands
  4. Model disagreement metric — ensemble uncertainty quantification (Upgraded to GP Epistemic Uncertainty)
  5. Residual Hurst exponent — empirical mean-reversion validation
  6. Apply button for predictors — staging → commit pattern
  7. Data staleness warning
  8. Improved divergence detection — swing-based, not point-to-point
  9. Backend Upgrade: BMR-SM (Bayesian Manifold Regime-Switching Model) integration.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.signal import argrelextrema
from datetime import datetime, timedelta
import logging
import warnings
import time

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ── Dependencies ────────────────────────────────────────────────────────────
try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    sm = None
    STATSMODELS_AVAILABLE = False

try:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, DotProduct
    from sklearn.mixture import GaussianMixture
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ── Constants ───────────────────────────────────────────────────────────────
VERSION = "v2.0.0"
PRODUCT_NAME = "Aarambh"
COMPANY = "Hemrek Capital"

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AARAMBH | Fair Value Breadth",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS (Hemrek Design System — shared with Arthagati, Nirnay, Pragyam)
# ══════════════════════════════════════════════════════════════════════════════

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
# MATHEMATICAL PRIMITIVES (shared theory base with Arthagati v2.1)
# ══════════════════════════════════════════════════════════════════════════════

def ornstein_uhlenbeck_estimate(series, dt=1.0):
    """
    Estimate OU parameters from residual series: dx = θ(μ−x)dt + σdW.
    Returns: (theta, mu, sigma)
    Used in: residual half-life, forward projection, normalization.
    """
    x = np.asarray(series, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return 0.05, 0.0, max(np.std(x), 1e-6) if len(x) > 1 else (0.05, 0.0, 1.0)
    
    x_lag = x[:-1]
    x_curr = x[1:]
    n = len(x_lag)
    
    sx = np.sum(x_lag)
    sy = np.sum(x_curr)
    sxx = np.sum(x_lag ** 2)
    sxy = np.sum(x_lag * x_curr)
    syy = np.sum(x_curr ** 2)
    
    denom = n * sxx - sx ** 2
    if abs(denom) < 1e-12:
        return 0.05, np.mean(x), max(np.std(x), 1e-6)
    
    a = (n * sxy - sx * sy) / denom
    b = (sy * sxx - sx * sxy) / denom
    
    a = np.clip(a, 1e-6, 1.0 - 1e-6)
    
    theta = -np.log(a) / dt
    mu = b / (1 - a)
    
    residuals = x_curr - a * x_lag - b
    sigma_sq = np.var(residuals)
    sigma = np.sqrt(max(sigma_sq * 2 * theta / (1 - a ** 2), 1e-12))
    
    return max(theta, 1e-4), mu, max(sigma, 1e-6)


def kalman_filter_1d(observations, process_var=None, measurement_var=None):
    """
    1D Kalman filter for conviction smoothing.
    Returns: (filtered_state, kalman_gains, estimate_variances)
    """
    obs = np.asarray(observations, dtype=np.float64)
    n = len(obs)
    if n == 0:
        return np.array([]), np.array([]), np.array([])
    
    if process_var is None:
        diffs = np.diff(obs[np.isfinite(obs)])
        process_var = max(np.var(diffs) * 0.1, 1e-8) if len(diffs) > 1 else 1e-3
    if measurement_var is None:
        clean = obs[np.isfinite(obs)]
        measurement_var = max(np.var(clean) * 0.5, 1e-8) if len(clean) > 1 else 1.0
    
    state = obs[0] if np.isfinite(obs[0]) else 0.0
    estimate_var = measurement_var
    filtered = np.zeros(n)
    gains = np.zeros(n)
    variances = np.zeros(n)
    filtered[0] = state
    variances[0] = estimate_var
    
    for i in range(1, n):
        pred_var = estimate_var + process_var
        if np.isfinite(obs[i]):
            K = pred_var / (pred_var + measurement_var)
            state = state + K * (obs[i] - state)
            estimate_var = (1 - K) * pred_var
            gains[i] = K
        else:
            estimate_var = pred_var
        filtered[i] = state
        variances[i] = estimate_var
    
    return filtered, gains, variances


def hurst_rs(series, max_lag=None):
    """Hurst exponent via Rescaled Range (R/S) analysis."""
    x = np.asarray(series, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 20:
        return 0.5
    
    if max_lag is None:
        max_lag = min(n // 2, 100)
    
    lags = []
    rs_values = []
    
    for lag in range(10, max_lag + 1, max(1, max_lag // 20)):
        rs_list = []
        for start in range(0, n - lag, lag):
            segment = x[start:start + lag]
            if len(segment) < 10:
                continue
            mean_seg = np.mean(segment)
            dev = np.cumsum(segment - mean_seg)
            R = np.max(dev) - np.min(dev)
            S = np.std(segment, ddof=1)
            if S > 1e-10:
                rs_list.append(R / S)
        
        if rs_list:
            lags.append(lag)
            rs_values.append(np.mean(rs_list))
    
    if len(lags) < 3:
        return 0.5
    
    log_lags = np.log(lags)
    log_rs = np.log(rs_values)
    
    slope, _, _, _, _ = stats.linregress(log_lags, log_rs)
    return np.clip(slope, 0.01, 0.99)


# ══════════════════════════════════════════════════════════════════════════════
# FAIR VALUE BREADTH ENGINE v2.0 (BMR-SM Integration)
# ══════════════════════════════════════════════════════════════════════════════

class FairValueEngine:
    """
    v2.0 Fair Value Breadth Engine - Powered by Bayesian Manifold Regime-Switching Model (BMR-SM).
    
    Key changes from v1.1:
      1. Walk-forward regression: at each time T, fit only on [0..T), predict T
      2. OU estimation on out-of-sample residuals
      3. Kalman-filtered conviction with confidence bands
      4. Model disagreement (ensemble spread) replaced with GP Epistemic Uncertainty
      5. Residual Hurst exponent for mean-reversion validation
      6. Swing-based divergence detection
      7. Component A: Gaussian Process mapping the equilibrium manifold with ARD
      8. Component C & D: GMM/HMM logic for extracting Z_t synthesis metric.
    """
    
    LOOKBACKS = [5, 10, 20, 50, 100]
    MIN_TRAIN_SIZE = 20  # Minimum expanding window before first prediction
    
    def __init__(self):
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
    
    def fit(self, X, y, feature_names=None, progress_callback=None):
        """Walk-forward fit: expanding window regression, then all analytics."""
        start_time = time.time()
        self.feature_names = feature_names or [f'X{i}' for i in range(X.shape[1])]
        self.n_samples = len(y)
        self.y = y.copy()
        self.X = X.copy()
        
        n = self.n_samples
        self.predictions = np.full(n, np.nan)
        self.model_spread = np.zeros(n)  # Std across model predictions (disagreement / uncertainty)
        
        refit_step = 5  # Refit models weekly (every 5 steps) to significantly optimize performance
        last_models = {'gp': None}
        current_scaler = None
        
        for t in range(n):
            if progress_callback and t % max(1, n // 20) == 0:
                progress_callback(t / n, f"Walking forward: {t}/{n} data points...")
                
            if t < self.MIN_TRAIN_SIZE:
                # Not enough data for regression — use expanding mean
                self.predictions[t] = np.mean(y[:t + 1])
                self.model_spread[t] = 0.0
            else:
                X_pred = X[t:t + 1]
                preds_at_t = []
                
                # Fit ensemble periodically instead of every single day to avoid extreme slowness
                if t == self.MIN_TRAIN_SIZE or t % refit_step == 0:
                    X_train = X[:t]
                    y_train = y[:t]
                    
                    if SKLEARN_AVAILABLE and self.scaler is not None:
                        scaler_t = StandardScaler()
                        X_train_s = scaler_t.fit_transform(X_train)
                        current_scaler = scaler_t
                        
                        # --- BMR-SM: Component A (Equilibrium Manifold via GP) ---
                        # Bound training window to 300 samples to keep O(N^3) GP fast in UI loop
                        MAX_GP_SAMPLES = 300
                        if len(X_train_s) > MAX_GP_SAMPLES:
                            X_train_gp = X_train_s[-MAX_GP_SAMPLES:]
                            y_train_gp = y_train[-MAX_GP_SAMPLES:]
                        else:
                            X_train_gp = X_train_s
                            y_train_gp = y_train
                            
                        n_features = X_train_gp.shape[1]
                        
                        try:
                            # Composite Kernel: ARD (RBF) + Linear (DotProduct) + Noise (WhiteKernel)
                            kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(n_features)) \
                                     + DotProduct() + WhiteKernel(noise_level=0.1)
                                     
                            gp = GaussianProcessRegressor(
                                kernel=kernel, 
                                n_restarts_optimizer=0, 
                                normalize_y=True,
                                random_state=42
                            )
                            gp.fit(X_train_gp, y_train_gp)
                            last_models['gp'] = gp
                        except Exception:
                            last_models['gp'] = None
                
                # Predict Step (using latest models)
                if SKLEARN_AVAILABLE and current_scaler is not None:
                    X_pred_s = current_scaler.transform(X_pred)
                    if last_models['gp'] is not None:
                        try:
                            mean_pred, std_pred = last_models['gp'].predict(X_pred_s, return_std=True)
                            preds_at_t.append(mean_pred[0])
                            # Map Epistemic Uncertainty directly to UI Model Spread
                            self.model_spread[t] = std_pred[0]
                        except Exception:
                            pass
                
                if preds_at_t:
                    self.predictions[t] = preds_at_t[0]
                    # Epistemic uncertainty is set during prediction
                else:
                    self.predictions[t] = np.mean(y[:t + 1])
                    self.model_spread[t] = 0.0
        
        if progress_callback:
            progress_callback(1.0, "Computing multi-lookback signals...")
            
        self.residuals = y - self.predictions
        
        # ── Final full-sample fit for model stats (OOS R² computed separately) ──
        self._compute_model_stats()
        
        # ── All downstream analytics on OOS residuals ───────────────────
        self._compute_multi_lookback_signals()
        self._compute_breadth_metrics()
        self._compute_kalman_conviction()
        self._find_pivots()
        self._compute_divergences()
        self._compute_forward_returns()
        self._compute_ou_diagnostics()
        self._compute_hurst()
        
        elapsed = time.time() - start_time
        logging.info(f"v2.0 engine [{n} obs, {len(self.feature_names)} features] in {elapsed:.1f}s")
        
        if progress_callback:
            progress_callback(1.0, "Done.")
            
        return self
    
    def _compute_model_stats(self):
        """Compute OOS model fit statistics (only on walk-forward portion)."""
        # Only count predictions from MIN_TRAIN_SIZE onwards (true OOS)
        oos_mask = np.arange(self.n_samples) >= self.MIN_TRAIN_SIZE
        y_oos = self.y[oos_mask]
        pred_oos = self.predictions[oos_mask]
        
        valid = np.isfinite(pred_oos)
        y_valid = y_oos[valid]
        p_valid = pred_oos[valid]
        
        if len(y_valid) > 2 and SKLEARN_AVAILABLE:
            self.model_stats = {
                'r2_oos': r2_score(y_valid, p_valid),
                'rmse_oos': np.sqrt(mean_squared_error(y_valid, p_valid)),
                'mae_oos': mean_absolute_error(y_valid, p_valid),
                'n_obs': len(y_valid),
                'n_features': len(self.feature_names),
                'avg_model_spread': np.mean(self.model_spread[oos_mask]),
            }
        else:
            ss_res = np.sum((y_valid - p_valid) ** 2)
            ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
            self.model_stats = {
                'r2_oos': 1 - ss_res / max(ss_tot, 1e-10),
                'rmse_oos': np.sqrt(np.mean((y_valid - p_valid) ** 2)),
                'mae_oos': np.mean(np.abs(y_valid - p_valid)),
                'n_obs': len(y_valid),
                'n_features': len(self.feature_names),
                'avg_model_spread': np.mean(self.model_spread[oos_mask]),
            }
    
    def _compute_multi_lookback_signals(self):
        """Compute z-scores and zones for each lookback period on OOS residuals."""
        r = self.residuals
        n = len(r)
        
        self.lookback_data = {}
        
        for lb in self.LOOKBACKS:
            if n < lb:
                continue
            
            rolling_mean = pd.Series(r).rolling(lb, min_periods=max(lb // 2, 5)).mean().values
            rolling_std = pd.Series(r).rolling(lb, min_periods=max(lb // 2, 5)).std().values
            z_scores = np.where(rolling_std > 1e-8, (r - rolling_mean) / rolling_std, 0)
            
            zones = np.full(n, 'N/A', dtype=object)
            for i in range(n):
                z = z_scores[i]
                if np.isnan(z):
                    continue
                elif z > 2:
                    zones[i] = 'Extreme Over'
                elif z > 1:
                    zones[i] = 'Overvalued'
                elif z > -1:
                    zones[i] = 'Fair Value'
                elif z > -2:
                    zones[i] = 'Undervalued'
                else:
                    zones[i] = 'Extreme Under'
            
            buy_signals = np.zeros(n, dtype=bool)
            sell_signals = np.zeros(n, dtype=bool)
            for i in range(1, n):
                if np.isfinite(z_scores[i]) and np.isfinite(z_scores[i - 1]):
                    if z_scores[i] < -1 and z_scores[i - 1] >= -1:
                        buy_signals[i] = True
                    if z_scores[i] > 1 and z_scores[i - 1] <= 1:
                        sell_signals[i] = True
            
            self.lookback_data[lb] = {
                'z_scores': z_scores,
                'zones': zones,
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'rolling_mean': rolling_mean,
                'rolling_std': rolling_std,
            }
        
        # Master time series
        self.ts_data = pd.DataFrame({
            'Actual': self.y,
            'FairValue': self.predictions,
            'Residual': self.residuals,
            'ModelSpread': self.model_spread,
        })
        
        for lb in self.lookback_data:
            self.ts_data[f'Z_{lb}'] = self.lookback_data[lb]['z_scores']
            self.ts_data[f'Zone_{lb}'] = self.lookback_data[lb]['zones']
            self.ts_data[f'Buy_{lb}'] = self.lookback_data[lb]['buy_signals']
            self.ts_data[f'Sell_{lb}'] = self.lookback_data[lb]['sell_signals']
    
    def _compute_breadth_metrics(self):
        """Compute breadth metrics across lookback periods."""
        n = len(self.ts_data)
        valid_lookbacks = [lb for lb in self.LOOKBACKS if lb in self.lookback_data]
        
        oversold_count = np.zeros(n)
        overbought_count = np.zeros(n)
        extreme_oversold = np.zeros(n)
        extreme_overbought = np.zeros(n)
        buy_signal_count = np.zeros(n)
        sell_signal_count = np.zeros(n)
        avg_z = np.zeros(n)
        
        for i in range(n):
            z_values = []
            for lb in valid_lookbacks:
                zone = self.lookback_data[lb]['zones'][i]
                z = self.lookback_data[lb]['z_scores'][i]
                
                if zone == 'Extreme Under':
                    extreme_oversold[i] += 1
                    oversold_count[i] += 1
                elif zone == 'Undervalued':
                    oversold_count[i] += 1
                elif zone == 'Extreme Over':
                    extreme_overbought[i] += 1
                    overbought_count[i] += 1
                elif zone == 'Overvalued':
                    overbought_count[i] += 1
                
                if self.lookback_data[lb]['buy_signals'][i]:
                    buy_signal_count[i] += 1
                if self.lookback_data[lb]['sell_signals'][i]:
                    sell_signal_count[i] += 1
                
                if not np.isnan(z):
                    z_values.append(z)
            
            if z_values:
                avg_z[i] = np.mean(z_values)
        
        num_lb = max(len(valid_lookbacks), 1)
        
        self.ts_data['OversoldBreadth'] = oversold_count / num_lb * 100
        self.ts_data['OverboughtBreadth'] = overbought_count / num_lb * 100
        self.ts_data['ExtremeOversold'] = extreme_oversold / num_lb * 100
        self.ts_data['ExtremeOverbought'] = extreme_overbought / num_lb * 100
        self.ts_data['BuySignalBreadth'] = buy_signal_count
        self.ts_data['SellSignalBreadth'] = sell_signal_count
        self.ts_data['AvgZ'] = avg_z
        
        # Raw conviction (before Kalman)
        self.ts_data['ConvictionRaw'] = (
            self.ts_data['OverboughtBreadth'] - self.ts_data['OversoldBreadth'] +
            (self.ts_data['ExtremeOverbought'] - self.ts_data['ExtremeOversold']) * 0.5
        )
    
    def _compute_kalman_conviction(self):
        """Apply Kalman filter to raw conviction for smooth, confident signal."""
        # --- BMR-SM: Component C & D (Regime Identification & Synthesis) ---
        r = self.ts_data['Residual'].values
        valid_mask = np.isfinite(r)
        r_valid = r[valid_mask]
        
        if SKLEARN_AVAILABLE and len(r_valid) > 30:
            # Fit HMM/GMM to identify 3 latent states: Discount, Equilibrium, Premium
            gmm = GaussianMixture(n_components=3, random_state=42)
            r_valid_2d = r_valid.reshape(-1, 1)
            gmm.fit(r_valid_2d)
            
            # Sort states by means: 0 = Discount (Negative), 1 = Equilibrium (~0), 2 = Premium (Positive)
            order = np.argsort(gmm.means_.flatten())
            means = gmm.means_.flatten()[order]
            stds = np.sqrt(gmm.covariances_.flatten())[order]
            
            # P(S_t = k | \delta_{1:t})
            probas = gmm.predict_proba(r_valid_2d)[:, order]
            
            # Synthesis: Z_t = \sum P(S_t=k) * (\delta_t - \mu_k) / \sigma_k
            z_t = np.zeros(len(r_valid))
            for k in range(3):
                safe_std = max(stds[k], 1e-6)
                z_t += probas[:, k] * ((r_valid - means[k]) / safe_std)
                
            Z_score_full = np.zeros(len(r))
            Z_score_full[valid_mask] = z_t
            
            # Scale BMR-SM Regime Score to UI Conviction scale (-100 to 100)
            raw_conviction = np.clip(Z_score_full * 33.33, -100, 100)
        else:
            raw_conviction = self.ts_data['ConvictionRaw'].values
            
        # Overwrite legacy breadth-based conviction with advanced BMR-SM metric
        self.ts_data['ConvictionRaw'] = raw_conviction
        
        filtered, gains, variances = kalman_filter_1d(raw_conviction)
        
        kalman_std = np.sqrt(np.maximum(variances, 0))
        
        self.ts_data['ConvictionScore'] = filtered
        self.ts_data['ConvictionUpper'] = np.clip(filtered + 1.96 * kalman_std, -100, 100)
        self.ts_data['ConvictionLower'] = np.clip(filtered - 1.96 * kalman_std, -100, 100)
        
        # Regime classification on smoothed conviction
        regimes = []
        for score in self.ts_data['ConvictionScore']:
            if score < -40:
                regimes.append('STRONGLY OVERSOLD')
            elif score < -20:
                regimes.append('OVERSOLD')
            elif score > 40:
                regimes.append('STRONGLY OVERBOUGHT')
            elif score > 20:
                regimes.append('OVERBOUGHT')
            else:
                regimes.append('NEUTRAL')
        self.ts_data['Regime'] = regimes
    
    def _compute_divergences(self):
        """Swing-based divergence detection (improved from v1.1 point-to-point)."""
        n = len(self.ts_data)
        price = self.y
        residual = self.residuals
        
        bull_div = np.zeros(n, dtype=bool)
        bear_div = np.zeros(n, dtype=bool)
        
        # Find swing lows/highs using local extrema
        order = 5
        if n < order * 3:
            self.ts_data['BullishDiv'] = bull_div
            self.ts_data['BearishDiv'] = bear_div
            return
        
        price_lows = argrelextrema(price, np.less, order=order)[0]
        price_highs = argrelextrema(price, np.greater, order=order)[0]
        
        # Bullish divergence: price makes lower low, residual makes higher low
        for i in range(1, len(price_lows)):
            idx_curr = price_lows[i]
            idx_prev = price_lows[i - 1]
            if price[idx_curr] < price[idx_prev] and residual[idx_curr] > residual[idx_prev]:
                if residual[idx_curr] < -np.std(residual[:idx_curr + 1]) * 0.5:
                    bull_div[idx_curr] = True
        
        # Bearish divergence: price makes higher high, residual makes lower high
        for i in range(1, len(price_highs)):
            idx_curr = price_highs[i]
            idx_prev = price_highs[i - 1]
            if price[idx_curr] > price[idx_prev] and residual[idx_curr] < residual[idx_prev]:
                if residual[idx_curr] > np.std(residual[:idx_curr + 1]) * 0.5:
                    bear_div[idx_curr] = True
        
        self.ts_data['BullishDiv'] = bull_div
        self.ts_data['BearishDiv'] = bear_div
    
    def _find_pivots(self, order=5):
        """Find pivot points in residuals."""
        r = self.residuals
        
        max_idx = argrelextrema(r, np.greater, order=order)[0]
        min_idx = argrelextrema(r, np.less, order=order)[0]
        
        self.pivots = {
            'tops': max_idx,
            'bottoms': min_idx,
            'avg_top': np.mean(r[max_idx]) if len(max_idx) > 0 else np.mean(r) + np.std(r),
            'avg_bottom': np.mean(r[min_idx]) if len(min_idx) > 0 else np.mean(r) - np.std(r),
        }
        
        self.ts_data['IsPivotTop'] = False
        self.ts_data['IsPivotBottom'] = False
        if len(max_idx) > 0:
            self.ts_data.loc[max_idx, 'IsPivotTop'] = True
        if len(min_idx) > 0:
            self.ts_data.loc[min_idx, 'IsPivotBottom'] = True
        
        self.residual_stats = {
            'mean': np.mean(r),
            'std': np.std(r),
            'current': r[-1],
            'current_zscore': (r[-1] - np.mean(r)) / max(np.std(r), 1e-8),
            'percentile': stats.percentileofscore(r, r[-1]),
            'min': np.min(r),
            'max': np.max(r),
        }
    
    def _compute_forward_returns(self):
        """Compute forward returns for signal validation."""
        n = len(self.ts_data)
        y = self.y
        for period in [5, 10, 20]:
            fwd_ret = np.full(n, np.nan)
            for i in range(n - period):
                if y[i] > 0:
                    fwd_ret[i] = (y[i + period] - y[i]) / y[i] * 100
            self.ts_data[f'FwdRet_{period}'] = fwd_ret
    
    def _compute_ou_diagnostics(self):
        """OU estimation on out-of-sample residuals."""
        r = self.residuals
        oos_r = r[self.MIN_TRAIN_SIZE:]
        
        if len(oos_r) > 30:
            theta, mu, sigma = ornstein_uhlenbeck_estimate(oos_r)
        else:
            theta, mu, sigma = 0.05, 0.0, max(np.std(r), 1e-6)
        
        self.ou_params = {
            'theta': theta,
            'mu': mu,
            'sigma': sigma,
            'half_life': np.log(2) / max(theta, 1e-4),
            'stationary_std': sigma / np.sqrt(2 * max(theta, 1e-4)),
        }
        
        # Forward projection: E[residual(t+n)] = μ + (r_current - μ) * exp(-θ*n)
        current_r = r[-1]
        proj_days = np.arange(1, 91)
        self.ou_projection = mu + (current_r - mu) * np.exp(-theta * proj_days)
    
    def _compute_hurst(self):
        """Hurst exponent of residuals — empirical mean-reversion test."""
        r = self.residuals
        oos_r = r[self.MIN_TRAIN_SIZE:]
        self.hurst = hurst_rs(oos_r) if len(oos_r) > 30 else 0.5
    
    def get_current_signal(self):
        """Get comprehensive current signal."""
        ts = self.ts_data
        current = ts.iloc[-1]
        
        conviction = current['ConvictionScore']
        regime = current['Regime']
        oversold_breadth = current['OversoldBreadth']
        overbought_breadth = current['OverboughtBreadth']
        model_spread = current['ModelSpread']
        
        if conviction < -60:
            signal, strength = 'BUY', 'STRONG'
        elif conviction < -40:
            signal, strength = 'BUY', 'MODERATE'
        elif conviction < -20:
            signal, strength = 'BUY', 'WEAK'
        elif conviction > 60:
            signal, strength = 'SELL', 'STRONG'
        elif conviction > 40:
            signal, strength = 'SELL', 'MODERATE'
        elif conviction > 20:
            signal, strength = 'SELL', 'WEAK'
        else:
            signal, strength = 'HOLD', 'NEUTRAL'
        
        if signal == 'BUY':
            confidence = 'HIGH' if oversold_breadth >= 80 else 'MEDIUM' if oversold_breadth >= 60 else 'LOW'
        elif signal == 'SELL':
            confidence = 'HIGH' if overbought_breadth >= 80 else 'MEDIUM' if overbought_breadth >= 60 else 'LOW'
        else:
            confidence = 'N/A'
        
        return {
            'signal': signal,
            'strength': strength,
            'confidence': confidence,
            'conviction_score': conviction,
            'conviction_upper': current['ConvictionUpper'],
            'conviction_lower': current['ConvictionLower'],
            'regime': regime,
            'oversold_breadth': oversold_breadth,
            'overbought_breadth': overbought_breadth,
            'residual': current['Residual'],
            'fair_value': current['FairValue'],
            'actual': current['Actual'],
            'avg_z': current['AvgZ'],
            'model_spread': model_spread,
            'has_bullish_div': current['BullishDiv'],
            'has_bearish_div': current['BearishDiv'],
            'ou_half_life': self.ou_params['half_life'],
            'hurst': self.hurst,
        }
    
    def get_model_stats(self):
        return self.model_stats
    
    def get_regime_stats(self):
        ts = self.ts_data
        regime_counts = ts['Regime'].value_counts()
        return {
            'strongly_oversold': regime_counts.get('STRONGLY OVERSOLD', 0),
            'oversold': regime_counts.get('OVERSOLD', 0),
            'neutral': regime_counts.get('NEUTRAL', 0),
            'overbought': regime_counts.get('OVERBOUGHT', 0),
            'strongly_overbought': regime_counts.get('STRONGLY OVERBOUGHT', 0),
            'current_regime': ts['Regime'].iloc[-1],
            'total_buy_signals': sum(ts['BuySignalBreadth']),
            'total_sell_signals': sum(ts['SellSignalBreadth']),
            'total_bull_div': ts['BullishDiv'].sum(),
            'total_bear_div': ts['BearishDiv'].sum(),
            'total_pivot_tops': ts['IsPivotTop'].sum(),
            'total_pivot_bottoms': ts['IsPivotBottom'].sum(),
        }
    
    def get_signal_performance(self):
        ts = self.ts_data
        results = {}
        for period in [5, 10, 20]:
            buy_returns = []
            sell_returns = []
            for i in range(len(ts) - period):
                if ts['ConvictionScore'].iloc[i] < -40:
                    fwd = ts[f'FwdRet_{period}'].iloc[i]
                    if not pd.isna(fwd):
                        buy_returns.append(fwd)
                if ts['ConvictionScore'].iloc[i] > 40:
                    fwd = ts[f'FwdRet_{period}'].iloc[i]
                    if not pd.isna(fwd):
                        sell_returns.append(-fwd)
            results[period] = {
                'buy_avg': np.mean(buy_returns) if buy_returns else 0,
                'buy_hit': np.mean([r > 0 for r in buy_returns]) if buy_returns else 0,
                'buy_count': len(buy_returns),
                'sell_avg': np.mean(sell_returns) if sell_returns else 0,
                'sell_hit': np.mean([r > 0 for r in sell_returns]) if sell_returns else 0,
                'sell_count': len(sell_returns),
            }
        return results


# ══════════════════════════════════════════════════════════════════════════════
# DATA UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def load_google_sheet(sheet_url):
    try:
        import re
        sheet_id_match = re.search(r'/d/([a-zA-Z0-9-_]+)', sheet_url)
        if not sheet_id_match:
            return None, "Invalid URL"
        sheet_id = sheet_id_match.group(1)
        gid_match = re.search(r'gid=(\d+)', sheet_url)
        gid = gid_match.group(1) if gid_match else '0'
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
        df = pd.read_csv(csv_url)
        return df, None
    except Exception as e:
        return None, str(e)


def clean_data(df, target, features, date_col=None):
    features_list = list(features)  # Enforce list type to prevent tuple concatenation errors
    cols = [target] + features_list
    if date_col and date_col != "None" and date_col in df.columns:
        cols.append(date_col)
    data = df[cols].copy()
    for col in [target] + features_list:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna()
    numeric_subset = data[[target] + features_list]
    is_finite = np.isfinite(numeric_subset).all(axis=1)
    data = data[is_finite]
    if date_col and date_col != "None" and date_col in data.columns:
        try:
            data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
            data = data.dropna(subset=[date_col])
            data = data.sort_values(date_col)
        except Exception:
            pass
    return data.reset_index(drop=True)


def update_chart_theme(fig):
    fig.update_layout(
        template="plotly_dark", plot_bgcolor="#1A1A1A", paper_bgcolor="#1A1A1A",
        font=dict(family="Inter", color="#EAEAEA"),
        xaxis=dict(gridcolor="#2A2A2A", zerolinecolor="#3A3A3A"),
        yaxis=dict(gridcolor="#2A2A2A", zerolinecolor="#3A3A3A"),
        margin=dict(t=40, l=20, r=20, b=20),
        hoverlabel=dict(bgcolor="#2A2A2A", font_size=12)
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# LANDING PAGE
# ══════════════════════════════════════════════════════════════════════════════

def render_landing_page():
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


def render_footer():
    from datetime import timezone
    utc_now = datetime.now(timezone.utc)
    ist_now = utc_now + timedelta(hours=5, minutes=30)
    current_time_ist = ist_now.strftime("%Y-%m-%d %H:%M:%S IST")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.caption(f"© 2026 {PRODUCT_NAME} | {COMPANY} | {VERSION} | {current_time_ist}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Sidebar ─────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; margin-bottom: 1rem;">
            <div style="font-size: 1.75rem; font-weight: 800; color: #FFC300;">AARAMBH</div>
            <div style="color: #888888; font-size: 0.75rem; margin-top: 0.25rem;">आरंभ | Fair Value Breadth</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-title">📁 Data Source</div>', unsafe_allow_html=True)
        data_source = st.radio("Source", ["📤 Upload", "📊 Google Sheets"], horizontal=True, label_visibility="collapsed")
        
        df = None
        
        if data_source == "📤 Upload":
            uploaded_file = st.file_uploader("CSV/Excel", type=['csv', 'xlsx'], label_visibility="collapsed")
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                except Exception as e:
                    st.error(f"Error: {e}")
                    return
        else:
            default_url = "https://docs.google.com/spreadsheets/d/1po7z42n3dYIQGAvn0D1-a4pmyxpnGPQ13TrNi3DB5_c/edit?gid=1938234952#gid=1938234952"
            sheet_url = st.text_input("Sheet URL", value=default_url, label_visibility="collapsed")
            if st.button("🔄 LOAD DATA", type="primary"):
                with st.spinner("Loading..."):
                    df, error = load_google_sheet(sheet_url)
                    if error:
                        st.error(f"Failed: {error}")
                        return
                    if 'engine' in st.session_state:
                        del st.session_state.engine
                    if 'engine_cache' in st.session_state:
                        del st.session_state.engine_cache
                    st.session_state['data'] = df
                    st.toast("Data loaded successfully!", icon="✅")
            if 'data' in st.session_state:
                df = st.session_state['data']
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # ── Landing page if no data ─────────────────────────────────────────
    if df is None:
        st.markdown("""
        <div class="premium-header">
            <h1>AARAMBH : Fair Value Breadth</h1>
            <div class="tagline">Walk-Forward Valuation · OU Mean-Reversion · Kalman Conviction | Quantitative Reversal Analysis</div>
        </div>
        """, unsafe_allow_html=True)
        render_landing_page()
        render_footer()
        return
    
    # ── Model Configuration (staging → commit) ──────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = df.columns.tolist()
    
    if len(numeric_cols) < 2:
        st.error("Need 2+ numeric columns.")
        return
    
    with st.sidebar:
        st.markdown('<div class="sidebar-title">🧠 Model Configuration</div>', unsafe_allow_html=True)
        
        default_target = "NIFTY50_PE" if "NIFTY50_PE" in numeric_cols else numeric_cols[0]
        default_preds = ["AD_RATIO", "COUNT", "REL_AD_RATIO", "REL_BREADTH", "IN10Y", "IN02Y",
                         "IN30Y", "INIRYY", "REPO", "US02Y", "US10Y", "US30Y", "NIFTY50_DY", "NIFTY50_PB"]
        
        active_target_state = st.session_state.get('active_target', default_target)
        if active_target_state not in numeric_cols:
             active_target_state = numeric_cols[0]
             
        target_col = st.selectbox("Target Variable", numeric_cols,
                                  index=numeric_cols.index(active_target_state))
        
        date_candidates = [c for c in all_cols if 'date' in c.lower()]
        default_date = date_candidates[0] if date_candidates else "None"
        active_date_state = st.session_state.get('active_date_col', default_date)
        if active_date_state not in (["None"] + all_cols):
             active_date_state = "None"
             
        date_col = st.selectbox("Date Column", ["None"] + all_cols,
                                index=(["None"] + all_cols).index(active_date_state))

        available = [c for c in numeric_cols if c != target_col]
        valid_defaults = [p for p in default_preds if p in available]
        
        # Initialize active predictors on first run
        if 'active_features' not in st.session_state:
            st.session_state['active_features'] = tuple(valid_defaults or available[:3])
        
        with st.expander("Predictor Columns", expanded=False):
            st.caption("Select predictors, then click Apply to recompute.")
            
            # Staging multiselect — user plays freely, no compute
            staging_features = st.multiselect(
                "Predictor Columns",
                options=available,
                default=[f for f in st.session_state['active_features'] if f in available],
                label_visibility="collapsed",
                help="These columns are used as predictors for walk-forward fair value regression."
            )
            
            if not staging_features:
                st.warning("⚠️ Select at least one predictor.")
                staging_features = [f for f in st.session_state['active_features'] if f in available]
            
            # Show diff between staging and active
            staging_set = set(staging_features)
            active_set = set(st.session_state['active_features'])
            has_pred_changes = staging_set != active_set
            
            # Also track target/date changes
            has_other_changes = (target_col != active_target_state) or (date_col != active_date_state)
            has_changes = has_pred_changes or has_other_changes
            
            if has_pred_changes:
                added = staging_set - active_set
                removed = active_set - staging_set
                changes = []
                if added:
                    changes.append(f"+{len(added)} added")
                if removed:
                    changes.append(f"−{len(removed)} removed")
                st.caption(f"Pending: {', '.join(changes)}")
            elif has_other_changes:
                st.caption("Pending: Target/Date changes")
            
            # Apply button — only this triggers recomputation
            apply_clicked = st.button(
                "✅ Apply Configuration" if has_changes else "No changes",
                use_container_width=True,
                disabled=not has_changes,
                type="primary" if has_changes else "secondary"
            )
            
            if apply_clicked and has_changes:
                st.session_state['active_target'] = target_col
                st.session_state['active_features'] = tuple(staging_features)
                st.session_state['active_date_col'] = date_col
                
                if 'engine' in st.session_state:
                    del st.session_state.engine
                if 'engine_cache' in st.session_state:
                    del st.session_state.engine_cache
                st.rerun()
            
            active_count = len(st.session_state['active_features'])
            total_count = len(available)
            if active_count != total_count:
                st.info(f"Active: {active_count}/{total_count} predictors")
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class='info-box'>
            <p style='font-size: 0.8rem; margin: 0; color: var(--text-muted); line-height: 1.5;'>
                <strong>Version:</strong> {VERSION}<br>
                <strong>Engine:</strong> Walk-Forward · OU · Kalman<br>
                <strong>Lookbacks:</strong> 5D, 10D, 20D, 50D, 100D
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # ── Resolve active config ───────────────────────────────────────────
    active_target = st.session_state.get('active_target', target_col)
    active_features = st.session_state.get('active_features', staging_features)
    active_date = st.session_state.get('active_date_col', date_col)
    
    # Explicitly cast to list to prevent list/tuple concatenation errors in clean_data
    feature_cols = list(active_features)
    
    # ── Header ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class="premium-header">
        <h1>AARAMBH : Fair Value Breadth</h1>
        <div class="tagline">Walk-Forward Valuation · OU Mean-Reversion · Kalman Conviction | Quantitative Reversal Analysis</div>
    </div>
    """, unsafe_allow_html=True)
    
    # ── Data staleness warning ──────────────────────────────────────────
    if active_date != "None" and active_date in df.columns:
        try:
            dates = pd.to_datetime(df[active_date], errors='coerce').dropna()
            if len(dates) > 0:
                latest_date = dates.max()
                from datetime import timezone as tz
                today = datetime.now(tz.utc) + timedelta(hours=5, minutes=30)
                data_age = (today - latest_date.to_pydatetime().replace(tzinfo=tz.utc)).days
                if data_age > 3:
                    st.markdown(f"""
                    <div style="background: rgba(239,68,68,0.1); border: 1px solid #ef4444; border-radius: 10px;
                                padding: 0.75rem 1.25rem; margin-bottom: 1rem; display: flex; align-items: center; gap: 12px;">
                        <span style="font-size: 1.4rem;">⚠️</span>
                        <div>
                            <span style="color: #ef4444; font-weight: 700;">Stale Data</span>
                            <span style="color: #888; font-size: 0.85rem;"> — Last data point is <b>{latest_date.strftime('%d %b %Y')}</b> ({data_age} days ago). Update your data source.</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        except Exception:
            pass
    
    # ── Clean & Fit ─────────────────────────────────────────────────────
    data = clean_data(df, active_target, feature_cols, active_date if active_date != "None" else None)
    
    if len(data) < 80:
        st.error("Need 80+ data points for walk-forward analysis (60 minimum training + 20 OOS).")
        return
    
    X = data[feature_cols].values
    y = data[active_target].values
    
    cache_key = f"{active_target}|{'|'.join(sorted(feature_cols))}|{len(data)}"
    
    if 'engine_cache' not in st.session_state or st.session_state.engine_cache != cache_key:
        with st.spinner("Preparing walk-forward engine..."):
            
            # Interactive Progress Tracker to eliminate the frozen UI feeling
            progress_bar = st.progress(0, text="Initializing engine...")
            
            def update_progress(frac, text):
                progress_bar.progress(frac, text=text)
            
            engine = FairValueEngine()
            engine.fit(X, y, feature_names=feature_cols, progress_callback=update_progress)
            
            st.session_state.engine = engine
            st.session_state.engine_cache = cache_key
            
            progress_bar.empty()
    
    engine = st.session_state.engine
    signal = engine.get_current_signal()
    model_stats = engine.get_model_stats()
    regime_stats = engine.get_regime_stats()
    ts = engine.ts_data.copy()
    
    if active_date != "None" and active_date in data.columns:
        ts['Date'] = pd.to_datetime(data[active_date].values)
    else:
        ts['Date'] = np.arange(len(ts))
    
    # ═══════════════════════════════════════════════════════════════════════
    # METRIC CARDS
    # ═══════════════════════════════════════════════════════════════════════
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 2])
    
    with c1:
        os_color = "success" if signal['oversold_breadth'] > 60 else "neutral"
        st.markdown(f'<div class="metric-card {os_color}"><h4>Oversold</h4><h2>{signal["oversold_breadth"]:.0f}%</h2><div class="sub-metric">Lookbacks in Zone</div></div>', unsafe_allow_html=True)
    
    with c2:
        ob_color = "danger" if signal['overbought_breadth'] > 60 else "neutral"
        st.markdown(f'<div class="metric-card {ob_color}"><h4>Overbought</h4><h2>{signal["overbought_breadth"]:.0f}%</h2><div class="sub-metric">Lookbacks in Zone</div></div>', unsafe_allow_html=True)
    
    with c3:
        conv_color = "success" if signal['conviction_score'] < -40 else "danger" if signal['conviction_score'] > 40 else "neutral"
        st.markdown(f'<div class="metric-card {conv_color}"><h4>Conviction</h4><h2>{signal["conviction_score"]:+.0f}</h2><div class="sub-metric">Kalman-Filtered</div></div>', unsafe_allow_html=True)
    
    with c4:
        sig_color = "success" if signal['signal'] == 'BUY' else "danger" if signal['signal'] == 'SELL' else "primary"
        st.markdown(f'<div class="metric-card {sig_color}"><h4>Signal</h4><h2>{signal["signal"]}</h2><div class="sub-metric">{signal["strength"]}</div></div>', unsafe_allow_html=True)
    
    with c5:
        reg_color = "success" if 'OVERSOLD' in signal['regime'] else "danger" if 'OVERBOUGHT' in signal['regime'] else "neutral"
        st.markdown(f'<div class="metric-card {reg_color}"><h4>Regime</h4><h2>{signal["regime"]}</h2><div class="sub-metric">Current State</div></div>', unsafe_allow_html=True)
    
    # ── Diagnostics Row ─────────────────────────────────────────────────
    d1, d2, d3, d4 = st.columns(4)
    
    with d1:
        hl = signal['ou_half_life']
        st.markdown(f'<div class="metric-card primary"><h4>OU Half-Life</h4><h2>{hl:.0f}d</h2><div class="sub-metric">Gap reversion time</div></div>', unsafe_allow_html=True)
    
    with d2:
        h = signal['hurst']
        h_label = 'Trending' if h > 0.55 else 'Random' if h > 0.45 else 'Mean-Reverting'
        h_class = 'danger' if h > 0.55 else 'neutral' if h > 0.45 else 'success'
        st.markdown(f'<div class="metric-card {h_class}"><h4>Residual Hurst</h4><h2>{h:.2f}</h2><div class="sub-metric">{h_label}</div></div>', unsafe_allow_html=True)
    
    with d3:
        r2 = model_stats['r2_oos']
        r2_class = 'success' if r2 > 0.7 else 'warning' if r2 > 0.4 else 'danger'
        st.markdown(f'<div class="metric-card {r2_class}"><h4>OOS R²</h4><h2>{r2:.3f}</h2><div class="sub-metric">Walk-Forward</div></div>', unsafe_allow_html=True)
    
    with d4:
        spread = model_stats['avg_model_spread']
        sp_class = 'success' if spread < 0.5 else 'warning' if spread < 1.5 else 'danger'
        st.markdown(f'<div class="metric-card {sp_class}"><h4>Model Spread</h4><h2>{spread:.2f}</h2><div class="sub-metric">Ensemble Disagreement</div></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════════════
    # TIMEFRAME FILTER
    # ═══════════════════════════════════════════════════════════════════════
    
    tf_col1, tf_col2 = st.columns([1, 6])
    with tf_col1:
        st.markdown("##### ⏱️ View Period")
    with tf_col2:
        time_filters = ["1M", "6M", "1Y", "2Y", "ALL"]
        selected_tf = st.radio("Timeframe", time_filters, index=2, horizontal=True, label_visibility="collapsed")
    
    ts_filtered = ts.copy()
    if selected_tf != "ALL":
        if active_date != "None" and pd.api.types.is_datetime64_any_dtype(ts['Date']):
            max_date = ts['Date'].max()
            offsets = {"1M": pd.DateOffset(months=1), "6M": pd.DateOffset(months=6),
                       "1Y": pd.DateOffset(years=1), "2Y": pd.DateOffset(years=2)}
            cutoff = max_date - offsets.get(selected_tf, pd.DateOffset(years=1))
            ts_filtered = ts[ts['Date'] >= cutoff]
        else:
            n_map = {"1M": 21, "6M": 126, "1Y": 252, "2Y": 504}
            ts_filtered = ts.iloc[max(0, len(ts) - n_map.get(selected_tf, 252)):]
    
    x_axis = ts_filtered['Date']
    x_title = "Date" if active_date != "None" else "Index"
    
    # ═══════════════════════════════════════════════════════════════════════
    # TABS
    # ═══════════════════════════════════════════════════════════════════════
    
    tab_regime, tab_signal, tab_zones, tab_signals, tab_data = st.tabs([
        "**🎯 Regime Analysis**",
        "**📊 Signal Dashboard**",
        "**📈 Zone Trends**",
        "**📉 Signal Trends**",
        "**📋 Data Table**"
    ])
    
    # ═══════════════════════════════════════════════════════════════════════
    # TAB: REGIME ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════
    with tab_regime:
        st.markdown("##### Kalman Conviction Score")
        st.markdown('<p style="color: #888;">Negative = Oversold bias | Positive = Overbought bias · Shaded = 95% Kalman confidence band</p>', unsafe_allow_html=True)
        
        fig_conv = go.Figure()
        
        # Confidence band
        if 'ConvictionUpper' in ts_filtered.columns:
            fig_conv.add_trace(go.Scatter(
                x=x_axis, y=ts_filtered['ConvictionUpper'],
                mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip',
            ))
            fig_conv.add_trace(go.Scatter(
                x=x_axis, y=ts_filtered['ConvictionLower'],
                mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip',
                fill='tonexty', fillcolor='rgba(255,195,0,0.08)', name='95% Band',
            ))
        
        # Positive/negative fills
        fig_conv.add_trace(go.Scatter(
            x=x_axis, y=ts_filtered['ConvictionScore'].clip(lower=0),
            fill='tozeroy', fillcolor='rgba(239,68,68,0.15)', line=dict(width=0), showlegend=False
        ))
        fig_conv.add_trace(go.Scatter(
            x=x_axis, y=ts_filtered['ConvictionScore'].clip(upper=0),
            fill='tozeroy', fillcolor='rgba(16,185,129,0.15)', line=dict(width=0), showlegend=False
        ))
        
        fig_conv.add_trace(go.Scatter(
            x=x_axis, y=ts_filtered['ConvictionScore'], mode='lines', name='Conviction (Kalman)',
            line=dict(color='#FFC300', width=2),
        ))
        
        # Raw conviction for comparison
        if 'ConvictionRaw' in ts_filtered.columns:
            fig_conv.add_trace(go.Scatter(
                x=x_axis, y=ts_filtered['ConvictionRaw'], mode='lines', name='Raw Conviction',
                line=dict(color='#555', width=1, dash='dot'), opacity=0.5,
            ))
        
        fig_conv.add_hline(y=40, line_dash="dash", line_color="rgba(239,68,68,0.5)")
        fig_conv.add_hline(y=-40, line_dash="dash", line_color="rgba(16,185,129,0.5)")
        fig_conv.add_hline(y=0, line_color="rgba(255,255,255,0.3)")
        
        fig_conv.update_layout(title="Conviction Score (Kalman-Filtered)", height=400,
                               xaxis_title=x_title, yaxis_title="Score", yaxis=dict(range=[-100, 100]))
        update_chart_theme(fig_conv)
        st.plotly_chart(fig_conv, use_container_width=True)
        
        st.markdown("---")
        st.markdown("##### Base Conviction Score")
        st.markdown('<p style="color: #888;">Negative = Oversold bias | Positive = Overbought bias</p>', unsafe_allow_html=True)
        
        if 'ConvictionRaw' in ts_filtered.columns:
            fig_raw = go.Figure()
            
            fig_raw.add_trace(go.Scatter(
                x=x_axis, y=ts_filtered['ConvictionRaw'].clip(lower=0),
                fill='tozeroy', fillcolor='rgba(239,68,68,0.15)',
                line=dict(width=0), showlegend=False
            ))
            
            fig_raw.add_trace(go.Scatter(
                x=x_axis, y=ts_filtered['ConvictionRaw'].clip(upper=0),
                fill='tozeroy', fillcolor='rgba(16,185,129,0.15)',
                line=dict(width=0), showlegend=False
            ))
            
            conv_colors = ['#10b981' if c < -40 else '#ef4444' if c > 40 else '#888' for c in ts_filtered['ConvictionRaw']]
            fig_raw.add_trace(go.Scatter(
                x=x_axis, y=ts_filtered['ConvictionRaw'], mode='lines+markers', name='Raw Conviction',
                line=dict(color='#FFC300', width=2), marker=dict(size=4, color=conv_colors)
            ))
            
            fig_raw.add_hline(y=40, line_dash="dash", line_color="rgba(239,68,68,0.5)")
            fig_raw.add_hline(y=-40, line_dash="dash", line_color="rgba(16,185,129,0.5)")
            fig_raw.add_hline(y=0, line_color="rgba(255,255,255,0.3)")
            
            fig_raw.update_layout(title="Base Conviction Score", height=400, xaxis_title=x_title, yaxis_title="Score",
                                   yaxis=dict(range=[-100, 100]))
            update_chart_theme(fig_raw)
            st.plotly_chart(fig_raw, use_container_width=True)

        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Regime Distribution")
            regime_data = {
                "Regime": ["🟢 Strongly Oversold", "🔵 Oversold", "⚪ Neutral", "🟠 Overbought", "🔴 Strongly Overbought"],
                "Count": [regime_stats['strongly_oversold'], regime_stats['oversold'], regime_stats['neutral'],
                          regime_stats['overbought'], regime_stats['strongly_overbought']],
                "Pct": [f"{regime_stats['strongly_oversold'] / len(ts) * 100:.1f}%",
                        f"{regime_stats['oversold'] / len(ts) * 100:.1f}%",
                        f"{regime_stats['neutral'] / len(ts) * 100:.1f}%",
                        f"{regime_stats['overbought'] / len(ts) * 100:.1f}%",
                        f"{regime_stats['strongly_overbought'] / len(ts) * 100:.1f}%"]
            }
            st.dataframe(pd.DataFrame(regime_data), width='stretch', hide_index=True)
        
        with col2:
            st.markdown("##### Current Regime & Diagnostics")
            curr_regime = signal['regime']
            regime_box_class = "success" if "OVERSOLD" in curr_regime else "danger" if "OVERBOUGHT" in curr_regime else ""
            
            st.markdown(f"""
            <div class="guide-box {regime_box_class}">
                <strong>Current: {curr_regime}</strong><br><br>
                {'Multiple timeframes showing oversold conditions — historically a buying opportunity.' if 'OVERSOLD' in curr_regime else 
                 'Multiple timeframes showing overbought conditions — historically a selling opportunity.' if 'OVERBOUGHT' in curr_regime else
                 'No strong directional bias across timeframes.'}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("##### Model Diagnostics")
            h_label = 'Mean-Reverting ✅' if signal['hurst'] < 0.45 else 'Trending ⚠️' if signal['hurst'] > 0.55 else 'Random Walk'
            st.markdown(f"""
            OOS R²: **{model_stats['r2_oos']:.4f}** | RMSE: **{model_stats['rmse_oos']:.4f}** | 
            OU Half-Life: **{signal['ou_half_life']:.0f} days** | 
            Hurst: **{signal['hurst']:.2f}** ({h_label}) |
            Model Spread: **{model_stats['avg_model_spread']:.3f}**
            """)
    
    # ═══════════════════════════════════════════════════════════════════════
    # TAB: SIGNAL DASHBOARD
    # ═══════════════════════════════════════════════════════════════════════
    with tab_signal:
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            st.markdown("##### Current Signal Analysis")
            signal_class = 'undervalued' if signal['signal'] == 'BUY' else 'overvalued' if signal['signal'] == 'SELL' else 'fair'
            signal_emoji = "🟢" if signal['signal'] == 'BUY' else "🔴" if signal['signal'] == 'SELL' else "🟡"
            
            st.markdown(f"""
            <div class="signal-card {signal_class}">
                <div class="label">WALK-FORWARD SIGNAL</div>
                <div class="value">{signal_emoji} {signal['signal']}</div>
                <div class="subtext">{signal['strength']} Strength • {signal['confidence']} Confidence • 
                OU t½ = {signal['ou_half_life']:.0f}d</div>
            </div>
            """, unsafe_allow_html=True)
            
            conv_pct = (signal['conviction_score'] + 100) / 2
            conv_color = '#10b981' if signal['conviction_score'] < -20 else '#ef4444' if signal['conviction_score'] > 20 else '#FFC300'
            
            st.markdown(f"""
            <div class="conviction-meter">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="color: #10b981; font-size: 0.75rem;">OVERSOLD</span>
                    <span style="color: #888; font-size: 0.75rem;">Conviction: {signal['conviction_score']:+.0f} [{signal['conviction_lower']:+.0f}, {signal['conviction_upper']:+.0f}]</span>
                    <span style="color: #ef4444; font-size: 0.75rem;">OVERBOUGHT</span>
                </div>
                <div class="conviction-bar">
                    <div class="conviction-fill" style="width: {conv_pct}%; background: {conv_color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if signal['has_bullish_div']:
                st.markdown('<span class="status-badge buy">🔔 BULLISH DIVERGENCE (Swing-Based)</span>', unsafe_allow_html=True)
            if signal['has_bearish_div']:
                st.markdown('<span class="status-badge sell">🔔 BEARISH DIVERGENCE (Swing-Based)</span>', unsafe_allow_html=True)
            
            # Model uncertainty warning
            if signal['model_spread'] > 1.0:
                st.markdown(f"""
                <div style="background: rgba(245,158,11,0.1); border: 1px solid #f59e0b; border-radius: 8px; padding: 0.5rem 1rem; margin-top: 0.5rem;">
                    <span style="color: #f59e0b; font-size: 0.8rem;">⚠️ High model disagreement ({signal['model_spread']:.2f}) — fair value estimate is uncertain. Signal confidence may be lower than indicated.</span>
                </div>
                """, unsafe_allow_html=True)
        
        with col_right:
            st.markdown("##### Lookback Breakdown")
            for lb in engine.LOOKBACKS:
                if lb not in engine.lookback_data:
                    continue
                z = engine.lookback_data[lb]['z_scores'][-1]
                zone = engine.lookback_data[lb]['zones'][-1]
                zone_color = '#10b981' if 'Under' in zone else '#ef4444' if 'Over' in zone else '#888'
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; padding: 0.5rem; border-bottom: 1px solid #2A2A2A;">
                    <span style="color: #888;">{lb}-Day</span>
                    <span style="color: {zone_color}; font-weight: 600;">{zone} ({z:+.2f}σ)</span>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Price vs Fair Value + OU projection
        st.markdown("##### Actual vs Walk-Forward Fair Value")
        
        fig = make_subplots(rows=2, cols=1, row_heights=[0.6, 0.4], shared_xaxes=True, vertical_spacing=0.05)
        
        fig.add_trace(go.Scatter(x=x_axis, y=ts_filtered['Actual'], mode='lines', name='Actual',
                                 line=dict(color='#FFC300', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_axis, y=ts_filtered['FairValue'], mode='lines', name='Fair Value (OOS)',
                                 line=dict(color='#06b6d4', width=2, dash='dash')), row=1, col=1)
        
        # Model spread as uncertainty band
        if 'ModelSpread' in ts_filtered.columns:
            upper = ts_filtered['FairValue'] + ts_filtered['ModelSpread']
            lower = ts_filtered['FairValue'] - ts_filtered['ModelSpread']
            fig.add_trace(go.Scatter(x=x_axis, y=upper, mode='lines', line=dict(width=0),
                                     showlegend=False, hoverinfo='skip'), row=1, col=1)
            fig.add_trace(go.Scatter(x=x_axis, y=lower, mode='lines', line=dict(width=0),
                                     fill='tonexty', fillcolor='rgba(6,182,212,0.08)',
                                     name='Model Uncertainty', hoverinfo='skip'), row=1, col=1)
        
        # Residual bar
        colors = ['#10b981' if r < 0 else '#ef4444' for r in ts_filtered['Residual']]
        fig.add_trace(go.Bar(x=x_axis, y=ts_filtered['Residual'], name='Residual (OOS)',
                             marker_color=colors, showlegend=False), row=2, col=1)
        fig.add_hline(y=0, line_color="#FFC300", line_width=1, row=2, col=1)
        
        # OU projection on residual pane
        if hasattr(engine, 'ou_projection') and pd.api.types.is_datetime64_any_dtype(ts['Date']):
            last_date = ts['Date'].iloc[-1]
            proj_dates = pd.date_range(start=last_date, periods=91, freq='D')[1:]
            fig.add_trace(go.Scatter(
                x=proj_dates, y=engine.ou_projection,
                mode='lines', name='OU Projection',
                line=dict(color='#FFC300', width=1.5, dash='dot'), opacity=0.5,
            ), row=2, col=1)
        
        fig.update_layout(height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02))
        fig.update_yaxes(title_text=active_target, row=1, col=1)
        fig.update_yaxes(title_text="Residual", row=2, col=1)
        update_chart_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    # ═══════════════════════════════════════════════════════════════════════
    # TAB: ZONE TRENDS
    # ═══════════════════════════════════════════════════════════════════════
    with tab_zones:
        st.markdown("##### Overbought / Oversold Breadth Over Time")
        st.markdown('<p style="color: #888;">% of lookback periods in oversold/overbought zones</p>', unsafe_allow_html=True)
        
        fig_zones = go.Figure()
        fig_zones.add_trace(go.Scatter(
            x=x_axis, y=ts_filtered['OversoldBreadth'],
            fill='tozeroy', fillcolor='rgba(16,185,129,0.2)',
            line=dict(color='#10b981', width=2), name='Oversold %'
        ))
        fig_zones.add_trace(go.Scatter(
            x=x_axis, y=ts_filtered['OverboughtBreadth'],
            fill='tozeroy', fillcolor='rgba(239,68,68,0.2)',
            line=dict(color='#ef4444', width=2), name='Overbought %'
        ))
        fig_zones.add_hline(y=60, line_dash="dash", line_color="rgba(255,195,0,0.3)")
        fig_zones.update_layout(title="Zone Breadth", height=400, xaxis_title=x_title, yaxis_title="% of Lookbacks",
                                yaxis=dict(range=[0, 100]))
        update_chart_theme(fig_zones)
        st.plotly_chart(fig_zones, use_container_width=True)
        
        st.markdown("---")
        st.markdown("##### Average Z-Score Across Lookbacks")
        
        fig_z = go.Figure()
        z_colors = ['#10b981' if z < -1 else '#ef4444' if z > 1 else '#888' for z in ts_filtered['AvgZ']]
        fig_z.add_trace(go.Bar(x=x_axis, y=ts_filtered['AvgZ'], marker_color=z_colors, name='Avg Z'))
        fig_z.add_hline(y=0, line_color="#FFC300", line_width=1)
        fig_z.add_hline(y=2, line_dash="dash", line_color="rgba(239,68,68,0.5)")
        fig_z.add_hline(y=-2, line_dash="dash", line_color="rgba(16,185,129,0.5)")
        fig_z.update_layout(title="Multi-Lookback Average Z-Score", height=350, xaxis_title=x_title, yaxis_title="Z-Score")
        update_chart_theme(fig_z)
        st.plotly_chart(fig_z, use_container_width=True)
    
    # ═══════════════════════════════════════════════════════════════════════
    # TAB: SIGNAL TRENDS
    # ═══════════════════════════════════════════════════════════════════════
    with tab_signals:
        st.markdown("##### Buy/Sell Signal Count by Period")
        
        fig_signals = go.Figure()
        fig_signals.add_trace(go.Bar(
            x=x_axis, y=ts_filtered['BuySignalBreadth'], name='Buy Signals',
            marker=dict(color='#10b981')
        ))
        fig_signals.add_trace(go.Bar(
            x=x_axis, y=-ts_filtered['SellSignalBreadth'], name='Sell Signals',
            marker=dict(color='#ef4444')
        ))
        fig_signals.update_layout(title="Signal Count by Period", height=350, xaxis_title=x_title,
                                  yaxis_title="Signal Count", barmode='relative')
        update_chart_theme(fig_signals)
        st.plotly_chart(fig_signals, use_container_width=True)
        
        st.markdown("---")
        st.markdown("##### Signal Statistics")
        
        perf = engine.get_signal_performance()
        perf_data = []
        for period in [5, 10, 20]:
            p = perf[period]
            perf_data.append({
                'Holding Period': f'{period} Days',
                'Buy Hit Rate': f"{p['buy_hit'] * 100:.1f}%" if p['buy_count'] > 0 else 'N/A',
                'Buy Avg Return': f"{p['buy_avg']:.2f}%" if p['buy_count'] > 0 else 'N/A',
                'Buy Count': p['buy_count'],
                'Sell Hit Rate': f"{p['sell_hit'] * 100:.1f}%" if p['sell_count'] > 0 else 'N/A',
                'Sell Avg Return': f"{p['sell_avg']:.2f}%" if p['sell_count'] > 0 else 'N/A',
                'Sell Count': p['sell_count'],
            })
        st.dataframe(pd.DataFrame(perf_data), width='stretch', hide_index=True)
    
    # ═══════════════════════════════════════════════════════════════════════
    # TAB: DATA TABLE
    # ═══════════════════════════════════════════════════════════════════════
    with tab_data:
        st.markdown(f"##### Time Series Data ({len(ts_filtered)} observations)")
        
        display_cols = ['Date', 'Actual', 'FairValue', 'Residual', 'ModelSpread', 'AvgZ',
                        'OversoldBreadth', 'OverboughtBreadth', 'ConvictionScore', 'Regime',
                        'BullishDiv', 'BearishDiv']
        display_cols = [c for c in display_cols if c in ts_filtered.columns]
        
        display_ts = ts_filtered[display_cols].copy()
        for col in ['Residual', 'ModelSpread', 'AvgZ', 'FairValue', 'ConvictionScore',
                     'OversoldBreadth', 'OverboughtBreadth']:
            if col in display_ts.columns:
                display_ts[col] = display_ts[col].round(3 if col in ['AvgZ', 'ModelSpread'] else 2 if col == 'FairValue' else 1)
        
        if 'BullishDiv' in display_ts.columns:
            display_ts['BullishDiv'] = display_ts['BullishDiv'].apply(lambda x: '🟢' if x else '')
        if 'BearishDiv' in display_ts.columns:
            display_ts['BearishDiv'] = display_ts['BearishDiv'].apply(lambda x: '🔴' if x else '')
        
        st.dataframe(display_ts, width='stretch', hide_index=True, height=500)
        
        csv_data = ts.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Full CSV", csv_data,
                           f"aarambh_{active_target}_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
    
    render_footer()


if __name__ == "__main__":
    main()
