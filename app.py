# -*- coding: utf-8 -*-
"""
AARAMBH (आरंभ) GEOMETRIC - Latent Equilibrium Manifold
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Manifold Learning · Geometric Distance · Complex System Equilibrium
Measures distance from the equilibrium structure of a multivariate phase space.

Architecture:
  1. Manifold Learning: Kernel PCA maps X_t to low-dim ψ_t
  2. Equilibrium Surface: y_fair = g(ψ_t)
  3. Geometric Distance: D_t = || X_t - Π_M(X_t) ||
  4. Mean Reversion: OU process estimating decay speed (κ)
  5. Metric Attribution: ∇f(X) via numerical Jacobian
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
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import KernelPCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ── Constants ───────────────────────────────────────────────────────────────
VERSION = "v4.0.0-Manifold-Optimized"
PRODUCT_NAME = "Aarambh Geometric"
COMPANY = "Hemrek Capital"

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AARAMBH | Geometric Equilibrium",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS (Hemrek Design System — STRICTLY UNCHANGED)
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
# MATHEMATICAL PRIMITIVES
# ══════════════════════════════════════════════════════════════════════════════

def kalman_filter_1d(observations, process_var=None, measurement_var=None):
    """1D Kalman filter for conviction smoothing."""
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

def ornstein_uhlenbeck_estimate(series, dt=1.0):
    """
    Estimate OU parameters: dx = κ(μ−x)dt + σdW.
    Returns: (kappa, mu, sigma)
    Used to estimate mean-reversion speed (κ).
    """
    x = np.asarray(series, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 20: return 0.05, 0.0, max(np.std(x), 1e-6) if len(x) > 1 else (0.05, 0.0, 1.0)
    
    x_lag, x_curr = x[:-1], x[1:]
    n = len(x_lag)
    sx, sy = np.sum(x_lag), np.sum(x_curr)
    sxx, sxy = np.sum(x_lag ** 2), np.sum(x_lag * x_curr)
    
    denom = n * sxx - sx ** 2
    if abs(denom) < 1e-12: return 0.05, np.mean(x), max(np.std(x), 1e-6)
    
    a = (n * sxy - sx * sy) / denom
    b = (sy * sxx - sx * sxy) / denom
    a = np.clip(a, 1e-6, 1.0 - 1e-6)
    
    kappa = -np.log(a) / dt
    mu = b / (1 - a)
    residuals = x_curr - a * x_lag - b
    sigma = np.sqrt(max(np.var(residuals) * 2 * kappa / (1 - a ** 2), 1e-12))
    
    return max(kappa, 1e-4), mu, max(sigma, 1e-6)

# ══════════════════════════════════════════════════════════════════════════════
# MANIFOLD PRICING ENGINE (GEOMETRIC - HIGHLY OPTIMIZED)
# ══════════════════════════════════════════════════════════════════════════════

class ManifoldPricingEngine:
    """
    Geometric Inference Engine.
    
    Optimizations built-in:
    - Bounded Rolling Window (O(N^3) -> O(1) constraints for KernelPCA)
    - Vectorized Gradient Attribution (removes iterative predict loops)
    - Smart Refit Cadence (21 days)
    - Vectorized Lookbacks (Pandas Rolling)
    """
    
    LOOKBACKS = [5, 10, 20, 50, 100]
    MIN_TRAIN_SIZE = 40
    # Capping training size to ~2 years strictly bounds O(N^3) kernel explosion
    MAX_TRAIN_SIZE = 504 
    # Refitting every month (21 days) perfectly captures topological regimes 
    # while saving thousands of computationally heavy operations
    REFIT_STEP = 21 
    
    def __init__(self):
        self.scaler = None
    
    def fit(self, X, y, feature_names=None, progress_callback=None):
        start_time = time.time()
        self.feature_names = feature_names or [f'X{i}' for i in range(X.shape[1])]
        self.n_samples = len(y)
        self.y = y.copy()
        self.X = X.copy()
        
        n = self.n_samples
        self.predictions = np.full(n, np.nan)
        self.D_t_series = np.zeros(n)
        self.top_drivers = np.full(n, "N/A", dtype=object)
        
        current_scaler = None
        kpca = None
        surface_model = None
        
        for t in range(n):
            if progress_callback and t % max(1, n // 10) == 0:
                progress_callback(t / n * 0.6, f"Learning equilibrium manifold: {t}/{n}...")
                
            if t < self.MIN_TRAIN_SIZE:
                self.predictions[t] = np.mean(y[:t + 1])
                self.D_t_series[t] = 0.0
            else:
                if t == self.MIN_TRAIN_SIZE or t % self.REFIT_STEP == 0:
                    # Bounded expanding window: solves the O(N^3) KPCA bottleneck
                    start_idx = max(0, t - self.MAX_TRAIN_SIZE)
                    X_train = X[start_idx:t]
                    y_train = y[start_idx:t]
                    
                    if SKLEARN_AVAILABLE:
                        scaler_t = StandardScaler()
                        X_train_s = scaler_t.fit_transform(X_train)
                        current_scaler = scaler_t
                        
                        try:
                            # 1. Manifold Learning: Φ : X -> R^k
                            k = min(4, X.shape[1])
                            kpca = KernelPCA(n_components=k, kernel='rbf', fit_inverse_transform=True, gamma=1.0/X.shape[1])
                            psi_train = kpca.fit_transform(X_train_s)
                            
                            # 2. Equilibrium Surface: y_fair = g(ψ)
                            surface_model = Ridge(alpha=1.0)
                            surface_model.fit(psi_train, y_train)
                        except Exception:
                            pass
                
                # Projection and Prediction for current state t
                X_pred = X[t:t + 1]
                if SKLEARN_AVAILABLE and current_scaler is not None and kpca is not None and surface_model is not None:
                    try:
                        X_pred_s = current_scaler.transform(X_pred)
                        
                        # Project onto manifold
                        psi_test = kpca.transform(X_pred_s)
                        
                        # Geometric Distance D_t = | X_t - Π_M(X_t) |
                        X_reconstructed = kpca.inverse_transform(psi_test)
                        self.D_t_series[t] = np.linalg.norm(X_pred_s - X_reconstructed)
                        
                        # Predict Fair Value
                        y_fair = surface_model.predict(psi_test)[0]
                        self.predictions[t] = y_fair
                        
                        # 5. Multi-Metric Contribution Attribution ∇f(X_t)
                        # VECTORIZED Jacobian bottleneck fix
                        epsilon = 0.01
                        num_features = X.shape[1]
                        
                        # Create a batch matrix instead of python looping
                        X_plus_batch = np.repeat(X_pred_s, num_features, axis=0)
                        np.fill_diagonal(X_plus_batch, X_plus_batch.diagonal() + epsilon)
                        
                        # Transform and predict the entire matrix at once
                        psi_plus_batch = kpca.transform(X_plus_batch)
                        y_plus_batch = surface_model.predict(psi_plus_batch)
                        
                        delta_y = (y_plus_batch - y_fair) / epsilon
                        top_idx = np.argmax(np.abs(delta_y))
                        self.top_drivers[t] = self.feature_names[top_idx][:12] 
                        
                    except Exception as e:
                        self.predictions[t] = self.predictions[t-1] if t > 0 else y[t]
                else:
                    self.predictions[t] = np.mean(y[:t + 1])
        
        # Deviation Δ_t = y_t - y_fair
        self.residuals = y - self.predictions
        
        if progress_callback:
            progress_callback(0.7, "Estimating OU Manifold Dynamics...")
            
        # Fast Vectorized Rolling Window for Z-Scores (eliminates nested for-loops)
        resid_series = pd.Series(self.residuals)
        self.lookback_data = {lb: {'Z_t': np.zeros(n)} for lb in self.LOOKBACKS}
        
        for lb in self.LOOKBACKS:
            # shifted by 1 prevents lookahead bias (exactly matches self.residuals[t-lb:t])
            rmean = resid_series.rolling(window=lb).mean().shift(1).fillna(0).values
            rstd = resid_series.rolling(window=lb).std().shift(1).fillna(1e-8).values + 1e-8
            
            z_t = (self.residuals - rmean) / rstd
            
            # Zero out periods before train bounds
            mask = np.arange(n) < max(self.MIN_TRAIN_SIZE, lb)
            z_t[mask] = 0.0
            
            self.lookback_data[lb]['Z_t'] = z_t
        
        # 4. OU Mean Reversion Speed (κ) estimation
        oos_resid = self.residuals[self.MIN_TRAIN_SIZE:]
        kappa, mu, sigma = ornstein_uhlenbeck_estimate(oos_resid)
        self.kappa = kappa
        
        if progress_callback:
            progress_callback(0.9, "Synthesizing Topological Score...")
            
        self._compute_model_stats()
        self._compute_composite_mispricing()
        self._compute_manifold_breadth()
        self._detect_regime_shifts()
        self._compute_forward_returns()
        
        elapsed = time.time() - start_time
        logging.info(f"Geometric engine [{n} obs, {len(self.feature_names)} features] in {elapsed:.1f}s")
        
        if progress_callback:
            progress_callback(1.0, "Done.")
            
        return self
    
    def _compute_model_stats(self):
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
                'avg_model_spread': np.mean(self.D_t_series[oos_mask]), # Mean D_t distance
            }
        else:
            self.model_stats = {'r2_oos': 0.0, 'rmse_oos': 0.0, 'avg_model_spread': 0.0}
    
    def _compute_composite_mispricing(self):
        """
        Calculates geometric conviction: Ω_t = αZ_t + βD_t * sign(Z_t)
        Distance from manifold amplifies the standard mispricing score.
        """
        D_z = (self.D_t_series - np.mean(self.D_t_series)) / (np.std(self.D_t_series) + 1e-8)
        Z_stat = (self.residuals - np.mean(self.residuals)) / (np.std(self.residuals) + 1e-8)
        
        alpha, beta = 20.0, 15.0
        
        Omega_raw = (alpha * Z_stat) + (beta * D_z * np.sign(Z_stat))
        
        filtered_omega, _, variances = kalman_filter_1d(Omega_raw)
        kalman_std = np.sqrt(np.maximum(variances, 0))
        
        self.ts_data = pd.DataFrame({
            'Actual': self.y,
            'FairValue': self.predictions,
            'Residual': self.residuals,
            'ModelSpread': self.D_t_series, # Map to Dist for UI
            'AvgZ': Z_stat,
            'TopDriver': self.top_drivers,
            'ConvictionRaw': Omega_raw,
            'ConvictionScore': np.clip(filtered_omega, -100, 100),
            'ConvictionUpper': np.clip(filtered_omega + 1.96 * kalman_std, -100, 100),
            'ConvictionLower': np.clip(filtered_omega - 1.96 * kalman_std, -100, 100)
        })
        
        regimes = []
        for score in self.ts_data['ConvictionScore']:
            if score < -40: regimes.append('STRONGLY OVERSOLD')
            elif score < -20: regimes.append('OVERSOLD')
            elif score > 40: regimes.append('STRONGLY OVERBOUGHT')
            elif score > 20: regimes.append('OVERBOUGHT')
            else: regimes.append('NEUTRAL')
        self.ts_data['Regime'] = regimes

    def _compute_manifold_breadth(self):
        """Multi-Scale Structure tracking combining Z_t and Manifold D_t"""
        n = len(self.y)
        oversold_count = np.zeros(n)
        overbought_count = np.zeros(n)
        
        D_threshold = np.percentile(self.D_t_series[self.MIN_TRAIN_SIZE:], 75)
        
        for i in range(n):
            for lb in self.LOOKBACKS:
                z = self.lookback_data[lb]['Z_t'][i]
                d = self.D_t_series[i]
                d_is_high = d > D_threshold
                
                if z < -1.5 and d_is_high: oversold_count[i] += 1
                elif z < -0.5: oversold_count[i] += 1
                    
                if z > 1.5 and d_is_high: overbought_count[i] += 1
                elif z > 0.5: overbought_count[i] += 1

        num_lb = len(self.LOOKBACKS)
        self.ts_data['OversoldBreadth'] = oversold_count / num_lb * 100
        self.ts_data['OverboughtBreadth'] = overbought_count / num_lb * 100
        
        self.ts_data['BuySignalBreadth'] = np.where(self.ts_data['OversoldBreadth'] > 60, 1, 0)
        self.ts_data['SellSignalBreadth'] = np.where(self.ts_data['OverboughtBreadth'] > 60, 1, 0)
        self.ts_data['IsPivotTop'] = False
        self.ts_data['IsPivotBottom'] = False

    def _detect_regime_shifts(self):
        """Regime shifts correspond to topological changes in manifold distances."""
        n = len(self.y)
        bull_div = np.zeros(n, dtype=bool)
        bear_div = np.zeros(n, dtype=bool)
        
        D_thresh = np.percentile(self.D_t_series[self.MIN_TRAIN_SIZE:], 90)
        
        for i in range(self.MIN_TRAIN_SIZE, n):
            if self.D_t_series[i] > D_thresh:
                if self.ts_data['AvgZ'].iloc[i] < -1.0: bull_div[i] = True
                elif self.ts_data['AvgZ'].iloc[i] > 1.0: bear_div[i] = True
                    
        self.ts_data['BullishDiv'] = bull_div
        self.ts_data['BearishDiv'] = bear_div
        
        # OU Decay Path Projection
        current_r = self.residuals[-1]
        self.ou_projection = current_r * np.exp(-self.kappa * np.arange(1, 91))
        
    def _compute_forward_returns(self):
        n = len(self.y)
        for period in [5, 10, 20]:
            fwd_ret = np.full(n, np.nan)
            for i in range(n - period):
                if self.y[i] > 0:
                    fwd_ret[i] = (self.y[i + period] - self.y[i]) / self.y[i] * 100
            self.ts_data[f'FwdRet_{period}'] = fwd_ret

    def get_current_signal(self):
        ts = self.ts_data
        current = ts.iloc[-1]
        
        conviction = current['ConvictionScore']
        regime = current['Regime']
        
        if conviction < -60: signal, strength = 'BUY', 'STRONG'
        elif conviction < -40: signal, strength = 'BUY', 'MODERATE'
        elif conviction < -20: signal, strength = 'BUY', 'WEAK'
        elif conviction > 60: signal, strength = 'SELL', 'STRONG'
        elif conviction > 40: signal, strength = 'SELL', 'MODERATE'
        elif conviction > 20: signal, strength = 'SELL', 'WEAK'
        else: signal, strength = 'HOLD', 'NEUTRAL'
        
        confidence = 'HIGH' if current['OversoldBreadth'] >= 80 or current['OverboughtBreadth'] >= 80 else 'MEDIUM' if current['OversoldBreadth'] >= 60 or current['OverboughtBreadth'] >= 60 else 'LOW'
        
        return {
            'signal': signal,
            'strength': strength,
            'confidence': confidence,
            'conviction_score': conviction,
            'conviction_upper': current['ConvictionUpper'],
            'conviction_lower': current['ConvictionLower'],
            'regime': regime,
            'oversold_breadth': current['OversoldBreadth'],
            'overbought_breadth': current['OverboughtBreadth'],
            'residual': current['Residual'],
            'fair_value': current['FairValue'],
            'actual': current['Actual'],
            'avg_z': current['AvgZ'],
            'model_spread': current['ModelSpread'], # Maps to D_t geometric distance
            'has_bullish_div': current['BullishDiv'],
            'has_bearish_div': current['BearishDiv'],
            'top_driver': current['TopDriver'],
            'kappa': self.kappa,
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
            'total_pivot_tops': 0,
            'total_pivot_bottoms': 0,
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
                    if not pd.isna(fwd): buy_returns.append(fwd)
                if ts['ConvictionScore'].iloc[i] > 40:
                    fwd = ts[f'FwdRet_{period}'].iloc[i]
                    if not pd.isna(fwd): sell_returns.append(-fwd)
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
        if not sheet_id_match: return None, "Invalid URL"
        sheet_id = sheet_id_match.group(1)
        gid_match = re.search(r'gid=(\d+)', sheet_url)
        gid = gid_match.group(1) if gid_match else '0'
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
        df = pd.read_csv(csv_url)
        return df, None
    except Exception as e:
        return None, str(e)


def clean_data(df, target, features, date_col=None):
    features_list = list(features)
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
        except Exception: pass
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
            <h3 style='color: var(--purple); margin-bottom: 0.5rem;'>🌌 Manifold Learning</h3>
            <p style='color: var(--text-muted); font-size: 0.9rem; line-height: 1.6;'>
                Instead of linear regression, the engine learns a low-dimensional manifold embedded in high-dimensional phase space.
            </p>
            <br>
            <p style='color: var(--text-secondary); font-size: 0.85rem;'>
                <strong>Method:</strong> Kernel PCA (RBF)<br>
                <strong>Surface:</strong> Equilibrium mapping g(ψ)<br>
                <strong>Attribution:</strong> ∇f(X) Jacobian
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card info' style='min-height: 280px; justify-content: flex-start;'>
            <h3 style='color: var(--info-cyan); margin-bottom: 0.5rem;'>📐 Geometric Distance</h3>
            <p style='color: var(--text-muted); font-size: 0.9rem; line-height: 1.6;'>
                Mispricing isn't just a statistical gap; it's a structural deformation. We measure distance from the latent equilibrium geometry.
            </p>
            <br>
            <p style='color: var(--text-secondary); font-size: 0.85rem;'>
                <strong>Projection:</strong> | Z_t - Π_M(Z_t) |<br>
                <strong>Metric:</strong> D_t (Manifold Dist)<br>
                <strong>Dynamics:</strong> OU Decay Speed (κ)
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card primary' style='min-height: 280px; justify-content: flex-start;'>
            <h3 style='color: var(--primary-color); margin-bottom: 0.5rem;'>📊 Composite Mispricing</h3>
            <p style='color: var(--text-muted); font-size: 0.9rem; line-height: 1.6;'>
                Integrates statistical Z-scores with structural distance D_t to form an ultimate geometric conviction score.
            </p>
            <br>
            <p style='color: var(--text-secondary); font-size: 0.85rem;'>
                <strong>Equation:</strong> Ω_t = αZ_t + βD_t<br>
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
        Select a <strong>Target</strong> and <strong>Predictors</strong>, then click <strong>Apply</strong> to run the geometric engine.</p>
    </div>
    """, unsafe_allow_html=True)

def render_footer():
    utc_now = datetime.now()
    current_time = utc_now.strftime("%Y-%m-%d %H:%M:%S")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.caption(f"© 2026 {PRODUCT_NAME} | {COMPANY} | {VERSION} | {current_time}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

def main():
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; margin-bottom: 1rem;">
            <div style="font-size: 1.75rem; font-weight: 800; color: #FFC300;">AARAMBH</div>
            <div style="color: #888888; font-size: 0.75rem; margin-top: 0.25rem;">आरंभ | Geometric Inference</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-title">📁 Data Source</div>', unsafe_allow_html=True)
        data_source = st.radio("Source", ["📤 Upload", "📊 Google Sheets"], horizontal=True, label_visibility="collapsed")
        
        df = None
        if data_source == "📤 Upload":
            uploaded_file = st.file_uploader("CSV/Excel", type=['csv', 'xlsx'], label_visibility="collapsed")
            if uploaded_file:
                try: df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                except Exception as e: st.error(f"Error: {e}"); return
        else:
            default_url = "https://docs.google.com/spreadsheets/d/1po7z42n3dYIQGAvn0D1-a4pmyxpnGPQ13TrNi3DB5_c/edit?gid=1938234952#gid=1938234952"
            sheet_url = st.text_input("Sheet URL", value=default_url, label_visibility="collapsed")
            if st.button("🔄 LOAD DATA", type="primary"):
                with st.spinner("Loading..."):
                    df, error = load_google_sheet(sheet_url)
                    if error: st.error(f"Failed: {error}"); return
                    if 'engine' in st.session_state: del st.session_state.engine
                    if 'engine_cache' in st.session_state: del st.session_state.engine_cache
                    st.session_state['data'] = df
                    st.toast("Data loaded successfully!", icon="✅")
            if 'data' in st.session_state: df = st.session_state['data']
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    if df is None:
        st.markdown("""
        <div class="premium-header">
            <h1>AARAMBH : Geometric Inference</h1>
            <div class="tagline">Latent Equilibrium Manifold · Geometric Distance · Complex System Equilibrium</div>
        </div>
        """, unsafe_allow_html=True)
        render_landing_page()
        render_footer()
        return
    
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
        if active_target_state not in numeric_cols: active_target_state = numeric_cols[0]
        target_col = st.selectbox("Target Variable", numeric_cols, index=numeric_cols.index(active_target_state))
        
        date_candidates = [c for c in all_cols if 'date' in c.lower()]
        default_date = date_candidates[0] if date_candidates else "None"
        active_date_state = st.session_state.get('active_date_col', default_date)
        if active_date_state not in (["None"] + all_cols): active_date_state = "None"
        date_col = st.selectbox("Date Column", ["None"] + all_cols, index=(["None"] + all_cols).index(active_date_state))

        available = [c for c in numeric_cols if c != target_col]
        valid_defaults = [p for p in default_preds if p in available]
        
        if 'active_features' not in st.session_state:
            st.session_state['active_features'] = tuple(valid_defaults or available[:3])
        
        with st.expander("Predictor Columns", expanded=False):
            st.caption("Select predictors, then click Apply to recompute.")
            staging_features = st.multiselect(
                "Predictor Columns", options=available,
                default=[f for f in st.session_state['active_features'] if f in available], label_visibility="collapsed"
            )
            
            if not staging_features:
                st.warning("⚠️ Select at least one predictor.")
                staging_features = [f for f in st.session_state['active_features'] if f in available]
            
            staging_set = set(staging_features)
            active_set = set(st.session_state['active_features'])
            has_pred_changes = staging_set != active_set
            has_other_changes = (target_col != active_target_state) or (date_col != active_date_state)
            has_changes = has_pred_changes or has_other_changes
            
            if has_pred_changes:
                added, removed = staging_set - active_set, active_set - staging_set
                changes = []
                if added: changes.append(f"+{len(added)} added")
                if removed: changes.append(f"−{len(removed)} removed")
                st.caption(f"Pending: {', '.join(changes)}")
            elif has_other_changes: st.caption("Pending: Target/Date changes")
            
            apply_clicked = st.button("✅ Apply Configuration" if has_changes else "No changes", use_container_width=True, disabled=not has_changes, type="primary" if has_changes else "secondary")
            
            if apply_clicked and has_changes:
                st.session_state['active_target'] = target_col
                st.session_state['active_features'] = tuple(staging_features)
                st.session_state['active_date_col'] = date_col
                if 'engine' in st.session_state: del st.session_state.engine
                if 'engine_cache' in st.session_state: del st.session_state.engine_cache
                st.rerun()
            
            active_count, total_count = len(st.session_state['active_features']), len(available)
            if active_count != total_count: st.info(f"Active: {active_count}/{total_count} predictors")
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class='info-box'>
            <p style='font-size: 0.8rem; margin: 0; color: var(--text-muted); line-height: 1.5;'>
                <strong>Version:</strong> {VERSION}<br>
                <strong>Engine:</strong> Latent Manifold · OU Reversion<br>
                <strong>Lookbacks:</strong> 5D, 10D, 20D, 50D, 100D
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    active_target = st.session_state.get('active_target', target_col)
    active_features = st.session_state.get('active_features', staging_features)
    active_date = st.session_state.get('active_date_col', date_col)
    feature_cols = list(active_features)
    
    st.markdown("""
    <div class="premium-header">
        <h1>AARAMBH : Geometric Inference</h1>
        <div class="tagline">Latent Equilibrium Manifold · Geometric Distance · Complex System Equilibrium</div>
    </div>
    """, unsafe_allow_html=True)
    
    if active_date != "None" and active_date in df.columns:
        try:
            dates = pd.to_datetime(df[active_date], errors='coerce').dropna()
            if len(dates) > 0:
                latest_date = dates.max()
                data_age = (datetime.now() - latest_date).days
                if data_age > 3:
                    st.markdown(f"""
                    <div style="background: rgba(239,68,68,0.1); border: 1px solid #ef4444; border-radius: 10px;
                                padding: 0.75rem 1.25rem; margin-bottom: 1rem; display: flex; align-items: center; gap: 12px;">
                        <span style="font-size: 1.4rem;">⚠️</span>
                        <div>
                            <span style="color: #ef4444; font-weight: 700;">Stale Data</span>
                            <span style="color: #888; font-size: 0.85rem;"> — Last data point is <b>{latest_date.strftime('%d %b %Y')}</b> ({data_age} days ago).</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        except Exception: pass
    
    data = clean_data(df, active_target, feature_cols, active_date if active_date != "None" else None)
    
    if len(data) < 80:
        st.error("Need 80+ data points for geometric manifold analysis.")
        return
    
    X, y = data[feature_cols].values, data[active_target].values
    cache_key = f"{VERSION}|{active_target}|{'|'.join(sorted(feature_cols))}|{len(data)}"
    
    if 'engine_cache' not in st.session_state or st.session_state.engine_cache != cache_key:
        with st.spinner("Preparing Fast Geometric Engine (Optimized)..."):
            progress_bar = st.progress(0, text="Initializing manifold learning space...")
            
            def update_progress(frac, text):
                progress_bar.progress(frac, text=text)
            
            engine = ManifoldPricingEngine()
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
    else: ts['Date'] = np.arange(len(ts))
    
    # ═══════════════════════════════════════════════════════════════════════
    # METRIC CARDS (UI UNCHANGED, BACKEND DYNAMICALLY MAPS GEOMETRIC CONCEPTS)
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 2])
    
    with c1:
        os_color = "success" if signal['oversold_breadth'] > 60 else "neutral"
        st.markdown(f'<div class="metric-card {os_color}"><h4>Geom Undervalued</h4><h2>{signal["oversold_breadth"]:.0f}%</h2><div class="sub-metric">Multi-Scale Support</div></div>', unsafe_allow_html=True)
    
    with c2:
        ob_color = "danger" if signal['overbought_breadth'] > 60 else "neutral"
        st.markdown(f'<div class="metric-card {ob_color}"><h4>Geom Overvalued</h4><h2>{signal["overbought_breadth"]:.0f}%</h2><div class="sub-metric">Multi-Scale Support</div></div>', unsafe_allow_html=True)
    
    with c3:
        conv_color = "success" if signal['conviction_score'] < -40 else "danger" if signal['conviction_score'] > 40 else "neutral"
        st.markdown(f'<div class="metric-card {conv_color}"><h4>Conviction</h4><h2>{signal["conviction_score"]:+.0f}</h2><div class="sub-metric">Kalman-Filtered</div></div>', unsafe_allow_html=True)
    
    with c4:
        sig_color = "success" if signal['signal'] == 'BUY' else "danger" if signal['signal'] == 'SELL' else "primary"
        st.markdown(f'<div class="metric-card {sig_color}"><h4>Signal</h4><h2>{signal["signal"]}</h2><div class="sub-metric">{signal["strength"]}</div></div>', unsafe_allow_html=True)
    
    with c5:
        reg_color = "success" if 'OVERSOLD' in signal['regime'] else "danger" if 'OVERBOUGHT' in signal['regime'] else "neutral"
        st.markdown(f'<div class="metric-card {reg_color}"><h4>Regime</h4><h2>{signal["regime"]}</h2><div class="sub-metric">Current State</div></div>', unsafe_allow_html=True)
    
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        hl = signal['model_spread']
        st.markdown(f'<div class="metric-card primary"><h4>Manifold Dist (D_t)</h4><h2>{hl:.3f}</h2><div class="sub-metric">Geometric Distance</div></div>', unsafe_allow_html=True)
    
    with d2:
        kappa_val = signal['kappa']
        k_label = 'Fast' if kappa_val > 0.1 else 'Slow' if kappa_val < 0.02 else 'Moderate'
        k_class = 'success' if kappa_val > 0.1 else 'warning' if kappa_val < 0.02 else 'neutral'
        st.markdown(f'<div class="metric-card {k_class}"><h4>Reversion (κ)</h4><h2>{kappa_val:.3f}</h2><div class="sub-metric">{k_label} OU Decay</div></div>', unsafe_allow_html=True)
    
    with d3:
        r2 = model_stats['r2_oos']
        r2_class = 'success' if r2 > 0.7 else 'warning' if r2 > 0.4 else 'danger'
        st.markdown(f'<div class="metric-card {r2_class}"><h4>Manifold R²</h4><h2>{r2:.3f}</h2><div class="sub-metric">Surface Fit (OOS)</div></div>', unsafe_allow_html=True)
    
    with d4:
        driver = signal['top_driver']
        st.markdown(f'<div class="metric-card purple"><h4>Top Driver</h4><h2 style="font-size: 1.25rem;">{driver}</h2><div class="sub-metric">∇f(X) Jacobian Eval</div></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════════════
    # TIMEFRAME FILTER
    # ═══════════════════════════════════════════════════════════════════════
    tf_col1, tf_col2 = st.columns([1, 6])
    with tf_col1: st.markdown("##### ⏱️ View Period")
    with tf_col2: selected_tf = st.radio("Timeframe", ["1M", "6M", "1Y", "2Y", "ALL"], index=2, horizontal=True, label_visibility="collapsed")
    
    ts_filtered = ts.copy()
    if selected_tf != "ALL":
        if active_date != "None" and pd.api.types.is_datetime64_any_dtype(ts['Date']):
            max_date = ts['Date'].max()
            offsets = {"1M": pd.DateOffset(months=1), "6M": pd.DateOffset(months=6), "1Y": pd.DateOffset(years=1), "2Y": pd.DateOffset(years=2)}
            ts_filtered = ts[ts['Date'] >= max_date - offsets.get(selected_tf, pd.DateOffset(years=1))]
        else:
            n_map = {"1M": 21, "6M": 126, "1Y": 252, "2Y": 504}
            ts_filtered = ts.iloc[max(0, len(ts) - n_map.get(selected_tf, 252)):]
    
    x_axis, x_title = ts_filtered['Date'], "Date" if active_date != "None" else "Index"
    
    # ═══════════════════════════════════════════════════════════════════════
    # TABS
    # ═══════════════════════════════════════════════════════════════════════
    tab_regime, tab_signal, tab_zones, tab_signals, tab_data = st.tabs([
        "**🎯 Regime Analysis**", "**📊 Signal Dashboard**", "**📈 Zone Trends**", "**📉 Signal Trends**", "**📋 Data Table**"
    ])
    
    with tab_regime:
        st.markdown("##### Kalman Conviction Score (Composite Ω_t)")
        st.markdown('<p style="color: #888;">Negative = Undervalued | Positive = Overvalued · Shaded = 95% Kalman confidence band</p>', unsafe_allow_html=True)
        
        fig_conv = go.Figure()
        if 'ConvictionUpper' in ts_filtered.columns:
            fig_conv.add_trace(go.Scatter(x=x_axis, y=ts_filtered['ConvictionUpper'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
            fig_conv.add_trace(go.Scatter(x=x_axis, y=ts_filtered['ConvictionLower'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip', fill='tonexty', fillcolor='rgba(255,195,0,0.08)'))
        fig_conv.add_trace(go.Scatter(x=x_axis, y=ts_filtered['ConvictionScore'].clip(lower=0), fill='tozeroy', fillcolor='rgba(239,68,68,0.15)', line=dict(width=0), showlegend=False))
        fig_conv.add_trace(go.Scatter(x=x_axis, y=ts_filtered['ConvictionScore'].clip(upper=0), fill='tozeroy', fillcolor='rgba(16,185,129,0.15)', line=dict(width=0), showlegend=False))
        fig_conv.add_trace(go.Scatter(x=x_axis, y=ts_filtered['ConvictionScore'], mode='lines', name='Composite Ω_t', line=dict(color='#FFC300', width=2)))
        
        if 'ConvictionRaw' in ts_filtered.columns:
            fig_conv.add_trace(go.Scatter(x=x_axis, y=ts_filtered['ConvictionRaw'], mode='lines', name='Raw Ω_t', line=dict(color='#555', width=1, dash='dot'), opacity=0.5))
        
        fig_conv.add_hline(y=40, line_dash="dash", line_color="rgba(239,68,68,0.5)")
        fig_conv.add_hline(y=-40, line_dash="dash", line_color="rgba(16,185,129,0.5)")
        fig_conv.add_hline(y=0, line_color="rgba(255,255,255,0.3)")
        fig_conv.update_layout(title="Composite Structure Index (Kalman-Filtered)", height=400, xaxis_title=x_title, yaxis_title="Score", yaxis=dict(range=[-100, 100]))
        st.plotly_chart(update_chart_theme(fig_conv), use_container_width=True)
        
        st.markdown("---")
        st.markdown("##### Base Conviction Score (Raw Ω_t)")
        
        if 'ConvictionRaw' in ts_filtered.columns:
            fig_raw = go.Figure()
            fig_raw.add_trace(go.Scatter(x=x_axis, y=ts_filtered['ConvictionRaw'].clip(lower=0), fill='tozeroy', fillcolor='rgba(239,68,68,0.15)', line=dict(width=0), showlegend=False))
            fig_raw.add_trace(go.Scatter(x=x_axis, y=ts_filtered['ConvictionRaw'].clip(upper=0), fill='tozeroy', fillcolor='rgba(16,185,129,0.15)', line=dict(width=0), showlegend=False))
            conv_colors = ['#10b981' if c < -40 else '#ef4444' if c > 40 else '#888' for c in ts_filtered['ConvictionRaw']]
            fig_raw.add_trace(go.Scatter(x=x_axis, y=ts_filtered['ConvictionRaw'], mode='lines+markers', name='Raw Ω_t', line=dict(color='#FFC300', width=2), marker=dict(size=4, color=conv_colors)))
            fig_raw.add_hline(y=40, line_dash="dash", line_color="rgba(239,68,68,0.5)")
            fig_raw.add_hline(y=-40, line_dash="dash", line_color="rgba(16,185,129,0.5)")
            fig_raw.add_hline(y=0, line_color="rgba(255,255,255,0.3)")
            fig_raw.update_layout(title="Base Structural Score", height=400, xaxis_title=x_title, yaxis_title="Score", yaxis=dict(range=[-100, 100]))
            st.plotly_chart(update_chart_theme(fig_raw), use_container_width=True)

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Regime Distribution")
            regime_data = {
                "Regime": ["🟢 Strongly Oversold", "🔵 Oversold", "⚪ Neutral", "🟠 Overbought", "🔴 Strongly Overbought"],
                "Count": [regime_stats['strongly_oversold'], regime_stats['oversold'], regime_stats['neutral'], regime_stats['overbought'], regime_stats['strongly_overbought']],
                "Pct": [f"{regime_stats['strongly_oversold'] / len(ts) * 100:.1f}%", f"{regime_stats['oversold'] / len(ts) * 100:.1f}%", f"{regime_stats['neutral'] / len(ts) * 100:.1f}%", f"{regime_stats['overbought'] / len(ts) * 100:.1f}%", f"{regime_stats['strongly_overbought'] / len(ts) * 100:.1f}%"]
            }
            st.dataframe(pd.DataFrame(regime_data), width='stretch', hide_index=True)
        
        with col2:
            st.markdown("##### Current Regime & Diagnostics")
            curr_regime = signal['regime']
            regime_box_class = "success" if "OVERSOLD" in curr_regime else "danger" if "OVERBOUGHT" in curr_regime else ""
            st.markdown(f"""
            <div class="guide-box {regime_box_class}">
                <strong>Current: {curr_regime}</strong><br><br>
                {'Manifold surface and statistical distance align on deep geometric undervaluation.' if 'OVERSOLD' in curr_regime else 
                 'Manifold surface and statistical distance align on extreme geometric overvaluation.' if 'OVERBOUGHT' in curr_regime else
                 'System remains in structural equilibrium.'}
            </div>
            """, unsafe_allow_html=True)
            k_label = 'Fast ✅' if signal['kappa'] > 0.1 else 'Slow ⚠️' if signal['kappa'] < 0.02 else 'Moderate'
            st.markdown(f"Manifold R²: **{model_stats['r2_oos']:.4f}** | RMSE: **{model_stats['rmse_oos']:.4f}** | Manifold Dist: **{signal['model_spread']:.3f}** | OU Decay (κ): **{signal['kappa']:.3f}** ({k_label}) | Top Driver: **{signal['top_driver']}**")
    
    with tab_signal:
        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.markdown("##### Current Signal Analysis")
            signal_class = 'undervalued' if signal['signal'] == 'BUY' else 'overvalued' if signal['signal'] == 'SELL' else 'fair'
            signal_emoji = "🟢" if signal['signal'] == 'BUY' else "🔴" if signal['signal'] == 'SELL' else "🟡"
            st.markdown(f"""
            <div class="signal-card {signal_class}">
                <div class="label">GEOMETRIC SIGNAL</div>
                <div class="value">{signal_emoji} {signal['signal']}</div>
                <div class="subtext">{signal['strength']} Strength • {signal['confidence']} Confidence • 
                Manifold Dist D_t = {signal['model_spread']:.3f}</div>
            </div>
            """, unsafe_allow_html=True)
            conv_pct, conv_color = (signal['conviction_score'] + 100) / 2, '#10b981' if signal['conviction_score'] < -20 else '#ef4444' if signal['conviction_score'] > 20 else '#FFC300'
            st.markdown(f"""
            <div class="conviction-meter">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="color: #10b981; font-size: 0.75rem;">OVERSOLD</span>
                    <span style="color: #888; font-size: 0.75rem;">Composite Ω_t: {signal['conviction_score']:+.0f} [{signal['conviction_lower']:+.0f}, {signal['conviction_upper']:+.0f}]</span>
                    <span style="color: #ef4444; font-size: 0.75rem;">OVERBOUGHT</span>
                </div>
                <div class="conviction-bar"><div class="conviction-fill" style="width: {conv_pct}%; background: {conv_color};"></div></div>
            </div>
            """, unsafe_allow_html=True)
            if signal['has_bullish_div']: st.markdown('<span class="status-badge buy">🔔 STRUCTURAL BREAK (Bullish Alert)</span>', unsafe_allow_html=True)
            if signal['has_bearish_div']: st.markdown('<span class="status-badge sell">🔔 STRUCTURAL BREAK (Bearish Alert)</span>', unsafe_allow_html=True)
            if signal['model_spread'] > np.percentile(engine.D_t_series, 90):
                st.markdown(f"""
                <div style="background: rgba(245,158,11,0.1); border: 1px solid #f59e0b; border-radius: 8px; padding: 0.5rem 1rem; margin-top: 0.5rem;">
                    <span style="color: #f59e0b; font-size: 0.8rem;">⚠️ Extreme Structural Deformation ({signal['model_spread']:.2f}) — regime shift in progress. Phase space geometry highly perturbed.</span>
                </div>
                """, unsafe_allow_html=True)
        
        with col_right:
            st.markdown("##### Multi-Scale Geom Breakdown")
            for lb in engine.LOOKBACKS:
                if lb not in engine.lookback_data: continue
                z_stat = engine.lookback_data[lb]['Z_t'][-1]
                d_geom = engine.D_t_series[-1]
                
                if z_stat < -1.0 and d_geom > np.percentile(engine.D_t_series, 75):
                    zone, zone_color = 'Extreme Under', '#10b981'
                elif z_stat < -0.5:
                    zone, zone_color = 'Undervalued', '#10b981'
                elif z_stat > 1.0 and d_geom > np.percentile(engine.D_t_series, 75):
                    zone, zone_color = 'Extreme Over', '#ef4444'
                elif z_stat > 0.5:
                    zone, zone_color = 'Overvalued', '#ef4444'
                else:
                    zone, zone_color = 'Equilibrium', '#888'
                
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; padding: 0.5rem; border-bottom: 1px solid #2A2A2A;">
                    <span style="color: #888;">{lb}-Day</span>
                    <span style="color: {zone_color}; font-weight: 600;">{zone}</span>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("##### Actual vs Equilibrium Manifold Surface")
        fig = make_subplots(rows=2, cols=1, row_heights=[0.6, 0.4], shared_xaxes=True, vertical_spacing=0.05)
        fig.add_trace(go.Scatter(x=x_axis, y=ts_filtered['Actual'], mode='lines', name='Actual', line=dict(color='#FFC300', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_axis, y=ts_filtered['FairValue'], mode='lines', name='Equilibrium Surface y_fair', line=dict(color='#06b6d4', width=2, dash='dash')), row=1, col=1)
        
        colors = ['#10b981' if r < 0 else '#ef4444' for r in ts_filtered['Residual']]
        fig.add_trace(go.Bar(x=x_axis, y=ts_filtered['Residual'], name='Deviation (OOS)', marker_color=colors, showlegend=False), row=2, col=1)
        fig.add_hline(y=0, line_color="#FFC300", line_width=1, row=2, col=1)
        
        if hasattr(engine, 'ou_projection') and pd.api.types.is_datetime64_any_dtype(ts['Date']):
            proj_dates = pd.date_range(start=ts['Date'].iloc[-1], periods=91, freq='D')[1:]
            fig.add_trace(go.Scatter(x=proj_dates, y=engine.ou_projection, mode='lines', name='OU Reversion Path', line=dict(color='#FFC300', width=1.5, dash='dot'), opacity=0.5), row=2, col=1)
        
        fig.update_layout(height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02))
        fig.update_yaxes(title_text=active_target, row=1, col=1)
        fig.update_yaxes(title_text="Deviation (Z_t)", row=2, col=1)
        st.plotly_chart(update_chart_theme(fig), use_container_width=True)
    
    with tab_zones:
        st.markdown("##### Geometric Support Over Time")
        st.markdown('<p style="color: #888;">% of timeframes supporting geometric mispricing</p>', unsafe_allow_html=True)
        fig_zones = go.Figure()
        fig_zones.add_trace(go.Scatter(x=x_axis, y=ts_filtered['OversoldBreadth'], fill='tozeroy', fillcolor='rgba(16,185,129,0.2)', line=dict(color='#10b981', width=2), name='Geom Undervalued %'))
        fig_zones.add_trace(go.Scatter(x=x_axis, y=ts_filtered['OverboughtBreadth'], fill='tozeroy', fillcolor='rgba(239,68,68,0.2)', line=dict(color='#ef4444', width=2), name='Geom Overvalued %'))
        fig_zones.add_hline(y=60, line_dash="dash", line_color="rgba(255,195,0,0.3)")
        fig_zones.update_layout(title="Geometric Multi-Scale Breadth", height=400, xaxis_title=x_title, yaxis_title="% of Timeframes", yaxis=dict(range=[0, 100]))
        st.plotly_chart(update_chart_theme(fig_zones), use_container_width=True)
        
        st.markdown("---")
        st.markdown("##### Average Statistical Deviation (Z_t)")
        fig_z = go.Figure()
        z_colors = ['#10b981' if z < -1 else '#ef4444' if z > 1 else '#888' for z in ts_filtered['AvgZ']]
        fig_z.add_trace(go.Bar(x=x_axis, y=ts_filtered['AvgZ'], marker_color=z_colors, name='Statistical Z'))
        fig_z.add_hline(y=0, line_color="#FFC300", line_width=1)
        fig_z.add_hline(y=2, line_dash="dash", line_color="rgba(239,68,68,0.5)")
        fig_z.add_hline(y=-2, line_dash="dash", line_color="rgba(16,185,129,0.5)")
        fig_z.update_layout(title="Multi-Scale Average Mispricing Deviation", height=350, xaxis_title=x_title, yaxis_title="Z-Score")
        st.plotly_chart(update_chart_theme(fig_z), use_container_width=True)
    
    with tab_signals:
        st.markdown("##### Buy/Sell Signal Count by Period")
        fig_signals = go.Figure()
        fig_signals.add_trace(go.Bar(x=x_axis, y=ts_filtered['BuySignalBreadth'], name='Buy Signals', marker=dict(color='#10b981')))
        fig_signals.add_trace(go.Bar(x=x_axis, y=-ts_filtered['SellSignalBreadth'], name='Sell Signals', marker=dict(color='#ef4444')))
        fig_signals.update_layout(title="Signal Count by Period", height=350, xaxis_title=x_title, yaxis_title="Signal Count", barmode='relative')
        st.plotly_chart(update_chart_theme(fig_signals), use_container_width=True)
        
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
    
    with tab_data:
        st.markdown(f"##### Time Series Data ({len(ts_filtered)} observations)")
        display_cols = ['Date', 'Actual', 'FairValue', 'Residual', 'ModelSpread', 'AvgZ', 'TopDriver',
                        'OversoldBreadth', 'OverboughtBreadth', 'ConvictionScore', 'Regime',
                        'BullishDiv', 'BearishDiv']
        display_cols = [c for c in display_cols if c in ts_filtered.columns]
        
        display_ts = ts_filtered[display_cols].copy()
        
        # Rename ModelSpread and AvgZ to reflect Geometric context for export clarity
        display_ts = display_ts.rename(columns={'ModelSpread': 'ManifoldDist_Dt', 'AvgZ': 'StatScore_Zt'})
        
        for col in ['Residual', 'ManifoldDist_Dt', 'StatScore_Zt', 'FairValue', 'ConvictionScore', 'OversoldBreadth', 'OverboughtBreadth']:
            if col in display_ts.columns:
                display_ts[col] = display_ts[col].round(3 if col in ['StatScore_Zt', 'ManifoldDist_Dt'] else 2 if col == 'FairValue' else 1)
        
        if 'BullishDiv' in display_ts.columns: display_ts['BullishDiv'] = display_ts['BullishDiv'].apply(lambda x: '🟢' if x else '')
        if 'BearishDiv' in display_ts.columns: display_ts['BearishDiv'] = display_ts['BearishDiv'].apply(lambda x: '🔴' if x else '')
        
        st.dataframe(display_ts, width='stretch', hide_index=True, height=500)
        st.download_button("📥 Download Full CSV", ts.to_csv(index=False).encode('utf-8'), f"aarambh_geom_{active_target}_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
    
    render_footer()

if __name__ == "__main__":
    main()
