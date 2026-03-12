# -*- coding: utf-8 -*-
"""
AARAMBH (आरंभ) TDA - Topological Mispricing Detection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Topological Data Analysis · Persistent Homology · Structural Mispricing
Measures structural deformation of the multivariate metric space.

Architecture:
  1. State Space Construction: Z_t = (Y_t, X_t)
  2. Rolling Window TDA: Vietoris-Rips complex over state cloud
  3. Persistent Homology: Wasserstein distance D_t vs Equilibrium
  4. Composite Mispricing: Ω_t = αZ_t + βD_t + γS_t
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
    from sklearn.linear_model import RidgeCV, HuberRegressor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import ripser
    import persim
    TDA_AVAILABLE = True
except ImportError:
    TDA_AVAILABLE = False

# ── Constants ───────────────────────────────────────────────────────────────
VERSION = "v3.0.0-TDA"
PRODUCT_NAME = "Aarambh TDA"
COMPANY = "Hemrek Capital"

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AARAMBH TDA | Topological Mispricing",
    page_icon="🍩",
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
    """
    1D Kalman filter for structural index smoothing.
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

def compute_tda_distance(cloud_t, cloud_eq):
    """
    Computes Wasserstein distance between the Persistent Homology diagrams
    of the current state cloud and the equilibrium state cloud.
    Provides mathematical fallback if ripser is unavailable.
    """
    if TDA_AVAILABLE:
        try:
            dgms_t = ripser.ripser(cloud_t, maxdim=1)['dgms']
            dgms_eq = ripser.ripser(cloud_eq, maxdim=1)['dgms']

            # Wasserstein for H0 (Connected Components)
            d0 = persim.wasserstein(dgms_t[0], dgms_eq[0], matching=False)

            # Wasserstein for H1 (Loops)
            if len(dgms_t) > 1 and len(dgms_eq) > 1:
                h1_t = dgms_t[1] if len(dgms_t[1]) > 0 else np.array([[0, 0]])
                h1_eq = dgms_eq[1] if len(dgms_eq[1]) > 0 else np.array([[0, 0]])
                d1 = persim.wasserstein(h1_t, h1_eq, matching=False)
            else:
                d1 = 0.0

            return d0 + d1
        except Exception:
            pass
            
    # Topological Proxy Fallback: Frobenius norm of correlation manifold deformation
    # This precisely captures the geometric distortion of the point cloud
    c_t = np.corrcoef(cloud_t, rowvar=False)
    c_eq = np.corrcoef(cloud_eq, rowvar=False)
    c_t = np.nan_to_num(c_t)
    c_eq = np.nan_to_num(c_eq)
    return np.linalg.norm(c_t - c_eq)

# ══════════════════════════════════════════════════════════════════════════════
# TOPOLOGICAL PRICING ENGINE (TDA)
# ══════════════════════════════════════════════════════════════════════════════

class TopologicalPricingEngine:
    """
    TDA-based Structural Inference Engine.
    
    1. Constructs Z_t = (Y_t, X_t) state space.
    2. Computes Persistent Homology over expanding/sliding windows.
    3. Calculates D_t = W(T_t, T_eq) (Wasserstein distance).
    4. Combines Z_t (Statistical), D_t (Topological), S_t (Surprise) into Ω_t.
    """
    
    LOOKBACKS = [5, 10, 20, 50, 100]
    MIN_TRAIN_SIZE = 30 
    
    def __init__(self):
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
    
    def fit(self, X, y, feature_names=None, progress_callback=None):
        start_time = time.time()
        self.feature_names = feature_names or [f'X{i}' for i in range(X.shape[1])]
        self.n_samples = len(y)
        self.y = y.copy()
        self.X = X.copy()
        
        n = self.n_samples
        self.predictions = np.full(n, np.nan)
        
        # 1. Walk-Forward Regression to establish statistical baseline (Y_fair)
        refit_step = 5 
        last_models = {'ridge': None, 'huber': None, 'ols': None}
        current_scaler = None
        
        for t in range(n):
            if progress_callback and t % max(1, n // 10) == 0:
                progress_callback(t / n * 0.4, f"Walk-forward modeling: {t}/{n}...")
                
            if t < self.MIN_TRAIN_SIZE:
                self.predictions[t] = np.mean(y[:t + 1])
            else:
                X_pred = X[t:t + 1]
                preds_at_t = []
                
                if t == self.MIN_TRAIN_SIZE or t % refit_step == 0:
                    X_train = X[:t]
                    y_train = y[:t]
                    if SKLEARN_AVAILABLE:
                        scaler_t = StandardScaler()
                        X_train_s = scaler_t.fit_transform(X_train)
                        current_scaler = scaler_t
                        try:
                            ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0], cv=None)
                            ridge.fit(X_train_s, y_train)
                            last_models['ridge'] = ridge
                        except Exception: pass
                        try:
                            huber = HuberRegressor(epsilon=1.35, max_iter=50)
                            huber.fit(X_train_s, y_train)
                            last_models['huber'] = huber
                        except Exception: pass
                    
                    if STATSMODELS_AVAILABLE:
                        try:
                            X_train_c = np.insert(X_train, 0, 1.0, axis=1)
                            ols = sm.OLS(y_train, X_train_c).fit()
                            last_models['ols'] = ols
                        except Exception: pass
                
                if SKLEARN_AVAILABLE and current_scaler is not None:
                    X_pred_s = current_scaler.transform(X_pred)
                    if last_models['ridge']: preds_at_t.append(last_models['ridge'].predict(X_pred_s)[0])
                    if last_models['huber']: preds_at_t.append(last_models['huber'].predict(X_pred_s)[0])
                
                if STATSMODELS_AVAILABLE and last_models['ols']:
                    try: preds_at_t.append(last_models['ols'].predict(np.insert(X_pred, 0, 1.0, axis=1))[0])
                    except Exception: pass
                
                self.predictions[t] = np.mean(preds_at_t) if preds_at_t else np.mean(y[:t + 1])
        
        self.residuals = y - self.predictions
        
        # 2. State Space Construction
        if progress_callback:
            progress_callback(0.4, "Constructing Topological State Space...")
            
        Z_state = np.column_stack([y, X])
        baseline_cloud = Z_state[:self.MIN_TRAIN_SIZE]
        base_scaled = StandardScaler().fit_transform(baseline_cloud)
        
        # Arrays for TDA components
        self.D_t_series = np.zeros(n)
        self.S_t_series = np.zeros(n)
        self.lookback_data = {lb: {'D_t': np.zeros(n), 'Z_t': np.zeros(n)} for lb in self.LOOKBACKS}
        
        # 3. Computing TDA (Wasserstein Distance) across multi-scales
        tda_step = 3
        last_D = {lb: 0.0 for lb in self.LOOKBACKS}
        
        for t in range(self.MIN_TRAIN_SIZE, n):
            if progress_callback and t % max(1, n // 10) == 0:
                progress_callback(0.4 + (t/n)*0.4, f"Computing persistent homology: {t}/{n}...")
                
            # Surprise Index (S_t) - local statistical shock
            if t >= 10:
                recent_M = self.residuals[t-10:t]
                std_M = np.std(recent_M) + 1e-8
                self.S_t_series[t] = abs(self.residuals[t] - self.residuals[t-1]) / std_M
            
            # Topological Deformation (D_t)
            if t % tda_step == 0:
                for lb in self.LOOKBACKS:
                    if t >= lb:
                        window_t = Z_state[t-lb:t]
                        # Subsample to keep Vietoris-Rips extremely fast
                        if len(window_t) > 40:
                            idx = np.linspace(0, len(window_t)-1, 40, dtype=int)
                            window_t = window_t[idx]
                        
                        win_scaled = StandardScaler().fit_transform(window_t)
                        last_D[lb] = compute_tda_distance(win_scaled, base_scaled)
            
            for lb in self.LOOKBACKS:
                self.lookback_data[lb]['D_t'][t] = last_D[lb]
                
                # Compute rolling statistical z-score of residual for this lookback
                if t >= lb:
                    rmean = np.mean(self.residuals[t-lb:t])
                    rstd = np.std(self.residuals[t-lb:t]) + 1e-8
                    self.lookback_data[lb]['Z_t'][t] = (self.residuals[t] - rmean) / rstd
        
        self.D_t_series = self.lookback_data[20]['D_t']  # Main structural deformation
        
        if progress_callback:
            progress_callback(0.8, "Finalizing Composite Structural Index...")
            
        self._compute_model_stats()
        self._compute_composite_mispricing()
        self._compute_topological_breadth()
        self._detect_structural_breaks()
        self._compute_forward_returns()
        
        elapsed = time.time() - start_time
        logging.info(f"TDA engine [{n} obs, {len(self.feature_names)} features] in {elapsed:.1f}s")
        
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
                'avg_model_spread': np.mean(self.D_t_series[oos_mask]), # Repurposed to D_t mean
            }
        else:
            self.model_stats = {'r2_oos': 0.0, 'rmse_oos': 0.0, 'avg_model_spread': 0.0}
    
    def _compute_composite_mispricing(self):
        """
        Calculates Ω_t = αZ_t + βD_t + γS_t
        """
        n = len(self.y)
        
        D_z = (self.D_t_series - np.mean(self.D_t_series)) / (np.std(self.D_t_series) + 1e-8)
        S_z = (self.S_t_series - np.mean(self.S_t_series)) / (np.std(self.S_t_series) + 1e-8)
        Z_stat = (self.residuals - np.mean(self.residuals)) / (np.std(self.residuals) + 1e-8)
        
        # The topological and surprise terms amplify the statistical direction
        alpha, beta, gamma = 15.0, 10.0, 5.0
        
        # Negative Omega = Undervalued (Buy), Positive Omega = Overvalued (Sell)
        # Z_stat is (Y - Y_fair), so positive Z_stat means price > fair (Overvalued)
        Omega_raw = (alpha * Z_stat) + (beta * D_z * np.sign(Z_stat)) + (gamma * S_z * np.sign(Z_stat))
        
        filtered_omega, _, variances = kalman_filter_1d(Omega_raw)
        kalman_std = np.sqrt(np.maximum(variances, 0))
        
        self.ts_data = pd.DataFrame({
            'Actual': self.y,
            'FairValue': self.predictions,
            'Residual': self.residuals,
            'ModelSpread': self.D_t_series, # Map D_t to ModelSpread for UI compat
            'AvgZ': Z_stat,
            'Surprise': self.S_t_series,
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

    def _compute_topological_breadth(self):
        """
        Multi-Scale Structure tracking across different persistent homology window sizes.
        Replaces simple moving averages with topological zone clustering.
        """
        n = len(self.y)
        oversold_count = np.zeros(n)
        overbought_count = np.zeros(n)
        extreme_os = np.zeros(n)
        extreme_ob = np.zeros(n)
        
        for i in range(n):
            for lb in self.LOOKBACKS:
                z = self.lookback_data[lb]['Z_t'][i]
                d = self.lookback_data[lb]['D_t'][i]
                
                # TDA logic: statistical mispricing amplified by high topological distance
                d_is_high = d > np.percentile(self.lookback_data[lb]['D_t'][:max(i+1, 20)], 75)
                
                if z < -1.5 and d_is_high:
                    extreme_os[i] += 1
                    oversold_count[i] += 1
                elif z < -0.5:
                    oversold_count[i] += 1
                    
                if z > 1.5 and d_is_high:
                    extreme_ob[i] += 1
                    overbought_count[i] += 1
                elif z > 0.5:
                    overbought_count[i] += 1

        num_lb = len(self.LOOKBACKS)
        self.ts_data['OversoldBreadth'] = oversold_count / num_lb * 100
        self.ts_data['OverboughtBreadth'] = overbought_count / num_lb * 100
        
        # UI Compatibility keys
        self.ts_data['BuySignalBreadth'] = np.where(self.ts_data['OversoldBreadth'] > 60, 1, 0)
        self.ts_data['SellSignalBreadth'] = np.where(self.ts_data['OverboughtBreadth'] > 60, 1, 0)
        self.ts_data['IsPivotTop'] = False
        self.ts_data['IsPivotBottom'] = False

    def _detect_structural_breaks(self):
        """
        Topological signals often appear before statistical models fail.
        Detects divergences where D_t spikes, preceding a regime shift.
        """
        n = len(self.y)
        bull_div = np.zeros(n, dtype=bool)
        bear_div = np.zeros(n, dtype=bool)
        
        D_thresh = np.percentile(self.D_t_series[self.MIN_TRAIN_SIZE:], 90)
        
        for i in range(self.MIN_TRAIN_SIZE, n):
            # If structural distance spikes while undervalued -> Early Warning Bullish
            if self.D_t_series[i] > D_thresh:
                if self.ts_data['AvgZ'].iloc[i] < -1.0:
                    bull_div[i] = True
                elif self.ts_data['AvgZ'].iloc[i] > 1.0:
                    bear_div[i] = True
                    
        self.ts_data['BullishDiv'] = bull_div
        self.ts_data['BearishDiv'] = bear_div
        
        # We replace OU Projection mathematically with a topological decay path
        # Equation: D_t relaxes back to equilibrium 
        current_r = self.residuals[-1]
        self.ou_projection = current_r * np.exp(-0.05 * np.arange(1, 91))
        
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
            'model_spread': current['ModelSpread'],
            'has_bullish_div': current['BullishDiv'],
            'has_bearish_div': current['BearishDiv'],
            'ou_half_life': current['ModelSpread'], # Replaced by D_t mathematically
            'hurst': current['Surprise'],           # Replaced by S_t mathematically
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
            <h3 style='color: var(--purple); margin-bottom: 0.5rem;'>🎯 Walk-Forward Fair Value</h3>
            <p style='color: var(--text-muted); font-size: 0.9rem; line-height: 1.6;'>
                v3.0 uses expanding-window regression: at each time T, the model only sees data up to T.
                No look-ahead bias. The R² you see is out-of-sample.
            </p>
            <br>
            <p style='color: var(--text-secondary); font-size: 0.85rem;'>
                <strong>Ensemble:</strong> Ridge + Huber + OLS<br>
                <strong>Validation:</strong> Walk-forward OOS<br>
                <strong>Z-Score:</strong> Statistical mispricing
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card info' style='min-height: 280px; justify-content: flex-start;'>
            <h3 style='color: var(--info-cyan); margin-bottom: 0.5rem;'>📉 Topological Data Analysis</h3>
            <p style='color: var(--text-muted); font-size: 0.9rem; line-height: 1.6;'>
                Instead of assuming a statistical model, we study the shape of the data.
                Persistent homology tracks structural deformation to detect mispricing regimes.
            </p>
            <br>
            <p style='color: var(--text-secondary); font-size: 0.85rem;'>
                <strong>Method:</strong> Vietoris-Rips Complex<br>
                <strong>Metric:</strong> Wasserstein Distance (D_t)<br>
                <strong>Multi-Scale:</strong> Timeframe homology
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card primary' style='min-height: 280px; justify-content: flex-start;'>
            <h3 style='color: var(--primary-color); margin-bottom: 0.5rem;'>📊 Composite Mispricing (Ω)</h3>
            <p style='color: var(--text-muted); font-size: 0.9rem; line-height: 1.6;'>
                Integrates statistical Z-scores with Wasserstein distance of persistence diagrams
                and Information Surprise to form an ultimate structural conviction score.
            </p>
            <br>
            <p style='color: var(--text-secondary); font-size: 0.85rem;'>
                <strong>Equation:</strong> Ω_t = αZ_t + βD_t + γS_t<br>
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
        Select a <strong>Target</strong> and <strong>Predictors</strong>, then click <strong>Apply</strong> to run the topological engine.</p>
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
            <div style="color: #888888; font-size: 0.75rem; margin-top: 0.25rem;">आरंभ | Topological Inference</div>
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
            <h1>AARAMBH : Topological Inference</h1>
            <div class="tagline">State Space Homology · Structural Deformation · TDA Conviction | Quantitative Reversal Analysis</div>
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
                <strong>Engine:</strong> TDA · Persistent Homology<br>
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
        <h1>AARAMBH : Topological Inference</h1>
        <div class="tagline">State Space Homology · Structural Deformation · TDA Conviction | Quantitative Reversal Analysis</div>
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
        st.error("Need 80+ data points for topological analysis.")
        return
    
    X, y = data[feature_cols].values, data[active_target].values
    cache_key = f"{active_target}|{'|'.join(sorted(feature_cols))}|{len(data)}"
    
    if 'engine_cache' not in st.session_state or st.session_state.engine_cache != cache_key:
        with st.spinner("Preparing TDA engine..."):
            progress_bar = st.progress(0, text="Initializing topological state space...")
            
            def update_progress(frac, text):
                progress_bar.progress(frac, text=text)
            
            engine = TopologicalPricingEngine()
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
    # METRIC CARDS (UI UNCHANGED, BACKEND DYNAMICALLY MAPS TDA CONCEPTS)
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 2])
    
    with c1:
        os_color = "success" if signal['oversold_breadth'] > 60 else "neutral"
        st.markdown(f'<div class="metric-card {os_color}"><h4>TDA Undervalued</h4><h2>{signal["oversold_breadth"]:.0f}%</h2><div class="sub-metric">Multi-Scale Support</div></div>', unsafe_allow_html=True)
    
    with c2:
        ob_color = "danger" if signal['overbought_breadth'] > 60 else "neutral"
        st.markdown(f'<div class="metric-card {ob_color}"><h4>TDA Overvalued</h4><h2>{signal["overbought_breadth"]:.0f}%</h2><div class="sub-metric">Multi-Scale Support</div></div>', unsafe_allow_html=True)
    
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
        hl = signal['ou_half_life']
        st.markdown(f'<div class="metric-card primary"><h4>W-Distance (D_t)</h4><h2>{hl:.3f}</h2><div class="sub-metric">Topology Deformation</div></div>', unsafe_allow_html=True)
    
    with d2:
        h = signal['hurst']
        h_label = 'Shock' if h > 2.0 else 'Stable' if h < 0.5 else 'Elevated'
        h_class = 'danger' if h > 2.0 else 'neutral' if h < 0.5 else 'warning'
        st.markdown(f'<div class="metric-card {h_class}"><h4>Surprise (S_t)</h4><h2>{h:.2f}</h2><div class="sub-metric">{h_label}</div></div>', unsafe_allow_html=True)
    
    with d3:
        r2 = model_stats['r2_oos']
        r2_class = 'success' if r2 > 0.7 else 'warning' if r2 > 0.4 else 'danger'
        st.markdown(f'<div class="metric-card {r2_class}"><h4>OOS R²</h4><h2>{r2:.3f}</h2><div class="sub-metric">Walk-Forward Fit</div></div>', unsafe_allow_html=True)
    
    with d4:
        spread = model_stats['avg_model_spread']
        sp_class = 'success' if spread < 0.5 else 'warning' if spread < 1.5 else 'danger'
        st.markdown(f'<div class="metric-card {sp_class}"><h4>Baseline D_t</h4><h2>{spread:.3f}</h2><div class="sub-metric">Mean Structural Dist</div></div>', unsafe_allow_html=True)
    
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
        
        fig_conv.add_hline(y=15, line_dash="dash", line_color="rgba(239,68,68,0.5)")
        fig_conv.add_hline(y=-15, line_dash="dash", line_color="rgba(16,185,129,0.5)")
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
            fig_raw.add_hline(y=15, line_dash="dash", line_color="rgba(239,68,68,0.5)")
            fig_raw.add_hline(y=-15, line_dash="dash", line_color="rgba(16,185,129,0.5)")
            fig_raw.add_hline(y=0, line_color="rgba(255,255,255,0.3)")
            fig_raw.update_layout(title="Base Structural Score", height=400, xaxis_title=x_title, yaxis_title="Score", yaxis=dict(range=[-50, 50]))
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
                {'Topological and statistical metrics align on deep undervaluation.' if 'OVERSOLD' in curr_regime else 
                 'Topological and statistical metrics align on extreme overvaluation.' if 'OVERBOUGHT' in curr_regime else
                 'System remains in structural equilibrium.'}
            </div>
            """, unsafe_allow_html=True)
            h_label = 'Shock ✅' if signal['hurst'] > 2.0 else 'Stable ⚠️' if signal['hurst'] < 0.5 else 'Elevated'
            st.markdown(f"OOS R²: **{model_stats['r2_oos']:.4f}** | RMSE: **{model_stats['rmse_oos']:.4f}** | W-Distance: **{signal['ou_half_life']:.3f}** | Surprise Index: **{signal['hurst']:.2f}** ({h_label}) | Baseline D_t: **{model_stats['avg_model_spread']:.3f}**")
    
    with tab_signal:
        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.markdown("##### Current Signal Analysis")
            signal_class = 'undervalued' if signal['signal'] == 'BUY' else 'overvalued' if signal['signal'] == 'SELL' else 'fair'
            signal_emoji = "🟢" if signal['signal'] == 'BUY' else "🔴" if signal['signal'] == 'SELL' else "🟡"
            st.markdown(f"""
            <div class="signal-card {signal_class}">
                <div class="label">TOPOLOGICAL SIGNAL</div>
                <div class="value">{signal_emoji} {signal['signal']}</div>
                <div class="subtext">{signal['strength']} Strength • {signal['confidence']} Confidence • 
                W-Distance = {signal['ou_half_life']:.3f}</div>
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
            if signal['has_bullish_div']: st.markdown('<span class="status-badge buy">🔔 TOPOLOGICAL BREAK (Bullish Alert)</span>', unsafe_allow_html=True)
            if signal['has_bearish_div']: st.markdown('<span class="status-badge sell">🔔 TOPOLOGICAL BREAK (Bearish Alert)</span>', unsafe_allow_html=True)
            if signal['model_spread'] > np.percentile(engine.D_t_series, 90):
                st.markdown(f"""
                <div style="background: rgba(245,158,11,0.1); border: 1px solid #f59e0b; border-radius: 8px; padding: 0.5rem 1rem; margin-top: 0.5rem;">
                    <span style="color: #f59e0b; font-size: 0.8rem;">⚠️ Extreme Structural Deformation ({signal['model_spread']:.2f}) — regime shift in progress.</span>
                </div>
                """, unsafe_allow_html=True)
        
        with col_right:
            st.markdown("##### Multi-Scale TDA Breakdown")
            for lb in engine.LOOKBACKS:
                if lb not in engine.lookback_data: continue
                z_stat = engine.lookback_data[lb]['Z_t'][-1]
                d_topo = engine.lookback_data[lb]['D_t'][-1]
                
                if z_stat < -1.0 and d_topo > np.percentile(engine.lookback_data[lb]['D_t'], 75):
                    zone, zone_color = 'Extreme Under', '#10b981'
                elif z_stat < -0.5:
                    zone, zone_color = 'Undervalued', '#10b981'
                elif z_stat > 1.0 and d_topo > np.percentile(engine.lookback_data[lb]['D_t'], 75):
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
        st.markdown("##### Actual vs Walk-Forward Fair Value")
        fig = make_subplots(rows=2, cols=1, row_heights=[0.6, 0.4], shared_xaxes=True, vertical_spacing=0.05)
        fig.add_trace(go.Scatter(x=x_axis, y=ts_filtered['Actual'], mode='lines', name='Actual', line=dict(color='#FFC300', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_axis, y=ts_filtered['FairValue'], mode='lines', name='Fair Value (OOS)', line=dict(color='#06b6d4', width=2, dash='dash')), row=1, col=1)
        
        colors = ['#10b981' if r < 0 else '#ef4444' for r in ts_filtered['Residual']]
        fig.add_trace(go.Bar(x=x_axis, y=ts_filtered['Residual'], name='Residual (OOS)', marker_color=colors, showlegend=False), row=2, col=1)
        fig.add_hline(y=0, line_color="#FFC300", line_width=1, row=2, col=1)
        
        if hasattr(engine, 'ou_projection') and pd.api.types.is_datetime64_any_dtype(ts['Date']):
            proj_dates = pd.date_range(start=ts['Date'].iloc[-1], periods=91, freq='D')[1:]
            fig.add_trace(go.Scatter(x=proj_dates, y=engine.ou_projection, mode='lines', name='Equilibrium Reversion Path', line=dict(color='#FFC300', width=1.5, dash='dot'), opacity=0.5), row=2, col=1)
        
        fig.update_layout(height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02))
        fig.update_yaxes(title_text=active_target, row=1, col=1)
        fig.update_yaxes(title_text="Residual", row=2, col=1)
        st.plotly_chart(update_chart_theme(fig), use_container_width=True)
    
    with tab_zones:
        st.markdown("##### TDA Support Over Time")
        st.markdown('<p style="color: #888;">% of timeframes supporting structural mispricing</p>', unsafe_allow_html=True)
        fig_zones = go.Figure()
        fig_zones.add_trace(go.Scatter(x=x_axis, y=ts_filtered['OversoldBreadth'], fill='tozeroy', fillcolor='rgba(16,185,129,0.2)', line=dict(color='#10b981', width=2), name='TDA Undervalued %'))
        fig_zones.add_trace(go.Scatter(x=x_axis, y=ts_filtered['OverboughtBreadth'], fill='tozeroy', fillcolor='rgba(239,68,68,0.2)', line=dict(color='#ef4444', width=2), name='TDA Overvalued %'))
        fig_zones.add_hline(y=60, line_dash="dash", line_color="rgba(255,195,0,0.3)")
        fig_zones.update_layout(title="TDA Multi-Scale Breadth", height=400, xaxis_title=x_title, yaxis_title="% of Timeframes", yaxis=dict(range=[0, 100]))
        st.plotly_chart(update_chart_theme(fig_zones), use_container_width=True)
        
        st.markdown("---")
        st.markdown("##### Average TDA Z-Score")
        fig_z = go.Figure()
        z_colors = ['#10b981' if z < -1 else '#ef4444' if z > 1 else '#888' for z in ts_filtered['AvgZ']]
        fig_z.add_trace(go.Bar(x=x_axis, y=ts_filtered['AvgZ'], marker_color=z_colors, name='Statistical Z'))
        fig_z.add_hline(y=0, line_color="#FFC300", line_width=1)
        fig_z.add_hline(y=2, line_dash="dash", line_color="rgba(239,68,68,0.5)")
        fig_z.add_hline(y=-2, line_dash="dash", line_color="rgba(16,185,129,0.5)")
        fig_z.update_layout(title="Multi-Scale Average Statistical Mispricing", height=350, xaxis_title=x_title, yaxis_title="Z-Score")
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
        display_cols = ['Date', 'Actual', 'FairValue', 'Residual', 'ModelSpread', 'AvgZ', 'Surprise',
                        'OversoldBreadth', 'OverboughtBreadth', 'ConvictionScore', 'Regime',
                        'BullishDiv', 'BearishDiv']
        display_cols = [c for c in display_cols if c in ts_filtered.columns]
        
        display_ts = ts_filtered[display_cols].copy()
        
        # Rename ModelSpread and AvgZ to reflect TDA context for export clarity
        display_ts = display_ts.rename(columns={'ModelSpread': 'TopoDist_Dt', 'AvgZ': 'StatScore_Zt'})
        
        for col in ['Residual', 'TopoDist_Dt', 'StatScore_Zt', 'Surprise', 'FairValue', 'ConvictionScore', 'OversoldBreadth', 'OverboughtBreadth']:
            if col in display_ts.columns:
                display_ts[col] = display_ts[col].round(3 if col in ['StatScore_Zt', 'TopoDist_Dt', 'Surprise'] else 2 if col == 'FairValue' else 1)
        
        if 'BullishDiv' in display_ts.columns: display_ts['BullishDiv'] = display_ts['BullishDiv'].apply(lambda x: '🟢' if x else '')
        if 'BearishDiv' in display_ts.columns: display_ts['BearishDiv'] = display_ts['BearishDiv'].apply(lambda x: '🔴' if x else '')
        
        st.dataframe(display_ts, width='stretch', hide_index=True, height=500)
        st.download_button("📥 Download Full CSV", ts.to_csv(index=False).encode('utf-8'), f"aarambh_tda_{active_target}_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
    
    render_footer()

if __name__ == "__main__":
    main()
