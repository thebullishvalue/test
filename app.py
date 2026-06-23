import streamlit as st
import json
import random
import warnings
import time
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import optuna
import yfinance as yf

warnings.filterwarnings("ignore")

# =============================================================================
# STREAMLIT UI SETUP & AESTHETICS
# =============================================================================
# Terminal-style institutional aesthetic configuration
st.set_page_config(page_title="HCI-Z Optimizer Terminal", page_icon="⚡", layout="wide")

st.title("⚡ HCI-Z Robust Parameter Optimization")
st.markdown("Dynamic Volatility Grid Intelligence — Adaptive Z-Score Edition.")

# Sidebar Configuration
st.sidebar.header("Data Environment")
DATA_PERIOD = st.sidebar.selectbox("Data Period", ["2y", "5y", "10y"], index=1)
MIN_TRADING_DAYS = 400
N_TRIALS = st.sidebar.number_input("Optuna Trials", min_value=50, max_value=2000, value=300, step=50)
TARGET_ASSETS = st.sidebar.number_input("Target Asset Count", min_value=10, max_value=150, value=100, step=10)

st.sidebar.header("Execution Assumptions")
FEE_PCT = st.sidebar.number_input("Fee % (Round Trip)", value=0.15, format="%.2f") / 100
SLIPPAGE_PCT = 0.0005
INITIAL_CAPITAL = 100_000

# Objective Weights
st.sidebar.header("Optimization Objectives")
WEIGHT_SHARPE = st.sidebar.slider("Sharpe Weight", 0.0, 1.0, 0.45)
WEIGHT_PROFIT_FACTOR = st.sidebar.slider("Profit Factor Weight", 0.0, 1.0, 0.25)
WEIGHT_CALMAR = st.sidebar.slider("Calmar Weight", 0.0, 1.0, 0.15)
WEIGHT_CONSISTENCY = st.sidebar.slider("Consistency Weight", 0.0, 1.0, 0.15)

# =============================================================================
# DIVERSE TICKER POOLS
# =============================================================================
TICKER_POOLS = {
    "indian_large": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS", "LT.NS", "AXISBANK.NS", "BAJFINANCE.NS", "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "NESTLEIND.NS", "POWERGRID.NS", "NTPC.NS"],
    "indian_mid": ["PIIND.NS", "DIXON.NS", "PERSISTENT.NS", "MPHASIS.NS", "COFORGE.NS", "LALPATHLAB.NS", "ALKEM.NS", "ASTRAL.NS", "ESCORTS.NS", "M&M.NS", "TATAPOWER.NS", "ADANIPORTS.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS"],
    "us_large": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "V", "PG", "JNJ", "XOM", "UNH", "MA", "HD", "CVX", "PFE"],
    "us_tech_growth": ["AMD", "AVGO", "ADBE", "CRM", "NOW", "SNOW", "PLTR", "PANW", "CRWD"],
    "global_indices_etf": ["SPY", "QQQ", "IWM", "VTI", "EFA", "EEM", "GLD", "SLV", "TLT", "^NSEI", "^BSESN", "^GSPC", "^IXIC"],
    "crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD", "AVAX-USD", "DOGE-USD"],
    "commodities": ["GC=F", "SI=F", "CL=F", "NG=F", "HG=F", "ZC=F", "ZS=F", "KC=F", "USO", "GLD", "SLV", "DBA", "DBC"]
}

def get_diverse_tickers(total: int = 100, seed: int = 42) -> List[str]:
    """Sample balanced across categories. Capped to available unique tickers."""
    random.seed(seed)
    selected = []
    per_category = max(3, total // len(TICKER_POOLS))
    
    for category, tickers in TICKER_POOLS.items():
        n = min(per_category, len(tickers))
        selected.extend(random.sample(tickers, n))
    
    all_tickers = list(set([t for pool in TICKER_POOLS.values() for t in pool]))
    safe_total = min(total, len(all_tickers)) 
    
    while len(selected) < safe_total:
        extra = random.choice(all_tickers)
        if extra not in selected:
            selected.append(extra)
    
    return selected[:safe_total]

# =============================================================================
# DATA FETCHING ENGINE
# =============================================================================
@st.cache_data(ttl="1d", show_spinner=False)
def download_and_cache(tickers: List[str], period: str) -> Dict[str, pd.DataFrame]:
    """Downloads data, flattens MultiIndex (yfinance fix), stores in RAM."""
    data_dict = {}
    
    for ticker in tickers:
        try:
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True, threads=False)
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            if not df.empty and len(df) >= MIN_TRADING_DAYS:
                df = df.rename(columns=str.title)
                if "Adj Close" in df.columns:
                    df = df.drop(columns=["Adj Close"])
                df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
                df.index = pd.to_datetime(df.index).tz_localize(None)
                data_dict[ticker] = df
        except Exception:
            pass
            
    return data_dict

# =============================================================================
# HCI-Z INDICATOR ENGINE (HIGH-SPEED VECTORIZATION)
# =============================================================================
def compute_hci_z(df, z_len=20, z_grid=0.5, max_cap=5.0, gravity=0.95, look=50, sig_len=20):
    """
    Lightning-fast Python port of the Hemrek Count Index [Z-Score].
    Uses numpy convolution for WMA and efficient iteration for recursive gravity.
    """
    out = df.copy()
    close = out["Close"].values
    
    # 1. Bar Returns
    ret_pct = np.zeros_like(close)
    ret_pct[1:] = (close[1:] - close[:-1]) / close[:-1] * 100.0
    
    # 2. Z-Score Engine (Volatility Window)
    ret_series = pd.Series(ret_pct)
    std_ret = ret_series.rolling(window=z_len, min_periods=1).std(ddof=0).values
    
    z_score = np.zeros_like(ret_pct)
    valid = std_ret > 0
    z_score[valid] = ret_pct[valid] / std_ret[valid]
    
    # 3. Capped Step Logic
    raw_step = np.floor(np.abs(z_score) / z_grid) * z_grid
    step_mag = np.minimum(raw_step, max_cap)
    
    current_step = np.zeros_like(z_score)
    current_step[z_score > 0] = step_mag[z_score > 0]
    current_step[z_score < 0] = -step_mag[z_score < 0]
    
    # 4. Apply Gravity & New Steps (Numpy Iteration)
    count = np.zeros_like(close)
    for i in range(1, len(count)):
        if current_step[i] != 0.0:
            count[i] = count[i-1] + current_step[i]
        else:
            count[i] = count[i-1] * gravity
            
    # 5. Detrend against Baseline Window
    count_series = pd.Series(count, index=out.index)
    sma_count = count_series.rolling(window=look, min_periods=1).mean()
    count_index = count_series - sma_count
    
    # 6. Ultra-fast WMA Signal Line (via Convolution)
    if sig_len > 1:
        weights = np.arange(1, sig_len + 1)
        weights = weights / weights.sum()
        # Convolve and pad the start with NaNs to match pandas rolling behavior
        sig_arr = np.convolve(count_index.values, weights[::-1], mode='full')[:len(count_index)]
        sig_arr[:sig_len-1] = np.nan
        count_signal = pd.Series(sig_arr, index=out.index)
    else:
        count_signal = count_index.copy()
    
    # 7. Crosses & Positioning
    cross_up = (count_index > count_signal) & (count_index.shift(1) <= count_signal.shift(1))
    cross_down = (count_index < count_signal) & (count_index.shift(1) >= count_signal.shift(1))
    
    position = pd.Series(0, index=out.index)
    position[cross_up.fillna(False)] = 1
    position[cross_down.fillna(False)] = 0
    out["position"] = position.ffill().fillna(0)
    
    return out

# =============================================================================
# VECTORIZED BACKTESTING
# =============================================================================
def backtest_asset(df, fee, slippage):
    if "position" not in df.columns or df["position"].sum() == 0:
        return {"sharpe": -10, "profit_factor": 0, "calmar": -10, "max_dd": 100, "trades": 0, "total_return": -99}
    
    ret = df["Close"].pct_change().fillna(0)
    pos_change = df["position"].diff().abs()
    cost = pos_change * (fee + slippage)
    
    strategy_ret = df["position"].shift(1) * ret - cost
    equity = (1 + strategy_ret).cumprod() * INITIAL_CAPITAL
    
    total_return = (equity.iloc[-1] / INITIAL_CAPITAL - 1) * 100
    daily_ret = strategy_ret
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else 0.0
    
    peak = equity.cummax()
    max_dd = ((equity - peak) / peak * 100).min()
    
    gains = daily_ret[daily_ret > 0].sum()
    losses = -daily_ret[daily_ret < 0].sum()
    profit_factor = gains / losses if losses > 0 else 10.0
    
    calmar = (total_return / 100) / (abs(max_dd) / 100) if max_dd != 0 else 0
    trades = int(pos_change.sum() / 2)
    
    return {
        "sharpe": round(sharpe, 3), "profit_factor": round(profit_factor, 3),
        "calmar": round(calmar, 3), "max_dd": round(max_dd, 2),
        "trades": trades, "total_return": round(total_return, 2)
    }

# =============================================================================
# OPTUNA OBJECTIVE
# =============================================================================
def objective(trial, data_dict):
    params = {
        "z_len": trial.suggest_int("z_len", 10, 50, step=2),
        "z_grid": trial.suggest_float("z_grid", 0.1, 1.5, step=0.1),
        "max_cap": trial.suggest_float("max_cap", 2.0, 8.0, step=0.5),
        "gravity": trial.suggest_float("gravity", 0.85, 0.99, step=0.01),
        "look": trial.suggest_int("look", 20, 100, step=5),
        "sig_len": trial.suggest_int("sig_len", 5, 40, step=2),
    }
    
    scores, pf_list, sharpe_list = [], [], []
    
    for ticker, df in data_dict.items():
        try:
            hci_df = compute_hci_z(df, **params)
            metrics = backtest_asset(hci_df, FEE_PCT, SLIPPAGE_PCT)
            if metrics["trades"] < 8 or metrics["sharpe"] < -1.5: continue
            
            sharpe_list.append(metrics["sharpe"])
            pf_list.append(metrics["profit_factor"])
            scores.append((WEIGHT_SHARPE * metrics["sharpe"]) + (WEIGHT_PROFIT_FACTOR * min(metrics["profit_factor"], 4.0)) + (WEIGHT_CALMAR * min(metrics["calmar"], 3.0)))
        except Exception: continue
    
    if len(scores) < max(5, len(data_dict) // 3): return -999
    
    mean_score = np.mean(scores)
    consistency = 1.0 / (1.0 + np.std(sharpe_list)) if len(sharpe_list) > 1 else 0.5
    final_objective = (mean_score * 0.85) + (WEIGHT_CONSISTENCY * consistency * 10)
    
    trial.set_user_attr("mean_sharpe", round(np.mean(sharpe_list), 3))
    trial.set_user_attr("median_pf", round(np.median(pf_list), 3))
    return final_objective

# =============================================================================
# MAIN APP EXECUTION
# =============================================================================
if st.button("🚀 Initialize Optimization Sequence"):
    start_time = time.time()
    
    with st.status("Engaging Engine...", expanded=True) as status:
        st.write("1. Assembling dynamic market breadth...")
        tickers = get_diverse_tickers(total=int(TARGET_ASSETS))
        
        st.write(f"2. Fetching institutional data stream ({len(tickers)} arrays)...")
        data_dict = download_and_cache(tickers, DATA_PERIOD)
        st.write(f"   *Live arrays acquired: {len(data_dict)}*")
        
        if len(data_dict) < 10:
            st.error("Data stream insufficient. Expand lookback window.")
            st.stop()

        st.write(f"3. Deploying Optuna Swarm ({N_TRIALS} epochs)...")
        opt_tickers = list(data_dict.keys())[:min(35, len(data_dict))]
        opt_data = {k: data_dict[k] for k in opt_tickers}
        
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        progress_bar = st.progress(0)
        
        for i in range(N_TRIALS):
            study.optimize(lambda t: objective(t, opt_data), n_trials=1, n_jobs=1)
            progress_bar.progress((i + 1) / N_TRIALS)

        status.update(label="Analysis Complete", state="complete", expanded=False)
        
    # --- Results Rendering ---
    st.success(f"✅ Sequence complete. Uptime: {(time.time() - start_time) / 60:.1f}m")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏆 Optimal Core Parameters")
        st.json(study.best_params)
        
    with col2:
        st.subheader("📊 Primary Metrics")
        st.metric("Objective Score", f"{study.best_value:.4f}")
        st.metric("Mean Sharpe", study.best_trial.user_attrs.get("mean_sharpe", "N/A"))
        st.metric("Median Profit Factor", study.best_trial.user_attrs.get("median_pf", "N/A"))

    # --- Breadth Validation ---
    st.subheader("🌍 Multi-Asset Validation Output")
    with st.spinner("Processing total matrix..."):
        val_results = []
        for ticker, df in data_dict.items():
            hci_df = compute_hci_z(df, **study.best_params)
            metrics = backtest_asset(hci_df, FEE_PCT, SLIPPAGE_PCT)
            metrics["ticker"] = ticker
            val_results.append(metrics)
            
        val_df = pd.DataFrame(val_results).set_index("ticker")
        st.dataframe(val_df.style.highlight_max(axis=0, color='#0ea5e9'))
        
    st.download_button("Export Config [JSON]", data=json.dumps(study.best_params, indent=2), file_name="hciz_params.json", mime="application/json")
    st.download_button("Export Matrix [CSV]", data=val_df.to_csv(), file_name="hciz_validation.csv", mime="text/csv")
else:
    st.info("Awaiting execution trigger.")
