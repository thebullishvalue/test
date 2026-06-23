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
# STREAMLIT UI SETUP
# =============================================================================
st.set_page_config(page_title="CSSI Optimizer", page_icon="📈", layout="wide")
st.title("📈 CSSI Robust Parameter Optimization")
st.markdown("Finds robust default values for the Conviction Streak Synergy Index (CSSI) across diverse asset classes.")

# Sidebar Configuration
st.sidebar.header("Configuration")
DATA_PERIOD = st.sidebar.selectbox("Data Period", ["2y", "5y", "10y"], index=1)
MIN_TRADING_DAYS = 400
N_TRIALS = st.sidebar.number_input("Optuna Trials", min_value=50, max_value=2000, value=300, step=50)
TARGET_ASSETS = st.sidebar.number_input("Target Asset Count", min_value=10, max_value=150, value=100, step=10)

st.sidebar.header("Backtest Settings")
FEE_PCT = st.sidebar.number_input("Fee % (Round Trip)", value=0.15, format="%.2f") / 100
SLIPPAGE_PCT = 0.0005
INITIAL_CAPITAL = 100_000

# Objective Weights
st.sidebar.header("Objective Weights")
WEIGHT_SHARPE = st.sidebar.slider("Sharpe Weight", 0.0, 1.0, 0.45)
WEIGHT_PROFIT_FACTOR = st.sidebar.slider("Profit Factor Weight", 0.0, 1.0, 0.25)
WEIGHT_CALMAR = st.sidebar.slider("Calmar Weight", 0.0, 1.0, 0.15)
WEIGHT_CONSISTENCY = st.sidebar.slider("Consistency Weight", 0.0, 1.0, 0.15)

SIG_TYPE = "sma"

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
    """Sample balanced across categories. BUG FIX: Capped to available unique tickers."""
    random.seed(seed)
    selected = []
    per_category = max(3, total // len(TICKER_POOLS))
    
    for category, tickers in TICKER_POOLS.items():
        n = min(per_category, len(tickers))
        selected.extend(random.sample(tickers, n))
    
    all_tickers = list(set([t for pool in TICKER_POOLS.values() for t in pool]))
    safe_total = min(total, len(all_tickers)) # Prevents infinite loops!
    
    while len(selected) < safe_total:
        extra = random.choice(all_tickers)
        if extra not in selected:
            selected.append(extra)
    
    return selected[:safe_total]


# =============================================================================
# DATA FETCHING (OPTIMIZED FOR STREAMLIT)
# =============================================================================
@st.cache_data(ttl="1d", show_spinner=False)
def download_and_cache(tickers: List[str], period: str) -> Dict[str, pd.DataFrame]:
    """Downloads data and stores it in Streamlit's RAM. Refreshes once a day."""
    data_dict = {}
    
    for ticker in tickers:
        try:
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True, threads=False)
            
            # --- YFINANCE FIX: Flatten MultiIndex columns ---
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            if not df.empty and len(df) >= MIN_TRADING_DAYS:
                df = df.rename(columns=str.title)
                if "Adj Close" in df.columns:
                    df = df.drop(columns=["Adj Close"])
                
                # Keep only the columns we need, ensuring they are 1D
                df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
                df.index = pd.to_datetime(df.index).tz_localize(None)
                data_dict[ticker] = df
        except Exception:
            pass
            
    return data_dict

# =============================================================================
# CSSI & BACKTESTING LOGIC
# =============================================================================
def compute_cssi(df, thres=1.0, look=50, sig_len=20, kvo_fast=34, kvo_slow=55, kvo_sig_len=13, gamma=0.85, norm_len=50):
    out = df.copy()
    close = out["Close"]
    hlc3 = (out["High"] + out["Low"] + close) / 3.0
    
    ret_pct = close.pct_change() * 100.0
    s = pd.Series(0.0, index=close.index)
    s[ret_pct > thres] = 1.0
    s[ret_pct < -thres] = -1.0
    
    delta_h = hlc3.diff()
    sv = np.where(delta_h >= 0, out["Volume"], -out["Volume"])
    sv = pd.Series(sv, index=close.index).fillna(0)
    
    kvo = sv.ewm(span=kvo_fast, adjust=False).mean() - sv.ewm(span=kvo_slow, adjust=False).mean()
    ksig = kvo.ewm(span=kvo_sig_len, adjust=False).mean()
    k_hist = kvo - ksig
    
    k_std = k_hist.rolling(norm_len).std()
    k_norm = k_hist / k_std.replace(0, np.nan)
    k_norm = k_norm.fillna(0)
    
    align_mod = s * k_norm
    eff_s = s * (1.0 + gamma * align_mod)
    eff_s = eff_s.clip(-2.0, 2.0)
    
    aug_count = eff_s.cumsum()
    aug_index = aug_count - aug_count.rolling(look, min_periods=look//2).mean()
    aug_signal = aug_index.rolling(sig_len, min_periods=sig_len//2).mean()
    
    cross_up = (aug_index > aug_signal) & (aug_index.shift(1) <= aug_signal.shift(1))
    cross_down = (aug_index < aug_signal) & (aug_index.shift(1) >= aug_signal.shift(1))
    
    position = pd.Series(0, index=out.index)
    position[cross_up.fillna(False)] = 1
    position[cross_down.fillna(False)] = 0
    out["position"] = position.ffill().fillna(0)
    
    return out

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

def objective(trial, data_dict):
    params = {
        "thres": trial.suggest_float("thres", 0.5, 2.2, step=0.1),
        "look": trial.suggest_int("look", 25, 110, step=5),
        "sig_len": trial.suggest_int("sig_len", 10, 35, step=2),
        "kvo_fast": trial.suggest_int("kvo_fast", 22, 48, step=2),
        "kvo_slow": trial.suggest_int("kvo_slow", 42, 75, step=3),
        "kvo_sig_len": trial.suggest_int("kvo_sig_len", 9, 22, step=1),
        "gamma": trial.suggest_float("gamma", 0.45, 1.6, step=0.05),
        "norm_len": trial.suggest_int("norm_len", 25, 100, step=5),
    }
    
    scores, pf_list, sharpe_list = [], [], []
    
    for ticker, df in data_dict.items():
        try:
            cssi_df = compute_cssi(df, **params)
            metrics = backtest_asset(cssi_df, FEE_PCT, SLIPPAGE_PCT)
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
if st.button("🚀 Run Optimization"):
    start_time = time.time()
    
    with st.status("Initializing Optimization Process...", expanded=True) as status:
        st.write("1. Building balanced asset universe...")
        tickers = get_diverse_tickers(total=int(TARGET_ASSETS))
        
        st.write(f"2. Fetching & Caching data for {len(tickers)} assets...")
        data_dict = download_and_cache(tickers, DATA_PERIOD)
        st.write(f"   *Loaded {len(data_dict)} assets successfully.*")
        
        if len(data_dict) < 10:
            st.error("Not enough valid assets fetched. Try a shorter timeframe.")
            st.stop()

        st.write(f"3. Starting Optuna Search ({N_TRIALS} trials)...")
        # Use only a subset for speed in Streamlit
        opt_tickers = list(data_dict.keys())[:min(35, len(data_dict))]
        opt_data = {k: data_dict[k] for k in opt_tickers}
        
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        
        # Progress bar for Streamlit
        progress_bar = st.progress(0)
        
        for i in range(N_TRIALS):
            study.optimize(lambda t: objective(t, opt_data), n_trials=1, n_jobs=1) # n_jobs=1 is safer for cloud memory
            progress_bar.progress((i + 1) / N_TRIALS)

        status.update(label="Optimization Complete!", state="complete", expanded=False)
        
    # --- Results UI ---
    st.success(f"✅ Optimization finished in {(time.time() - start_time) / 60:.1f} minutes!")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏆 Best Parameters")
        st.json(study.best_params)
        
    with col2:
        st.subheader("📊 Best Trial Metrics")
        st.metric("Objective Score", f"{study.best_value:.4f}")
        st.metric("Mean Sharpe", study.best_trial.user_attrs.get("mean_sharpe", "N/A"))
        st.metric("Median Profit Factor", study.best_trial.user_attrs.get("median_pf", "N/A"))

    # --- Validation Phase ---
    st.subheader("🌍 Validating on Full Universe")
    with st.spinner("Running full backtest with best parameters..."):
        val_results = []
        for ticker, df in data_dict.items():
            cssi_df = compute_cssi(df, **study.best_params)
            metrics = backtest_asset(cssi_df, FEE_PCT, SLIPPAGE_PCT)
            metrics["ticker"] = ticker
            val_results.append(metrics)
            
        val_df = pd.DataFrame(val_results).set_index("ticker")
        st.dataframe(val_df.style.highlight_max(axis=0, color='darkgreen'))
        
    # Download Buttons
    st.download_button("Download Parameters (JSON)", data=json.dumps(study.best_params, indent=2), file_name="cssi_params.json", mime="application/json")
    st.download_button("Download Validation Results (CSV)", data=val_df.to_csv(), file_name="validation_results.csv", mime="text/csv")
else:
    st.info("Adjust the settings in the sidebar and click 'Run Optimization' to begin.")
