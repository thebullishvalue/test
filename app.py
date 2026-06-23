#!/usr/bin/env python3
"""
HCI-Z Parameter Optimization Framework
======================================
Fetches & caches 100+ diverse instruments (Indian equity, US equity, crypto, commodities)
then runs an intelligent parameter study (Optuna) to find robust default values
for the Hemrek Count Index [Z-Score Edition] (HCI-Z).

Author: Grok (built for @BullishValue) / Refined by Gemini
Date: June 2026
"""

import os
import json
import random
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import time
import optuna

warnings.filterwarnings("ignore")

# =============================================================================
# LOGGING SETUP (Detailed terminal output)
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("HCI-Z-Optimizer")

optuna_logger = logging.getLogger("optuna")
optuna_logger.setLevel(logging.WARNING)

# =============================================================================
# CONFIGURATION
# =============================================================================
CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(exist_ok=True)

DATA_PERIOD = "5y"          
MIN_TRADING_DAYS = 400      

# Optimization settings
N_TRIALS = 800              
N_JOBS = 4                  
STUDY_NAME = "hci_z_robust_v1"

# Backtest settings
FEE_PCT = 0.0015            
SLIPPAGE_PCT = 0.0005
INITIAL_CAPITAL = 100_000

# Objective weights
WEIGHT_SHARPE = 0.45
WEIGHT_PROFIT_FACTOR = 0.25
WEIGHT_CALMAR = 0.15
WEIGHT_CONSISTENCY = 0.15   

# =============================================================================
# DIVERSE TICKER POOLS
# =============================================================================
TICKER_POOLS = {
    "indian_large": [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
        "LT.NS", "AXISBANK.NS", "BAJFINANCE.NS", "MARUTI.NS", "SUNPHARMA.NS"
    ],
    "indian_mid": [
        "PIIND.NS", "DIXON.NS", "PERSISTENT.NS", "MPHASIS.NS", "COFORGE.NS",
        "LALPATHLAB.NS", "ALKEM.NS", "ASTRAL.NS", "ESCORTS.NS", "M&M.NS"
    ],
    "us_large": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
        "JPM", "V", "PG", "JNJ", "XOM", "UNH", "MA"
    ],
    "us_tech_growth": [
        "AMD", "AVGO", "ADBE", "CRM", "NOW", "SNOW", "PLTR", "PANW", "CRWD"
    ],
    "global_indices_etf": [
        "SPY", "QQQ", "IWM", "VTI", "EFA", "EEM", "GLD", "SLV", "TLT",
        "^NSEI", "^BSESN"
    ],
    "crypto": [
        "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD"
    ],
    "commodities": [
        "GC=F", "SI=F", "CL=F", "NG=F", "HG=F", "USO", "DBC"
    ]
}

def get_diverse_tickers(total: int = 100, seed: int = 42) -> List[str]:
    """Sample balanced across categories for a robust study."""
    random.seed(seed)
    selected = []
    per_category = max(3, total // len(TICKER_POOLS))
    
    for tickers in TICKER_POOLS.values():
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
# DATA FETCHING + CACHING
# =============================================================================
def download_and_cache(tickers: List[str], period: str = DATA_PERIOD) -> Dict[str, pd.DataFrame]:
    import yfinance as yf
    
    data_dict = {}
    failed = []
    
    for ticker in tqdm(tickers, desc="Fetching & caching data", unit="ticker"):
        cache_file = CACHE_DIR / f"{ticker.replace('=', '_').replace('^', '')}.parquet"
        
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                if len(df) >= MIN_TRADING_DAYS:
                    data_dict[ticker] = df
                    continue
            except Exception:
                pass 
        
        try:
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True, threads=False)
            if df.empty or len(df) < MIN_TRADING_DAYS:
                failed.append(ticker)
                continue
            
            df = df.rename(columns=str.title)
            if "Adj Close" in df.columns:
                df = df.drop(columns=["Adj Close"])
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            
            df.to_parquet(cache_file, compression="zstd")
            data_dict[ticker] = df
            
        except Exception as e:
            failed.append(ticker)
            logger.debug(f"Failed {ticker}: {e}")
            
    return data_dict


# =============================================================================
# HCI-Z INDICATOR ENGINE
# =============================================================================
def compute_hci_z(
    df: pd.DataFrame,
    z_len: int = 20,
    z_grid: float = 0.5,
    max_cap: float = 5.0,
    gravity: float = 0.95,
    look: int = 50,
    sig_len: int = 20,
) -> pd.DataFrame:
    """
    Python implementation of the Hemrek Count Index [Z-Score Edition].
    """
    out = df.copy()
    close = out["Close"].values
    
    # 1. Per-bar return (%)
    ret_pct = np.zeros_like(close)
    ret_pct[1:] = (close[1:] - close[:-1]) / close[:-1] * 100.0
    
    # 2. Z-Score calculation (Zero-Mean anchored)
    s_ret = pd.Series(ret_pct)
    std_ret = s_ret.rolling(z_len).std().values
    
    z_score = np.zeros_like(ret_pct)
    mask = (std_ret != 0) & (~np.isnan(std_ret))
    z_score[mask] = ret_pct[mask] / std_ret[mask]
    
    # 3. Capped Step Logic
    raw_step = np.floor(np.abs(z_score) / z_grid) * z_grid
    step_mag = np.minimum(raw_step, max_cap)
    
    current_step = np.zeros_like(z_score)
    current_step[z_score > 0] = step_mag[z_score > 0]
    current_step[z_score < 0] = -step_mag[z_score < 0]
    
    # 4. Fast path-dependent Loop for Count & Gravity
    counts = np.zeros_like(current_step)
    curr = 0.0
    for i in range(len(current_step)):
        step = current_step[i]
        if step != 0.0:
            curr = curr + step
        else:
            curr = curr * gravity
        counts[i] = curr
        
    s_counts = pd.Series(counts, index=out.index)
    
    # 5. Detrend against baseline
    baseline = s_counts.rolling(look, min_periods=look//2).mean()
    count_index = s_counts - baseline
    
    # 6. Signal Line (Using EMA for fast calculation proxy of WMA)
    count_signal = count_index.ewm(span=sig_len, adjust=False).mean()
    
    # 7. Generate Signals
    out["count_index"] = count_index
    out["count_signal"] = count_signal
    
    cross_up = (count_index > count_signal) & (count_index.shift(1) <= count_signal.shift(1))
    cross_down = (count_index < count_signal) & (count_index.shift(1) >= count_signal.shift(1))
    
    position = pd.Series(0, index=out.index)
    position[cross_up] = 1
    position[cross_down] = 0
    out["position"] = position.ffill().fillna(0)
    
    return out


# =============================================================================
# VECTORIZED BACKTEST
# =============================================================================
def backtest_asset(df: pd.DataFrame, fee: float = FEE_PCT, slippage: float = SLIPPAGE_PCT) -> Dict[str, float]:
    if "position" not in df.columns or df["position"].sum() == 0:
        return {"sharpe": -10, "profit_factor": 0, "calmar": -10, "max_dd": 100, "trades": 0, "total_return": -99}
    
    close = df["Close"]
    ret = close.pct_change().fillna(0)
    
    pos_change = df["position"].diff().abs()
    cost = pos_change * (fee + slippage)
    
    strategy_ret = df["position"].shift(1) * ret - cost
    equity = (1 + strategy_ret).cumprod() * INITIAL_CAPITAL
    
    total_return = (equity.iloc[-1] / INITIAL_CAPITAL - 1) * 100
    
    daily_ret = strategy_ret
    sharpe = 0.0 if daily_ret.std() == 0 else (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)
    
    peak = equity.cummax()
    max_dd = ((equity - peak) / peak * 100).min()
    
    gains = daily_ret[daily_ret > 0].sum()
    losses = -daily_ret[daily_ret < 0].sum()
    profit_factor = gains / losses if losses > 0 else 10.0
    
    calmar = (total_return / 100) / (abs(max_dd) / 100) if max_dd != 0 else 0
    trades = int(pos_change.sum() / 2)
    
    return {
        "sharpe": round(sharpe, 3),
        "profit_factor": round(profit_factor, 3),
        "calmar": round(calmar, 3),
        "max_dd": round(max_dd, 2),
        "trades": trades,
        "total_return": round(total_return, 2)
    }


# =============================================================================
# OPTUNA OBJECTIVE
# =============================================================================
def log_best_callback(study, trial):
    if trial.number % 50 == 0 or trial.number == study.best_trial.number:
        logger.info(
            f"Trial {trial.number:4d} | Best Score: {study.best_value:.4f} | "
            f"Mean Sharpe: {study.best_trial.user_attrs.get('mean_sharpe', 'N/A')} | "
            f"Assets tested: {study.best_trial.user_attrs.get('n_assets_tested', 'N/A')}"
        )

def objective(trial, data_dict: Dict[str, pd.DataFrame]) -> float:
    # HCI-Z specific hyperparameter space
    params = {
        "z_len": trial.suggest_int("z_len", 10, 60, step=5),
        "z_grid": trial.suggest_float("z_grid", 0.1, 1.5, step=0.1),
        "max_cap": trial.suggest_float("max_cap", 2.0, 10.0, step=0.5),
        "gravity": trial.suggest_float("gravity", 0.85, 1.0, step=0.01),
        "look": trial.suggest_int("look", 20, 100, step=5),
        "sig_len": trial.suggest_int("sig_len", 10, 40, step=2),
    }
    
    scores, pf_list, sharpe_list = [], [], []
    
    for ticker, df in data_dict.items():
        try:
            hci_df = compute_hci_z(df, **params)
            metrics = backtest_asset(hci_df)
            
            if metrics["trades"] < 8 or metrics["sharpe"] < -1.5:
                continue
                
            sharpe_list.append(metrics["sharpe"])
            pf_list.append(metrics["profit_factor"])
            
            score = (
                WEIGHT_SHARPE * metrics["sharpe"] +
                WEIGHT_PROFIT_FACTOR * min(metrics["profit_factor"], 4.0) +
                WEIGHT_CALMAR * min(metrics["calmar"], 3.0)
            )
            scores.append(score)
            
        except Exception:
            continue
    
    if len(scores) < max(5, len(data_dict) // 3):
        return -999 
    
    mean_score = np.mean(scores)
    consistency = 1.0 / (1.0 + np.std(sharpe_list)) if len(sharpe_list) > 1 else 0.5
    
    final_objective = (mean_score * 0.85) + (WEIGHT_CONSISTENCY * consistency * 10)
    
    trial.set_user_attr("mean_sharpe", round(np.mean(sharpe_list), 3))
    trial.set_user_attr("median_pf", round(np.median(pf_list), 3))
    trial.set_user_attr("n_assets_tested", len(scores))
    
    return final_objective


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    start_time = time.time()
    logger.info("=" * 72)
    logger.info("HCI-Z ROBUST PARAMETER OPTIMIZATION  •  Z-Score Edition")
    logger.info("=" * 72)

    tickers = get_diverse_tickers(total=100)
    data_dict = download_and_cache(tickers)

    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=50)
    )

    opt_tickers = list(data_dict.keys())[:35]
    opt_data = {k: data_dict[k] for k in opt_tickers}

    logger.info(f"\n[3/5] Starting Optuna study ({N_TRIALS} trials)...")
    study.optimize(
        lambda trial: objective(trial, opt_data),
        n_trials=N_TRIALS,
        n_jobs=N_JOBS,
        show_progress_bar=True,
        callbacks=[log_best_callback]
    )

    best_params = study.best_params
    logger.info("\n" + "=" * 72)
    logger.info("🏆 BEST HCI-Z PARAMETERS (Recommended Defaults)")
    logger.info("=" * 72)
    for k, v in best_params.items():
        logger.info(f"   {k:15s} = {v}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("optimization_results")
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / f"best_hciz_params_{timestamp}.json", "w") as f:
        json.dump(best_params, f, indent=2)

    logger.info("\n[5/5] Validating on FULL cached universe...")
    validation_results = []
    for ticker, df in tqdm(data_dict.items(), desc="Validating"):
        try:
            hci_df = compute_hci_z(df, **best_params)
            metrics = backtest_asset(hci_df)
            metrics["ticker"] = ticker
            validation_results.append(metrics)
        except Exception:
            pass

    val_df = pd.DataFrame(validation_results)
    summary = val_df[["sharpe", "profit_factor", "calmar", "max_dd", "trades"]].describe().round(2)
    
    logger.info("\nValidation Summary:")
    for line in summary.to_string().split("\n"):
        logger.info("   " + line)

    logger.info(f"\n✅ RUN COMPLETE in {(time.time() - start_time)/60:.1f} minutes")

if __name__ == "__main__":
    main()
