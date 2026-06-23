#!/usr/bin/env python3
"""
CSSI Parameter Optimization Framework
=====================================
Fetches & caches 100+ diverse instruments (Indian equity, US equity, crypto, commodities)
then runs an intelligent parameter study (Optuna) to find robust default values
for the Conviction Streak Synergy Index (CSSI) that maximize risk-adjusted
profitability + signal quality across asset classes.

Author: Grok (built for @BullishValue)
Date: June 2026
"""

import os
import json
import random
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

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
logger = logging.getLogger("CSSI-Optimizer")

# Silence Optuna's own logger unless we want it
optuna_logger = logging.getLogger("optuna")
optuna_logger.setLevel(logging.WARNING)

# =============================================================================
# CONFIGURATION
# =============================================================================
CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(exist_ok=True)

# How much history to fetch
DATA_PERIOD = "5y"          # yfinance period
MIN_TRADING_DAYS = 400      # Skip assets with too little data

# Optimization settings
N_TRIALS = 800              # Optuna trials (increase to 1500+ for deeper search)
N_JOBS = 4                  # Parallel trials (set to 1 if memory issues)
STUDY_NAME = "cssi_robust_v1"

# Backtest settings
FEE_PCT = 0.0015            # 0.15% round-trip (conservative for India + slippage)
SLIPPAGE_PCT = 0.0005
INITIAL_CAPITAL = 100_000

# Objective weights (you can tune these)
WEIGHT_SHARPE = 0.45
WEIGHT_PROFIT_FACTOR = 0.25
WEIGHT_CALMAR = 0.15
WEIGHT_CONSISTENCY = 0.15   # penalizes high variance across assets

# Fixed params (reduce search space)
SIG_TYPE = "sma"
ROC_LEN = 3

# =============================================================================
# DIVERSE TICKER POOLS (expand as needed)
# =============================================================================
TICKER_POOLS = {
    "indian_large": [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
        "LT.NS", "AXISBANK.NS", "BAJFINANCE.NS", "MARUTI.NS", "SUNPHARMA.NS",
        "TITAN.NS", "ULTRACEMCO.NS", "NESTLEIND.NS", "POWERGRID.NS", "NTPC.NS"
    ],
    "indian_mid": [
        "PIIND.NS", "DIXON.NS", "PERSISTENT.NS", "MPHASIS.NS", "COFORGE.NS",
        "LALPATHLAB.NS", "ALKEM.NS", "ASTRAL.NS", "ESCORTS.NS", "M&M.NS",
        "TATAPOWER.NS", "ADANIPORTS.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS"
    ],
    "us_large": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
        "JPM", "V", "PG", "JNJ", "XOM", "UNH", "MA", "HD", "CVX", "PFE"
    ],
    "us_tech_growth": [
        "AMD", "AVGO", "ADBE", "CRM", "NOW", "SNOW", "PLTR", "PANW", "CRWD"
    ],
    "global_indices_etf": [
        "SPY", "QQQ", "IWM", "VTI", "EFA", "EEM", "GLD", "SLV", "TLT",
        "^NSEI", "^BSESN", "^GSPC", "^IXIC"
    ],
    "crypto": [
        "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD",
        "AVAX-USD", "DOGE-USD"
    ],
    "commodities": [
        "GC=F", "SI=F", "CL=F", "NG=F", "HG=F", "ZC=F", "ZS=F", "KC=F",
        "USO", "GLD", "SLV", "DBA", "DBC"
    ]
}

def get_diverse_tickers(total: int = 100, seed: int = 42) -> List[str]:
    """Sample balanced across categories for a robust study."""
    random.seed(seed)
    selected = []
    per_category = max(3, total // len(TICKER_POOLS))
    
    for category, tickers in TICKER_POOLS.items():
        n = min(per_category, len(tickers))
        selected.extend(random.sample(tickers, n))
    
    # Fill remaining randomly if needed
    all_tickers = [t for pool in TICKER_POOLS.values() for t in pool]
    while len(selected) < total:
        extra = random.choice(all_tickers)
        if extra not in selected:
            selected.append(extra)
    
    return selected[:total]


# =============================================================================
# DATA FETCHING + CACHING
# =============================================================================
def download_and_cache(
    tickers: List[str],
    period: str = DATA_PERIOD,
    force_refresh: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Download OHLCV data and cache as Parquet.
    Returns dict {ticker: dataframe with columns ['Open','High','Low','Close','Volume']}
    """
    import yfinance as yf
    
    data_dict = {}
    failed = []
    
    for ticker in tqdm(tickers, desc="Fetching & caching data", unit="ticker"):
        cache_file = CACHE_DIR / f"{ticker.replace('=', '_').replace('^', '')}.parquet"
        
        if cache_file.exists() and not force_refresh:
            try:
                df = pd.read_parquet(cache_file)
                if len(df) >= MIN_TRADING_DAYS:
                    data_dict[ticker] = df
                    continue
            except Exception:
                pass  # corrupted cache → re-download
        
        try:
            df = yf.download(
                ticker,
                period=period,
                progress=False,
                auto_adjust=True,
                threads=False
            )
            if df.empty or len(df) < MIN_TRADING_DAYS:
                failed.append(ticker)
                continue
            
            # Standardize columns
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
    
    if failed:
        logger.warning(f"Failed to fetch {len(failed)} tickers: {failed[:8]}...")
    
    logger.info(f"Cached/loaded {len(data_dict)} assets → {CACHE_DIR}")
    return data_dict


# =============================================================================
# CSSI INDICATOR (Python port of the Pine Script logic)
# =============================================================================
def compute_cssi(
    df: pd.DataFrame,
    thres: float = 1.0,
    look: int = 50,
    sig_len: int = 20,
    kvo_fast: int = 34,
    kvo_slow: int = 55,
    kvo_sig_len: int = 13,
    gamma: float = 0.85,
    norm_len: int = 50,
) -> pd.DataFrame:
    """
    Vectorized implementation of Conviction Streak Synergy Index (CSSI).
    Returns dataframe with aug_index, aug_signal, cross_up, cross_down, position, etc.
    """
    out = df.copy()
    close = out["Close"]
    hlc3 = (out["High"] + out["Low"] + close) / 3.0
    
    # 1. Price step (Hemrek core)
    ret_pct = close.pct_change() * 100.0
    s = pd.Series(0.0, index=close.index)
    s[ret_pct > thres] = 1.0
    s[ret_pct < -thres] = -1.0
    
    # 2. Klinger signed volume
    delta_h = hlc3.diff()
    sv = np.where(delta_h >= 0, out["Volume"], -out["Volume"])
    sv = pd.Series(sv, index=close.index).fillna(0)
    
    kvo = sv.ewm(span=kvo_fast, adjust=False).mean() - sv.ewm(span=kvo_slow, adjust=False).mean()
    ksig = kvo.ewm(span=kvo_sig_len, adjust=False).mean()
    k_hist = kvo - ksig
    
    # 3. Normalization
    k_std = k_hist.rolling(norm_len).std()
    k_norm = k_hist / k_std.replace(0, np.nan)
    k_norm = k_norm.fillna(0)
    
    # 4. Alignment & effective step
    align_mod = s * k_norm
    eff_s = s * (1.0 + gamma * align_mod)
    eff_s = eff_s.clip(-2.0, 2.0)
    
    # 5. Augmented count + detrend
    aug_count = eff_s.cumsum()
    aug_index = aug_count - aug_count.rolling(look, min_periods=look//2).mean()
    
    # 6. Signal line
    if SIG_TYPE == "sma":
        aug_signal = aug_index.rolling(sig_len, min_periods=sig_len//2).mean()
    elif SIG_TYPE == "ema":
        aug_signal = aug_index.ewm(span=sig_len, adjust=False).mean()
    else:
        aug_signal = aug_index.rolling(sig_len, min_periods=sig_len//2).mean()
    
    # 7. Signals
    cross_up = (aug_index > aug_signal) & (aug_index.shift(1) <= aug_signal.shift(1))
    cross_down = (aug_index < aug_signal) & (aug_index.shift(1) >= aug_signal.shift(1))
    
    out["aug_index"] = aug_index
    out["aug_signal"] = aug_signal
    out["cross_up"] = cross_up.fillna(False)
    out["cross_down"] = cross_down.fillna(False)
    
    # Simple position logic: long on cross_up, flat on cross_down (long-only for robustness)
    position = pd.Series(0, index=out.index)
    position[out["cross_up"]] = 1
    position[out["cross_down"]] = 0
    out["position"] = position.ffill().fillna(0)
    
    return out


# =============================================================================
# VECTORIZED BACKTEST
# =============================================================================
def backtest_asset(
    df: pd.DataFrame,
    fee: float = FEE_PCT,
    slippage: float = SLIPPAGE_PCT
) -> Dict[str, float]:
    """
    Simple long-only vectorized backtest on the 'position' column from compute_cssi.
    Returns performance metrics.
    """
    if "position" not in df.columns or df["position"].sum() == 0:
        return {"sharpe": -10, "profit_factor": 0, "calmar": -10, 
                "max_dd": 100, "trades": 0, "total_return": -99}
    
    close = df["Close"]
    ret = close.pct_change().fillna(0)
    
    # Apply costs only on position changes
    pos_change = df["position"].diff().abs()
    cost = pos_change * (fee + slippage)
    
    strategy_ret = df["position"].shift(1) * ret - cost
    equity = (1 + strategy_ret).cumprod() * INITIAL_CAPITAL
    
    # Metrics
    total_return = (equity.iloc[-1] / INITIAL_CAPITAL - 1) * 100
    
    # Daily returns for Sharpe (annualized)
    daily_ret = strategy_ret
    if daily_ret.std() == 0:
        sharpe = 0.0
    else:
        sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)
    
    # Max Drawdown
    peak = equity.cummax()
    dd = (equity - peak) / peak * 100
    max_dd = dd.min()
    
    # Profit Factor
    gains = daily_ret[daily_ret > 0].sum()
    losses = -daily_ret[daily_ret < 0].sum()
    profit_factor = gains / losses if losses > 0 else 10.0
    
    # Calmar
    calmar = (total_return / 100) / (abs(max_dd) / 100) if max_dd != 0 else 0
    
    # Number of trades (position flips)
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
# OPTUNA CALLBACK FOR LIVE LOGGING
# =============================================================================
def log_best_callback(study, trial):
    """Prints best score and parameters every 50 trials for visibility."""
    if trial.number % 50 == 0 or trial.number == study.best_trial.number:
        logger.info(
            f"Trial {trial.number:4d} | Best Score: {study.best_value:.4f} | "
            f"Mean Sharpe: {study.best_trial.user_attrs.get('mean_sharpe', 'N/A')} | "
            f"Assets tested: {study.best_trial.user_attrs.get('n_assets_tested', 'N/A')}"
        )


# =============================================================================
# OPTUNA OBJECTIVE (Robust across assets)
# =============================================================================
def objective(trial, data_dict: Dict[str, pd.DataFrame]) -> float:
    """
    Objective: Find parameters that deliver good risk-adjusted performance
    on AVERAGE across many different asset classes (robustness > single-asset peak).
    """
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
    
    scores = []
    pf_list = []
    sharpe_list = []
    
    for ticker, df in data_dict.items():
        try:
            cssi_df = compute_cssi(df, **params)
            metrics = backtest_asset(cssi_df)
            
            if metrics["trades"] < 8:          # too few signals → unreliable
                continue
            if metrics["sharpe"] < -1.5:       # terrible
                continue
                
            sharpe_list.append(metrics["sharpe"])
            pf_list.append(metrics["profit_factor"])
            
            # Composite per-asset score
            score = (
                WEIGHT_SHARPE * metrics["sharpe"] +
                WEIGHT_PROFIT_FACTOR * min(metrics["profit_factor"], 4.0) +
                WEIGHT_CALMAR * min(metrics["calmar"], 3.0)
            )
            scores.append(score)
            
        except Exception:
            continue
    
    if len(scores) < max(5, len(data_dict) // 3):
        return -999  # not enough valid assets
    
    mean_score = np.mean(scores)
    consistency = 1.0 / (1.0 + np.std(sharpe_list)) if len(sharpe_list) > 1 else 0.5
    
    final_objective = (
        mean_score * 0.85 +
        WEIGHT_CONSISTENCY * consistency * 10
    )
    
    # Store extra info for analysis
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
    logger.info("CSSI ROBUST PARAMETER OPTIMIZATION  •  v2 (Upgraded Logging)")
    logger.info("=" * 72)
    logger.info(f"Configuration → Trials: {N_TRIALS} | Jobs: {N_JOBS} | Optimization assets: ~35")
    logger.info(f"                  Fee: {FEE_PCT*100:.2f}% | Data period: {DATA_PERIOD}")
    logger.info("-" * 72)

    # 1. Get diverse tickers
    logger.info("[1/5] Building balanced universe of 100 instruments...")
    tickers = get_diverse_tickers(total=100, seed=42)
    logger.info(f"      → Sampled across 7 categories (Indian Large/Mid, US Large/Tech, Indices, Crypto, Commodities)")

    # 2. Download / load from cache
    phase_start = time.time()
    logger.info("\n[2/5] Downloading & caching OHLCV data (first run can take 5-15 min)...")
    data_dict = download_and_cache(tickers)
    logger.info(f"      → Loaded {len(data_dict)} assets with ≥{MIN_TRADING_DAYS} days of history "
                f"(took {time.time() - phase_start:.1f}s)")

    if len(data_dict) < 15:
        logger.error("❌ Not enough valid assets. Exiting.")
        return

    # 3. Create Optuna study
    logger.info(f"\n[3/5] Starting Optuna study ({N_TRIALS} trials on ~35 representative assets)...")
    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=50)
    )

    opt_tickers = list(data_dict.keys())[:35]
    opt_data = {k: data_dict[k] for k in opt_tickers}
    logger.info(f"      → Using {len(opt_data)} diverse assets for parameter search")

    # Run optimization with live logging callback
    phase_start = time.time()
    study.optimize(
        lambda trial: objective(trial, opt_data),
        n_trials=N_TRIALS,
        n_jobs=N_JOBS,
        show_progress_bar=True,
        callbacks=[log_best_callback]
    )
    logger.info(f"      → Optimization finished in {time.time() - phase_start:.1f}s")

    # 4. Results
    logger.info("\n[4/5] Optimization complete. Extracting best robust parameters...")
    best_params = study.best_params
    best_value = study.best_value

    logger.info("\n" + "=" * 72)
    logger.info("🏆 BEST ROBUST PARAMETERS (Recommended Defaults)")
    logger.info("=" * 72)
    for k, v in best_params.items():
        logger.info(f"   {k:15s} = {v}")
    logger.info("-" * 72)
    logger.info(f"   Best objective score : {best_value:.4f}")
    logger.info(f"   Mean Sharpe (best)   : {study.best_trial.user_attrs.get('mean_sharpe', 'N/A')}")
    logger.info(f"   Median Profit Factor : {study.best_trial.user_attrs.get('median_pf', 'N/A')}")
    logger.info(f"   Assets used in search: {study.best_trial.user_attrs.get('n_assets_tested', 'N/A')}")

    # Save artifacts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("optimization_results")
    results_dir.mkdir(exist_ok=True)

    params_file = results_dir / f"best_cssi_params_{timestamp}.json"
    with open(params_file, "w") as f:
        json.dump(best_params, f, indent=2)

    trials_file = results_dir / f"cssi_trials_{timestamp}.csv"
    study.trials_dataframe().to_csv(trials_file, index=False)

    logger.info(f"\n   ✅ Best params saved → {params_file}")
    logger.info(f"   ✅ Full trial history → {trials_file}")

    # 5. Validation on full universe
    logger.info("\n[5/5] Validating best parameters on FULL cached universe...")
    phase_start = time.time()
    validation_results = []

    for ticker, df in tqdm(data_dict.items(), desc="Validating best params", unit="asset"):
        try:
            cssi_df = compute_cssi(df, **best_params)
            metrics = backtest_asset(cssi_df)
            metrics["ticker"] = ticker
            validation_results.append(metrics)
        except Exception as e:
            logger.debug(f"Skipped {ticker}: {e}")

    val_df = pd.DataFrame(validation_results)
    logger.info(f"      → Validation completed in {time.time() - phase_start:.1f}s on {len(val_df)} assets")

    # Summary stats
    logger.info("\nValidation Summary (Best Params across all assets):")
    summary = val_df[["sharpe", "profit_factor", "calmar", "max_dd", "trades"]].describe().round(2)
    for line in summary.to_string().split("\n"):
        logger.info("   " + line)

    val_file = results_dir / f"validation_full_universe_{timestamp}.csv"
    val_df.to_csv(val_file, index=False)
    logger.info(f"\n   ✅ Full validation CSV saved → {val_file}")

    # Final timing
    total_time = time.time() - start_time
    logger.info("\n" + "=" * 72)
    logger.info(f"✅ RUN COMPLETE in {total_time/60:.1f} minutes")
    logger.info("=" * 72)
    logger.info("NEXT STEPS:")
    logger.info("1. Copy values from best_cssi_params_*.json into your Pine Script CSSI inputs.")
    logger.info("2. Review validation_full_universe CSV → check consistency across Indian vs US vs Crypto.")
    logger.info("3. Increase N_TRIALS or opt_data size for even more robust defaults.")
    logger.info("4. Re-run anytime — caching makes it fast after first execution.")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
