"""
PRAGYAM (प्रज्ञम) - Portfolio Intelligence | A Hemrek Capital Product
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Walk-forward portfolio curation with regime-aware strategy allocation.
Multi-strategy backtesting and capital optimization engine.
"""

import os
# --- Optimization: Prevent NumPy from thread-thrashing under Streamlit multiprocessing ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import plotly.graph_objects as go
import logging

logger = logging.getLogger("pragyam")
from typing import List, Dict, Tuple, Optional
import io
import base64
import warnings
from scipy.stats import theilslopes

# --- Suppress known NumPy warnings during backtest warm-up ---
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in divide')
# --- End suppression ---

# --- Import Unified Chart Components ---
from charts import (
    COLORS, get_chart_layout,
    create_equity_drawdown_chart, create_rolling_metrics_chart,
    create_correlation_heatmap, create_tier_sharpe_heatmap,
    create_risk_return_scatter, create_factor_radar,
    create_weight_evolution_chart, create_signal_heatmap
)

# --- Import Strategies from strategies.py ---
try:
    from strategies import BaseStrategy, discover_strategies, STRATEGY_REGISTRY
except ImportError:
    st.error("Fatal Error: `strategies.py` not found. Please ensure it's in the same directory.")
    st.stop()

# --- Import Live Data Generation from backdata.py ---
try:
    from backdata import (
        generate_historical_data, 
        load_symbols_from_file, 
        MAX_INDICATOR_PERIOD,
        SYMBOLS_UNIVERSE
    )
except ImportError:
    st.error("Fatal Error: `backdata.py` not found. Please ensure it's in the same directory.")
    st.stop()

# --- Import Trigger Data Loader from strategy_selection.py ---
try:
    from strategy_selection import (
        load_breadth_data,
        compute_adaptive_thresholds,
        SIP_TRIGGER, SWING_BUY_TRIGGER, SWING_SELL_TRIGGER,
        BREADTH_SHEET_URL
    )
    STRATEGY_SELECTION_AVAILABLE = True
except ImportError:
    STRATEGY_SELECTION_AVAILABLE = False
    logger.warning("strategy_selection.py not found. Trigger data must be uploaded manually.")

# --- Import Unified Backtest Engine for Dynamic Strategy Selection ---
try:
    from backtest_engine import (
        UnifiedBacktestEngine,
        DynamicPortfolioStylesGenerator,
        PerformanceMetrics,
        compute_risk_metrics,
        estimate_transaction_cost_bps,
    )
except ImportError:
    st.error("Fatal Error: `backtest_engine.py` not found. Please ensure it's in the same directory.")
    st.stop()

DYNAMIC_SELECTION_AVAILABLE = True


# --- System Configuration ---
st.set_page_config(page_title="PRAGYAM | Portfolio Intelligence", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")

# --- Constants ---
VERSION = "v4.0.0"  # Adversarial audit resolutions: DSR, adaptive quantile gates, RMT signal extraction, market impact, breadth-based regimes, conformal dependence correction
PRODUCT_NAME = "Pragyam"
COMPANY = "Hemrek Capital"

# --- Trigger-Based Backtest Configuration ---
# Thresholds derived from strategy_selection.py (research-backed, NOT arbitrary)
TRIGGER_CONFIG = {
    'SIP Investment': {
        'buy_threshold': SIP_TRIGGER,           # from strategy_selection constants
        'sell_threshold': SWING_SELL_TRIGGER,    # from strategy_selection constants
        'sell_enabled': False,   # SIP accumulates, no sell
        'use_sprt': False,       # REC-3: set True to use SPRT evidence accumulation
        'description': 'Systematic accumulation on regime dips'
    },
    'Swing Trading': {
        'buy_threshold': SWING_BUY_TRIGGER,     # from strategy_selection constants
        'sell_threshold': SWING_SELL_TRIGGER,    # from strategy_selection constants
        'sell_enabled': True,
        'use_sprt': False,       # REC-3: set True to use SPRT evidence accumulation
        'description': 'Tactical entry/exit on regime signals'
    },
    'All Weather': {
        'buy_threshold': SIP_TRIGGER,           # from strategy_selection constants
        'sell_threshold': SWING_SELL_TRIGGER,
        'sell_enabled': False,
        'use_sprt': False,       # REC-3: set True to use SPRT evidence accumulation
        'description': 'Balanced regime-aware allocation'
    }
}

# --- CSS Styling (Hemrek Capital Design System) ---
_css_path = os.path.join(os.path.dirname(__file__), "style.css")
with open(_css_path) as _f:
    st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)

# --- Session State Management ---
if 'performance' not in st.session_state: st.session_state.performance = None
if 'portfolio' not in st.session_state: st.session_state.portfolio = None
if 'current_df' not in st.session_state: st.session_state.current_df = None
if 'selected_date' not in st.session_state: st.session_state.selected_date = None
if 'suggested_mix' not in st.session_state: st.session_state.suggested_mix = None
if 'regime_display' not in st.session_state: st.session_state.regime_display = None # For sidebar display
if 'min_pos_pct' not in st.session_state: st.session_state.min_pos_pct = 1.0
if 'max_pos_pct' not in st.session_state: st.session_state.max_pos_pct = 10.0

# --- Base Classes and Utilities ---
def create_export_link(data_bytes, filename):
    """Create downloadable CSV link"""
    b64 = base64.b64encode(data_bytes).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-link">Download Portfolio CSV</a>'
    return href

TRANSACTION_COST_BPS = 20

def compute_portfolio_return(
    portfolio: pd.DataFrame,
    next_prices: pd.DataFrame,
    is_rebalance: bool = False,
    prev_portfolio: Optional[pd.DataFrame] = None,
    current_prices: Optional[pd.DataFrame] = None,
) -> float:
    """Compute single-period portfolio return with turnover-proportional costs."""
    if portfolio.empty or 'value' not in portfolio.columns or portfolio['value'].sum() == 0:
        return 0.0
    merged = portfolio.merge(next_prices[['symbol', 'price']], on='symbol', how='left', suffixes=('_prev', '_next'))
    if merged.empty:
        return 0.0
    merged['price_next'] = merged['price_next'].fillna(merged['price_prev'])
    safe_prev = np.where(merged['price_prev'] > 0, merged['price_prev'], 1e-10)
    returns = (merged['price_next'] - safe_prev) / safe_prev
    gross_return = float(np.average(returns, weights=merged['value']))

    if is_rebalance and TRANSACTION_COST_BPS > 0:
        if prev_portfolio is not None and not prev_portfolio.empty and 'value' in prev_portfolio.columns:
            prices_df = current_prices if current_prices is not None else portfolio
            drifted_prev = prev_portfolio.merge(prices_df[['symbol', 'price']], on='symbol', how='left')
            drifted_prev['price_y'] = drifted_prev['price_y'].fillna(drifted_prev['price_x'])
            drifted_prev['drifted_value'] = drifted_prev['units'] * drifted_prev['price_y']
            
            curr_total = portfolio['value'].sum()
            prev_total = drifted_prev['drifted_value'].sum()
            if curr_total > 0 and prev_total > 0:
                curr_v = portfolio.set_index('symbol')['value']
                prev_v = drifted_prev.set_index('symbol')['drifted_value']
                all_symbols = curr_v.index.union(prev_v.index)
                curr_vals = curr_v.reindex(all_symbols, fill_value=0.0)
                prev_vals = prev_v.reindex(all_symbols, fill_value=0.0)
                trade_value = float(np.abs(curr_vals - prev_vals).sum())
                turnover = (trade_value / curr_total) / 2.0
            else:
                turnover = 1.0
        else:
            turnover = 1.0

        gross_return -= turnover * TRANSACTION_COST_BPS / 10000.0
    return gross_return

def calculate_advanced_metrics(returns_with_dates: List[Dict]) -> Tuple[Dict, float]:
    """Calculate comprehensive risk-adjusted performance metrics."""
    default_metrics = {
        'total_return': 0, 'annual_return': 0, 'volatility': 0,
        'sharpe': 0, 'sortino': 0, 'max_drawdown': 0, 'calmar': 0,
        'win_rate': 0, 'kelly_criterion': 0, 'omega_ratio': 1.0,
        'tail_ratio': 1.0, 'gain_to_pain': 0, 'profit_factor': 1.0
    }
    if len(returns_with_dates) < 2:
        return default_metrics, 52.0

    returns_df = pd.DataFrame(returns_with_dates).sort_values('date').set_index('date')
    date_range = returns_df.index
    if hasattr(date_range, 'min') and hasattr(date_range, 'max'):
        total_calendar_days = (date_range.max() - date_range.min()).days
        if total_calendar_days > 0:
            periods_per_year = len(date_range) * 365.25 / total_calendar_days
        else:
            periods_per_year = 52.0
    else:
        periods_per_year = 52.0
    periods_per_year = float(np.clip(periods_per_year, 12, 365))

    returns = returns_df['return'].values
    core = compute_risk_metrics(returns, periods_per_year=periods_per_year)

    r = np.asarray(returns, dtype=np.float64)
    gains = r[r > 0]
    losses = r[r < 0]
    
    total_gains = gains.sum() if len(gains) > 0 else 0
    total_losses = abs(losses.sum()) if len(losses) > 0 else 0

    mu = r.mean()
    var = r.var()
    if var > 1e-8:
        std = np.sqrt(var)
        z_scores = np.clip((r - mu) / std, -3.0, 3.0)
        skewness = float(np.mean(z_scores ** 3))
        base_kelly = mu / var
        skew_penalty = max(0.0, 1.0 + (mu * skewness) / std)
        kelly = base_kelly * skew_penalty
    else:
        kelly = 0.0
    kelly = float(np.clip(kelly, -1, 1))

    omega_ratio = total_gains / total_losses if total_losses > 0.0001 else (total_gains * 10 if total_gains > 0 else 1.0)
    omega_ratio = float(np.clip(omega_ratio, 0, 50))
    profit_factor = omega_ratio

    upper_tail = np.percentile(r, 95) if len(r) >= 20 else r.max()
    lower_tail = abs(np.percentile(r, 5)) if len(r) >= 20 else abs(r.min())
    tail_ratio = upper_tail / lower_tail if lower_tail > 0.0001 else (10.0 if upper_tail > 0 else 1.0)
    tail_ratio = float(np.clip(tail_ratio, 0, 20))

    pain = abs(losses.sum()) if len(losses) > 0 else 0
    gain_to_pain = r.sum() / pain if pain > 0.0001 else (r.sum() * 10 if r.sum() > 0 else 0)
    gain_to_pain = float(np.clip(gain_to_pain, -20, 20))

    metrics = {
        'total_return': core['total_return'],
        'annual_return': core['ann_return'],
        'volatility': core['volatility'],
        'sharpe': core['sharpe'],
        'sortino': core['sortino'],
        'max_drawdown': core['max_drawdown'],
        'calmar': core['calmar'],
        'win_rate': core['win_rate'],
        'kelly_criterion': kelly,
        'omega_ratio': omega_ratio,
        'tail_ratio': tail_ratio,
        'gain_to_pain': gain_to_pain,
        'profit_factor': profit_factor,
    }
    return metrics, periods_per_year


def calculate_strategy_weights(
    performance: Dict,
    method: str = 'softmax_sharpe',
    returns_data: Dict[str, np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute strategy allocation weights.

    Methods:
        'softmax_sharpe': Original Sharpe-based softmax (backward compatible).
        'rmt_min_variance': Minimum variance using RMT-cleaned covariance.
        'rmt_risk_parity': Equal risk contribution using RMT-cleaned covariance.
        'hrp': Hierarchical Risk Parity (REC-5) — robust to estimation error.
        'equal': Uniform 1/N allocation.

    Args:
        performance: performance dict with 'strategy' key.
        method: weighting method to use.
        returns_data: {strategy_name: 1D returns array} for RMT methods.
    """
    strat_names = list(performance['strategy'].keys())
    if not strat_names:
        return {}

    # RMT-based methods (including HRP — REC-5)
    if method in ('rmt_min_variance', 'rmt_risk_parity', 'hrp') and returns_data:
        try:
            from rmt_core import rmt_minimum_variance_weights, rmt_risk_parity_weights, hrp_weights

            # Align returns to common length
            available = [n for n in strat_names if n in returns_data and len(returns_data[n]) >= 20]
            if len(available) >= 2:
                min_len = min(len(returns_data[n]) for n in available)
                returns_matrix = np.column_stack([returns_data[n][:min_len] for n in available])

                if method == 'rmt_min_variance':
                    weights_dict = rmt_minimum_variance_weights(returns_matrix, available)
                elif method == 'hrp':
                    weights_dict = hrp_weights(returns_matrix, available)
                else:
                    weights_dict = rmt_risk_parity_weights(returns_matrix, available)

                # Add zero weight for strategies not in returns_data
                for name in strat_names:
                    if name not in weights_dict:
                        weights_dict[name] = 0.0
                # Renormalize
                total = sum(weights_dict.values())
                if total > 0:
                    return {n: w / total for n, w in weights_dict.items()}
        except Exception:
            pass  # Fall through to softmax

    if method == 'equal':
        # Epistemic Warning: Evaluate spectral redundancy before naive 1/N allocation
        if returns_data and len(returns_data) >= 2:
            try:
                from rmt_core import compute_spectral_diagnostics
                available = [n for n in strat_names if n in returns_data and len(returns_data[n]) >= 20]
                if len(available) >= 2:
                    min_len = min(len(returns_data[n]) for n in available)
                    rm = np.column_stack([returns_data[n][:min_len] for n in available])
                    diag = compute_spectral_diagnostics(rm)
                    if diag.effective_rank < len(available) * 0.5:
                        logger.warning(f"EPISTEMIC RISK: Using 1/N 'equal' allocation on highly redundant strategies. "
                                       f"Apparent strategies: {len(available)} | Effective independent bets: {diag.effective_rank:.1f}. "
                                       f"Portfolio variance will scale quadratically for redundant clusters.")
            except Exception:
                pass
        return {name: 1.0 / len(strat_names) for name in strat_names}

    # Default: softmax_sharpe with ADAPTIVE temperature scaling.
    # HIGH-2 fix: κ = c / σ_Sharpe makes allocation invariant to the
    # scale of performance differences.  c=1.5 gives meaningful
    # differentiation without extreme concentration.
    SOFTMAX_CONCENTRATION = 1.5  # desired concentration constant

    sharpe_values = []
    for name in strat_names:
        strat_data = performance['strategy'][name]
        if isinstance(strat_data, dict) and 'metrics' in strat_data and isinstance(strat_data['metrics'], dict):
            sharpe = strat_data['metrics'].get('sharpe', 0)
        else:
            sharpe = strat_data.get('sharpe', 0)
        if not isinstance(sharpe, (int, float)) or not np.isfinite(sharpe):
            sharpe = 0
        sharpe_values.append(sharpe)

    # C-2: Deflated Sharpe Ratio — haircut Sharpe values for multiple testing
    if returns_data and len(returns_data) >= 2:
        try:
            from quant_core import compute_family_dsr
            dsr_results = compute_family_dsr(returns_data)
            
            # FDR Control (Benjamini-Hochberg procedure, Harvey, Liu & Zhu 2016)
            p_values = sorted([(name, dsr.p_value) for name, dsr in dsr_results.items()], key=lambda x: x[1])
            M = len(p_values)
            alpha_fdr = 0.05
            max_k = 0
            for k, (name, p_val) in enumerate(p_values, 1):
                if p_val <= (k / M) * alpha_fdr:
                    max_k = k
            
            significant_strats = set(name for name, _ in p_values[:max_k])
            
            for i, name in enumerate(strat_names):
                if name in dsr_results and name not in significant_strats:
                    # Mask out noise strategies: exp(-inf) approaches 0 weight
                    sharpe_values[i] = -np.inf
        except Exception:
            pass  # DSR is enhancement, not critical path

    sharpe_values = np.array(sharpe_values)

    if sharpe_values.size == 0:
        return {name: 1.0 / len(strat_names) for name in strat_names} if strat_names else {}

    # Epistemic Fix: Robust Adaptive Temperature for Leptokurtic Distributions
    # np.std() squares outliers, artificially collapsing the temperature for the cluster.
    # We replace L2 variance with L1 Median Absolute Deviation (MAD) for robust scaling.
    median_sharpe = np.median(sharpe_values)
    mad = np.median(np.abs(sharpe_values - median_sharpe))
    robust_std = mad * 1.4826  # scale factor for normal distribution equivalence
    
    # Cap outlier z-scores to prevent winner-take-all softmax degeneration
    z_scores = (sharpe_values - median_sharpe) / max(robust_std, 0.05)
    robust_sharpes = np.clip(z_scores, -3.0, 3.0)
    
    stable_sharpes = SOFTMAX_CONCENTRATION * (robust_sharpes - np.max(robust_sharpes))
    exp_sharpes = np.exp(stable_sharpes)
    total_score = np.sum(exp_sharpes)

    if total_score == 0 or not np.isfinite(total_score):
        return {name: 1.0 / len(strat_names) for name in strat_names}

    weights = exp_sharpes / total_score
    return {name: weights[i] for i, name in enumerate(strat_names)}

def _calculate_performance_on_window(
    window_data: List[Tuple[datetime, pd.DataFrame]], 
    strategies: Dict[str, BaseStrategy], 
    training_capital: float, 
    precomputed_cache: Dict = None,
    cache_start_idx: int = 0
) -> Dict:
    performance = {name: {'returns': []} for name in strategies}
    subset_performance = {name: {} for name in strategies}
    
    if precomputed_cache:
        # O(1) Cache slice retrieval instead of O(T^2) regeneration
        window_end_idx = cache_start_idx + len(window_data) - 1
        for name in strategies:
            performance[name]['returns'] = precomputed_cache['strategy'][name][cache_start_idx:window_end_idx]
            for tier_name, rets in precomputed_cache['subset'].get(name, {}).items():
                if tier_name not in subset_performance[name]: subset_performance[name][tier_name] = []
                subset_performance[name][tier_name] = rets[cache_start_idx:window_end_idx]
    else:
        for i in range(len(window_data) - 1):
            date, df = window_data[i]
            next_date, next_df = window_data[i+1]
            for name, strategy in strategies.items():
                try:
                    portfolio = strategy.generate_portfolio(df, training_capital)
                    if portfolio.empty: continue
                    performance[name]['returns'].append({'return': compute_portfolio_return(portfolio, next_df), 'date': next_date})
                    n, tier_size = len(portfolio), 10
                    num_tiers = n // tier_size
                    if num_tiers == 0: continue
                    for j in range(num_tiers):
                        tier_name = f'tier_{j+1}'
                        if tier_name not in subset_performance[name]: subset_performance[name][tier_name] = []
                        sub_df = portfolio.iloc[j*tier_size : (j+1)*tier_size]
                        if not sub_df.empty:
                            sub_ret = compute_portfolio_return(sub_df, next_df)
                            subset_performance[name][tier_name].append({'return': sub_ret, 'date': next_date})
                except Exception as e: logger.error(f"Window Calc Error ({name}, {date}): {e}")
                
    final_performance = {}
    for name, perf in performance.items():
        metrics = calculate_advanced_metrics(perf['returns'])[0]
        final_performance[name] = {'metrics': metrics, 'sharpe': metrics['sharpe']}
    final_sub_performance = {}
    for name, data in subset_performance.items():
        final_sub_performance[name] = {
            sub: calculate_advanced_metrics(sub_perf)[0]['sharpe']
            for sub, sub_perf in data.items() if sub_perf
        }
    return {'strategy': final_performance, 'subset': final_sub_performance}


def run_trigger_based_backtest(
    strategies: Dict[str, BaseStrategy],
    historical_data: List[Tuple[datetime, pd.DataFrame]],
    trigger_df: Optional[pd.DataFrame],
    buy_col: str = 'REL_BREADTH',
    buy_threshold: float = 0.42,
    sell_col: str = 'REL_BREADTH',
    sell_threshold: float = 1.5,
    sell_enabled: bool = False,
    capital: float = 2500000.0,
    deployment_style: str = 'SIP',
    progress_bar = None,
    status_text = None
) -> Dict:
    """Wrapper to run trigger-based backtest using the UnifiedBacktestEngine."""
    logger.info(f"TRIGGER-BASED BACKTEST: {deployment_style} mode | {len(strategies)} strategies")
    logger.info(f"  Buy: {buy_col} < {buy_threshold} | Sell: {sell_col} > {sell_threshold} (enabled={sell_enabled})")
    
    if not historical_data:
        logger.error("No historical data provided")
        return {}

    is_sip = 'SIP' in deployment_style
    mode = 'sip' if is_sip else 'swing'

    engine = UnifiedBacktestEngine(capital=capital)
    engine._historical_data = historical_data
    engine._strategies = strategies

    all_results = engine.run_backtest(
        mode=mode,
        external_trigger_df=trigger_df,
        buy_col=buy_col,
        sell_col=sell_col,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        progress_callback=lambda p, m: progress_bar.progress(0.3 + p * 0.4, text=m) if progress_bar else None
    )

    logger.info(f"TRIGGER-BASED BACKTEST COMPLETE: {len(all_results)} strategies processed")
    return all_results


def evaluate_historical_performance_trigger_based(
    _strategies: Dict[str, BaseStrategy],
    historical_data: List[Tuple[datetime, pd.DataFrame]],
    trigger_df: Optional[pd.DataFrame] = None,
    deployment_style: str = 'SIP Investment',
    trigger_config: Optional[Dict] = None,
    test_window_size: int = 50
) -> Dict:
    """
    Walk-forward evaluation of strategies using trigger-based buy/sell signals.
    
    UNIFIED PIPELINE (eliminates the previous fragmentation):
    1. Build trigger masks from REL_BREADTH data
    2. Walk forward through time: train on past → curate on trigger → test OOS
    3. Produce System_Curated returns + per-strategy OOS returns
    4. Calculate metrics compatible with curate_final_portfolio
    
    This replaces BOTH the old trigger-only backtest AND the dead walk-forward code.
    """
    # Get trigger configuration
    if trigger_config is None:
        trigger_config = TRIGGER_CONFIG.get(deployment_style, TRIGGER_CONFIG['SIP Investment'])
    
    buy_threshold = trigger_config.get('buy_threshold', 0.42)
    sell_threshold = trigger_config.get('sell_threshold', 0.50)
    sell_enabled = trigger_config.get('sell_enabled', False)
    is_sip = 'SIP' in deployment_style
    
    MIN_TRAIN_DAYS = 5
    # REC-2: Embargo period between training window and test observation.
    # Prevents information leakage through serial correlation of indicator
    # features.  Set to 1 day (conservative; the actual indicator lookback
    # is handled by backdata.py's warmup, but returns computed from the
    # last training observation can correlate with the first OOS observation).
    # Reference: Lopez de Prado (2018), "Advances in Financial Machine
    # Learning", Ch. 7 — purged/embargoed cross-validation.
    EMBARGO_DAYS = 1
    TRAINING_CAPITAL = 2500000.0

    logger.info("=" * 70)
    logger.info("WALK-FORWARD EVALUATION (TRIGGER-INTEGRATED)")
    logger.info("=" * 70)
    logger.info(f"  Style: {deployment_style} | Buy < {buy_threshold} | Sell >= {sell_threshold} (enabled={sell_enabled})")
    logger.info(f"  Strategies: {list(_strategies.keys())}")
    logger.info(f"  Data points: {len(historical_data)} | Embargo: {EMBARGO_DAYS} days")

    if len(historical_data) < MIN_TRAIN_DAYS + EMBARGO_DAYS + 2:
        logger.error(f"Not enough data ({len(historical_data)}) for walk-forward. Need {MIN_TRAIN_DAYS + 2}+")
        return {}
    
    # ─────────────────────────────────────────────────────────────────────
    # BUILD TRIGGER MASKS
    # ─────────────────────────────────────────────────────────────────────
    simulation_dates = []
    for d, _ in historical_data:
        sim_date = d.date() if hasattr(d, 'date') else d
        simulation_dates.append(sim_date)
    
    buy_mask = [False] * len(historical_data)
    sell_mask = [False] * len(historical_data)
    
    use_sprt = trigger_config.get('use_sprt', False)

    if trigger_df is not None and not trigger_df.empty and 'REL_BREADTH' in trigger_df.columns:
        if use_sprt:
            # REC-3: Use SPRT evidence accumulation instead of fixed thresholds
            try:
                from strategy_selection import get_sprt_trigger_dates
                sprt_buy_dates, sprt_sell_dates = get_sprt_trigger_dates(trigger_df)
                sprt_buy_set = set(d.date() if hasattr(d, 'date') else d for d in sprt_buy_dates)
                sprt_sell_set = set(d.date() if hasattr(d, 'date') else d for d in sprt_sell_dates)
                for i, sim_date in enumerate(simulation_dates):
                    if sim_date in sprt_buy_set:
                        buy_mask[i] = True
                    if sell_enabled and sim_date in sprt_sell_set:
                        sell_mask[i] = True
                logger.info(f"  SPRT trigger masks: {sum(buy_mask)} buy days, {sum(sell_mask)} sell days")
            except Exception as e:
                logger.warning(f"  SPRT trigger failed ({e}), falling back to fixed thresholds")
                use_sprt = False  # fall through to fixed thresholds below

        if not use_sprt:
            # Fixed-threshold triggers (original method)
            if hasattr(trigger_df.index, 'date'):
                trigger_map = {idx.date(): val for idx, val in trigger_df['REL_BREADTH'].items() if pd.notna(val)}
            else:
                trigger_map = {pd.to_datetime(idx).date(): val for idx, val in trigger_df['REL_BREADTH'].items() if pd.notna(val)}

            for i, sim_date in enumerate(simulation_dates):
                if sim_date in trigger_map:
                    breadth = trigger_map[sim_date]
                    if breadth < buy_threshold:
                        buy_mask[i] = True
                    if sell_enabled and breadth >= sell_threshold:
                        sell_mask[i] = True

        logger.info(f"  Trigger masks: {sum(buy_mask)} buy days, {sum(sell_mask)} sell days")
    else:
        # No trigger data: treat every day as active (standard walk-forward)
        logger.warning("  No trigger data: using standard walk-forward (every day active)")
        for i in range(len(historical_data)):
            buy_mask[i] = True
    
    # ─────────────────────────────────────────────────────────────────────
    # WALK-FORWARD LOOP
    # ─────────────────────────────────────────────────────────────────────
    all_names = list(_strategies.keys()) + ['System_Curated']
    oos_perf = {name: {'returns': []} for name in all_names}
    weight_entropies = []
    strategy_weights_history = []
    subset_weights_history = []
    spectral_history = []

    # --- O(T^2) Bottleneck Fix: Precompute all strategy returns once ---
    precalc_cache = {'strategy': {n: [] for n in _strategies}, 'subset': {n: {} for n in _strategies}}
    logger.info("Precomputing strategy returns for walk-forward cache...")
    
    # C-1 FIX: Inject IndicatorHistoryAccumulator for adaptive quantile gates
    try:
        from quant_core import IndicatorHistoryAccumulator
        global_acc = IndicatorHistoryAccumulator()
    except ImportError:
        global_acc = None
        
    for j in range(len(historical_data) - 1):
        df_date, df = historical_data[j]
        next_date, next_df = historical_data[j+1]
        if global_acc is not None:
            global_acc.update(df)
        for name, strategy in _strategies.items():
            try:
                if global_acc is not None:
                    strategy.accumulator = global_acc
                port = strategy.generate_portfolio(df, TRAINING_CAPITAL)
                ret = compute_portfolio_return(port, next_df) if not port.empty else 0.0
                precalc_cache['strategy'][name].append({'return': ret, 'date': next_date})
                if not port.empty:
                    num_tiers = len(port) // 10
                    for t in range(num_tiers):
                        t_name = f'tier_{t+1}'
                        if t_name not in precalc_cache['subset'][name]: precalc_cache['subset'][name][t_name] = []
                        t_port = port.iloc[t*10:(t+1)*10]
                        t_ret = compute_portfolio_return(t_port, next_df) if not t_port.empty else 0.0
                        precalc_cache['subset'][name][t_name].append({'return': t_ret, 'date': next_date})
            except Exception:
                precalc_cache['strategy'][name].append({'return': 0.0, 'date': next_date})
    # -------------------------------------------------------------------

    # State for SIP accumulation tracking
    sip_portfolio_units = {}  # strategy -> {symbol: units}
    swing_in_position = {}   # strategy -> bool

    # Track last held portfolios for computing returns on non-trigger days
    last_curated_port = pd.DataFrame()
    last_strategy_ports = {}  # strategy_name -> portfolio DataFrame
    
    start_idx = max(MIN_TRAIN_DAYS + EMBARGO_DAYS, len(historical_data) - test_window_size - 1)
    progress_bar = st.progress(0, text="Initializing walk-forward...")
    total_steps = len(historical_data) - start_idx - 1

    if total_steps <= 0:
        progress_bar.empty()
        logger.error("Not enough data for walk-forward steps")
        return {}

    step_count = 0
    for i in range(start_idx, len(historical_data) - 1):
        # REC-2: train on data[:i-EMBARGO], skip embargo gap, test at i
        # Execution Fix: Shift from 50-day Rolling Window to Expanding Window.
        # RMT and Sharpe estimators require maximum T/N ratio to converge accurately. 
        # The O(1) cache ensures this costs 0 extra computation time.
        train_start = 0
        train_window = historical_data[train_start : i - EMBARGO_DAYS]
        test_date, test_df = historical_data[i]
        next_date, next_df = historical_data[i + 1]
        
        if global_acc is not None:
            global_acc.update(test_df)
            
        is_buy_day = buy_mask[i]
        is_sell_day = sell_mask[i]
        
        step_count += 1
        pct = step_count / total_steps
        progress_bar.progress(min(pct, 0.99), text=f"Walk-forward step {step_count}/{total_steps}")
        
        # ─── SYSTEM CURATED: Walk-Forward Portfolio ───
        if is_buy_day or (not is_sip and is_sell_day):
            try:
                # Train on historical window to get performance-based weights
                in_sample_perf = _calculate_performance_on_window(train_window, _strategies, TRAINING_CAPITAL, precalc_cache, cache_start_idx=train_start)
                
                curated_port, strat_wts, sub_wts, _ = curate_final_portfolio(
                    _strategies, in_sample_perf, test_df, TRAINING_CAPITAL, 30, 1.0, 10.0
                )
                
                strategy_weights_history.append({'date': test_date, **strat_wts})
                subset_weights_history.append({'date': test_date, **sub_wts})
                
                if curated_port.empty:
                    oos_perf['System_Curated']['returns'].append({'return': 0, 'date': next_date})
                else:
                    prev_curated = last_curated_port if not last_curated_port.empty else None
                    last_curated_port = curated_port.copy()
                    oos_ret = compute_portfolio_return(curated_port, next_df, is_rebalance=True, prev_portfolio=prev_curated, current_prices=test_df)
                    oos_perf['System_Curated']['returns'].append({'return': oos_ret, 'date': next_date})

                    # Track weight entropy
                    weights = curated_port['weightage_pct'] / 100
                    valid_weights = weights[weights > 0]
                    if len(valid_weights) > 0:
                        entropy = -np.sum(valid_weights * np.log2(valid_weights))
                        weight_entropies.append(entropy)

            except Exception as e:
                logger.error(f"Walk-forward curation error ({test_date}): {e}")
                oos_perf['System_Curated']['returns'].append({'return': 0, 'date': next_date})
        else:
            # Non-trigger day: compute return from held positions
            if not last_curated_port.empty:
                held_ret = compute_portfolio_return(last_curated_port, next_df)
                oos_perf['System_Curated']['returns'].append({'return': held_ret, 'date': next_date})
                # Update held portfolio prices for next day's return calc
                merged = last_curated_port.merge(next_df[['symbol', 'price']], on='symbol', how='left')
                if 'price_y' in merged.columns:
                    merged['price'] = merged['price_y'].fillna(merged['price_x'])
                    merged = merged.drop(columns=['price_x', 'price_y'])
                    merged['value'] = merged['units'] * merged['price']
                    last_curated_port = merged
            else:
                oos_perf['System_Curated']['returns'].append({'return': 0, 'date': next_date})
        
        # ─── PER-STRATEGY OOS Returns ───
        for name, strategy in _strategies.items():
            try:
                if is_buy_day:
                    if global_acc is not None:
                        strategy.accumulator = global_acc
                    portfolio = strategy.generate_portfolio(test_df, TRAINING_CAPITAL)
                    if not portfolio.empty:
                        prev_strat_port = last_strategy_ports.get(name)
                        if prev_strat_port is not None and prev_strat_port.empty:
                            prev_strat_port = None
                        last_strategy_ports[name] = portfolio.copy()
                        oos_perf[name]['returns'].append({
                            'return': compute_portfolio_return(portfolio, next_df, is_rebalance=True, prev_portfolio=prev_strat_port, current_prices=test_df),
                            'date': next_date
                        })
                    else:
                        oos_perf[name]['returns'].append({'return': 0, 'date': next_date})
                elif name in last_strategy_ports and not last_strategy_ports[name].empty:
                    # Held position: compute actual return
                    held_ret = compute_portfolio_return(last_strategy_ports[name], next_df)
                    oos_perf[name]['returns'].append({'return': held_ret, 'date': next_date})
                else:
                    oos_perf[name]['returns'].append({'return': 0, 'date': next_date})
            except Exception as e:
                logger.error(f"OOS Strategy Error ({name}, {test_date}): {e}")
                oos_perf[name]['returns'].append({'return': 0, 'date': next_date})

        # ─── SPECTRAL TRACKING (every 5th step) — uses return time-series, not cross-section ───
        if step_count % 5 == 0 and i >= 20:
            try:
                from rmt_core import compute_spectral_diagnostics
                # Build T×N return matrix from trailing window of per-strategy OOS returns
                strat_keys = [n for n in _strategies.keys()]
                ret_cols = {}
                for sn in strat_keys:
                    r_list = oos_perf[sn]['returns']
                    if len(r_list) >= 10:
                        ret_cols[sn] = np.array([
                            r['return'] if isinstance(r, dict) else r for r in r_list
                        ], dtype=float)
                if len(ret_cols) >= 2:
                    min_len = min(len(v) for v in ret_cols.values())
                    if min_len >= 10:
                        ret_matrix = np.column_stack([v[:min_len] for v in ret_cols.values()])
                        ret_matrix = np.nan_to_num(ret_matrix, nan=0.0)
                        spec_diag = compute_spectral_diagnostics(ret_matrix)
                        spectral_history.append({
                            'date': test_date,
                            'absorption_ratio': spec_diag.absorption_ratio,
                            'effective_rank': spec_diag.effective_rank,
                            'condition_number': spec_diag.condition_number,
                            'largest_eigenvalue': float(spec_diag.eigenvalues[0]),
                        })
            except Exception:
                pass

    progress_bar.empty()

    # ─────────────────────────────────────────────────────────────────────
    # COMPUTE FINAL METRICS
    # ─────────────────────────────────────────────────────────────────────
    final_oos_perf = {}
    for name, data in oos_perf.items():
        metrics, _ = calculate_advanced_metrics(data['returns'])
        final_oos_perf[name] = {
            'returns': data['returns'],
            'metrics': metrics
        }

    if weight_entropies:
        final_oos_perf['System_Curated']['metrics']['avg_weight_entropy'] = np.mean(weight_entropies)

    # Tier-level performance from TRAIN window only (CRITICAL-2 fix:
    # using full historical_data here caused look-ahead bias — the tier
    # Sharpes included OOS data that the walk-forward loop had not yet seen).
    train_cutoff = max(MIN_TRAIN_DAYS, len(historical_data) * 2 // 3)
    full_history_subset_perf = _calculate_performance_on_window(
        historical_data[:train_cutoff], _strategies, TRAINING_CAPITAL, precomputed_cache=precalc_cache, cache_start_idx=0
    )['subset']

    logger.info("=" * 70)
    curated_metrics = final_oos_perf.get('System_Curated', {}).get('metrics', {})
    logger.info(f"WALK-FORWARD COMPLETE | System_Curated CAGR: {curated_metrics.get('annual_return', 0):.1%} | "
                 f"Sharpe: {curated_metrics.get('sharpe', 0):.2f} | MaxDD: {curated_metrics.get('max_drawdown', 0):.1%}")
    logger.info("=" * 70)

    # Spectral summary
    spectral_summary = {}
    if spectral_history:
        ar_vals = [s['absorption_ratio'] for s in spectral_history]
        eff_vals = [s['effective_rank'] for s in spectral_history]
        spectral_summary = {
            'mean_absorption_ratio': float(np.mean(ar_vals)),
            'ar_volatility': float(np.std(ar_vals)),
            'mean_effective_rank': float(np.mean(eff_vals)),
            'n_observations': len(spectral_history),
        }

    # REC-4: Conformal prediction intervals for strategy returns
    conformal_intervals = {}
    try:
        from rmt_core import conformal_strategy_intervals
        strat_returns_for_ci = {}
        for name, data in final_oos_perf.items():
            if name == 'System_Curated':
                continue
            ret_list = data.get('returns', [])
            if len(ret_list) >= 20:
                strat_returns_for_ci[name] = np.array(
                    [r['return'] if isinstance(r, dict) else r for r in ret_list], dtype=float
                )
        if strat_returns_for_ci:
            conformal_intervals = conformal_strategy_intervals(strat_returns_for_ci, alpha=0.1)
    except Exception:
        pass

    # REC-1: Strategy dimensionality reduction summary
    strategy_factors = {}
    try:
        from rmt_core import reduce_strategy_space
        strat_returns_for_pca = {}
        for name, data in final_oos_perf.items():
            if name == 'System_Curated':
                continue
            ret_list = data.get('returns', [])
            if len(ret_list) >= 20:
                strat_returns_for_pca[name] = np.array(
                    [r['return'] if isinstance(r, dict) else r for r in ret_list], dtype=float
                )
        if len(strat_returns_for_pca) >= 3:
            strategy_factors = reduce_strategy_space(strat_returns_for_pca)
    except Exception:
        pass

    return {
        'strategy': final_oos_perf,
        'subset': full_history_subset_perf,
        'strategy_weights_history': strategy_weights_history,
        'subset_weights_history': subset_weights_history,
        'backtest_mode': 'walk_forward_trigger',
        'trigger_config': trigger_config,
        'spectral_history': spectral_history,
        'spectral_summary': spectral_summary,
        'conformal_intervals': conformal_intervals,
        'strategy_factors': strategy_factors,
    }


def evaluate_historical_performance(
    _strategies: Dict[str, BaseStrategy],
    historical_data: List[Tuple[datetime, pd.DataFrame]],
    test_window_size: int = 50
) -> Dict:
    """
    Standard walk-forward evaluation WITHOUT trigger signals.
    Every day is a rebalancing day.
    """
    MIN_TRAIN_FILES = 2
    EMBARGO_DAYS = 1  # REC-2: embargo gap
    TRAINING_CAPITAL = 2500000.0

    if len(historical_data) < MIN_TRAIN_FILES + EMBARGO_DAYS + 1:
        st.error(f"Not enough historical data. Need at least {MIN_TRAIN_FILES + EMBARGO_DAYS + 1} files.")
        return {}

    all_names = list(_strategies.keys()) + ['System_Curated']
    oos_perf = {name: {'returns': []} for name in all_names}
    weight_entropies = []
    strategy_weights_history = []
    subset_weights_history = []
    spectral_history = []

    # --- O(T^2) Bottleneck Fix: Precompute all strategy returns once ---
    precalc_cache = {'strategy': {n: [] for n in _strategies}, 'subset': {n: {} for n in _strategies}}
    logger.info("Precomputing strategy returns for walk-forward cache...")
    
    try:
        from quant_core import IndicatorHistoryAccumulator
        global_acc = IndicatorHistoryAccumulator()
    except ImportError:
        global_acc = None
        
    for j in range(len(historical_data) - 1):
        df_date, df = historical_data[j]
        next_date, next_df = historical_data[j+1]
        if global_acc is not None:
            global_acc.update(df)
        for name, strategy in _strategies.items():
            try:
                if global_acc is not None:
                    strategy.accumulator = global_acc
                port = strategy.generate_portfolio(df, TRAINING_CAPITAL)
                ret = compute_portfolio_return(port, next_df) if not port.empty else 0.0
                precalc_cache['strategy'][name].append({'return': ret, 'date': next_date})
                if not port.empty:
                    num_tiers = len(port) // 10
                    for t in range(num_tiers):
                        t_name = f'tier_{t+1}'
                        if t_name not in precalc_cache['subset'][name]: precalc_cache['subset'][name][t_name] = []
                        t_port = port.iloc[t*10:(t+1)*10]
                        t_ret = compute_portfolio_return(t_port, next_df) if not t_port.empty else 0.0
                        precalc_cache['subset'][name][t_name].append({'return': t_ret, 'date': next_date})
            except Exception:
                precalc_cache['strategy'][name].append({'return': 0.0, 'date': next_date})
    # -------------------------------------------------------------------

    # Track previous portfolios for turnover-proportional cost (CRITICAL-3 wiring)
    last_curated_port = pd.DataFrame()
    last_strategy_ports: Dict[str, pd.DataFrame] = {}

    start_idx = max(MIN_TRAIN_FILES + EMBARGO_DAYS, len(historical_data) - test_window_size - 1)
    progress_bar = st.progress(0, text="Initializing backtest...")
    total_steps = len(historical_data) - start_idx - 1

    if total_steps <= 0:
        st.error(f"Not enough data for backtest steps. Need at least {MIN_TRAIN_FILES + EMBARGO_DAYS + 2} days.")
        progress_bar.empty()
        return {}

    for i in range(start_idx, len(historical_data) - 1):
        # REC-2: train on data[:i-EMBARGO], skip embargo gap, test at i
        # Execution Fix: Shift from 50-day Rolling Window to Expanding Window.
        # RMT and Sharpe estimators require maximum T/N ratio to converge accurately. 
        # The O(1) cache ensures this costs 0 extra computation time.
        train_start = 0
        train_window = historical_data[train_start : i - EMBARGO_DAYS]
        test_date, test_df = historical_data[i]
        next_date, next_df = historical_data[i + 1]

        if global_acc is not None:
            global_acc.update(test_df)

        step_idx = i - start_idx + 1
        progress_text = f"Processing step {step_idx}/{total_steps}"
        progress_bar.progress(step_idx / total_steps, text=progress_text)

        in_sample_perf = _calculate_performance_on_window(train_window, _strategies, TRAINING_CAPITAL, precalc_cache, cache_start_idx=train_start)

        try:
            curated_port, strat_wts, sub_wts, _ = curate_final_portfolio(
                _strategies, in_sample_perf, test_df, TRAINING_CAPITAL, 30, 1.0, 10.0
            )

            strategy_weights_history.append({'date': test_date, **strat_wts})
            subset_weights_history.append({'date': test_date, **sub_wts})

            if curated_port.empty:
                oos_perf['System_Curated']['returns'].append({'return': 0, 'date': next_date})
            else:
                prev_curated = last_curated_port if not last_curated_port.empty else None
                last_curated_port = curated_port.copy()
                oos_perf['System_Curated']['returns'].append({
                    'return': compute_portfolio_return(curated_port, next_df, is_rebalance=True, prev_portfolio=prev_curated, current_prices=test_df),
                    'date': next_date
                })
                weights = curated_port['weightage_pct'] / 100
                valid_w = weights[weights > 0]
                if len(valid_w) > 0:
                    entropy = -np.sum(valid_w * np.log2(valid_w))
                    weight_entropies.append(entropy)
        except Exception as e:
            logger.error(f"OOS Curation Error ({test_date.date()}): {e}")
            oos_perf['System_Curated']['returns'].append({'return': 0, 'date': next_date})

        for name, strategy in _strategies.items():
            try:
                if global_acc is not None:
                    strategy.accumulator = global_acc
                portfolio = strategy.generate_portfolio(test_df, TRAINING_CAPITAL)
                prev_strat_port = last_strategy_ports.get(name)
                if prev_strat_port is not None and prev_strat_port.empty:
                    prev_strat_port = None
                last_strategy_ports[name] = portfolio.copy() if not portfolio.empty else pd.DataFrame()
                oos_perf[name]['returns'].append({
                    'return': compute_portfolio_return(portfolio, next_df, is_rebalance=True, prev_portfolio=prev_strat_port, current_prices=test_df),
                    'date': next_date
                })
            except Exception as e:
                logger.error(f"OOS Strategy Error ({name}, {test_date.date()}): {e}")
                oos_perf[name]['returns'].append({'return': 0, 'date': next_date})

        # ─── SPECTRAL TRACKING (every 5th step) — uses return time-series ───
        step_count = i - MIN_TRAIN_FILES - EMBARGO_DAYS
        if step_count % 5 == 0 and i >= 20:
            try:
                from rmt_core import compute_spectral_diagnostics
                strat_keys = list(_strategies.keys())
                ret_cols = {}
                for sn in strat_keys:
                    r_list = oos_perf[sn]['returns']
                    if len(r_list) >= 10:
                        ret_cols[sn] = np.array([
                            r['return'] if isinstance(r, dict) else r for r in r_list
                        ], dtype=float)
                if len(ret_cols) >= 2:
                    min_len = min(len(v) for v in ret_cols.values())
                    if min_len >= 10:
                        ret_matrix = np.column_stack([v[:min_len] for v in ret_cols.values()])
                        ret_matrix = np.nan_to_num(ret_matrix, nan=0.0)
                        spec_diag = compute_spectral_diagnostics(ret_matrix)
                        spectral_history.append({
                            'date': test_date,
                            'absorption_ratio': spec_diag.absorption_ratio,
                            'effective_rank': spec_diag.effective_rank,
                            'condition_number': spec_diag.condition_number,
                            'largest_eigenvalue': float(spec_diag.eigenvalues[0]),
                        })
            except Exception:
                pass

    progress_bar.empty()

    final_oos_perf = {}
    for name, data in oos_perf.items():
        metrics, _ = calculate_advanced_metrics(data['returns'])
        final_oos_perf[name] = {
            'returns': data['returns'],
            'metrics': metrics
        }
    
    if weight_entropies:
        final_oos_perf['System_Curated']['metrics']['avg_weight_entropy'] = np.mean(weight_entropies)
    
    # CRITICAL-2 fix: use train-only window for tier Sharpe calculation
    train_cutoff = max(MIN_TRAIN_FILES, len(historical_data) * 2 // 3)
    full_history_subset_perf = _calculate_performance_on_window(
        historical_data[:train_cutoff], _strategies, TRAINING_CAPITAL, precomputed_cache=precalc_cache, cache_start_idx=0
    )['subset']

    # Spectral summary (market correlation regime)
    spectral_summary = {}
    if spectral_history:
        ar_vals = [s['absorption_ratio'] for s in spectral_history]
        eff_vals = [s['effective_rank'] for s in spectral_history]
        spectral_summary = {
            'mean_absorption_ratio': float(np.mean(ar_vals)),
            'ar_volatility': float(np.std(ar_vals)) if len(ar_vals) > 1 else 0.0,
            'mean_effective_rank': float(np.mean(eff_vals)),
            'n_observations': len(spectral_history),
        }

    # Cross-strategy spectral metrics (strategy independence)
    cross_strategy_spectral = {}
    try:
        from rmt_core import detect_redundant_strategies
        strat_returns = {}
        for name, data in final_oos_perf.items():
            if name == 'System_Curated':
                continue
            ret_list = data.get('returns', [])
            if len(ret_list) >= 20:
                rets = np.array([r['return'] if isinstance(r, dict) else r for r in ret_list], dtype=float)
                rets = np.nan_to_num(rets, nan=0.0)
                strat_returns[name] = rets
        if len(strat_returns) >= 2:
            redundancy = detect_redundant_strategies(strat_returns)
            cross_strategy_spectral = {
                'effective_strategy_count': redundancy['effective_strategy_count'],
                'noise_fraction': redundancy['noise_fraction'],
                'strategy_clusters': redundancy['clusters'],
            }
    except Exception:
        pass

    # REC-4: Conformal prediction intervals for strategy returns
    conformal_intervals = {}
    try:
        from rmt_core import conformal_strategy_intervals
        strat_returns_for_ci = {}
        for name, data in final_oos_perf.items():
            if name == 'System_Curated':
                continue
            ret_list = data.get('returns', [])
            if len(ret_list) >= 20:
                strat_returns_for_ci[name] = np.array(
                    [r['return'] if isinstance(r, dict) else r for r in ret_list], dtype=float
                )
        if strat_returns_for_ci:
            conformal_intervals = conformal_strategy_intervals(strat_returns_for_ci, alpha=0.1)
    except Exception:
        pass

    # REC-1: Strategy dimensionality reduction summary
    strategy_factors = {}
    try:
        from rmt_core import reduce_strategy_space
        strat_returns_for_pca = {}
        for name, data in final_oos_perf.items():
            if name == 'System_Curated':
                continue
            ret_list = data.get('returns', [])
            if len(ret_list) >= 20:
                strat_returns_for_pca[name] = np.array(
                    [r['return'] if isinstance(r, dict) else r for r in ret_list], dtype=float
                )
        if len(strat_returns_for_pca) >= 3:
            strategy_factors = reduce_strategy_space(strat_returns_for_pca)
    except Exception:
        pass

    return {
        'strategy': final_oos_perf,
        'subset': full_history_subset_perf,
        'strategy_weights_history': strategy_weights_history,
        'subset_weights_history': subset_weights_history,
        'spectral_history': spectral_history,
        'spectral_summary': spectral_summary,
        'cross_strategy_spectral': cross_strategy_spectral,
        'conformal_intervals': conformal_intervals,
        'strategy_factors': strategy_factors,
    }


def curate_final_portfolio(strategies: Dict[str, BaseStrategy], performance: Dict, current_df: pd.DataFrame, sip_amount: float, num_positions: int, min_pos_pct: float, max_pos_pct: float) -> Tuple[pd.DataFrame, Dict, Dict, Optional[float]]:
    # Build returns_data for RMT weight methods
    returns_data = {}
    for name in strategies:
        strat_data = performance.get('strategy', {}).get(name, {})
        if isinstance(strat_data, dict) and 'returns' in strat_data:
            ret_list = strat_data['returns']
            if isinstance(ret_list, list) and len(ret_list) >= 10:
                # Extract numeric returns from list of dicts
                rets = np.array([r['return'] if isinstance(r, dict) else r for r in ret_list], dtype=float)
                rets = np.nan_to_num(rets, nan=0.0)
                if len(rets) >= 10:
                    returns_data[name] = rets

    strategy_weights = calculate_strategy_weights(
        performance,
        method='hrp' if len(returns_data) >= 2 else 'softmax_sharpe',
        returns_data=returns_data if returns_data else None,
    )
    subset_weights = {}
    for name in strategies:
        sub_perfs = performance.get('subset', {}).get(name, {})
        tier_names = sorted(sub_perfs.keys())
        if not tier_names:
            subset_weights[name] = {}
            continue

        tier_sharpes = np.array([sub_perfs.get(tier, 1.0 - (int(tier.split('_')[1]) * 0.05)) + 2 for tier in tier_names])
        
        if tier_sharpes.size == 0:
            subset_weights[name] = {}
            continue

        stable_sharpes = tier_sharpes - np.max(tier_sharpes)
        exp_sharpes = np.exp(stable_sharpes)
        total_exp = np.sum(exp_sharpes)

        if total_exp > 0 and np.isfinite(total_exp):
            subset_weights[name] = {tier: exp_sharpes[i] / total_exp for i, tier in enumerate(tier_names)}
        else:
            equal_weight = 1.0 / len(tier_names) if tier_names else 0
            subset_weights[name] = {tier: equal_weight for tier in tier_names}

    # H-3 FIX: Additive capital decomposition instead of multiplicative weight composition.
    # Each strategy gets a capital slice (strategy_weight * sip_amount).
    # Within each strategy, each tier gets a slice (tier_weight * strategy_capital).
    # Within each tier, stocks are weighted by their relative weightage_pct (already sums to ~100%).
    # This prevents the triple-product compression that made final weights dominated
    # by strategy-level weights rather than stock-level conviction.
    aggregated_holdings = {}
    for name, strategy in strategies.items():
        strat_w = strategy_weights.get(name, 0)
        if strat_w <= 0:
            continue
        port = strategy.generate_portfolio(current_df, sip_amount)
        if port.empty:
            continue
        n, tier_size = len(port), 10
        num_tiers = n // tier_size
        if num_tiers == 0:
            continue

        # Total tier weight for this strategy (for normalization within strategy)
        tier_ws = subset_weights.get(name, {})
        tier_total = sum(tier_ws.get(f'tier_{j+1}', 0) for j in range(num_tiers))
        if tier_total <= 0:
            tier_total = 1.0

        for j in range(num_tiers):
            tier_name = f'tier_{j+1}'
            if tier_name not in tier_ws:
                continue
            sub_df = port.iloc[j*tier_size:(j+1)*tier_size]
            tier_weight = tier_ws[tier_name] / tier_total  # normalized within strategy

            # Stock weights within this tier: normalize to sum to 1
            tier_stock_weights = sub_df['weightage_pct'].values.copy()
            tier_stock_total = tier_stock_weights.sum()
            if tier_stock_total > 0:
                tier_stock_weights = tier_stock_weights / tier_stock_total
            else:
                tier_stock_weights = np.ones(len(sub_df)) / len(sub_df)

            for k, (_, row) in enumerate(sub_df.iterrows()):
                symbol, price = row['symbol'], row['price']
                # Capital allocation: strategy_share × tier_share × stock_share
                final_weight = strat_w * tier_weight * tier_stock_weights[k]
                if symbol in aggregated_holdings:
                    aggregated_holdings[symbol]['weight'] += final_weight
                else:
                    aggregated_holdings[symbol] = {'price': price, 'weight': final_weight}
    if not aggregated_holdings:
        return pd.DataFrame(), {}, {}, None
        
    final_port = pd.DataFrame([{'symbol': s, **d} for s, d in aggregated_holdings.items()]).sort_values('weight', ascending=False).head(num_positions)
    total_weight = final_port['weight'].sum()
    
    if total_weight > 0:
        final_port['weightage_pct'] = final_port['weight'] * 100 / total_weight
    else:
        final_port['weightage_pct'] = 100.0 / len(final_port)
        
    final_port['weightage_pct'] = final_port['weightage_pct'].clip(lower=min_pos_pct, upper=max_pos_pct)
    weight_sum = final_port['weightage_pct'].sum()
    if weight_sum > 0:
        final_port['weightage_pct'] = (final_port['weightage_pct'] / weight_sum) * 100
        
    safe_price = np.where(final_port['price'] > 0, final_port['price'], 1e-10)
    final_port['units'] = np.floor((sip_amount * final_port['weightage_pct'] / 100) / safe_price)
    final_port['value'] = final_port['units'] * final_port['price']
    
    final_port_df = final_port.sort_values('weightage_pct', ascending=False).reset_index(drop=True)

    # Compute diversification ratio using RMT-cleaned covariance
    diversification_ratio = None
    try:
        from rmt_core import compute_spectral_diagnostics, compute_diversification_ratio
        strat_names = list(strategies.keys())
        # Reuse returns_data already built above
        strat_returns = {n: returns_data[n] for n in strat_names if n in returns_data}
        if len(strat_returns) >= 2:
            available = [n for n in strat_names if n in strat_returns]
            min_len = min(len(strat_returns[n]) for n in available)
            returns_matrix = np.column_stack([strat_returns[n][:min_len] for n in available])
            diagnostics = compute_spectral_diagnostics(returns_matrix)
            vols = np.std(returns_matrix, axis=0)
            vols = np.where(vols > 1e-10, vols, 1e-10)
            cleaned_cov = diagnostics.cleaned_corr * np.outer(vols, vols)
            weights_vec = np.array([strategy_weights.get(n, 0) for n in available])
            if weights_vec.sum() > 0:
                weights_vec = weights_vec / weights_vec.sum()
                diversification_ratio = compute_diversification_ratio(weights_vec, cleaned_cov)
    except Exception:
        pass

    return final_port_df, strategy_weights, subset_weights, diversification_ratio

# --- NEW: Production-Grade Market Regime Detection System (v2 - Corrected Logic) ---
class MarketRegimeDetectorV2:
    """
    Institutional-grade market regime detection (v2) with corrected scoring and
    classification logic.
    """
    
    def __init__(self):
        self.regime_thresholds = {
            'CRISIS': {'score': -1.0, 'confidence': 0.85},
            'BEAR': {'score': -0.5, 'confidence': 0.75},
            'WEAK_BEAR': {'score': -0.1, 'confidence': 0.65},
            'CHOP': {'score': 0.1, 'confidence': 0.60},
            'WEAK_BULL': {'score': 0.5, 'confidence': 0.65},
            'BULL': {'score': 1.0, 'confidence': 0.75},
            'STRONG_BULL': {'score': 1.5, 'confidence': 0.85},
        }
    
    def detect_regime(self, historical_data: list) -> Tuple[str, str, float, Dict]:
        if len(historical_data) < 10:
            return "INSUFFICIENT_DATA", "🐂 Bull Market Mix", 0.3, {}
        
        analysis_window = historical_data[-10:]
        latest_date, latest_df = analysis_window[-1]
        
        metrics = {
            'momentum': self._analyze_momentum_regime(analysis_window),
            'trend': self._analyze_trend_quality(analysis_window),
            'breadth': self._analyze_market_breadth(latest_df),
            'volatility': self._analyze_volatility_regime(analysis_window),
            'extremes': self._analyze_statistical_extremes(latest_df),
            'correlation': self._analyze_correlation_regime(latest_df),
            'velocity': self._analyze_velocity(analysis_window)
        }
        
        regime_score = self._calculate_composite_score(metrics)
        regime_name, confidence = self._classify_regime(regime_score, metrics)
        mix_name = self._map_regime_to_mix(regime_name)
        explanation = self._generate_explanation(regime_name, confidence, metrics, regime_score)
        
        return regime_name, mix_name, confidence, {
            'score': regime_score,
            'metrics': metrics,
            'explanation': explanation,
            'analysis_date': latest_date.strftime('%Y-%m-%d')
        }

    def _analyze_momentum_regime(self, window: list) -> Dict:
        # HIGH-1 fix: separate cross-sectional BREADTH from time-series
        # MOMENTUM.  The cross-sectional mean of individual RSIs is a
        # breadth indicator, NOT a momentum indicator.  For momentum we
        # use the median price change (a proxy for a cap-equal index
        # return) and its Theil-Sen trend.
        #
        # Breadth: fraction of stocks with RSI > 50 (participation)
        # Momentum: median % change trend (market direction)
        breadth_values = [(df['rsi latest'] > 50).mean() for _, df in window]
        pct_change_medians = [df['% change'].median() for _, df in window]

        current_breadth = breadth_values[-1]
        breadth_trend = theilslopes(breadth_values, range(len(breadth_values)))[0]

        # Momentum from actual price returns (not RSI levels)
        current_momentum = pct_change_medians[-1]
        momentum_trend = theilslopes(pct_change_medians, range(len(pct_change_medians)))[0]

        # Also keep OSC for supplementary signal
        osc_values = [df['osc latest'].mean() for _, df in window]
        current_osc = osc_values[-1]
        osc_trend = theilslopes(osc_values, range(len(osc_values)))[0]

        # Classification using BOTH breadth and momentum
        if current_breadth > 0.65 and momentum_trend > 0 and current_momentum > 0:
            strength, score = 'STRONG_BULLISH', 2.0
        elif current_breadth > 0.55 and momentum_trend >= 0:
            strength, score = 'BULLISH', 1.0
        elif current_breadth < 0.30 and momentum_trend < 0 and current_momentum < 0:
            strength, score = 'STRONG_BEARISH', -2.0
        elif current_breadth < 0.45 and momentum_trend <= 0:
            strength, score = 'BEARISH', -1.0
        else:
            strength, score = 'NEUTRAL', 0.0

        return {
            'strength': strength, 'score': score,
            'current_rsi': current_breadth * 100,  # backward-compat: scale to 0-100
            'rsi_trend': breadth_trend,
            'current_breadth': current_breadth,
            'momentum_trend': momentum_trend,
            'current_osc': current_osc, 'osc_trend': osc_trend
        }

    def _analyze_trend_quality(self, window: list) -> Dict:
        above_ma200_pct = [(df['price'] > df['ma200 latest']).mean() for _, df in window]
        ma_alignment = [(df['ma90 latest'] > df['ma200 latest']).mean() for _, df in window]
        
        current_above_200 = above_ma200_pct[-1]
        current_alignment = ma_alignment[-1]
        trend_consistency = theilslopes(above_ma200_pct, range(len(above_ma200_pct)))[0]
        
        if current_above_200 > 0.75 and current_alignment > 0.70 and trend_consistency >= 0:
            quality, score = 'STRONG_UPTREND', 2.0
        elif current_above_200 > 0.60 and current_alignment > 0.55:
            quality, score = 'UPTREND', 1.0
        elif current_above_200 < 0.30 and current_alignment < 0.30 and trend_consistency < 0:
            quality, score = 'STRONG_DOWNTREND', -2.0
        elif current_above_200 < 0.45 and current_alignment < 0.45:
            quality, score = 'DOWNTREND', -1.0
        else:
            quality, score = 'TRENDLESS', 0.0
            
        return {'quality': quality, 'score': score, 'above_200dma': current_above_200, 'ma_alignment': current_alignment, 'trend_consistency': trend_consistency}

    def _analyze_market_breadth(self, df: pd.DataFrame) -> Dict:
        rsi_bullish = (df['rsi latest'] > 50).mean()
        osc_positive = (df['osc latest'] > 0).mean()
        rsi_weak = (df['rsi latest'] < 40).mean()
        osc_oversold = (df['osc latest'] < -50).mean()
        divergence = abs(rsi_bullish - osc_positive)
        
        if rsi_bullish > 0.70 and osc_positive > 0.60 and divergence < 0.15:
            quality, score = 'STRONG_BROAD', 2.0
        elif rsi_bullish > 0.55 and osc_positive > 0.45:
            quality, score = 'HEALTHY', 1.0
        elif rsi_weak > 0.60 and osc_oversold > 0.50:
            quality, score = 'CAPITULATION', -2.0
        elif rsi_weak > 0.45 and osc_oversold > 0.35:
            quality, score = 'WEAK', -1.0
        elif divergence > 0.25:
            quality, score = 'DIVERGENT', -0.5
        else:
            quality, score = 'MIXED', 0.0
            
        return {'quality': quality, 'score': score, 'rsi_bullish_pct': rsi_bullish, 'osc_positive_pct': osc_positive, 'divergence': divergence}

    def _analyze_volatility_regime(self, window: list) -> Dict:
        # HIGH-4 fix: use np.where to avoid division by near-zero MA20
        # (instruments with MA20 < 1.0 would produce enormous BBW with +1e-6 guard)
        bb_widths = []
        for _, df in window:
            ma20 = df['ma20 latest'].values
            dev20 = df['dev20 latest'].values
            safe_ma20 = np.where(np.abs(ma20) > 1.0, ma20, np.nan)
            bbw = (4 * dev20) / safe_ma20
            bb_widths.append(float(np.nanmean(bbw)))
        current_bbw = bb_widths[-1]
        vol_trend = theilslopes(bb_widths, range(len(bb_widths)))[0]
        
        if current_bbw < 0.08 and vol_trend < 0:
            regime, score = 'SQUEEZE', 0.5 
        elif current_bbw > 0.15 and vol_trend > 0:
            regime, score = 'PANIC', -1.0 
        elif current_bbw > 0.12:
            regime, score = 'ELEVATED', -0.5
        else:
            regime, score = 'NORMAL', 0.0
            
        return {'regime': regime, 'score': score, 'current_bbw': current_bbw, 'vol_trend': vol_trend}

    def _analyze_statistical_extremes(self, df: pd.DataFrame) -> Dict:
        extreme_oversold = (df['zscore latest'] < -2.0).mean()
        extreme_overbought = (df['zscore latest'] > 2.0).mean()
        
        if extreme_oversold > 0.40:
            extreme_type, score = 'DEEPLY_OVERSOLD', 1.5 
        elif extreme_overbought > 0.40:
            extreme_type, score = 'DEEPLY_OVERBOUGHT', -1.5
        elif extreme_oversold > 0.20:
            extreme_type, score = 'OVERSOLD', 0.75
        elif extreme_overbought > 0.20:
            extreme_type, score = 'OVERBOUGHT', -0.75
        else:
            extreme_type, score = 'NORMAL', 0.0
            
        return {'type': extreme_type, 'score': score, 'zscore_extreme_oversold_pct': extreme_oversold, 'zscore_extreme_overbought_pct': extreme_overbought}

    def _analyze_correlation_regime(self, df: pd.DataFrame) -> Dict:
        """
        Analyze cross-sectional correlation structure using spectral analysis.

        Uses the absorption ratio from RMT eigendecomposition of the
        cross-sectional indicator matrix. The absorption ratio measures what
        fraction of total variance is captured by the top eigenvectors —
        a direct measure of correlation concentration (herding).

        Falls back to the indicator-agreement heuristic when RMT is unavailable
        or the cross-section is too small.
        """
        try:
            from rmt_core import compute_spectral_diagnostics

            indicator_cols = ['rsi latest', 'osc latest', 'zscore latest', 'rsi weekly', 'osc weekly']
            available_cols = [c for c in indicator_cols if c in df.columns]

            if len(available_cols) >= 3 and len(df) >= 8:
                indicator_matrix = df[available_cols].ffill().fillna(df[available_cols].median()).values
                diagnostics = compute_spectral_diagnostics(indicator_matrix)
                ar = diagnostics.absorption_ratio

                if ar > 0.7:
                    regime, score = 'HIGH_CORRELATION', -0.5
                elif ar < 0.4:
                    regime, score = 'LOW_CORRELATION', 0.5
                else:
                    regime, score = 'NORMAL', 0.0

                return {
                    'regime': regime,
                    'score': score,
                    'correlation_score': ar,
                    'dispersion': 1 - ar,
                    'indicator_agreement': ar,
                    'effective_factors': diagnostics.effective_rank,
                    'spectral_diagnostics': diagnostics,
                }
        except Exception:
            pass

        return self._fallback_correlation_regime(df)

    def _fallback_correlation_regime(self, df: pd.DataFrame) -> Dict:
        """Original indicator-agreement heuristic as fallback."""
        rsi_median = df['rsi latest'].median()
        osc_median = df['osc latest'].median()

        rsi_above = (df['rsi latest'] > rsi_median).mean()
        rsi_agreement = max(rsi_above, 1 - rsi_above)

        osc_above = (df['osc latest'] > osc_median).mean()
        osc_agreement = max(osc_above, 1 - osc_above)

        both_oversold = ((df['rsi latest'] < 40) & (df['osc latest'] < -30)).mean()
        both_overbought = ((df['rsi latest'] > 60) & (df['osc latest'] > 30)).mean()
        indicator_agreement = both_oversold + both_overbought

        rsi_dispersion = df['rsi latest'].std() / 50
        osc_dispersion = df['osc latest'].std() / 100
        avg_dispersion = (rsi_dispersion + osc_dispersion) / 2

        correlation_score = (rsi_agreement + osc_agreement) / 2 * (1 - avg_dispersion) + indicator_agreement * 0.3
        correlation_score = np.clip(correlation_score, 0, 1)

        if correlation_score > 0.7:
            regime, score = 'HIGH_CORRELATION', -0.5
        elif correlation_score < 0.4:
            regime, score = 'LOW_CORRELATION', 0.5
        else:
            regime, score = 'NORMAL', 0.0

        return {
            'regime': regime,
            'score': score,
            'correlation_score': correlation_score,
            'dispersion': avg_dispersion,
            'indicator_agreement': indicator_agreement,
        }

    def _analyze_velocity(self, window: list) -> Dict:
        """
        Analyze momentum velocity and acceleration.
        
        Velocity: First derivative of RSI (rate of change)
        Acceleration: Second derivative (rate of change of velocity)
        
        Positive acceleration with positive velocity = strengthening momentum
        Negative acceleration with positive velocity = momentum fading
        """
        if len(window) < 5: 
            return {'acceleration': 'UNKNOWN', 'score': 0.0, 'avg_velocity': 0.0, 'acceleration_value': 0.0}
        
        recent_rsis = np.array([w[1]['rsi latest'].mean() for w in window[-5:]])
        
        # Velocity: First differences (first derivative)
        velocity = np.diff(recent_rsis)  # 4 values
        avg_velocity = np.mean(velocity)
        current_velocity = velocity[-1]
        
        # Acceleration: Second differences (second derivative)
        acceleration_values = np.diff(velocity)  # 3 values
        avg_acceleration = np.mean(acceleration_values)
        current_acceleration = acceleration_values[-1]
        
        # Classification based on velocity and acceleration
        if avg_velocity > 1.5 and current_acceleration > 0:
            velocity_regime, score = 'ACCELERATING_UP', 1.5
        elif avg_velocity > 1.0 and current_acceleration >= 0:
            velocity_regime, score = 'RISING_FAST', 1.0
        elif avg_velocity > 0.5:
            velocity_regime, score = 'RISING', 0.5
        elif avg_velocity < -1.5 and current_acceleration < 0:
            velocity_regime, score = 'ACCELERATING_DOWN', -1.5
        elif avg_velocity < -1.0 and current_acceleration <= 0:
            velocity_regime, score = 'FALLING_FAST', -1.0
        elif avg_velocity < -0.5:
            velocity_regime, score = 'FALLING', -0.5
        elif abs(avg_velocity) < 0.5 and abs(current_acceleration) > 0.5:
            # Momentum building from stable base
            if current_acceleration > 0:
                velocity_regime, score = 'COILING_UP', 0.3
            else:
                velocity_regime, score = 'COILING_DOWN', -0.3
        else:
            velocity_regime, score = 'STABLE', 0.0
            
        return {
            'acceleration': velocity_regime, 
            'score': score, 
            'avg_velocity': avg_velocity,
            'current_velocity': current_velocity,
            'acceleration_value': current_acceleration
        }

    def _calculate_composite_score(self, metrics: Dict) -> float:
        weights = { 'momentum': 0.25, 'trend': 0.25, 'breadth': 0.15, 'volatility': 0.05, 'extremes': 0.10, 'correlation': 0.10, 'velocity': 0.10 }
        return sum(metrics[factor]['score'] * weight for factor, weight in weights.items())
    
    def _classify_regime(self, score: float, metrics: Dict) -> Tuple[str, float]:
        if metrics['volatility']['regime'] == 'PANIC' and score < -0.5 and metrics['breadth']['quality'] == 'CAPITULATION':
            return 'CRISIS', 0.90
            
        sorted_thresholds = sorted(self.regime_thresholds.items(), key=lambda item: item[1]['score'])
        
        for regime, threshold in reversed(sorted_thresholds):
            if score >= threshold['score']:
                confidence = threshold['confidence'] * 0.75 if metrics['breadth']['quality'] == 'DIVERGENT' else threshold['confidence']
                return regime, confidence

        return 'CRISIS', 0.85
    
    def _map_regime_to_mix(self, regime: str) -> str:
        mapping = {
            'STRONG_BULL': 'Bull Market Mix', 'BULL': 'Bull Market Mix',
            'WEAK_BULL': 'Chop/Consolidate Mix', 'CHOP': 'Chop/Consolidate Mix',
            'WEAK_BEAR': 'Chop/Consolidate Mix', 'BEAR': 'Bear Market Mix',
            'CRISIS': 'Bear Market Mix'
        }
        return mapping.get(regime, 'Chop/Consolidate Mix')
    
    def _generate_explanation(self, regime: str, confidence: float, metrics: Dict, score: float) -> str:
        lines = [f"**Detected Regime:** {regime} (Score: {score:.2f}, Confidence: {confidence:.0%})", ""]
        rationales = {
            'STRONG_BULL': "Strong upward momentum with broad participation. Favor momentum strategies.",
            'BULL': "Positive trend with healthy breadth. Conditions support growth strategies.",
            'WEAK_BULL': "Uptrend showing signs of fatigue or divergence. Rotate to defensive positions.",
            'CHOP': "No clear directional bias. Favors mean reversion and relative value strategies.",
            'WEAK_BEAR': "Downtrend developing. Begin defensive positioning.",
            'BEAR': "Established downtrend with weak breadth. Favor defensive strategies.",
            'CRISIS': "Severe market stress. Focus on capital preservation and oversold opportunities."
        }
        lines.append(f"**Rationale:** {rationales.get(regime, 'Market conditions unclear.')}")
        if metrics['breadth']['quality'] == 'DIVERGENT':
            lines.append("⚠️ **Warning:** Breadth divergence detected - narrow leadership may not be sustainable.")
        lines.append("\n**Key Factors:**")
        lines.append(f"• **Momentum:** {metrics['momentum']['strength']} (RSI: {metrics['momentum']['current_rsi']:.1f})")
        lines.append(f"• **Trend:** {metrics['trend']['quality']} ({metrics['trend']['above_200dma']:.0%} > 200DMA)")
        lines.append(f"• **Breadth:** {metrics['breadth']['quality']} ({metrics['breadth']['rsi_bullish_pct']:.0%} bullish)")
        lines.append(f"• **Volatility:** {metrics['volatility']['regime']} (BBW: {metrics['volatility']['current_bbw']:.3f})")
        if metrics['extremes']['type'] != 'NORMAL':
            lines.append(f"• **Extremes:** {metrics['extremes']['type']} detected")
        return "\n".join(lines)

@st.cache_data(ttl=3600, show_spinner=False)
def get_market_mix_suggestion_v3(end_date: datetime) -> Tuple[str, str, float, Dict]:
    detector = MarketRegimeDetectorV2()
    regime_days_to_fetch = int(MAX_INDICATOR_PERIOD * 1.5) + 30 
    fetch_start_date = end_date - timedelta(days=regime_days_to_fetch)
    
    try:
        historical_data = generate_historical_data(
            symbols_to_process=SYMBOLS_UNIVERSE,
            start_date=fetch_start_date,
            end_date=end_date
        )
        
        if len(historical_data) < 10:
            return (
                "Bull Market Mix",
                "⚠️ Insufficient historical data (< 10 periods). Defaulting to Bull Mix.",
                0.30, {}
            )
            
        regime_name, mix_name, confidence, details = detector.detect_regime(historical_data)
        return mix_name, details['explanation'], confidence, details

    except Exception as e:
        logger.error(f"Error in get_market_mix_suggestion_v3: {e}")
        return (
            "Bull Market Mix",
            f"⚠️ Error during regime detection: {e}. Defaulting to Bull Mix.",
            0.30, {}
        )

# =========================================================================
# --- Live Data Loading Function (Refactored) ---
@st.cache_data(ttl=3600, show_spinner=False)
def load_historical_data(end_date: datetime, lookback_files: int) -> List[Tuple[datetime, pd.DataFrame]]:
    """Fetches and processes all historical data on-the-fly."""
    logger.info(f"--- START: Live Data Generation (End Date: {end_date.date()}, Lookback: {lookback_files}) ---")
    total_days_to_fetch = int((lookback_files + MAX_INDICATOR_PERIOD) * 1.5) + 30
    fetch_start_date = end_date - timedelta(days=total_days_to_fetch)
    logger.info(f"Calculated fetch start date: {fetch_start_date.date()}")

    try:
        live_data = generate_historical_data(SYMBOLS_UNIVERSE, fetch_start_date, end_date)
        logger.info(f"--- SUCCESS: {len(live_data)} total valid days generated. ---")
        return live_data
    except Exception as e:
        logger.error(f"Error during load_historical_data: {e}")
        st.error(f"Failed to fetch or process live data: {e}")
        return []


# --- UI & Visualization Functions ---

# ═══════════════════════════════════════════════════════════════════════════════
# UI PRIMITIVES — Helpers shared by all tabs
# ═══════════════════════════════════════════════════════════════════════════════

def _section_header(title: str, subtitle: str = "") -> str:
    sub = f"<p class='section-subtitle'>{subtitle}</p>" if subtitle else ""
    return f"""<div class='section'><div class='section-header'><h3 class='section-title'>{title}</h3>{sub}</div></div>"""

def _section_divider():
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

def _metric_card(label: str, value: str, sub: str = "", cls: str = "neutral") -> str:
    sub_html = f"<div class='sub-metric'>{sub}</div>" if sub else ""
    return f"""<div class='metric-card {cls}'><h4>{label}</h4><h2>{value}</h2>{sub_html}</div>"""

def _render_cards(cards_list):
    """Render a row of metric cards from a list of (label, value, sub, cls) tuples."""
    cols = st.columns(min(len(cards_list), 6))
    for i, (label, val, sub, cls) in enumerate(cards_list):
        with cols[i % len(cols)]:
            st.markdown(_metric_card(label, val, sub, cls), unsafe_allow_html=True)

def _plot_area_evolution(weight_history: List[Dict], y_format: str = ".0%"):
    fig = create_weight_evolution_chart(weight_history)
    if fig.data:
        st.plotly_chart(fig, width='stretch')

def _build_returns_df(performance: Dict) -> pd.DataFrame:
    frames = {}
    for name, perf in performance.get('strategy', {}).items():
        if perf.get('returns'):
            df_raw = pd.DataFrame(perf['returns'])
            df = df_raw.drop_duplicates(subset='date', keep='last').set_index('date')
            frames[name] = df['return']
    return pd.DataFrame(frames) if frames else pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════

def display_performance_metrics(performance: Dict):
    if not performance:
        st.warning("Performance data not available. Please run an analysis.")
        return

    curated = performance.get('strategy', {}).get('System_Curated', {})
    m = curated.get('metrics', {})
    rets = curated.get('returns', [])

    # ── Hero Metrics ──
    st.markdown(_section_header("Performance Overview",
                                "System Curated portfolio — walk-forward out-of-sample results"), unsafe_allow_html=True)
    _render_cards([
        ("CAGR",         f"{m.get('annual_return',0):.1%}",  "Annualized return",
         'success' if m.get('annual_return',0) > 0 else 'danger'),
        ("Total Return", f"{m.get('total_return',0):.1%}",   "Cumulative",
         'success' if m.get('total_return',0) > 0 else 'danger'),
        ("Volatility",   f"{m.get('volatility',0):.1%}",     "Annualized σ",
         'warning' if m.get('volatility',0) > 0.20 else 'info'),
        ("Max Drawdown", f"{m.get('max_drawdown',0):.1%}",   "Peak-to-trough",
         'danger' if m.get('max_drawdown',0) < -0.10 else 'warning' if m.get('max_drawdown',0) < -0.05 else 'success'),
        ("Sharpe",       f"{m.get('sharpe',0):.2f}",          "Risk-adj. return",
         'success' if m.get('sharpe',0) > 1 else 'warning' if m.get('sharpe',0) > 0 else 'danger'),
        ("Sortino",      f"{m.get('sortino',0):.2f}",         "Downside-adj.",
         'success' if m.get('sortino',0) > 1 else 'warning' if m.get('sortino',0) > 0 else 'danger'),
    ])

    # ── Equity Curve & Drawdown ──
    if rets:
        _section_divider()
        st.markdown(_section_header("Equity Curve & Drawdown",
                                    "Growth of ₹1 investment with underwater periods"), unsafe_allow_html=True)
        df_ret = pd.DataFrame(rets).sort_values('date')
        fig_eq = create_equity_drawdown_chart(df_ret, date_col='date', return_col='return')
        st.plotly_chart(fig_eq, width='stretch')

    # ── Extended Risk Metrics ──
    _section_divider()
    st.markdown(_section_header("Extended Risk Metrics",
                                "Higher-order risk-adjusted performance ratios"), unsafe_allow_html=True)
    _render_cards([
        ("Calmar",        f"{m.get('calmar',0):.2f}",        "CAGR / MaxDD",
         'success' if m.get('calmar',0) > 1 else 'warning' if m.get('calmar',0) > 0 else 'danger'),
        ("Omega",         f"{m.get('omega_ratio',1):.2f}",   "Gain/loss ratio",
         'success' if m.get('omega_ratio',1) > 1.5 else 'warning' if m.get('omega_ratio',1) > 1 else 'danger'),
        ("Win Rate",      f"{m.get('win_rate',0):.0%}",      "Batting average",
         'success' if m.get('win_rate',0) > 0.55 else 'warning' if m.get('win_rate',0) > 0.45 else 'danger'),
        ("Profit Factor", f"{m.get('profit_factor',1):.2f}",  "Gross win/loss",
         'success' if m.get('profit_factor',1) > 1.5 else 'warning' if m.get('profit_factor',1) > 1 else 'danger'),
        ("Tail Ratio",    f"{m.get('tail_ratio',1):.2f}",     "Right/left tail",
         'info' if m.get('tail_ratio',1) > 1 else 'warning'),
        ("Gain/Pain",     f"{m.get('gain_to_pain',0):.2f}",   "Net efficiency",
         'success' if m.get('gain_to_pain',0) > 0.5 else 'warning' if m.get('gain_to_pain',0) > 0 else 'danger'),
    ])

    # ── Rolling Risk-Adjusted Performance ──
    if rets and len(rets) >= 5:
        _section_divider()
        st.markdown(_section_header("Rolling Risk-Adjusted Performance",
                                    "Time-varying Sharpe & Sortino to detect regime shifts"), unsafe_allow_html=True)
        df_ret = pd.DataFrame(rets).sort_values('date')
        win = max(3, len(df_ret) // 5)
        fig_roll = create_rolling_metrics_chart(df_ret, window=win, date_col='date', return_col='return')
        st.plotly_chart(fig_roll, width='stretch')

    # ── Strategy Attribution Table ──
    _section_divider()
    st.markdown(_section_header("Strategy Attribution",
                                "Walk-forward performance comparison across all strategies"), unsafe_allow_html=True)
    rows = []
    for name, perf in performance.get('strategy', {}).items():
        pm = perf.get('metrics', {})
        rows.append({'Strategy': name, 'CAGR': pm.get('annual_return', 0),
                     'Vol': pm.get('volatility', 0), 'Sharpe': pm.get('sharpe', 0),
                     'Sortino': pm.get('sortino', 0), 'Max DD': pm.get('max_drawdown', 0),
                     'Win Rate': pm.get('win_rate', 0)})
    if rows:
        df_a = pd.DataFrame(rows).sort_values('Sharpe', ascending=False)
        fmt = df_a.copy()
        for c, f in [('CAGR','{:.1%}'),('Vol','{:.1%}'),('Sharpe','{:.2f}'),
                      ('Sortino','{:.2f}'),('Max DD','{:.1%}'),('Win Rate','{:.0%}')]:
            fmt[c] = fmt[c].apply(lambda x, ff=f: ff.format(x))
        st.dataframe(fmt, width='stretch', hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: RISK INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════════

def _render_risk_intelligence(performance: Dict):
    if not performance:
        st.warning("Performance data not available. Please run an analysis.")
        return

    returns_df = _build_returns_df(performance)
    spectral_hist = performance.get('spectral_history', [])
    spectral_summ = performance.get('spectral_summary', {})
    cross_spec    = performance.get('cross_strategy_spectral', {})
    dr = getattr(st.session_state, 'diversification_ratio', None)

    # ── RMT Overview Cards ──
    if cross_spec or dr or spectral_summ:
        st.markdown(_section_header("RMT Portfolio Diagnostics",
                                    "Marchenko-Pastur signal-noise separation applied to your portfolio"), unsafe_allow_html=True)
        cards = []
        if cross_spec:
            eff = cross_spec.get('effective_strategy_count', 0)
            cards.append(("Effective Strategies", f"{eff:.1f}", "Spectrally independent bets",
                          'success' if eff >= 3 else 'warning' if eff >= 2 else 'danger'))
            nf = cross_spec.get('noise_fraction', 0)
            cards.append(("Noise Fraction", f"{nf:.0%}", "Correlations that are noise",
                          'success' if nf < 0.5 else 'warning' if nf < 0.7 else 'danger'))
        if dr:
            cards.append(("Diversification Ratio", f"{dr:.2f}", "DR > 1 = genuine benefit",
                          'success' if dr > 1.2 else 'info' if dr > 1.0 else 'warning'))
        if spectral_summ:
            ar = spectral_summ.get('mean_absorption_ratio', 0)
            cards.append(("Absorption Ratio", f"{ar:.3f}", "AR > 0.7 = herding",
                          'danger' if ar > 0.7 else 'warning' if ar > 0.5 else 'success'))
            er = spectral_summ.get('mean_effective_rank', 0)
            cards.append(("Effective Rank", f"{er:.1f}", "Independent factors",
                          'success' if er > 3 else 'warning' if er > 2 else 'info'))
            av = spectral_summ.get('ar_volatility', 0)
            cards.append(("AR Volatility", f"{av:.3f}", "Regime stability",
                          'warning' if av > 0.1 else 'info'))
        if cards:
            _render_cards(cards)

    # ── Strategy Correlation Matrix ──
    if not returns_df.empty and len(returns_df.columns) > 1:
        _section_divider()
        st.markdown(_section_header("Strategy Correlation",
                                    "Pairwise return correlations — lower is better for diversification"), unsafe_allow_html=True)
        corr = returns_df.corr()
        fig_c = create_correlation_heatmap(corr)
        st.plotly_chart(fig_c, width='stretch')

        od_mask = ~np.eye(len(corr), dtype=bool)
        avg = corr.values[od_mask].mean()
        regime = "Well Diversified" if avg < 0.5 else ("Moderate" if avg < 0.7 else "Concentrated")
        cls = 'success' if avg < 0.5 else 'warning' if avg < 0.7 else 'danger'
        _render_cards([
            ("Avg Correlation", f"{avg:.2f}", "Off-diagonal mean", cls),
            ("Regime", regime, "Diversification quality", cls),
        ])

    # ── Spectral Charts ──
    if spectral_hist or cross_spec:
        from charts import (create_eigenvalue_histogram, create_cleaned_vs_raw_correlation,
                            create_absorption_ratio_chart, create_factor_loading_heatmap,
                            create_spectral_risk_dashboard)

        if spectral_hist:
            _section_divider()
            st.markdown(_section_header("Rolling Absorption Ratio",
                                        "Systemic risk indicator over time"), unsafe_allow_html=True)
            st.plotly_chart(create_absorption_ratio_chart(spectral_hist), width='stretch')

        if not returns_df.empty and len(returns_df.columns) > 1:
            try:
                from rmt_core import compute_spectral_diagnostics
                raw = returns_df.dropna().values
                if raw.shape[0] >= 10 and raw.shape[1] >= 2:
                    diag = compute_spectral_diagnostics(raw)
                    labels = list(returns_df.columns)

                    _section_divider()
                    st.markdown(_section_header("Eigenvalue Decomposition",
                                                "Strategy return eigenvalues vs Marchenko-Pastur noise boundary"), unsafe_allow_html=True)
                    st.plotly_chart(create_eigenvalue_histogram(
                        diag.eigenvalues, diag.mp_dist.lambda_plus,
                        diag.mp_dist.lambda_minus, diag.mp_dist.gamma,
                        diag.mp_dist.sigma_sq), width='stretch')

                    _section_divider()
                    st.markdown(_section_header("Raw vs RMT-Cleaned Correlation",
                                                "Noise removed from correlation structure"), unsafe_allow_html=True)
                    st.plotly_chart(create_cleaned_vs_raw_correlation(
                        returns_df.corr().values, diag.cleaned_corr, labels), width='stretch')

                    if diag.eigenvectors.shape[1] >= 2:
                        _section_divider()
                        st.markdown(_section_header("Factor Loadings",
                                                    "Which strategies load on which hidden risk factors"), unsafe_allow_html=True)
                        st.plotly_chart(create_factor_loading_heatmap(
                            diag.eigenvectors, labels, diag.eigenvalues,
                            n_factors=min(5, len(labels))), width='stretch')
            except Exception:
                pass

        if spectral_hist and len(spectral_hist) >= 3:
            _section_divider()
            st.markdown(_section_header("Spectral Risk Dashboard",
                                        "Multi-panel tracking of spectral risk indicators"), unsafe_allow_html=True)
            st.plotly_chart(create_spectral_risk_dashboard(spectral_hist), width='stretch')

    # ── Strategy Weight Evolution ──
    wh = performance.get('strategy_weights_history', [])
    if wh:
        _section_divider()
        st.markdown(_section_header("Strategy Weight Evolution",
                                    "How portfolio weights shifted across the walk-forward window"), unsafe_allow_html=True)
        _plot_area_evolution(wh)

    # ── REC-4: Conformal Prediction Intervals ──
    ci = performance.get('conformal_intervals', {})
    if ci:
        _section_divider()
        st.markdown(_section_header("Conformal Prediction Intervals (90%)",
                                    "Distribution-free coverage guarantee — next-period return likely falls within these bounds"), unsafe_allow_html=True)
        ci_rows = []
        for name, interval in ci.items():
            lower, point_est, upper = interval
            ci_rows.append({
                'Strategy': name,
                'Lower (90%)': f"{lower:.4f}",
                'Point Est': f"{point_est:.4f}",
                'Upper (90%)': f"{upper:.4f}",
                'Width': f"{upper - lower:.4f}",
            })
        if ci_rows:
            st.dataframe(pd.DataFrame(ci_rows).set_index('Strategy'), width='stretch')

    # ── REC-1: Strategy Dimensionality Reduction ──
    sf = performance.get('strategy_factors', {})
    if sf and 'n_factors' in sf:
        _section_divider()
        st.markdown(_section_header("Strategy Factor Decomposition",
                                    "RMT-based identification of true independent factors among strategies"), unsafe_allow_html=True)
        n_factors = sf.get('n_factors', 0)
        factor_labels = sf.get('factor_labels', [])
        n_strats = len(sf.get('strategy_factor_map', {}))
        explained_var = sf.get('explained_variance', np.array([]))
        total_var_explained = float(explained_var.sum()) if len(explained_var) > 0 else 0.0
        factor_cards = [
            ("Strategies", str(n_strats), "Total strategies analyzed", 'info'),
            ("Signal Factors", str(n_factors), "Above MP noise boundary", 'success' if n_factors >= 3 else 'warning'),
            ("Variance Explained", f"{total_var_explained:.0%}", "By signal factors", 'success' if total_var_explained > 0.5 else 'info'),
            ("Redundancy", f"{max(0, n_strats - n_factors)}", "Noise-dominated strategies", 'warning' if n_strats - n_factors > n_factors else 'info'),
        ]
        _render_cards(factor_cards)

        mapping = sf.get('strategy_factor_map', {})
        if mapping and factor_labels:
            factor_groups = {}
            for strat, factor_idx in mapping.items():
                label = factor_labels[factor_idx] if factor_idx < len(factor_labels) else f"Factor_{factor_idx}"
                factor_groups.setdefault(label, []).append(strat)
            for label, strats in sorted(factor_groups.items()):
                st.markdown(f"**{label}**: {', '.join(strats)}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: STRATEGY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def _render_strategy_analysis(performance: Dict, strategies_dict: Dict, current_df):
    strat_names = [k for k in performance.get('strategy', {}).keys() if k != 'System_Curated']
    if not strat_names:
        st.warning("No individual strategy data available for analysis.")
        return

    # ── Risk-Return Scatter & Factor Radar ──
    st.markdown(_section_header("Risk-Return Profile",
                                "Strategy positioning and multi-factor fingerprints"), unsafe_allow_html=True)
    scatter_data, factor_data = [], []
    for name in strat_names:
        pm = performance.get('strategy', {}).get(name, {}).get('metrics', {})
        if pm:
            scatter_data.append({'Strategy': name, 'Volatility': pm.get('volatility', 0),
                                 'CAGR': pm.get('annual_return', 0), 'Sharpe': pm.get('sharpe', 0),
                                 'Max DD': pm.get('max_drawdown', 0)})
            factor_data.append({'Strategy': name,
                                'Return Factor': min(max(pm.get('annual_return',0)/0.30,-1),1),
                                'Risk Control': min(max(-pm.get('max_drawdown',-0.20)/0.20,0),1),
                                'Consistency': pm.get('win_rate', 0.5),
                                'Efficiency': min(max(pm.get('sharpe',0)/2,-1),1),
                                'Tail Risk': min(max(pm.get('tail_ratio',1),0),2)/2})

    col_l, col_r = st.columns(2)
    with col_l:
        if scatter_data:
            st.plotly_chart(create_risk_return_scatter(scatter_data), width='stretch')

    with col_r:
        if factor_data:
            st.plotly_chart(create_factor_radar(factor_data, max_strategies=4), width='stretch')

    # ── Tier Sharpe Heatmap ──
    subset_perf = performance.get('subset', {})
    if subset_perf:
        _section_divider()
        st.markdown(_section_header("Sharpe by Position Tier",
                                    "Performance decay across 10-stock tiers"), unsafe_allow_html=True)
        fig_t = create_tier_sharpe_heatmap(subset_perf, strat_names)
        if fig_t is not None:
            st.plotly_chart(fig_t, width='stretch')

            # Tier insight cards
            max_tier = 0
            for s in strat_names:
                if s in subset_perf and subset_perf[s]:
                    nums = [int(t.split('_')[1]) for t in subset_perf[s].keys()]
                    if nums: max_tier = max(max_tier, max(nums))
            if max_tier > 0:
                hm = {}
                for s in strat_names:
                    hm[s] = [subset_perf.get(s,{}).get(f'tier_{i+1}', np.nan) for i in range(max_tier)]
                df_h = pd.DataFrame(hm).T
                df_h.columns = [f'T{i+1}' for i in range(df_h.shape[1])]
                tier_means = df_h.mean(axis=0)
                _render_cards([
                    ("Best Tier", tier_means.idxmax(), f"Sharpe {tier_means.max():.2f}", "success"),
                    ("Worst Tier", tier_means.idxmin(), f"Sharpe {tier_means.min():.2f}", "danger"),
                    ("Dispersion", f"{tier_means.std():.2f}", "Cross-tier σ", "info"),
                ])

    # ── Cross-Strategy Conviction ──
    strats_for_hm = {n: strategies_dict[n] for n in strat_names if n in strategies_dict}
    if strats_for_hm and current_df is not None:
        _section_divider()
        st.markdown(_section_header("Cross-Strategy Conviction",
                                    "Signal overlap and consensus across selected strategies"), unsafe_allow_html=True)
        all_signals, signal_counts = [], {}
        for name, s in strats_for_hm.items():
            try:
                port = s.generate_portfolio(current_df)
                if not port.empty:
                    if 'composite_score' not in port.columns:
                        port['composite_score'] = port['weightage_pct']
                    for _, row in port.head(20).iterrows():
                        all_signals.append({'symbol': row['symbol'], 'strategy': name,
                                            'conviction': row['composite_score']})
                    for sym in port.head(10)['symbol']:
                        signal_counts[sym] = signal_counts.get(sym, 0) + 1
            except Exception:
                pass

        if all_signals:
            df_sig = pd.DataFrame(all_signals)
            hm_df = df_sig.pivot(index='symbol', columns='strategy', values='conviction').fillna(0)
            cs = [[0.0, '#2563eb'], [0.5, '#4B5563'], [1.0, '#dc2626']]
            
            # Prevent Streamlit JSON serialization crash on zero-variance matrices
            z_min, z_max = hm_df.values.min(), hm_df.values.max()
            if z_min == z_max:
                z_min, z_max = -0.1, 0.1
                
            fig_hm = go.Figure(data=go.Heatmap(
                z=hm_df.values, x=hm_df.columns, y=hm_df.index,
                colorscale=cs, zmid=0, zmin=z_min, zmax=z_max,
                text=np.round(hm_df.values, 2), texttemplate='%{text:.2f}',
                textfont=dict(size=9, color='rgba(255,255,255,0.8)'),
                hovertemplate='%{y} × %{x}<br>Conviction: %{z:.2f}<extra></extra>',
                colorbar=dict(title=dict(text='Score', font=dict(color='#6B7280', size=10)),
                              tickfont=dict(color='#6B7280', size=9), thickness=8, outlinewidth=0),
                xgap=1, ygap=1,
            ))
            layout = get_chart_layout(height=max(320, len(hm_df) * 20), show_legend=False)
            layout['hovermode'] = 'closest'
            layout['margin'] = dict(l=80, r=16, t=36, b=36)
            fig_hm.update_layout(**layout)
            fig_hm.update_xaxes(tickfont=dict(size=9, color='#6B7280'), showgrid=False)
            fig_hm.update_yaxes(tickfont=dict(size=9, color='#6B7280'), showgrid=False)
            st.plotly_chart(fig_hm, width='stretch')

        if signal_counts:
            sorted_sigs = sorted(signal_counts.items(), key=lambda x: x[1], reverse=True)
            threshold = len(strats_for_hm) / 2
            high_conv = [s for s, c in sorted_sigs if c >= threshold]
            avg_agr = np.mean([c for _, c in sorted_sigs]) / len(strats_for_hm)
            top_pick = sorted_sigs[0] if sorted_sigs else ("—", 0)
            _render_cards([
                ("High Conviction", str(len(high_conv)), f"≥{threshold:.0f} strategy agreement", "primary"),
                ("Signal Agreement", f"{avg_agr:.0%}", "Mean strategy overlap", "info"),
                ("Top Consensus", top_pick[0], f"{top_pick[1]}/{len(strats_for_hm)} strategies", "success"),
            ])

    # ── Tier Allocation History ──
    swh = performance.get('subset_weights_history', [])
    if swh:
        _section_divider()
        st.markdown(_section_header("Tier Allocation History",
                                    "How subset tier weights evolved through the walk-forward window"), unsafe_allow_html=True)
        sel = st.selectbox("Select Strategy", options=strat_names, key="tier_evo_select")
        if sel:
            tier_hist = [{'date': r['date'], **r.get(sel, {})} for r in swh if r.get(sel)]
            _plot_area_evolution(tier_hist)

    # ── Adaptive Selection Ranking ──
    _section_divider()
    st.markdown(_section_header("Adaptive Selection Ranking",
                                "Dispersion-weighted rank composite"), unsafe_allow_html=True)
    summary_rows = []
    for name in strat_names:
        pm = performance.get('strategy', {}).get(name, {}).get('metrics', {})
        sub = performance.get('subset', {}).get(name, {})
        summary_rows.append({
            'Strategy': name, 'Sharpe': pm.get('sharpe',0), 'Sortino': pm.get('sortino',0),
            'Calmar': pm.get('calmar',0), 'Max DD': pm.get('max_drawdown',0),
            'Win Rate': pm.get('win_rate',0),
            'T1 Sharpe': sub.get('tier_1', np.nan) if sub else np.nan})
    if summary_rows:
        df_s = pd.DataFrame(summary_rows)
        for c in ['Sharpe','Sortino','Calmar','Win Rate']:
            df_s[f'{c}_Rank'] = df_s[c].rank(pct=True)
        df_s['DD_Rank'] = df_s['Max DD'].rank(pct=True, ascending=False)
        r_cols = [c for c in df_s.columns if c.endswith('_Rank')]
        dispersions = {c: df_s[c].std() for c in r_cols}
        total_d = sum(dispersions.values()) or 1
        weights = {c: d/total_d for c, d in dispersions.items()}
        df_s['Score'] = sum(df_s[c]*w for c, w in weights.items())
        df_s = df_s.sort_values('Score', ascending=False)
        show = df_s[['Strategy','Sharpe','Sortino','Calmar','Max DD','Win Rate','T1 Sharpe','Score']].copy()
        for c, f in [('Sharpe','{:.2f}'),('Sortino','{:.2f}'),('Calmar','{:.2f}'),
                      ('Max DD','{:.1%}'),('Win Rate','{:.0%}'),('Score','{:.2f}')]:
            show[c] = show[c].apply(lambda x, ff=f: ff.format(x))
        show['T1 Sharpe'] = show['T1 Sharpe'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
        st.dataframe(show, width='stretch', hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: BACKTEST DATA
# ═══════════════════════════════════════════════════════════════════════════════

def _render_backtest_data(performance: Dict):
    strat_keys = list(performance.get('strategy', {}).keys())
    if not strat_keys:
        st.warning("No strategy data available. Run analysis first.")
        return

    # ── Phase 3: Walk-Forward Metrics ──
    st.markdown(_section_header("Phase 3 — Walk-Forward Performance",
                                "Out-of-sample metrics for selected strategies"), unsafe_allow_html=True)
    rows = []
    for name in strat_keys:
        sp = performance.get('strategy', {}).get(name, {})
        pm = sp.get('metrics', {})
        rows.append({
            'Strategy': name,
            'Total Return': pm.get('total_return', 0),
            'CAGR': pm.get('annual_return', pm.get('ann_return', 0)),
            'Volatility': pm.get('volatility', 0),
            'Sharpe Ratio': pm.get('sharpe', 0),
            'Sortino Ratio': pm.get('sortino', 0),
            'Calmar Ratio': pm.get('calmar', 0),
            'Max Drawdown': pm.get('max_drawdown', pm.get('max_dd', 0)),
            'Win Rate': pm.get('win_rate', 0),
            'Profit Factor': pm.get('profit_factor', 0),
            'Omega Ratio': pm.get('omega_ratio', 0),
            'Tail Ratio': pm.get('tail_ratio', 0),
            'Gain/Pain': pm.get('gain_to_pain', 0),
            'Trading Days': len(sp.get('returns', []))})
    df_p3 = pd.DataFrame(rows).sort_values('Sharpe Ratio', ascending=False).reset_index(drop=True)

    best = df_p3.iloc[0] if len(df_p3) else None
    _render_cards([
        ("Strategies", str(len(df_p3)), "Phase 3 walk-forward", "primary"),
        ("Avg Sharpe", f"{df_p3['Sharpe Ratio'].mean():.2f}", "Cross-strategy mean",
         'success' if df_p3['Sharpe Ratio'].mean() > 0.5 else 'warning' if df_p3['Sharpe Ratio'].mean() > 0 else 'danger'),
        ("Avg Return", f"{df_p3['Total Return'].mean():.1%}", "Mean total return",
         'success' if df_p3['Total Return'].mean() > 0 else 'danger'),
        ("Top Strategy", str(best['Strategy'])[:15] if best is not None else "-",
         f"Sharpe {best['Sharpe Ratio']:.2f}" if best is not None else "", "success"),
    ])
    _section_divider()

    fmt_cols = {'Total Return':'{:.2%}','CAGR':'{:.2%}','Volatility':'{:.2%}',
                'Sharpe Ratio':'{:.3f}','Sortino Ratio':'{:.3f}','Calmar Ratio':'{:.3f}',
                'Max Drawdown':'{:.2%}','Win Rate':'{:.1%}','Profit Factor':'{:.2f}',
                'Omega Ratio':'{:.2f}','Tail Ratio':'{:.2f}','Gain/Pain':'{:.2f}',
                'Trading Days':'{:.0f}'}
    styled = df_p3.style.format(fmt_cols)
    grad = [c for c in ['Sharpe Ratio','Sortino Ratio','Calmar Ratio','Total Return'] if c in df_p3.columns]
    if grad:
        try: styled = styled.background_gradient(subset=grad, cmap='RdYlGn')
        except Exception: pass
    st.dataframe(styled, width='stretch', hide_index=True)

    st.download_button("Download Phase 3 CSV", data=df_p3.to_csv(index=False),
                       file_name=f"phase3_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                       mime="text/csv")

    # ── Phase 2: All Strategies ──
    p2 = st.session_state.get('phase2_strategy_metrics', {})
    if p2:
        _section_divider()
        st.markdown(_section_header("Phase 2 — All Strategy Evaluations",
                                    "Trigger-based backtest results used for dynamic strategy selection"), unsafe_allow_html=True)
        p2_rows = []
        for name, data in p2.items():
            if not isinstance(data, dict) or data.get('status') != 'ok': continue
            pm = data.get('metrics', {})
            p2_rows.append({
                'Strategy': name, 'Total Return': pm.get('total_return',0),
                'Ann. Return': pm.get('ann_return',0), 'Volatility': pm.get('volatility',0),
                'Sharpe': pm.get('sharpe',0), 'Sortino': pm.get('sortino',0),
                'Calmar': pm.get('calmar',0), 'Max DD': pm.get('max_dd',0),
                'Win Rate': pm.get('win_rate',0), 'Trades': pm.get('trade_events',0)})
        if p2_rows:
            df_p2 = pd.DataFrame(p2_rows).sort_values('Sharpe', ascending=False).reset_index(drop=True)
            n_sel = len([k for k in performance.get('strategy',{}).keys() if k != 'System_Curated'])
            n_tot = len(df_p2)
            sr = n_sel / n_tot * 100 if n_tot > 0 else 0
            _render_cards([
                ("Evaluated", str(n_tot), "Total strategies", "primary"),
                ("Selected", str(n_sel), f"{sr:.0f}% selection rate", "success"),
                ("Avg Sharpe", f"{df_p2['Sharpe'].mean():.2f}", "All strategies",
                 'success' if df_p2['Sharpe'].mean() > 0.5 else 'warning' if df_p2['Sharpe'].mean() > 0 else 'danger'),
                ("Best Strategy", str(df_p2.iloc[0]['Strategy'])[:15],
                 f"Sharpe {df_p2.iloc[0]['Sharpe']:.2f}", "info"),
            ])

            sel_strats = [k for k in performance.get('strategy',{}).keys() if k != 'System_Curated']
            df_p2.insert(1, 'Selected', df_p2['Strategy'].apply(lambda x: '✅' if x in sel_strats else ''))
            p2_fmt = {'Total Return':'{:.2%}','Ann. Return':'{:.2%}','Volatility':'{:.2%}',
                      'Sharpe':'{:.2f}','Sortino':'{:.2f}','Calmar':'{:.2f}',
                      'Max DD':'{:.2%}','Win Rate':'{:.1%}','Trades':'{:.0f}'}
            p2_styled = df_p2.style.format(p2_fmt)
            try: p2_styled = p2_styled.background_gradient(subset=['Sharpe','Sortino','Calmar'], cmap='RdYlGn')
            except Exception: pass
            st.dataframe(p2_styled, width='stretch', hide_index=True, height=600)
            st.caption(f"Showing {n_tot} strategies evaluated in Phase 2. Y = selected for curation.")
            st.download_button("Download Phase 2 CSV", data=df_p2.to_csv(index=False),
                               file_name=f"phase2_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                               mime="text/csv")
    else:
        _section_divider()
        st.info("Phase 2 strategy selection metrics not available. Run analysis with dynamic selection enabled.")

# ═══════════════════════════════════════════════════════════════════════════
# DYNAMIC STRATEGY SELECTION ENGINE v2.1
# ═══════════════════════════════════════════════════════════════════════════

if 'dynamic_strategies_cache' not in st.session_state:
    st.session_state.dynamic_strategies_cache = None

# Configure module logger
_dss_logger = logger.getChild("DynamicSelection")


def _run_dynamic_strategy_selection(
    historical_data: List[Tuple[datetime, pd.DataFrame]], 
    all_strategies: Dict[str, BaseStrategy],
    selected_style: str,
    progress_bar=None,
    status_text=None,
    trigger_df: Optional[pd.DataFrame] = None,
    trigger_config: Optional[Dict] = None,
) -> Tuple[Optional[List[str]], Dict[str, Dict]]:
    """
    Backtest all strategies using the UnifiedBacktestEngine and select top 4.
    This function is a high-level wrapper around the backtest_engine module,
    replacing the previous internal backtesting logic.
    """
    
    # 1. Configuration
    is_sip = "SIP" in selected_style
    mode = 'sip' if is_sip else 'swing'
    
    if trigger_config is None:
        trigger_config = TRIGGER_CONFIG.get(selected_style, TRIGGER_CONFIG.get('SIP Investment', {}))
    
    _dss_logger.info("=" * 70)
    _dss_logger.info("DYNAMIC STRATEGY SELECTION (via UnifiedBacktestEngine)")
    _dss_logger.info("=" * 70)
    _dss_logger.info(f"Investment Style: {selected_style} (mode: {mode})")
    
    if not DYNAMIC_SELECTION_AVAILABLE:
        _dss_logger.warning("backtest_engine.py not available - using static selection")
        return None, {}
    
    if not historical_data or len(historical_data) < 10:
        _dss_logger.warning(f"Insufficient data ({len(historical_data) if historical_data else 0} days) - using static selection")
        return None, {}
        
        
    # 2. Initialize the Engine
    # Capital is standardized for comparison during selection phase
    engine = UnifiedBacktestEngine(capital=10_000_000) 
    engine._historical_data = historical_data
    engine._strategies = all_strategies
    
    # 3. Run Backtest
    if status_text:
        status_text.text(f"Running {mode} backtest on {len(all_strategies)} strategies...")
    
    backtest_results = engine.run_backtest(
        mode=mode,
        external_trigger_df=trigger_df,
        buy_col='REL_BREADTH',
        sell_col='REL_BREADTH',
        buy_threshold=trigger_config.get('buy_threshold', 0.42),
        sell_threshold=trigger_config.get('sell_threshold', 0.52),
        progress_callback=lambda p, m: progress_bar.progress(0.25 + p * 0.35, text=m) if progress_bar else None
    )
    
    if not backtest_results:
        _dss_logger.error("Backtest engine returned no results.")
        return None, {}
        
    # 4. Select Top Strategies
    if status_text:
        status_text.text("Selecting top strategies...")
        
    selected_strategies = engine.select_top_strategies(
        results=backtest_results,
        mode=mode,
        n_strategies=4,
        diversify=True # Use RMT diversification
    )
    
    # 5. Format results for UI compatibility
    # The UI expects a specific format for the metrics dictionary.
    formatted_results = {}
    for name, data in backtest_results.items():
        if name.startswith('__'): continue
        
        metrics = data.get('metrics', {})
        
        # Remap keys to match what the UI expects in _render_backtest_data
        ui_metrics = {
            'total_return': metrics.get('total_return', 0.0),
            'ann_return': metrics.get('annualized_return', 0.0),
            'volatility': metrics.get('volatility', 0.0),
            'sharpe': metrics.get('sharpe_ratio', 0.0),
            'sortino': metrics.get('sortino_ratio', 0.0),
            'calmar': metrics.get('calmar_ratio', 0.0),
            'max_dd': metrics.get('max_drawdown', 0.0),
            'win_rate': metrics.get('win_rate', 0.0),
            'trade_events': metrics.get('trade_events', 0),
            'buy_events': metrics.get('buy_events', 0),
        }
        
        formatted_results[name] = {
            'status': 'ok',
            'metrics': ui_metrics,
            'daily_data': data.get('daily_data')
        }

    _dss_logger.info(f"SELECTED (dynamic): {', '.join(selected_strategies)}")
    
    return selected_strategies, formatted_results


# --- Main Application ---
def main():
    strategies = discover_strategies()

    # Fallback static PORTFOLIO_STYLES (used if dynamic selection fails)
    PORTFOLIO_STYLES = {
        "Swing Trading": {
            "description": "Short-term (3-21 day) holds to capture rapid momentum and volatility.",
            "mixes": {
                "Bull Market Mix": {
                    "strategies": ['GameTheoreticStrategy', 'NebulaMomentumStorm', 'VolatilitySurfer', 'CelestialAlphaForge'],
                    "rationale": "Dynamically selected based on highest Sortino Ratio from backtest results."
                },
                
                "Bear Market Mix": {
                    "strategies": ['MomentumAccelerator', 'VolatilitySurfer', 'AdaptiveVolBreakout', 'GameTheoreticStrategy'],
                    "rationale": "Dynamically selected based on highest Sortino Ratio from backtest results."
                },
                
                "Chop/Consolidate Mix": {
                    "strategies": ['MomentumAccelerator', 'VolatilitySurfer', 'AdaptiveVolBreakout', 'GameTheoreticStrategy'],
                    "rationale": "Dynamically selected based on highest Sortino Ratio from backtest results."
                }
            }
        },
        
        "SIP Investment": {
            "description": "Systematic long-term (3-12+ months) wealth accumulation. Focus on consistency and drawdown protection.",
            "mixes": {
                "Bull Market Mix": {
                    "strategies": ['GameTheoreticStrategy', 'MomentumAccelerator', 'VolatilitySurfer', 'DivineMomentumOracle'],
                    "rationale": "Dynamically selected based on highest Calmar Ratio from backtest results."
                },
                
                "Bear Market Mix": {
                    "strategies": ['MomentumAccelerator', 'VolatilitySurfer', 'AdaptiveVolBreakout', 'GameTheoreticStrategy'],
                    "rationale": "Dynamically selected based on highest Calmar Ratio from backtest results."
                },
                
                "Chop/Consolidate Mix": {
                    "strategies": ['MomentumAccelerator', 'VolatilitySurfer', 'AdaptiveVolBreakout', 'GameTheoreticStrategy'],
                    "rationale": "Dynamically selected based on highest Calmar Ratio from backtest results."
                }
            }
        }
    }
    
    def update_regime_suggestion():
        """
        Called when the analysis date changes. Fetches *just enough*
        data to run the regime model and updates the session state.
        """
        selected_date_obj = st.session_state.get('analysis_date_str') 
        if not selected_date_obj:
            return
            
        selected_date = datetime.combine(selected_date_obj, datetime.min.time())
        
        toast_msg = f"Fetching regime data for {selected_date.date()}..."
        st.toast(toast_msg, icon="🧠")
        
        mix_name, explanation, confidence, details = get_market_mix_suggestion_v3(selected_date)
        
        st.session_state.suggested_mix = mix_name
        
        # --- NEW: Store detailed regime info for sidebar display ---
        st.session_state.regime_display = {
            'mix': mix_name,
            'confidence': confidence,
            'explanation': explanation
        }


    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; margin-bottom: 1rem;">
            <div style="font-size: 1.75rem; font-weight: 800; color: #FFC300;">PRAGYAM</div>
            <div style="color: #888888; font-size: 0.75rem; margin-top: 0.25rem;">प्रज्ञम | Portfolio Intelligence</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-title">📅 Analysis Configuration</div>', unsafe_allow_html=True)
        
        today = datetime.now()
        selected_date_str = st.date_input(
            "Select Analysis Date",
            value=today,
            min_value=today - timedelta(days=5*365),
            max_value=today,
            help="Choose a date to run the portfolio curation.",
            key='analysis_date_str',
            on_change=update_regime_suggestion 
        )

        # --- NEW: Dynamic Market Regime Info Card ---
        # Trigger initial calculation if needed
        if st.session_state.suggested_mix is None:
             update_regime_suggestion()

        if st.session_state.regime_display:
            data = st.session_state.regime_display
            # Using HTML/CSS to blend with the existing sidebar UI (metric-card/info-box style)
            st.markdown(f"""
            <div style="background-color: var(--secondary-background-color); border: 1px solid var(--border-color); border-radius: 8px; padding: 12px; margin: 10px 0 20px 0; border-left: 0px solid var(--primary-color); box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <div style="color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; margin-bottom: 4px;">Market Regime</div>
                <div style="color: var(--text-primary); font-size: 1.1rem; font-weight: 700; line-height: 1.2;">{data['mix']}</div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 8px;">
                    <span style="color: var(--text-muted); font-size: 0.8rem;">Confidence</span>
                    <span style="color: var(--primary-color); font-weight: 600; font-size: 0.8rem;">{data['confidence']:.0%}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        # --- END NEW CARD ---
        
        st.markdown('<div class="sidebar-title">💼 Portfolio Style</div>', unsafe_allow_html=True)

        options_list = list(PORTFOLIO_STYLES.keys())
        default_index = 0 
        if "SIP Investment" in options_list:
            default_index = options_list.index("SIP Investment")

        selected_main_branch = st.selectbox(
            "1. Select Investment Style",
            options=options_list,
            index=default_index,
            help="Choose your primary investment objective (e.g., short-term trading or long-term investing)."
        )
        
        mix_options = list(PORTFOLIO_STYLES[selected_main_branch]["mixes"].keys())
        
        st.markdown('<div class="sidebar-title">⚙️ Portfolio Parameters</div>', unsafe_allow_html=True)
        capital = st.number_input("Capital (₹)", 1000, 100000000, 2500000, 1000, help="Total capital to allocate")
        num_positions = st.slider("Number of Positions", 5, 100, 30, 5, help="Maximum positions in the final portfolio")

        # ═══════════════════════════════════════════════════════════════════
        # TRIGGER-BASED BACKTEST CONFIGURATION
        # ═══════════════════════════════════════════════════════════════════
        st.markdown('<div class="sidebar-title">🎯 Backtest Trigger Settings</div>', unsafe_allow_html=True)
        
        use_trigger_backtest = st.checkbox(
            "Enable Trigger-Based Backtest",
            value=True,
            help="Use REL_BREADTH trigger signals for buy/sell timing (aligned with backtest.py methodology)"
        )
        
        trigger_df = None
        trigger_config = TRIGGER_CONFIG.get(selected_main_branch, TRIGGER_CONFIG['SIP Investment'])
        
        if use_trigger_backtest:
            trigger_df = None  # Initialize before conditional branches to prevent NameError
            with st.expander("⚙️ Trigger Configuration", expanded=False):
                # Auto-fetch from Google Sheets
                if STRATEGY_SELECTION_AVAILABLE:
                    st.caption("📡 Trigger Source: Google Sheets (REL_BREADTH)")
                    try:
                        @st.cache_data(ttl=3600, show_spinner=False)
                        def _fetch_breadth_data():
                            return load_breadth_data(lookback_rows=600)
                        
                        breadth_df = _fetch_breadth_data()
                        if not breadth_df.empty:
                            trigger_df = breadth_df.copy()
                            trigger_df = trigger_df.set_index('DATE')
                            st.success(f"✅ Loaded {len(trigger_df)} trigger entries from Google Sheets")
                            
                            # Show recent data preview
                            recent = breadth_df.tail(5).copy()
                            recent['DATE'] = recent['DATE'].dt.strftime('%Y-%m-%d')
                            st.dataframe(recent[['DATE', 'REL_BREADTH']], hide_index=True, width='stretch')
                        else:
                            st.warning("⚠️ No data from Google Sheets. Using first-day entry fallback.")
                    except Exception as e:
                        st.error(f"Error fetching from Google Sheets: {e}")
                        trigger_df = None
                else:
                    # Fallback: file upload if strategy_selection not available
                    trigger_file = st.file_uploader(
                        "Upload Trigger File (CSV/XLSX)",
                        type=['csv', 'xlsx', 'xls'],
                        help="File with DATE and REL_BREADTH columns."
                    )
                    
                    if trigger_file is not None:
                        try:
                            if trigger_file.name.endswith('.csv'):
                                trigger_df = pd.read_csv(trigger_file)
                            else:
                                trigger_df = pd.read_excel(trigger_file)
                            
                            if 'DATE' in trigger_df.columns:
                                trigger_df['DATE'] = pd.to_datetime(trigger_df['DATE'], format='%d-%m-%Y', errors='coerce')
                                trigger_df = trigger_df.dropna(subset=['DATE']).set_index('DATE')
                                st.success(f"✅ Loaded {len(trigger_df)} trigger entries")
                            else:
                                st.warning("File must contain a 'DATE' column")
                                trigger_df = None
                        except Exception as e:
                            st.error(f"Error loading trigger file: {e}")
                            trigger_df = None
                
                # C-1: Compute adaptive thresholds from breadth distribution
                adaptive_buy = trigger_config['buy_threshold']
                adaptive_sell = trigger_config['sell_threshold']
                if STRATEGY_SELECTION_AVAILABLE and trigger_df is not None and 'REL_BREADTH' in trigger_df.columns:
                    try:
                        adaptive_buy, adaptive_sell = compute_adaptive_thresholds(
                            trigger_df['REL_BREADTH'], buy_pct=25, sell_pct=60
                        )
                        st.info(f"Adaptive thresholds: Buy < {adaptive_buy:.3f} | Sell >= {adaptive_sell:.3f}")
                    except Exception:
                        pass

                st.markdown(f"**Mode:** {selected_main_branch}")

                col1, col2 = st.columns(2)
                with col1:
                    buy_thresh = st.number_input(
                        "Buy Threshold",
                        value=adaptive_buy,
                        min_value=0.0,
                        max_value=2.0,
                        step=0.01,
                        help="Buy when REL_BREADTH < this value (adaptive default from breadth distribution)"
                    )
                with col2:
                    sell_thresh = st.number_input(
                        "Sell Threshold",
                        value=adaptive_sell,
                        min_value=0.1,
                        max_value=3.0,
                        step=0.01,
                        help="Sell when REL_BREADTH >= this value (adaptive default from breadth distribution)"
                    )
                
                sell_enabled = st.checkbox(
                    "Enable Sell Trigger",
                    value=trigger_config['sell_enabled'],
                    help="If disabled, positions are held until end of period"
                )
                
                # Update trigger config
                trigger_config = {
                    'buy_threshold': buy_thresh,
                    'sell_threshold': sell_thresh,
                    'sell_enabled': sell_enabled
                }
                
                st.caption(f"📊 Buy: REL_BREADTH < {buy_thresh} | Sell: REL_BREADTH >= {sell_thresh} ({'enabled' if sell_enabled else 'disabled'})")
        
        # Store in session state for later use
        st.session_state.use_trigger_backtest = use_trigger_backtest
        st.session_state.trigger_df = trigger_df
        st.session_state.trigger_config = trigger_config

        if st.button("Run Analysis", width='stretch', type="primary"):
            
            lookback_files = 200
            
            selected_date_obj = st.session_state.get('analysis_date_str')
            if not selected_date_obj:
                st.error("Analysis date is missing. Please select a date.")
                st.stop()
                
            selected_date_dt = datetime.combine(selected_date_obj, datetime.min.time())

            # --- Create Progress Tracking UI ---
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0, text="Initializing...")
                status_text = st.empty()

            total_days_to_fetch = int((lookback_files + MAX_INDICATOR_PERIOD) * 1.5) + 30
            fetch_start_date = selected_date_dt - timedelta(days=total_days_to_fetch)
            
            # ═══════════════════════════════════════════════════════════════════
            # PHASE 1: DATA FETCHING
            # ═══════════════════════════════════════════════════════════════════
            progress_bar.progress(0.05, text="Fetching market data...")
            status_text.text(f"Downloading {len(SYMBOLS_UNIVERSE)} symbols")
            
            logger.info("=" * 70)
            logger.info("PRAGYAM ANALYSIS ENGINE v2.1")
            logger.info("=" * 70)
            logger.info(f"[PHASE 1/4] DATA FETCHING")
            logger.info(f"  Symbols: {len(SYMBOLS_UNIVERSE)}")
            logger.info(f"  Period: {fetch_start_date.date()} to {selected_date_dt.date()}")
            
            all_historical_data = load_historical_data(selected_date_dt, lookback_files)
            
            if not all_historical_data:
                progress_bar.empty()
                status_text.empty()
                st.error("Application cannot start: No historical data could be loaded or generated.")
                st.stop()

            progress_bar.progress(0.20, text="Data loaded. Preparing...")
            logger.info(f"  Result: {len(all_historical_data)} trading days loaded")

            current_date, current_df = all_historical_data[-1]
            training_data = all_historical_data[:-1]
            
            if len(training_data) > lookback_files:
                training_data_window = training_data[-lookback_files:]
            else:
                training_data_window = training_data
            
            training_data_window_with_current = training_data_window + [(current_date, current_df)]
            
            st.session_state.current_df = current_df
            st.session_state.selected_date = current_date.strftime('%Y-%m-%d')
            
            if len(training_data_window_with_current) < 10:
                progress_bar.empty()
                status_text.empty()
                st.error(f"Not enough training data loaded ({len(training_data_window_with_current)} days). Need at least 10. Check date range or lookback period.")
                st.stop()
                
            if not st.session_state.suggested_mix:
                progress_bar.empty()
                status_text.empty()
                st.error("Market regime could not be determined. Please select a valid date. Analysis cannot run.")
                st.stop()
                
            final_mix_to_use = st.session_state.suggested_mix

            # CRITICAL: Prevent Lookahead Bias by strictly separating Phase 2 (Selection) and Phase 3 (Validation) data
            PHASE3_LOOKBACK = 50
            phase2_data = training_data_window_with_current[:-PHASE3_LOOKBACK] if len(training_data_window_with_current) > PHASE3_LOOKBACK + 10 else training_data_window_with_current
            
            # ═══════════════════════════════════════════════════════════════════
            # PHASE 2: DYNAMIC STRATEGY SELECTION (TRIGGER-BASED)
            # ═══════════════════════════════════════════════════════════════════
            progress_bar.progress(0.25, text="Running strategy selection...")
            status_text.text(f"Analyzing {len(strategies)} strategies...")
            
            # Get trigger configuration for strategy selection
            use_trigger = st.session_state.get('use_trigger_backtest', True)
            trigger_df = st.session_state.get('trigger_df', None)
            trigger_config = st.session_state.get('trigger_config', TRIGGER_CONFIG.get(selected_main_branch, {}))
            
            logger.info("-" * 70)
            logger.info(f"[PHASE 2/4] DYNAMIC STRATEGY SELECTION (TRIGGER-BASED)")
            logger.info(f"  Investment Style: {selected_main_branch}")
            logger.info(f"  Market Regime: {final_mix_to_use}")
            logger.info(f"  Trigger Mode: {use_trigger}")
            if use_trigger:
                logger.info(f"  Buy Threshold: {trigger_config.get('buy_threshold', 0.42)}")
                logger.info(f"  Sell Enabled: {trigger_config.get('sell_enabled', False)}")
            
            dynamic_strategies, strategy_metrics = _run_dynamic_strategy_selection(
                phase2_data, 
                strategies, 
                selected_main_branch,
                progress_bar=progress_bar,
                status_text=status_text,
                trigger_df=trigger_df if use_trigger else None,
                trigger_config=trigger_config if use_trigger else None
            )
            
            # Store Phase 2 metrics for Strategy Metrics tab
            st.session_state.phase2_strategy_metrics = strategy_metrics
            
            # Determine which strategies to use
            if dynamic_strategies and len(dynamic_strategies) >= 4:
                style_strategies = dynamic_strategies
                selection_mode = "DYNAMIC"
                logger.info(f"  Mode: DYNAMIC - Selected {len(dynamic_strategies)} strategies")
                st.toast(f"Selected: {', '.join(style_strategies[:2])}...", icon="✅")
            else:
                style_strategies = PORTFOLIO_STYLES[selected_main_branch]["mixes"][final_mix_to_use]['strategies']
                selection_mode = "STATIC"
                logger.info(f"  Mode: STATIC (fallback) - Using predefined strategies")
                st.toast(f"Using default strategies", icon="ℹ️")
            
            # Filter to only available strategies
            strategies_to_run = {name: strategies[name] for name in style_strategies if name in strategies}
            
            if not strategies_to_run:
                progress_bar.empty()
                status_text.empty()
                st.error(f"None of the selected strategies are available: {style_strategies}")
                st.stop()
            
            logger.info(f"  Strategies for execution: {list(strategies_to_run.keys())}")
            
            # ═══════════════════════════════════════════════════════════════════
            # PHASE 3: WALK-FORWARD PORTFOLIO CURATION (Pure — No Triggers)
            # ═══════════════════════════════════════════════════════════════════
            # Phase 3 uses a shorter window (50 days) than Phase 2 (100 days)
            # - Phase 2 needs breadth for robust strategy selection across regimes
            # - Phase 3 needs recency for adaptive walk-forward weights
            #
            # ARCHITECTURAL PRINCIPLE:
            # Triggers (REL_BREADTH) are used in Phase 2 to SELECT strategies
            # that perform best under our timing system. Phase 3 evaluates the
            # CURATION QUALITY: "Given these strategies, how well can we pick 
            # stocks day-by-day?" This is a pure walk-forward process — every
            # day is a rebalancing day. The resulting metrics (Sharpe, Sortino,
            # MaxDD) measure stock-picking ability, not timing ability.
            
            progress_bar.progress(0.65, text="Running walk-forward portfolio curation...")
            status_text.text(f"Walk-forward: {len(strategies_to_run)} strategies over {PHASE3_LOOKBACK} days...")
            
            logger.info("-" * 70)
            logger.info(f"[PHASE 3/4] WALK-FORWARD PORTFOLIO CURATION (PURE)")
            logger.info(f"  Mode: {selected_main_branch}")
            logger.info(f"  Strategies: {list(strategies_to_run.keys())}")
            logger.info(f"  Phase 3 Window: {PHASE3_LOOKBACK} days (of {len(training_data_window_with_current)} total)")
            logger.info(f"  Method: Daily rebalancing walk-forward (no trigger dependency)")
            
            st.session_state.performance = evaluate_historical_performance(
                strategies_to_run, 
                training_data_window_with_current,
                test_window_size=PHASE3_LOOKBACK
            )
            
            # ═══════════════════════════════════════════════════════════════════
            # PHASE 4: PORTFOLIO CURATION
            # ═══════════════════════════════════════════════════════════════════
            if st.session_state.performance:
                progress_bar.progress(0.90, text="Curating final portfolio...")
                status_text.text("Optimizing position weights...")
                
                logger.info("-" * 70)
                logger.info(f"[PHASE 4/4] PORTFOLIO CURATION")
                logger.info(f"  Capital: ₹{capital:,}")
                logger.info(f"  Max Positions: {num_positions}")
                
                st.session_state.portfolio, _, _, st.session_state.diversification_ratio = curate_final_portfolio(
                    strategies_to_run,
                    st.session_state.performance,
                    st.session_state.current_df,
                    capital,
                    num_positions,
                    st.session_state.min_pos_pct,
                    st.session_state.max_pos_pct
                )
                
                progress_bar.progress(1.0, text="Complete!")
                status_text.text(f"Portfolio: {len(st.session_state.portfolio)} positions")
                
                logger.info(f"  Result: {len(st.session_state.portfolio)} positions curated")
                logger.info("=" * 70)
                logger.info(f"ANALYSIS COMPLETE | Mode: {selection_mode} | Strategies: {len(strategies_to_run)} | Positions: {len(st.session_state.portfolio)}")
                logger.info("=" * 70)
                
                progress_bar.empty()
                status_text.empty()
                
                st.toast("Analysis Complete!", icon="✅")
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class='info-box'>
            <p style='font-size: 0.8rem; margin: 0; color: var(--text-muted); line-height: 1.5;'>
                <strong>Version:</strong> {VERSION}<br>
                <strong>Engine:</strong> Walk-Forward Curation<br> 
                <strong>Data:</strong> Live Generated
            </p>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.portfolio is None or st.session_state.performance is None:
        # Show header only on landing page
        st.markdown(f"""
        <div class="premium-header">
            <h1>PRAGYAM | Portfolio Intelligence</h1>
            <div class="tagline">Walk-Forward Curation with Regime-Aware Strategy Allocation</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-box welcome'>
            <h4>Welcome to the Pragyam Curation System</h4>
            <p>
                This platform uses a walk-forward engine to backtest and curate a final portfolio
                based on a dynamic mix of quantitative strategies.
            </p>
            <strong>To begin, please follow these steps:</strong>
            <ol style="margin-left: 20px; margin-top: 10px;">
                <li>Select your desired <strong>Analysis Date</strong> in the sidebar.</li>
                <li>Choose your <strong>Investment Style</strong> (e.g., SIP Investment, Swing Trading).</li>
                <li>Adjust your <strong>Capital</strong> and desired <strong>Number of Positions</strong>.</li>
                <li>Click the <strong>Run Analysis</strong> button to start the curation.</li>
            </ol>
            <p style="margin-top: 1rem; font-weight: 600; color: var(--primary-color);">
                The system will automatically detect the market regime and select the optimal strategy mix for you.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(_metric_card("Regime-Aware", "Auto-Detects", "Bull, Bear, or Chop Mix", "info"), unsafe_allow_html=True)
        with col2:
            st.markdown(_metric_card("Dynamic", "Strategy Curation", "Weights strategies by performance", "success"), unsafe_allow_html=True)
        with col3:
            st.markdown(_metric_card("Walk-Forward", "Robust Backtesting", "Avoids lookahead bias", "primary"), unsafe_allow_html=True)

    else:
        total_value = st.session_state.portfolio['value'].sum()
        cash_remaining = capital - total_value

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(_metric_card("Cash Utilized", f"{total_value:,.2f}", "Total capital deployed", "primary"), unsafe_allow_html=True)
        with col2:
            st.markdown(_metric_card("Cash Remaining", f"{cash_remaining:,.2f}", "Available capital", "neutral"), unsafe_allow_html=True)
        with col3:
            st.markdown(_metric_card("Positions", f"{len(st.session_state.portfolio)}", "Active holdings", "info"), unsafe_allow_html=True)
        _section_divider()

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["**Portfolio**", "**Performance**", "**Risk Intelligence**", "**Strategy Analysis**", "**Backtest Data**"])

        with tab1:
            st.markdown(_section_header("Curated Portfolio Holdings", f"{len(st.session_state.portfolio)} positions from multi-strategy walk-forward curation"), unsafe_allow_html=True)
            display_df = st.session_state.portfolio[['symbol', 'price', 'units', 'weightage_pct', 'value']]
            
            styled_df = display_df.style.format({
                'price': '{:,.2f}', 
                'value': '{:,.2f}', 
                'units': '{:,.0f}',
                'weightage_pct': '{:.2f}%'
            }).set_table_attributes(
                'class="stMarkdown table"' 
            ).hide(
                axis="index"
            )
            
            st.markdown(styled_df.to_html(), unsafe_allow_html=True)
            
            _section_divider()
            portfolio_df = st.session_state.portfolio
            first_cols = ['symbol', 'price', 'units']
            other_cols = [col for col in portfolio_df.columns if col not in first_cols]
            new_order = first_cols + other_cols
            download_df = portfolio_df[new_order]

            csv_buf = io.BytesIO()
            download_df.to_csv(csv_buf, index=False, encoding='utf-8-sig')
            csv_bytes = csv_buf.getvalue()
            st.markdown(
                create_export_link(csv_bytes, f"curated_portfolio_{datetime.now().strftime('%Y%m%d')}.csv"), 
                unsafe_allow_html=True
            )

        with tab2:
            display_performance_metrics(st.session_state.performance)

        with tab3:
            _render_risk_intelligence(st.session_state.performance)

        with tab4:
            _render_strategy_analysis(st.session_state.performance, strategies, st.session_state.current_df)

        with tab5:
            _render_backtest_data(st.session_state.performance)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Dynamic footer with IST time (timezone-aware)
    utc_now = datetime.now(timezone.utc)
    ist_now = utc_now + timedelta(hours=5, minutes=30)
    current_time_ist = ist_now.strftime("%Y-%m-%d %H:%M:%S IST")
    st.caption(f"© 2026 {PRODUCT_NAME} | {COMPANY} | {VERSION} | {current_time_ist}")

if __name__ == "__main__":
    main()
