"""
Backtest Engine - Unified Integration Module for Pragyam
=========================================================

This module provides institutional-grade backtesting capabilities that integrate
seamlessly with Pragyam's data pipeline and strategy ecosystem.

Key Features:
- Unified data fetching through backdata.py (shared resources)
- Dynamic strategy selection based on performance metrics
- SIP Mode: Top 4 strategies by Calmar Ratio
- Swing Mode: Top 4 strategies by Sortino Ratio
- Performance-optimized with intelligent caching

Author: Hemrek Capital
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger("BacktestEngine")


# ============================================================================
# CANONICAL RISK METRICS — single source of truth for Sharpe, Sortino, etc.
# ============================================================================

def compute_risk_metrics(
    returns: np.ndarray,
    periods_per_year: float = 252.0,
    risk_free_rate: float = 0.0,
    total_return: float | None = None,
    ann_return: float | None = None,
) -> Dict[str, float]:
    """Canonical risk-metric computation used by every analytics path.

    Operates on a 1-D array of periodic returns (daily, weekly, or
    per-trigger-day).  All ratio math lives here — callers should
    **not** re-derive Sharpe / Sortino / Calmar independently.

    Args:
        returns: 1-D array-like of periodic simple returns.
        periods_per_year: Annualization factor (252 for daily, 52 for
            weekly, or an empirically estimated value).
        risk_free_rate: Annual risk-free rate subtracted from return.
        total_return: Pre-computed total (geometric) return.  If *None*,
            computed from ``returns`` via ``prod(1+r) - 1``.
        ann_return: Pre-computed annualized return / CAGR.  If *None*,
            derived from *total_return* and the number of periods.

    Returns:
        Dict with keys: total_return, ann_return, volatility, sharpe,
        sortino, max_drawdown, calmar, win_rate, best_period, worst_period.
    """
    empty = {
        'total_return': 0.0, 'ann_return': 0.0, 'volatility': 0.0,
        'sharpe': 0.0, 'sortino': 0.0, 'max_drawdown': 0.0,
        'calmar': 0.0, 'win_rate': 0.0, 'best_period': 0.0, 'worst_period': 0.0,
    }

    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return empty

    # ── total return ──
    if total_return is None:
        total_return = float(np.prod(1 + r) - 1)

    # ── annualized return ──
    n = len(r)
    years = n / periods_per_year
    if ann_return is None:
        if years > 0 and total_return > -1:
            ann_return = (1 + total_return) ** (1 / years) - 1
        else:
            ann_return = 0.0

    # ── volatility (annualized) ──
    ann_factor = np.sqrt(periods_per_year)
    volatility = float(np.std(r, ddof=1) * ann_factor)

    # ── Sharpe ──
    excess = ann_return - risk_free_rate
    sharpe = excess / volatility if volatility > 0.001 else 0.0
    sharpe = float(np.clip(sharpe, -10, 10))

    # ── Sortino (RMS of downside) ──
    downside = np.minimum(r, 0.0)
    downside_vol = float(np.sqrt(np.mean(downside ** 2)) * ann_factor)
    sortino = excess / downside_vol if downside_vol > 0.001 else 0.0
    sortino = float(np.clip(sortino, -10, 10))

    # ── max drawdown ──
    cumulative = np.cumprod(1 + r)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = float(np.min(drawdowns))

    # ── Calmar ──
    calmar = ann_return / abs(max_drawdown) if max_drawdown < -0.001 else 0.0
    calmar = float(np.clip(calmar, -20, 20))

    # ── Win rate & extremes ──
    win_rate = float(np.mean(r > 0))
    best_period = float(np.max(r))
    worst_period = float(np.min(r))

    return {
        'total_return': total_return,
        'ann_return': ann_return,
        'volatility': volatility,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_drawdown,
        'calmar': calmar,
        'win_rate': win_rate,
        'best_period': best_period,
        'worst_period': worst_period,
    }


# ============================================================================
# PERFORMANCE METRICS CALCULATOR (legacy class — delegates to canonical)
# ============================================================================

class PerformanceMetrics:
    """
    Institutional-grade performance metrics calculator.
    Computes all major risk-adjusted return metrics with proper bounds checking.
    """
    
    @staticmethod
    def calculate(
        daily_values: pd.DataFrame,
        risk_free_rate: float = 0.0,
        periods_per_year: float = 252.0
    ) -> Dict[str, float]:
        """Calculate comprehensive performance metrics from daily portfolio values.

        Delegates core ratio math to :func:`compute_risk_metrics`.
        """
        if daily_values.empty or len(daily_values) < 2:
            return PerformanceMetrics._empty_metrics()

        values = daily_values['value'].values
        initial_value = daily_values['investment'].iloc[0]
        final_value = values[-1]

        if initial_value <= 0 or final_value <= 0:
            return PerformanceMetrics._empty_metrics()

        daily_returns = pd.Series(values).pct_change().dropna()
        daily_returns = daily_returns.replace([np.inf, -np.inf], np.nan).dropna().clip(-1.0, 1.0)

        if daily_returns.empty or len(daily_returns) < 2:
            return PerformanceMetrics._empty_metrics()

        total_return = (final_value - initial_value) / initial_value
        years = len(daily_values) / periods_per_year
        cagr = (final_value / initial_value) ** (1 / years) - 1 if years > 0 else 0

        core = compute_risk_metrics(
            daily_returns.values,
            periods_per_year=periods_per_year,
            risk_free_rate=risk_free_rate,
            total_return=total_return,
            ann_return=cagr,
        )

        return {
            'total_return': core['total_return'],
            'annualized_return': core['ann_return'],
            'cagr': cagr,
            'volatility': core['volatility'],
            'sharpe_ratio': core['sharpe'],
            'sortino_ratio': core['sortino'],
            'max_drawdown': core['max_drawdown'],
            'calmar_ratio': core['calmar'],
            'win_rate': core['win_rate'],
            'best_day': core['best_period'],
            'worst_day': core['worst_period'],
            'trading_days': len(daily_values),
            'final_value': final_value,
        }
    
    @staticmethod
    def _empty_metrics() -> Dict[str, float]:
        """Return empty metrics structure."""
        return {
            'total_return': 0, 'annualized_return': 0, 'cagr': 0,
            'volatility': 0, 'sharpe_ratio': 0, 'sortino_ratio': 0,
            'max_drawdown': 0, 'calmar_ratio': 0, 'win_rate': 0,
            'best_day': 0, 'worst_day': 0, 'trading_days': 0, 'final_value': 0
        }


# ============================================================================
# DATA CACHE MANAGER
# ============================================================================

class DataCacheManager:
    """
    In-memory data cache to avoid redundant API calls.

    Implements a singleton pattern so every caller shares one cache.
    """

    _instance: Optional["DataCacheManager"] = None
    _CACHE_TTL_MINUTES: int = 30

    def __new__(cls) -> "DataCacheManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._cache: Dict[str, Any] = {}
            cls._instance._cache_timestamps: Dict[str, datetime] = {}
        return cls._instance

    @staticmethod
    def _generate_key(symbols: List[str], start_date: datetime, end_date: datetime) -> str:
        """Deterministic cache key from query parameters."""
        return f"{'-'.join(sorted(symbols))}_{start_date:%Y%m%d}_{end_date:%Y%m%d}"
    
    def get(self, symbols: List[str], start_date: datetime, end_date: datetime) -> Optional[List[Tuple[datetime, pd.DataFrame]]]:
        """Retrieve cached data if valid."""
        key = self._generate_key(symbols, start_date, end_date)
        
        if key in self._cache:
            cached_time = self._cache_timestamps.get(key)
            if cached_time and (datetime.now() - cached_time).total_seconds() < self._CACHE_TTL_MINUTES * 60:
                logger.info(f"Cache HIT for key: {key[:8]}...")
                return self._cache[key]
            else:
                # Expired - remove from cache
                self._cache.pop(key, None)
                self._cache_timestamps.pop(key, None)
        
        logger.info(f"Cache MISS for key: {key[:8]}...")
        return None
    
    def set(self, symbols: List[str], start_date: datetime, end_date: datetime, data: List[Tuple[datetime, pd.DataFrame]]):
        """Store data in cache."""
        key = self._generate_key(symbols, start_date, end_date)
        self._cache[key] = data
        self._cache_timestamps[key] = datetime.now()
        logger.info(f"Cached data for key: {key[:8]}... ({len(data)} snapshots)")
    
    def clear(self):
        """Clear all cached data."""
        self._cache.clear()
        self._cache_timestamps.clear()
        logger.info("Cache cleared")


# ============================================================================
# UNIFIED BACKTEST ENGINE
# ============================================================================

class UnifiedBacktestEngine:
    """
    Unified backtesting engine that integrates with Pragyam's ecosystem.
    
    Supports both SIP (Systematic Investment) and Swing Trading modes
    with dynamic strategy selection based on performance metrics.
    """
    
    def __init__(
        self,
        capital: float = 10_000_000,
        risk_free_rate: float = 0.0
    ):
        """
        Initialize the backtest engine.
        
        Args:
            capital: Initial capital (default: 1 Crore)
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.capital = capital
        self.risk_free_rate = risk_free_rate
        self.cache_manager = DataCacheManager()
        self._historical_data: Optional[List[Tuple[datetime, pd.DataFrame]]] = None
        self._strategies: Dict[str, Any] = {}
        
    def load_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        use_cache: bool = True,
        progress_callback: Optional[Callable] = None
    ) -> List[Tuple[datetime, pd.DataFrame]]:
        """
        Load historical data using Pragyam's backdata module.
        Utilizes caching to avoid redundant API calls.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date for data
            end_date: End date for data
            use_cache: Whether to use cached data
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of (date, DataFrame) tuples
        """
        # Import backdata here to avoid circular imports
        try:
            from backdata import generate_historical_data, MAX_INDICATOR_PERIOD
        except ImportError:
            logger.error("backdata.py not found. Cannot load data.")
            return []
        
        # Check cache first
        if use_cache:
            cached_data = self.cache_manager.get(symbols, start_date, end_date)
            if cached_data is not None:
                self._historical_data = cached_data
                return cached_data
        
        # Calculate fetch start date (need extra days for indicator warmup)
        fetch_start = start_date - timedelta(days=int(MAX_INDICATOR_PERIOD * 1.5) + 30)
        
        if progress_callback:
            progress_callback(0.1, "Fetching market data...")
        
        logger.info(f"Fetching data from {fetch_start.date()} to {end_date.date()} for {len(symbols)} symbols")
        
        # Generate historical data
        all_data = generate_historical_data(symbols, fetch_start, end_date)
        
        if not all_data:
            logger.error("Failed to fetch any historical data")
            return []
        
        # Filter to requested date range
        filtered_data = [
            (date, df) for date, df in all_data
            if start_date <= date <= end_date
        ]
        
        if progress_callback:
            progress_callback(0.5, f"Loaded {len(filtered_data)} trading days")
        
        # Cache the data
        if use_cache and filtered_data:
            self.cache_manager.set(symbols, start_date, end_date, filtered_data)
        
        self._historical_data = filtered_data
        return filtered_data
    
    def load_strategies(self, strategy_filter: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Load all available strategies from strategies.py.
        
        Args:
            strategy_filter: Optional list of strategy names to include
            
        Returns:
            Dictionary of strategy instances
        """
        try:
            from strategies import BaseStrategy
            import strategies as strategies_module
            import inspect
            
            all_strategies = {}
            
            # Find all BaseStrategy subclasses
            for name, obj in inspect.getmembers(strategies_module, inspect.isclass):
                if issubclass(obj, BaseStrategy) and obj is not BaseStrategy:
                    if strategy_filter is None or name in strategy_filter:
                        try:
                            all_strategies[name] = obj()
                            logger.debug(f"Loaded strategy: {name}")
                        except Exception as e:
                            logger.warning(f"Could not instantiate strategy {name}: {e}")
            
            self._strategies = all_strategies
            logger.info(f"Loaded {len(all_strategies)} strategies")
            return all_strategies
            
        except ImportError as e:
            logger.error(f"Failed to import strategies module: {e}")
            return {}
    
    def run_backtest(
        self,
        mode: str = 'swing',
        external_trigger_df: Optional[pd.DataFrame] = None,
        buy_col: Optional[str] = None,
        sell_col: Optional[str] = None,
        buy_threshold: float = 0.42,
        sell_threshold: float = 0.52,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Dict]:
        """
        Run backtest on all loaded strategies.
        
        Args:
            mode: 'swing' for lumpsum or 'sip' for systematic investment
            external_trigger_df: DataFrame with trigger signals (optional)
            buy_col: Column name for buy signals
            sell_col: Column name for sell signals
            buy_threshold: Threshold for buy trigger
            sell_threshold: Threshold for sell trigger
            progress_callback: Callback for progress updates
            
        Returns:
            Dictionary with metrics for each strategy
        """
        if not self._historical_data:
            logger.error("No historical data loaded. Call load_data() first.")
            return {}
        
        if not self._strategies:
            logger.error("No strategies loaded. Call load_strategies() first.")
            return {}
        
        all_results = {}
        total_strategies = len(self._strategies)
        
        # Prepare trigger masks
        buy_dates_mask, sell_dates_mask = self._prepare_trigger_masks(
            external_trigger_df, buy_col, sell_col, buy_threshold, sell_threshold
        )
        
        for i, (name, strategy) in enumerate(self._strategies.items()):
            if progress_callback:
                progress_callback(
                    0.5 + (0.4 * (i + 1) / total_strategies),
                    f"Backtesting: {name} ({i+1}/{total_strategies})"
                )
            
            try:
                if mode.lower() == 'sip':
                    metrics, daily_data = self._run_sip_backtest(strategy, name)
                else:
                    metrics, daily_data = self._run_swing_backtest(
                        strategy, name, buy_dates_mask, sell_dates_mask
                    )
                
                all_results[name] = {
                    'metrics': metrics,
                    'daily_data': daily_data
                }
                
            except Exception as e:
                logger.error(f"Error backtesting {name}: {e}")
                all_results[name] = {
                    'metrics': PerformanceMetrics._empty_metrics(),
                    'daily_data': pd.DataFrame()
                }
        
        # Compute cross-strategy spectral metrics
        try:
            from rmt_core import detect_redundant_strategies
            returns_dict = {}
            for name, data in all_results.items():
                dd = data.get('daily_data')
                if dd is not None and not dd.empty and 'value' in dd.columns:
                    rets = dd['value'].pct_change().dropna().values
                    if len(rets) >= 20:
                        returns_dict[name] = rets
            if len(returns_dict) >= 3:
                redundancy = detect_redundant_strategies(returns_dict)
                all_results['__spectral_metrics__'] = {
                    'effective_strategy_count': redundancy['effective_strategy_count'],
                    'noise_fraction': redundancy['noise_fraction'],
                    'strategy_clusters': redundancy['clusters'],
                    'signal_eigenvalues': redundancy['signal_eigenvalues'].tolist() if len(redundancy['signal_eigenvalues']) > 0 else [],
                    'diagnostics': redundancy.get('diagnostics'),
                }
        except Exception:
            pass

        if progress_callback:
            progress_callback(1.0, "Backtest complete")

        return all_results
    
    def _prepare_trigger_masks(
        self,
        trigger_df: Optional[pd.DataFrame],
        buy_col: Optional[str],
        sell_col: Optional[str],
        buy_threshold: float,
        sell_threshold: float
    ) -> Tuple[List[bool], List[bool]]:
        """Prepare buy/sell trigger masks from external trigger DataFrame."""
        
        if trigger_df is None or trigger_df.empty:
            # No external triggers - use simple hold strategy
            return [True] + [False] * (len(self._historical_data) - 1), [False] * len(self._historical_data)
        
        buy_mask = [False] * len(self._historical_data)
        sell_mask = [False] * len(self._historical_data)
        
        if buy_col and buy_col in trigger_df.columns:
            external_buy_dates = trigger_df[trigger_df[buy_col] < buy_threshold].index.date
            buy_mask = [d.date() in external_buy_dates for d, _ in self._historical_data]
        
        if sell_col and sell_col in trigger_df.columns:
            external_sell_dates = trigger_df[trigger_df[sell_col] > sell_threshold].index.date
            sell_mask = [d.date() in external_sell_dates for d, _ in self._historical_data]
        
        return buy_mask, sell_mask
    
    def _run_swing_backtest(
        self,
        strategy,
        name: str,
        buy_mask: List[bool],
        sell_mask: List[bool]
    ) -> Tuple[Dict, pd.DataFrame]:
        """Run swing trading (lumpsum) backtest for a single strategy."""
        
        daily_values = []
        portfolio_units = {}
        buy_signal_active = False
        current_capital = self.capital
        
        for j, (date, df) in enumerate(self._historical_data):
            is_buy_day = buy_mask[j]
            actual_buy_trigger = is_buy_day and not buy_signal_active
            
            if is_buy_day:
                buy_signal_active = True
            elif not is_buy_day:
                buy_signal_active = False
            
            # Sell Logic
            if sell_mask[j] and portfolio_units:
                prices_today = df.set_index('symbol')['price']
                current_capital += sum(
                    units * prices_today.get(symbol, 0)
                    for symbol, units in portfolio_units.items()
                )
                portfolio_units = {}
                buy_signal_active = False
            
            # Buy Logic
            if actual_buy_trigger and not portfolio_units and current_capital > 1000:
                try:
                    buy_portfolio = strategy.generate_portfolio(df, current_capital)
                    if not buy_portfolio.empty and 'units' in buy_portfolio.columns:
                        portfolio_units = pd.Series(
                            buy_portfolio['units'].values,
                            index=buy_portfolio['symbol']
                        ).to_dict()
                        current_capital -= buy_portfolio['value'].sum()
                except Exception as e:
                    logger.debug(f"Portfolio generation failed for {name} on {date}: {e}")
            
            # Valuation
            portfolio_value = 0
            if portfolio_units:
                prices_today = df.set_index('symbol')['price']
                portfolio_value = sum(
                    units * prices_today.get(symbol, 0)
                    for symbol, units in portfolio_units.items()
                )
            
            daily_values.append({
                'date': date,
                'value': portfolio_value + current_capital,
                'investment': self.capital
            })
        
        if not daily_values:
            return PerformanceMetrics._empty_metrics(), pd.DataFrame()
        
        daily_df = pd.DataFrame(daily_values)
        metrics = PerformanceMetrics.calculate(daily_df, self.risk_free_rate)
        
        return metrics, daily_df
    
    def _run_sip_backtest(self, strategy, name: str) -> Tuple[Dict, pd.DataFrame]:
        """Run SIP (systematic investment) backtest for a single strategy."""
        
        # SIP: Invest fixed amount periodically (weekly in this case)
        sip_amount = self.capital / 52  # Weekly SIP
        total_invested = 0
        portfolio_units = {}
        daily_values = []
        last_sip_week = None
        
        for date, df in self._historical_data:
            iso = date.isocalendar()
            week_key = (iso[0], iso[1])  # (year, week) to avoid cross-year collision

            # Weekly SIP
            if week_key != last_sip_week:
                try:
                    buy_portfolio = strategy.generate_portfolio(df, sip_amount)
                    if not buy_portfolio.empty and 'units' in buy_portfolio.columns:
                        for _, row in buy_portfolio.iterrows():
                            symbol = row['symbol']
                            units = row['units']
                            if symbol in portfolio_units:
                                portfolio_units[symbol] += units
                            else:
                                portfolio_units[symbol] = units
                        total_invested += buy_portfolio['value'].sum()
                        last_sip_week = week_key
                except Exception as e:
                    logger.debug(f"SIP generation failed for {name} on {date}: {e}")
            
            # Valuation
            portfolio_value = 0
            if portfolio_units:
                prices_today = df.set_index('symbol')['price']
                portfolio_value = sum(
                    units * prices_today.get(symbol, 0)
                    for symbol, units in portfolio_units.items()
                )
            
            daily_values.append({
                'date': date,
                'value': portfolio_value,
                'investment': total_invested if total_invested > 0 else sip_amount
            })
        
        if not daily_values:
            return PerformanceMetrics._empty_metrics(), pd.DataFrame()
        
        daily_df = pd.DataFrame(daily_values)
        metrics = PerformanceMetrics.calculate(daily_df, self.risk_free_rate)
        
        return metrics, daily_df
    
    def select_top_strategies(
        self,
        results: Dict[str, Dict],
        mode: str,
        n_strategies: int = 4,
        diversify: bool = True,
        max_correlation: float = 0.7
    ) -> List[str]:
        """
        Select top strategies based on mode-specific metrics, with optional
        RMT-based redundancy filtering to ensure spectral independence.

        When diversify=True, uses cleaned (denoised) correlations to avoid
        selecting strategies that are just noisy copies of each other.

        Args:
            results: Backtest results dictionary
            mode: 'sip' for Calmar-based selection, 'swing' for Sortino-based
            n_strategies: Number of strategies to select
            diversify: If True, apply RMT redundancy filter
            max_correlation: Maximum cleaned correlation allowed between selected strategies

        Returns:
            List of top strategy names
        """
        if not results:
            return []

        # Determine selection metric based on mode
        if mode.lower() == 'sip':
            metric_key = 'calmar_ratio'
            logger.info("Selecting strategies by Calmar Ratio (SIP mode)")
        else:
            metric_key = 'sortino_ratio'
            logger.info("Selecting strategies by Sortino Ratio (Swing mode)")

        # Extract metrics
        strategy_scores = []
        for name, data in results.items():
            metrics = data.get('metrics', {})
            score = metrics.get(metric_key, 0)

            # Filter out strategies with invalid/negative scores
            if np.isfinite(score) and score > -100:
                strategy_scores.append((name, score))

        # Sort by score descending
        strategy_scores.sort(key=lambda x: x[1], reverse=True)

        # RMT-based redundancy-aware selection
        if diversify and len(strategy_scores) > n_strategies:
            try:
                from rmt_core import detect_redundant_strategies, greedy_diversified_select

                returns_dict = {}
                for name, data in results.items():
                    daily_data = data.get('daily_data')
                    if daily_data is not None and not daily_data.empty and 'value' in daily_data.columns:
                        rets = daily_data['value'].pct_change().dropna().values
                        if len(rets) >= 20:
                            returns_dict[name] = rets

                if len(returns_dict) >= n_strategies:
                    redundancy = detect_redundant_strategies(returns_dict)
                    if redundancy.get('cleaned_corr') is not None and redundancy['diagnostics'] is not None:
                        top_strategies = greedy_diversified_select(
                            strategy_scores,
                            redundancy['cleaned_corr'],
                            redundancy['strategy_names'],
                            n_select=n_strategies,
                            max_correlation=max_correlation,
                        )
                        eff_count = redundancy['effective_strategy_count']
                        noise_frac = redundancy['noise_fraction']
                        logger.info(
                            f"RMT diversified selection: {top_strategies} "
                            f"(effective={eff_count:.1f}, noise={noise_frac:.1%})"
                        )
                        return top_strategies
            except Exception as e:
                logger.debug(f"RMT diversification unavailable: {e}")

        # Fallback: pure metric-based selection
        top_strategies = [name for name, score in strategy_scores[:n_strategies]]

        logger.info(f"Selected top {len(top_strategies)} strategies: {top_strategies}")
        for name, score in strategy_scores[:n_strategies]:
            logger.info(f"  {name}: {metric_key}={score:.4f}")

        return top_strategies


# ============================================================================
# DYNAMIC PORTFOLIO STYLES GENERATOR
# ============================================================================

class DynamicPortfolioStylesGenerator:
    """
    Generates PORTFOLIO_STYLES dictionary dynamically based on backtest results.
    Replaces hardcoded strategy selections with data-driven choices.
    """
    
    def __init__(self, engine: UnifiedBacktestEngine):
        """
        Initialize with a configured backtest engine.
        
        Args:
            engine: UnifiedBacktestEngine instance with loaded data and strategies
        """
        self.engine = engine
        self._backtest_results: Dict[str, Dict] = {}
        self._sip_results: Dict[str, Dict] = {}
        self._swing_results: Dict[str, Dict] = {}
    
    def run_comprehensive_backtest(
        self,
        external_trigger_df: Optional[pd.DataFrame] = None,
        buy_col: Optional[str] = None,
        sell_col: Optional[str] = None,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[Dict, Dict]:
        """
        Run backtests for both SIP and Swing modes.
        
        Returns:
            Tuple of (sip_results, swing_results)
        """
        # Run SIP backtest
        if progress_callback:
            progress_callback(0.1, "Running SIP backtest...")
        
        self._sip_results = self.engine.run_backtest(
            mode='sip',
            progress_callback=lambda p, m: progress_callback(0.1 + p * 0.4, m) if progress_callback else None
        )
        
        # Run Swing backtest
        if progress_callback:
            progress_callback(0.5, "Running Swing backtest...")
        
        self._swing_results = self.engine.run_backtest(
            mode='swing',
            external_trigger_df=external_trigger_df,
            buy_col=buy_col,
            sell_col=sell_col,
            progress_callback=lambda p, m: progress_callback(0.5 + p * 0.4, m) if progress_callback else None
        )
        
        return self._sip_results, self._swing_results
    
    def generate_portfolio_styles(
        self,
        n_strategies: int = 4
    ) -> Dict[str, Dict]:
        """
        Generate dynamic PORTFOLIO_STYLES based on backtest results.
        
        Args:
            n_strategies: Number of strategies per mix
            
        Returns:
            PORTFOLIO_STYLES dictionary
        """
        # Select strategies for each mode
        sip_strategies = self.engine.select_top_strategies(
            self._sip_results, mode='sip', n_strategies=n_strategies
        )
        
        swing_strategies = self.engine.select_top_strategies(
            self._swing_results, mode='swing', n_strategies=n_strategies
        )
        
        # Build rationale strings with metrics
        sip_rationale = self._build_rationale(self._sip_results, sip_strategies, 'calmar_ratio')
        swing_rationale = self._build_rationale(self._swing_results, swing_strategies, 'sortino_ratio')
        
        # Generate PORTFOLIO_STYLES
        portfolio_styles = {
            "Swing Trading": {
                "description": "Short-term (3-21 day) holds to capture rapid momentum and volatility.",
                "mixes": {
                    "Bull Market Mix": {
                        "strategies": swing_strategies,
                        "rationale": f"Dynamically selected based on highest Sortino Ratio from backtest. {swing_rationale}"
                    },
                    "Bear Market Mix": {
                        "strategies": swing_strategies,
                        "rationale": f"Dynamically selected based on highest Sortino Ratio from backtest. {swing_rationale}"
                    },
                    "Chop/Consolidate Mix": {
                        "strategies": swing_strategies,
                        "rationale": f"Dynamically selected based on highest Sortino Ratio from backtest. {swing_rationale}"
                    }
                }
            },
            "SIP Investment": {
                "description": "Systematic long-term (3-12+ months) wealth accumulation. Focus on consistency and drawdown protection.",
                "mixes": {
                    "Bull Market Mix": {
                        "strategies": sip_strategies,
                        "rationale": f"Dynamically selected based on highest Calmar Ratio from backtest. {sip_rationale}"
                    },
                    "Bear Market Mix": {
                        "strategies": sip_strategies,
                        "rationale": f"Dynamically selected based on highest Calmar Ratio from backtest. {sip_rationale}"
                    },
                    "Chop/Consolidate Mix": {
                        "strategies": sip_strategies,
                        "rationale": f"Dynamically selected based on highest Calmar Ratio from backtest. {sip_rationale}"
                    }
                }
            }
        }
        
        return portfolio_styles
    
    def _build_rationale(
        self,
        results: Dict[str, Dict],
        strategies: List[str],
        metric_key: str
    ) -> str:
        """Build rationale string with metric values."""
        if not results or not strategies:
            return "No backtest data available."
        
        parts = []
        for name in strategies:
            if name in results:
                score = results[name].get('metrics', {}).get(metric_key, 0)
                parts.append(f"{name}: {score:.2f}")
        
        return " | ".join(parts) if parts else "Metrics unavailable."
    
    def get_strategy_leaderboard(self, mode: str = 'swing') -> pd.DataFrame:
        """
        Get a formatted leaderboard of strategy performance.
        
        Args:
            mode: 'sip' or 'swing'
            
        Returns:
            DataFrame with strategy rankings
        """
        results = self._sip_results if mode.lower() == 'sip' else self._swing_results
        
        if not results:
            return pd.DataFrame()
        
        leaderboard_data = []
        for name, data in results.items():
            metrics = data.get('metrics', {})
            leaderboard_data.append({
                'Strategy': name,
                'Total Return': metrics.get('total_return', 0),
                'CAGR': metrics.get('cagr', 0),
                'Volatility': metrics.get('volatility', 0),
                'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
                'Sortino Ratio': metrics.get('sortino_ratio', 0),
                'Max Drawdown': metrics.get('max_drawdown', 0),
                'Calmar Ratio': metrics.get('calmar_ratio', 0),
                'Win Rate': metrics.get('win_rate', 0)
            })
        
        df = pd.DataFrame(leaderboard_data)
        
        # Sort by primary metric for mode
        sort_col = 'Calmar Ratio' if mode.lower() == 'sip' else 'Sortino Ratio'
        df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
        
        return df


# ============================================================================
# INTEGRATION HELPER FUNCTIONS
# ============================================================================

def initialize_backtest_engine(
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    capital: float = 10_000_000,
    progress_callback: Optional[Callable] = None
) -> Tuple[UnifiedBacktestEngine, DynamicPortfolioStylesGenerator]:
    """
    Initialize and configure the backtest engine with data.
    
    This is the main entry point for Pragyam integration.
    
    Args:
        symbols: List of stock symbols
        start_date: Backtest start date
        end_date: Backtest end date
        capital: Initial capital
        progress_callback: Optional progress callback
        
    Returns:
        Tuple of (engine, generator) ready for use
    """
    engine = UnifiedBacktestEngine(capital=capital)
    
    # Load data
    engine.load_data(symbols, start_date, end_date, progress_callback=progress_callback)
    
    # Load all strategies
    engine.load_strategies()
    
    # Create generator
    generator = DynamicPortfolioStylesGenerator(engine)
    
    return engine, generator


def get_dynamic_portfolio_styles(
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    capital: float = 10_000_000,
    external_trigger_df: Optional[pd.DataFrame] = None,
    buy_col: Optional[str] = None,
    sell_col: Optional[str] = None,
    n_strategies: int = 4,
    progress_callback: Optional[Callable] = None
) -> Dict[str, Dict]:
    """
    Main entry point: Generate dynamic PORTFOLIO_STYLES based on backtest.
    
    This function replaces the hardcoded PORTFOLIO_STYLES in app.py.
    
    Args:
        symbols: List of stock symbols
        start_date: Backtest start date  
        end_date: Backtest end date
        capital: Initial capital for backtest
        external_trigger_df: Optional trigger signal DataFrame
        buy_col: Buy trigger column name
        sell_col: Sell trigger column name
        n_strategies: Number of strategies to select per mode
        progress_callback: Optional callback for progress updates
        
    Returns:
        PORTFOLIO_STYLES dictionary with dynamically selected strategies
    """
    # Initialize engine
    engine, generator = initialize_backtest_engine(
        symbols, start_date, end_date, capital,
        progress_callback=lambda p, m: progress_callback(p * 0.3, m) if progress_callback else None
    )
    
    # Run comprehensive backtest
    generator.run_comprehensive_backtest(
        external_trigger_df=external_trigger_df,
        buy_col=buy_col,
        sell_col=sell_col,
        progress_callback=lambda p, m: progress_callback(0.3 + p * 0.6, m) if progress_callback else None
    )
    
    # Generate portfolio styles
    if progress_callback:
        progress_callback(0.95, "Generating portfolio styles...")
    
    portfolio_styles = generator.generate_portfolio_styles(n_strategies=n_strategies)
    
    if progress_callback:
        progress_callback(1.0, "Complete")
    
    return portfolio_styles


# ============================================================================
# STREAMLIT UI MODULE (Standalone Operation)
# ============================================================================

def run_streamlit_ui():
    """
    Run standalone Streamlit UI for backtesting.
    Can be used independently or integrated into Pragyam.
    """
    try:
        import streamlit as st
    except ImportError:
        logger.error("Streamlit not installed. Cannot run UI.")
        return
    
    st.set_page_config(
        page_title="Backtest Engine | Pragyam",
        page_icon="⚙️",
        layout="wide"
    )
    
    st.title("⚙️ Unified Backtest Engine")
    st.markdown("*Dynamic Strategy Selection for Pragyam*")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        capital = st.number_input(
            "Capital (₹)",
            min_value=100000,
            max_value=100000000,
            value=10000000,
            step=100000
        )
        
        n_strategies = st.slider(
            "Strategies per Mix",
            min_value=2,
            max_value=8,
            value=4
        )
        
        run_button = st.button("🚀 Run Backtest", type="primary", width='stretch')
    
    if run_button:
        try:
            from backdata import SYMBOLS_UNIVERSE, MAX_INDICATOR_PERIOD
        except ImportError:
            st.error("Could not import backdata.py")
            return
        
        if not SYMBOLS_UNIVERSE:
            st.error("No symbols found in symbols.txt")
            return
        
        # Set date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        # Progress
        progress_bar = st.progress(0, text="Initializing...")
        
        def update_progress(p, msg):
            progress_bar.progress(p, text=msg)
        
        # Run backtest
        portfolio_styles = get_dynamic_portfolio_styles(
            symbols=SYMBOLS_UNIVERSE,
            start_date=start_date,
            end_date=end_date,
            capital=capital,
            n_strategies=n_strategies,
            progress_callback=update_progress
        )
        
        progress_bar.empty()
        st.success("✅ Backtest Complete!")
        
        # Display results
        st.header("📊 Dynamic Portfolio Styles")
        
        for style_name, style_data in portfolio_styles.items():
            with st.expander(f"**{style_name}**", expanded=True):
                st.markdown(f"*{style_data['description']}*")
                
                for mix_name, mix_data in style_data['mixes'].items():
                    st.subheader(mix_name)
                    st.markdown(f"**Strategies:** `{'`, `'.join(mix_data['strategies'])}`")
                    st.caption(mix_data['rationale'])


__all__ = [
    'compute_risk_metrics',
    'PerformanceMetrics',
    'DataCacheManager',
    'UnifiedBacktestEngine',
    'DynamicPortfolioStylesGenerator',
    'initialize_backtest_engine',
    'get_dynamic_portfolio_styles',
    'run_streamlit_ui',
]

if __name__ == "__main__":
    run_streamlit_ui()
