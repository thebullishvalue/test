"""
PRAGYAM Strategy Selection & Performance Framework
===================================================

Evaluates trading strategies by simulating SIP and Swing investment behaviors
based on market breadth triggers.

Data Source:
    REL_BREADTH from Google Sheets
    https://docs.google.com/spreadsheets/d/1po7z42n3dYIQGAvn0D1-a4pmyxpnGPQ13TrNi3DB5_c

Execution Modes:
----------------
MODE 1: SIP Investment
    - Trigger: Every date where REL_BREADTH < 0.42
    - Process: Generate portfolio for each strategy, accumulate into Master Portfolio
    - Evaluation: Terminal Portfolio metrics

MODE 2: Swing Trading
    - Buy Trigger: REL_BREADTH < 0.42
    - Sell Trigger: REL_BREADTH >= 0.50
    - Process: Execute buy-sell cycles repeatedly
    - Evaluation: Net performance from completed + open trades

Author: Hemrek Capital
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import bisect
from typing import Dict, List, Tuple, Optional, Any
import logging
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')

try:
    from backtest_engine import compute_risk_metrics
except ImportError:
    compute_risk_metrics = None  # graceful degradation

logger = logging.getLogger("StrategySelection")


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS - RESEARCH-DERIVED TRIGGERS (NOT ARBITRARY)
# ══════════════════════════════════════════════════════════════════════════════

# Defaults — override via environment: PRAGYAM_SIP_TRIGGER, PRAGYAM_SWING_BUY, PRAGYAM_SWING_SELL
import os as _os
SIP_TRIGGER = float(_os.environ.get('PRAGYAM_SIP_TRIGGER', '0.42'))
SWING_BUY_TRIGGER = float(_os.environ.get('PRAGYAM_SWING_BUY', '0.42'))
SWING_SELL_TRIGGER = float(_os.environ.get('PRAGYAM_SWING_SELL', '0.50'))


def compute_adaptive_thresholds(breadth_series: 'pd.Series', buy_pct: float = 25, sell_pct: float = 60) -> tuple:
    """Compute adaptive buy/sell thresholds from historical breadth distribution.

    Args:
        breadth_series: Historical REL_BREADTH values.
        buy_pct: Percentile below which to trigger buys (default 25th).
        sell_pct: Percentile above which to trigger sells (default 60th).

    Returns:
        (buy_threshold, sell_threshold) tuple.
    """
    clean = breadth_series.dropna()
    if len(clean) < 20:
        return SIP_TRIGGER, SWING_SELL_TRIGGER
    buy_thresh = float(np.percentile(clean, buy_pct))
    sell_thresh = float(np.percentile(clean, sell_pct))
    
    # Epistemic Fix: Absolute Reality Bounds
    # Prevent absurd thresholds during heavily skewed/multimodal regimes (e.g. buying at 0.80).
    buy_thresh = max(0.10, min(buy_thresh, 0.48))
    sell_thresh = max(0.40, min(sell_thresh, 0.85))
    
    # Ensure a strict minimum mathematical spread to prevent whipsaw clustering
    if sell_thresh - buy_thresh < 0.05:
        midpoint = (buy_thresh + sell_thresh) / 2.0
        buy_thresh = max(0.10, midpoint - 0.05)
        sell_thresh = min(0.85, midpoint + 0.05)
        if buy_thresh >= sell_thresh: # Ultimate fallback
            return SIP_TRIGGER, SWING_SELL_TRIGGER
            
    return round(buy_thresh, 3), round(sell_thresh, 3)

BREADTH_SHEET_ID = "1po7z42n3dYIQGAvn0D1-a4pmyxpnGPQ13TrNi3DB5_c"
BREADTH_SHEET_URL = f"https://docs.google.com/spreadsheets/d/{BREADTH_SHEET_ID}/export?format=csv"

DEFAULT_LOOKBACK_ROWS = 400
DEFAULT_SIP_AMOUNT = 100000.0


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_breadth_data(lookback_rows: int = DEFAULT_LOOKBACK_ROWS) -> pd.DataFrame:
    """
    Load REL_BREADTH data from Google Sheets.
    
    Returns DataFrame with columns: DATE, REL_BREADTH
    """
    try:
        logger.info(f"Fetching breadth data from Google Sheets...")
        df = pd.read_csv(BREADTH_SHEET_URL)
        
        # Standardize column names
        df.columns = [c.strip().upper() for c in df.columns]
        
        # Find DATE column
        if 'DATE' not in df.columns:
            date_cols = [c for c in df.columns if 'DATE' in c.upper()]
            if date_cols:
                df = df.rename(columns={date_cols[0]: 'DATE'})
            else:
                raise ValueError(f"No DATE column. Available: {list(df.columns)}")
        
        # Find REL_BREADTH column
        if 'REL_BREADTH' not in df.columns:
            breadth_cols = [c for c in df.columns if 'BREADTH' in c.upper()]
            if breadth_cols:
                df = df.rename(columns={breadth_cols[0]: 'REL_BREADTH'})
            else:
                raise ValueError(f"No BREADTH column. Available: {list(df.columns)}")
        
        # Parse and clean
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
        df['REL_BREADTH'] = pd.to_numeric(df['REL_BREADTH'], errors='coerce')
        df = df.dropna(subset=['DATE', 'REL_BREADTH'])
        df = df.sort_values('DATE', ascending=True)
        df = df.tail(lookback_rows).reset_index(drop=True)
        
        logger.info(f"Loaded {len(df)} rows: {df['DATE'].min().date()} to {df['DATE'].max().date()}")
        
        return df[['DATE', 'REL_BREADTH']]
        
    except Exception as e:
        logger.error(f"Error loading breadth data: {e}")
        return pd.DataFrame(columns=['DATE', 'REL_BREADTH'])


def load_strategies() -> Dict[str, Any]:
    """Load all strategy classes from strategies.py."""
    try:
        from strategies import BaseStrategy
        import strategies as strategies_module
        import inspect
        
        all_strategies = {}
        for name, obj in inspect.getmembers(strategies_module, inspect.isclass):
            if issubclass(obj, BaseStrategy) and obj is not BaseStrategy:
                try:
                    all_strategies[name] = obj()
                except Exception as e:
                    logger.warning(f"Could not instantiate {name}: {e}")
        
        logger.info(f"Loaded {len(all_strategies)} strategies")
        return all_strategies
        
    except ImportError as e:
        logger.error(f"Could not import strategies: {e}")
        return {}


def load_historical_data(
    symbols: List[str],
    start_date: datetime,
    end_date: datetime
) -> List[Tuple[datetime, pd.DataFrame]]:
    """
    Load historical indicator snapshots from backdata.py.
    
    Returns list of (date, indicator_df) tuples.
    """
    try:
        from backdata import generate_historical_data, MAX_INDICATOR_PERIOD
        
        # Extend start for indicator warmup
        fetch_start = start_date - timedelta(days=int(MAX_INDICATOR_PERIOD * 1.5) + 30)
        
        logger.info(f"Fetching historical data for {len(symbols)} symbols...")
        all_data = generate_historical_data(symbols, fetch_start, end_date)
        
        # Filter to requested range
        filtered = [(date, df) for date, df in all_data if start_date <= date <= end_date]
        
        logger.info(f"Loaded {len(filtered)} trading days of indicator data")
        return filtered
        
    except ImportError as e:
        logger.error(f"Could not import backdata: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# TRIGGER IDENTIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def get_sip_trigger_dates(
    breadth_data: pd.DataFrame,
    buy_threshold: Optional[float] = None
) -> List[datetime]:
    """
    Get all dates where REL_BREADTH < buy_threshold.

    Args:
        breadth_data: DataFrame with 'DATE' and 'REL_BREADTH' columns.
        buy_threshold: Threshold below which to trigger. Defaults to SIP_TRIGGER.
    """
    if breadth_data.empty:
        return []

    if buy_threshold is None:
        buy_threshold = SIP_TRIGGER

    mask = breadth_data['REL_BREADTH'] < buy_threshold
    dates = breadth_data.loc[mask, 'DATE'].tolist()

    logger.info(f"SIP: Found {len(dates)} trigger dates (REL_BREADTH < {buy_threshold})")
    return dates


def get_swing_cycles(
    breadth_data: pd.DataFrame,
    buy_threshold: Optional[float] = None,
    sell_threshold: Optional[float] = None
) -> List[Dict]:
    """
    Identify buy-sell cycles based on breadth triggers.

    Args:
        breadth_data: DataFrame with 'DATE' and 'REL_BREADTH' columns.
        buy_threshold: Buy when REL_BREADTH < this. Defaults to SWING_BUY_TRIGGER.
        sell_threshold: Sell when REL_BREADTH >= this. Defaults to SWING_SELL_TRIGGER.
    """
    if breadth_data.empty:
        return []

    if buy_threshold is None:
        buy_threshold = SWING_BUY_TRIGGER
    if sell_threshold is None:
        sell_threshold = SWING_SELL_TRIGGER

    cycles = []
    in_position = False
    entry_date = None
    entry_breadth = None

    for _, row in breadth_data.iterrows():
        date = row['DATE']
        breadth = row['REL_BREADTH']

        if not in_position and breadth < buy_threshold:
            # BUY signal
            in_position = True
            entry_date = date
            entry_breadth = breadth

        elif in_position and breadth >= sell_threshold:
            # SELL signal
            cycles.append({
                'entry_date': entry_date,
                'exit_date': date,
                'entry_breadth': entry_breadth,
                'exit_breadth': breadth,
                'status': 'closed'
            })
            in_position = False
            entry_date = None
    
    # Handle open position
    if in_position and entry_date is not None:
        cycles.append({
            'entry_date': entry_date,
            'exit_date': breadth_data['DATE'].iloc[-1],
            'entry_breadth': entry_breadth,
            'exit_breadth': breadth_data['REL_BREADTH'].iloc[-1],
            'status': 'open'
        })
    
    closed = len([c for c in cycles if c['status'] == 'closed'])
    logger.info(f"Swing: Found {len(cycles)} cycles ({closed} closed, {len(cycles)-closed} open)")
    
    return cycles


# ══════════════════════════════════════════════════════════════════════════════
# CUMULATIVE MASTER PORTFOLIO TRACKER
# ══════════════════════════════════════════════════════════════════════════════

class MasterPortfolio:
    """
    Tracks cumulative holdings across multiple SIP entries.
    
    Maintains:
    - Holdings: symbol -> {units, avg_cost, current_price}
    - Cash balance
    - Value history for performance calculation
    """
    
    def __init__(self):
        self.holdings: Dict[str, Dict] = {}  # symbol -> {units, avg_cost}
        self.total_invested = 0.0
        self.history: List[Dict] = []
        
    def add_portfolio(self, portfolio_df: pd.DataFrame, date: datetime, investment: float):
        """
        Add a new portfolio's holdings to the master portfolio.
        
        Args:
            portfolio_df: DataFrame with columns [symbol, price, units, value]
            date: Investment date
            investment: Amount invested
        """
        if portfolio_df.empty:
            return
        
        self.total_invested += investment
        
        for _, row in portfolio_df.iterrows():
            symbol = row['symbol']
            price = row['price']
            units = row.get('units', 0)
            
            if units <= 0 or price <= 0:
                continue
            
            if symbol in self.holdings:
                # Average into existing position
                existing = self.holdings[symbol]
                total_units = existing['units'] + units
                total_cost = existing['units'] * existing['avg_cost'] + units * price
                self.holdings[symbol] = {
                    'units': total_units,
                    'avg_cost': total_cost / total_units,
                    'current_price': price
                }
            else:
                self.holdings[symbol] = {
                    'units': units,
                    'avg_cost': price,
                    'current_price': price
                }
        
        # Record snapshot
        self._record_snapshot(date, 'SIP_BUY', cash_flow=investment)
    
    def update_prices(self, price_data: pd.DataFrame, date: datetime):
        """
        Update current prices for all holdings.
        
        Args:
            price_data: DataFrame with columns [symbol, price]
            date: Price date
        """
        if price_data.empty:
            return
        
        prices = price_data.set_index('symbol')['price'].to_dict()
        
        for symbol in self.holdings:
            if symbol in prices:
                self.holdings[symbol]['current_price'] = prices[symbol]
        
        self._record_snapshot(date, 'PRICE_UPDATE', cash_flow=0.0)
    
    def get_current_value(self) -> float:
        """Get total current value of all holdings."""
        return sum(
            h['units'] * h.get('current_price', h['avg_cost'])
            for h in self.holdings.values()
        )
    
    def get_total_return(self) -> float:
        """Get total return percentage."""
        if self.total_invested <= 0:
            return 0.0
        return (self.get_current_value() - self.total_invested) / self.total_invested
    
    def _record_snapshot(self, date: datetime, action: str, cash_flow: float = 0.0):
        """Record portfolio state."""
        self.history.append({
            'date': date,
            'action': action,
            'total_invested': self.total_invested,
            'cash_flow': cash_flow,
            'current_value': self.get_current_value(),
            'num_holdings': len(self.holdings),
            'return': self.get_total_return()
        })
    
    def get_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics.

        Delegates core ratio math to :func:`backtest_engine.compute_risk_metrics`
        when available, with inline fallback for standalone usage.
        """
        if not self.history or self.total_invested <= 0:
            return self._empty_metrics()

        total_return = self.get_total_return()
        terminal_value = self.get_current_value()

        # Extract per-snapshot returns
        returns = []
        for i in range(1, len(self.history)):
            prev_val = self.history[i - 1]['current_value']
            curr_val = self.history[i]['current_value']
            cf = self.history[i]['cash_flow']
            if prev_val > 0:
                # CRITICAL FIX: Modified Dietz Time-Weighted Return
                returns.append((curr_val - cf - prev_val) / prev_val)
            elif prev_val == 0 and cf > 0:
                returns.append(0.0)
            else:
                returns.append(0.0)

        returns = np.array(returns) if returns else np.array([total_return])

        # Estimate periods_per_year from ACTUAL calendar time
        if len(self.history) >= 2:
            start_date = self.history[0]['date']
            end_date = self.history[-1]['date']
            days = (end_date - start_date).days
            years = days / 365.25 if days > 0 else 1.0
            periods_per_year = max(1.0, len(returns) / years)
        else:
            periods_per_year = 252.0

        if compute_risk_metrics is not None and len(returns) >= 2:
            core = compute_risk_metrics(
                returns,
                periods_per_year=periods_per_year,
                total_return=total_return,
            )
            sharpe = core['sharpe']
            sortino = core['sortino']
            max_dd = core['max_drawdown']
            win_rate = core['win_rate']
            calmar = core['calmar']
        else:
            # Fallback: inline computation for standalone usage
            ann_factor = np.sqrt(periods_per_year)
            if len(returns) > 1 and returns.std() > 0:
                sharpe = (returns.mean() / returns.std()) * ann_factor
                downside = returns[returns < 0]
                downside_dev = np.sqrt(np.mean(downside ** 2)) if len(downside) > 0 else 0
                sortino = (returns.mean() / downside_dev) * ann_factor if downside_dev > 0 else sharpe
            else:
                sharpe = total_return if total_return > 0 else 0
                sortino = sharpe

            values = [h['current_value'] for h in self.history]
            if values:
                peak = np.maximum.accumulate(values)
                drawdown = (np.array(values) - peak) / np.where(peak > 0, peak, 1)
                max_dd = drawdown.min()
            else:
                max_dd = 0
            win_rate = (returns > 0).mean() if len(returns) > 0 else 0
            calmar = total_return / abs(max_dd) if max_dd < -0.001 else 0

        return {
            'total_invested': self.total_invested,
            'terminal_value': terminal_value,
            'total_return': total_return,
            'sharpe': sharpe,
            'sortino': sortino,
            'max_drawdown': max_dd,
            'calmar': calmar,
            'win_rate': win_rate,
            'num_holdings': len(self.holdings)
        }
    
    def _empty_metrics(self) -> Dict:
        return {
            'total_invested': 0,
            'terminal_value': 0,
            'total_return': 0,
            'sharpe': 0,
            'sortino': 0,
            'max_drawdown': 0,
            'calmar': 0,
            'win_rate': 0,
            'num_holdings': 0
        }
    
    def reset(self):
        """Reset to empty state."""
        self.holdings = {}
        self.total_invested = 0.0
        self.history = []


# ══════════════════════════════════════════════════════════════════════════════
# SIP MODE EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

def execute_sip_mode(
    strategies: Dict[str, Any],
    historical_data: List[Tuple[datetime, pd.DataFrame]],
    breadth_data: pd.DataFrame,
    sip_amount: float = DEFAULT_SIP_AMOUNT,
    buy_threshold: Optional[float] = None
) -> Dict[str, Dict]:
    """
    Execute SIP mode: Accumulate on every trigger date.

    Process:
    1. Find all dates where REL_BREADTH < buy_threshold
    2. For each trigger date:
       - Get indicator data for that date
       - Run each strategy's generate_portfolio()
       - Add holdings to cumulative master portfolio
    3. Update to terminal prices
    4. Calculate selection metrics from terminal portfolio

    Args:
        strategies: Dict of strategy_name -> strategy instance
        historical_data: List of (date, indicator_df) from backdata
        breadth_data: DataFrame with DATE and REL_BREADTH
        sip_amount: Amount to invest per trigger
        buy_threshold: Adaptive buy threshold. Defaults to SIP_TRIGGER.

    Returns:
        Dict of strategy_name -> performance metrics
    """
    # Get trigger dates
    trigger_dates = get_sip_trigger_dates(breadth_data, buy_threshold=buy_threshold)
    
    if not trigger_dates:
        logger.warning("No SIP trigger dates found")
        return {}
    
    # Create date lookup for indicator data
    data_by_date = {}
    for date, df in historical_data:
        key = date.date() if isinstance(date, datetime) else date
        data_by_date[key] = df
    
    sorted_data_dates = sorted(data_by_date.keys())
    
    def get_indicator_data(target_date) -> Optional[pd.DataFrame]:
        """Get indicator data for target date (or nearest available)."""
        target = target_date.date() if isinstance(target_date, datetime) else target_date
        
        if target in data_by_date:
            return data_by_date[target]
        
        # Execution Fix: O(log N) binary search replaces O(N) linear scan
        idx = bisect.bisect_left(sorted_data_dates, target)
        if idx < len(sorted_data_dates):
            return data_by_date[sorted_data_dates[idx]]
        
        return data_by_date[sorted_data_dates[-1]] if sorted_data_dates else None
    
    results = {}
    
    for strategy_name, strategy in strategies.items():
        logger.info(f"SIP Mode: Evaluating {strategy_name}...")
        
        master = MasterPortfolio()
        entries_made = 0
        
        for trigger_date in trigger_dates:
            indicator_df = get_indicator_data(trigger_date)
            
            if indicator_df is None or indicator_df.empty:
                continue
            
            try:
                # Generate portfolio for this date
                # Memory Fix: _clean_data handles numpy bounds, remove deep copy
                portfolio = strategy.generate_portfolio(indicator_df, sip_amount=sip_amount)
                
                if portfolio.empty:
                    continue
                
                # Add to master portfolio
                master.add_portfolio(portfolio, trigger_date, sip_amount)
                entries_made += 1
                
            except Exception as e:
                logger.warning(f"Error generating portfolio for {strategy_name} on {trigger_date}: {e}")
                continue
        
        # Update to terminal prices (last available data)
        if sorted_data_dates and master.holdings:
            final_df = data_by_date[sorted_data_dates[-1]]
            master.update_prices(final_df[['symbol', 'price']], datetime.now())
        
        # Get metrics
        metrics = master.get_metrics()
        metrics['mode'] = 'SIP'
        metrics['trigger'] = f'REL_BREADTH < {SIP_TRIGGER}'
        metrics['num_entries'] = entries_made
        metrics['sip_amount'] = sip_amount
        
        results[strategy_name] = metrics
        
        logger.info(f"  {strategy_name}: {entries_made} entries, Return: {metrics['total_return']:.2%}")
    
    return results


# ══════════════════════════════════════════════════════════════════════════════
# SWING MODE EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

def execute_swing_mode(
    strategies: Dict[str, Any],
    historical_data: List[Tuple[datetime, pd.DataFrame]],
    breadth_data: pd.DataFrame,
    capital_per_trade: float = DEFAULT_SIP_AMOUNT,
    buy_threshold: Optional[float] = None,
    sell_threshold: Optional[float] = None
) -> Dict[str, Dict]:
    """
    Execute Swing mode: Buy-sell cycles based on breadth triggers.

    Process:
    1. Identify cycles: Buy when REL_BREADTH < buy_threshold, Sell when >= sell_threshold
    2. For each cycle:
       - Generate portfolio at entry
       - Track to exit date
       - Calculate cycle return
    3. Aggregate all cycle returns for final metrics

    Args:
        strategies: Dict of strategy_name -> strategy instance
        historical_data: List of (date, indicator_df) from backdata
        breadth_data: DataFrame with DATE and REL_BREADTH
        capital_per_trade: Capital allocated per swing trade
        buy_threshold: Adaptive buy threshold. Defaults to SWING_BUY_TRIGGER.
        sell_threshold: Adaptive sell threshold. Defaults to SWING_SELL_TRIGGER.

    Returns:
        Dict of strategy_name -> performance metrics
    """
    # Get swing cycles
    cycles = get_swing_cycles(breadth_data, buy_threshold=buy_threshold, sell_threshold=sell_threshold)
    
    if not cycles:
        logger.warning("No swing cycles found")
        return {}
    
    # Create date lookup
    data_by_date = {}
    for date, df in historical_data:
        key = date.date() if isinstance(date, datetime) else date
        data_by_date[key] = df
    
    sorted_data_dates = sorted(data_by_date.keys())
    
    def get_indicator_data(target_date) -> Optional[pd.DataFrame]:
        target = target_date.date() if isinstance(target_date, datetime) else target_date
        if target in data_by_date:
            return data_by_date[target]
        idx = bisect.bisect_left(sorted_data_dates, target)
        if idx < len(sorted_data_dates):
            return data_by_date[sorted_data_dates[idx]]
        return data_by_date[sorted_data_dates[-1]] if sorted_data_dates else None
    
    results = {}
    
    for strategy_name, strategy in strategies.items():
        logger.info(f"Swing Mode: Evaluating {strategy_name}...")
        
        cycle_results = []
        
        for cycle in cycles:
            entry_date = cycle['entry_date']
            exit_date = cycle['exit_date']
            
            entry_df = get_indicator_data(entry_date)
            exit_df = get_indicator_data(exit_date)
            
            if entry_df is None or entry_df.empty:
                continue
            
            try:
                # Generate portfolio at entry
                portfolio = strategy.generate_portfolio(entry_df, sip_amount=capital_per_trade)
                
                if portfolio.empty:
                    continue
                
                # Calculate entry value
                entry_value = portfolio['value'].sum() if 'value' in portfolio.columns else 0
                
                if entry_value <= 0:
                    continue
                
                # Calculate exit value
                if exit_df is not None and not exit_df.empty:
                    exit_prices = exit_df.set_index('symbol')['price'].to_dict()
                    
                    exit_value = 0
                    for _, row in portfolio.iterrows():
                        symbol = row['symbol']
                        units = row.get('units', 0)
                        exit_price = exit_prices.get(symbol, row['price'])
                        exit_value += units * exit_price
                else:
                    exit_value = entry_value
                
                cycle_return = (exit_value - entry_value) / entry_value
                
                cycle_results.append({
                    'entry_date': entry_date,
                    'exit_date': exit_date,
                    'entry_value': entry_value,
                    'exit_value': exit_value,
                    'return': cycle_return,
                    'status': cycle['status'],
                    'num_holdings': len(portfolio)
                })
                
            except Exception as e:
                logger.warning(f"Error in swing cycle for {strategy_name}: {e}")
                continue
        
        # Aggregate metrics
        if cycle_results:
            df_cycles = pd.DataFrame(cycle_results)
            closed = df_cycles[df_cycles['status'] == 'closed']
            
            if len(closed) > 0:
                returns = closed['return'].values
                avg_return = returns.mean()
                win_rate = (returns > 0).mean()

                # Compounded total return
                total_return = (1 + closed['return']).prod() - 1

                # Calculate actual time span for accurate annualization
                date_start = pd.to_datetime(closed['entry_date'].min())
                date_end = pd.to_datetime(closed['exit_date'].max())
                days = (date_end - date_start).days
                years = days / 365.25 if days > 0 else 1.0
                periods_per_year = max(1.0, len(returns) / years)

                # Risk metrics — delegate to canonical when available
                if compute_risk_metrics is not None and len(returns) >= 2:
                    core = compute_risk_metrics(returns, periods_per_year=periods_per_year, total_return=total_return)
                    sharpe = core['sharpe']
                    sortino = core['sortino']
                    max_dd = core['max_drawdown']
                    calmar = core['calmar']
                elif len(returns) > 1 and returns.std() > 0:
                    ann_factor = np.sqrt(max(len(returns), 1))
                    sharpe = (avg_return / returns.std()) * ann_factor
                    downside = np.minimum(returns, 0)
                    downside_dev = np.sqrt(np.mean(downside ** 2))
                    sortino = (avg_return / downside_dev) * ann_factor if downside_dev > 0 else sharpe
                    cumulative = np.cumprod(1 + returns)
                    peak = np.maximum.accumulate(cumulative)
                    max_dd = float(np.min((cumulative - peak) / peak))
                    calmar = total_return / abs(max_dd) if max_dd < -0.001 else 0
                else:
                    sharpe = avg_return * 10 if avg_return > 0 else 0
                    sortino = sharpe
                    max_dd = 0
                    calmar = 0
                
            else:
                # Only open trades
                avg_return = df_cycles['return'].mean()
                total_return = avg_return
                win_rate = (df_cycles['return'] > 0).mean()
                sharpe = sortino = 0
                max_dd = 0
                calmar = 0
            
            n_closed = len(closed)
            results[strategy_name] = {
                'mode': 'Swing',
                'buy_trigger': f'REL_BREADTH < {SWING_BUY_TRIGGER}',
                'sell_trigger': f'REL_BREADTH >= {SWING_SELL_TRIGGER}',
                'completed_trades': n_closed,
                'open_trades': len(df_cycles) - n_closed,
                'avg_return_per_trade': avg_return,
                'total_return': total_return,
                'win_rate': win_rate,
                'sharpe': sharpe,
                'sortino': sortino,
                'max_drawdown': max_dd,
                'calmar': calmar
            }
            
            logger.info(f"  {strategy_name}: {len(cycle_results)} cycles, Return: {total_return:.2%}")
        else:
            results[strategy_name] = {
                'mode': 'Swing',
                'completed_trades': 0,
                'open_trades': 0,
                'total_return': 0,
                'sharpe': 0,
                'sortino': 0,
                'max_drawdown': 0,
                'calmar': 0,
                'win_rate': 0
            }
    
    return results


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY RANKING (DISPERSION-WEIGHTED - NO FIXED FORMULAS)
# ══════════════════════════════════════════════════════════════════════════════

def rank_strategies(metrics: Dict[str, Dict]) -> pd.DataFrame:
    """
    Rank strategies using dispersion-weighted scoring.
    
    NO FIXED FORMULAS like "0.30×Sharpe + 0.25×Sortino"
    
    Instead:
    - Each metric gets percentile rank (0-1)
    - Weights derived from cross-sectional dispersion
    - Metrics that better differentiate strategies get higher weight
    """
    if not metrics:
        return pd.DataFrame()
    
    df = pd.DataFrame.from_dict(metrics, orient='index')
    df['strategy'] = df.index
    df = df.reset_index(drop=True)
    
    # CRITICAL MATH FIX: max_drawdown is naturally computed as a negative number (e.g. -0.25).
    # Therefore, mathematically, higher (-0.10 > -0.25) is ALREADY better.
    # Placing it in negative_metrics inverted the logic, rewarding the worst drawdowns.
    positive_metrics = ['sharpe', 'sortino', 'calmar', 'win_rate', 'total_return', 'max_drawdown']
    # Penalty metrics (metrics where strictly larger positive magnitude is worse)
    negative_metrics = []
    
    # Compute normalized scales instead of uniform percentiles to preserve dispersion
    for metric in positive_metrics:
        if metric in df.columns:
            clean_s = df[metric].replace([np.inf, -np.inf], np.nan).dropna()
            if not clean_s.empty:
                col_min, col_max = clean_s.min(), clean_s.max()
                df[f'{metric}_rank'] = (df[metric] - col_min) / (col_max - col_min) if col_max > col_min else 0.5
            else:
                df[f'{metric}_rank'] = 0.5
    
    for metric in negative_metrics:
        if metric in df.columns:
            clean_s = df[metric].replace([np.inf, -np.inf], np.nan).dropna()
            if not clean_s.empty:
                col_min, col_max = clean_s.min(), clean_s.max()
                df[f'{metric}_rank'] = (col_max - df[metric]) / (col_max - col_min) if col_max > col_min else 0.5
            else:
                df[f'{metric}_rank'] = 0.5
    
    rank_cols = [c for c in df.columns if c.endswith('_rank')]
    
    if not rank_cols:
        df['score'] = 0
        return df
    
    # Compute dispersion-weighted score with absolute safety against NaN and negative values
    dispersions = {}
    for col in rank_cols:
        s = df[col].std(ddof=0)
        if pd.isna(s) or s <= 0:
            dispersions[col] = 0.01
        else:
            dispersions[col] = max(float(s), 0.01)
            
    total_disp = sum(dispersions.values())
    weights = {col: disp / total_disp for col, disp in dispersions.items()}
    
    df['score'] = sum(df[col] * weight for col, weight in weights.items())
    df = df.sort_values('score', ascending=False)
    
    # Store weights for transparency
    df.attrs['weights'] = {k.replace('_rank', ''): v for k, v in weights.items()}
    
    return df


def get_strategy_weights(rankings: pd.DataFrame, concentration: float = 0.5) -> Dict[str, float]:
    """
    Compute allocation weights from rankings.
    
    concentration: 0 = equal weight, 1 = score-proportional
    """
    if rankings.empty or 'score' not in rankings.columns:
        return {}
    
    scores = rankings.set_index('strategy')['score']
    
    score_min, score_max = scores.min(), scores.max()
    if score_max > score_min:
        normalized = (scores - score_min) / (score_max - score_min)
    else:
        normalized = pd.Series(1.0 / len(scores), index=scores.index)
    
    equal_weight = 1.0 / len(normalized)
    score_weight = normalized / normalized.sum()
    
    blended = (1 - concentration) * equal_weight + concentration * score_weight
    return (blended / blended.sum()).to_dict()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENGINE CLASS
# ══════════════════════════════════════════════════════════════════════════════

class StrategySelectionEngine:
    """
    Complete strategy selection engine.
    
    Usage:
        engine = StrategySelectionEngine()
        engine.initialize(symbols, start_date, end_date)
        
        # Run SIP evaluation
        sip_results = engine.run_sip_mode()
        
        # Run Swing evaluation  
        swing_results = engine.run_swing_mode()
        
        # Get rankings and weights
        rankings = engine.get_rankings()
        weights = engine.get_weights()
    """
    
    def __init__(
        self,
        sip_amount: float = DEFAULT_SIP_AMOUNT,
        lookback_rows: int = DEFAULT_LOOKBACK_ROWS
    ):
        self.sip_amount = sip_amount
        self.lookback_rows = lookback_rows
        
        self.breadth_data: Optional[pd.DataFrame] = None
        self.strategies: Dict[str, Any] = {}
        self.historical_data: List[Tuple[datetime, pd.DataFrame]] = []

        self.sip_results: Dict[str, Dict] = {}
        self.swing_results: Dict[str, Dict] = {}
        self.current_mode: Optional[str] = None
        self.rankings: Optional[pd.DataFrame] = None
        self.weights: Dict[str, float] = {}

        # Adaptive thresholds (computed during initialize)
        self.buy_threshold: Optional[float] = None
        self.sell_threshold: Optional[float] = None
    
    def initialize(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> bool:
        """
        Initialize engine with data.
        
        Args:
            symbols: List of stock symbols
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            True if initialization successful
        """
        # Load breadth data
        self.breadth_data = load_breadth_data(self.lookback_rows)
        if self.breadth_data.empty:
            logger.error("Failed to load breadth data")
            return False

        # Compute adaptive thresholds from breadth distribution
        self.buy_threshold, self.sell_threshold = compute_adaptive_thresholds(
            self.breadth_data['REL_BREADTH']
        )
        logger.info(f"Adaptive thresholds: buy < {self.buy_threshold}, sell >= {self.sell_threshold}")

        # Load strategies
        self.strategies = load_strategies()
        if not self.strategies:
            logger.error("Failed to load strategies")
            return False
        
        # Load historical data
        self.historical_data = load_historical_data(symbols, start_date, end_date)
        if not self.historical_data:
            logger.error("Failed to load historical data")
            return False
        
        logger.info(f"Initialized: {len(self.strategies)} strategies, {len(self.historical_data)} trading days")
        return True
    
    def run_sip_mode(self) -> Dict[str, Dict]:
        """Run SIP mode evaluation."""
        if not self._check_ready():
            return {}
        
        self.sip_results = execute_sip_mode(
            self.strategies,
            self.historical_data,
            self.breadth_data,
            self.sip_amount,
            buy_threshold=self.buy_threshold
        )
        
        self.current_mode = 'SIP'
        self.rankings = rank_strategies(self.sip_results)
        
        return self.sip_results
    
    def run_swing_mode(self) -> Dict[str, Dict]:
        """Run Swing mode evaluation."""
        if not self._check_ready():
            return {}
        
        self.swing_results = execute_swing_mode(
            self.strategies,
            self.historical_data,
            self.breadth_data,
            self.sip_amount,
            buy_threshold=self.buy_threshold,
            sell_threshold=self.sell_threshold
        )
        
        self.current_mode = 'Swing'
        self.rankings = rank_strategies(self.swing_results)
        
        return self.swing_results
    
    def get_rankings(self) -> pd.DataFrame:
        """Get strategy rankings."""
        return self.rankings if self.rankings is not None else pd.DataFrame()
    
    def get_weights(self, concentration: float = 0.5) -> Dict[str, float]:
        """Get strategy allocation weights."""
        if self.rankings is not None:
            self.weights = get_strategy_weights(self.rankings, concentration)
        return self.weights
    
    def get_triggers(self) -> Dict:
        """Get trigger configuration."""
        return {
            'sip_trigger': SIP_TRIGGER,
            'swing_buy': SWING_BUY_TRIGGER,
            'swing_sell': SWING_SELL_TRIGGER
        }
    
    def _check_ready(self) -> bool:
        """Check if engine is ready to run."""
        if self.breadth_data is None or self.breadth_data.empty:
            logger.error("Breadth data not loaded")
            return False
        if not self.strategies:
            logger.error("Strategies not loaded")
            return False
        if not self.historical_data:
            logger.error("Historical data not loaded")
            return False
        return True
    
    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            "═" * 60,
            "STRATEGY SELECTION ENGINE",
            "═" * 60,
            "",
            "TRIGGERS (Research-Derived):",
            f"  SIP:        REL_BREADTH < {SIP_TRIGGER}",
            f"  Swing Buy:  REL_BREADTH < {SWING_BUY_TRIGGER}",
            f"  Swing Sell: REL_BREADTH >= {SWING_SELL_TRIGGER}",
            "",
            f"Lookback: {self.lookback_rows} rows",
            f"SIP Amount: ₹{self.sip_amount:,.0f}",
        ]
        
        if self.breadth_data is not None and not self.breadth_data.empty:
            sip_dates = get_sip_trigger_dates(self.breadth_data)
            cycles = get_swing_cycles(self.breadth_data)
            lines.extend([
                "",
                f"SIP Trigger Dates: {len(sip_dates)}",
                f"Swing Cycles: {len(cycles)}",
            ])
        
        if self.strategies:
            lines.append(f"\nStrategies Loaded: {len(self.strategies)}")
        
        if self.current_mode:
            lines.append(f"\nCurrent Mode: {self.current_mode}")
        
        if self.weights:
            lines.append("\nTOP STRATEGY WEIGHTS:")
            for name, weight in sorted(self.weights.items(), key=lambda x: -x[1])[:5]:
                lines.append(f"  {name}: {weight:.1%}")
        
        lines.append("═" * 60)
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# SEQUENTIAL PROBABILITY RATIO TEST (REC-3)
# ══════════════════════════════════════════════════════════════════════════════
# Reference: Wald (1945), "Sequential Analysis";
#            Shiryaev (1963), "On optimum methods in quickest detection".
#
# Replaces fixed-threshold triggers (REL_BREADTH < 0.42) with a
# statistically principled sequential test that accumulates evidence
# for regime change and fires when the evidence crosses a significance
# boundary.

class SPRTRegimeTrigger:
    """
    Sequential Probability Ratio Test for regime change detection.

    Instead of a fixed threshold on REL_BREADTH, this test:
    1. Models REL_BREADTH under two hypotheses:
       H0 (neutral): REL_BREADTH ~ N(μ0, σ²)  — normal market
       H1 (stress):  REL_BREADTH ~ N(μ1, σ²)  — stressed market (buy opp)
    2. Accumulates log-likelihood ratio evidence sequentially
    3. Fires BUY when evidence exceeds upper bound (confirms H1)
    4. Fires SELL when evidence falls below lower bound (confirms H0)

    The bounds are set by Type I/II error rates (α, β), providing
    statistical control over false signals.

    Usage:
        sprt = SPRTRegimeTrigger()
        sprt.fit(historical_breadth_series)

        for breadth_value in new_observations:
            signal = sprt.update(breadth_value)
            if signal == 'BUY':
                ...
            elif signal == 'SELL':
                ...
    """

    def __init__(
        self,
        alpha: float = 0.05,   # P(false buy signal) — Type I error
        beta: float = 0.10,    # P(missed buy opportunity) — Type II error
        stress_shift: float = -0.5,  # how many σ below mean constitutes stress
        decay: float = 0.95,   # Leakage factor to prevent lock-out in chop
    ):
        # Strictly clamp error probabilities to prevent boundary calculation blow-ups
        self.alpha = max(1e-5, min(alpha, 0.5))
        self.beta = max(1e-5, min(beta, 0.5))
        self.stress_shift = stress_shift
        self.decay = decay

        # Wald boundaries (log scale)
        self.upper_bound = np.log((1 - self.beta) / self.alpha)   # accept H1 (BUY)
        self.lower_bound = np.log(self.beta / (1 - self.alpha))   # accept H0 (SELL)

        # Distribution parameters (fitted from data)
        self.mu0 = 0.5   # neutral mean
        self.mu1 = 0.3   # stress mean
        self.sigma = 0.15

        # Running state
        self.log_lr = 0.0  # cumulative log-likelihood ratio
        self.fitted = False

    def fit(self, breadth_series: Any) -> 'SPRTRegimeTrigger':
        """
        Estimate distribution parameters from historical breadth data.

        Args:
            breadth_series: Historical REL_BREADTH values.

        Returns:
            self (for chaining).
        """
        # Memory Fix: Direct numpy casting bypasses heavy Pandas series constructors in loops
        arr = np.asarray(breadth_series, dtype=float)
        clean = arr[~np.isnan(arr)]
        if len(clean) < 20:
            return self

        self.mu0 = float(np.median(clean))  # robust central tendency
        self.sigma = float(np.std(clean))
        if self.sigma < 0.01:
            self.sigma = 0.15

        # Stress hypothesis: shift below median
        self.mu1 = self.mu0 + self.stress_shift * self.sigma

        # CRITICAL FIX: Do NOT reset log_lr if already accumulating evidence!
        if not self.fitted:
            self.log_lr = 0.0
        self.fitted = True
        return self

    def update(self, breadth_value: float) -> str:
        """
        Process a new breadth observation and return signal.

        Args:
            breadth_value: Current REL_BREADTH observation.

        Returns:
            'BUY' if evidence confirms stress regime (H1),
            'SELL' if evidence confirms neutral regime (H0),
            'HOLD' if still accumulating evidence.
        """
        if not self.fitted:
            # Fallback to fixed threshold
            if breadth_value < SIP_TRIGGER:
                return 'BUY'
            elif breadth_value >= SWING_SELL_TRIGGER:
                return 'SELL'
            return 'HOLD'

        # Log-likelihood ratio increment
        # log(f(x|H1) / f(x|H0)) for Gaussian
        ll_h1 = -0.5 * ((breadth_value - self.mu1) / self.sigma) ** 2
        ll_h0 = -0.5 * ((breadth_value - self.mu0) / self.sigma) ** 2
        # Clip increment and apply decay to prevent explosive non-convergence and lock-outs in chop regimes
        self.log_lr = (self.log_lr * self.decay) + np.clip(ll_h1 - ll_h0, -50.0, 50.0)

        if self.log_lr >= self.upper_bound:
            # Evidence confirms stress → BUY
            self.log_lr = 0.0  # reset for next sequence
            return 'BUY'
        elif self.log_lr <= self.lower_bound:
            # Evidence confirms neutral → SELL
            self.log_lr = 0.0  # reset
            return 'SELL'
        else:
            return 'HOLD'

    def get_evidence_level(self) -> float:
        """Return current log-likelihood ratio (for monitoring)."""
        return self.log_lr

    def get_buy_probability(self) -> float:
        """
        Approximate probability of the stress hypothesis given current evidence.

        Uses the posterior odds: P(H1|data) / P(H0|data) = exp(log_lr)
        assuming equal priors.
        """
        odds = np.exp(np.clip(self.log_lr, -20, 20))
        return float(odds / (1 + odds))

    def reset(self):
        """Reset accumulated evidence."""
        self.log_lr = 0.0


def get_sprt_trigger_dates(
    breadth_data: 'pd.DataFrame',
    alpha: float = 0.05,
    beta: float = 0.10,
) -> tuple:
    """
    Apply SPRT to breadth data and return buy/sell trigger dates.

    This is a drop-in replacement for get_sip_trigger_dates and
    get_swing_cycles when SPRT mode is enabled.

    Args:
        breadth_data: DataFrame with DATE and REL_BREADTH columns.
        alpha: Type I error rate (false buy).
        beta: Type II error rate (missed buy).

    Returns:
        (buy_dates, sell_dates) — lists of datetime objects.
    """
    if breadth_data.empty or 'REL_BREADTH' not in breadth_data.columns:
        return [], []

    sprt = SPRTRegimeTrigger(alpha=alpha, beta=beta)

    buy_dates = []
    sell_dates = []
    historical_buffer = []

    has_date_col = 'DATE' in breadth_data.columns

    for idx, row in breadth_data.iterrows():
        val = row['REL_BREADTH']
        historical_buffer.append(val)
        
        # Fit only on past data to prevent lookahead bias
        # Execution Fix: Bound fit to rolling 100-day window to prevent O(N^2) memory 
        # accumulation and allow the baseline to track current regime macro shifts.
        if len(historical_buffer) > 20:
            sprt.fit(historical_buffer[-100:-1])
            
        signal = sprt.update(val)
        date_val = row['DATE'] if has_date_col else idx
        if signal == 'BUY':
            buy_dates.append(date_val)
        elif signal == 'SELL':
            sell_dates.append(date_val)

    logger.info(
        f"SPRT triggers: {len(buy_dates)} buy, {len(sell_dates)} sell "
        f"(α={alpha}, β={beta})"
    )
    return buy_dates, sell_dates


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    'StrategySelectionEngine',
    'load_breadth_data',
    'load_strategies',
    'load_historical_data',
    'get_sip_trigger_dates',
    'get_swing_cycles',
    'execute_sip_mode',
    'execute_swing_mode',
    'rank_strategies',
    'get_strategy_weights',
    'MasterPortfolio',
    'SPRTRegimeTrigger',
    'get_sprt_trigger_dates',
    'compute_adaptive_thresholds',
    'SIP_TRIGGER',
    'SWING_BUY_TRIGGER',
    'SWING_SELL_TRIGGER',
    'BREADTH_SHEET_URL',
]
