import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import warnings
import os
from typing import List, Tuple, Dict, Any
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("backdata")

warnings.filterwarnings("ignore", category=FutureWarning)


class LiquidityOscillator:
    """Calculates the Liquidity Oscillator indicator."""

    def __init__(self, length: int = 20, impact_window: int = 3):
        if length <= 0 or impact_window <= 0:
            raise ValueError("length and impact_window must be positive integers.")
        self.length = length
        self.impact_window = impact_window

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        required_columns = {'open', 'high', 'low', 'close', 'volume'}
        if not required_columns.issubset(data.columns):
            return pd.Series(dtype=float)

        df = data.copy()
        df['spread'] = (df['high'] + df['low']) / 2 - df['open']
        df['vol_ma'] = df['volume'].rolling(window=self.length).mean()
        safe_vol_ma = df['vol_ma'].replace(0, np.nan)
        df['vwap_spread'] = (df['spread'] * df['volume'] / safe_vol_ma).rolling(window=self.length).mean()
        close_shifted = df['close'].shift(self.impact_window)
        df['price_impact'] = ((df['close'] - close_shifted) * df['volume'] / safe_vol_ma).rolling(window=self.length).mean()
        df['liquidity_score'] = df['vwap_spread'] - df['price_impact']
        df['source_value'] = df['close'] + df['liquidity_score']
        df['lowest_value'] = df['source_value'].rolling(window=self.length).min()
        df['highest_value'] = df['source_value'].rolling(window=self.length).max()
        range_value = df['highest_value'] - df['lowest_value']
        safe_range_value = range_value.replace(0, np.nan)
        oscillator = 200 * (df['source_value'] - df['lowest_value']) / safe_range_value - 100
        return oscillator.rename('liquidity_oscillator')

def resample_data(df: pd.DataFrame, rule: str = 'W-FRI') -> pd.DataFrame:
    """Resample daily OHLCV data to a different timeframe.

    Uses last-available trading day per week to handle NSE holidays
    and incomplete current weeks, preventing temporal lookahead bias 
    and ensuring current-week data is never dropped during reindexing.
    """
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return pd.DataFrame()
        
    # Group by standard pandas period to maintain week boundaries
    grouped = df.groupby(pd.Grouper(freq=rule))
    agg_map = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    resampled = grouped.agg(agg_map)
    
    # CRITICAL FIX: Align index to the actual LAST available trading day in that period.
    # This prevents dropping the current incomplete week (e.g., Wed) during daily reindexing.
    # Execution Fix: Fully vectorized retrieval bypasses O(N) lambda evaluation
    last_dates = df.index.to_series().groupby(pd.Grouper(freq=rule)).last()
    resampled.index = last_dates
    resampled = resampled[resampled.index.notna()]
    
    # Drop past weeks with < 2 trading days, but explicitly retain the current (last) week 
    # even if it only has 1 day (e.g., running backtest on a Monday).
    counts = grouped['close'].count()
    counts.index = last_dates
    valid_mask = (counts >= 2) | (counts.index == df.index[-1])
    
    return resampled[valid_mask]


def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI) using Wilder's smoothing."""
    if data.empty or 'close' not in data.columns or len(data) < period:
        return pd.Series(index=data.index, dtype=float)

    delta = data['close'].diff(1)
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    # CRITICAL FIX: adjust=False strictly enforces J.W. Wilder's original exponential smoothing.
    # Default adjust=True calculates a standard EMA, causing RSI divergence vs canonical platforms.
    avg_gain = gain.ewm(com=period - 1, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period, adjust=False).mean()

    # CRITICAL FIX: Mathematically isolate exact 0.0 division without polluting warmup NaNs
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = pd.Series(100.0 - (100.0 / (1.0 + rs)), index=data.index)
    rsi = np.where((avg_loss == 0.0) & (avg_gain > 0.0), 100.0, rsi)
    
    return pd.Series(rsi, index=data.index)

def calculate_all_indicators(
    symbol_data: pd.DataFrame,
    oscillator_calculator: LiquidityOscillator
) -> pd.DataFrame | None:
    """
    Calculate all indicators for a single symbol's full history.

    Returns a DataFrame indexed by date with columns for price, returns,
    oscillators, RSI, moving averages, deviations, and z-scores across
    daily and weekly timeframes.  Returns ``None`` on empty input.
    """
    daily_data = symbol_data.copy()
    if daily_data.empty:
        return None

    weekly_data = resample_data(daily_data, 'W-FRI')
    
    all_results_df = pd.DataFrame(index=daily_data.index)
    all_results_df['price'] = daily_data['close']
    all_results_df['% change'] = daily_data['close'].pct_change()

    timeframes = {'latest': daily_data, 'weekly': weekly_data}
    
    for tf_name, df in timeframes.items():
        if len(df) < 2:
            continue
        
        osc = oscillator_calculator.calculate(df)
        if not osc.dropna().empty:
            all_results_df[f'osc {tf_name}'] = osc
            all_results_df[f'9ema osc {tf_name}'] = osc.ewm(span=9).mean()
            all_results_df[f'21ema osc {tf_name}'] = osc.ewm(span=21).mean()

            if len(osc.dropna()) >= 20:
                osc_sma20 = osc.rolling(window=20).mean()
                osc_std20 = osc.rolling(window=20).std()
                safe_std20 = osc_std20.replace(0, np.nan)
                all_results_df[f'zscore {tf_name}'] = (osc - osc_sma20) / safe_std20

        rsi_series = calculate_rsi(df)
        if rsi_series is not None and not rsi_series.dropna().empty:
            all_results_df[f'rsi {tf_name}'] = rsi_series

        for period in [20, 90, 200]:
            if len(df) >= period:
                all_results_df[f'ma{period} {tf_name}'] = df['close'].rolling(window=period).mean()
                if period == 20:
                    all_results_df[f'dev{period} {tf_name}'] = df['close'].rolling(window=period).std()

    all_results_df = all_results_df.reindex(daily_data.index)
    
    weekly_cols = [col for col in all_results_df.columns if 'weekly' in col]
    all_results_df[weekly_cols] = all_results_df[weekly_cols].ffill()
    
    return all_results_df


def load_symbols_from_file(filepath: str = "symbols.txt") -> List[str]:
    """Load a list of stock symbols from a text file (one per line)."""
    if not os.path.exists(filepath):
        logger.error("Symbol file not found at: %s", filepath)
        return []

    try:
        with open(filepath, 'r') as f:
            symbols = [line.strip().upper() for line in f if line.strip()]
        logger.info("Loaded %d symbols from %s", len(symbols), filepath)
        # MEDIUM-2: Survivorship bias warning — a static symbol file only
        # contains stocks that exist TODAY.  Any stock removed from the index
        # (delisted, merged, or dropped from NIFTY) is absent, systematically
        # overstating backtest returns for mean-reversion strategies.
        logger.warning(
            "SURVIVORSHIP BIAS: symbols.txt is a static file containing only "
            "current constituents.  Historical backtest results may be upward-"
            "biased because delisted or removed stocks are excluded.  Consider "
            "using a point-in-time constituent list for rigorous backtesting."
        )
        return symbols
    except Exception as e:
        logger.error("Error reading symbol file %s: %s", filepath, e)
        return []

# Load the fixed universe
SYMBOLS_UNIVERSE = load_symbols_from_file()

# Define the column order here so it can be used by the generator
COLUMN_ORDER = [
    'date', 'symbol', 'price', 'rsi latest', 'rsi weekly',
    '% change', 'osc latest', 'osc weekly',
    '9ema osc latest', '9ema osc weekly',
    '21ema osc latest', '21ema osc weekly',
    'zscore latest', 'zscore weekly',
    'ma20 latest', 'ma90 latest', 'ma200 latest',
    'ma20 weekly', 'ma90 weekly', 'ma200 weekly',
    'dev20 latest', 'dev20 weekly'
]

# --- NEW: Export max indicator period ---
INDICATOR_PERIODS = [20, 90, 200]
MAX_INDICATOR_PERIOD = max(INDICATOR_PERIODS)


def generate_historical_data(
    symbols_to_process: List[str],
    start_date: datetime,
    end_date: datetime,
) -> List[Tuple[datetime, pd.DataFrame]]:
    """
    Generate historical indicator snapshots for a list of symbols.

    Args:
        symbols_to_process: Stock ticker symbols (e.g. ``["RELIANCE.NS"]``).
        start_date: Beginning of the download window (must include warmup).
        end_date: End of the snapshot window.

    Returns:
        Chronologically ordered list of ``(date, indicator_df)`` tuples.
    """
    if not symbols_to_process:
        logger.error("No symbols provided to generate_historical_data.")
        return []

    # --- Download ---
    try:
        logger.info("yfinance: downloading %d symbols...", len(symbols_to_process))
        all_data = yf.download(
            symbols_to_process,
            start=start_date,
            end=end_date + timedelta(days=1),  # yfinance is end-exclusive
            progress=False,
            auto_adjust=True,  # Prevent stock splits from causing massive false indicator spikes
        )
    except Exception as e:
        logger.error("yf.download failed: %s", e)
        all_data = pd.DataFrame()

    # Prevent after-hours lookahead bias
    if not all_data.empty:
        if all_data.index.tz is not None:
            all_data.index = all_data.index.tz_localize(None)
        # Strip time and strictly clip to requested end_date to drop Yahoo's live/partial injections
        all_data.index = all_data.index.normalize()
        end_boundary = pd.to_datetime(end_date).normalize()
        all_data = all_data.loc[all_data.index <= end_boundary]

    if all_data.empty or all_data['Close'].dropna(how='all').empty:
        logger.error("yf.download returned empty or all-NaN Close data.")
        return []

    # --- Remove failed tickers ---
    if len(symbols_to_process) > 1:
        valid_tickers = all_data['Close'].dropna(how='all', axis=1).columns
        invalid_tickers = [s for s in symbols_to_process if s not in valid_tickers]

        if invalid_tickers:
            logger.warning("Skipping tickers with no data: %s", ", ".join(invalid_tickers))
            all_data = all_data.loc[:, (slice(None), valid_tickers)]
            symbols_to_process = list(valid_tickers)

            if not symbols_to_process:
                logger.error("No valid tickers remaining after download.")
                return []

    logger.info("Download OK. Shape: %s, tickers: %d", all_data.shape, len(symbols_to_process))

    all_data.columns.names = ['Indicator', 'Symbol']
    oscillator_calculator = LiquidityOscillator(length=20, impact_window=3)
    
    # 2. --- Pre-calculate all indicators for all symbols ---
    ticker_indicator_cache = {}
    for i, ticker in enumerate(symbols_to_process):
        try:
            if len(symbols_to_process) > 1:
                symbol_df = all_data.xs(ticker, level='Symbol', axis=1).copy()
            else:
                symbol_df = all_data.copy()
                
            symbol_df.columns = [col.lower() for col in symbol_df.columns]
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in symbol_df.columns:
                    symbol_df[col] = pd.to_numeric(symbol_df[col], errors='coerce')
            
            symbol_df = symbol_df.dropna(subset=['close', 'volume'])
            symbol_df.name = ticker
            
            if not symbol_df.empty:
                indicators_df = calculate_all_indicators(symbol_df, oscillator_calculator)
                ticker_indicator_cache[ticker] = indicators_df
                
        except (pd.errors.DataError, KeyError, IndexError) as e:
            logger.warning("Skipping %s: data quality error during indicator calculation (%s)", ticker, e)
            continue

    # 3. --- Generate Daily Snapshots in Memory ---
    pragati_data_list: List[Tuple[datetime, pd.DataFrame]] = []
    
    if not ticker_indicator_cache:
        return []

    # Memory Optimization: Apply temporal boundary before concatenation.
    # Prevents O(N * Warmup) memory bloat by strictly joining required rows.
    all_ticker_dfs = []
    valid_start = start_date + timedelta(days=MAX_INDICATOR_PERIOD)
    
    for ticker, df in ticker_indicator_cache.items():
        mask = (df.index >= valid_start) & (df.index <= end_date)
        sliced_df = df.loc[mask]
        if not sliced_df.empty:
            # .assign() creates a rapid shallow copy to prevent memory fragmentation
            all_ticker_dfs.append(sliced_df.assign(symbol=ticker.replace('.NS', '')))
            
    if not all_ticker_dfs:
        return []
        
    master_df = pd.concat(all_ticker_dfs)
    master_df = master_df.dropna(subset=['price'])
    
    # Vectorized formatting
    master_df['date'] = master_df.index.strftime('%d %b')
    master_df['% change'] = master_df['% change'] * 100
    
    # Ensure column contract
    for col in COLUMN_ORDER:
        if col not in master_df.columns:
            # Prevent Pandas extension type pollution which crashes downstream RMT/NumPy math
            master_df[col] = np.nan
    master_df = master_df[COLUMN_ORDER]
    
    # Group by index (date) and build the final list
    for snapshot_date, group_df in master_df.groupby(master_df.index):
        pragati_data_list.append((snapshot_date, group_df.reset_index(drop=True)))
        
    # Ensure chronological order
    pragati_data_list.sort(key=lambda x: x[0])

    return pragati_data_list


def main():
    """Standalone Streamlit UI for generating indicator snapshots."""
    import streamlit as st
    import zipfile
    import shutil

    st.set_page_config(
        page_title="Indicator Snapshot Generator (Optimized)",
        page_icon="⚡",
        layout="wide"
    )
    
    st.title("📊 Daily Indicator Snapshot Generator (Optimized)")

    with st.sidebar:
        st.header("1. Select Date Range")
        today = datetime.now()
        # --- UPDATED: Default start date to be far enough back for indicators
        default_start = today - timedelta(days=MAX_INDICATOR_PERIOD + 90)
        start_date = st.date_input("Start Date", default_start)
        end_date = st.date_input("End Date", today)

        st.header("2. Ticker Universe")
        if SYMBOLS_UNIVERSE:
            st.info(f"Using fixed universe from `symbols.txt` ({len(SYMBOLS_UNIVERSE)} tickers).")
            with st.expander("View Tickers"):
                st.dataframe(SYMBOLS_UNIVERSE, width='stretch')
        else:
            st.error("`symbols.txt` not found or is empty. Cannot proceed.")
        
        st.header("3. Generate")
        process_button = st.button("Generate Snapshots", type="primary", width='stretch')

    if process_button:
        if start_date > end_date:
            st.error("Error: Start date cannot be after end date.")
        elif not SYMBOLS_UNIVERSE:
            st.error("Error: Cannot generate data. `symbols.txt` is missing or empty.")
        else:
            symbols_to_process = SYMBOLS_UNIVERSE
            
            fetch_start_date = start_date - timedelta(days=int(MAX_INDICATOR_PERIOD * 1.5) + 30)

            with st.spinner(f"Generating historical data from {fetch_start_date} to {end_date}..."):
                all_generated_data = generate_historical_data(
                    symbols_to_process, 
                    fetch_start_date, # Pass the earlier date for indicator warmup
                    end_date
                )
            
            if not all_generated_data:
                st.error("Failed to generate any data.")
                return

            # --- Filter the generated data to *only* the user's requested date range
            all_generated_data = [
                (date, df) for date, df in all_generated_data 
                if date.date() >= start_date and date.date() <= end_date
            ]
            
            if not all_generated_data:
                st.warning("Data was fetched, but no valid trading days found in the selected Start/End range.")
                return

            base_dir = "data"
            reports_dir = os.path.join(base_dir, "historical")
            zip_dir = os.path.join(base_dir, "zip")

            if os.path.exists(base_dir):
                shutil.rmtree(base_dir)
            os.makedirs(reports_dir)
            os.makedirs(zip_dir)

            st.info("Saving daily snapshots to 'data/historical' folder...")
            progress_bar = st.progress(0)
            last_day_df = pd.DataFrame()

            if all_generated_data:
                for i, (snapshot_date, final_df) in enumerate(all_generated_data):
                    if not final_df.empty:
                        last_day_df = final_df
                        filename = os.path.join(reports_dir, f"{snapshot_date.strftime('%Y-%m-%d')}.csv")
                        final_df.to_csv(filename, index=False, float_format='%.2f')
                    
                    progress_bar.progress((i + 1) / len(all_generated_data))
            
                zip_file_name_only = f"indicator_reports_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.zip"
                zip_full_path = os.path.join(zip_dir, zip_file_name_only)
                
                with zipfile.ZipFile(zip_full_path, 'w') as zipf:
                    for root, _, files in os.walk(reports_dir):
                        for file in files:
                            zipf.write(os.path.join(root, file), os.path.join(os.path.basename(root), file))

                st.success("✅ Snapshots and Zip file generated successfully in the 'data' folder!")
                
                st.subheader(f"Data for {end_date.strftime('%Y-%m-%d')} (Last Day)")
                if not last_day_df.empty:
                    st.dataframe(last_day_df[COLUMN_ORDER].round(2))
                
                with open(zip_full_path, "rb") as fp:
                    st.download_button(
                        label="⬇️ Download All Reports (.zip)",
                        data=fp,
                        file_name=zip_file_name_only,
                        mime="application/zip"
                    )
            else:
                st.warning("No data was generated for the selected date range.")

__all__ = [
    'LiquidityOscillator',
    'resample_data',
    'calculate_rsi',
    'calculate_all_indicators',
    'load_symbols_from_file',
    'generate_historical_data',
    'SYMBOLS_UNIVERSE',
    'MAX_INDICATOR_PERIOD',
]

if __name__ == "__main__":
    main()
