"""
MASTER × Pragyam — Phase 0: Market Status Vector Engine
=========================================================

Constructs the market status vector mτ from NIFTY50 index data,
following the MASTER paper (Li et al., AAAI-24). The vector encodes
current market price plus rolling statistics of price and volume over
multiple windows, providing a compact representation of market regime.

Uses NIFTYIETF.NS (already in symbols.txt) — zero additional API calls.

Author: Hemrek Capital
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import logging

logger = logging.getLogger("master_market")

# Rolling windows for market statistics (from MASTER paper)
ROLLING_WINDOWS = [5, 10, 20, 30, 60]

# Market proxy symbol (already in symbols.txt)
MARKET_PROXY_SYMBOL = "NIFTYIETF.NS"

# Dimension of the market status vector mτ
# 1 current price + 5 price means + 5 price stds + 5 vol means + 5 vol stds
MARKET_VECTOR_DIM = 1 + len(ROLLING_WINDOWS) * 4  # = 21


class MarketStatusVector:
    """Constructs the market status vector mτ from index price and volume data.

    The vector captures current market state through rolling statistics:
        mτ = [current_close, price_mean_5, ..., price_mean_60,
               price_std_5, ..., price_std_60,
               vol_mean_5, ..., vol_mean_60,
               vol_std_5, ..., vol_std_60]

    Args:
        market_symbol: Ticker symbol for the market index proxy.
            Defaults to NIFTYIETF.NS.
    """

    def __init__(self, market_symbol: str = MARKET_PROXY_SYMBOL):
        self.market_symbol = market_symbol
        self._price_series: Optional[pd.Series] = None
        self._volume_series: Optional[pd.Series] = None
        self._precomputed: Optional[pd.DataFrame] = None

    def fit(self, all_data: pd.DataFrame) -> "MarketStatusVector":
        """Extract market proxy price/volume from the yfinance batch download.

        Args:
            all_data: Multi-level DataFrame from yf.download() with
                columns levels ['Indicator', 'Symbol'], or a single-symbol
                DataFrame with columns ['open', 'high', 'low', 'close', 'volume'].

        Returns:
            self, for method chaining.
        """
        try:
            if isinstance(all_data.columns, pd.MultiIndex):
                symbol_df = all_data.xs(
                    self.market_symbol, level='Symbol', axis=1
                ).copy()
                symbol_df.columns = [c.lower() for c in symbol_df.columns]
            else:
                symbol_df = all_data.copy()
                symbol_df.columns = [c.lower() for c in symbol_df.columns]

            symbol_df = symbol_df.dropna(subset=['close', 'volume'])

            self._price_series = pd.to_numeric(
                symbol_df['close'], errors='coerce'
            )
            self._volume_series = pd.to_numeric(
                symbol_df['volume'], errors='coerce'
            )

            # Pre-compute all rolling statistics for efficiency
            self._precomputed = self._precompute_rolling()

            logger.info(
                "MarketStatusVector fitted on %s: %d trading days",
                self.market_symbol, len(self._price_series),
            )
        except (KeyError, IndexError) as e:
            logger.error(
                "Failed to extract %s from downloaded data: %s",
                self.market_symbol, e,
            )
            self._price_series = None
            self._volume_series = None
            self._precomputed = None

        return self

    def _precompute_rolling(self) -> pd.DataFrame:
        """Pre-compute all rolling statistics for all dates."""
        records = {}

        price = self._price_series
        volume = self._volume_series

        records['current_price'] = price

        for d in ROLLING_WINDOWS:
            records[f'price_mean_{d}'] = price.rolling(window=d, min_periods=d).mean()
            records[f'price_std_{d}'] = price.rolling(window=d, min_periods=d).std()
            records[f'vol_mean_{d}'] = volume.rolling(window=d, min_periods=d).mean()
            records[f'vol_std_{d}'] = volume.rolling(window=d, min_periods=d).std()

        return pd.DataFrame(records)

    def get_vector(self, snapshot_date: pd.Timestamp) -> Optional[np.ndarray]:
        """Return the market status vector mτ for a given date.

        Args:
            snapshot_date: Trading date for which to retrieve mτ.

        Returns:
            numpy array of shape (21,) or None if data unavailable.
        """
        if self._precomputed is None:
            logger.warning("MarketStatusVector not fitted. Call fit() first.")
            return None

        # Normalize the date
        snapshot_date = pd.Timestamp(snapshot_date).normalize()

        if snapshot_date not in self._precomputed.index:
            # Try the most recent available date before snapshot_date
            valid_dates = self._precomputed.index[
                self._precomputed.index <= snapshot_date
            ]
            if valid_dates.empty:
                logger.warning(
                    "No market data available for or before %s", snapshot_date
                )
                return None
            snapshot_date = valid_dates[-1]

        row = self._precomputed.loc[snapshot_date]

        if row.isnull().any():
            logger.debug(
                "Partial NaN in market vector for %s (early history).",
                snapshot_date,
            )
            # Fill NaN with 0 for early dates where rolling windows aren't full
            row = row.fillna(0.0)

        return row.values.astype(np.float64)

    def get_vector_normalized(
        self, snapshot_date: pd.Timestamp
    ) -> Optional[np.ndarray]:
        """Return a z-score normalized market status vector.

        Normalizes each component by its historical mean and std up to
        (and including) snapshot_date to prevent look-ahead bias.

        Args:
            snapshot_date: Trading date for which to retrieve mτ.

        Returns:
            numpy array of shape (21,) or None if data unavailable.
        """
        if self._precomputed is None:
            return None

        snapshot_date = pd.Timestamp(snapshot_date).normalize()

        # Use only data up to snapshot_date for normalization
        historical = self._precomputed.loc[
            self._precomputed.index <= snapshot_date
        ]

        if len(historical) < 60:
            # Not enough history for stable normalization
            return self.get_vector(snapshot_date)

        means = historical.mean()
        stds = historical.std().replace(0, 1.0)

        raw = self.get_vector(snapshot_date)
        if raw is None:
            return None

        return ((raw - means.values) / stds.values).astype(np.float64)

    @property
    def is_fitted(self) -> bool:
        """Whether the market status vector has been fitted with data."""
        return self._precomputed is not None

    @property
    def dim(self) -> int:
        """Dimensionality of the market status vector."""
        return MARKET_VECTOR_DIM
