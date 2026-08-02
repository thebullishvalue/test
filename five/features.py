"""
Return construction and causal feature engineering.

Every transform here is *causal*: the value at time t uses only information
available at t. That is a hard requirement — the engine is evaluated
walk-forward, and a single centred or full-sample statistic would leak the
future into the oscillator and make the backtest meaningless.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .universe import YIELD_SYMBOLS


def compute_returns(prices: pd.DataFrame, winsor_z: float = 8.0) -> pd.DataFrame:
    """Period-over-period changes, per-instrument in the appropriate space.

    Prices use log returns. Yield series (^TNX and friends) use simple
    differences: a yield moving 0.30 -> 0.05 is a 25bp move, not a -179%
    return, and logs of near-zero yields explode.

    Extreme observations are winsorised against a *trailing* robust scale so
    a single bad tick or a stock split artefact cannot dominate a covariance
    estimate. The clip level is causal (shifted), so no future information
    enters.
    """
    px = prices.copy()

    yield_cols = [c for c in px.columns if c in YIELD_SYMBOLS]
    price_cols = [c for c in px.columns if c not in YIELD_SYMBOLS]

    rets = pd.DataFrame(index=px.index, columns=px.columns, dtype="float64")
    if price_cols:
        safe = px[price_cols].where(px[price_cols] > 0)
        rets[price_cols] = np.log(safe).diff()
    if yield_cols:
        rets[yield_cols] = px[yield_cols].diff() / 100.0  # percent -> decimal

    rets = rets.replace([np.inf, -np.inf], np.nan)

    if winsor_z and winsor_z > 0:
        scale = (rets.abs().rolling(252, min_periods=60).median()
                 .shift(1).replace(0, np.nan))
        limit = winsor_z * 1.4826 * scale
        rets = rets.clip(lower=-limit, upper=limit, axis=0)

    # A missing return is "no information", which is zero in standardised
    # space. Instruments that are structurally absent were dropped upstream.
    return rets.fillna(0.0)


def rolling_zscore(df: pd.DataFrame | pd.Series, window: int = 63,
                   min_periods: int = 30, robust: bool = False):
    """Trailing standardisation, inclusive of the current observation.

    With ``robust`` the location/scale pair is median/MAD, which stops a
    single crisis print from flattening months of subsequent readings.
    """
    if robust:
        loc = df.rolling(window, min_periods=min_periods).median()
        mad = (df - loc).abs().rolling(window, min_periods=min_periods).median()
        scale = (1.4826 * mad).replace(0, np.nan)
    else:
        loc = df.rolling(window, min_periods=min_periods).mean()
        scale = df.rolling(window, min_periods=min_periods).std().replace(0, np.nan)

    z = (df - loc) / scale
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def ewma_vol(rets: pd.DataFrame | pd.Series, halflife: float = 21.0,
             annualise: bool = False):
    """Exponentially-weighted volatility, causal."""
    v = rets.ewm(halflife=halflife, min_periods=10).std()
    return v * np.sqrt(252) if annualise else v


def cross_sectional_dispersion(rets: pd.DataFrame) -> pd.Series:
    """Cross-sectional standard deviation of returns — a dispersion proxy.

    High dispersion with low index volatility is the classic signature of a
    stock-picker's / mean-reverting market; low dispersion with high index
    volatility signals correlated risk-off.
    """
    return rets.std(axis=1)


def average_pairwise_correlation(factor_scores: np.ndarray,
                                 explained: np.ndarray) -> float:
    """Approximate mean pairwise correlation from a factor decomposition.

    If the first principal component explains a share ``p`` of total variance
    across ``n`` standardised series, mean pairwise correlation is
    approximately ``(n * p - 1) / (n - 1)``.
    """
    if explained is None or len(explained) == 0:
        return np.nan
    n = max(factor_scores.shape[1], 2)
    p = float(explained[0])
    return float((n * p - 1.0) / (n - 1.0))


def sample_weights(n: int, halflife_days: float, ages: np.ndarray | None = None
                   ) -> np.ndarray:
    """Exponential recency weights over a calibration window.

    ``halflife_days <= 0`` yields uniform weights.
    """
    if ages is None:
        ages = np.arange(n)[::-1].astype(float)   # 0 == most recent
    if halflife_days is None or halflife_days <= 0:
        return np.ones(n)
    return np.power(0.5, ages / float(halflife_days))


def realised_beta(target: pd.Series, factor: pd.Series, window: int = 63) -> pd.Series:
    """Rolling univariate beta of target on a factor, causal."""
    cov = target.rolling(window, min_periods=window // 2).cov(factor)
    var = factor.rolling(window, min_periods=window // 2).var().replace(0, np.nan)
    return (cov / var).fillna(0.0)
