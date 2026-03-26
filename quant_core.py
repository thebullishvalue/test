"""
Quantitative Core — Research-Grounded Mathematical Infrastructure for Pragyam.

Implements fixes and enhancements identified in the adversarial audit:
  C-1: Adaptive Quantile Gates (replaces fixed thresholds)
  C-2: Deflated Sharpe Ratio (corrects multiple testing bias)
  C-4: Empirical Quantile Rank (replaces Gaussian z-scores on non-Gaussian data)
  C-5: Validated Liquidity Indicators (Amihud, Corwin-Schultz)
  H-4: Market Impact Model (Almgren-Chriss inspired)
  H-2: Rank-Based Cross-Sectional Normalization (replaces StandardScaler)
  H-5: Conformal Intervals with Serial Dependence Correction
  M-2: Distribution-Aware NaN Imputation

References:
  [1] Bailey & López de Prado (2014), "The Deflated Sharpe Ratio"
  [2] Amihud (2002), "Illiquidity and Stock Returns"
  [3] Corwin & Schultz (2012), "A Simple Way to Estimate Bid-Ask Spreads"
  [4] Almgren & Chriss (2001), "Optimal Execution of Portfolio Transactions"
  [5] Harvey, Liu & Zhu (2016), "...and the Cross-Section of Expected Returns"
  [6] Barber, Candès, Ramdas & Tibshirani (2023), "Conformal Prediction Beyond
      Exchangeability"

Zero coupling to Streamlit.  Dependencies: numpy, scipy, pandas.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger("quant_core")

_EPS = 1e-10


# ═══════════════════════════════════════════════════════════════════════════════
# C-1: ADAPTIVE QUANTILE GATES
# ═══════════════════════════════════════════════════════════════════════════════
# Replaces every hardcoded threshold (RSI < 30, OSC < -80, etc.) with its
# empirical quantile from an expanding window. Makes thresholds adaptive to
# the evolving distribution and eliminates Gaussian assumptions.
#
# Reference: López de Prado (2018), "Advances in Financial Machine Learning",
# Ch. 5 — fractionally differentiated features & adaptive thresholds.
# ═══════════════════════════════════════════════════════════════════════════════

class AdaptiveQuantileGate:
    """Expanding-window quantile threshold estimator.

    Instead of ``indicator < -80`` (fixed), use::

        gate = AdaptiveQuantileGate(percentile=5)
        threshold = gate.threshold(indicator_history)
        signal = indicator_current < threshold

    The percentile is the *desired tail probability*. The actual threshold
    value adapts to the empirical distribution of the indicator.

    For exponentially weighted quantiles (to handle regime transitions),
    set ``halflife`` to a positive integer (number of observations).
    """

    def __init__(
        self,
        percentile: float,
        min_observations: int = 50,
        halflife: Optional[int] = None,
        fallback_value: float = 0.0,
    ):
        if not 0 < percentile < 100:
            raise ValueError(f"percentile must be in (0, 100), got {percentile}")
        self.percentile = percentile
        self.min_observations = min_observations
        self.halflife = halflife
        self.fallback_value = fallback_value

    def threshold(self, history: np.ndarray) -> float:
        """Compute the adaptive threshold from an expanding window.

        Args:
            history: 1-D array of past indicator values (newest last).

        Returns:
            The threshold value at the requested percentile.
        """
        h = np.asarray(history, dtype=np.float64)
        h = h[np.isfinite(h)]

        if len(h) < self.min_observations:
            return self.fallback_value

        if self.halflife is not None and self.halflife > 0:
            # Exponentially weighted quantile via weighted percentile
            n = len(h)
            decay = np.log(2) / self.halflife
            weights = np.exp(-decay * np.arange(n - 1, -1, -1))
            sorted_idx = np.argsort(h)
            sorted_h = h[sorted_idx]
            sorted_w = weights[sorted_idx]
            cumw = np.cumsum(sorted_w)
            cumw /= cumw[-1]
            idx = np.searchsorted(cumw, self.percentile / 100.0)
            idx = min(idx, len(sorted_h) - 1)
            return float(sorted_h[idx])
        else:
            return float(np.percentile(h, self.percentile))

    def threshold_vectorized(self, history_matrix: np.ndarray) -> np.ndarray:
        """Compute thresholds for multiple indicators at once.

        Args:
            history_matrix: 2-D array (T x N) where each column is an indicator.

        Returns:
            1-D array of thresholds, one per column.
        """
        T, N = history_matrix.shape
        thresholds = np.full(N, self.fallback_value)
        if T < self.min_observations:
            return thresholds
        for j in range(N):
            col = history_matrix[:, j]
            finite = col[np.isfinite(col)]
            if len(finite) >= self.min_observations:
                thresholds[j] = float(np.percentile(finite, self.percentile))
        return thresholds


def compute_adaptive_multiplier(
    value: float,
    history: np.ndarray,
    percentile_tiers: List[float],
    multiplier_tiers: List[float],
    min_obs: int = 50,
    halflife: Optional[int] = None,
) -> float:
    """Compute an adaptive multiplier using quantile-gated tiers.

    Replaces the fixed np.select() cascades in all strategies.

    Args:
        value: Current indicator value for one stock.
        history: Expanding-window history of that indicator across dates.
        percentile_tiers: Ascending list of percentiles (e.g. [5, 10, 20, 30, 50]).
        multiplier_tiers: Corresponding multiplier for each tier (same length + 1
            for the default above the last percentile).
        min_obs: Minimum observations before adaptive thresholds activate.
        halflife: Exponential weighting halflife (None = uniform).

    Returns:
        The multiplier for the current value.

    Example::

        # Old: np.select([rsi < 30, rsi < 50, rsi < 70], [3.5, 2.0, 1.0], default=0.3)
        # New:
        mult = compute_adaptive_multiplier(
            value=rsi_current,
            history=rsi_history,
            percentile_tiers=[10, 30, 60],   # adaptive equivalents of 30, 50, 70
            multiplier_tiers=[3.5, 2.0, 1.0, 0.3],  # len = len(tiers) + 1
        )
    """
    h = np.asarray(history, dtype=np.float64)
    h = h[np.isfinite(h)]

    if len(h) < min_obs:
        # Insufficient data: return the middle multiplier (neutral)
        mid = len(multiplier_tiers) // 2
        return multiplier_tiers[mid]

    # Compute quantile thresholds
    if halflife is not None and halflife > 0:
        gate = AdaptiveQuantileGate(percentile=50, halflife=halflife)
        thresholds = []
        for p in percentile_tiers:
            gate.percentile = p
            thresholds.append(gate.threshold(h))
    else:
        thresholds = [float(np.percentile(h, p)) for p in percentile_tiers]

    # Find which tier the current value falls into
    for i, thresh in enumerate(thresholds):
        if value < thresh:
            return multiplier_tiers[i]
    return multiplier_tiers[-1]  # default (above all thresholds)


def compute_adaptive_multiplier_vectorized(
    values: np.ndarray,
    history: np.ndarray,
    percentile_tiers: List[float],
    multiplier_tiers: List[float],
    min_obs: int = 50,
) -> np.ndarray:
    """Vectorized version: compute adaptive multipliers for a cross-section.

    Args:
        values: 1-D array of current indicator values (one per stock, length N).
        history: 1-D array of historical indicator values across all stocks/dates
            (the pooled expanding window for this indicator).
        percentile_tiers: Ascending percentiles.
        multiplier_tiers: Multipliers (len = len(tiers) + 1).
        min_obs: Minimum history length.

    Returns:
        1-D array of multipliers (length N).
    """
    h = np.asarray(history, dtype=np.float64)
    h = h[np.isfinite(h)]
    v = np.asarray(values, dtype=np.float64)
    default_idx = len(multiplier_tiers) // 2

    if len(h) < min_obs:
        return np.full(len(v), multiplier_tiers[default_idx])

    thresholds = np.array([np.percentile(h, p) for p in percentile_tiers])
    mults = np.full(len(v), multiplier_tiers[-1])

    # Iterate tiers in reverse so lower tiers override higher ones
    for i in range(len(thresholds) - 1, -1, -1):
        mask = v < thresholds[i]
        mults[mask] = multiplier_tiers[i]

    return mults


# ═══════════════════════════════════════════════════════════════════════════════
# C-2: DEFLATED SHARPE RATIO
# ═══════════════════════════════════════════════════════════════════════════════
# Corrects observed Sharpe ratio for:
#   (a) Number of strategies tested (multiple comparisons)
#   (b) Non-normality of returns (skewness, kurtosis)
#   (c) Sample length
#
# Reference: Bailey & López de Prado (2014), Eq. 7.
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DSRResult:
    """Result of Deflated Sharpe Ratio computation."""
    observed_sharpe: float
    deflated_sharpe: float
    p_value: float           # prob that true Sharpe <= 0 after correction
    is_significant: bool     # p_value < significance_level
    expected_max_sharpe: float  # E[max(SR)] under null for M trials
    haircut_pct: float       # percentage reduction from observed to deflated


def deflated_sharpe_ratio(
    returns: np.ndarray,
    num_strategies_tested: int,
    periods_per_year: float = 252.0,
    significance_level: float = 0.05,
) -> DSRResult:
    """Compute the Deflated Sharpe Ratio per Bailey & López de Prado (2014).

    The DSR adjusts the observed Sharpe for the expected maximum Sharpe
    under the null hypothesis that all M strategies have zero true Sharpe.

    Args:
        returns: 1-D array of strategy returns.
        num_strategies_tested: Total number of strategies/configurations tested
            (M in the paper). This includes strategies that were discarded.
        periods_per_year: Annualization factor.
        significance_level: Threshold for statistical significance.

    Returns:
        DSRResult with deflated Sharpe and significance flag.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    T = len(r)

    if T < 10 or num_strategies_tested < 1:
        return DSRResult(
            observed_sharpe=0.0, deflated_sharpe=0.0, p_value=1.0,
            is_significant=False, expected_max_sharpe=0.0, haircut_pct=100.0,
        )

    # Observed annualized Sharpe
    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=1))
    if sigma < _EPS:
        return DSRResult(
            observed_sharpe=0.0, deflated_sharpe=0.0, p_value=1.0,
            is_significant=False, expected_max_sharpe=0.0, haircut_pct=100.0,
        )

    sr_observed = (mu / sigma) * np.sqrt(periods_per_year)

    # Skewness and kurtosis of returns
    skew = float(sp_stats.skew(r, bias=False))
    kurt = float(sp_stats.kurtosis(r, bias=False))  # excess kurtosis

    # Standard error of the Sharpe ratio (Lo, 2002)
    # SE(SR) = sqrt((1 - skew*SR + (kurt-1)/4 * SR^2) / (T-1))
    sr_per_period = mu / sigma  # non-annualized
    se_sr_sq = (
        1.0
        - skew * sr_per_period
        + ((kurt - 1) / 4.0) * sr_per_period ** 2
    ) / max(T - 1, 1)
    se_sr = np.sqrt(max(se_sr_sq, _EPS))

    # Expected maximum Sharpe under null (all M strategies have SR=0)
    # E[max(Z_1,...,Z_M)] ≈ (1 - γ) * Φ^{-1}(1 - 1/M) + γ * Φ^{-1}(1 - 1/(M*e))
    # where γ ≈ 0.5772 (Euler-Mascheroni)
    M = max(num_strategies_tested, 1)
    gamma_em = 0.5772156649
    if M > 1:
        z1 = sp_stats.norm.ppf(1.0 - 1.0 / M)
        z2 = sp_stats.norm.ppf(1.0 - 1.0 / (M * np.e))
        e_max_sr = (1.0 - gamma_em) * z1 + gamma_em * z2
    else:
        e_max_sr = 0.0

    # Annualize the expected max SR
    e_max_sr_ann = e_max_sr * se_sr * np.sqrt(periods_per_year)

    # Deflated Sharpe = (observed - E[max]) / SE
    se_sr_ann = se_sr * np.sqrt(periods_per_year)
    if se_sr_ann < _EPS:
        return DSRResult(
            observed_sharpe=sr_observed, deflated_sharpe=0.0, p_value=1.0,
            is_significant=False, expected_max_sharpe=e_max_sr_ann,
            haircut_pct=100.0,
        )

    dsr = (sr_observed - e_max_sr_ann) / se_sr_ann

    # p-value: probability that true SR <= 0 after correction
    p_value = float(sp_stats.norm.cdf(-dsr))  # one-sided test

    haircut = (1.0 - dsr / sr_observed) * 100.0 if abs(sr_observed) > _EPS else 100.0

    return DSRResult(
        observed_sharpe=sr_observed,
        deflated_sharpe=float(dsr),
        p_value=p_value,
        is_significant=p_value < significance_level,
        expected_max_sharpe=e_max_sr_ann,
        haircut_pct=float(np.clip(haircut, 0, 100)),
    )


def compute_family_dsr(
    strategy_returns: Dict[str, np.ndarray],
    periods_per_year: float = 252.0,
    significance_level: float = 0.05,
) -> Dict[str, DSRResult]:
    """Compute DSR for each strategy in a family, using M = total strategies.

    Args:
        strategy_returns: {name: 1-D returns array}.
        periods_per_year: Annualization factor.
        significance_level: Significance threshold.

    Returns:
        {name: DSRResult} for each strategy.
    """
    M = len(strategy_returns)
    results = {}
    for name, rets in strategy_returns.items():
        results[name] = deflated_sharpe_ratio(
            rets, M, periods_per_year, significance_level,
        )
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# C-4: EMPIRICAL QUANTILE RANK
# ═══════════════════════════════════════════════════════════════════════════════
# Replaces Gaussian z-scores on non-Gaussian quantities with
# distribution-free empirical quantile ranks.
#
# For a value x in a distribution F, the rank is F(x) ∈ [0, 1].
# This is invariant to the shape of F and provides uniform marginals.
# ═══════════════════════════════════════════════════════════════════════════════

def empirical_quantile_rank(
    current_values: np.ndarray,
    history: np.ndarray,
    min_observations: int = 20,
) -> np.ndarray:
    """Compute empirical quantile rank of current values within history.

    Returns values in [0, 1] where 0 = minimum ever observed,
    1 = maximum ever observed. This is distribution-free and does not
    assume Gaussian, bounded, or any other parametric form.

    Args:
        current_values: 1-D array of current indicator values (N stocks).
        history: 1-D array of pooled historical values for this indicator.
        min_observations: Below this, returns 0.5 (uninformative prior).

    Returns:
        1-D array of quantile ranks in [0, 1].
    """
    h = np.asarray(history, dtype=np.float64)
    h = h[np.isfinite(h)]
    v = np.asarray(current_values, dtype=np.float64)

    if len(h) < min_observations:
        return np.full(len(v), 0.5)

    sorted_h = np.sort(h)
    ranks = np.searchsorted(sorted_h, v, side='right') / len(sorted_h)
    return np.clip(ranks, 0.0, 1.0)


def empirical_quantile_rank_timeseries(
    series: np.ndarray,
    expanding_min: int = 50,
) -> np.ndarray:
    """Compute expanding-window empirical quantile rank for a time series.

    At each time t, rank(t) = fraction of values in series[0:t] that are
    less than or equal to series[t].

    This replaces the rolling z-score in backdata.py with a distribution-free
    alternative.

    Args:
        series: 1-D time series.
        expanding_min: Minimum observations before computing (NaN before).

    Returns:
        1-D array of quantile ranks (NaN for the first expanding_min points).
    """
    s = np.asarray(series, dtype=np.float64)
    n = len(s)
    ranks = np.full(n, np.nan)

    for t in range(expanding_min, n):
        window = s[:t + 1]
        finite = window[np.isfinite(window)]
        if len(finite) < expanding_min:
            continue
        ranks[t] = float(np.sum(finite <= s[t])) / len(finite)

    return ranks


# ═══════════════════════════════════════════════════════════════════════════════
# C-5: VALIDATED LIQUIDITY INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════

def amihud_illiquidity(
    returns: np.ndarray,
    volume: np.ndarray,
    window: int = 20,
) -> np.ndarray:
    """Amihud (2002) illiquidity ratio: |r_t| / Volume_t.

    Higher values indicate lower liquidity (more price impact per unit traded).

    The rolling mean smooths the daily ratio into a stable indicator.

    Args:
        returns: 1-D array of daily returns.
        volume: 1-D array of daily volume (in shares or currency).
        window: Rolling window for averaging.

    Returns:
        1-D array of rolling Amihud illiquidity (same length as input, NaN-padded).
    """
    r = np.asarray(returns, dtype=np.float64)
    v = np.asarray(volume, dtype=np.float64)
    n = len(r)

    safe_v = np.where(v > 0, v, np.nan)
    daily_illiq = np.abs(r) / safe_v

    # Rolling mean
    result = np.full(n, np.nan)
    for t in range(window - 1, n):
        w = daily_illiq[t - window + 1: t + 1]
        finite = w[np.isfinite(w)]
        if len(finite) > 0:
            result[t] = float(np.mean(finite))

    return result


def corwin_schultz_spread(
    high: np.ndarray,
    low: np.ndarray,
    window: int = 20,
) -> np.ndarray:
    """Corwin & Schultz (2012) high-low spread estimator.

    Estimates the bid-ask spread from daily high-low prices using the
    insight that the high is typically a buy and the low is typically a sell.

    S = (2 * (e^α - 1)) / (1 + e^α)

    where α is derived from the ratio of two-day to one-day high-low ranges.

    Args:
        high: 1-D array of daily high prices.
        low: 1-D array of daily low prices.
        window: Rolling window for averaging the spread estimate.

    Returns:
        1-D array of estimated spreads (same length, NaN-padded).
    """
    h = np.asarray(high, dtype=np.float64)
    lo = np.asarray(low, dtype=np.float64)
    n = len(h)

    # Prevent log(0) or log(negative)
    safe_h = np.maximum(h, _EPS)
    safe_lo = np.maximum(lo, _EPS)
    ratio = np.log(safe_h / safe_lo)

    # β = E[ln(H_t/L_t)]^2
    beta = ratio ** 2

    # γ = [ln(max(H_t, H_{t+1}) / min(L_t, L_{t+1}))]^2
    gamma = np.full(n, np.nan)
    for t in range(n - 1):
        h2 = max(h[t], h[t + 1])
        l2 = min(lo[t], lo[t + 1])
        if l2 > 0 and h2 > 0:
            gamma[t] = np.log(h2 / l2) ** 2

    # α = (sqrt(2*β) - sqrt(β)) / (3 - 2*sqrt(2)) - sqrt(γ / (3 - 2*sqrt(2)))
    sqrt2 = np.sqrt(2.0)
    denom = 3.0 - 2.0 * sqrt2

    spread = np.full(n, np.nan)
    for t in range(window, n):
        beta_mean = np.nanmean(beta[t - window: t])
        gamma_mean = np.nanmean(gamma[t - window: t])
        if np.isnan(beta_mean) or np.isnan(gamma_mean):
            continue

        alpha = (np.sqrt(2 * beta_mean) - np.sqrt(beta_mean)) / denom
        alpha -= np.sqrt(max(gamma_mean, 0) / denom)

        if alpha > 0:
            spread[t] = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
        else:
            spread[t] = 0.0

    return np.clip(np.nan_to_num(spread, nan=np.nan), 0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# H-2: RANK-BASED CROSS-SECTIONAL NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════════════
# Replaces StandardScaler with rank-based normalization that is:
#   - Invariant to outliers
#   - Independent of distributional assumptions
#   - Stable across universe changes
# ═══════════════════════════════════════════════════════════════════════════════

def rank_normalize(
    df: pd.DataFrame,
    columns: List[str],
    method: str = 'average',
) -> pd.DataFrame:
    """Replace values with their cross-sectional percentile ranks.

    Each column is independently ranked and scaled to [0, 1].
    Unlike StandardScaler, this is:
      - Outlier-robust (ranks are bounded)
      - Distribution-free (no Gaussian assumption)
      - Stable across universe composition changes

    Args:
        df: DataFrame with the columns to normalize.
        columns: Column names to rank-normalize.
        method: Ranking method ('average', 'min', 'max', 'dense', 'ordinal').

    Returns:
        DataFrame with specified columns replaced by their rank percentiles.
    """
    result = df.copy()
    for col in columns:
        if col in result.columns:
            ranks = result[col].rank(method=method, pct=True)
            result[col] = ranks.fillna(0.5)  # missing → median rank
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# H-4: MARKET IMPACT MODEL
# ═══════════════════════════════════════════════════════════════════════════════
# Almgren-Chriss inspired market impact for realistic transaction cost
# estimation. Impact = η * σ * √(V/ADV) where:
#   η = market impact coefficient (calibrated to NSE)
#   σ = daily volatility
#   V = trade volume
#   ADV = average daily volume
#
# Reference: Almgren & Chriss (2001)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MarketImpactEstimate:
    """Estimated transaction costs including market impact."""
    spread_cost_bps: float    # bid-ask spread component
    impact_cost_bps: float    # market impact component
    total_cost_bps: float     # total one-way cost
    turnover: float           # fraction of portfolio traded


def estimate_market_impact(
    trade_value: float,
    stock_adv: float,
    stock_volatility: float,
    spread_bps: float = 5.0,
    impact_coefficient: float = 0.1,
) -> MarketImpactEstimate:
    """Estimate the market impact of a single trade.

    Uses the square-root law: impact ∝ σ * √(V/ADV).

    Args:
        trade_value: Value of the trade (currency units).
        stock_adv: Average daily trading value of the stock.
        stock_volatility: Daily return volatility of the stock.
        spread_bps: Half-spread in basis points.
        impact_coefficient: η calibration parameter.
            0.1 is conservative for liquid NSE large-caps.
            0.3-0.5 for mid-caps.

    Returns:
        MarketImpactEstimate with cost breakdown.
    """
    if stock_adv <= 0 or stock_volatility <= 0 or trade_value <= 0:
        return MarketImpactEstimate(
            spread_cost_bps=spread_bps,
            impact_cost_bps=0.0,
            total_cost_bps=spread_bps,
            turnover=0.0,
        )

    participation_rate = trade_value / stock_adv
    impact_bps = (
        impact_coefficient
        * stock_volatility
        * 10000  # convert to bps
        * np.sqrt(participation_rate)
    )

    return MarketImpactEstimate(
        spread_cost_bps=spread_bps,
        impact_cost_bps=float(impact_bps),
        total_cost_bps=float(spread_bps + impact_bps),
        turnover=float(participation_rate),
    )


def estimate_portfolio_impact(
    portfolio: pd.DataFrame,
    adv_data: Optional[pd.DataFrame] = None,
    default_adv: float = 1e8,
    default_volatility: float = 0.02,
    impact_coefficient: float = 0.1,
) -> float:
    """Estimate total portfolio transaction cost in basis points.

    Args:
        portfolio: Portfolio DataFrame with 'symbol', 'value' columns.
        adv_data: Optional DataFrame with 'symbol', 'adv', 'volatility'.
        default_adv: Default ADV if stock-level data unavailable.
        default_volatility: Default daily vol if unavailable.
        impact_coefficient: Impact coefficient.

    Returns:
        Estimated total cost in basis points (value-weighted average).
    """
    if portfolio.empty or 'value' not in portfolio.columns:
        return 10.0  # conservative default

    total_value = portfolio['value'].sum()
    if total_value <= 0:
        return 10.0

    weighted_cost = 0.0
    for _, row in portfolio.iterrows():
        tv = row.get('value', 0)
        if tv <= 0:
            continue

        sym = row.get('symbol', '')
        adv = default_adv
        vol = default_volatility

        if adv_data is not None and not adv_data.empty:
            sym_data = adv_data[adv_data['symbol'] == sym]
            if not sym_data.empty:
                adv = sym_data['adv'].iloc[0] if 'adv' in sym_data.columns else default_adv
                vol = sym_data['volatility'].iloc[0] if 'volatility' in sym_data.columns else default_volatility

        impact = estimate_market_impact(tv, adv, vol, impact_coefficient=impact_coefficient)
        weighted_cost += impact.total_cost_bps * (tv / total_value)

    return float(weighted_cost)


# ═══════════════════════════════════════════════════════════════════════════════
# H-5: CONFORMAL INTERVALS WITH SERIAL DEPENDENCE CORRECTION
# ═══════════════════════════════════════════════════════════════════════════════
# Standard conformal prediction assumes exchangeability. For serially
# correlated strategy returns, we apply the block bootstrap correction
# from Barber et al. (2023).
#
# The key idea: use non-overlapping blocks of size b to construct
# calibration residuals, reducing effective sample size by factor b.
# ═══════════════════════════════════════════════════════════════════════════════

def conformal_intervals_with_dependence(
    returns: np.ndarray,
    alpha: float = 0.10,
    block_size: Optional[int] = None,
) -> Tuple[float, float, float]:
    """Compute conformal prediction interval corrected for serial dependence.

    Args:
        returns: 1-D array of strategy returns (time-ordered).
        alpha: Miscoverage level (0.10 = 90% coverage).
        block_size: Size of non-overlapping blocks for dependence correction.
            If None, estimated from the lag-1 autocorrelation.

    Returns:
        (lower, point_estimate, upper) tuple.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    n = len(r)

    if n < 10:
        return (0.0, 0.0, 0.0)

    # Auto-estimate block size from lag-1 autocorrelation if not provided
    if block_size is None:
        rho = np.corrcoef(r[:-1], r[1:])[0, 1] if n > 2 else 0.0
        rho = max(abs(rho), 0.01)
        # Optimal block size for dependent data: b ≈ (2*ρ/(1-ρ²))^(1/3) * n^(1/3)
        # Politis & Romano (1994)
        b_raw = ((2.0 * rho / max(1.0 - rho ** 2, 0.01)) ** (1 / 3)) * (n ** (1 / 3))
        block_size = max(2, min(int(np.ceil(b_raw)), n // 3))

    # Block-based calibration: use block means as pseudo-residuals
    n_blocks = n // block_size
    if n_blocks < 5:
        # Insufficient blocks: fall back to standard conformal with Bonferroni
        # adjustment (conservative but valid)
        adjusted_alpha = alpha / max(1, block_size)
        lower = float(np.percentile(r, 100 * adjusted_alpha / 2))
        upper = float(np.percentile(r, 100 * (1 - adjusted_alpha / 2)))
        point = float(np.median(r))
        return (lower, point, upper)

    # Non-overlapping block residuals
    block_means = np.array([
        np.mean(r[i * block_size: (i + 1) * block_size])
        for i in range(n_blocks)
    ])

    # Conformal quantiles on block means
    lower_q = float(np.percentile(block_means, 100 * alpha / 2))
    upper_q = float(np.percentile(block_means, 100 * (1 - alpha / 2)))
    point = float(np.median(r))

    # Scale block-mean intervals back to individual return scale
    # Block means have reduced variance by factor sqrt(block_size)
    scale = np.sqrt(block_size)
    width_expansion = scale * 0.5 + 0.5  # partial expansion (conservative blend)

    center = (lower_q + upper_q) / 2
    half_width = (upper_q - lower_q) / 2 * width_expansion

    return (
        float(center - half_width),
        float(point),
        float(center + half_width),
    )


def conformal_strategy_intervals_corrected(
    strategy_returns: Dict[str, np.ndarray],
    alpha: float = 0.10,
) -> Dict[str, Tuple[float, float, float]]:
    """Compute dependence-corrected conformal intervals for all strategies.

    Drop-in replacement for rmt_core.conformal_strategy_intervals.

    Args:
        strategy_returns: {name: 1-D returns array}.
        alpha: Miscoverage level.

    Returns:
        {name: (lower, point, upper)} for each strategy.
    """
    results = {}
    for name, rets in strategy_returns.items():
        results[name] = conformal_intervals_with_dependence(rets, alpha)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# M-2: DISTRIBUTION-AWARE NAN IMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════
# Replaces the current imputation (RSI→50, OSC→median, z→0, MA→price)
# with a scheme that produces genuinely uninformative values that do NOT
# generate active trading signals.
# ═══════════════════════════════════════════════════════════════════════════════

def impute_uninformative(
    df: pd.DataFrame,
    indicator_configs: Optional[Dict[str, Dict]] = None,
) -> pd.DataFrame:
    """Impute missing values with genuinely uninformative defaults.

    Unlike the current scheme (RSI→50, z→0, MA→price), this imputation
    produces values that the adaptive quantile gates will classify as
    "middle of the distribution" — generating a multiplier of 1.0 (neutral).

    Strategy:
      - RSI: Cross-sectional median (not 50, which is an active signal)
      - Oscillators: Cross-sectional median
      - Z-scores / Quantile ranks: 0.5 (median rank — truly uninformative)
      - Moving averages: Cross-sectional median of MA values (not price)
      - Deviations: Cross-sectional median of deviation values

    Args:
        df: DataFrame with indicator columns.
        indicator_configs: Optional mapping of column patterns to imputation
            strategies. If None, uses defaults.

    Returns:
        DataFrame with NaN values imputed.
    """
    result = df.copy()

    if indicator_configs is None:
        indicator_configs = {
            'rsi': {'method': 'cross_sectional_median'},
            'osc': {'method': 'cross_sectional_median'},
            'ema osc': {'method': 'cross_sectional_median'},
            'zscore': {'method': 'fixed', 'value': 0.0},
            'qrank': {'method': 'fixed', 'value': 0.5},
            'ma': {'method': 'cross_sectional_median'},
            'dev': {'method': 'cross_sectional_median'},
        }

    for col in result.columns:
        if result[col].isna().sum() == 0:
            continue

        col_lower = col.lower()
        imputed = False

        for pattern, config in indicator_configs.items():
            if pattern in col_lower:
                method = config.get('method', 'cross_sectional_median')
                if method == 'cross_sectional_median':
                    median_val = result[col].median()
                    if np.isnan(median_val):
                        median_val = 0.0
                    result[col] = result[col].fillna(median_val)
                elif method == 'fixed':
                    result[col] = result[col].fillna(config.get('value', 0.0))
                imputed = True
                break

        if not imputed:
            # Fallback: cross-sectional median
            median_val = result[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            result[col] = result[col].fillna(median_val)

    # Final safety: replace any remaining NaN/Inf
    result = result.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# INDICATOR HISTORY ACCUMULATOR
# ═══════════════════════════════════════════════════════════════════════════════
# Maintains expanding-window histories for all indicators across dates.
# This is the data structure that enables adaptive quantile gates.
# ═══════════════════════════════════════════════════════════════════════════════

class IndicatorHistoryAccumulator:
    """Accumulates indicator values across dates for adaptive thresholds.

    Usage in walk-forward::

        acc = IndicatorHistoryAccumulator()
        for date, df in historical_data:
            acc.update(df)
            thresholds = acc.get_thresholds(percentile=10)
            # Use thresholds in strategy
    """

    def __init__(self):
        self._histories: Dict[str, List[float]] = {}

    def update(self, df: pd.DataFrame, indicator_columns: Optional[List[str]] = None):
        """Add current cross-section values to history.

        Args:
            df: Current day's indicator DataFrame.
            indicator_columns: Columns to track. If None, tracks all numeric columns.
        """
        if indicator_columns is None:
            indicator_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in indicator_columns:
            if col not in df.columns:
                continue
            values = df[col].dropna().values
            if col not in self._histories:
                self._histories[col] = []
            self._histories[col].extend(values.tolist())

    def get_history(self, indicator: str) -> np.ndarray:
        """Get the full expanding-window history for an indicator."""
        return np.array(self._histories.get(indicator, []))

    def get_threshold(
        self,
        indicator: str,
        percentile: float,
        min_obs: int = 50,
        halflife: Optional[int] = None,
    ) -> float:
        """Get the adaptive threshold for an indicator at a given percentile."""
        gate = AdaptiveQuantileGate(percentile, min_obs, halflife)
        return gate.threshold(self.get_history(indicator))

    def get_quantile_ranks(
        self,
        df: pd.DataFrame,
        indicator: str,
        min_obs: int = 20,
    ) -> np.ndarray:
        """Get empirical quantile ranks for current cross-section values."""
        if indicator not in df.columns:
            return np.full(len(df), 0.5)
        return empirical_quantile_rank(
            df[indicator].values,
            self.get_history(indicator),
            min_obs,
        )

    @property
    def indicators(self) -> List[str]:
        return list(self._histories.keys())

    def __len__(self) -> int:
        if not self._histories:
            return 0
        return max(len(v) for v in self._histories.values())


# ═══════════════════════════════════════════════════════════════════════════════
# POINT-IN-TIME UNIVERSE INFRASTRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════
# Provides the framework for point-in-time constituent tracking.
# When actual historical data is unavailable, flags the survivorship bias
# and provides a degraded-but-honest mode.
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class UniverseSnapshot:
    """Point-in-time universe snapshot."""
    date: pd.Timestamp
    constituents: List[str]
    source: str  # 'historical' or 'static_with_survivorship_warning'


def load_point_in_time_universe(
    date: pd.Timestamp,
    historical_constituents_path: Optional[str] = None,
    static_symbols: Optional[List[str]] = None,
) -> UniverseSnapshot:
    """Load the universe of stocks valid at a specific date.

    If historical constituent data is available (CSV with columns:
    date, symbol, action [ADD/REMOVE]), uses it to reconstruct
    the point-in-time universe.

    Otherwise, falls back to the static symbol list with a survivorship
    bias warning.

    Args:
        date: The target date.
        historical_constituents_path: Path to historical constituent file.
        static_symbols: Fallback static symbol list.

    Returns:
        UniverseSnapshot with constituents and source metadata.
    """
    if historical_constituents_path is not None:
        try:
            df = pd.read_csv(historical_constituents_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df[df['date'] <= date].sort_values('date')

            # Reconstruct universe by replaying ADD/REMOVE actions
            universe = set()
            for _, row in df.iterrows():
                sym = row['symbol'].strip().upper()
                action = row['action'].strip().upper()
                if action == 'ADD':
                    universe.add(sym)
                elif action == 'REMOVE':
                    universe.discard(sym)

            if universe:
                return UniverseSnapshot(
                    date=date,
                    constituents=sorted(universe),
                    source='historical',
                )
        except Exception as e:
            logger.warning(f"Could not load historical constituents: {e}")

    # Fallback to static universe
    symbols = static_symbols or []
    logger.warning(
        "SURVIVORSHIP BIAS: Using static symbol list. Historical constituent "
        "data unavailable. Backtest results may be upward-biased."
    )
    return UniverseSnapshot(
        date=date,
        constituents=symbols,
        source='static_with_survivorship_warning',
    )


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # C-1: Adaptive Quantile Gates
    'AdaptiveQuantileGate',
    'compute_adaptive_multiplier',
    'compute_adaptive_multiplier_vectorized',
    'IndicatorHistoryAccumulator',
    # C-2: Deflated Sharpe Ratio
    'DSRResult',
    'deflated_sharpe_ratio',
    'compute_family_dsr',
    # C-4: Empirical Quantile Rank
    'empirical_quantile_rank',
    'empirical_quantile_rank_timeseries',
    # C-5: Validated Indicators
    'amihud_illiquidity',
    'corwin_schultz_spread',
    # H-2: Rank Normalization
    'rank_normalize',
    # H-4: Market Impact
    'MarketImpactEstimate',
    'estimate_market_impact',
    'estimate_portfolio_impact',
    # H-5: Conformal with Dependence
    'conformal_intervals_with_dependence',
    'conformal_strategy_intervals_corrected',
    # M-2: Imputation
    'impute_uninformative',
    # Point-in-Time Universe
    'UniverseSnapshot',
    'load_point_in_time_universe',
]
