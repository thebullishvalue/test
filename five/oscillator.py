"""
The Fair Value Oscillator (FVO).

Pipeline:  log(P / FV)  ->  adaptive normalisation  ->  bounded saturation
           ->  optional state-space smoothing  ->  dynamic thresholds.

Design notes
------------
*Adaptive normalisation* is what makes a 2019 reading comparable to a 2020
one: the raw log gap is scaled by its own trailing dispersion, so a ±2 reading
means "two standard deviations wide *for this volatility regime*", not a fixed
percentage.

*Bounded saturation* uses tanh rather than a hard clip. A clip destroys all
information beyond the boundary — every extreme prints the same -100 — while
tanh compresses monotonically, so a -97 and a -99 remain distinguishable.

*Thresholds are lagged by one session.* The overbought/oversold quantiles at
time t are computed from data up to t-1. Including the current observation in
its own threshold makes signals look sharper than they could ever be live.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Normalisation & saturation
# ---------------------------------------------------------------------------
def normalise(signal: pd.Series, window: int = 126, robust: bool = True,
              min_periods: int | None = None) -> pd.Series:
    """Adaptive standardisation of the raw mispricing signal."""
    mp = min_periods or max(20, window // 4)
    if robust:
        loc = signal.rolling(window, min_periods=mp).median()
        mad = (signal - loc).abs().rolling(window, min_periods=mp).median()
        scale = 1.4826 * mad
    else:
        loc = signal.rolling(window, min_periods=mp).mean()
        scale = signal.rolling(window, min_periods=mp).std()

    # A robust scale can collapse to zero in a dead-flat stretch; fall back to
    # the classical estimate before giving up on the observation.
    scale = scale.replace(0, np.nan)
    fallback = signal.rolling(window, min_periods=mp).std().replace(0, np.nan)
    scale = scale.fillna(fallback)

    z = (signal - loc) / scale
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def saturate(z: pd.Series, kappa: float = 2.5, bound: float = 100.0) -> pd.Series:
    """Map an unbounded z-score onto (-bound, +bound) monotonically."""
    return bound * np.tanh(z / max(kappa, 1e-6))


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------
def kalman_local_level(x: pd.Series, span: int = 6
                       ) -> tuple[pd.Series, pd.Series]:
    """Local-level (random walk + noise) filter.

    The signal-to-noise ratio is derived from the data: observation variance
    from the one-step differences, process variance scaled by ``span`` so the
    control behaves like an EMA span (larger => smoother).
    """
    v = x.values.astype("float64")
    if len(v) == 0:
        return x.copy(), x.copy()

    dv = np.diff(v)
    r = float(np.nanvar(dv)) if np.isfinite(np.nanvar(dv)) else 1.0
    r = max(r, 1e-8) * max(span, 1) * 0.5     # observation noise
    q = max(float(np.nanvar(v)) * 1e-3, 1e-8)  # process noise

    out = np.empty_like(v)
    var = np.empty_like(v)
    xh = v[0] if np.isfinite(v[0]) else 0.0
    P = 1.0
    for i, obs in enumerate(v):
        P += q                                   # predict
        if np.isfinite(obs):                     # update
            K = P / (P + r)
            xh += K * (obs - xh)
            P *= (1 - K)
        out[i], var[i] = xh, P
    return (pd.Series(out, index=x.index),
            pd.Series(np.sqrt(np.maximum(var, 0.0)), index=x.index))


def smooth_oscillator(raw: pd.Series, method: str = "Kalman", span: int = 6
                      ) -> tuple[pd.Series, pd.Series]:
    """Return ``(smoothed, one-sigma band)`` for the requested filter."""
    if method == "EMA":
        s = raw.ewm(span=max(span, 2), min_periods=1).mean()
        band = (raw - s).rolling(63, min_periods=10).std()
    elif method == "Kalman":
        s, sd = kalman_local_level(raw, span=span)
        resid = (raw - s).rolling(63, min_periods=10).std()
        band = pd.concat([sd, resid], axis=1).max(axis=1)
    else:
        s = raw.copy()
        band = raw.rolling(63, min_periods=10).std()
    return s, band.bfill().fillna(5.0)


# ---------------------------------------------------------------------------
# Dynamic thresholds
# ---------------------------------------------------------------------------
def dynamic_thresholds(fvo: pd.Series, q: float = 0.12, window: int = 504,
                       min_periods: int = 120) -> tuple[pd.Series, pd.Series]:
    """Rolling empirical overbought/oversold levels, lagged one session.

    Falls back to an expanding quantile early in the sample so thresholds
    exist before ``window`` observations have accumulated.
    """
    roll_hi = fvo.rolling(window, min_periods=min_periods).quantile(1 - q)
    roll_lo = fvo.rolling(window, min_periods=min_periods).quantile(q)
    exp_hi = fvo.expanding(min_periods=60).quantile(1 - q)
    exp_lo = fvo.expanding(min_periods=60).quantile(q)

    ob = roll_hi.fillna(exp_hi).shift(1)
    os_ = roll_lo.fillna(exp_lo).shift(1)
    return ob, os_


# ---------------------------------------------------------------------------
# Divergence detection
# ---------------------------------------------------------------------------
def find_pivots(s: pd.Series, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Confirmed swing highs/lows: an extremum of a 2k+1 window centred on t.

    A pivot at t is only *known* at t+k. Callers must therefore timestamp any
    resulting signal at the confirmation bar, not at the pivot itself.
    """
    v = s.values.astype("float64")
    n = len(v)
    highs, lows = [], []
    for i in range(k, n - k):
        w = v[i - k:i + k + 1]
        if not np.isfinite(w).all():
            continue
        if v[i] == w.max() and (w.argmax() == k):
            highs.append(i)
        if v[i] == w.min() and (w.argmin() == k):
            lows.append(i)
    return np.array(highs, dtype=int), np.array(lows, dtype=int)


def detect_divergences(price: pd.Series, fvo: pd.Series, k: int = 5,
                       min_gap: int = 8, min_move: float = 4.0
                       ) -> tuple[pd.Series, pd.Series]:
    """Classic pivot divergences between price and the oscillator.

    Bearish: price makes a higher high while the oscillator makes a lower
    high — the market is extending without any improvement in market-relative
    valuation. Bullish is the mirror image.

    Both series are timestamped at the *confirmation* bar (pivot + k), so they
    are usable as live signals.
    """
    bear = pd.Series(False, index=price.index)
    bull = pd.Series(False, index=price.index)
    n = len(price)
    highs, lows = find_pivots(price, k=k)

    pv, fv = price.values, fvo.values

    for arr, is_high in ((highs, True), (lows, False)):
        for a, b in zip(arr[:-1], arr[1:]):
            if b - a < min_gap:
                continue
            confirm = min(b + k, n - 1)
            if is_high:
                if pv[b] > pv[a] and fv[b] < fv[a] - min_move:
                    bear.iloc[confirm] = True
            else:
                if pv[b] < pv[a] and fv[b] > fv[a] + min_move:
                    bull.iloc[confirm] = True
    return bear, bull


# ---------------------------------------------------------------------------
# Mean reversion (Ornstein-Uhlenbeck)
# ---------------------------------------------------------------------------
def ou_fit(signal: pd.Series, window: int = 126, step: int = 5
           ) -> tuple[pd.Series, pd.Series]:
    """Rolling OU parameters from the discrete analogue ds_t = a + b·s_{t-1} + e.

    Returns ``(theta, sigma)`` where theta = -b is the pull-to-mean speed per
    session and sigma is the residual diffusion. Estimated every ``step``
    sessions and forward-filled — the parameters move slowly and a daily refit
    buys nothing but noise.
    """
    s = signal.astype("float64")
    ds = s.diff()
    lag = s.shift(1)

    theta = pd.Series(np.nan, index=s.index)
    sigma = pd.Series(np.nan, index=s.index)

    y_all, x_all = ds.values, lag.values
    for i in range(window, len(s), max(step, 1)):
        y = y_all[i - window + 1:i + 1]
        x = x_all[i - window + 1:i + 1]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < window // 2 or np.std(x[m]) < 1e-12:
            continue
        b, a = np.polyfit(x[m], y[m], 1)
        resid = y[m] - (a + b * x[m])
        theta.iloc[i] = -b
        sigma.iloc[i] = float(np.std(resid))

    theta = theta.ffill().clip(1e-3, 0.9)
    sigma = sigma.ffill()
    return theta.fillna(0.05), sigma.fillna(sigma.median() if sigma.notna().any() else 0.3)


def mean_reversion_probability(z: pd.Series, theta: pd.Series, sigma: pd.Series,
                               horizon: int = 10, target: float = 0.5) -> pd.Series:
    """P(|z| falls inside ±target within ``horizon`` sessions), under OU.

    For dz = -θ·z·dt + σ·dW the conditional law is Gaussian:
        z_{t+h} | z_t ~ N( z_t·e^{-θh},  σ²(1 - e^{-2θh}) / (2θ) )
    so the probability follows analytically from the normal CDF. This replaces
    the usual hand-tuned logistic, which has no distributional meaning.
    """
    th = theta.clip(1e-3, 0.9).values
    sg = np.maximum(sigma.values, 1e-6)
    z0 = z.values.astype("float64")
    h = max(int(horizon), 1)

    decay = np.exp(-th * h)
    mu = z0 * decay
    var = (sg ** 2) * (1 - np.exp(-2 * th * h)) / (2 * th)
    sd = np.sqrt(np.maximum(var, 1e-12))

    p = norm.cdf((target - mu) / sd) - norm.cdf((-target - mu) / sd)
    return pd.Series(100.0 * np.clip(p, 0, 1), index=z.index)


def half_life(theta: pd.Series) -> pd.Series:
    """Sessions for a dislocation to decay by half, from the OU speed."""
    return (np.log(2) / theta.clip(1e-3, 0.9)).clip(1, 250)


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------
def threshold_signals(fvo: pd.Series, ob: pd.Series, os_: pd.Series
                      ) -> tuple[pd.Series, pd.Series]:
    """Entry markers: oscillator crossing back inside its dynamic bands.

    Crossing *back* matters. Entering the moment an extreme is first touched
    fights the dislocation while it is still widening; waiting for the cross
    back requires evidence the move has turned.
    """
    long_entry = (fvo.shift(1) < os_.shift(1)) & (fvo >= os_)
    short_entry = (fvo.shift(1) > ob.shift(1)) & (fvo <= ob)
    return long_entry.fillna(False), short_entry.fillna(False)
