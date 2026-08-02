"""
Market regime classification.

Regimes are *discovered*, not hand-labelled: a k-means model clusters the
market's state vector, and clusters are then named by their statistical
signature. The classifier is refit walk-forward on trailing data only, so a
label at time t never depends on what happened after t. (Full-sample
clustering is the usual shortcut here, and it silently leaks the future into
every regime-conditioned weight and backtest downstream.)

State vector, all causal:
    trend        21d mean market return, scaled by trailing volatility
    volatility   21d market volatility relative to its own 252d level
    dispersion   cross-sectional return dispersion
    correlation  average pairwise correlation proxy
    persistence  63d autocorrelation of market returns
    drawdown     distance below the trailing 252d high
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .config import REGIME_NAMES

FEATURES = ["trend", "volatility", "dispersion", "correlation", "persistence", "drawdown"]


def market_state(rets: pd.DataFrame) -> pd.DataFrame:
    """Build the causal market-state feature matrix from the return panel."""
    mkt = rets.mean(axis=1)                       # equal-weighted market proxy
    vol21 = mkt.rolling(21, min_periods=10).std()
    vol252 = mkt.rolling(252, min_periods=60).std()

    trend = mkt.rolling(21, min_periods=10).mean() / vol21.replace(0, np.nan)
    volatility = vol21 / vol252.replace(0, np.nan)
    dispersion = rets.std(axis=1).rolling(21, min_periods=10).mean()

    # var(mean) / mean(var) is the standard average-pairwise-correlation proxy
    # for an equal-weighted basket.
    var_mkt = mkt.rolling(21, min_periods=10).var()
    mean_var = rets.rolling(21, min_periods=10).var().mean(axis=1)
    correlation = (var_mkt / mean_var.replace(0, np.nan)).clip(0, 1)

    persistence = mkt.rolling(63, min_periods=30).apply(
        lambda x: pd.Series(x).autocorr(lag=1) if np.std(x) > 0 else 0.0, raw=False)

    level = mkt.cumsum()
    drawdown = level - level.rolling(252, min_periods=60).max()

    X = pd.concat([trend, volatility, dispersion, correlation, persistence, drawdown],
                  axis=1)
    X.columns = FEATURES
    return X.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)


def _name_clusters(centroids: np.ndarray, columns: list[str]) -> dict[int, str]:
    """Map cluster ids to regime names by their centroid signature.

    The rule is deterministic so names stay comparable across refits:
      * the most volatile cluster is HIGH-VOL;
      * of the rest, the weakest-trend cluster is RISK-OFF and the
        strongest-trend cluster is RISK-ON;
      * of what remains, the more autocorrelated cluster is TREND, the other
        MEAN-REV.
    """
    C = pd.DataFrame(centroids, columns=columns)
    names: dict[int, str] = {}
    remaining = list(C.index)

    hv = C.loc[remaining, "volatility"].idxmax()
    names[hv] = "HIGH-VOL"
    remaining.remove(hv)

    if remaining:
        ro = C.loc[remaining, "trend"].idxmin()
        names[ro] = "RISK-OFF"
        remaining.remove(ro)
    if remaining:
        ron = C.loc[remaining, "trend"].idxmax()
        names[ron] = "RISK-ON"
        remaining.remove(ron)
    if len(remaining) >= 2:
        order = C.loc[remaining, "persistence"].sort_values(ascending=False).index.tolist()
        names[order[0]] = "TREND"
        names[order[1]] = "MEAN-REV"
        for extra in order[2:]:
            names[extra] = "TREND"
    elif remaining:
        names[remaining[0]] = "TREND"
    return names


def classify_regimes(rets: pd.DataFrame, n_regimes: int = 5, train: int = 252,
                     refit_every: int = 63, seed: int = 7
                     ) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Walk-forward regime labels.

    Returns ``(labels, stability, state)`` where ``stability`` is the share of
    the trailing 21 sessions that carried the current label — a cheap measure
    of how settled the regime is.
    """
    state = market_state(rets)
    n = len(state)
    train = int(min(max(train, 120), max(n - 1, 120)))

    labels = pd.Series(index=state.index, dtype="object")
    model: KMeans | None = None
    scaler: StandardScaler | None = None
    names: dict[int, str] = {}

    t = min(train, n)
    while True:
        hist = state.iloc[:t]
        k = int(min(n_regimes, max(2, len(hist) // 40)))
        scaler = StandardScaler().fit(hist.values)
        model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(
            scaler.transform(hist.values))
        names = _name_clusters(model.cluster_centers_, list(state.columns))

        if t == min(train, n):  # warm-up block, labelled in-sample by design
            lab = model.predict(scaler.transform(hist.values))
            labels.iloc[:t] = [names[c] for c in lab]

        end = min(t + refit_every, n)
        if end > t:
            fwd = state.iloc[t:end]
            lab = model.predict(scaler.transform(fwd.values))
            labels.iloc[t:end] = [names[c] for c in lab]
        if end >= n:
            break
        t = end

    labels = labels.ffill().fillna("TREND").astype("object")
    for name in labels.unique():
        if name not in REGIME_NAMES:
            labels = labels.replace(name, "TREND")

    codes = pd.Series(pd.Categorical(labels, categories=REGIME_NAMES).codes,
                      index=labels.index)
    stability = codes.rolling(21, min_periods=5).apply(
        lambda w: float(np.mean(w == w[-1])), raw=True).fillna(0.5)

    return labels, stability, state


def regime_spans(labels: pd.Series) -> list[tuple]:
    """Compress a label series into ``(start, end, label)`` spans for shading."""
    spans: list[tuple] = []
    if labels.empty:
        return spans
    prev, start = labels.iloc[0], labels.index[0]
    for ts, val in labels.items():
        if val != prev:
            spans.append((start, ts, prev))
            prev, start = val, ts
    spans.append((start, labels.index[-1], prev))
    return spans


def regime_summary(labels: pd.Series, target_rets: pd.Series,
                   oscillator: pd.Series) -> pd.DataFrame:
    """Per-regime behaviour of the target and of the oscillator."""
    rows = []
    for name in REGIME_NAMES:
        mask = labels == name
        if mask.sum() == 0:
            rows.append({"regime": name, "sessions": 0, "share %": 0.0,
                         "target μ (bp/d)": np.nan, "target σ (bp/d)": np.nan,
                         "FVO mean": np.nan, "FVO σ": np.nan})
            continue
        rows.append({
            "regime": name,
            "sessions": int(mask.sum()),
            "share %": 100.0 * float(mask.mean()),
            "target μ (bp/d)": 1e4 * float(target_rets[mask].mean()),
            "target σ (bp/d)": 1e4 * float(target_rets[mask].std()),
            "FVO mean": float(oscillator[mask].mean()),
            "FVO σ": float(oscillator[mask].std()),
        })
    return pd.DataFrame(rows)
