"""
The Fair Value Engine — walk-forward, regime-aware, cross-sectional valuation.

What the model actually says
----------------------------
At each session the market's whole cross-section is compressed into a handful
of latent factors plus a set of *orthogonal peers* — instruments that still
carry information about the target after the common factors are removed. A
ridge regression maps that market state onto the target's return. Fair value
is then the price path implied by rolling that model-implied return forward
from an anchor ``H`` sessions back:

    FV_t = P_{t-H} · exp( Σ_{s=t-H+1..t} ŷ_s )

so the mispricing

    m_t = log(P_t / FV_t) = (realised H-period return) − (implied H-period return)

is exactly the target's cumulative excess move versus what the rest of the
world did. That quantity is stationary and interpretable. Accumulating implied
returns from a single fixed anchor — the tempting alternative — makes the gap
an integral of residuals, i.e. a random walk that drifts arbitrarily far from
price and can no longer be read as a valuation.

Everything below is strictly walk-forward: a model calibrated on
``[t-lookback, t)`` produces the estimates for ``[t, t+refit_every)`` and is
never asked about its own training window.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV

from .config import EngineConfig, REGIME_NAMES
from .factors import FittedFactors, fit_factors, loading_drift
from .features import compute_returns, rolling_zscore, sample_weights
from .oscillator import (dynamic_thresholds, detect_divergences, half_life,
                         mean_reversion_probability, normalise, ou_fit,
                         saturate, smooth_oscillator, threshold_signals)
from .regimes import classify_regimes
from .universe import CLASS_OF, YIELD_SYMBOLS

ALPHA_GRID = np.array([0.05, 0.2, 1.0, 5.0, 20.0, 100.0, 500.0])


# ---------------------------------------------------------------------------
# Per-refit record
# ---------------------------------------------------------------------------
@dataclass
class Calibration:
    """One walk-forward calibration and the block it was responsible for."""

    start: int                    # index of the first forecast session
    end: int                      # exclusive
    coef: np.ndarray
    intercept: float
    alpha: float
    peers: list[str]
    in_sample_r2: float
    factors: FittedFactors
    design_mean: np.ndarray
    design_scale: np.ndarray
    resid_coef: np.ndarray        # peers ~ factors projection, frozen
    factor_drift: float = np.nan
    coef_drift: float = np.nan
    regime: str = "TREND"
    feature_names: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _residualiser_fit(S: np.ndarray, X: np.ndarray) -> np.ndarray:
    S1 = np.column_stack([np.ones(len(S)), S])
    coef, *_ = np.linalg.lstsq(S1, X, rcond=None)
    return coef


def _residualiser_apply(coef: np.ndarray, S: np.ndarray, X: np.ndarray) -> np.ndarray:
    S1 = np.column_stack([np.ones(len(S)), S])
    return X - S1 @ coef


def _safe_corr(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Column-wise correlation of matrix A with vector y, NaN-safe."""
    Ac = A - A.mean(axis=0)
    yc = y - y.mean()
    den = np.sqrt((Ac ** 2).sum(axis=0) * (yc ** 2).sum())
    den = np.where(den <= 1e-12, np.nan, den)
    return np.nan_to_num((Ac * yc[:, None]).sum(axis=0) / den, nan=0.0)


def _cusum_breaks(std_resid: pd.Series, drift: float = 0.5, threshold: float = 5.0
                  ) -> tuple[pd.Series, list[pd.Timestamp]]:
    """Two-sided CUSUM over standardised residuals.

    Accumulates evidence that the residual mean has shifted; when either arm
    exceeds ``threshold`` a structural break is flagged and both arms reset.
    """
    hi = lo = 0.0
    path, breaks = [], []
    for ts, e in std_resid.items():
        if not np.isfinite(e):
            path.append(max(hi, lo))
            continue
        hi = max(0.0, hi + e - drift)
        lo = max(0.0, lo - e - drift)
        if max(hi, lo) > threshold:
            breaks.append(ts)
            hi = lo = 0.0
        path.append(max(hi, lo))
    return pd.Series(path, index=std_resid.index), breaks


def _rolling_oos_r2(resid: pd.Series, actual: pd.Series, window: int = 63) -> pd.Series:
    """Out-of-sample R² of the walk-forward forecasts, rolling."""
    sse = resid.pow(2).rolling(window, min_periods=window // 2).sum()
    mu = actual.rolling(window, min_periods=window // 2).mean()
    sst = (actual - mu).pow(2).rolling(window, min_periods=window // 2).sum()
    return (1.0 - sse / sst.replace(0, np.nan))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_engine(prices: pd.DataFrame, cfg: EngineConfig, progress=None) -> dict:
    """Calibrate the engine and return every artefact the dashboard needs.

    ``progress`` is an optional ``callable(fraction, message)``.
    """
    t_start = time.time()

    def say(frac: float, msg: str) -> None:
        if progress:
            progress(min(max(frac, 0.0), 1.0), msg)

    target = cfg.target
    if target not in prices.columns:
        raise ValueError(f"Target {target!r} is not in the loaded panel.")

    # ---- 1. returns & features -------------------------------------------
    say(0.05, f"Building the return panel for {prices.shape[1]} instruments…")
    rets = compute_returns(prices)
    Z = rolling_zscore(rets, window=63, min_periods=30)

    others = [c for c in rets.columns if c != target]
    if cfg.exclude_same_class:
        tcls = CLASS_OF.get(target)
        others = [c for c in others if CLASS_OF.get(c) != tcls]
    if len(others) < 10:
        raise RuntimeError("Fewer than 10 explanatory instruments remain after "
                           "exclusions — widen the universe.")

    T = len(rets)
    lookback = int(min(cfg.lookback, max(60, T // 3)))
    if T < lookback + 40:
        raise RuntimeError(f"Only {T} sessions available; need at least "
                           f"{lookback + 40} for a walk-forward calibration.")

    # ---- 2. regimes (walk-forward) ---------------------------------------
    say(0.15, "Classifying market regimes walk-forward…")
    regimes, regime_stability, state = classify_regimes(
        rets, n_regimes=cfg.n_regimes, train=lookback, refit_every=63)

    # ---- 3. walk-forward calibration -------------------------------------
    say(0.25, "Running walk-forward fair-value calibration…")
    y_all = rets[target].values
    X_all = Z[others].values
    # Raw (un-standardised) returns, kept solely for the proxy-exclusion test.
    # That test has to run in raw space: the rolling z-score rescales every
    # series by its own trailing volatility, which attenuates measured
    # correlation by 2-3 points — enough for a 0.999-correlated index proxy to
    # slip under a 0.99 threshold and quietly drive fair value onto price.
    R_all = rets[others].values

    yhat = np.full(T, np.nan)
    calibrations: list[Calibration] = []
    contrib_blocks: list[pd.DataFrame] = []
    prev_factors: FittedFactors | None = None
    prev_coef: np.ndarray | None = None

    n_blocks = max(1, (T - lookback + cfg.refit_every - 1) // cfg.refit_every)
    block_i = 0
    t = lookback
    while t < T:
        block_i += 1
        say(0.25 + 0.45 * block_i / n_blocks,
            f"Calibration {block_i}/{n_blocks} — window ending {rets.index[t].date()}")

        tr = slice(t - lookback, t)
        end = min(t + cfg.refit_every, T)
        fw = slice(t, end)

        Xtr = X_all[tr]
        ytr = y_all[tr]

        # -- 3a. latent factors, aligned to the previous calibration
        ff = fit_factors(Xtr, cfg.method, cfg.n_factors, previous=prev_factors)
        Str = ff.transform(Xtr)

        # -- 3b. orthogonal peers: what survives removing the common factors
        rc = _residualiser_fit(Str, Xtr)
        Rtr = _residualiser_apply(rc, Str, Xtr)
        y_res = _residualiser_apply(_residualiser_fit(Str, ytr[:, None]),
                                    Str, ytr[:, None]).ravel()

        corr_res = _safe_corr(Rtr, y_res)
        raw_corr = np.abs(_safe_corr(R_all[tr], ytr))
        eligible = raw_corr < cfg.exclude_corr_above          # drop near-proxies
        score = np.where(eligible, np.abs(corr_res), -np.inf)
        n_peers = int(min(cfg.n_peers, max(0, int(np.isfinite(score).sum()))))
        peer_pos = np.argsort(score)[::-1][:n_peers] if n_peers else np.array([], int)
        peers = [others[i] for i in peer_pos]

        # -- 3c. design matrix: factors + residualised peers, frozen scaling
        Dtr = np.column_stack([Str, Rtr[:, peer_pos]]) if n_peers else Str
        d_mean = Dtr.mean(axis=0)
        d_scale = Dtr.std(axis=0)
        d_scale[(d_scale == 0) | ~np.isfinite(d_scale)] = 1.0
        Dtr_s = (Dtr - d_mean) / d_scale

        # -- 3d. weighted ridge
        w = sample_weights(lookback, cfg.halflife_days)
        cur_reg = regimes.iloc[t - 1]
        if cfg.regime_weighting:
            same = (regimes.iloc[tr].values == cur_reg)
            w = w * np.where(same, 1.6, 0.7)
        w = w / w.mean()

        if cfg.auto_alpha:
            model = RidgeCV(alphas=ALPHA_GRID).fit(Dtr_s, ytr, sample_weight=w)
            alpha = float(model.alpha_)
        else:
            model = Ridge(alpha=cfg.ridge_alpha).fit(Dtr_s, ytr, sample_weight=w)
            alpha = float(cfg.ridge_alpha)

        in_r2 = float(model.score(Dtr_s, ytr, sample_weight=w))

        # -- 3e. forecast the forward block with the frozen model
        Xfw = X_all[fw]
        Sfw = ff.transform(Xfw)
        Rfw = _residualiser_apply(rc, Sfw, Xfw)
        Dfw = np.column_stack([Sfw, Rfw[:, peer_pos]]) if n_peers else Sfw
        Dfw_s = (Dfw - d_mean) / d_scale
        yhat[fw] = Dfw_s @ model.coef_ + model.intercept_

        names = [f"F{i + 1}" for i in range(Str.shape[1])] + peers
        contrib_blocks.append(pd.DataFrame(
            Dfw_s * model.coef_ * 1e4,                     # basis points per day
            index=rets.index[fw], columns=names))

        calibrations.append(Calibration(
            start=t, end=end, coef=model.coef_.copy(), intercept=float(model.intercept_),
            alpha=alpha, peers=peers, in_sample_r2=in_r2, factors=ff,
            design_mean=d_mean, design_scale=d_scale, resid_coef=rc,
            factor_drift=loading_drift(ff, prev_factors),
            coef_drift=(np.nan if prev_coef is None or prev_coef.shape != model.coef_.shape
                        else float(np.mean(np.abs(model.coef_ - prev_coef)))),
            regime=cur_reg, feature_names=names))

        prev_factors, prev_coef = ff, model.coef_.copy()
        t = end

    # ---- 4. fair value paths ---------------------------------------------
    say(0.75, "Reconstructing fair-value paths…")
    price = prices[target]
    implied = pd.Series(yhat, index=rets.index)
    cum = implied.fillna(0.0).cumsum()

    # Residual = the part of the target's move the market state did not
    # explain. Needed here to size the fair-value uncertainty halo, and reused
    # for the stability diagnostics further down.
    resid = rets[target] - implied
    resid[implied.isna()] = np.nan
    resid_vol = resid.rolling(42, min_periods=15).std()

    # Yield series are modelled in differences, not log returns, so their fair
    # value must be rolled forward additively. Compounding a yield change the
    # way a price return compounds would be nonsense.
    is_yield = target in YIELD_SYMBOLS

    def fv_path(H: int) -> pd.Series:
        """Anchored fair value: roll the implied change H sessions forward."""
        anchor = price.shift(H)
        drift = cum - cum.shift(H)
        fv = anchor + 100.0 * drift if is_yield else anchor * np.exp(drift)
        fv.iloc[:lookback + H] = np.nan           # before the model existed
        return fv.where(fv > 0)                   # log() below needs positivity

    H = int(cfg.fv_horizon)
    fv = fv_path(H)
    gap = price - fv
    mis = np.log(price / fv)
    mis_pct = (np.exp(mis) - 1.0) * 100.0

    # Residual-risk halo around the fair-value path, in the target's own space.
    halo = resid_vol * np.sqrt(H)
    fv_lo = fv - 100.0 * halo if is_yield else fv * np.exp(-halo)
    fv_hi = fv + 100.0 * halo if is_yield else fv * np.exp(halo)

    # ---- 5. oscillator ----------------------------------------------------
    say(0.82, "Normalising the oscillator and deriving thresholds…")
    z = normalise(mis, window=cfg.norm_window, robust=cfg.robust_norm)
    fvo_raw = saturate(z, kappa=cfg.saturation)
    fvo, band = smooth_oscillator(fvo_raw, method=cfg.smoothing, span=cfg.smooth_span)
    ob, os_ = dynamic_thresholds(fvo, q=cfg.threshold_q, window=cfg.threshold_window)

    long_entry, short_entry = threshold_signals(fvo, ob, os_)
    div_bear, div_bull = detect_divergences(price, fvo, k=5)

    theta, sigma_ou = ou_fit(z, window=max(cfg.norm_window, 126))
    p_mr = mean_reversion_probability(z, theta, sigma_ou, horizon=cfg.mr_horizon)
    hl = half_life(theta)

    # multi-timeframe: same model, different anchor horizons
    mtf: dict[str, pd.Series] = {}
    mtf_z: dict[str, pd.Series] = {}
    for h in cfg.mtf_horizons:
        m_h = np.log(price / fv_path(int(h)))
        z_h = normalise(m_h, window=cfg.norm_window, robust=cfg.robust_norm)
        mtf[f"{int(h)}d"] = saturate(z_h, kappa=cfg.saturation)
        mtf_z[f"{int(h)}d"] = z_h

    # ---- 6. diagnostics ---------------------------------------------------
    say(0.90, "Computing stability diagnostics and confidence…")
    oos_r2 = _rolling_oos_r2(resid, rets[target], window=63)
    sigma_e = resid_vol * 100.0
    resid_scale = resid.rolling(126, min_periods=40).std().replace(0, np.nan)
    std_resid = (resid / resid_scale)
    cusum, breaks = _cusum_breaks(std_resid.fillna(0.0))

    # per-session drift series, held flat within each calibration block
    factor_drift = pd.Series(np.nan, index=rets.index)
    coef_drift = pd.Series(np.nan, index=rets.index)
    alpha_path = pd.Series(np.nan, index=rets.index)
    in_r2_path = pd.Series(np.nan, index=rets.index)
    for c in calibrations:
        sl = slice(c.start, c.end)
        factor_drift.iloc[sl] = c.factor_drift
        coef_drift.iloc[sl] = c.coef_drift
        alpha_path.iloc[sl] = c.alpha
        in_r2_path.iloc[sl] = c.in_sample_r2
    factor_drift = factor_drift.ffill()
    coef_drift = coef_drift.ffill()

    coverage = prices.notna().mean(axis=1)

    fit_term = oos_r2.clip(0, 0.9) / 0.9
    beta_term = np.exp(-6.0 * coef_drift.fillna(coef_drift.median()))
    fac_term = np.exp(-6.0 * factor_drift.fillna(factor_drift.median()))
    conf = 100.0 * (0.40 * fit_term.fillna(0.0)
                    + 0.20 * beta_term.clip(0, 1)
                    + 0.15 * fac_term.clip(0, 1)
                    + 0.15 * regime_stability.clip(0, 1)
                    + 0.10 * coverage.clip(0, 1))
    conf = conf.clip(3, 99)

    # ---- 7. attribution history ------------------------------------------
    contrib = (pd.concat(contrib_blocks, axis=0).sort_index()
               if contrib_blocks else pd.DataFrame(index=rets.index))
    contrib = contrib.loc[:, ~contrib.columns.duplicated()]

    say(1.0, f"Calibrated in {time.time() - t_start:.1f}s")

    return dict(
        # identity / inputs
        config=cfg, target=target, dates=rets.index, prices=prices, rets=rets, Z=Z,
        others=others, lookback=lookback, n_assets=prices.shape[1],
        # valuation
        price=price, fv=fv, fv_lo=fv_lo, fv_hi=fv_hi, gap=gap, mis=mis,
        mis_pct=mis_pct, implied=implied, target_is_yield=is_yield,
        # oscillator
        z=z, fvo_raw=fvo_raw, fvo=fvo, band=band, ob=ob, os_=os_,
        long_entry=long_entry, short_entry=short_entry,
        div_bear=div_bear, div_bull=div_bull,
        theta=theta, sigma_ou=sigma_ou, p_mr=p_mr, half_life=hl,
        mtf=mtf, mtf_z=mtf_z,
        # regimes
        regimes=regimes, regime_stability=regime_stability, market_state=state,
        # diagnostics
        resid=resid, oos_r2=oos_r2, sigma_e=sigma_e, cusum=cusum, breaks=breaks,
        factor_drift=factor_drift, coef_drift=coef_drift, alpha_path=alpha_path,
        in_sample_r2=in_r2_path, confidence=conf, coverage=coverage,
        # explainability
        contrib=contrib, calibrations=calibrations, last=calibrations[-1],
        elapsed=time.time() - t_start,
    )


def snapshot(E: dict) -> dict:
    """Latest-session summary of every headline output."""
    i = -1
    price = float(E["price"].iloc[i])
    fv_series = E["fv"].dropna()
    fv = float(fv_series.iloc[i]) if len(fv_series) else np.nan
    mis_pct = float(E["mis_pct"].iloc[i]) if np.isfinite(fv) else np.nan
    fvo = float(E["fvo"].iloc[i])
    z = float(E["z"].iloc[i])
    ob = float(E["ob"].iloc[i]) if np.isfinite(E["ob"].iloc[i]) else np.nan
    os_ = float(E["os_"].iloc[i]) if np.isfinite(E["os_"].iloc[i]) else np.nan

    if np.isfinite(ob) and fvo > ob:
        verdict, vcolor = "RICH", "#ff5d6c"
    elif np.isfinite(os_) and fvo < os_:
        verdict, vcolor = "CHEAP", "#2fd08c"
    elif abs(fvo) < 15:
        verdict, vcolor = "FAIRLY VALUED", "#f2b544"
    else:
        verdict, vcolor = ("MILDLY RICH", "#e0a94a") if fvo > 0 else ("MILDLY CHEAP", "#5fd39a")

    hist = E["fvo"].dropna()
    pctile = float((hist < fvo).mean() * 100) if len(hist) else np.nan

    return dict(
        price=price, fv=fv, gap=price - fv if np.isfinite(fv) else np.nan,
        mis_pct=mis_pct, fvo=fvo, z=z, ob=ob, os_=os_, percentile=pctile,
        verdict=verdict, verdict_color=vcolor,
        regime=str(E["regimes"].iloc[i]),
        confidence=float(E["confidence"].iloc[i]),
        p_mr=float(E["p_mr"].iloc[i]),
        half_life=float(E["half_life"].iloc[i]),
        sigma_e=float(E["sigma_e"].iloc[i]) if np.isfinite(E["sigma_e"].iloc[i]) else np.nan,
        oos_r2=float(E["oos_r2"].iloc[i]) if np.isfinite(E["oos_r2"].iloc[i]) else np.nan,
        day_change=float(E["price"].iloc[-1] / E["price"].iloc[-2] - 1.0),
        asof=E["dates"][-1],
    )
