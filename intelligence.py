"""
Sanket Priority Engine — Self-Calibration via Bayesian search.

Methodology:
- Train / validation split by date (default 70 / 30) — out-of-sample IR is reported.
- Objective: per-horizon Information Ratio of daily Spearman IC between
  Priority and forward return, averaged across hold periods. Avoids the
  overlapping-window double-counting of v1's multi-horizon-sum Sharpe.
- L2 regularization on factor weights to discourage runaway fits.
- Boundary rows (last max(hold_periods) dates and any NaN-return rows) dropped.
- Real Optuna fANOVA param-importance, with weight-share fallback.

Performance:
- Weight-invariant arrays (factor matrix, penalties, damping constant, return
  ranks) are precomputed once at __init__. Each Optuna trial reduces to a
  matrix-vector product, a tier-mult lookup, two grouped rank passes, and a
  vectorized per-date Pearson via np.add.reduceat. Roughly 50× faster than
  calling pe.compute_priority per (date, trial).
"""
import numpy as np
import pandas as pd
import optuna
import priority_engine as pe

optuna.logging.set_verbosity(optuna.logging.WARNING)

_VR_W = {'LOW': 1.20, 'NORMAL': 1.00, 'HIGH': 0.85, 'EXTREME': 0.55}
_TIER_IDX = {
    'A: Long': 0, 'A: Short': 0,
    'B: Long': 1, 'B: Short': 1,
    'C: Long': 2, 'C: Short': 2,
}
_TIER_DEFAULT_IDX = 3


# ────────────────────────────────────────────────────────────────────────
# Vectorized helpers (numpy, no pandas in the hot path)
# ────────────────────────────────────────────────────────────────────────
def _rank_grouped(values, starts, ends):
    """Within-group ordinal rank (1..n). Input must be sorted by group."""
    out = np.empty(len(values), dtype=np.float64)
    for s, e in zip(starts, ends):
        seg = values[s:e]
        order = seg.argsort()
        ranks = np.empty(e - s, dtype=np.float64)
        ranks[order] = np.arange(1, e - s + 1, dtype=np.float64)
        out[s:e] = ranks
    return out


def _rank_pct_grouped(values, starts, ends, sizes):
    """Within-group rank percentile (rank / n) ∈ (0, 1]."""
    out = np.empty(len(values), dtype=np.float64)
    for i, (s, e) in enumerate(zip(starts, ends)):
        n = sizes[i]
        seg = values[s:e]
        order = seg.argsort()
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(1, n + 1, dtype=np.float64)
        out[s:e] = ranks / n
    return out


def _pearson_grouped(x, y, starts, sizes, min_n=5):
    """
    Per-group Pearson correlation, fully vectorized via add.reduceat.
    Returns array of shape (n_groups,); NaN where group is too small or degenerate.
    Input must be sorted by group; `starts` is the group-start index array.
    """
    n = sizes.astype(np.float64)
    sx  = np.add.reduceat(x,     starts)
    sy  = np.add.reduceat(y,     starts)
    sxx = np.add.reduceat(x * x, starts)
    syy = np.add.reduceat(y * y, starts)
    sxy = np.add.reduceat(x * y, starts)

    mx = sx / n
    my = sy / n
    cov  = sxy / n - mx * my
    varx = sxx / n - mx * mx
    vary = syy / n - my * my

    denom = np.sqrt(np.maximum(varx, 0.0) * np.maximum(vary, 0.0))
    valid = (sizes >= min_n) & (denom > 1e-12)
    out = np.full(len(starts), np.nan, dtype=np.float64)
    np.divide(cov, denom, out=out, where=valid)
    return out


# ────────────────────────────────────────────────────────────────────────
# Precomputed dataset — built once per (train, val) split
# ────────────────────────────────────────────────────────────────────────
class _PrecomputedDataset:
    """Weight-invariant arrays for fast per-trial IC scoring."""

    def __init__(self, df: pd.DataFrame, hold_periods, min_xsect: int):
        df = df.copy()
        if 'Date' not in df.columns:
            self.empty = True
            return
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        # Drop any row with NaN forward return at any tracked horizon —
        # guarantees per-horizon IC uses the same row set, lets us pre-rank
        # returns once.
        ret_cols = [f'Ret_{h}b' for h in hold_periods if f'Ret_{h}b' in df.columns]
        if ret_cols:
            df = df.dropna(subset=ret_cols).reset_index(drop=True)

        # Drop dates with insufficient cross-section
        if not df.empty:
            counts = df['Date'].value_counts()
            keep = counts[counts >= min_xsect].index
            df = df[df['Date'].isin(keep)].sort_values('Date').reset_index(drop=True)

        self.empty = df.empty
        if self.empty:
            return

        # ── Group boundaries (data is sorted by Date) ──
        dates = df['Date'].values
        change = np.concatenate(([True], dates[1:] != dates[:-1]))
        self.starts = np.flatnonzero(change).astype(np.int64)
        self.sizes  = np.diff(np.append(self.starts, len(dates))).astype(np.int64)
        self.ends   = self.starts + self.sizes
        self.n_rows = len(df)
        self.n_groups = len(self.starts)
        self.dates_unique = dates[self.starts]

        # ── Factor matrix M (N, 5): F1, F2, F3=conv/20, F4, F5 ──
        f1   = self._col_f(df, 'F1_PriceMom', 0.0)
        f2   = self._col_f(df, 'F2_VolQual',  0.0)
        conv = self._col_f(df, 'Conviction',  0.0)
        f3   = conv / 20.0
        f4   = self._col_f(df, 'Pulse', 0.0)
        f5   = self._col_f(df, 'HMM_Bull', 0.33) - self._col_f(df, 'HMM_Bear', 0.33)
        f7   = self._col_f(df, 'LO', 0.0) / 100.0   # liquidity range-extension (reversion)
        self.M = np.column_stack([f1, f2, f3, f4, f5, f7])

        # ── Penalty matrices (long_rev, long_div) and (short_rev, short_div) ──
        wt1 = self._col_f(df, 'Wave', 0.0)
        if 'WT1_5ago' in df.columns:
            wt5_raw = df['WT1_5ago'].to_numpy(dtype=np.float64)
            wt5 = np.where(np.isnan(wt5_raw), wt1, wt5_raw)
        else:
            wt5 = wt1.copy()
        travel = wt1 - wt5

        long_rev = np.where(
            (wt1 > 60) & (travel < 0),
            np.minimum(1.0, ((wt1 - 60) / 40.0) * (np.abs(conv) / 50.0)),
            0.0,
        )
        short_rev = np.where(
            (wt1 < -60) & (travel > 0),
            np.minimum(1.0, ((-wt1 - 60) / 40.0) * (np.abs(conv) / 50.0)),
            0.0,
        )

        bull_div = self._col_b(df, 'Bullish_Div')
        bear_div = self._col_b(df, 'Bearish_Div')
        long_div  = (bear_div & (conv > 30)).astype(np.float64)
        short_div = (bull_div & (conv < -30)).astype(np.float64)

        self.P_long  = np.column_stack([long_rev,  long_div])
        self.P_short = np.column_stack([short_rev, short_div])

        # ── Tier index (constant per row) ──
        st = (df['SignalType'].fillna('-').to_numpy()
              if 'SignalType' in df.columns
              else np.full(len(df), '-', dtype=object))
        self.tier_idx = np.fromiter(
            (_TIER_IDX.get(s, _TIER_DEFAULT_IDX) for s in st),
            dtype=np.int8, count=len(st),
        )

        # ── Constant damping factor (everything that doesn't depend on weights) ──
        if 'Vol_Regime' in df.columns:
            vr_raw = df['Vol_Regime'].fillna('NORMAL').to_numpy()
        else:
            vr_raw = np.full(len(df), 'NORMAL', dtype=object)
        vr_w = np.fromiter((_VR_W.get(v, 1.0) for v in vr_raw),
                           dtype=np.float64, count=len(vr_raw))
        conf = self._col_f(df, 'Regime_Confidence', 0.5)
        cp   = self._col_b(df, 'Change_Point').astype(np.float64)
        self.damp_const = vr_w * (0.6 + 0.4 * conf) * (1.0 - 0.35 * cp)

        # ── Pre-ranked forward returns per horizon (within each date) ──
        self.ret_ranks = {}
        self.has_horizon = []
        for h in hold_periods:
            col = f'Ret_{h}b'
            if col not in df.columns:
                continue
            r = df[col].to_numpy(dtype=np.float64)
            self.ret_ranks[h] = _rank_grouped(r, self.starts, self.ends)
            self.has_horizon.append(h)

    @staticmethod
    def _col_f(df, name, default):
        if name in df.columns:
            return df[name].fillna(default).to_numpy(dtype=np.float64)
        return np.full(len(df), float(default), dtype=np.float64)

    @staticmethod
    def _col_b(df, name):
        if name in df.columns:
            return df[name].fillna(False).astype(bool).to_numpy()
        return np.zeros(len(df), dtype=bool)


# ────────────────────────────────────────────────────────────────────────
# Fast scoring kernel — pure numpy on precomputed arrays
# ────────────────────────────────────────────────────────────────────────
def _evaluate_ic(precomp: _PrecomputedDataset, weights: dict, min_xsect: int = 5) -> float:
    """Asymmetric kernel — separate long/short β and γ, shared tier mults."""
    if precomp.empty:
        return -100.0

    w_beta_long = np.array([
        weights['beta_F1_pricemom_long'],
        weights['beta_F2_volqual_long'],
        weights['beta_F3_wave_long'],
        weights['beta_F4_pulse_long'],
        weights['beta_F5_regime_long'],
        weights.get('beta_F7_liq_long', 0.0),
    ], dtype=np.float64)
    w_beta_short = np.array([
        weights['beta_F1_pricemom_short'],
        weights['beta_F2_volqual_short'],
        weights['beta_F3_wave_short'],
        weights['beta_F4_pulse_short'],
        weights['beta_F5_regime_short'],
        weights.get('beta_F7_liq_short', 0.0),
    ], dtype=np.float64)
    w_gamma_long = np.array([
        weights['gamma_reversion_long'],
        weights['gamma_divergence_long'],
    ], dtype=np.float64)
    w_gamma_short = np.array([
        weights['gamma_reversion_short'],
        weights['gamma_divergence_short'],
    ], dtype=np.float64)
    w_F6_long  = float(weights['beta_F6_xsect_long'])
    w_F6_short = float(weights['beta_F6_xsect_short'])
    tier_arr = np.array([
        weights['tier_A_mult'],
        weights['tier_B_mult'],
        weights['tier_C_mult'],
        weights['tier_default_mult'],
    ], dtype=np.float64)

    Mb_long  = precomp.M @ w_beta_long
    Mb_short = precomp.M @ w_beta_short
    PgL = precomp.P_long  @ w_gamma_long
    PgS = precomp.P_short @ w_gamma_short

    base_long  =  Mb_long  - PgL
    base_short = -Mb_short - PgS   # short flips factor signs

    damp = precomp.damp_const * tier_arr[precomp.tier_idx]

    pre_long  = base_long  * damp
    pre_short = base_short * damp

    F6_long  = (_rank_pct_grouped(pre_long,  precomp.starts, precomp.ends, precomp.sizes) - 0.5) * 2.0
    F6_short = (_rank_pct_grouped(pre_short, precomp.starts, precomp.ends, precomp.sizes) - 0.5) * 2.0

    priority_long  = (base_long  + w_F6_long  * F6_long)  * damp
    priority_short = (base_short + w_F6_short * F6_short) * damp

    pl_ranks = _rank_grouped(priority_long,  precomp.starts, precomp.ends)
    ps_ranks = _rank_grouped(priority_short, precomp.starts, precomp.ends)

    # corr(rank(x), rank(-y)) = -corr(rank(x), rank(y)) — reuse the same ret_ranks.
    horizon_irs = []
    for h in precomp.has_horizon:
        r = precomp.ret_ranks[h]
        ic_l =  _pearson_grouped(pl_ranks, r, precomp.starts, precomp.sizes, min_xsect)
        ic_s = -_pearson_grouped(ps_ranks, r, precomp.starts, precomp.sizes, min_xsect)
        ics = np.concatenate([ic_l, ic_s])
        ics = ics[~np.isnan(ics)]
        if len(ics) >= 3:
            horizon_irs.append(ics.mean() / max(ics.std(), 1e-6))

    if not horizon_irs:
        return -100.0
    return float(np.mean(horizon_irs))


# ────────────────────────────────────────────────────────────────────────
# Public class
# ────────────────────────────────────────────────────────────────────────
class PriorityTuner:
    def __init__(self,
                 historical_data: pd.DataFrame,
                 hold_periods=None,
                 train_frac: float = 0.70,
                 l2_alpha: float = 0.001,
                 min_xsect: int = 5,
                 enable_f7: bool = False):
        # enable_f7: whether the LO range-extension factor (F7) participates in the
        # ranking search. OFF by default — F7 is collinear with the existing
        # WaveTrend reversion penalty + F3, so on thin data the optimizer can hand it
        # large, unstable, partly-cancelling weights that move live rankings without
        # adding real out-of-sample edge. Kept dormant (pinned to 0) until validated:
        # the factor, its math, and the Set-B liquidity confidence features all remain;
        # only its *ranking weight* is gated. Flip on to A/B-test F7's fANOVA
        # importance + val IR against a no-F7 baseline before trusting it.
        self.hold_periods = list(hold_periods) if hold_periods is not None else list(pe.HOLD_HORIZONS)
        self.train_frac   = train_frac
        self.l2_alpha     = l2_alpha
        self.min_xsect    = min_xsect
        self.enable_f7    = enable_f7
        self.best_weights = pe.DEFAULT_W.copy()
        self.study        = None
        self.train_score  = None
        self.val_score    = None

        df = historical_data.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')

        # Drop the trailing max(hold_periods) dates whose forward returns are NaN.
        max_h = max(self.hold_periods)
        sorted_dates = sorted(df['Date'].unique())
        if len(sorted_dates) > max_h:
            cutoff = sorted_dates[-max_h]
            df = df[df['Date'] < cutoff]
            sorted_dates = sorted(df['Date'].unique())

        # Train / validation split by date (chronological).
        n_total = len(sorted_dates)
        n_train = max(1, int(n_total * train_frac))
        train_dates = set(sorted_dates[:n_train])
        val_dates   = set(sorted_dates[n_train:])

        self.train_data = df[df['Date'].isin(train_dates)].copy()
        self.val_data   = (df[df['Date'].isin(val_dates)].copy()
                           if val_dates else df.iloc[0:0].copy())

        # Precompute weight-invariant arrays once per split.
        self._train_pre = _PrecomputedDataset(self.train_data, self.hold_periods, min_xsect)
        self._val_pre   = _PrecomputedDataset(self.val_data,   self.hold_periods, min_xsect)

    def _ic_score(self, precomp: _PrecomputedDataset, weights: dict) -> float:
        return _evaluate_ic(precomp, weights, self.min_xsect)

    def optimize(self, n_trials: int = 50, progress_callback=None):
        def objective(trial):
            w = {
                # Long-side factor weights
                'beta_F1_pricemom_long':  trial.suggest_float('beta_F1_pricemom_long',  0.0, 40.0),
                'beta_F2_volqual_long':   trial.suggest_float('beta_F2_volqual_long',   0.0, 30.0),
                'beta_F3_wave_long':      trial.suggest_float('beta_F3_wave_long',     0.0, 30.0),
                'beta_F4_pulse_long':     trial.suggest_float('beta_F4_pulse_long',    0.0, 40.0),
                'beta_F5_regime_long':    trial.suggest_float('beta_F5_regime_long',   0.0, 50.0),
                'beta_F6_xsect_long':     trial.suggest_float('beta_F6_xsect_long',    0.0, 40.0),
                # F7 searched only when explicitly enabled; pinned to 0 otherwise so it
                # can't acquire spurious (collinear) weight on thin data — see __init__.
                'beta_F7_liq_long':       (trial.suggest_float('beta_F7_liq_long',  -40.0, 40.0)
                                           if self.enable_f7 else 0.0),
                'gamma_reversion_long':   trial.suggest_float('gamma_reversion_long',  0.0, 40.0),
                'gamma_divergence_long':  trial.suggest_float('gamma_divergence_long', 0.0, 40.0),
                # Short-side factor weights
                'beta_F1_pricemom_short': trial.suggest_float('beta_F1_pricemom_short', 0.0, 40.0),
                'beta_F2_volqual_short':  trial.suggest_float('beta_F2_volqual_short',  0.0, 30.0),
                'beta_F3_wave_short':     trial.suggest_float('beta_F3_wave_short',     0.0, 30.0),
                'beta_F4_pulse_short':    trial.suggest_float('beta_F4_pulse_short',    0.0, 40.0),
                'beta_F5_regime_short':   trial.suggest_float('beta_F5_regime_short',   0.0, 50.0),
                'beta_F6_xsect_short':    trial.suggest_float('beta_F6_xsect_short',    0.0, 40.0),
                'beta_F7_liq_short':      (trial.suggest_float('beta_F7_liq_short', -40.0, 40.0)
                                           if self.enable_f7 else 0.0),
                'gamma_reversion_short':  trial.suggest_float('gamma_reversion_short',  0.0, 40.0),
                'gamma_divergence_short': trial.suggest_float('gamma_divergence_short', 0.0, 40.0),
                # Shared tier multipliers
                'tier_A_mult':       trial.suggest_float('tier_A_mult',       0.5, 2.0),
                'tier_B_mult':       trial.suggest_float('tier_B_mult',       0.5, 2.0),
                'tier_C_mult':       trial.suggest_float('tier_C_mult',       0.5, 2.0),
                'tier_default_mult': trial.suggest_float('tier_default_mult', 0.5, 2.0),
            }

            score = self._ic_score(self._train_pre, w)

            # L2 on β / γ — normalize by count so the penalty stays scale-invariant
            # across symmetric (8 keys) vs asymmetric (16 keys) search spaces.
            beta_keys = [k for k in w if k.startswith(('beta_', 'gamma_'))]
            n_beta = len(beta_keys) or 1
            l2 = sum(w[k] ** 2 for k in beta_keys) / n_beta / 125.0  # mean(w²) / 125
            score -= self.l2_alpha * l2

            if progress_callback:
                progress_callback(trial.number, n_trials, score)
            return score

        self.study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        self.study.optimize(objective, n_trials=n_trials)

        self.best_weights = {**pe.DEFAULT_W, **self.study.best_params}
        self.train_score  = float(self.study.best_value)
        return self.best_weights, self.train_score

    def evaluate_validation(self) -> float:
        """Compute validation IR on held-out dates using best weights. Call after optimize()."""
        self.val_score = self._ic_score(self._val_pre, self.best_weights)
        return self.val_score

    def get_param_importance(self) -> dict:
        """Optuna fANOVA importance over trial history. Falls back to weight-share."""
        if self.study is None or len(self.study.trials) < 2:
            return {}
        try:
            imp = optuna.importance.get_param_importances(self.study)
            total = sum(imp.values())
            if total <= 0:
                raise ValueError("fANOVA returned zero importance.")
            return {k: (v / total) * 100.0 for k, v in imp.items()}
        except Exception:
            total = sum(abs(v) for v in self.best_weights.values()) or 1.0
            return {k: abs(v) / total * 100.0 for k, v in self.best_weights.items()}

    def get_sensitivity(self):
        return self.get_param_importance()


# ────────────────────────────────────────────────────────────────────────
# Signal-Confidence Calibration (Layer 2)
#
# The PriorityTuner above optimizes *ranking* (cross-sectional IC) over the
# whole universe. It does NOT learn whether an individual fired A/B/C signal
# is a true or false positive. This calibrator does exactly that: on the
# harvested panel it labels each fired signal by whether its forward return
# over `horizon` bars moved the signal's way, then fits a per-set logistic
# P(true | regime/context features). Validated out-of-sample by date, same as
# the IC objective. The result feeds priority_engine.compute_signal_confidence
# so the screener's Intel_Confidence becomes a calibrated probability.
# ────────────────────────────────────────────────────────────────────────
def _fit_logistic(X, y, l2=1.0, iters=100, tol=1e-7):
    """Ridge-regularized logistic regression via Newton-IRLS (numpy only).

    X already standardized (no intercept column — intercept fit unpenalized).
    Returns (coef[k], intercept). Robust to separation via the L2 term and a
    capped step; falls back gracefully if the Hessian is singular.
    """
    n, k = X.shape
    Xb = np.column_stack([np.ones(n), X])           # intercept in column 0
    beta = np.zeros(k + 1)
    reg = np.full(k + 1, l2)
    reg[0] = 0.0                                     # do not penalize intercept
    for _ in range(iters):
        eta = np.clip(Xb @ beta, -35.0, 35.0)
        p = 1.0 / (1.0 + np.exp(-eta))
        W = np.clip(p * (1.0 - p), 1e-6, None)
        grad = Xb.T @ (p - y) + reg * beta
        H = (Xb * W[:, None]).T @ Xb + np.diag(reg)
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        beta_new = beta - step
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    return beta[1:], float(beta[0])


def _auc(y, scores):
    """ROC AUC via the Mann–Whitney U statistic (ties handled by mid-ranks)."""
    y = np.asarray(y)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty(len(scores), dtype=float)
    s = scores[order]
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    # average ranks for ties
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[order[j + 1]] == s[order[i]]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    sum_pos = ranks[y == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def calibrate_signal_confidence(ts_df: pd.DataFrame,
                                horizon: int = 5,
                                train_frac: float = 0.70,
                                l2: float = 1.0,
                                deadband_frac: float = 0.10,
                                min_set_samples: int = 120,
                                min_total_samples: int = 150) -> dict:
    """Fit per-set P(true) logistic on harvested fired signals. None if too sparse.

    Label (multi-horizon, magnitude-aware): a fired signal is "true" if its mean
    directional forward return across all tracked horizons (Ret_2b…Ret_13b)
    clearly beats a deadband — not just a sign flip on a single horizon. The
    deadband is ``deadband_frac`` × the typical |mean directional return| over
    fired rows, so a barely-positive drift (or going nowhere) counts as a false
    positive, not a win. Asset-agnostic (self-scaling). Features come from
    pe.signal_conf_features so train and inference are identical. Split
    chronologically by date. Returns a model dict {feat_mean, feat_std,
    sets:{A,B,C,_pooled:{coef,intercept,...}}, horizon, horizons, deadband,
    val_auc, val_precision_lift, base_rate, n_train}.
    """
    if ts_df is None or ts_df.empty or not {'Date', 'SignalType'}.issubset(ts_df.columns):
        return None

    df = ts_df.copy()
    df['Date'] = pd.to_datetime(df['Date'])

    # Tracked forward-return horizons present in the harvest (Ret_<h>b).
    ret_cols = [f'Ret_{h}b' for h in pe.HOLD_HORIZONS if f'Ret_{h}b' in df.columns]
    if not ret_cols:
        return None
    horizons_used = [int(c[4:-1]) for c in ret_cols]

    X_all, dir_sign, set_letter, fired = pe.signal_conf_features(df)
    d = np.nan_to_num(dir_sign, nan=0.0)

    # Mean directional forward return across horizons (NaN where no horizon resolves).
    R = df[ret_cols].to_numpy(dtype=float) * d[:, None]
    with np.errstate(invalid='ignore'):
        dr = np.nanmean(R, axis=1)
    fired = np.asarray(fired) & np.isfinite(dr)
    if fired.sum() < min_total_samples:
        return None

    # Magnitude deadband, self-scaled to the universe's typical move.
    typ = float(np.median(np.abs(dr[fired]))) if fired.any() else 0.0
    deadband = deadband_frac * typ
    y_all = (dr > deadband).astype(float)

    # Chronological split by date over fired rows only.
    dates = df['Date'].to_numpy()
    fired_dates = np.unique(dates[fired])
    if len(fired_dates) < 8:
        return None
    n_train = max(1, int(len(fired_dates) * train_frac))
    is_train = np.isin(dates, fired_dates[:n_train])   # type-safe over datetime64

    tr = fired & is_train
    va = fired & ~is_train
    if tr.sum() < min_total_samples:
        return None

    # Standardize features on the training subset.
    Xtr_raw = X_all[tr]
    mean = Xtr_raw.mean(axis=0)
    std = Xtr_raw.std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)

    def _standardize(mask):
        return (X_all[mask] - mean) / std

    sets_out = {}
    set_arr = set_letter

    def _fit_subset(mask):
        if mask.sum() < min_set_samples:
            return None
        Xz = _standardize(mask)
        y = y_all[mask]
        if y.sum() < 5 or (len(y) - y.sum()) < 5:   # need both classes
            return None
        coef, b0 = _fit_logistic(Xz, y, l2=l2)
        return {
            'coef': [float(c) for c in coef],
            'intercept': float(b0),
            'n_train': int(mask.sum()),
            'base_rate': float(y.mean()),
        }

    # Pooled model (all sets) — always the fallback.
    pooled = _fit_subset(tr)
    if pooled is None:
        return None
    sets_out['_pooled'] = pooled

    for s in ('A', 'B', 'C'):
        m = _fit_subset(tr & (set_arr == s))
        if m is not None:
            sets_out[s] = m

    model = {
        'version': 2,
        'horizon': int(horizon),
        'horizons': horizons_used,
        'deadband': float(deadband),
        'label': 'mean directional return over horizons > deadband',
        'feature_names': list(pe.CONF_FEATURES),
        'feat_mean': [float(v) for v in mean],
        'feat_std': [float(v) for v in std],
        'sets': sets_out,
        'base_rate': float(y_all[tr].mean()),
        'n_train': int(tr.sum()),
    }

    # Out-of-sample diagnostics: AUC and precision lift (top-half conf vs base).
    if va.sum() >= 30:
        val_df = df[va].copy()
        proba = pe.predict_signal_confidence(val_df, model)
        yv = y_all[va]
        ok = ~np.isnan(proba)
        if ok.sum() >= 30 and 0 < yv[ok].sum() < ok.sum():
            model['val_auc'] = _auc(yv[ok], proba[ok])
            base = float(yv[ok].mean())
            hi = proba[ok] >= np.median(proba[ok])
            top_prec = float(yv[ok][hi].mean()) if hi.sum() else float('nan')
            model['base_rate_val'] = base
            model['val_top_half_precision'] = top_prec
            model['val_precision_lift'] = (top_prec - base) if base > 0 else float('nan')
            model['n_val'] = int(ok.sum())

    return model


# ════════════════════════════════════════════════════════════════════════
# Layer 3 — Meta-Conviction calibration
# ════════════════════════════════════════════════════════════════════════

def _spearman_ir(dates, scores, ret_mat, min_n: int = 5) -> float:
    """Cross-sectional rank-IR of ``scores`` against DIRECTIONAL forward returns.

    Mirrors the Priority Engine's IC methodology: per horizon, per date, the
    Spearman IC (rank-correlation) of scores vs directional return across that
    date's signals; IR_h = mean / std of the per-date ICs; IR = mean over
    horizons. ``ret_mat`` is (n, H) of direction-signed returns, so a good
    long OR short reads as positive — making meta and priority directly
    comparable. Dates with < ``min_n`` usable rows are skipped.
    """
    dates = np.asarray(dates)
    scores = np.asarray(scores, dtype=float)
    ret_mat = np.asarray(ret_mat, dtype=float)
    if ret_mat.ndim == 1:
        ret_mat = ret_mat[:, None]
    uniq = np.unique(dates)
    irs = []
    for h in range(ret_mat.shape[1]):
        rh = ret_mat[:, h]
        ics = []
        for d in uniq:
            m = (dates == d) & np.isfinite(rh) & np.isfinite(scores)
            if int(m.sum()) < min_n:
                continue
            s = scores[m]; r = rh[m]
            if np.std(s) < 1e-9 or np.std(r) < 1e-9:
                continue
            sr = pd.Series(s).rank().to_numpy()
            rr = pd.Series(r).rank().to_numpy()
            ic = np.corrcoef(sr, rr)[0, 1]
            if np.isfinite(ic):
                ics.append(ic)
        if len(ics) >= 3:
            a = np.asarray(ics)
            irs.append(a.mean() / max(a.std(), 1e-6))
    return float(np.mean(irs)) if irs else float('nan')


def calibrate_meta_conviction(ts_df: pd.DataFrame,
                              weights: dict,
                              train_frac: float = 0.70,
                              l2: float = 1.0,
                              deadband_frac: float = 0.10,
                              min_total_samples: int = 150,
                              min_xsect: int = 5) -> dict:
    """Fit the Layer-3 meta-conviction model on harvested fired signals. None if sparse.

    Fuses the two orthogonal inputs — cross-sectional Priority rank and per-signal
    Intel confidence — into a single calibrated P(true). The harvested panel does
    not carry Priority_*_pct (compute_priority is cross-sectional and only runs
    live), so we materialize it here with a per-date pass using the just-tuned
    ``weights``. Features come from pe.meta_conf_features so train and inference
    are identical. Split chronologically by date.

    Probation: the returned model is ``active`` only if its out-of-sample rank-IR
    BEAT naked Priority's rank-IR (and is positive). An inactive model is kept
    for annotation but must not drive Hide/filter actions. Returns a model dict
    {coef, intercept, feat_mean, feat_std, feature_names, deadband, horizons,
    val_auc, meta_val_ir, priority_val_ir, active, n_train, n_val, base_rate}.
    """
    if ts_df is None or ts_df.empty or not {'Date', 'SignalType'}.issubset(ts_df.columns):
        return None
    df = ts_df.copy()
    df['Date'] = pd.to_datetime(df['Date'])

    ret_cols = [f'Ret_{h}b' for h in pe.HOLD_HORIZONS if f'Ret_{h}b' in df.columns]
    if not ret_cols:
        return None
    horizons_used = [int(c[4:-1]) for c in ret_cols]

    # ── Materialize cross-sectional Priority rank per date (the orthogonal input) ──
    # Explicit per-date loop (not groupby.apply) so the Date column survives and we
    # avoid the grouping-column deprecation; compute_priority is cross-sectional.
    try:
        parts = [pe.compute_priority(g, weights=weights)
                 for _, g in df.groupby('Date', sort=False)]
        df = pd.concat(parts) if parts else df
    except Exception:
        return None
    if 'Priority_Long_pct' not in df.columns:
        return None

    # Intel_Confidence should already be on the harvest panel; compute if not.
    if 'Intel_Confidence' not in df.columns:
        try:
            df = pe.compute_signal_confidence(
                df, weights=weights, conf_model=pe.get_active_conf_model(),
                compute_flags=False)
        except Exception:
            return None

    X_all, dir_sign, fired = pe.meta_conf_features(df)
    d = np.nan_to_num(dir_sign, nan=0.0)

    # Direction-signed forward returns (good long OR short → positive).
    R = df[ret_cols].to_numpy(dtype=float) * d[:, None]
    with np.errstate(invalid='ignore'):
        dr = np.nanmean(R, axis=1)
    fired = np.asarray(fired) & np.isfinite(dr)
    if int(fired.sum()) < min_total_samples:
        return None

    typ = float(np.median(np.abs(dr[fired]))) if fired.any() else 0.0
    deadband = deadband_frac * typ
    y_all = (dr > deadband).astype(float)

    dates = df['Date'].to_numpy()
    fired_dates = np.unique(dates[fired])
    if len(fired_dates) < 8:
        return None
    n_train = max(1, int(len(fired_dates) * train_frac))
    is_train = np.isin(dates, fired_dates[:n_train])
    tr = fired & is_train
    va = fired & ~is_train
    if (int(tr.sum()) < min_total_samples
            or y_all[tr].sum() < 5 or (tr.sum() - y_all[tr].sum()) < 5):
        return None

    Xtr = X_all[tr]
    mean = Xtr.mean(axis=0)
    std = Xtr.std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    Xz_tr = (Xtr - mean) / std
    coef, b0 = _fit_logistic(Xz_tr, y_all[tr], l2=l2)

    model = {
        'version': 1,
        'feature_names': list(pe.META_FEATURES),
        'feat_mean': [float(v) for v in mean],
        'feat_std': [float(v) for v in std],
        'coef': [float(c) for c in coef],
        'intercept': float(b0),
        'deadband': float(deadband),
        'horizons': horizons_used,
        'label': 'mean directional return over horizons > deadband',
        'base_rate': float(y_all[tr].mean()),
        'n_train': int(tr.sum()),
        'active': False,
    }

    # ── Probation: meta OOS rank-IR vs naked Priority OOS rank-IR ──
    if int(va.sum()) >= max(30, min_xsect):
        proba = pe.predict_meta_conviction(df, model)
        meta_scores = proba[va]
        prio_scores = X_all[va, pe.META_FEATURES.index('rank_pct')]
        Rva = R[va]
        dv = dates[va]
        ok = np.isfinite(meta_scores)
        if int(ok.sum()) >= max(30, min_xsect):
            meta_ir = _spearman_ir(dv[ok], meta_scores[ok], Rva[ok], min_n=min_xsect)
            prio_ir = _spearman_ir(dv[ok], prio_scores[ok], Rva[ok], min_n=min_xsect)
            yv = y_all[va][ok]
            model['val_auc'] = (_auc(yv, meta_scores[ok])
                                if 0 < yv.sum() < len(yv) else float('nan'))
            model['meta_val_ir'] = meta_ir
            model['priority_val_ir'] = prio_ir
            model['n_val'] = int(ok.sum())
            # Filter/reorder ONLY if the meta layer beat naked priority OOS + adds edge.
            model['active'] = bool(np.isfinite(meta_ir) and np.isfinite(prio_ir)
                                   and meta_ir > prio_ir and meta_ir > 0.0)

    return model
