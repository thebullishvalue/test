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
    'D: Long': 3, 'D: Short': 3,
}
_TIER_DEFAULT_IDX = 4


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
        self.M = np.column_stack([f1, f2, f3, f4, f5])

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
    ], dtype=np.float64)
    w_beta_short = np.array([
        weights['beta_F1_pricemom_short'],
        weights['beta_F2_volqual_short'],
        weights['beta_F3_wave_short'],
        weights['beta_F4_pulse_short'],
        weights['beta_F5_regime_short'],
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
        weights['tier_D_mult'],
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
                 hold_periods=(2, 3, 5, 8, 13),
                 train_frac: float = 0.70,
                 l2_alpha: float = 0.001,
                 min_xsect: int = 5):
        self.hold_periods = list(hold_periods)
        self.train_frac   = train_frac
        self.l2_alpha     = l2_alpha
        self.min_xsect    = min_xsect
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
                'gamma_reversion_long':   trial.suggest_float('gamma_reversion_long',  0.0, 40.0),
                'gamma_divergence_long':  trial.suggest_float('gamma_divergence_long', 0.0, 40.0),
                # Short-side factor weights
                'beta_F1_pricemom_short': trial.suggest_float('beta_F1_pricemom_short', 0.0, 40.0),
                'beta_F2_volqual_short':  trial.suggest_float('beta_F2_volqual_short',  0.0, 30.0),
                'beta_F3_wave_short':     trial.suggest_float('beta_F3_wave_short',     0.0, 30.0),
                'beta_F4_pulse_short':    trial.suggest_float('beta_F4_pulse_short',    0.0, 40.0),
                'beta_F5_regime_short':   trial.suggest_float('beta_F5_regime_short',   0.0, 50.0),
                'beta_F6_xsect_short':    trial.suggest_float('beta_F6_xsect_short',    0.0, 40.0),
                'gamma_reversion_short':  trial.suggest_float('gamma_reversion_short',  0.0, 40.0),
                'gamma_divergence_short': trial.suggest_float('gamma_divergence_short', 0.0, 40.0),
                # Shared tier multipliers
                'tier_A_mult':       trial.suggest_float('tier_A_mult',       0.5, 2.0),
                'tier_B_mult':       trial.suggest_float('tier_B_mult',       0.5, 2.0),
                'tier_C_mult':       trial.suggest_float('tier_C_mult',       0.5, 2.0),
                'tier_D_mult':       trial.suggest_float('tier_D_mult',       0.5, 2.0),
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

        self.study = optuna.create_study(direction='maximize')
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
