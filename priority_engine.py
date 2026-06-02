"""
Sanket Priority Engine — Asymmetric Orthogonal Composition (vectorized).

Composes Priority from six factors (F1–F6) plus reversion / divergence
penalties, with tier and regime damping. Output is in basis-point-equivalent
units of forward return.

Asymmetric: each β and γ has separate `_long` and `_short` variants. The
optimizer can learn that, e.g., Pulse predicts long returns more strongly
than short returns. Tier multipliers (signal-class quality) stay shared
since they are direction-agnostic. Legacy v1 profiles (single symmetric
weight per factor) auto-migrate by fan-out — same value to both sides.

Profiles are persisted per universe key (`<universe> · <selected_index>`)
in ``~/.sanket/profiles.json`` so switching universes auto-loads the
matching profile rather than silently re-using a stale one.
"""
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

# Optional cross-process file lock for the shared profiles JSON. On multi-user
# deployments (Streamlit Cloud) two simultaneous calibrations would otherwise
# read-modify-write the same file and clobber each other (lost-update race).
# Degrades to a no-op if filelock isn't installed, so the app never hard-fails on
# a missing optional dependency — it just loses the cross-process guarantee.
try:
    from filelock import FileLock, Timeout as _LockTimeout
    _HAVE_FILELOCK = True
except Exception:
    _HAVE_FILELOCK = False

VOL_REGIME_W   = {'LOW': 1.20, 'NORMAL': 1.00, 'HIGH': 0.85, 'EXTREME': 0.55}
VOL_REGIME_IDX = {'LOW': 0,    'NORMAL': 1,    'HIGH': 2,    'EXTREME': 3}

# Fibonacci-spaced forward-return horizons (bars) used by the calibration engine
# and the timeseries harvest loop.  Single source of truth shared by sanket.py
# and intelligence.py (both import priority_engine).
HOLD_HORIZONS = [2, 3, 5, 8, 13]

# Asymmetric defaults — symmetric values to start; the optimizer finds the asymmetry.
DEFAULT_W = {
    # Factor weights, per side
    'beta_F1_pricemom_long':   15.0,   'beta_F1_pricemom_short':   15.0,
    'beta_F2_volqual_long':     8.0,   'beta_F2_volqual_short':     8.0,
    'beta_F3_wave_long':       10.0,   'beta_F3_wave_short':       10.0,
    'beta_F4_pulse_long':      12.0,   'beta_F4_pulse_short':      12.0,
    'beta_F5_regime_long':     18.0,   'beta_F5_regime_short':     18.0,
    'beta_F6_xsect_long':      12.0,   'beta_F6_xsect_short':      12.0,
    # F7 = LO range-extension (reversion/liquidity). Default 0 → uncalibrated
    # profiles ignore it; the optimizer (range allows negative) learns its sign.
    'beta_F7_liq_long':         0.0,   'beta_F7_liq_short':         0.0,
    # Penalty weights, per side
    'gamma_reversion_long':    20.0,   'gamma_reversion_short':    20.0,
    'gamma_divergence_long':   18.0,   'gamma_divergence_short':   18.0,
    # Tier multipliers (signal-class quality) — direction-agnostic
    'tier_A_mult':              1.00,
    'tier_B_mult':              1.30,
    'tier_C_mult':              0.85,
    'tier_default_mult':        0.90,
}

# v1 → v2 key migration. Old profiles used a single symmetric value per factor;
# we fan that value out to both _long and _short.
_LEGACY_KEY_MAP = {
    'beta_F1_pricemom':   ('beta_F1_pricemom_long',   'beta_F1_pricemom_short'),
    'beta_F2_volqual':    ('beta_F2_volqual_long',    'beta_F2_volqual_short'),
    'beta_F3_wave':       ('beta_F3_wave_long',       'beta_F3_wave_short'),
    'beta_F4_pulse':      ('beta_F4_pulse_long',      'beta_F4_pulse_short'),
    'beta_F5_regime':     ('beta_F5_regime_long',     'beta_F5_regime_short'),
    'beta_F6_xsect':      ('beta_F6_xsect_long',      'beta_F6_xsect_short'),
    'gamma_reversion':    ('gamma_reversion_long',    'gamma_reversion_short'),
    'gamma_divergence':   ('gamma_divergence_long',   'gamma_divergence_short'),
}


def _migrate_legacy_keys(weights: dict) -> dict:
    """Fan out v1 symmetric keys to v2 _long/_short pairs."""
    out = dict(weights)
    for old_key, new_keys in _LEGACY_KEY_MAP.items():
        if old_key in out and not any(nk in out for nk in new_keys):
            v = out.pop(old_key)
            for nk in new_keys:
                out[nk] = v
        else:
            out.pop(old_key, None)  # superseded — drop the legacy key
    return out


active_W = DEFAULT_W.copy()


def set_active_weights(weights_dict: dict):
    """Set global active weights, with v1 → v2 migration + DEFAULT_W backfill."""
    global active_W
    migrated = _migrate_legacy_keys(weights_dict)
    active_W = {**DEFAULT_W, **migrated}


def get_active_weights():
    return active_W.copy()


# ──────────────────────────────────────────────────────────────────────
# Profile persistence — per-universe, single JSON file
#
# File layout: { "<universe_key>": <opt_results_dict>, ... }
# ``<universe_key>`` = "Universe · SelectedIndex" (e.g., "India Indexes · NIFTY 50").
#
# Best-effort durability: survives Python process restarts within a deployment.
# On Streamlit Cloud container rebuilds (code pushes) the home dir is reset,
# so the JSON Export from the sidebar Passport remains the only channel
# guaranteed to outlive a redeploy.
# ──────────────────────────────────────────────────────────────────────
PROFILES_PATH = Path.home() / ".sanket" / "profiles.json"
_LEGACY_SINGLE_PATH = Path.home() / ".sanket" / "profile.json"
_PROFILES_LOCK_PATH = Path.home() / ".sanket" / "profiles.json.lock"


@contextmanager
def _profiles_lock(timeout: float = 10.0):
    """Hold an exclusive lock around a profiles read-modify-write.

    No-op when filelock is unavailable or the lock can't be acquired in time —
    best-effort, never blocks the UI indefinitely or crashes on a missing dep.
    """
    if not _HAVE_FILELOCK:
        yield
        return
    lock = None
    try:
        _PROFILES_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(_PROFILES_LOCK_PATH), timeout=timeout)
        lock.acquire()
    except Exception:
        lock = None   # couldn't lock (timeout/FS issue) — proceed best-effort
    try:
        yield
    finally:
        if lock is not None:
            try:
                lock.release()
            except Exception:
                pass


def _profile_key(universe, selected_index, timeframe=None) -> str:
    """Stable string key combining universe + index + timeframe."""
    if not universe:
        return "—"
    parts = [str(universe)]
    if selected_index:
        parts.append(str(selected_index))
    if timeframe:
        parts.append(str(timeframe).lower())
    return " · ".join(parts)


def _profiles_load_all() -> dict:
    """Return all stored profiles ({key: opt_results}). Empty dict if missing/invalid."""
    try:
        if not PROFILES_PATH.exists():
            return {}
        with open(PROFILES_PATH) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _profiles_save_all(profiles: dict) -> bool:
    # Atomic write: serialize to a sibling temp file, then os-replace into place.
    # Prevents a crash/concurrent-read mid-write from leaving a truncated JSON.
    try:
        PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = PROFILES_PATH.with_suffix('.json.tmp')
        with open(tmp_path, "w") as f:
            json.dump(profiles, f, indent=2, default=str)
        tmp_path.replace(PROFILES_PATH)
        return True
    except Exception:
        return False


def _maybe_migrate_legacy_profile():
    """One-shot migration of the v1 single-profile file to the v2 dict-of-profiles file."""
    if PROFILES_PATH.exists() or not _LEGACY_SINGLE_PATH.exists():
        return
    try:
        with open(_LEGACY_SINGLE_PATH) as f:
            old = json.load(f)
        if not isinstance(old, dict) or "weights" not in old:
            return
        key = _profile_key(old.get("universe"), old.get("selected_index"))
        if _profiles_save_all({key: old}):
            try:
                _LEGACY_SINGLE_PATH.unlink()
            except Exception:
                pass
    except Exception:
        pass


# Run migration eagerly at import — idempotent if already done.
_maybe_migrate_legacy_profile()


def save_profile(opt_results: dict) -> bool:
    """Persist a calibration profile, keyed by universe + timeframe it was fit on."""
    universe       = opt_results.get("universe")
    selected_index = opt_results.get("selected_index")
    timeframe      = opt_results.get("timeframe")
    key = _profile_key(universe, selected_index, timeframe)
    # Lock the whole read-modify-write so a concurrent calibration on another
    # universe can't read the file before our write and then clobber our key.
    with _profiles_lock():
        profiles = _profiles_load_all()
        profiles[key] = {
            "weights":        opt_results.get("weights", {}),
            "train_score":    opt_results.get("train_score"),
            "val_score":      opt_results.get("val_score"),
            "sensitivity":    opt_results.get("sensitivity", {}),
            "signal_conf":    opt_results.get("signal_conf"),   # Layer 2 calibrated confidence model
            "meta_conviction": opt_results.get("meta_conviction"),  # Layer 3 meta-conviction model
            "timestamp":      opt_results.get("timestamp"),
            "universe":       universe,
            "selected_index": selected_index,
            "timeframe":      timeframe,
        }
        return _profiles_save_all(profiles)


def load_profile_for(universe, selected_index, timeframe=None):
    """Load the profile that was fit on the given universe + timeframe. None if missing."""
    if not universe:
        return None
    key = _profile_key(universe, selected_index, timeframe)
    profiles = _profiles_load_all()
    p = profiles.get(key)
    if p and isinstance(p, dict) and isinstance(p.get("weights"), dict):
        return p
    return None


def load_profile():
    """Return the most recently calibrated profile across all universes (best-effort).

    Used as a fallback when no universe context is available. Prefer
    ``load_profile_for(universe, selected_index)`` whenever possible.
    """
    profiles = _profiles_load_all()
    if not profiles:
        return None
    valid = [p for p in profiles.values()
             if isinstance(p, dict) and isinstance(p.get("weights"), dict)]
    if not valid:
        return None
    valid.sort(key=lambda p: p.get("timestamp") or "", reverse=True)
    return valid[0]


def delete_profile(universe=None, selected_index=None, timeframe=None) -> bool:
    """Remove a specific universe+timeframe profile. With ``universe=None``, wipes ALL profiles."""
    if universe is None:
        try:
            if PROFILES_PATH.exists():
                PROFILES_PATH.unlink()
            return True
        except Exception:
            return False

    key = _profile_key(universe, selected_index, timeframe)
    with _profiles_lock():
        profiles = _profiles_load_all()
        if key in profiles:
            del profiles[key]
            return _profiles_save_all(profiles)
    return True


def list_profiles() -> list:
    """List stored profile summaries — for UI directories of available calibrations."""
    profiles = _profiles_load_all()
    return [
        {
            "key":            k,
            "universe":       v.get("universe"),
            "selected_index": v.get("selected_index"),
            "timestamp":      v.get("timestamp"),
            "train_score":    v.get("train_score"),
            "val_score":      v.get("val_score"),
        }
        for k, v in profiles.items()
        if isinstance(v, dict)
    ]


# ──────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────
def _z_robust(s: pd.Series) -> pd.Series:
    med = s.median()
    mad = (s - med).abs().median() * 1.4826
    return (s - med) / max(mad, 1e-6)


def _col(df: pd.DataFrame, name: str, default):
    if name in df.columns:
        return df[name]
    return pd.Series(default, index=df.index)


def _tier_map(weights: dict) -> dict:
    a = weights['tier_A_mult']
    b = weights['tier_B_mult']
    c = weights['tier_C_mult']
    return {
        'A: Long': a, 'A: Short': a,
        'B: Long': b, 'B: Short': b,
        'C: Long': c, 'C: Short': c,
    }


# ──────────────────────────────────────────────────────────────────────
# compute_priority — vectorized, asymmetric per side
# ──────────────────────────────────────────────────────────────────────
def compute_priority(df: pd.DataFrame, weights=None) -> pd.DataFrame:
    """
    Vectorized asymmetric priority computation. Returns a copy of df with
    Priority_Long / Priority_Short / *_z / *_pct columns added, sorted by
    long-side tiebreak.

    Long and short use independent weight vectors (β_long, γ_long, β_short,
    γ_short). The optimizer in ``intelligence.py`` searches them jointly.
    """
    if df.empty:
        return df

    if weights is None:
        weights = active_W
    weights = _migrate_legacy_keys(weights)
    weights = {**DEFAULT_W, **weights}

    df = df.copy()

    # ── Factors ──
    F1   = _col(df, 'F1_PriceMom', 0.0).astype(float)
    F2   = _col(df, 'F2_VolQual',  0.0).astype(float)
    conv = _col(df, 'Conviction',  0.0).astype(float)
    F3   = conv / 20.0
    F4   = _col(df, 'Pulse', 0.0).astype(float)
    F5   = _col(df, 'HMM_Bull', 0.33).astype(float) - _col(df, 'HMM_Bear', 0.33).astype(float)
    F7   = _col(df, 'LO', 0.0).astype(float) / 100.0   # liquidity range-extension (reversion)

    wt1    = _col(df, 'Wave', 0.0).astype(float)
    travel = wt1 - _col(df, 'WT1_5ago', wt1).astype(float)

    # ── Reversion risk (per side) ──
    long_rev = pd.Series(
        np.where((wt1 > 60) & (travel < 0),
                 np.minimum(1.0, ((wt1 - 60) / 40.0) * (conv.abs() / 50.0)),
                 0.0),
        index=df.index)
    short_rev = pd.Series(
        np.where((wt1 < -60) & (travel > 0),
                 np.minimum(1.0, ((-wt1 - 60) / 40.0) * (conv.abs() / 50.0)),
                 0.0),
        index=df.index)

    # ── Divergence penalty ──
    bull_div = _col(df, 'Bullish_Div', False).astype(bool)
    bear_div = _col(df, 'Bearish_Div', False).astype(bool)
    long_div  = (bear_div & (conv > 30)).astype(float)
    short_div = (bull_div & (conv < -30)).astype(float)

    # ── Damping cascade (shared between sides — tier is direction-agnostic) ──
    vr_w     = _col(df, 'Vol_Regime', 'NORMAL').map(VOL_REGIME_W).fillna(1.0)
    tier_w   = _col(df, 'SignalType', '-').map(_tier_map(weights)).fillna(weights['tier_default_mult'])
    conf     = _col(df, 'Regime_Confidence', 0.5).astype(float).fillna(0.5)
    cp       = _col(df, 'Change_Point', False).astype(float).fillna(0.0)
    damp     = vr_w * tier_w * (0.6 + 0.4 * conf) * (1.0 - 0.35 * cp)

    # ── Inner score — asymmetric per side ──
    inner_long = (weights['beta_F1_pricemom_long'] * F1
                + weights['beta_F2_volqual_long']  * F2
                + weights['beta_F3_wave_long']     * F3
                + weights['beta_F4_pulse_long']    * F4
                + weights['beta_F5_regime_long']   * F5
                + weights['beta_F7_liq_long']      * F7
                - weights['gamma_reversion_long']  * long_rev
                - weights['gamma_divergence_long'] * long_div)
    inner_short = (weights['beta_F1_pricemom_short'] * (-F1)
                 + weights['beta_F2_volqual_short']  * (-F2)
                 + weights['beta_F3_wave_short']     * (-F3)
                 + weights['beta_F4_pulse_short']    * (-F4)
                 + weights['beta_F5_regime_short']   * (-F5)
                 + weights['beta_F7_liq_short']      * (-F7)
                 - weights['gamma_reversion_short']  * short_rev
                 - weights['gamma_divergence_short'] * short_div)

    # ── F6: rank of damped inner score within the cross-section [-1, +1] ──
    # A cross-sectional factor is meaningless with fewer than 2 names: rank(pct=True)
    # of a lone asset is 1.0 → F6 = +1.0, awarding maximum "outperformance" points for
    # outperforming nobody. Neutralize F6 to 0 when n < 2 so a single-asset screen
    # (e.g. an isolated correlation target) doesn't get an inflated baseline priority.
    pre_long  = inner_long  * damp
    pre_short = inner_short * damp
    if len(df) < 2:
        F6_long  = pd.Series(0.0, index=df.index)
        F6_short = pd.Series(0.0, index=df.index)
    else:
        F6_long  = (pre_long.rank(pct=True)  - 0.5) * 2.0
        F6_short = (pre_short.rank(pct=True) - 0.5) * 2.0

    # ── Final priority: F6 added inside the damping path (uniform treatment) ──
    df['Priority_Long']  = (inner_long  + weights['beta_F6_xsect_long']  * F6_long)  * damp
    df['Priority_Short'] = (inner_short + weights['beta_F6_xsect_short'] * F6_short) * damp

    for col in ('Priority_Long', 'Priority_Short'):
        df[col + '_z']   = _z_robust(df[col])
        df[col + '_pct'] = df[col].rank(pct=True) * 100

    # ── Tiebreaker chain ──
    vr_idx = _col(df, 'Vol_Regime', 'NORMAL').map(VOL_REGIME_IDX).fillna(1)
    df['_tb_long'] = list(zip(
        -df['Priority_Long'],
        -conf,
        vr_idx,
        -F1.abs(),
    ))
    df['_tb_short'] = list(zip(
        -df['Priority_Short'],
        -conf,
        vr_idx,
        -F1.abs(),
    ))

    df = df.sort_values('_tb_long', ascending=True)
    return df


# ──────────────────────────────────────────────────────────────────────
# Intelligence Confirmation — per-signal confidence (Layer 1)
#
# The signal sets (A/B/C) fire as pure boolean crossings, blind to regime.
# This scores each *fired* signal in [0, 1] by re-using the regime
# intelligence that compute_priority already consumes for the cross-sectional
# damping cascade — but here it is applied per signal and direction-aware, so a
# bull-cross fired into a BEAR / EXTREME-vol / change-point context scores low.
#
# Score = evidence × trust, both in [0, 1]:
#   evidence (corroboration, can push high or low)
#     • regime alignment : HMM_Bull−HMM_Bear (long) mapped to [0, 1]
#     • rank agreement   : Priority_<side>_pct / 100 — does the calibrated
#                          rank engine agree with the boolean?
#   trust (multipliers, can only reduce)
#     • vol regime quality (VOL_REGIME_W normalized)
#     • regime confidence (0.6 + 0.4·conf)
#     • change-point      (×0.65 on a detected break)
#     • reversion risk    (directional, mirrors compute_priority)
#     • divergence contradiction (bear-div under a long, etc.)
#
# Non-fired rows (Zone labels / '-') get NaN — they are not signals.
# ──────────────────────────────────────────────────────────────────────
_SIG_DIR = {
    'A: Long': 1,  'B: Long': 1,  'C: Long': 1,
    'A: Short': -1, 'B: Short': -1, 'C: Short': -1,
}

# Map SignalType → set letter, for per-set calibrated models (Layer 2).
_SIG_SET = {
    'A: Long': 'A', 'A: Short': 'A',
    'B: Long': 'B', 'B: Short': 'B',
    'C: Long': 'C', 'C: Short': 'C',
}

# Directional feature names for the signal-confidence logistic (Layer 2).
# All features are signed by signal direction so a single coefficient vector
# serves both long and short (a supportive context is always positive).
CONF_FEATURES = [
    'hmm_align',    # dir·(HMM_Bull − HMM_Bear) — regime corroboration
    'vol_quality',  # vol-regime weight (LOW high → calm = trustworthy)
    'regime_conf',  # HMM confidence
    'change_point', # 1.0 on a detected regime break (noise)
    'reversion',    # directional reversion-exhaustion risk [0,1]
    'div_contra',   # 1.0 when an opposite-side divergence contradicts the signal
    'mom_align',    # dir·F1_PriceMom
    'conv_align',   # dir·Conviction/50
    'pulse_align',  # dir·Pulse
    'liq_support',  # dir·Liquidity_Osc/100 — microstructure liquidity backing the move
    'liq_exhaust',  # LO range-extension against the signal [0,1] — liquidity exhaustion risk
]


def signal_conf_features(df: pd.DataFrame):
    """Build the directional feature matrix for the signal-confidence model.

    Shared by the calibrator (intelligence.py, on harvested bars) and the
    apply path (compute_signal_confidence, on the live cross-section) so train
    and inference see identical features. Returns (X, dir_sign, set_letter,
    fired_mask) where X is (n, len(CONF_FEATURES)).
    """
    sig = _col(df, 'SignalType', '-')
    dir_sign  = sig.map(_SIG_DIR).to_numpy(dtype=float)      # +1/-1/NaN
    set_letter = sig.map(_SIG_SET).to_numpy(dtype=object)    # 'A'/'B'/'C'/NaN
    fired = ~pd.isna(dir_sign)
    d = np.nan_to_num(dir_sign, nan=0.0)

    hmm_bull = _col(df, 'HMM_Bull', 0.33).astype(float).to_numpy()
    hmm_bear = _col(df, 'HMM_Bear', 0.33).astype(float).to_numpy()
    hmm_align = d * (hmm_bull - hmm_bear)

    vr_w = _col(df, 'Vol_Regime', 'NORMAL').map(VOL_REGIME_W).fillna(1.0).astype(float).to_numpy()

    regime_conf = _col(df, 'Regime_Confidence', 0.5).astype(float).fillna(0.5).to_numpy()
    change_point = _col(df, 'Change_Point', False).astype(bool).to_numpy().astype(float)

    wt1 = _col(df, 'Wave', np.nan).astype(float)
    if wt1.isna().all():
        wt1 = _col(df, 'WT1', 0.0).astype(float)
    wt1 = wt1.to_numpy()
    conv = _col(df, 'Conviction', 0.0).astype(float).to_numpy()
    wt1_5ago = _col(df, 'WT1_5ago', np.nan).astype(float)
    wt1_5ago = np.where(np.isnan(wt1_5ago.to_numpy()), wt1, wt1_5ago.to_numpy())
    travel = wt1 - wt1_5ago
    long_rev = np.where((wt1 > 60) & (travel < 0),
                        np.minimum(1.0, ((wt1 - 60) / 40.0) * (np.abs(conv) / 50.0)), 0.0)
    short_rev = np.where((wt1 < -60) & (travel > 0),
                         np.minimum(1.0, ((-wt1 - 60) / 40.0) * (np.abs(conv) / 50.0)), 0.0)
    reversion = np.where(d > 0, long_rev, np.where(d < 0, short_rev, 0.0))

    bull_div = _col(df, 'Bullish_Div', False).astype(bool).to_numpy()
    bear_div = _col(df, 'Bearish_Div', False).astype(bool).to_numpy()
    div_contra = np.where(d > 0, bear_div & (conv > 30),
                 np.where(d < 0, bull_div & (conv < -30), False)).astype(float)

    f1 = _col(df, 'F1_PriceMom', 0.0).astype(float).to_numpy()
    pulse = _col(df, 'Pulse', 0.0).astype(float).to_numpy()
    mom_align  = d * f1
    conv_align = d * conv / 50.0
    pulse_align = d * pulse

    # ── Liquidity (LO) — two faces ──
    # Support: signed microstructure liquidity in the signal's direction (+ = backed).
    liq_osc = _col(df, 'Liquidity_Osc', 0.0).astype(float).fillna(0.0).to_numpy()
    liq_support = d * liq_osc / 100.0
    # Exhaustion: how far the LO range-stochastic is stretched into the signal's
    # direction (a fresh long into an already-topped LO = chasing). [0,1].
    lo = _col(df, 'LO', 0.0).astype(float).fillna(0.0).to_numpy()
    liq_exhaust = np.clip(d * lo / 100.0, 0.0, 1.0)

    X = np.column_stack([
        hmm_align, vr_w, regime_conf, change_point,
        reversion, div_contra, mom_align, conv_align, pulse_align,
        liq_support, liq_exhaust,
    ])
    return X, dir_sign, set_letter, fired


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))


def predict_signal_confidence(df: pd.DataFrame, model: dict) -> np.ndarray:
    """Calibrated P(true) per row from a fitted signal-confidence model.

    Returns an array aligned to df.index; NaN where no fired signal or no
    usable per-set/pooled model. Standardization + per-set coefficients (with
    pooled fallback) are read from ``model``.
    """
    X, dir_sign, set_letter, fired = signal_conf_features(df)
    n = len(df)
    out = np.full(n, np.nan)
    if not model or 'sets' not in model:
        return out
    # Align the live feature matrix to the model's stored feature order by NAME,
    # so a model trained on an older/smaller feature set (e.g. before liq_* were
    # added) still scores correctly. If the model expects a feature we no longer
    # build, fall back to heuristic (return all-NaN).
    model_names = model.get('feature_names', CONF_FEATURES)
    try:
        idx = [CONF_FEATURES.index(fn) for fn in model_names]
    except ValueError:
        return out
    Xsel = X[:, idx]
    mean = np.asarray(model.get('feat_mean', np.zeros(Xsel.shape[1])), dtype=float)
    std  = np.asarray(model.get('feat_std',  np.ones(Xsel.shape[1])),  dtype=float)
    std  = np.where(std < 1e-9, 1.0, std)
    Xz = (Xsel - mean) / std
    sets = model['sets']
    pooled = sets.get('_pooled')
    for i in range(n):
        if not fired[i]:
            continue
        m = sets.get(set_letter[i]) or pooled
        if not m:
            continue
        beta = np.asarray(m['coef'], dtype=float)
        b0 = float(m['intercept'])
        out[i] = _sigmoid(b0 + Xz[i] @ beta)
    return out


# Active calibrated signal-confidence model (Layer 2), parallel to active_W.
active_conf_model = None


def set_active_conf_model(model):
    """Install the calibrated signal-confidence model (or None to clear)."""
    global active_conf_model
    active_conf_model = model if (model and isinstance(model, dict)) else None


def get_active_conf_model():
    return active_conf_model


def compute_signal_confidence(df: pd.DataFrame, weights=None, conf_model='__active__',
                              compute_flags: bool = True) -> pd.DataFrame:
    """Add Intel_Confidence (0–1), Intel_Stars (1–5), Intel_Flags (str) per row.

    Fully per-symbol: every term is read off the symbol's own regime/momentum
    state — no cross-sectional ranking against peers. Vectorized; returns a copy.
    Rows whose SignalType is not a fired A/B/C signal get NaN confidence.

    Two-tier score:
      • Heuristic (Layer 1): regime alignment × own-factor agreement × trust,
        always available, no training required.
      • Calibrated (Layer 2): when a fitted ``conf_model`` is present, the
        per-signal probability P(true) learned on harvested outcomes replaces
        the heuristic for sets the model covers. ``conf_model='__active__'``
        uses the module-global model; pass None to force pure heuristic.
    The Intel_Flags diagnostics are computed the same way in both modes.
    """
    if df.empty:
        return df
    df = df.copy()

    sig = _col(df, 'SignalType', '-')
    direction = sig.map(_SIG_DIR)            # +1 long, -1 short, NaN otherwise
    is_long  = (direction == 1)
    is_short = (direction == -1)
    fired    = direction.notna()

    # ── Evidence 1: regime alignment, direction-aware → [0, 1] ──
    hmm_bull = _col(df, 'HMM_Bull', 0.33).astype(float)
    hmm_bear = _col(df, 'HMM_Bear', 0.33).astype(float)
    hmm_dir  = np.where(is_long, hmm_bull - hmm_bear,
               np.where(is_short, hmm_bear - hmm_bull, 0.0))     # [-1, 1]
    align01  = np.clip((hmm_dir + 1.0) / 2.0, 0.0, 1.0)

    # ── Evidence 2: per-symbol directional factor agreement → [0, 1] ──
    # Purely local — no cross-sectional rank. Asks whether the symbol's OWN
    # momentum and conviction lean the way the signal points. Squashed with tanh
    # so it is bounded and neutral (0.5) when the symbol has no directional lean.
    f1     = _col(df, 'F1_PriceMom', 0.0).astype(float).fillna(0.0)
    conv   = _col(df, 'Conviction', 0.0).astype(float).fillna(0.0)
    d_sign = np.where(is_long, 1.0, np.where(is_short, -1.0, 0.0))
    factor_lean = d_sign * (0.5 * f1 + 0.5 * conv / 50.0)
    local01 = 0.5 * (np.tanh(factor_lean) + 1.0)

    # ── Evidence 3: liquidity support — is the microstructure backing the move? ──
    liq_osc = _col(df, 'Liquidity_Osc', 0.0).astype(float).fillna(0.0).to_numpy()
    liq01 = 0.5 * (np.tanh(d_sign * liq_osc / 50.0) + 1.0)   # [0,1], 0.5 = neutral

    evidence = 0.40 * align01 + 0.35 * local01 + 0.25 * liq01

    # ── Trust 1: vol regime quality (normalized so LOW=1.0) ──
    vr_w = _col(df, 'Vol_Regime', 'NORMAL').map(VOL_REGIME_W).fillna(1.0).astype(float)
    vq   = vr_w / max(VOL_REGIME_W.values())

    # ── Trust 2: regime confidence ──
    conf = _col(df, 'Regime_Confidence', 0.5).astype(float).fillna(0.5)
    cf   = 0.6 + 0.4 * conf

    # ── Trust 3: change-point ──
    cp       = _col(df, 'Change_Point', False).astype(bool)
    cp_mult  = np.where(cp, 0.65, 1.0)

    # ── Trust 4: reversion risk (directional, mirrors compute_priority) ──
    wt1    = _col(df, 'Wave', 0.0).astype(float)
    if 'Wave' not in df.columns:                 # screener rows store WT1 under 'Wave'
        wt1 = _col(df, 'WT1', 0.0).astype(float)
    travel = wt1 - _col(df, 'WT1_5ago', wt1).astype(float)
    long_rev = np.where((wt1 > 60) & (travel < 0),
                        np.minimum(1.0, ((wt1 - 60) / 40.0) * (conv.abs() / 50.0)), 0.0)
    short_rev = np.where((wt1 < -60) & (travel > 0),
                         np.minimum(1.0, ((-wt1 - 60) / 40.0) * (conv.abs() / 50.0)), 0.0)
    rev      = np.where(is_long, long_rev, np.where(is_short, short_rev, 0.0))
    rev_mult = 1.0 - rev

    # ── Trust 5: divergence contradiction (signal vs opposite-side divergence) ──
    bull_div = _col(df, 'Bullish_Div', False).astype(bool)
    bear_div = _col(df, 'Bearish_Div', False).astype(bool)
    div_contra = np.where(is_long,  bear_div & (conv > 30),
                 np.where(is_short, bull_div & (conv < -30), False)).astype(float)
    div_mult   = 1.0 - 0.5 * div_contra

    # ── Trust 6: liquidity exhaustion — LO stretched into the signal's direction ──
    lo = _col(df, 'LO', 0.0).astype(float).fillna(0.0)
    d_arr = np.where(is_long, 1.0, np.where(is_short, -1.0, 0.0))
    liq_exhaust = np.clip(d_arr * lo.to_numpy() / 100.0, 0.0, 1.0)
    liq_mult = 1.0 - 0.4 * liq_exhaust

    trust = vq * cf * cp_mult * rev_mult * div_mult * liq_mult
    score = np.clip(evidence * trust, 0.0, 1.0)

    # Layer 2: where a calibrated model covers the signal's set, use its
    # learned probability instead of the heuristic. Falls back to heuristic
    # per-row when the model lacks that set.
    model = get_active_conf_model() if conf_model == '__active__' else conf_model
    have = np.zeros(len(df), dtype=bool)
    if model:
        cal = predict_signal_confidence(df, model)
        have = ~np.isnan(cal)
        score = np.where(have, cal, score)

    fired_arr = np.asarray(fired)
    df['Intel_Confidence'] = np.where(fired_arr, score, np.nan)
    df['Intel_Source'] = np.where(have, 'calibrated',
                                  np.where(fired_arr, 'heuristic', ''))

    # 1–5 stars on fixed bands so the rating is comparable across runs.
    bands = np.array([0.20, 0.35, 0.50, 0.65])
    stars = 1 + np.digitize(np.where(fired, score, -1.0), bands)
    df['Intel_Stars'] = np.where(fired, stars, 0).astype(int)

    # ── Flags: the dominant contradictions, for spotting likely false positives ──
    if compute_flags:
        flag_specs = [
            (np.asarray(is_long) & (np.asarray(hmm_dir) < -0.1), 'bear-regime'),
            (np.asarray(is_short) & (np.asarray(hmm_dir) < -0.1), 'bull-regime'),
            (vr_w.to_numpy() <= VOL_REGIME_W['EXTREME'] + 1e-9, 'extreme-vol'),
            (cp.to_numpy(), 'change-pt'),
            (rev > 0.25, 'rev-risk'),
            (div_contra > 0, 'div-contra'),
            (np.asarray(local01) < 0.35, 'mom-against'),
            (liq_exhaust > 0.5, 'liq-exhaust'),
            (np.asarray(liq01) < 0.35, 'liq-against'),
        ]
        flags = []
        for r in range(len(df)):
            if not bool(fired.iloc[r]):
                flags.append('')
                continue
            row_flags = [label for mask, label in flag_specs if bool(mask[r])]
            flags.append(' · '.join(row_flags))
        df['Intel_Flags'] = flags

    return df


def signal_confidence_at(window_df: pd.DataFrame, side: str, set_letter: str,
                         weights=None, conf_model='__active__'):
    """Per-bar Intel confidence over a recent window, for a fixed side + set.

    Forces every bar in ``window_df`` to be treated as a fired ``set_letter`` /
    ``side`` signal, so confidence can be read at the bar a signal actually fired
    (its *fire bar*) rather than only at the snapshot date. This keeps the score
    consistent with how the Layer-2 model was trained (features AT the fire bar).

    Returns (conf_array, source_array) aligned to ``window_df`` rows in their
    given (chronological) order. The window must carry the per-bar feature
    columns (HMM_*, Vol_Regime, Regime_Confidence, Change_Point, *_Div, WT1,
    WT1_5ago, Conviction, F1_PriceMom, Pulse).
    """
    if window_df is None or len(window_df) == 0:
        return np.array([]), np.array([], dtype=object)
    d = window_df.copy()
    side_word = 'Long' if side == 'long' else 'Short'
    d['SignalType'] = f"{(set_letter or 'A')}: {side_word}"
    scored = compute_signal_confidence(d, weights=weights, conf_model=conf_model,
                                       compute_flags=False)
    return (scored['Intel_Confidence'].to_numpy(),
            scored['Intel_Source'].to_numpy())


# ══════════════════════════════════════════════════════════════════════════
# Layer 3 — Meta-Conviction
#
# The final intelligence layer. Layers 1/2 (compute_signal_confidence) score a
# fired signal *per symbol* — its own regime / momentum fit, blind to peers.
# The Priority Engine (compute_priority) is the *cross-sectional* view — where a
# name ranks within today's universe. These are informationally orthogonal, and
# the old Layer-3 collapsed them to a single user threshold on Intel_Confidence,
# discarding the rank and the agreement between the two.
#
# Layer 3 fuses them. On the assembled cross-section it builds a small feature
# vector per fired signal — cross-sectional rank percentile, per-signal
# confidence, their interaction, and whether the confidence is calibrated — and
# emits a final Conviction ∈ [0,1] plus a 0–3 Tier and a human reason.
#
#   • Calibrated (model active): a walk-forward-validated logistic P(true) that
#     is allowed to REORDER/FILTER only if it beat naked Priority ranking
#     out-of-sample (model['active'] == True). Same probation discipline as the
#     rest of the stack — it refuses to act on unproven edge.
#   • Fallback (no active model): Conviction = rank_pct × confidence, ADVISORY
#     only (annotate, never hide).
#   • Abstention: if the cross-section shows no spread in Conviction (no
#     differentiating information today), callers fall back to raw Priority
#     order. compute_conviction flags this via Conviction_Spread.
#
# Non-fired rows get NaN Conviction / tier 0 — they are not signals.
# ══════════════════════════════════════════════════════════════════════════

# Meta features, in fixed order. All available on BOTH the harvested panel
# (after a per-date compute_priority pass) and the live cross-section, so train
# and inference see identical inputs (the lesson that bit the per-set models).
META_FEATURES = ['rank_pct', 'conf', 'rank_x_conf', 'is_calibrated']


def meta_conf_features(df: pd.DataFrame):
    """Build the meta-conviction feature matrix for fired signals.

    Shared by ``calibrate_meta_conviction`` (intelligence.py, on harvested bars
    that already carry Priority_*_pct + Intel_Confidence) and the apply path
    (compute_conviction, on the live cross-section). Returns
    (X, dir_sign, fired_mask) where X is (n, len(META_FEATURES)).

    A row is ``fired`` only if it carries a fired A/B/C signal AND both a
    cross-sectional rank percentile and a confidence score (else the meta model
    has nothing orthogonal to fuse).
    """
    sig = _col(df, 'SignalType', '-')
    dir_sign = sig.map(_SIG_DIR).to_numpy(dtype=float)        # +1 long / -1 short / NaN
    is_long  = dir_sign == 1
    is_short = dir_sign == -1

    pl_pct = _col(df, 'Priority_Long_pct',  np.nan).astype(float).to_numpy()
    ps_pct = _col(df, 'Priority_Short_pct', np.nan).astype(float).to_numpy()
    rank_pct = np.where(is_long, pl_pct, np.where(is_short, ps_pct, np.nan)) / 100.0

    conf = _col(df, 'Intel_Confidence', np.nan).astype(float).to_numpy()
    src  = _col(df, 'Intel_Source', '').astype(str).to_numpy()
    is_cal = (src == 'calibrated').astype(float)

    fired = (~np.isnan(dir_sign)) & np.isfinite(rank_pct) & np.isfinite(conf)

    rp = np.nan_to_num(rank_pct, nan=0.5)
    cf = np.nan_to_num(conf,     nan=0.5)
    X = np.column_stack([rp, cf, rp * cf, is_cal])
    return X, dir_sign, fired


def predict_meta_conviction(df: pd.DataFrame, model: dict) -> np.ndarray:
    """Calibrated meta P(true) per row from a fitted meta-conviction model.

    Returns an array aligned to df.index; NaN where no fired signal or no usable
    model. Standardization + coefficients are read from ``model`` and aligned to
    the stored feature order by NAME, so an older/narrower model still scores.
    """
    X, _dir, fired = meta_conf_features(df)
    n = len(df)
    out = np.full(n, np.nan)
    if not model or 'coef' not in model:
        return out
    model_names = model.get('feature_names', META_FEATURES)
    try:
        idx = [META_FEATURES.index(fn) for fn in model_names]
    except ValueError:
        return out
    Xsel = X[:, idx]
    mean = np.asarray(model.get('feat_mean', np.zeros(Xsel.shape[1])), dtype=float)
    std  = np.asarray(model.get('feat_std',  np.ones(Xsel.shape[1])),  dtype=float)
    std  = np.where(std < 1e-9, 1.0, std)
    Xz = (Xsel - mean) / std
    beta = np.asarray(model['coef'], dtype=float)
    b0 = float(model.get('intercept', 0.0))
    scores = _sigmoid(b0 + Xz @ beta)
    out = np.where(fired, scores, np.nan)
    return out


# Active calibrated meta-conviction model (Layer 3), parallel to active_conf_model.
active_meta_model = None


def set_active_meta_model(model):
    """Install the calibrated meta-conviction model (or None to clear)."""
    global active_meta_model
    active_meta_model = model if (model and isinstance(model, dict)) else None


def get_active_meta_model():
    return active_meta_model


# Fixed Conviction → Tier bands, so a tier means the same thing across runs.
_CONV_BANDS = np.array([0.35, 0.55, 0.70])   # tiers 0 / 1 / 2 / 3


def compute_conviction(df: pd.DataFrame, meta_model='__active__',
                       spread_eps: float = 0.03) -> pd.DataFrame:
    """Add Meta_Conviction (0–1), Conviction_Tier (0–3), Conviction_Source,
    Conviction_Reason, Conviction_Active, Conviction_Spread per row.

    The scalar is Meta_Conviction, NOT 'Conviction' — the engine's structural
    Conviction (±100) already owns the 'Conviction' column and must not be touched.

    Requires Priority_*_pct (from compute_priority) and Intel_Confidence (from
    compute_signal_confidence) already on the frame. Fired rows the meta layer
    cannot fuse, and all non-fired rows, get NaN Conviction / tier 0.

      • Calibrated: when an ``active`` model is present, Conviction is its learned
        P(true). ``meta_model='__active__'`` uses the module-global model; pass
        None to force the fallback.
      • Fallback: rank_pct × confidence — advisory only.

    Conviction_Active mirrors the model's probation flag: True only when a model
    that beat naked Priority OOS is driving the score. Callers must restrict
    Hide/filter actions to Conviction_Active rows (advisory scores annotate only).
    Conviction_Spread is the cross-sectional std of Conviction over fired rows;
    near-zero means abstain (no differentiating information today).
    """
    if df.empty:
        return df
    df = df.copy()

    X, dir_sign, fired = meta_conf_features(df)
    fired_arr = np.asarray(fired)
    rank_pct = X[:, META_FEATURES.index('rank_pct')]
    conf     = X[:, META_FEATURES.index('conf')]

    model = get_active_meta_model() if meta_model == '__active__' else meta_model
    active = bool(model and model.get('active'))

    if model and 'coef' in model:
        cal = predict_meta_conviction(df, model)
        have = ~np.isnan(cal)
        # Fall back to the rank×conf product wherever the model didn't score a fired row.
        fb = np.clip(rank_pct * conf, 0.0, 1.0)
        score = np.where(have, cal, fb)
        source = np.where(have, 'meta', np.where(fired_arr, 'fallback', ''))
    else:
        score = np.clip(rank_pct * conf, 0.0, 1.0)
        source = np.where(fired_arr, 'fallback', '')

    # NOTE: the scalar is Meta_Conviction (NOT 'Conviction') — the engine's structural
    # Conviction (±100) already owns the 'Conviction' column upstream; reusing it here
    # clobbered it with NaN on non-fired rows.
    df['Meta_Conviction']   = np.where(fired_arr, np.clip(score, 0.0, 1.0), np.nan)
    df['Conviction_Source'] = source
    df['Conviction_Active'] = bool(active)

    tier = 1 + np.digitize(np.where(fired_arr, score, -1.0), _CONV_BANDS)
    df['Conviction_Tier'] = np.where(fired_arr, tier, 0).astype(int)

    # Abstention signal: spread of conviction across today's fired cross-section.
    fired_scores = score[fired_arr]
    spread = float(np.std(fired_scores)) if fired_scores.size >= 2 else 0.0
    df['Conviction_Spread'] = spread
    abstain = spread < spread_eps

    # Human reason — rank + confidence, plus any contradiction flags already on
    # the row (live screen carries Intel_Flags; the harvest panel does not).
    flags = (df['Intel_Flags'].astype(str).to_numpy()
             if 'Intel_Flags' in df.columns else np.array([''] * len(df), dtype=object))
    reasons = []
    for i in range(len(df)):
        if not fired_arr[i]:
            reasons.append('')
            continue
        parts = [f"rank {rank_pct[i]*100:.0f}%", f"conf {conf[i]*100:.0f}%"]
        if source[i] == 'meta':
            parts.append('meta' if active else 'meta·advisory')
        if abstain:
            parts.append('abstain (flat cross-section)')
        if flags[i]:
            parts.append(flags[i])
        reasons.append(' · '.join(parts))
    df['Conviction_Reason'] = reasons

    return df
