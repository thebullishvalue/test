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
from pathlib import Path

import numpy as np
import pandas as pd

VOL_REGIME_W   = {'LOW': 1.20, 'NORMAL': 1.00, 'HIGH': 0.85, 'EXTREME': 0.55}
VOL_REGIME_IDX = {'LOW': 0,    'NORMAL': 1,    'HIGH': 2,    'EXTREME': 3}

# Asymmetric defaults — symmetric values to start; the optimizer finds the asymmetry.
DEFAULT_W = {
    # Factor weights, per side
    'beta_F1_pricemom_long':   15.0,   'beta_F1_pricemom_short':   15.0,
    'beta_F2_volqual_long':     8.0,   'beta_F2_volqual_short':     8.0,
    'beta_F3_wave_long':       10.0,   'beta_F3_wave_short':       10.0,
    'beta_F4_pulse_long':      12.0,   'beta_F4_pulse_short':      12.0,
    'beta_F5_regime_long':     18.0,   'beta_F5_regime_short':     18.0,
    'beta_F6_xsect_long':      12.0,   'beta_F6_xsect_short':      12.0,
    # Penalty weights, per side
    'gamma_reversion_long':    20.0,   'gamma_reversion_short':    20.0,
    'gamma_divergence_long':   18.0,   'gamma_divergence_short':   18.0,
    # Tier multipliers (signal-class quality) — direction-agnostic
    'tier_A_mult':              1.00,
    'tier_B_mult':              1.30,
    'tier_C_mult':              0.85,
    'tier_D_mult':              0.75,
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


def _profile_key(universe, selected_index) -> str:
    """Stable string key combining universe + index."""
    if not universe:
        return "—"
    parts = [str(universe)]
    if selected_index:
        parts.append(str(selected_index))
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
    try:
        PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PROFILES_PATH, "w") as f:
            json.dump(profiles, f, indent=2, default=str)
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
    """Persist a calibration profile, keyed by the universe it was fit on."""
    universe       = opt_results.get("universe")
    selected_index = opt_results.get("selected_index")
    key = _profile_key(universe, selected_index)
    profiles = _profiles_load_all()
    profiles[key] = {
        "weights":        opt_results.get("weights", {}),
        "train_score":    opt_results.get("train_score"),
        "val_score":      opt_results.get("val_score"),
        "sensitivity":    opt_results.get("sensitivity", {}),
        "timestamp":      opt_results.get("timestamp"),
        "universe":       universe,
        "selected_index": selected_index,
    }
    return _profiles_save_all(profiles)


def load_profile_for(universe, selected_index):
    """Load the profile that was fit on the given universe selection. None if missing."""
    if not universe:
        return None
    key = _profile_key(universe, selected_index)
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


def delete_profile(universe=None, selected_index=None) -> bool:
    """Remove a specific universe's profile. With ``universe=None``, wipes ALL profiles."""
    if universe is None:
        try:
            if PROFILES_PATH.exists():
                PROFILES_PATH.unlink()
            return True
        except Exception:
            return False

    key = _profile_key(universe, selected_index)
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
    d = weights['tier_D_mult']
    return {
        'A: Long': a, 'A: Short': a,
        'B: Long': b, 'B: Short': b,
        'C: Long': c, 'C: Short': c,
        'D: Long': d, 'D: Short': d,
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
                - weights['gamma_reversion_long']  * long_rev
                - weights['gamma_divergence_long'] * long_div)
    inner_short = (weights['beta_F1_pricemom_short'] * (-F1)
                 + weights['beta_F2_volqual_short']  * (-F2)
                 + weights['beta_F3_wave_short']     * (-F3)
                 + weights['beta_F4_pulse_short']    * (-F4)
                 + weights['beta_F5_regime_short']   * (-F5)
                 - weights['gamma_reversion_short']  * short_rev
                 - weights['gamma_divergence_short'] * short_div)

    # ── F6: rank of damped inner score within the cross-section [-1, +1] ──
    pre_long  = inner_long  * damp
    pre_short = inner_short * damp
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
