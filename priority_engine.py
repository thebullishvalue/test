"""
Sanket Priority Engine — Orthogonal Additive Composition.

Replaces the multiplicative-redundant 8-layer model. Now composes Priority
from 6 maximally-independent factors with hand-picked but rational weights
(future: regression-calibrated against forward returns).

Public API unchanged: compute_priority(df) returns df with
Priority_Long, Priority_Short, *_z, *_pct columns.
"""
import numpy as np
import pandas as pd

VOL_REGIME_W   = {'LOW': 1.20, 'NORMAL': 1.00, 'HIGH': 0.85, 'EXTREME': 0.55}
VOL_REGIME_IDX = {'LOW': 0,    'NORMAL': 1,    'HIGH': 2,    'EXTREME': 3}
TIER_W         = {'B: Long': 1.30, 'B: Short': 1.30,
                  'A: Long': 1.00, 'A: Short': 1.00,
                  'C: Long': 0.85, 'C: Short': 0.85,
                  'D: Long': 0.75, 'D: Short': 0.75}

# Factor weights — output is in basis points of expected 5-bar forward return.
# These are hand-picked rational defaults. They can be later refined via OLS
# regression on logged predictions vs. realized 5-bar returns; nothing in the
# rest of the system depends on the magnitudes — only on the rank ordering.
W = {
    'beta_F1_pricemom':   15.0,   # raw price momentum (ATR-normalized 5-bar log return)
    'beta_F2_volqual':     8.0,   # signed-volume z-score, 5-bar smoothed
    'beta_F3_wave':       10.0,   # Conviction (oscillator structure) / 20
    'beta_F4_pulse':      12.0,   # Pulse (acceleration edge)
    'beta_F5_regime':     18.0,   # Joint-state HMM bull-prob differential
    'beta_F6_xsect':      12.0,   # cross-sectional rank
    'gamma_reversion':    20.0,   # subtractive: extended-zone mean-reversion risk
    'gamma_divergence':   18.0,   # subtractive: contra-divergence haircut
}


def _z_robust(s: pd.Series) -> pd.Series:
    med = s.median()
    mad = (s - med).abs().median() * 1.4826
    return (s - med) / max(mad, 1e-6)


def _reversion_risk(row, side):
    wt1 = row.get('Wave', 0)
    travel = wt1 - row.get('WT1_5ago', wt1)
    if side == 'long' and wt1 > 60 and travel < 0:
        return min(1.0, (wt1 - 60) / 40 * abs(row.get('Conviction', 0)) / 50)
    if side == 'short' and wt1 < -60 and travel > 0:
        return min(1.0, (-wt1 - 60) / 40 * abs(row.get('Conviction', 0)) / 50)
    return 0.0


def _divergence_pen(row, side):
    conv = row.get('Conviction', 0)
    if side == 'long' and row.get('Bearish_Div', False) and conv > 30:  return 1.0
    if side == 'short' and row.get('Bullish_Div', False) and conv < -30: return 1.0
    return 0.0


def _priority_one(row, side):
    """Returns Priority for one side, in basis-point-equivalent units."""
    sign = 1 if side == 'long' else -1

    # Six orthogonal factors (raw values — z-scoring happens cross-sectionally below)
    F1 = sign * row.get('F1_PriceMom', 0)
    F2 = sign * row.get('F2_VolQual', 0)
    F3 = sign * row.get('Conviction', 0) / 20.0     # rescale [-100, +100] → [-5, +5]
    F4 = sign * row.get('Pulse', 0)
    F5_long = row.get('HMM_Bull', 0.33) - row.get('HMM_Bear', 0.33)  # [-1, +1]
    F5 = F5_long if side == 'long' else -F5_long
    # F6 set later cross-sectionally; placeholder here, overwritten in compute_priority
    F6 = 0.0

    rev = _reversion_risk(row, side)
    div = _divergence_pen(row, side)

    raw = (W['beta_F1_pricemom'] * F1
         + W['beta_F2_volqual']  * F2
         + W['beta_F3_wave']     * F3
         + W['beta_F4_pulse']    * F4
         + W['beta_F5_regime']   * F5
         + W['beta_F6_xsect']    * F6
         - W['gamma_reversion']  * rev
         - W['gamma_divergence'] * div)

    # Vol-regime softening (high-vol tape is noisier — discount predictions)
    vr_w = VOL_REGIME_W.get(row.get('Vol_Regime', 'NORMAL'), 1.0)
    raw  *= vr_w

    # Tier weight as a-priori signal-quality prior
    raw *= TIER_W.get(row.get('SignalType', '-'), 0.90)

    # Confidence dampening (low HMM-confidence => softer prediction)
    conf = row.get('Regime_Confidence', 0.5)
    raw *= (0.6 + 0.4 * conf)

    # Change-point dampening — transitions are noisy
    if row.get('Change_Point', False):
        raw *= 0.65

    return raw


def compute_priority(df: pd.DataFrame) -> pd.DataFrame:
    """Adds Priority_Long / Priority_Short / *_z / *_pct to df. Returns df."""
    if df.empty:
        return df

    # Cross-sectional factor F6: how strong is this symbol vs. universe peers?
    # Built from the four per-symbol factors that already exist.
    composite_long  = (df['F1_PriceMom'] + df['F2_VolQual']
                      + 0.5 * df['Conviction'] / 20.0
                      + 0.5 * df['Pulse'])
    composite_short = -composite_long
    df['_F6_Long']  = (composite_long.rank(pct=True)  - 0.5) * 2   # [-1, +1]
    df['_F6_Short'] = (composite_short.rank(pct=True) - 0.5) * 2

    # Compute base priority for both sides (without F6, then add it)
    pl = df.apply(lambda r: _priority_one(r, 'long'),  axis=1)
    ps = df.apply(lambda r: _priority_one(r, 'short'), axis=1)

    pl = pl + W['beta_F6_xsect'] * df['_F6_Long']
    ps = ps + W['beta_F6_xsect'] * df['_F6_Short']

    df['Priority_Long']  = pl
    df['Priority_Short'] = ps

    # Robust z-score
    for col in ('Priority_Long', 'Priority_Short'):
        df[col + '_z']   = _z_robust(df[col])
        df[col + '_pct'] = df[col].rank(pct=True) * 100

    # Tie-breaker chain
    df['_tb_long'] = list(zip(
        -df['Priority_Long'],
        -df['Regime_Confidence'],
        df['Vol_Regime'].map(VOL_REGIME_IDX).fillna(1),
        -df['F1_PriceMom'].abs(),
    ))
    df['_tb_short'] = list(zip(
        -df['Priority_Short'],
        -df['Regime_Confidence'],
        df['Vol_Regime'].map(VOL_REGIME_IDX).fillna(1),
        -df['F1_PriceMom'].abs(),
    ))

    df = df.sort_values('_tb_long', ascending=True)
    df = df.drop(columns=['_F6_Long', '_F6_Short'])
    return df
