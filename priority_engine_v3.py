# ══════════════════════════════════════════════════════════════════════════════
# PRIORITY ENGINE v3 — DIRECTIONAL CONVICTION 2.0
# ══════════════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd

VOL_REGIME_W   = {'LOW': 1.20, 'NORMAL': 1.00, 'HIGH': 0.85, 'EXTREME': 0.55}
VOL_REGIME_IDX = {'LOW': 0,    'NORMAL': 1,    'HIGH': 2,    'EXTREME': 3}
TIER_W         = {'B: Long': 1.30, 'B: Short': 1.30,
                  'A: Long': 1.00, 'A: Short': 1.00,
                  'C: Long': 0.85, 'C: Short': 0.85,
                  'D: Long': 0.75, 'D: Short': 0.75}
BULL_NARRS = {'SQUEEZE','HYPER','IGNITE','ORGANIC','LOAD','HARDEN','STEALTH'}
BEAR_NARRS = {'CRASH','CAPITUL','LIQUID','POP','EXHAUST','CHAOS','TRAP'}

def _direction_gate(c, p, cd, side):
    s = 1 if side == 'long' else -1
    # Sharp sigmoid: misaligned (d<0) → near zero; aligned (d>0) → near 1
    d = (0.55*np.tanh(s*c/25.0) + 0.30*np.tanh(s*p/2.5) + 0.15*np.tanh(s*cd/4.0))
    return 1.0 / (1.0 + np.exp(-3.5 * d))

def _core_strength(c, p, cd, side):
    s = 1 if side == 'long' else -1
    # Output range: [0, 100]; ~50 at "moderately strong", ~85 at "very strong"
    return 100.0 * (0.50*np.tanh(max(s*c,0)/25.0)
                  + 0.30*np.tanh(max(s*p,0)/2.5)
                  + 0.20*np.tanh(max(s*cd,0)/4.0))

def _regime_mult(hb, hbear, vr, cp, conf, side):
    hmm = hb if side == 'long' else hbear
    # Map hmm_align ∈ [0,1] to a multiplicative band [0.5, 1.5]
    return float(np.clip((0.5 + 1.0*hmm)
                         * VOL_REGIME_W.get(vr, 1.0)
                         * (0.6 + 0.4*conf)
                         * (0.65 if cp else 1.0),
                         0.30, 1.60))

def _zone_iq(wt1, wt1_5ago, side):
    travel = wt1 - wt1_5ago
    extension_penalty = max(0.0, abs(wt1) - 60.0) / 40.0
    if side == 'long':
        ts = 0.5 + 0.5 * np.tanh(travel / 25.0)
        ps = np.exp(-((wt1 - 30.0)**2) / (2 * 35.0**2))
    else:
        ts = 0.5 + 0.5 * np.tanh(-travel / 25.0)
        ps = np.exp(-((wt1 + 30.0)**2) / (2 * 35.0**2))
    rr = 1.0 - extension_penalty
    return 0.5 + 1.3 * (0.45*ts + 0.35*ps + 0.20*rr)

def _confluence(row, side):
    keys = (['LA_5d','LB_5d','LC_5d','LD_5d'] if side == 'long'
            else ['SA_5d','SB_5d','SC_5d','SD_5d'])
    sets = [row.get(k, '—') != '—' for k in keys]
    tier_w = [1.00, 1.30, 0.85, 0.75]
    n = sum(t * int(s) for t, s in zip(tier_w, sets))
    nar = row.get('Narrative', 'NEUTRAL')
    if side == 'long':
        nw = 1.20 if nar in BULL_NARRS else (0.85 if nar in BEAR_NARRS else 1.00)
        dv = 1.25 if row.get('Bullish_Div', False) else 1.0
    else:
        nw = 1.20 if nar in BEAR_NARRS else (0.85 if nar in BULL_NARRS else 1.00)
        dv = 1.25 if row.get('Bearish_Div', False) else 1.0
    return float(np.clip(0.7 + 0.45 * n * nw * dv, 0.7, 2.5))

def _vol_flow(voltrend, side):
    s = 1 if side == 'long' else -1
    return float(np.clip(1.0 + 0.40 * np.tanh(s * voltrend / 10.0 * 1.5), 0.7, 1.4))

def _freshness(row, side):
    prefix = 'L' if side == 'long' else 'S'
    youngest = None
    # Check sets A, B, C, D across Today, 1d, 2d, 3d
    for age, D in enumerate(('Today','1d','2d','3d')):
        for X in 'ABCD':
            if row.get(f'{prefix}{X}_{D}', '—') != '—':
                if youngest is None or age < youngest:
                    youngest = age
                break
        if youngest is not None: break
        
    if youngest is None:
        return 0.4
    return float(np.exp(-youngest / 2.5))

def _reversion_pen(wt1, wt1_5ago, conv, side):
    travel = wt1 - wt1_5ago
    if side == 'long' and wt1 > 60 and travel < 0:
        return 0.20 * (wt1 - 60) / 40 * abs(conv) / 50
    if side == 'short' and wt1 < -60 and travel > 0:
        return 0.20 * (-wt1 - 60) / 40 * abs(conv) / 50
    return 0.0

def _div_pen(row, conv, side):
    if side == 'long' and row.get('Bearish_Div', False) and conv > 30:  return 0.30
    if side == 'short' and row.get('Bullish_Div', False) and conv < -30: return 0.30
    return 0.0

def _priority_one(row, side):
    c, p, cd = row['Conviction'], row['Pulse'], row['Conviction_Delta']
    
    # Layer decomposition
    dg = _direction_gate(c, p, cd, side)
    cs = _core_strength(c, p, cd, side)
    rm = _regime_mult(row['HMM_Bull'], row['HMM_Bear'], row['Vol_Regime'],
                        row['Change_Point'], row['Confidence'], side)
    zi = _zone_iq(row['Wave'], row.get('WT1_5ago', row['Wave']), side)
    cf = _confluence(row, side)
    vf = _vol_flow(row.get('VolTrend', 0.0), side)
    fd = _freshness(row, side)
    
    raw = dg * cs * rm * zi * cf * vf * fd
    
    # Subtractive guards (multiplicative haircuts)
    raw *= (1.0 - _reversion_pen(row['Wave'], row.get('WT1_5ago', row['Wave']), c, side))
    raw *= (1.0 - _div_pen(row, c, side))
    
    # Tier weighting based on signal set quality
    raw *= TIER_W.get(row.get('SignalType', '-'), 0.90)
    
    return max(raw, 0.0)

def compute_priority_v3(df: pd.DataFrame) -> pd.DataFrame:
    """Adds Priority_Long_v3 / Priority_Short_v3 / *_z / *_pct to df. Returns df."""
    if df.empty: return df
    df['Priority_Long_v3']  = df.apply(lambda r: _priority_one(r, 'long'),  axis=1)
    df['Priority_Short_v3'] = df.apply(lambda r: _priority_one(r, 'short'), axis=1)
    
    # Robust z-score (using median/MAD)
    for col in ('Priority_Long_v3', 'Priority_Short_v3'):
        med = df[col].median()
        mad = (df[col] - med).abs().median() * 1.4826 or 1e-6
        df[col + '_z']   = (df[col] - med) / mad
        df[col + '_pct'] = df[col].rank(pct=True) * 100
        
    # Tie-breaker chain: Multi-column tuple for stable sorting
    df['_tb_long'] = list(zip(
        -df['Priority_Long_v3'],
        -df['Confidence'],
        df['Vol_Regime'].map(VOL_REGIME_IDX),
        -df['Conviction_Delta'].abs()
    ))
    df['_tb_short'] = list(zip(
        -df['Priority_Short_v3'],
        -df['Confidence'],
        df['Vol_Regime'].map(VOL_REGIME_IDX),
        -df['Conviction_Delta'].abs()
    ))
    
    # Force deterministic sort (default: by Longs)
    df = df.sort_values('_tb_long', ascending=True)
    
    return df
