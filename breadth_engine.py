"""
Sanket v3.5.0 — Breadth Engine: market & sector advance/decline intelligence.

A shared, dependency-light module (numpy + pandas only) that turns the universe
close panel the screener already holds into breadth signals. Ported from the
Hemrek "Market Breadth" app (Relative_Breadth = Fibonacci-MA blend of an
EMA-smoothed A/(A+D) oscillator), with two additions for the Sanket stack:

  • Universe_Breadth + Breadth_Momentum  — one market-wide series per date.
      Drives Path A (regime tilt in compute_priority) and Path B (a Layer-2
      signal-confidence feature). Same value for every stock on a date — a
      timing/regime signal, NOT a cross-sectional one.

  • Sector_Rel_Breadth = sector_breadth − universe_breadth  — per (sector, date).
      Drives Path C (the F8 cross-sectional factor). De-meaning against the
      universe makes it orthogonal to Path A, so the two never double-count the
      market-wide level: F8 ranks *which groups* are participating, the tilt
      handles *how bullish the tape* is.

The engine computes NOTHING the screener can't already see — it reads the same
``data_dict`` (ticker → OHLCV) used to rank, so there is zero new data
dependency. Built once per run, before the per-stock loop, and attached
identically in the live-screener and calibration-harvest paths so train and
apply features match bar-for-bar.
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd

# Oversold 0.40 / overbought 0.50 (Hemrek bands) → 0.45 is the neutral midpoint.
BREADTH_NEUTRAL = 0.45
# Smoothing window for the EMA-smoothed A/(A+D) oscillator (Hemrek "Custom Breadth").
_BREADTH_PERIOD = 10
# Fibonacci MA periods averaged into the "Relative Breadth" oscillator.
_FIB_PERIODS = (2, 3, 5, 8, 13, 21)
# Minimum constituents with a return on a date for breadth to be defined.
_MIN_NAMES = 10
# A sector needs at least this many members for its breadth to be trustworthy.
_MIN_SECTOR_NAMES = 5

# NSE sectoral indexes used to build the ticker → sector map (India universes).
# Each name is what get_index_stock_list() / the NSE equity-stockIndices API expects.
NSE_SECTOR_INDEXES = [
    "NIFTY BANK", "NIFTY IT", "NIFTY AUTO", "NIFTY PHARMA", "NIFTY FMCG",
    "NIFTY METAL", "NIFTY MEDIA", "NIFTY REALTY", "NIFTY ENERGY",
    "NIFTY FINANCIAL SERVICES", "NIFTY PSU BANK", "NIFTY PRIVATE BANK",
    "NIFTY CONSUMER DURABLES", "NIFTY HEALTHCARE INDEX", "NIFTY OIL & GAS",
]


def _seeded_ema(x: pd.Series, period: int = _BREADTH_PERIOD) -> pd.Series:
    """EMA seeded by the first valid SMA(period) — faithful to the Hemrek loop.

    Standard ``ewm(adjust=False)`` seeds on the first sample; the breadth
    oscillator instead seeds on the first full SMA so the smoothing only begins
    once ``period`` observations exist. NaN inputs hold the previous value.
    """
    x = x.astype(float)
    sma = x.rolling(period, min_periods=period).mean()
    out = np.full(len(x), np.nan, dtype=float)
    fv = sma.first_valid_index()
    if fv is None:
        return pd.Series(out, index=x.index)
    pos = x.index.get_loc(fv)
    if isinstance(pos, slice):           # duplicate timestamp guard
        pos = pos.start
    out[pos] = sma.iloc[pos]
    c = 2.0 / (period + 1)
    xv = x.to_numpy()
    for i in range(pos + 1, len(x)):
        prev = out[i - 1]
        cur = xv[i]
        out[i] = prev if np.isnan(cur) else prev + c * (cur - prev)
    return pd.Series(out, index=x.index)


def _relative_breadth(close_df: pd.DataFrame) -> pd.Series:
    """Relative Breadth oscillator (~[0,1]) from a wide close panel (dates × tickers).

    Ports Breadth-main ``compute_timeseries``: per-date advance/decline →
    AD_Ratio → x = A/(A+D) → EMA(10) Custom Breadth → blend with six Fibonacci
    SMAs. NaN on dates with < _MIN_NAMES participating constituents.
    """
    if close_df is None or close_df.shape[1] == 0:
        return pd.Series(dtype=float)
    close_df = close_df.sort_index()
    pct = close_df.pct_change(fill_method=None)
    adv = (pct > 0).sum(axis=1).astype(float)
    dec = (pct < 0).sum(axis=1).astype(float)
    have = pct.notna().sum(axis=1).astype(float)          # names with a defined return

    ad_ratio = np.where(dec.to_numpy() > 0,
                        adv.to_numpy() / np.maximum(dec.to_numpy(), 1e-9),
                        adv.to_numpy())
    ad_ratio = pd.Series(ad_ratio, index=close_df.index)
    x = ad_ratio / (ad_ratio + 1.0)                       # A/(A+D)-equivalent, [0,1]

    cb = _seeded_ema(x, _BREADTH_PERIOD)                  # Custom Breadth
    z = sum(cb.rolling(p, min_periods=1).mean() for p in _FIB_PERIODS) / len(_FIB_PERIODS)
    rel = (z + cb) / 2.0
    rel[have.to_numpy() < _MIN_NAMES] = np.nan            # insufficient breadth
    return rel


def _close_panel(data_dict: dict, tickers=None) -> pd.DataFrame:
    """Assemble a wide close DataFrame (index=dates, cols=tickers) from data_dict."""
    cols = {}
    src = tickers if tickers is not None else data_dict.keys()
    for t in src:
        d = data_dict.get(t)
        if d is None or 'Close' not in getattr(d, 'columns', []):
            continue
        s = d['Close']
        if s is None or s.empty:
            continue
        cols[t] = s
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols).sort_index()


# Process-lifetime cache for the (slow, network-derived) sector map, keyed by
# universe+index. get_index_stock_list already caches constituents ~1h; this
# avoids even re-assembling the reverse map within a run.
_SECTOR_MAP_CACHE: dict = {}
_SECTOR_MAP_TTL = 3600.0


def build_sector_map(universe: str, selected_index: str, get_index_stock_list) -> dict:
    """ticker → sector dict for India universes, from NSE sectoral-index membership.

    Best-effort and gated to Indian equities (the only universe where "sector"
    is meaningful and constituent lists are fetchable). Returns {} for every
    other universe, or on any failure — callers treat an empty map as "Path C
    off" (Sector_Rel_Breadth = 0 → F8 inert). Cached process-wide.
    """
    if universe not in ("India Indexes",):
        return {}
    key = f"{universe}::{selected_index}"
    hit = _SECTOR_MAP_CACHE.get(key)
    if hit and (time.time() - hit[0]) < _SECTOR_MAP_TTL:
        return hit[1]

    mapping: dict = {}
    fetched_any = False
    for n_idx, sector_idx in enumerate(NSE_SECTOR_INDEXES):
        try:
            symbols, _msg = get_index_stock_list(sector_idx)
        except Exception:
            symbols = None
        # Fail-fast: if the first couple of indexes return nothing, assume the
        # NSE source is down/blocked and bail rather than burning a full timeout
        # on all 15 (worst case otherwise ~15 × timeout). Path C then degrades to
        # off (empty map) — the screener is never stalled by breadth.
        if not symbols:
            if not fetched_any and n_idx >= 1:
                break
            continue
        fetched_any = True
        sector = sector_idx.replace("NIFTY ", "").replace(" INDEX", "").strip().title()
        for sym in symbols:
            # First-wins: a name in multiple indexes (e.g. a bank in BANK and
            # FINANCIAL SERVICES) keeps its first, most-specific sector.
            mapping.setdefault(sym, sector)

    _SECTOR_MAP_CACHE[key] = (time.time(), mapping)
    return mapping


class BreadthPanel:
    """Computed-once breadth signals for a run, attachable to each per-stock frame.

    Attributes
    ----------
    universe : pd.Series      date → Relative_Breadth (~[0,1]); market-wide.
    universe_mom : pd.Series  date → 3-bar-smoothed breadth momentum.
    sector_rel : dict         sector → (sector_breadth − universe_breadth) series.
    ticker_sector : dict      ticker → sector.
    """

    def __init__(self, universe, universe_mom, sector_rel, ticker_sector):
        self.universe = universe
        self.universe_mom = universe_mom
        self.sector_rel = sector_rel
        self.ticker_sector = ticker_sector

    @property
    def available(self) -> bool:
        return self.universe is not None and not self.universe.dropna().empty

    def _rel_for_ticker(self, ticker: str) -> pd.Series:
        sec = self.ticker_sector.get(ticker)
        if sec is None or sec not in self.sector_rel:
            return pd.Series(0.0, index=self.universe.index)
        return self.sector_rel[sec]

    def attach(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Attach Universe_Breadth / Breadth_Momentum / Sector_Rel_Breadth to df.

        Reindexed onto df.index (forward-filled), so it works whether df is daily
        or weekly-resampled — both the screener and the harvest call this the same
        way, keeping train/apply features identical.
        """
        idx = df.index
        if self.universe is None:
            df['Universe_Breadth'] = BREADTH_NEUTRAL
            df['Breadth_Momentum'] = 0.0
            df['Sector_Rel_Breadth'] = 0.0
            return df
        df['Universe_Breadth'] = self.universe.reindex(idx, method='ffill').fillna(BREADTH_NEUTRAL).to_numpy()
        df['Breadth_Momentum'] = self.universe_mom.reindex(idx, method='ffill').fillna(0.0).to_numpy()
        df['Sector_Rel_Breadth'] = self._rel_for_ticker(ticker).reindex(idx, method='ffill').fillna(0.0).to_numpy()
        return df


def build_breadth_panel(data_dict: dict, sector_map: dict | None = None) -> BreadthPanel:
    """Build the run's BreadthPanel from the universe close panel.

    Parameters
    ----------
    data_dict : {ticker: OHLCV DataFrame}  — the same panel the screener ranks.
    sector_map : {ticker: sector} or None  — from build_sector_map (India only).

    Universe breadth uses the whole panel; sector breadth uses each sector's
    member subset (≥ _MIN_SECTOR_NAMES). Sector_Rel = sector − universe (de-meaned).
    """
    close_df = _close_panel(data_dict)
    universe = _relative_breadth(close_df)
    if universe.dropna().empty:
        return BreadthPanel(None, None, {}, {})

    universe_mom = universe.diff().rolling(3, min_periods=1).mean()

    sector_rel: dict = {}
    ticker_sector: dict = {}
    if sector_map:
        # group tickers present in this run by sector
        by_sector: dict = {}
        for t in close_df.columns:
            sec = sector_map.get(t)
            if sec is None:
                continue
            ticker_sector[t] = sec
            by_sector.setdefault(sec, []).append(t)
        for sec, members in by_sector.items():
            if len(members) < _MIN_SECTOR_NAMES:
                # too thin to be a reliable group → leave its members on F8=0
                for t in members:
                    ticker_sector.pop(t, None)
                continue
            sec_breadth = _relative_breadth(close_df[members])
            # De-mean against the universe so F8 is orthogonal to the Path-A tilt.
            sector_rel[sec] = (sec_breadth - universe).reindex(universe.index)

    return BreadthPanel(universe, universe_mom, sector_rel, ticker_sector)


# Self-test: synthetic panel → sanity-check shapes, ranges, and de-meaning.
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=120, freq="B")
    tickers = [f"T{i}.NS" for i in range(40)]
    # Two groups with different drift so sector breadth diverges from universe.
    closes = {}
    for i, t in enumerate(tickers):
        drift = 0.0008 if i < 20 else -0.0004
        steps = rng.normal(drift, 0.015, len(dates))
        closes[t] = pd.Series(100 * np.exp(np.cumsum(steps)), index=dates)
        closes[t] = pd.DataFrame({'Close': closes[t]})
    data = {t: closes[t] for t in tickers}
    smap = {t: ("Strong" if i < 20 else "Weak") for i, t in enumerate(tickers)}

    panel = build_breadth_panel(data, sector_map=smap)
    assert panel.available, "panel should be available"
    u = panel.universe.dropna()
    assert ((u >= 0) & (u <= 1)).all(), "universe breadth must be in [0,1]"
    assert "Strong" in panel.sector_rel and "Weak" in panel.sector_rel
    # Strong group should sit above the universe more often than the weak group.
    s_rel = panel.sector_rel["Strong"].dropna().mean()
    w_rel = panel.sector_rel["Weak"].dropna().mean()
    assert s_rel > w_rel, f"strong sector should out-participate weak ({s_rel:.4f} vs {w_rel:.4f})"

    df = data[tickers[0]].copy()
    df = panel.attach(df, tickers[0])
    for c in ("Universe_Breadth", "Breadth_Momentum", "Sector_Rel_Breadth"):
        assert c in df.columns and df[c].notna().all(), f"{c} should be fully attached"
    print(f"OK · universe∈[{u.min():.3f},{u.max():.3f}] "
          f"strong_rel={s_rel:+.4f} weak_rel={w_rel:+.4f} "
          f"attached={list(df.columns[-3:])}")
