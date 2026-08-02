"""
Data acquisition for the Fair Value Engine.

Architecture adapted from the Tattva system's ``data/fetcher.py``. Every
external call is wrapped with:

1. **Two-tier cache** (memory + disk, TTL + versioned keys) — :mod:`fve.cache`
2. **Circuit breaker** per service — :mod:`fve.circuit_breaker`
3. **Retry with backoff** for transient whole-batch failures
4. **Partial-success completion** — a batch is one yfinance call, but yfinance
   rate-limits a few tickers per batch, so it can come back incomplete. Rather
   than caching that incomplete result, the missing symbols get one targeted
   re-fetch; anything still absent is backfilled from the last good snapshot.
5. **Stale fallback** — if a fetch fails and the circuit is open, the last good
   snapshot is served so the dashboard keeps working through an outage.

The **explanatory panel is fetched separately from the target**. The panel is
target-agnostic and caches as one unit keyed by (symbols, years); a single
equity target is a one-symbol fetch joined on afterwards. Folding the target
into the batch would give every new ticker a different cache key and force a
full multi-minute re-download each time.
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .cache import panel_cache, symbol_cache, symbol_fail_cache, target_cache
from .circuit_breaker import (CircuitBreakerError, RetryWithBackoff,
                              yfinance_circuit)
from .config import DataConfig
from .universe import CLASS_OF, YIELD_SYMBOLS, calendar_references

log = logging.getLogger(__name__)

# A backfilled column whose true last observation is further behind than this
# is dropped rather than filled: forward-filling a flat line across weeks would
# silently flatten that instrument's contribution to the factor structure.
STALE_BACKFILL_DAYS = 10


@dataclass
class DataBundle:
    """Prices plus provenance and quality metadata."""

    prices: pd.DataFrame
    source: str                     # LIVE | CACHE | CACHE (stale) | SIMULATED
    asof: pd.Timestamp
    requested: int
    coverage: pd.Series
    dropped: dict[str, str]
    fetch_seconds: float = 0.0
    backfilled: dict[str, str] = field(default_factory=dict)
    target: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def n_assets(self) -> int:
        return self.prices.shape[1]

    @property
    def n_sessions(self) -> int:
        return self.prices.shape[0]


# ---------------------------------------------------------------------------
# Raw yfinance calls (retry-wrapped; the circuit wraps these in turn)
# ---------------------------------------------------------------------------
def _yf_download(symbols: tuple[str, ...], period: str, timeout: int = 20) -> pd.DataFrame:
    """One raw yfinance batch call."""
    import yfinance as yf

    raw = yf.download(list(symbols), period=period, auto_adjust=True,
                      progress=False, threads=True, group_by="column",
                      timeout=timeout)
    if raw is None or (hasattr(raw, "empty") and raw.empty):
        raise ValueError("Empty yfinance response")
    return raw


# Retrying variant for bulk fetches, where a transient failure costs a whole
# batch and is worth waiting out.
_yf_batch = RetryWithBackoff(max_retries=2, initial_delay=1.5,
                             backoff_factor=2.0)(_yf_download)


def _extract_close(raw: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """Pull a wide Close-price frame out of any yfinance response layout."""
    if raw is None or len(raw) == 0:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        level0 = set(raw.columns.get_level_values(0))
        if "Close" in level0:
            px = raw["Close"]
        elif "Adj Close" in level0:
            px = raw["Adj Close"]
        else:                                    # group_by="ticker" layout
            try:
                px = raw.xs("Close", axis=1, level=1)
            except KeyError:
                return pd.DataFrame()
    else:
        px = raw[["Close"]] if "Close" in raw.columns else raw
        if len(symbols) == 1:
            px = px.copy()
            px.columns = symbols[:1]

    if isinstance(px, pd.Series):
        px = px.to_frame(symbols[0])
    px = px.astype("float64")
    if getattr(px.index, "tz", None) is not None:
        px.index = px.index.tz_localize(None)
    return px.sort_index()


def _download_batch(symbols: list[str], period: str, timeout: int = 20,
                    retry: bool = True) -> pd.DataFrame:
    """Circuit-protected batch download. Returns empty on any failure.

    ``retry=False`` is for interactive probes: a user waiting on a typed ticker
    should get an answer in seconds, not after the full retry ladder.
    """
    if not symbols:
        return pd.DataFrame()
    try:
        raw = yfinance_circuit.call(_yf_batch if retry else _yf_download,
                                    tuple(symbols), period, timeout)
    except CircuitBreakerError as exc:
        log.warning("yfinance circuit open, skipping batch: %s", exc)
        return pd.DataFrame()
    except Exception as exc:  # noqa: BLE001
        log.warning("yfinance batch failed (%s…): %s", symbols[0], exc)
        return pd.DataFrame()
    return _extract_close(raw, symbols)


# ---------------------------------------------------------------------------
# Snapshot backfill
# ---------------------------------------------------------------------------
def _backfill_missing(px: pd.DataFrame, symbols: list[str]
                      ) -> tuple[pd.DataFrame, dict[str, str]]:
    """Refill columns a partial fetch dropped, from the newest usable snapshot.

    yfinance routinely rate-limits a handful of tickers per batch while the
    rest succeed. The partial frame is non-empty, so it bypasses the
    all-or-nothing stale fallback and would be cached with holes in it —
    silently shrinking the cross-section the factor model sees.

    Returns the frame plus a ``{symbol: last native date}`` registry of what
    was carried over, so the UI can disclose it.
    """
    if px.empty:
        return px, {}

    missing = [s for s in symbols
               if s not in px.columns or px[s].notna().sum() == 0]
    if not missing:
        return px, {}

    filled: dict[str, str] = {}
    dropped: list[str] = []
    frame_end = px.index.max()

    for snap in panel_cache.snapshots_newest_first():
        if not missing:
            break
        if not isinstance(snap, pd.DataFrame) or snap.empty:
            continue
        aligned = snap.reindex(px.index).ffill()
        for sym in list(missing):
            if sym not in aligned.columns or aligned[sym].isna().all():
                continue
            native = snap[sym].dropna()
            if len(native):
                last_native = pd.Timestamp(native.index.max())
                behind = int(np.busday_count(last_native.date(),
                                             pd.Timestamp(frame_end).date()))
                if behind > STALE_BACKFILL_DAYS:
                    dropped.append(sym)
                    missing.remove(sym)
                    continue
                filled[sym] = str(last_native.date())
            px[sym] = aligned[sym]
            missing.remove(sym)

    if filled:
        log.info("Backfilled %d instruments from snapshot: %s",
                 len(filled), list(filled)[:8])
    if dropped:
        log.warning("Dropped %d instruments (snapshot >%d sessions stale): %s",
                    len(dropped), STALE_BACKFILL_DAYS, dropped[:8])
        px = px.drop(columns=dropped, errors="ignore")
    return px, filled


# ---------------------------------------------------------------------------
# Panel fetch
# ---------------------------------------------------------------------------
def fetch_panel(symbols: list[str], cfg: DataConfig, progress=None
                ) -> tuple[pd.DataFrame, str, float, dict[str, str]]:
    """Fetch the explanatory panel with the full fault-tolerance stack.

    ``progress`` is an optional ``callable(done, total, message)``.
    """
    key = tuple(sorted(symbols))
    period = f"{cfg.years}y"

    cached = panel_cache.get(key, period)
    if cached is not None and isinstance(cached, pd.DataFrame) and not cached.empty:
        if progress:
            progress(1, 1, f"Loaded {cached.shape[1]} instruments from cache")
        return cached, "CACHE", 0.0, {}

    t0 = time.time()
    chunks = [symbols[i:i + cfg.chunk_size]
              for i in range(0, len(symbols), cfg.chunk_size)]
    total = len(chunks) + 1
    frames: list[pd.DataFrame] = []

    for i, chunk in enumerate(chunks):
        if progress:
            progress(i, total, f"Fetching batch {i + 1}/{len(chunks)} "
                               f"({len(chunk)} instruments)…")
        part = _download_batch(chunk, period)
        if not part.empty:
            frames.append(part)

    px = pd.concat(frames, axis=1) if frames else pd.DataFrame()
    if not px.empty:
        px = px.loc[:, ~px.columns.duplicated()]

    # Targeted recovery of whatever the batches dropped. A transport failure
    # kills a whole batch, not just the ticker that stalled, so writing off 45
    # instruments because one DNS lookup timed out is not acceptable.
    missing = [s for s in symbols
               if s not in px.columns or px[s].notna().sum() == 0]
    if missing and frames:
        if progress:
            progress(len(chunks), total, f"Recovering {len(missing)} instruments…")
        recovered: list[pd.DataFrame] = []
        for j in range(0, len(missing), 8):
            part = _download_batch(missing[j:j + 8], period, timeout=15)
            if not part.empty:
                recovered.append(part)
        if recovered:
            extra = pd.concat(recovered, axis=1)
            extra = extra.loc[:, ~extra.columns.duplicated()]
            keep = [c for c in extra.columns if extra[c].notna().sum() > 0]
            if keep:
                px = (extra[keep] if px.empty else
                      px.drop(columns=keep, errors="ignore").join(extra[keep],
                                                                 how="outer"))

    if progress:
        progress(total, total, "Download complete")

    if px.empty:
        stale = panel_cache.get_stale(key, period)
        if stale is not None and not stale.empty:
            log.warning("Panel fetch empty; serving last-good snapshot")
            return stale, "CACHE (stale)", time.time() - t0, {}
        raise RuntimeError(
            "Yahoo Finance returned nothing for any batch. The provider may be "
            "unreachable or rate-limiting. Switch on Demo mode to run against a "
            "synthetic market, or retry in a minute.")

    px.index = pd.to_datetime(px.index)
    if getattr(px.index, "tz", None) is not None:
        px.index = px.index.tz_localize(None)
    px = px.sort_index()

    px, backfilled = _backfill_missing(px, symbols)
    panel_cache.put(key, period, value=px)
    return px, "LIVE", time.time() - t0, backfilled


def fetch_series(ticker: str, years: int = 8, probe: bool = False) -> pd.Series | None:
    """Close series for a single instrument — the free-form target path.

    Also doubles as the "does this ticker exist" probe used by symbol
    resolution, so a successful resolution makes the subsequent analysis fetch
    a cache hit rather than a second network call. Set ``probe`` for the
    interactive path: it fails fast instead of working through the retry ladder.
    """
    period = f"{years}y"
    cached = target_cache.get(ticker, period)
    if cached is not None:
        return cached if isinstance(cached, pd.Series) and len(cached) else None

    px = _download_batch([ticker], period, timeout=8 if probe else 15,
                         retry=not probe)
    if px.empty:
        stale = target_cache.get_stale(ticker, period)
        return stale if isinstance(stale, pd.Series) and len(stale) else None

    col = ticker if ticker in px.columns else px.columns[0]
    s = pd.to_numeric(px[col], errors="coerce").dropna()
    if len(s) < 60:
        return None
    s.name = ticker
    target_cache.put(ticker, period, value=s)
    return s


def resolve_symbol(raw: str, market: str, years: int = 8) -> tuple[str | None, str]:
    """Resolve a typed symbol to a yfinance ticker, with listing auto-detect.

    ``market='india'`` respects an explicit ``.NS``/``.BO`` suffix, otherwise
    probes ``SYMBOL.NS`` first then ``SYMBOL.BO`` (NSE takes precedence for
    dual-listed names). ``market='us'`` uppercases and translates ``.`` to
    ``-`` (yfinance convention: ``BRK.B`` -> ``BRK-B``).

    Returns ``(ticker, exchange_label)`` or ``(None, error_message)``.
    """
    cleaned = (raw or "").strip().upper()
    if not cleaned or " " in cleaned or len(cleaned) > 20:
        return None, f"'{raw}' is not a valid ticker symbol."

    cached = symbol_cache.get(cleaned, market)
    if cached is not None:
        return tuple(cached)  # type: ignore[return-value]

    # Failures are memoised on a short TTL. Streamlit reruns this on every
    # keystroke elsewhere in the sidebar, and re-probing a symbol that does not
    # exist would cost a network round trip each time. The TTL is deliberately
    # short so a genuinely new listing is not blacklisted for the session.
    failed = symbol_fail_cache.get(cleaned, market)
    if failed is not None:
        return None, str(failed)

    if market == "india":
        if cleaned.endswith((".NS", ".BO")):
            candidates = [(cleaned, "NSE" if cleaned.endswith(".NS") else "BSE")]
        else:
            candidates = [(f"{cleaned}.NS", "NSE"), (f"{cleaned}.BO", "BSE")]
    else:
        candidates = [(cleaned.replace(".", "-"), "US")]

    for ticker, exchange in candidates:
        try:
            s = fetch_series(ticker, years, probe=True)
        except Exception as exc:  # noqa: BLE001
            log.warning("Symbol probe %s failed: %s", ticker, exc)
            s = None
        if s is not None and not s.empty:
            symbol_cache.put(cleaned, market, value=(ticker, exchange))
            return ticker, exchange

    tried = " or ".join(f"{t} ({e})" for t, e in candidates)
    if market == "india":
        msg = f"'{cleaned}' not found on NSE or BSE via Yahoo Finance (tried {tried})."
    else:
        msg = f"'{cleaned}' not found on Yahoo Finance (tried {tried})."
    symbol_fail_cache.put(cleaned, market, value=msg)
    return None, msg


# ---------------------------------------------------------------------------
# Cleaning / alignment
# ---------------------------------------------------------------------------
def align_and_clean(px: pd.DataFrame, cfg: DataConfig, target: str | None = None
                    ) -> tuple[pd.DataFrame, pd.Series, dict[str, str]]:
    """Put the panel on one trading calendar and drop unusable series."""
    dropped: dict[str, str] = {}
    required = [target] if target else []

    # 1. Master calendar from a liquid reference for the target's own market.
    ref = next((r for r in calendar_references(target)
                if r in px.columns and px[r].notna().sum() > 50), None)
    if ref is not None:
        calendar = px.index[px[ref].notna()]
    else:
        density = px.notna().mean(axis=1)
        calendar = px.index[density > 0.5]
        if len(calendar) < 100:
            calendar = px.index
    px = px.reindex(calendar)

    # 2. Weekend rows are artefacts of 7-day instruments (crypto, some FX
    #    crosses). Equities and commodities do not trade then, so such a row
    #    would be a fully forward-filled duplicate of Friday masquerading as a
    #    session.
    px = px[px.index.dayofweek < 5]

    # 3. Bridge short gaps only — a long gap means the instrument was not
    #    trading, and filling it would fabricate a run of zero returns.
    px = px.ffill(limit=cfg.max_ffill)

    coverage = px.notna().mean()
    for sym in px.columns:
        if sym in required:
            continue
        if coverage[sym] < cfg.min_coverage:
            dropped[sym] = f"coverage {coverage[sym]:.0%} < {cfg.min_coverage:.0%}"

    for sym in px.columns:
        if sym in dropped or sym in required:
            continue
        s = px[sym].dropna()
        if len(s) < 60:
            dropped[sym] = "fewer than 60 observations"
        elif sym not in YIELD_SYMBOLS and (s <= 0).any():
            dropped[sym] = "non-positive price observations"
        elif s.nunique() < 10:
            dropped[sym] = "constant / near-constant series"

    px = px[[c for c in px.columns if c not in dropped]]

    # 4. Trim the leading stretch where too little of the panel exists.
    if px.shape[1] > 5:
        density = px.notna().mean(axis=1)
        good = np.flatnonzero(density.values > 0.8)
        if len(good) and len(px) - good[0] > 250:
            px = px.iloc[good[0]:]

    px = px.dropna(axis=0, how="all")
    return px, coverage, dropped


def load_universe(symbols: list[str], cfg: DataConfig, target: str | None = None,
                  progress=None) -> DataBundle:
    """End-to-end loader: fetch panel, join the target, align, clean, report.

    ``symbols`` is the explanatory universe and must be **target-independent** —
    pass the tier list as-is, without appending a free-form ticker. The panel
    cache is keyed on it, so letting the target perturb the list would give
    every new ticker a different key and re-download the whole universe. A
    target that is not already in the panel is fetched separately and joined.
    """
    notes: list[str] = []
    backfilled: dict[str, str] = {}

    panel_symbols = list(symbols)

    if cfg.synthetic:
        sim_symbols = panel_symbols + ([target] if target and target not in panel_symbols
                                       else [])
        px = simulate_universe(sim_symbols, cfg.years, cfg.seed)
        source, secs = "SIMULATED", 0.0
    else:
        px, source, secs, backfilled = fetch_panel(panel_symbols, cfg, progress=progress)

        # Only a target outside the universe needs its own call.
        if target and (target not in px.columns or px[target].notna().sum() == 0):
            if progress:
                progress(1, 1, f"Fetching target {target}…")
            series = fetch_series(target, cfg.years)
            if series is None or series.empty:
                raise RuntimeError(
                    f"No usable price history returned for target '{target}'.")
            px = px.drop(columns=[target], errors="ignore").join(series, how="outer")
            notes.append(f"{target} fetched separately and joined onto the panel.")

    if backfilled:
        notes.append(f"{len(backfilled)} instruments carried from the last good "
                     f"snapshot after a partial provider response.")

    clean, coverage, dropped = align_and_clean(px, cfg, target=target)

    if target and target not in clean.columns:
        raise RuntimeError(
            f"Target '{target}' did not survive cleaning "
            f"({dropped.get(target, 'insufficient overlapping history')}).")

    if clean.shape[1] < 20:
        raise RuntimeError(
            f"Only {clean.shape[1]} instruments survived cleaning — not enough "
            "cross-section to build a market state model.")

    return DataBundle(prices=clean, source=source, asof=clean.index[-1],
                      requested=len(symbols), coverage=coverage, dropped=dropped,
                      fetch_seconds=secs, backfilled=backfilled, target=target,
                      notes=notes)


def provider_status() -> dict:
    """Circuit and cache telemetry for the diagnostics view."""
    return {
        "circuit": yfinance_circuit.status(),
        "panel_cache": panel_cache.stats(),
        "target_cache": target_cache.stats(),
        "symbol_cache": symbol_cache.stats(),
    }


# ---------------------------------------------------------------------------
# Synthetic universe (offline demo / reproducible tests)
# ---------------------------------------------------------------------------
_SIM_FACTORS = ["MKT", "RATES", "INFL", "MOM", "USD", "CRYPTO"]

_EQUITY_CLASSES = {"Equity Indices", "US Sector ETFs", "US Style/Factor ETFs",
                   "Global Equity ETFs", "India Equity", "US Stocks", "India Stocks"}


def _sim_loadings(cls: str, sym: str, rng: np.random.Generator) -> np.ndarray:
    """Economically-shaped factor loadings so the simulation is not pure noise."""
    L = dict(MKT=0.0, RATES=0.0, INFL=0.0, MOM=0.0, USD=0.0, CRYPTO=0.0)
    if cls in _EQUITY_CLASSES:
        L.update(MKT=1.0, RATES=-0.22, INFL=0.08, MOM=0.30, USD=-0.10)
        if cls in ("Global Equity ETFs", "India Equity"):
            L["USD"] = -0.40
        if sym in ("SMH", "SPHB", "^NDX", "QQQ", "XLK"):
            L.update(MKT=1.35, MOM=0.50, CRYPTO=0.18)
        if sym in ("XLE", "XOP", "OIH"):
            L.update(INFL=0.55, MKT=0.85)
    elif cls in ("Rates & Treasuries", "Credit", "Inflation-Linked"):
        L.update(MKT=0.10, RATES=0.85, INFL=0.12)
        if cls == "Credit":
            L.update(MKT=0.35, RATES=0.55)
        if cls == "Inflation-Linked":
            L.update(INFL=0.45, RATES=0.60)
        if sym in YIELD_SYMBOLS:
            L.update(MKT=-0.05, RATES=-0.90, INFL=0.30)
    elif cls == "Commodities":
        L.update(MKT=0.20, INFL=0.85, USD=-0.45)
        if sym in ("GC=F", "GLD", "SI=F", "SLV", "GDX"):
            L.update(MKT=0.05, INFL=0.60, USD=-0.70, RATES=0.25)
    elif cls == "FX":
        usd_short = (any(sym.startswith(p) for p in ("EUR", "GBP", "AUD", "NZD"))
                     or sym in ("FXE", "FXY", "FXB", "FXF", "CEW"))
        L.update(MKT=0.10, RATES=0.15, USD=-0.90 if usd_short else 0.90)
    elif cls == "Crypto":
        L.update(MKT=0.45, MOM=0.45, CRYPTO=0.95)
    elif cls == "Volatility":
        L.update(MKT=-1.30, MOM=-0.25)
    elif cls == "Real Assets":
        L.update(MKT=0.80, RATES=0.45, INFL=0.15)
    return np.array([L[k] for k in _SIM_FACTORS]) + rng.normal(0, 0.10, len(_SIM_FACTORS))


def simulate_universe(symbols: list[str], years: int, seed: int) -> pd.DataFrame:
    """Latent-factor DGP with a 5-state Markov regime chain.

    Deterministic for a given seed. Note it contains **no mean-reverting
    mispricing** by construction: an honest engine should report roughly zero
    signal on it, which makes it a useful leakage control as well as a demo.
    """
    rng = np.random.default_rng(seed)
    n_days = int(years * 252)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    n = len(symbols)

    P = np.array([[.965, .015, .008, .008, .004],
                  [.012, .965, .012, .007, .004],
                  [.010, .014, .960, .010, .006],
                  [.012, .008, .010, .950, .020],
                  [.010, .004, .006, .025, .955]])
    reg = np.zeros(n_days, dtype=int)
    for t in range(1, n_days):
        reg[t] = rng.choice(5, p=P[reg[t - 1]])

    mu = {"MKT": [.0009, .0003, .0000, .0000, -.0012],
          "RATES": [.0001, .0001, -.0001, .0002, -.0003],
          "INFL": [.0001, .0001, .0002, .0004, .0002],
          "MOM": [.0002, .0002, .0000, .0000, -.0004],
          "USD": [-.0001, .0000, .0001, .0002, .0004],
          "CRYPTO": [.0020, .0005, .0000, .0005, -.0030]}
    sd = {"MKT": [.0060, .0045, .0050, .0110, .0095],
          "RATES": [.0030, .0025, .0030, .0050, .0045],
          "INFL": [.0050, .0040, .0045, .0080, .0070],
          "MOM": [.0040, .0035, .0030, .0060, .0055],
          "USD": [.0030, .0025, .0025, .0045, .0040],
          "CRYPTO": [.0220, .0160, .0140, .0300, .0260]}
    F = np.column_stack([rng.normal(np.array(mu[k])[reg], np.array(sd[k])[reg])
                         for k in _SIM_FACTORS])
    F[:, 3] = pd.Series(F[:, 3]).ewm(alpha=.12).mean().values

    beta_mult = np.array([1.0, 1.0, 1.0, 1.35, 1.60])[reg]
    idio_mult = np.array([0.85, 0.90, 1.00, 1.50, 1.35])[reg]

    # Per-symbol streams, seeded from the symbol name rather than its position.
    # Positional seeding would give every free-form target the same RNG state
    # (it is always appended last), so AAPL and RELIANCE.NS would simulate to
    # byte-identical series — which reads as a broken dashboard.
    sym_rng = {s: np.random.default_rng(
        int(hashlib.md5(f"{seed}|{s}".encode()).hexdigest()[:8], 16)) for s in symbols}

    B = np.vstack([_sim_loadings(CLASS_OF.get(s, "US Stocks"), s, sym_rng[s])
                   for s in symbols])
    idio_by_class = {
        "Equity Indices": .007, "US Sector ETFs": .008, "US Style/Factor ETFs": .006,
        "Global Equity ETFs": .009, "India Equity": .010, "Rates & Treasuries": .004,
        "Credit": .004, "Inflation-Linked": .003, "Commodities": .014, "FX": .005,
        "Crypto": .030, "Volatility": .050, "Real Assets": .011,
    }
    idio_base = np.array([idio_by_class.get(CLASS_OF.get(s, ""), .013) for s in symbols])
    alpha = np.array([sym_rng[s].normal(0.0002, .0004) for s in symbols])

    idio = np.column_stack([sym_rng[s].normal(0, 1, n_days) for s in symbols])
    R = (alpha + (F @ B.T) * beta_mult[:, None]
         + idio * (idio_base * idio_mult[:, None]))

    p0 = np.empty(n)
    for i, s in enumerate(symbols):
        cls = CLASS_OF.get(s, "US Stocks")
        rng = sym_rng[s]
        if cls == "Crypto":
            p0[i] = np.exp(rng.uniform(np.log(0.3), np.log(70000)))
        elif cls == "FX":
            p0[i] = 150.0 if "JPY" in s else (85.0 if "INR" in s else rng.uniform(0.6, 1.7))
        elif s in YIELD_SYMBOLS:
            p0[i] = rng.uniform(1.5, 5.0)
        elif cls == "Volatility":
            p0[i] = rng.uniform(12, 40)
        elif cls == "Equity Indices" or s in ("^NSEI", "^BSESN", "^NSEBANK"):
            p0[i] = rng.uniform(3000, 42000)
        elif cls == "Commodities":
            p0[i] = 2400.0 if "GC" in s else rng.uniform(18, 260)
        else:
            p0[i] = np.exp(rng.uniform(np.log(25), np.log(700)))

    return pd.DataFrame(p0 * np.exp(np.cumsum(R, axis=0)), index=dates, columns=symbols)
