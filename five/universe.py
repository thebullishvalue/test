"""
Cross-asset instrument universe for the Fair Value Engine.

Two distinct concepts live here, and keeping them separate is the point:

**The explanatory universe** (~237 instruments) is the market state the engine
prices a target against. It is deliberately built from *aggregates* — indices,
sector/style/regional ETFs, rates, credit, commodities, FX, crypto,
volatility — rather than individual company stocks. A basket of single names
adds little a sector ETF does not already carry, while multiplying idiosyncratic
noise and download time. This universe is target-agnostic, so it batches and
caches as one unit.

**The target** is whatever the user wants valued. Index, ETF, commodity, FX
cross and crypto targets come from the explanatory universe itself. Individual
equities — US or India — are entered as a free-form ticker and fetched
separately, then joined onto the panel. Keeping the target's fetch off the
batch is what stops a dynamically varying symbol set from busting the panel
cache on every new ticker.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Explanatory universe: asset class -> tickers (Yahoo Finance symbology)
# ---------------------------------------------------------------------------
UNIVERSE: dict[str, list[str]] = {
    "Equity Indices": [
        "^GSPC", "^IXIC", "^DJI", "^RUT", "^NDX", "^FTSE", "^GDAXI", "^FCHI",
        "^STOXX50E", "^N225", "^HSI", "^AXJO", "^GSPTSE", "^KS11", "^MXX", "^TWII",
    ],
    "US Sector ETFs": [
        "XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLB", "XLU", "XLRE",
        "XLC", "SMH", "XBI", "KRE", "ITB", "XOP", "OIH", "XME", "JETS", "IYT",
    ],
    "US Style/Factor ETFs": [
        "SPY", "QQQ", "IWM", "DIA", "MDY", "VTV", "VUG", "MTUM", "QUAL", "USMV",
        "SPHB", "SPLV", "VYM", "SIZE", "RSP", "EQAL",
    ],
    "Global Equity ETFs": [
        "EFA", "EEM", "VWO", "IEFA", "ACWI", "VEU", "EWJ", "EWG", "EWU", "EWC",
        "EWA", "EWZ", "EWW", "EWY", "EWT", "EWS", "FXI", "KWEB", "MCHI", "EZU",
        "VGK", "ILF", "EPP", "AAXJ",
    ],
    # India context. Required whenever an India stock is the target — without a
    # local cross-section the model would price an NSE name purely off US
    # macro, which is a materially worse conditioning set.
    "India Equity": [
        "^NSEI", "^NSEBANK", "^BSESN", "^CNX100", "^CNXIT", "^CNXAUTO",
        "^CNXPHARMA", "^CNXFMCG", "^CNXMETAL", "^CNXENERGY", "^CNXREALTY",
        "^CNXPSUBANK", "^CNXINFRA", "^CNXMEDIA", "^NSMIDCP", "^CNXSC",
        "NIFTYBEES.NS", "BANKBEES.NS", "JUNIORBEES.NS", "ITBEES.NS",
        "PHARMABEES.NS", "PSUBNKBEES.NS", "CPSEETF.NS", "MON100.NS",
        "INDA", "INDY", "EPI", "SMIN",
    ],
    "Rates & Treasuries": [
        "^IRX", "^FVX", "^TNX", "^TYX", "SHY", "IEI", "IEF", "TLH", "TLT",
        "GOVT", "SHV", "BIL", "SGOV", "VGSH", "VGIT", "VGLT",
    ],
    "Credit": [
        "LQD", "HYG", "JNK", "AGG", "BND", "BNDX", "EMB", "VCSH", "VCIT",
        "VCLT", "MBB", "BKLN", "PFF", "CWB", "FALN", "SJNK",
    ],
    "Inflation-Linked": ["TIP", "VTIP", "SCHP", "STIP"],
    "Commodities": [
        "GC=F", "SI=F", "HG=F", "PL=F", "PA=F", "CL=F", "BZ=F", "NG=F", "RB=F",
        "HO=F", "ZC=F", "ZS=F", "ZW=F", "KC=F", "SB=F", "CC=F", "CT=F", "LE=F",
        "GLD", "SLV", "USO", "UNG", "DBC", "DBA", "DBB", "GSG", "PDBC", "GDX",
        "GDXJ", "COPX", "URA", "LIT", "REMX", "WOOD",
    ],
    "FX": [
        "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X", "USDCAD=X",
        "NZDUSD=X", "USDCNY=X", "USDMXN=X", "USDBRL=X", "USDKRW=X", "USDINR=X",
        "USDSEK=X", "USDNOK=X", "USDZAR=X", "USDTRY=X", "USDTWD=X", "USDSGD=X",
        "DX-Y.NYB", "EURJPY=X", "EURGBP=X", "EURINR=X", "FXE", "FXY", "FXB",
        "FXF", "UUP", "CEW",
    ],
    "Crypto": [
        "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "BNB-USD", "ADA-USD",
        "DOGE-USD", "AVAX-USD", "LINK-USD", "DOT-USD", "LTC-USD", "BCH-USD",
        "TRX-USD", "XLM-USD", "ATOM-USD", "ETC-USD",
    ],
    "Volatility": [
        "^VIX", "^VIX9D", "^VVIX", "^VXN", "^RVX", "^OVX", "^GVZ", "^MOVE",
        "VIXY", "VXX", "SVXY",
    ],
    "Real Assets": ["VNQ", "VNQI", "REET", "SCHH", "IYR", "IGF", "PAVE", "TIPX"],
}

# ---------------------------------------------------------------------------
# Core tier — a liquid, class-representative subset for fast loads
# ---------------------------------------------------------------------------
CORE: set[str] = {
    "^GSPC", "^IXIC", "^DJI", "^RUT", "^NDX", "^FTSE", "^GDAXI", "^STOXX50E",
    "^N225", "^HSI", "^GSPTSE",
    "XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLB", "XLU", "XLRE",
    "XLC", "SMH", "XBI", "KRE", "XOP",
    "SPY", "QQQ", "IWM", "DIA", "VTV", "VUG", "MTUM", "USMV", "SPHB", "RSP",
    "EFA", "EEM", "VWO", "ACWI", "EWJ", "EWG", "EWZ", "EWY", "EWT", "FXI",
    "KWEB", "VGK",
    "^NSEI", "^NSEBANK", "^BSESN", "^CNXIT", "^CNXAUTO", "^CNXPHARMA",
    "^CNXFMCG", "^CNXMETAL", "^NSMIDCP", "^CNXSC", "NIFTYBEES.NS",
    "BANKBEES.NS", "INDA", "EPI",
    "^IRX", "^FVX", "^TNX", "^TYX", "SHY", "IEF", "TLT", "GOVT",
    "LQD", "HYG", "JNK", "AGG", "EMB", "BKLN", "PFF", "TIP",
    "GC=F", "SI=F", "HG=F", "CL=F", "BZ=F", "NG=F", "ZC=F", "ZW=F",
    "GLD", "SLV", "USO", "DBC", "DBA", "GDX", "COPX", "URA",
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X", "USDCAD=X",
    "USDCNY=X", "USDMXN=X", "USDINR=X", "DX-Y.NYB", "UUP", "FXE", "FXY", "CEW",
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "BNB-USD", "DOGE-USD", "LINK-USD",
    "^VIX", "^VXN", "^MOVE", "^OVX", "VIXY",
    "VNQ", "REET", "IGF",
}

# ---------------------------------------------------------------------------
# Target selection
# ---------------------------------------------------------------------------
# Categories whose target is typed in rather than picked from a list.
# Value is the market key consumed by symbol resolution.
FREEFORM_MARKETS: dict[str, str] = {
    "US Stocks": "us",
    "India Stocks": "india",
}

MARKET_HINTS: dict[str, dict[str, str]] = {
    "us": {"placeholder": "e.g. AAPL, MSFT, BRK.B",
           "hint": "US listing · symbol as typed ( . becomes - )"},
    "india": {"placeholder": "e.g. RELIANCE, TCS, TATASTEEL",
              "hint": "NSE (.NS) checked first, then BSE (.BO)"},
}

# Classes offered as dropdown targets. Single-name equity baskets are
# deliberately absent — those are the free-form markets above.
DROPDOWN_TARGET_CLASSES: list[str] = [
    "US Style/Factor ETFs", "Equity Indices", "US Sector ETFs",
    "Global Equity ETFs", "India Equity", "Commodities", "FX", "Crypto",
    "Rates & Treasuries", "Credit", "Volatility", "Real Assets",
    "Inflation-Linked",
]


def target_categories(available: list[str] | None = None) -> dict[str, list[str]]:
    """Category -> selectable targets. Free-form markets map to an empty list."""
    out: dict[str, list[str]] = {name: [] for name in FREEFORM_MARKETS}
    pool = set(available) if available is not None else None
    for cls in DROPDOWN_TARGET_CLASSES:
        syms = UNIVERSE.get(cls, [])
        if pool is not None:
            syms = [s for s in syms if s in pool]
        if syms:
            out[cls] = syms
    return out


# ---------------------------------------------------------------------------
# Instruments quoted as yields rather than prices. Period changes are taken as
# simple differences: a yield moving 0.30 -> 0.05 is a 25bp move, not a -179%
# return, and logs of near-zero yields explode.
# ---------------------------------------------------------------------------
YIELD_SYMBOLS: set[str] = {"^IRX", "^FVX", "^TNX", "^TYX"}

# Reference instruments establishing the trading calendar, tried in order.
# Crypto is excluded on purpose: it trades weekends, and using the union of all
# instrument calendars would insert rows that are empty for every other asset.
CALENDAR_REFERENCES: list[str] = ["SPY", "^GSPC", "QQQ", "IWM", "EFA"]
# For an India target the local calendar governs instead (NSE holidays differ
# from NYSE, and Diwali/Holi sessions would otherwise be dropped entirely).
INDIA_CALENDAR_REFERENCES: list[str] = ["^NSEI", "NIFTYBEES.NS", "^BSESN", "^NSEBANK"]

TAPE: dict[str, str] = {
    "SPY": "US EQ", "QQQ": "NASDAQ", "IWM": "SMALL CAP", "EFA": "INTL EQ",
    "EEM": "EM EQ", "^NSEI": "NIFTY 50", "^NSEBANK": "BANK NIFTY",
    "TLT": "30Y UST", "^TNX": "10Y YIELD", "HYG": "HIGH YIELD", "GLD": "GOLD",
    "CL=F": "WTI", "DX-Y.NYB": "DXY", "EURUSD=X": "EUR/USD",
    "USDINR=X": "USD/INR", "BTC-USD": "BITCOIN", "^VIX": "VIX",
}

CLASS_COLORS: dict[str, str] = {
    "Equity Indices": "#9aa7ff",
    "US Sector ETFs": "#38c7dc",
    "US Style/Factor ETFs": "#4f9cf9",
    "Global Equity ETFs": "#2fd08c",
    "India Equity": "#ff9d5c",
    "Rates & Treasuries": "#f2b544",
    "Credit": "#e0a94a",
    "Inflation-Linked": "#d4b483",
    "Commodities": "#f2884b",
    "FX": "#b7a4f3",
    "Crypto": "#d08cf0",
    "Volatility": "#ff5d6c",
    "Real Assets": "#c58fb0",
    "US Stocks": "#54c9e8",
    "India Stocks": "#ffb37a",
    "Target": "#eef3ff",
    "Latent Factor": "#f2b544",
}

# ---------------------------------------------------------------------------
# Derived lookups
# ---------------------------------------------------------------------------
ALL_SYMBOLS: list[str] = []
CLASS_OF: dict[str, str] = {}
for _cls, _syms in UNIVERSE.items():
    for _s in _syms:
        if _s not in CLASS_OF:
            ALL_SYMBOLS.append(_s)
            CLASS_OF[_s] = _cls

CORE_SYMBOLS: list[str] = [s for s in ALL_SYMBOLS if s in CORE]


def symbols_for_tier(tier: str) -> list[str]:
    """Explanatory instruments for a universe tier ('core' or 'extended')."""
    return list(CORE_SYMBOLS) if str(tier).lower().startswith("core") else list(ALL_SYMBOLS)


def class_breakdown(symbols: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in symbols:
        cls = CLASS_OF.get(s, "Target")
        out[cls] = out.get(cls, 0) + 1
    return out


def color_of(symbol_or_class: str) -> str:
    if symbol_or_class in CLASS_COLORS:
        return CLASS_COLORS[symbol_or_class]
    return CLASS_COLORS.get(CLASS_OF.get(symbol_or_class, ""), "#8fa3c4")


def is_india_symbol(symbol: str) -> bool:
    """True for NSE/BSE listings and India index tickers."""
    if symbol.endswith((".NS", ".BO")):
        return True
    return CLASS_OF.get(symbol) == "India Equity"


def calendar_references(target: str | None = None) -> list[str]:
    """Calendar anchors, ordered so an India target uses the India session."""
    if target and is_india_symbol(target):
        return INDIA_CALENDAR_REFERENCES + CALENDAR_REFERENCES
    return CALENDAR_REFERENCES + INDIA_CALENDAR_REFERENCES
