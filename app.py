# app.py
# ============================================================================
#  FVE — FAIR VALUE ENGINE
#  Market-relative valuation across a 200+ instrument cross-asset universe.
#  Run:  streamlit run app.py
# ============================================================================
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA, FastICA, FactorAnalysis
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="FVE — Fair Value Engine", layout="wide",
                    initial_sidebar_state="expanded")

# Hide the main Streamlit header and menu bar for a cleaner layout
st.markdown("""<style>
    #header {visibility: hidden;}
    .css-18e3th9 {display: none;}
    .css-1d391u8 {display: none;}
    </style>""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# 1. UNIVERSE DEFINITION  (212 instruments, 9 asset classes)
# ----------------------------------------------------------------------------
UNIVERSE = {
    "US Equities": """AAPL MSFT NVDA GOOGL AMZN META TSLA AVGO AMD INTC CRM ORCL ADBE NFLX PANW
CRWD PLTR SNOW NET DDOG ZS OKTA QCOM TXN MU AMAT LRCX KLAC SNPS CDNS V MA JPM BAC GS MS WFC
C AXP BLK SCHW COIN HOOD PYPL XYZ UBER ABNB SHOP SPOT RBLX DKNG PINS SNAP ROKU F GM T VZ DIS
CMCSA CHTR KO PEP WMT COST MCD NKE SBUX TGT HD LOW CAT DE GE HON MMM UPS RTX LMT NOC BA GD UNP
CSX NSC XOM CVX COP SLB OXY JNJ PFE UNH MRK ABBV LLY TMO ABT DHR BMY AMGN GILD ISRG MDT SYK
BSX EL CL GIS K MBT?""",
    "Intl Equities": """TSM ASML NVO AZN SHEL BHP RIO VALE PBR SAP SIE.DE MC.PA TTE.PA NESN.SW
7203.T 6758.T 8306.T 9984.T 005930.KS RELIANCE.NS""",
    "Indices": """^GSPC ^IXIC ^DJI ^RUT ^FTSE ^GDAXI ^N225 ^HSI ^STOXX50E ^GSPTSE ^BSESN ^KS11""",
    "ETFs": """SPY QQQ IWM DIA XLF XLK XLE XLV XLI XLP XLU XLB XLRE XLC TLT IEF HYG LQD TIP
AGG BND GLD SLV USO VNQ GDX KWEB EEM EFA VWO EWJ FXI""",
    "FX": """EURUSD=X GBPUSD=X USDJPY=X USDCHF=X AUDUSD=X USDCAD=X NZDUSD=X USDCNH=X DX-Y.NYB
EURJPY=X""",
    "Rates": """^IRX ^FVX ^TNX ^TYX ZB=F ZN=F ZF=F ZT=F""",
    "Commodities": """CL=F BZ=F NG=F HG=F GC=F SI=F PL=F PA=F ZC=F ZS=F ZW=F KC=F CC=F CT=F SB=F
ZL=F""",
    "Crypto": """BTC-USD ETH-USD SOL-USD XRP-USD ADA-USD DOGE-USD AVAX-USD DOT-USD LINK-USD
MATIC-USD LTC-USD BCH-USD BNB-USD TRX-USD""",
    "Volatility": """^VIX ^VXN ^MOVE VIXY UVXY SVXY""",
}
# Fix known issues in the original strings
UNIVERSE["US Equities"] = UNIVERSE["US Equities"].replace("K ", "").replace("MBT?", "MBT")
UNIVERSE["FX"] = UNIVERSE["FX"].replace("USDCNH=X ", "")

# ---- Extended universe (live yfinance tickers) ----
UNIVERSE["US Treasuries"] = """BIL SHV SGOV SHY VGSH IEI IEF VGIT TLH TLT VGLT GOVT"""
UNIVERSE["Yield Indices"] = """^IRX ^FVX ^TNX ^TYX"""
UNIVERSE["Inflation-Protected"] = """TIP VTIP WIP"""
UNIVERSE["Aggregate Bonds"] = """BSV BLV AGG BND FLOT BNDW BNDX"""
UNIVERSE["Corporate IG"] = """LQD VCSH VCIT VCLT"""
UNIVERSE["High Yield"] = """HYG JNK GHYG BGRN PFF CWB FALN"""
UNIVERSE["Structured"] = """MBB VMBS BKLN"""
UNIVERSE["Municipals"] = """MUB VTEB"""
UNIVERSE["Intl Govt Bonds"] = """IGOV BWX IBND IEGA.L IEAC.L IBGL.L SDEU.L IGLT.L INXG.L SLXX.L"""
UNIVERSE["APAC Bonds"] = """VGB.AX XBB.TO"""
UNIVERSE["Equity Benchmarks"] = """ACWI EFA EW EM EWJ EZU EWY EWT EWU"""
UNIVERSE["Regional Equity"] = """VNM EPHE EIDO EWS UAE INDA"""
UNIVERSE["Country Indices"] = """^GDAXI ^FCHI ^STOXX50E ^FTSE ^IBEX ^AEX ^SSMI"""
UNIVERSE["Volatility"] = """^VIX ^MOVE VIXM"""
UNIVERSE["Energy/Commodity Equities"] = """EWZ EWA"""
UNIVERSE["Sectors"] = """XLB XME PICK XLE XLI XLF KRE GLTR PALL LIT URA SLX REMX WOOD IGF"""
UNIVERSE["FX Major"] = """DX-Y.NYB UDN USDU FXE FXY FXB FXF FXA FXC CEW"""
UNIVERSE["FX INR"] = """INR=X EURINR=X GBPINR=X JPYINR=X AUDINR=X NZDINR=X CADINR=X CHFINR=X CNYINR=X SGDINR=X HKDINR=X INRUSD=X BDT=X"""
UNIVERSE["FX Asia EM"] = """CNY=X CNH=X JPY=X KRW=X MXN=X BRL=X ZAR=X THB=X TWD=X MYR=X"""
UNIVERSE["Style Factors"] = """VTV VUG MTUM USMV SPHB VYM"""
UNIVERSE["REITs"] = """VNQ VNQI REET"""
UNIVERSE["Commodity Baskets"] = """DBC GSG DBB DBA GLTR"""
UNIVERSE["Thematic"] = """SMH RINF HYDR"""

CLASS_COLORS = {
    "US Equities": "#4f9cf9", "Intl Equities": "#7fb3ff", "Indices": "#9aa7ff",
    "ETFs": "#38c7dc", "FX": "#5fd39a", "Rates": "#f2b544",
    "Commodities": "#f2884b", "Crypto": "#b7a4f3", "Volatility": "#ff5d6c",
    "US Treasuries": "#ffe5b4", "Yield Indices": "#ff9999",
    "Inflation-Protected": "#bebada", "Aggregate Bonds": "#d4b8ff",
    "Corporate IG": "#cce5ff", "High Yield": "#ffcccc",
    "Structured": "#d4edda", "Municipals": "#fff3cd",
    "Intl Govt Bonds": "#e2e3e5", "APAC Bonds": "#f8d7da",
    "Equity Benchmarks": "#c3e6cb", "Regional Equity": "#d1ecf1",
    "Country Indices": "#f5f5f5", "FX Major": "#e8e8e8",
    "FX INR": "#e8e8e8", "FX Asia EM": "#e8e8e8",
    "Style Factors": "#e8e8e8", "REITs": "#e8e8e8",
    "Commodity Baskets": "#e8e8e8", "Thematic": "#e8e8e8",
    "Energy/Commodity Equities": "#e8e8e8", "Sectors": "#e8e8e8",
}

for cls, blob in list(UNIVERSE.items()):
    if cls not in CLASS_COLORS:
        CLASS_COLORS[cls] = "#888888"

SYMBOLS, CLASSES = [], []
for _cls, _blob in UNIVERSE.items():
    for _s in _blob.split():
        if _s and _s not in SYMBOLS:
            SYMBOLS.append(_s); CLASSES.append(_cls)
N_ASSETS = len(SYMBOLS)
CLASS_OF = dict(zip(SYMBOLS, CLASSES))

REGIME_META = {
    "RISK-ON":   {"color": "#2fd08c", "icon": "▲"},
    "TREND":     {"color": "#38c7dc", "icon": "↗"},
    "MEAN-REV":  {"color": "#b7a4f3", "icon": "⇄"},
    "HIGH-VOL":  {"color": "#f2b544", "icon": "≈"},
    "RISK-OFF":  {"color": "#ff5d6c", "icon": "▼"},
}

# ----------------------------------------------------------------------------# ----------------------------------------------------------------------------
# 2. THEME + CSS SHELL
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"]  { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background:
  radial-gradient(1100px 700px at 88% -12%, rgba(56,132,255,.10), transparent 60%),
  radial-gradient(900px 650px at -8% 108%, rgba(242,181,68,.07), transparent 55%),
  radial-gradient(700px 500px at 55% 118%, rgba(63,216,201,.05), transparent 60%),
  #0c1322; }
.stApp::before { content:""; position:fixed; inset:0; pointer-events:none; z-index:0;
  background-image: linear-gradient(rgba(120,150,210,.045) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(120,150,210,.045) 1px, transparent 1px);
  background-size: 44px 44px;
  mask-image: radial-gradient(1200px 800px at 50% 0%, black 30%, transparent 85%); }
.block-container { padding-top: 1.4rem; }
div[data-testid="stVerticalBlock"] { position: relative; z-index: 1; }

/* ---------- header ---------- */
.fve-head { display:flex; align-items:flex-end; justify-content:space-between; gap:16px;
  flex-wrap:wrap; margin-bottom:6px; }
.fve-brand { font-family:'Space Grotesk'; font-weight:700; font-size:30px; letter-spacing:-.02em;
  color:#eef3ff; line-height:1; }
.fve-brand em { font-style:normal; color:#f2b544; }
.fve-sub { font-family:'IBM Plex Mono'; font-size:11px; letter-spacing:.16em; color:#7f93b8;
  text-transform:uppercase; margin-top:7px; }
.fve-chips { display:flex; gap:8px; flex-wrap:wrap; }
.chip { font-family:'IBM Plex Mono'; font-size:10.5px; letter-spacing:.08em; color:#a9bcdd;
  border:1px solid rgba(140,165,215,.22); background:rgba(18,28,46,.7); padding:5px 10px;
  border-radius:6px; display:inline-flex; align-items:center; gap:7px; transition:.25s; }
.chip:hover { border-color:rgba(63,216,201,.5); color:#dff6f2; transform:translateY(-1px); }
.dot { width:7px; height:7px; border-radius:50%; background:#2fd08c;
  box-shadow:0 0 0 0 rgba(47,208,140,.6); animation:pulse 1.8s infinite; }
@keyframes pulse { 70% { box-shadow:0 0 0 8px rgba(47,208,140,0);} 100% { box-shadow:0 0 0 0 rgba(47,208,140,0);} }

/* ---------- ticker tape ---------- */
.tape-wrap { overflow:hidden; border-top:1px solid rgba(140,165,215,.14);
  border-bottom:1px solid rgba(140,165,215,.14); margin:12px 0 20px; padding:8px 0;
  mask-image:linear-gradient(90deg, transparent, black 6%, black 94%, transparent); }
.tape { display:inline-flex; gap:34px; white-space:nowrap; animation:tape 38s linear infinite;
  font-family:'IBM Plex Mono'; font-size:12px; }
.tape-wrap:hover .tape { animation-play-state:paused; }
@keyframes tape { from{transform:translateX(0)} to{transform:translateX(-50%)} }
.tape b { color:#c9d7f2; font-weight:500; } .tape .up{color:#2fd08c} .tape .dn{color:#ff5d6c}

/* ---------- KPI cards ---------- */
.kpis { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:12px; }
.kpi { background:linear-gradient(180deg, rgba(21,32,52,.92), rgba(15,23,39,.92));
  border:1px solid rgba(140,165,215,.16); border-radius:10px; padding:14px 16px 13px;
  transition:transform .22s, border-color .22s, box-shadow .22s; position:relative; overflow:hidden; }
.kpi:hover { transform:translateY(-3px); border-color:rgba(63,216,201,.45);
  box-shadow:0 10px 28px -14px rgba(63,216,201,.35); }
.kpi .lab { font-family:'IBM Plex Mono'; font-size:9.5px; letter-spacing:.16em; color:#7f93b8;
  text-transform:uppercase; margin-bottom:8px; }
.kpi .val { font-family:'IBM Plex Mono'; font-size:27px; font-weight:600; color:#eef3ff;
  line-height:1.05; }
.kpi .aux { font-size:11.5px; margin-top:7px; color:#93a7c9; }
.kpi .aux .up{color:#2fd08c; font-weight:600} .kpi .aux .dn{color:#ff5d6c; font-weight:600}
.kpi .aux .amber{color:#f2b544; font-weight:600}
.kpi.accent-amber { border-top:2px solid #f2b544; } .kpi.accent-cyan { border-top:2px solid #3fd8c9; }
.kpi.accent-green { border-top:2px solid #2fd08c; } .kpi.accent-red { border-top:2px solid #ff5d6c; }
.kpi.accent-blue { border-top:2px solid #4f9cf9; } .kpi.accent-violet { border-top:2px solid #b7a4f3; }
.meter { height:5px; border-radius:3px; background:rgba(140,165,215,.15); margin-top:10px;
  overflow:hidden; }
.meter i { display:block; height:100%; border-radius:3px; transition:width .8s ease; }
.pill { display:inline-block; font-family:'IBM Plex Mono'; font-size:11px; letter-spacing:.1em;
  padding:4px 11px; border-radius:20px; font-weight:600; }

/* ---------- section headers ---------- */
.sec-kick { font-family:'IBM Plex Mono'; font-size:10px; letter-spacing:.22em; color:#f2b544;
  text-transform:uppercase; }
.sec-title { font-family:'Space Grotesk'; font-size:21px; font-weight:600; color:#eef3ff;
  margin:2px 0 2px; letter-spacing:-.01em; }
.sec-desc { font-size:12.5px; color:#8ba0c4; margin-bottom:8px; }

/* ---------- streamlit chrome ---------- */
[data-testid="stTabs"] [data-baseweb="tab-list"] { gap:6px; border-bottom:1px solid rgba(140,165,215,.18); }
[data-testid="stTabs"] [data-baseweb="tab"] { background:transparent; border-radius:8px 8px 0 0;
  padding:8px 16px; font-family:'Space Grotesk'; font-size:13px; color:#93a7c9; }
[data-testid="stTabs"] [aria-selected="true"] { background:rgba(63,216,201,.08); color:#eef3ff !important;
  box-shadow: inset 0 -2px 0 #3fd8c9; }
.stButton > button { width:100%; background:linear-gradient(180deg,#17324e,#122档); }
.stButton > button { background:linear-gradient(180deg, #17324e, #12253c); color:#dff6f2;
  border:1px solid rgba(63,216,201,.4); border-radius:8px; font-family:'Space Grotesk';
  font-weight:600; letter-spacing:.04em; transition:.25s; }
.stButton > button:hover { border-color:#3fd8c9; box-shadow:0 0 18px -4px rgba(63,216,201,.5);
  transform:translateY(-1px); color:#fff; }
div[data-baseweb="select"] > div, .stTextInput input, .stTextArea textarea {
  background:rgba(18,28,46,.85) !important; border-color:rgba(140,165,215,.25) !important; }
[data-testid="stMetric"] { display:none; }
footer { visibility:hidden; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# 3. PLOTTING HELPERS
# ----------------------------------------------------------------------------
AX = dict(gridcolor="rgba(140,165,215,.08)", zerolinecolor="rgba(140,165,215,.14)",
          tickfont=dict(family="IBM Plex Mono", size=10, color="#8fa3c4"),
          title_font=dict(family="IBM Plex Mono", size=10, color="#7f93b8"))
def fig_base(fig, height=420, legend=True):
    fig.update_layout(
        template="plotly_dark", height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,20,35,.55)",
        margin=dict(l=8, r=8, t=30, b=8), hoverlabel=dict(bgcolor="#101a2c",
            bordercolor="rgba(63,216,201,.4)", font=dict(family="IBM Plex Mono", size=11)),
        showlegend=legend, legend=dict(orientation="h", y=1.08, x=0,
            font=dict(family="IBM Plex Mono", size=10, color="#a9bcdd")),
        xaxis=dict(**AX), yaxis=dict(**AX))
    return fig

def fmt_px(v):
    a = abs(v)
    if a >= 10000: return f"{v:,.0f}"
    if a >= 100:   return f"{v:,.2f}"
    if a >= 1:     return f"{v:.3f}"
    return f"{v:.5f}"

# ----------------------------------------------------------------------------
# 4. DATA LAYER
# ----------------------------------------------------------------------------
FACTOR_NAMES = ["MKT", "RATES", "INFL", "MOM", "USD", "CRY"]

def _class_loadings(cls, sym, rng):
    L = dict(zip(FACTOR_NAMES, [0.0]*6))
    if cls in ("US Equities", "Intl Equities"):
        L = dict(MKT=1.0, RATES=-0.25, INFL=0.10, MOM=0.35, USD=-0.10, CRY=0.0)
        if sym in ("NVDA","TSLA","AMD","PLTR","COIN","CRWD","SNOW","MSTR","HOOD","RBLX"):
            L["MKT"], L["MOM"], L["CRY"] = 1.45, 0.55, 0.25
        if sym in ("XOM","CVX","COP","SLB","OXY"): L["INFL"] = 0.55
        if cls == "Intl Equities": L["USD"] = -0.35
    elif cls == "Indices":
        L = dict(MKT=1.0, RATES=-0.15, INFL=0.05, MOM=0.30, USD=-0.15, CRY=0.0)
    elif cls == "ETFs":
        L = dict(MKT=0.9, RATES=-0.10, INFL=0.10, MOM=0.25, USD=-0.05, CRY=0.0)
        if sym in ("TLT","IEF","AGG","BND"): L = dict(MKT=0.10, RATES=0.85, INFL=0.15, MOM=0.1, USD=0.0, CRY=0.0)
        if sym in ("HYG","LQD","TIP"):       L = dict(MKT=0.35, RATES=0.60, INFL=0.20, MOM=0.1, USD=0.0, CRY=0.0)
        if sym in ("GLD","SLV","GDX"):       L = dict(MKT=0.15, RATES=0.10, INFL=0.75, MOM=0.2, USD=-0.55, CRY=0.0)
        if sym in ("USO","XLE"):             L = dict(MKT=0.30, RATES=0.0,  INFL=0.95, MOM=0.2, USD=-0.35, CRY=0.0)
        if sym in ("VNQ","XLRE"):            L["RATES"] = 0.45
        if sym in ("KWEB","FXI","EEM","VWO","EWJ","EFA"): L["USD"] = -0.40
    elif cls == "FX":
        usd_short = any(sym.startswith(p) for p in ("EUR","GBP","AUD","NZD"))
        L = dict(MKT=0.10, RATES=0.20, INFL=0.05, MOM=0.15, USD=-0.90 if usd_short else 0.90, CRY=0.0)
    elif cls == "Rates":
        L = dict(MKT=-0.10, RATES=0.90, INFL=0.25, MOM=0.10, USD=0.05, CRY=0.0)
    elif cls == "Commodities":
        L = dict(MKT=0.20, RATES=0.05, INFL=0.85, MOM=0.20, USD=-0.40, CRY=0.0)
    elif cls == "Crypto":
        L = dict(MKT=0.45, RATES=-0.05, INFL=0.05, MOM=0.50, USD=-0.10, CRY=0.95)
    elif cls == "Volatility":
        L = dict(MKT=-1.25, RATES=-0.05, INFL=0.0, MOM=-0.20, USD=0.0, CRY=-0.05)
    noise = rng.normal(0, 0.12, 6)
    return np.array([L[k] for k in FACTOR_NAMES]) + noise

@st.cache_data(show_spinner=False, max_entries=6)
def simulate_universe(seed, n_days):
    """Latent-factor DGP with a Markov regime chain — deterministic per seed."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end="2026-07-31", periods=n_days)
    n = N_ASSETS
    # --- regime chain: 0 RISK-ON  1 TREND  2 MEAN-REV  3 HIGH-VOL  4 RISK-OFF
    P = np.array([[.965,.015,.008,.008,.004],
                  [.012,.965,.012,.007,.004],
                  [.010,.014,.960,.010,.006],
                  [.012,.008,.010,.950,.020],
                  [.010,.004,.006,.025,.955]])
    reg = np.zeros(n_days, dtype=int)
    for t in range(1, n_days):
        reg[t] = rng.choice(5, p=P[reg[t-1]])
    # --- factor shocks (regime-dependent drift / vol)
    mu = {  "MKT":  [.0009,.0003,.0000,.0000,-.0012],
            "RATES":[.0001,.0001,-.0001,.0002,-.0003],
            "INFL": [.0001,.0001,.0002,.0004,.0002],
            "MOM":  [.0002,.0002,.0000,.0000,-.0004],
            "USD":  [-.0001,.0000,.0001,.0002,.0004],
            "CRY":  [.0020,.0005,.0000,.0005,-.0030]}
    sd = {  "MKT":  [.0060,.0045,.0050,.0110,.0095],
            "RATES":[.0030,.0025,.0030,.0050,.0045],
            "INFL": [.0050,.0040,.0045,.0080,.0070],
            "MOM":  [.0040,.0035,.0030,.0060,.0055],
            "USD":  [.0030,.0025,.0025,.0045,.0040],
            "CRY":  [.0220,.0160,.0140,.0300,.0260]}
    F = np.column_stack([rng.normal(np.array(mu[k])[reg], np.array(sd[k])[reg]) for k in FACTOR_NAMES])
    F[:, 3] = pd.Series(F[:, 3]).ewm(alpha=.12).mean().values          # momentum persistence
    beta_mult = np.array([1.0, 1.0, 1.0, 1.35, 1.6])[reg]              # beta expansion in stress
    idio_mult = np.array([0.85, 0.9, 1.0, 1.5, 1.35])[reg]
    # --- per-asset structure
    B = np.vstack([_class_loadings(CLASSES[i], SYMBOLS[i], rng) for i in range(n)])
    idio_base = np.array([
        {"US Equities":.011,"Intl Equities":.012,"Indices":.008,"ETFs":.009,"FX":.005,
         "Rates":.004,"Commodities":.013,"Crypto":.028,"Volatility":.05}[c] for c in CLASSES])
    alpha = rng.normal(0.0002, .0004, n)
    R = alpha + (F @ B.T) * beta_mult[:, None] + rng.normal(0, 1, (n_days, n)) * (idio_base * idio_mult[:, None])
    p0 = np.empty(n)
    for i, (s, c) in enumerate(zip(SYMBOLS, CLASSES)):
        if c == "Crypto":      p0[i] = np.exp(rng.uniform(np.log(.08), np.log(60000)))
        elif c == "FX":        p0[i] = 150.0 if "JPY" in s else rng.uniform(.6, 1.7)
        elif c == "Rates":     p0[i] = rng.uniform(.5, 5.0)
        elif c == "Volatility":p0[i] = rng.uniform(12, 35)
        elif c == "Indices":   p0[i] = rng.uniform(3000, 42000)
        elif c == "Commodities":p0[i] = 2400.0 if "GC" in s else rng.uniform(18, 260)
        else:                  p0[i] = np.exp(rng.uniform(np.log(25), np.log(700)))
    prices = pd.DataFrame(p0 * np.exp(np.cumsum(R, 0)), index=dates, columns=SYMBOLS)
    return prices, pd.Series(reg, index=dates)

@st.cache_data(show_spinner=False, max_entries=2)
def load_live_universe():
    """Fetch the full live universe from Yahoo Finance (max available history)."""
    import yfinance as yf
    raw = yf.download(SYMBOLS, period="max", auto_adjust=True, progress=False, threads=True)
    px = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
    px = px.ffill().dropna(axis=1, thresh=int(0.7 * len(px)))
    if px.shape[1] < 120:
        raise RuntimeError(f"only {px.shape[1]} tickers returned")
    return px

def get_prices():
    """Fetch live prices via yfinance — no simulation fallback."""
    prices = load_live_universe()
    return prices, "LIVE"

# ----------------------------------------------------------------------------
# 5. STATISTICAL CORE
# ----------------------------------------------------------------------------
def rolling_zscore(df, w=63, minp=30):
    mu = df.rolling(w, min_periods=minp).mean()
    sd = df.rolling(w, min_periods=minp).std().replace(0, np.nan)
    return ((df - mu) / sd).fillna(0.0)

def detect_regimes(rz, pc1):
    """KMeans on market features → labelled regimes (data-driven, not the DGP's labels)."""
    mkt = pc1.rolling(21).mean()
    vol = pc1.rolling(21).std()
    disp = rz.std(axis=1).rolling(21).mean()
    X = pd.concat([mkt, vol, disp], axis=1).dropna()
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=5, n_init=10, random_state=7).fit(Xs)
    lab = pd.Series(km.labels_, index=X.index)
    names = {}
    stats_c = pd.DataFrame({c: [lab.eq(c).mean(),
                                X.loc[lab.eq(c), X.columns[0]].mean(),
                                X.loc[lab.eq(c), X.columns[1]].mean(),
                                X.loc[lab.eq(c), X.columns[2]].mean()]
                            for c in range(5)}, index=["share","ret","vol","disp"]).T
    order_vol = stats_c["vol"].sort_values(ascending=False).index.tolist()
    hv = order_vol[0]; names[hv] = "HIGH-VOL"
    rest = [c for c in order_vol[1:]]
    rets = stats_c.loc[rest, "ret"]
    ro, ron = rets.idxmin(), rets.idxmax()
    names[ro], names[ron] = "RISK-OFF", "RISK-ON"
    mid = [c for c in rest if c not in (ro, ron)]
    d0, d1 = stats_c.loc[mid, "disp"].sort_values(ascending=False).index.tolist()
    names[d0], names[d1] = "MEAN-REV", "TREND"
    full = pd.Series("TREND", index=rz.index)
    full.loc[lab.index] = lab.map(names)
    return full

def fit_factor_model(Xw, method, ncomp):
    Xc = Xw - Xw.mean(0)
    if method == "PCA":
        m = PCA(n_components=ncomp, random_state=0)
    elif method == "FastICA":
        m = FastICA(n_components=ncomp, random_state=0, max_iter=500)
    else:
        m = FactorAnalysis(n_components=ncomp, random_state=0)
    S = m.fit_transform(Xc)
    ev = (m.explained_variance_ratio_ if hasattr(m, "explained_variance_ratio_")
          and m.explained_variance_ratio_ is not None and len(m.explained_variance_ratio_) == ncomp
          else np.full(ncomp, np.nan))
    return m, S, ev

def run_engine(prices, target, method, ncomp, lookback, refit, peers_k, alpha,
               decay, regime_weight, norm_win, bound, smooth, span, thr_q):
    t0 = time.time()
    status = st.status("Calibrating Fair Value Engine…", expanded=True)
    def step(msg): status.write(f"▸ {msg}")

    step(f"Preparing {prices.shape[1]}-instrument return matrix…")
    rets = np.log(prices).diff().fillna(0)
    rz = rolling_zscore(rets)
    others = [c for c in rets.columns if c != target]

    step(f"Extracting {ncomp} latent factors ({method}) + classifying regimes…")
    m0, S0, _ = fit_factor_model(rz[others].iloc[:lookback].values, method, ncomp)
    pc1 = pd.Series(m0.transform(rz[others].values)[:, 0], index=rets.index)
    regimes = detect_regimes(rz, pc1)

    step("Walk-forward fair-value regression (regime-weighted)…")
    T = len(rets)
    yhat = np.full(T, np.nan)
    chunk_regimes, chunk_r2, chunk_beta = [], [], []
    last_model = None
    t = lookback
    while t < T:
        w = slice(t - lookback, t)
        Xw = rz[others].iloc[w].values
        m, S, _ = fit_factor_model(Xw, method, ncomp)
        corr = rets[others].iloc[w].corrwith(rets[target].iloc[w]).abs()
        peers = corr.sort_values(ascending=False).head(peers_k).index.tolist()
        X = np.hstack([S, rz[peers].iloc[w].values])
        y = rets[target].iloc[w].values
        cur_reg = regimes.iloc[t - 1]
        age = np.arange(lookback)[::-1]
        wts = np.power(decay, age / 21.0)
        if regime_weight:
            wts = wts * np.where(regimes.iloc[w].values == cur_reg, 1.6, 0.7)
        mdl = Ridge(alpha=alpha).fit(X, y, sample_weight=wts)
        end = min(t + refit, T)
        Xf = np.hstack([m.transform(rz[others].iloc[t:end].values),
                        rz[peers].iloc[t:end].values])
        yhat[t:end] = Xf @ mdl.coef_ + mdl.intercept_
        in_r2 = mdl.score(X, y, sample_weight=wts)
        chunk_regimes += [cur_reg] * (end - t)
        chunk_r2 += [in_r2] * (end - t)
        chunk_beta.append((mdl.coef_.copy(), peers, t))
        last_model = (mdl, m, peers, w, in_r2)
        t = end

    step("Assembling fair value path, oscillator & diagnostics…")
    price = prices[target]
    start = lookback
    fv = pd.Series(np.nan, index=rets.index)
    fv.iloc[start:] = price.iloc[start] * np.exp(np.cumsum(yhat[start:] - rets[target].iloc[start]))
    # fair value accumulates model-implied returns from the anchor price
    impl = pd.Series(yhat, index=rets.index).fillna(0.0)
    fv = price.iloc[start] * np.exp(impl.iloc[start:].cumsum())
    mis = np.log(price / fv)                                   # log mispricing
    mispct = (np.exp(mis) - 1) * 100
    gap = price - fv

    mu = mis.rolling(norm_win, min_periods=20).mean()
    sd = mis.rolling(norm_win, min_periods=20).std().replace(0, np.nan)
    z = ((mis - mu) / sd).fillna(0.0)
    fvo_raw = 100 * np.tanh(z / bound)

    if smooth == "EMA":
        fvo_s = fvo_raw.ewm(span=span, min_periods=1).mean()
        band = (fvo_raw - fvo_s).rolling(42, min_periods=10).std()
    elif smooth == "Kalman":
        q, r_ = 0.8, max(np.nanvar(np.diff(fvo_raw.values)), 1e-6) * span
        x = fvo_raw.values.astype(float); out = np.zeros_like(x); P = np.zeros_like(x)
        xv, Pv = x[0], 1.0
        for i, v in enumerate(x):
            Pv += q; K = Pv / (Pv + r_); xv += K * (v - xv); Pv *= (1 - K)
            out[i], P[i] = xv, Pv
        fvo_s = pd.Series(out, index=fvo_raw.index)
        band = 1.96 * np.sqrt(P)
    else:
        fvo_s, band = fvo_raw.copy(), fvo_raw.rolling(42, min_periods=10).std()
    band = pd.Series(band, index=fvo_raw.index).fillna(8)

    ob = fvo_s.expanding(min_periods=60).quantile(1 - thr_q)
    os_ = fvo_s.expanding(min_periods=60).quantile(thr_q)

    # rolling fit & residual diagnostics
    resid = rets[target] - pd.Series(yhat, index=rets.index)
    r2 = (1 - resid.pow(2).rolling(63).mean() / rets[target].pow(2).rolling(63).mean()).clip(0, 1)
    sigma_e = resid.rolling(42).std() * 100
    std_res = (resid / resid.rolling(63).std()).fillna(0)
    cus, brk, s = [], [], 0.0
    for i, e in enumerate(std_res.values):
        s = max(0.0, s + abs(e) - 0.5)
        cus.append(s)
        if s > 6: brk.append(i); s = 0.0
        else: brk.append(np.nan)
    cusum = pd.Series(cus, index=resid.index)

    # confidence score
    # Convert regime labels to numeric codes for rolling calculation
    regime_codes = regimes.astype('category').cat.codes
    reg_purity = regime_codes.rolling(21).apply(lambda x: (x == x.iloc[-1]).mean(), raw=False)
    beta_drift = pd.Series(np.nan, index=rets.index)
    prev = None
    for coef, _, tidx in chunk_beta:
        d = 0.0 if prev is None else float(np.mean(np.abs(coef - prev)))
        beta_drift.iloc[tidx] = d; prev = coef
    beta_drift = beta_drift.ffill().fillna(0)
    stab = np.exp(-8 * beta_drift)
    conf = (100 * (0.55 * r2.ffill().fillna(0) + 0.25 * reg_purity + 0.20 * stab)).clip(5, 99)

    # OU half-life → mean-reversion probability
    dmis = mis.diff(); lag = mis.shift(1)
    df_ou = pd.concat([dmis, lag], axis=1).dropna()
    theta = pd.Series(np.nan, index=mis.index)
    vals = df_ou.values
    for i in range(63, len(df_ou), 5):
        yy, xx = vals[i-63:i, 0], vals[i-63:i, 1]
        b = np.polyfit(xx, yy, 1)[0]
        theta.iloc[df_ou.index.get_loc(df_ou.index[i])] = -b
    theta = theta.ffill().clip(0.005, 0.6)
    halflife = (np.log(2) / theta).clip(2, 120)
    horizon = st.session_state.get("mr_horizon", 10)
    p_mr = ((1 - np.exp(-theta * horizon)) *
            (1 / (1 + np.exp(-(z.abs() - 1.2) / 0.6))) * (0.4 + 0.6 * conf / 100) * 100).clip(0, 99)

    # signals & divergences
    sig_dn = (fvo_s.shift(1) > os_.shift(1)) & (fvo_s <= os_)
    sig_up = (fvo_s.shift(1) < ob.shift(1)) & (fvo_s >= ob)
    phi, plo = price.rolling(20).max(), price.rolling(20).min()
    fhi, flo = fvo_s.rolling(20).max(), fvo_s.rolling(20).min()
    div_bear = (price >= phi.shift(1)) & (fvo_s < fhi.shift(1) - 6)
    div_bull = (price <= plo.shift(1)) & (fvo_s > flo.shift(1) + 6)
    for arr in (div_bear, div_bull):
        idx = np.flatnonzero(arr.values); last = -30
        for i in idx:
            if i - last < 15: arr.iloc[i] = False
            else: last = i

    # explainability — exact linear attribution on the final window
    mdl, m, peers, w, in_r2 = last_model
    feat_names = [f"PC{i+1}" for i in range(ncomp)] + peers
    Xlast = np.hstack([m.transform(rz[others].iloc[-63:].values), rz[peers].iloc[-63:].values])
    ylast = rets[target].iloc[-63:].values
    contrib = pd.DataFrame(Xlast * mdl.coef_, index=rets.index[-63:], columns=feat_names) * 1e4  # bp/day
    base_r2 = mdl.score(Xlast, ylast)
    rng_l = np.random.default_rng(1)
    perm = {}
    for j, name in enumerate(feat_names):
        Xp = Xlast.copy(); Xp[:, j] = rng_l.permutation(Xp[:, j])
        perm[name] = max(base_r2 - mdl.score(Xp, ylast), 0.0)
    perm_imp = pd.Series(perm).sort_values(ascending=False)
    factor_exp = pd.Series(mdl.coef_[:ncomp] * 1e4, index=feat_names[:ncomp])
    comp_var = (m.explained_variance_ratio_ * 100) if hasattr(m, "explained_variance_ratio_") \
               and m.explained_variance_ratio_ is not None else np.full(ncomp, np.nan)

    # multi-timeframe oscillator
    mtf = {}
    for lab_, w_ in (("Intraday / 21d", 21), ("Swing / 63d", 63), ("Position / 126d", 126)):
        zw = ((mis - mis.rolling(w_, min_periods=10).mean()) /
              mis.rolling(w_, min_periods=10).std().replace(0, np.nan)).fillna(0)
        mtf[lab_] = 100 * np.tanh(zw / bound)

    status.update(label=f"Engine calibrated in {time.time()-t0:.1f}s", state="complete", expanded=False)
    return dict(dates=rets.index, price=price, fv=fv, gap=gap, mis=mis, mispct=mispct,
                z=z, fvo_raw=fvo_raw, fvo=fvo_s, band=band, ob=ob, os_=os_, r2=r2,
                sigma_e=sigma_e, cusum=cusum, breaks=np.array(brk, dtype=float),
                regimes=regimes, sig_dn=sig_dn, sig_up=sig_up, div_bear=div_bear,
                div_bull=div_bull, conf=conf, p_mr=p_mr, halflife=halflife,
                contrib=contrib, perm_imp=perm_imp, factor_exp=factor_exp,
                comp_var=comp_var, mtf=mtf, rets=rets, peers=peers, in_r2=in_r2,
                ncomp=ncomp, method=method, n_assets=prices.shape[1])

# ----------------------------------------------------------------------------
# 6. BACKTESTER
# ----------------------------------------------------------------------------
def backtest(E, allow_short, exit_lvl, max_hold):
    fvo, ob, os_ = E["fvo"].values, E["ob"].values, E["os_"].values
    r = E["rets"][ [c for c in E["rets"].columns if c == E["price"].name][0] ].values \
        if E["price"].name in E["rets"].columns else E["rets"].iloc[:, 0].values
    r = E["rets"].get(E["price"].name, E["rets"].iloc[:, 0]).values
    T = len(fvo); pos, entry, held = 0, 0, 0
    strat = np.zeros(T); trades = []
    for t in range(T - 1):
        strat[t + 1] = pos * r[t + 1]
        if pos == 0:
            if np.isfinite(os_[t]) and fvo[t] < os_[t]:
                pos, entry, held = 1, t, 0
            elif allow_short and np.isfinite(ob[t]) and fvo[t] > ob[t]:
                pos, entry, held = -1, t, 0
        else:
            held += 1
            exit_long = pos == 1 and (fvo[t] >= exit_lvl or held >= max_hold)
            exit_shrt = pos == -1 and (fvo[t] <= -exit_lvl or held >= max_hold)
            if exit_long or exit_shrt:
                pr = E["price"].values
                ret_t = pos * (np.log(pr[t]) - np.log(pr[entry]))
                trades.append(dict(entry=E["dates"][entry].date(), exit=E["dates"][t].date(),
                                   side="LONG" if pos == 1 else "SHORT", holds=held,
                                   ret=100 * (np.exp(ret_t) - 1)))
                pos = 0
    eq = np.exp(np.nancumsum(strat)); bh = np.exp(np.nancumsum(np.nan_to_num(r)))
    sd = np.std(strat[lookback_min:]) if (lookback_min := 60) < T else np.std(strat)
    sharpe = np.mean(strat[60:]) / (sd + 1e-9) * np.sqrt(252)
    tr = pd.DataFrame(trades)
    stats_d = dict(n=len(tr), winrate=100 * (tr["ret"] > 0).mean() if len(tr) else np.nan,
                   avg=tr["ret"].mean() if len(tr) else np.nan,
                   best=tr["ret"].max() if len(tr) else np.nan,
                   worst=tr["ret"].min() if len(tr) else np.nan,
                   hold=tr["holds"].mean() if len(tr) else np.nan, sharpe=sharpe,
                   cagr=100 * (eq[-1] ** (252 / max(T, 1)) - 1))
    return pd.Series(eq, index=E["dates"]), pd.Series(bh, index=E["dates"]), tr, stats_d

# ----------------------------------------------------------------------------
# 7. UI — HEADER, SIDEBAR, KPIs
# ----------------------------------------------------------------------------
st.markdown("""
<div class="fve-head">
  <div>
    <div class="fve-brand">FVE <em>//</em> FAIR VALUE ENGINE</div>
    <div class="fve-sub">Market-relative valuation · cross-asset latent-state model · v2.4</div>
  </div>
  <div class="fve-chips">
    <span class="chip"><span class="dot"></span>ENGINE LIVE</span>
    <span class="chip">UNIVERSE&nbsp;<b style="color:#eef3ff">""" + str(N_ASSETS) + """</b>&nbsp;INSTRUMENTS</span>
    <span class="chip">9 ASSET CLASSES</span>
    <span class="chip">WALK-FORWARD · REGIME-AWARE</span>
  </div>
</div>""", unsafe_allow_html=True)

# ---- sidebar ----
with st.sidebar:
    st.markdown("<div class='sec-kick'>Engine Controls</div><div class='sec-title' style='font-size:17px'>Live Data Calibration</div>", unsafe_allow_html=True)
    # Hierarchical asset selector: class → symbol
    asset_classes = list(UNIVERSE.keys())
    default_class = CLASS_OF.get("QQQ", asset_classes[0])
    selected_class = st.selectbox("Asset Class", asset_classes, index=asset_classes.index(default_class))
    symbols_in_class = UNIVERSE[selected_class].split()
    default_symbol = "QQQ" if "QQQ" in symbols_in_class else symbols_in_class[0]
    target = st.selectbox("Target Asset", symbols_in_class, index=symbols_in_class.index(default_symbol))
    st.markdown("<div class='sec-kick' style='margin-top:14px'>Model</div>", unsafe_allow_html=True)
    method = st.selectbox("Latent-factor method", ["PCA", "FastICA", "FactorAnalysis"])
    ncomp = st.slider("Latent factors", 4, 16, 8)
    lookback = st.select_slider("Calibration window (sessions)", [126, 189, 252, 378], 252)
    refit = st.select_slider("Recalibration frequency (sessions)", [5, 10, 21, 42], 21)
    peers_k = st.slider("Orthogonal peer instruments", 5, 30, 15)
    alpha = st.select_slider("Ridge penalty λ", [0.1, 1.0, 10.0, 100.0], 10.0)
    decay = st.select_slider("Sample half-life decay", [0.80, 0.88, 0.94, 1.0], 0.94,
                            format_func=lambda x: "uniform" if x == 1.0 else f"{x}")
    regime_weight = st.toggle("Regime-aware sample weighting", True)
    st.markdown("<div class='sec-kick' style='margin-top:14px'>Oscillator</div>", unsafe_allow_html=True)
    norm_win = st.select_slider("Adaptive normalization window", [21, 42, 63, 126], 63)
    bound = st.select_slider("Saturation bound (z → ±100)", [1.5, 2.0, 2.5, 3.0], 2.5)
    smooth = st.selectbox("Smoothing filter", ["Kalman", "EMA", "None"])
    span = st.slider("Filter gain / span", 2, 21, 6)
    thr_q = st.select_slider("Dynamic threshold quantile", [0.05, 0.08, 0.12, 0.18, 0.25], 0.12)
    st.session_state["mr_horizon"] = st.slider("Mean-reversion horizon (sessions)", 1, 42, 10)
    st.markdown("<div class='sec-kick' style='margin-top:14px'>Backtest</div>", unsafe_allow_html=True)
    allow_short = st.toggle("Allow short signals", True)
    exit_lvl = st.slider("Exit at FVO ≥", -40, 40, -10)
    max_hold = st.slider("Max holding period (sessions)", 5, 60, 21)
    st.markdown("---")
    # Run Analysis button
    if st.button("RUN ANALYSIS"):
        st.session_state["run_id"] = st.session_state.get("run_id", 0) + 1
        st.session_state["analyzing"] = True
        st.rerun()
    st.caption("Research tool – live data via yfinance. Not investment advice.")

# ---- load + compute ----
if st.session_state.get("analyzing", False):
    prices, feed = get_prices()
    if target not in prices.columns:
        target = prices.columns[0]
    run_id = st.session_state.get("run_id", 0)

    with st.spinner("Loading live data and calibrating engine…"):
        E = run_engine(prices, target, method, ncomp, lookback, refit, peers_k, alpha,
                       decay, regime_weight, norm_win, bound, smooth, span, thr_q)
else:
    st.info("Click **RUN ANALYSIS** to fetch live data and run the engine.")
    st.stop()

# ---- ticker tape ----
proxies = {"SPY": "US EQ", "EFA": "INTL EQ", "TLT": "RATES", "GLD": "GOLD",
           "CL=F": "WTI", "BTC-USD": "BTC", "EURUSD=X": "EUR/USD", "^VIX": "VIX",
           "XLE": "ENERGY", "KWEB": "CHINA"}
items = ""
for s, lab in proxies.items():
    if s in prices.columns and len(prices[s].dropna()) > 2:
        chg = 100 * (prices[s].iloc[-1] / prices[s].iloc[-2] - 1)
        cls = "up" if chg >= 0 else "dn"; arrow = "▲" if chg >= 0 else "▼"
        items += f"<span><b>{lab}</b> <span class='{cls}'>{arrow} {chg:+.2f}%</span></span>"
st.markdown(f'<div class="tape-wrap"><div class="tape">{items}{items}</div></div>',
            unsafe_allow_html=True)

# ---- KPI console ----
i = -1
px_v, fv_v = E["price"].iloc[i], E["fv"].iloc[i]
gap_v, mis_v = E["gap"].iloc[i], E["mispct"].iloc[i]
fvo_v, z_v = E["fvo"].iloc[i], E["z"].iloc[i]
reg_now = E["regimes"].iloc[i]
rm = REGIME_META[reg_now]
conf_v, pmr_v = E["conf"].iloc[i], E["p_mr"].iloc[i]
sig_e = E["sigma_e"].iloc[i]
verdict = "EXPENSIVE" if mis_v > 1 else ("CHEAP" if mis_v < -1 else "FAIRLY VALUED")
v_col = "#ff5d6c" if mis_v > 1 else ("#2fd08c" if mis_v < -1 else "#f2b544")
d1 = E["price"].iloc[-1] / E["price"].iloc[-2] - 1
hl = E["halflife"].iloc[i]

fvo_pos = float(np.clip((fvo_v + 100) / 200 * 100, 0, 100))
kpi_html = f"""
<div class="kpis">
  <div class="kpi accent-blue"><div class="lab">{target} · Market Price · {CLASS_OF[target]}</div>
    <div class="val">{fmt_px(px_v)}</div>
    <div class="aux"><span class="{'up' if d1>=0 else 'dn'}">{'▲' if d1>=0 else '▼'} {100*d1:+.2f}%</span> 1D · feed {feed}</div></div>
  <div class="kpi accent-amber"><div class="lab">Model Fair Value</div>
    <div class="val" style="color:#f2b544">{fmt_px(fv_v)}</div>
    <div class="aux">implied by {E['n_assets']-1}-instrument state · {E['method']}×{E['ncomp']}F</div></div>
  <div class="kpi accent-{'red' if gap_v>0 else 'green'}"><div class="lab">Fair Value Gap</div>
    <div class="val" style="color:{'#ff5d6c' if gap_v>0 else '#2fd08c'}">{('+' if gap_v>0 else '')+fmt_px(gap_v)}</div>
    <div class="aux"><span class="amber">{mis_v:+.2f}%</span> mispricing · <b style="color:{v_col}">{verdict}</b></div></div>
  <div class="kpi accent-cyan"><div class="lab">Fair Value Oscillator</div>
    <div class="val" style="color:#3fd8c9">{fvo_v:+.1f}</div>
    <div class="meter"><i style="width:{fvo_pos:.0f}%;background:linear-gradient(90deg,#2fd08c,#f2b544,#ff5d6c)"></i></div>
    <div class="aux">z = {z_v:+.2f}σ · adaptive · bounded ±100</div></div>
</div>
<div class="kpis">
  <div class="kpi" style="border-top:2px solid {rm['color']}"><div class="lab">Market Regime</div>
    <div class="val" style="font-size:20px"><span class="pill" style="color:{rm['color']};border:1px solid {rm['color']}66;background:{rm['color']}1a">{rm['icon']} {reg_now}</span></div>
    <div class="aux">data-driven cluster · 5-state</div></div>
  <div class="kpi accent-green"><div class="lab">Confidence Score</div>
    <div class="val">{conf_v:.0f}<span style="font-size:14px;color:#7f93b8">/100</span></div>
    <div class="meter"><i style="width:{conf_v:.0f}%;background:#2fd08c"></i></div>
    <div class="aux">fit {E['r2'].iloc[i]:.2f} R² · regime purity-weighted</div></div>
  <div class="kpi accent-violet"><div class="lab">Mean-Reversion P({st.session_state['mr_horizon']}d)</div>
    <div class="val">{pmr_v:.0f}%</div>
    <div class="meter"><i style="width:{pmr_v:.0f}%;background:#b7a4f3"></i></div>
    <div class="aux">OU half-life ≈ {hl:.0f} sessions</div></div>
  <div class="kpi accent-red"><div class="lab">Residual Risk</div>
    <div class="val">{sig_e:.2f}<span style="font-size:14px;color:#7f93b8">%/d</span></div>
    <div class="aux">unexplained idiosyncratic vol · 42d</div></div>
</div>"""
st.markdown(kpi_html, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# 8. TABS
# ----------------------------------------------------------------------------
tab_val, tab_osc, tab_drv, tab_reg, tab_bt, tab_mtf = st.tabs(
    ["◈ Valuation", "◈ Oscillator Lab", "◈ Drivers & Explainability",
     "◈ Regime & Stability", "◈ Signal Backtest", "◈ Multi-Timeframe"])

D = E["dates"]
regime_spans = []
prev_r, start_i = E["regimes"].iloc[0], 0
for k in range(1, len(E["regimes"])):
    if E["regimes"].iloc[k] != prev_r:
        regime_spans.append((D[start_i], D[k], prev_r)); prev_r, start_i = E["regimes"].iloc[k], k
regime_spans.append((D[start_i], D[-1], prev_r))

def paint_regimes(fig, row=1, alpha_=0.07):
    for a, b, r in regime_spans[-40:]:
        fig.add_vrect(x0=a, x1=b, fillcolor=REGIME_META[r]["color"], opacity=alpha_,
                      layer="below", line_width=0, row=row, col=1)

# ---------------- TAB 1 · VALUATION ----------------
with tab_val:
    st.markdown("<div class='sec-kick'>Core Output</div><div class='sec-title'>Price vs. Model-Implied Fair Value</div>"
                "<div class='sec-desc'>Fair value is the price path implied by the full market state "
                "(latent factors + orthogonal peers). Bands are ±1σ model uncertainty; markers flag "
                "statistically significant oscillator crossings and price/oscillator divergences.</div>",
                unsafe_allow_html=True)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38],
                        vertical_spacing=0.05, subplot_titles=("", "FAIR VALUE OSCILLATOR"))
    paint_regimes(fig, 1)
    fv_lo = E["fv"] * np.exp(-E["sigma_e"] / 100 * np.sqrt(norm_win))
    fv_hi = E["fv"] * np.exp(E["sigma_e"] / 100 * np.sqrt(norm_win))
    fig.add_trace(go.Scatter(x=D, y=fv_hi, line=dict(width=0), showlegend=False, hoverinfo="skip"), 1, 1)
    fig.add_trace(go.Scatter(x=D, y=fv_lo, fill="tonexty", fillcolor="rgba(242,181,68,.10)",
                             line=dict(width=0), name="±1σ FV band"), 1, 1)
    fig.add_trace(go.Scatter(x=D, y=E["price"], name=target, line=dict(color="#e8eefc", width=1.6)), 1, 1)
    fig.add_trace(go.Scatter(x=D, y=E["fv"], name="Fair Value", line=dict(color="#f2b544", width=1.6, dash="dot")), 1, 1)
    sd_ = E["sig_dn"]; su_ = E["sig_up"]
    fig.add_trace(go.Scatter(x=D[sd_], y=E["price"][sd_], mode="markers", name="undervalued entry",
                             marker=dict(symbol="triangle-up", size=11, color="#2fd08c",
                                         line=dict(color="#0c1322", width=1))), 1, 1)
    fig.add_trace(go.Scatter(x=D[su_], y=E["price"][su_], mode="markers", name="overvalued entry",
                             marker=dict(symbol="triangle-down", size=11, color="#ff5d6c",
                                         line=dict(color="#0c1322", width=1))), 1, 1)
    db, du = E["div_bear"], E["div_bull"]
    fig.add_trace(go.Scatter(x=D[db], y=E["price"][db], mode="markers", name="bearish divergence",
                             marker=dict(symbol="diamond", size=9, color="#ff5d6c", opacity=.85)), 1, 1)
    fig.add_trace(go.Scatter(x=D[du], y=E["price"][du], mode="markers", name="bullish divergence",
                             marker=dict(symbol="diamond", size=9, color="#2fd08c", opacity=.85)), 1, 1)
    # oscillator pane
    fig.add_trace(go.Scatter(x=D, y=E["ob"], line=dict(color="rgba(255,93,108,.55)", width=1, dash="dash"),
                             name="dynamic OB"), 2, 1)
    fig.add_trace(go.Scatter(x=D, y=E["os_"], line=dict(color="rgba(47,208,140,.55)", width=1, dash="dash"),
                             name="dynamic OS"), 2, 1)
    fig.add_trace(go.Scatter(x=D, y=np.zeros(len(D)), line=dict(color="rgba(140,165,215,.35)", width=1),
                             name="fair-value centerline"), 2, 1)
    fig.add_trace(go.Scatter(x=D, y=E["fvo"] + E["band"], line=dict(width=0), showlegend=False, hoverinfo="skip"), 2, 1)
    fig.add_trace(go.Scatter(x=D, y=E["fvo"] - E["band"], fill="tonexty", fillcolor="rgba(63,216,201,.10)",
                             line=dict(width=0), name="confidence band"), 2, 1)
    fig.add_trace(go.Scatter(x=D, y=E["fvo_raw"], line=dict(color="rgba(63,216,201,.28)", width=1),
                             name="raw FVO"), 2, 1)
    fig.add_trace(go.Scatter(x=D, y=E["fvo"], line=dict(color="#3fd8c9", width=2), name="FVO (smoothed)"), 2, 1)
    fig.add_trace(go.Scatter(x=D[sd_], y=E["fvo"][sd_], mode="markers", showlegend=False,
                             marker=dict(color="#2fd08c", size=8)), 2, 1)
    fig.add_trace(go.Scatter(x=D[su_], y=E["fvo"][su_], mode="markers", showlegend=False,
                             marker=dict(color="#ff5d6c", size=8)), 2, 1)
    fig.update_yaxes(title_text="price", row=1, col=1, **AX)
    fig.update_yaxes(title_text="FVO", range=[-105, 105], row=2, col=1, **AX)
    fig.update_layout(annotations=[dict(text="FAIR VALUE OSCILLATOR · 0 = fairly valued", x=0, y=0.365,
        xref="paper", yref="paper", showarrow=False, font=dict(family="IBM Plex Mono", size=9, color="#7f93b8"))])
    st.plotly_chart(fig_base(fig, 640), use_container_width=True)

    c1, c2 = st.columns([3, 2])
    with c1:
        fig2 = go.Figure(go.Bar(x=D, y=E["mispct"], marker_color=np.where(E["mispct"] >= 0, "#ff5d6c", "#2fd08c"), opacity=.8))
        fig2.add_hline(y=0, line_color="rgba(140,165,215,.3)")
        fig2.update_layout(title=dict(text="FAIR VALUE GAP (% of price)", font=dict(family="IBM Plex Mono", size=10, color="#7f93b8")))
        st.plotly_chart(fig_base(fig2, 240), use_container_width=True)
    with c2:
        fig3 = go.Figure(go.Histogram(x=E["mispct"].dropna(), nbinsx=60,
                                      marker_color="rgba(63,216,201,.55)"))
        fig3.add_vline(x=mis_v, line_color="#f2b544", line_width=2,
                       annotation_text=f"now {mis_v:+.2f}%", annotation_font_color="#f2b544")
        fig3.update_layout(title=dict(text="MISPRICING DISTRIBUTION", font=dict(family="IBM Plex Mono", size=10, color="#7f93b8")))
        st.plotly_chart(fig_base(fig3, 240), use_container_width=True)

# ---------------- TAB 2 · OSCILLATOR LAB ----------------
with tab_osc:
    st.markdown("<div class='sec-kick'>Signal Anatomy</div><div class='sec-title'>Oscillator Lab</div>"
                "<div class='sec-desc'>Adaptive normalization keeps readings comparable across volatility regimes; "
                "thresholds are historical quantiles of the smoothed oscillator, not fixed levels.</div>",
                unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=D, y=E["fvo_raw"], name="raw", line=dict(color="rgba(63,216,201,.3)", width=1)))
        fig.add_trace(go.Scatter(x=D, y=E["fvo"], name=f"smoothed ({smooth})", line=dict(color="#3fd8c9", width=2)))
        fig.add_trace(go.Scatter(x=D, y=E["ob"], name="OB quantile", line=dict(color="rgba(255,93,108,.6)", width=1, dash="dash")))
        fig.add_trace(go.Scatter(x=D, y=E["os_"], name="OS quantile", line=dict(color="rgba(47,208,140,.6)", width=1, dash="dash")))
        fig.add_hline(y=0, line_color="rgba(140,165,215,.35)")
        st.plotly_chart(fig_base(fig, 360), use_container_width=True)
    with c2:
        cur = E["fvo"].dropna()
        fig = go.Figure(go.Histogram(x=cur, nbinsx=55, marker_color="rgba(183,164,243,.6)"))
        fig.add_vline(x=cur.quantile(1 - thr_q), line_dash="dash", line_color="#ff5d6c")
        fig.add_vline(x=cur.quantile(thr_q), line_dash="dash", line_color="#2fd08c")
        fig.add_vline(x=fvo_v, line_color="#f2b544", line_width=2, annotation_text=f"now {fvo_v:+.0f}")
        fig.update_layout(title=dict(text="FVO EMPIRICAL DISTRIBUTION + DYNAMIC THRESHOLDS",
                                     font=dict(family="IBM Plex Mono", size=10, color="#7f93b8")))
        st.plotly_chart(fig_base(fig, 360), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=D, y=E["z"], name="mispricing z", line=dict(color="#f2b544", width=1.6)))
        fig.add_hrect(y0=1, y1=3, fillcolor="rgba(255,93,108,.08)", line_width=0)
        fig.add_hrect(y0=-3, y1=-1, fillcolor="rgba(47,208,140,.08)", line_width=0)
        st.plotly_chart(fig_base(fig, 260), use_container_width=True)
        st.caption("**Adaptive z-score** of log(P/FV) — the pre-saturation signal.")
    with c2:
        pctl = float((cur < fvo_v).mean() * 100)
        fig = go.Figure(go.Indicator(mode="gauge+number", value=float(fvo_v),
                number=dict(font=dict(family="IBM Plex Mono", size=34, color="#3fd8c9")),
                gauge=dict(axis=dict(range=[-100, 100], tickfont=dict(color="#8fa3c4", size=9)),
                           bar=dict(color="#3fd8c9"), bgcolor="rgba(13,20,35,.6)",
                           borderwidth=1, bordercolor="rgba(140,165,215,.25)",
                           steps=[dict(range=[-100, float(cur.quantile(thr_q))], color="rgba(47,208,140,.12)"),
                                  dict(range=[float(cur.quantile(1-thr_q)), 100], color="rgba(255,93,108,.12)")],
                           threshold=dict(line=dict(color="#f2b544", width=2), thickness=.75, value=float(fvo_v)))))
        fig.update_layout(height=260, margin=dict(t=30, b=0), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Current reading sits at the **{pctl:.0f}th percentile** of history.")
    with c3:
        n_bear, n_bull = int(db.sum()), int(du.sum())
        st.markdown(f"""
        <div class="kpi accent-red" style="margin-bottom:10px"><div class="lab">Bearish Divergences (12m)</div>
        <div class="val">{n_bear}</div><div class="aux">price new-high, FVO failing to confirm</div></div>
        <div class="kpi accent-green"><div class="lab">Bullish Divergences (12m)</div>
        <div class="val">{n_bull}</div><div class="aux">price new-low, FVO refusing to follow</div></div>
        <div class="kpi accent-violet" style="margin-top:10px"><div class="lab">OU Mean-Reversion Half-Life</div>
        <div class="val">{hl:.0f}d</div><div class="aux">speed at which mispricing historically decays</div></div>
        """, unsafe_allow_html=True)

# ---------------- TAB 3 · DRIVERS ----------------
with tab_drv:
    st.markdown("<div class='sec-kick'>Explainability</div><div class='sec-title'>What Is Driving Fair Value Today?</div>"
                "<div class='sec-desc'>The fair-value regression is linear, so attributions are exact: "
                "βⱼ·xⱼ decomposes the model-implied return into factor and peer contributions (bp/day).</div>",
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    contrib_last = E["contrib"].iloc[-1].sort_values(key=abs, ascending=False)
    with c1:
        top = contrib_last.head(14)
        fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h",
                               marker_color=["#2fd08c" if v < 0 else "#ff5d6c" for v in top.values]))
        fig.update_layout(title=dict(text="TOP CONTRIBUTORS TO IMPLIED RETURN (bp/day) — green pulls FV down (cheap), red up",
                                     font=dict(family="IBM Plex Mono", size=9, color="#7f93b8")), yaxis=dict(autorange="reversed", **AX))
        st.plotly_chart(fig_base(fig, 380), use_container_width=True)
    with c2:
        cls_agg = E["contrib"].iloc[-1].groupby(
            [CLASS_OF.get(c, "Latent Factor") for c in E["contrib"].columns]).sum()
        cls_agg = cls_agg.reindex(cls_agg.abs().sort_values(ascending=False).index)
        fig = go.Figure(go.Bar(x=cls_agg.values, y=cls_agg.index, orientation="h",
                               marker_color=[CLASS_COLORS.get(i, "#3fd8c9") for i in cls_agg.index]))
        fig.update_layout(title=dict(text="CONTRIBUTION BY ASSET CLASS (bp/day)",
                                     font=dict(family="IBM Plex Mono", size=9, color="#7f93b8")), yaxis=dict(autorange="reversed", **AX))
        st.plotly_chart(fig_base(fig, 200), use_container_width=True)
        fe = E["factor_exp"]
        fig = go.Figure(go.Bar(x=fe.index, y=fe.values,
                               marker_color=["#f2b544" if v >= 0 else "#4f9cf9" for v in fe.values]))
        fig.update_layout(title=dict(text="LATENT FACTOR EXPOSURES (bp/day per σ)",
                                     font=dict(family="IBM Plex Mono", size=9, color="#7f93b8")), **{"xaxis": dict(**AX)})
        st.plotly_chart(fig_base(fig, 168), use_container_width=True)

    heat_cols = contrib_last.abs().head(14).index.tolist()
    fig = go.Figure(go.Heatmap(
        z=E["contrib"][heat_cols].T.values, x=D[-40:], y=heat_cols,
        colorscale=[[0, "#2fd08c"], [0.5, "#101a2c"], [1, "#ff5d6c"]], zmid=0,
        colorbar=dict(title=dict(text="bp/d", font=dict(size=9, color="#8fa3c4")),
                      tickfont=dict(size=9, color="#8fa3c4"), thickness=10)))
    fig.update_layout(title=dict(text="CONTRIBUTION HEATMAP · LAST 40 SESSIONS",
                                 font=dict(family="IBM Plex Mono", size=10, color="#7f93b8")),
                      yaxis=dict(autorange="reversed", **AX))
    st.plotly_chart(fig_base(fig, 380), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        pi = E["perm_imp"].head(12)
        fig = go.Figure(go.Bar(x=pi.values, y=pi.index, orientation="h",
                               marker_color="rgba(242,181,68,.8)"))
        fig.update_layout(title=dict(text="PERMUTATION IMPORTANCE (ΔR² on final window)",
                                     font=dict(family="IBM Plex Mono", size=9, color="#7f93b8")), yaxis=dict(autorange="reversed", **AX))
        st.plotly_chart(fig_base(fig, 300), use_container_width=True)
    with c2:
        ev = E["comp_var"]
        fig = go.Figure(go.Bar(x=[f"PC{i+1}" for i in range(len(ev))], y=ev,
                               marker_color="rgba(79,156,249,.8)"))
        fig.update_layout(title=dict(text="VARIANCE CAPTURED PER LATENT FACTOR (%)",
                                     font=dict(family="IBM Plex Mono", size=9, color="#7f93b8")))
        st.plotly_chart(fig_base(fig, 300), use_container_width=True)

# ---------------- TAB 4 · REGIME & STABILITY ----------------
with tab_reg:
    st.markdown("<div class='sec-kick'>Diagnostics</div><div class='sec-title'>Regime Timeline & Model Stability</div>"
                "<div class='sec-desc'>Regimes are KMeans clusters over market momentum, volatility and dispersion. "
                "CUSUM tracks structural breaks in the fair-value residual; rolling R² tracks explanatory power.</div>",
                unsafe_allow_html=True)
    fig = go.Figure()
    for a, b, r in regime_spans:
        fig.add_vrect(x0=a, x1=b, fillcolor=REGIME_META[r]["color"], opacity=0.75, line_width=0)
    fig.update_layout(height=90, yaxis=dict(visible=False, range=[0, 1]), xaxis=dict(**AX), margin=dict(t=5, b=5))
    st.plotly_chart(fig_base(fig, 90, legend=False), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=D, y=E["r2"], name="rolling R²", line=dict(color="#2fd08c", width=1.6)))
        fig.add_trace(go.Scatter(x=D, y=E["conf"] / 100, name="confidence /100", line=dict(color="#b7a4f3", width=1.2, dash="dot")))
        fig.update_layout(title=dict(text="ROLLING EXPLANATORY POWER (63d)", font=dict(family="IBM Plex Mono", size=10, color="#7f93b8")))
        st.plotly_chart(fig_base(fig, 300), use_container_width=True)
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=D, y=E["cusum"], name="CUSUM(|resid|)", line=dict(color="#f2b544", width=1.4)))
        brk_idx = np.flatnonzero(np.isfinite(E["breaks"]))
        for bi in brk_idx:
            fig.add_vline(x=D[bi], line_color="#ff5d6c", line_width=1, opacity=.7)
        fig.add_hline(y=6, line_dash="dash", line_color="rgba(255,93,108,.5)")
        fig.update_layout(title=dict(text=f"STRUCTURAL BREAK DETECTOR · {len(brk_idx)} breaks flagged",
                                     font=dict(family="IBM Plex Mono", size=10, color="#7f93b8")))
        st.plotly_chart(fig_base(fig, 300), use_container_width=True)

    rdf = pd.DataFrame([
        {"regime": r,
         "target μ (bp/d)": 1e4 * E["rets"][target][E["regimes"] == r].mean(),
         "target σ (bp/d)": 1e4 * E["rets"][target][E["regimes"] == r].std(),
         "FVO mean": E["fvo"][E["regimes"] == r].mean(),
         "days": int((E["regimes"] == r).sum()),
         "share %": 100 * (E["regimes"] == r).mean()}
        for r in REGIME_META])
    st.dataframe(rdf.style.format({"target μ (bp/d)": "{:+.1f}", "target σ (bp/d)": "{:.1f}",
                                   "FVO mean": "{:+.1f}", "share %": "{:.1f}"}),
                 use_container_width=True, hide_index=True)

# ---------------- TAB 5 · BACKTEST ----------------
with tab_bt:
    st.markdown("<div class='sec-kick'>Historical Replay</div><div class='sec-title'>Oscillator Signal Backtest</div>"
                "<div class='sec-desc'>Long when FVO crosses below its dynamic oversold quantile, short above overbought "
                "(if enabled); exit near the centerline or at max hold. Walk-forward — signals use only past information.</div>",
                unsafe_allow_html=True)
    eq, bh, tr, stt = backtest(E, allow_short, exit_lvl, max_hold)
    c = st.columns(6)
    labels = ["TRADES", "WIN RATE", "AVG RET", "AVG HOLD", "SHARPE", "CAGR"]
    vals = [f"{stt['n']:.0f}", f"{stt['winrate']:.0f}%", f"{stt['avg']:+.2f}%",
            f"{stt['hold']:.0f}d", f"{stt['sharpe']:.2f}", f"{stt['cagr']:+.1f}%"]
    cols = ["#eef3ff", "#2fd08c", "#3fd8c9", "#eef3ff", "#f2b544", "#b7a4f3"]
    for k in range(6):
        with c[k]:
            st.markdown(f"<div class='kpi'><div class='lab'>{labels[k]}</div>"
                        f"<div class='val' style='font-size:22px;color:{cols[k]}'>{vals[k]}</div></div>",
                        unsafe_allow_html=True)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=D, y=eq, name="FVO strategy", line=dict(color="#3fd8c9", width=2)), 1, 1)
    fig.add_trace(go.Scatter(x=D, y=bh, name="buy & hold", line=dict(color="rgba(232,238,252,.5)", width=1.2)), 1, 1)
    dd = eq / eq.cummax() - 1
    fig.add_trace(go.Scatter(x=D, y=dd * 100, name="drawdown %", fill="tozeroy",
                             fillcolor="rgba(255,93,108,.18)", line=dict(color="#ff5d6c", width=1)), 2, 1)
    st.plotly_chart(fig_base(fig, 460), use_container_width=True)
    if len(tr):
        st.dataframe(tr.tail(15).iloc[::-1].style.format({"ret": "{:+.2f}%"}),
                     use_container_width=True, hide_index=True)
    else:
        st.info("No completed trades at current thresholds — loosen the entry quantile in the sidebar.")

# ---------------- TAB 6 · MULTI-TIMEFRAME ----------------
with tab_mtf:
    st.markdown("<div class='sec-kick'>Horizon Scan</div><div class='sec-title'>Multi-Timeframe Fair Value</div>"
                "<div class='sec-desc'>One model, three normalization horizons — dislocations that align across "
                "timeframes carry materially higher conviction.</div>", unsafe_allow_html=True)
    cols = st.columns(3)
    for k, (lab_, s_) in enumerate(E["mtf"].items()):
        v_ = float(s_.iloc[-1]); pct = float((s_.dropna() < v_).mean() * 100)
        sig = "OVERVALUED" if v_ > 45 else ("UNDERVALUED" if v_ < -45 else "NEUTRAL")
        scol = "#ff5d6c" if v_ > 45 else ("#2fd08c" if v_ < -45 else "#8fa3c4")
        with cols[k]:
            st.markdown(f"""<div class="kpi accent-cyan"><div class="lab">{lab_}</div>
            <div class="val" style="color:#3fd8c9">{v_:+.1f}</div>
            <div class="aux"><b style="color:{scol}">{sig}</b> · {pct:.0f}th pctile</div></div>""",
                        unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_hrect(y0=45, y1=100, fillcolor="rgba(255,93,108,.07)", line_width=0)
            fig.add_hrect(y0=-100, y1=-45, fillcolor="rgba(47,208,140,.07)", line_width=0)
            fig.add_trace(go.Scatter(x=D, y=s_, line=dict(color="#3fd8c9", width=1.4), showlegend=False))
            fig.add_hline(y=0, line_color="rgba(140,165,215,.3)")
            fig.update_yaxes(range=[-105, 105], **AX)
            st.plotly_chart(fig_base(fig, 190, legend=False), use_container_width=True)

st.markdown("<div style='margin-top:26px;padding-top:12px;border-top:1px solid rgba(140,165,215,.14);"
            "font-family:IBM Plex Mono;font-size:10px;letter-spacing:.12em;color:#5f7396'>"
            f"FVE v2.4 · {E['n_assets']} INSTRUMENTS · {E['method']}×{E['ncomp']} FACTORS · "
            f"REFIT {refit}D · FEED: {feed} · RESEARCH TOOL — NOT INVESTMENT ADVICE</div>",
            unsafe_allow_html=True)