"""
Sanket - Market Signal Screener | A Pragyam Product Family Member
WRCI Engine Quantitative Signal Screener Terminal
"""

import html
import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
import numpy as np
import plotly.graph_objects as go
import requests
import io
import urllib3
from priority_engine_v3 import compute_priority_v3
import warnings
import logging
import time
from scipy.stats import spearmanr
from nsepython import nse_get_advances_declines
from logger import console

# UI — Obsidian Quant Terminal System
from ui.theme import inject_css, apply_chart_theme, progress_bar
import ui.components as ui

# ── SVG ICON SYSTEM ────────────────────────────────────────────────────────
SVGS = {
    "CHECK": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>',
    "LONG": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m5 12 7-7 7 7"/><path d="M12 19V5"/></svg>',
    "SHORT": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5v14"/><path d="m19 12-7 7-7-7"/></svg>',
    "DOT": '<svg width="8" height="8" viewBox="0 0 24 24" fill="currentColor" style="display: inline-block; vertical-align: middle; margin-right: 4px;"><circle cx="12" cy="12" r="10"/></svg>',
    "UP": '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="display: inline-block; vertical-align: middle; margin-right: 4px;"><path d="m5 12 7-7 7 7"/><path d="M12 19V5"/></svg>',
    "DOWN": '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="display: inline-block; vertical-align: middle; margin-right: 4px;"><path d="M12 5v14"/><path d="m19 12-7 7-7-7"/></svg>',
    "ZAP": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m13 2-2 10h3L11 22l2-10h-3l2-10z"/></svg>',
    "CHART": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>',
    "STRENGTH": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 16a4 4 0 1 0 0-8 4 4 0 0 0 0 8Z"/><path d="M8 8V4h8v4"/><path d="M16 16v4H8v-4"/></svg>',
    "SETTINGS": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.1a2 2 0 0 1-1-1.72v-.51a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/></svg>'
}

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Silence noisy warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
np.seterr(divide="ignore", invalid="ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="SANKET | Market Signal Screener",
    layout="wide",
    initial_sidebar_state="expanded",
)

VERSION = "v3.0.0"

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ══════════════════════════════════════════════════════════════════════════════

if "results_df" not in st.session_state:
    st.session_state["results_df"] = None
if "run_screener_flag" not in st.session_state:
    st.session_state["run_screener_flag"] = False
if "run_timeseries_flag" not in st.session_state:
    st.session_state["run_timeseries_flag"] = False
if "timeseries_done" not in st.session_state:
    st.session_state["timeseries_done"] = False
if "run_error" not in st.session_state:
    st.session_state["run_error"] = None
if "corr_data" not in st.session_state:
    st.session_state["corr_data"] = None
if "run_correlation_flag" not in st.session_state:
    st.session_state["run_correlation_flag"] = False

# ══════════════════════════════════════════════════════════════════════════════
# INITIALIZE UI
# ══════════════════════════════════════════════════════════════════════════════
inject_css()
ui.render_theme_toggle()

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & UNIVERSE DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

INDEX_LIST = [
    "F&O Stocks",
    # Broad market
    "NIFTY 50", "NIFTY NEXT 50", "NIFTY 100", "NIFTY 200", "NIFTY 500",
    # Midcap
    "NIFTY MIDCAP 50", "NIFTY MIDCAP 100", "NIFTY MIDCAP 150", "NIFTY MID SELECT",
    # Smallcap
    "NIFTY SMLCAP 50", "NIFTY SMLCAP 100", "NIFTY SMLCAP 250",
    # Sectoral
    "NIFTY BANK", "NIFTY PRIVATE BANK", "NIFTY PSU BANK",
    "NIFTY FIN SERVICE",
    "NIFTY IT", "NIFTY AUTO", "NIFTY FMCG", "NIFTY PHARMA",
    "NIFTY METAL", "NIFTY ENERGY", "NIFTY INFRA", "NIFTY REALTY",
    "NIFTY MEDIA",
    # All indexes as instruments
    "Benchmark Indexes",
]

# Broad-market + sectoral index instruments (traded as tickers, not constituents)
BENCHMARK_INDEXES_LIST = [
    # Broad market — NSE
    "^NSEI",           # Nifty 50
    "^NSMIDCP",        # Nifty Next 50
    "NIFTY_100.NS",    # Nifty 100
    "NIFTY_200.NS",    # Nifty 200
    "NIFTY_500.NS",    # Nifty 500
    "^NSEMDCP50",      # Nifty Midcap 50
    "NIFTY_MIDCAP_100.NS",    # Nifty Midcap 100
    "NIFTY_MIDCAP_150.NS",    # Nifty Midcap 150
    "NIFTY_MID_SELECT.NS",    # Nifty Midcap Select
    "NIFTYSMLCAP50.NS",       # Nifty Smallcap 50
    "NIFTY_SMALLCAP_100.NS",  # Nifty Smallcap 100
    "NIFTY_SMALLCAP_250.NS",  # Nifty Smallcap 250
    # Volatility
    "^INDIAVIX",       # India VIX
    # Broad market — BSE
    "^BSESN",          # S&P BSE Sensex
    "BSE-100.BO",      # BSE 100
    "BSE-200.BO",      # BSE 200
    "BSE-500.BO",      # BSE 500
    # Sectoral — NSE
    "^NSEBANK",        # Nifty Bank
    "^CNXFIN",         # Nifty Financial Services
    "^CNXIT",          # Nifty IT
    "^CNXAUTO",        # Nifty Auto
    "^CNXFMCG",        # Nifty FMCG
    "^CNXPHARMA",      # Nifty Pharma
    "^CNXMETAL",       # Nifty Metal
    "^CNXREALTY",      # Nifty Realty
    "^CNXENERGY",      # Nifty Energy
    "^CNXINFRA",       # Nifty Infrastructure
    "^CNXPSUBANK",     # Nifty PSU Bank
    "NIFTY_PRIVATE_BANK.NS",  # Nifty Private Bank
    "^CNXMEDIA",       # Nifty Media
]

BASE_URL = "https://archives.nseindia.com/content/indices/"
INDEX_URL_MAP = {
    "NIFTY 50": f"{BASE_URL}ind_nifty50list.csv",
    "NIFTY NEXT 50": f"{BASE_URL}ind_niftynext50list.csv",
    "NIFTY 100": f"{BASE_URL}ind_nifty100list.csv",
    "NIFTY 200": f"{BASE_URL}ind_nifty200list.csv",
    "NIFTY 500": f"{BASE_URL}ind_nifty500list.csv",
    "NIFTY MIDCAP 50": f"{BASE_URL}ind_niftymidcap50list.csv",
    "NIFTY MIDCAP 100": f"{BASE_URL}ind_niftymidcap100list.csv",
    "NIFTY MIDCAP 150": f"{BASE_URL}ind_niftymidcap150list.csv",
    "NIFTY MID SELECT": f"{BASE_URL}ind_niftymidcapselectlist.csv",
    "NIFTY SMLCAP 50":  f"{BASE_URL}ind_niftysmallcap50list.csv",
    "NIFTY SMLCAP 100": f"{BASE_URL}ind_niftysmallcap100list.csv",
    "NIFTY SMLCAP 250": f"{BASE_URL}ind_niftysmallcap250list.csv",
    "NIFTY BANK": f"{BASE_URL}ind_niftybanklist.csv",
    "NIFTY PRIVATE BANK": f"{BASE_URL}ind_niftypvtbanklist.csv",
    "NIFTY PSU BANK": f"{BASE_URL}ind_niftypsubanklist.csv",
    "NIFTY AUTO": f"{BASE_URL}ind_niftyautolist.csv",
    "NIFTY FIN SERVICE": f"{BASE_URL}ind_niftyfinancelist.csv",
    "NIFTY FMCG": f"{BASE_URL}ind_niftyfmcglist.csv",
    "NIFTY IT": f"{BASE_URL}ind_niftyitlist.csv",
    "NIFTY PHARMA": f"{BASE_URL}ind_niftypharmalist.csv",
    "NIFTY METAL": f"{BASE_URL}ind_niftymetallist.csv",
    "NIFTY ENERGY": f"{BASE_URL}ind_niftyenergylist.csv",
    "NIFTY INFRA": f"{BASE_URL}ind_niftyinfrastructurelist.csv",
    "NIFTY REALTY": f"{BASE_URL}ind_niftyrealtylist.csv",
    "NIFTY MEDIA": f"{BASE_URL}ind_niftymedialist.csv",
}

WIKI_URL_MAP = {
    "NIFTY 50": "https://en.wikipedia.org/wiki/NIFTY_50",
    "NIFTY NEXT 50": "https://en.wikipedia.org/wiki/NIFTY_Next_50",
    "NIFTY BANK": "https://en.wikipedia.org/wiki/NIFTY_Bank",
    "NIFTY IT": "https://en.wikipedia.org/wiki/NIFTY_IT",
    "NIFTY FIN SERVICE": "https://en.wikipedia.org/wiki/Nifty_Financial_Services_Index",
}

UNIVERSE_OPTIONS = ["India Indexes", "Global Indexes", "US Indexes", "ETF Index", "Commodities", "Currency", "Crypto", "Global Macro"]
TIMEFRAME_OPTIONS = ["Daily", "Weekly"]

# ETF Universe (from Pragyam)
ETF_LIST = [
    "CHEMICAL.NS", "NIFTYIETF.NS", "MON100.NS", "MAKEINDIA.NS", "SILVERIETF.NS",
    "HEALTHIETF.NS", "CONSUMIETF.NS", "GOLDIETF.NS", "INFRAIETF.NS", "CPSEETF.NS",
    "TNIDETF.NS", "COMMOIETF.NS", "MODEFENCE.NS", "MOREALTY.NS", "PSUBNKIETF.NS",
    "MASPTOP50.NS", "FMCGIETF.NS", "GROWWPOWER.NS", "ITIETF.NS", "EVINDIA.NS",
    "MNC.NS", "FINIETF.NS", "AUTOIETF.NS", "PVTBANIETF.NS", "MONIFTY500.NS",
    "ECAPINSURE.NS", "MIDCAPIETF.NS", "MOSMALL250.NS", "OILIETF.NS", "METALIETF.NS"
]

# US Index list
US_INDEX_LIST = ["S&P 500", "DOW JONES", "NASDAQ 100"]

# Hardcoded DOW 30 fallback (as of late 2024 — used only when Wikipedia is unreachable)
_DOW30_FALLBACK = [
    "AAPL", "AMGN", "AMZN", "AXP", "BA",  "CAT", "CRM", "CSCO", "CVX", "DIS",
    "DOW",  "GS",   "HD",   "HON", "IBM",  "JNJ", "JPM", "KO",   "MCD", "MRK",
    "MSFT", "NKE",  "NVDA", "PG",  "SHW",  "TRV", "UNH", "V",    "VZ",  "WMT",
]

# Commodities list (Yahoo Finance) — Expanded from Pragyam
COMMODITY_MAP = {
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Platinum": "PL=F",
    "Palladium": "PA=F",
    "Copper": "HG=F",
    "Crude Oil WTI": "CL=F",
    "Brent Crude": "BZ=F",
    "Natural Gas": "NG=F",
    "Gasoline RBOB": "RB=F",
    "Heating Oil": "HO=F",
    "Corn": "ZC=F",
    "Wheat": "ZW=F",
    "Soybeans": "ZS=F",
    "Soybean Meal": "ZM=F",
    "Soybean Oil": "ZL=F",
    "Cotton": "CT=F",
    "Coffee": "KC=F",
    "Sugar": "SB=F",
    "Cocoa": "CC=F",
    "Orange Juice": "OJ=F",
    "Lumber": "LBS=F",
    "Live Cattle": "LE=F",
    "Lean Hogs": "HE=F",
    "Feeder Cattle": "GF=F",
}
COMMODITY_LIST = list(COMMODITY_MAP.keys())

# Currency pairs (Yahoo Finance) — Expanded from Pragyam
CURRENCY_MAP = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "USD/CHF": "USDCHF=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "USDCAD=X",
    "NZD/USD": "NZDUSD=X",
    "USD/INR": "USDINR=X",
    "EUR/GBP": "EURGBP=X",
    "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X",
    "AUD/JPY": "AUDJPY=X",
    "EUR/CHF": "EURCHF=X",
    "EUR/AUD": "EURAUD=X",
    "GBP/CHF": "GBPCHF=X",
    "GBP/AUD": "GBPAUD=X",
    "USD/SGD": "USDSGD=X",
    "USD/HKD": "USDHKD=X",
    "USD/CNH": "USDCNH=X",
    "USD/ZAR": "USDZAR=X",
    "USD/MXN": "USDMXN=X",
    "USD/TRY": "USDTRY=X",
    "USD/BRL": "USDBRL=X",
    "USD/KRW": "USDKRW=X",
}
CURRENCY_LIST = list(CURRENCY_MAP.keys())

# Crypto universe (Yahoo Finance)
CRYPTO_MAP = {
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Solana": "SOL-USD",
    "Binance Coin": "BNB-USD",
    "Ripple (XRP)": "XRP-USD",
    "Cardano": "ADA-USD",
    "Dogecoin": "DOGE-USD",
    "Tron": "TRX-USD",
    "Chainlink": "LINK-USD",
    "Polkadot": "DOT-USD",
    "Polygon (POL)": "POL-USD",
    "Litecoin": "LTC-USD",
    "Bitcoin Cash": "BCH-USD",
    "Shiba Inu": "SHIB-USD",
    "Avalanche": "AVAX-USD",
    "Near Protocol": "NEAR-USD",
    "Uniswap": "UNI-USD",
    "Stellar": "XLM-USD",
    "Ethereum Classic": "ETC-USD",
    "Monero": "XMR-USD",
    "Cosmos": "ATOM-USD"
}
CRYPTO_LIST = list(CRYPTO_MAP.keys())

# Global Macro Bond ETF Universe — proxy for global yield dynamics via yfinance-available instruments
GLOBAL_MACRO_MAP = {
    # ── US Treasuries (Full Curve) ─────────────────────────────────────────────
    "US Treasury 1-3 Month":             "BIL",
    "US Treasury Ultra-Short (0-1Y)":    "SHV",
    "US Treasury 0-3 Month (SGOV)":      "SGOV",
    "US Treasury Short (1-3Y)":          "SHY",
    "US Treasury Short (1-3Y) Vanguard": "VGSH",
    "US Treasury Intermediate (3-7Y)":   "IEI",
    "US Treasury Intermediate (7-10Y)":  "IEF",
    "US Treasury Intermediate Vanguard": "VGIT",
    "US Treasury Long (10-20Y)":         "TLH",
    "US Treasury Long (20Y+)":           "TLT",
    "US Treasury Long Vanguard":         "VGLT",
    "US Treasury Total Market":          "GOVT",
    # ── Direct Yield Indices (Raw %) ──────────────────────────────────────────
    "US 13-Week T-Bill Yield":           "^IRX",
    "US 5-Year Treasury Yield":          "^FVX",
    "US 10-Year Treasury Yield":         "^TNX",
    "US 30-Year Treasury Yield":         "^TYX",
    # ── Inflation-Protected (TIPS) ─────────────────────────────────────────────
    "US TIPS Broad Market":              "TIP",
    "US TIPS Short-Term":                "VTIP",
    "International Govt Inflation-Linked": "WIP",
    # ── Aggregate / Multi-Sector ───────────────────────────────────────────────
    "US Core Aggregate Bond":            "AGG",
    "US Total Bond Market":              "BND",
    "US Floating Rate Notes":            "FLOT",
    "Global Aggregate Bond (Hedged)":    "BNDW",
    "Total International Bond (ex-US)":  "BNDX",
    # ── US Corporate: Investment Grade ────────────────────────────────────────
    "US Corporate Investment Grade":     "LQD",
    "US Corporate Short-Term (1-5Y)":    "VCSH",
    "US Corporate Intermediate":         "VCIT",
    "US Corporate Long-Term":            "VCLT",
    # ── High Yield & Alternative Credit ───────────────────────────────────────
    "US High Yield Corporate":           "HYG",
    "US High Yield Corporate SPDR":      "JNK",
    "Global High Yield Bond":            "GHYG",
    "Global Green Bond":                 "BGRN",
    "Preferred Stock (Hybrid)":          "PFF",
    "Convertible Bonds":                 "CWB",
    "Fallen Angels (Recent HY)":         "FALN",
    # ── Structured & Asset-Backed ─────────────────────────────────────────────
    "US Mortgage-Backed Securities":     "MBB",
    "US Mortgage-Backed Vanguard":       "VMBS",
    "US Senior Loan (Floating Rate)":    "BKLN",
    # ── Municipal Bonds ───────────────────────────────────────────────────────
    "US Municipal National":             "MUB",
    "US Municipal Tax-Exempt Vanguard":  "VTEB",
    # ── Developed Markets Sovereign (Europe) ─────────────────────────────────
    "International Treasury (ex-US)":    "IGOV",
    "International Treasury SPDR":       "BWX",
    "International Corporate Bonds":     "IBND",
    "Eurozone Government Bond":          "IEGA.L",
    "Eurozone Corporate Bond (IG)":      "IEAC.L",
    "Germany Govt Bonds (Bunds/Long)":   "BUNL.L",
    "Germany Short-Term (Schatz)":       "SDEU.L",
    "UK Gilts":                          "IGLT.L",
    "UK Gilts (Inflation-Linked)":       "INXG.L",
    "UK Corporate Bonds":                "SLXX.L",
    # ── Developed Markets Sovereign (Asia-Pacific) ────────────────────────────
    "Japan Government Bonds (Broad)":    "JGBL.L",
    "Australia Government Bonds":        "VGB.AX",
    "Canada Broad Aggregate Bond":       "XBB.TO",
    # ── India Fixed Income ────────────────────────────────────────────────────
    "India Gov Bonds (LSE Proxy)":       "IIND.L",
    "India 8-13Y G-Sec":                 "LTGILTBEES.NS",
    "India 5Y G-Sec":                    "GILT5YBEES.NS",
    "India AAA PSU Bond (Bharat 2030)":  "EBBETF0430.NS",
    "India Overnight Rate (Liquid)":     "LIQUIDBEES.NS",
    # ── Emerging Markets ──────────────────────────────────────────────────────
    "EM Sovereign Debt (USD)":           "EMB",
    "EM Sovereign Debt USD Invesco":     "PCY",
    "EM Sovereign (Local Currency)":     "EMLC",
    "EM High Yield Corporate":           "EMHY",
    "China Government Bonds":            "CBON",
    "China CNY Local Bonds":             "CNYB.L",
    # ── Broad Duration Proxies ────────────────────────────────────────────────
    "Short-Term Broad Bond":             "BSV",
    "Long-Term Broad Bond":              "BLV",
}

# Global Benchmark Indexes Universe — primary national equity index per country.
# Futures proxies used where the cash index is not available on Yahoo Finance.
GLOBAL_INDEXES_MAP = {
    # ── North America ──────────────────────────────────────────────────────────
    "S&P 500 (USA)":                     "^GSPC",
    "Dow Jones (USA)":                   "^DJI",
    "NASDAQ 100 (USA)":                  "^NDX",
    "Russell 2000 (USA)":                "^RUT",
    "TSX Composite (Canada)":            "^GSPTSE",
    "IPC (Mexico)":                      "^MXX",
    "Bovespa (Brazil)":                  "^BVSP",
    "Merval (Argentina)":                "^MERV",
    "IPSA (Chile)":                      "^IPSA",
    "COLCAP (Colombia)":                 "^COLCAP",
    # ── Europe ─────────────────────────────────────────────────────────────────
    "FTSE 100 (UK)":                     "^FTSE",
    "DAX (Germany)":                     "^GDAXI",
    "CAC 40 (France)":                   "^FCHI",
    "IBEX 35 (Spain)":                   "^IBEX",
    "FTSE MIB (Italy)":                  "FTSEMIB.MI",
    "AEX (Netherlands)":                 "^AEX",
    "SMI (Switzerland)":                 "^SSMI",
    "OMX Stockholm 30 (Sweden)":         "^OMXS30",
    "Oslo Bors All-Share (Norway)":      "^OSEAX",
    "OMX Copenhagen 25 (Denmark)":       "^OMXC25",
    "ATX (Austria)":                     "^ATX",
    "BEL 20 (Belgium)":                  "^BFX",
    "WIG 20 (Poland)":                   "^WIG20",
    "BIST 100 (Turkey)":                 "XU100.IS",
    "PSI 20 (Portugal)":                 "^PSI20",
    "ASE General (Greece)":              "^ATG",
    "OMX Helsinki 25 (Finland)":         "^OMXH25",
    "PX (Czech Republic)":               "^PX",
    "BUX (Hungary)":                     "^BUX",
    "MOEX (Russia)":                     "IMOEX.ME",
    # ── Asia-Pacific ───────────────────────────────────────────────────────────
    "Nikkei 225 (Japan)":                "^N225",
    "TOPIX (Japan)":                     "^TOPX",
    "Shanghai Composite (China)":        "000001.SS",
    "CSI 300 (China)":                   "000300.SS",
    "Hang Seng (Hong Kong)":             "^HSI",
    "KOSPI (South Korea)":               "^KS11",
    "KOSDAQ (South Korea)":              "^KQ11",
    "TAIEX (Taiwan)":                    "^TWII",
    "Nifty 50 (India)":                  "^NSEI",
    "Sensex (India)":                    "^BSESN",
    "ASX 200 (Australia)":               "^AXJO",
    "All Ordinaries (Australia)":        "^AORD",
    "STI (Singapore)":                   "^STI",
    "KLCI (Malaysia)":                   "^KLSE",
    "SET Composite (Thailand)":          "^SET",
    "Jakarta Composite (Indonesia)":     "^JKSE",
    "PSEi (Philippines)":                "PSEi.PS",
    "NZX 50 (New Zealand)":              "^NZ50",
    "VN-Index (Vietnam)":                "^VNINDEX",
    "KSE 100 (Pakistan)":                "^KSE",
    # ── Middle East & Africa ───────────────────────────────────────────────────
    "TA-125 (Israel)":                   "^TA125.TA",
    "Tadawul (Saudi Arabia)":            "^TASI.SR",
    "DFM General (UAE)":                 "^DFMGI",
    "QE Index (Qatar)":                  "^QSI",
    "JSE All-Share (South Africa)":      "J203.JO",
    "EGX 30 (Egypt)":                    "^CASE",
}

# Asset Name Lookup for friendly display (Reverse map tickers to names)
ASSET_NAME_LOOKUP = {v: k for k, v in {**COMMODITY_MAP, **CURRENCY_MAP, **CRYPTO_MAP, **GLOBAL_MACRO_MAP, **GLOBAL_INDEXES_MAP}.items()}

# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def get_fno_stock_list():
    """Fetch F&O eligible stocks from NSE with multiple fallback sources."""
    try:
        url = "https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.nseindia.com/market-data/live-equity-market?symbol=NIFTY%20FIN%20SERVICE',
        }

        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=10)

        response = session.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'data' in data:
                symbols = [item['symbol'] for item in data['data'] if 'symbol' in item]
                if symbols:
                    symbols_ns = [str(s) + ".NS" for s in symbols if s and str(s).strip()]
                    return symbols_ns, f"✓ Fetched {len(symbols_ns)} F&O securities"
    except Exception:
        pass

    try:
        stock_data = nse_get_advances_declines()
        if isinstance(stock_data, pd.DataFrame):
            symbols = None
            if 'SYMBOL' in stock_data.columns:
                symbols = stock_data['SYMBOL'].tolist()
            elif 'symbol' in stock_data.columns:
                symbols = stock_data['symbol'].tolist()
            elif len(stock_data.index) > 0 and not isinstance(stock_data.index, pd.RangeIndex):
                symbols = stock_data.index.tolist()

            if symbols:
                symbols_ns = [str(s) + ".NS" for s in symbols if s and str(s).strip()]
                if symbols_ns:
                    return symbols_ns, f"✓ Fetched {len(symbols_ns)} F&O securities"
    except Exception:
        pass

    try:
        url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, verify=False, timeout=10)
        if response.status_code == 200:
            csv_file = io.StringIO(response.text)
            stock_df = pd.read_csv(csv_file)
            if 'Symbol' in stock_df.columns:
                symbols = stock_df['Symbol'].tolist()
                symbols_ns = [str(s) + ".NS" for s in symbols if s and str(s).strip()]
                return symbols_ns, f"✓ Fetched {len(symbols_ns)} stocks (NIFTY 500 fallback)"
    except Exception:
        pass

    return None, "Failed to fetch F&O list from all sources"


def get_index_stock_list(index):
    if index == "F&O Stocks":
        return get_fno_stock_list()

    if index == "Benchmark Indexes":
        return BENCHMARK_INDEXES_LIST, f"✓ Loaded {len(BENCHMARK_INDEXES_LIST)} benchmark index instruments"

    # --- Source 1: NSE JSON API (most reliable, same endpoint as F&O) ---
    try:
        import urllib.parse
        api_url = f"https://www.nseindia.com/api/equity-stockIndices?index={urllib.parse.quote(index)}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.nseindia.com/market-data/live-equity-market',
        }
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        response = session.get(api_url, headers=headers, timeout=15)
        if response.status_code == 200:
            data = response.json()
            if 'data' in data:
                symbols = [item['symbol'] for item in data['data'] if 'symbol' in item]
                # Skip the first entry — it's always the index itself, not a constituent
                symbols = [s for s in symbols[1:] if s and str(s).strip()]
                if symbols:
                    symbols_ns = [str(s) + ".NS" for s in symbols]
                    return symbols_ns, f"✓ Fetched {len(symbols_ns)} constituents (NSE API)"
    except Exception:
        pass

    # --- Source 2: NSE archives CSV ---
    url = INDEX_URL_MAP.get(index)
    if url:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Cache-Control': 'max-age=0',
            }
            session = requests.Session()
            session.get("https://archives.nseindia.com", headers=headers, verify=False, timeout=10)
            response = session.get(url, headers=headers, verify=False, timeout=15)
            response.raise_for_status()
            stock_df = pd.read_csv(io.StringIO(response.text))
            symbol_col = next((c for c in stock_df.columns if c.lower() == 'symbol'), None)
            if symbol_col:
                symbols = stock_df[symbol_col].tolist()
                symbols_ns = [str(s) + ".NS" for s in symbols if s and str(s).strip()]
                if symbols_ns:
                    return symbols_ns, f"✓ Fetched {len(symbols_ns)} constituents (NSE archive)"
        except Exception:
            pass

    # --- Source 3: Wikipedia fallback ---
    wiki_result = _fetch_index_from_wikipedia(index)
    if wiki_result[0]:
        return wiki_result

    return None, f"Could not fetch constituents for '{index}'. NSE API, archive CSV, and Wikipedia all failed."


def _fetch_index_from_wikipedia(index):
    wiki_url = WIKI_URL_MAP.get(index)
    if not wiki_url:
        return None, f"No Wikipedia fallback for {index}"
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(wiki_url, headers=headers, timeout=15)
        response.raise_for_status()
        tables = pd.read_html(io.StringIO(response.text))
        for table in tables:
            cols_lower = [str(c).lower() for c in table.columns]
            symbol_col = None
            for candidate in ('symbol', 'ticker', 'nse code', 'code'):
                for i, c in enumerate(cols_lower):
                    if candidate in c:
                        symbol_col = table.columns[i]
                        break
                if symbol_col is not None:
                    break
            if symbol_col is None:
                continue
            symbols = [str(s).strip() for s in table[symbol_col].dropna().tolist()]
            symbols_ns = [s + ".NS" for s in symbols if s and s.lower() != 'nan']
            if symbols_ns:
                return symbols_ns, f"✓ Fetched {len(symbols_ns)} constituents (Wikipedia fallback)"
        return None, "No symbol table found on Wikipedia page"
    except Exception as e:
        return None, f"Wikipedia fallback error: {e}"


def _fetch_us_index_from_wikipedia(index_name):
    """Scrape constituent tickers for a US index from Wikipedia."""
    wiki_urls = {
        "S&P 500":    "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "NASDAQ 100": "https://en.wikipedia.org/wiki/Nasdaq-100",
        "DOW JONES":  "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
    }
    url = wiki_urls.get(index_name)
    if not url:
        return None, f"No Wikipedia URL configured for {index_name}"
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        tables = pd.read_html(io.StringIO(response.text))
        for table in tables:
            cols_lower = [str(c).lower() for c in table.columns]
            symbol_col = None
            for candidate in ('symbol', 'ticker'):
                for i, c in enumerate(cols_lower):
                    if candidate in c:
                        symbol_col = table.columns[i]
                        break
                if symbol_col is not None:
                    break
            if symbol_col is None:
                continue
            raw = [str(s).strip() for s in table[symbol_col].dropna().tolist()]
            # Normalise BRK.B → BRK-B style; drop header echoes and junk rows
            symbols = []
            for s in raw:
                s = s.replace('.', '-')
                if s and s.lower() not in ('symbol', 'ticker', 'nan') and 1 <= len(s) <= 6:
                    symbols.append(s)
            if len(symbols) >= 10:
                return symbols, f"✓ Fetched {len(symbols)} constituents (Wikipedia)"
        return None, "No valid symbol table found on Wikipedia page"
    except Exception as e:
        return None, f"Wikipedia fetch error: {e}"


def get_us_index_symbols(index_name):
    """Get constituent stock tickers for a US index.

    Primary source: Wikipedia scrape. Fallback: hardcoded list for DOW JONES.
    Returns plain NYSE/NASDAQ tickers (no exchange suffix).
    """
    symbols, msg = _fetch_us_index_from_wikipedia(index_name)
    if symbols:
        return symbols, msg
    if index_name == "DOW JONES":
        return _DOW30_FALLBACK.copy(), f"✓ Loaded {len(_DOW30_FALLBACK)} DOW constituents (hardcoded fallback)"
    return None, f"Could not fetch constituents for '{index_name}': {msg}"


def get_global_macro_symbols():
    """Return the Global Macro bond ETF universe."""
    symbols = list(GLOBAL_MACRO_MAP.values())
    return symbols, f"✓ Loaded {len(symbols)} Global Macro instruments"


def get_global_index_symbols():
    """Return the Global Indexes universe — one benchmark index per country."""
    symbols = list(GLOBAL_INDEXES_MAP.values())
    return symbols, f"✓ Loaded {len(symbols)} global benchmark indexes"


def get_commodity_symbols(commodity_type=None):
    """Get commodity futures symbols."""
    if commodity_type is None:
        return list(COMMODITY_MAP.values()), f"✓ Fetched {len(COMMODITY_MAP)} commodities"
    symbol = COMMODITY_MAP.get(commodity_type)
    if symbol:
        return [symbol], f"✓ Fetched {commodity_type}"
    return None, f"Unknown commodity: {commodity_type}"


def get_currency_symbols(currency_pair=None):
    """Get currency pair symbols."""
    if currency_pair is None:
        return list(CURRENCY_MAP.values()), f"✓ Fetched {len(CURRENCY_MAP)} currency pairs"
    symbol = CURRENCY_MAP.get(currency_pair)
    if symbol:
        return [symbol], f"✓ Fetched {currency_pair}"
    return None, f"Unknown currency pair: {currency_pair}"


def get_crypto_symbols(crypto_name=None):
    """Get cryptocurrency symbols."""
    if crypto_name is None:
        return list(CRYPTO_MAP.values()), f"✓ Fetched {len(CRYPTO_MAP)} digital assets"
    symbol = CRYPTO_MAP.get(crypto_name)
    if symbol:
        return [symbol], f"✓ Fetched {crypto_name}"
    return None, f"Unknown crypto asset: {crypto_name}"


def get_etf_symbols():
    """Return the fixed ETF universe for analysis"""
    return ETF_LIST, f"✓ Loaded {len(ETF_LIST)} ETFs"


@st.cache_data(ttl=300, show_spinner=False)
def fetch_batch_data(stock_list, end_date=None, days_back=300, include_live=True):
    if end_date is None:
        end_date = datetime.date.today()
    
    download_end = end_date + datetime.timedelta(days=5)
    start_date = end_date - datetime.timedelta(days=days_back + 365)
    
    try:
        all_data = yf.download(
            stock_list,
            start=start_date,
            end=download_end,
            progress=False,
            auto_adjust=True,
            group_by='ticker',
            threads=True,
        )
        
        if all_data.empty:
            return None, "No data returned"
            
        if isinstance(all_data, pd.DataFrame) and isinstance(all_data.columns, pd.MultiIndex):
            data_dict = {}
            for ticker in stock_list:
                try:
                    ticker_df = all_data.xs(ticker, level=0, axis=1)
                    if not ticker_df.empty and not ticker_df['Close'].isnull().all():
                        data_dict[ticker] = ticker_df.copy()
                except KeyError:
                    pass
        elif isinstance(all_data, dict):
            data_dict = {t:df.copy() for t,df in all_data.items() if not df.empty and not df['Close'].isnull().all()}
        else:
             return None, "Unexpected data structure"

        if include_live and end_date == datetime.date.today() and data_dict:
            sample_df = list(data_dict.values())[0]
            sample_df.index = pd.to_datetime(sample_df.index)
            if sample_df.index.tz is not None:
                 sample_df.index = sample_df.index.tz_localize(None)
            
            has_today = any(idx.date() == datetime.date.today() for idx in sample_df.index)
            if not has_today:
                try:
                    live_data = yf.download(list(data_dict.keys()), period="1d", progress=False, auto_adjust=True, group_by='ticker')
                    if not live_data.empty:
                        for ticker in data_dict.keys():
                            try:
                                live_ticker = live_data.xs(ticker, level=0, axis=1)
                                if not live_ticker.empty and not live_ticker['Close'].isnull().all():
                                    hist_df = data_dict[ticker]
                                    hist_df.index = pd.to_datetime(hist_df.index)
                                    if hist_df.index.tz is not None: hist_df.index = hist_df.index.tz_localize(None)
                                    live_ticker.index = pd.to_datetime(live_ticker.index)
                                    if live_ticker.index.tz is not None: live_ticker.index = live_ticker.index.tz_localize(None)
                                    new_dates = live_ticker.index.difference(hist_df.index)
                                    if len(new_dates) > 0:
                                        data_dict[ticker] = pd.concat([hist_df, live_ticker.loc[new_dates]]).sort_index()
                            except KeyError: pass
                except Exception: pass
        return data_dict, f"✓ Downloaded {len(data_dict)} tickers"
    except Exception as e:
        return None, f"Download error: {e}"


def resample_to_weekly(df):
    if df is None or df.empty:
        return df
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    weekly = df.resample('W-MON', closed='left', label='left').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    return weekly

# ══════════════════════════════════════════════════════════════════════════════
# WRCI ENGINE: WAVE-REGIME COMPOSITE INDEX CALCULATION
# ══════════════════════════════════════════════════════════════════════════════

def calculate_wma(series, length):
    if length <= 1:
        return series
    weights = np.arange(1, length + 1)
    return series.rolling(window=length).apply(lambda vars: np.dot(vars, weights) / weights.sum(), raw=True)


def calculate_hma(series, length):
    if length <= 1:
        return series
    half_length = int(length / 2)
    sqrt_length = int(np.sqrt(length))
    wma_half = calculate_wma(series, half_length)
    wma_full = calculate_wma(series, length)
    diff = 2 * wma_half - wma_full
    return calculate_wma(diff, sqrt_length)


def calculate_ema(series, length):
    """
    Exponential Moving Average matched to TradingView's ta.ema.
    Initializes with SMA and follows the recursive formula.
    """
    if length <= 1:
        return series
    
    # Calculate initial SMA for startup
    sma = series.rolling(window=length, min_periods=length).mean()
    
    # Find the first valid SMA index
    first_idx = sma.first_valid_index()
    if first_idx is None:
        return pd.Series(np.nan, index=series.index)
        
    start_pos = series.index.get_loc(first_idx)
    alpha = 2.0 / (length + 1)
    
    # Recursive calculation
    values = series.values
    ema_values = np.empty(len(series))
    ema_values.fill(np.nan)
    ema_values[start_pos] = sma.loc[first_idx]
    
    for i in range(start_pos + 1, len(series)):
        if np.isnan(values[i]):
            # If current price is NaN, EMA is NaN but state is preserved
            ema_values[i] = np.nan
        else:
            # If previous EMA was NaN (due to NaN price), find the last valid EMA
            prev_ema = ema_values[i-1]
            if np.isnan(prev_ema):
                # Look back for last valid EMA to continue recursion
                # Standard TV behavior: if price was NaN, recursion skips it
                j = i - 1
                while j >= start_pos and np.isnan(ema_values[j]):
                    j -= 1
                prev_ema = ema_values[j] if j >= start_pos else values[i]
            
            ema_values[i] = (values[i] - prev_ema) * alpha + prev_ema
            
    return pd.Series(ema_values, index=series.index)


def calculate_linreg(series, length, offset=0):
    """Calculate the Linear Regression endpoint."""
    def _linreg_val(y):
        if np.isnan(y).any():
            return np.nan
        x = np.arange(len(y))
        slope, intercept = np.polyfit(x, y, 1)
        return slope * (len(y) - 1 - offset) + intercept

    return series.rolling(window=length).apply(_linreg_val, raw=True)


def calculate_true_range(df):
    """Standard True Range calculation."""
    prev_close = df['Close'].shift(1)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - prev_close).abs()
    tr3 = (df['Low'] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def compute_rsi(series, length=14):
    """RSI calculation using RMA (TradingView standard)."""
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    alpha = 1.0 / length
    roll_up = up.ewm(alpha=alpha, adjust=False).mean()
    roll_down = down.ewm(alpha=alpha, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def calculate_trend_count(series, length):
    trend = pd.Series(0.0, index=series.index)
    for i in range(1, length + 1):
        trend += np.where(series > series.shift(i), 1, -1)
    return trend


def run_full_analysis(df, reg_len=20, n1=10, n2=21, obLevel1=80, obLevel2=40, osLevel1=-80, osLevel2=-40):
    hlc3 = (df['High'] + df['Low'] + df['Close']) / 3.0
    vol = df['Volume']
    
    # Institutional Volume Fallback: Historically used for VWMA-based indicators on indexes; 
    # maintained for volume-trend calculations even after transition to EMA core.
    if vol.sum() == 0:
        vol = pd.Series(1.0, index=df.index)
    
    hma_p = calculate_hma(hlc3, 15)
    hma_v = calculate_hma(vol, 15)

    trend = calculate_trend_count(hma_p, reg_len)
    voltrend_raw = calculate_trend_count(hma_v, reg_len)

    coeff = 10.0 / reg_len
    norm_trend = (trend * coeff) * 10.0
    voltrend = voltrend_raw * coeff

    ap = hlc3
    esa = calculate_ema(ap, n1)
    d = calculate_ema((ap - esa).abs(), n1)
    ci = (ap - esa) / (0.015 * d).replace(0, np.nan)
    tci = calculate_ema(ci, n2)

    wt1 = tci
    wt2 = wt1.rolling(window=4).mean()

    # ── CONVICTION V3 ENGINE ────────────────────────────────────────────────
    # Component 1: Trend Strength (structural, slow, magnitude-aware)
    # Slope per bar = linreg endpoint(today) − linreg endpoint shifted back 1 bar on the same line.
    # Mirrors Pine's `ta.linreg(src, len, 0) - ta.linreg(src, len, 1)`.
    hma_close = calculate_hma(df['Close'], reg_len)
    slope     = calculate_linreg(hma_close, reg_len, offset=0) \
              - calculate_linreg(hma_close, reg_len, offset=1)
    avg_price = df['Close'].rolling(reg_len).mean().replace(0, np.nan)
    slope_pct = slope / avg_price
    
    tr = calculate_true_range(df)
    atr = tr.rolling(reg_len).mean()
    atr_pct = (atr / avg_price).replace(0, np.nan)
    
    trend_str = (slope_pct / atr_pct).clip(-3, 3) / 3.0  # bounded [-1, +1]

    # Component 2: Momentum Quality (tactical, medium)
    wt_sep = wt1 - wt2
    wt_sep_p = wt_sep / wt_sep.rolling(60).std().replace(0, 1)
    wt_sep_n = np.tanh(wt_sep_p / 2.0)  # bounded [-1, +1]

    # Component 3: Participation (volume + RSI confluence)
    vol_z = (df['Volume'] - df['Volume'].rolling(20).mean()) / df['Volume'].rolling(20).std().replace(0, 1)
    vol_n = np.tanh(vol_z / 2.0)
    price_dir = np.sign(df['Close'] - df['Close'].shift(5))
    participation = price_dir * vol_n.abs()  # [-1, +1]
    
    rsi_14 = compute_rsi(df['Close'], 14)
    rsi_norm = (rsi_14 - 50) / 50.0  # [-1, +1]
    flow = 0.7 * participation + 0.3 * rsi_norm

    # Composite Conviction [−100, +100]
    conviction = (100 * (0.50 * trend_str + 0.30 * wt_sep_n + 0.20 * flow)).fillna(0)

    # ── PULSE V3 ENGINE ─────────────────────────────────────────────────────
    # Use 3-bar velocity and 30-bar baseline (no overlap)
    conv_vel_3 = conviction - conviction.shift(3)
    baseline_30 = conviction.shift(3).rolling(27).mean()
    baseline_std = conviction.shift(3).rolling(27).std()
    
    NOISE_FLOOR = 1.5
    denom = np.maximum(baseline_std, NOISE_FLOOR)
    conv_z_3 = ((conviction - baseline_30) / denom).clip(-5, 5)
    
    vel_baseline = conv_vel_3.rolling(60).mean()
    vel_std = conv_vel_3.rolling(60).std()
    vel_z = ((conv_vel_3 - vel_baseline) / np.maximum(vel_std, NOISE_FLOOR)).clip(-5, 5)
    
    # Volume Factor [0.7, 1.0]
    vol_align = np.tanh(vol_z / 2.0) * np.sign(conv_vel_3)
    vol_factor = 0.85 + 0.15 * vol_align
    
    # Price-Action Factor [0.7, 1.0]
    ret_3 = (df['Close'] - df['Close'].shift(3)) / df['Close'].shift(3).replace(0, np.nan)
    atr_14 = tr.rolling(14).mean()
    atr_pct_14 = (atr_14 / df['Close']).replace(0, 1)
    ret_z = (ret_3 / atr_pct_14).clip(-5, 5)
    price_align = np.tanh(ret_z * np.sign(conv_vel_3) / 2.0)
    price_factor = 0.85 + 0.15 * price_align
    
    # Roll-over Correction (Amplify sign misalignment)
    sign_misalign = (np.sign(conviction) != np.sign(conv_vel_3)).astype(float)
    turn_amplifier = 1.0 + 0.30 * sign_misalign * np.tanh(np.abs(conv_vel_3) / 8.0)
    
    # Composite Pulse [−6, +6]
    pulse_core = np.sign(conv_vel_3) * np.sqrt((vel_z * conv_z_3).abs())
    pulse = (pulse_core * vol_factor * price_factor * turn_amplifier).clip(-6, 6).fillna(0)

    # ── UPDATE DATAFRAME ────────────────────────────────────────────────────
    df['Unified_Osc']      = wt1
    df['Signal_Line']      = wt2
    df['WT1']              = wt1
    df['Norm_Trend']       = norm_trend
    df['Conviction']       = conviction
    df['Pulse']            = pulse
    df['VolTrend']         = voltrend
    df['WT1_5ago']         = wt1.shift(5)
    df['Recent_Travel']    = wt1 - wt1.shift(5)
    df['Conviction_Delta'] = conviction.diff().fillna(0)
    df['Pulse_Delta']      = pulse.diff().fillna(0)

    # ── Step 2: Self-normalized thresholds for narrative matrix ───────────────
    # Rolling percentile boundaries replace hardcoded ±5
    pd_roll_std = df['Pulse_Delta'].rolling(30, min_periods=10).std().replace(0, 1.0)
    cd_roll_std = df['Conviction_Delta'].rolling(30, min_periods=10).std().replace(0, 1.0)

    # Normalize deltas to their local volatility (z-score of the delta)
    pd_norm = df['Pulse_Delta'] / pd_roll_std
    cd_norm = df['Conviction_Delta'] / cd_roll_std

    def categorize_narrative_v2(p_norm, c_norm):
        """
        Direction-aware narrative matrix.
        Uses normalized deltas (σ-units) instead of hardcoded ±5.
        Thresholds: ±1σ for ">>"/"<<", 0 for ">/"<"
        """
        p_state = ">>" if p_norm > 1.0 else ">" if p_norm > 0 else "<" if p_norm > -1.0 else "<<"
        c_state = ">>" if c_norm > 1.0 else ">" if c_norm > 0 else "<" if c_norm > -1.0 else "<<"

        matrix = {
            (">>", ">>"): ("SQUEEZE",  2.0, "#22c55e"),
            (">>", ">"):  ("HYPER",    1.6, "#34d399"),
            (">>", "<"):  ("EXHAUST",  0.6, "#fbbf24"),
            (">>", "<<"): ("CHAOS",    0.4, "#f87171"),
            (">",  ">>"): ("IGNITE",   1.4, "#4a9eff"),
            (">",  ">"):  ("ORGANIC",  1.2, "#38bdf8"),
            (">",  "<"):  ("EXPAND",   0.9, "#94a3b8"),
            (">",  "<<"): ("POP",      0.7, "#f87171"),
            ("<",  ">>"): ("HARDEN",   1.5, "#22c55e"),
            ("<",  ">"):  ("STEALTH",  1.3, "#818cf8"),
            ("<",  "<"):  ("RETRACE",  0.8, "#cbd5e1"),
            ("<",  "<<"): ("LIQUID",   0.3, "#ef4444"),
            ("<<", ">>"): ("LOAD",     1.7, "#4a9eff"),
            ("<<", ">"):  ("TRAP",     0.5, "#fbbf24"),
            ("<<", "<"):  ("CAPITUL",  0.2, "#ef4444"),
            ("<<", "<<"): ("CRASH",    0.1, "#b91c1c"),
        }
        return matrix.get((p_state, c_state), ("NEUTRAL", 1.0, "#94a3b8"))

    narrative_results = [
        categorize_narrative_v2(p, c)
        for p, c in zip(pd_norm.fillna(0), cd_norm.fillna(0))
    ]
    df['Narrative']      = [n[0] for n in narrative_results]
    df['Priority_Mult']  = [n[1] for n in narrative_results]
    df['Narrative_Color']= [n[2] for n in narrative_results]

    # ── Step 3: Zone Depth Factor ─────────────────────────────────────────────
    # Rewards depth of oscillator position (composite_line is WRCI WT1)
    osc_val = wt1
    
    # Zone boundaries (consistent with WRCI script)
    obLevel1, obLevel2 = 80, 40

    # Bullish zone depth: how deep into oversold are we? (OS = negative WT1)
    bull_zone_depth = ((-osc_val - obLevel2) / (obLevel1 - obLevel2)).clip(0, 1).fillna(0)
    # Bearish zone depth: how deep into overbought are we?
    bear_zone_depth = ((osc_val - obLevel2) / (obLevel1 - obLevel2)).clip(0, 1).fillna(0)

    df['Bull_Zone_Depth'] = bull_zone_depth
    df['Bear_Zone_Depth'] = bear_zone_depth

    # Base Crossings
    sig_bull_cross = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
    sig_bear_cross = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1))

    # Set B: Crossover — Contrarian signals inside extreme zones
    crossover_long  = sig_bear_cross & (wt1 < osLevel2)
    crossover_short = sig_bull_cross & (wt1 > obLevel2)

    # Set A: Momentum — Trend signals (Mutually Exclusive with Set B)
    momentum_long   = sig_bull_cross & (~crossover_short)
    momentum_short  = sig_bear_cross & (~crossover_long)

    # Set C: Threshold — freshly entering OS/OB zone with signal-line validation
    threshold_long  = (wt1 < osLevel2) & (wt1.shift(1) >= osLevel2) & (wt2 > osLevel2)
    threshold_short = (wt1 > obLevel2) & (wt1.shift(1) <= obLevel2) & (wt2 < obLevel2)

    df['long_cond'] = momentum_long
    df['short_cond'] = momentum_short
    df['long_cond_comp'] = crossover_long
    df['short_cond_comp'] = crossover_short
    df['long_cond_wt'] = threshold_long
    df['short_cond_wt'] = threshold_short

    df['Condition'] = np.select(
        [wt1 > obLevel1, wt1 > obLevel2, wt1 < osLevel1, wt1 < osLevel2],
        ['OB Extreme', 'OB', 'OS Extreme', 'OS'],
        default='Neutral'
    )

    # Set D: Squeeze — Squeeze Momentum
    df = compute_squeeze_momentum(df, length=20)
    sqz_release = df['SQZ_Off'] & df['SQZ_On'].shift(1)
    df['long_cond_sqz'] = sqz_release & (df['SQZ_Val'] > df['SQZ_Val'].shift(1))
    df['short_cond_sqz'] = sqz_release & (df['SQZ_Val'] < df['SQZ_Val'].shift(1))

    return df

# ══════════════════════════════════════════════════════════════════════════════
# SQUEEZE MOMENTUM ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def compute_squeeze_momentum(df: pd.DataFrame, length: int = 20, mult: float = 1.5, 
                             length_kc: int = 20, mult_kc: float = 1.5, 
                             use_true_range: bool = True) -> pd.DataFrame:
    """
    Squeeze Momentum Indicator
    Ported exactly from Pine Script.
    """
    df = df.copy()
    
    # Calculate BB
    source = df['Close']
    basis = source.rolling(window=length).mean()
    dev = mult * source.rolling(window=length).std(ddof=0)
    upperBB = basis + dev
    lowerBB = basis - dev
    
    # Calculate KC
    ma = source.rolling(window=length_kc).mean()
    if use_true_range:
        prev_close = df['Close'].shift(1)
        tr1 = df['High'] - df['Low']
        tr2 = (df['High'] - prev_close).abs()
        tr3 = (df['Low'] - prev_close).abs()
        range_val = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    else:
        range_val = df['High'] - df['Low']
        
    rangema = range_val.rolling(window=length_kc).mean()
    upperKC = ma + rangema * mult_kc
    lowerKC = ma - rangema * mult_kc
    
    # Squeeze States
    sqzOn = (lowerBB > lowerKC) & (upperBB < upperKC)
    sqzOff = (lowerBB < lowerKC) & (upperBB > upperKC)
    noSqz = (~sqzOn) & (~sqzOff)
    
    # Linear Regression of Delta
    highest_high = df['High'].rolling(window=length).max()
    lowest_low = df['Low'].rolling(window=length).min()
    
    avg_hl = (highest_high + lowest_low) / 2.0
    sma_close = df['Close'].rolling(window=length).mean()
    
    inner_avg = (avg_hl + sma_close) / 2.0
    delta = source - inner_avg
    
    # Real ta.linreg endpoint
    val = calculate_linreg(delta, length, 0)
    
    # Assign Colors
    prev_val = val.shift(1).fillna(0)
    bcolor = np.where(
        val > 0,
        np.where(val > prev_val, "lime", "green"),
        np.where(val < prev_val, "red", "maroon")
    )
    scolor = np.where(noSqz, "blue", np.where(sqzOn, "black", "gray"))
    
    # Add to DataFrame
    df['SQZ_Val'] = val
    df['SQZ_BColor'] = bcolor
    df['SQZ_SColor'] = scolor
    df['SQZ_On'] = sqzOn
    df['SQZ_Off'] = sqzOff
    df['SQZ_NoSqz'] = noSqz
    
    return df

# ══════════════════════════════════════════════════════════════════════════════
# REGIME INTELLIGENCE ENGINE (NIRNAY FEATURES)
# ══════════════════════════════════════════════════════════════════════════════

class AdaptiveHMM:
    """Hidden Markov Model for regime state discovery - classifies WRCI signals"""
    
    def __init__(self):
        self.n_states = 3
        self.transition_matrix = np.array([
            [0.85, 0.10, 0.05],
            [0.10, 0.80, 0.10],
            [0.05, 0.10, 0.85]
        ])
        self.emission_means = np.array([0.6, 0.0, -0.6])
        self.emission_stds = np.array([0.3, 0.25, 0.3])
        self.state_probabilities = np.array([0.33, 0.34, 0.33])
        self.observation_history = []
        self.state_history = []
    
    def _gaussian_pdf(self, x, mean, std):
        if std < 1e-8:
            return 1.0 if abs(x - mean) < 1e-8 else 0.0
        return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
    
    def update(self, observation):
        self.observation_history.append(observation)
        predicted = self.transition_matrix.T @ self.state_probabilities
        emissions = np.array([self._gaussian_pdf(observation, self.emission_means[s], self.emission_stds[s]) for s in range(3)])
        updated = emissions * predicted
        total = updated.sum()
        if total > 1e-10:
            updated /= total
        else:
            updated = np.array([0.33, 0.34, 0.33])
        self.state_probabilities = updated
        most_likely = np.argmax(updated)
        self.state_history.append(most_likely)
        
        if len(self.observation_history) >= 10:
            recent_obs = np.array(self.observation_history[-50:])
            recent_states = self.state_history[-len(recent_obs):]
            for state in range(3):
                mask = np.array(recent_states) == state
                if mask.sum() >= 2:
                    state_obs = recent_obs[mask]
                    self.emission_means[state] = 0.9 * self.emission_means[state] + 0.1 * np.mean(state_obs)
                    self.emission_stds[state] = 0.9 * self.emission_stds[state] + 0.1 * max(np.std(state_obs), 0.1)
        
        return {"BULL": updated[0], "NEUTRAL": updated[1], "BEAR": updated[2]}
    
    def reset(self):
        self.state_probabilities = np.array([0.33, 0.34, 0.33])
        self.observation_history = []
        self.state_history = []


class GARCHDetector:
    """GARCH-inspired volatility regime detection for WRCI signal variance"""
    
    def __init__(self):
        self.current_variance = 0.04
        self.omega = 0.0001
        self.alpha = 0.1
        self.beta = 0.85
        self.long_term_mean = 0.04
        self.shock_history = []
    
    def update(self, shock):
        self.shock_history.append(shock)
        shock_sq = shock ** 2
        new_var = self.omega + self.alpha * shock_sq + self.beta * self.current_variance
        self.current_variance = np.clip(new_var, 0.001, 1.0)
        
        if len(self.shock_history) >= 10:
            realized = np.var(self.shock_history[-min(50, len(self.shock_history)):])
            self.long_term_mean = 0.95 * self.long_term_mean + 0.05 * realized
        
        return np.sqrt(self.current_variance)
    
    def get_regime(self):
        current_vol = np.sqrt(self.current_variance)
        long_term_vol = np.sqrt(self.long_term_mean)
        ratio = current_vol / long_term_vol if long_term_vol > 0 else 1.0
        
        if ratio < 0.6:
            return "LOW", 1.3
        elif ratio < 0.9:
            return "NORMAL", 1.0
        elif ratio < 1.4:
            return "HIGH", 0.8
        else:
            return "EXTREME", 0.6
    
    def reset(self):
        self.current_variance = 0.04
        self.shock_history = []


class CUSUMDetector:
    """CUSUM change point detection for WRCI signal regime shifts"""
    
    def __init__(self, threshold=4.0, drift=0.5):
        self.threshold = threshold
        self.drift = drift
        self.positive_cusum = 0.0
        self.negative_cusum = 0.0
        self.value_history = []
        self.running_mean = 0.0
        self.running_std = 1.0
    
    def update(self, value):
        self.value_history.append(value)
        
        if len(self.value_history) >= 3:
            recent = self.value_history[-min(20, len(self.value_history)):]
            self.running_mean = np.mean(recent)
            self.running_std = max(np.std(recent), 0.1)
        
        z = (value - self.running_mean) / self.running_std
        self.positive_cusum = max(0, self.positive_cusum + z - self.drift)
        self.negative_cusum = max(0, self.negative_cusum - z - self.drift)
        
        change_detected = self.positive_cusum > self.threshold or self.negative_cusum > self.threshold
        
        if change_detected:
            self.positive_cusum = 0
            self.negative_cusum = 0
        
        return change_detected
    
    def reset(self):
        self.positive_cusum = 0.0
        self.negative_cusum = 0.0
        self.value_history = []


class AdaptiveKalmanFilter:
    """Kalman filter for WRCI signal smoothing"""
    
    def __init__(self, process_var=0.01, measurement_var=0.1):
        self.estimate = 0.0
        self.error_covariance = 1.0
        self.process_variance = process_var
        self.measurement_variance = measurement_var
        self.innovation_history = []
    
    def update(self, measurement):
        predicted_estimate = self.estimate
        predicted_covariance = self.error_covariance + self.process_variance
        innovation = measurement - predicted_estimate
        self.innovation_history.append(innovation)
        if len(self.innovation_history) > 50:
            self.innovation_history.pop(0)
        innovation_cov = predicted_covariance + self.measurement_variance
        kalman_gain = predicted_covariance / innovation_cov
        self.estimate = predicted_estimate + kalman_gain * innovation
        self.error_covariance = (1 - kalman_gain) * predicted_covariance
        
        if len(self.innovation_history) >= 5:
            innovation_var = np.var(self.innovation_history[-min(20, len(self.innovation_history)):])
            self.measurement_variance = 0.9 * self.measurement_variance + 0.1 * innovation_var
        
        return self.estimate
    
    def reset(self, initial=0.0):
        self.estimate = initial
        self.error_covariance = 1.0
        self.innovation_history = []


def run_regime_analysis(df):
    """Apply Regime Intelligence to WRCI-computed dataframe"""
    hmm = AdaptiveHMM()
    garch = GARCHDetector()
    cusum = CUSUMDetector()
    kalman = AdaptiveKalmanFilter()
    
    regimes = []
    hmm_bulls = []
    hmm_bears = []
    vol_regimes = []
    change_points = []
    confidences = []
    signal_history = []
    
    unified_vals = df['Unified_Osc'].values
    
    for i in range(len(df)):
        sig = unified_vals[i] if not np.isnan(unified_vals[i]) else 0
        filtered = kalman.update(sig / 10.0)
        
        shock = sig - signal_history[-1] if signal_history else 0
        garch.update(shock)
        vol_regime, _ = garch.get_regime()
        
        hmm_probs = hmm.update(filtered)
        change = cusum.update(filtered)
        
        bull_p = hmm_probs['BULL']
        bear_p = hmm_probs['BEAR']
        
        if change:
            regime = "TRANSITION"
        elif bull_p > 0.6:
            regime = "BULL"
        elif bear_p > 0.6:
            regime = "BEAR"
        elif bull_p > 0.4:
            regime = "WEAK_BULL"
        elif bear_p > 0.4:
            regime = "WEAK_BEAR"
        else:
            regime = "NEUTRAL"
        
        regimes.append(regime)
        hmm_bulls.append(bull_p)
        hmm_bears.append(bear_p)
        vol_regimes.append(vol_regime)
        change_points.append(change)
        confidences.append(max(bull_p, bear_p, hmm_probs['NEUTRAL']))
        signal_history.append(sig)
    
    df['Regime'] = regimes
    df['HMM_Bull'] = hmm_bulls
    df['HMM_Bear'] = hmm_bears
    df['Vol_Regime'] = vol_regimes
    df['Change_Point'] = change_points
    df['Confidence'] = confidences
    
    return df


def calculate_divergences(df):
    """Calculate bullish and bearish divergences for WRCI signals"""
    osc_rising = df['Unified_Osc'] > df['Unified_Osc'].shift(1)
    price_falling = df['Close'] < df['Close'].shift(1)
    osc_falling = df['Unified_Osc'] < df['Unified_Osc'].shift(1)
    price_rising = df['Close'] > df['Close'].shift(1)
    
    df['Bullish_Div'] = osc_rising & price_falling & (df['Unified_Osc'] < -5)
    df['Bearish_Div'] = osc_falling & price_rising & (df['Unified_Osc'] > 5)
    
    return df

# ══════════════════════════════════════════════════════════════════════════════
# DATA HANDLING & UTILITIES
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# UI HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def render_footer():
    """Render app footer with copyright and version info."""
    ist = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5, minutes=30)))
    st.markdown(f"""
    <div class="app-footer">
        <div class="content">
            © {ist.year} <strong>Sanket</strong> &nbsp;·&nbsp; @thebullishvalue &nbsp;·&nbsp; {VERSION} &nbsp;·&nbsp; {ist.strftime("%Y-%m-%d %H:%M:%S IST")}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_landing_page():
    """Render landing page with system overview."""
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class='system-card portfolio'>
            <h3>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
                PULSE ENGINE
            </h3>
            <p>Sanket Pulse Engine identifies Abnormal Acceleration (Velocity * Z-Score) to surface high-conviction ignition events.</p>
            <div class='spec'>
                <span>Primary:</span> Abnormal Acceleration (Pulse)<br>
                <span>Secondary:</span> Signal Conviction Score<br>
                <span>Metric:</span> 5D Velocity * 20D Vol Z-Score<br>
                <span>Sorting:</span> Rank by Pulse Strength
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='system-card regime'>
            <h3>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/></svg>
                SIGNAL STRUCTURE
            </h3>
            <p>Hierarchical signal generation (Sets A-D) contextualized by Pulse and structural trend regime alignment.</p>
            <div class='spec'>
                <span>Sets:</span> Momentum / Crossover / Threshold / Squeeze<br>
                <span>Ranking:</span> Sorted by Abnormal Acceleration<br>
                <span>Long/Short:</span> Dual-sided directional logic<br>
                <span>Timing:</span> Age-weighted signal aging
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='system-card strategies'>
            <h3>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>
                UNIVERSE COVERAGE
            </h3>
            <p>Scan F&O stocks or entire index constituents. Filter by timeframe. Customize sensitivity thresholds.</p>
            <div class='spec'>
                <span>Universes:</span> F&O + Indices + ETFs<br>
                <span>Timeframes:</span> Daily · Weekly<br>
                <span>Symbols:</span> Up to 500<br>
                <span>Modes:</span> Point + Time Series
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class='landing-prompt'>
        <h4>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8"/></svg>
            AWAITING ANALYSIS PARAMETERS
        </h4>
        <p>Configure via the <strong>Sidebar</strong>: select <strong>Universe</strong>, <strong>Timeframe</strong>, <strong>Analysis Mode</strong>, and <strong>Engine Settings</strong>.<br>
           Click <strong>RUN SCREENER</strong> to analyze and discover today's signals.<br>
           <span style="color:var(--ink-secondary); font-size:0.85em; margin-top:0.5rem; display:inline-block;">System will compute Wave Trend oscillations · Analyze Abnormal Acceleration · Rank by Pulse strength</span></p>
    </div>
    """, unsafe_allow_html=True)


def get_signal_strength_score(row):
    """Calculate signal strength from magnitude with diminishing returns above 50.

    Returns: Strength score (0-100) where magnitude 0-50 = linear, >50 = diminishing returns.
    """
    base_score = abs(row.get('Signal', 0))
    if base_score > 50:
        base_score = 50 + (base_score - 50) * 0.5
    return min(100, base_score)


def render_signal_detail_card(symbol, price, signal_val, trend_val, zone, signal_type, rsi_val, osc_val, zscore_val, ma_count):
    """Render detailed signal card with strength indicator and technical confirmations.

    Displays signal magnitude, trend direction, zone status, and technical confirmations
    (RSI levels, oscillator state) to provide comprehensive signal context.

    Returns: Renders to Streamlit; no return value.
    """
    signal_strength = get_signal_strength_score({'Signal': signal_val})

    # Determine signal quality
    if signal_strength >= 65:
        icon = SVGS["DOT"].replace('currentColor', 'var(--emerald)')
        label = "Strong"
    elif signal_strength >= 50:
        icon = SVGS["DOT"].replace('currentColor', 'var(--info)')
        label = "Moderate"
    elif signal_strength >= 35:
        icon = SVGS["DOT"].replace('currentColor', 'var(--amber)')
        label = "Weak"
    else:
        icon = SVGS["DOT"].replace('currentColor', 'var(--rose)')
        label = "Very Weak"

    # Technical confirmation indicators
    confirmations = []
    if pd.notna(rsi_val):
        if rsi_val > 70:
            confirmations.append(("RSI Overbought", SVGS["UP"].replace('currentColor', 'var(--rose)'), "var(--rose)"))
        elif rsi_val < 30:
            confirmations.append(("RSI Oversold", SVGS["DOWN"].replace('currentColor', 'var(--emerald)'), "var(--emerald)"))
        else:
            confirmations.append(("RSI Neutral", "—", "var(--amber)"))

    trend_label = "Strong" if abs(trend_val) > 30 else "Moderate" if abs(trend_val) > 15 else "Weak"
    trend_icon = SVGS["UP"].replace('currentColor', 'var(--emerald)') if trend_val > 0 else SVGS["DOWN"].replace('currentColor', 'var(--rose)')

    st.markdown(f"""
    <div style="background: linear-gradient(145deg, var(--glass) 0%, rgba(17, 24, 39, 0.4) 100%);
                border: 1px solid var(--border); border-radius: 10px; padding: 1.25rem; margin-bottom: 0.75rem;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.75rem;">
            <div>
                <div style="font-family: var(--display); font-size: 1rem; font-weight: 700; color: var(--ink-primary);">
                    {symbol.replace('.NS', '')}
                </div>
                <div style="font-family: var(--data); font-size: 0.8rem; color: var(--ink-secondary);">
                    {price:,.2f}
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-family: var(--data); font-size: 1.25rem; font-weight: 700; color: var(--amber);">
                    {signal_strength:.0f}%
                </div>
                <div style="font-family: var(--data); font-size: 0.7rem; color: var(--ink-secondary); text-transform: uppercase; letter-spacing: 0.05em;">
                    Strength
                </div>
            </div>
        </div>

        <div style="background: rgba(255,255,255,0.02); border-radius: 6px; padding: 0.75rem; margin-bottom: 0.75rem;">
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; font-family: var(--data); font-size: 0.75rem;">
                <div>
                    <span style="color: var(--ink-tertiary); text-transform: uppercase; font-size: 0.65rem;">Signal Type</span><br>
                    <span style="color: var(--ink-primary); font-weight: 600;">{icon} {signal_type}</span>
                </div>
                <div>
                    <span style="color: var(--ink-tertiary); text-transform: uppercase; font-size: 0.65rem;">Trend Strength</span><br>
                    <span style="color: var(--ink-primary); font-weight: 600;">{trend_icon} {trend_label}</span>
                </div>
                <div>
                    <span style="color: var(--ink-tertiary); text-transform: uppercase; font-size: 0.65rem;">Zone</span><br>
                    <span style="color: var(--ink-primary); font-weight: 600;">{zone}</span>
                </div>
                <div>
                    <span style="color: var(--ink-tertiary); text-transform: uppercase; font-size: 0.65rem;">MA Alignment</span><br>
                    <span style="color: var(--ink-primary); font-weight: 600;">{int(ma_count) if pd.notna(ma_count) else 0}/5</span>
                </div>
            </div>
        </div>

        <div style="display: flex; gap: 0.5rem; flex-wrap: wrap; font-family: var(--data); font-size: 0.7rem;">
            <div style="padding: 0.35rem 0.75rem; background: rgba(212,168,83,0.1); border-radius: 4px; border: 1px solid rgba(212,168,83,0.2); color: var(--amber);">
                ◈ Signal: {signal_val:+.2f}
            </div>
            <div style="padding: 0.35rem 0.75rem; background: rgba(45,212,168,0.1); border-radius: 4px; border: 1px solid rgba(45,212,168,0.2); color: var(--emerald);">
                ≈ Wave: {osc_val:+.2f}
            </div>
            {f'<div style="padding: 0.35rem 0.75rem; background: rgba(232,85,90,0.1); border-radius: 4px; border: 1px solid rgba(232,85,90,0.2); color: var(--rose);">RSI: {rsi_val:.0f}</div>' if pd.notna(rsi_val) else ''}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# UI COMPONENTS & SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        # Centered Masthead
        st.markdown("""
        <div style="text-align:center; padding:0.75rem 0 1.5rem 0;">
            <div style="font-family:var(--display); font-size:1.5rem; font-weight:800; color:var(--amber); letter-spacing:-0.02em;">SANKET</div>
            <div style="font-family:var(--data); color:var(--ink-tertiary); font-size:0.65rem; margin-top:0.2rem; letter-spacing:0.08em; text-transform:uppercase;">संकेत | Signal Screener</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Analysis Depth
        st.markdown('<div class="sidebar-title">Analysis Depth</div>', unsafe_allow_html=True)
        timeframe = st.radio("Timeframe", TIMEFRAME_OPTIONS, horizontal=True, label_visibility="collapsed")
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Universe Selection
        st.markdown('<div class="sidebar-title">Universe Selection</div>', unsafe_allow_html=True)
        universe = st.selectbox("Universe", UNIVERSE_OPTIONS, label_visibility="collapsed")
        selected_index = None

        if universe == "India Indexes":
            selected_index = st.selectbox("Index", INDEX_LIST, index=INDEX_LIST.index("Benchmark Indexes"), label_visibility="collapsed")
        elif universe == "Global Indexes":
            selected_index = "Global Benchmark Indexes"
        elif universe == "US Indexes":
            selected_index = st.selectbox("Index", US_INDEX_LIST, index=US_INDEX_LIST.index("DOW JONES"), label_visibility="collapsed")
        elif universe == "ETF Index":
            selected_index = "NSE ETF Universe"
        elif universe == "Commodities":
            selected_index = "Global Commodities"
        elif universe == "Currency":
            selected_index = "Major FX Pairs"
        elif universe == "Crypto":
            selected_index = "Digital Assets (Top 20)"
        elif universe == "Global Macro":
            selected_index = "Global Macro Bonds"

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        analysis_mode = st.radio("Mode", ["Single Date", "Historical Range", "Correlation Analysis", "Pulse Narrative"], horizontal=True, label_visibility="collapsed")

        if analysis_mode in ["Single Date", "Pulse Narrative"]:
            st.markdown('<div class="sidebar-title">Analysis Date</div>', unsafe_allow_html=True)
            analysis_date = st.date_input("Date", datetime.date.today(), max_value=datetime.date.today(), label_visibility="collapsed")
            start_date_hist, end_date_hist = None, None
            corr_target_ticker, corr_lookback, corr_method = None, 90, "Pearson"
        elif analysis_mode == "Historical Range":
            st.markdown('<div class="sidebar-title">Analysis Date</div>', unsafe_allow_html=True)
            analysis_date = datetime.date.today()
            col_date1, col_date2 = st.columns(2)
            with col_date1: start_date_hist = st.date_input("Start", datetime.date.today() - datetime.timedelta(days=300), label_visibility="collapsed")
            with col_date2: end_date_hist = st.date_input("End", datetime.date.today(), label_visibility="collapsed")
            corr_target_ticker, corr_lookback, corr_method = None, 90, "Pearson"
        else:  # Correlation Analysis mode
            st.markdown('<div class="sidebar-title">Analysis Date</div>', unsafe_allow_html=True)
            analysis_date = st.date_input("Analysis Date", datetime.date.today(), max_value=datetime.date.today(), label_visibility="collapsed")
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            start_date_hist, end_date_hist = None, None

            # Target Asset Panel
            st.markdown('<div class="sidebar-title">Target Asset</div>', unsafe_allow_html=True)
            target_class = st.selectbox("Asset Class", ["Commodities", "Currency", "Crypto", "Global Indexes"], label_visibility="collapsed")

            # Build target asset options from maps
            if target_class == "Commodities":
                target_map = COMMODITY_MAP
                target_display_names = list(COMMODITY_MAP.keys())
            elif target_class == "Currency":
                target_map = CURRENCY_MAP
                target_display_names = list(CURRENCY_MAP.keys())
            elif target_class == "Crypto":
                target_map = CRYPTO_MAP
                target_display_names = list(CRYPTO_MAP.keys())
            else:  # Global Indexes
                target_map = GLOBAL_INDEXES_MAP
                target_display_names = list(GLOBAL_INDEXES_MAP.keys())

            target_selected = st.selectbox("Asset", target_display_names, label_visibility="collapsed")
            corr_target_ticker = target_map.get(target_selected, target_selected)

            # Correlation params
            st.markdown('<div class="sidebar-title">Analysis Params</div>', unsafe_allow_html=True)
            corr_lookback_str = st.selectbox("Lookback", ["30D", "60D", "90D", "180D"], label_visibility="collapsed")
            corr_lookback = int(corr_lookback_str.replace("D", ""))
            corr_method = st.selectbox("Method", ["Pearson", "Spearman"], label_visibility="collapsed")

        # WRCI Engine — hardcoded defaults
        reg_len, wt_n1, wt_n2 = 20, 10, 21
        obLevel1, obLevel2, osLevel1, osLevel2 = 80, 40, -80, -40

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Run Button
        run_clicked = st.button("◈ RUN SCREENER", type="primary", width='stretch')

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # System Spec Card
        try:
            if universe == "India Indexes" and selected_index:
                symbols_count = len(get_index_stock_list(selected_index)[0] or [])
                universe_display = selected_index
            elif universe == "Global Indexes":
                symbols_count = len(GLOBAL_INDEXES_MAP)
                universe_display = "Global Benchmark Indexes"
            elif universe == "US Indexes" and selected_index:
                symbols_count = len(get_us_index_symbols(selected_index)[0] or [])
                universe_display = selected_index
            elif universe == "Commodities" and selected_index:
                symbols_count = len(get_commodity_symbols(selected_index)[0] or [])
                universe_display = selected_index
            elif universe == "Currency" and selected_index:
                symbols_count = len(get_currency_symbols(selected_index)[0] or [])
                universe_display = selected_index
            elif universe == "ETF Index":
                symbols_count = len(ETF_LIST)
                universe_display = "NSE ETFs"
            elif universe == "Global Macro":
                symbols_count = len(GLOBAL_MACRO_MAP)
                universe_display = "Global Macro Bonds"
            else:
                symbols_count = "—"
                universe_display = universe
        except:
            symbols_count = "—"
            universe_display = universe

        # Build system spec card
        spec_html = f"""
        <div class="system-spec">
            <div class="spec-row"><span class="spec-label">Version</span><span class="spec-value">{VERSION}</span></div>
            <div class="spec-row"><span class="spec-label">Universe</span><span class="spec-value" style="font-size:0.7rem;">{universe_display}</span></div>
            <div class="spec-row"><span class="spec-label">Timeframe</span><span class="spec-value">{timeframe}</span></div>
            <div class="spec-row"><span class="spec-label">Mode</span><span class="spec-value" style="font-size:0.7rem;">{analysis_mode}</span></div>
        """

        # Add Target row only in Correlation Analysis mode
        if analysis_mode == "Correlation Analysis":
            spec_html += f'<div class="spec-row"><span class="spec-label">Target</span><span class="spec-value" style="font-size:0.7rem;">{target_selected}</span></div>'

        spec_html += """
        </div>
        """
        st.markdown(spec_html, unsafe_allow_html=True)

        return universe, selected_index, analysis_date, reg_len, wt_n1, wt_n2, (obLevel1, obLevel2, osLevel1, osLevel2), timeframe, analysis_mode, start_date_hist, end_date_hist, run_clicked, corr_target_ticker, corr_lookback, corr_method


# ══════════════════════════════════════════════════════════════════════════════
# MAIN SCREENER FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def run_screener_analysis(universe, selected_index, analysis_date, reg_len, wt_n1, wt_n2, levels, timeframe, show_progress=True, external_progress_slot=None, progress_offset=0, progress_scale=100):
    """Execute WRCI momentum analysis on universe symbols and return ranked signals.

    Fetches market data for universe, computes Wave Trend oscillations, calculates
    signal magnitude and trend values, detects overbought/oversold zones.

    Args:
        external_progress_slot: Optional Streamlit container for external progress tracking (e.g., from correlation analysis)
        progress_offset: Starting percentage for external progress tracking (default 0)
        progress_scale: Scale factor for progress percentage within external slot (default 100 = full)

    Returns: DataFrame with signals ranked by magnitude, or None on error.
    """
    obLevel1, obLevel2, osLevel1, osLevel2 = levels
    progress_slot = external_progress_slot if external_progress_slot is not None else (st.empty() if show_progress else None)

    if show_progress or external_progress_slot is not None:
        pct_val = progress_offset + (5 * progress_scale / 100)
        progress_bar(progress_slot, pct_val, "Initializing WRCI engine", f"Universe: {universe}")
    
    console.start_phase("DATA ACQUISITION", 1, 2)
    console.section("Universe Configuration")
    console.item("Universe", universe)
    console.item("Selected Index", selected_index)
    console.item("Timeframe", timeframe)

    if universe == "India Indexes":
        stock_list, msg = get_index_stock_list(selected_index)
    elif universe == "Global Indexes":
        stock_list, msg = get_global_index_symbols()
    elif universe == "US Indexes":
        stock_list, msg = get_us_index_symbols(selected_index)
    elif universe == "Commodities":
        stock_list, msg = get_commodity_symbols(None)
    elif universe == "Currency":
        stock_list, msg = get_currency_symbols(None)
    elif universe == "Crypto":
        stock_list, msg = get_crypto_symbols(None)
    elif universe == "ETF Index":
        stock_list, msg = get_etf_symbols()
    elif universe == "Global Macro":
        stock_list, msg = get_global_macro_symbols()
    else:
        stock_list, msg = None, f"Unknown universe: {universe}"

    if not stock_list:
        console.error(msg)
        st.error(msg)
        return None

    console.success(f"Fetched {len(stock_list)} symbols for {selected_index}")
    console.section("Market Data Fetch")
    if show_progress or external_progress_slot is not None:
        pct_val = progress_offset + (15 * progress_scale / 100)
        progress_bar(progress_slot, pct_val, "Fetching Market Data", f"{len(stock_list)} stocks")
    data_dict, fetch_msg = fetch_batch_data(stock_list, end_date=analysis_date)

    if not data_dict:
        console.error(fetch_msg)
        st.error(fetch_msg)
        return None

    console.success(f"Successfully downloaded data for {len(data_dict)} stocks")
    console.end_phase("DATA ACQUISITION")

    console.start_phase("WRCI MOMENTUM ANALYSIS", 2, 2)

    console.section("Analysis Parameters")
    console.item("Timeframe", timeframe)
    console.item("Regression Length", reg_len)
    console.item("Wave Trend", f"N1={wt_n1}  N2={wt_n2}")
    console.item("OB Levels", f"{obLevel1} / {obLevel2}")
    console.item("OS Levels", f"{osLevel1} / {osLevel2}")
    console.item("Instruments", f"{len(data_dict)} of {len(stock_list)} fetched successfully")
    if show_progress or external_progress_slot is not None:
        pct_val = progress_offset + (20 * progress_scale / 100)
        progress_bar(progress_slot, pct_val, "Analyzing WRCI momentum", f"{len(data_dict)} stocks")

    results = []

    _tf_label = "weekly" if timeframe == "Weekly" else "daily"
    console.section(f"Signal Analysis — {len(data_dict)} {_tf_label} instruments")

    for i, (ticker, df) in enumerate(data_dict.items()):
        try:
            pct = int(progress_offset + (20 + (i + 1) / len(data_dict) * 75) * progress_scale / 100)
            if show_progress or external_progress_slot is not None:
                progress_bar(progress_slot, pct, f"Analyzing Signals", f"{i + 1}/{len(data_dict)} stocks")

            if timeframe == "Weekly":
                df = resample_to_weekly(df)

            if len(df) < reg_len + 30:
                console.detail(f"{ticker}: Skipped (Insufficient data: {len(df)} rows)")
                continue

            df = run_full_analysis(df, reg_len, wt_n1, wt_n2, obLevel1, obLevel2, osLevel1, osLevel2)
            df = run_regime_analysis(df)        # adds HMM_Bull/Bear, Vol_Regime, Change_Point, Confidence
            df = calculate_divergences(df)      # adds Bullish_Div, Bearish_Div

            # Sample at analysis_date or last available
            df.index = pd.to_datetime(df.index)
            target_dt = pd.to_datetime(analysis_date)

            if target_dt in df.index:
                idx_pos = df.index.get_loc(target_dt)
            else:
                idx_pos = len(df) - 1

            if idx_pos < 5:
                continue

            # Get historical signals for tracking (Today, 1d, 2d, 3d, Within 5d)
            sample_range = df.iloc[max(0, idx_pos - 5) : idx_pos + 1]

            last_row = df.iloc[idx_pos]

            # Build Signal String — priority: Set B > Set A > Set C > Set D > Zone
            signal_type = "-"
            if last_row['long_cond_comp']:
                signal_type = "B: Long"
            elif last_row['short_cond_comp']:
                signal_type = "B: Short"
            elif last_row['long_cond']:
                signal_type = "A: Long"
            elif last_row['short_cond']:
                signal_type = "A: Short"
            elif last_row['long_cond_wt']:
                signal_type = "C: Long"
            elif last_row['short_cond_wt']:
                signal_type = "C: Short"
            elif last_row['long_cond_sqz']:
                signal_type = "D: Long"
            elif last_row['short_cond_sqz']:
                signal_type = "D: Short"
            elif last_row['Condition'] != 'Neutral':
                signal_type = last_row['Condition']

            # Clean display names
            simple_name = ticker.replace(".NS", "").lstrip("^")
            friendly_name = ASSET_NAME_LOOKUP.get(ticker)
            if friendly_name:
                display_name = f"{ticker} ({friendly_name})"
            else:
                display_name = simple_name

            # Calculate % change from previous close (day-over-day)
            prev_close = df.iloc[idx_pos - 1]['Close'] if idx_pos > 0 else last_row['Close']
            pct_change = ((last_row['Close'] - prev_close) / prev_close * 100) if prev_close > 0 else 0.0

            results.append({
                "Symbol": ticker,
                "DisplayName": display_name,
                "SimpleName": simple_name,
                "Signal": round(last_row['Unified_Osc'], 2) if not pd.isna(last_row['Unified_Osc']) else 0.0,
                "Trend": round(last_row['Norm_Trend'], 2) if not pd.isna(last_row['Norm_Trend']) else 0.0,
                "Conviction": round(last_row['Conviction'], 2) if not pd.isna(last_row['Conviction']) else 0.0,
                "Conviction_Delta": round(last_row['Conviction_Delta'], 2) if not pd.isna(last_row['Conviction_Delta']) else 0.0,
                "Pulse": round(last_row['Pulse'], 2) if not pd.isna(last_row['Pulse']) else 0.0,
                "Pulse_Delta": round(last_row['Pulse_Delta'], 2) if not pd.isna(last_row['Pulse_Delta']) else 0.0,
                "Narrative": last_row['Narrative'],
                "Narrative_Color": last_row['Narrative_Color'],
                "Wave": round(last_row['WT1'], 2) if not pd.isna(last_row['WT1']) else 0.0,
                "Zone": last_row['Condition'],
                "SignalType": signal_type,
                "Price": round(last_row['Close'], 2),
                "PctChange": round(pct_change, 2),
                # v3 Metrics for Engine 2.0
                "WT1_5ago":      round(df.iloc[idx_pos-5]['WT1'], 2) if idx_pos >= 5 else 0.0,
                "VolTrend":      round(last_row.get('VolTrend', 0), 3),
                "HMM_Bull":      float(last_row.get('HMM_Bull', 0.33)),
                "HMM_Bear":      float(last_row.get('HMM_Bear', 0.33)),
                "Vol_Regime":    str(last_row.get('Vol_Regime', 'NORMAL')),
                "Change_Point":  bool(last_row.get('Change_Point', False)),
                "Confidence":    float(last_row.get('Confidence', 0.0)),
                "Bullish_Div":   bool(last_row.get('Bullish_Div', False)),
                "Bearish_Div":   bool(last_row.get('Bearish_Div', False)),
                # Set C: Momentum — Historical Long Signals (kept for Range Study compat)
                "L_Today": "●" if sample_range.iloc[-1]['long_cond'] else "—",
                "L_1d": "●" if sample_range.iloc[-2]['long_cond'] else "—",
                "L_2d": "●" if sample_range.iloc[-3]['long_cond'] else "—",
                "L_3d": "●" if sample_range.iloc[-4]['long_cond'] else "—",
                "L_5d": "●" if sample_range.tail(5)['long_cond'].any() else "—",
                # Set C: Momentum — Historical Short Signals
                "S_Today": "●" if sample_range.iloc[-1]['short_cond'] else "—",
                "S_1d": "●" if sample_range.iloc[-2]['short_cond'] else "—",
                "S_2d": "●" if sample_range.iloc[-3]['short_cond'] else "—",
                "S_3d": "●" if sample_range.iloc[-4]['short_cond'] else "—",
                "S_5d": "●" if sample_range.tail(5)['short_cond'].any() else "—",
                # Set A: Momentum — Historical Long Signals
                "LA_Today": "●" if sample_range.iloc[-1]['long_cond'] else "—",
                "LA_1d": "●" if sample_range.iloc[-2]['long_cond'] else "—",
                "LA_2d": "●" if sample_range.iloc[-3]['long_cond'] else "—",
                "LA_3d": "●" if sample_range.iloc[-4]['long_cond'] else "—",
                "LA_5d": "●" if sample_range.tail(5)['long_cond'].any() else "—",
                # Set A: Momentum — Historical Short Signals
                "SA_Today": "●" if sample_range.iloc[-1]['short_cond'] else "—",
                "SA_1d": "●" if sample_range.iloc[-2]['short_cond'] else "—",
                "SA_2d": "●" if sample_range.iloc[-3]['short_cond'] else "—",
                "SA_3d": "●" if sample_range.iloc[-4]['short_cond'] else "—",
                "SA_5d": "●" if sample_range.tail(5)['short_cond'].any() else "—",
                # Set B: Crossover — Historical Long Signals
                "LB_Today": "●" if sample_range.iloc[-1]['long_cond_comp'] else "—",
                "LB_1d": "●" if sample_range.iloc[-2]['long_cond_comp'] else "—",
                "LB_2d": "●" if sample_range.iloc[-3]['long_cond_comp'] else "—",
                "LB_3d": "●" if sample_range.iloc[-4]['long_cond_comp'] else "—",
                "LB_5d": "●" if sample_range.tail(5)['long_cond_comp'].any() else "—",
                # Set B: Crossover — Historical Short Signals
                "SB_Today": "●" if sample_range.iloc[-1]['short_cond_comp'] else "—",
                "SB_1d": "●" if sample_range.iloc[-2]['short_cond_comp'] else "—",
                "SB_2d": "●" if sample_range.iloc[-3]['short_cond_comp'] else "—",
                "SB_3d": "●" if sample_range.iloc[-4]['short_cond_comp'] else "—",
                "SB_5d": "●" if sample_range.tail(5)['short_cond_comp'].any() else "—",
                # Set C: Threshold — Historical Long Signals
                "LC_Today": "●" if sample_range.iloc[-1]['long_cond_wt'] else "—",
                "LC_1d": "●" if sample_range.iloc[-2]['long_cond_wt'] else "—",
                "LC_2d": "●" if sample_range.iloc[-3]['long_cond_wt'] else "—",
                "LC_3d": "●" if sample_range.iloc[-4]['long_cond_wt'] else "—",
                "LC_5d": "●" if sample_range.tail(5)['long_cond_wt'].any() else "—",
                # Set C: Threshold — Historical Short Signals
                "SC_Today": "●" if sample_range.iloc[-1]['short_cond_wt'] else "—",
                "SC_1d": "●" if sample_range.iloc[-2]['short_cond_wt'] else "—",
                "SC_2d": "●" if sample_range.iloc[-3]['short_cond_wt'] else "—",
                "SC_3d": "●" if sample_range.iloc[-4]['short_cond_wt'] else "—",
                "SC_5d": "●" if sample_range.tail(5)['short_cond_wt'].any() else "—",
                # Set D: Squeeze — Historical Long Signals
                "LD_Today": "●" if sample_range.iloc[-1]['long_cond_sqz'] else "—",
                "LD_1d": "●" if sample_range.iloc[-2]['long_cond_sqz'] else "—",
                "LD_2d": "●" if sample_range.iloc[-3]['long_cond_sqz'] else "—",
                "LD_3d": "●" if sample_range.iloc[-4]['long_cond_sqz'] else "—",
                "LD_5d": "●" if sample_range.tail(5)['long_cond_sqz'].any() else "—",
                # Set D: Squeeze — Historical Short Signals
                "SD_Today": "●" if sample_range.iloc[-1]['short_cond_sqz'] else "—",
                "SD_1d": "●" if sample_range.iloc[-2]['short_cond_sqz'] else "—",
                "SD_2d": "●" if sample_range.iloc[-3]['short_cond_sqz'] else "—",
                "SD_3d": "●" if sample_range.iloc[-4]['short_cond_sqz'] else "—",
                "SD_5d": "●" if sample_range.tail(5)['short_cond_sqz'].any() else "—",
                # Additional fields for detail cards
                "Osc_Value": round(last_row.get('Unified_Osc', 0), 2),
                "MA_Alignment": 5,  # Placeholder
                "ZScore_Value": 0,  # Placeholder
            })
            
            console.detail(f"[{i+1}/{len(data_dict)}] {ticker}: Signal={last_row['Unified_Osc']:+.2f}  Zone={last_row['Condition']}  Status={signal_type}")
            
        except Exception as e:
            console.failure(f"Analysis Failed: {ticker}", str(e))
            continue

    console.end_phase("WRCI MOMENTUM ANALYSIS")
    
    console.summary("RUN SUMMARY", {
        "Universe": universe,
        "Universe Index": selected_index,
        "Total Symbols": len(stock_list),
        "Data Success": len(data_dict),
        "Analyzed Stocks": len(results),
        "Analysis Date": analysis_date,
        "Status": "COMPLETE"
    })
    console.line('═', 70)
    
    if show_progress or external_progress_slot is not None:
        pct_val = progress_offset + (95 * progress_scale / 100) if external_progress_slot else 100
        progress_bar(progress_slot, pct_val, "Analysis Complete", f"{len(results)} stocks analyzed")
        if show_progress and external_progress_slot is None:
            progress_slot.empty()

    if not results:
        st.warning("No stocks met the analysis criteria.")
        # Return empty DataFrame with expected columns to prevent downstream KeyErrors
        expected_cols = [
            "Symbol", "DisplayName", "SimpleName", "Signal", "Trend", "Wave", "Zone", "SignalType", "Price", "PctChange",
            "L_Today", "L_1d", "L_2d", "L_3d", "L_5d", "S_Today", "S_1d", "S_2d", "S_3d", "S_5d",
            "LA_Today", "LA_1d", "LA_2d", "LA_3d", "LA_5d", "SA_Today", "SA_1d", "SA_2d", "SA_3d", "SA_5d",
            "LB_Today", "LB_1d", "LB_2d", "LB_3d", "LB_5d", "SB_Today", "SB_1d", "SB_2d", "SB_3d", "SB_5d",
            "LC_Today", "LC_1d", "LC_2d", "LC_3d", "LC_5d", "SC_Today", "SC_1d", "SC_2d", "SC_3d", "SC_5d",
            "LD_Today", "LD_1d", "LD_2d", "LD_3d", "LD_5d", "SD_Today", "SD_1d", "SD_2d", "SD_3d", "SD_5d",
            "Osc_Value", "MA_Alignment", "ZScore_Value",
        ]
        return pd.DataFrame(columns=expected_cols)

    results_df = pd.DataFrame(results)
    
    # Global ranking via Priority Engine v3
    if not results_df.empty:
        results_df = compute_priority_v3(results_df)
        # Default sort by Long Percentile for the global table
        results_df = results_df.sort_values('Priority_Long_v3', ascending=False)
        
    return results_df


def run_timeseries_analysis(universe, selected_index, start_date, end_date, reg_len, wt_n1, wt_n2, levels, timeframe):
    """Execute WRCI analysis across historical date range for signal evolution tracking.

    Differs from run_screener_analysis: processes 500+ days of history to detect
    signal emergence, persistence, and fade patterns over time for timeline visualization.

    Returns: Dict with per-date results for historical signal tracking.
    """
    progress_slot = st.empty()
    progress_bar(progress_slot, 5, "Fetching Historical Depth", f"Date range: {start_date} to {end_date}")

    console.start_phase("HISTORICAL ACQUISITION", 1, 2)
    console.section("Range Configuration")
    console.item("Universe", universe)
    console.item("Selected Index", selected_index)
    console.item("Start Date", start_date)
    console.item("End Date", end_date)
    console.item("Timeframe", timeframe)

    if universe == "India Indexes":
        stock_list, _ = get_index_stock_list(selected_index)
    elif universe == "Global Indexes":
        stock_list, _ = get_global_index_symbols()
    elif universe == "US Indexes":
        stock_list, _ = get_us_index_symbols(selected_index)
    elif universe == "Commodities":
        stock_list, _ = get_commodity_symbols(None)
    elif universe == "Currency":
        stock_list, _ = get_currency_symbols(None)
    elif universe == "Crypto":
        stock_list, _ = get_crypto_symbols(None)
    elif universe == "ETF Index":
        stock_list, _ = get_etf_symbols()
    elif universe == "Global Macro":
        stock_list, _ = get_global_macro_symbols()
    else:
        stock_list = None

    if not stock_list:
        console.error("Failed to retrieve stock list")
        st.error("Failed to retrieve stock list")
        return

    console.success(f"Fetched {len(stock_list)} symbols for {selected_index}")
    console.section("Mass Historical Download")
    data_dict, msg = fetch_batch_data(stock_list, end_date=end_date, days_back=500)

    if not data_dict:
        console.error("No historical data available")
        st.error("No historical data available for selected range.")
        return

    console.success(f"Downloaded depth for {len(data_dict)} entities")
    console.end_phase("HISTORICAL ACQUISITION")

    progress_bar(progress_slot, 15, "Processing WRCI + Regime Intelligence", f"{len(data_dict)} stocks")
    all_results = []

    for i, (ticker, df) in enumerate(data_dict.items()):
        try:
            pct = int(15 + (i + 1) / len(data_dict) * 70)
            progress_bar(progress_slot, pct, f"Analyzing Signals", f"{i + 1}/{len(data_dict)} stocks")
            if timeframe == "Weekly":
                df = resample_to_weekly(df)
            df = run_full_analysis(df, reg_len, wt_n1, wt_n2, *levels)
            df = run_regime_analysis(df)
            df = calculate_divergences(df)

            mask = (df.index.date >= start_date) & (df.index.date <= end_date)
            range_df = df.loc[mask]

            for date, row in range_df.iterrows():
                all_results.append({
                    'Date': date,
                    'Symbol': ticker,
                    'Signal': row['Unified_Osc'],
                    'Trend': row['Norm_Trend'],
                    'Conviction': row['Conviction'],
                    'Wave': row['WT1'],
                    'Zone': row['Condition'],
                    'LongSignal': row['long_cond'],
                    'ShortSignal': row['short_cond'],
                    # Regime Intelligence columns
                    'Regime': row.get('Regime', 'NEUTRAL'),
                    'HMM_Bull': row.get('HMM_Bull', 0),
                    'HMM_Bear': row.get('HMM_Bear', 0),
                    'Vol_Regime': row.get('Vol_Regime', 'NORMAL'),
                    'Change_Point': row.get('Change_Point', False),
                    'Confidence': row.get('Confidence', 0),
                    'Bullish_Div': row.get('Bullish_Div', False),
                    'Bearish_Div': row.get('Bearish_Div', False),
                })
            
            console.detail(f"[{i+1}/{len(data_dict)}] {ticker}: {len(range_df)} data points processed")
            
        except Exception as e:
            console.failure(f"Range Analysis Failed: {ticker}", str(e))
            continue

    console.end_phase("WRCI RANGE ANALYSIS")

    progress_slot.empty()
    if not all_results:
        st.error("No results generated for the selected timeframe.")
        return

    ts_df = pd.DataFrame(all_results)
    ts_df['Date'] = pd.to_datetime(ts_df['Date'])
    ts_df = ts_df.sort_values('Date')

    # Aggregate daily metrics - WRCI + Regime Intelligence
    daily_agg = ts_df.groupby('Date').agg({
        'Signal': 'mean',
        'Trend': 'mean',
        'Wave': 'mean',
        'LongSignal': 'sum',
        'ShortSignal': 'sum',
        'Zone': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Neutral',
        # Regime aggregations
        'Regime': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'NEUTRAL',
        'HMM_Bull': 'mean',
        'HMM_Bear': 'mean',
        'Vol_Regime': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'NORMAL',
        'Change_Point': 'sum',
        'Confidence': 'mean',
        'Bullish_Div': 'sum',
        'Bearish_Div': 'sum',
    })

    daily_agg['TotalSignals'] = daily_agg['LongSignal'] + daily_agg['ShortSignal']
    daily_agg['L_S_Ratio'] = daily_agg['LongSignal'] / (daily_agg['ShortSignal'] + 0.01)
    daily_agg['Conviction'] = daily_agg['Signal'].abs()

    # Compute zone percentages
    zone_counts = ts_df.groupby('Date')['Zone'].apply(lambda x: (x.isin(['OB Extreme', 'OB'])).sum())
    os_counts = ts_df.groupby('Date')['Zone'].apply(lambda x: (x.isin(['OS Extreme', 'OS'])).sum())
    total_per_day = ts_df.groupby('Date').size()
    daily_agg['Oversold_Pct'] = (zone_counts / total_per_day * 100).fillna(0)
    daily_agg['Overbought_Pct'] = (os_counts / total_per_day * 100).fillna(0)

    # Compute regime percentages
    regime_bull = ts_df.groupby('Date')['Regime'].apply(lambda x: x.str.contains('BULL', na=False).sum())
    regime_bear = ts_df.groupby('Date')['Regime'].apply(lambda x: x.str.contains('BEAR', na=False).sum())
    regime_trans = ts_df.groupby('Date')['Regime'].apply(lambda x: (x == 'TRANSITION').sum())
    daily_agg['Regime_Bull_Pct'] = (regime_bull / total_per_day * 100).fillna(0)
    daily_agg['Regime_Bear_Pct'] = (regime_bear / total_per_day * 100).fillna(0)
    daily_agg['Regime_Transition_Pct'] = (regime_trans / total_per_day * 100).fillna(0)

    # Summary metrics
    total_signals = daily_agg['TotalSignals'].sum()
    avg_signal = daily_agg['Signal'].mean()
    overall_ratio = daily_agg['LongSignal'].sum() / max(daily_agg['ShortSignal'].sum(), 1)
    most_common_zone = ts_df['Zone'].mode()[0] if len(ts_df['Zone'].mode()) > 0 else 'Neutral'
    dominant_regime = ts_df['Regime'].mode()[0] if len(ts_df['Regime'].mode()) > 0 else 'NEUTRAL'
    
    avg_oversold = daily_agg['Oversold_Pct'].mean()
    avg_overbought = daily_agg['Overbought_Pct'].mean()
    total_buys = int(daily_agg['LongSignal'].sum())
    total_sells = int(daily_agg['ShortSignal'].sum())
    avg_bull_regime = daily_agg['Regime_Bull_Pct'].mean()
    avg_bear_regime = daily_agg['Regime_Bear_Pct'].mean()
    total_change_points = int(daily_agg['Change_Point'].sum())

    console.summary("HISTORICAL RANGE SUMMARY", {
        "Universe": universe,
        "Universe Index": selected_index,
        "Historical Range": f"{start_date} to {end_date}",
        "Total Signals Generated": int(total_signals),
        "Avg Signal Strength": round(avg_signal, 2),
        "Bias Ratio (L/S)": round(overall_ratio, 2),
        "Dominant Zone": most_common_zone,
        "HMM Regime": dominant_regime,
        "Status": "COMPLETE"
    })
    console.line('═', 70)

    progress_bar(progress_slot, 100, "Historical Range Complete", f"{int(total_signals)} signals analyzed")
    progress_slot.empty()
    st.session_state["timeseries_done"] = True

    ui.render_section_header(f"Historical Range ({start_date} to {end_date})", icon="history", accent="violet")

    # Summary metric cards
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    timeframe_label = "Weekly Average" if timeframe == 'Weekly' else "Daily Average"
    with c1:
        ui.render_metric_card("Total Signals", str(int(total_signals)), f"{total_buys} long · {total_sells} short", "info")
    with c2:
        ui.render_metric_card("Avg Oversold", f"{avg_oversold:.1f}%", timeframe_label, "success")
    with c3:
        ui.render_metric_card("Avg Overbought", f"{avg_overbought:.1f}%", timeframe_label, "danger")
    with c4:
        ui.render_metric_card("Period Regime", dominant_regime, f"Bull: {avg_bull_regime:.0f}% | Bear: {avg_bear_regime:.0f}%", "warning")
    with c5:
        ui.render_metric_card("L/S Ratio", f"{overall_ratio:.2f}", f"{'Bullish' if overall_ratio > 1 else 'Bearish'} bias", "info")
    with c6:
        ui.render_metric_card("Trading Days", str(len(daily_agg)), "Analyzed", "neutral")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Create 4 tabs like NIRNAY
    tab1, tab2, tab3, tab4 = st.tabs([
        "Signal Dashboard",
        "Transaction Dynamics", 
        "Regime Analysis",
        "Data Terminal"
    ])

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 1: SIGNAL DASHBOARD
    # ═══════════════════════════════════════════════════════════════════
    with tab1:
        ui.render_section_header("Extreme Signal Trends", "Overbought / Oversold Distribution Over Time", icon="activity", accent="cyan")
        
        fig_zones = go.Figure()
        fig_zones.add_trace(go.Scatter(
            x=daily_agg.index, y=daily_agg['Oversold_Pct'],
            mode='lines', name='Oversold %',
            fill='tozeroy', fillcolor='rgba(52,211,153,0.12)',
            line=dict(color='#2DD4A8', width=2)
        ))
        fig_zones.add_trace(go.Scatter(
            x=daily_agg.index, y=daily_agg['Overbought_Pct'],
            mode='lines', name='Overbought %',
            fill='tozeroy', fillcolor='rgba(251,113,133,0.12)',
            line=dict(color='#E8555A', width=2)
        ))
        ymax = max(daily_agg['Oversold_Pct'].max(), daily_agg['Overbought_Pct'].max()) * 1.15
        fig_zones.update_layout(title='', height=350, hovermode='x unified', yaxis=dict(range=[0, ymax]))
        apply_chart_theme(fig_zones)
        st.plotly_chart(fig_zones, width='stretch')

        st.markdown("<br>", unsafe_allow_html=True)
        ui.render_section_header("Signal Volume Trends", "Raw Counts Over Time", icon="bar-chart", accent="info")
        
        fig_counts = go.Figure()
        fig_counts.add_trace(go.Bar(
            x=daily_agg.index, y=daily_agg['LongSignal'],
            name='Oversold', 
            marker=dict(color='#2DD4A8', line=dict(color='#2DD4A8', width=1))
        ))
        fig_counts.add_trace(go.Bar(
            x=daily_agg.index, y=daily_agg['ShortSignal'],
            name='Overbought', 
            marker=dict(color='#E8555A', line=dict(color='#E8555A', width=1))
        ))
        fig_counts.update_layout(title='', height=300, hovermode='x unified', barmode='group')
        apply_chart_theme(fig_counts)
        st.plotly_chart(fig_counts, width='stretch')

    # ════════════════════════════════════���═���════════════════════════════════════
    # TAB 2: TRANSACTION DYNAMICS
    # ═══════════════════════════════════════════════════════════════════════════
    with tab2:
        ui.render_section_header("Transaction Signal Trends", "Buy / Sell Signal Counts Over Time", icon="zap", accent="emerald")
        
        fig_signals = go.Figure()
        fig_signals.add_trace(go.Scatter(
            x=daily_agg.index, y=daily_agg['LongSignal'],
            mode='lines+markers', name='Long Signals',
            line=dict(color='#2DD4A8', width=2),
            marker=dict(size=6, color='#2DD4A8')
        ))
        fig_signals.add_trace(go.Scatter(
            x=daily_agg.index, y=daily_agg['ShortSignal'],
            mode='lines+markers', name='Short Signals',
            line=dict(color='#E8555A', width=2),
            marker=dict(size=6, color='#E8555A')
        ))
        fig_signals.update_layout(title='', height=300, hovermode='x unified')
        apply_chart_theme(fig_signals)
        st.plotly_chart(fig_signals, width='stretch')

        st.markdown("<br>", unsafe_allow_html=True)
        ui.render_section_header("Divergence Persistence", "Divergence Signals Over Time", icon="trending-up", accent="amber")
        
        fig_div = go.Figure()
        fig_div.add_trace(go.Bar(
            x=daily_agg.index, y=daily_agg['Bullish_Div'],
            name='Bullish Divergence', 
            marker=dict(color='#D4A853', line=dict(color='#D4A853', width=1))
        ))
        fig_div.add_trace(go.Bar(
            x=daily_agg.index, y=-daily_agg['Bearish_Div'],
            name='Bearish Divergence', 
            marker=dict(color='#06B6D4', line=dict(color='#06B6D4', width=1))
        ))
        fig_div.update_layout(title='', height=300, hovermode='x unified', barmode='relative')
        apply_chart_theme(fig_div)
        st.plotly_chart(fig_div, width='stretch')

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 3: REGIME ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    with tab3:
        ui.render_section_header("Aggregate Signal Momentum", "Average Signal Value Over Time", icon="activity", accent="rose")
        
        colors = ['#2DD4A8' if v < -20 else '#E8555A' if v > 20 else '#64748B' for v in daily_agg['Signal']]
        
        fig_avg = go.Figure()
        fig_avg.add_trace(go.Scatter(
            x=daily_agg.index, y=daily_agg['Signal'].clip(lower=0),
            fill='tozeroy', fillcolor='rgba(232,85,90,0.05)',
            line=dict(width=0), showlegend=False, hoverinfo='skip'
        ))
        fig_avg.add_trace(go.Scatter(
            x=daily_agg.index, y=daily_agg['Signal'].clip(upper=0),
            fill='tozeroy', fillcolor='rgba(45,212,168,0.05)',
            line=dict(width=0), showlegend=False, hoverinfo='skip'
        ))
        fig_avg.add_trace(go.Scatter(
            x=daily_agg.index, y=daily_agg['Signal'],
            mode='lines+markers', name='Avg Signal',
            line=dict(color='#D4A853', width=2),
            marker=dict(size=6, color=colors)
        ))
        fig_avg.add_hline(y=20, line=dict(color='rgba(239,68,68,0.5)', width=1, dash='dash'))
        fig_avg.add_hline(y=-20, line=dict(color='rgba(16,185,129,0.5)', width=1, dash='dash'))
        fig_avg.add_hline(y=0, line=dict(color='rgba(255,255,255,0.3)', width=1))
        fig_avg.update_layout(title='', height=300, hovermode='x unified', yaxis=dict(range=[-80, 80]))
        apply_chart_theme(fig_avg)
        st.plotly_chart(fig_avg, width='stretch')

        st.markdown("<br>", unsafe_allow_html=True)
        ui.render_section_header("HMM Regime Distribution Over Time", "Percentage of symbols in each HMM regime daily", icon="activity", accent="cyan")
        
        fig_regime = go.Figure()
        fig_regime.add_trace(go.Scatter(
            x=daily_agg.index, y=daily_agg['Regime_Bull_Pct'],
            mode='lines', name='Bull Regime %',
            fill='tozeroy', fillcolor='rgba(52,211,153,0.12)',
            line=dict(color='#2DD4A8', width=2)
        ))
        fig_regime.add_trace(go.Scatter(
            x=daily_agg.index, y=daily_agg['Regime_Bear_Pct'],
            mode='lines', name='Bear Regime %',
            fill='tozeroy', fillcolor='rgba(232,85,90,0.12)',
            line=dict(color='#E8555A', width=2)
        ))
        fig_regime.update_layout(title='', height=300, hovermode='x unified', yaxis=dict(range=[0, 100]))
        apply_chart_theme(fig_regime)
        st.plotly_chart(fig_regime, width='stretch')

        st.markdown("<br>", unsafe_allow_html=True)
        ui.render_section_header("Volatility Dynamics", "Volatility Regime & Change Points Over Time", icon="shield", accent="amber")
        
        # Compute high vol percentage
        vol_high = ts_df.groupby('Date')['Vol_Regime'].apply(lambda x: (x.isin(['HIGH', 'EXTREME'])).sum() / len(x) * 100)
        
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=daily_agg.index, y=vol_high.fillna(0),
            mode='lines+markers', name='High Vol %',
            line=dict(color='#D4A853', width=2),
            marker=dict(size=5)
        ))
        fig_vol.add_trace(go.Bar(
            x=daily_agg.index, y=daily_agg['Change_Point'],
            name='Change Points',
            marker=dict(color='#A855F7', opacity=0.7)
        ))
        fig_vol.update_layout(title='', height=250, hovermode='x unified')
        apply_chart_theme(fig_vol)
        st.plotly_chart(fig_vol, width='stretch')

        st.markdown("<br>", unsafe_allow_html=True)
        col_r1, col_r2 = st.columns(2)
        
        with col_r1:
            ui.render_section_header("State Transition Metrics", "HMM Regime Statistics", icon="bar-chart", accent="emerald")
            regime_stats = {
                "Metric": ["Avg Bull Regime %", "Avg Bear Regime %", "Total Change Points", "Avg High Vol %"],
                "Value": [f"{avg_bull_regime:.1f}%", f"{avg_bear_regime:.1f}%", f"{total_change_points}", f"{vol_high.mean():.1f}%"]
            }
            st.dataframe(pd.DataFrame(regime_stats), width='stretch', hide_index=True)
        
        with col_r2:
            ui.render_section_header("Distribution Metrics", "Signal Statistics", icon="database", accent="rose")
            signal_stats = {
                "Metric": ["Mean Signal", "Median Signal", "Min Signal", "Max Signal", "Std Dev"],
                "Value": [
                    f"{daily_agg['Signal'].mean():.2f}",
                    f"{daily_agg['Signal'].median():.2f}",
                    f"{daily_agg['Signal'].min():.2f}",
                    f"{daily_agg['Signal'].max():.2f}",
                    f"{daily_agg['Signal'].std():.2f}"
                ]
            }
            st.dataframe(pd.DataFrame(signal_stats), width='stretch', hide_index=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 4: DATA TERMINAL
    # ═══════════════════════════════════════════════════════════════════════════
    with tab4:
        timeframe_label = "Weekly Time Series" if timeframe == 'Weekly' else "Daily Time Series"
        ui.render_section_header("Analytical Data", f"{timeframe_label} ({len(daily_agg)} periods)", icon="list", accent="cyan")
        
        display_ts = daily_agg.copy()
        display_ts.index = display_ts.index.strftime('%Y-%m-%d')
        display_ts = display_ts.reset_index().rename(columns={'Date': 'Date'})
        
        # Select columns to display
        display_cols = ['Date', 'LongSignal', 'ShortSignal', 'Signal', 'Oversold_Pct', 'Overbought_Pct', 
                      'Regime_Bull_Pct', 'Regime_Bear_Pct', 'Change_Point']
        display_ts = display_ts[display_cols]
        display_ts.columns = ['Date', 'Long Sig', 'Short Sig', 'Avg Signal', 'Oversold %', 'Overbought %',
                           'Bull Regime %', 'Bear Regime %', 'Change Pts']
        
        st.dataframe(display_ts, width='stretch', hide_index=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        csv_data = ts_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Report (CSV)",
            data=csv_data,
            file_name=f"sanket_range_study_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    render_footer()


# ══════════════════════════════════════════════════════════════════════════════
# CORRELATION MODE ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def run_correlation_analysis(universe, selected_index, target_ticker, lookback, method, timeframe, analysis_date=None):
    """Execute correlation analysis between universe constituents and a target asset.

    Returns a dict with correlation data, rolling correlations, prices, and returns,
    plus WRCI confluence scoring for trade intelligence.
    """
    if analysis_date is None:
        analysis_date = datetime.date.today()
    progress_slot = st.empty()
    progress_bar(progress_slot, 5, "Initializing Correlation Engine", "Fetching Market Data")

    try:
        # Fetch universe symbols
        if universe == "India Indexes":
            stock_list, msg = get_index_stock_list(selected_index)
        elif universe == "Global Indexes":
            stock_list, msg = get_global_index_symbols()
        elif universe == "US Indexes":
            stock_list, msg = get_us_index_symbols(selected_index)
        elif universe == "Commodities":
            stock_list, msg = get_commodity_symbols(None)
        elif universe == "Currency":
            stock_list, msg = get_currency_symbols(None)
        elif universe == "Crypto":
            stock_list, msg = get_crypto_symbols(None)
        elif universe == "ETF Index":
            stock_list, msg = get_etf_symbols()
        else:
            st.error(f"Universe '{universe}' not supported")
            return None

        if not stock_list:
            st.error(f"Failed to fetch universe symbols: {msg}")
            return None

        console.item("Symbols fetched", len(stock_list))

        # Combine with target asset
        combined_list = list(set(stock_list + [target_ticker]))
        console.item("Combined symbols", len(combined_list))

        progress_bar(progress_slot, 15, "Fetching OHLCV Data", f"Symbols: {len(combined_list)}")

        # Fetch data
        data_dict, fetch_msg = fetch_batch_data(combined_list, days_back=lookback + 60)
        if data_dict is None:
            st.error(f"Data fetch failed: {fetch_msg}")
            console.item("Data fetch error", fetch_msg)
            return None

        console.item("Data fetched for symbols", len(data_dict))

        progress_bar(progress_slot, 25, "Building Price Matrix", "Pivoting Close Prices")

        # Build Close price matrix — handle MultiIndex columns from yfinance
        close_dict = {}
        for ticker, data in data_dict.items():
            if len(data) > 0:
                if 'Close' in data.columns:
                    close_dict[ticker] = data['Close']
                else:
                    # Handle MultiIndex case
                    try:
                        close_dict[ticker] = data[data.columns[data.columns.get_level_values(-1) == 'Close'][0]]
                    except (IndexError, KeyError):
                        console.item(f"Skipping {ticker}", "No Close column found")

        if not close_dict:
            st.error("No valid price data found for universe")
            console.item("Error", "No Close prices extracted")
            return None

        console.item("Close prices extracted for", len(close_dict))

        close_df = pd.DataFrame(close_dict)
        close_df = close_df.dropna(axis=1, how='all')

        console.item("Close DataFrame shape", f"{close_df.shape}")

        if len(close_df) < lookback + 10:
            st.error(f"Insufficient historical data for correlation analysis (only {len(close_df)} rows, need {lookback + 10})")
            console.item("Error", f"Only {len(close_df)} rows, need {lookback + 10}")
            return None

        # Resample to weekly if needed
        if timeframe == "Weekly":
            close_df = resample_to_weekly(close_df)

        progress_bar(progress_slot, 40, "Computing Returns", f"Method: {method}")

        # Compute log returns — drop rows only where all values are NaN
        returns_df = np.log(close_df / close_df.shift(1)).dropna(how='all')

        if target_ticker not in returns_df.columns:
            st.error(f"Target asset '{target_ticker}' not in data")
            console.item("Error", f"Target {target_ticker} not in returns columns")
            return None

        target_returns = returns_df[target_ticker].dropna()
        console.item("Target returns available", len(target_returns))

        # Filter to common dates with target
        common_idx = returns_df.index.intersection(target_returns.index)
        if len(common_idx) < lookback + 10:
            st.error(f"Insufficient overlapping data (only {len(common_idx)} days). Try a shorter lookback period.")
            console.item("Error", f"Only {len(common_idx)} common dates, need {lookback + 10}")
            return None

        returns_df = returns_df.loc[common_idx]
        target_returns = target_returns.loc[common_idx]
        universe_returns = returns_df.drop(columns=[target_ticker])

        console.item("Universe returns shape", f"{universe_returns.shape}")
        console.item("Target returns shape", target_returns.shape)

        progress_bar(progress_slot, 60, "Computing Rolling Correlation", f"Lookback: {lookback} bars")

        # Compute rolling correlation — use vectorized rolling correlation
        rolling_corr_dict = {}
        console.item("Computing rolling correlations", f"method={method}, lookback={lookback}, cols={len(universe_returns.columns)}")

        try:
            # Convert to numpy for faster computation
            target_vals = target_returns.values

            for col in universe_returns.columns:
                try:
                    col_vals = universe_returns[col].ffill().bfill().values

                    # Compute rolling correlation using DataFrame.rolling.corr()
                    temp_df = pd.DataFrame({
                        'col': col_vals,
                        'target': target_vals
                    })
                    rolling_corr = temp_df['col'].rolling(window=lookback).corr(temp_df['target'])
                    rolling_corr_dict[col] = rolling_corr
                except Exception as col_err:
                    console.item(f"Skipping column {col}", str(col_err)[:50])
                    continue

            console.item("Rolling corr dict entries", len(rolling_corr_dict))

            if len(rolling_corr_dict) == 0:
                st.error("Could not compute rolling correlations for any column")
                return None

            rolling_corr_df = pd.DataFrame(rolling_corr_dict)
            console.item("Rolling corr DataFrame shape", rolling_corr_df.shape)
        except Exception as e:
            st.error(f"Error in rolling correlation: {str(e)}")
            console.item("Rolling corr computation error", str(e)[:100])
            return None

        if rolling_corr_df.empty or len(rolling_corr_df) == 0:
            st.error("Could not compute rolling correlations. Check data availability.")
            console.item("Error", "Rolling correlation DataFrame is empty")
            return None

        # Get current and average correlations
        current_corr = rolling_corr_df.iloc[-1]
        avg_corr = rolling_corr_df.mean()
        corr_trend = current_corr - avg_corr

        # Compute tiers
        def get_corr_tier(corr):
            if pd.isna(corr):
                return "Neutral"
            abs_corr = abs(corr)
            if corr > 0:
                if abs_corr >= 0.6: return "Strong+"
                elif abs_corr >= 0.4: return "Moderate+"
                elif abs_corr >= 0.2: return "Weak+"
                else: return "Neutral"
            else:
                if abs_corr >= 0.6: return "Strong-"
                elif abs_corr >= 0.4: return "Moderate-"
                elif abs_corr >= 0.2: return "Weak-"
                else: return "Neutral"

        # Run WRCI analysis on the universe for confluence
        # Pass progress tracking to show detailed per-symbol analysis (75-90% of correlation progress)
        wrci_results = run_screener_analysis(
            universe, selected_index, analysis_date, 20, 10, 21, (80, 40, -80, -40), timeframe,
            show_progress=False, external_progress_slot=progress_slot, progress_offset=75, progress_scale=15
        )

        progress_bar(progress_slot, 90, "Building Results DataFrame", "Computing Divergence Metrics")

        # Build correlation results dataframe
        corr_data_list = []
        for symbol in universe_returns.columns:
            if symbol not in close_df.columns or symbol not in current_corr.index:
                continue

            # Get current data
            current_price = close_df[symbol].iloc[-1] if symbol in close_df.columns else np.nan
            price_change = (close_df[symbol].pct_change().iloc[-1] * 100) if symbol in close_df.columns else np.nan
            target_price = close_df[target_ticker].iloc[-1]
            target_change = (close_df[target_ticker].pct_change().iloc[-1] * 100)

            # Get WRCI data if available
            wrci_signal = np.nan
            wrci_zone = "—"
            wrci_signal_type = "Neutral"
            if wrci_results is not None and len(wrci_results) > 0:
                wrci_row = wrci_results[wrci_results['SimpleName'] == symbol.replace('.NS', '').replace('^', '')]
                if len(wrci_row) > 0:
                    wrci_signal = wrci_row['Signal'].values[0]
                    wrci_zone = wrci_row['Zone'].values[0]
                    wrci_signal_type = wrci_row['SignalType'].values[0]

            # Compute divergence
            expected_change = current_corr[symbol] * target_change
            divergence = price_change - expected_change

            corr_data_list.append({
                'Symbol': symbol,
                'DisplayName': symbol,
                'SimpleName': symbol.replace('.NS', '').replace('^', ''),
                'Corr_Current': current_corr[symbol],
                'Corr_Avg': avg_corr[symbol],
                'Corr_Trend': corr_trend[symbol],
                'Corr_Tier': get_corr_tier(current_corr[symbol]),
                'Price': current_price,
                'PctChange': price_change,
                'Target_Pct': target_change,
                'Expected_Change': expected_change,
                'Divergence': divergence,
                'WRCI_Signal': wrci_signal,
                'WRCI_Zone': wrci_zone,
                'WRCI_Signal_Type': wrci_signal_type,
            })

        corr_df = pd.DataFrame(corr_data_list)
        if len(corr_df) == 0:
            st.error("No correlation data could be computed")
            console.item("Error", "Empty correlation DataFrame")
            return None

        corr_df = corr_df.sort_values('Corr_Current', key=abs, ascending=False)

        # Compute confluence score
        corr_df['Confluence_Score'] = abs(corr_df['Corr_Current']) * (abs(corr_df['WRCI_Signal'].fillna(0)) / 80.0)

        # Get target name from maps (maps are display_name -> ticker, so reverse lookup)
        target_name = target_ticker
        for map_dict in [COMMODITY_MAP, CURRENCY_MAP, CRYPTO_MAP, GLOBAL_INDEXES_MAP]:
            if target_ticker in map_dict.values():
                target_name = [k for k, v in map_dict.items() if v == target_ticker][0]
                break
            elif target_ticker in map_dict.keys():
                target_name = target_ticker
                break

        progress_bar(progress_slot, 100, "Analysis Complete", "Ready to display")
        time.sleep(0.3)
        progress_slot.empty()

        return {
            "corr_df": corr_df,
            "rolling_corr": rolling_corr_df,
            "target_ticker": target_ticker,
            "target_name": target_name,
            "prices": close_df,
            "returns": returns_df,
            "lookback": lookback,
            "method": method,
            "timeframe": timeframe,
        }

    except Exception as e:
        st.error(f"Correlation analysis error: {str(e)}")
        console.item("Exception", str(e))
        import traceback
        console.item("Traceback", traceback.format_exc()[-500:])
        return None


# ══════════════════════════════════════════════════════════════════════════════
# CORRELATION MODE — HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _build_confluence_table_html(df: pd.DataFrame) -> str:
    """Build ranked HTML table for confluence setups.

    Displays symbol, correlation, zone, signal, actual/expected/divergence, and confluence score.

    Returns: Complete HTML document string ready for st.components.v1.html().
    """
    import html as html_module

    table_rows = []
    if df.empty:
        table_rows.append(f"""
        <tr>
            <td colspan="8" style="
                text-align: center;
                color: #374151;
                font-family: 'IBM Plex Mono', monospace;
                font-size: 0.72rem;
                letter-spacing: 0.06em;
                padding: 2.25rem 1rem;
            ">— no setups —</td>
        </tr>
        """)
    else:
        for idx, (_, row) in enumerate(df.iterrows(), 1):
            symbol = html_module.escape(str(row.get('SimpleName', '')))
            corr = float(row.get('Corr_Current', 0))
            zone = html_module.escape(str(row.get('WRCI_Zone', 'Neutral')))
            signal_type = html_module.escape(str(row.get('WRCI_Signal_Type', '—')))
            actual = float(row.get('PctChange', 0))
            expected = float(row.get('Expected_Change', 0))
            divergence = float(row.get('Divergence', 0))
            confluence = float(row.get('Confluence_Score', 0))

            corr_color = "#34D399" if corr > 0 else "#FB7185"
            div_color = "#34D399" if divergence > 0 else "#FB7185"
            conf_color = "#A78BFA"

            rank_str = f"{idx:02d}"

            table_rows.append(f"""
            <tr>
                <td class="numeric" style="color: #D4A853; font-weight: 700;">{rank_str}</td>
                <td class="symbol">{symbol}</td>
                <td class="numeric" style="color: {corr_color}; font-weight: 600;">{corr:+.3f}</td>
                <td class="numeric">{zone}</td>
                <td class="numeric">{signal_type}</td>
                <td class="numeric" style="color: #94A3B8;">{actual:+.2f}%</td>
                <td class="numeric" style="color: #94A3B8;">{expected:+.2f}%</td>
                <td class="numeric" style="color: {div_color}; font-weight: 600;">{divergence:+.2f}%</td>
                <td class="numeric" style="color:{conf_color}; font-weight:600;">{confluence:.2f}</td>
            </tr>
            """)

    # Build full HTML
    table_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'IBM Plex Mono', monospace;
            background: transparent;
            color: #F1F5F9;
            padding: 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        thead th {{
            background: transparent;
            color: #4B5563;
            font-size: 0.62rem !important;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            padding: 0.5rem 0.5rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            text-align: left;
        }}
        thead th.numeric {{ text-align: right; }}
        tbody tr {{
            border-bottom: 1px solid rgba(255, 255, 255, 0.03);
        }}
        tbody tr:hover {{ background: rgba(139, 92, 246, 0.05); }}
        tbody td {{
            padding: 0.5rem 0.5rem;
            color: #F1F5F9;
            font-size: 0.72rem !important;
        }}
        tbody td.symbol {{
            font-weight: 700;
            font-size: 0.75rem;
            letter-spacing: 0.02em;
        }}
        tbody td.numeric {{
            text-align: right;
            font-variant-numeric: tabular-nums;
        }}
    </style>
    </head>
    <body>
    <table>
        <thead>
            <tr>
                <th class="numeric">Rank</th>
                <th>Symbol</th>
                <th class="numeric">Corr</th>
                <th>Zone</th>
                <th>Type</th>
                <th class="numeric">Actual %</th>
                <th class="numeric">Expected %</th>
                <th class="numeric">Div %</th>
                <th class="numeric">Confluence</th>
            </tr>
        </thead>
        <tbody>
            {"".join(table_rows)}
        </tbody>
    </table>
    </body>
    </html>
    """
    return table_html


# ══════════════════════════════════════════════════════════════════════════════
# CORRELATION MODE — RESULTS RENDERER
# ══════════════════════════════════════════════════════════════════════════════

def render_correlation_results(corr_data: dict) -> None:
    """Render Correlation mode 4-tab results interface."""
    corr_df = corr_data["corr_df"]
    rolling_corr_df = corr_data["rolling_corr"]
    target_ticker = corr_data["target_ticker"]
    target_name = corr_data["target_name"]
    lookback = corr_data["lookback"]
    method = corr_data["method"]

    tab1, tab2, tab3 = st.tabs([
        "Correlation Dashboard",
        "Trade Intelligence",
        "Heatmap Matrix"
    ])

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 1: CORRELATION DASHBOARD
    # ═══════════════════════════════════════════════════════════════════════════
    with tab1:
        ui.render_section_header(
            "Correlation Dashboard",
            f"Target: {target_name} ({target_ticker}) | {lookback}D Rolling {method}",
            icon="crosshair",
            accent="violet"
        )

        # Summary metrics
        strong_corr_count = len(corr_df[corr_df['Corr_Current'] >= 0.6])
        strong_inv_count = len(corr_df[corr_df['Corr_Current'] <= -0.6])
        avg_abs_corr = abs(corr_df['Corr_Current']).mean()
        target_change = corr_df['Target_Pct'].iloc[0] if len(corr_df) > 0 else 0

        metrics = [
            {"label": "Target Performance", "value": f"{target_change:+.2f}%", "kind": "success" if target_change >= 0 else "danger"},
            {"label": "Highly Correlated", "value": str(strong_corr_count), "kind": "info"},
            {"label": "Highly Inverse", "value": str(strong_inv_count), "kind": "warning"},
            {"label": "Avg |Correlation|", "value": f"{avg_abs_corr:.2f}", "kind": "neutral"},
            {"label": "Correlation Signal", "value": "CONCENTRATED" if strong_corr_count > len(corr_df) * 0.3 else "DIVERSIFIED", "kind": "violet"},
        ]

        cols = st.columns(len(metrics))
        for i, m in enumerate(metrics):
            with cols[i]:
                ui.render_metric_card(m["label"], m["value"], color_class=m["kind"])

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

        # Ranked lists
        col_pos, col_neg = st.columns(2)

        with col_pos:
            ui.render_section_header("Top Positively Correlated", icon="trending", accent="emerald")
            pos_corr = corr_df[corr_df['Corr_Current'] > 0].head(7)
            for _, row in pos_corr.iterrows():
                trend_arrow = "↑" if row['Corr_Trend'] > 0.05 else "↓" if row['Corr_Trend'] < -0.05 else "→"
                corr_val = row['Corr_Current']
                tier_class = row['Corr_Tier'].lower().replace("+", "-pos").replace("-", "-neg")

                st.markdown(f"""
                <div class="corr-row">
                    <div>
                        <div class="name">{row['SimpleName']}</div>
                        <div class="sub">{row['PctChange']:+.2f}% | Expected: {row['Expected_Change']:+.2f}%</div>
                    </div>
                    <div style="display:flex; gap:8px; align-items:center;">
                        <span class="corr-tier {tier_class}">{corr_val:.3f}</span>
                        <div class="corr-bar-track">
                            <div class="corr-bar-center"></div>
                            <div class="corr-bar-fill pos" style="width:{abs(corr_val)*50}px;"></div>
                        </div>
                        <span style="font-size:0.75rem; color:var(--ink-secondary);">{trend_arrow}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with col_neg:
            ui.render_section_header("Top Inversely Correlated", icon="trending", accent="rose")
            neg_corr = corr_df[corr_df['Corr_Current'] < 0].head(7)
            for _, row in neg_corr.iterrows():
                trend_arrow = "↑" if row['Corr_Trend'] > 0.05 else "↓" if row['Corr_Trend'] < -0.05 else "→"
                corr_val = row['Corr_Current']
                tier_class = row['Corr_Tier'].lower().replace("+", "-pos").replace("-", "-neg")

                st.markdown(f"""
                <div class="corr-row">
                    <div>
                        <div class="name">{row['SimpleName']}</div>
                        <div class="sub">{row['PctChange']:+.2f}% | Expected: {row['Expected_Change']:+.2f}%</div>
                    </div>
                    <div style="display:flex; gap:8px; align-items:center;">
                        <span class="corr-tier {tier_class}">{corr_val:.3f}</span>
                        <div class="corr-bar-track">
                            <div class="corr-bar-center"></div>
                            <div class="corr-bar-fill neg" style="width:{abs(corr_val)*50}px;"></div>
                        </div>
                        <span style="font-size:0.75rem; color:var(--ink-secondary);">{trend_arrow}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 2: TRADE INTELLIGENCE
    # ═══════════════════════════════════════════════════════════════════════════
    with tab2:
        ui.render_section_header(
            "Trade Intelligence",
            "Confluence: Correlation × Momentum Signals",
            icon="zap",
            accent="cyan"
        )

        # How to read this tab - styled as interpretation card
        st.markdown("""
        <div style="background:rgba(56,189,248,0.08); border:1px solid rgba(56,189,248,0.2);
                    border-radius:8px; padding:1rem; margin:1.5rem 0; font-family:var(--data); font-size:0.75rem;">
            <div style="color:#38BDF8; font-weight:700; text-transform:uppercase; margin-bottom:0.75rem; letter-spacing:0.06em;">
                How to Read
            </div>
            <div style="color:#F1F5F9; line-height:1.6;">
                Each setup type is ranked by <span style="color:#38BDF8; font-weight:600;">Confluence Score</span> (0-1).
                Highest rank = strongest opportunity. Look for: <span style="font-weight:600;">(1) Score >0.7</span>,
                <span style="font-weight:600;">(2) |Div %| >3%</span>, <span style="font-weight:600;">(3) Confirmed Zone (OB/OS Extreme)</span>
            </div>
            <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:0.5rem; margin-top:0.75rem;">
                <div style="font-family:var(--data); font-size:0.65rem; color:var(--ink-secondary);">
                    <span style="color:#38BDF8; font-weight:600;">Corr</span> — Correlation strength
                </div>
                <div style="font-family:var(--data); font-size:0.65rem; color:var(--ink-secondary);">
                    <span style="color:#38BDF8; font-weight:600;">Zone</span> — Momentum extreme (OB/OS)
                </div>
                <div style="font-family:var(--data); font-size:0.65rem; color:var(--ink-secondary);">
                    <span style="color:#38BDF8; font-weight:600;">Div %</span> — Actual vs Expected
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Trade setup classification
        def classify_setup(row):
            corr = row['Corr_Current']
            div = row['Divergence']
            zone = row['WRCI_Zone']

            if corr > 0.4 and div > 2 and zone in ['OS', 'OS Extreme']:
                return "LAGGARD"
            elif corr > 0.4 and div < -2 and zone in ['OB', 'OB Extreme']:
                return "RUNAWAY"
            elif abs(corr) < 0.2:
                return "CONVERGING"
            elif corr < -0.4 and div < -2 and zone in ['OB', 'OB Extreme']:
                return "CONTRA"
            else:
                return "NEUTRAL"

        corr_df['Setup'] = corr_df.apply(classify_setup, axis=1)

        # Summary metrics
        laggard_count = len(corr_df[corr_df['Setup'] == 'LAGGARD'])
        runaway_count = len(corr_df[corr_df['Setup'] == 'RUNAWAY'])
        converging_count = len(corr_df[corr_df['Setup'] == 'CONVERGING'])
        contra_count = len(corr_df[corr_df['Setup'] == 'CONTRA'])
        avg_confluence = corr_df[corr_df['Setup'] != 'NEUTRAL']['Confluence_Score'].mean()

        metrics = [
            {"label": "Laggard Setups", "value": str(laggard_count), "kind": "success"},
            {"label": "Runaway Setups", "value": str(runaway_count), "kind": "danger"},
            {"label": "Converging", "value": str(converging_count), "kind": "warning"},
            {"label": "Contra Setups", "value": str(contra_count), "kind": "info"},
            {"label": "Avg Confluence", "value": f"{avg_confluence:.2f}", "kind": "neutral"},
        ]

        cols = st.columns(len(metrics))
        for i, m in enumerate(metrics):
            with cols[i]:
                ui.render_metric_card(m["label"], m["value"], color_class=m["kind"])

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Render each setup type as a section
        setup_configs = [
            {
                "name": "LAGGARD",
                "title": "Laggard Setups",
                "description": "High corr + oversold + underperforming — expect catch-up rally",
                "color": "#34D399",
                "bg_color": "rgba(45, 212, 168, 0.1)",
                "border_color": "rgba(45, 212, 168, 0.25)"
            },
            {
                "name": "RUNAWAY",
                "title": "Runaway Setups",
                "description": "High corr + overbought + overextended — expect pullback",
                "color": "#FB7185",
                "bg_color": "rgba(232, 85, 90, 0.1)",
                "border_color": "rgba(232, 85, 90, 0.25)"
            },
            {
                "name": "CONVERGING",
                "title": "Converging Setups",
                "description": "Low corr or normalizing — expect tightening after divergence",
                "color": "#D4A853",
                "bg_color": "rgba(212, 168, 83, 0.1)",
                "border_color": "rgba(212, 168, 83, 0.25)"
            },
            {
                "name": "CONTRA",
                "title": "Contra Setups",
                "description": "Strong negative corr + overbought — expect rally vs target decline",
                "color": "#A78BFA",
                "bg_color": "rgba(139, 92, 246, 0.1)",
                "border_color": "rgba(139, 92, 246, 0.25)"
            }
        ]

        # Setup interpretation guide
        setup_interpretation = {
            "LAGGARD": {
                "action": "BUY",
                "rationale": "Stock lagging expected move — expect catch-up rally to target's pace",
                "validate": "Check that Zone is OS/OS Extreme and Div % is positive & large (>3%)",
                "risk": "Correlation may break; stock continues lagging instead of catching up"
            },
            "RUNAWAY": {
                "action": "SHORT",
                "rationale": "Stock overextended vs expected move — expect pullback to fair value",
                "validate": "Check that Zone is OB/OB Extreme and Div % is negative & large (<-3%)",
                "risk": "Stock may continue running; wait for Zone to weaken before shorting"
            },
            "CONVERGING": {
                "action": "DE-RISK",
                "rationale": "Correlation collapsing — pair-trade falling apart, avoid new entries",
                "validate": "Corr close to 0 or unstable; watch for re-correlation before re-entering",
                "risk": "Old positions may unwind suddenly; previous divergence trades may fail"
            },
            "CONTRA": {
                "action": "LONG (vs target down)",
                "rationale": "Negative correlation + target down — expect rally when target recovers",
                "validate": "Check Corr is strongly negative (<-0.6) and Zone is OB/OB Extreme",
                "risk": "Negative correlations are unstable; requires conviction and risk management"
            }
        }

        for config in setup_configs:
            setup_data = corr_df[corr_df['Setup'] == config['name']].nlargest(10, 'Confluence_Score')

            if len(setup_data) > 0:
                st.markdown(f"""
                <div style="display:flex; align-items:baseline; gap:0.65rem; margin:1.75rem 0 0.9rem 0;
                             padding-bottom:0.6rem; border-bottom:1px solid {config['border_color']};">
                    <span style="font-family:var(--display); font-size:0.62rem; font-weight:700;
                                 letter-spacing:0.12em; text-transform:uppercase; color:{config['color']};
                                 padding:0.18rem 0.5rem; background:{config['bg_color']};
                                 border:1px solid {config['border_color']}; border-radius:4px;">
                        {config['name']}</span>
                    <span style="font-family:var(--display); font-size:1rem; font-weight:700;
                                 color:#F1F5F9; letter-spacing:0.04em;">{config['title']}</span>
                    <span style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem; color:#6B7280;">
                        {config['description']}</span>
                    <span style="margin-left:auto; font-family:'IBM Plex Mono',monospace; font-size:0.72rem;
                                 color:{config['color']};">→ {len(setup_data)}</span>
                </div>
                """, unsafe_allow_html=True)

                # Interpretation card
                interp = setup_interpretation[config['name']]
                st.markdown(f"""
                <div style="background:{config['bg_color']}; border:1px solid {config['border_color']};
                            border-radius:8px; padding:0.75rem 1rem; margin-bottom:1rem; font-family:var(--data); font-size:0.75rem;">
                    <div style="display:grid; grid-template-columns:auto 1fr; gap:0.5rem 1rem; color:#F1F5F9;">
                        <span style="color:{config['color']}; font-weight:700; text-transform:uppercase;">Action</span>
                        <span>{interp['action']}</span>
                        <span style="color:{config['color']}; font-weight:700; text-transform:uppercase;">Rationale</span>
                        <span>{interp['rationale']}</span>
                        <span style="color:{config['color']}; font-weight:700; text-transform:uppercase;">Validate</span>
                        <span>{interp['validate']}</span>
                        <span style="color:#FB7185; font-weight:700; text-transform:uppercase;">⚠ Risk</span>
                        <span style="color:#FB7185;">{interp['risk']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Display as two-column table
                col_left, col_right = st.columns(2)
                with col_left:
                    st.markdown(f"""<p style="font-family:'IBM Plex Mono',monospace; font-size:0.62rem; font-weight:600;
                                   text-transform:uppercase; letter-spacing:0.1em; color:{config['color']};
                                   margin:0 0 0.4rem 0; display:flex; align-items:center; gap:0.35rem;">
                        Top Confluence</p>""", unsafe_allow_html=True)
                    top_half = setup_data.head(5)
                    if len(top_half) > 0:
                        st.components.v1.html(_build_confluence_table_html(top_half), height=100 + len(top_half) * 48)
                with col_right:
                    st.markdown(f"""<p style="font-family:'IBM Plex Mono',monospace; font-size:0.62rem; font-weight:600;
                                   text-transform:uppercase; letter-spacing:0.1em; color:{config['color']};
                                   margin:0 0 0.4rem 0; display:flex; align-items:center; gap:0.35rem;">
                        Also Considered</p>""", unsafe_allow_html=True)
                    bottom_half = setup_data.iloc[5:10]
                    if len(bottom_half) > 0:
                        st.components.v1.html(_build_confluence_table_html(bottom_half), height=100 + len(bottom_half) * 48)
                    else:
                        st.info("No additional setups")

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 3: HEATMAP MATRIX
    # ═══════════════════════════════════════════════════════════════════════════
    with tab3:
        ui.render_section_header("Correlation Matrix", "Top constituents by |correlation|", icon="grid", accent="violet")

        # Build heatmap data using Symbol (original ticker) to match rolling_corr_df columns
        top_by_corr = corr_df.copy()
        top_by_corr['AbsCorr'] = abs(top_by_corr['Corr_Current'])
        top_rows = top_by_corr.nlargest(30, 'AbsCorr')
        top_symbols = top_rows['Symbol'].tolist()
        valid_symbols = [s for s in top_symbols if s in rolling_corr_df.columns]
        heatmap_data = rolling_corr_df[valid_symbols].iloc[-1:].T if valid_symbols else pd.DataFrame()

        if len(heatmap_data) > 0:
            # Filter to only the top symbols that exist in rolling_corr_df
            heatmap_rows = corr_df[corr_df['Symbol'].isin(valid_symbols)].copy()
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_rows['Corr_Current'].values.reshape(-1, 1),
                x=["Correlation"],
                y=heatmap_rows['SimpleName'].values,
                colorscale=[[0, "#E8555A"], [0.5, "#1a2133"], [1, "#2DD4A8"]],
                zmid=0,
                zmin=-1,
                zmax=1,
                text=heatmap_rows['Corr_Current'].values.reshape(-1, 1),
                texttemplate='%{text:.2f}',
                textfont={"size": 8, "color": "#94A3B8"},
                colorbar=dict(title="Corr", thickness=15, len=0.7)
            ))
            apply_chart_theme(fig)
            fig.update_layout(height=600, margin=dict(l=150, r=50, t=50, b=50))
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("No correlation data available for heatmap")


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS FOR TAB RENDERING
# ══════════════════════════════════════════════════════════════════════════════

def _bucket_signals_by_age(results_df: pd.DataFrame, side: str = 'long', condition_set: str = 'C', timeframe: str = 'Daily') -> dict:
    """Bucket signals by age (Today, 1d, 2d, 3d, 5d) with stats for timeline display.

    condition_set: 'A' = Momentum (LA_/SA_), 'B' = Crossover (LB_/SB_), 'C' = Threshold (LC_/SC_)
    timeframe: 'Daily' or 'Weekly' — determines age label names
    """
    if condition_set == 'A':
        prefix = 'LA' if side == 'long' else 'SA'
    elif condition_set == 'B':
        prefix = 'LB' if side == 'long' else 'SB'
    elif condition_set == 'C':
        prefix = 'LC' if side == 'long' else 'SC'
    elif condition_set == 'D':
        prefix = 'LD' if side == 'long' else 'SD'
    else:
        prefix = 'L' if side == 'long' else 'S'
    target_indicator = "●"

    if timeframe == 'Weekly':
        age_labels = ["This Week", "1 Week Ago", "2 Weeks Ago", "3 Weeks Ago", "Within 5 Weeks"]
    else:
        age_labels = ["Today", "1 Day Ago", "2 Days Ago", "3 Days Ago", "Within 5 Days"]

    buckets = {label: [] for label in age_labels}
    col_map = {
        age_labels[0]: f"{prefix}_Today",
        age_labels[1]: f"{prefix}_1d",
        age_labels[2]: f"{prefix}_2d",
        age_labels[3]: f"{prefix}_3d",
        age_labels[4]: f"{prefix}_5d"
    }
    seen = set()

    for age in buckets.keys():
        col = col_map[age]
        subset = results_df[(results_df[col] == target_indicator) & (~results_df['Symbol'].isin(seen))]
        for _, r in subset.iterrows():
            buckets[age].append(r)
            seen.add(r['Symbol'])

    # Compute stats for each bucket
    stats = {}
    for age, rows in buckets.items():
        if rows:
            signals = [r['Signal'] for r in rows]
            pct_changes = [r.get('PctChange', 0) for r in rows]
            convictions = [r.get('Conviction', 0) for r in rows]
            avg_signal = np.mean(signals)
            avg_pct_change = np.mean(pct_changes)
            avg_conviction = np.mean(convictions)
            count = len(rows)
            stats[age] = {
                'count': count,
                'avg_signal': avg_signal,
                'avg_pct_change': avg_pct_change,
                'avg_conviction': avg_conviction,
                'rows': rows
            }
        else:
            stats[age] = {'count': 0, 'avg_signal': 0, 'avg_pct_change': 0, 'rows': []}

    # Calculate trend: are signals strengthening (newer) or weakening (older)?
    newest_label = age_labels[0]  # "Today" or "This Week"
    older_labels = age_labels[1:]  # Rest of the labels

    newest_avg = stats[newest_label]['avg_signal'] if stats[newest_label]['count'] > 0 else 0
    older_avg = np.mean([stats[age]['avg_signal'] for age in older_labels if stats[age]['count'] > 0]) if any(stats[age]['count'] for age in older_labels) else 0

    if newest_avg > older_avg + 5:
        trend = f"{SVGS['UP'].replace('12','14').replace('12','14')} Strengthening"
        trend_color = "#2DD4A8"
    elif newest_avg < older_avg - 5:
        trend = f"{SVGS['DOWN'].replace('12','14').replace('12','14')} Weakening"
        trend_color = "#E8555A"
    else:
        trend = "— Stable"
        trend_color = "#D4A853"

    return buckets, stats, trend, trend_color


def _render_signal_legend(side: str = 'long', condition_set: str = 'A') -> None:
    """Render context-aware interpretation legend below a timing table.

    condition_set: 'A' = Momentum, 'B' = Crossover, 'C' = Threshold
    """
    if condition_set == 'A':
        if side == 'long':
            signal_desc  = "Positive WRCI value — the oscillator has crossed upward, indicating building bullish momentum. Higher magnitude = stronger push."
            trend_desc   = "Positive = uptrend confirming the signal. Negative = downtrend still in place despite the bullish cross."
            timing_desc  = "Older bullish signals are more reliable — the upside shift has had time to prove itself. Today&rsquo;s signal is fresh and may still be developing."
            together_good = "Signal &#x2B; | Trend &#x2B; = high conviction long — momentum and direction fully aligned."
            together_mixed = "Signal &#x2B; | Trend &#x2212; = bullish cross against a downtrend. Likely a counter-trend bounce — wait for Trend to turn positive before committing."
        else:
            signal_desc  = "Negative WRCI value — the oscillator has crossed downward, indicating building selling pressure. Higher magnitude (more negative) = stronger push."
            trend_desc   = "Negative = downtrend confirming the signal. Positive = uptrend still in place despite the bearish cross."
            timing_desc  = "Older bearish signals are more reliable — the downside shift has confirmed over time. Today&rsquo;s signal is fresh and may still be developing."
            together_good = "Signal &#x2212; | Trend &#x2212; = high conviction short — momentum and direction fully aligned."
            together_mixed = "Signal &#x2212; | Trend &#x2B; = bearish cross inside an uptrend. Possible exhaustion or pullback — not a clean short until the trend turns negative."
    elif condition_set == 'B':
        if side == 'long':
            signal_desc  = "Composite Line crosses below Signal Line while both are in oversold zone (&lt;&minus;40). Momentum exhaustion inside OS territory — a precise reversal-timing signal."
            trend_desc   = "Negative = downtrend confirming the OS crossover — highest conviction. Positive = crossover in an uptrend&rsquo;s dip — potential sharp recovery."
            timing_desc  = "Recent crossovers carry the most urgency — the cross just occurred inside an extreme zone. Older Crossover signals may have already played out."
            together_good = "Signal &lt;&minus;40 | Crossover &#x2713; | Trend &minus; = full alignment — momentum exhausted in oversold, downtrend may be reversing."
            together_mixed = "Signal &lt;&minus;40 | Crossover &#x2713; | Trend &#x2B; = OS crossover in bullish trend. Likely a dip-buy setup — strongest when trend is gently positive."
        else:
            signal_desc  = "Composite Line crosses above Signal Line while both are in overbought zone (&gt;&#x2B;40). Momentum exhaustion inside OB territory — a precise reversal-timing signal."
            trend_desc   = "Positive = uptrend confirming the OB crossover — highest conviction. Negative = crossover in a downtrend&rsquo;s rally — potential sharp reversal."
            timing_desc  = "Recent crossovers carry the most urgency — the cross just occurred inside an extreme zone. Older Crossover signals may have already resolved."
            together_good = "Signal &gt;&#x2B;40 | Crossover &#x2713; | Trend &#x2B; = full alignment — momentum exhausted in overbought, uptrend may be topping."
            together_mixed = "Signal &gt;&#x2B;40 | Crossover &#x2713; | Trend &minus; = OB crossover in bearish trend. Likely a dead-cat rally — short with tight stops."
    else:  # condition_set == 'C': Threshold
        if side == 'long':
            signal_desc  = "Composite Line has freshly dropped below &minus;40 from neutral, while Signal Line stays above &minus;40. Marks the first bar of oversold entry — a zone-breach signal."
            trend_desc   = "Negative = downtrend confirming oversold entry. Positive = pullback into OS within an uptrend — potential bounce setup."
            timing_desc  = "Fresher signals are more actionable — the line just entered the zone. Older signals may have already resolved or deepened."
            together_good = "Signal &lt;&minus;40 | Trend &minus; = confirmed oversold entry in a downtrend — strongest Threshold long context."
            together_mixed = "Signal &lt;&minus;40 | Trend &#x2B; = oversold entry in an uptrend. Likely a shallow pullback — watch for quick recovery rather than full reversal."
        else:
            signal_desc  = "Composite Line has freshly risen above &#x2B;40 from neutral, while Signal Line stays below &#x2B;40. Marks the first bar of overbought entry — a zone-breach signal."
            trend_desc   = "Positive = uptrend confirming overbought. Negative = relief rally into OB within a downtrend — potential fade setup."
            timing_desc  = "Fresher signals are more urgent — the line just entered the zone. Older overbought entries may have already faded or extended."
            together_good = "Signal &gt;&#x2B;40 | Trend &#x2B; = confirmed overbought in an uptrend — strongest Threshold short context."
            together_mixed = "Signal &gt;&#x2B;40 | Trend &minus; = overbought in a downtrend. Likely a counter-trend rally — fade quickly or wait for re-entry below &#x2B;40."

    # Removed "How to read this table" section from signal tabs to free up mobile space
    # This content will be moved to System Data tab for reference
    # signal_desc, trend_desc, timing_desc, together_good, together_mixed are still available
    # in the function scope for future use if needed
    pass


def _build_signal_table_html(stats: dict, side: str = 'long', timeframe: str = 'Daily') -> str:
    """Build organized HTML table for signals grouped by age with section headers."""
    import html as html_module

    accent_light = "#34D399" if side == 'long' else "#FB7185"
    border_color = "rgba(45, 212, 168, 0.3)" if side == 'long' else "rgba(232, 85, 90, 0.3)"
    header_bg = "rgba(45, 212, 168, 0.15)" if side == 'long' else "rgba(232, 85, 90, 0.15)"

    table_rows = []
    if timeframe == 'Weekly':
        age_order = ["This Week", "1 Week Ago", "2 Weeks Ago", "3 Weeks Ago", "Within 5 Weeks"]
    else:
        age_order = ["Today", "1 Day Ago", "2 Days Ago", "3 Days Ago", "Within 5 Days"]

    for age in age_order:
        if stats[age]['count'] == 0:
            continue

        # Section header for this age group
        avg_signal = stats[age]['avg_signal']
        avg_pct = stats[age].get('avg_pct_change', 0)
        avg_conv = stats[age].get('avg_conviction', 0)
        count = stats[age]['count']
        table_rows.append(f"""
        <tr style="background: {header_bg}; border-bottom: 2px solid {border_color};">
            <td colspan="9" style="padding: 0.75rem 1rem; font-family: 'IBM Plex Mono', monospace !important; font-size: 0.8rem !important; font-weight: 700; color: {accent_light}; text-transform: uppercase; letter-spacing: 0.05em;">
                {age} · {count} signal{'s' if count != 1 else ''} · Avg Conv: {avg_conv:+.1f} · Avg %: {avg_pct:+.1f}
            </td>
        </tr>
        """)

        # Data rows for this age group
        _zone_colors = {"OB Extreme": "#FB7185", "OB": "#FCA5A5",
                        "OS Extreme": "#34D399", "OS": "#86EFAC"}
        for row in stats[age]['rows']:
            symbol = html_module.escape(str(row.get('DisplayName', row.get('Symbol', ''))))
            price = float(row.get('Price', 0))
            pct_change = float(row.get('PctChange', 0))
            signal = float(row.get('Signal', 0))
            trend = float(row.get('Trend', 0))
            conviction = float(row.get('Conviction', 0))
            pulse = float(row.get('Pulse', 0))
            conv_delta = float(row.get('Conviction_Delta', 0))
            pulse_delta = float(row.get('Pulse_Delta', 0))
            narrative = str(row.get('Narrative', 'NEUTRAL'))
            narr_color = str(row.get('Narrative_Color', '#94a3b8'))
            signal_type = str(row.get('SignalType', '-'))

            # Color % change: green for positive, red for negative
            pct_color = "#34D399" if pct_change >= 0 else "#FB7185"
            
            # Delta formatting
            conv_delta_color = "#34D399" if conv_delta >= 0 else "#FB7185"
            pulse_delta_color = "#4a9eff" if pulse_delta >= 0 else "#D4A853"
            conv_delta_arrow = "↑" if conv_delta >= 0 else "↓"
            pulse_delta_arrow = "↑" if pulse_delta >= 0 else "↓"

            table_rows.append(f"""
            <tr>
                <td class="symbol">{symbol}</td>
                <td class="numeric currency">{price:,.2f}</td>
                <td class="numeric" style="color: {pct_color}; font-weight: 600;">{pct_change:+.2f}%</td>
                <td class="numeric" style="color: {accent_light}; font-weight: 600;">{signal:+.2f}</td>
                <td class="numeric" style="color: #D4A853; font-weight: 600;">{conviction:+.2f}</td>
                <td class="numeric" style="color: {conv_delta_color}; font-size: 0.65rem; font-weight: 600;">{conv_delta_arrow}{abs(conv_delta):.1f}</td>
                <td class="numeric" style="color: #4a9eff; font-weight: 600;">{pulse:+.2f}</td>
                <td class="numeric" style="color: {pulse_delta_color}; font-size: 0.65rem; font-weight: 600;">{pulse_delta_arrow}{abs(pulse_delta):.1f}</td>
                <td class="numeric" style="color: {narr_color}; font-weight: 700; letter-spacing: 0.05em; font-size: 0.68rem !important;">{narrative}</td>
            </tr>
            """)

    if not table_rows:
        table_rows.append(f"""
        <tr>
            <td colspan="9" style="text-align:center; color:#374151; font-family:'IBM Plex Mono',monospace;
                font-size:0.72rem; letter-spacing:0.06em; padding:2.25rem 1rem;">
                — no signals detected —
            </td>
        </tr>""")

    table_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        * {{
            -webkit-text-size-adjust: 100%;
            -moz-text-size-adjust: 100%;
            text-size-adjust: 100%;
        }}
        body {{
            font-family: 'IBM Plex Mono', monospace;
            background: transparent;
            color: #F1F5F9;
            padding: 0.5rem 0.5rem 1.5rem 0.5rem;
            font-size: 16px !important;
        }}
        @media (max-width: 768px) {{
            body {{
                font-size: 16px !important;
            }}
        }}
        .portfolio-table {{
            width: 100%;
            border-radius: 10px;
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            border: 1px solid rgba(255, 255, 255, 0.05);
            background: linear-gradient(145deg, rgba(17, 24, 39, 0.45) 0%, rgba(17, 24, 39, 0.4) 100%);
        }}
        .portfolio-table table {{
            width: 100%;
            min-width: 480px;
            border-collapse: collapse;
        }}
        .portfolio-table thead th {{
            background: linear-gradient(180deg, rgba(10, 14, 23, 0.95) 0%, rgba(10, 14, 23, 0.85) 100%);
            color: #4B5563;
            font-size: 0.62rem !important;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            padding: 0.75rem 0.75rem;
            border-bottom: 2px solid {border_color};
            text-align: left;
        }}
        .portfolio-table thead th.numeric {{ text-align: right; }}
        .portfolio-table tbody tr {{
            border-bottom: 1px solid rgba(255, 255, 255, 0.03);
            transition: background 0.2s ease;
        }}
        .portfolio-table tbody tr:nth-child(odd) {{ background: rgba(255, 255, 255, 0.01); }}
        .portfolio-table tbody tr:nth-child(even) {{ background: rgba(255, 255, 255, 0.005); }}
        .portfolio-table tbody tr:hover {{ background: {border_color}; }}
        .portfolio-table tbody td {{
            padding: 0.75rem 0.75rem;
            color: #F1F5F9;
            vertical-align: middle;
            font-size: 0.75rem !important;
        }}
        .portfolio-table tbody td.symbol {{
            font-weight: 700;
            font-size: 0.78rem;
            letter-spacing: 0.02em;
            font-family: 'Space Grotesk', sans-serif;
        }}
        .portfolio-table tbody td.numeric {{
            text-align: right;
            font-variant-numeric: tabular-nums;
        }}
    </style>
    </head>
    <body>
    <div class="portfolio-table">
        <table>
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th class="numeric">Price</th>
                    <th class="numeric">% Change</th>
                    <th class="numeric">Signal</th>
                    <th class="numeric">Conv</th>
                    <th class="numeric">Δ Conv</th>
                    <th class="numeric">Pulse</th>
                    <th class="numeric">Δ Pulse</th>
                    <th class="numeric">Narrative</th>
                </tr>
            </thead>
            <tbody>
                {"".join(table_rows)}
            </tbody>
        </table>
    </div>
    </body>
    </html>
    """
    return table_html

def _build_narrative_table_html(df: pd.DataFrame, side: str = 'long') -> str:
    """Build a simplified HTML table for Pulse Narrative mode showing all symbols."""
    import html as html_module

    accent_light = "#34D399" if side == 'long' else "#FB7185"
    border_color = "rgba(45, 212, 168, 0.3)" if side == 'long' else "rgba(232, 85, 90, 0.3)"
    
    table_rows = []
    if df.empty:
        table_rows.append(f"""
        <tr>
            <td colspan="9" style="text-align:center; color:#374151; font-family:'IBM Plex Mono',monospace;
                font-size:0.72rem; letter-spacing:0.06em; padding:2.25rem 1rem;">
                — no data available —
            </td>
        </tr>""")
    else:
        for _, row in df.iterrows():
            symbol = html_module.escape(str(row.get('DisplayName', row.get('Symbol', ''))))
            price = float(row.get('Price', 0))
            pct_change = float(row.get('PctChange', 0))
            signal = float(row.get('Signal', 0))
            conviction = float(row.get('Conviction', 0))
            pulse = float(row.get('Pulse', 0))
            conv_delta = float(row.get('Conviction_Delta', 0))
            pulse_delta = float(row.get('Pulse_Delta', 0))
            narrative = str(row.get('Narrative', 'NEUTRAL'))
            narr_color = str(row.get('Narrative_Color', '#94a3b8'))

            pct_color = "#34D399" if pct_change >= 0 else "#FB7185"
            conv_delta_color = "#34D399" if conv_delta >= 0 else "#FB7185"
            pulse_delta_color = "#4a9eff" if pulse_delta >= 0 else "#D4A853"
            conv_delta_arrow = "↑" if conv_delta >= 0 else "↓"
            pulse_delta_arrow = "↑" if pulse_delta >= 0 else "↓"

            table_rows.append(f"""
            <tr>
                <td class="symbol" style="color: #F1F5F9;">{symbol}</td>
                <td class="numeric currency">{price:,.2f}</td>
                <td class="numeric" style="color: {pct_color}; font-weight: 600;">{pct_change:+.2f}%</td>
                <td class="numeric" style="color: #60A5FA; font-weight: 600;">{signal:+.2f}</td>
                <td class="numeric" style="color: #D4A853; font-weight: 600;">{conviction:+.2f}</td>
                <td class="numeric" style="color: {conv_delta_color}; font-size: 0.65rem; font-weight: 600;">{conv_delta_arrow}{abs(conv_delta):.1f}</td>
                <td class="numeric" style="color: #4a9eff; font-weight: 600;">{pulse:+.2f}</td>
                <td class="numeric" style="color: {pulse_delta_color}; font-size: 0.65rem; font-weight: 600;">{pulse_delta_arrow}{abs(pulse_delta):.1f}</td>
                <td class="numeric" style="color: {narr_color}; font-weight: 700; letter-spacing: 0.05em; font-size: 0.68rem !important;">{narrative}</td>
            </tr>
            """)

    table_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'IBM Plex Mono', monospace;
            background: transparent;
            color: #F1F5F9;
            padding: 0.5rem;
            font-size: 14px;
        }}
        .portfolio-table {{
            width: 100%;
            border-radius: 8px;
            overflow-x: auto;
            border: 1px solid rgba(255, 255, 255, 0.05);
            background: rgba(10, 14, 23, 0.4);
        }}
        .portfolio-table table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .portfolio-table thead th {{
            background: rgba(15, 23, 42, 0.9);
            color: #94A3B8;
            font-size: 0.65rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            padding: 0.75rem;
            border-bottom: 2px solid {border_color};
            text-align: left;
        }}
        .portfolio-table thead th.numeric {{ text-align: right; }}
        .portfolio-table tbody tr {{
            border-bottom: 1px solid rgba(255, 255, 255, 0.03);
        }}
        .portfolio-table tbody tr:hover {{ background: rgba(255, 255, 255, 0.04); }}
        .portfolio-table tbody td {{
            padding: 0.85rem 0.75rem;
            vertical-align: middle;
            font-size: 0.75rem;
            white-space: nowrap;
        }}
        .portfolio-table tbody td.symbol {{
            font-weight: 700;
            font-family: 'Space Grotesk', sans-serif;
        }}
        .portfolio-table tbody td.numeric {{
            text-align: right;
            font-variant-numeric: tabular-nums;
        }}
    </style>
    </head>
    <body>
    <div class="portfolio-table">
        <table>
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th class="numeric">Price</th>
                    <th class="numeric">% Change</th>
                    <th class="numeric">Signal</th>
                    <th class="numeric">Conv</th>
                    <th class="numeric">Δ Conv</th>
                    <th class="numeric">Pulse</th>
                    <th class="numeric">Δ Pulse</th>
                    <th class="numeric">Narrative</th>
                </tr>
            </thead>
            <tbody>
                {''.join(table_rows)}
            </tbody>
        </table>
    </div>
    </body>
    </html>
    """
    return table_html



def _build_signal_strength_table_html(df: pd.DataFrame, side: str = 'long') -> str:
    """Build ranked HTML table for top signals by Abnormal Acceleration (Pulse).

    Creates styled HTML table with colored accent for side (long=green, short=red),
    displaying symbol, price, signal magnitude, trend direction, and zone status.
    Prioritizes Pulse (Velocity * Z-Score) as the ranking metric.

    Returns: Complete HTML document string ready for st.components.v1.html().
    """
    import html as html_module

    accent_light = "#34D399" if side == 'long' else "#FB7185"
    border_color = "rgba(45, 212, 168, 0.3)" if side == 'long' else "rgba(232, 85, 90, 0.3)"

    table_rows = []
    if df.empty:
        table_rows.append(f"""
        <tr>
            <td colspan="7" style="
                text-align: center;
                color: #374151;
                font-family: 'IBM Plex Mono', monospace;
                font-size: 0.72rem;
                letter-spacing: 0.06em;
                padding: 2.25rem 1rem;
            ">— no signals detected —</td>
        </tr>
        """)
    else:
        _zone_colors = {"OB Extreme": "#FB7185", "OB": "#FCA5A5",
                        "OS Extreme": "#34D399", "OS": "#86EFAC"}
        for idx, (_, row) in enumerate(df.iterrows(), 1):
            symbol = html_module.escape(str(row.get('DisplayName', row.get('Symbol', ''))))
            price = float(row.get('Price', 0))
            pct_change = float(row.get('PctChange', 0))
            signal = float(row.get('Signal', 0))
            conviction = float(row.get('Conviction', 0))
            pulse = float(row.get('Pulse', 0))
            conv_delta = float(row.get('Conviction_Delta', 0))
            narrative = str(row.get('Narrative', 'NEUTRAL'))
            narr_color = str(row.get('Narrative_Color', '#94a3b8'))

            rank_str = f"{idx:02d}"
            pct_color = "#34D399" if pct_change >= 0 else "#FB7185"
            conv_delta_color = "#34D399" if conv_delta >= 0 else "#FB7185"
            conv_delta_arrow = "↑" if conv_delta >= 0 else "↓"

            # v3 Metrics
            pct_rank = float(row.get(f'Priority_{side.capitalize()}_v3_pct', 0))
            hmm_bull = float(row.get('HMM_Bull', 0.5))
            hmm_bear = float(row.get('HMM_Bear', 0.5))
            vol_reg  = str(row.get('Vol_Regime', 'NORMAL'))
            
            # Regime Logic
            regime_tag = "NEUTRAL"
            regime_color = "#94a3b8"
            if side == 'long':
                if hmm_bull > 0.7: regime_tag, regime_color = "BULL", "#34D399"
                elif hmm_bull < 0.3: regime_tag, regime_color = "BEAR", "#FB7185"
            else:
                if hmm_bear > 0.7: regime_tag, regime_color = "BEAR", "#FB7185"
                elif hmm_bear < 0.3: regime_tag, regime_color = "BULL", "#34D399"
                
            vol_color = {"LOW": "#60a5fa", "NORMAL": "#94a3b8", "HIGH": "#fbbf24", "EXTREME": "#f87171"}.get(vol_reg, "#94a3b8")

            table_rows.append(f"""
            <tr>
                <td class="numeric" style="color: #D4A853; font-weight: 700;">{rank_str}</td>
                <td class="symbol">{symbol}</td>
                <td class="numeric" style="color: #4a9eff; font-weight: 700;">TOP {101-pct_rank:,.1f}%</td>
                <td class="numeric currency">{price:,.2f}</td>
                <td class="numeric" style="color: {pct_color}; font-weight: 600;">{pct_change:+.2f}%</td>
                <td class="numeric" style="color: {accent_light}; font-weight: 600;">{signal:+.2f}</td>
                <td class="numeric" style="color: #D4A853; font-weight: 600;">{conviction:+.2f}</td>
                <td class="numeric" style="color: {conv_delta_color}; font-size: 0.65rem; font-weight: 600;">{conv_delta_arrow}{abs(conv_delta):.1f}</td>
                <td class="numeric" style="color: #4a9eff; font-weight: 600;">{pulse:+.2f}</td>
                <td class="numeric" style="color: {regime_color}; font-weight: 700; font-size: 0.65rem;">{regime_tag}</td>
                <td class="numeric" style="color: {vol_color}; font-weight: 700; font-size: 0.65rem;">{vol_reg}</td>
                <td class="numeric" style="color: {narr_color}; font-weight: 700; letter-spacing: 0.05em; font-size: 0.68rem !important;">{narrative}</td>
            </tr>
            """)

    table_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'IBM Plex Mono', monospace;
            background: transparent;
            color: #F1F5F9;
            padding: 0.5rem;
        }}
        .portfolio-table {{
            width: 100%;
            border-radius: 10px;
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            border: 1px solid rgba(255, 255, 255, 0.05);
            background: linear-gradient(145deg, rgba(17, 24, 39, 0.45) 0%, rgba(17, 24, 39, 0.4) 100%);
        }}
        .portfolio-table table {{
            width: 100%;
            min-width: 480px;
            border-collapse: collapse;
        }}
        .portfolio-table thead th {{
            background: linear-gradient(180deg, rgba(10, 14, 23, 0.95) 0%, rgba(10, 14, 23, 0.85) 100%);
            color: #4B5563;
            font-size: 0.62rem !important;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            padding: 0.75rem 0.75rem;
            border-bottom: 2px solid {border_color};
            text-align: left;
        }}
        .portfolio-table thead th.numeric {{ text-align: right; }}
        .portfolio-table tbody tr {{
            border-bottom: 1px solid rgba(255, 255, 255, 0.03);
            transition: background 0.2s ease;
        }}
        .portfolio-table tbody tr:nth-child(odd) {{ background: rgba(255, 255, 255, 0.01); }}
        .portfolio-table tbody tr:nth-child(even) {{ background: rgba(255, 255, 255, 0.005); }}
        .portfolio-table tbody tr:hover {{ background: {border_color}; }}
        .portfolio-table tbody td {{
            padding: 0.85rem 0.75rem;
            color: #F1F5F9;
            vertical-align: middle;
            font-size: 0.75rem !important;
            white-space: nowrap;
        }}
        .portfolio-table tbody td.symbol {{
            font-weight: 700;
            font-size: 0.78rem;
            letter-spacing: 0.02em;
            font-family: 'Space Grotesk', sans-serif;
        }}
        .portfolio-table tbody td.numeric {{
            text-align: right;
            font-variant-numeric: tabular-nums;
        }}
    </style>
    </head>
    <body>
    <div class="portfolio-table">
        <table>
            <thead>
                <tr>
                    <th class="numeric">Rank</th>
                    <th>Symbol</th>
                    <th class="numeric">Percentile</th>
                    <th class="numeric">Price</th>
                    <th class="numeric">% Change</th>
                    <th class="numeric">Signal</th>
                    <th class="numeric">Conv</th>
                    <th class="numeric">Δ Conv</th>
                    <th class="numeric">Pulse</th>
                    <th class="numeric">Regime</th>
                    <th class="numeric">Vol</th>
                    <th class="numeric">Narrative</th>
                </tr>
            </thead>
            <tbody>
                {"".join(table_rows)}
            </tbody>
        </table>
    </div>
    </body>
    </html>
    """
    return table_html




def main():
    """Main app entry point with state-based flow."""
    # Render sidebar and get parameters + run button state
    sidebar_out = render_sidebar()
    universe, selected_index, analysis_date, reg_len, wt_n1, wt_n2, levels, timeframe, mode, start_date, end_date, run_clicked, corr_target_ticker, corr_lookback, corr_method = sidebar_out

    # Handle run button click
    if run_clicked:
        st.session_state["run_screener_flag"] = True
        st.session_state["timeseries_done"] = False
        st.session_state["run_correlation_flag"] = False
        st.session_state["results_df"] = None
        st.session_state["corr_data"] = None
        st.session_state["run_error"] = None
        st.rerun()

    # Reset display flags if mode switches
    if mode == "Single Date" and st.session_state.get("timeseries_done"):
        st.session_state["timeseries_done"] = False
        st.rerun()
    if mode == "Single Date" and st.session_state.get("corr_data"):
        st.session_state["corr_data"] = None
        st.rerun()

    # Show landing page if no results yet AND not in display modes
    if st.session_state["results_df"] is None and st.session_state.get("corr_data") is None and not st.session_state.get("run_screener_flag") and not st.session_state.get("timeseries_done"):
        ui.render_header("SANKET", "Market Signal Screener · संकेत · WRCI Engine")
        if st.session_state.get("run_error"):
            st.error(st.session_state["run_error"])
        render_landing_page()
        render_footer()
    else:
        # Run analysis if flagged
        if st.session_state.get("run_screener_flag"):
            if mode in ["Single Date", "Pulse Narrative"]:
                header_text = "Institutional Signal Screener" if mode == "Single Date" else "Pulse Narrative Analysis"
                console.header(f"SANKET TERMINAL — {header_text}", VERSION)
                console.main_header("ANALYSIS RUN START", {
                    "Universe": universe, "Index": selected_index, "Timeframe": timeframe, "Target Date": analysis_date, "Mode": mode
                })

                results_df = run_screener_analysis(universe, selected_index, analysis_date, reg_len, wt_n1, wt_n2, levels, timeframe)
                if results_df is None:
                    st.session_state["run_error"] = f"Failed to fetch constituents for '{selected_index}'."
                else:
                    st.session_state["run_error"] = None
                st.session_state["results_df"] = results_df
                st.session_state["run_screener_flag"] = False
                st.rerun()
            elif mode == "Historical Range":
                console.header("SANKET TERMINAL — Bulk Range Intelligence", VERSION)
                run_timeseries_analysis(universe, selected_index, start_date, end_date, reg_len, wt_n1, wt_n2, levels, timeframe)
                st.session_state["run_screener_flag"] = False
            else:  # Correlation
                corr_data = run_correlation_analysis(universe, selected_index, corr_target_ticker, corr_lookback, corr_method, timeframe, analysis_date)
                st.session_state["corr_data"] = corr_data
                st.session_state["run_screener_flag"] = False
                st.rerun()

        # Display single-date results
        if st.session_state["results_df"] is not None and not st.session_state.get("timeseries_done"):
            results_df = st.session_state["results_df"]
            
            # Safety: Ensure required columns exist
            if 'SimpleName' not in results_df.columns and not results_df.empty:
                results_df['SimpleName'] = results_df['Symbol'].str.replace(".NS", "", regex=False).str.lstrip("^")
            for _col in ['LA_Today','LA_1d','LA_2d','LA_3d','LA_5d','SA_Today','SA_1d','SA_2d','SA_3d','SA_5d','LB_Today','LB_1d','LB_2d','LB_3d','LB_5d','SB_Today','SB_1d','SB_2d','SB_3d','SB_5d','LC_Today','LC_1d','LC_2d','LC_3d','LC_5d','SC_Today','SC_1d','SC_2d','SC_3d','SC_5d']:
                if _col not in results_df.columns: results_df[_col] = "—"

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            if mode == "Pulse Narrative":
                tab_narrative, tab_strength, tab_raw = st.tabs(["Pulse Narrative Dashboard", "Signal Strength", "System Data"])
                with tab_narrative:
                    ui.render_section_header(f"Pulse Narrative — {timeframe} Universe State", "Full universe ranking by Abnormal Acceleration.", icon="zap", accent="amber")
                    avg_pulse = results_df['Pulse'].mean()
                    avg_conv = results_df['Conviction'].mean()
                    strong_pulse = len(results_df[results_df['Pulse'].abs() > 10])
                    bullish_bias = (results_df['Signal'] > 0).sum() / len(results_df) * 100 if len(results_df) > 0 else 0
                    m1, m2, m3, m4 = st.columns(4)
                    with m1: ui.render_metric_card("Universe Pulse", f"{avg_pulse:+.2f}", "Avg Acceleration", "neutral")
                    with m2: ui.render_metric_card("Universe Conv", f"{avg_conv:+.2f}", "Avg Conviction", "neutral")
                    with m3: ui.render_metric_card("High Pulse", str(strong_pulse), f"{strong_pulse/len(results_df)*100 if len(results_df)>0 else 0:.0f}% Universe", "info")
                    with m4: ui.render_metric_card("Bullish Bias", f"{bullish_bias:.0f}%", "Signal > 0", "success" if bullish_bias > 50 else "danger")
                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                    bull_narr_tab, bear_narr_tab = st.tabs(["Bullish Priority Ranking", "Bearish Priority Ranking"])
                    with bull_narr_tab:
                        bull_priority_df = results_df.sort_values('Priority_Long_v3', ascending=False)
                        st.components.v1.html(_build_narrative_table_html(bull_priority_df, side='long'), height=min(1200, 150 + len(bull_priority_df) * 52))
                    with bear_narr_tab:
                        bear_priority_df = results_df.sort_values('Priority_Short_v3', ascending=False)
                        st.components.v1.html(_build_narrative_table_html(bear_priority_df, side='short'), height=min(1200, 150 + len(bear_priority_df) * 52))
            else:
                tab_signals, tab_strength, tab_raw = st.tabs(["Action Dashboard", "Signal Strength", "System Data"])
                with tab_signals:
                    timeframe_label = "This Week's" if timeframe == 'Weekly' else "Today's"
                    ui.render_section_header(
                        f"{timeframe_label} Signals",
                        "Multi-condition momentum signals — Momentum (A) · Crossover (B) · Threshold (C) · Squeeze (D)",
                        icon="zap",
                        accent="amber"
                    )

                    # Set C: Momentum — sorted by Directional Priority v3
                    longs_df = results_df[results_df['L_5d'] != "—"].copy().sort_values('Priority_Long_v3', ascending=False)
                    shorts_df = results_df[results_df['S_5d'] != "—"].copy().sort_values('Priority_Short_v3', ascending=False)

                    # Set A: Momentum — broad crossover anywhere
                    has_bullish_crossover = (results_df[['LB_Today', 'LB_1d', 'LB_2d', 'LB_3d', 'LB_5d']] != "—").any(axis=1)
                    has_bearish_crossover = (results_df[['SB_Today', 'SB_1d', 'SB_2d', 'SB_3d', 'SB_5d']] != "—").any(axis=1)

                    longs_a_df = results_df[(results_df['LA_5d'] != "—") & ~has_bearish_crossover].copy().sort_values('Priority_Long_v3', ascending=False)
                    shorts_a_df = results_df[(results_df['SA_5d'] != "—") & ~has_bullish_crossover].copy().sort_values('Priority_Short_v3', ascending=False)

                    # Set B: Crossover
                    longs_b_df = results_df[results_df['LB_5d'] != "—"].copy().sort_values('Priority_Long_v3', ascending=False)
                    shorts_b_df = results_df[results_df['SB_5d'] != "—"].copy().sort_values('Priority_Short_v3', ascending=False)

                    # Set C: Threshold
                    longs_c_df = results_df[results_df['LC_5d'] != "—"].copy().sort_values('Priority_Long_v3', ascending=False)
                    shorts_c_df = results_df[results_df['SC_5d'] != "—"].copy().sort_values('Priority_Short_v3', ascending=False)

                    # Set D: Squeeze
                    longs_d_df = results_df[results_df['LD_5d'] != "—"].copy().sort_values('Priority_Long_v3', ascending=False)
                    shorts_d_df = results_df[results_df['SD_5d'] != "—"].copy().sort_values('Priority_Short_v3', ascending=False)

                    if timeframe == 'Weekly':
                        _age_order = ["This Week", "1 Week Ago", "2 Weeks Ago", "3 Weeks Ago", "Within 5 Weeks"]
                    else:
                        _age_order = ["Today", "1 Day Ago", "2 Days Ago", "3 Days Ago", "Within 5 Days"]

                    has_signals = any(not df_.empty for df_ in [longs_a_df, shorts_a_df, longs_b_df, shorts_b_df, longs_c_df, shorts_c_df, longs_d_df, shorts_d_df])

                    if has_signals:
                        total_longs  = len(longs_a_df) + len(longs_b_df) + len(longs_c_df) + len(longs_d_df)
                        total_shorts = len(shorts_a_df) + len(shorts_b_df) + len(shorts_c_df) + len(shorts_d_df)
                        all_longs  = pd.concat([longs_a_df, longs_b_df, longs_c_df, longs_d_df]).drop_duplicates('Symbol').sort_values('Priority_Long_v3', ascending=False)
                        all_shorts = pd.concat([shorts_a_df, shorts_b_df, shorts_c_df, shorts_d_df]).drop_duplicates('Symbol').sort_values('Priority_Short_v3', ascending=False)

                        mc1, mc2, mc3, mc4 = st.columns(4)
                        with mc1: ui.render_metric_card("Long Signals", str(total_longs), f"A:{len(longs_a_df)} B:{len(longs_b_df)} C:{len(longs_c_df)} D:{len(longs_d_df)}", "success")
                        with mc2: ui.render_metric_card("Short Signals", str(total_shorts), f"A:{len(shorts_a_df)} B:{len(shorts_b_df)} C:{len(shorts_c_df)} D:{len(shorts_d_df)}", "danger")
                        with mc3:
                            strongest_long = all_longs.iloc[0] if not all_longs.empty else None
                            ui.render_metric_card("Strongest Long", strongest_long['SimpleName'] if strongest_long is not None else "—", f"Signal: {strongest_long['Signal']:.1f}" if strongest_long is not None else "No signals", "info")
                        with mc4:
                            strongest_short = all_shorts.iloc[0] if not all_shorts.empty else None
                            ui.render_metric_card("Strongest Short", strongest_short['SimpleName'] if strongest_short is not None else "—", f"Signal: {strongest_short['Signal']:.1f}" if strongest_short is not None else "No signals", "info")

                        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                        bull_tab, bear_tab = st.tabs(["Bullish Signals by Timing", "Bearish Signals by Timing"])
                        with bull_tab:
                            mom_bull_tab, cross_bull_tab, thresh_bull_tab, sqz_bull_tab = st.tabs(["Momentum", "Crossover", "Threshold", "Squeeze"])
                            with mom_bull_tab:
                                _, la_stats, _, _ = _bucket_signals_by_age(longs_a_df, side='long', condition_set='A', timeframe=timeframe)
                                la_html = _build_signal_table_html(la_stats, side='long', timeframe=timeframe)
                                _g = sum(1 for a in _age_order if la_stats[a]['count'] > 0)
                                _r = sum(la_stats[a]['count'] for a in _age_order)
                                st.components.v1.html(la_html, height=max(120 + _g * 60 + _r * 56, 150))
                            with cross_bull_tab:
                                _, lb_stats, _, _ = _bucket_signals_by_age(longs_b_df, side='long', condition_set='B', timeframe=timeframe)
                                lb_html = _build_signal_table_html(lb_stats, side='long', timeframe=timeframe)
                                _g = sum(1 for a in _age_order if lb_stats[a]['count'] > 0)
                                _r = sum(lb_stats[a]['count'] for a in _age_order)
                                st.components.v1.html(lb_html, height=max(70 + _g * 46 + _r * 44, 110))
                            with thresh_bull_tab:
                                _, lc_stats, _, _ = _bucket_signals_by_age(longs_c_df, side='long', condition_set='C', timeframe=timeframe)
                                lc_html = _build_signal_table_html(lc_stats, side='long', timeframe=timeframe)
                                _g = sum(1 for a in _age_order if lc_stats[a]['count'] > 0)
                                _r = sum(lc_stats[a]['count'] for a in _age_order)
                                st.components.v1.html(lc_html, height=max(70 + _g * 46 + _r * 44, 110))
                            with sqz_bull_tab:
                                _, ld_stats, _, _ = _bucket_signals_by_age(longs_d_df, side='long', condition_set='D', timeframe=timeframe)
                                ld_html = _build_signal_table_html(ld_stats, side='long', timeframe=timeframe)
                                _g = sum(1 for a in _age_order if ld_stats[a]['count'] > 0)
                                _r = sum(ld_stats[a]['count'] for a in _age_order)
                                st.components.v1.html(ld_html, height=max(70 + _g * 46 + _r * 44, 110))
                        with bear_tab:
                            mom_bear_tab, cross_bear_tab, thresh_bear_tab, sqz_bear_tab = st.tabs(["Momentum", "Crossover", "Threshold", "Squeeze"])
                            with mom_bear_tab:
                                _, sa_stats, _, _ = _bucket_signals_by_age(shorts_a_df, side='short', condition_set='A', timeframe=timeframe)
                                sa_html = _build_signal_table_html(sa_stats, side='short', timeframe=timeframe)
                                _g = sum(1 for a in _age_order if sa_stats[a]['count'] > 0)
                                _r = sum(sa_stats[a]['count'] for a in _age_order)
                                st.components.v1.html(sa_html, height=max(70 + _g * 46 + _r * 44, 110))
                            with cross_bear_tab:
                                _, sb_stats, _, _ = _bucket_signals_by_age(shorts_b_df, side='short', condition_set='B', timeframe=timeframe)
                                sb_html = _build_signal_table_html(sb_stats, side='short', timeframe=timeframe)
                                _g = sum(1 for a in _age_order if sb_stats[a]['count'] > 0)
                                _r = sum(sb_stats[a]['count'] for a in _age_order)
                                st.components.v1.html(sb_html, height=max(70 + _g * 46 + _r * 44, 110))
                            with thresh_bear_tab:
                                _, sc_stats, _, _ = _bucket_signals_by_age(shorts_c_df, side='short', condition_set='C', timeframe=timeframe)
                                sc_html = _build_signal_table_html(sc_stats, side='short', timeframe=timeframe)
                                _g = sum(1 for a in _age_order if sc_stats[a]['count'] > 0)
                                _r = sum(sc_stats[a]['count'] for a in _age_order)
                                st.components.v1.html(sc_html, height=max(70 + _g * 46 + _r * 44, 110))
                            with sqz_bear_tab:
                                _, sd_stats, _, _ = _bucket_signals_by_age(shorts_d_df, side='short', condition_set='D', timeframe=timeframe)
                                sd_html = _build_signal_table_html(sd_stats, side='short', timeframe=timeframe)
                                _g = sum(1 for a in _age_order if sd_stats[a]['count'] > 0)
                                _r = sum(sd_stats[a]['count'] for a in _age_order)
                                st.components.v1.html(sd_html, height=max(70 + _g * 46 + _r * 44, 110))
                    else:
                        st.info("No signals detected in the specified universe and timeframe.")


            # Pre-compute top-10 rankings for Strength and Export tabs
            _longs_base = results_df[results_df['L_5d'] != "—"].copy()
            _shorts_base = results_df[results_df['S_5d'] != "—"].copy()
            top_longs = _longs_base.sort_values('Priority_Long_v3', ascending=False).head(10)
            top_shorts = _shorts_base.sort_values('Priority_Short_v3', ascending=False).head(10)

            # Tab 2 & 4 are common to both dashboards

            # ════ TAB 2: SIGNAL STRENGTH ANALYSIS ════════════════════════════════════════
            with tab_strength:
                ui.render_section_header(
                    "Abnormal Acceleration (Pulse)",
                    "Top signals ranked by Pulse — Momentum (A) · Crossover (B) · Threshold (C) · Squeeze (D)",
                    icon="zap",
                    accent="amber"
                )

                # Strength metrics
                avg_pulse = results_df['Pulse'].abs().mean()
                avg_conv = results_df['Conviction'].abs().mean()
                strong_pulse_count = len(results_df[results_df['Pulse'].abs() > 10])
                strong_trend_count = len(results_df[results_df['Trend'].abs() > 30])

                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                with col_s1: ui.render_metric_card("Avg Pulse", f"{avg_pulse:.1f}", "Abnormal Acceleration", "neutral")
                with col_s2: ui.render_metric_card("Avg Conviction", f"{avg_conv:.1f}", "Blended confluence", "neutral")
                with col_s3: ui.render_metric_card("Strong Pulse", str(strong_pulse_count), f"{strong_pulse_count/len(results_df)*100:.0f}% of universe", "info")
                with col_s4: ui.render_metric_card("Strong Trends", str(strong_trend_count), f"{strong_trend_count/len(results_df)*100:.0f}% of universe", "info")


                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

                # ── column label renderer ──
                def _col_label(side_label, side):
                    arrow = SVGS['LONG'].replace('currentColor', 'var(--emerald)') if side == 'long' else SVGS['SHORT'].replace('currentColor', 'var(--rose)')
                    color = 'var(--emerald)' if side == 'long' else 'var(--rose)'
                    return f"""
                    <p style="font-family:'IBM Plex Mono',monospace; font-size:0.62rem; font-weight:600;
                               text-transform:uppercase; letter-spacing:0.1em; color:{color};
                               margin:0 0 0.4rem 0; display:flex; align-items:center; gap:0.35rem;">
                        {arrow} {side_label}
                    </p>"""

                st.markdown(f"""
                <div style="display:flex; align-items:baseline; gap:0.65rem; margin:1.75rem 0 0.9rem 0;
                             padding-bottom:0.6rem; border-bottom:1px solid rgba(212,168,83,0.2);">
                    <span style="font-family:var(--display); font-size:0.62rem; font-weight:700;
                                 letter-spacing:0.12em; text-transform:uppercase; color:#D4A853;
                                 padding:0.18rem 0.5rem; background:rgba(212,168,83,0.1);
                                 border:1px solid rgba(212,168,83,0.3); border-radius:4px;">PRORITY ENGINE v3</span>
                    <span style="font-family:var(--display); font-size:1rem; font-weight:700;
                                 color:#F1F5F9; letter-spacing:0.04em;">Top 10 Rankings</span>
                </div>
                """, unsafe_allow_html=True)

                _col_l, _col_s = st.columns(2)
                with _col_l:
                    st.markdown(_col_label("Top 10 Bullish", "long"), unsafe_allow_html=True)
                    st.components.v1.html(_build_signal_strength_table_html(top_longs, side='long'), height=150 + len(top_longs)*55)
                with _col_s:
                    st.markdown(_col_label("Top 10 Bearish", "short"), unsafe_allow_html=True)
                    st.components.v1.html(_build_signal_strength_table_html(top_shorts, side='short'), height=150 + len(top_shorts)*55)

            # ════ TAB 4: SYSTEM DATA ════════════════════════════════════════════════════
            with tab_raw:
                ui.render_section_header(
                    "System Raw Data",
                    "Complete underlying data for model validation",
                    icon="database",
                    accent="cyan"
                )



                display_df = results_df.sort_values("Priority_Long_v3", ascending=False)[[
                    "DisplayName", "Price", "Signal", "Narrative", "Priority_Long_v3", "Conviction", "Pulse", "Wave"
                ]]
                st.dataframe(display_df, width='stretch', height=600)


                # ── SIGNAL TYPES REFERENCE SECTION ─────────────────────────────────────
                if mode != "Pulse Narrative":
                    st.markdown('<div class="section-divider" style="margin-top: 2rem;"></div>', unsafe_allow_html=True)
                    ui.render_section_header(
                        "Signal Types Reference",
                        "Four signal generation conditions — Momentum (A) · Crossover (B) · Threshold (C) · Squeeze (D)",
                        icon="layers",
                        accent="amber"
                    )
                    st.markdown("""
                    <p style="font-family: var(--data); font-size: 0.8rem; color: var(--ink-secondary); margin-bottom: 1.5rem;">
                        Reference guide for understanding the four signal generation methodologies and their key metrics.
                    </p>
                    """, unsafe_allow_html=True)
                    # Signal guide grid
                    st.markdown('<div class="signal-guide-grid"><div class="signal-type momentum"><div class="signal-type-label">Set A: Momentum</div><div class="signal-type-desc">Composite Line crosses Signal Line anywhere • No zone filter • Captures building momentum</div></div><div class="signal-type crossover"><div class="signal-type-label">Set B: Crossover</div><div class="signal-type-desc">Lines cross in extreme zones (±40) • Momentum exhaustion • High precision timing</div></div><div class="signal-type threshold"><div class="signal-type-label">Set C: Threshold</div><div class="signal-type-desc">Freshly enters OS/OB zone from neutral • Earliest actionable signal • Sorted by Pulse</div></div><div class="signal-type squeeze"><div class="signal-type-label">Set D: Squeeze</div><div class="signal-type-desc">Volatility squeeze firing • Bollinger Bands inside Keltner Channels • Expansion incoming</div></div></div>', unsafe_allow_html=True)

                # ── NARRATIVE MATRIX REFERENCE ──────────────────────────────────────────
                st.markdown('<div class="section-divider" style="margin-top: 2rem;"></div>', unsafe_allow_html=True)


                ui.render_section_header(
                    "Narrative Matrix",
                    "16 institutional states classified by Conviction Δ (Y) × Pulse Δ (X)",
                    icon="grid",
                    accent="indigo"
                )

                st.markdown("""
                <div style="margin-bottom: 1.5rem;">
                    <p style="font-family: var(--data); font-size: 0.8rem; color: var(--ink-secondary);">
                        The Intelligence Matrix fuses <b>Structural Conviction</b> (Trend Depth) with <b>Abnormal Pulse</b> (Acceleration) to classify every signal into one of 16 institutional states.
                    </p>
                </div>
                
                <div class="narrative-guide-grid">
                    <div class="signal-type crush-card"><div class="signal-type-label">LOAD</div><div class="narr-desc">Crush | Lead<br>Whale Entry</div></div>
                    <div class="signal-type soft-card"><div class="signal-type-label">HARDEN</div><div class="narr-desc">Soft | Lead<br>Base Building</div></div>
                    <div class="signal-type firm-card"><div class="signal-type-label">IGNITE</div><div class="narr-desc">Firm | Lead<br>Strong Flow</div></div>
                    <div class="signal-type surge-card"><div class="signal-type-label">SQUEEZE</div><div class="narr-desc">Surge | Lead<br>Max Ignition</div></div>
                    <div class="signal-type crush-card"><div class="signal-type-label">TRAP</div><div class="narr-desc">Crush | Deep<br>Bear/Bull Trap</div></div>
                    <div class="signal-type soft-card"><div class="signal-type-label">STEALTH</div><div class="narr-desc">Soft | Deep<br>Quiet Acc.</div></div>
                    <div class="signal-type firm-card"><div class="signal-type-label">ORGANIC</div><div class="narr-desc">Firm | Deep<br>Healthy Trend</div></div>
                    <div class="signal-type surge-card"><div class="signal-type-label">HYPER</div><div class="narr-desc">Surge | Deep<br>Accelerated</div></div>
                    <div class="signal-type crush-card"><div class="signal-type-label">CAPITUL</div><div class="narr-desc">Crush | Light<br>Final Wash</div></div>
                    <div class="signal-type soft-card"><div class="signal-type-label">RETRACE</div><div class="narr-desc">Soft | Light<br>Orderly Pull</div></div>
                    <div class="signal-type firm-card"><div class="signal-type-label">EXPAND</div><div class="narr-desc">Firm | Light<br>Volatility</div></div>
                    <div class="signal-type surge-card"><div class="signal-type-label">EXHAUST</div><div class="narr-desc">Surge | Light<br>Blow-off Top</div></div>
                    <div class="signal-type crush-card"><div class="signal-type-label">CRASH</div><div class="narr-desc">Crush | Hollow<br>Full Collapse</div></div>
                    <div class="signal-type soft-card"><div class="signal-type-label">LIQUID</div><div class="narr-desc">Soft | Hollow<br>Liquidation</div></div>
                    <div class="signal-type firm-card"><div class="signal-type-label">POP</div><div class="narr-desc">Firm | Hollow<br>Flash Burst</div></div>
                    <div class="signal-type surge-card"><div class="signal-type-label">CHAOS</div><div class="narr-desc">Surge | Hollow<br>Noise Bubble</div></div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown('<div class="section-divider" style="margin-top: 2rem;"></div>', unsafe_allow_html=True)
                ui.render_section_header(
                    "Export Quant Dataset",
                    "Signal archives by timing and top-ranked strength lists",
                    icon="download",
                    accent="cyan"
                )

                # Row 1 — full signal lists
                st.markdown('<p style="font-family:\'IBM Plex Mono\',monospace; font-size:0.65rem; color:#4B5563; text-transform:uppercase; letter-spacing:0.08em; margin: 0.5rem 0 0.4rem 0;">Signals by Timing</p>', unsafe_allow_html=True)
                dl_col1, dl_col2 = st.columns(2)
                with dl_col1:
                    st.download_button(
                        label="↑  Bullish Signals",
                        data=results_df[results_df['Signal'] > 0].to_csv(index=False).encode('utf-8'),
                        file_name=f"bullish_signals_{analysis_date}.csv",
                        mime="text/csv",
                        key="dl_bullish_timing",
                        help="All active bullish signals grouped by timing",
                        use_container_width=True
                    )
                with dl_col2:
                    st.download_button(
                        label="↓  Bearish Signals",
                        data=results_df[results_df['Signal'] < 0].to_csv(index=False).encode('utf-8'),
                        file_name=f"bearish_signals_{analysis_date}.csv",
                        mime="text/csv",
                        key="dl_bearish_timing",
                        help="All active bearish signals grouped by timing",
                        use_container_width=True
                    )

                # Row 2 — top 10 by strength
                st.markdown('<p style="font-family:\'IBM Plex Mono\',monospace; font-size:0.65rem; color:#4B5563; text-transform:uppercase; letter-spacing:0.08em; margin: 0.9rem 0 0.4rem 0;">Top 10 by Strength</p>', unsafe_allow_html=True)
                dl_col3, dl_col4 = st.columns(2)
                with dl_col3:
                    st.download_button(
                        label="↑  Top 10 Bullish",
                        data=top_longs.to_csv(index=False).encode('utf-8'),
                        file_name=f"top10_bullish_{analysis_date}.csv",
                        mime="text/csv",
                        key="dl_top10_bullish",
                        help="Top 10 bullish signals ranked by Abnormal Acceleration (Pulse)",
                        use_container_width=True
                    )
                with dl_col4:
                    st.download_button(
                        label="↓  Top 10 Bearish",
                        data=top_shorts.to_csv(index=False).encode('utf-8'),
                        file_name=f"top10_bearish_{analysis_date}.csv",
                        mime="text/csv",
                        key="dl_top10_bearish",
                        help="Top 10 bearish signals ranked by Abnormal Acceleration (Pulse)",
                        use_container_width=True
                    )

                render_footer()

        # Display Correlation results
        if st.session_state.get("corr_data") is not None:
            render_correlation_results(st.session_state["corr_data"])
            render_footer()
if __name__ == "__main__":
    main()
