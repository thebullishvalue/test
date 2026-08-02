"""
FVE — Fair Value Engine
=======================

Streamlit dashboard for cross-sectional, market-relative valuation.

    streamlit run app.py

The engine estimates a target asset's fair value as the price path implied by
a 200+ instrument cross-asset universe, and publishes the deviation as a
bounded, adaptively-normalised oscillator. All modelling lives in the ``fve``
package; this file is presentation and wiring only.
"""
from __future__ import annotations

import warnings
from dataclasses import asdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fve import __version__
from fve.backtest import (decile_forward_returns, information_coefficients,
                          run_backtest, signal_event_study)
from fve.cache import begin_force_refresh
from fve.config import BacktestConfig, DataConfig, EngineConfig, REGIME_META
from fve.data import load_universe, provider_status, resolve_symbol
from fve.engine import run_engine, snapshot
from fve import explain as ex
from fve import theme, viz
from fve.regimes import regime_summary
from fve.universe import (CLASS_OF, FREEFORM_MARKETS, MARKET_HINTS, TAPE,
                          UNIVERSE, class_breakdown, symbols_for_tier,
                          target_categories)

warnings.filterwarnings("ignore")

st.set_page_config(page_title="FVE — Fair Value Engine", layout="wide",
                   initial_sidebar_state="expanded", page_icon="◈")
st.markdown(theme.CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached compute
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=4)
def load_data(tier: str, years: int, synthetic: bool, seed: int, target: str):
    cfg = DataConfig(years=years, synthetic=synthetic, seed=seed)
    # The explanatory universe is passed target-independent on purpose: the
    # panel cache is keyed on this list, so appending a free-form ticker would
    # mint a new key per target and re-download the universe every time. A
    # target outside the universe is fetched separately by load_universe.
    symbols = symbols_for_tier(tier)

    bar = st.progress(0.0, text="Contacting data provider…")

    def report(done, total, msg):
        bar.progress(min(done / max(total, 1), 1.0), text=msg)

    try:
        bundle = load_universe(symbols, cfg, target=target, progress=report)
    finally:
        bar.empty()
    return bundle


@st.cache_data(show_spinner=False, max_entries=6)
def calibrate(prices: pd.DataFrame, cfg_key: tuple, _cfg: EngineConfig):
    """Calibrate, memoised on ``(prices, cfg_key)``.

    ``cfg_key`` is never read — it exists so the cache keys on the
    configuration. The config itself is passed underscore-prefixed because
    Streamlit cannot hash a dataclass, and would otherwise refuse the call.
    """
    bar = st.progress(0.0, text="Calibrating…")

    def report(frac, msg):
        bar.progress(frac, text=msg)

    try:
        result = run_engine(prices, _cfg, progress=report)
    finally:
        bar.empty()
    return result


# ---------------------------------------------------------------------------
# Sidebar — controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(theme.section("Engine Controls", "Configuration"), unsafe_allow_html=True)

    with st.expander("◈  Data source", expanded=True):
        n_core = len(symbols_for_tier("core"))
        n_ext = len(symbols_for_tier("extended"))
        tier = st.radio("Universe tier", [f"Core ({n_core})", f"Extended ({n_ext})"],
                        horizontal=True,
                        help="The explanatory cross-section the target is priced "
                             "against. Extended gives the factor model more to work "
                             "with but takes noticeably longer to download.")
        tier_key = "core" if tier.startswith("Core") else "extended"
        years = st.select_slider("History (years)", [3, 5, 8, 10], value=8)
        synthetic = st.toggle(
            "Demo mode — synthetic market", value=False,
            help="Generates a reproducible factor-driven market offline. Useful "
                 "when the data provider is unreachable, and as a control: the "
                 "simulator contains no exploitable mispricing, so the engine "
                 "should report roughly zero signal on it.")
        if st.button("↻  Force refresh from provider"):
            begin_force_refresh()
            load_data.clear()
            st.toast("Cache bypassed for this session — next run refetches live.")

    with st.expander("◈  Target asset", expanded=True):
        # Asset class → target. Individual equities have no curated list: they
        # are entered as a ticker and resolved against the provider, which is
        # both open-ended and honest about what actually exists.
        categories = target_categories(symbols_for_tier(tier_key))
        cat_names = list(categories.keys())

        st.session_state.setdefault("target_category", "US Style/Factor ETFs")
        if st.session_state["target_category"] not in cat_names:
            st.session_state["target_category"] = cat_names[0]
        sel_cat = st.selectbox("Asset class", cat_names, key="target_category")

        target: str | None = None
        target_label = ""

        if sel_cat in FREEFORM_MARKETS:
            market = FREEFORM_MARKETS[sel_cat]
            meta = MARKET_HINTS[market]
            raw_symbol = st.text_input(
                "Ticker symbol", key=f"symbol_{market}",
                placeholder=meta["placeholder"],
                help="Any listed equity. The symbol is resolved against Yahoo "
                     "Finance before the engine will run.")
            if raw_symbol and raw_symbol.strip():
                if synthetic:
                    # No provider to probe in demo mode — accept the symbol and
                    # simulate a series for it.
                    target = raw_symbol.strip().upper()
                    target_label = f"{target} · simulated"
                    st.caption(f"Demo mode — **{target}** will be simulated.")
                else:
                    with st.spinner("Resolving symbol…"):
                        ticker, exchange = resolve_symbol(raw_symbol, market, years)
                    if ticker is None:
                        st.error(exchange)
                    else:
                        target = ticker
                        target_label = f"{ticker} · {exchange}"
                        st.caption(f"**{raw_symbol.strip().upper()} → {ticker}** · "
                                   f"{exchange}")
            else:
                st.caption(meta["hint"])
        else:
            options = categories[sel_cat]
            if st.session_state.get("target_select") not in options:
                st.session_state["target_select"] = (
                    "QQQ" if "QQQ" in options else options[0])
            target = st.selectbox("Target", options, key="target_select")
            target_label = f"{target} · {CLASS_OF.get(target, '')}"

    with st.expander("◈  Factor model", expanded=False):
        method = st.selectbox("Latent factor method",
                              ["PCA", "ICA", "FactorAnalysis", "Autoencoder"],
                              help="PCA maximises explained variance; ICA seeks "
                                   "statistically independent drivers; "
                                   "FactorAnalysis models idiosyncratic noise "
                                   "explicitly; Autoencoder allows non-linear "
                                   "compression.")
        n_factors = st.slider("Latent factors", 3, 20, 8)
        n_peers = st.slider("Orthogonal peers", 0, 30, 12,
                            help="Instruments still correlated with the target "
                                 "*after* the common factors are removed.")
        exclude_corr = st.slider("Drop instruments correlated above", 0.80, 1.0, 0.99, 0.01,
                                 help="Guards against near-perfect proxies. With "
                                      "an index proxy in the explanatory set, fair "
                                      "value collapses onto price and the "
                                      "oscillator becomes noise.")
        exclude_class = st.toggle("Exclude the target's own asset class", False)

    with st.expander("◈  Calibration", expanded=False):
        lookback = st.select_slider("Calibration window (sessions)",
                                    [126, 189, 252, 378, 504], value=252)
        refit = st.select_slider("Recalibration interval (sessions)",
                                 [5, 10, 21, 42, 63], value=21)
        auto_alpha = st.toggle("Auto-select ridge penalty (GCV)", True)
        ridge_alpha = st.select_slider("Ridge penalty λ", [0.1, 1.0, 10.0, 100.0, 1000.0],
                                       value=10.0, disabled=auto_alpha)
        halflife = st.select_slider("Sample weight half-life (sessions)",
                                    [0, 21, 42, 63, 126, 252], value=63,
                                    format_func=lambda x: "uniform" if x == 0 else str(x))
        regime_weighting = st.toggle("Regime-aware sample weighting", True,
                                     help="Up-weights history recorded in the same "
                                          "regime as today.")

    with st.expander("◈  Oscillator", expanded=False):
        fv_horizon = st.select_slider("Fair value anchor horizon H (sessions)",
                                      [10, 21, 42, 63, 126], value=63,
                                      help="Mispricing is the target's cumulative "
                                           "excess return versus the market-implied "
                                           "path over the last H sessions.")
        norm_window = st.select_slider("Adaptive normalisation window",
                                       [42, 63, 126, 252], value=126)
        robust_norm = st.toggle("Robust normalisation (median / MAD)", True)
        saturation = st.select_slider("Saturation κ", [1.5, 2.0, 2.5, 3.0, 4.0], value=2.5)
        smoothing = st.selectbox("Smoothing filter", ["Kalman", "EMA", "None"])
        smooth_span = st.slider("Filter span / gain", 2, 21, 6)
        threshold_q = st.select_slider("Dynamic threshold quantile",
                                       [0.05, 0.08, 0.12, 0.18, 0.25], value=0.12)
        mr_horizon = st.slider("Mean-reversion horizon (sessions)", 1, 42, 10)

    with st.expander("◈  Signal backtest", expanded=False):
        allow_short = st.toggle("Allow short signals", True)
        exit_level = st.slider("Exit when |FVO| falls below", 0, 60, 10)
        max_hold = st.slider("Maximum holding period (sessions)", 5, 90, 21)
        cost_bps = st.slider("Cost per side (bp)", 0.0, 20.0, 2.0, 0.5)
        size_by_conf = st.toggle("Size positions by confidence", False)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    run_clicked = st.button("▶  RUN ANALYSIS", type="primary", disabled=target is None)
    if target is None:
        st.caption("Choose or enter a target asset to enable the run.")
    st.caption("Research tool. Market data via Yahoo Finance. "
               "Not investment advice.")

# ---------------------------------------------------------------------------
# Staged configuration
#
# Nothing below executes off the live widgets. Selections are assembled into a
# pending configuration and only committed when RUN ANALYSIS is pressed.
# Streamlit reruns the whole script on every widget interaction, so binding the
# engine directly to the widgets would kick off a multi-minute download and a
# full recalibration each time the user touched a control — including while
# they were still part-way through choosing one.
# ---------------------------------------------------------------------------
pending_engine = EngineConfig(
    target=target or "", tier=tier_key, method=method, n_factors=n_factors,
    lookback=lookback, refit_every=refit, n_peers=n_peers,
    ridge_alpha=ridge_alpha, auto_alpha=auto_alpha, halflife_days=float(halflife),
    regime_weighting=regime_weighting, exclude_corr_above=exclude_corr,
    exclude_same_class=exclude_class, fv_horizon=fv_horizon,
    norm_window=norm_window, robust_norm=robust_norm, saturation=saturation,
    smoothing=smoothing, smooth_span=smooth_span, threshold_q=threshold_q,
    mr_horizon=mr_horizon,
)
pending_bt = BacktestConfig(allow_short=allow_short, exit_level=float(exit_level),
                            max_hold=max_hold, cost_bps=cost_bps,
                            size_by_confidence=size_by_conf)
pending_sig = (tier_key, years, synthetic, target, pending_engine.key(),
               tuple(sorted(asdict(pending_bt).items())))

if run_clicked and target is not None:
    st.session_state["active"] = dict(
        engine=pending_engine, bt=pending_bt, tier=tier_key, years=years,
        synthetic=synthetic, target=target, label=target_label, sig=pending_sig)

active = st.session_state.get("active")

# ---------------------------------------------------------------------------
# Landing state
# ---------------------------------------------------------------------------
if active is None:
    st.markdown(theme.header(len(symbols_for_tier(tier_key)), len(UNIVERSE),
                             "STANDBY", pd.Timestamp.today(), __version__),
                unsafe_allow_html=True)
    st.markdown(theme.section(
        "Getting started", "A market-relative valuation model",
        "The engine prices one asset against the state of everything else. Each "
        "session, the cross-section of 200+ instruments is compressed into latent "
        "factors plus the orthogonal peers that still matter once those factors are "
        "removed; a walk-forward, regime-weighted ridge regression maps that market "
        "state onto the target's return. Fair value is the price path that state "
        "implies, and the Fair Value Oscillator is the normalised deviation from it."
    ), unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(theme.kpi("Selected target", target or "— none —",
                              target_label or "enter a ticker or pick from the list",
                              accent="cyan"), unsafe_allow_html=True)
    with c2:
        st.markdown(theme.kpi("Explanatory universe", f"{len(symbols_for_tier(tier_key))}",
                              f"{len(UNIVERSE)} asset classes · {years}y history",
                              accent="blue"), unsafe_allow_html=True)
    with c3:
        st.markdown(theme.kpi("Model", f"{method} × {n_factors}",
                              f"{lookback}d window · refit every {refit}d",
                              accent="amber"), unsafe_allow_html=True)

    st.markdown(
        "<div class='note'>Configure the target and model, then press "
        "<b>RUN ANALYSIS</b>. Nothing is fetched or calibrated until you do. The "
        "first live download of a few hundred instruments takes several minutes; "
        "it is cached to disk, so later runs start immediately and only a changed "
        "universe or history length triggers a refetch. Switch on <b>Demo mode</b> "
        "for an instant offline run against a synthetic market.</div>",
        unsafe_allow_html=True)
    st.stop()

# Everything below reads the COMMITTED configuration, never the live widgets.
engine_cfg = active["engine"]
bt_cfg = active["bt"]
target = active["target"]
target_label = active["label"]
tier_key, years, synthetic = active["tier"], active["years"], active["synthetic"]
smoothing = engine_cfg.smoothing
threshold_q = engine_cfg.threshold_q
robust_norm = engine_cfg.robust_norm
auto_alpha = engine_cfg.auto_alpha
halflife = engine_cfg.halflife_days
regime_weighting = engine_cfg.regime_weighting
allow_short = bt_cfg.allow_short
exit_level = bt_cfg.exit_level
max_hold = bt_cfg.max_hold
cost_bps = bt_cfg.cost_bps

settings_changed = active["sig"] != pending_sig

# ---------------------------------------------------------------------------
# Load + calibrate
# ---------------------------------------------------------------------------
try:
    bundle = load_data(tier_key, years, synthetic, 7, target)
except Exception as exc:  # noqa: BLE001 — surface any provider failure to the user
    st.error(f"**Data load failed.** {exc}")
    st.info("Enable **Demo mode** in the sidebar to run the engine against a "
            "synthetic market while the provider is unavailable.")
    st.stop()

if target not in bundle.prices.columns:
    st.error(f"**{target}** returned no usable history and was dropped "
             f"({bundle.dropped.get(target, 'unknown reason')}). Choose another target.")
    st.stop()

try:
    E = calibrate(bundle.prices, engine_cfg.key(), engine_cfg)
except Exception as exc:  # noqa: BLE001
    st.error(f"**Calibration failed.** {exc}")
    st.stop()

S = snapshot(E)
D = E["dates"]

# ---------------------------------------------------------------------------
# Header + tape
# ---------------------------------------------------------------------------
st.markdown(theme.header(bundle.n_assets, len(class_breakdown(list(bundle.prices.columns))),
                         bundle.source, bundle.asof, __version__),
            unsafe_allow_html=True)

if settings_changed:
    st.warning("Sidebar settings have changed. The results below are from the "
               "last completed run — press **RUN ANALYSIS** to apply them.",
               icon="⚠️")

for _note in bundle.notes:
    st.caption(f"· {_note}")

tape_items = []
for sym, label in TAPE.items():
    if sym in bundle.prices.columns:
        s = bundle.prices[sym].dropna()
        if len(s) > 2:
            tape_items.append((label, 100.0 * (s.iloc[-1] / s.iloc[-2] - 1.0)))
st.markdown(theme.tape(tape_items), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# KPI console
# ---------------------------------------------------------------------------
reg_meta = REGIME_META.get(S["regime"], {"color": "#8fa3c4", "icon": "◇", "desc": ""})
gap_colour = viz.RED if S["gap"] > 0 else viz.GREEN

st.markdown(theme.kpi_row([
    theme.kpi(f"{target} · market price · {CLASS_OF.get(target, '')}",
              viz.fmt_price(S["price"]),
              f"<span class='{'up' if S['day_change'] >= 0 else 'dn'}'>"
              f"{'▲' if S['day_change'] >= 0 else '▼'} {100 * S['day_change']:+.2f}%</span>"
              f" 1D · feed {bundle.source}", accent="blue"),
    theme.kpi("Model fair value", viz.fmt_price(S["fv"]),
              f"implied by {len(E['others'])} instruments · "
              f"{E['config'].method}×{E['config'].n_factors}F · H={E['config'].fv_horizon}d",
              accent="amber", value_color=viz.AMBER),
    theme.kpi("Fair value gap",
              ("+" if S["gap"] > 0 else "") + viz.fmt_price(S["gap"]),
              f"<span class='am'>{S['mis_pct']:+.2f}%</span> mispricing · "
              f"<b style='color:{S['verdict_color']}'>{S['verdict']}</b>",
              accent="red" if S["gap"] > 0 else "green", value_color=gap_colour),
    theme.kpi("Fair value oscillator", f"{S['fvo']:+.1f}",
              f"z = {S['z']:+.2f}σ · {S['percentile']:.0f}th percentile of history",
              accent="cyan", value_color=viz.CYAN,
              meter=(S["fvo"] + 100) / 2,
              meter_color="linear-gradient(90deg,#2fd08c,#f2b544,#ff5d6c)"),
]), unsafe_allow_html=True)

st.markdown(theme.kpi_row([
    theme.kpi("Market regime",
              f"<span class='pill' style='color:{reg_meta['color']};"
              f"border:1px solid {reg_meta['color']}66;background:{reg_meta['color']}1a'>"
              f"{reg_meta['icon']} {S['regime']}</span>",
              reg_meta.get("desc", ""), accent=""),
    theme.kpi("Confidence", f"{S['confidence']:.0f}<span style='font-size:14px;"
                            f"color:#7f93b8'>/100</span>",
              f"out-of-sample R² {S['oos_r2']:.2f} · stability & coverage weighted",
              accent="green", meter=S["confidence"], meter_color=viz.GREEN),
    theme.kpi(f"Mean reversion P({E['config'].mr_horizon}d)", f"{S['p_mr']:.0f}%",
              f"OU half-life ≈ {S['half_life']:.0f} sessions",
              accent="violet", meter=S["p_mr"], meter_color=viz.VIOLET),
    theme.kpi("Residual risk", f"{S['sigma_e']:.2f}<span style='font-size:14px;"
                               f"color:#7f93b8'>%/day</span>",
              "idiosyncratic volatility the market cannot explain · 42d",
              accent="red"),
]), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tabs = st.tabs(["◈ Valuation", "◈ Oscillator Lab", "◈ Drivers", "◈ Regime & Stability",
                "◈ Signal Evidence", "◈ Universe & Data"])

# ---------------------------------------------------------- 1 · VALUATION ---
with tabs[0]:
    st.markdown(theme.section(
        "Core output", "Price versus model-implied fair value",
        f"Fair value is anchored {E['config'].fv_horizon} sessions back and rolled "
        f"forward on the return the market state implies, so the gap reads as "
        f"{target}'s cumulative excess move against the rest of the world. The band "
        f"is ±1σ of unexplained residual risk over that horizon; markers flag "
        f"oscillator crossings back inside the dynamic thresholds and price / "
        f"oscillator divergences."), unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    show_raw = c1.checkbox("Show unsmoothed oscillator", True)
    show_reg = c2.checkbox("Shade market regimes", True)
    st.plotly_chart(viz.price_and_oscillator(E, show_raw, show_reg), width="stretch")

    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(viz.gap_bars(E), width="stretch")
    c2.plotly_chart(viz.mispricing_histogram(E, S["mis_pct"]), width="stretch")

    hist = E["mis_pct"].dropna()
    if len(hist):
        wider = float((hist.abs() > abs(S["mis_pct"])).mean() * 100)
        st.markdown(
            f"<div class='note'>The current gap of <b>{S['mis_pct']:+.2f}%</b> has been "
            f"exceeded in absolute terms on <b>{wider:.1f}%</b> of sessions in this "
            f"sample. Mean absolute gap: {hist.abs().mean():.2f}% · "
            f"95th percentile: {hist.abs().quantile(0.95):.2f}%.</div>",
            unsafe_allow_html=True)

# ------------------------------------------------------ 2 · OSCILLATOR LAB ---
with tabs[1]:
    st.markdown(theme.section(
        "Signal anatomy", "Oscillator laboratory",
        "Adaptive normalisation keeps a reading comparable across volatility "
        "regimes; thresholds are trailing empirical quantiles rather than fixed "
        "levels, and are lagged one session so they never contain the observation "
        "they are judging."), unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2])
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=D, y=E["fvo_raw"], name="raw",
                                 line=dict(color="rgba(63,216,201,.32)", width=1)))
        fig.add_trace(go.Scatter(x=D, y=E["fvo"], name=f"smoothed · {smoothing}",
                                 line=dict(color=viz.CYAN, width=2)))
        fig.add_trace(go.Scatter(x=D, y=E["ob"], name="overbought",
                                 line=dict(color="rgba(255,93,108,.6)", width=1, dash="dash")))
        fig.add_trace(go.Scatter(x=D, y=E["os_"], name="oversold",
                                 line=dict(color="rgba(47,208,140,.6)", width=1, dash="dash")))
        fig.add_hline(y=0, line_color="rgba(140,165,215,.35)")
        st.plotly_chart(viz.style(fig, 360, title="OSCILLATOR & DYNAMIC THRESHOLDS"),
                        width="stretch")
    with c2:
        st.plotly_chart(viz.oscillator_distribution(E, S["fvo"], threshold_q),
                        width="stretch")

    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        st.plotly_chart(viz.gauge(S["fvo"], S["os_"], S["ob"]), width="stretch")
        st.caption(f"Reading sits at the **{S['percentile']:.0f}th percentile** of its "
                   f"own history. Bands mark the current dynamic thresholds.")
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=D, y=E["z"], name="mispricing z",
                                 line=dict(color=viz.AMBER, width=1.5)))
        fig.add_hrect(y0=1, y1=5, fillcolor="rgba(255,93,108,.07)", line_width=0)
        fig.add_hrect(y0=-5, y1=-1, fillcolor="rgba(47,208,140,.07)", line_width=0)
        fig.add_hline(y=0, line_color="rgba(140,165,215,.3)")
        st.plotly_chart(viz.style(fig, 260, legend=False,
                                  title="PRE-SATURATION Z-SCORE OF log(P / FV)"),
                        width="stretch")
        st.caption("The signal before the tanh bound is applied — useful for judging "
                   "how far into the tail a saturated reading really is.")
    with c3:
        n_bear, n_bull = int(E["div_bear"].sum()), int(E["div_bull"].sum())
        st.markdown(
            theme.kpi("Bearish divergences", f"{n_bear}",
                      "price higher high, oscillator lower high", accent="red")
            + "<div style='height:10px'></div>"
            + theme.kpi("Bullish divergences", f"{n_bull}",
                        "price lower low, oscillator higher low", accent="green")
            + "<div style='height:10px'></div>"
            + theme.kpi("OU half-life", f"{S['half_life']:.0f}d",
                        f"P(revert to ±0.5σ within {E['config'].mr_horizon}d) = "
                        f"<b>{S['p_mr']:.0f}%</b>", accent="violet"),
            unsafe_allow_html=True)

    st.markdown(theme.section(
        "Horizon scan", "Multi-timeframe fair value",
        "One calibration, several anchor horizons. A short-horizon dislocation is a "
        "tactical stretch; when short and long horizons agree, the target is "
        "dislocated against the market on every timescale the model can see."),
        unsafe_allow_html=True)
    cols = st.columns(len(E["mtf"]))
    for col, (label, series) in zip(cols, E["mtf"].items()):
        v = float(series.iloc[-1])
        clean = series.dropna()
        pct = float((clean < v).mean() * 100) if len(clean) else np.nan
        if v > 45:
            tag, colour = "RICH", viz.RED
        elif v < -45:
            tag, colour = "CHEAP", viz.GREEN
        else:
            tag, colour = "NEUTRAL", viz.MUTED
        with col:
            st.markdown(theme.kpi(f"anchor {label}", f"{v:+.1f}",
                                  f"<b style='color:{colour}'>{tag}</b> · "
                                  f"{pct:.0f}th percentile",
                                  accent="cyan", value_color=viz.CYAN),
                        unsafe_allow_html=True)
            st.plotly_chart(viz.mtf_panel(series), width="stretch")

# ------------------------------------------------------------ 3 · DRIVERS ---
with tabs[2]:
    st.markdown(theme.section(
        "Explainability", "What is setting fair value right now?",
        "The fair-value map is linear in its transformed features, so attribution is "
        "exact rather than approximated: β·x decomposes the implied return into a "
        "contribution per latent factor and per orthogonal peer, in basis points per "
        "day. Positive contributions push fair value up, which makes the asset look "
        "cheaper at an unchanged price."), unsafe_allow_html=True)

    if E["contrib"].empty:
        st.info("No attribution available for this configuration.")
    else:
        row = E["contrib"].iloc[-1].dropna()
        row = row.reindex(row.abs().sort_values(ascending=False).index)
        c1, c2 = st.columns([3, 2])
        with c1:
            st.plotly_chart(
                viz.contribution_bars(row.head(16),
                                      "TOP DRIVERS OF IMPLIED RETURN (bp/day)"),
                width="stretch")
        with c2:
            st.plotly_chart(
                viz.contribution_bars(ex.contribution_by_class(row).head(10),
                                      "CONTRIBUTION BY ASSET CLASS (bp/day)",
                                      height=380, by_class=True),
                width="stretch")

        st.plotly_chart(viz.contribution_heatmap(E["contrib"]), width="stretch")

        c1, c2 = st.columns(2)
        with c1:
            with st.spinner("Computing permutation importance…"):
                imp = ex.permutation_importance(E).head(14)
            fig = go.Figure(go.Bar(x=imp.values[::-1], y=list(imp.index)[::-1],
                                   orientation="h", marker_color="rgba(242,181,68,.85)",
                                   marker_line=dict(width=0)))
            st.plotly_chart(viz.style(fig, 380, legend=False,
                                      title="PERMUTATION IMPORTANCE · ΔR² WHEN SCRAMBLED"),
                            width="stretch")
        with c2:
            with st.spinner("Computing SHAP attributions…"):
                sv, backend = ex.shap_values(E)
            mean_abs = sv.abs().mean().sort_values(ascending=False).head(14)
            fig = go.Figure(go.Bar(x=mean_abs.values[::-1], y=list(mean_abs.index)[::-1],
                                   orientation="h", marker_color="rgba(79,156,249,.85)",
                                   marker_line=dict(width=0)))
            st.plotly_chart(viz.style(fig, 380, legend=False,
                                      title="MEAN |SHAP| · bp/day"), width="stretch")
            st.caption(f"Computed via `{backend}`. For a linear model the "
                       f"interventional SHAP value is exactly β·(x − E[x]), so this "
                       f"agrees with the contribution decomposition by construction.")

        st.markdown(theme.section(
            "Latent structure", "What the factors actually are",
            "Principal components arrive unnamed. Mapping the loadings back onto "
            "instruments and asset classes is what turns 'F3' into an economic "
            "statement."), unsafe_allow_html=True)
        st.dataframe(ex.factor_interpretation(E).style.format(
            {"variance %": "{:.2f}", "β (bp/day per σ)": "{:+.1f}"}),
            width="stretch", hide_index=True)

        c1, c2 = st.columns([2, 3])
        with c1:
            k = E["last"].factors.loadings.shape[0]
            which = st.selectbox("Inspect factor", [f"F{i + 1}" for i in range(k)])
            idx = int(which[1:]) - 1
            st.plotly_chart(
                viz.factor_loadings_chart(ex.factor_loadings_frame(E, idx, 18),
                                          f"{which} — LARGEST LOADINGS"),
                width="stretch")
        with c2:
            hist_coef = ex.coefficient_history(E)
            fig = go.Figure()
            palette = [viz.CYAN, viz.AMBER, viz.VIOLET, viz.GREEN, viz.BLUE,
                       viz.RED, "#7fb3ff", "#d08cf0"]
            for i, col in enumerate(hist_coef.columns):
                fig.add_trace(go.Scatter(x=hist_coef.index, y=hist_coef[col], name=col,
                                         line=dict(width=1.4,
                                                   color=palette[i % len(palette)])))
            fig.add_hline(y=0, line_color="rgba(140,165,215,.3)")
            st.plotly_chart(
                viz.style(fig, 360,
                          title="FACTOR COEFFICIENT PATHS ACROSS RECALIBRATIONS (bp/day per σ)"),
                width="stretch")
            st.caption("Components are sign- and order-aligned between refits, so a "
                       "line changing sign here is a genuine change in the "
                       "relationship — not the arbitrary sign flip PCA would "
                       "otherwise produce.")

        st.markdown(theme.section("Peer stability", "Which instruments keep mattering", ""),
                    unsafe_allow_html=True)
        pp = ex.peer_persistence(E)
        if len(pp):
            st.dataframe(pp.head(25).style.format({"share %": "{:.1f}"}),
                         width="stretch", hide_index=True)
            churn = 100.0 - float(pp["share %"].head(E["config"].n_peers or 1).mean())
            st.caption(f"A peer set that survives across refits reflects a stable "
                       f"economic relationship. Mean turnover of the current peer "
                       f"count: **{churn:.0f}%**.")

# ------------------------------------------------- 4 · REGIME & STABILITY ---
with tabs[3]:
    st.markdown(theme.section(
        "Diagnostics", "Regime timeline and model stability",
        "Regimes are k-means clusters over the market's state vector, refit "
        "walk-forward on trailing data only — a full-sample clustering would leak "
        "the future into every regime-conditioned weight below. CUSUM accumulates "
        "evidence that the residual relationship has shifted."), unsafe_allow_html=True)

    st.plotly_chart(viz.regime_ribbon(E["regimes"]), width="stretch")
    legend = " &nbsp;·&nbsp; ".join(
        f"<span style='color:{m['color']}'>{m['icon']} {n}</span>"
        for n, m in REGIME_META.items())
    st.markdown(f"<div style='font-family:JetBrains Mono;font-size:10.5px;"
                f"letter-spacing:.08em;margin:-8px 0 14px'>{legend}</div>",
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    c1.plotly_chart(viz.diagnostics_chart(E), width="stretch")
    c2.plotly_chart(viz.cusum_chart(E), width="stretch")

    c1, c2 = st.columns(2)
    c1.plotly_chart(viz.stability_chart(E), width="stretch")
    c2.plotly_chart(viz.market_state_chart(E["market_state"]), width="stretch")

    st.markdown(theme.section("Regime behaviour", "How the target and oscillator behave "
                              "in each state", ""), unsafe_allow_html=True)
    summary = regime_summary(E["regimes"], E["rets"][target], E["fvo"])
    st.dataframe(summary.style.format({
        "share %": "{:.1f}", "target μ (bp/d)": "{:+.1f}", "target σ (bp/d)": "{:.1f}",
        "FVO mean": "{:+.1f}", "FVO σ": "{:.1f}"}), width="stretch", hide_index=True)

    if E["breaks"]:
        recent = ", ".join(str(pd.Timestamp(b).date()) for b in E["breaks"][-6:])
        st.markdown(f"<div class='note'><b>{len(E['breaks'])} structural breaks</b> "
                    f"flagged in the fair-value residual. Most recent: {recent}. "
                    f"A break means the relationship between the target and the market "
                    f"state shifted faster than the recalibration schedule could track."
                    f"</div>", unsafe_allow_html=True)

# --------------------------------------------------- 5 · SIGNAL EVIDENCE ---
with tabs[4]:
    st.markdown(theme.section(
        "Does the signal carry information?", "Evidence before strategy",
        "The first two panels are parameter-free: no thresholds, no trading rule, "
        "nothing to overfit. If the information coefficient is flat and the bucket "
        "profile is unsloped, no strategy built on this oscillator will work, however "
        "attractive the equity curve below happens to look."), unsafe_allow_html=True)

    c1, c2 = st.columns([2, 3])
    with c1:
        ic = information_coefficients(E)
        st.dataframe(ic.style.format({"IC": "{:+.3f}", "p-value": "{:.3f}"})
                     .background_gradient(subset=["IC"], cmap="RdYlGn", vmin=-0.15, vmax=0.15),
                     width="stretch", hide_index=True)
        st.caption("Spearman rank correlation between the **negated** oscillator and "
                   "subsequent returns, so positive means the signal works as "
                   "intended. Daily observations overlap, so p-values are optimistic; "
                   "treat |IC| < 0.03 as noise.")
    with c2:
        dec = decile_forward_returns(E, horizon=E["config"].mr_horizon)
        if len(dec):
            st.plotly_chart(viz.decile_chart(dec, E["config"].mr_horizon), width="stretch")
        else:
            st.info("Not enough history to form oscillator buckets.")

    ev = signal_event_study(E, horizon=21)
    if len(ev.columns):
        st.plotly_chart(viz.event_study_chart(ev), width="stretch")

    st.markdown(theme.section(
        "Historical replay", "Trading rule backtest",
        f"Long when the oscillator crosses back above its dynamic oversold "
        f"threshold{', short on the mirror condition' if allow_short else ''}; exit "
        f"when the dislocation closes to ±{exit_level} or after {max_hold} sessions. "
        f"Signals use only trailing information, execution is at the next session, "
        f"and {cost_bps:.1f}bp is charged per side."), unsafe_allow_html=True)

    bt = run_backtest(E, bt_cfg)
    stats = bt["stats"]
    cards = [
        theme.kpi("Trades", f"{stats['trades']:.0f}",
                  f"exposure {stats['exposure %']:.0f}% of sessions"),
        theme.kpi("Win rate", f"{stats['win rate %']:.0f}%"
                  if np.isfinite(stats["win rate %"]) else "—",
                  f"profit factor {stats['profit factor']:.2f}"
                  if np.isfinite(stats["profit factor"]) else "", accent="green"),
        theme.kpi("Sharpe", f"{stats['sharpe']:.2f}"
                  if np.isfinite(stats["sharpe"]) else "—",
                  f"buy &amp; hold {stats['bh sharpe']:.2f}", accent="cyan"),
        theme.kpi("CAGR", f"{stats['cagr %']:+.1f}%",
                  f"buy &amp; hold {stats['bh cagr %']:+.1f}%", accent="amber"),
    ]
    st.markdown(theme.kpi_row(cards), unsafe_allow_html=True)
    st.markdown(theme.kpi_row([
        theme.kpi("Max drawdown", f"{stats['max drawdown %']:.1f}%",
                  f"buy &amp; hold {stats['bh max dd %']:.1f}%", accent="red"),
        theme.kpi("Average trade", f"{stats['avg trade %']:+.2f}%"
                  if np.isfinite(stats["avg trade %"]) else "—",
                  f"held {stats['avg hold (d)']:.0f} sessions on average"
                  if np.isfinite(stats["avg hold (d)"]) else ""),
        theme.kpi("Best trade", f"{stats['best %']:+.1f}%"
                  if np.isfinite(stats["best %"]) else "—", "", accent="green"),
        theme.kpi("Worst trade", f"{stats['worst %']:+.1f}%"
                  if np.isfinite(stats["worst %"]) else "—", "", accent="red"),
    ]), unsafe_allow_html=True)

    st.plotly_chart(viz.equity_chart(bt), width="stretch")

    if len(bt["trades"]):
        st.dataframe(bt["trades"].tail(20).iloc[::-1].style.format(
            {"ret_pct": "{:+.2f}", "entry_fvo": "{:+.0f}", "exit_fvo": "{:+.0f}"}),
            width="stretch", hide_index=True)
    else:
        st.info("No completed trades at these thresholds — widen the entry quantile "
                "or lengthen the maximum holding period.")

    st.markdown("<div class='note'>A single backtest path is one draw from a large "
                "parameter space. The sidebar exposes enough degrees of freedom to "
                "fit any curve you like; the parameter-free evidence above is the "
                "part that resists that temptation.</div>", unsafe_allow_html=True)

# ---------------------------------------------------- 6 · UNIVERSE & DATA ---
with tabs[5]:
    st.markdown(theme.section(
        "Provenance", "Universe composition and data quality",
        "Every instrument the model saw, what was discarded, and why. The panel is "
        "aligned to a liquid equity trading calendar rather than the union of all "
        "instrument calendars — otherwise crypto's weekend sessions would inject "
        "rows that are empty for everything else."), unsafe_allow_html=True)

    breakdown = class_breakdown(list(bundle.prices.columns))
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(theme.kpi("Instruments loaded", f"{bundle.n_assets}",
                          f"of {bundle.requested} requested", accent="cyan"),
                unsafe_allow_html=True)
    c2.markdown(theme.kpi("Asset classes", f"{len(breakdown)}", "", accent="blue"),
                unsafe_allow_html=True)
    c3.markdown(theme.kpi("Sessions", f"{bundle.n_sessions:,}",
                          f"{D[0]:%b %Y} → {D[-1]:%b %Y}", accent="amber"),
                unsafe_allow_html=True)
    c4.markdown(theme.kpi("Median coverage",
                          f"{100 * bundle.coverage.median():.1f}%",
                          f"{len(bundle.dropped)} instruments dropped", accent="green"),
                unsafe_allow_html=True)

    c1, c2 = st.columns([2, 3])
    with c1:
        bd = pd.Series(breakdown).sort_values(ascending=True)
        fig = go.Figure(go.Bar(x=bd.values, y=bd.index, orientation="h",
                               marker_color=[viz.color_of(i) for i in bd.index],
                               marker_line=dict(width=0)))
        st.plotly_chart(viz.style(fig, 420, legend=False,
                                  title="INSTRUMENTS PER ASSET CLASS"), width="stretch")
    with c2:
        corr = E["rets"].corrwith(E["rets"][target]).drop(labels=[target], errors="ignore")
        corr = corr.reindex(corr.abs().sort_values(ascending=False).index).head(24)
        fig = go.Figure(go.Bar(x=corr.values[::-1], y=list(corr.index)[::-1],
                               orientation="h",
                               marker_color=[viz.color_of(str(i)) for i in corr.index][::-1],
                               marker_line=dict(width=0)))
        fig.add_vline(x=0, line_color="rgba(140,165,215,.3)")
        st.plotly_chart(viz.style(fig, 420, legend=False,
                                  title=f"RAW CORRELATION WITH {target} · FULL SAMPLE"),
                        width="stretch")
        st.caption("Raw correlation is what the engine deliberately moves beyond: "
                   "peers are selected on correlation that survives removing the "
                   "common factors, not on this ranking.")

    st.markdown(theme.section(
        "Provider health", "Fetch infrastructure",
        "Every provider call runs through a retry-with-backoff, a circuit breaker "
        "that stops hammering a failing service, and a two-tier cache whose expired "
        "entries are retained as a last-good snapshot. A partial response — Yahoo "
        "rate-limits a few tickers per batch — triggers a targeted re-fetch of just "
        "the missing symbols, then a snapshot backfill, so the cross-section stays "
        "whole instead of silently shrinking."), unsafe_allow_html=True)

    status = provider_status()
    circuit = status["circuit"]
    circuit_colour = {"closed": viz.GREEN, "half_open": viz.AMBER,
                      "open": viz.RED}.get(circuit["state"], viz.MUTED)
    pc = status["panel_cache"]
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(theme.kpi("Provider circuit", circuit["state"].upper().replace("_", "-"),
                          f"{circuit['failures']} failures · {circuit['successes']} successes",
                          value_color=circuit_colour), unsafe_allow_html=True)
    c2.markdown(theme.kpi("Panel cache hit rate", f"{100 * pc['hit_rate']:.0f}%",
                          f"{pc['hits']} hits · {pc['misses']} misses · "
                          f"{pc['stale_hits']} stale", accent="cyan"),
                unsafe_allow_html=True)
    c3.markdown(theme.kpi("Fetch time", f"{bundle.fetch_seconds:.0f}s"
                          if bundle.fetch_seconds else "cached",
                          f"feed {bundle.source}", accent="blue"), unsafe_allow_html=True)
    c4.markdown(theme.kpi("Snapshot backfills", f"{len(bundle.backfilled)}",
                          "instruments carried from a prior snapshot",
                          accent="amber" if bundle.backfilled else ""),
                unsafe_allow_html=True)

    if bundle.backfilled:
        with st.expander(f"Backfilled instruments ({len(bundle.backfilled)})"):
            st.dataframe(pd.DataFrame({
                "instrument": list(bundle.backfilled.keys()),
                "class": [CLASS_OF.get(s, "—") for s in bundle.backfilled],
                "last native observation": list(bundle.backfilled.values()),
            }), width="stretch", hide_index=True)
            st.caption("These came back empty from the provider and were restored "
                       "from the most recent good snapshot. Anything more than 10 "
                       "sessions stale is dropped instead of filled — a flat line "
                       "forward-filled across weeks would distort its factor loading.")

    if bundle.dropped:
        with st.expander(f"Dropped instruments ({len(bundle.dropped)})"):
            st.dataframe(pd.DataFrame({
                "instrument": list(bundle.dropped.keys()),
                "class": [CLASS_OF.get(s, "—") for s in bundle.dropped],
                "reason": list(bundle.dropped.values()),
            }), width="stretch", hide_index=True)

    with st.expander("Current model specification"):
        spec = pd.DataFrame({
            "setting": ["target", "universe tier", "instruments used", "factor method",
                        "latent factors", "orthogonal peers", "calibration window",
                        "recalibration interval", "ridge penalty", "sample half-life",
                        "regime weighting", "fair value anchor H", "normalisation",
                        "saturation κ", "smoothing", "threshold quantile",
                        "calibrations run", "calibration time"],
            # Cast uniformly to str: a mixed int/str column cannot be
            # serialised to Arrow and Streamlit would have to coerce it.
            "value": [str(v) for v in [
                target, tier_key, len(E["others"]), E["config"].method,
                E["config"].n_factors, len(E["last"].peers),
                f"{E['lookback']} sessions", f"{E['config'].refit_every} sessions",
                f"{E['last'].alpha:g}" + (" (auto)" if auto_alpha else ""),
                "uniform" if not halflife else f"{halflife} sessions",
                "on" if regime_weighting else "off",
                f"{E['config'].fv_horizon} sessions",
                f"{'robust median/MAD' if robust_norm else 'mean/std'} "
                f"over {E['config'].norm_window}",
                E["config"].saturation, E["config"].smoothing,
                E["config"].threshold_q, len(E["calibrations"]),
                f"{E['elapsed']:.1f}s"]],
        })
        st.dataframe(spec, width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
st.markdown(
    f"<div style='margin-top:26px;padding-top:12px;border-top:1px solid "
    f"rgba(140,165,215,.14);font-family:JetBrains Mono;font-size:10px;"
    f"letter-spacing:.11em;color:#5f7396'>FVE v{__version__} · {bundle.n_assets} "
    f"INSTRUMENTS · {E['config'].method}×{E['config'].n_factors} FACTORS · "
    f"{len(E['calibrations'])} WALK-FORWARD CALIBRATIONS · FEED {bundle.source} · "
    f"RESEARCH TOOL — NOT INVESTMENT ADVICE</div>",
    unsafe_allow_html=True)
