"""Plotly chart builders and the dashboard's visual language."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import REGIME_META
from .regimes import regime_spans
from .universe import color_of

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
INK = "#eef3ff"
MUTED = "#8fa3c4"
CYAN = "#3fd8c9"
BLUE = "#4f9cf9"
AMBER = "#f2b544"
GREEN = "#2fd08c"
RED = "#ff5d6c"
VIOLET = "#b7a4f3"
GRID = "rgba(140,165,215,.07)"

MONO = "JetBrains Mono, IBM Plex Mono, ui-monospace, monospace"
SANS = "Plus Jakarta Sans, Inter, system-ui, sans-serif"

AX = dict(
    gridcolor=GRID,
    zerolinecolor="rgba(140,165,215,.16)",
    tickfont=dict(family=MONO, size=10, color=MUTED),
    title_font=dict(family=MONO, size=10, color="#7f93b8"),
)


def style(fig: go.Figure, height: int = 420, legend: bool = True,
          title: str | None = None) -> go.Figure:
    """Apply the dashboard's chart theme."""
    fig.update_layout(
        template="plotly_dark", height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(11,18,31,.55)",
        margin=dict(l=14, r=14, t=42 if title else 28, b=16),
        hoverlabel=dict(bgcolor="#0d1726", bordercolor="rgba(63,216,201,.45)",
                        font=dict(family=MONO, size=11)),
        showlegend=legend,
        legend=dict(orientation="h", y=1.10, x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(family=MONO, size=10, color="#a9bcdd")),
        hovermode="x unified",
    )
    if title:
        fig.update_layout(title=dict(
            text=title, x=0, xanchor="left", y=0.98,
            font=dict(family=MONO, size=10.5, color="#7f93b8")))
    fig.update_xaxes(**AX)
    fig.update_yaxes(**AX)
    return fig


def fmt_price(v: float) -> str:
    if not np.isfinite(v):
        return "—"
    a = abs(v)
    if a >= 10000:
        return f"{v:,.0f}"
    if a >= 100:
        return f"{v:,.2f}"
    if a >= 1:
        return f"{v:,.3f}"
    return f"{v:.5f}"


def paint_regimes(fig: go.Figure, labels: pd.Series, row: int | None = None,
                  opacity: float = 0.07, max_spans: int = 260) -> None:
    """Shade the background by regime."""
    spans = regime_spans(labels)[-max_spans:]
    for a, b, r in spans:
        meta = REGIME_META.get(r)
        if not meta:
            continue
        kw = dict(x0=a, x1=b, fillcolor=meta["color"], opacity=opacity,
                  layer="below", line_width=0)
        if row is not None:
            fig.add_vrect(row=row, col=1, **kw)
        else:
            fig.add_vrect(**kw)


# ---------------------------------------------------------------------------
# Core chart: price vs fair value, with the oscillator beneath
# ---------------------------------------------------------------------------
def price_and_oscillator(E: dict, show_raw: bool = True,
                         show_regimes: bool = True, height: int = 660) -> go.Figure:
    D = E["dates"]
    target = E["target"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38],
                        vertical_spacing=0.055)

    if show_regimes:
        paint_regimes(fig, E["regimes"], row=1)
        paint_regimes(fig, E["regimes"], row=2, opacity=0.05)

    # --- fair value uncertainty band: ±1σ of residual risk over the horizon
    fig.add_trace(go.Scatter(x=D, y=E["fv_hi"], line=dict(width=0),
                             showlegend=False, hoverinfo="skip"), 1, 1)
    fig.add_trace(go.Scatter(x=D, y=E["fv_lo"], fill="tonexty",
                             fillcolor="rgba(242,181,68,.10)", line=dict(width=0),
                             name="fair value ±1σ"), 1, 1)

    fig.add_trace(go.Scatter(x=D, y=E["price"], name=f"{target} price",
                             line=dict(color=INK, width=1.7)), 1, 1)
    fig.add_trace(go.Scatter(x=D, y=E["fv"], name="model fair value",
                             line=dict(color=AMBER, width=1.7, dash="dot")), 1, 1)

    for mask, label, sym, colour in (
            (E["long_entry"], "undervalued signal", "triangle-up", GREEN),
            (E["short_entry"], "overvalued signal", "triangle-down", RED)):
        if mask.any():
            fig.add_trace(go.Scatter(
                x=D[mask], y=E["price"][mask], mode="markers", name=label,
                marker=dict(symbol=sym, size=11, color=colour,
                            line=dict(color="#0c1322", width=1))), 1, 1)
    for mask, label, colour in ((E["div_bull"], "bullish divergence", GREEN),
                                (E["div_bear"], "bearish divergence", RED)):
        if mask.any():
            fig.add_trace(go.Scatter(
                x=D[mask], y=E["price"][mask], mode="markers", name=label,
                marker=dict(symbol="diamond-open", size=10, color=colour,
                            line=dict(width=1.6))), 1, 1)

    # --- oscillator pane
    fig.add_trace(go.Scatter(x=D, y=E["ob"], name="dynamic overbought",
                             line=dict(color="rgba(255,93,108,.6)", width=1, dash="dash")), 2, 1)
    fig.add_trace(go.Scatter(x=D, y=E["os_"], name="dynamic oversold",
                             line=dict(color="rgba(47,208,140,.6)", width=1, dash="dash")), 2, 1)
    fig.add_trace(go.Scatter(x=D, y=E["fvo"] + E["band"], line=dict(width=0),
                             showlegend=False, hoverinfo="skip"), 2, 1)
    fig.add_trace(go.Scatter(x=D, y=E["fvo"] - E["band"], fill="tonexty",
                             fillcolor="rgba(63,216,201,.12)", line=dict(width=0),
                             name="oscillator confidence band"), 2, 1)
    if show_raw:
        fig.add_trace(go.Scatter(x=D, y=E["fvo_raw"], name="raw FVO",
                                 line=dict(color="rgba(63,216,201,.30)", width=1)), 2, 1)
    fig.add_trace(go.Scatter(x=D, y=E["fvo"], name="FVO",
                             line=dict(color=CYAN, width=2.1)), 2, 1)
    fig.add_hline(y=0, line_color="rgba(140,165,215,.45)", line_width=1, row=2, col=1)

    fig.update_yaxes(title_text="price", row=1, col=1)
    fig.update_yaxes(title_text="FVO   (0 = fair)", range=[-105, 105], row=2, col=1)
    style(fig, height=height)
    fig.update_layout(hovermode="x unified")
    return fig


# ---------------------------------------------------------------------------
# Supporting charts
# ---------------------------------------------------------------------------
def mispricing_histogram(E: dict, current: float, height: int = 250) -> go.Figure:
    vals = E["mis_pct"].dropna()
    fig = go.Figure(go.Histogram(x=vals, nbinsx=70, marker_color="rgba(63,216,201,.5)",
                                 marker_line=dict(width=0)))
    if np.isfinite(current):
        fig.add_vline(x=current, line_color=AMBER, line_width=2,
                      annotation_text=f"now {current:+.2f}%",
                      annotation_font=dict(color=AMBER, family=MONO, size=10))
    return style(fig, height, legend=False,
                 title="MISPRICING DISTRIBUTION · % OF PRICE")


def gap_bars(E: dict, height: int = 250) -> go.Figure:
    v = E["mis_pct"]
    fig = go.Figure(go.Bar(x=E["dates"], y=v, marker_line=dict(width=0),
                           marker_color=np.where(v >= 0, RED, GREEN), opacity=0.85))
    fig.add_hline(y=0, line_color="rgba(140,165,215,.35)")
    return style(fig, height, legend=False,
                 title="FAIR VALUE GAP · POSITIVE = TRADING ABOVE MODEL")


def oscillator_distribution(E: dict, current: float, q: float,
                            height: int = 330) -> go.Figure:
    vals = E["fvo"].dropna()
    fig = go.Figure(go.Histogram(x=vals, nbinsx=60, marker_color="rgba(183,164,243,.55)",
                                 marker_line=dict(width=0)))
    fig.add_vline(x=float(vals.quantile(1 - q)), line_dash="dash", line_color=RED)
    fig.add_vline(x=float(vals.quantile(q)), line_dash="dash", line_color=GREEN)
    fig.add_vline(x=current, line_color=AMBER, line_width=2,
                  annotation_text=f"now {current:+.0f}",
                  annotation_font=dict(color=AMBER, family=MONO, size=10))
    return style(fig, height, legend=False,
                 title="FVO EMPIRICAL DISTRIBUTION & DYNAMIC THRESHOLDS")


def gauge(value: float, low: float, high: float, height: int = 260) -> go.Figure:
    lo = low if np.isfinite(low) else -60.0
    hi = high if np.isfinite(high) else 60.0
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=float(value),
        number=dict(font=dict(family=MONO, size=34, color=CYAN),
                    valueformat="+.1f"),
        gauge=dict(
            axis=dict(range=[-100, 100], tickfont=dict(color=MUTED, size=9)),
            bar=dict(color=CYAN, thickness=0.25),
            bgcolor="rgba(13,20,35,.6)", borderwidth=1,
            bordercolor="rgba(140,165,215,.25)",
            steps=[dict(range=[-100, lo], color="rgba(47,208,140,.18)"),
                   dict(range=[hi, 100], color="rgba(255,93,108,.18)")],
            threshold=dict(line=dict(color=AMBER, width=3), thickness=0.8,
                           value=float(value)))))
    fig.update_layout(height=height, margin=dict(t=28, b=10, l=20, r=20),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def contribution_bars(series: pd.Series, title: str, height: int = 380,
                      by_class: bool = False) -> go.Figure:
    s = series.iloc[::-1]
    colours = ([color_of(str(i)) for i in s.index] if by_class
               else [RED if v > 0 else GREEN for v in s.values])
    fig = go.Figure(go.Bar(x=s.values, y=[str(i) for i in s.index], orientation="h",
                           marker_color=colours, marker_line=dict(width=0)))
    fig.add_vline(x=0, line_color="rgba(140,165,215,.3)")
    return style(fig, height, legend=False, title=title)


def contribution_heatmap(contrib: pd.DataFrame, n_rows: int = 16,
                         n_cols: int = 60, height: int = 420) -> go.Figure:
    if contrib.empty:
        return style(go.Figure(), height, legend=False, title="NO ATTRIBUTION AVAILABLE")
    recent = contrib.tail(n_cols)
    rank = recent.abs().mean().sort_values(ascending=False)
    cols = rank.head(n_rows).index.tolist()
    Zm = recent[cols].T.values
    lim = float(np.nanpercentile(np.abs(Zm), 98)) or 1.0
    fig = go.Figure(go.Heatmap(
        z=Zm, x=recent.index, y=cols, zmid=0, zmin=-lim, zmax=lim,
        colorscale=[[0, GREEN], [0.5, "#0f1729"], [1, RED]],
        colorbar=dict(title=dict(text="bp/d", font=dict(size=9, color=MUTED)),
                      tickfont=dict(size=9, color=MUTED), thickness=10)))
    fig.update_yaxes(autorange="reversed")
    return style(fig, height, legend=False,
                 title=f"DRIVER ATTRIBUTION · LAST {len(recent)} SESSIONS (bp/day)")


def regime_ribbon(labels: pd.Series, height: int = 96) -> go.Figure:
    fig = go.Figure()
    for a, b, r in regime_spans(labels):
        meta = REGIME_META.get(r)
        if meta:
            fig.add_vrect(x0=a, x1=b, fillcolor=meta["color"], opacity=0.8, line_width=0)
    fig.update_yaxes(visible=False, range=[0, 1])
    fig.update_layout(height=height, margin=dict(t=6, b=6, l=14, r=14))
    return style(fig, height, legend=False)


def diagnostics_chart(E: dict, height: int = 300) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=E["dates"], y=E["oos_r2"], name="out-of-sample R² (63d)",
                            line=dict(color=GREEN, width=1.6)))
    fig.add_trace(go.Scatter(x=E["dates"], y=E["confidence"] / 100.0,
                             name="confidence / 100",
                             line=dict(color=VIOLET, width=1.2, dash="dot")))
    fig.add_hline(y=0, line_color="rgba(140,165,215,.3)")
    return style(fig, height, title="EXPLANATORY POWER — GENUINELY OUT-OF-SAMPLE")


def stability_chart(E: dict, height: int = 300) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=E["dates"], y=E["coef_drift"], name="coefficient drift",
                             line=dict(color=BLUE, width=1.5)))
    fig.add_trace(go.Scatter(x=E["dates"], y=E["factor_drift"], name="factor loading drift",
                             line=dict(color=AMBER, width=1.5)))
    return style(fig, height, title="ROLLING STABILITY · MEAN |Δ| BETWEEN RECALIBRATIONS")


def cusum_chart(E: dict, height: int = 300) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=E["dates"], y=E["cusum"], name="CUSUM statistic",
                             line=dict(color=AMBER, width=1.5)))
    fig.add_hline(y=5.0, line_dash="dash", line_color="rgba(255,93,108,.6)")
    for ts in E["breaks"]:
        fig.add_vline(x=ts, line_color=RED, line_width=1, opacity=0.55)
    return style(fig, height,
                 title=f"STRUCTURAL BREAK DETECTOR · {len(E['breaks'])} BREAKS FLAGGED")


def mtf_panel(series: pd.Series, height: int = 200) -> go.Figure:
    fig = go.Figure()
    fig.add_hrect(y0=45, y1=105, fillcolor="rgba(255,93,108,.07)", line_width=0)
    fig.add_hrect(y0=-105, y1=-45, fillcolor="rgba(47,208,140,.07)", line_width=0)
    fig.add_trace(go.Scatter(x=series.index, y=series.values,
                             line=dict(color=CYAN, width=1.5), showlegend=False))
    fig.add_hline(y=0, line_color="rgba(140,165,215,.35)")
    fig.update_yaxes(range=[-105, 105])
    return style(fig, height, legend=False)


def equity_chart(bt: dict, height: int = 440) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32],
                        vertical_spacing=0.06)
    fig.add_trace(go.Scatter(x=bt["equity"].index, y=bt["equity"], name="FVO strategy",
                             line=dict(color=CYAN, width=2)), 1, 1)
    fig.add_trace(go.Scatter(x=bt["buyhold"].index, y=bt["buyhold"], name="buy & hold",
                             line=dict(color="rgba(232,238,252,.45)", width=1.3)), 1, 1)
    fig.add_trace(go.Scatter(x=bt["drawdown"].index, y=bt["drawdown"] * 100,
                             name="drawdown %", fill="tozeroy",
                             fillcolor="rgba(255,93,108,.18)",
                             line=dict(color=RED, width=1)), 2, 1)
    fig.update_yaxes(title_text="growth of 1", row=1, col=1)
    fig.update_yaxes(title_text="drawdown %", row=2, col=1)
    return style(fig, height)


def decile_chart(df: pd.DataFrame, horizon: int, height: int = 320) -> go.Figure:
    col = f"mean fwd {horizon}d %"
    fig = go.Figure(go.Bar(
        x=df["bucket"], y=df[col], marker_line=dict(width=0),
        marker_color=[GREEN if v > 0 else RED for v in df[col]],
        text=[f"{v:+.2f}%" for v in df[col]], textposition="outside",
        textfont=dict(family=MONO, size=9, color=MUTED)))
    fig.add_hline(y=0, line_color="rgba(140,165,215,.35)")
    fig.update_xaxes(title_text="FVO bucket   (1 = cheapest · 10 = richest)")
    fig.update_yaxes(title_text=f"mean {horizon}d forward return %")
    return style(fig, height, legend=False,
                 title="FORWARD RETURN BY OSCILLATOR BUCKET")


def event_study_chart(df: pd.DataFrame, height: int = 320) -> go.Figure:
    colours = {"undervalued": GREEN, "overvalued": RED,
               "bullish": CYAN, "bearish": VIOLET}
    fig = go.Figure()
    for col in df.columns:
        key = next((k for k in colours if col.startswith(k)), None)
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col,
                                 line=dict(color=colours.get(key, MUTED), width=1.8)))
    fig.add_hline(y=0, line_color="rgba(140,165,215,.35)")
    fig.update_xaxes(title_text="sessions after signal")
    fig.update_yaxes(title_text="mean cumulative return %")
    return style(fig, height, title="EVENT STUDY · AVERAGE PATH AFTER EACH SIGNAL")


def factor_loadings_chart(loadings: pd.Series, title: str,
                          height: int = 360) -> go.Figure:
    s = loadings.iloc[::-1]
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h",
                           marker_color=[color_of(str(i)) for i in s.index],
                           marker_line=dict(width=0)))
    fig.add_vline(x=0, line_color="rgba(140,165,215,.3)")
    return style(fig, height, legend=False, title=title)


def market_state_chart(state: pd.DataFrame, height: int = 320) -> go.Figure:
    fig = go.Figure()
    palette = [CYAN, AMBER, VIOLET, GREEN, BLUE, RED]
    for i, col in enumerate(state.columns):
        s = state[col]
        norm = (s - s.rolling(504, min_periods=60).mean()) / \
               s.rolling(504, min_periods=60).std().replace(0, np.nan)
        fig.add_trace(go.Scatter(x=state.index, y=norm.fillna(0), name=col,
                                 line=dict(color=palette[i % len(palette)], width=1.2)))
    return style(fig, height, title="MARKET STATE VECTOR · STANDARDISED")
