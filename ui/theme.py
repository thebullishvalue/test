"""
Pragyam v3.5.0 — Shared CSS, chart theming, and color constants for the UI layer.

UI — "Obsidian Quant" Institutional Research Terminal design language.

Aesthetic: "Obsidian Quant" — Institutional Research Terminal
Precision-instrument design language for quantitative finance.
- Display/UI:  Syne (geometric, authoritative, distinctive)
- Body/Data:   JetBrains Mono (refined monospace, tabular precision)
- Palette:     Obsidian (#0A0E17 -> #050810), Amber Gold (#D4A853)
- Surfaces:    Frameless glass panels with thin border strokes.
"""

from __future__ import annotations

import html
import time
from pathlib import Path

import streamlit as st

VERSION = "v3.5.0"
PRODUCT_NAME = "Sanket"
COMPANY = "@thebullishvalue"

# Path to external CSS file
CSS_PATH = Path(__file__).parent / "theme.css"

# ── Shared Plotly layout config ─────────────────────────────────────────────
# Eliminates massive duplication across all tab files.

PLOTLY_FONT = dict(family="JetBrains Mono, monospace", color="#94A3B8", size=10)
PLOTLY_HOVERLABEL = dict(
    bgcolor="rgba(10, 14, 23, 0.95)",
    font=dict(family="JetBrains Mono, monospace", size=11, color="#F1F5F9"),
    bordercolor="rgba(255,255,255,0.08)",
    align="left",
)
PLOTLY_LEGEND = dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1,
    font=dict(size=10, family="JetBrains Mono, monospace"),
    bgcolor="rgba(0,0,0,0)",
)
PLOTLY_MARGIN = dict(t=20, l=50, r=20, b=40)
PLOTLY_GRID = "rgba(255,255,255,0.035)"
PLOTLY_GRID_ZERO = "rgba(255,255,255,0.06)"

# Interactive chart config — click + zoom + pan
PLOTLY_MODEBAR = dict(
    modeBarButtonsToRemove=["lasso2d", "select2d"],
    modeBarButtonsToAdd=[
        "drawline",
        "eraseshape",
    ],
    displaylogo=False,
)


def chart_layout(
    height: int = 360,
    show_legend: bool = True,
    margin: dict | None = None,
    responsive: bool = False,
) -> dict:
    """Return a base Plotly layout dict for the Obsidian Quant theme.

    Args:
        height: Fixed pixel height for the chart.
        show_legend: Whether to show the legend.
        margin: Custom margin dict.
        responsive: If True, adds CSS-based responsive sizing via autosize.
    """
    base = dict(
        height=height,
        showlegend=show_legend,
        legend=PLOTLY_LEGEND if show_legend else None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=PLOTLY_FONT,
        hovermode="x unified",
        hoverlabel=PLOTLY_HOVERLABEL,
        margin=margin or PLOTLY_MARGIN,
        spikedistance=-1,
    )
    if responsive:
        base["autosize"] = True
    return base


def style_axes(fig, y_title: str = "", x_title: str = "", y_range=None, row=None, col=None) -> None:
    """Apply consistent axis styling to a Plotly figure."""
    kw = {}
    if row is not None:
        kw["row"] = row
    if col is not None:
        kw["col"] = col

    fig.update_xaxes(
        showgrid=True,
        gridcolor=PLOTLY_GRID,
        gridwidth=0.5,
        zeroline=False,
        linecolor="rgba(255,255,255,0.04)",
        title_text=x_title,
        tickfont=dict(size=9, family="JetBrains Mono, monospace", color="#64748B"),
        # Vertical crosshair — dashed dim grey
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikethickness=0.5,
        spikedash="dash",
        spikecolor="rgba(148,163,184,0.18)",
        **kw,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=PLOTLY_GRID,
        gridwidth=0.5,
        zeroline=True,
        zerolinecolor=PLOTLY_GRID_ZERO,
        zerolinewidth=0.5,
        linecolor="rgba(255,255,255,0.04)",
        title_text=y_title,
        range=y_range,
        tickfont=dict(size=9, family="JetBrains Mono, monospace", color="#64748B"),
        hoverformat=".2f",
        **kw,
    )


@st.cache_resource
def _load_theme_css() -> str:
    """Read theme.css once per process; cached so reruns skip the file read."""
    if CSS_PATH.exists():
        return CSS_PATH.read_text()
    return "/* theme.css not found */"


def inject_css() -> None:
    """Inject the Obsidian Quant Terminal CSS into the Streamlit app.

    CSS is read once per process (cached) and deduped by Streamlit, so
    repeated reruns pay zero I/O and zero DOM patching cost.

    NOTE: do NOT guard this behind a "once per session" flag. Streamlit rebuilds
    the element tree every rerun; an st.markdown <style> block only persists if it
    is re-emitted each run. Skipping re-injection removes the style and unstyles
    the app on the next interaction. The cost here is already minimal — _load_theme_css
    is @st.cache_resource (zero re-read) and Streamlit dedups the identical element.
    """
    st.markdown(f"<style>{_load_theme_css()}</style>", unsafe_allow_html=True)


_PROGRESS_THROTTLE: dict = {}   # id(slot) -> (last_pct, last_time)
_PROGRESS_MIN_INTERVAL = 0.25   # seconds between re-renders of the same slot


def progress_bar(slot, pct: int, label: str, sub: str = "") -> None:
    """Render a themed progress card into an ``st.empty()`` slot.

    Throttled: inside a tight per-stock loop the bar can be asked to redraw
    hundreds of times a second, and each redraw is a websocket round-trip. Skip a
    redraw when the integer pct hasn't advanced AND <0.25s has elapsed for this
    slot. Completion (pct>=100) and the first draw always render, so the user
    never sees a stalled or unfinished bar — only the redundant intermediate
    frames that would look identical are dropped.
    """
    pct = int(pct)
    _now = time.monotonic()
    _key = id(slot)
    _last = _PROGRESS_THROTTLE.get(_key)
    if _last is not None and pct < 100:
        _last_pct, _last_t = _last
        # Suppress only a same-pct redraw within the interval. A pct that moved
        # (up = progress, or down = a new/reused slot starting over) always renders.
        if pct == _last_pct and (_now - _last_t) < _PROGRESS_MIN_INTERVAL:
            return
    _PROGRESS_THROTTLE[_key] = (pct, _now)
    if pct >= 100:
        _PROGRESS_THROTTLE.pop(_key, None)   # reset so a reused slot starts clean

    is_complete = pct >= 100
    bar_color = "#34D399" if is_complete else "#D4A853" if pct > 50 else "#22D3EE"
    dot_class = "pulse-dot complete" if is_complete else "pulse-dot"
    slot.markdown(
        f"""
    <div class="progress-card">
        <div class="progress-label">
            <span class="{dot_class}"></span>{html.escape(label)}
        </div>
        {f'<div class="progress-sub">{html.escape(sub)}</div>' if sub else ''}
        <div class="progress-track">
            <div class="progress-fill" style="width:{pct}%;background:{bar_color};box-shadow:0 0 10px {bar_color};"></div>
        </div>
        <div class="progress-pct">{pct}%</div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def apply_chart_theme(fig) -> None:
    """Apply the Obsidian Quant Terminal theme to a Plotly figure (mutates in place)."""
    fig.update_layout(**chart_layout())
    style_axes(fig)
