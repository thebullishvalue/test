"""
Arthagati v2.7.0 — Shared CSS, chart theming, and colour constants.
अर्थगति (Arthagati) — "Market sentiment / movement of meaning"

UI thesis: "Obsidian Quant" Institutional Research Terminal.
- Display/UI:  Space Grotesk (geometric, authoritative)
- Body/Data:   JetBrains Mono / IBM Plex Mono (tabular precision)
- Palette:     Obsidian (#0A0E17 -> #050810) backgrounds,
               Amber Gold (#D4A853), Cyan, Emerald, Rose accents
- Surfaces:    Frameless glass panels, thin border strokes
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

VERSION = "v2.7.0"
PRODUCT_NAME = "Arthagati"
COMPANY = "@thebullishvalue"

# ── Obsidian Quant colour tokens (mirror :root in theme.css) ────────────────
C_AMBER         = "#D4A853"
C_AMBER_BRIGHT  = "#E8C478"
C_AMBER_DIM     = "rgba(212, 168, 83, 0.6)"
C_AMBER_GLOW    = "rgba(212, 168, 83, 0.25)"
C_CYAN          = "#06B6D4"
C_EMERALD       = "#2DD4A8"
C_EMERALD_BRIGHT = "#6EE7C8"
C_ROSE          = "#E8555A"
C_ROSE_BRIGHT   = "#F07075"
C_VIOLET        = "#8B5CF6"
C_ORANGE        = "#F59E0B"
C_SLATE_WARM    = "#8B7E6A"

# Semantic shortcuts used throughout the engine code paths
C_PRIMARY = C_AMBER
C_GREEN   = C_EMERALD
C_RED     = C_ROSE
C_MUTED   = "#4B5563"
C_TEXT    = "#F1F5F9"
C_BG_DEEP = "#050810"
C_BG_BASE = "#0A0E17"
C_BG_CARD = "#111827"
C_BG_GRID = "rgba(255,255,255,0.035)"

# Path to external CSS file
CSS_PATH = Path(__file__).parent / "theme.css"

# ── Shared Plotly layout configuration ───────────────────────────────────────
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

# Shared base layout — paper/plot backgrounds are transparent so the page's
# glass containers show through.
PLOTLY_BASE: dict = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=PLOTLY_FONT,
    hoverlabel=PLOTLY_HOVERLABEL,
)


def chart_layout(
    height: int = 360,
    show_legend: bool = True,
    margin: dict | None = None,
    responsive: bool = False,
) -> dict:
    """Return a base Plotly layout dict for the Obsidian Quant theme."""
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


def inject_css() -> None:
    """Inject the Obsidian Quant Terminal CSS into the Streamlit app."""
    if CSS_PATH.exists():
        css = CSS_PATH.read_text()
    else:
        css = "/* theme.css not found */"
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def progress_bar(slot, pct: int, label: str, sub: str = "") -> None:
    """Render a themed progress card into an ``st.empty()`` slot."""
    is_complete = pct >= 100
    bar_color = C_EMERALD if is_complete else C_AMBER if pct > 50 else C_CYAN
    dot_class = "pulse-dot complete" if is_complete else "pulse-dot"
    sub_html = f'<div class="progress-sub">{sub}</div>' if sub else ""
    slot.markdown(
        f"""
    <div class="progress-card">
        <div class="progress-label">
            <span class="{dot_class}"></span>{label}
        </div>
        {sub_html}
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
