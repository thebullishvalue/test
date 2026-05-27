"""
Arthagati v2.7.0 — Reusable UI components.
अर्थगति (Arthagati) — "Market sentiment / movement of meaning"

UI thesis: Obsidian Quant Institutional Research Terminal.
"""

from __future__ import annotations

import html as html_mod
import re

import streamlit as st


# ── SVG Icons (inline, no external deps) ────────────────────────────────────

ICONS = {
    "chart":      '<svg aria-label="Chart icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
    "cube":       '<svg aria-label="Cube icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/></svg>',
    "target":     '<svg aria-label="Target icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
    "layers":     '<svg aria-label="Layers icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>',
    "bar-chart":  '<svg aria-label="Bar chart icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>',
    "activity":   '<svg aria-label="Activity icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
    "crosshair":  '<svg aria-label="Crosshair icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><line x1="22" y1="12" x2="18" y2="12"/><line x1="6" y1="12" x2="2" y2="12"/><line x1="12" y1="6" x2="12" y2="2"/><line x1="12" y1="22" x2="12" y2="18"/></svg>',
    "cpu":        '<svg aria-label="CPU icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>',
    "zap":        '<svg aria-label="Zap icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>',
    "shield":     '<svg aria-label="Shield icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>',
    "grid":       '<svg aria-label="Grid icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>',
    "database":   '<svg aria-label="Database icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>',
    "trending":   '<svg aria-label="Trending icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>',
    "eye":        '<svg aria-label="Eye icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>',
    "play":       '<svg aria-label="Play icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8"/></svg>',
    "search":     '<svg aria-label="Search icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>',
    "file-text":  '<svg aria-label="File icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/><path d="M16 13H8"/><path d="M16 17H8"/><path d="M10 9H8"/></svg>',
    "settings":   '<svg aria-label="Settings icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>',
    "compass":    '<svg aria-label="Compass icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/></svg>',
    "arrow-up":   '<svg aria-label="Up" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="19" x2="12" y2="5"/><polyline points="5 12 12 5 19 12"/></svg>',
    "arrow-down": '<svg aria-label="Down" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"/><polyline points="19 12 12 19 5 12"/></svg>',
    "alert-triangle": '<svg aria-label="Alert icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    "circle":     '<svg aria-label="Circle" role="img" viewBox="0 0 24 24" fill="currentColor" stroke="none"><circle cx="12" cy="12" r="10"/></svg>',
    "check-circle": '<svg aria-label="Check" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
    "minus-circle": '<svg aria-label="Minus" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="8" y1="12" x2="16" y2="12"/></svg>',
    "trending-up":   '<svg aria-label="Bull" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg>',
    "trending-down": '<svg aria-label="Bear" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 17 13.5 8.5 8.5 13.5 2 7"/><polyline points="16 17 22 17 22 11"/></svg>',
    "globe":      '<svg aria-label="Globe" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>',
    "play-circle": '<svg aria-label="Run" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8"/></svg>',
}


def get_icon(name: str, size: int = 18, stroke_width: float = 1.5) -> str:
    """Return an SVG icon string with custom size and stroke width."""
    base_svg = ICONS.get(name, ICONS["chart"])
    base_svg = re.sub(r'\s+width="[^"]*"', "", base_svg)
    base_svg = re.sub(r'\s+height="[^"]*"', "", base_svg)
    base_svg = re.sub(r'\s+stroke-width="[^"]*"', "", base_svg)
    return base_svg.replace("<svg", f'<svg width="{size}" height="{size}" stroke-width="{stroke_width}"')


# ── Page elements ───────────────────────────────────────────────────────────

def render_header(title: str, tagline: str) -> None:
    """Render the terminal masthead — title with अर्थगति serif overlay."""
    st.markdown(
        f'<div class="premium-header">'
        f"<h1>{html_mod.escape(title)}</h1>"
        f'<div class="tagline">{html_mod.escape(tagline)}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_section_header(
    title: str,
    description: str = "",
    icon: str = "chart",
    accent: str = "",
) -> None:
    """Render a section header with icon badge, title, and optional description.

    accent: ``""`` (amber, default), ``"cyan"``, ``"emerald"``, ``"violet"``, ``"rose"``.
    """
    svg = get_icon(icon, size=16, stroke_width=1.8)
    icon_class = f"icon {accent}" if accent else "icon"
    hdr_class = f"section-hdr {accent}" if accent else "section-hdr"
    desc_html = f'<div class="desc">{html_mod.escape(description)}</div>' if description else ""
    st.markdown(
        f'<div class="{hdr_class}">'
        f'<div class="{icon_class}">{svg}</div>'
        f'<div class="text">'
        f'<h3>{html_mod.escape(title)}</h3>'
        f"{desc_html}"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def section_gap() -> None:
    """Insert vertical spacing between major sections."""
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)


def section_divider() -> None:
    """Insert a thin gradient divider line."""
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ── Cards ───────────────────────────────────────────────────────────────────

def render_metric_card(
    label: str,
    value: str,
    subtext: str = "",
    color_class: str = "neutral",
    tooltip: str = "",
    icon: str = "",
) -> None:
    """Render a terminal-styled metric card.

    color_class: ``"neutral"`` | ``"success"`` | ``"danger"`` | ``"warning"`` | ``"info"`` | ``"violet"``.
    """
    tooltip_html = ""
    if tooltip:
        tooltip_html = (
            f'<div class="metric-tooltip" data-tooltip="{html_mod.escape(tooltip)}">'
            f'<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">'
            f'<circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>'
            f'<line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
            f'<span class="metric-tooltip-text">{html_mod.escape(tooltip)}</span>'
            f"</div>"
        )
    sub_html = f'<div class="sub-metric">{html_mod.escape(subtext)}</div>' if subtext else ""
    icon_html = f'<span class="card-icon">{get_icon(icon, size=12, stroke_width=2)}</span> ' if icon else ""
    st.markdown(
        f'<div class="metric-card {html_mod.escape(color_class)}">'
        f"<h4>{icon_html}{html_mod.escape(label)}</h4>"
        f"<h2>{html_mod.escape(value)}</h2>"
        f"{sub_html}"
        f"{tooltip_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_info_box(title: str, content: str) -> None:
    """Render a cyan-accented info box."""
    st.markdown(
        f'<div class="info-box">'
        f"<h4>{html_mod.escape(title)}</h4>"
        f"<p>{html_mod.escape(content)}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_warning_box(title: str, content: str) -> None:
    """Render an amber-accented warning box."""
    st.markdown(
        f'<div class="warning-box">'
        f'<div class="icon"></div>'
        f"<div>"
        f'<div class="title">{html_mod.escape(title)}</div>'
        f'<div class="content">{html_mod.escape(content)}</div>'
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_interpretation_card(title: str, body: str, color: str = "neutral") -> None:
    """Render a state-aware interpretation card.

    color: ``"neutral"`` | ``"success"`` | ``"danger"`` | ``"warning"`` | ``"info"``.
    Body is raw HTML — caller is trusted.
    """
    st.markdown(
        f'<div class="interp-card {html_mod.escape(color)}">'
        f'<div class="interp-title">{html_mod.escape(title)}</div>'
        f'<div class="interp-body">{body}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_system_card(
    title: str,
    description: str,
    specs: list[tuple[str, str]],
    card_class: str = "mood",
    icon: str = "chart",
) -> None:
    """Render a landing-page feature card.

    card_class: ``"mood"`` | ``"similar"`` | ``"corr"`` (Arthagati variants).
    """
    spec_html = "".join(
        f"<span>{html_mod.escape(label)}</span> {html_mod.escape(value)}<br>"
        for label, value in specs
    )
    svg = get_icon(icon, size=18, stroke_width=1.8)
    st.markdown(
        f"""
        <div class='system-card {html_mod.escape(card_class)}'>
            <h3>
                {svg}
                {html_mod.escape(title)}
            </h3>
            <p>{html_mod.escape(description)}</p>
            <div class='spec'>{spec_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_landing_prompt(title: str, body_html: str) -> None:
    """Render the highlighted 'awaiting action' prompt on the landing page."""
    st.markdown(
        f"""
        <div class='landing-prompt'>
            <h4>
                {get_icon('play', size=14, stroke_width=2)}
                {html_mod.escape(title)}
            </h4>
            <p>{body_html}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Sidebar helpers ─────────────────────────────────────────────────────────

def sidebar_title(text: str, icon: str = "") -> None:
    """Render a sidebar section heading."""
    icon_html = f'<span class="card-icon" style="margin-right:0.4rem;">{get_icon(icon, size=12, stroke_width=2)}</span>' if icon else ""
    st.markdown(
        f'<div class="sidebar-title">{icon_html}{html_mod.escape(text)}</div>',
        unsafe_allow_html=True,
    )


def sidebar_masthead(product: str, sanskrit: str, subtitle: str) -> None:
    """Render the sidebar masthead — product name + Sanskrit + subtitle."""
    st.markdown(
        f"""
        <div style="text-align:center; padding:1rem 0 0.75rem 0;">
            <div style="font-family:var(--display); font-size:1.4rem; font-weight:800;
                        letter-spacing:0.06em; color:var(--amber); text-shadow:0 0 18px var(--amber-glow);">
                {html_mod.escape(product)}
            </div>
            <div style="font-family:var(--display-serif); font-size:0.95rem; color:var(--amber-dim);
                        margin-top:0.2rem; letter-spacing:0.04em;">
                {html_mod.escape(sanskrit)}
            </div>
            <div style="font-family:var(--data); font-size:0.6rem; color:var(--ink-tertiary);
                        margin-top:0.35rem; text-transform:uppercase; letter-spacing:0.18em;">
                {html_mod.escape(subtitle)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_passport(version: str, engine: str, data_label: str) -> None:
    """Render a small terminal-style 'system spec' card in the sidebar footer."""
    st.markdown(
        f"""
        <div class='system-spec'>
            <div class='spec-row'>
                <span class='spec-label'>Version</span>
                <span class='spec-value'>{html_mod.escape(version)}</span>
            </div>
            <div class='spec-row'>
                <span class='spec-label'>Engine</span>
                <span class='spec-value'>{html_mod.escape(engine)}</span>
            </div>
            <div class='spec-row'>
                <span class='spec-label'>Data</span>
                <span class='spec-value'>{html_mod.escape(data_label)}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Footer ──────────────────────────────────────────────────────────────────

def render_footer(product: str, company: str, version: str, timestamp: str) -> None:
    """Render the app footer."""
    st.markdown(
        f"""
        <div class='app-footer'>
            <div class='content'>
                © 2026 <strong>{html_mod.escape(product)}</strong> &nbsp;·&nbsp;
                {html_mod.escape(company)} &nbsp;·&nbsp;
                <strong>{html_mod.escape(version)}</strong> &nbsp;·&nbsp;
                {html_mod.escape(timestamp)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
