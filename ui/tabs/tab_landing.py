"""
Arthagati landing page — three system cards + methodology + awaiting-data prompt.
Mirrors Nishkarsh's tab_landing structure & visual fidelity.
"""

from __future__ import annotations

import streamlit as st

from ui.components import (
    render_header,
    render_section_header,
    render_system_card,
    render_metric_card,
    render_landing_prompt,
    render_interpretation_card,
    section_gap,
)


def render_landing_page(version: str, n_predictors: int) -> None:
    """Informational landing page shown before analysis starts."""

    # ── Masthead ────────────────────────────────────────────────────
    render_header(
        title="Arthagati",
        tagline="Ornstein-Uhlenbeck  ·  Kalman  ·  Decay-Spearman  ·  Adaptive Percentiles  |  Quantitative Market Physics",
    )

    section_gap()

    # ── Three system feature cards ──────────────────────────────────
    col1, col2, col3 = st.columns(3, gap="small")
    with col1:
        render_system_card(
            title="Historical Mood",
            description=(
                "Full sentiment timeline with OU forward projection, Kalman confidence "
                "bands, and regime transition markers on a TradingView-style chart."
            ),
            specs=[
                ("Range:", "Mood Score −100 → +100"),
                ("Confirmation:", "MSF Spread oscillator"),
                ("Projection:", "90-day OU mean-reversion"),
            ],
            card_class="mood",
            icon="chart",
        )
    with col2:
        render_system_card(
            title="Similar Periods",
            description=(
                "Historical analog matching against the full dataset with forward-return "
                "outcomes, aggregate win-rates, and a backtest scatter."
            ),
            specs=[
                ("Distance:", "Mahalanobis (55%)"),
                ("Shape:", "Trajectory cosine (35%)"),
                ("Recency:", "Decay weight (10%)"),
            ],
            card_class="similar",
            icon="search",
        )
    with col3:
        render_system_card(
            title="Correlation Analysis",
            description=(
                "Full transparency into which variables drive the mood score and which "
                "are noise, ranked by the engine's own quality formula."
            ),
            specs=[
                ("Anchors:", "PE  &  Earnings Yield"),
                ("Method:", "Decay-Spearman + Entropy"),
                ("Output:", "Keep / Useful / Weak"),
            ],
            card_class="corr",
            icon="file-text",
        )

    section_gap()

    # ── Methodology — three coloured interpretation cards ──────────
    render_section_header(
        title="Analysis Methodology",
        description="Physics-informed scoring pipeline · confirmation · regime detection",
        icon="cpu",
    )

    m1, m2, m3 = st.columns(3, gap="small")
    with m1:
        render_interpretation_card(
            title="Mood Engine — 5 Layers",
            body=(
                "<ul style='margin:0; padding-left:1.1rem; line-height:1.8;'>"
                "<li><strong>Decay-Spearman</strong> correlations (504d half-life)</li>"
                "<li><strong>Entropy weighting</strong> — noisy variables suppressed</li>"
                "<li><strong>Adaptive percentiles</strong> — decay-weighted CDF</li>"
                "<li><strong>OU normalisation</strong> → [−100, +100]</li>"
                "<li><strong>Kalman smoothing</strong> + ±1.96σ band</li>"
                "</ul>"
            ),
            color="success",
        )
    with m2:
        render_interpretation_card(
            title="MSF Spread — Confirmation",
            body=(
                "<ul style='margin:0; padding-left:1.1rem; line-height:1.8;'>"
                "<li><strong>Momentum</strong> — NIFTY ROC z-score (14d)</li>"
                "<li><strong>Structure</strong> — mood trend divergence</li>"
                "<li><strong>Flow</strong> — breadth participation</li>"
                "<li><strong>Regime</strong> — adaptive directional count</li>"
                "<li><strong>Weights</strong> — inverse-variance (Markowitz)</li>"
                "</ul>"
            ),
            color="info",
        )
    with m3:
        render_interpretation_card(
            title="Regime Detection",
            body=(
                "<ul style='margin:0; padding-left:1.1rem; line-height:1.8;'>"
                "<li><strong>Trending</strong> — momentum strategies favoured</li>"
                "<li><strong>Volatile Trend</strong> — directional with swings</li>"
                "<li><strong>Mean-Reverting</strong> — contrarian strategies</li>"
                "<li><strong>Choppy</strong> — reduce size, avoid</li>"
                "<li><strong>Output</strong> — scales MSF weights + OU horizon</li>"
                "</ul>"
            ),
            color="warning",
        )

    section_gap()

    # ── Mood score interpretation zones ─────────────────────────────
    render_section_header(
        title="Mood Score Interpretation",
        description="Score thresholds and operating regimes",
        icon="target",
        accent="cyan",
    )

    z1, z2, z3 = st.columns(3, gap="small")
    with z1:
        render_interpretation_card(
            title="Bullish Zone (> +20)",
            body=(
                "Positive sentiment. Trend-following strategies favoured. "
                "At extremes (&gt; +60, <strong>Euphoric</strong>) mean-reversion risk rises sharply."
            ),
            color="success",
        )
    with z2:
        render_interpretation_card(
            title="Neutral Zone (−20 to +20)",
            body=(
                "No strong directional bias. Await macro confirmation or use "
                "MSF Spread and Similar Periods for additional context."
            ),
            color="info",
        )
    with z3:
        render_interpretation_card(
            title="Bearish Zone (&lt; −20)",
            body=(
                "Negative sentiment. Defensive positioning warranted. "
                "At extremes (&lt; −60, <strong>Capitulation</strong>) contrarian signals may emerge."
            ),
            color="danger",
        )

    section_gap()

    # ── System coverage strip ───────────────────────────────────────
    render_section_header(
        title="System Coverage",
        description="Anchors · predictors · mathematical primitives",
        icon="layers",
        accent="violet",
    )

    c1, c2, c3, c4, c5 = st.columns(5, gap="small")
    with c1:
        render_metric_card("Score Anchors", "2", "PE · Earnings Yield", color_class="neutral")
    with c2:
        render_metric_card("Predictors", f"{n_predictors}", "Macro + breadth vars", color_class="neutral")
    with c3:
        render_metric_card("Math Primitives", "12", "Pure NumPy functions", color_class="neutral")
    with c4:
        render_metric_card("OU Projection", "90d", "Forward reversion path", color_class="neutral")
    with c5:
        render_metric_card("Analog Returns", "3", "30 · 60 · 90 day", color_class="neutral")

    section_gap()

    # ── Awaiting-data prompt ────────────────────────────────────────
    render_landing_prompt(
        title="Awaiting Run",
        body_html=(
            "Click <strong>Run Analysis</strong> in the sidebar to fetch live data from Google Sheets "
            "and execute the full 5-layer sentiment pipeline. Once loaded, switch between "
            "<strong>Historical Mood</strong>, <strong>Similar Periods</strong>, and "
            "<strong>Correlation Analysis</strong> views — or tune the active predictor set in "
            "<strong>Model Configuration</strong>."
        ),
    )
