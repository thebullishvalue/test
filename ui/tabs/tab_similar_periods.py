"""
Arthagati — Similar Periods view (analog matching + forward returns + backtest).
"""

from __future__ import annotations

import html as html_mod

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ui.components import (
    render_section_header,
    render_metric_card,
    render_interpretation_card,
    render_warning_box,
    section_divider,
    get_icon,
)
from ui.theme import (
    C_AMBER,
    C_CYAN,
    C_EMERALD,
    C_ROSE,
    C_MUTED,
    PLOTLY_BASE,
    PLOTLY_GRID,
    PLOTLY_GRID_ZERO,
)


def _render_period_card(period: dict) -> None:
    """Render one analog-period card in the unified glass / corner-dot system."""
    mood_val = period["mood_score"]
    if mood_val > 20:
        sig_class, sig_label = "buy", "Bullish"
    elif mood_val < -20:
        sig_class, sig_label = "sell", "Bearish"
    else:
        sig_class, sig_label = "hold", "Neutral"

    fwd_html = ""
    for horizon, key in [(30, "fwd_30d"), (60, "fwd_60d"), (90, "fwd_90d")]:
        val = period.get(key)
        if val is None:
            fwd_html += (
                f'<div class="position-signal">'
                f'<span class="position-signal-label">+{horizon}D</span>'
                f'<span class="position-signal-value">—</span>'
                f"</div>"
            )
        else:
            cls = "bullish" if val > 0 else "bearish"
            fwd_html += (
                f'<div class="position-signal">'
                f'<span class="position-signal-label">+{horizon}D</span>'
                f'<span class="position-signal-value {cls}">{val:+.1f}%</span>'
                f"</div>"
            )

    similarity_pct = period["similarity"] * 100
    tier_class = f"tier-{'buy' if sig_class == 'buy' else 'caution' if sig_class == 'sell' else 'hold'}"

    st.markdown(
        f"""
        <div class="position-card {tier_class}">
            <div class="position-card-header">
                <div class="position-card-symbol">{html_mod.escape(period['date'])}</div>
                <div class="position-card-conviction">
                    <span class="position-card-score">{mood_val:+.1f}</span>
                    <span class="position-card-badge badge-{sig_class}">{sig_label}</span>
                </div>
            </div>
            <div class="position-card-signals">
                <div class="position-signal">
                    <span class="position-signal-label">Similarity</span>
                    <span class="position-signal-value" style="color:var(--amber-bright);">{similarity_pct:.1f}%</span>
                </div>
                <div class="position-signal">
                    <span class="position-signal-label">NIFTY</span>
                    <span class="position-signal-value">{period['nifty']:,.0f}</span>
                </div>
                {fwd_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render(mood_df, *, find_similar_periods, backtest_horizon) -> None:
    """Render Similar Periods view — analog cards + forward-return summary + backtest."""

    render_section_header(
        title="Similar Historical Periods",
        description="Mahalanobis + trajectory matching · forward NIFTY returns from each analog",
        icon="search",
        accent="emerald",
    )

    similar_periods = find_similar_periods(mood_df)
    if not similar_periods:
        st.warning("Not enough historical data to find similar periods.")
        return

    # ── Forward return summary cards ─────────────────────────────────────
    fwd_30 = [p["fwd_30d"] for p in similar_periods if p["fwd_30d"] is not None]
    fwd_60 = [p["fwd_60d"] for p in similar_periods if p["fwd_60d"] is not None]
    fwd_90 = [p["fwd_90d"] for p in similar_periods if p["fwd_90d"] is not None]

    if fwd_30 or fwd_60 or fwd_90:
        cols = st.columns(3, gap="small")
        for col, horizon, values in zip(cols, [30, 60, 90], [fwd_30, fwd_60, fwd_90]):
            if not values:
                continue
            median_ret = np.median(values)
            positive_pct = sum(1 for v in values if v > 0) / len(values) * 100
            with col:
                render_metric_card(
                    label=f"+{horizon}D Median Return",
                    value=f"{median_ret:+.1f}%",
                    subtext=f"{positive_pct:.0f}% positive ({len(values)} analogs)",
                    color_class="success" if median_ret > 0 else "danger",
                    icon="trending-up" if median_ret > 0 else "trending-down",
                )

    section_divider()

    # ── Analog period cards ──────────────────────────────────────────────
    render_section_header(
        title="Top Analog Periods",
        description="Top 10 historical matches by similarity score",
        icon="layers",
    )

    st.markdown('<div class="position-guide-list">', unsafe_allow_html=True)
    for period in similar_periods[:10]:
        _render_period_card(period)
    st.markdown("</div>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════
    # BACKTEST SANITY CHECK
    # ═══════════════════════════════════════════════════════════════════════
    section_divider()
    render_section_header(
        title="Backtest · Mood Score vs Forward NIFTY Return",
        description="Each dot = one historical day · pattern indicates predictive relationship",
        icon="chart",
        accent="rose",
    )

    render_warning_box(
        title="Hindsight Regime Fit",
        content=(
            "Historical points are evaluated using parameters learned from today's active "
            "correlation regime — treat as descriptive, not predictive."
        ),
    )

    n = len(mood_df)
    horizon = backtest_horizon
    if n <= horizon + 10:
        st.caption("Insufficient data points for backtest.")
        return

    bt_mood  = mood_df["Mood_Score"].values[: n - horizon]
    bt_nifty = mood_df["NIFTY"].values
    bt_fwd   = (bt_nifty[horizon:] / bt_nifty[: n - horizon] - 1) * 100

    valid = np.isfinite(bt_mood) & np.isfinite(bt_fwd)
    bt_mood_clean = bt_mood[valid]
    bt_fwd_clean  = bt_fwd[valid]

    if len(bt_mood_clean) <= 20:
        st.caption("Insufficient data points for backtest.")
        return

    from scipy.stats import spearmanr as _spearmanr

    split_idx = int(len(bt_mood_clean) * 0.7)
    train_m, train_r = bt_mood_clean[:split_idx], bt_fwd_clean[:split_idx]
    test_m,  test_r  = bt_mood_clean[split_idx:], bt_fwd_clean[split_idx:]

    bt_pearson  = np.corrcoef(train_m, train_r)[0, 1] if len(train_m) > 2 else 0
    bt_spearman, _ = _spearmanr(train_m, train_r)
    if not np.isfinite(bt_spearman):
        bt_spearman = 0.0
    oos_pearson = np.corrcoef(test_m, test_r)[0, 1] if len(test_m) > 2 else 0
    oos_spearman, _ = _spearmanr(test_m, test_r) if len(test_m) > 2 else (0.0, 1.0)
    if not np.isfinite(oos_spearman):
        oos_spearman = 0.0

    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scattergl(
        x=train_m, y=train_r, mode="markers",
        marker=dict(size=4, color=np.where(train_m > 0, C_EMERALD, C_ROSE), opacity=0.4),
        hovertemplate="Mood: %{x:.1f}<br>+30d Return: %{y:.1f}%<extra></extra>",
        name=f"Train (70%, n={len(train_m)})",
    ))
    fig_bt.add_trace(go.Scattergl(
        x=test_m, y=test_r, mode="markers",
        marker=dict(size=6, color=np.where(test_m > 0, C_EMERALD, C_ROSE),
                    opacity=0.85, symbol="diamond"),
        hovertemplate="Mood: %{x:.1f}<br>+30d Return: %{y:.1f}%<extra></extra>",
        name=f"Test (30%, n={len(test_m)})",
    ))

    if len(train_m) > 10:
        x_range = np.linspace(bt_mood_clean.min(), bt_mood_clean.max(), 50)
        z1 = np.polyfit(train_m, train_r, 1)
        fig_bt.add_trace(go.Scatter(
            x=x_range, y=z1[0] * x_range + z1[1],
            mode="lines", line=dict(color=C_AMBER, width=2, dash="dash"),
            name=f"Linear (train ρ={bt_pearson:.2f}, test ρ={oos_pearson:.2f})",
        ))
        z2 = np.polyfit(train_m, train_r, 2)
        fig_bt.add_trace(go.Scatter(
            x=x_range, y=z2[0] * x_range ** 2 + z2[1] * x_range + z2[2],
            mode="lines", line=dict(color=C_CYAN, width=2, dash="dot"),
            name=f"Quadratic (train ρ_s={bt_spearman:.2f}, test ρ_s={oos_spearman:.2f})",
        ))

    fig_bt.add_hline(y=0, line_color="rgba(148,163,184,0.35)", line_width=1, line_dash="dot")
    fig_bt.add_vline(x=0, line_color="rgba(148,163,184,0.35)", line_width=1, line_dash="dot")

    fig_bt.update_layout(
        **PLOTLY_BASE,
        height=420,
        hovermode="closest",
        showlegend=True,
        margin=dict(l=60, r=20, t=20, b=50),
        xaxis=dict(
            title=dict(text="Mood Score at T",
                       font=dict(size=11, color=C_MUTED, family="JetBrains Mono, monospace")),
            showgrid=True, gridcolor=PLOTLY_GRID, gridwidth=0.5,
            zeroline=True, zerolinecolor=PLOTLY_GRID_ZERO,
            tickfont=dict(size=9, family="JetBrains Mono, monospace", color="#64748B"),
        ),
        yaxis=dict(
            title=dict(text="NIFTY Return T+30d (%)",
                       font=dict(size=11, color=C_MUTED, family="JetBrains Mono, monospace")),
            showgrid=True, gridcolor=PLOTLY_GRID, gridwidth=0.5,
            zeroline=True, zerolinecolor=PLOTLY_GRID_ZERO,
            tickfont=dict(size=9, family="JetBrains Mono, monospace", color="#64748B"),
        ),
        legend=dict(
            x=0.02, y=0.98,
            bgcolor="rgba(10,14,23,0.85)",
            bordercolor="rgba(255,255,255,0.06)", borderwidth=1,
            font=dict(size=10, family="JetBrains Mono, monospace"),
        ),
    )

    st.markdown('<div class="chart-container similar">', unsafe_allow_html=True)
    st.plotly_chart(fig_bt, use_container_width=True, config={"displayModeBar": False, "displaylogo": False})
    st.markdown("</div>", unsafe_allow_html=True)

    # Interpretation — driven by OOS results
    oos_stronger = oos_spearman if abs(oos_spearman) > abs(oos_pearson) else oos_pearson
    if abs(oos_stronger) > 0.3:
        strength  = "strong" if abs(oos_stronger) > 0.5 else "moderate"
        direction = "positive" if oos_stronger > 0 else "negative"
        body = (
            f"<strong>Out-of-sample (30%):</strong> Pearson {oos_pearson:.2f} · Spearman {oos_spearman:.2f} — "
            f"{strength} {direction} relationship holds on unseen data.<br>"
            f"<span style='color:var(--ink-tertiary);'>In-sample (70%): Pearson {bt_pearson:.2f} · "
            f"Spearman {bt_spearman:.2f}</span><br><br>"
            + (
                "Higher mood scores have historically been followed by positive NIFTY returns."
                if oos_stronger > 0 else
                "Higher mood scores have historically been followed by negative NIFTY returns "
                "(contrarian signal)."
            )
        )
        render_interpretation_card("Predictive Relationship Holds", body, color="success")
    else:
        body = (
            f"<strong>Out-of-sample (30%):</strong> Pearson {oos_pearson:.2f} · Spearman {oos_spearman:.2f} — "
            f"weak out-of-sample relationship at 30-day horizon.<br>"
            f"<span style='color:var(--ink-tertiary);'>In-sample (70%): Pearson {bt_pearson:.2f} · "
            f"Spearman {bt_spearman:.2f}</span><br><br>"
            "The mood score's predictive power may be non-linear (check the quadratic curve) "
            "or work better at different horizons."
        )
        render_interpretation_card("Weak Out-of-Sample Fit", body, color="warning")
