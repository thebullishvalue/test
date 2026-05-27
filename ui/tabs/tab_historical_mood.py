"""
Arthagati — Historical Mood view (TradingView-style 2-pane chart).

Obsidian Quant fidelity: glass surfaces, JetBrains Mono ticks, dashed
spike crosshairs, transparent plot/paper backgrounds.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ui.components import (
    render_section_header,
    render_metric_card,
    section_divider,
)
from ui.theme import (
    C_AMBER,
    C_AMBER_BRIGHT,
    C_CYAN,
    C_EMERALD,
    C_ROSE,
    C_MUTED,
    PLOTLY_BASE,
    PLOTLY_GRID,
    PLOTLY_GRID_ZERO,
    PLOTLY_HOVERLABEL,
    PLOTLY_LEGEND,
)


def render(mood_df, msf_df, *, timeframes, regime_styles, mood_scale, ou_proj_days) -> None:
    """TradingView-style mood + MSF spread chart + period summary + MSF breakdown."""

    render_section_header(
        title="Market Mood Terminal",
        description="TradingView-style analysis · Mood Score + MSF Spread indicator",
        icon="activity",
    )

    # ── Timeframe selector (Google-Finance style row) ─────────────────────
    if "tf_selected" not in st.session_state:
        st.session_state.tf_selected = "1Y"

    tf_cols = st.columns(len(timeframes))
    for i, tf in enumerate(timeframes.keys()):
        with tf_cols[i]:
            btn_type = "primary" if st.session_state.tf_selected == tf else "secondary"
            if st.button(tf, key=f"tf_{tf}", use_container_width=True, type=btn_type):
                st.session_state.tf_selected = tf
                st.rerun()

    selected_tf = st.session_state.tf_selected
    if selected_tf == "YTD":
        today = datetime.now()
        days_back = (today - datetime(today.year, 1, 1)).days + 1
    else:
        days_back = timeframes[selected_tf]

    if days_back and days_back < len(mood_df):
        df = mood_df.tail(days_back).copy()
        msf_filtered = msf_df.tail(days_back).copy()
    else:
        df = mood_df.copy()
        msf_filtered = msf_df.copy()

    if df.empty:
        st.warning("No data available for selected timeframe.")
        return

    # ═══════════════════════════════════════════════════════════════════════
    # CHART — Mood Score (row 1) + MSF Spread (row 2)
    # ═══════════════════════════════════════════════════════════════════════
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.08, row_heights=[0.65, 0.35],
    )

    # Kalman confidence band
    if "Confidence_Upper" in df.columns and "Confidence_Lower" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["DATE"], y=df["Confidence_Upper"],
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df["DATE"], y=df["Confidence_Lower"],
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
            fill="tonexty", fillcolor="rgba(212,168,83,0.10)",
            name="95% Confidence",
        ), row=1, col=1)

    # Mood Score line
    fig.add_trace(go.Scattergl(
        x=df["DATE"], y=df["Mood_Score"],
        mode="lines", name="Mood Score",
        line=dict(color=C_AMBER, width=2),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Mood: %{y:.2f}<extra></extra>",
    ), row=1, col=1)

    fig.add_hline(y=0, line_color="rgba(148,163,184,0.4)", line_width=1, line_dash="dash", row=1, col=1)

    last_point = df.iloc[-1]
    fig.add_annotation(
        x=last_point["DATE"], y=last_point["Mood_Score"],
        text=f"<b>{last_point['Mood_Score']:.1f}</b>",
        showarrow=True, arrowhead=2, arrowcolor=C_AMBER,
        ax=40, ay=0,
        bgcolor="rgba(10,14,23,0.85)", bordercolor=C_AMBER, borderwidth=1,
        font=dict(color=C_AMBER_BRIGHT, size=11, family="JetBrains Mono, monospace"),
        row=1, col=1,
    )

    # ── OU forward projection ────────────────────────────────────────────
    ou_theta = float(last_point.get("OU_Theta", 0.05))
    ou_mu    = float(last_point.get("OU_Mu",    0.0))
    ou_sigma = float(last_point.get("OU_Sigma", 1.0))
    ou_std   = ou_sigma / np.sqrt(2.0 * max(ou_theta, 1e-4))

    last_date  = last_point["DATE"]
    proj_dates = pd.date_range(start=last_date, periods=ou_proj_days + 1, freq="D")[1:]
    proj_n     = np.arange(1, ou_proj_days + 1, dtype=np.float64)

    x_current_ou = last_point["Mood_Score"] / mood_scale * max(ou_std, 1e-6) + ou_mu
    proj_ou   = ou_mu + (x_current_ou - ou_mu) * np.exp(-ou_theta * proj_n)
    proj_mood = np.clip((proj_ou - ou_mu) / max(ou_std, 1e-6) * mood_scale, -100, 100)

    fig.add_trace(go.Scatter(
        x=proj_dates, y=proj_mood,
        mode="lines", name="OU Projection",
        line=dict(color=C_AMBER, width=1.5, dash="dot"),
        opacity=0.55,
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Projected: %{y:.1f}<extra></extra>",
    ), row=1, col=1)

    fig.add_annotation(
        x=proj_dates[-1], y=0.0,
        text=f"EQ ({last_point.get('OU_Half_Life', 0):.0f}d t½)",
        showarrow=False,
        font=dict(color="#64748B", size=9, family="JetBrains Mono, monospace"),
        xanchor="left", xshift=5, row=1, col=1,
    )

    # Dynamic y-bounds
    _yc = [df["Mood_Score"].values]
    if "Confidence_Upper" in df.columns:
        _yc.append(df["Confidence_Upper"].values)
    if "Confidence_Lower" in df.columns:
        _yc.append(df["Confidence_Lower"].values)
    _yc.append(proj_mood)
    _y_all = np.concatenate([c[np.isfinite(c)] for c in _yc])
    _y_min, _y_max = float(_y_all.min()) if len(_y_all) else -100, float(_y_all.max()) if len(_y_all) else 100
    _y_pad = max((_y_max - _y_min) * 0.08, 2.0)
    mood_y_lo, mood_y_hi = _y_min - _y_pad, _y_max + _y_pad

    # Regime transition markers
    if "Regime" in df.columns:
        regimes = df["Regime"].values
        dates   = df["DATE"].values
        groups: dict[str, tuple[list, list]] = {}
        for i in range(1, len(regimes)):
            if regimes[i] != regimes[i - 1] and regimes[i] != "Unknown":
                color = regime_styles.get(regimes[i], (C_MUTED, "neutral"))[0]
                groups.setdefault(color, ([], []))
                xg, yg = groups[color]
                xg.extend([dates[i], dates[i], None])
                yg.extend([mood_y_lo, mood_y_hi, None])
        for color, (xg, yg) in groups.items():
            fig.add_trace(go.Scattergl(
                x=xg, y=yg, mode="lines",
                line=dict(color=color, width=1, dash="dot"),
                opacity=0.5, showlegend=False, hoverinfo="skip",
            ), row=1, col=1)

    # ── Row 2: MSF Spread ─────────────────────────────────────────────────
    msf_values = msf_filtered["msf_spread"].values
    fig.add_trace(go.Scattergl(
        x=df["DATE"], y=msf_values,
        mode="lines", name="MSF Spread",
        line=dict(color=C_CYAN, width=2),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>MSF: %{y:.2f}<extra></extra>",
    ), row=2, col=1)
    fig.add_hline(y=0, line_color="rgba(148,163,184,0.4)", line_width=1, row=2, col=1)

    # Divergence triangles
    lookback = 10
    mood_series = df["Mood_Score"]
    msf_series  = pd.Series(msf_values, index=df.index)
    rmm_min = mood_series.rolling(lookback + 1, min_periods=1).min()
    rmm_max = mood_series.rolling(lookback + 1, min_periods=1).max()
    rms_min = msf_series.rolling(lookback + 1, min_periods=1).min()
    rms_max = msf_series.rolling(lookback + 1, min_periods=1).max()
    p_mood_min, p_msf_min = rmm_min.shift(lookback), rms_min.shift(lookback)
    p_mood_max, p_msf_max = rmm_max.shift(lookback), rms_max.shift(lookback)
    bear_mask = (mood_series == rmm_min) & (mood_series < p_mood_min) & (rms_min > p_msf_min)
    bull_mask = (mood_series == rmm_max) & (mood_series > p_mood_max) & (rms_max < p_msf_max)
    valid = np.zeros(len(df), dtype=bool)
    valid[lookback * 2 : len(df) - 1] = True
    red_idx   = np.where(bear_mask & valid)[0]
    green_idx = np.where(bull_mask & valid)[0]

    if len(red_idx):
        fig.add_trace(go.Scatter(
            x=[df["DATE"].iloc[i] for i in red_idx], y=[5] * len(red_idx),
            mode="markers", name="Bearish Signal",
            marker=dict(symbol="triangle-down", size=9, color=C_ROSE,
                        line=dict(color=C_ROSE, width=1)),
            hoverinfo="skip", showlegend=False,
        ), row=2, col=1)
    if len(green_idx):
        fig.add_trace(go.Scatter(
            x=[df["DATE"].iloc[i] for i in green_idx], y=[-5] * len(green_idx),
            mode="markers", name="Bullish Signal",
            marker=dict(symbol="triangle-up", size=9, color=C_EMERALD,
                        line=dict(color=C_EMERALD, width=1)),
            hoverinfo="skip", showlegend=False,
        ), row=2, col=1)

    # ── Layout — Obsidian Quant ───────────────────────────────────────────
    fig.update_layout(
        **PLOTLY_BASE,
        height=750,
        hovermode="x unified",
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1,
            font=dict(size=10, family="JetBrains Mono, monospace"),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=60, r=20, t=60, b=40),
        spikedistance=-1,
        yaxis=dict(
            title=dict(text="Mood Score", font=dict(size=11, color=C_MUTED, family="JetBrains Mono, monospace")),
            showgrid=True, gridcolor=PLOTLY_GRID, gridwidth=0.5,
            zeroline=True, zerolinecolor=PLOTLY_GRID_ZERO, zerolinewidth=0.5,
            linecolor="rgba(255,255,255,0.04)",
            tickfont=dict(size=9, family="JetBrains Mono, monospace", color="#64748B"),
            range=[mood_y_hi, mood_y_lo],
        ),
        yaxis2=dict(
            title=dict(text="MSF Spread", font=dict(size=11, color=C_MUTED, family="JetBrains Mono, monospace")),
            showgrid=True, gridcolor=PLOTLY_GRID, gridwidth=0.5,
            zeroline=True, zerolinecolor=PLOTLY_GRID_ZERO, zerolinewidth=0.5,
            linecolor="rgba(255,255,255,0.04)",
            tickfont=dict(size=9, family="JetBrains Mono, monospace", color="#64748B"),
        ),
        xaxis=dict(
            showgrid=False, linecolor="rgba(255,255,255,0.04)",
            showspikes=True, spikemode="across", spikesnap="cursor",
            spikethickness=0.5, spikedash="dash",
            spikecolor="rgba(148,163,184,0.18)",
        ),
        xaxis2=dict(
            showgrid=True, gridcolor=PLOTLY_GRID, gridwidth=0.5, type="date",
            linecolor="rgba(255,255,255,0.04)",
            tickfont=dict(size=9, family="JetBrains Mono, monospace", color="#64748B"),
            showspikes=True, spikemode="across", spikesnap="cursor",
            spikethickness=0.5, spikedash="dash",
            spikecolor="rgba(148,163,184,0.18)",
        ),
    )

    # Thin separator line between panes
    fig.add_shape(
        type="line", xref="paper", yref="paper",
        x0=0, y0=0.38, x1=1, y1=0.38,
        line=dict(color="rgba(255,255,255,0.06)", width=1),
    )

    st.markdown('<div class="chart-container mood">', unsafe_allow_html=True)
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": True,
            "scrollZoom": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════
    # PERIOD SUMMARY METRICS
    # ═══════════════════════════════════════════════════════════════════════
    section_divider()
    render_section_header(
        title="Period Summary",
        description=f"Mood & MSF statistics across the {selected_tf} window",
        icon="bar-chart",
        accent="cyan",
    )

    period_high = df["Mood_Score"].max()
    period_low  = df["Mood_Score"].min()
    period_avg  = df["Mood_Score"].mean()
    msf_avg     = msf_filtered["msf_spread"].mean()

    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        render_metric_card("Period High", f"{period_high:.1f}", "Most bullish", color_class="success", icon="arrow-up")
    with sc2:
        render_metric_card("Period Low", f"{period_low:.1f}", "Most bearish", color_class="danger", icon="arrow-down")
    with sc3:
        avg_cls = "success" if period_avg > 0 else "danger" if period_avg < 0 else "neutral"
        render_metric_card("Average Mood", f"{period_avg:.1f}", f"{selected_tf} period", color_class=avg_cls)
    with sc4:
        # Lower MSF is "more oversold" → success; higher → danger
        msf_cls = "success" if msf_avg < 0 else "danger" if msf_avg > 0 else "neutral"
        render_metric_card("Avg MSF Spread", f"{msf_avg:+.2f}", f"{selected_tf} period", color_class=msf_cls)

    # ═══════════════════════════════════════════════════════════════════════
    # MSF COMPONENT DECOMPOSITION
    # ═══════════════════════════════════════════════════════════════════════
    section_divider()
    render_section_header(
        title="MSF Component Breakdown",
        description="Current contribution of each component · weights = inverse-variance (auto-calibrated)",
        icon="layers",
        accent="violet",
    )

    msf_idx = min(len(msf_filtered) - 1, len(df) - 1)
    if msf_idx >= 0 and not msf_filtered.empty:
        comps = [
            ("momentum",  "Momentum",  "var(--amber)"),
            ("structure", "Structure", "var(--violet)"),
            ("regime",    "Regime",    "var(--emerald)"),
            ("flow",      "Flow",      "var(--cyan)"),
        ]
        c_cols = st.columns(4, gap="small")
        for j, (name, label, color) in enumerate(comps):
            val = msf_filtered[name].iloc[msf_idx] if name in msf_filtered.columns else 0
            period_val = msf_filtered[name].mean() if name in msf_filtered.columns else 0
            bar_pct = max(0, min(100, (val + 10) / 20 * 100))
            with c_cols[j]:
                st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(145deg, rgba(17,24,39,0.45) 0%, rgba(17,24,39,0.4) 100%);
                        border: 1px solid var(--border);
                        border-radius: var(--r-md);
                        padding: var(--sp-4) var(--sp-4);
                        backdrop-filter: blur(8px);
                    ">
                        <div style="display:flex; justify-content:space-between;
                                    align-items:center; margin-bottom:0.6rem;">
                            <span style="font-family:var(--data); font-size:0.62rem; color:var(--ink-tertiary);
                                         font-weight:600; text-transform:uppercase; letter-spacing:0.1em;">
                                {label}
                            </span>
                            <span style="font-family:var(--display); font-size:1.1rem; font-weight:700;
                                         color:{color}; font-variant-numeric:tabular-nums;">
                                {val:+.1f}
                            </span>
                        </div>
                        <div style="height:4px; background:rgba(255,255,255,0.04);
                                    border-radius:2px; position:relative;">
                            <div style="position:absolute; left:50%; top:0; width:1px; height:4px;
                                        background:rgba(255,255,255,0.12);"></div>
                            <div style="width:{bar_pct:.0f}%; height:100%; background:{color};
                                        border-radius:2px; opacity:0.85; box-shadow:0 0 8px {color};"></div>
                        </div>
                        <div style="font-family:var(--data); font-size:0.6rem; color:var(--ink-tertiary);
                                    margin-top:0.4rem; letter-spacing:0.04em;">
                            Period avg: <span style="color:var(--ink-secondary);">{period_val:+.1f}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
