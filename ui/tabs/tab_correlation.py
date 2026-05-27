"""
Arthagati — Correlation & Predictor Analysis view.

Decay-weighted Spearman vs PE/EY anchors + entropy quality ranking.
"""

from __future__ import annotations

import html as html_mod

import pandas as pd
import streamlit as st

from ui.components import (
    render_section_header,
    render_warning_box,
    render_interpretation_card,
    section_divider,
    get_icon,
)


def _render_corr_row(variable: str, corr_val: float) -> None:
    """One correlation row (variable name + bar + numeric value)."""
    color = "var(--emerald-bright)" if corr_val > 0 else "var(--rose-bright)"
    bar_pct = min(abs(corr_val) * 100, 100)
    if abs(corr_val) >= 0.5:
        dot_color, dot_label = "var(--emerald)", "strong"
    elif abs(corr_val) >= 0.3:
        dot_color, dot_label = "var(--amber)", "moderate"
    else:
        dot_color, dot_label = "var(--ink-tertiary)", "weak"

    st.markdown(
        f"""
        <div style="
            display:flex; align-items:center; margin-bottom:0.4rem;
            padding:0.55rem 0.8rem;
            background: linear-gradient(145deg, rgba(17,24,39,0.45) 0%, rgba(17,24,39,0.35) 100%);
            backdrop-filter: blur(6px);
            border:1px solid var(--border); border-radius:var(--r-sm);
            transition: all 200ms cubic-bezier(0.16,1,0.3,1);
        ">
            <span style="width:8px; height:8px; border-radius:50%;
                         background:{dot_color}; box-shadow:0 0 6px {dot_color};
                         margin-right:0.7rem;"
                  title="{dot_label}"></span>
            <span style="width:140px; font-family:var(--data); font-size:0.78rem;
                         color:var(--ink-primary); font-weight:500;">
                {html_mod.escape(variable)}
            </span>
            <div style="flex:1; height:5px; background:rgba(255,255,255,0.04);
                        border-radius:3px; margin:0 12px; position:relative;">
                <div style="width:{bar_pct}%; height:100%; background:{color};
                            border-radius:3px; box-shadow:0 0 6px {color};"></div>
            </div>
            <span style="width:64px; text-align:right; font-family:var(--data);
                         font-size:0.78rem; color:{color}; font-weight:700;
                         font-variant-numeric:tabular-nums;">
                {corr_val:+.2f}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_quality_row(rank: int, row: dict, max_quality: float) -> None:
    """One predictor-quality row (ranked, with bar + recommendation badge)."""
    bar_pct = (row["quality"] / max_quality) * 100 if max_quality else 0

    if row["quality"] >= max_quality * 0.5 and row["coverage"] > 50:
        rec_label, badge_class = "KEEP", "badge-strong-buy"
    elif row["quality"] >= max_quality * 0.2 and row["coverage"] > 30:
        rec_label, badge_class = "USEFUL", "badge-hold"
    elif row["coverage"] < 10:
        rec_label, badge_class = "NO DATA", "badge-caution"
    else:
        rec_label, badge_class = "WEAK", "badge-caution"

    active_dot = (
        '<span style="color:var(--amber);">●</span>'
        if row["active"]
        else '<span style="color:var(--ink-tertiary);">○</span>'
    )
    active_label = "Active" if row["active"] else "Inactive"

    st.markdown(
        f"""
        <div style="
            display:flex; align-items:center; gap:var(--sp-3);
            margin-bottom:0.4rem; padding:0.6rem 0.85rem;
            background: linear-gradient(145deg, rgba(17,24,39,0.45) 0%, rgba(17,24,39,0.35) 100%);
            backdrop-filter: blur(6px);
            border:1px solid {'rgba(212,168,83,0.18)' if row['active'] else 'var(--border)'};
            border-radius:var(--r-sm);
        ">
            <span style="width:22px; font-family:var(--data); font-size:0.7rem;
                         color:var(--ink-tertiary); font-weight:700;">
                {rank:02d}
            </span>
            <span style="width:140px; font-family:var(--data); font-size:0.8rem;
                         color:var(--ink-primary); font-weight:600;">
                {html_mod.escape(row['variable'])}
            </span>
            <div style="flex:1; height:5px; background:rgba(255,255,255,0.04);
                        border-radius:3px; position:relative;">
                <div style="width:{bar_pct:.0f}%; height:100%;
                            background:linear-gradient(90deg, var(--amber) 0%, var(--amber-bright) 100%);
                            border-radius:3px; box-shadow:0 0 8px var(--amber-glow);"></div>
            </div>
            <span style="width:64px; text-align:right; font-family:var(--data);
                         font-size:0.7rem; color:var(--ink-secondary);
                         font-variant-numeric:tabular-nums;">
                |ρ| {row['avg_corr']:.2f}
            </span>
            <span style="width:64px; text-align:right; font-family:var(--data);
                         font-size:0.7rem; color:var(--ink-secondary);
                         font-variant-numeric:tabular-nums;">
                H {row['entropy']:.2f}
            </span>
            <span class="position-card-badge {badge_class}" style="font-size:0.62rem; min-width:78px; justify-content:center;">
                {rec_label}
            </span>
            <span style="width:90px; text-align:right; font-family:var(--data);
                         font-size:0.7rem; color:var(--ink-secondary);">
                {active_dot} {active_label}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render(
    raw_df,
    *,
    active_preds,
    non_predictor_cols,
    calculate_anchor_correlations,
    shannon_entropy,
) -> None:
    """Render the Correlation & Predictor Analysis view."""

    render_section_header(
        title="Correlation & Predictor Analysis",
        description="Decay-weighted Spearman correlations vs PE & EY anchors · entropy-weighted predictor quality",
        icon="file-text",
        accent="cyan",
    )

    # ── Anchor health diagnostic ─────────────────────────────────────────
    anchors = {"NIFTY50_PE": "PE Ratio", "NIFTY50_EY": "Earnings Yield"}
    anchor_health: dict[str, dict] = {}
    for col, label in anchors.items():
        if col in raw_df.columns:
            nunique = raw_df[col].nunique()
            has_variance = nunique > 3 and raw_df[col].std() > 1e-6
            anchor_health[col] = {"label": label, "ok": has_variance, "nunique": nunique}
        else:
            anchor_health[col] = {"label": label, "ok": False, "nunique": 0}

    bad_anchors = [v["label"] for v in anchor_health.values() if not v["ok"]]
    if bad_anchors:
        render_warning_box(
            title="Data Quality Issue",
            content=(
                f"{', '.join(bad_anchors)} has insufficient variance in the source data. "
                "If Earnings Yield is empty in the sheet, it is auto-derived from PE (1/PE × 100). "
                "Check that your Google Sheet has valid data for these columns."
            ),
        )

    # ── Correlation bars (PE + EY side by side) ──────────────────────────
    section_divider()
    col1, col2 = st.columns(2, gap="small")

    def _render_corr_block(parent_col, anchor_col: str, title: str, icon: str):
        with parent_col:
            render_section_header(
                title=title,
                description=f"Variables ranked by |ρ| with {anchor_col}",
                icon=icon,
                accent="cyan" if anchor_col == "NIFTY50_PE" else "emerald",
            )
            if not anchor_health.get(anchor_col, {}).get("ok", False):
                st.caption(f"{anchor_col} has insufficient data variance — correlations may be unreliable.")
            corrs = calculate_anchor_correlations(raw_df, anchor_col, active_preds)
            if corrs.empty:
                st.caption("No correlations computed. Check data source.")
                return corrs
            corrs_display = corrs.sort_values("correlation", key=abs, ascending=False)
            for _, r in corrs_display.iterrows():
                _render_corr_row(r["variable"], r["correlation"])
            return corrs

    pe_corrs = _render_corr_block(col1, "NIFTY50_PE", "PE Ratio Correlations", "chart")
    ey_corrs = _render_corr_block(col2, "NIFTY50_EY", "Earnings Yield Correlations", "bar-chart")

    # ── Predictor quality assessment ─────────────────────────────────────
    section_divider()
    render_section_header(
        title="Predictor Quality Assessment",
        description="Quality = |ρ| × (1 − entropy) — exactly how the mood engine weights predictors internally",
        icon="target",
        accent="violet",
    )

    all_vars = [
        c for c in raw_df.columns
        if c not in non_predictor_cols and pd.api.types.is_numeric_dtype(raw_df[c])
    ]
    quality_rows = []
    for var in all_vars:
        pe_corr = 0.0
        if pe_corrs is not None and not pe_corrs.empty:
            m = pe_corrs.loc[pe_corrs["variable"] == var]
            if len(m) > 0:
                pe_corr = abs(m.iloc[0]["correlation"])
        ey_corr = 0.0
        if ey_corrs is not None and not ey_corrs.empty:
            m = ey_corrs.loc[ey_corrs["variable"] == var]
            if len(m) > 0:
                ey_corr = abs(m.iloc[0]["correlation"])
        avg_corr = (pe_corr + ey_corr) / 2

        var_returns = raw_df[var].pct_change().dropna().values
        entropy = shannon_entropy(var_returns) if len(var_returns) > 10 else 0.5
        info_quality = 1.0 - entropy
        quality_score = avg_corr * max(info_quality, 0.1)
        non_zero_pct = (raw_df[var] != 0).mean() * 100
        quality_rows.append({
            "variable": var,
            "pe_corr": pe_corr,
            "ey_corr": ey_corr,
            "avg_corr": avg_corr,
            "entropy": entropy,
            "quality": quality_score,
            "coverage": non_zero_pct,
            "active": var in active_preds,
        })

    quality_rows.sort(key=lambda x: x["quality"], reverse=True)
    if not quality_rows:
        return

    max_quality = max(r["quality"] for r in quality_rows) or 1.0
    for rank, row in enumerate(quality_rows, 1):
        _render_quality_row(rank, row, max_quality)

    # ── Summary interpretation ───────────────────────────────────────────
    keep_count = sum(1 for r in quality_rows if r["quality"] >= max_quality * 0.5 and r["coverage"] > 50)
    useful_count = sum(
        1 for r in quality_rows
        if max_quality * 0.2 <= r["quality"] < max_quality * 0.5 and r["coverage"] > 30
    )
    weak_count = len(quality_rows) - keep_count - useful_count

    summary_body = (
        f"<strong style='color:var(--emerald);'>{keep_count} strong</strong> predictors "
        f"(high correlation × low entropy) · "
        f"<strong style='color:var(--amber);'>{useful_count} useful</strong> (moderate signal) · "
        f"<strong style='color:var(--ink-tertiary);'>{weak_count} weak</strong> (low signal or noisy).<br><br>"
        "<span style='font-size:0.72rem; color:var(--ink-tertiary);'>"
        "|ρ| = average |correlation| with PE &amp; EY anchors · "
        "H = Shannon entropy of returns (lower = more structured) · "
        "Quality = |ρ| × (1−H) — same formula the mood engine uses for predictor weighting."
        "</span>"
    )
    render_interpretation_card(
        title="Recommendation Summary",
        body=summary_body,
        color="info",
    )
