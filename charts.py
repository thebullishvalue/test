"""
PRAGYAM — Institutional Chart Components
══════════════════════════════════════════════════════════════════════════════

Hemrek Capital Design System v2.0
Minimalist, data-forward visualizations for quantitative finance.

Design principles:
  - Transparent plot backgrounds (CSS card handles container)
  - Ultra-subtle grid (nearly invisible reference structure)
  - Thin, precise lines (1.5px data, 0.5-1px reference)
  - Restrained fills (alpha ≤ 0.10 for most areas)
  - Color used sparingly and with meaning
  - No chart-level titles (section headers handle context)
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple

# ══════════════════════════════════════════════════════════════════════════════
# DESIGN TOKENS
# ══════════════════════════════════════════════════════════════════════════════

COLORS = {
    # Brand
    'primary': '#FFC300',
    'primary_rgb': '255, 195, 0',

    # Surfaces
    'background': '#0F0F0F',
    'card': '#1A1A1A',
    'elevated': '#2A2A2A',

    # Borders
    'border': '#2A2A2A',
    'border_light': '#3A3A3A',

    # Typography
    'text': '#EAEAEA',
    'text_secondary': '#B0B0B0',
    'muted': '#888888',

    # Semantic
    'success': '#10b981',
    'danger': '#ef4444',
    'warning': '#f59e0b',
    'info': '#06b6d4',
    'neutral': '#888888',

    # Trading
    'bull': '#10b981',
    'bear': '#ef4444',

    # Multi-series palette (desaturated for cleaner stacking)
    'palette': [
        '#FFC300', '#34d399', '#22d3ee', '#fbbf24',
        '#a78bfa', '#fb7185', '#a3e635', '#fb923c',
    ],
}

# Internal tokens — not exported but used by every chart
_GRID = 'rgba(255,255,255,0.035)'
_ZERO = 'rgba(255,255,255,0.08)'
_TICK = '#6B7280'
_LABEL = '#9CA3AF'
_FONT = 'Inter, -apple-system, BlinkMacSystemFont, sans-serif'


def _axis(**overrides) -> dict:
    """Standard axis configuration."""
    base = dict(
        showgrid=True,
        gridcolor=_GRID,
        gridwidth=1,
        zeroline=False,
        linewidth=0,
        tickfont=dict(color=_TICK, size=10),
        title_font=dict(color=_LABEL, size=11),
    )
    base.update(overrides)
    return base


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert #RRGGBB to rgba(r,g,b,a)."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def get_chart_layout(
    title: str = "",
    height: int = 400,
    show_legend: bool = True,
    legend_position: str = 'top',
) -> dict:
    """Standardized Plotly layout for Hemrek Design System."""
    legend_configs = {
        'top': dict(
            orientation='h', y=1.02, x=0.0,
            xanchor='left', yanchor='bottom',
            font=dict(size=10, color=_TICK),
        ),
        'bottom': dict(
            orientation='h', y=-0.18, x=0.5,
            xanchor='center', yanchor='top',
            font=dict(size=10, color=_TICK),
        ),
        'right': dict(
            orientation='v', y=0.5, x=1.02,
            xanchor='left', yanchor='middle',
            font=dict(size=10, color=_TICK),
        ),
    }

    config = {
        'template': 'plotly_dark',
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'height': height,
        'margin': dict(l=52, r=16, t=36 if not title else 56, b=44),
        'font': dict(family=_FONT, color=COLORS['text'], size=12),
        'showlegend': show_legend,
        'legend': legend_configs.get(legend_position, legend_configs['top']),
        'hovermode': 'x unified',
        'hoverlabel': dict(
            bgcolor=COLORS['elevated'],
            font_size=11,
            font_family=_FONT,
            bordercolor=COLORS['border_light'],
        ),
    }

    if title:
        config['title'] = dict(
            text=title, font=dict(size=13, color=_LABEL),
            x=0.0, xanchor='left', y=0.98, yanchor='top',
        )
    else:
        config['title'] = dict(text='', font=dict(size=1))

    return config


# ══════════════════════════════════════════════════════════════════════════════
# EQUITY & PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

def create_equity_drawdown_chart(
    returns_df: pd.DataFrame,
    date_col: str = 'date',
    return_col: str = 'return',
) -> go.Figure:
    """Dual-panel equity curve with underwater analysis."""
    df = returns_df.copy().sort_values(date_col)
    df['equity'] = (1 + df[return_col]).cumprod()
    df['peak'] = df['equity'].expanding().max()
    df['drawdown'] = (df['equity'] / df['peak']) - 1

    eq_min, eq_max = df['equity'].min(), df['equity'].max()
    pad = (eq_max - eq_min) * 0.08
    y_lo = max(0.85, eq_min - pad)
    y_hi = eq_max + pad

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.06, row_heights=[0.74, 0.26],
    )
    fig.layout.annotations = ()

    # Baseline anchor (invisible) — must come BEFORE equity for fill='tonexty'
    fig.add_trace(go.Scatter(
        x=df[date_col], y=[y_lo] * len(df),
        mode='lines', line=dict(width=0, color='rgba(0,0,0,0)'),
        showlegend=False, hoverinfo='skip',
    ), row=1, col=1)

    # Equity line with soft gradient fill
    fig.add_trace(go.Scatter(
        x=df[date_col], y=df['equity'],
        mode='lines', name='Portfolio',
        line=dict(color=COLORS['primary'], width=1.8),
        fill='tonexty',
        fillcolor=_hex_to_rgba(COLORS['primary'], 0.07),
        hovertemplate='%{y:.4f}<extra>Portfolio</extra>',
    ), row=1, col=1)

    # High water mark
    fig.add_trace(go.Scatter(
        x=df[date_col], y=df['peak'],
        mode='lines', name='High Water Mark',
        line=dict(color='rgba(255,255,255,0.15)', width=1, dash='dot'),
        hovertemplate='%{y:.4f}<extra>HWM</extra>',
    ), row=1, col=1)

    # Drawdown area
    fig.add_trace(go.Scatter(
        x=df[date_col], y=df['drawdown'],
        mode='lines', name='Drawdown', fill='tozeroy',
        line=dict(color=_hex_to_rgba(COLORS['danger'], 0.5), width=1),
        fillcolor=_hex_to_rgba(COLORS['danger'], 0.10),
        hovertemplate='%{y:.1%}<extra>DD</extra>',
    ), row=2, col=1)

    layout = get_chart_layout(height=460, show_legend=True, legend_position='top')
    layout['margin'] = dict(l=52, r=16, t=40, b=36)
    fig.update_layout(**layout)

    fig.update_yaxes(range=[y_lo, y_hi], title_text='', row=1, col=1, **_axis())
    fig.update_yaxes(title_text='', tickformat='.0%', row=2, col=1, **_axis())
    fig.update_xaxes(row=1, col=1, **_axis(showticklabels=False))
    fig.update_xaxes(row=2, col=1, **_axis())

    return fig


def create_rolling_metrics_chart(
    returns_df: pd.DataFrame,
    window: int = 12,
    date_col: str = 'date',
    return_col: str = 'return',
    periods_per_year: int = 52,
) -> go.Figure:
    """Rolling Sharpe and Sortino with zone shading."""
    df = returns_df.copy().sort_values(date_col)

    r_mean = df[return_col].rolling(window=window).mean()
    r_std = df[return_col].rolling(window=window).std()
    r_sharpe = (r_mean / r_std.replace(0, np.nan)) * np.sqrt(periods_per_year)

    ds = df[return_col].clip(upper=0)
    r_sortino = (r_mean / ds.rolling(window=window).std().replace(0, np.nan)) * np.sqrt(periods_per_year)

    fig = go.Figure()

    # Zone shading: below zero = faint red, above 1 = faint green
    fig.add_hrect(y0=-10, y1=0, fillcolor=_hex_to_rgba(COLORS['danger'], 0.03), line_width=0)
    fig.add_hrect(y0=1, y1=20, fillcolor=_hex_to_rgba(COLORS['success'], 0.03), line_width=0)

    # Zero line
    fig.add_hline(y=0, line_color=_ZERO, line_width=1)
    fig.add_hline(y=1, line_color='rgba(255,255,255,0.05)', line_width=1, line_dash='dot')

    fig.add_trace(go.Scatter(
        x=df[date_col], y=r_sharpe, mode='lines',
        name=f'Sharpe ({window}w)',
        line=dict(color=COLORS['primary'], width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=df[date_col], y=r_sortino, mode='lines',
        name=f'Sortino ({window}w)',
        line=dict(color=COLORS['success'], width=1.5),
    ))

    # Clip visible y-range to avoid extreme outliers
    all_vals = pd.concat([r_sharpe, r_sortino]).dropna()
    if not all_vals.empty:
        q_lo, q_hi = all_vals.quantile(0.02), all_vals.quantile(0.98)
        pad = (q_hi - q_lo) * 0.15
        y_range = [max(q_lo - pad, -5), min(q_hi + pad, 8)]
    else:
        y_range = [-2, 4]

    layout = get_chart_layout(height=320, show_legend=True)
    fig.update_layout(**layout)
    fig.update_xaxes(**_axis())
    fig.update_yaxes(range=y_range, **_axis())

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# CORRELATION & HEATMAP
# ══════════════════════════════════════════════════════════════════════════════

def create_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "",
) -> go.Figure:
    """Adaptive correlation heatmap — green-to-red for typical portfolios."""
    vals = corr_matrix.values.flatten()
    od = ~np.eye(len(corr_matrix), dtype=bool).flatten()
    c_min = float(np.nanmin(vals[od]))
    n = len(corr_matrix)

    if c_min > -0.1:
        cs = [
            [0.0, '#059669'], [0.30, '#34d399'], [0.50, '#fbbf24'],
            [0.75, '#f97316'], [1.0, '#dc2626'],
        ]
        zmin = max(0, round(c_min - 0.05, 1))
        zmax, zmid = 1.0, (zmin + 1.0) / 2
    else:
        cs = [
            [0.0, '#2563eb'], [0.25, '#60a5fa'], [0.5, '#4B5563'],
            [0.75, '#f87171'], [1.0, '#dc2626'],
        ]
        zmin, zmax, zmid = -1, 1, 0

    txt_size = 10 if n <= 6 else (9 if n <= 10 else 8)

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.index,
        colorscale=cs, zmid=zmid, zmin=zmin, zmax=zmax,
        text=np.round(corr_matrix.values, 2), texttemplate='%{text}',
        textfont=dict(size=txt_size, color='rgba(255,255,255,0.85)'),
        hovertemplate='%{x} × %{y}<br>ρ = %{z:.3f}<extra></extra>',
        colorbar=dict(
            title=dict(text='ρ', font=dict(color=_TICK, size=12)),
            tickfont=dict(color=_TICK, size=9),
            thickness=10, len=0.7, outlinewidth=0,
        ),
        xgap=1, ygap=1,
    ))

    h = max(300, n * 38)
    layout = get_chart_layout(title=title, height=h, show_legend=False)
    layout['hovermode'] = 'closest'
    fig.update_layout(**layout)
    fig.update_xaxes(tickangle=45, tickfont=dict(size=10, color=_TICK), showgrid=False)
    fig.update_yaxes(tickfont=dict(size=10, color=_TICK), showgrid=False)

    return fig


def create_tier_sharpe_heatmap(
    subset_perf: Dict,
    strategy_names: List[str],
) -> go.Figure:
    """Sharpe ratio by position tier — rows sorted by average."""
    max_tier = 0
    for s in strategy_names:
        if s in subset_perf and subset_perf[s]:
            nums = [int(t.split('_')[1]) for t in subset_perf[s].keys()]
            if nums:
                max_tier = max(max_tier, max(nums))
    if max_tier == 0:
        return None

    hm = {}
    for s in strategy_names:
        hm[s] = [subset_perf.get(s, {}).get(f'tier_{i+1}', np.nan) for i in range(max_tier)]

    df = pd.DataFrame(hm).T
    df.columns = [f'T{i+1}' for i in range(df.shape[1])]
    df['_avg'] = df.mean(axis=1)
    df = df.sort_values('_avg', ascending=False).drop('_avg', axis=1)

    cs = [
        [0.0, '#dc2626'], [0.35, '#fbbf24'], [0.5, '#4B5563'],
        [0.65, '#6ee7b7'], [1.0, '#059669'],
    ]
    n = len(df)
    txt_size = 10 if n <= 8 else 9

    fig = go.Figure(data=go.Heatmap(
        z=df.values, x=df.columns, y=df.index,
        colorscale=cs, zmid=0,
        text=np.round(df.values, 2), texttemplate='%{text:.2f}',
        textfont=dict(size=txt_size, color='rgba(255,255,255,0.85)'),
        hovertemplate='%{y}<br>%{x}: Sharpe %{z:.3f}<extra></extra>',
        colorbar=dict(
            title=dict(text='Sharpe', font=dict(color=_TICK, size=11)),
            tickfont=dict(color=_TICK, size=9),
            thickness=10, len=0.7, outlinewidth=0,
        ),
        xgap=2, ygap=2,
    ))

    layout = get_chart_layout(height=max(280, n * 32 + 60), show_legend=False)
    layout['hovermode'] = 'closest'
    layout['margin'] = dict(l=120, r=16, t=36, b=36)
    fig.update_layout(**layout)
    fig.update_xaxes(tickfont=dict(size=10, color=_TICK), showgrid=False, side='top')
    fig.update_yaxes(tickfont=dict(size=10, color=_TICK), showgrid=False)

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SCATTER & FRONTIER
# ══════════════════════════════════════════════════════════════════════════════

def create_risk_return_scatter(
    strategy_data: List[Dict],
    show_cml: bool = True,
) -> go.Figure:
    """Risk-return bubble chart with optional CML."""
    df = pd.DataFrame(strategy_data)
    if df.empty:
        return go.Figure()

    df['vol_pct'] = df['Volatility'] * 100
    df['cagr_pct'] = df['CAGR'] * 100
    df['size'] = np.clip(np.abs(df['Max DD']) * 120 + 8, 14, 42)

    fig = go.Figure()

    # CML (behind bubbles)
    if show_cml and len(df) > 2:
        best = df.loc[df['Sharpe'].idxmax()]
        tv, tr = best['vol_pct'], best['cagr_pct']
        end_v = min(tv * 1.8, df['vol_pct'].max() * 1.25)
        end_r = tr * (end_v / tv) if tv > 0 else 0
        fig.add_trace(go.Scatter(
            x=[0, end_v], y=[0, end_r], mode='lines', name='CML',
            line=dict(color=COLORS['muted'], dash='dash', width=1.5),
            showlegend=False, hoverinfo='skip',
        ))
        fig.add_trace(go.Scatter(
            x=[tv], y=[tr], mode='markers', name='Optimal',
            marker=dict(size=14, color=COLORS['primary'], symbol='diamond',
                        line=dict(width=1.5, color='rgba(255,255,255,0.6)')),
            showlegend=False,
            hovertemplate=f'<b>{best["Strategy"]}</b> (optimal)<extra></extra>',
        ))

    # Bubbles with text labels
    fig.add_trace(go.Scatter(
        x=df['vol_pct'], y=df['cagr_pct'],
        mode='markers+text',
        marker=dict(
            size=df['size'],
            color=df['Sharpe'],
            colorscale=[
                [0.0, '#dc2626'], [0.33, '#fbbf24'],
                [0.5, '#a3a3a3'], [0.66, '#6ee7b7'], [1.0, '#059669'],
            ],
            cmin=-1, cmax=2,
            showscale=True,
            colorbar=dict(
                title=dict(text='Sharpe', font=dict(color=_TICK, size=10)),
                tickfont=dict(color=_TICK, size=9),
                thickness=10, len=0.6, outlinewidth=0, x=1.02,
            ),
            line=dict(width=2, color='rgba(255,255,255,0.8)'),
            opacity=0.95,
        ),
        text=df['Strategy'].apply(lambda x: x[:12] + '..' if len(x) > 12 else x),
        textposition='top center',
        textfont=dict(size=10, color=COLORS['text']),
        customdata=np.column_stack([df['Strategy'], df['Max DD'], df['Sharpe']]),
        hovertemplate=(
            '<b>%{customdata[0]}</b><br>'
            'CAGR: %{y:.1f}%<br>Vol: %{x:.1f}%<br>'
            'Sharpe: %{customdata[2]:.2f}<br>'
            'Max DD: %{customdata[1]:.1%}<extra></extra>'
        ),
        name='Strategies',
    ))

    # Axis ranges with tight padding
    vr = df['vol_pct'].max() - df['vol_pct'].min()
    cr = df['cagr_pct'].max() - df['cagr_pct'].min()
    vp, cp = max(vr * 0.12, 1.0), max(cr * 0.12, 0.5)

    layout = get_chart_layout(height=400, show_legend=False)
    layout['hovermode'] = 'closest'
    fig.update_layout(**layout)
    fig.update_xaxes(
        title_text='Volatility (%)',
        range=[max(0, df['vol_pct'].min() - vp), df['vol_pct'].max() + vp],
        **_axis(),
    )
    fig.update_yaxes(
        title_text='CAGR (%)',
        range=[df['cagr_pct'].min() - cp, df['cagr_pct'].max() + cp],
        **_axis(),
    )

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# RADAR & FACTOR
# ══════════════════════════════════════════════════════════════════════════════

def create_factor_radar(
    factor_data: List[Dict],
    max_strategies: int = 4,
) -> go.Figure:
    """Multi-strategy factor fingerprint radar."""
    df = pd.DataFrame(factor_data)
    if df.empty:
        return go.Figure()

    if 'Efficiency' in df.columns:
        df = df.nlargest(min(max_strategies, len(df)), 'Efficiency')
    else:
        df = df.head(max_strategies)

    cats = [c for c in df.columns if c != 'Strategy']
    fig = go.Figure()

    for idx, (_, row) in enumerate(df.iterrows()):
        vals = [row[c] for c in cats] + [row[cats[0]]]
        color = COLORS['palette'][idx % len(COLORS['palette'])]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats + [cats[0]],
            fill='toself',
            name=row['Strategy'][:18],
            line=dict(color=color, width=2),
            fillcolor=_hex_to_rgba(color, 0.20),
            opacity=0.85,
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 1],
                showticklabels=True,
                tickfont=dict(size=9, color=COLORS['muted']),
                gridcolor='rgba(255,255,255,0.08)', linewidth=0,
                tick0=0, dtick=0.25,
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color=COLORS['text']),
                gridcolor='rgba(255,255,255,0.08)', linewidth=0,
            ),
            bgcolor='rgba(0,0,0,0)',
        ),
        showlegend=True,
        legend=dict(
            orientation='h', y=-0.15, x=0.5, xanchor='center',
            font=dict(size=9, color=_TICK),
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family=_FONT, color=COLORS['text']),
        height=380,
        margin=dict(l=56, r=56, t=28, b=56),
    )

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# WEIGHT EVOLUTION
# ══════════════════════════════════════════════════════════════════════════════

def create_weight_evolution_chart(
    weight_history: List[Dict],
    title: str = "",
) -> go.Figure:
    """Stacked area for weight evolution over time."""
    if not weight_history:
        return go.Figure()

    df = pd.DataFrame(weight_history)
    if 'date' not in df.columns or df.empty:
        return go.Figure()

    df['date'] = pd.to_datetime(df['date'])
    cols = [c for c in df.columns if c != 'date']
    if not cols:
        return go.Figure()

    fig = go.Figure()
    for idx, col in enumerate(cols):
        color = COLORS['palette'][idx % len(COLORS['palette'])]
        fig.add_trace(go.Scatter(
            x=df['date'], y=df[col],
            mode='lines', name=col[:20],
            stackgroup='one',
            line=dict(width=0.3, color=color),
            fillcolor=_hex_to_rgba(color, 0.65),
            hovertemplate=f'{col[:15]}: ' + '%{y:.1%}<extra></extra>',
        ))

    layout = get_chart_layout(title=title, height=380, show_legend=True, legend_position='bottom')
    fig.update_layout(**layout)
    fig.update_xaxes(**_axis())
    fig.update_yaxes(tickformat='.0%', **_axis())

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# CONVICTION & SIGNAL
# ══════════════════════════════════════════════════════════════════════════════

def create_signal_heatmap(
    signal_data: pd.DataFrame,
    value_col: str = 'signal',
    symbol_col: str = 'symbol',
) -> go.Figure:
    """Grid heatmap of signal strength by symbol."""
    if signal_data.empty:
        return go.Figure()

    sdf = signal_data.sort_values(value_col, ascending=False)
    symbols = sdf[symbol_col].apply(lambda x: x.replace('.NS', '')).tolist()
    values = sdf[value_col].tolist()

    n_cols = min(8, len(symbols))
    n_rows = (len(symbols) + n_cols - 1) // n_cols
    while len(symbols) < n_rows * n_cols:
        symbols.append('')
        values.append(0)

    z = np.array(values).reshape(n_rows, n_cols)
    text = np.array(symbols).reshape(n_rows, n_cols)

    cs = [[0.0, COLORS['danger']], [0.5, '#4B5563'], [1.0, COLORS['success']]]

    fig = go.Figure(data=go.Heatmap(
        z=z, text=text, texttemplate='%{text}',
        textfont=dict(size=10, color='rgba(255,255,255,0.85)'),
        colorscale=cs, zmid=0, showscale=True,
        colorbar=dict(
            title=dict(text='Signal', font=dict(color=_TICK, size=10)),
            tickfont=dict(color=_TICK, size=9),
            thickness=10, outlinewidth=0,
        ),
        hovertemplate='%{text}<br>Signal: %{z:.2f}<extra></extra>',
        xgap=2, ygap=2,
    ))

    layout = get_chart_layout(height=max(180, n_rows * 48), show_legend=False)
    layout['hovermode'] = 'closest'
    fig.update_layout(**layout)
    fig.update_xaxes(showticklabels=False, showgrid=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, autorange='reversed')

    return fig


def create_bar_chart(
    data: List[Dict],
    x_col: str,
    y_col: str,
    title: str = "",
    color_by_value: bool = True,
    horizontal: bool = False,
) -> go.Figure:
    """Standardized bar chart with conditional coloring."""
    df = pd.DataFrame(data)
    if df.empty:
        return go.Figure()

    if color_by_value:
        colors = [COLORS['success'] if v >= 0 else COLORS['danger'] for v in df[y_col]]
    else:
        colors = COLORS['primary']

    if horizontal:
        fig = go.Figure(go.Bar(
            x=df[y_col], y=df[x_col], orientation='h',
            marker_color=colors, marker_line_width=0,
            text=[f'{v:+.2f}' if isinstance(v, (int, float)) else str(v) for v in df[y_col]],
            textposition='outside', textfont=dict(color=_TICK, size=10),
        ))
        fig.add_vline(x=0, line_color=_ZERO, line_width=1)
    else:
        fig = go.Figure(go.Bar(
            x=df[x_col], y=df[y_col],
            marker_color=colors, marker_line_width=0,
            text=[f'{v:.2f}' if isinstance(v, (int, float)) else str(v) for v in df[y_col]],
            textposition='outside', textfont=dict(color=_TICK, size=10),
        ))
        fig.add_hline(y=0, line_color=_ZERO, line_width=1)

    layout = get_chart_layout(title=title, height=380)
    fig.update_layout(**layout, bargap=0.3)
    fig.update_xaxes(**_axis())
    fig.update_yaxes(**_axis())

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SPECTRAL ANALYSIS (Random Matrix Theory)
# ══════════════════════════════════════════════════════════════════════════════

def create_eigenvalue_histogram(
    eigenvalues: np.ndarray,
    mp_lambda_plus: float,
    mp_lambda_minus: float,
    gamma: float,
    sigma_sq: float = 1.0,
    title: str = "",
) -> go.Figure:
    """Eigenvalue distribution with Marchenko-Pastur overlay and signal markers."""
    n_signal = int(np.sum(eigenvalues > mp_lambda_plus))
    n_noise = len(eigenvalues) - n_signal
    signal_eigs = eigenvalues[eigenvalues > mp_lambda_plus]
    noise_eigs = eigenvalues[eigenvalues <= mp_lambda_plus]

    fig = go.Figure()

    # Noise histogram
    if len(noise_eigs) > 0:
        fig.add_trace(go.Histogram(
            x=noise_eigs, nbinsx=max(8, len(noise_eigs) // 3),
            name=f'Noise ({n_noise})',
            marker=dict(color=_hex_to_rgba('#6B7280', 0.5), line=dict(width=0.5, color=_hex_to_rgba('#6B7280', 0.7))),
            histnorm='probability density',
        ))

    # MP theoretical curve
    x_mp = np.linspace(max(0, mp_lambda_minus - 0.15), mp_lambda_plus + 0.3, 300)
    mp_pdf = np.zeros_like(x_mp)
    inside = (x_mp > mp_lambda_minus + 1e-12) & (x_mp < mp_lambda_plus - 1e-12)
    if np.any(inside):
        xi = x_mp[inside]
        mp_pdf[inside] = np.sqrt((mp_lambda_plus - xi) * (xi - mp_lambda_minus)) / (2 * np.pi * sigma_sq * xi / gamma)

    mp_peak = float(np.max(mp_pdf)) if np.any(mp_pdf > 0) else 0.5

    # MP curve as filled area
    fig.add_trace(go.Scatter(
        x=x_mp, y=mp_pdf, mode='lines', name='MP Theory',
        line=dict(color=COLORS['danger'], width=1.5),
        fill='tozeroy', fillcolor=_hex_to_rgba(COLORS['danger'], 0.06),
        hovertemplate='λ: %{x:.3f}<br>PDF: %{y:.4f}<extra></extra>',
    ))

    # Signal eigenvalue markers
    if len(signal_eigs) > 0:
        h = mp_peak * 1.3
        for i, eig in enumerate(signal_eigs):
            fig.add_trace(go.Scatter(
                x=[eig, eig], y=[0, h], mode='lines',
                line=dict(color=COLORS['primary'], width=2.5),
                name=f'Signal λ={eig:.2f}' if i < 5 else None,
                showlegend=(i < 5),
                hovertemplate=f'Signal: {eig:.3f}<extra></extra>',
            ))

    # λ+ threshold
    fig.add_vline(
        x=mp_lambda_plus, line_color=COLORS['warning'],
        line_width=1.5, line_dash='dot',
    )
    fig.add_annotation(
        x=mp_lambda_plus, y=mp_peak * 1.15, text=f'λ+ = {mp_lambda_plus:.2f}',
        showarrow=False, font=dict(color=COLORS['warning'], size=10),
        xanchor='left', xshift=6,
    )

    # Dynamic range
    x_max = float(max(eigenvalues)) * 1.1 if len(eigenvalues) else mp_lambda_plus + 1
    x_min = max(0, float(min(eigenvalues)) * 0.9) if len(eigenvalues) else 0

    layout = get_chart_layout(title=title, height=380)
    fig.update_layout(**layout)
    fig.update_xaxes(title_text='Eigenvalue (λ)', range=[x_min, x_max], **_axis())
    fig.update_yaxes(title_text='Density', **_axis())

    # Info annotation
    fig.add_annotation(
        x=0.98, y=0.95, xref='paper', yref='paper', xanchor='right',
        text=f'Signal: {n_signal}  ·  Noise: {n_noise}  ·  γ = {gamma:.2f}',
        showarrow=False,
        font=dict(color=_TICK, size=10),
        bgcolor=_hex_to_rgba(COLORS['elevated'], 0.8),
        borderpad=5,
    )

    return fig


def create_cleaned_vs_raw_correlation(
    raw_corr: np.ndarray,
    cleaned_corr: np.ndarray,
    labels: List[str],
    title: str = "",
) -> go.Figure:
    """Three-panel heatmap: raw, cleaned, and noise removed."""
    diff = raw_corr - cleaned_corr
    n = len(labels)

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=['Raw Sample', 'RMT-Cleaned', 'Noise Removed'],
        horizontal_spacing=0.08,
    )

    corr_cs = [
        [0.0, '#2563eb'], [0.25, '#60a5fa'], [0.5, '#4B5563'],
        [0.75, '#f87171'], [1.0, '#dc2626'],
    ]
    diff_cs = [
        [0.0, '#0891b2'], [0.5, '#1f2937'], [1.0, '#d97706'],
    ]

    txt_sz = 9 if n <= 8 else 7
    diff_max = max(abs(np.nanmin(diff)), abs(np.nanmax(diff)), 0.01)

    for col_idx, (z, cs, zmin, zmax, zmid, ht) in enumerate([
        (raw_corr, corr_cs, -1, 1, 0, 'Raw ρ: %{z:.3f}'),
        (cleaned_corr, corr_cs, -1, 1, 0, 'Clean ρ: %{z:.3f}'),
        (diff, diff_cs, -diff_max, diff_max, 0, 'Noise: %{z:.3f}'),
    ], 1):
        fig.add_trace(go.Heatmap(
            z=z, x=labels, y=labels,
            colorscale=cs, zmin=zmin, zmax=zmax, zmid=zmid,
            text=np.round(z, 2), texttemplate='%{text}',
            textfont=dict(size=txt_sz, color='rgba(255,255,255,0.8)'),
            showscale=(col_idx == 3),
            colorbar=dict(
                title=dict(text='Δρ', font=dict(color=_TICK, size=10)),
                tickfont=dict(color=_TICK, size=9),
                thickness=8, len=0.5, outlinewidth=0,
            ) if col_idx == 3 else None,
            hovertemplate=f'%{{x}} × %{{y}}<br>{ht}<extra></extra>',
            xgap=1, ygap=1,
        ), row=1, col=col_idx)

    h = max(300, n * 30 + 80)
    layout = get_chart_layout(title=title, height=h, show_legend=False)
    layout['hovermode'] = 'closest'
    fig.update_layout(**layout)

    for ann in fig.layout.annotations:
        ann.font = dict(size=11, color=_LABEL)

    for i in range(1, 4):
        fig.update_xaxes(tickangle=45, tickfont=dict(size=8, color=_TICK), showgrid=False, row=1, col=i)
        fig.update_yaxes(tickfont=dict(size=8, color=_TICK), showgrid=False, row=1, col=i)

    return fig


def create_absorption_ratio_chart(
    spectral_history: List[Dict],
    title: str = "",
) -> go.Figure:
    """Rolling absorption ratio with stress zone bands."""
    dates = [s['date'] for s in spectral_history]
    ar = [s['absorption_ratio'] for s in spectral_history]

    fig = go.Figure()

    # Zone bands
    fig.add_hrect(y0=0.8, y1=1.05, fillcolor=_hex_to_rgba(COLORS['danger'], 0.05), line_width=0)
    fig.add_hrect(y0=0.0, y1=0.4, fillcolor=_hex_to_rgba(COLORS['success'], 0.04), line_width=0)

    # Threshold lines
    fig.add_hline(y=0.8, line_color=_hex_to_rgba(COLORS['danger'], 0.3), line_width=1, line_dash='dot')
    fig.add_hline(y=0.4, line_color=_hex_to_rgba(COLORS['success'], 0.3), line_width=1, line_dash='dot')

    # AR line
    fig.add_trace(go.Scatter(
        x=dates, y=ar, mode='lines', name='Absorption Ratio',
        line=dict(color=COLORS['primary'], width=1.5),
        fill='tozeroy', fillcolor=_hex_to_rgba(COLORS['primary'], 0.05),
        hovertemplate='%{x}<br>AR: %{y:.3f}<extra></extra>',
    ))

    # Zone labels
    fig.add_annotation(x=0.99, y=0.90, xref='paper', yref='y', text='Systemic Risk',
                       showarrow=False, font=dict(color=_hex_to_rgba(COLORS['danger'], 0.5), size=9), xanchor='right')
    fig.add_annotation(x=0.99, y=0.20, xref='paper', yref='y', text='Diversified',
                       showarrow=False, font=dict(color=_hex_to_rgba(COLORS['success'], 0.5), size=9), xanchor='right')

    layout = get_chart_layout(title=title, height=340, show_legend=False)
    fig.update_layout(**layout)
    fig.update_xaxes(**_axis())
    fig.update_yaxes(range=[0, 1.05], **_axis())

    return fig


def create_factor_loading_heatmap(
    eigenvectors: np.ndarray,
    labels: List[str],
    eigenvalues: np.ndarray = None,
    n_factors: int = 5,
    title: str = "",
) -> go.Figure:
    """Factor loading heatmap with explained variance labels."""
    n_factors = min(n_factors, eigenvectors.shape[1])
    loadings = eigenvectors[:, :n_factors]

    if eigenvalues is not None and len(eigenvalues) >= n_factors:
        tot = eigenvalues.sum()
        f_labels = [f'F{i+1} ({eigenvalues[i]/tot*100:.1f}%)' for i in range(n_factors)]
    else:
        f_labels = [f'F{i+1}' for i in range(n_factors)]

    abs_max = max(abs(np.nanmin(loadings)), abs(np.nanmax(loadings)), 0.01)
    n = len(labels)
    txt_sz = 10 if n <= 8 else 9

    cs = [[0.0, '#2563eb'], [0.5, '#1f2937'], [1.0, '#dc2626']]

    fig = go.Figure(data=go.Heatmap(
        z=loadings, x=f_labels, y=labels,
        colorscale=cs, zmin=-abs_max, zmax=abs_max, zmid=0,
        text=np.round(loadings, 2), texttemplate='%{text}',
        textfont=dict(size=txt_sz, color='rgba(255,255,255,0.8)'),
        hovertemplate='%{y}<br>%{x}<br>Loading: %{z:.3f}<extra></extra>',
        colorbar=dict(
            title=dict(text='Loading', font=dict(color=_TICK, size=10)),
            tickfont=dict(color=_TICK, size=9),
            thickness=8, len=0.7, outlinewidth=0,
        ),
        xgap=2, ygap=2,
    ))

    layout = get_chart_layout(title=title, height=max(260, n * 26 + 80), show_legend=False)
    layout['hovermode'] = 'closest'
    fig.update_layout(**layout)
    fig.update_xaxes(tickfont=dict(size=10, color=_TICK), showgrid=False, side='top')
    fig.update_yaxes(tickfont=dict(size=10, color=_TICK), showgrid=False)

    return fig


def create_spectral_risk_dashboard(
    spectral_history: List[Dict],
    title: str = "",
) -> go.Figure:
    """2x2 multi-panel: AR, effective rank, condition number, largest eigenvalue."""
    dates = [s['date'] for s in spectral_history]
    ar = [s.get('absorption_ratio', 0) for s in spectral_history]
    eff = [s.get('effective_rank', 1) for s in spectral_history]
    cond = [s.get('condition_number', 1) for s in spectral_history]
    lam1 = [s.get('largest_eigenvalue', 1) for s in spectral_history]

    panels = [
        ('Absorption Ratio', ar, COLORS['primary'], '.3f'),
        ('Effective Rank', eff, COLORS['info'], '.1f'),
        ('Condition Number', cond, COLORS['warning'], '.1f'),
        ('Largest Eigenvalue', lam1, COLORS['danger'], '.2f'),
    ]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[p[0] for p in panels],
        vertical_spacing=0.14, horizontal_spacing=0.10,
    )

    for idx, (_, vals, color, fmt) in enumerate(panels):
        r, c = divmod(idx, 2)
        fig.add_trace(go.Scatter(
            x=dates, y=vals, mode='lines', name=panels[idx][0],
            line=dict(color=color, width=1.5),
            fill='tozeroy', fillcolor=_hex_to_rgba(color, 0.04),
            hovertemplate=f'%{{x}}<br>%{{y:{fmt}}}<extra></extra>',
        ), row=r + 1, col=c + 1)

    # AR threshold
    fig.add_hline(y=0.8, line_color=_hex_to_rgba(COLORS['danger'], 0.3),
                  line_width=1, line_dash='dot', row=1, col=1)

    layout = get_chart_layout(title=title, height=520, show_legend=False)
    layout['margin'] = dict(l=48, r=16, t=48, b=36)
    fig.update_layout(**layout)

    for ann in fig.layout.annotations:
        ann.font = dict(size=11, color=_LABEL)

    for r in range(1, 3):
        for c in range(1, 3):
            fig.update_xaxes(row=r, col=c, **_axis(tickfont=dict(color=_TICK, size=9)))
            fig.update_yaxes(row=r, col=c, **_axis(tickfont=dict(color=_TICK, size=9)))

    fig.update_yaxes(type='log', row=2, col=1)

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# MASTER ATTENTION CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def create_attention_heatmap(
    attention_matrix: np.ndarray,
    labels: List[str],
    title: str = "",
) -> go.Figure:
    """Inter-stock attention heatmap (neural correlation analog).

    Args:
        attention_matrix: shape (n_stocks, n_stocks), mean over timesteps.
        labels: Stock/ETF ticker labels.
        title: Optional chart title.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure(data=go.Heatmap(
        z=attention_matrix,
        x=labels,
        y=labels,
        colorscale=[
            [0, COLORS['background']],
            [0.3, _hex_to_rgba(COLORS['info'], 0.5)],
            [0.6, _hex_to_rgba(COLORS['primary'], 0.7)],
            [1.0, COLORS['primary']],
        ],
        colorbar=dict(
            title=dict(text='Attention', font=dict(color=_LABEL, size=10)),
            tickfont=dict(color=_TICK, size=9),
        ),
        hovertemplate='%{y} → %{x}: %{z:.3f}<extra></extra>',
    ))

    layout = get_chart_layout(title=title, height=500, show_legend=False)
    layout['xaxis'] = _axis(tickangle=-45, tickfont=dict(color=_TICK, size=8))
    layout['yaxis'] = _axis(tickfont=dict(color=_TICK, size=8), autorange='reversed')
    fig.update_layout(**layout)

    return fig


def create_cross_time_correlation_chart(
    cross_time_matrix: np.ndarray,
    source_label: str = "Source",
    target_label: str = "Target",
    title: str = "",
) -> go.Figure:
    """Cross-time correlation heatmap between two stocks.

    Args:
        cross_time_matrix: shape (tau, tau).
        source_label: Source stock ticker.
        target_label: Target stock ticker.
        title: Optional chart title.

    Returns:
        Plotly Figure.
    """
    tau = cross_time_matrix.shape[0]
    time_labels = [f"t-{tau - 1 - i}" for i in range(tau)]

    fig = go.Figure(data=go.Heatmap(
        z=cross_time_matrix,
        x=time_labels,
        y=time_labels,
        colorscale=[
            [0, COLORS['background']],
            [0.5, _hex_to_rgba(COLORS['info'], 0.4)],
            [1.0, COLORS['primary']],
        ],
        colorbar=dict(
            title=dict(text='Correlation', font=dict(color=_LABEL, size=10)),
            tickfont=dict(color=_TICK, size=9),
        ),
        hovertemplate=f'{target_label}[%{{y}}] ← {source_label}[%{{x}}]: %{{z:.4f}}<extra></extra>',
    ))

    layout = get_chart_layout(title=title, height=400, show_legend=False)
    layout['xaxis'] = _axis(title=f'{source_label} (time)', tickfont=dict(color=_TICK, size=9))
    layout['yaxis'] = _axis(title=f'{target_label} (time)', tickfont=dict(color=_TICK, size=9), autorange='reversed')
    fig.update_layout(**layout)

    return fig


def create_attention_entropy_chart(
    entropy_values: List[float],
    dates: List[str],
    title: str = "",
) -> go.Figure:
    """Time series of inter-stock attention entropy.

    Args:
        entropy_values: Normalized entropy [0, 1] per snapshot.
        dates: Date labels.
        title: Optional chart title.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates,
        y=entropy_values,
        mode='lines',
        line=dict(color=COLORS['primary'], width=1.5),
        fill='tozeroy',
        fillcolor=_hex_to_rgba(COLORS['primary'], 0.08),
        name='Attention Entropy',
    ))

    # Threshold lines
    fig.add_hline(y=0.80, line_color=_hex_to_rgba(COLORS['success'], 0.4),
                  line_width=1, line_dash='dot',
                  annotation_text='Healthy', annotation_position='top left',
                  annotation_font=dict(color=COLORS['success'], size=9))
    fig.add_hline(y=0.50, line_color=_hex_to_rgba(COLORS['danger'], 0.4),
                  line_width=1, line_dash='dot',
                  annotation_text='Crisis', annotation_position='bottom left',
                  annotation_font=dict(color=COLORS['danger'], size=9))

    layout = get_chart_layout(title=title, height=350, show_legend=False)
    layout['yaxis'] = _axis(title='Normalized Entropy', range=[0, 1.05])
    fig.update_layout(**layout)

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    'COLORS',
    'get_chart_layout',
    'create_equity_drawdown_chart',
    'create_rolling_metrics_chart',
    'create_correlation_heatmap',
    'create_tier_sharpe_heatmap',
    'create_risk_return_scatter',
    'create_factor_radar',
    'create_weight_evolution_chart',
    'create_signal_heatmap',
    'create_bar_chart',
    'create_eigenvalue_histogram',
    'create_cleaned_vs_raw_correlation',
    'create_absorption_ratio_chart',
    'create_factor_loading_heatmap',
    'create_spectral_risk_dashboard',
    'create_attention_heatmap',
    'create_cross_time_correlation_chart',
    'create_attention_entropy_chart',
]
