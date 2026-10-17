"""Plotly figures. Tier colours are always paired with an emoji/text label in the UI (never colour-only)."""
import math
import plotly.graph_objects as go
from .config import BRAKE, EMOJI
from .dna import CATEGORIES

TIER_COLOR = {'LOW': '#34d399', 'MODERATE': '#fbbf24', 'HIGH': '#fb923c', 'CRITICAL': '#f87171'}
KIND_STYLE = {'source': ('#94a6c4', 'Meridian'), 'vendor': ('#38bdf8', 'Vendor'), 'service': ('#60a5fa', 'Subprocessor'),
              'ai': ('#f87171', 'AI / LLM'), 'unknown': ('#6f819f', 'Unknown destination')}
TXT, GRID = '#e8eef9', '#1f2b47'


def _base(fig, h=380):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font=dict(color=TXT, size=12, family='Inter, system-ui, Segoe UI, Roboto, sans-serif'), legend=dict(orientation='h', y=-0.15))
    return fig


def _layout(G):
    depth = {}
    from collections import deque
    q = deque([n for n, d in G.nodes(data=True) if d.get('kind') == 'source']); [depth.setdefault(n, 0) for n in q]
    while q:
        u = q.popleft()
        for v in G.successors(u):
            if v not in depth:
                depth[v] = depth[u] + 1; q.append(v)
    cols = {}
    for n, d in depth.items(): cols.setdefault(d, []).append(n)
    pos = {}
    for d, ns in cols.items():
        for i, n in enumerate(sorted(ns)):
            pos[n] = (d, (len(ns) - 1) / 2 - i)
    return pos


def twin_figure(G, path=None):
    """Meridian -> Vendor -> Subprocessor -> AI, with edge probabilities printed on the edges."""
    pos = _layout(G); fig = go.Figure(); on_path = set(zip(path, path[1:])) if path else set()
    for u, v, d in G.edges(data=True):
        (x0, y0), (x1, y1) = pos[u], pos[v]; hot = (u, v) in on_path
        dashed = G.nodes[v].get('kind') == 'unknown'
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', hoverinfo='skip', showlegend=False,
                                 line=dict(width=1.5 + 5 * d['p'], color='#f59e0b' if hot else '#475569', dash='dot' if dashed else 'solid')))
        fig.add_annotation(x=x1 - (x1 - x0) * 0.18, y=y1 - (y1 - y0) * 0.18, ax=x0 + (x1 - x0) * 0.2, ay=y0 + (y1 - y0) * 0.2, xref='x', yref='y',
                           axref='x', ayref='y', showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=1.5, arrowcolor='#f59e0b' if hot else '#64748b')
        fig.add_annotation(x=(x0 + x1) / 2, y=(y0 + y1) / 2, text=f"<b>p = {d['p']:.2f}</b>", showarrow=False, font=dict(size=12, color='#0b1220'),
                           bgcolor='#fbbf24' if hot else '#cbd5e1', bordercolor='#0b1220', borderpad=3, opacity=0.95)
    for kind, (color, label) in KIND_STYLE.items():
        ns = [n for n in G if G.nodes[n].get('kind') == kind]
        if not ns: continue
        fig.add_trace(go.Scatter(x=[pos[n][0] for n in ns], y=[pos[n][1] for n in ns], mode='markers+text', name=label, text=ns, textposition='bottom center',
                                 marker=dict(size=34, color=color, line=dict(width=2, color='#0b1220'), symbol='diamond' if kind == 'ai' else 'circle'),
                                 hovertext=[f'{n} ({label})' for n in ns], hoverinfo='text'))
    xs = [p[0] for p in pos.values()]
    fig.update_xaxes(visible=False, range=[min(xs) - 0.6, max(xs) + 0.6]); fig.update_yaxes(visible=False, range=[min(p[1] for p in pos.values()) - 1, max(p[1] for p in pos.values()) + 1])
    return _base(fig, 420)


def dna_radar(before, after=None, title=''):
    fig = go.Figure(); cats = CATEGORIES + [CATEGORIES[0]]
    fig.add_trace(go.Scatterpolar(r=[before[c] * 100 for c in CATEGORIES] + [before[CATEGORIES[0]] * 100], theta=cats, fill='toself', name='Original document',
                                  line=dict(color='#ef4444'), fillcolor='rgba(239,68,68,0.20)'))
    if after is not None:
        fig.add_trace(go.Scatterpolar(r=[after[c] * 100 for c in CATEGORIES] + [after[CATEGORIES[0]] * 100], theta=cats, fill='toself', name='After redaction (exposed)',
                                      line=dict(color='#38bdf8'), fillcolor='rgba(56,189,248,0.25)'))
    fig.update_layout(polar=dict(bgcolor='rgba(0,0,0,0)', radialaxis=dict(range=[0, 100], ticksuffix='%', gridcolor=GRID), angularaxis=dict(gridcolor=GRID)), title=title)
    return _base(fig, 380)


def waterfall(bd):
    """BASE -> after P_ai -> after criticality -> after history = final allowance."""
    a = bd['base']; b = a * bd['p_ai_factor']; c = b * bd['criticality_factor']; d = c * bd['history_factor']
    fig = go.Figure(go.Waterfall(x=['BASE', f"AI exposure ({bd['p_ai']:.0%})", f"Criticality ({bd['criticality']})", 'Historical exposure', 'Final allowance'],
                                 measure=['absolute', 'relative', 'relative', 'relative', 'total'], y=[a, b - a, c - b, d - c, d],
                                 text=[f'{v:,.0f}' for v in (a, b - a, c - b, d - c, d)], textposition='outside',
                                 decreasing=dict(marker=dict(color='#f97316')), totals=dict(marker=dict(color='#38bdf8')), increasing=dict(marker=dict(color='#22c55e')),
                                 connector=dict(line=dict(color=GRID))))
    fig.update_yaxes(gridcolor=GRID, title='points'); return _base(fig, 340)


def forecast_fig(ttf, allowance):
    xs, ys = zip(*ttf['forecast']); fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', fill='tozeroy', name='Projected balance' if ttf['is_projection'] else 'Forecast balance (from history)',
                             line=dict(color='#38bdf8', dash='dash' if ttf['is_projection'] else 'solid'), fillcolor='rgba(56,189,248,0.15)'))
    for frac, name, col in ((BRAKE['AUTO'], 'AUTO floor', '#eab308'), (BRAKE['STRICT'], 'STRICT floor', '#f97316'), (BRAKE['REVIEW'], 'BLOCK floor', '#ef4444')):
        fig.add_hline(y=frac * allowance, line=dict(color=col, dash='dot', width=1), annotation_text=name, annotation_position='top right', annotation_font_color=col)
    fig.update_xaxes(title='days from today', gridcolor=GRID); fig.update_yaxes(title='leak budget (points)', gridcolor=GRID, rangemode='tozero'); return _base(fig, 340)


def debt_fig(series, base):
    fig = go.Figure()
    if series:
        m, s, c = zip(*series)
        fig.add_trace(go.Bar(x=m, y=s, name='Spend in month', marker_color='#38bdf8'))
        fig.add_trace(go.Scatter(x=m, y=c, name='Cumulative privacy debt', mode='lines+markers', line=dict(color='#f97316', width=3)))
        for mult, name in ((0.5, 'MODERATE'), (1.0, 'HIGH'), (2.0, 'CRITICAL')):
            fig.add_hline(y=mult * base, line=dict(color=TIER_COLOR[name], dash='dot', width=1), annotation_text=f'{EMOJI[name]} {name} at {mult:g}x BASE', annotation_position='top left')
    fig.update_yaxes(gridcolor=GRID, title='points'); return _base(fig, 340)


def blast_fig(b):
    items = [('Direct vendors', b['direct_vendors'], '#38bdf8'), ('Subprocessors', b['subprocessors'], '#a78bfa'), ('AI systems', b['ai_systems'], '#ef4444'),
             ('External AI providers', b['external_ai_providers'], '#f43f5e'), ('Unknown destinations', b['unknown_destinations'], '#64748b')]
    fig = go.Figure(go.Bar(y=[i[0] for i in items][::-1], x=[i[1] for i in items][::-1], orientation='h', marker_color=[i[2] for i in items][::-1],
                           text=[i[1] for i in items][::-1], textposition='outside'))
    fig.update_xaxes(gridcolor=GRID, dtick=1, rangemode='tozero'); return _base(fig, 250)
