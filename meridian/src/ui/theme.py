"""Meridian design system: tokens -> CSS -> reusable components.

Direction: midnight / graphite surfaces, ONE accent family (cyan -> blue), semantic state colours reserved for risk
tiers and always paired with a text label (never colour alone). Glass blur is used in exactly two places (sidebar and the
page hero). Motion is limited to hover lifts, a focus ring and a progress shimmer, and is disabled under
prefers-reduced-motion. No external fonts or scripts: system font stacks only (works offline and in locked-down networks).
"""
from __future__ import annotations

import html as _html

import streamlit as st

TOKENS = {
    "bg": "#070b14", "surface": "#0d1424", "surface2": "#121b30", "surface3": "#18233d", "line": "#1f2b47", "line_strong": "#2d3d63",
    "text": "#e8eef9", "muted": "#94a6c4", "faint": "#6f819f",
    "accent": "#38bdf8", "accent2": "#60a5fa", "accent_ink": "#04121f", "accent_soft": "rgba(56,189,248,.12)",
    "ok": "#34d399", "warn": "#fbbf24", "high": "#fb923c", "crit": "#f87171", "info": "#60a5fa",
}
TONE = {"LOW": TOKENS["ok"], "MODERATE": TOKENS["warn"], "HIGH": TOKENS["high"], "CRITICAL": TOKENS["crit"],
        "AUTO": TOKENS["ok"], "STRICT": TOKENS["warn"], "REVIEW": TOKENS["high"], "BLOCK": TOKENS["crit"],
        "ok": TOKENS["ok"], "warn": TOKENS["warn"], "high": TOKENS["high"], "crit": TOKENS["crit"], "info": TOKENS["info"],
        "accent": TOKENS["accent"], "muted": TOKENS["muted"]}
SPACE = {"xs": "4px", "sm": "8px", "md": "14px", "lg": "22px", "xl": "34px"}
RADIUS = {"sm": "8px", "md": "12px", "lg": "18px", "pill": "999px"}
FONT_SANS = 'Inter, "SF Pro Text", "Segoe UI", system-ui, -apple-system, Roboto, "Helvetica Neue", Arial, sans-serif'
FONT_MONO = '"JetBrains Mono", "SF Mono", ui-monospace, Menlo, Consolas, monospace'


# --------------------------------------------------------------------------- accessibility helper (also unit-tested)
def _lin(c: float) -> float:
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def luminance(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def contrast_ratio(fg: str, bg: str) -> float:
    a, b = sorted((luminance(fg), luminance(bg)), reverse=True)
    return (a + 0.05) / (b + 0.05)


def esc(x) -> str:
    """Everything that originates outside the app (OSINT text, filenames, vendor names) is escaped before it enters HTML."""
    return _html.escape("" if x is None else str(x), quote=True)


# --------------------------------------------------------------------------- CSS
def css() -> str:
    t = TOKENS
    vars_ = "".join(f"--m-{k.replace('_', '-')}:{v};" for k, v in t.items())
    return f"""<style>
:root{{{vars_}--m-font:{FONT_SANS};--m-mono:{FONT_MONO};--m-r-sm:{RADIUS['sm']};--m-r-md:{RADIUS['md']};--m-r-lg:{RADIUS['lg']};
--m-ease:cubic-bezier(.2,.7,.2,1);}}
html,body,[data-testid="stAppViewContainer"],.stApp{{background:var(--m-bg)!important;color:var(--m-text);font-family:var(--m-font);}}
.stApp{{background:radial-gradient(1100px 520px at 12% -8%,rgba(56,189,248,.10),transparent 60%),radial-gradient(900px 480px at 100% 0%,rgba(96,165,250,.07),transparent 55%),var(--m-bg)!important;}}
[data-testid="stHeader"]{{background:transparent!important}}
.block-container{{max-width:1440px;padding:1.6rem 2rem 4rem}}
h1,h2,h3,h4{{font-family:var(--m-font);letter-spacing:-.01em;color:var(--m-text)}}
h1{{font-weight:750}} h2{{font-size:1.15rem;font-weight:650;margin-top:1.6rem}} h3{{font-size:1rem;font-weight:650}}
p,li,label,span{{color:inherit}} a{{color:var(--m-accent)}} code,pre{{font-family:var(--m-mono)!important}}
hr{{border-color:var(--m-line)!important}}
::selection{{background:rgba(56,189,248,.35)}}
/* ---- sidebar (glass #1) ---- */
[data-testid="stSidebar"]{{background:rgba(10,16,30,.72)!important;backdrop-filter:blur(14px) saturate(130%);-webkit-backdrop-filter:blur(14px) saturate(130%);border-right:1px solid var(--m-line)}}
[data-testid="stSidebar"] .stRadio>div{{gap:2px}}
[data-testid="stSidebar"] .stRadio label{{border-radius:10px;padding:7px 10px;border:1px solid transparent;transition:background .16s var(--m-ease),border-color .16s var(--m-ease);cursor:pointer;width:100%}}
[data-testid="stSidebar"] .stRadio label:hover{{background:var(--m-surface2);border-color:var(--m-line)}}
[data-testid="stSidebar"] .stRadio label:has(input:checked){{background:var(--m-accent-soft);border-color:rgba(56,189,248,.45);color:var(--m-text)}}
[data-testid="stSidebar"] .stRadio label>div:first-child{{display:none}}
.brand{{display:flex;align-items:center;gap:10px;margin:2px 0 6px}}
.brand .mark{{width:28px;height:28px;border-radius:9px;background:conic-gradient(from 210deg,var(--m-accent),var(--m-accent2),#1d4ed8,var(--m-accent));box-shadow:0 0 18px rgba(56,189,248,.35)}}
.brand b{{font-size:1.1rem;letter-spacing:.02em}} .brand span{{color:var(--m-muted);font-size:.72rem;display:block;margin-top:-2px}}
/* ---- focus (keyboard) ---- */
:focus-visible,button:focus-visible,input:focus-visible,[role="tab"]:focus-visible,[data-baseweb="select"]:focus-within{{outline:2px solid var(--m-accent)!important;outline-offset:2px!important;border-radius:8px}}
/* ---- native widgets ---- */
.stButton>button,.stDownloadButton>button{{border-radius:10px;border:1px solid var(--m-line-strong);background:var(--m-surface2);color:var(--m-text);font-weight:600;transition:transform .16s var(--m-ease),border-color .16s,background .16s,box-shadow .16s}}
.stButton>button:hover,.stDownloadButton>button:hover{{border-color:var(--m-accent);background:var(--m-surface3);transform:translateY(-1px)}}
.stButton>button:active{{transform:translateY(0)}}
.stButton>button[kind="primary"]{{background:linear-gradient(135deg,var(--m-accent),var(--m-accent2));color:var(--m-accent-ink);border:0;box-shadow:0 6px 22px rgba(56,189,248,.28)}}
.stButton>button[kind="primary"]:hover{{box-shadow:0 8px 28px rgba(56,189,248,.42);filter:brightness(1.06)}}
.stButton>button:disabled{{opacity:.45;transform:none}}
[data-baseweb="select"]>div,.stTextInput input,.stNumberInput input,.stTextArea textarea{{background:var(--m-surface)!important;border-color:var(--m-line-strong)!important;border-radius:10px!important}}
[data-testid="stMetric"]{{background:var(--m-surface);border:1px solid var(--m-line);border-radius:var(--m-r-md);padding:12px 14px;transition:border-color .16s,transform .16s var(--m-ease)}}
[data-testid="stMetric"]:hover{{border-color:var(--m-line-strong);transform:translateY(-1px)}}
[data-testid="stMetricLabel"]{{color:var(--m-muted)}} [data-testid="stMetricValue"]{{font-weight:700}}
[data-testid="stTabs"] [role="tablist"]{{gap:4px;border-bottom:1px solid var(--m-line)}}
[data-testid="stTabs"] [role="tab"]{{border-radius:8px 8px 0 0;color:var(--m-muted)}} [data-testid="stTabs"] [aria-selected="true"]{{color:var(--m-text)}}
[data-testid="stExpander"]{{border:1px solid var(--m-line);border-radius:var(--m-r-md);background:var(--m-surface)}}
[data-testid="stDataFrame"]{{border:1px solid var(--m-line);border-radius:var(--m-r-md);overflow:hidden}}
[data-testid="stAlert"]{{border-radius:var(--m-r-md);border:1px solid var(--m-line)}}
.stProgress>div>div>div>div{{background:linear-gradient(90deg,var(--m-accent),var(--m-accent2));}}
/* ---- components ---- */
.m-hero{{position:relative;border:1px solid var(--m-line);border-radius:var(--m-r-lg);padding:22px 26px;margin:0 0 18px;overflow:hidden;
 background:linear-gradient(135deg,rgba(18,27,48,.78),rgba(13,20,36,.62));backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px)}}
.m-hero:before{{content:"";position:absolute;inset:0 0 auto 0;height:2px;background:linear-gradient(90deg,transparent,var(--m-accent),var(--m-accent2),transparent);opacity:.8}}
.m-kicker{{font-size:.72rem;letter-spacing:.14em;text-transform:uppercase;color:var(--m-accent);font-weight:700}}
.m-title{{font-size:1.85rem;font-weight:760;letter-spacing:-.02em;margin:4px 0 4px;line-height:1.15}}
.m-sub{{color:var(--m-muted);font-size:.95rem;max-width:78ch;line-height:1.5}}
.m-card{{background:var(--m-surface);border:1px solid var(--m-line);border-radius:var(--m-r-md);padding:14px 16px;margin-bottom:12px;transition:border-color .18s,transform .18s var(--m-ease),box-shadow .18s}}
.m-card.lift:hover{{border-color:var(--m-line-strong);transform:translateY(-2px);box-shadow:0 10px 30px rgba(0,0,0,.35)}}
.m-card h4{{margin:0;font-size:1rem;font-weight:650}} .m-card .m-sub2{{color:var(--m-muted);font-size:.78rem;margin:2px 0 10px}}
.m-kv{{display:flex;justify-content:space-between;gap:12px;font-size:.86rem;padding:3px 0;color:var(--m-muted)}} .m-kv b{{color:var(--m-text);font-weight:600;text-align:right}}
.m-bar{{height:7px;border-radius:6px;background:var(--m-line);overflow:hidden;margin:6px 0 8px}} .m-bar>i{{display:block;height:100%;border-radius:6px;transition:width .5s var(--m-ease)}}
.m-badge{{display:inline-flex;align-items:center;gap:6px;white-space:nowrap;font-weight:700;letter-spacing:.05em;font-size:.72rem;padding:2px 10px;border-radius:999px;border:1px solid;background:rgba(255,255,255,.02)}}
.m-badge:before{{content:"";width:6px;height:6px;border-radius:50%;background:currentColor}}
.m-kpi{{background:var(--m-surface);border:1px solid var(--m-line);border-radius:var(--m-r-md);padding:14px 16px;position:relative;overflow:hidden}}
.m-kpi .l{{color:var(--m-muted);font-size:.74rem;letter-spacing:.06em;text-transform:uppercase}} .m-kpi .v{{font-size:1.9rem;font-weight:750;line-height:1.15;margin-top:2px}} .m-kpi .d{{font-size:.78rem;color:var(--m-muted);margin-top:2px}}
.m-kpi:after{{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--tone,var(--m-accent))}}
.m-verdict{{border:1px solid var(--tone);border-radius:var(--m-r-lg);padding:18px 22px;background:linear-gradient(135deg,color-mix(in srgb,var(--tone) 14%,var(--m-surface)),var(--m-surface))}}
.m-verdict .big{{font-size:2rem;font-weight:800;letter-spacing:.04em;color:var(--tone)}} .m-verdict .s{{color:var(--m-muted)}}
.m-callout{{border:1px solid var(--m-line);border-left:3px solid var(--tone);border-radius:var(--m-r-md);padding:10px 14px;background:var(--m-surface);margin:8px 0;font-size:.9rem;color:var(--m-text)}}
.m-empty{{border:1px dashed var(--m-line-strong);border-radius:var(--m-r-lg);padding:30px;text-align:center;color:var(--m-muted);background:rgba(13,20,36,.5)}} .m-empty b{{display:block;color:var(--m-text);font-size:1.05rem;margin-bottom:4px}}
.m-src{{display:grid;grid-template-columns:auto 1fr auto;gap:4px 12px;align-items:center;padding:10px 14px;border:1px solid var(--m-line);border-radius:var(--m-r-md);background:var(--m-surface);margin-bottom:8px}}
.m-src .u{{font-family:var(--m-mono);font-size:.78rem;color:var(--m-muted);overflow-wrap:anywhere}} .m-src .n{{font-weight:600}}
.m-step{{display:flex;gap:10px;align-items:center;color:var(--m-muted);font-size:.82rem;margin:2px 0 12px}} .m-step b{{color:var(--m-text)}} .m-step .dot{{width:22px;height:22px;border-radius:50%;display:inline-grid;place-items:center;font-size:.72rem;font-weight:700;background:var(--m-surface2);border:1px solid var(--m-line-strong)}}
.m-step .dot.on{{background:var(--m-accent);color:var(--m-accent-ink);border-color:var(--m-accent)}}
.m-quote{{font-size:.88rem;color:var(--m-text);border-left:2px solid var(--m-line-strong);padding:2px 0 2px 10px;margin:4px 0;overflow-wrap:anywhere}}
.m-tag{{font-size:.7rem;padding:1px 8px;border-radius:6px;background:var(--m-surface2);color:var(--m-muted);border:1px solid var(--m-line)}}
.m-mono{{font-family:var(--m-mono);font-size:.78rem;color:var(--m-muted)}}
.m-shimmer{{background:linear-gradient(90deg,var(--m-line),var(--m-surface3),var(--m-line));background-size:200% 100%;animation:m-sh 1.6s linear infinite;border-radius:6px;height:8px}}
@keyframes m-sh{{to{{background-position:-200% 0}}}}
.m-sr{{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}}
/* ---- responsive ---- */
@media (max-width:900px){{.block-container{{padding:1rem 1rem 3rem}} .m-title{{font-size:1.45rem}} .m-hero{{padding:16px}} .m-kpi .v{{font-size:1.5rem}} .m-src{{grid-template-columns:1fr}}}}
@media (max-width:640px){{[data-testid="stHorizontalBlock"]{{flex-wrap:wrap!important}} [data-testid="stHorizontalBlock"]>[data-testid="stColumn"]{{min-width:100%!important}}}}
/* ---- reduced motion ---- */
@media (prefers-reduced-motion:reduce){{*,*:before,*:after{{animation:none!important;transition:none!important;scroll-behavior:auto!important}}
 .m-card.lift:hover,.stButton>button:hover,[data-testid="stMetric"]:hover{{transform:none!important}}}}
@media (forced-colors:active){{.m-badge,.m-kpi,.m-card,.m-verdict{{border:1px solid CanvasText}}}}
</style>"""


def inject() -> None:
    st.markdown(css(), unsafe_allow_html=True)


# --------------------------------------------------------------------------- components (return HTML; render() draws it)
def render(html: str) -> None:
    st.markdown(html, unsafe_allow_html=True)


def badge(text: str, tone: str = "accent") -> str:
    c = TONE.get(tone, tone)
    return f'<span class="m-badge" style="color:{c};border-color:{c}">{esc(text)}</span>'


def page_header(title: str, sub: str = "", kicker: str = "") -> None:
    render(f'<header class="m-hero" role="banner">{f"<div class=m-kicker>{esc(kicker)}</div>" if kicker else ""}<h1 class="m-title">{esc(title)}</h1>'
           f'{f"<div class=m-sub>{esc(sub)}</div>" if sub else ""}</header>')


def kpi(label: str, value, detail: str = "", tone: str = "accent") -> str:
    return f'<div class="m-kpi" style="--tone:{TONE.get(tone, tone)}"><div class="l">{esc(label)}</div><div class="v">{esc(value)}</div><div class="d">{esc(detail)}</div></div>'


def kpi_row(items: list[tuple]) -> None:
    cols = st.columns(len(items))
    for c, it in zip(cols, items):
        c.markdown(kpi(*it), unsafe_allow_html=True)


def callout(text: str, tone: str = "info") -> None:
    render(f'<div class="m-callout" role="note" style="--tone:{TONE.get(tone, tone)}">{esc(text)}</div>')


def empty_state(title: str, text: str) -> None:
    render(f'<div class="m-empty"><b>{esc(title)}</b>{esc(text)}</div>')


def meter(frac: float, tone: str = "accent", label: str = "") -> str:
    f = max(0.0, min(1.0, float(frac)))
    return (f'<div class="m-bar" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="{f * 100:.0f}" aria-label="{esc(label)}">'
            f'<i style="width:{f * 100:.0f}%;background:{TONE.get(tone, tone)}"></i></div>')


def kv(rows: list[tuple]) -> str:
    return "".join(f'<div class="m-kv"><span>{esc(a)}</span><b>{b if isinstance(b, _Raw) else esc(b)}</b></div>' for a, b in rows)


class _Raw(str):
    """Marks pre-built, already-escaped HTML (e.g. a badge) for kv()."""


def raw(html: str) -> _Raw:
    return _Raw(html)


def card(inner: str, tone: str | None = None, lift: bool = True) -> str:
    style = f' style="border-top:3px solid {TONE.get(tone, tone)}"' if tone else ""
    return f'<div class="m-card{" lift" if lift else ""}"{style}>{inner}</div>'


def verdict(label: str, sub: str, tone: str) -> None:
    render(f'<div class="m-verdict" role="status" style="--tone:{TONE.get(tone, tone)}"><div class="big">{esc(label)}</div><div class="s">{esc(sub)}</div></div>')


def stepper(steps: list[str], active: int) -> None:
    render('<div class="m-step" role="list">' + " &nbsp;›&nbsp; ".join(
        f'<span role="listitem"><span class="dot{" on" if i == active else ""}">{i + 1}</span> <b>{esc(s)}</b></span>' if i == active
        else f'<span role="listitem"><span class="dot">{i + 1}</span> {esc(s)}</span>' for i, s in enumerate(steps)) + "</div>")


def brand() -> None:
    render('<div class="brand"><div class="mark" aria-hidden="true"></div><div><b>Meridian</b><span>Third-party AI exposure control</span></div></div>')


def plotly_layout(fig, height=380):
    t = TOKENS
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=t["text"], size=12, family=FONT_SANS), legend=dict(orientation="h", y=-0.15), colorway=[t["accent"], t["accent2"], t["warn"], t["high"], t["crit"]])
    return fig
