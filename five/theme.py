"""Dashboard chrome: CSS shell and small HTML component builders."""
from __future__ import annotations

import numpy as np

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
  --bg:#080d16;
  --card:linear-gradient(135deg, rgba(17,26,44,.85) 0%, rgba(13,20,35,.85) 100%);
  --border:rgba(140,165,215,.18);
  --cyan:#3fd8c9; --blue:#4f9cf9; --amber:#f2b544;
  --green:#2fd08c; --red:#ff5d6c; --violet:#b7a4f3;
  --ink:#eef3ff; --muted:#7f93b8;
}

html, body, [class*="css"] { font-family:'Plus Jakarta Sans', system-ui, sans-serif; }

.stApp {
  background:
    radial-gradient(1400px 900px at 85% -15%, rgba(79,156,249,.12), transparent 60%),
    radial-gradient(1100px 750px at -10% 110%, rgba(242,181,68,.08), transparent 55%),
    radial-gradient(900px 600px at 50% 115%, rgba(63,216,201,.06), transparent 60%),
    var(--bg);
}
.stApp::before {
  content:""; position:fixed; inset:0; pointer-events:none; z-index:0;
  background-image:linear-gradient(rgba(120,150,210,.04) 1px, transparent 1px),
                   linear-gradient(90deg, rgba(120,150,210,.04) 1px, transparent 1px);
  background-size:36px 36px;
  mask-image:radial-gradient(1300px 850px at 50% 0%, black 40%, transparent 90%);
}
::-webkit-scrollbar { width:7px; height:7px; }
::-webkit-scrollbar-track { background:rgba(9,14,23,.8); }
::-webkit-scrollbar-thumb { background:rgba(140,165,215,.25); border-radius:4px; }
::-webkit-scrollbar-thumb:hover { background:var(--cyan); }

/* ---------- header ---------- */
.fve-head {
  display:flex; align-items:center; justify-content:space-between; gap:16px;
  flex-wrap:wrap; margin:0 0 14px; padding:16px 20px;
  background:linear-gradient(135deg, rgba(17,26,44,.75) 0%, rgba(13,20,35,.75) 100%);
  border:1px solid var(--border); border-radius:14px;
  backdrop-filter:blur(12px); box-shadow:0 8px 32px rgba(0,0,0,.37);
}
.fve-brand {
  font-family:'Space Grotesk', sans-serif; font-weight:800; font-size:27px;
  letter-spacing:-.02em; color:var(--ink); line-height:1;
  display:flex; align-items:center; gap:10px;
}
.fve-brand em { font-style:normal; color:var(--amber); text-shadow:0 0 12px rgba(242,181,68,.4); }
.fve-sub {
  font-family:'JetBrains Mono', monospace; font-size:10.5px; letter-spacing:.14em;
  color:var(--muted); text-transform:uppercase; margin-top:7px;
}
.fve-chips { display:flex; gap:8px; flex-wrap:wrap; }
.chip {
  font-family:'JetBrains Mono', monospace; font-size:10.5px; letter-spacing:.08em;
  color:#a9bcdd; border:1px solid rgba(140,165,215,.22); background:rgba(18,28,46,.75);
  padding:6px 12px; border-radius:8px; display:inline-flex; align-items:center; gap:8px;
  transition:all .25s ease;
}
.chip:hover { border-color:rgba(63,216,201,.5); color:#dff6f2; transform:translateY(-1px); }
.chip b { color:var(--ink); }
.dot {
  width:7px; height:7px; border-radius:50%; background:var(--green);
  box-shadow:0 0 0 0 rgba(47,208,140,.6); animation:pulse 1.8s infinite;
}
.dot.sim { background:var(--amber); box-shadow:0 0 0 0 rgba(242,181,68,.6); }
@keyframes pulse { 70%{box-shadow:0 0 0 8px rgba(47,208,140,0);} 100%{box-shadow:0 0 0 0 rgba(47,208,140,0);} }

/* ---------- ticker tape ---------- */
.tape-wrap {
  overflow:hidden; border-top:1px solid rgba(140,165,215,.14);
  border-bottom:1px solid rgba(140,165,215,.14); margin:0 0 18px; padding:9px 0;
  background:rgba(13,20,35,.45);
  mask-image:linear-gradient(90deg, transparent, black 4%, black 96%, transparent);
}
.tape {
  display:inline-flex; gap:34px; white-space:nowrap; animation:tape 46s linear infinite;
  font-family:'JetBrains Mono', monospace; font-size:11.5px;
}
.tape-wrap:hover .tape { animation-play-state:paused; }
@keyframes tape { from{transform:translateX(0)} to{transform:translateX(-50%)} }
.tape b { color:#c9d7f2; font-weight:600; }
.tape .up { color:var(--green); background:rgba(47,208,140,.12); padding:2px 6px; border-radius:4px; }
.tape .dn { color:var(--red);   background:rgba(255,93,108,.12); padding:2px 6px; border-radius:4px; }

/* ---------- KPI cards ---------- */
.kpis { display:grid; grid-template-columns:repeat(4,1fr); gap:13px; margin-bottom:13px; }
@media (max-width:1100px){ .kpis { grid-template-columns:repeat(2,1fr); } }
.kpi {
  background:var(--card); border:1px solid var(--border); border-radius:12px;
  padding:15px 17px 14px; position:relative; overflow:hidden;
  transition:transform .25s cubic-bezier(.16,1,.3,1), border-color .25s, box-shadow .25s;
}
.kpi:hover {
  transform:translateY(-3px); border-color:rgba(63,216,201,.45);
  box-shadow:0 12px 32px -12px rgba(63,216,201,.3);
}
.kpi .lab {
  font-family:'JetBrains Mono', monospace; font-size:9.5px; letter-spacing:.15em;
  color:var(--muted); text-transform:uppercase; margin-bottom:8px; font-weight:500;
}
.kpi .val {
  font-family:'Space Grotesk', sans-serif; font-size:27px; font-weight:700;
  color:var(--ink); line-height:1.05; letter-spacing:-.01em;
}
.kpi .aux { font-size:11.5px; margin-top:8px; color:#93a7c9; line-height:1.45; }
.kpi .aux .up{color:var(--green);font-weight:600}
.kpi .aux .dn{color:var(--red);font-weight:600}
.kpi .aux .am{color:var(--amber);font-weight:600}
.kpi.b-amber{border-top:3px solid var(--amber)} .kpi.b-cyan{border-top:3px solid var(--cyan)}
.kpi.b-green{border-top:3px solid var(--green)} .kpi.b-red{border-top:3px solid var(--red)}
.kpi.b-blue{border-top:3px solid var(--blue)}   .kpi.b-violet{border-top:3px solid var(--violet)}
.meter { height:6px; border-radius:3px; background:rgba(140,165,215,.15); margin-top:10px; overflow:hidden; }
.meter i { display:block; height:100%; border-radius:3px; transition:width .8s cubic-bezier(.16,1,.3,1); }
.pill {
  display:inline-block; font-family:'JetBrains Mono', monospace; font-size:12px;
  letter-spacing:.09em; padding:5px 13px; border-radius:20px; font-weight:600;
}

/* ---------- section headers ---------- */
.kick {
  font-family:'JetBrains Mono', monospace; font-size:10px; letter-spacing:.22em;
  color:var(--amber); text-transform:uppercase; font-weight:600;
}
.title {
  font-family:'Space Grotesk', sans-serif; font-size:21px; font-weight:700;
  color:var(--ink); margin:3px 0 4px; letter-spacing:-.01em;
}
.desc { font-size:12.5px; color:#8ba0c4; margin-bottom:14px; line-height:1.6; max-width:110ch; }
.note {
  font-size:11.5px; color:#8ba0c4; line-height:1.6; border-left:2px solid rgba(63,216,201,.4);
  padding:2px 0 2px 12px; margin:10px 0 4px;
}

/* ---------- streamlit chrome ---------- */
[data-testid="stSidebar"] {
  background:rgba(10,16,28,.94) !important; border-right:1px solid var(--border) !important;
}
[data-testid="stTabs"] [data-baseweb="tab-list"] {
  gap:7px; border-bottom:1px solid rgba(140,165,215,.18); margin-bottom:16px;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
  background:rgba(18,28,46,.4); border-radius:10px 10px 0 0; padding:9px 17px;
  font-family:'Space Grotesk', sans-serif; font-size:13px; font-weight:600;
  color:#93a7c9; border:1px solid transparent; border-bottom:none; transition:all .2s ease;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover { color:var(--ink); background:rgba(63,216,201,.05); }
[data-testid="stTabs"] [aria-selected="true"] {
  background:rgba(63,216,201,.12) !important; color:var(--ink) !important;
  border-color:rgba(63,216,201,.3) !important;
  box-shadow:inset 0 -3px 0 var(--cyan) !important;
}
.stButton > button {
  background:linear-gradient(135deg,#173b60 0%,#0e243d 100%) !important; color:#dff6f2 !important;
  border:1px solid rgba(63,216,201,.45) !important; border-radius:10px !important;
  font-family:'Space Grotesk', sans-serif !important; font-size:13.5px !important;
  font-weight:700 !important; letter-spacing:.06em !important; padding:10px 18px !important;
  transition:all .25s ease !important; width:100%;
}
.stButton > button:hover {
  border-color:var(--cyan) !important; box-shadow:0 0 24px -2px rgba(63,216,201,.5) !important;
  transform:translateY(-2px) !important; color:#fff !important;
}
div[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input {
  background:rgba(18,28,46,.9) !important; border-color:rgba(140,165,215,.25) !important;
  border-radius:8px !important; color:var(--ink) !important;
}
[data-testid="stMetric"] { background:rgba(17,26,44,.5); border:1px solid var(--border);
  border-radius:10px; padding:12px 14px; }
footer { visibility:hidden; }
#MainMenu { visibility:hidden; }
</style>
"""


def header(n_assets: int, n_classes: int, source: str, asof, version: str = "3.0") -> str:
    live = source != "SIMULATED"
    return f"""
<div class="fve-head">
  <div>
    <div class="fve-brand">FVE <em>//</em> FAIR VALUE ENGINE</div>
    <div class="fve-sub">Cross-sectional market-relative valuation · latent factor state model · v{version}</div>
  </div>
  <div class="fve-chips">
    <span class="chip"><span class="dot{'' if live else ' sim'}"></span>{source}</span>
    <span class="chip">UNIVERSE&nbsp;<b>{n_assets}</b>&nbsp;INSTRUMENTS</span>
    <span class="chip"><b>{n_classes}</b>&nbsp;ASSET CLASSES</span>
    <span class="chip">AS OF&nbsp;<b>{asof:%d %b %Y}</b></span>
  </div>
</div>"""


def section(kick: str, title: str, desc: str = "") -> str:
    body = f"<div class='desc'>{desc}</div>" if desc else ""
    return f"<div class='kick'>{kick}</div><div class='title'>{title}</div>{body}"


def kpi(label: str, value: str, aux: str = "", accent: str = "", value_color: str = "",
        meter: float | None = None, meter_color: str = "#3fd8c9") -> str:
    cls = f"kpi b-{accent}" if accent else "kpi"
    vstyle = f" style='color:{value_color}'" if value_color else ""
    bar = ""
    if meter is not None and np.isfinite(meter):
        bar = (f"<div class='meter'><i style='width:{np.clip(meter, 0, 100):.0f}%;"
               f"background:{meter_color}'></i></div>")
    aux_html = f"<div class='aux'>{aux}</div>" if aux else ""
    return (f"<div class='{cls}'><div class='lab'>{label}</div>"
            f"<div class='val'{vstyle}>{value}</div>{bar}{aux_html}</div>")


def kpi_row(cards: list[str]) -> str:
    return f"<div class='kpis'>{''.join(cards)}</div>"


def tape(items: list[tuple[str, float]]) -> str:
    """Scrolling ticker tape from (label, pct change) pairs."""
    if not items:
        return ""
    html = ""
    for label, chg in items:
        cls = "up" if chg >= 0 else "dn"
        arrow = "▲" if chg >= 0 else "▼"
        html += f"<span><b>{label}</b> <span class='{cls}'>{arrow} {chg:+.2f}%</span></span>"
    return f"<div class='tape-wrap'><div class='tape'>{html}{html}</div></div>"
