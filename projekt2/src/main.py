"""
================================================================
Multivariate Zeitreihenanalyse – Streamlit Dashboard
================================================================
Datensatz  : Global Fuel Prices 2020–2026 (Kaggle)
Länder     : France, Germany, Indonesia, Italy
Variable   : petrol_usd_liter  →  Fuel Price Index
Methode    : Box-Jenkins → ARIMA (automatisiert)

Ausführen
---------
    streamlit run src/main.py

Tabs
----
    1 · Übersicht      – alle Zeitreihen + KPI-Karten
    2 · Einzelanalyse  – Box-Jenkins-Schritte pro Land
    3 · Modellvergleich – Metriken aller Länder
    4 · Prognose       – 10-Wochen-Forecast (Apr–Jun 2026)
================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os
import itertools
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from statsmodels.tsa.stattools    import adfuller, acf, pacf
from statsmodels.tsa.arima.model  import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox


# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Fuel Price ARIMA Dashboard",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════
# DESIGN SYSTEM
# ══════════════════════════════════════════════════════════════

BG      = "#F8FAFC"
SURFACE = "#FFFFFF"
BORDER  = "#E2E8F0"
TEXT    = "#0F172A"
TEXT2   = "#475569"
MUTED   = "#94A3B8"
ACCENT  = "#2563EB"
POS     = "#059669"
NEG     = "#DC2626"

LAENDER = {
    "France":    {"farbe": "#1D4ED8", "flag": "🇫🇷"},
    "Germany":   {"farbe": "#B91C1C", "flag": "🇩🇪"},
    "Indonesia": {"farbe": "#15803D", "flag": "🇮🇩"},
    "Italy":     {"farbe": "#C2410C", "flag": "🇮🇹"},
}

DATENPFAD = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "raw", "global_fuel_prices_2020_2026.csv",
)

_FONT = "'Plus Jakarta Sans','Inter',sans-serif"
_GRID = "#F1F5F9"
_AXIS = dict(
    gridcolor=_GRID, gridwidth=1,
    zerolinecolor=BORDER, linecolor=BORDER,
    tickfont=dict(color=MUTED, size=11),
)
CHART_DEFAULTS = dict(
    paper_bgcolor=SURFACE,
    plot_bgcolor="#FAFBFD",
    font=dict(family=_FONT, color=TEXT, size=12),
    margin=dict(l=10, r=10, t=52, b=10),
    hoverlabel=dict(
        bgcolor=SURFACE, bordercolor=BORDER,
        font=dict(family=_FONT, size=12, color=TEXT),
    ),
    legend=dict(
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor=BORDER, borderwidth=1,
        font=dict(size=11, color=TEXT2),
        orientation="h",
        yanchor="bottom", y=-0.22,
        xanchor="center", x=0.5,
    ),
)


# ══════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════

def _inject_css() -> None:
    st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

    <style>
    *, *::before, *::after { box-sizing: border-box; }

    /* ── APP ────────────────────────────────────── */
    .stApp { background: #F8FAFC !important; font-family: 'Plus Jakarta Sans','Inter',sans-serif !important; }
    .block-container { padding-top: 1.75rem !important; padding-bottom: 3rem !important; max-width: 1380px !important; }

    /* ── SIDEBAR ────────────────────────────────── */
    [data-testid="stSidebar"] { background: #FFFFFF !important; border-right: 1px solid #E2E8F0 !important; }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem !important; }

    /* ── HEADER ─────────────────────────────────── */
    .dash-header { padding-bottom: 1.4rem; border-bottom: 1px solid #E2E8F0; margin-bottom: 1.75rem; }
    .dash-title  { font-size: 1.55rem; font-weight: 800; color: #0F172A; letter-spacing: -0.5px; margin: 0; }
    .dash-title em { font-style: normal; color: #2563EB; }
    .dash-sub    { font-size: 0.79rem; color: #94A3B8; margin-top: 0.3rem; }
    .dash-pills  { display: flex; gap: 0.45rem; margin-top: 0.7rem; flex-wrap: wrap; }
    .dash-pill   {
        display: inline-flex; align-items: center;
        padding: 0.22rem 0.65rem;
        background: #EFF6FF; border: 1px solid #BFDBFE;
        border-radius: 999px; font-size: 0.68rem; font-weight: 600;
        color: #1D4ED8; text-transform: uppercase; letter-spacing: 0.4px;
    }

    /* ── KPI CARDS ──────────────────────────────── */
    .kpi-row { display: grid; grid-template-columns: repeat(4,1fr); gap: 1rem; margin-bottom: 1.75rem; }
    .kpi-card {
        background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px;
        padding: 1.2rem 1.4rem; position: relative; overflow: hidden;
        transition: box-shadow 0.2s, transform 0.15s;
    }
    .kpi-card:hover { box-shadow: 0 6px 20px rgba(0,0,0,0.07); transform: translateY(-2px); }
    .kpi-card::after {
        content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 3px;
        background: var(--c, #2563EB); border-radius: 0 0 14px 14px;
    }
    .kpi-top   { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.55rem; }
    .kpi-flag  { font-size: 1.4rem; }
    .kpi-badge { font-size: 0.67rem; font-weight: 700; padding: 0.18rem 0.5rem; border-radius: 999px; text-transform: uppercase; letter-spacing: 0.4px; }
    .kpi-pos   { background: #ECFDF5; color: #059669; }
    .kpi-neg   { background: #FEF2F2; color: #DC2626; }
    .kpi-label { font-size: 0.67rem; font-weight: 600; color: #94A3B8; text-transform: uppercase; letter-spacing: 1px; }
    .kpi-value { font-size: 1.7rem; font-weight: 800; color: #0F172A; font-family: 'JetBrains Mono',monospace; letter-spacing: -0.5px; line-height: 1.1; }
    .kpi-sub   { font-size: 0.71rem; color: #94A3B8; margin-top: 0.18rem; }

    /* ── SECTION HEADER ─────────────────────────── */
    .sec-eye  { font-size: 0.66rem; font-weight: 700; color: #2563EB; text-transform: uppercase; letter-spacing: 1.8px; margin-bottom: 0.18rem; }
    .sec-h2   { font-size: 1.12rem; font-weight: 700; color: #0F172A; letter-spacing: -0.3px; margin-bottom: 0.18rem; }
    .sec-desc { font-size: 0.82rem; color: #64748B; line-height: 1.6; margin-bottom: 1.1rem; }

    /* ── BEST MODEL CARD ────────────────────────── */
    .best {
        background: linear-gradient(135deg, #EFF6FF, #F8FAFF);
        border: 1px solid #BFDBFE; border-radius: 14px;
        padding: 1.15rem 1.4rem; margin-bottom: 1.4rem;
    }
    .best-eye  { font-size: 0.66rem; font-weight: 700; color: #2563EB; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 0.28rem; }
    .best-name { font-size: 1.5rem; font-weight: 800; color: #1D4ED8; font-family: 'JetBrains Mono',monospace; letter-spacing: -0.5px; margin-bottom: 0.65rem; }
    .best-row  { display: flex; gap: 1.75rem; flex-wrap: wrap; }
    .bm        { display: flex; flex-direction: column; }
    .bm-l      { font-size: 0.62rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 600; }
    .bm-v      { font-size: 0.95rem; font-weight: 700; color: #0F172A; font-family: 'JetBrains Mono',monospace; }

    /* ── STAT PILLS ─────────────────────────────── */
    .pills    { display: flex; gap: 0.6rem; flex-wrap: wrap; margin: 0.7rem 0 1.35rem; }
    .pill     { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 0.52rem 0.95rem; min-width: 125px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
    .pill-lbl { font-size: 0.62rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }
    .pill-val { font-size: 0.93rem; font-weight: 700; color: #0F172A; font-family: 'JetBrains Mono',monospace; }
    .p-blue   { border-left: 3px solid #2563EB; }
    .p-green  { border-left: 3px solid #059669; }
    .p-red    { border-left: 3px solid #DC2626; }
    .p-amber  { border-left: 3px solid #D97706; }

    /* ── INSIGHT CARDS ──────────────────────────── */
    .insight {
        background: #FFFFFF; border: 1px solid #E2E8F0;
        border-left: 3px solid #2563EB; border-radius: 0 10px 10px 0;
        padding: 0.9rem 1.1rem; margin-bottom: 0.7rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03); transition: box-shadow 0.2s;
    }
    .insight:hover { box-shadow: 0 4px 12px rgba(37,99,235,0.08); }
    .insight-t { font-size: 0.71rem; font-weight: 700; color: #2563EB; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 0.28rem; }
    .insight-b { font-size: 0.82rem; color: #475569; line-height: 1.6; }

    /* ── DIVIDER ────────────────────────────────── */
    .divider { border: none; border-top: 1px solid #E2E8F0; margin: 1.5rem 0; }

    /* ── SIDEBAR COMPONENTS ─────────────────────── */
    .sb-logo  { padding-bottom: 1.15rem; margin-bottom: 1.15rem; border-bottom: 1px solid #E2E8F0; }
    .sb-title { font-size: 1.05rem; font-weight: 800; color: #0F172A; letter-spacing: -0.3px; }
    .sb-sub   { font-size: 0.67rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 1.1px; margin-top: 0.12rem; }
    .sb-lbl   { font-size: 0.62rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 1.3px; margin: 1.1rem 0 0.45rem; }
    .sb-row   { display: flex; justify-content: space-between; padding: 0.32rem 0; border-bottom: 1px solid #F1F5F9; font-size: 0.79rem; }
    .sb-k     { color: #64748B; }
    .sb-v     { color: #0F172A; font-family: 'JetBrains Mono',monospace; font-weight: 600; font-size: 0.77rem; }

    /* ── TABS ───────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] { background: transparent !important; border-bottom: 1px solid #E2E8F0 !important; gap: 0 !important; }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important; color: #64748B !important;
        border: none !important; border-bottom: 2px solid transparent !important;
        padding: 0.55rem 1.15rem !important;
        font-family: 'Plus Jakarta Sans',sans-serif !important;
        font-size: 0.82rem !important; font-weight: 500 !important;
        border-radius: 0 !important; transition: color 0.15s !important;
    }
    .stTabs [aria-selected="true"] { color: #0F172A !important; font-weight: 700 !important; border-bottom: 2px solid #2563EB !important; }
    .stTabs [data-baseweb="tab"]:hover { color: #0F172A !important; }
    .stTabs [data-baseweb="tab-panel"] { padding-top: 1.6rem !important; }

    /* ── STREAMLIT METRIC OVERRIDE ──────────────── */
    [data-testid="stMetric"] {
        background: #FFFFFF !important; border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important; padding: 1rem 1.15rem !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
    }
    [data-testid="stMetricLabel"] p { font-size: 0.67rem !important; color: #94A3B8 !important; text-transform: uppercase !important; letter-spacing: 1px !important; font-weight: 600 !important; }
    [data-testid="stMetricValue"]   { font-size: 1.35rem !important; font-weight: 800 !important; color: #0F172A !important; font-family: 'JetBrains Mono',monospace !important; }
    [data-testid="stMetricDelta"]   { font-size: 0.74rem !important; font-weight: 600 !important; }

    /* ── FORM LABELS ────────────────────────────── */
    .stSelectbox label, .stSlider label, .stMultiSelect label {
        font-size: 0.71rem !important; font-weight: 600 !important;
        color: #64748B !important; text-transform: uppercase !important; letter-spacing: 0.9px !important;
    }

    /* ── DATAFRAME ──────────────────────────────── */
    .stDataFrame { border-radius: 10px !important; overflow: hidden !important; border: 1px solid #E2E8F0 !important; }

    /* ── ALERTS ─────────────────────────────────── */
    .stSuccess, .stWarning { border-radius: 10px !important; }
    </style>
    """, unsafe_allow_html=True)

_inject_css()


# ══════════════════════════════════════════════════════════════
# SHARED UI HELPERS
# ══════════════════════════════════════════════════════════════

def sec(eyebrow: str, title: str, desc: str = "") -> None:
    st.markdown(f'<div class="sec-eye">{eyebrow}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sec-h2">{title}</div>',   unsafe_allow_html=True)
    if desc:
        st.markdown(f'<div class="sec-desc">{desc}</div>', unsafe_allow_html=True)


def _vline(fig: go.Figure, x_str: str,
           color: str = "#94A3B8", label: str = "", y: float = 0.96) -> None:
    """Pandas-2.x–safe vertical dashed line (no add_vline arithmetic)."""
    fig.add_shape(type="line", xref="x", yref="paper",
                  x0=x_str, x1=x_str, y0=0, y1=1,
                  line=dict(color=color, width=1, dash="dot"))
    if label:
        fig.add_annotation(xref="x", yref="paper", x=x_str, y=y,
                           text=f" {label}", showarrow=False,
                           font=dict(size=10, color=color),
                           xanchor="left", yanchor="top",
                           bgcolor="rgba(255,255,255,0.85)", borderpad=2)


def _vrect(fig: go.Figure, x0: str, x1: str,
           color: str = "gray", opacity: float = 0.07, label: str = "") -> None:
    """Pandas-2.x–safe shaded rect (no add_vrect arithmetic)."""
    fig.add_shape(type="rect", xref="x", yref="paper",
                  x0=x0, x1=x1, y0=0, y1=1,
                  fillcolor=color, opacity=opacity, line=dict(width=0))
    if label:
        fig.add_annotation(xref="x", yref="paper", x=x0, y=0.97,
                           text=f" {label}", showarrow=False,
                           font=dict(size=10, color=TEXT2),
                           xanchor="left", yanchor="top",
                           bgcolor="rgba(255,255,255,0.82)", borderpad=2)


def _apply(fig: go.Figure, height: int = 420,
           hovermode: str = "x unified") -> go.Figure:
    """Apply shared chart defaults + axis style."""
    fig.update_layout(**CHART_DEFAULTS, height=height, hovermode=hovermode)
    fig.update_xaxes(**_AXIS)
    fig.update_yaxes(**_AXIS)
    return fig


# ══════════════════════════════════════════════════════════════
# DATA & ARIMA — CACHED
# ══════════════════════════════════════════════════════════════

@st.cache_data
def lade_daten(pfad: str) -> dict:
    df = pd.read_csv(pfad, parse_dates=["date"])
    daten = {}
    for land in LAENDER:
        sub = (df[df["country"] == land]
               .sort_values("date")
               .reset_index(drop=True))
        serie = sub.set_index("date")["petrol_usd_liter"]
        serie.index.freq = pd.tseries.frequencies.to_offset("W-MON")
        daten[land] = serie
    return daten


@st.cache_data
def adf_test(serie_values, _serie_index):
    serie = pd.Series(serie_values, index=_serie_index)
    res = adfuller(serie.dropna(), autolag="AIC")
    return {
        "stat": round(res[0], 4), "pval": round(res[1], 4),
        "cv_5": round(res[4]["5%"], 4), "ok": res[1] < 0.05,
    }


@st.cache_data
def berechne_arima(serie_values, _serie_index, p_max=3, q_max=3):
    serie = pd.Series(serie_values, index=_serie_index)
    log_s = np.log(serie)
    rows  = []
    for p, q in itertools.product(range(p_max + 1), range(q_max + 1)):
        if p == 0 and q == 0:
            continue
        try:
            fit    = ARIMA(log_s.dropna(), order=(p, 1, q)).fit()
            fitted = np.exp(fit.fittedvalues.dropna())
            actual = serie.iloc[1:len(fitted) + 1]
            n      = min(len(fitted), len(actual))
            rmse   = np.sqrt(np.mean((actual.values[:n] - fitted.values[:n]) ** 2))
            mae    = np.mean(np.abs(actual.values[:n]  - fitted.values[:n]))
            rows.append({
                "Modell": f"ARIMA({p},1,{q})",
                "p": p, "d": 1, "q": q,
                "AIC":  round(fit.aic, 2),  "BIC":  round(fit.bic, 2),
                "RMSE": round(rmse, 4),      "MAE":  round(mae,  4),
                "LogL": round(fit.llf, 2),   "_fit": fit,
            })
        except Exception:
            pass
    df_res = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
    best   = df_res.iloc[0]
    return best["_fit"], df_res.drop(columns="_fit"), int(best["p"]), int(best["q"])


@st.cache_data
def berechne_prognose(serie_values, _serie_index, p, q):
    serie = pd.Series(serie_values, index=_serie_index)
    fit   = ARIMA(np.log(serie).dropna(), order=(p, 1, q)).fit()
    fc    = fit.get_forecast(steps=10)
    mean  = np.exp(fc.predicted_mean)
    ci    = np.exp(fc.conf_int(alpha=0.05))
    idx   = pd.date_range(
        start=serie.index[-1] + pd.Timedelta(weeks=1),
        periods=10, freq="W-MON",
    )
    mean.index = idx
    ci.index   = idx
    return mean, ci


# ══════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════

with st.spinner("Daten werden geladen…"):
    daten = lade_daten(DATENPFAD)


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
        <div class="sb-title">⛽ Fuel Analytics</div>
        <div class="sb-sub">ARIMA Dashboard · v1.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-lbl">Datensatz</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-row"><span class="sb-k">Quelle</span><span class="sb-v">Kaggle</span></div>
    <div class="sb-row"><span class="sb-k">Periode</span><span class="sb-v">2020–2026</span></div>
    <div class="sb-row"><span class="sb-k">Frequenz</span><span class="sb-v">Wöchentlich</span></div>
    <div class="sb-row"><span class="sb-k">Obs.</span><span class="sb-v">≈ 327 / Land</span></div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-lbl">Länder</div>', unsafe_allow_html=True)
    for land, info in LAENDER.items():
        st.markdown(
            f'<div class="sb-row">'
            f'<span class="sb-k">{info["flag"]} {land}</span>'
            f'<span class="sb-v" style="color:{info["farbe"]};">●</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sb-lbl">Methode</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-row"><span class="sb-k">Ansatz</span><span class="sb-v">Box-Jenkins</span></div>
    <div class="sb-row"><span class="sb-k">Modell</span><span class="sb-v">ARIMA(p,1,q)</span></div>
    <div class="sb-row"><span class="sb-k">Selektion</span><span class="sb-v">Min. AIC</span></div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<hr style="border:none;border-top:1px solid #E2E8F0;margin:1.1rem 0 0.5rem;">',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sb-lbl">Einzelanalyse</div>', unsafe_allow_html=True)
    ausgewaehltes_land = st.selectbox(
        "Land auswählen",
        list(LAENDER.keys()),
        format_func=lambda x: f"{LAENDER[x]['flag']}  {x}",
        label_visibility="collapsed",
    )
    p_max = st.slider("Max. AR-Ordnung (p)", 1, 4, 3)
    q_max = st.slider("Max. MA-Ordnung (q)", 1, 4, 3)

    # Live stats for selected country
    s_sb  = daten[ausgewaehltes_land]
    wow   = (s_sb.iloc[-1] - s_sb.iloc[-2]) / s_sb.iloc[-2] * 100
    sign  = "+" if wow >= 0 else ""
    clr   = POS if wow >= 0 else NEG
    st.markdown('<div class="sb-lbl">Aktuell — Gewähltes Land</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="sb-row"><span class="sb-k">Letzter Wert</span><span class="sb-v">{s_sb.iloc[-1]:.3f}</span></div>
    <div class="sb-row"><span class="sb-k">WoW</span><span class="sb-v" style="color:{clr};">{sign}{wow:.2f}%</span></div>
    <div class="sb-row"><span class="sb-k">Min</span><span class="sb-v">{s_sb.min():.3f}</span></div>
    <div class="sb-row"><span class="sb-k">Max</span><span class="sb-v">{s_sb.max():.3f}</span></div>
    <div class="sb-row"><span class="sb-k">Mittelwert</span><span class="sb-v">{s_sb.mean():.3f}</span></div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE HEADER  +  KPI CARDS
# ══════════════════════════════════════════════════════════════

st.markdown("""
<div class="dash-header">
    <div class="dash-title">Multivariate Zeitreihenanalyse — <em>Benzinpreise</em></div>
    <div class="dash-sub">Box-Jenkins ARIMA · Wöchentliche Daten · Global Fuel Prices 2020–2026</div>
    <div class="dash-pills">
        <span class="dash-pill">🇫🇷 France</span>
        <span class="dash-pill">🇩🇪 Germany</span>
        <span class="dash-pill">🇮🇩 Indonesia</span>
        <span class="dash-pill">🇮🇹 Italy</span>
        <span class="dash-pill">ARIMA(p,1,q)</span>
        <span class="dash-pill">327 Beobachtungen</span>
    </div>
</div>
""", unsafe_allow_html=True)

kpi_cols = st.columns(4)
for i, (land, info) in enumerate(LAENDER.items()):
    s    = daten[land]
    val  = s.iloc[-1]
    wow  = (val - s.iloc[-2]) / s.iloc[-2] * 100
    sign = "+" if wow >= 0 else ""
    cls  = "kpi-pos" if wow >= 0 else "kpi-neg"
    arr  = "▲" if wow >= 0 else "▼"
    with kpi_cols[i]:
        st.markdown(f"""
        <div class="kpi-card" style="--c:{info['farbe']};">
            <div class="kpi-top">
                <span class="kpi-flag">{info['flag']}</span>
                <span class="kpi-badge {cls}">{arr} {sign}{wow:.2f}%</span>
            </div>
            <div class="kpi-label">{land}</div>
            <div class="kpi-value">{val:.3f}</div>
            <div class="kpi-sub">Fuel Price Index · letzter Wert</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "01  Übersicht",
    "02  Einzelanalyse",
    "03  Modellvergleich",
    "04  Prognose",
])


# ──────────────────────────────────────────────────────────────
# TAB 1 · ÜBERSICHT
# ──────────────────────────────────────────────────────────────

with tab1:
    sec("MARKTÜBERBLICK", "Alle Zeitreihen im Vergleich",
        "Wöchentliche Benzinpreise aller vier Länder, Januar 2020 bis April 2026. "
        "Erkennbar: COVID-Einbruch (2020), Energiekrise & Ukraine-Krieg (2022), "
        "anschließende Stabilisierung auf unterschiedlichem Preisniveau.")

    fig1 = go.Figure()
    for land, info in LAENDER.items():
        s = daten[land]
        fig1.add_trace(go.Scatter(
            x=s.index, y=s.values,
            name=f"{info['flag']} {land}",
            line=dict(color=info["farbe"], width=2),
            hovertemplate=(
                f"<b>{land}</b><br>"
                "%{x|%d.%m.%Y}<br>"
                "Index: <b>%{y:.3f}</b><extra></extra>"
            ),
        ))
    _vrect(fig1, "2020-03-01", "2020-06-01", color="gray",    opacity=0.07, label="COVID-19")
    _vrect(fig1, "2022-02-24", "2022-08-01", color="#DC2626", opacity=0.05, label="Energiekrise")
    _apply(fig1, height=460, hovermode="x unified")
    fig1.update_layout(
        title=dict(
            text="<b>Fuel Price Index</b> — alle Länder 2020–2026",
            font=dict(size=15, color=TEXT), x=0, xanchor="left",
        ),
        yaxis_title="Fuel Price Index",
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    sec("KENNZAHLEN", "Statistischer Überblick")

    m1, m2, m3, m4 = st.columns(4)
    for col, (land, info) in zip([m1, m2, m3, m4], LAENDER.items()):
        s = daten[land]
        with col:
            st.markdown(f"**{info['flag']} {land}**")
            st.metric("Mittelwert", f"{s.mean():.3f}")
            st.metric("Maximum",   f"{s.max():.3f}",
                      delta=s.idxmax().strftime("%d.%m.%Y"), delta_color="off")
            st.metric("Aktuell",   f"{s.iloc[-1]:.3f}",
                      delta=f"{s.iloc[-1]-s.iloc[-2]:+.3f} vs. Vorwoche")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    sec("KONTEXT", "Makroökonomische Ereignisse")
    ca, cb = st.columns(2)
    with ca:
        st.markdown("""
        <div class="insight">
            <div class="insight-t">🦠 COVID-19 · März 2020</div>
            <div class="insight-b">Nachfrageeinbruch durch globale Lockdowns. Benzinpreise
            fielen auf Mehrjahrestiefs — in Indonesien durch staatliche Subventionen gedämpft.</div>
        </div>
        <div class="insight">
            <div class="insight-t">⚡ Energiekrise · Herbst 2021</div>
            <div class="insight-b">Nachholnachfrage trifft knappes Angebot. Stärkster
            Preisanstieg im gesamten Datensatz, besonders ausgeprägt in Westeuropa.</div>
        </div>
        """, unsafe_allow_html=True)
    with cb:
        st.markdown("""
        <div class="insight">
            <div class="insight-t">🌍 Ukraine-Krieg · Feb. 2022</div>
            <div class="insight-b">Versorgungsunterbrechungen und Sanktionen gegen russische
            Energie erhöhen die Volatilität. Frankreich, Deutschland und Italien
            am stärksten betroffen.</div>
        </div>
        <div class="insight">
            <div class="insight-t">🛢 OPEC+ Förderkürzungen · 2023</div>
            <div class="insight-b">Koordinierte Produktionssenkungen stabilisieren die Preise
            auf erhöhtem Niveau und erhöhen die Prognoseunsicherheit bis 2024.</div>
        </div>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# TAB 2 · EINZELANALYSE
# ──────────────────────────────────────────────────────────────

with tab2:
    land  = ausgewaehltes_land
    info  = LAENDER[land]
    serie = daten[land]

    sec("BOX-JENKINS", f"Einzelanalyse — {info['flag']} {land}",
        f"Vollständige ARIMA-Modellierung für {land}: ADF-Test, "
        "ACF/PACF-Analyse, Koeffizientenschätzung und Ljung-Box-Diagnose.")

    with st.spinner(f"ARIMA Grid-Search für {land} läuft…"):
        modell, df_modelle, p_b, q_b = berechne_arima(
            serie.values, serie.index, p_max, q_max,
        )

    best_row = df_modelle.iloc[0]
    st.markdown(f"""
    <div class="best">
        <div class="best-eye">⭐ Bestes Modell — Minimum AIC</div>
        <div class="best-name">ARIMA({p_b},1,{q_b})</div>
        <div class="best-row">
            <div class="bm"><span class="bm-l">AIC</span> <span class="bm-v">{best_row['AIC']}</span></div>
            <div class="bm"><span class="bm-l">BIC</span> <span class="bm-v">{best_row['BIC']}</span></div>
            <div class="bm"><span class="bm-l">RMSE</span><span class="bm-v">{best_row['RMSE']}</span></div>
            <div class="bm"><span class="bm-l">MAE</span> <span class="bm-v">{best_row['MAE']}</span></div>
            <div class="bm"><span class="bm-l">Log-L</span><span class="bm-v">{best_row['LogL']}</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── ADF-Test ─────────────────────────────────────────────

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    sec("SCHRITT 1–2", "Stationaritätstest (ADF)",
        "Die Rohdaten zeigen einen stochastischen Trend (I(1)). "
        "Erste Differenz des Log-Preises macht die Reihe stationär → d = 1.")

    log_s  = np.log(serie)
    diff_s = log_s.diff().dropna()
    adf_r  = adf_test(serie.values,  serie.index)
    adf_d  = adf_test(diff_s.values, diff_s.index)

    st.markdown(f"""
    <div class="pills">
        <div class="pill p-amber">
            <div class="pill-lbl">ADF Rohdaten</div>
            <div class="pill-val">{adf_r['stat']}</div>
        </div>
        <div class="pill p-amber">
            <div class="pill-lbl">p-Wert (Roh)</div>
            <div class="pill-val">{adf_r['pval']}</div>
        </div>
        <div class="pill {'p-green' if adf_d['ok'] else 'p-red'}">
            <div class="pill-lbl">ADF Δlog</div>
            <div class="pill-val">{adf_d['stat']}</div>
        </div>
        <div class="pill {'p-green' if adf_d['ok'] else 'p-red'}">
            <div class="pill-lbl">p-Wert (Δlog)</div>
            <div class="pill-val">{adf_d['pval']}</div>
        </div>
        <div class="pill p-blue">
            <div class="pill-lbl">Integrationsord.</div>
            <div class="pill-val">d = 1</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    ca, cb = st.columns(2)
    with ca:
        df_adf = pd.DataFrame([
            {"Reihe": "Rohdaten",
             "ADF-Stat": adf_r["stat"], "p-Wert": adf_r["pval"],
             "KV 5%": adf_r["cv_5"],   "Stationär": "✗" if not adf_r["ok"] else "✓"},
            {"Reihe": "Δ log(Index)",
             "ADF-Stat": adf_d["stat"], "p-Wert": adf_d["pval"],
             "KV 5%": adf_d["cv_5"],   "Stationär": "✓" if adf_d["ok"]  else "✗"},
        ])
        st.dataframe(df_adf, use_container_width=True, hide_index=True)
        st.caption("d = 1 → I(1)-Prozess. Erste Differenz des Log-Index ist stationär.")
    with cb:
        fig_adf = make_subplots(
            rows=2, cols=1, vertical_spacing=0.14,
            subplot_titles=["Log-Benzinpreis (trend-behaftet)",
                            "Δ log(Benzinpreis) — stationär"],
        )
        fig_adf.add_trace(go.Scatter(
            x=log_s.index, y=log_s.values,
            line=dict(color=info["farbe"], width=1.8), showlegend=False,
        ), row=1, col=1)
        fig_adf.add_trace(go.Scatter(
            x=diff_s.index, y=diff_s.values,
            line=dict(color=ACCENT, width=1.5), showlegend=False,
        ), row=2, col=1)
        fig_adf.add_hline(y=0, line_dash="dash", line_color=MUTED,
                          line_width=0.9, row=2, col=1)
        _apply(fig_adf, height=360, hovermode="x")
        fig_adf.update_annotations(font=dict(color=TEXT2, size=11))
        st.plotly_chart(fig_adf, use_container_width=True)

    # ── ACF & PACF ───────────────────────────────────────────

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    sec("SCHRITT 3", "ACF & PACF der stationären Reihe",
        "Autokorrelations- und partielle Autokorrelationsfunktion zur "
        "Identifikation der ARIMA-Ordnungen p und q.")

    acf_v  = acf(diff_s.dropna(),  nlags=25, alpha=0.05)
    pacf_v = pacf(diff_s.dropna(), nlags=25, alpha=0.05, method="ywm")
    lags   = list(range(1, 26))
    ci_bnd = 1.96 / np.sqrt(len(diff_s))

    fig_ap = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.1,
        subplot_titles=["ACF — Autokorrelationsfunktion",
                        "PACF — Partielle Autokorrelationsfunktion"],
    )
    for vals, col, clr in [
        (acf_v[0][1:],  1, info["farbe"]),
        (pacf_v[0][1:], 2, ACCENT),
    ]:
        bar_clrs = [NEG if abs(v) > ci_bnd else clr for v in vals]
        fig_ap.add_trace(go.Bar(
            x=lags, y=vals,
            marker_color=bar_clrs, marker_line_width=0,
            showlegend=False,
        ), row=1, col=col)
        for sign in [1, -1]:
            fig_ap.add_hline(
                y=sign * ci_bnd, line_dash="dash",
                line_color=MUTED, line_width=1, row=1, col=col,
            )
    fig_ap.update_xaxes(title_text="Lag (Wochen)", **_AXIS)
    fig_ap.update_yaxes(**_AXIS)
    _apply(fig_ap, height=360, hovermode="x")
    fig_ap.update_annotations(font=dict(color=TEXT2, size=12))
    st.plotly_chart(fig_ap, use_container_width=True)
    st.caption(
        "Balken außerhalb der gestrichelten 95%-Grenzen (rot markiert) → statistisch bedeutsam."
    )

    # ── Koeffizienten ─────────────────────────────────────────

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    sec("SCHRITT 4–6", f"Koeffizienten — ARIMA({p_b},1,{q_b})")

    params   = modell.params
    std_err  = modell.bse
    t_stats  = modell.tvalues
    p_values = modell.pvalues

    df_koeff = pd.DataFrame({
        "Koeffizient": params.index,
        "Wert":        params.round(5).values,
        "Std.Fehler":  std_err.round(5).values,
        "t-Statistik": t_stats.round(4).values,
        "p-Wert":      p_values.round(4).values,
        "Signifikanz": [
            "*** p<0.05" if p < 0.05 else
            ("** p<0.10"  if p < 0.10 else "n.s.")
            for p in p_values
        ],
    })
    st.dataframe(df_koeff, use_container_width=True, hide_index=True)
    st.caption("*** p < 0.05 · ** p < 0.10 · n.s. = nicht signifikant")

    lb    = acorr_ljungbox(modell.resid.dropna(), lags=[5, 10, 20], return_df=True)
    lb_ok = all(lb["lb_pvalue"] > 0.05)
    if lb_ok:
        st.success(
            f"✓ Ljung-Box (Lags 5/10/20): Keine Autokorrelation in den Residuen — "
            f"ARIMA({p_b},1,{q_b}) ist gut spezifiziert."
        )
    else:
        st.warning(
            "⚠ Ljung-Box: Autokorrelation in Residuen erkannt. "
            "Höhere Ordnungen könnten das Modell verbessern."
        )


# ──────────────────────────────────────────────────────────────
# TAB 3 · MODELLVERGLEICH
# ──────────────────────────────────────────────────────────────

with tab3:
    sec("MODELLVERGLEICH", "Evaluationsmetriken — alle Länder",
        "Für jedes Land wird automatisch das beste ARIMA-Modell via Grid-Search gefunden. "
        "Verglichen nach AIC (Akaike) und RMSE — niedrigere Werte sind besser.")

    zusammenfassung = []
    alle_modelle    = {}

    bar = st.progress(0, text="Modelle werden berechnet…")
    for i, (land, info) in enumerate(LAENDER.items()):
        s = daten[land]
        mod, df_m, p_b_, q_b_ = berechne_arima(s.values, s.index, p_max, q_max)
        alle_modelle[land] = (mod, p_b_, q_b_)
        best = df_m.iloc[0]
        zusammenfassung.append({
            "Land":   f"{info['flag']} {land}",
            "Modell": best["Modell"],
            "AIC":    best["AIC"],
            "BIC":    best["BIC"],
            "RMSE":   best["RMSE"],
            "MAE":    best["MAE"],
            "Log-L":  best["LogL"],
        })
        bar.progress((i + 1) / len(LAENDER), text=f"{land} ✓")
    bar.empty()

    df_sum = pd.DataFrame(zusammenfassung)

    # Summary pills
    br = df_sum.loc[df_sum["RMSE"].idxmin()]
    ba = df_sum.loc[df_sum["AIC"].idxmin()]
    n_models = (p_max + 1) * (q_max + 1) - 1
    st.markdown(f"""
    <div class="pills">
        <div class="pill p-green">
            <div class="pill-lbl">Bestes RMSE</div>
            <div class="pill-val">{br['Land']}</div>
        </div>
        <div class="pill p-blue">
            <div class="pill-lbl">Bestes AIC</div>
            <div class="pill-val">{ba['Land']}</div>
        </div>
        <div class="pill p-amber">
            <div class="pill-lbl">Modelle geprüft</div>
            <div class="pill-val">{n_models} / Land</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(df_sum, use_container_width=True, hide_index=True)
    st.caption(
        "AIC/BIC: Modellgüte (niedriger = besser) · "
        "RMSE/MAE: In-Sample Vorhersagefehler · Log-L: Log-Likelihood (höher = besser)"
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    ca, cb = st.columns(2)

    with ca:
        sec("", "AIC-Vergleich")
        fig_aic = go.Figure()
        for _, row in df_sum.iterrows():
            lk  = row["Land"].split(" ")[-1]
            clr = LAENDER.get(lk, {}).get("farbe", ACCENT)
            fig_aic.add_trace(go.Bar(
                x=[row["Land"]], y=[row["AIC"]],
                marker_color=clr, marker_line_width=0,
                text=[f"{row['AIC']:.1f}"], textposition="outside",
                textfont=dict(size=11, color=TEXT2),
                showlegend=False,
                hovertemplate=f"<b>{row['Land']}</b><br>AIC: {row['AIC']}<extra></extra>",
            ))
        _apply(fig_aic, height=340)
        fig_aic.update_layout(
            title=dict(text="<b>AIC</b> — niedriger ist besser",
                       font=dict(size=13, color=TEXT), x=0, xanchor="left"),
            yaxis_title="AIC", xaxis_title="", bargap=0.38,
        )
        st.plotly_chart(fig_aic, use_container_width=True)

    with cb:
        sec("", "RMSE-Vergleich")
        fig_rmse = go.Figure()
        for _, row in df_sum.iterrows():
            lk  = row["Land"].split(" ")[-1]
            clr = LAENDER.get(lk, {}).get("farbe", ACCENT)
            fig_rmse.add_trace(go.Bar(
                x=[row["Land"]], y=[row["RMSE"]],
                marker_color=clr, marker_line_width=0,
                text=[f"{row['RMSE']:.4f}"], textposition="outside",
                textfont=dict(size=11, color=TEXT2),
                showlegend=False,
                hovertemplate=f"<b>{row['Land']}</b><br>RMSE: {row['RMSE']}<extra></extra>",
            ))
        _apply(fig_rmse, height=340)
        fig_rmse.update_layout(
            title=dict(text="<b>RMSE</b> — niedriger ist besser",
                       font=dict(size=13, color=TEXT), x=0, xanchor="left"),
            yaxis_title="RMSE (Index-Einheiten)", xaxis_title="", bargap=0.38,
        )
        st.plotly_chart(fig_rmse, use_container_width=True)


# ──────────────────────────────────────────────────────────────
# TAB 4 · PROGNOSE
# ──────────────────────────────────────────────────────────────

with tab4:
    sec("10-WOCHEN-PROGNOSE", "April bis Juni 2026",
        "Prognose der wöchentlichen Fuel Price Indices für alle vier Länder. "
        "Das 95%-Konfidenzband zeigt die Unsicherheit der Vorhersage.")

    fig_fc  = go.Figure()
    prog_tab = []

    for land, info in LAENDER.items():
        s = daten[land]
        _, p_b_, q_b_ = alle_modelle.get(
            land,
            (None,
             *berechne_arima(s.values, s.index, p_max, q_max)[2:])
        )
        fc_mean, fc_ci = berechne_prognose(s.values, s.index, p_b_, q_b_)
        hist = s.iloc[-52:]

        # Historical
        fig_fc.add_trace(go.Scatter(
            x=hist.index, y=hist.values,
            name=f"{info['flag']} {land}",
            line=dict(color=info["farbe"], width=2),
            hovertemplate=(
                f"<b>{land}</b> historisch<br>"
                "%{x|%d.%m.%Y}<br>Index: <b>%{y:.3f}</b><extra></extra>"
            ),
        ))
        # Seamless connector
        fig_fc.add_trace(go.Scatter(
            x=[hist.index[-1], fc_mean.index[0]],
            y=[hist.values[-1], fc_mean.values[0]],
            line=dict(color=info["farbe"], width=2, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ))
        # Forecast line
        fig_fc.add_trace(go.Scatter(
            x=fc_mean.index, y=fc_mean.values,
            name=f"{info['flag']} {land} Prognose",
            line=dict(color=info["farbe"], width=2.5, dash="dash"),
            mode="lines+markers", marker=dict(size=5, symbol="circle"),
            hovertemplate=(
                f"<b>{land}</b> Prognose<br>"
                "%{x|%d.%m.%Y}<br>Index: <b>%{y:.3f}</b><extra></extra>"
            ),
        ))
        # 95 % CI band
        fig_fc.add_trace(go.Scatter(
            x=list(fc_ci.index) + list(fc_ci.index[::-1]),
            y=list(fc_ci.iloc[:, 0]) + list(fc_ci.iloc[:, 1][::-1]),
            fill="toself", fillcolor=info["farbe"],
            opacity=0.08, line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))
        # Forecast table rows
        for dt, val, (lo, hi) in zip(fc_mean.index, fc_mean.values, fc_ci.values):
            prog_tab.append({
                "Land":           f"{info['flag']} {land}",
                "Datum":          dt.strftime("%d.%m.%Y"),
                "Prognose":       round(val, 4),
                "KI Unten (95%)": round(lo,  4),
                "KI Oben (95%)":  round(hi,  4),
            })

    # Forecast split line (pandas-safe)
    split = daten["France"].index[-1].strftime("%Y-%m-%d")
    _vline(fig_fc, split, color=MUTED, label="Prognose →", y=0.96)

    _apply(fig_fc, height=500, hovermode="x unified")
    fig_fc.update_layout(
        title=dict(
            text="<b>10-Wochen-Prognose</b> — April bis Juni 2026 · 95%-Konfidenzband",
            font=dict(size=15, color=TEXT), x=0, xanchor="left",
        ),
        yaxis_title="Fuel Price Index",
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    # ── Forecast table ────────────────────────────────────────

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    sec("DATENTABELLE", "Wöchentliche Prognosewerte")

    df_prog   = pd.DataFrame(prog_tab)
    land_opts = [f"{LAENDER[l]['flag']} {l}" for l in LAENDER]
    land_sel  = st.multiselect(
        "Länder filtern",
        land_opts, default=land_opts,
        label_visibility="collapsed",
    )
    st.dataframe(
        df_prog[df_prog["Land"].isin(land_sel)],
        use_container_width=True, hide_index=True,
    )

    # ── Used models footer ────────────────────────────────────

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    sec("MODELLINFO", "Verwendete ARIMA-Modelle")

    mi_cols = st.columns(4)
    for i, (land, info) in enumerate(LAENDER.items()):
        _, p_, q_ = alle_modelle.get(land, (None, "?", "?"))
        with mi_cols[i]:
            st.markdown(f"""
            <div class="best" style="margin-bottom:0;">
                <div class="best-eye">{info['flag']} {land}</div>
                <div class="best-name">ARIMA({p_},1,{q_})</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown(
    '<div style="font-size:0.74rem;color:#94A3B8;text-align:center;padding-bottom:0.5rem;">'
    "Multivariate Zeitreihenanalyse · Box-Jenkins ARIMA · "
    "Global Fuel Prices 2020–2026 (Kaggle) · Streamlit Dashboard"
    "</div>",
    unsafe_allow_html=True,
)
