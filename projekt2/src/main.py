"""
================================================================
Multivariate Zeitreihenanalyse – Streamlit Dashboard
================================================================
Datensatz  : Global Fuel Prices 2020–2026 (Kaggle)
Länder     : France, Germany, Indonesia, Italy
Variable   : petrol_usd_liter
Methode    : Box-Jenkins → ARIMA (automatisiert)

Ausführen
---------
    streamlit run src/main.py

Struktur
--------
    Tab 1 : Übersicht – alle Zeitreihen im Vergleich
    Tab 2 : Einzelanalyse – ARIMA pro Land (alle 8 Schritte)
    Tab 3 : Modellvergleich – Evaluationsmetriken aller Länder
    Tab 4 : Prognose – 10-Wochen-Forecast für alle Länder
================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import itertools
from statsmodels.tsa.stattools    import adfuller
from statsmodels.tsa.arima.model  import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
import os

# ══════════════════════════════════════════════════════════════
# KONFIGURATION
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Fuel Price ARIMA Dashboard",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded",
)

LAENDER = {
    "France":    {"farbe": "#003f88", "flag": "🇫🇷"},
    "Germany":   {"farbe": "#c0392b", "flag": "🇩🇪"},
    "Indonesia": {"farbe": "#1a7a4a", "flag": "🇮🇩"},
    "Italy":     {"farbe": "#e67e22", "flag": "🇮🇹"},
}

DATENPFAD = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "raw", "global_fuel_prices_2020_2026.csv"
)


# ══════════════════════════════════════════════════════════════
# DATEN & ARIMA – GECACHT
# ══════════════════════════════════════════════════════════════

@st.cache_data
def lade_daten(pfad: str) -> dict:
    """Lädt CSV und gibt pro Land eine saubere Zeitreihe zurück."""
    df = pd.read_csv(pfad, parse_dates=["date"])
    daten = {}
    for land in LAENDER:
        sub = df[df["country"] == land].copy()
        sub = sub.sort_values("date").reset_index(drop=True)
        serie = sub.set_index("date")["petrol_usd_liter"]
        serie.index.freq = pd.tseries.frequencies.to_offset("W-MON")
        daten[land] = serie
    return daten


@st.cache_data
def adf_test(serie_values, serie_index):
    """ADF-Test – gibt Statistik, p-Wert und Ergebnis zurück."""
    serie = pd.Series(serie_values, index=serie_index)
    res = adfuller(serie.dropna(), autolag="AIC")
    return {
        "stat":   round(res[0], 4),
        "pval":   round(res[1], 4),
        "cv_5":   round(res[4]["5%"], 4),
        "ok":     res[1] < 0.05,
    }


@st.cache_data
def berechne_arima(serie_values, serie_index, p_max=3, q_max=3):
    """
    Grid-Search ARIMA(p,1,q) – gibt bestes Modell + Ergebnistabelle zurück.

    Evaluationsmetriken
    -------------------
    AIC  : Akaike Information Criterion     (niedriger = besser)
    BIC  : Bayesian Information Criterion   (niedriger = besser)
    RMSE : Root Mean Squared Error          (niedriger = besser)
    MAE  : Mean Absolute Error              (niedriger = besser)
    """
    serie = pd.Series(serie_values, index=serie_index)
    log_s = np.log(serie)
    ergebnisse = []

    for p, q in itertools.product(range(p_max + 1), range(q_max + 1)):
        if p == 0 and q == 0:
            continue
        try:
            fit = ARIMA(log_s.dropna(), order=(p, 1, q)).fit()
            resid    = fit.resid.dropna()
            fitted   = np.exp(fit.fittedvalues.dropna())
            actual   = serie.iloc[1:len(fitted) + 1]
            min_len  = min(len(fitted), len(actual))
            rmse = np.sqrt(np.mean((actual.values[:min_len] - fitted.values[:min_len]) ** 2))
            mae  = np.mean(np.abs(actual.values[:min_len] - fitted.values[:min_len]))
            ergebnisse.append({
                "Modell": f"ARIMA({p},1,{q})",
                "p": p, "d": 1, "q": q,
                "AIC":  round(fit.aic, 2),
                "BIC":  round(fit.bic, 2),
                "RMSE": round(rmse, 4),
                "MAE":  round(mae,  4),
                "LogL": round(fit.llf, 2),
                "_fit": fit,
            })
        except Exception:
            pass

    df_res = pd.DataFrame(ergebnisse).sort_values("AIC").reset_index(drop=True)
    bestes = df_res.iloc[0]
    return bestes["_fit"], df_res.drop(columns="_fit"), int(bestes["p"]), int(bestes["q"])


@st.cache_data
def berechne_prognose(serie_values, serie_index, p, q):
    """10-Wochen-Prognose mit 95%-Konfidenzintervall."""
    serie  = pd.Series(serie_values, index=serie_index)
    log_s  = np.log(serie)
    fit    = ARIMA(log_s.dropna(), order=(p, 1, q)).fit()
    fc     = fit.get_forecast(steps=10)
    mean   = np.exp(fc.predicted_mean)
    ci     = np.exp(fc.conf_int(alpha=0.05))
    letztes = serie.index[-1]
    idx    = pd.date_range(
        start=letztes + pd.Timedelta(weeks=1),
        periods=10, freq="W-MON"
    )
    mean.index = idx
    ci.index   = idx
    return mean, ci


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("⛽ Fuel Price ARIMA")
    st.markdown("---")
    st.markdown("**Datensatz**")
    st.caption("Global Fuel Prices 2020–2026")
    st.caption("Quelle: Kaggle")
    st.markdown("---")
    st.markdown("**Länder**")
    for land, info in LAENDER.items():
        st.markdown(f"{info['flag']} {land}")
    st.markdown("---")
    st.markdown("**Methode**")
    st.caption("Box-Jenkins → ARIMA(p,1,q)")
    st.caption("Grid-Search via AIC")
    st.markdown("---")
    ausgewaehltes_land = st.selectbox(
        "Land für Einzelanalyse:",
        list(LAENDER.keys()),
        format_func=lambda x: f"{LAENDER[x]['flag']} {x}"
    )
    p_max = st.slider("Max. AR-Ordnung (p)", 1, 4, 3)
    q_max = st.slider("Max. MA-Ordnung (q)", 1, 4, 3)


# ══════════════════════════════════════════════════════════════
# DATEN LADEN
# ══════════════════════════════════════════════════════════════

with st.spinner("Daten werden geladen..."):
    daten = lade_daten(DATENPFAD)

# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════

st.title("📈 Multivariate Zeitreihenanalyse – Benzinpreise")
st.markdown(
    "**Box-Jenkins ARIMA** | 327 wöchentliche Beobachtungen | 2020–2026 | "
    "France 🇫🇷 · Germany 🇩🇪 · Indonesia 🇮🇩 · Italy 🇮🇹"
)
st.markdown("---")

# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Übersicht",
    "🔬 Einzelanalyse",
    "📋 Modellvergleich",
    "🔮 Prognose"
])


# ──────────────────────────────────────────────────────────────
# TAB 1: ÜBERSICHT
# ──────────────────────────────────────────────────────────────

with tab1:
    st.subheader("Alle Zeitreihen im Vergleich")
    st.markdown(
        "Wöchentliche Benzinpreise aller vier Länder von Januar 2020 bis April 2026. "
        "Deutlich erkennbar: der COVID-Einbruch (2020), die Energiekrise (2022) und die "
        "anschließende Stabilisierung auf unterschiedlichem Preisniveau."
    )

    fig = go.Figure()
    for land, info in LAENDER.items():
        s = daten[land]
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values,
            name=f"{info['flag']} {land}",
            line=dict(color=info["farbe"], width=1.8),
            hovertemplate=f"<b>{land}</b><br>%{{x|%d.%m.%Y}}<br>%{{y:.3f}} USD/L<extra></extra>"
        ))
    fig.add_vrect(x0="2020-03-01", x1="2020-06-01",
                  fillcolor="gray", opacity=0.08,
                  annotation_text="COVID", annotation_position="top left")
    fig.add_vrect(x0="2022-02-24", x1="2022-08-01",
                  fillcolor="red", opacity=0.06,
                  annotation_text="Energiekrise", annotation_position="top left")
    fig.update_layout(
        height=450, hovermode="x unified",
        xaxis_title="Datum", yaxis_title="USD / Liter",
        legend=dict(orientation="h", y=-0.15),
        plot_bgcolor="#f8f9fa", paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)

    # KPI-Karten
    st.markdown("#### Kennzahlen")
    cols = st.columns(4)
    for i, (land, info) in enumerate(LAENDER.items()):
        s = daten[land]
        with cols[i]:
            st.markdown(f"**{info['flag']} {land}**")
            st.metric("Mittelwert",  f"{s.mean():.3f} $/L")
            st.metric("Maximum",     f"{s.max():.3f} $/L",
                      delta=f"am {s.idxmax().strftime('%d.%m.%Y')}")
            st.metric("Aktuell",     f"{s.iloc[-1]:.3f} $/L",
                      delta=f"{s.iloc[-1]-s.iloc[-2]:+.3f} vs. Vorwoche")


# ──────────────────────────────────────────────────────────────
# TAB 2: EINZELANALYSE
# ──────────────────────────────────────────────────────────────

with tab2:
    land  = ausgewaehltes_land
    info  = LAENDER[land]
    serie = daten[land]

    st.subheader(f"{info['flag']} Einzelanalyse – {land}")

    with st.spinner(f"ARIMA Grid-Search für {land}..."):
        modell, df_modelle, p_b, q_b = berechne_arima(
            serie.values, serie.index, p_max, q_max
        )

    st.success(f"Bestes Modell: **ARIMA({p_b},1,{q_b})** | "
               f"AIC = {df_modelle.iloc[0]['AIC']} | "
               f"BIC = {df_modelle.iloc[0]['BIC']}")

    st.markdown("---")

    # ── Schritt 2: ADF-Test ──
    st.markdown("#### Schritt 1–2: Stationaritätstest (ADF)")
    col1, col2 = st.columns(2)

    with col1:
        adf_roh  = adf_test(serie.values, serie.index)
        log_s    = np.log(serie)
        diff_s   = log_s.diff().dropna()
        adf_diff = adf_test(diff_s.values, diff_s.index)

        df_adf = pd.DataFrame([
            {"Reihe": "Rohdaten",         "ADF-Stat": adf_roh["stat"],  "p-Wert": adf_roh["pval"],  "Stationär": "✗" if not adf_roh["ok"]  else "✓"},
            {"Reihe": "Δ log(Benzinpreis)", "ADF-Stat": adf_diff["stat"], "p-Wert": adf_diff["pval"], "Stationär": "✓" if adf_diff["ok"] else "✗"},
        ])
        st.dataframe(df_adf, use_container_width=True, hide_index=True)
        st.caption("d = 1 → I(1) Prozess. Erste Differenz macht die Reihe stationär.")

    with col2:
        fig2 = make_subplots(rows=2, cols=1, subplot_titles=["Log-Benzinpreis", "Δ log(Benzinpreis) – stationär"])
        fig2.add_trace(go.Scatter(x=log_s.index, y=log_s.values,
                                   line=dict(color=info["farbe"], width=1.5), showlegend=False), row=1, col=1)
        fig2.add_trace(go.Scatter(x=diff_s.index, y=diff_s.values,
                                   line=dict(color="#c0392b", width=1.2), showlegend=False), row=2, col=1)
        fig2.add_hline(y=0, line_dash="dash", line_color="black", line_width=0.8, row=2, col=1)
        fig2.update_layout(height=380, plot_bgcolor="#f8f9fa", paper_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # ── Schritt 4: ACF & PACF (als Balken) ──
    st.markdown("#### Schritt 3: ACF & PACF der stationären Reihe")

    from statsmodels.tsa.stattools import acf, pacf
    acf_vals  = acf(diff_s.dropna(),  nlags=25, alpha=0.05)
    pacf_vals = pacf(diff_s.dropna(), nlags=25, alpha=0.05, method="ywm")
    lags = list(range(1, 26))

    fig3 = make_subplots(rows=1, cols=2,
                          subplot_titles=["ACF – Autokorrelationsfunktion",
                                          "PACF – Partielle Autokorrelationsfunktion"])
    # ACF
    fig3.add_trace(go.Bar(x=lags, y=acf_vals[0][1:],
                           marker_color=info["farbe"], name="ACF"), row=1, col=1)
    fig3.add_hline(y=1.96/np.sqrt(len(diff_s)),  line_dash="dash", line_color="gray", row=1, col=1)
    fig3.add_hline(y=-1.96/np.sqrt(len(diff_s)), line_dash="dash", line_color="gray", row=1, col=1)
    # PACF
    fig3.add_trace(go.Bar(x=lags, y=pacf_vals[0][1:],
                           marker_color="#c0392b", name="PACF"), row=1, col=2)
    fig3.add_hline(y=1.96/np.sqrt(len(diff_s)),  line_dash="dash", line_color="gray", row=1, col=2)
    fig3.add_hline(y=-1.96/np.sqrt(len(diff_s)), line_dash="dash", line_color="gray", row=1, col=2)
    fig3.update_layout(height=350, plot_bgcolor="#f8f9fa",
                        paper_bgcolor="white", showlegend=False)
    fig3.update_xaxes(title_text="Lag (Wochen)")
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("Gestrichelte Linien = 95%-Signifikanzgrenzen. Balken außerhalb → statistisch bedeutsam.")

    st.markdown("---")

    # ── Schritt 7: Koeffizienten & t-Statistiken ──
    st.markdown(f"#### Schritt 4–6: Koeffizienten ARIMA({p_b},1,{q_b})")
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
        "Signifikanz": ["***" if p < 0.05 else ("**" if p < 0.10 else "n.s.")
                        for p in p_values],
    })
    st.dataframe(df_koeff, use_container_width=True, hide_index=True)
    st.caption("*** p < 0.05  |  ** p < 0.10  |  n.s. = nicht signifikant")

    # Ljung-Box
    residuen = modell.resid.dropna()
    lb = acorr_ljungbox(residuen, lags=[5, 10, 20], return_df=True)
    lb_ok = all(lb["lb_pvalue"] > 0.05)
    if lb_ok:
        st.success("✓ Ljung-Box Test: Keine Autokorrelation in Residuen → Modell gut spezifiziert")
    else:
        st.warning("⚠ Ljung-Box Test: Autokorrelation in Residuen erkannt")


# ──────────────────────────────────────────────────────────────
# TAB 3: MODELLVERGLEICH
# ──────────────────────────────────────────────────────────────

with tab3:
    st.subheader("Modellvergleich – alle Länder")
    st.markdown(
        "Für jedes Land wird automatisch das beste ARIMA-Modell via Grid-Search gefunden. "
        "Verglichen wird nach **AIC** (Akaike Information Criterion) – niedrigere Werte sind besser."
    )

    st.markdown("#### Evaluationsmetriken")
    st.caption("AIC/BIC: Modellgüte (niedriger = besser) | RMSE/MAE: Vorhersagefehler (niedriger = besser)")

    zusammenfassung = []
    alle_modelle    = {}

    progress = st.progress(0, text="Berechne Modelle...")
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
        })
        progress.progress((i + 1) / len(LAENDER), text=f"{land} ✓")

    progress.empty()

    df_summary = pd.DataFrame(zusammenfassung)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)

    # AIC-Vergleichsplot
    fig4 = px.bar(df_summary, x="Land", y="AIC", color="Land",
                  color_discrete_map={
                      f"{LAENDER[l]['flag']} {l}": LAENDER[l]["farbe"]
                      for l in LAENDER
                  },
                  title="AIC-Vergleich der besten Modelle pro Land",
                  text="AIC")
    fig4.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig4.update_layout(height=380, showlegend=False,
                        plot_bgcolor="#f8f9fa", paper_bgcolor="white",
                        yaxis_title="AIC", xaxis_title="")
    st.plotly_chart(fig4, use_container_width=True)

    # RMSE-Vergleich
    fig5 = px.bar(df_summary, x="Land", y="RMSE", color="Land",
                  color_discrete_map={
                      f"{LAENDER[l]['flag']} {l}": LAENDER[l]["farbe"]
                      for l in LAENDER
                  },
                  title="RMSE-Vergleich – Vorhersagefehler pro Land",
                  text="RMSE")
    fig5.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig5.update_layout(height=380, showlegend=False,
                        plot_bgcolor="#f8f9fa", paper_bgcolor="white",
                        yaxis_title="RMSE (USD/Liter)", xaxis_title="")
    st.plotly_chart(fig5, use_container_width=True)


# ──────────────────────────────────────────────────────────────
# TAB 4: PROGNOSE
# ──────────────────────────────────────────────────────────────

with tab4:
    st.subheader("10-Wochen-Prognose – April bis Juni 2026")
    st.markdown(
        "Prognose der wöchentlichen Benzinpreise für alle Länder. "
        "Das **95%-Konfidenzintervall** zeigt die Unsicherheit der Vorhersage."
    )

    # Alle Prognosen berechnen
    fig6 = go.Figure()
    prog_tabelle = []

    for land, info in LAENDER.items():
        s = daten[land]
        _, p_b_, q_b_ = alle_modelle.get(land) or berechne_arima(s.values, s.index, p_max, q_max)[1:]
        fc_mean, fc_ci = berechne_prognose(s.values, s.index, p_b_, q_b_)

        # Historisch (letzte 52 Wochen)
        hist = s.iloc[-52:]
        fig6.add_trace(go.Scatter(
            x=hist.index, y=hist.values,
            name=f"{info['flag']} {land}",
            line=dict(color=info["farbe"], width=1.8),
            hovertemplate=f"<b>{land}</b> historisch<br>%{{x|%d.%m.%Y}}<br>%{{y:.3f}} $/L<extra></extra>"
        ))

        # Nahtloser Übergang
        fig6.add_trace(go.Scatter(
            x=[hist.index[-1], fc_mean.index[0]],
            y=[hist.values[-1], fc_mean.values[0]],
            line=dict(color=info["farbe"], width=1.8, dash="dot"),
            showlegend=False
        ))

        # Prognose
        fig6.add_trace(go.Scatter(
            x=fc_mean.index, y=fc_mean.values,
            name=f"{info['flag']} {land} Prognose",
            line=dict(color=info["farbe"], width=2, dash="dash"),
            mode="lines+markers", marker=dict(size=5),
            hovertemplate=f"<b>{land}</b> Prognose<br>%{{x|%d.%m.%Y}}<br>%{{y:.3f}} $/L<extra></extra>"
        ))

        # Konfidenzband
        fig6.add_trace(go.Scatter(
            x=list(fc_ci.index) + list(fc_ci.index[::-1]),
            y=list(fc_ci.iloc[:, 0]) + list(fc_ci.iloc[:, 1][::-1]),
            fill="toself", fillcolor=info["farbe"],
            opacity=0.07, line=dict(width=0),
            showlegend=False, hoverinfo="skip"
        ))

        # Tabelle
        for dt, val, (lo, hi) in zip(fc_mean.index, fc_mean.values, fc_ci.values):
            prog_tabelle.append({
                "Land":   f"{info['flag']} {land}",
                "Datum":  dt.strftime("%d.%m.%Y"),
                "Prognose ($/L)": round(val, 4),
                "KI Unten ($/L)": round(lo,  4),
                "KI Oben ($/L)":  round(hi,  4),
            })

    fig6.add_vline(x=daten["France"].index[-1],
                   line_dash="dot", line_color="gray", line_width=1)
    fig6.add_annotation(
        x=daten["France"].index[-1], y=0,
        text="  Prognose →", showarrow=False,
        font=dict(color="gray", size=10), yref="paper", y=0.02
    )
    fig6.update_layout(
        height=500, hovermode="x unified",
        xaxis_title="Datum", yaxis_title="USD / Liter",
        legend=dict(orientation="h", y=-0.18),
        plot_bgcolor="#f8f9fa", paper_bgcolor="white",
    )
    st.plotly_chart(fig6, use_container_width=True)

    # Prognosetabelle
    st.markdown("#### Prognosetabelle")
    df_prog = pd.DataFrame(prog_tabelle)
    land_filter = st.multiselect(
        "Länder filtern:",
        [f"{LAENDER[l]['flag']} {l}" for l in LAENDER],
        default=[f"{LAENDER[l]['flag']} {l}" for l in LAENDER]
    )
    st.dataframe(
        df_prog[df_prog["Land"].isin(land_filter)],
        use_container_width=True, hide_index=True
    )

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Multivariate Zeitreihenanalyse · Box-Jenkins ARIMA · "
    "Global Fuel Prices 2020–2026 (Kaggle) · "
    "Streamlit Dashboard"
)
