"""
================================================================
Univariate Zeitreihenanalyse – Wöchentliche Benzinpreise FR
================================================================
Datensatz  : Global Fuel Prices 2020–2026 (Kaggle)
Land       : Frankreich
Variable   : petrol_usd_liter (Benzinpreis in USD/Liter)
Zeitraum   : 06.01.2020 – 06.04.2026  (wöchentlich, 327 Beob.)
Methode    : Box-Jenkins → ARIMA

Besonderheiten der Zeitreihe
-----------------------------
  - COVID-19 Schock (März–Mai 2020): starker Preiseinbruch
  - Erholung & Energiekrise (2021–2022): rasanter Anstieg
  - Stabilisierung auf hohem Niveau (2023–2026)

Struktur
--------
  Schritt 1 : Daten laden & explorative Analyse
  Schritt 2 : Stationaritätstest (ADF)
  Schritt 3 : Transformation → schwache Stationarität
  Schritt 4 : ACF & PACF interpretieren
  Schritt 5 : Modellselektion (AIC/BIC Grid-Search)
  Schritt 6 : Residualanalyse (Ljung-Box, Jarque-Bera)
  Schritt 7 : t-Statistiken der Koeffizienten
  Schritt 8 : 10-Wochen-Prognose mit Konfidenzintervallen
================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import itertools
import os

from statsmodels.tsa.stattools    import adfuller
from statsmodels.tsa.arima.model  import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats

# ── Ordner anlegen ─────────────────────────────────────────────
os.makedirs("docs", exist_ok=True)

# ── Plot-Stil ──────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor" : "white",
    "axes.facecolor"   : "#f8f9fa",
    "axes.grid"        : True,
    "grid.color"       : "#dee2e6",
    "grid.linestyle"   : "--",
    "grid.linewidth"   : 0.6,
    "font.family"      : "sans-serif",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.titlesize"   : 12,
    "axes.labelsize"   : 10,
})

BLAU   = "#003f88"
ROT    = "#c0392b"
GRUEN  = "#1a7a4a"
ORANGE = "#e67e22"


# ══════════════════════════════════════════════════════════════
# SCHRITT 1: DATEN LADEN & EXPLORATIVE ANALYSE
# ══════════════════════════════════════════════════════════════
print("=" * 62)
print("  SCHRITT 1: Daten laden & explorative Analyse")
print("=" * 62)

df = pd.read_csv("fuel_prices_france.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

serie = df.set_index("date")["petrol_usd_liter"]
serie.index.freq = "W-MON"
serie.name = "Benzinpreis Frankreich (USD/Liter)"

print(f"  Land         : Frankreich")
print(f"  Zeitraum     : {serie.index[0].date()} – {serie.index[-1].date()}")
print(f"  Frequenz     : wöchentlich")
print(f"  Beobacht.    : {len(serie)}")
print(f"  Mittelwert   : {serie.mean():.4f} USD/Liter")
print(f"  Std. Abw.    : {serie.std():.4f}")
print(f"  Min          : {serie.min():.4f}  ({serie.idxmin().date()})")
print(f"  Max          : {serie.max():.4f}  ({serie.idxmax().date()})")
print(f"  Fehlende W.  : {serie.isna().sum()}")

# ── Plot mit annotierten Ereignissen ──────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(serie.index, serie.values, color=BLAU, linewidth=1.6, zorder=3)
ax.fill_between(serie.index, serie.values, alpha=0.08, color=BLAU)

# Ereignisse markieren
ereignisse = {
    "2020-03-16": ("COVID-19\nLockdown",   ROT,    "bottom"),
    "2022-02-24": ("Ukraine-\nKrieg",      ORANGE, "bottom"),
    "2022-06-06": ("Preishoch\n5.06$/L",   ROT,    "top"),
    "2020-04-20": ("Tiefpunkt\n1.90$/L",   GRUEN,  "top"),
}
for datum, (label, farbe, va) in ereignisse.items():
    x = pd.Timestamp(datum)
    y = serie.asof(x)
    ax.axvline(x, color=farbe, linewidth=1.2, linestyle=":", alpha=0.8, zorder=2)
    offset = 0.15 if va == "top" else -0.15
    ax.annotate(label, xy=(x, y), xytext=(x, y + offset),
                fontsize=8, color=farbe, ha="center",
                arrowprops=dict(arrowstyle="-", color=farbe, lw=0.8))

ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.set_title("Wöchentliche Benzinpreise Frankreich 2020–2026  (USD/Liter)",
             fontweight="bold", pad=12)
ax.set_xlabel("Datum")
ax.set_ylabel("USD / Liter")
plt.tight_layout()
plt.savefig("docs/01_zeitreihe_roh.png", dpi=150, bbox_inches="tight")
plt.show()
print("  → docs/01_zeitreihe_roh.png\n")


# ══════════════════════════════════════════════════════════════
# SCHRITT 2: STATIONARITÄTSTEST (ADF)
# ══════════════════════════════════════════════════════════════
print("=" * 62)
print("  SCHRITT 2: Augmented Dickey-Fuller Test (ADF)")
print("=" * 62)
print("  H0 : Einheitswurzel vorhanden → NICHT stationär")
print("  H1 : Zeitreihe ist stationär")
print()

def adf_test(ts, name):
    res = adfuller(ts.dropna(), autolag="AIC")
    stat, p, cv = res[0], res[1], res[4]
    ok = p < 0.05
    print(f"  ── {name} ──")
    print(f"  ADF-Statistik  : {stat:>10.4f}")
    print(f"  p-Wert         : {p:>10.4f}")
    print(f"  Krit. W. 1%    : {cv['1%']:>10.4f}")
    print(f"  Krit. W. 5%    : {cv['5%']:>10.4f}")
    print(f"  Ergebnis       : {'✓ STATIONÄR  (p < 0.05)' if ok else '✗ NICHT STATIONÄR  (p ≥ 0.05)'}")
    print()
    return ok

adf_test(serie, "Rohdaten – petrol_usd_liter")


# ══════════════════════════════════════════════════════════════
# SCHRITT 3: TRANSFORMATION → SCHWACHE STATIONARITÄT
# ══════════════════════════════════════════════════════════════
print("=" * 62)
print("  SCHRITT 3: Transformation zur Stationarität")
print("=" * 62)
print("  Log-Transformation  → stabilisiert die Varianz")
print("  1. Differenz        → entfernt stochastischen Trend")
print("  → Ergebnis: I(1) Prozess  (d = 1)")
print()

log_serie  = np.log(serie)
log_serie.name = "log(Benzinpreis)"

diff_serie = log_serie.diff().dropna()
diff_serie.name = "Δ log(Benzinpreis)  [wöchentliche Log-Rendite]"

print("  ADF nach Log-Transformation:")
adf_test(log_serie, "log(petrol_usd_liter)")

print("  ADF nach 1. Differenz:")
ok_diff = adf_test(diff_serie, "Δ log(petrol_usd_liter)")

d = 1
print(f"  → Integrationsordnung : d = {d}  ({'✓ stationär' if ok_diff else '2. Differenz nötig'})\n")

# ── Plot Transformation ────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(13, 7))

axes[0].plot(log_serie.index, log_serie.values, color=BLAU, linewidth=1.5)
axes[0].set_title("Log-Benzinpreis  log(USD/Liter)", fontweight="bold")
axes[0].set_ylabel("log(USD/Liter)")
axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

axes[1].plot(diff_serie.index, diff_serie.values, color=ROT, linewidth=1.2, alpha=0.85)
axes[1].axhline(0, color="black", linewidth=0.9, linestyle="--")
axes[1].set_title("Erste Differenz  Δlog(Benzinpreis)  – stationäre Reihe", fontweight="bold")
axes[1].set_ylabel("Δ log(USD/Liter)")
axes[1].set_xlabel("Datum")
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout()
plt.savefig("docs/02_transformation.png", dpi=150, bbox_inches="tight")
plt.show()
print("  → docs/02_transformation.png\n")


# ══════════════════════════════════════════════════════════════
# SCHRITT 4: ACF & PACF
# ══════════════════════════════════════════════════════════════
print("=" * 62)
print("  SCHRITT 4: ACF & PACF der stationären Reihe")
print("=" * 62)
print("  ACF  bricht nach Lag q ab → MA(q)")
print("  PACF bricht nach Lag p ab → AR(p)")
print("  Exponentielles Abklingen  → gemischter ARMA-Prozess")
print()

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

plot_acf(diff_serie.dropna(), lags=30, ax=axes[0],
         color=BLAU, vlines_kwargs={"colors": BLAU}, alpha=0.05)
axes[0].set_title("ACF – Autokorrelationsfunktion\nΔ log(Benzinpreis)", fontweight="bold")
axes[0].set_xlabel("Lag (Wochen)")
axes[0].set_ylabel("Autokorrelation")

plot_pacf(diff_serie.dropna(), lags=30, ax=axes[1],
          color=ROT, vlines_kwargs={"colors": ROT}, method="ywm", alpha=0.05)
axes[1].set_title("PACF – Partielle Autokorrelationsfunktion\nΔ log(Benzinpreis)", fontweight="bold")
axes[1].set_xlabel("Lag (Wochen)")
axes[1].set_ylabel("Partielle Autokorrelation")

plt.tight_layout()
plt.savefig("docs/03_acf_pacf.png", dpi=150, bbox_inches="tight")
plt.show()
print("  → docs/03_acf_pacf.png\n")


# ══════════════════════════════════════════════════════════════
# SCHRITT 5: MODELLSELEKTION (AIC / BIC GRID-SEARCH)
# ══════════════════════════════════════════════════════════════
print("=" * 62)
print("  SCHRITT 5: Modellselektion – AIC/BIC Grid-Search")
print("=" * 62)
print("  Kandidaten: ARIMA(p,1,q)  mit p ∈ {0..3}, q ∈ {0..3}")
print()

ergebnisse = []
for p, q in itertools.product(range(4), range(4)):
    if p == 0 and q == 0:
        continue
    try:
        fit = ARIMA(log_serie.dropna(), order=(p, 1, q)).fit()
        ergebnisse.append({
            "Modell": f"ARIMA({p},1,{q})",
            "p": p, "d": 1, "q": q,
            "AIC":  round(fit.aic, 3),
            "BIC":  round(fit.bic, 3),
            "LogL": round(fit.llf, 3),
        })
    except Exception:
        pass

df_mod = pd.DataFrame(ergebnisse).sort_values("AIC").reset_index(drop=True)

print("  Top 8 nach AIC:")
print(df_mod.head(8).to_string(index=False))
print()

best      = df_mod.iloc[0]
p_b, q_b  = int(best["p"]), int(best["q"])
print(f"  → Bestes Modell : ARIMA({p_b},1,{q_b})")
print(f"     AIC = {best['AIC']}   BIC = {best['BIC']}\n")

# ── AIC-Vergleichsplot ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
top = df_mod.head(10).iloc[::-1].copy()
farben = [BLAU if m == best["Modell"] else "#adb5bd" for m in top["Modell"]]
bars = ax.barh(top["Modell"], top["AIC"], color=farben, edgecolor="white", height=0.65)
for bar, val in zip(bars, top["AIC"]):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}", va="center", fontsize=9)
ax.set_title("ARIMA Modellvergleich – AIC  (niedriger = besser)",
             fontweight="bold")
ax.set_xlabel("AIC")
plt.tight_layout()
plt.savefig("docs/04_modellvergleich.png", dpi=150, bbox_inches="tight")
plt.show()
print("  → docs/04_modellvergleich.png\n")

modell = ARIMA(log_serie.dropna(), order=(p_b, 1, q_b)).fit()


# ══════════════════════════════════════════════════════════════
# SCHRITT 6: RESIDUALANALYSE
# ══════════════════════════════════════════════════════════════
print("=" * 62)
print(f"  SCHRITT 6: Residualanalyse – ARIMA({p_b},1,{q_b})")
print("=" * 62)
print("  Gut spezifiziertes Modell:")
print("  ✓ Residuen ≈ weißes Rauschen (keine Autokorrelation)")
print("  ✓ Annähernde Normalverteilung")
print()

residuen = modell.resid.dropna()

fig = plt.figure(figsize=(13, 8))
gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

# Residuen über Zeit
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(residuen.index, residuen.values, color="#6c757d", linewidth=1.0)
ax1.axhline(0, color=ROT, linewidth=1.2, linestyle="--")
ax1.fill_between(residuen.index,
                 residuen.values, 0,
                 where=residuen.values > 0, color=BLAU,  alpha=0.2)
ax1.fill_between(residuen.index,
                 residuen.values, 0,
                 where=residuen.values < 0, color=ROT, alpha=0.2)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.set_title(f"Residuen – ARIMA({p_b},1,{q_b})", fontweight="bold")
ax1.set_ylabel("Residuum")
ax1.set_xlabel("Datum")

# Histogramm
ax2 = fig.add_subplot(gs[1, 0])
ax2.hist(residuen, bins=25, color=BLAU, alpha=0.75,
         edgecolor="white", density=True)
xr = np.linspace(residuen.min(), residuen.max(), 200)
ax2.plot(xr, stats.norm.pdf(xr, residuen.mean(), residuen.std()),
         color=ROT, linewidth=2, label="Normalverteilung")
ax2.set_title("Verteilung der Residuen", fontweight="bold")
ax2.set_xlabel("Residuum")
ax2.legend(fontsize=9)

# ACF der Residuen
ax3 = fig.add_subplot(gs[1, 1])
plot_acf(residuen, lags=20, ax=ax3,
         color=BLAU, vlines_kwargs={"colors": BLAU}, alpha=0.05)
ax3.set_title("ACF der Residuen", fontweight="bold")
ax3.set_xlabel("Lag (Wochen)")

plt.suptitle("Residualanalyse", fontsize=14, fontweight="bold")
plt.savefig("docs/05_residualanalyse.png", dpi=150, bbox_inches="tight")
plt.show()
print("  → docs/05_residualanalyse.png\n")

# Tests
lb = acorr_ljungbox(residuen, lags=[5, 10, 20], return_df=True)
print("  Ljung-Box Test  (H0: keine Autokorrelation in Residuen):")
print(lb.to_string())
print()
if all(lb["lb_pvalue"] > 0.05):
    print("  ✓ p > 0.05 bei allen Lags → Modell gut spezifiziert\n")
else:
    print("  ⚠ Autokorrelation vorhanden → Modell prüfen\n")

jb_stat, jb_p = stats.jarque_bera(residuen)
print(f"  Jarque-Bera Test  (H0: Normalverteilung):")
print(f"  Statistik: {jb_stat:.4f}   p-Wert: {jb_p:.4f}")
if jb_p > 0.05:
    print("  ✓ Normalverteilung nicht abgelehnt\n")
else:
    print("  ⚠ Leichte Abweichung von Normalverteilung")
    print("    → Bei wöchentlichen Preisen durch Schocks erwartet\n")


# ══════════════════════════════════════════════════════════════
# SCHRITT 7: T-STATISTIKEN DER KOEFFIZIENTEN
# ══════════════════════════════════════════════════════════════
print("=" * 62)
print("  SCHRITT 7: Koeffizientenschätzung & t-Statistiken")
print("=" * 62)
print("  t-Statistik = Koeffizient / Standardfehler")
print("  |t| > 1.96  →  signifikant auf 5%-Niveau   ***")
print("  |t| > 1.645 →  signifikant auf 10%-Niveau  **")
print("  |t| ≤ 1.645 →  nicht signifikant           n.s.")
print()

params   = modell.params
std_err  = modell.bse
t_stats  = modell.tvalues
p_values = modell.pvalues

header = f"  {'Koeffizient':<18} {'Wert':>9} {'Std.Fehler':>11} " \
         f"{'t-Stat':>9} {'p-Wert':>9}  Signifikanz"
print(header)
print("  " + "─" * 72)
for name in params.index:
    sig = "***" if p_values[name] < 0.05 else \
          ("**"  if p_values[name] < 0.10 else "n.s.")
    print(f"  {name:<18} {params[name]:>9.5f} {std_err[name]:>11.5f} "
          f"{t_stats[name]:>9.4f} {p_values[name]:>9.4f}  {sig}")

print()
print(f"  Log-Likelihood : {modell.llf:.4f}")
print(f"  AIC            : {modell.aic:.4f}")
print(f"  BIC            : {modell.bic:.4f}\n")


# ══════════════════════════════════════════════════════════════
# SCHRITT 8: 10-WOCHEN-PROGNOSE MIT KONFIDENZINTERVALLEN
# ══════════════════════════════════════════════════════════════
print("=" * 62)
print("  SCHRITT 8: 10-Wochen-Prognose  (April–Juni 2026)")
print("=" * 62)

forecast    = modell.get_forecast(steps=10)
fc_mean     = forecast.predicted_mean
fc_ci       = forecast.conf_int(alpha=0.05)

# Rücktransformation (exp wegen Log)
fc_mean_orig = np.exp(fc_mean)
fc_ci_orig   = np.exp(fc_ci)

letztes_datum = serie.index[-1]
prog_idx = pd.date_range(
    start=letztes_datum + pd.Timedelta(weeks=1),
    periods=10, freq="W"
)
fc_mean_orig.index = prog_idx
fc_ci_orig.index   = prog_idx

print()
print(f"  {'Datum':<14} {'Prognose ($/L)':>15}  "
      f"{'95%-KI unten':>14}  {'95%-KI oben':>13}")
print("  " + "─" * 60)
for dt, val, (lo, hi) in zip(prog_idx, fc_mean_orig.values, fc_ci_orig.values):
    print(f"  {str(dt.date()):<14} {val:>15.4f}  {lo:>14.4f}  {hi:>13.4f}")

# ── Prognose-Plot ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))

# Historisch (letzte 52 Wochen für Lesbarkeit)
hist = serie.iloc[-52:]
ax.plot(hist.index, hist.values,
        color=BLAU, linewidth=1.8, label="Historisch (letztes Jahr)")

# Nahtloser Übergang: letzten historischen Punkt + Prognose verbinden
uebergang_x = [hist.index[-1], prog_idx[0]]
uebergang_y = [hist.values[-1], fc_mean_orig.values[0]]
ax.plot(uebergang_x, uebergang_y, color=GRUEN, linewidth=2, linestyle="--")

# Prognose
ax.plot(prog_idx, fc_mean_orig.values,
        color=GRUEN, linewidth=2.2, linestyle="--",
        marker="o", markersize=5, label=f"Prognose ARIMA({p_b},1,{q_b})")

# Konfidenzband
ax.fill_between(prog_idx,
                fc_ci_orig.iloc[:, 0],
                fc_ci_orig.iloc[:, 1],
                color=GRUEN, alpha=0.15, label="95%-Konfidenzintervall")

# Trennlinie
ax.axvline(letztes_datum, color="gray", linewidth=1,
           linestyle=":", alpha=0.7)
ax.text(letztes_datum + pd.Timedelta(days=4),
        ax.get_ylim()[0] + 0.05,
        "Prognose →", color="gray", fontsize=9)

ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=30)
ax.set_title(f"ARIMA({p_b},1,{q_b}) – 10-Wochen-Prognose Benzinpreise Frankreich",
             fontweight="bold", pad=12)
ax.set_xlabel("Datum")
ax.set_ylabel("Benzinpreis (USD/Liter)")
ax.legend(loc="upper left", fontsize=9)
plt.tight_layout()
plt.savefig("docs/06_prognose.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n  → docs/06_prognose.png")

print()
print("=" * 62)
print(f"  ✓ ANALYSE ABGESCHLOSSEN")
print(f"  Bestes Modell : ARIMA({p_b},1,{q_b})")
print(f"  Datensatz     : 327 wöchentliche Beobachtungen")
print(f"  Prognose      : 10 Wochen  (April–Juni 2026)")
print(f"  Grafiken      : docs/  (6 Dateien)")
print("=" * 62)
