# Kraftstoffpreise: Warum sich Europa und Indonesien voellig anders verhalten

**Wöchentliche Zeitreihenanalyse (2020–2026) | Deutschland, Frankreich, Indonesien**

---

## Zwei Welten, ein Rohstoff

![Benzinpreise Vergleich](docs/plots_aus_comparative_forecasting/01_benzinpreise_vergleich.png)

Europaeische Kraftstoffpreise schwanken zwischen 2 und 5 USD pro Liter. Indonesien bewegt sich kaum – unter 1 USD, fast eine Gerade. Gleicher Rohstoff, gleicher Weltmarkt, voellig unterschiedliches Preisverhalten.

![Europa vs Indonesien](docs/plots_aus_comparative_forecasting/06_europa_vs_indonesien.png)

---

## Warum verhalten sich diese Laender so unterschiedlich?

Die Antwort ist nicht Datenqualitaet. Es ist **Marktstruktur**.

|                | Europa (DE, FR)        | Indonesien                      |
| -------------- | ---------------------- | ------------------------------- |
| Preisbildung   | Marktgetrieben         | Staatlich administriert         |
| Subventionen   | Niedrig                | Hoch                            |
| Preisanpassung | Graduell, marktbasiert | Sprunghaft, politisch gesteuert |
| Volatilitaet   | Hoch                   | Niedrig                         |

Deutschland und Frankreich sind Teil des europaeischen Binnenmarkts – Brent-Rohoel, Raffineriemargen und Steuern bestimmen den Endpreis. Preisschocks (Ukraine-Krise 2022) schlagen direkt durch.

Indonesien betreibt ein **staatliches Subventionssystem**: Der Staat begrenzt Preisanpassungen, federt Weltmarktschocks ab und greift aktiv in die Preisbildung ein. Das Ergebnis ist ein fundamental anderes Preisregime – nicht schlechtere Daten, sondern ein anderes oekonomisches System.

![Volatilitaet](docs/plots_aus_comparative_forecasting/04_volatilitaet_vergleich.png)

---

## Der gemeinsame Treiber: Brent-Rohoelpreis

![Brent Crude](docs/plots_aus_comparative_forecasting/03_brent_crude.png)

Alle drei Laender beziehen denselben Rohstoff vom Weltmarkt. Aber wie stark kommt dieser Preis beim Endverbraucher an?

![Korrelation Heatmaps](docs/plots_aus_comparative_forecasting/05_korrelation_heatmaps.png)

In Europa: **starke Korrelation** zwischen Brent und Benzinpreis – der Marktmechanismus funktioniert. In Indonesien: **schwaehere Korrelation** – die Subventionen brechen die Transmission.

Granger-Kausalitaetstests bestaetigen: Brent verbessert die Prognose des Benzinpreises in allen drei Laendern signifikant – aber der Effekt ist in Europa deutlich staerker.

---

## Welches Modell prognostiziert am besten?

Vier Prognosemodelle treten gegeneinander an – auf einem realistischen **8-Wochen-Horizont**.

| Modell                | Typ              | Kernidee                                     |
| --------------------- | ---------------- | -------------------------------------------- |
| **ARIMA**             | Univariat        | Autokorrelation der differenzierten Reihe    |
| **VAR**               | Multivariat      | Gemeinsame Dynamik von Benzin, Diesel, Brent |
| **State Space (UCM)** | Strukturell      | Zerlegung in Trend, Zyklus, Stoerung         |
| **TimeGPT**           | Foundation Model | Zero-Shot Forecasting (Nixtla API)           |

### Ergebnis

![Modell Ranking](docs/plots_aus_comparative_forecasting/08_modell_ranking_rmse.png)

![MAPE Vergleich](docs/plots_aus_comparative_forecasting/09_mape_vergleich.png)

**Was faellt auf:**

- In Europa gewinnt **VAR** – die multivariate Information (Brent, Diesel) hilft, weil der Marktmechanismus funktioniert.
- In Indonesien gewinnt **State Space** – ein strukturelles Modell, das den stabilen Trend besser abbildet als volatile Marktmodelle.
- ARIMA ist ueberall solide, aber selten das beste Modell.

### Prognose vs. Realitaet

![Forecast Vergleich](docs/plots_aus_comparative_forecasting/10_forecast_vergleich.png)

---

## Klassische Modelle vs. AI: Wer gewinnt?

Die ehrliche Antwort: **Es kommt darauf an.**

|                         | Klassisch (ARIMA, VAR, UCM)             | AI (TimeGPT)  |
| ----------------------- | --------------------------------------- | ------------- |
| Interpretierbarkeit     | Hoch – man versteht, was das Modell tut | Black Box     |
| Setup                   | Feature-Engineering noetig              | Plug & Play   |
| Performance (Kurzfrist) | Oft konkurrenzfaehig                    | Oft aehnlich  |
| Kosten                  | Kostenlos                               | API-Kosten    |
| Reproduzierbarkeit      | Deterministisch                         | API-abhaengig |

Foundation Models sind nicht automatisch ueberlegen. Bei stabilen, gut verstandenen Zeitreihen – wie subventionierten Kraftstoffpreisen – koennen klassische Modelle gleichwertige Ergebnisse liefern, mit dem Vorteil vollstaendiger Interpretierbarkeit.

**Wann lohnt sich AI-Forecasting?** Wenn man viele heterogene Zeitreihen schnell prognostizieren muss, ohne jede einzelne zu verstehen. Fuer tiefe Einzelanalysen bleiben klassische Modelle ueberlegen.

---

## Fazit

1. **Marktstruktur bestimmt das Prognoseverhalten** – nicht die Datenqualitaet. Indonesien ist nicht schwieriger zu prognostizieren, es funktioniert anders.
2. **Europaeische Integration ist messbar** – Deutschland und Frankreich verhalten sich prognostisch nahezu identisch.
3. **8 Wochen sind der richtige Horizont** – laengere Prognosen verlieren bei allen Modellen rapide an Genauigkeit.
4. **Das beste Modell haengt vom Marktregime ab** – VAR fuer integrierte Maerkte, State Space fuer administrierte Preise.
5. **AI-Modelle sind kein Allheilmittel** – Interpretierbarkeit und Reproduzierbarkeit bleiben relevante Entscheidungskriterien.

---

## Limitationen

- USD-normierte Preise – lokale Waehrungseffekte nicht beruecksichtigt
- Kein Out-of-Sample Backtesting (rollende Evaluation)
- Begrenzte exogene Variablen (keine Geopolitik, Lagerbestaende, Raffinerie-Auslastung)

---

## Projektstruktur

```
projekt2/
├── data/
│   ├── raw/                    # Originaldaten
│   └── processed/              # Aufbereitete Laenderdaten
├── docs/
│   └──france
│   └──germany
│   └──indonesia
│   └── plots_aus_comparative_forecasting/                  # Exportierte Visualisierungen
├── notebooks/
│   └── niklas
│   └── nikita
│   └── christina
├── src/
└── comparative_forecasting.ipynb  # Hauptanalyse
├── requirements.txt
└── README.md
```

## Setup & Ausfuehrung

```bash
# 1. Virtuelle Umgebung aktivieren
python -m venv .venv
source .venv/bin/activate

# 2. Abhaengigkeiten installieren
pip install -r requirements.txt

# 3. Notebook oeffnen
jupyter notebook notebooks/comparative_forecasting.ipynb
```

### TimeGPT aktivieren

Im Notebook die Zeile aendern:

```python
TIMEGPT_API_KEY = "PASTE_API_KEY_HERE"  # ← eigenen Key einfuegen
```

Alles andere laeuft automatisch.

---

**Technologien:** Python, pandas, statsmodels, scikit-learn, matplotlib, seaborn, Nixtla TimeGPT
