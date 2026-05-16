# Kraftstoffpreise: Warum sich Europa und Indonesien voellig anders verhalten

**Wöchentliche Zeitreihenanalyse (2020–2026) | Deutschland, Frankreich, Indonesien**

---

## Zwei Welten, ein Rohstoff

![Benzinpreise Vergleich](docs/plots_aus_comparative_forecasting/01_benzinpreise_vergleich.png)

Europa schwankt zwischen 2 und 5 USD pro Liter. Indonesien bewegt sich kaum – unter 1 USD, fast eine Gerade. Gleicher Rohstoff, gleicher Weltmarkt, voellig unterschiedliches Preisverhalten. Warum?

![Europa vs Indonesien](docs/plots_aus_comparative_forecasting/06_europa_vs_indonesien.png)

|                | Europa (DE, FR)        | Indonesien                      |
| -------------- | ---------------------- | ------------------------------- |
| Preisbildung   | Marktgetrieben         | Staatlich administriert         |
| Subventionen   | Niedrig                | Hoch                            |
| Preisanpassung | Graduell, marktbasiert | Sprunghaft, politisch gesteuert |
| Volatilitaet   | Hoch                   | Niedrig                         |

Deutschland und Frankreich sind im europaeischen Binnenmarkt – Brent-Rohoel, Raffineriemargen und Steuern bestimmen den Endpreis. Preisschocks wie die Ukraine-Krise 2022 schlagen direkt durch. Indonesien betreibt ein **staatliches Subventionssystem**: Der Staat federt Weltmarktschocks ab. Das ist kein schlechtes Datenset – es ist ein anderes oekonomisches System.

![Volatilitaet](docs/plots_aus_comparative_forecasting/04_volatilitaet_vergleich.png)

---

## Der gemeinsame Treiber: Brent-Rohoelpreis

![Brent Crude](docs/plots_aus_comparative_forecasting/03_brent_crude.png)

Alle drei Laender beziehen denselben Rohstoff. Aber wie stark kommt der Weltmarktpreis beim Verbraucher an?

![Korrelation Heatmaps](docs/plots_aus_comparative_forecasting/05_korrelation_heatmaps.png)

In Europa: **starke Korrelation** zwischen Brent und Benzinpreis. In Indonesien: **schwach** – die Subventionen brechen die Transmission. Granger-Kausalitaetstests bestaetigen: Brent verbessert die Prognose in allen drei Laendern signifikant, aber der Effekt ist in Europa deutlich staerker.

---

## Prognosemodelle: 8-Wochen-Horizont

Drei klassische Modelle prognostizieren den Benzinpreis 8 Wochen voraus:

| Modell                | Typ         | Kernidee                                     |
| --------------------- | ----------- | -------------------------------------------- |
| **ARIMA**             | Univariat   | Autokorrelation der differenzierten Reihe    |
| **VAR**               | Multivariat | Gemeinsame Dynamik von Benzin, Diesel, Brent |
| **State Space (UCM)** | Strukturell | Zerlegung in Trend, Zyklus, Stoerung         |

### Ergebnisse: Wer prognostiziert am besten?

![Modell Ranking](docs/plots_aus_comparative_forecasting/08_modell_ranking_rmse.png)

#### Deutschland

| Modell      | RMSE   | MAE    | MAPE   | Rang |
| ----------- | ------ | ------ | ------ | ---- |
| **VAR**     | 0.2265 | 0.2014 | 3.96%  | 1    |
| State Space | 0.2833 | 0.2491 | 4.88%  | 2    |
| ARIMA       | 0.2940 | 0.2580 | 5.06%  | 3    |

#### Frankreich

| Modell      | RMSE   | MAE    | MAPE   | Rang |
| ----------- | ------ | ------ | ------ | ---- |
| **VAR**     | 0.2489 | 0.2170 | 4.30%  | 1    |
| State Space | 0.3051 | 0.2605 | 5.15%  | 2    |
| ARIMA       | 0.3180 | 0.2712 | 5.36%  | 3    |

#### Indonesien

| Modell          | RMSE   | MAE    | MAPE   | Rang |
| --------------- | ------ | ------ | ------ | ---- |
| **State Space** | 0.0518 | 0.0462 | 4.37%  | 1    |
| ARIMA           | 0.0528 | 0.0472 | 4.48%  | 2    |
| VAR             | 0.0540 | 0.0484 | 4.59%  | 3    |

![MAPE Vergleich](docs/plots_aus_comparative_forecasting/09_mape_vergleich.png)

**Kernbefund:** In Europa gewinnt **VAR** – die multivariate Information (Brent, Diesel) hilft, weil der Marktmechanismus funktioniert. In Indonesien gewinnt **State Space** – ein strukturelles Modell, das den stabilen Trend besser abbildet als volatile Marktmodelle.

### Prognose vs. Realitaet

![Forecast Vergleich](docs/plots_aus_comparative_forecasting/10_forecast_vergleich.png)

---

## TimeGPT: Foundation Model im Vergleich

Zusaetzlich zu den klassischen Modellen testen wir **TimeGPT** (Nixtla) – ein vortrainiertes Foundation Model, das ohne manuelles Training prognostiziert (Zero-Shot Forecasting).

Die vollstaendige TimeGPT-Analyse deckt ab:

| Feature                      | Ergebnis                                                                 |
| ---------------------------- | ------------------------------------------------------------------------ |
| **Forecasting at Scale**     | Alle 3 Laender gleichzeitig, ohne separates Training                     |
| **Uncertainty Quantification** | Konfidenzintervalle passen sich automatisch an die Volatilitaet an     |
| **Exogenous Variables**      | Brent und Diesel als zusaetzliche Informationsquellen                    |
| **Cross-Validation**         | Ergebnisse ueber 3 Zeitfenster hinweg stabil                            |
| **Fine-Tuning**              | Anpassung an domaenenspezifische Muster (10 und 50 Steps)               |
| **Anomaly Detection**        | Erkennt automatisch ungewoehnliche Preisbewegungen (z.B. Ukraine-Krise) |

![TimeGPT Konfidenzintervalle](docs/plots_aus_timegpt/01_timegpt_konfidenzintervalle.png)

![TimeGPT Anomalien](docs/plots_aus_timegpt/05_timegpt_anomalien.png)

![TimeGPT Varianten Vergleich](docs/plots_aus_timegpt/03_timegpt_varianten_vergleich.png)

---

## Klassisch vs. AI: Wer gewinnt?

|                         | Klassisch (ARIMA, VAR, UCM)             | AI (TimeGPT)  |
| ----------------------- | --------------------------------------- | ------------- |
| Interpretierbarkeit     | Hoch – man versteht, was das Modell tut | Black Box     |
| Setup                   | Feature-Engineering noetig              | Plug & Play   |
| Performance (Kurzfrist) | Oft konkurrenzfaehig                    | Oft aehnlich  |
| Kosten                  | Kostenlos                               | API-Kosten    |
| Reproduzierbarkeit      | Deterministisch                         | API-abhaengig |

Foundation Models sind nicht automatisch ueberlegen. Bei stabilen, gut verstandenen Zeitreihen koennen klassische Modelle gleichwertige Ergebnisse liefern – mit dem Vorteil vollstaendiger Interpretierbarkeit.

---

## Fazit

1. **Marktstruktur bestimmt das Prognoseverhalten** – nicht die Datenqualitaet. Indonesien ist nicht schwieriger zu prognostizieren, es funktioniert anders.
2. **Europaeische Integration ist messbar** – Deutschland und Frankreich verhalten sich prognostisch nahezu identisch (MAPE ~4-5%).
3. **Das beste Modell haengt vom Marktregime ab** – VAR fuer integrierte Maerkte (RMSE 0.23), State Space fuer administrierte Preise (RMSE 0.05).
4. **8 Wochen sind der richtige Horizont** – laengere Prognosen verlieren bei allen Modellen rapide an Genauigkeit.
5. **AI-Modelle sind kein Allheilmittel** – Interpretierbarkeit und Reproduzierbarkeit bleiben relevante Entscheidungskriterien.

### Limitationen

- USD-normierte Preise – lokale Waehrungseffekte nicht beruecksichtigt
- Kein Out-of-Sample Backtesting (rollende Evaluation)
- Begrenzte exogene Variablen (keine Geopolitik, Lagerbestaende, Raffinerie-Auslastung)

---

## Projektstruktur

```
projekt2/
├── data/
│   ├── raw/                               # Originaldaten
│   └── processed/                         # Aufbereitete Laenderdaten
├── docs/
│   ├── plots_aus_comparative_forecasting/ # Plots klassische Modelle
│   └── plots_aus_timegpt/                 # Plots TimeGPT
├── notebooks/                             # Explorative Notebooks
├── src/
│   ├── comparative_forecasting.ipynb      # Hauptanalyse (ARIMA, VAR, State Space)
│   └── TimeGPT.ipynb                     # TimeGPT-Analyse (Foundation Model)
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook src/comparative_forecasting.ipynb
```

**Technologien:** Python, pandas, statsmodels, scikit-learn, matplotlib, seaborn, Nixtla TimeGPT
