"""
Zeitreihen-Projekt – src-Paket.

Module:
  data_loader    – einheitliches Laden der 4 Länderreihen
  diagnostics    – ADF, KPSS, Chow, QLR
  evaluation     – Train/Test-Split, Rolling-Origin CV, MSE/RMSE/MAE/MAPE
  models         – Kandidaten für das gemeinsame Modell
  compare_models – Orchestrator: alle Kandidaten evaluieren + Vergleichstabelle
"""
from .data_loader import (
    LAENDER, EUROPA, ZIEL_SPALTE, EXOG_SPALTE,
    serie_pro_land, panel_dataframe, exogene_serie, datensatz_info,
)
from .evaluation import (
    evaluate_holdout, rolling_origin_cv,
    vergleichstabelle, detail_pro_land,
    metriken_dict,
)

__all__ = [
    "LAENDER", "EUROPA", "ZIEL_SPALTE", "EXOG_SPALTE",
    "serie_pro_land", "panel_dataframe", "exogene_serie", "datensatz_info",
    "evaluate_holdout", "rolling_origin_cv",
    "vergleichstabelle", "detail_pro_land",
    "metriken_dict",
]
