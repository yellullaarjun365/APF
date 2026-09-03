"""APF V1 -- Shared feature engineering.
Used identically by training (src/models/train_baseline.py) and serving
(src/api/main.py). Do not fork this logic.
"""
import numpy as np
import pandas as pd

TARGET = "total_yield_kg"
DROP_COLS = ["total_yield_kg", "yield_kg_per_ha", "final_survival_count",
             "final_weight_g", "fcr_effective", "dataset_version", "generated_at"]

def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df["log_stocking_count"] = np.log1p(df["stocking_count"])
    df["log_culture_days"] = np.log1p(df["culture_days"])
    df["log_pond_area"] = np.log1p(df["pond_area_ha"])
    df["temp_range"] = df["max_temp_c"] - df["min_temp_c"]
    df["do_deficit"] = np.clip(6.0 - df["min_do_mg_l"], 0, None)
    df["temp_stress_degdays"] = np.clip(df["max_temp_c"] - 32.0, 0, None) * df["culture_days"]
    df["season"] = df["season"].astype(str)
    df["intensity"] = df["intensity"].astype(str)
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    y = df[TARGET] if TARGET in df.columns else None
    return X, y
