"""APF V1 -- Shared feature engineering.

Used identically by training (src/models/train_baseline.py) and serving
(src/api/main.py). Do not fork this logic.
"""
import numpy as np
import pandas as pd

TARGET = "total_yield_kg"

# Dropped: all outcome variables that a farmer does not know on day one
DROP_COLS = [
    "total_yield_kg", "yield_kg_per_ha", "final_survival_count",
    "final_weight_g", "fcr_effective", "total_feed_kg", "survival_rate",
    "dataset_version", "generated_at"
]


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()

    # ------------------------------------------------------------------
    # Compute derived fields if missing (serving path vs training path)
    # ------------------------------------------------------------------
    if "stocking_density_fish_ha" not in df.columns:
        df["stocking_density_fish_ha"] = df["stocking_count"] / df["pond_area_ha"]

    # Log transforms for skewed count/area variables
    df["log_stocking_count"] = np.log1p(df["stocking_count"])
    df["log_culture_days"] = np.log1p(df["culture_days"])
    df["log_pond_area"] = np.log1p(df["pond_area_ha"])

    # Temperature features
    df["temp_range"] = df["max_temp_c"] - df["min_temp_c"]
    df["temp_stress_degdays"] = np.clip(df["max_temp_c"] - 32.0, 0, None) * df["culture_days"]
    df["temp_above_optimal_days"] = (
        np.clip(df["max_temp_c"] - 30.0, 0, None)
        / df["temp_range"].replace(0, 1)
        * df["culture_days"]
    )

    # DO features
    df["do_deficit"] = np.clip(6.0 - df["min_do_mg_l"], 0, None)
    df["do_stress_severity"] = np.clip(4.0 - df["min_do_mg_l"], 0, None)
    df["do_critical_severity"] = np.clip(1.5 - df["min_do_mg_l"], 0, None)

    # pH features
    df["ph_stress_low"] = np.clip(6.5 - df["min_ph"], 0, None)
    df["ph_stress_high"] = np.clip(df["min_ph"] - 8.5, 0, None)
    df["ph_stress_total"] = df["ph_stress_low"] + df["ph_stress_high"]

    # Interaction features
    df["density_x_temp_stress"] = df["stocking_density_fish_ha"] * df["temp_stress_degdays"] / 1000
    df["density_x_do_deficit"] = df["stocking_density_fish_ha"] * df["do_deficit"] / 1000
    df["do_temp_interaction"] = df["mean_do_mg_l"] * df["mean_temperature_c"]

    # Categorical encoding
    df["season"] = df["season"].astype(str)
    df["intensity"] = df["intensity"].astype(str)

    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    y = df[TARGET] if TARGET in df.columns else None

    return X, y
