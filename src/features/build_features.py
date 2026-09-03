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
    # 1. Derived fields that the generator pre-computes but the API may
    #    not provide (farmers know stocking_count + pond_area, not density)
    # ------------------------------------------------------------------
    if "stocking_density_fish_ha" not in df.columns:
        df["stocking_density_fish_ha"] = df["stocking_count"] / df["pond_area_ha"]

    # ------------------------------------------------------------------
    # 2. Log transforms for skewed count/area variables
    # ------------------------------------------------------------------
    df["log_stocking_count"] = np.log1p(df["stocking_count"])
    df["log_culture_days"] = np.log1p(df["culture_days"])
    df["log_pond_area"] = np.log1p(df["pond_area_ha"])

    # ------------------------------------------------------------------
    # 3. Temperature features
    # ------------------------------------------------------------------
    df["temp_range"] = df["max_temp_c"] - df["min_temp_c"]
    df["temp_stress_degdays"] = np.clip(df["max_temp_c"] - 32.0, 0, None) * df["culture_days"]
    df["temp_above_optimal_days"] = (
        np.clip(df["max_temp_c"] - 30.0, 0, None)
        / df["temp_range"].replace(0, 1)
        * df["culture_days"]
    )

    # ------------------------------------------------------------------
    # 4. DO features
    # ------------------------------------------------------------------
    df["do_deficit"] = np.clip(6.0 - df["min_do_mg_l"], 0, None)
    df["do_stress_severity"] = np.clip(4.0 - df["min_do_mg_l"], 0, None)
    df["do_critical_severity"] = np.clip(1.5 - df["min_do_mg_l"], 0, None)

    # ------------------------------------------------------------------
    # 5. pH features
    # ------------------------------------------------------------------
    df["ph_stress_low"] = np.clip(6.5 - df["min_ph"], 0, None)
    df["ph_stress_high"] = np.clip(df["min_ph"] - 8.5, 0, None)
    df["ph_stress_total"] = df["ph_stress_low"] + df["ph_stress_high"]

    # ------------------------------------------------------------------
    # 6. Stress-day aggregates (generator creates these from daily sim;
    #    serving path must estimate them from summary stats above)
    # ------------------------------------------------------------------
    if "temp_stress_days" not in df.columns:
        temp_stress_frac = np.clip((df["max_temp_c"] - 32.0) / 5.0, 0, 0.35)
        df["temp_stress_days"] = (temp_stress_frac * df["culture_days"]).round().astype(int)

    if "do_stress_days" not in df.columns:
        do_stress_frac = np.clip(df["do_deficit"] / 6.0, 0, 0.30)
        df["do_stress_days"] = (do_stress_frac * df["culture_days"]).round().astype(int)

    if "ph_stress_days" not in df.columns:
        ph_stress_frac = np.clip(df["ph_stress_total"] / 2.0, 0, 0.25)
        df["ph_stress_days"] = (ph_stress_frac * df["culture_days"]).round().astype(int)

    if "stress_days" not in df.columns:
        df["stress_days"] = df[["temp_stress_days", "do_stress_days", "ph_stress_days"]].max(axis=1)

    # ------------------------------------------------------------------
    # 7. Interaction features
    # ------------------------------------------------------------------
    df["density_x_temp_stress"] = df["stocking_density_fish_ha"] * df["temp_stress_degdays"] / 1000
    df["density_x_do_deficit"] = df["stocking_density_fish_ha"] * df["do_deficit"] / 1000
    df["do_temp_interaction"] = df["mean_do_mg_l"] * df["mean_temperature_c"]

    # ------------------------------------------------------------------
    # 8. Categorical encoding
    # ------------------------------------------------------------------
    df["season"] = df["season"].astype(str)
    df["intensity"] = df["intensity"].astype(str)

    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    y = df[TARGET] if TARGET in df.columns else None
    return X, y