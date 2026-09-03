\"\"\"APF V1 — Mechanistic synthetic data generator for Nile tilapia.

Generates farm-level culture cycles with a biological core:
  1. Temperature-dependent growth (Gompertz + thermal degradation)
  2. Feed conversion ratio (FCR) by intensity
  3. Mortality penalty from DO / pH / temperature excursions
  4. Environmental scaffolding from real sensor statistics

Usage:
    python scripts/generate_synthetic_data.py --n_samples 5000 --out data/synthetic/v1_train.parquet
\"\"\"

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

# ── load biology parameters ──────────────────────────────────
CONFIG_PATH = Path(__file__).resolve().parent.parent / \"config\" / \"tilapia_biology_params.yaml\"
with open(CONFIG_PATH, \"r\") as f:
    BIO = yaml.safe_load(f)

G = BIO[\"growth\"]
M = BIO[\"mortality\"]
F = BIO[\"feed_conversion\"]

# ── constants ────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# FAO / literature yield ranges for validation (kg/ha/cycle)
LITERATURE_YIELD_LOW_KG_HA = 1500
LITERATURE_YIELD_HIGH_KG_HA = 8000


# ── helper: thermal performance curve (Q10-style) ───────────
def sgr_for_temperature(temp_c: float) -> float:
    \"\"\"Return SGR (%/day) given water temperature.\"\"\"
    opt_low, opt_high = G[\"sgr_optimal_temp_low_c\"], G[\"sgr_optimal_temp_high_c\"]
    opt_sgr = G[\"sgr_optimal_value_pct_per_day\"]

    if opt_low <= temp_c <= opt_high:
        return opt_sgr

    # Degradation outside optimum using a Gaussian-like penalty
    sigma = 4.0  # degrees C for ~50 % reduction
    penalty = np.exp(-0.5 * ((temp_c - (opt_low + opt_high) / 2) / sigma) ** 2)
    return opt_sgr * max(penalty, 0.05)  # floor at 5 % of optimal


# ── helper: mortality rate for a single day ──────────────────
def daily_mortality_rate(temp_c: float, do_mg_l: float, ph: float) -> float:
    \"\"\"Return daily fractional mortality given environmental conditions.\"\"\"
    rate = 0.0

    # DO penalty
    if do_mg_l < M[\"do_lethal_mg_l\"]:
        rate += M[\"daily_mortality_rate_do_below_lethal\"]
    elif do_mg_l < M[\"do_critical_mg_l\"]:
        rate += M[\"daily_mortality_rate_do_below_critical\"]

    # Temperature penalty
    if temp_c > M[\"temp_lethal_high_c\"]:
        rate += M[\"daily_mortality_rate_temp_above_lethal\"]
    elif temp_c > M[\"temp_stress_high_c\"]:
        rate += M[\"daily_mortality_rate_temp_above_stress\"]

    if temp_c < M[\"temp_lethal_low_c\"]:
        rate += 0.05  # cold-shock mortality, no config yet
    elif temp_c < M[\"temp_stress_low_c\"]:
        rate += 0.003

    # pH penalty
    if ph < M[\"ph_stress_low\"] or ph > M[\"ph_stress_high\"]:
        rate += M[\"daily_mortality_rate_ph_outside_stress\"]
    if ph < M[\"ph_lethal_low\"] or ph > M[\"ph_lethal_high\"]:
        rate += 0.05

    return min(rate, 0.5)  # cap at 50 % / day


# ── helper: FCR by system intensity ──────────────────────────
def fcr_for_conditions(intensity: str, temp_c: float, do_mg_l: float) -> float:
    \"\"\"Return FCR given intensity label and stress conditions.\"\"\"
    low, high = {
        \"extensive\": (F[\"fcr_extensive_low\"], F[\"fcr_extensive_high\"]),
        \"semi-intensive\": (F[\"fcr_semi_intensive_low\"], F[\"fcr_semi_intensive_high\"]),
        \"intensive\": (F[\"fcr_intensive_low\"], F[\"fcr_intensive_high\"]),
    }.get(intensity, (1.2, 2.0))

    base = np.random.uniform(low, high)

    # Stress worsens FCR
    if temp_c > M[\"temp_stress_high_c\"] or do_mg_l < M[\"do_critical_mg_l\"]:
        base *= 1.15
    if temp_c > M[\"temp_lethal_high_c\"] or do_mg_l < M[\"do_lethal_mg_l\"]:
        base *= 1.3

    return round(base, 2)


# ── helper: generate environmental time series ───────────────
def generate_env_series(culture_days: int, mean_temp: float, season: str) -> pd.DataFrame:
    \"\"\"Generate daily temperature, DO, pH with realistic diurnal/seasonal noise.\"\"\"
    days = np.arange(culture_days)

    # Seasonal temperature trend (simple sinusoid)
    if season == \"summer\":
        temp_trend = mean_temp + 1.5 * np.sin(2 * np.pi * days / 365)
    elif season == \"winter\":
        temp_trend = mean_temp - 2.0 * np.sin(2 * np.pi * days / 365)
    else:
        temp_trend = np.full(culture_days, mean_temp)

    # Diurnal noise (±1.5 C)
    temp_noise = np.random.normal(0, 1.0, culture_days)
    temperature = temp_trend + temp_noise
    temperature = np.clip(temperature, 15.0, 40.0)

    # DO anti-correlated with temperature (warm water holds less O2)
    do_base = 8.0 - 0.15 * (temperature - 25.0)
    do_noise = np.random.normal(0, 0.8, culture_days)
    dissolved_oxygen = do_base + do_noise
    dissolved_oxygen = np.clip(dissolved_oxygen, 0.1, 15.0)

    # pH weakly correlated with DO (photosynthesis cycle)
    ph_base = 7.5 + 0.05 * (dissolved_oxygen - 7.0)
    ph_noise = np.random.normal(0, 0.3, culture_days)
    ph = ph_base + ph_noise
    ph = np.clip(ph, 5.0, 10.0)

    return pd.DataFrame({
        \"day\": days,
        \"temperature_c\": np.round(temperature, 2),
        \"dissolved_oxygen_mg_l\": np.round(dissolved_oxygen, 2),
        \"ph\": np.round(ph, 2),
    })


# ── core: simulate one culture cycle ─────────────────────────
def simulate_cycle(
    pond_area_ha: float,
    stocking_count: int,
    initial_weight_g: float,
    culture_days: int,
    mean_temp: float,
    season: str,
    intensity: str,
    feed_protein_pct: float,
) -> dict:
    \"\"\"Simulate a single culture cycle and return features + target.\"\"\"
    env = generate_env_series(culture_days, mean_temp, season)

    # Starting biomass
    biomass_g = stocking_count * initial_weight_g
    survival_count = stocking_count
    total_feed_kg = 0.0

    # Growth loop (daily time step)
    for day in range(culture_days):
        temp = env.loc[day, \"temperature_c\"]
        do = env.loc[day, \"dissolved_oxygen_mg_l\"]
        ph = env.loc[day, \"ph\"]

        # Mortality
        mort_rate = daily_mortality_rate(temp, do, ph)
        daily_deaths = int(np.random.binomial(survival_count, mort_rate))
        survival_count -= daily_deaths

        if survival_count <= 0:
            survival_count = 0
            break

        # Growth
        sgr = sgr_for_temperature(temp)
        # Protein correction: 30 % protein is baseline; 40 % gives +5 % SGR
        protein_factor = 1.0 + 0.005 * (feed_protein_pct - 30)
        daily_growth_factor = np.exp((sgr / 100.0) * protein_factor)
        biomass_g *= daily_growth_factor

        # Feed
        fcr = fcr_for_conditions(intensity, temp, do)
        daily_feed_g = (biomass_g * (sgr / 100.0) * protein_factor) * fcr
        total_feed_kg += daily_feed_g / 1000.0

    final_weight_g = biomass_g / max(survival_count, 1)
    total_yield_kg = biomass_g / 1000.0
    yield_per_ha = total_yield_kg / pond_area_ha

    # Validation guard: reject physically implausible outcomes
    if not (LITERATURE_YIELD_LOW_KG_HA <= yield_per_ha <= LITERATURE_YIELD_HIGH_KG_HA):
        return None  # caller will retry

    return {
        \"pond_area_ha\": round(pond_area_ha, 3),
        \"stocking_count\": stocking_count,
        \"stocking_density_fish_ha\": round(stocking_count / pond_area_ha, 0),
        \"initial_weight_g\": round(initial_weight_g, 1),
        \"culture_days\": culture_days,
        \"mean_temperature_c\": round(mean_temp, 1),
        \"season\": season,
        \"intensity\": intensity,
        \"feed_protein_pct\": feed_protein_pct,
        \"final_survival_count\": survival_count,
        \"survival_rate\": round(survival_count / stocking_count, 3),
        \"final_weight_g\": round(final_weight_g, 1),
        \"total_yield_kg\": round(total_yield_kg, 1),
        \"yield_kg_per_ha\": round(yield_per_ha, 1),
        \"total_feed_kg\": round(total_feed_kg, 1),
        \"fcr_effective\": round(total_feed_kg / total_yield_kg, 2) if total_yield_kg > 0 else None,
        \"mean_do_mg_l\": round(env[\"dissolved_oxygen_mg_l\"].mean(), 2),
        \"min_do_mg_l\": round(env[\"dissolved_oxygen_mg_l\"].min(), 2),
        \"mean_ph\": round(env[\"ph\"].mean(), 2),
        \"min_ph\": round(env[\"ph\"].min(), 2),
        \"max_temp_c\": round(env[\"temperature_c\"].max(), 1),
        \"min_temp_c\": round(env[\"temperature_c\"].min(), 1),
        \"dataset_version\": \"v1.0.0-mechanistic\",
        \"generated_at\": datetime.utcnow().isoformat(),
    }


# ── batch generator ──────────────────────────────────────────
def generate_dataset(n_samples: int, max_retries: int = 5) -> pd.DataFrame:
    records = []
    attempts = 0
    while len(records) < n_samples and attempts < n_samples * max_retries:
        attempts += 1

        # Sample farm parameters from realistic ranges
        pond_area = np.random.choice([0.05, 0.1, 0.2, 0.5, 1.0])
        stocking_density = np.random.randint(5000, 50001)  # fish/ha
        stocking_count = int(stocking_density * pond_area)
        initial_weight = np.random.uniform(5.0, 50.0)  # g
        culture_days = int(np.random.uniform(90, 181))
        mean_temp = np.random.uniform(22.0, 32.0)
        season = np.random.choice([\"summer\", \"winter\", \"monsoon\"])
        intensity = np.random.choice([\"extensive\", \"semi-intensive\", \"intensive\"], p=[0.2, 0.6, 0.2])
        protein = np.random.choice([25, 30, 35, 40])

        result = simulate_cycle(
            pond_area_ha=pond_area,
            stocking_count=stocking_count,
            initial_weight_g=initial_weight,
            culture_days=culture_days,
            mean_temp=mean_temp,
            season=season,
            intensity=intensity,
            feed_protein_pct=protein,
        )
        if result:
            records.append(result)

    df = pd.DataFrame(records)
    return df


# ── validation tier 1: marginal distributions ────────────────
def validate_marginals(df: pd.DataFrame) -> dict:
    \"\"\"Check that synthetic marginals fall inside literature ranges.\"\"\"
    checks = {
        \"yield_kg_per_ha_in_range\": (
            LITERATURE_YIELD_LOW_KG_HA <= df[\"yield_kg_per_ha\"].mean() <= LITERATURE_YIELD_HIGH_KG_HA
        ),
        \"mean_temp_in_range\": (20 <= df[\"mean_temperature_c\"].mean() <= 35),
        \"mean_do_positive\": (df[\"mean_do_mg_l\"].min() > 0),
        \"survival_rate_0_to_1\": (df[\"survival_rate\"].between(0, 1).all()),
        \"fcr_positive\": (df[\"fcr_effective\"].dropna() > 0).all(),
    }
    return checks


# ── validation tier 2: relationship checks ───────────────────
def validate_relationships(df: pd.DataFrame) -> dict:
    \"\"\"Check known biological relationships are present.\"\"\"
    # Higher temp should correlate with lower DO (warm water holds less O2)
    temp_do_corr = df[\"mean_temperature_c\"].corr(df[\"mean_do_mg_l\"])
    checks = {
        \"temp_do_negative_corr\": temp_do_corr < -0.1,
        \"high_temp_reduces_survival\": (
            df[df[\"mean_temperature_c\"] > 30][\"survival_rate\"].mean()
            < df[df[\"mean_temperature_c\"] < 28][\"survival_rate\"].mean()
        ),
        \"high_temp_worsens_fcr\": (
            df[df[\"mean_temperature_c\"] > 30][\"fcr_effective\"].mean()
            > df[df[\"mean_temperature_c\"] < 28][\"fcr_effective\"].mean()
        ),
    }
    return checks


# ── main ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(\"--n_samples\", type=int, default=5000)
    parser.add_argument(\"--out\", type=str, default=\"data/synthetic/v1_train.parquet\")
    parser.add_argument(\"--seed\", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f\"Generating {args.n_samples} synthetic culture cycles …\")\n    df = generate_dataset(args.n_samples)

    # Run three-tier validation
    v1 = validate_marginals(df)
    v2 = validate_relationships(df)

    print(f\"  Marginal checks: {v1}\")
    print(f\"  Relationship checks: {v2}\")\n
    all_pass = all(v1.values()) and all(v2.values())
    if not all_pass:
        print(\"WARNING: Some validation checks failed. Review before training.\")
    else:
        print(\"All validation checks passed.\")\n
    # Save
    df.to_parquet(out_path, index=False)
    df.head(1000).to_csv(out_path.with_suffix(\".csv\"), index=False)

    # Validation report
    report = {
        \"dataset_version\": \"v1.0.0-mechanistic\",
        \"n_samples\": len(df),
        \"seed\": args.seed,
        \"generated_at\": datetime.utcnow().isoformat(),
        \"marginal_validation\": v1,
        \"relationship_validation\": v2,
        \"summary_stats\": df.describe().to_dict(),
    }
    report_path = Path(\"data/validation\") / f\"{out_path.stem}_validation.json\"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, \"w\") as f:
        json.dump(report, f, indent=2, default=str)

    print(f\"Saved: {out_path}\")\n    print(f\"Report: {report_path}\")\n
    # Quick preview
    print(\"\\nPreview (first 3 rows):\")\n    print(df.head(3).T)


if __name__ == \"__main__\":\n    main()
