"""APF V1 -- Mechanistic synthetic data generator for Nile tilapia."""
import argparse, json, random, sys
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd, yaml
from scipy import stats

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "tilapia_biology_params.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    BIO = yaml.safe_load(f)

G = BIO["growth"]; M = BIO["mortality"]; F = BIO["feed_conversion"]
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED); random.seed(RANDOM_SEED)
LITERATURE_YIELD_LOW_KG_HA = 1500
LITERATURE_YIELD_HIGH_KG_HA = 8000

def sgr_for_temperature(temp_c: float) -> float:
    opt_low, opt_high = G["sgr_optimal_temp_low_c"], G["sgr_optimal_temp_high_c"]
    opt_sgr = G["sgr_optimal_value_pct_per_day"]
    if opt_low <= temp_c <= opt_high:
        return opt_sgr
    sigma = 4.0
    penalty = np.exp(-0.5 * ((temp_c - (opt_low + opt_high) / 2) / sigma) ** 2)
    return opt_sgr * max(penalty, 0.05)

def daily_mortality_rate(temp_c: float, do_mg_l: float, ph: float) -> float:
    rate = 0.0
    if do_mg_l < M["do_lethal_mg_l"]:
        rate += M["daily_mortality_rate_do_below_lethal"]
    elif do_mg_l < M["do_critical_mg_l"]:
        rate += M["daily_mortality_rate_do_below_critical"]
    if temp_c > M["temp_lethal_high_c"]:
        rate += M["daily_mortality_rate_temp_above_lethal"]
    elif temp_c > M["temp_stress_high_c"]:
        rate += M["daily_mortality_rate_temp_above_stress"]
    if temp_c < M["temp_lethal_low_c"]:
        rate += 0.05
    elif temp_c < M["temp_stress_low_c"]:
        rate += 0.003
    if ph < M["ph_stress_low"] or ph > M["ph_stress_high"]:
        rate += M["daily_mortality_rate_ph_outside_stress"]
    if ph < M["ph_lethal_low"] or ph > M["ph_lethal_high"]:
        rate += 0.05
    return min(rate, 0.5)

def fcr_for_conditions(intensity: str, temp_c: float, do_mg_l: float) -> float:
    low, high = {
        "extensive": (F["fcr_extensive_low"], F["fcr_extensive_high"]),
        "semi-intensive": (F["fcr_semi_intensive_low"], F["fcr_semi_intensive_high"]),
        "intensive": (F["fcr_intensive_low"], F["fcr_intensive_high"]),
    }.get(intensity, (1.2, 2.0))
    base = np.random.uniform(low, high)
    if temp_c > M["temp_stress_high_c"] or do_mg_l < M["do_critical_mg_l"]:
        base *= 1.15
    if temp_c > M["temp_lethal_high_c"] or do_mg_l < M["do_lethal_mg_l"]:
        base *= 1.3
    return round(base, 2)

def generate_env_series(culture_days: int, mean_temp: float, season: str) -> pd.DataFrame:
    days = np.arange(culture_days)
    if season == "summer":
        temp_trend = mean_temp + 1.5 * np.sin(2 * np.pi * days / 365)
    elif season == "winter":
        temp_trend = mean_temp - 2.0 * np.sin(2 * np.pi * days / 365)
    else:
        temp_trend = np.full(culture_days, mean_temp)
    temp_noise = np.random.normal(0, 1.0, culture_days)
    temperature = np.clip(temp_trend + temp_noise, 15.0, 40.0)
    do_base = 8.0 - 0.15 * (temperature - 25.0)
    do_noise = np.random.normal(0, 0.8, culture_days)
    dissolved_oxygen = np.clip(do_base + do_noise, 0.1, 15.0)
    ph_base = 7.5 + 0.05 * (dissolved_oxygen - 7.0)
    ph_noise = np.random.normal(0, 0.3, culture_days)
    ph = np.clip(ph_base + ph_noise, 5.0, 10.0)
    return pd.DataFrame({
        "day": days,
        "temperature_c": np.round(temperature, 2),
        "dissolved_oxygen_mg_l": np.round(dissolved_oxygen, 2),
        "ph": np.round(ph, 2),
    })

def simulate_cycle(pond_area_ha, stocking_count, initial_weight_g, culture_days, mean_temp, season, intensity, feed_protein_pct):
    env = generate_env_series(culture_days, mean_temp, season)
    biomass_g = stocking_count * initial_weight_g
    survival_count = stocking_count
    total_feed_kg = 0.0
    for day in range(culture_days):
        temp = env.loc[day, "temperature_c"]
        do = env.loc[day, "dissolved_oxygen_mg_l"]
        ph = env.loc[day, "ph"]
        mort_rate = daily_mortality_rate(temp, do, ph)
        daily_deaths = int(np.random.binomial(survival_count, mort_rate))
        survival_count -= daily_deaths
        if survival_count <= 0:
            survival_count = 0; break
        sgr = sgr_for_temperature(temp)
        protein_factor = 1.0 + 0.005 * (feed_protein_pct - 30)
        daily_growth_factor = np.exp((sgr / 100.0) * protein_factor)
        biomass_g *= daily_growth_factor
        fcr = fcr_for_conditions(intensity, temp, do)
        daily_feed_g = (biomass_g * (sgr / 100.0) * protein_factor) * fcr
        total_feed_kg += daily_feed_g / 1000.0
    final_weight_g = biomass_g / max(survival_count, 1)
    total_yield_kg = biomass_g / 1000.0
    yield_per_ha = total_yield_kg / pond_area_ha
    if not (LITERATURE_YIELD_LOW_KG_HA <= yield_per_ha <= LITERATURE_YIELD_HIGH_KG_HA):
        return None
    return {
        "pond_area_ha": round(pond_area_ha, 3),
        "stocking_count": stocking_count,
        "stocking_density_fish_ha": round(stocking_count / pond_area_ha, 0),
        "initial_weight_g": round(initial_weight_g, 1),
        "culture_days": culture_days,
        "mean_temperature_c": round(mean_temp, 1),
        "season": season,
        "intensity": intensity,
        "feed_protein_pct": feed_protein_pct,
        "final_survival_count": survival_count,
        "survival_rate": round(survival_count / stocking_count, 3),
        "final_weight_g": round(final_weight_g, 1),
        "total_yield_kg": round(total_yield_kg, 1),
        "yield_kg_per_ha": round(yield_per_ha, 1),
        "total_feed_kg": round(total_feed_kg, 1),
        "fcr_effective": round(total_feed_kg / total_yield_kg, 2) if total_yield_kg > 0 else None,
        "mean_do_mg_l": round(env["dissolved_oxygen_mg_l"].mean(), 2),
        "min_do_mg_l": round(env["dissolved_oxygen_mg_l"].min(), 2),
        "mean_ph": round(env["ph"].mean(), 2),
        "min_ph": round(env["ph"].min(), 2),
        "max_temp_c": round(env["temperature_c"].max(), 1),
        "min_temp_c": round(env["temperature_c"].min(), 1),
        "dataset_version": "v1.0.0-mechanistic",
        "generated_at": datetime.utcnow().isoformat(),
    }

def generate_dataset(n_samples: int, max_retries: int = 5) -> pd.DataFrame:
    records = []
    attempts = 0
    while len(records) < n_samples and attempts < n_samples * max_retries:
        attempts += 1
        pond_area = np.random.choice([0.05, 0.1, 0.2, 0.5, 1.0])
        stocking_density = np.random.randint(5000, 50001)
        stocking_count = int(stocking_density * pond_area)
        initial_weight = np.random.uniform(5.0, 50.0)
        culture_days = int(np.random.uniform(90, 181))
        mean_temp = np.random.uniform(22.0, 32.0)
        season = np.random.choice(["summer", "winter", "monsoon"])
        intensity = np.random.choice(["extensive", "semi-intensive", "intensive"], p=[0.2, 0.6, 0.2])
        protein = np.random.choice([25, 30, 35, 40])
        result = simulate_cycle(pond_area, stocking_count, initial_weight, culture_days, mean_temp, season, intensity, protein)
        if result:
            records.append(result)
    return pd.DataFrame(records)

def validate_marginals(df: pd.DataFrame) -> dict:
    return {
        "yield_kg_per_ha_in_range": (LITERATURE_YIELD_LOW_KG_HA <= df["yield_kg_per_ha"].mean() <= LITERATURE_YIELD_HIGH_KG_HA),
        "mean_temp_in_range": (20 <= df["mean_temperature_c"].mean() <= 35),
        "mean_do_positive": (df["mean_do_mg_l"].min() > 0),
        "survival_rate_0_to_1": (df["survival_rate"].between(0, 1).all()),
        "fcr_positive": (df["fcr_effective"].dropna() > 0).all(),
    }

def validate_relationships(df: pd.DataFrame) -> dict:
    temp_do_corr = df["mean_temperature_c"].corr(df["mean_do_mg_l"])
    return {
        "temp_do_negative_corr": temp_do_corr < -0.1,
        "high_temp_reduces_survival": df[df["mean_temperature_c"] > 30]["survival_rate"].mean() < df[df["mean_temperature_c"] < 28]["survival_rate"].mean(),
        "high_temp_worsens_fcr": df[df["mean_temperature_c"] > 30]["fcr_effective"].mean() > df[df["mean_temperature_c"] < 28]["fcr_effective"].mean(),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--out", type=str, default="data/synthetic/v1_train.parquet")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    np.random.seed(args.seed); random.seed(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Generating {args.n_samples} synthetic culture cycles ...")
    df = generate_dataset(args.n_samples)
    v1 = validate_marginals(df)
    v2 = validate_relationships(df)
    print(f"  Marginal checks: {v1}")
    print(f"  Relationship checks: {v2}")
    all_pass = all(v1.values()) and all(v2.values())
    print("All validation checks passed." if all_pass else "WARNING: Some validation checks failed. Review before training.")
    df.to_parquet(out_path, index=False)
    df.head(1000).to_csv(out_path.with_suffix(".csv"), index=False)
    report = {
        "dataset_version": "v1.0.0-mechanistic",
        "n_samples": len(df),
        "seed": args.seed,
        "generated_at": datetime.utcnow().isoformat(),
        "marginal_validation": v1,
        "relationship_validation": v2,
        "summary_stats": df.describe().to_dict(),
    }
    report_path = Path("data/validation") / f"{out_path.stem}_validation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Saved: {out_path}")
    print(f"Report: {report_path}")
    print("\nPreview (first 3 rows):")
    print(df.head(3).T)

if __name__ == "__main__":
    main()
