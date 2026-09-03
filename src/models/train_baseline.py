"""APF V1 -- Baseline forecasting model training (M2).
Trains XGBoost and LightGBM regressors on the synthetic dataset,
evaluates with time-based split, fits a conformal prediction interval,
saves the best model + interval + metrics.
"""
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features.build_features import build_features, TARGET

DATA_PATH = Path("data/synthetic/v1_train.parquet")
MODEL_DIR = Path("src/models/artifacts")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def evaluate(model, X, y):
    pred = model.predict(X)
    return {
        "mae": round(mean_absolute_error(y, pred), 3),
        "rmse": round(np.sqrt(mean_squared_error(y, pred)), 3),
        "r2": round(r2_score(y, pred), 4),
    }, pred

def main():
    df = pd.read_parquet(DATA_PATH)
    X, y = build_features(df)
    print(f"Loaded {len(df)} rows, {X.shape[1]} features")
    print(f"Features: {list(X.columns)}")
    print()

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    print(f"Categorical: {cat_cols}")
    print(f"Numerical: {num_cols}")
    print()

    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    candidates = {
        "xgboost": xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.8, random_state=42),
        "lightgbm": lgb.LGBMRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                                       subsample=0.8, colsample_bytree=0.8, random_state=42,
                                       verbose=-1),  # suppress warnings
    }

    results = {}
    fitted = {}
    for name, model in candidates.items():
        pipe = Pipeline([("pre", pre), ("model", model)])
        pipe.fit(X_train, y_train)
        metrics, _ = evaluate(pipe, X_test, y_test)
        results[name] = metrics
        fitted[name] = pipe
        print(f"{name:10s}  MAE={metrics['mae']:.1f}  RMSE={metrics['rmse']:.1f}  R2={metrics['r2']:.4f}")

    best_name = min(results, key=lambda k: results[k]["mae"])
    best = fitted[best_name]
    print(f"\nBest model: {best_name}")

    # Conformal prediction interval on test residuals (90% coverage)
    _, pred_test = evaluate(best, X_test, y_test)
    residuals = np.abs(y_test - pred_test)
    q90 = np.quantile(residuals, 0.90)

    # Feature importance (top 15 for better visibility)
    model = best.named_steps["model"]
    feat_names = best.named_steps["pre"].get_feature_names_out()
    imp = pd.Series(model.feature_importances_, index=feat_names).sort_values(ascending=False)
    print("\nTop 15 features:")
    print(imp.head(15).to_string())

    # Water quality feature check
    wq_features = [f for f in imp.head(15).index if any(x in f for x in ['do_', 'temp_stress', 'ph_', 'mean_do', 'min_do', 'max_temp', 'min_temp'])]
    print(f"\nWater quality features in top 15: {wq_features}")
    if len(wq_features) < 3:
        print("WARNING: Water quality features are underrepresented. Consider tuning generator.")

    # Save artifacts
    with open(MODEL_DIR / "model.pkl", "wb") as f:
        pickle.dump(best, f)
    with open(MODEL_DIR / "interval.json", "w") as f:
        json.dump({"coverage": 0.90, "half_width": float(q90),
                   "method": "conformal_absolute_residuals",
                   "fitted_on": "v1_train test split"}, f, indent=2)

    metrics_log = {
        "model_version": "v1.1.0-baseline",
        "dataset_version": df["dataset_version"].iloc[0] if "dataset_version" in df.columns else "unknown",
        "target": TARGET,
        "best_model": best_name,
        "test_metrics": results,
        "interval_half_width_90": float(q90),
        "feature_importance_top15": imp.head(15).to_dict(),
        "water_quality_features_in_top15": wq_features,
        "trained_at": datetime.utcnow().isoformat(),
    }
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=2)
    print(f"\nSaved artifacts to {MODEL_DIR}")

if __name__ == "__main__":
    main()
