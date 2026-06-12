"""Train and cross-validate five regression models for groundwater recharge.

Part of the analysis pipeline for:
    Rausch R, Holler JK: Machine-learning-based estimation of groundwater
    recharge from land use in a small crystalline catchment of the Odenwald
    Mountains, Germany.

The models (ridge regression, random forest, gradient boosting, XGBoost,
support-vector regression) estimate the specific dry-weather discharge
(l s^-1 km^-2, a proxy for groundwater recharge) of 38 sub-catchments from
land-use composition, topography and precipitation. A plausibility filter
of 100-380 mm/a (Larisch 2024) reduces the training set to n = 28.

Two feature parameterisations are evaluated:
  - "km2": absolute land-use areas in km^2 plus total catchment area
  - "pct": relative land-use shares in percent (size-invariant)

Performance is reported via leave-one-out cross-validation (LOOCV); models
with hyperparameters use nested LOOCV (outer n folds, inner 5-fold grid
search). A single 80/20 hold-out split and the in-sample fit are reported
alongside for transparency.

Inputs
------
data/discharge_measurements.csv

Outputs
-------
results/metrics.csv        — LOOCV / hold-out / training metrics per model
results/predictions.csv    — LOOCV predictions per measurement and model
results/best_model.pkl     — refitted best model (highest LOOCV R^2)

Usage
-----
python scripts/02_train_models.py        (or simply: python run_all.py)
"""
from __future__ import annotations

import warnings
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import xgboost as xgb

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
RESULTS = PROJECT / "results"
RANDOM_STATE = 42

FEATURE_SETS = {
    "km2": [
        "urban_km2", "forest_km2", "cropland_km2", "meadow_km2", "area_km2",
        "elevation_mean_m", "elevation_diff_m", "relief_ratio", "precipitation_mm",
    ],
    "pct": [
        "urban_pct", "forest_pct", "cropland_pct", "meadow_pct",
        "elevation_mean_m", "elevation_diff_m", "relief_ratio", "precipitation_mm",
    ],
}
TARGET = "discharge_per_area_l_s_km2"

# Plausibility filter applied in the underlying measurement campaign
# (Larisch 2024): only recharge rates between 100 and 380 mm/a are
# considered plausible. Converted to l/(s km^2) via 1 l/(s km^2) = 31.5576 mm/a.
PLAUSIBILITY_MIN_L_S_KM2 = 100.0 / 31.5576
PLAUSIBILITY_MAX_L_S_KM2 = 380.0 / 31.5576


@dataclass
class ModelSpec:
    """A named estimator plus its hyperparameter grid (None = no tuning)."""
    name: str
    estimator: Any
    param_grid: dict | None


def make_specs() -> list[ModelSpec]:
    """The five model families compared in the manuscript.

    Scale-sensitive models (ridge, SVR) are wrapped in a pipeline that
    standardises the features first; tree ensembles are scale-invariant
    and used as-is.
    """
    return [
        ModelSpec(
            "Ridge",
            Pipeline([("scale", StandardScaler()),
                      ("est", Ridge(random_state=RANDOM_STATE))]),
            {"est__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        ),
        ModelSpec(
            "RandomForest",
            RandomForestRegressor(random_state=RANDOM_STATE),
            {"n_estimators": [100, 300, 500], "max_depth": [3, 5, None]},
        ),
        ModelSpec(
            "GradientBoosting",
            GradientBoostingRegressor(random_state=RANDOM_STATE),
            {"n_estimators": [100, 300], "learning_rate": [0.03, 0.1],
             "max_depth": [2, 3]},
        ),
        ModelSpec(
            "XGBoost",
            xgb.XGBRegressor(random_state=RANDOM_STATE,
                             objective="reg:squarederror", verbosity=0),
            {"n_estimators": [100, 300], "learning_rate": [0.03, 0.1],
             "max_depth": [2, 3], "reg_alpha": [0, 0.1]},
        ),
        ModelSpec(
            "SVR",
            Pipeline([("scale", StandardScaler()),
                      ("est", SVR(kernel="rbf"))]),
            {"est__C": [0.1, 1.0, 10.0, 100.0],
             "est__gamma": ["scale", 0.01, 0.1, 1.0]},
        ),
    ]


def fit_one(spec: ModelSpec, X: pd.DataFrame, y: pd.Series):
    """Fit a spec on (X, y); tune hyperparameters via inner 5-fold CV."""
    if spec.param_grid is None:
        est = deepcopy(spec.estimator).fit(X, y)
        return est
    inner = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(
        deepcopy(spec.estimator),
        spec.param_grid,
        scoring="neg_mean_squared_error",
        cv=inner,
        n_jobs=-1,
    ).fit(X, y)
    return gs


def loo_predict(spec: ModelSpec, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """Leave-one-out predictions; hyperparameters are re-tuned in every fold
    (nested CV), so no information from the held-out sample leaks into the
    model selection."""
    loo = LeaveOneOut()
    preds = np.empty(len(y))
    for train_idx, test_idx in loo.split(X):
        est = fit_one(spec, X.iloc[train_idx], y.iloc[train_idx])
        preds[test_idx[0]] = float(np.atleast_1d(est.predict(X.iloc[test_idx]))[0])
    return preds


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """R^2, RMSE, MAE and MAPE (with zero-protection for the latter)."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    safe_true = np.where(y_true == 0, 1e-9, y_true)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / safe_true)) * 100),
    }


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    df = pd.read_csv(DATA / "discharge_measurements.csv")
    df = df[~df["is_gauge_reference"]].reset_index(drop=True)
    n_total = len(df)
    plausible = df[TARGET].between(PLAUSIBILITY_MIN_L_S_KM2, PLAUSIBILITY_MAX_L_S_KM2)
    excluded_ids = df.loc[~plausible, "measurement_id"].tolist()
    df = df[plausible].reset_index(drop=True)
    print(
        f"Plausibility filter (100-380 mm/a): kept {len(df)} of {n_total} "
        f"measurements; excluded IDs = {excluded_ids}\n"
    )

    metrics_rows = []
    predictions_rows = []
    fitted: dict[tuple[str, str], Any] = {}

    for fs_name, feats in FEATURE_SETS.items():
        X = df[feats]
        y = df[TARGET]
        print(f"=== Feature set: {fs_name} ===")
        for spec in make_specs():
            # 1) Leave-one-out cross-validation (primary, rigorous estimate)
            preds_loo = loo_predict(spec, X, y)
            m_loo = evaluate(y.to_numpy(), preds_loo)

            # 2) Single 80/20 hold-out split (for direct comparison with the
            #    original campaign report; included for transparency, not for
            #    decision-making)
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=RANDOM_STATE
            )
            est_h = fit_one(spec, X_tr, y_tr)
            preds_hold = est_h.predict(X_te)
            m_hold = evaluate(y_te.to_numpy(), preds_hold)

            # 3) Training fit on all data (in-sample R^2; quantifies overfitting)
            est_full = fit_one(spec, X, y)
            preds_train = est_full.predict(X)
            m_train = evaluate(y.to_numpy(), preds_train)

            metrics_rows.append({
                "model": spec.name, "feature_set": fs_name,
                **m_loo,
                "r2_train": m_train["r2"], "rmse_train": m_train["rmse"],
                "r2_holdout": m_hold["r2"], "rmse_holdout": m_hold["rmse"],
            })
            for tid, yt, yp in zip(df["measurement_id"], y, preds_loo):
                predictions_rows.append({
                    "model": spec.name, "feature_set": fs_name,
                    "measurement_id": int(tid),
                    "y_true": float(yt), "y_pred": float(yp),
                })
            fitted[(spec.name, fs_name)] = est_full
            print(
                f"  {spec.name:>17}  "
                f"LOOCV R2={m_loo['r2']:+.3f}  "
                f"Hold-out R2={m_hold['r2']:+.3f}  "
                f"Train R2={m_train['r2']:+.3f}"
            )

    metrics = pd.DataFrame(metrics_rows).sort_values(
        ["feature_set", "r2"], ascending=[True, False]
    )
    metrics.to_csv(RESULTS / "metrics.csv", index=False)
    pd.DataFrame(predictions_rows).to_csv(RESULTS / "predictions.csv", index=False)

    # Persist the best model (highest LOOCV R^2 across all model x
    # feature-set combinations) for the downstream explainability and
    # counterfactual scripts.
    best_overall = metrics.sort_values("r2", ascending=False).iloc[0]
    best_key = (best_overall["model"], best_overall["feature_set"])
    best_est = fitted[best_key]
    joblib.dump(
        {
            "name": best_key[0],
            "feature_set": best_key[1],
            "estimator": best_est,
            "feature_names": FEATURE_SETS[best_key[1]],
        },
        RESULTS / "best_model.pkl",
    )
    print(f"\nBest: {best_key[0]} on {best_key[1]} (R2={best_overall['r2']:+.3f})")
    print("Saved -> results/best_model.pkl")


if __name__ == "__main__":
    main()
