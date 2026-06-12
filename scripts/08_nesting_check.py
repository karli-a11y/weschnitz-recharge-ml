"""Nesting (spatial-independence) robustness check.

Part of the analysis pipeline for:
    Holler JK, Rausch R: Machine-learning-based estimation of groundwater
    recharge from land use in a small crystalline catchment of the Odenwald
    Mountains, Germany.

The sub-catchments of a dry-weather discharge campaign are partially
nested: downstream stations integrate the area of upstream stations
(e.g. the two main-stem Weschnitz stations of > 40 km^2 contain most of
the headwater sub-catchments, and the lower Schlierbach station contains
the upper Schlierbach ones). Leave-one-out cross-validation treats the
measurements as independent, so this script quantifies how strongly the
results depend on the most aggregated (and therefore most overlapping)
stations.

Procedure: the six stations with a catchment area > 10 km^2 (IDs 1, 3,
5, 7, 26, 32) are removed from the filtered training set (n = 28 -> 22),
leaving sub-catchments whose mutual area overlap is small. The best
model spec (random forest, relative shares) is then re-evaluated under
the identical nested-LOOCV protocol, and the SHAP ranking is recomputed
on the refit model.

Outputs
-------
results/nesting_sensitivity_metrics.csv  — LOOCV metrics, full vs. reduced set
results/nesting_sensitivity_shap_rank.csv — SHAP ranking, full vs. reduced set

Requires the output of script 01 (input CSVs).

Usage
-----
python scripts/08_nesting_check.py        (or simply: python run_all.py)
"""
from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
RESULTS = PROJECT / "results"
TARGET = "discharge_per_area_l_s_km2"

# Stations whose upstream area aggregates several other sampled
# sub-catchments; 10 km^2 separates them cleanly from the rest.
NESTED_AREA_THRESHOLD_KM2 = 10.0


def _import_train():
    """Import 02_train_models.py despite the leading digit in the filename."""
    spec = importlib.util.spec_from_file_location(
        "train_models", PROJECT / "scripts" / "02_train_models.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_models"] = mod
    spec.loader.exec_module(mod)
    return mod


def shap_rank(est, X) -> pd.DataFrame:
    """Mean |SHAP| per feature with rank (1 = most important)."""
    try:
        explainer = shap.Explainer(est, X)
        sv = explainer(X).values
    except Exception:
        explainer = shap.Explainer(est.predict, X)
        sv = explainer(X).values
    if sv.ndim == 3:
        sv = sv[..., 0]
    means = np.abs(sv).mean(axis=0)
    out = pd.DataFrame({"feature": X.columns, "mean_abs_shap": means})
    out["rank"] = out["mean_abs_shap"].rank(ascending=False).astype(int)
    return out.sort_values("rank")


def evaluate(df: pd.DataFrame, train, label: str) -> tuple[dict, pd.DataFrame]:
    """LOOCV metrics and SHAP ranking of the (RandomForest, pct) spec."""
    feats = train.FEATURE_SETS["pct"]
    X, y = df[feats], df[TARGET]
    spec = next(s for s in train.make_specs() if s.name == "RandomForest")

    preds = train.loo_predict(spec, X, y)
    m = train.evaluate(y.to_numpy(), preds)
    row = {"setting": label, "n": len(df), **m}
    print(f"{label:>28}  n = {len(df):2d}  LOOCV R² = {m['r2']:+.3f}  "
          f"RMSE = {m['rmse']:.2f}")

    est = train.fit_one(spec, X, y)
    rank = shap_rank(est, X).assign(setting=label)
    return row, rank


def main() -> None:
    train = _import_train()
    df = pd.read_csv(DATA / "discharge_measurements.csv")
    df = df[~df["is_gauge_reference"]].reset_index(drop=True)
    df = df[df[TARGET].between(
        train.PLAUSIBILITY_MIN_L_S_KM2, train.PLAUSIBILITY_MAX_L_S_KM2
    )].reset_index(drop=True)

    nested = df[df["area_km2"] > NESTED_AREA_THRESHOLD_KM2]
    print(f"Removing {len(nested)} aggregated stations "
          f"(area > {NESTED_AREA_THRESHOLD_KM2:.0f} km²): "
          f"IDs = {nested['measurement_id'].tolist()}\n")
    df_reduced = df[df["area_km2"] <= NESTED_AREA_THRESHOLD_KM2].reset_index(drop=True)

    row_full, rank_full = evaluate(df, train, "full (n = 28)")
    row_red, rank_red = evaluate(df_reduced, train, "non-nested only")

    metrics = pd.DataFrame([row_full, row_red])
    metrics.to_csv(RESULTS / "nesting_sensitivity_metrics.csv", index=False)
    ranks = pd.concat([rank_full, rank_red], ignore_index=True)
    ranks.to_csv(RESULTS / "nesting_sensitivity_shap_rank.csv", index=False)

    print("\nSHAP ranking comparison:")
    cmp = (rank_full.set_index("feature")["rank"].rename("rank_full")
           .to_frame()
           .join(rank_red.set_index("feature")["rank"].rename("rank_reduced")))
    print(cmp.to_string())
    print(f"\nWrote {RESULTS / 'nesting_sensitivity_metrics.csv'}")
    print(f"Wrote {RESULTS / 'nesting_sensitivity_shap_rank.csv'}")


if __name__ == "__main__":
    main()
