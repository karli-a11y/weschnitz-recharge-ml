"""Plausibility-filter sensitivity analysis.

Part of the analysis pipeline for:
    Holler JK, Rausch R: Machine-learning-based estimation of groundwater
    recharge from land use in a small crystalline catchment of the Odenwald
    Mountains, Germany.

Reruns the full model comparison (a) with the 100-380 mm/a plausibility
filter active and (b) on the unfiltered data set, then writes a paired
comparison of LOOCV R^2 and SHAP-derived feature rankings.

This tests whether the qualitative land-use ranking depends on the
plausibility filter — i.e., whether the conclusion would change if all
38 measurements were used.

Requires the output of script 01 (input CSVs).
Runtime: several minutes (the full nested-LOOCV comparison runs twice).

Outputs
-------
results/filter_sensitivity_metrics.csv
results/filter_sensitivity_shap_rank.csv

Usage
-----
python scripts/07_filter_sensitivity.py        (or simply: python run_all.py)
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


def _import_train():
    """Import 02_train_models.py despite the leading digit in the filename."""
    spec = importlib.util.spec_from_file_location(
        "train_models", PROJECT / "scripts" / "02_train_models.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_models"] = mod
    spec.loader.exec_module(mod)
    return mod


def shap_rank(est, X) -> list[tuple[str, float]]:
    """Return (feature, mean |SHAP|) pairs sorted descending."""
    try:
        explainer = shap.Explainer(est, X)
        sv = explainer(X).values
    except Exception:
        explainer = shap.Explainer(est.predict, X)
        sv = explainer(X).values
    if sv.ndim == 3:
        sv = sv[..., 0]
    means = np.abs(sv).mean(axis=0)
    pairs = sorted(zip(X.columns, means), key=lambda t: -t[1])
    return pairs


def evaluate_setting(df: pd.DataFrame, train, label: str) -> tuple[pd.DataFrame, list[tuple[str, float]]]:
    """Train all specs on `df` (already filtered or not), return LOOCV
    metrics and the SHAP ranking of the best (RandomForest, pct) model."""
    print(f"\n=== {label} (n = {len(df)}) ===")
    rows = []
    fitted = {}
    for fs_name, feats in train.FEATURE_SETS.items():
        X, y = df[feats], df[TARGET]
        for spec in train.make_specs():
            preds = train.loo_predict(spec, X, y)
            m = train.evaluate(y.to_numpy(), preds)
            rows.append({
                "filter": label, "model": spec.name,
                "feature_set": fs_name, **m,
            })
            est = train.fit_one(spec, X, y)
            fitted[(spec.name, fs_name)] = (est, X, y)
            print(f"  {spec.name:>17}  {fs_name:>4}  LOOCV R²={m['r2']:+.3f}")
    metrics = pd.DataFrame(rows)

    # SHAP ranking for the RF/pct model under this filter setting.
    rf_est, X_rf, _ = fitted[("RandomForest", "pct")]
    sv_ranking = shap_rank(rf_est, X_rf)
    return metrics, sv_ranking


def main() -> None:
    train = _import_train()
    df_full = pd.read_csv(DATA / "discharge_measurements.csv")
    df_full = df_full[~df_full["is_gauge_reference"]].reset_index(drop=True)

    df_filtered = df_full[df_full[TARGET].between(
        train.PLAUSIBILITY_MIN_L_S_KM2, train.PLAUSIBILITY_MAX_L_S_KM2
    )].reset_index(drop=True)

    m_filt, rank_filt = evaluate_setting(df_filtered, train, "filtered")
    m_unfilt, rank_unfilt = evaluate_setting(df_full, train, "unfiltered")

    all_metrics = pd.concat([m_filt, m_unfilt], ignore_index=True)
    all_metrics.to_csv(RESULTS / "filter_sensitivity_metrics.csv", index=False)
    print(f"\nWrote {RESULTS / 'filter_sensitivity_metrics.csv'}")

    # Side-by-side SHAP ranking, RandomForest on pct, under both settings.
    feats_pct = train.FEATURE_SETS["pct"]
    rank_filt_df = (pd.DataFrame(rank_filt, columns=["feature", "mean_abs_shap"])
                    .assign(filter="filtered")
                    .set_index("feature").reindex(feats_pct))
    rank_unfilt_df = (pd.DataFrame(rank_unfilt, columns=["feature", "mean_abs_shap"])
                      .assign(filter="unfiltered")
                      .set_index("feature").reindex(feats_pct))
    rank_table = pd.DataFrame({
        "mean_abs_shap_filtered":   rank_filt_df["mean_abs_shap"],
        "rank_filtered":            rank_filt_df["mean_abs_shap"]
                                      .rank(ascending=False).astype(int),
        "mean_abs_shap_unfiltered": rank_unfilt_df["mean_abs_shap"],
        "rank_unfiltered":          rank_unfilt_df["mean_abs_shap"]
                                      .rank(ascending=False).astype(int),
    })
    rank_table.to_csv(RESULTS / "filter_sensitivity_shap_rank.csv")
    print(f"Wrote {RESULTS / 'filter_sensitivity_shap_rank.csv'}")
    print(rank_table.to_string())


if __name__ == "__main__":
    main()
