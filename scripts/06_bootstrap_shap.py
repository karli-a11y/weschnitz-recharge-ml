"""Bootstrap robustness of the SHAP feature ranking.

Part of the analysis pipeline for:
    Holler JK, Rausch R: Machine-learning-based estimation of groundwater
    recharge from land use in a small crystalline catchment of the Odenwald
    Mountains, Germany.

Procedure: B = 500 bootstrap resamples (with replacement, size = n) of the
filtered training set. For each resample we (a) refit the best model spec
(random forest, with hyperparameters re-tuned via inner 5-fold CV) and
(b) recompute the SHAP-based mean |SHAP| ranking on the in-bag training
matrix.

The output gives, for each feature, the empirical distribution of its
mean-absolute-SHAP value and of its rank position over the B resamples.
This quantifies how stable the land-use feature ranking is against random
perturbation of the n = 28 training set.

Requires the outputs of scripts 01 and 02 (input CSVs, best_model.pkl).
Runtime: a few minutes (500 grid searches).

Outputs
-------
results/bootstrap_shap.csv          — long-format SHAP value per (run, feature)
results/bootstrap_shap_summary.csv  — per-feature summary (P[rank == 1], etc.)
figures/fig12_bootstrap_shap.{pdf,png}   (manuscript Fig. 10)

Usage
-----
python scripts/06_bootstrap_shap.py        (or simply: python run_all.py)
"""
from __future__ import annotations

import importlib.util
import sys
import warnings
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, KFold

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
RESULTS = PROJECT / "results"
FIGURES = PROJECT / "figures"
TARGET = "discharge_per_area_l_s_km2"
N_BOOTSTRAP = 500
RANDOM_STATE = 42

FEATURE_DISPLAY = {
    "urban_pct":        "Urban share (%)",
    "forest_pct":       "Forest share (%)",
    "cropland_pct":     "Cropland share (%)",
    "meadow_pct":       "Meadow share (%)",
    "elevation_mean_m": "Mean elevation",
    "elevation_diff_m": "Relief",
    "relief_ratio":     "Relief ratio",
    "precipitation_mm": "Precipitation",
}


def _import_train():
    """Import 02_train_models.py despite the leading digit in the filename."""
    spec = importlib.util.spec_from_file_location(
        "train_models", PROJECT / "scripts" / "02_train_models.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_models"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_data() -> pd.DataFrame:
    """Training set: gauge reference removed, plausibility filter applied
    (identical to 02_train_models.py)."""
    train = _import_train()
    df = pd.read_csv(DATA / "discharge_measurements.csv")
    df = df[~df["is_gauge_reference"]].reset_index(drop=True)
    plausible = df[TARGET].between(
        train.PLAUSIBILITY_MIN_L_S_KM2, train.PLAUSIBILITY_MAX_L_S_KM2
    )
    return df[plausible].reset_index(drop=True)


def get_best_spec(train):
    """Return the random-forest spec — the final model of the manuscript."""
    for spec in train.make_specs():
        if spec.name == "RandomForest":
            return spec
    raise RuntimeError("RandomForest spec not found")


def fit_with_inner_cv(spec, X, y):
    """Refit a spec with inner 5-fold grid search (same protocol as 02)."""
    inner = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(
        deepcopy(spec.estimator),
        spec.param_grid,
        scoring="neg_mean_squared_error",
        cv=inner, n_jobs=-1,
    )
    gs.fit(X, y)
    return gs.best_estimator_


def shap_means(est, X) -> np.ndarray:
    """Mean |SHAP| per feature on the in-bag matrix."""
    try:
        explainer = shap.Explainer(est, X)
        sv = explainer(X).values
    except Exception:
        explainer = shap.Explainer(est.predict, X)
        sv = explainer(X).values
    if sv.ndim == 3:
        sv = sv[..., 0]
    return np.abs(sv).mean(axis=0)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    train = _import_train()
    df = load_data()
    feats = train.FEATURE_SETS["pct"]
    X = df[feats]
    y = df[TARGET]
    spec = get_best_spec(train)
    n = len(df)
    rng = np.random.default_rng(RANDOM_STATE)

    long_rows = []
    print(f"Running {N_BOOTSTRAP} bootstrap resamples (n = {n}); "
          f"may take a couple of minutes.")
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)  # sampling with replacement
        Xb, yb = X.iloc[idx], y.iloc[idx]
        est = fit_with_inner_cv(spec, Xb, yb)
        means = shap_means(est, Xb)
        ranks = (-means).argsort().argsort() + 1  # 1 == most important
        for f, m, r in zip(feats, means, ranks):
            long_rows.append({"run": b, "feature": f,
                              "mean_abs_shap": float(m),
                              "rank": int(r)})
        if (b + 1) % 50 == 0:
            print(f"  {b + 1}/{N_BOOTSTRAP} done")

    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(RESULTS / "bootstrap_shap.csv", index=False)

    # Per-feature summary: empirical P(rank == 1), P(rank <= 2), mean rank, etc.
    summary_rows = []
    for f in feats:
        sub = long_df[long_df["feature"] == f]
        summary_rows.append({
            "feature": f,
            "mean_abs_shap_mean": sub["mean_abs_shap"].mean(),
            "mean_abs_shap_p05":  sub["mean_abs_shap"].quantile(0.05),
            "mean_abs_shap_p95":  sub["mean_abs_shap"].quantile(0.95),
            "rank_mean":          sub["rank"].mean(),
            "rank_median":        sub["rank"].median(),
            "p_rank_eq_1":        float((sub["rank"] == 1).mean()),
            "p_rank_le_2":        float((sub["rank"] <= 2).mean()),
            "p_rank_le_3":        float((sub["rank"] <= 3).mean()),
        })
    summary = pd.DataFrame(summary_rows).sort_values("rank_mean")
    summary.to_csv(RESULTS / "bootstrap_shap_summary.csv", index=False)
    print("\nBootstrap summary:")
    print(summary.to_string(index=False))

    # ------- Rank-distribution heatmap (manuscript Fig. 10) ----------------
    feats_sorted = summary["feature"].tolist()
    n_feats = len(feats_sorted)
    counts = np.zeros((n_feats, n_feats), dtype=int)
    for i, f in enumerate(feats_sorted):
        sub = long_df[long_df["feature"] == f]
        for r in range(1, n_feats + 1):
            counts[i, r - 1] = int((sub["rank"] == r).sum())
    probs = counts / counts.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(probs, cmap="Blues", aspect="auto", vmin=0, vmax=1.0)
    ax.set_xticks(range(n_feats))
    ax.set_xticklabels([str(r + 1) for r in range(n_feats)])
    ax.set_yticks(range(n_feats))
    ax.set_yticklabels([FEATURE_DISPLAY.get(f, f) for f in feats_sorted])
    ax.set_xlabel("SHAP importance rank (1 = most important)")
    ax.set_title(f"Bootstrap rank distribution of mean |SHAP| "
                 f"(B = {N_BOOTSTRAP}, random forest, relative shares)")
    # Annotate probabilities >= 0.05
    for i in range(n_feats):
        for j in range(n_feats):
            if probs[i, j] >= 0.05:
                ax.text(j, i, f"{probs[i, j]:.2f}",
                        ha="center", va="center",
                        color="black" if probs[i, j] < 0.5 else "white",
                        fontsize=9)
    fig.colorbar(im, ax=ax, label="P(rank)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig12_bootstrap_shap.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig12_bootstrap_shap.png",
                bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"\nWrote {FIGURES / 'fig12_bootstrap_shap.png'}")


if __name__ == "__main__":
    main()
