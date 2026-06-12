"""Explainability layer: permutation importance, SHAP, partial dependence.

Part of the analysis pipeline for:
    Holler JK, Rausch R: Machine-learning-based estimation of groundwater
    recharge from land use in a small crystalline catchment of the Odenwald
    Mountains, Germany.

Requires the outputs of scripts 01 and 02 (input CSVs, best_model.pkl).

Outputs
-------
results/permutation_importance.csv
results/shap_values.npy
figures/fig08_shap_summary.{pdf,png}        (manuscript Fig. 8)
figures/fig09_partial_dependence.{pdf,png}  (manuscript Fig. 11)
figures/fig10_shap_dependence.{pdf,png}     (manuscript Fig. 9)

Note: figure file names follow the pipeline order in which they were first
created; the manuscript figure numbers follow the citation order in the
text. The mapping is given in the README.

Usage
-----
python scripts/03_explainability.py        (or simply: python run_all.py)
"""
from __future__ import annotations

import importlib.util
import sys
import warnings
from copy import deepcopy
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, KFold

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
RESULTS = PROJECT / "results"
FIGURES = PROJECT / "figures"
TARGET = "discharge_per_area_l_s_km2"
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

# Human-readable feature names for figure axes and legends.
FEATURE_DISPLAY = {
    "urban_pct":               "Urban share (%)",
    "forest_pct":              "Forest share (%)",
    "cropland_pct":            "Cropland share (%)",
    "meadow_pct":              "Meadow share (%)",
    "urban_km2":               "Urban area (km²)",
    "forest_km2":              "Forest area (km²)",
    "cropland_km2":            "Cropland area (km²)",
    "meadow_km2":              "Meadow area (km²)",
    "area_km2":                "Catchment area (km²)",
    "elevation_mean_m":        "Mean elevation (m a.s.l.)",
    "elevation_diff_m":        "Relief (m)",
    "relief_ratio":            "Relief ratio (–)",
    "precipitation_mm":        "Precipitation (mm a⁻¹)",
    "geo_buntsandstein":       "Geology: Buntsandstein",
    "geo_diorite_schiefer":    "Geology: Diorite / schist",
    "geo_crystalline_granite": "Geology: crystalline granite",
}


def label(name: str) -> str:
    return FEATURE_DISPLAY.get(name, name)


def import_train_module():
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
    df = pd.read_csv(DATA / "discharge_measurements.csv")
    df = df[~df["is_gauge_reference"]].reset_index(drop=True)
    train = import_train_module()
    plausible = df[TARGET].between(
        train.PLAUSIBILITY_MIN_L_S_KM2, train.PLAUSIBILITY_MAX_L_S_KM2
    )
    return df[plausible].reset_index(drop=True)


def refit_specs(df: pd.DataFrame):
    """Refit every model x feature-set combination on the full training set
    (with inner-CV hyperparameter tuning) for the permutation analysis."""
    train = import_train_module()
    specs = train.make_specs()
    fitted: dict[tuple[str, str], object] = {}
    inner = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for fs_name, feats in FEATURE_SETS.items():
        X, y = df[feats], df[TARGET]
        for spec in specs:
            est = deepcopy(spec.estimator)
            if spec.param_grid is None:
                est = est.fit(X, y)
            else:
                est = (
                    GridSearchCV(est, spec.param_grid,
                                 scoring="neg_mean_squared_error",
                                 cv=inner, n_jobs=-1)
                    .fit(X, y)
                    .best_estimator_
                )
            fitted[(spec.name, fs_name)] = est
    return fitted


def permutation_table(fitted, df) -> pd.DataFrame:
    """Permutation importance (50 repeats) for every fitted model."""
    rows = []
    for (name, fs_name), est in fitted.items():
        feats = FEATURE_SETS[fs_name]
        X, y = df[feats], df[TARGET]
        r = permutation_importance(
            est, X, y, n_repeats=50, random_state=RANDOM_STATE, n_jobs=-1
        )
        for f, m, s in zip(feats, r.importances_mean, r.importances_std):
            rows.append({
                "model": name, "feature_set": fs_name, "feature": f,
                "importance_mean": float(m), "importance_std": float(s),
            })
    return pd.DataFrame(rows)


def shap_for_best(df) -> tuple[np.ndarray, list[str], dict]:
    """SHAP values of the best model on the training matrix."""
    best = joblib.load(RESULTS / "best_model.pkl")
    feats = best["feature_names"]
    X = df[feats]
    est = best["estimator"]
    # shap.Explainer picks the appropriate algorithm per estimator type; the
    # fallback to the model-agnostic explainer covers pipeline estimators
    # that the direct path cannot unwrap.
    try:
        explainer = shap.Explainer(est, X)
        sv = explainer(X)
    except Exception:
        explainer = shap.Explainer(est.predict, X)
        sv = explainer(X)
    values = sv.values
    if values.ndim == 3:
        values = values[..., 0]
    np.save(RESULTS / "shap_values.npy", values)
    return values, feats, best


def plot_shap_summary(values, feats, df) -> None:
    """SHAP beeswarm summary of the best model (manuscript Fig. 8)."""
    X = df[feats]
    display_names = [label(f) for f in feats]
    plt.figure(figsize=(8, 5))
    shap.summary_plot(values, X, feature_names=display_names,
                      show=False, plot_size=None)
    plt.tight_layout()
    plt.savefig(FIGURES / "fig08_shap_summary.pdf", bbox_inches="tight")
    plt.savefig(FIGURES / "fig08_shap_summary.png", bbox_inches="tight", dpi=300)
    plt.close()


def plot_partial_dependence(fitted, df) -> None:
    """Partial-dependence figure for the best model (manuscript Fig. 11).

    Layout:
      - Top row: one wide panel combining the four land-use shares
        (urban, forest, cropland, meadow) as colour-coded curves on a
        common 'Share (%)' x-axis, with one sample-distribution histogram
        strip per class below it.
      - Bottom row: four separate panels for the topographic / climatic
        features (mean elevation, relief, precipitation, relief ratio),
        each with its own histogram strip.
    """
    best = joblib.load(RESULTS / "best_model.pkl")
    name, fs = best["name"], best["feature_set"]
    est = fitted[(name, fs)]
    feats_all = FEATURE_SETS[fs]
    X = df[feats_all]

    from sklearn.inspection import partial_dependence
    from matplotlib.ticker import MaxNLocator

    # ---- land-use group (only if pct features are in the active set) ----
    lu_feats = [f for f in ["urban_pct", "forest_pct",
                            "cropland_pct", "meadow_pct"] if f in feats_all]
    lu_colors = {
        "urban_pct":    "#7f7f7f",
        "forest_pct":   "#2ca02c",
        "cropland_pct": "#bcbd22",
        "meadow_pct":   "#17becf",
    }
    lu_labels = {
        "urban_pct":    "Urban",
        "forest_pct":   "Forest",
        "cropland_pct": "Cropland",
        "meadow_pct":   "Meadow",
    }

    # ---- topographic / climatic features (kept as separate panels) ----
    topo_feats = [f for f in ["elevation_mean_m", "elevation_diff_m",
                              "precipitation_mm", "relief_ratio"]
                  if f in feats_all]

    # Layout: top spans full width and is split into (PDP panel) +
    # (one histogram strip per land-use class). Bottom row keeps the
    # topographic / climatic features as individual panels.
    fig = plt.figure(figsize=(17, 12))
    n_topo = len(topo_feats)
    outer = fig.add_gridspec(
        2, n_topo, height_ratios=[1.65, 1.0],
        hspace=0.32, wspace=0.34,
    )
    n_strips = len(lu_feats)
    height_ratios = [7] + [1.7] * n_strips
    top_inner = outer[0, :].subgridspec(1 + n_strips, 1,
                                        height_ratios=height_ratios,
                                        hspace=0.08)
    ax_lu = fig.add_subplot(top_inner[0])
    hist_axes = [fig.add_subplot(top_inner[i + 1], sharex=ax_lu)
                 for i in range(n_strips)]

    for f in lu_feats:
        pd_res = partial_dependence(est, X, [f], kind="average",
                                    grid_resolution=50)
        xs = pd_res["grid_values"][0]
        ys = pd_res["average"][0]
        ax_lu.plot(xs, ys, lw=2.3, color=lu_colors[f], label=lu_labels[f])

    ax_lu.set_ylabel("Predicted specific discharge\n(l s$^{-1}$ km$^{-2}$)",
                     fontsize=13)
    ax_lu.set_title("Partial dependence on land-use shares", fontsize=14, pad=8)
    ax_lu.legend(title="Land-use class", loc="best", fontsize=11,
                 title_fontsize=11, ncol=len(lu_feats), frameon=True)
    ax_lu.tick_params(axis="y", labelsize=11)
    ax_lu.grid(alpha=0.3)
    ax_lu.tick_params(labelbottom=False)

    # One histogram strip per land-use class; all strips share the same
    # y-axis maximum so the bar heights are visually comparable.
    hist_bins = np.linspace(0, max(X[lu_feats].max().max(), 100), 21)
    counts_all = []
    for f in lu_feats:
        c, _ = np.histogram(X[f].values, bins=hist_bins)
        counts_all.append(c)
    y_max = max((c.max() for c in counts_all), default=1)
    for i, (f, c, ax_h) in enumerate(zip(lu_feats, counts_all, hist_axes)):
        ax_h.bar(hist_bins[:-1], c, width=np.diff(hist_bins),
                 align="edge", color=lu_colors[f],
                 edgecolor="white", linewidth=0.3, alpha=0.9)
        ax_h.set_ylim(0, y_max * 1.15)
        ax_h.set_ylabel(lu_labels[f], rotation=0, ha="right", va="center",
                        fontsize=11, labelpad=14,
                        color=lu_colors[f], fontweight="bold")
        ax_h.tick_params(axis="y", labelsize=9)
        ax_h.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=3))
        ax_h.grid(alpha=0.2, axis="y")
        # Hide x-tick labels on all but the last strip
        is_last = (i == n_strips - 1)
        ax_h.tick_params(labelbottom=is_last)
        if is_last:
            ax_h.tick_params(axis="x", labelsize=11)
            ax_h.set_xlabel("Land-use share (%)", fontsize=13, labelpad=6)
        ax_h.set_xlim(ax_lu.get_xlim())

    # ---- Bottom row: one PDP + histogram strip per topo / climatic feature ----
    topo_pdp_axes = []
    for k, f in enumerate(topo_feats):
        inner = outer[1, k].subgridspec(2, 1, height_ratios=[7, 1], hspace=0.05)
        ax = fig.add_subplot(inner[0])
        ax_h = fig.add_subplot(inner[1], sharex=ax)

        pd_res = partial_dependence(est, X, [f], kind="average",
                                    grid_resolution=50)
        xs = pd_res["grid_values"][0]
        ys = pd_res["average"][0]
        ax.plot(xs, ys, lw=2.2, color="#1f77b4")
        ax.grid(alpha=0.3)
        ax.tick_params(labelbottom=False, labelsize=10)
        topo_pdp_axes.append(ax)

        ax_h.hist(X[f].values, bins=15, color="#7a7a7a",
                  edgecolor="white", linewidth=0.3)
        ax_h.set_xlabel(label(f), fontsize=12, labelpad=5)
        ax_h.set_ylabel("n", fontsize=10)
        ax_h.tick_params(axis="x", labelsize=10)
        ax_h.tick_params(axis="y", labelsize=9)
        ax_h.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
        ax_h.grid(alpha=0.2, axis="y")

    if topo_pdp_axes:
        topo_pdp_axes[0].set_ylabel(
            "Predicted specific discharge\n(l s$^{-1}$ km$^{-2}$)",
            fontsize=12,
        )

    pretty = {
        "RandomForest":     "random forest",
        "GradientBoosting": "gradient boosting",
        "Ridge":            "ridge regression",
        "SVR":              "support-vector regression",
        "XGBoost":          "XGBoost",
    }.get(name, name)
    fig.suptitle(
        f"Partial dependence of the predicted specific discharge — best model "
        f"({pretty}, {'relative shares' if fs == 'pct' else 'absolute areas'})",
        fontsize=15, y=0.99,
    )
    fig.savefig(FIGURES / "fig09_partial_dependence.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig09_partial_dependence.png",
                bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_shap_dependence(values, feats, df) -> None:
    """SHAP dependence plots of the two most important features
    (manuscript Fig. 9)."""
    X = df[feats]
    display_names = [label(f) for f in feats]
    order = np.argsort(np.abs(values).mean(axis=0))[::-1][:2]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, idx in zip(axes, order):
        shap.dependence_plot(
            int(idx), values, X, feature_names=display_names,
            ax=ax, show=False,
        )
    fig.tight_layout()
    fig.savefig(FIGURES / "fig10_shap_dependence.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig10_shap_dependence.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    df = load_data()
    fitted = refit_specs(df)
    perm = permutation_table(fitted, df)
    perm.to_csv(RESULTS / "permutation_importance.csv", index=False)
    print(f"Permutation importance written ({len(perm)} rows)")
    values, feats, _ = shap_for_best(df)
    plot_shap_summary(values, feats, df)
    plot_partial_dependence(fitted, df)
    plot_shap_dependence(values, feats, df)
    print("Explainability figures written (SHAP summary, partial dependence, "
          "SHAP dependence).")


if __name__ == "__main__":
    main()
