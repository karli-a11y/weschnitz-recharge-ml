"""Descriptive and model-performance figures.

Part of the analysis pipeline for:
    Holler JK, Rausch R: Machine-learning-based estimation of groundwater
    recharge from land use in a small crystalline catchment of the Odenwald
    Mountains, Germany.

The study-area map (manuscript Fig. 1) is produced from GIS data outside
this pipeline and is therefore not generated here.

Requires the outputs of scripts 01-03 (input CSVs, metrics.csv,
permutation_importance.csv).

Outputs
-------
figures/fig02_landuse_composition.{pdf,png}          (manuscript Fig. 3)
figures/fig03_momnq_trend.{pdf,png}                  (manuscript Fig. 2)
figures/fig04_discharge_vs_dominant_landuse.{pdf,png} (manuscript Fig. 4)
figures/fig05_model_performance.{pdf,png}            (manuscript Fig. 5)
figures/fig06_predicted_vs_observed.{pdf,png}        (manuscript Fig. 6)
figures/fig07_permutation_importance.{pdf,png}       (manuscript Fig. 7)

Note: figure file names follow the pipeline order in which they were first
created; the manuscript figure numbers follow the citation order in the
text. The mapping is given in the README.

Usage
-----
python scripts/04_descriptive_figures.py        (or simply: python run_all.py)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymannkendall as mk

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
RESULTS = PROJECT / "results"
FIGURES = PROJECT / "figures"

# One fixed colour per land-use class, reused across all figures.
LU_COLORS = {
    "urban": "#7f7f7f",
    "forest": "#2ca02c",
    "cropland": "#bcbd22",
    "meadow": "#17becf",
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


def label_of(name: str) -> str:
    return FEATURE_DISPLAY.get(name, name)


def save(fig, name: str) -> None:
    """Save a figure as publication PDF plus 300-dpi PNG."""
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def fig02_landuse_composition() -> None:
    """Land-use composition of the whole catchment (manuscript Fig. 3)."""
    df = pd.read_csv(DATA / "discharge_measurements.csv")
    ref = df[df["is_gauge_reference"]].iloc[0]
    shares = {
        "urban": ref["urban_km2"], "forest": ref["forest_km2"],
        "cropland": ref["cropland_km2"], "meadow": ref["meadow_km2"],
    }
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(
        shares.values(),
        labels=[k.capitalize() for k in shares],
        autopct="%1.1f%%",
        colors=[LU_COLORS[k] for k in shares],
        wedgeprops={"edgecolor": "white"},
    )
    ax.set_title("Land-use composition of the entire catchment\n"
                 "(43.79 km², gauge Fahrenbach)")
    save(fig, "fig02_landuse_composition")


def fig03_momnq_trend() -> None:
    """Long-term MoMNQ recharge with Mann-Kendall trend test
    (manuscript Fig. 2)."""
    df = pd.read_csv(DATA / "momnq_timeseries.csv")
    mk_res = mk.original_test(df["momnq_mm_per_year"].values)
    # Robustness against serial correlation (value cited in the manuscript):
    mk_hr = mk.hamed_rao_modification_test(df["momnq_mm_per_year"].values)
    print(f"Mann-Kendall (original):  trend = {mk_res.trend}, p = {mk_res.p:.4f}, "
          f"Sen slope = {mk_res.slope:.2f} mm/a/yr")
    print(f"Mann-Kendall (Hamed-Rao): trend = {mk_hr.trend}, p = {mk_hr.p:.4f}")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["hydrological_year"], df["momnq_mm_per_year"], "o-", lw=1, ms=3,
            color="#1f77b4", label="MoMNQ (Wundt 1958)")
    rm = df["momnq_mm_per_year"].rolling(10, center=True).mean()
    ax.plot(df["hydrological_year"], rm, "-", color="#d62728", lw=2,
            label="10-yr rolling mean")
    ax.set_xlabel("Hydrological year")
    ax.set_ylabel("Groundwater recharge (mm a$^{-1}$)")
    ax.set_title("Long-term MoMNQ recharge, gauge Fahrenbach (1960-2023)")
    ax.text(
        0.02, 0.97,
        f"Mann-Kendall: trend = {mk_res.trend}\n"
        f"p = {mk_res.p:.4f}, Sen slope = {mk_res.slope:.2f} mm/a/yr",
        transform=ax.transAxes, va="top", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    save(fig, "fig03_momnq_trend")


def fig04_discharge_vs_dominant_landuse() -> None:
    """Specific discharge vs. catchment area, coloured by dominant land use,
    with the 100-380 mm/a plausibility window marked (manuscript Fig. 4)."""
    df = pd.read_csv(DATA / "discharge_measurements.csv")
    df = df[~df["is_gauge_reference"]].reset_index(drop=True)
    pct = df[["urban_pct", "forest_pct", "cropland_pct", "meadow_pct"]]
    dom = pct.idxmax(axis=1).str.replace("_pct", "")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for lu in LU_COLORS:
        m = dom == lu
        ax.scatter(
            df.loc[m, "area_km2"], df.loc[m, "discharge_per_area_l_s_km2"],
            c=LU_COLORS[lu], label=lu.capitalize(), s=55,
            edgecolor="black", linewidth=0.5,
        )
    # Mark the plausibility envelope (100-380 mm/a, converted to l/(s km^2)).
    ax.axhline(100 / 31.5576, color="grey", linestyle=":", lw=1,
               label="plausibility window\n(100 - 380 mm/a)")
    ax.axhline(380 / 31.5576, color="grey", linestyle=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("Sub-catchment area (km²)")
    ax.set_ylabel("Discharge per area (l s$^{-1}$ km$^{-2}$)")
    ax.set_title("Specific discharge vs. dominant land use")
    ax.legend(title="Dominant land use", frameon=True, loc="upper left",
              fontsize=8)
    ax.grid(alpha=0.3)
    save(fig, "fig04_discharge_vs_dominant_landuse")


def fig05_model_performance() -> None:
    """LOOCV R^2 and RMSE per model and feature set (manuscript Fig. 5)."""
    m = pd.read_csv(RESULTS / "metrics.csv")
    # Order models by their best (pct) LOOCV R² descending so the strongest
    # model is on the left; the same order is reused in the manuscript tables.
    models = (
        m[m["feature_set"] == "pct"]
        .sort_values("r2", ascending=False)["model"]
        .tolist()
    )
    # Human-readable model labels (avoid camelCase in published figure).
    pretty = {
        "RandomForest":     "Random forest",
        "GradientBoosting": "Gradient boosting",
        "Ridge":            "Ridge",
        "SVR":              "SVR",
        "XGBoost":          "XGBoost",
    }
    model_labels = [pretty.get(m_, m_) for m_ in models]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    metrics_to_plot = [("r2", "R²"), ("rmse", "RMSE (l s$^{-1}$ km$^{-2}$)")]
    width = 0.4
    fs_display = {"km2": "km²", "pct": "pct"}
    for ax, (col, lab) in zip(axes, metrics_to_plot):
        x = np.arange(len(models))
        for i, fs in enumerate(["km2", "pct"]):
            sub = m[m["feature_set"] == fs].set_index("model").reindex(models)
            ax.bar(x + (i - 0.5) * width, sub[col], width, label=fs_display[fs])
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, rotation=30, ha="right")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3, axis="y")
    axes[0].axhline(0, color="black", lw=0.5)
    axes[0].legend(title="Feature set")
    fig.suptitle("Model performance (Leave-one-out cross-validation)")
    fig.tight_layout()
    save(fig, "fig05_model_performance")


def fig06_predicted_vs_observed() -> None:
    """Predicted vs. observed scatter for the three best models, showing
    training and hold-out points separately (manuscript Fig. 6)."""
    import importlib.util, sys
    from sklearn.model_selection import train_test_split

    # Import 02_train_models.py to reuse its feature sets, model specs and
    # plausibility filter (filename starts with a digit, hence importlib).
    spec_mod = importlib.util.spec_from_file_location(
        "train_models", PROJECT / "scripts" / "02_train_models.py"
    )
    train = importlib.util.module_from_spec(spec_mod)
    sys.modules["train_models"] = train
    spec_mod.loader.exec_module(train)

    df = pd.read_csv(DATA / "discharge_measurements.csv")
    df = df[~df["is_gauge_reference"]].reset_index(drop=True)
    plausible = df[train.TARGET].between(
        train.PLAUSIBILITY_MIN_L_S_KM2, train.PLAUSIBILITY_MAX_L_S_KM2
    )
    df = df[plausible].reset_index(drop=True)

    metrics = pd.read_csv(RESULTS / "metrics.csv").sort_values("r2", ascending=False)
    top = metrics.head(3)[["model", "feature_set"]].to_records(index=False).tolist()

    specs_by_name = {s.name: s for s in train.make_specs()}
    color_train = "#1f77b4"
    color_test = "#d62728"

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), sharex=True, sharey=True)
    for ax, (name, fs) in zip(axes, top):
        feats = train.FEATURE_SETS[fs]
        X = df[feats]
        y = df[train.TARGET]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=train.RANDOM_STATE,
        )
        est = train.fit_one(specs_by_name[name], X_tr, y_tr)
        yp_tr = est.predict(X_tr)
        yp_te = est.predict(X_te)

        n_total = len(X)
        n_tr = len(X_tr)
        n_te = len(X_te)
        pct_tr = 100.0 * n_tr / n_total
        pct_te = 100.0 * n_te / n_total
        ax.scatter(y_tr, yp_tr, s=55, edgecolor="black", linewidth=0.5,
                   color=color_train, alpha=0.85,
                   label=f"Training (n={n_tr}, {pct_tr:.1f} %)")
        ax.scatter(y_te, yp_te, s=90, edgecolor="black", linewidth=0.6,
                   color=color_test, alpha=0.95, marker="D",
                   label=f"Test (n={n_te}, {pct_te:.1f} %)")

        all_x = np.concatenate([y_tr.values, y_te.values, yp_tr, yp_te])
        lo, hi = float(all_x.min()) * 0.9, float(all_x.max()) * 1.05
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

        r2 = metrics[(metrics.model == name)
                     & (metrics.feature_set == fs)].r2.iloc[0]
        r2_tr = metrics[(metrics.model == name)
                        & (metrics.feature_set == fs)].r2_train.iloc[0]
        r2_ho = metrics[(metrics.model == name)
                        & (metrics.feature_set == fs)].r2_holdout.iloc[0]
        # Human-readable model name (avoid programming-style names in titles).
        pretty = {
            "RandomForest":     "random forest",
            "GradientBoosting": "gradient boosting",
            "Ridge":            "ridge regression",
            "SVR":              "support-vector regression",
            "XGBoost":          "XGBoost",
        }.get(name, name)
        ax.set_title(
            f"{pretty} ({fs})\n"
            f"R²: train {r2_tr:+.2f} | hold-out {r2_ho:+.2f} | LOOCV {r2:+.2f}",
            fontsize=10,
        )
        ax.set_xlabel("Observed (l s$^{-1}$ km$^{-2}$)")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

    axes[0].set_ylabel("Predicted (l s$^{-1}$ km$^{-2}$)")
    save(fig, "fig06_predicted_vs_observed")


def fig07_permutation_importance() -> None:
    """Permutation importance per model on the pct feature set
    (manuscript Fig. 7)."""
    perm = pd.read_csv(RESULTS / "permutation_importance.csv")
    sub = perm[perm["feature_set"] == "pct"].copy()

    feature_order = [
        "meadow_pct", "forest_pct", "urban_pct", "cropland_pct",
        "elevation_mean_m", "elevation_diff_m", "relief_ratio", "precipitation_mm",
    ]
    palette = {
        "urban_pct":               LU_COLORS["urban"],
        "forest_pct":              LU_COLORS["forest"],
        "cropland_pct":            LU_COLORS["cropland"],
        "meadow_pct":              LU_COLORS["meadow"],
        "elevation_mean_m":        "#8c564b",
        "elevation_diff_m":        "#9467bd",
        "relief_ratio":            "#e377c2",
        "precipitation_mm":        "#1f77b4",
    }

    fig, ax = plt.subplots(figsize=(11, 5))
    pivot = sub.pivot(index="model", columns="feature", values="importance_mean")
    pivot = pivot[feature_order]
    pivot.columns = [label_of(c) for c in pivot.columns]
    pivot.plot(kind="bar", ax=ax, edgecolor="black", linewidth=0.4,
               color=[palette[c] for c in feature_order])
    ax.set_ylabel("Permutation importance (Δ R²)")
    ax.set_title("Permutation feature importance — land use, topography, precipitation")
    ax.set_xlabel("")
    ax.legend(title="Feature", loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
    ax.axhline(0, color="black", lw=0.5)
    ax.grid(alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    save(fig, "fig07_permutation_importance")


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    fig02_landuse_composition()
    fig03_momnq_trend()
    fig04_discharge_vs_dominant_landuse()
    fig05_model_performance()
    fig06_predicted_vs_observed()
    fig07_permutation_importance()
    print("Descriptive and performance figures written.")


if __name__ == "__main__":
    main()
