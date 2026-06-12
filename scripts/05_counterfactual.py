"""Counterfactual recharge prediction for hypothetical pure land-use scenarios.

Part of the analysis pipeline for:
    Holler JK, Rausch R: Machine-learning-based estimation of groundwater
    recharge from land use in a small crystalline catchment of the Odenwald
    Mountains, Germany.

For each land-use class (urban, forest, cropland, meadow), the best model
predicts recharge in a synthetic sub-catchment that consists 100 % of that
class. All other features (topography, precipitation, geology) are held at
representative values to make the comparison fair.

To bracket the prediction, three representative settings are used:
  - "valley":  low mean elevation (5th percentile), low relief, low precipitation
  - "typical": median values of all topographic and precipitation features
  - "upland":  high mean elevation (95th percentile), high relief, high precipitation

The geology is set to crystalline_granite for all scenarios because (a) it is
the dominant lithology in the catchment and (b) the geology one-hot features
have very low permutation importance in the models.

Requires the outputs of scripts 01 and 02 (input CSVs, best_model.pkl).

Outputs
-------
results/counterfactual_predictions.csv
figures/fig11_counterfactual_landuse.{pdf,png}   (manuscript Fig. 12)

Usage
-----
python scripts/05_counterfactual.py        (or simply: python run_all.py)
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
RESULTS = PROJECT / "results"
FIGURES = PROJECT / "figures"
TARGET = "discharge_per_area_l_s_km2"

LU_COLORS = {
    "urban": "#7f7f7f",
    "forest": "#2ca02c",
    "cropland": "#bcbd22",
    "meadow": "#17becf",
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


def topo_settings(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Three topographic baselines: valley / typical / upland."""
    qs = df[["elevation_mean_m", "elevation_diff_m",
             "relief_ratio", "precipitation_mm"]]
    return {
        "valley": {
            "elevation_mean_m": float(qs["elevation_mean_m"].quantile(0.05)),
            "elevation_diff_m": float(qs["elevation_diff_m"].quantile(0.05)),
            "relief_ratio":     float(qs["relief_ratio"].quantile(0.05)),
            "precipitation_mm": float(qs["precipitation_mm"].quantile(0.05)),
        },
        "typical": {
            "elevation_mean_m": float(qs["elevation_mean_m"].median()),
            "elevation_diff_m": float(qs["elevation_diff_m"].median()),
            "relief_ratio":     float(qs["relief_ratio"].median()),
            "precipitation_mm": float(qs["precipitation_mm"].median()),
        },
        "upland": {
            "elevation_mean_m": float(qs["elevation_mean_m"].quantile(0.95)),
            "elevation_diff_m": float(qs["elevation_diff_m"].quantile(0.95)),
            "relief_ratio":     float(qs["relief_ratio"].quantile(0.95)),
            "precipitation_mm": float(qs["precipitation_mm"].quantile(0.95)),
        },
    }


def build_row(land_use: str, topo: dict[str, float], feature_names: list[str]) -> pd.DataFrame:
    """Return a single-row DataFrame describing the synthetic catchment."""
    row = {f: 0.0 for f in feature_names}
    if f"{land_use}_pct" in row:
        row[f"{land_use}_pct"] = 100.0
    elif f"{land_use}_km2" in row:
        # For the km2 feature set a catchment area must be fixed as well.
        row[f"{land_use}_km2"] = topo.get("area_km2_assumed", 5.0)
        if "area_km2" in row:
            row["area_km2"] = topo.get("area_km2_assumed", 5.0)
    for k, v in topo.items():
        if k in row:
            row[k] = v
    if "geo_crystalline_granite" in row:
        row["geo_crystalline_granite"] = 1.0
    return pd.DataFrame([row], columns=feature_names)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    df = load_data()
    settings = topo_settings(df)

    best = joblib.load(RESULTS / "best_model.pkl")
    est = best["estimator"]
    feats = best["feature_names"]
    name = best["name"]
    fs = best["feature_set"]
    print(f"Best model: {name} on {fs}")

    land_uses = ["urban", "forest", "cropland", "meadow"]
    rows = []
    for lu in land_uses:
        for setting_name, topo in settings.items():
            X = build_row(lu, topo, feats)
            yhat = float(est.predict(X)[0])
            rows.append({
                "land_use": lu, "setting": setting_name,
                "elevation_mean_m": topo["elevation_mean_m"],
                "precipitation_mm": topo["precipitation_mm"],
                "predicted_q_l_s_km2": yhat,
                "predicted_recharge_mm_yr": yhat * 31.5576,
            })
    cf = pd.DataFrame(rows)
    cf.to_csv(RESULTS / "counterfactual_predictions.csv", index=False)
    print(cf.to_string(index=False))

    # ------------- Counterfactual bar chart (manuscript Fig. 12) -----------
    fig, ax = plt.subplots(figsize=(10, 6))

    pivot = cf.pivot(index="land_use", columns="setting",
                     values="predicted_q_l_s_km2").reindex(land_uses)
    pivot = pivot[["valley", "typical", "upland"]]
    pivot.plot(kind="bar", ax=ax, edgecolor="black", linewidth=0.4,
               color=["#aec7e8", "#1f77b4", "#08306b"], width=0.75)

    # Human-readable model name (avoid programming-style names in the title).
    pretty_name = {
        "RandomForest":     "random forest",
        "GradientBoosting": "gradient boosting",
        "Ridge":            "ridge regression",
        "SVR":              "support-vector regression",
        "XGBoost":          "XGBoost",
    }.get(name, name)

    ax.set_ylabel("Predicted specific discharge (l s$^{-1}$ km$^{-2}$)",
                  fontsize=13)
    ax.set_xlabel("")
    ax.set_title(
        f"100 %-land-use counterfactuals — {pretty_name} "
        f"({'relative shares' if fs == 'pct' else 'absolute areas'})",
        fontsize=14,
    )
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=11)
    ax.legend(title="Topographic setting", loc="lower right", fontsize=11,
              title_fontsize=11)
    ax.grid(alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=0)

    # Secondary y-axis: same quantity in mm a^-1 (1 l/(s km^2) = 31.5576 mm/a).
    ax2 = ax.twinx()
    y0, y1 = ax.get_ylim()
    ax2.set_ylim(y0 * 31.5576, y1 * 31.5576)
    ax2.set_ylabel("Predicted recharge (mm a$^{-1}$)", fontsize=13)
    ax2.tick_params(axis="y", labelsize=11)

    fig.tight_layout()
    fig.savefig(FIGURES / "fig11_counterfactual_landuse.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig11_counterfactual_landuse.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("\nCounterfactual figure written.")


if __name__ == "__main__":
    main()
