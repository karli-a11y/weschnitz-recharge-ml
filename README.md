# Machine-learning-based estimation of groundwater recharge from land use

Data and analysis code for:

> Rausch R, Holler JK: *Machine-learning-based estimation of groundwater
> recharge from land use in a small crystalline catchment of the Odenwald
> Mountains, Germany.*

The pipeline trains five regression models (ridge regression, random forest,
gradient boosting, XGBoost, support-vector regression) on 28 dry-weather
discharge measurements from the upper Weschnitz catchment (43.79 km²,
gauge Fahrenbach) to estimate groundwater recharge from land-use
composition, topography and precipitation, and explains the fitted models
with permutation importance, SHAP values, partial dependence,
100 %-land-use counterfactuals, a bootstrap robustness analysis and
sensitivity checks for the plausibility filter and the spatial nesting
of the sub-catchments.

## Quick start

Requires Python ≥ 3.10.

```bash
git clone https://github.com/karli-a11y/weschnitz-recharge-ml.git
cd weschnitz-recharge-ml
python -m pip install -r requirements.txt
python run_all.py
```

`run_all.py` runs the entire pipeline (data preparation → model training →
explainability → figures → counterfactuals → bootstrap → sensitivity) and
writes all outputs to `results/` and `figures/`. Total runtime is roughly
10–20 minutes on a typical laptop; the bootstrap (step 06) and the
sensitivity analysis (step 07) dominate. All random seeds are fixed, so
repeated runs reproduce identical numbers.

Each script can also be run individually, in numerical order:

```bash
python scripts/01_prepare_data.py
python scripts/02_train_models.py
# ...
```

## Repository layout

```
run_all.py            one-click reproduction of the full pipeline
requirements.txt      pinned package versions
data/                 input tables (also rebuilt by script 01)
  discharge_measurements.csv   38 dry-weather measurements + gauge reference
  momnq_timeseries.csv         yearly MoMNQ recharge 1960-2023
scripts/
  01_prepare_data.py           build the input CSVs and derived features
  02_train_models.py           train + nested-LOOCV the five models
  03_explainability.py         permutation importance, SHAP, partial dependence
  04_descriptive_figures.py    descriptive and performance figures
  05_counterfactual.py         100 %-land-use counterfactual predictions
  06_bootstrap_shap.py         bootstrap robustness of the SHAP ranking
  07_filter_sensitivity.py     plausibility-filter sensitivity analysis
  08_nesting_check.py          nesting (spatial-independence) robustness check
results/              metrics, predictions, model artefacts (generated)
figures/              publication figures, PDF + PNG (generated)
```

## Data

The input tables were transcribed from the appendix of the underlying
measurement campaign (Larisch 2024, Master's thesis, TU Darmstadt,
Institute of Applied Geosciences):

- **`data/discharge_measurements.csv`** — 38 dry-weather discharge
  measurements with land-use breakdown (urban, forest, cropland, meadow),
  topographic descriptors and derived model features per sub-catchment,
  plus the whole-catchment gauge reference (flagged via
  `is_gauge_reference`).
- **`data/momnq_timeseries.csv`** — yearly MoMNQ groundwater-recharge
  values (mm/a) 1960–2023 for the gauge Fahrenbach.

`scripts/01_prepare_data.py` documents the provenance of every column and
rebuilds both CSVs bit-for-bit.

## Figure file names vs. manuscript figure numbers

Figure file names follow the pipeline order in which they were first
created; the manuscript numbers the figures by citation order in the text.

| File in `figures/` | Manuscript |
|---|---|
| `fig02_landuse_composition` | Fig. 3 |
| `fig03_momnq_trend` | Fig. 2 |
| `fig04_discharge_vs_dominant_landuse` | Fig. 4 |
| `fig05_model_performance` | Fig. 5 |
| `fig06_predicted_vs_observed` | Fig. 6 |
| `fig07_permutation_importance` | Fig. 7 |
| `fig08_shap_summary` | Fig. 8 |
| `fig09_partial_dependence` | Fig. 11 |
| `fig10_shap_dependence` | Fig. 9 |
| `fig11_counterfactual_landuse` | Fig. 12 |
| `fig12_bootstrap_shap` | Fig. 10 |

The study-area map (manuscript Fig. 1) is produced from GIS data outside
this pipeline.

## Citation

If you use this code or data, please cite the paper above. A formal
citation with journal, year and DOI will be added here upon publication.

## License

The code is released under the MIT License (see `LICENSE`). The data
tables originate from Larisch (2024); please credit the original
measurement campaign when reusing them.
