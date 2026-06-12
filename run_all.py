"""One-click reproduction of the full analysis pipeline.

Part of the analysis pipeline for:
    Holler JK, Rausch R: Machine-learning-based estimation of groundwater
    recharge from land use in a small crystalline catchment of the Odenwald
    Mountains, Germany.

Runs all seven analysis scripts in order:
    01  build input CSVs (data/)
    02  train and cross-validate the five models (results/)
    03  permutation importance, SHAP, partial dependence
    04  descriptive and performance figures
    05  100 %-land-use counterfactuals
    06  bootstrap robustness of the SHAP ranking  (~ a few minutes)
    07  plausibility-filter sensitivity           (~ several minutes)
    08  nesting (spatial-independence) robustness check

Usage
-----
python run_all.py

All paths are resolved relative to this file, so the working directory
does not matter. Total runtime is roughly 10-20 minutes on a typical
laptop; steps 06 and 07 dominate.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

# Third-party packages required by the pipeline (see requirements.txt).
REQUIRED_PACKAGES = [
    "numpy", "pandas", "sklearn", "xgboost", "shap",
    "matplotlib", "pymannkendall", "joblib",
]

STEPS = [
    "scripts/01_prepare_data.py",
    "scripts/02_train_models.py",
    "scripts/03_explainability.py",
    "scripts/04_descriptive_figures.py",
    "scripts/05_counterfactual.py",
    "scripts/06_bootstrap_shap.py",
    "scripts/07_filter_sensitivity.py",
    "scripts/08_nesting_check.py",
]


def check_dependencies() -> None:
    """Fail early with a clear message if a required package is missing."""
    missing = [p for p in REQUIRED_PACKAGES
               if importlib.util.find_spec(p) is None]
    if missing:
        sys.exit(
            "Missing required packages: " + ", ".join(missing) + "\n"
            "Install them first with:\n"
            f"    {PY} -m pip install -r requirements.txt"
        )


def main() -> None:
    if sys.version_info < (3, 10):
        sys.exit("Python >= 3.10 is required "
                 f"(found {sys.version.split()[0]}).")
    check_dependencies()
    for step in STEPS:
        print(f"\n=== Running {step} ===")
        rc = subprocess.call([PY, str(ROOT / step)])
        if rc != 0:
            sys.exit(rc)
    print("\nAll done. Outputs are in results/ and figures/.")


if __name__ == "__main__":
    main()
