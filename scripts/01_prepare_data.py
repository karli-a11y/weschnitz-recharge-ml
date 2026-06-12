"""Build the two input tables of the study as CSV files in data/.

Part of the analysis pipeline for:
    Rausch R, Holler JK: Machine-learning-based estimation of groundwater
    recharge from land use in a small crystalline catchment of the Odenwald
    Mountains, Germany.

The raw values were transcribed from the appendix tables of the underlying
measurement campaign (Larisch 2024, Master's thesis, TU Darmstadt):
  - Annex 2: yearly MoMNQ groundwater-recharge values 1960-2023 for the
    gauge Fahrenbach (whole catchment, 43.79 km^2)
  - Annex 3: 38 dry-weather discharge measurements with land-use breakdown
    and topographic descriptors per sub-catchment

Besides transcribing, this script derives the model features documented in
the manuscript (relative land-use shares, mean elevation, relief ratio,
orographic precipitation estimate, recharge in mm/a, geology one-hots).

The two CSVs are also committed to the repository, so the rest of the
pipeline runs without this script; rerunning it reproduces them bit-for-bit.

Outputs
-------
data/momnq_timeseries.csv
data/discharge_measurements.csv

Usage
-----
python scripts/01_prepare_data.py        (or simply: python run_all.py)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"


def build_momnq() -> pd.DataFrame:
    """Yearly MoMNQ recharge (mm/a) 1960-2023, gauge Fahrenbach (Annex 2).

    MoMNQ = mean of the monthly lowest discharges (Wundt 1958), used as a
    proxy for groundwater recharge under dry-weather conditions.
    """
    rows = [
        (1960, 154), (1961, 383), (1962, 338), (1963, 193), (1964, 192),
        (1965, 191), (1966, 363), (1967, 377), (1968, 351), (1969, 345),
        (1970, 377), (1971, 212), (1972, 159), (1973, 155), (1974, 210),
        (1975, 294), (1976, 124), (1977, 158), (1978, 212), (1979, 214),
        (1980, 254), (1981, 310), (1982, 284), (1983, 314), (1984, 181),
        (1985, 210), (1986, 163), (1987, 304), (1988, 306), (1989, 209),
        (1990, 114), (1991, 123), (1992, 110), (1993, 140), (1994, 201),
        (1995, 276), (1996, 163), (1997, 194), (1998, 167), (1999, 314),
        (2000, 253), (2001, 299), (2002, 238), (2003, 229), (2004, 104),
        (2005, 178), (2006, 118), (2007, 183), (2008, 231), (2009, 138),
        (2010, 200), (2011, 182), (2012, 138), (2013, 200), (2014, 150),
        (2015, 173), (2016, 175), (2017, 138), (2018, 242), (2019, 133),
        (2020, 188), (2021, 132), (2022, 150), (2023, 168),
    ]
    return pd.DataFrame(rows, columns=["hydrological_year", "momnq_mm_per_year"])


def build_discharge_measurements() -> pd.DataFrame:
    """38 dry-weather discharge measurements plus gauge reference (Annex 3).

    Row 39 (last) is the gauge "Pegel Fahrenbach" (whole-catchment
    reference); it is kept in the table but flagged via is_gauge_reference
    so it can be excluded from model training.
    The source groups "agriculture_total" = "cropland + meadow"; all three
    columns are kept so the modelling granularity remains a free choice.
    """
    # Topographic descriptors per measurement point:
    #   (id, elevation at the measurement point, highest point in the
    #    sub-catchment, elevation difference), all in m a.s.l. / m.
    elevation_rows = [
        (1, 179.83, 550, 370.17), (2, 180.04, 527, 346.96),
        (3, 179.68, 550, 370.32), (4, 227.04, 499, 271.96),
        (5, 191.11, 475, 283.89), (6, 191.69, 433, 241.31),
        (7, 191.88, 475, 283.12), (8, 200.78, 472, 271.22),
        (9, 197.82, 472, 274.18), (10, 235.60, 480, 244.40),
        (11, 235.96, 480, 244.04), (12, 235.92, 504, 268.08),
        (13, 324.11, 505, 180.89), (14, 328.39, 470, 141.61),
        (15, 215.81, 480, 264.19), (16, 206.42, 480, 273.58),
        (17, 205.79, 360, 154.21), (18, 222.14, 360, 137.86),
        (19, 220.31, 345, 124.69), (20, 283.30, 456, 172.70),
        (21, 266.45, 533, 266.55), (22, 238.75, 375, 136.25),
        (23, 220.99, 417, 196.01), (24, 221.13, 423, 201.87),
        (25, 222.50, 450, 227.50), (26, 205.23, 550, 344.77),
        (27, 272.74, 525, 252.26), (28, 313.28, 527, 213.72),
        (29, 383.46, 550, 166.54), (30, 277.85, 482, 204.15),
        (31, 297.89, 521, 223.11), (32, 187.28, 550, 362.72),
        (33, 223.20, 480, 256.80), (34, 222.76, 428, 205.24),
        (35, 236.31, 547, 310.69), (36, 232.74, 550, 317.26),
        (37, 276.80, 506, 229.20), (38, 446.45, 475, 28.55),
        (39, 179.84, 550, 370.16),
    ]
    elev_df = pd.DataFrame(
        elevation_rows,
        columns=["measurement_id", "elevation_measurement_m",
                 "elevation_max_m", "elevation_diff_m"],
    )

    # Dominant lithology per sub-catchment, derived from an overlay of the
    # measurement-point map, the geological map and the recharge-distribution
    # map of the source thesis. Three classes:
    #   - "buntsandstein": eastern margin (Triassic sandstone)
    #   - "diorite_schiefer": Schlierbach catchment (north-west)
    #   - "crystalline_granite": default; biotite and hornblende granite of
    #     the Weschnitz pluton with minor alluvium along valley floors.
    # The reference gauge row (id 39) is assigned to the dominant class.
    buntsandstein_ids = {13, 14, 37, 38}
    diorite_schiefer_ids = {26, 32, 35, 36}

    def _geology_class(mid: int) -> str:
        if mid in buntsandstein_ids:
            return "buntsandstein"
        if mid in diorite_schiefer_ids:
            return "diorite_schiefer"
        return "crystalline_granite"

    elev_df["geology"] = elev_df["measurement_id"].map(_geology_class)

    # Discharge and land-use areas per measurement point:
    #   (id, Q in m^3/s, Q in l/s, catchment area km^2, specific discharge
    #    l/(s km^2), urban km^2, forest km^2, agriculture-total km^2,
    #    cropland km^2, meadow km^2)
    rows = [
        (1, 0.33, 329.84, 43.79, 7.53, 5.52, 19.69, 18.58, 10.42, 8.16),
        (2, 0.01, 10.60, 3.31, 3.20, 0.17, 1.65, 1.50, 0.67, 0.83),
        (3, 0.42, 415.66, 40.60, 10.24, 5.34, 18.20, 17.05, 9.73, 7.32),
        (4, 0.04, 36.50, 3.85, 9.49, 0.17, 2.58, 1.10, 0.63, 0.46),
        (5, 0.09, 94.52, 16.16, 5.85, 1.77, 8.06, 6.33, 4.43, 1.90),
        (6, 0.00, 4.40, 1.98, 2.22, 0.53, 0.75, 0.70, 0.61, 0.09),
        (7, 0.08, 81.12, 14.17, 5.72, 1.23, 7.31, 5.63, 3.82, 1.81),
        (8, 0.05, 45.36, 8.10, 5.60, 0.40, 5.05, 2.65, 1.49, 1.16),
        (9, 0.03, 28.39, 5.75, 4.94, 0.59, 2.26, 2.91, 2.27, 0.64),
        (10, 0.01, 9.86, 1.71, 5.75, 0.03, 1.19, 0.50, 0.20, 0.30),
        (11, 0.03, 32.04, 7.02, 4.57, 0.18, 5.03, 1.80, 0.87, 0.93),
        (12, 0.02, 21.66, 5.30, 4.09, 0.16, 3.84, 1.30, 0.67, 0.63),
        (13, 0.02, 15.20, 3.78, 4.03, 0.11, 2.82, 0.85, 0.54, 0.31),
        (14, 0.00, 1.12, 0.77, 1.46, 0.01, 0.67, 0.09, 0.05, 0.04),
        (15, 0.01, 8.66, 2.77, 3.12, 0.06, 1.76, 0.95, 0.69, 0.26),
        (16, 0.01, 12.82, 3.51, 3.66, 0.37, 1.76, 1.38, 1.04, 0.33),
        (17, 0.02, 18.16, 1.85, 9.80, 0.09, 0.50, 1.27, 0.96, 0.31),
        (18, 0.00, 3.66, 0.50, 7.33, 0.04, 0.26, 0.20, 0.15, 0.05),
        (19, 0.01, 6.62, 0.28, 24.01, 0.04, 0.12, 0.12, 0.12, 0.00),
        (20, 0.01, 7.58, 0.25, 30.24, 0.08, 0.14, 0.03, 0.00, 0.03),
        (21, 0.01, 9.46, 1.26, 7.49, 0.36, 0.80, 0.10, 0.02, 0.08),
        (22, 0.00, 3.66, 0.31, 11.68, 0.04, 0.13, 0.15, 0.10, 0.05),
        (23, 0.01, 7.38, 0.91, 8.09, 0.28, 0.22, 0.41, 0.23, 0.19),
        (24, 0.00, 2.76, 0.54, 5.13, 0.00, 0.37, 0.17, 0.14, 0.03),
        (25, 0.00, 4.31, 0.65, 6.59, 0.00, 0.53, 0.13, 0.06, 0.06),
        (26, 0.12, 122.86, 13.87, 8.86, 1.79, 6.18, 5.89, 2.38, 3.51),
        (27, 0.02, 15.54, 1.87, 8.32, 0.13, 0.98, 0.75, 0.41, 0.34),
        (28, 0.04, 41.99, 2.38, 17.66, 0.12, 1.43, 0.83, 0.04, 0.79),
        (29, 0.02, 15.21, 1.35, 11.26, 0.18, 0.41, 0.76, 0.49, 0.27),
        (30, 0.00, 2.38, 1.46, 1.63, 0.00, 1.22, 0.24, 0.11, 0.12),
        (31, 0.00, 2.89, 7.44, 0.39, 0.00, 7.11, 0.33, 0.27, 0.06),
        (32, 0.19, 189.99, 18.14, 10.47, 2.09, 7.33, 8.71, 4.07, 4.64),
        (33, 0.00, 3.01, 1.38, 2.18, 0.06, 1.04, 0.27, 0.16, 0.11),
        (34, 0.00, 3.46, 0.76, 4.57, 0.00, 0.40, 0.36, 0.28, 0.09),
        (35, 0.09, 89.83, 1.77, 50.75, 0.19, 1.48, 0.11, 0.05, 0.05),
        (36, 0.07, 69.22, 9.74, 7.11, 0.82, 4.74, 4.18, 1.45, 2.73),
        (37, 0.03, 34.25, 3.77, 9.08, 0.15, 2.41, 1.21, 0.62, 0.59),
        (38, 0.00, 1.57, 0.17, 9.13, 0.00, 0.02, 0.15, 0.06, 0.09),
        (39, 0.25, 248.00, 43.79, 5.66, 5.52, 19.69, 18.58, 10.42, 8.16),
    ]
    cols = [
        "measurement_id",
        "discharge_m3_s",
        "discharge_l_s",
        "area_km2",
        "discharge_per_area_l_s_km2",
        "urban_km2",
        "forest_km2",
        "agriculture_total_km2",
        "cropland_km2",
        "meadow_km2",
    ]
    df = pd.DataFrame(rows, columns=cols)
    df["is_gauge_reference"] = df["measurement_id"] == 39

    # Merge in the topographic descriptors.
    df = df.merge(elev_df, on="measurement_id", how="left")

    # Mean elevation of the sub-catchment, estimated as the midpoint between
    # the measurement point (catchment outlet) and the highest point in the
    # catchment. This is an approximation; without a true hypsometric curve
    # the midpoint is the most defensible single estimate.
    df["elevation_mean_m"] = (
        df["elevation_measurement_m"] + df["elevation_max_m"]
    ) / 2.0

    # Relief ratio: elevation difference per characteristic length sqrt(area).
    # Larger values -> steeper catchment, expected to suppress recharge.
    df["relief_ratio"] = df["elevation_diff_m"] / np.sqrt(df["area_km2"] * 1e6)

    # Annual precipitation estimate from the regional orographic gradient
    # of the Odenwald. Based on DWD HYRAS climatology and the values cited
    # in Larisch (2024) and references therein, mean annual precipitation
    # in the upper Weschnitz catchment scales approximately as:
    #   P(elev) = 600 + 0.9 * elev_mean  [mm a^-1]
    # which gives ~770 mm at 190 m a.s.l. (gauge Fahrenbach) and ~1095 mm
    # at 550 m a.s.l. (highest crests). This is documented as a derived
    # feature in the manuscript.
    df["precipitation_mm"] = 600.0 + 0.9 * df["elevation_mean_m"]

    # Derive relative land-use shares (%). The four primary classes sum to
    # the catchment area (within rounding), so they are normalised by their
    # own sum rather than by area_km2.
    primary_sum = (
        df["urban_km2"] + df["forest_km2"] + df["cropland_km2"] + df["meadow_km2"]
    )
    df["urban_pct"] = df["urban_km2"] / primary_sum * 100
    df["forest_pct"] = df["forest_km2"] / primary_sum * 100
    df["cropland_pct"] = df["cropland_km2"] / primary_sum * 100
    df["meadow_pct"] = df["meadow_km2"] / primary_sum * 100

    # Unit conversion: 1 l/(s km^2) = 31.5576 mm/a.
    df["recharge_mm_per_year"] = df["discharge_per_area_l_s_km2"] * 31.5576

    # One-hot encode geology.
    df["geo_buntsandstein"] = (df["geology"] == "buntsandstein").astype(int)
    df["geo_diorite_schiefer"] = (df["geology"] == "diorite_schiefer").astype(int)
    df["geo_crystalline_granite"] = (df["geology"] == "crystalline_granite").astype(int)
    return df


def main() -> None:
    DATA.mkdir(exist_ok=True)

    momnq = build_momnq()
    out_momnq = DATA / "momnq_timeseries.csv"
    momnq.to_csv(out_momnq, index=False)
    print(f"Wrote {out_momnq} with {len(momnq)} rows")

    discharge = build_discharge_measurements()
    out_discharge = DATA / "discharge_measurements.csv"
    discharge.to_csv(out_discharge, index=False)
    print(f"Wrote {out_discharge} with {len(discharge)} rows")


if __name__ == "__main__":
    main()
