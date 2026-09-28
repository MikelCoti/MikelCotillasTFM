"""Estimate negative-binomial dispersion for the Executive Summary.

Run from project root:
    python src/calibrate_demand_distribution.py

Assumption:
    Var(Y) = mu + alpha * mu^2

alpha is estimated by maximum likelihood from LightGBM validation-period
seven-day forecast errors. M5 contains observed sales, not verified
unconstrained demand or stockout labels.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import nbinom

from config import FEATURES, MODELS, OUTPUTS
from train_model import TARGET, load_panel, split

MODEL_FILE = MODELS / "lightgbm_7d.joblib"
CALIBRATION_FILE = OUTPUTS / "demand_dispersion_calibration.csv"
COVERAGE_FILE = OUTPUTS / "demand_dispersion_coverage.csv"

ALPHA_MIN = 1e-5
ALPHA_MAX = 10.0
MU_EPS = 1e-8
COVERAGE_LEVELS = (0.50, 0.70, 0.80, 0.90, 0.95)


def validate_inputs(actual, forecast):
    y = np.asarray(actual, dtype=float)
    mu = np.asarray(forecast, dtype=float)
    if y.ndim != 1 or mu.ndim != 1 or len(y) != len(mu) or len(y) == 0:
        raise ValueError("Actual and forecast arrays must be non-empty and aligned.")
    if np.any(~np.isfinite(y)) or np.any(~np.isfinite(mu)):
        raise ValueError("Calibration data contain non-finite values.")
    if np.any(y < 0) or np.any(mu < 0):
        raise ValueError("Demand and forecasts must be non-negative.")
    if not np.allclose(y, np.rint(y), atol=1e-8):
        raise ValueError("Seven-day demand targets must be integer counts.")
    return np.rint(y).astype(np.int64), np.maximum(mu, MU_EPS)


def negative_log_likelihood(log_alpha, actual, forecast):
    alpha = float(np.exp(log_alpha))
    r = 1.0 / alpha
    p = r / (r + forecast)
    logp = nbinom.logpmf(actual, r, p)
    if np.any(~np.isfinite(logp)):
        return np.finfo(float).max / 100.0
    return float(-logp.sum())


def fit_alpha(actual, forecast):
    y, mu = validate_inputs(actual, forecast)
    result = minimize_scalar(
        negative_log_likelihood,
        args=(y, mu),
        method="bounded",
        bounds=(np.log(ALPHA_MIN), np.log(ALPHA_MAX)),
        options={"xatol": 1e-8, "maxiter": 500},
    )
    if not result.success:
        raise RuntimeError(f"Dispersion optimization failed: {result.message}")
    return float(np.exp(result.x)), float(result.fun)


def coverage_rows(actual, forecast, alpha, level, category):
    y, mu = validate_inputs(actual, forecast)
    r = 1.0 / alpha
    p = r / (r + mu)
    rows = []
    for nominal in COVERAGE_LEVELS:
        q = nbinom.ppf(nominal, r, p)
        empirical = float(np.mean(y <= q))
        rows.append({
            "level": level,
            "category": category,
            "alpha": alpha,
            "nominal_coverage": nominal,
            "empirical_coverage": empirical,
            "coverage_error_pp": 100.0 * (empirical - nominal),
            "n_observations": len(y),
        })
    return rows


def summarize(frame, level, category):
    actual = frame[TARGET].to_numpy(float)
    forecast = frame["forecast_7d"].to_numpy(float)
    alpha, nll = fit_alpha(actual, forecast)
    total_actual = actual.sum()
    bias = ((forecast - actual).sum() / total_actual) if total_actual > 0 else np.nan
    summary = {
        "forecast_model": "LightGBM",
        "level": level,
        "category": category,
        "alpha": alpha,
        "n_observations": len(frame),
        "mean_actual_7d": float(actual.mean()),
        "mean_forecast_7d": float(forecast.mean()),
        "aggregate_bias_pct": float(100 * bias) if np.isfinite(bias) else np.nan,
        "negative_log_likelihood": nll,
    }
    return summary, coverage_rows(actual, forecast, alpha, level, category)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-file", type=Path, default=MODEL_FILE)
    parser.add_argument("--calibration-file", type=Path, default=CALIBRATION_FILE)
    parser.add_argument("--coverage-file", type=Path, default=COVERAGE_FILE)
    args = parser.parse_args()

    if not args.model_file.exists():
        raise FileNotFoundError(
            f"Missing model: {args.model_file}. Run `python src/train_model.py` first."
        )

    panel = load_panel()
    _, validation, _ = split(panel)
    model = joblib.load(args.model_file)

    cal = validation[["item_id", "store_id", "cat_id", "d_num", TARGET]].copy()
    cal["forecast_7d"] = np.maximum(model.predict(validation[FEATURES]), 0.0)

    summaries, coverages = [], []

    s, c = summarize(cal, "global", "ALL")
    summaries.append(s)
    coverages.extend(c)

    for category, group in cal.groupby("cat_id", sort=True):
        s, c = summarize(group, "category", str(category))
        summaries.append(s)
        coverages.extend(c)

    summary_df = pd.DataFrame(summaries)
    coverage_df = pd.DataFrame(coverages)

    args.calibration_file.parent.mkdir(parents=True, exist_ok=True)
    args.coverage_file.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.calibration_file, index=False)
    coverage_df.to_csv(args.coverage_file, index=False)

    print("\nCALIBRATED DISPERSION")
    print(summary_df[
        ["level", "category", "alpha", "n_observations",
         "mean_actual_7d", "mean_forecast_7d", "aggregate_bias_pct"]
    ].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    printable = coverage_df.copy()
    printable["nominal_coverage"] *= 100
    printable["empirical_coverage"] *= 100
    print("\nEMPIRICAL COVERAGE ON THE CALIBRATION PERIOD")
    print(printable[
        ["level", "category", "nominal_coverage",
         "empirical_coverage", "coverage_error_pp"]
    ].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print(f"\nSaved: {args.calibration_file}")
    print(f"Saved: {args.coverage_file}")
    print(
        "\nCaveat: LightGBM already used this validation period for early stopping. "
        "These are calibration diagnostics, not an independent final validation."
    )


if __name__ == "__main__":
    main()
