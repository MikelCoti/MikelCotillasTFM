# Empirical demand-dispersion calibration

This update removes the hard-coded `alpha = 0.3` from the real-data path of the
landing dashboard.

## Files

- `src/calibrate_demand_distribution.py`
- `Executive_Summary.py`

## Install

1. Copy `src/calibrate_demand_distribution.py` into your project's `src/`.
2. Replace the old root `app.py` with `Executive_Summary.py`.
   Keeping the new filename is intentional: with Streamlit's conventional
   multipage layout, it gives the main page the navigation label
   **Executive Summary**.
3. Leave the existing `pages/` files unchanged.

## Run

From `retail-risk-gdl-multicategory`:

```powershell
python src/calibrate_demand_distribution.py
streamlit run Executive_Summary.py
```

The calibration script uses the existing trained
`models/lightgbm_7d.joblib` and `data/processed/m5_panel.parquet`.

It creates:

- `outputs/demand_dispersion_calibration.csv`
- `outputs/demand_dispersion_coverage.csv`

The dashboard then loads category-specific LightGBM dispersion estimates
(FOODS, HOUSEHOLD and HOBBIES), with the calibrated global estimate as a
fallback.

## Method

The predictive distribution is Negative Binomial NB2:

`Var(Y) = mu + alpha * mu^2`

where `mu` is the LightGBM seven-day point forecast. `alpha` is estimated by
maximum likelihood from validation-period seven-day forecast errors.

The script also reports empirical coverage at 50%, 70%, 80%, 90% and 95%.

## Important limitations

This is stronger than an arbitrary dispersion assumption, but it is not a
calibration against real Walmart stockout events. M5 provides observed sales,
not verified inventory, stockout labels or uncensored demand.

LightGBM also used the validation period for early stopping, so the coverage
table is a calibration diagnostic rather than an independent final validation.
