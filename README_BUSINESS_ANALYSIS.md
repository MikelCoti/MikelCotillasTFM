# Business analyses — multi-category retail-risk MVP

Copy `src/analyze_inventory.py`, `src/run_policy_sensitivity.py`, and
`pages/2_Business_Analysis.py` to the **same relative locations** in your existing
`retail-risk-gdl-multicategory` project. Existing model weights and Streamlit
pages are unchanged. Run commands below from the project root.

## Inputs

Complete directories from the existing inventory simulator, containing
`inventory_summary.csv`, `inventory_weekly.csv`, `inventory_products.csv`,
`inventory_daily.csv`, and `simulation_config.json`.

## A. Category and product analysis (no retraining or simulation)

```powershell
python src/analyze_inventory.py --scenarios outputs/inventory_budget_36000 outputs/inventory_budget_25000 --output-dir outputs/business_analysis
```

The script first performs the existing inventory/accounting audit, verifies
that scenarios have the same product-day demand and starting inventory, and
checks that the **forecasts actually used by the simulator** match across
scenarios. It then writes:

- `category_inventory.csv`: category KPIs by model and scenario; percentages
  are calculated from total units, **not** averages of per-SKU percentages.
- `category_comparison.csv`: differences vs. LightGBM for each category.
- `category_weekly.csv`: category KPIs by week and model.
- `category_forecasting.csv`: MAE, WAPE, aggregate bias by category for the
  six seven-day forecasts used in the inventory replay.
- `product_model_metrics.csv`: model forecast error and inventory outcomes for
  each product-store pair.
- `product_comparison.csv`: *paired* SKU-store comparison vs. LightGBM with
  MAE reduction, unmet-unit reduction, stockout-day reduction, and spend delta.
- `product_patterns.csv`: counts by lower/higher MAE and fewer/more shortages.
- `weekly_budget_audit.csv`: weekly spending and budget remaining.

Positive `forecast_mae_improvement` means lower forecasting MAE than LightGBM;
positive `unmet_reduction_units` means fewer unmet units; negative
`spend_difference` means lower purchasing expenditure.

## B. Policy sensitivity (reruns the same existing simulator)

For a small starting grid (8 policy combinations x 4 models):

```powershell
python src/run_policy_sensitivity.py --models lightgbm node_only cross_store full --seeds 42 --budgets 36000 25000 --lead-days 1 3 --safety-days 0 2 --output-dir outputs/business_analysis
```

For the full grid (18 policy combinations x 4 models):

```powershell
python src/run_policy_sensitivity.py --models lightgbm node_only cross_store full --seeds 42 --budgets 36000 25000 --lead-days 1 3 5 --safety-days 0 2 4 --output-dir outputs/business_analysis
```

The runner loads the model forecasts once and applies the same starting stock,
daily demand proxy, cost assumptions, and budget to every model under each
lead-time/safety-stock combination. It writes **summaries only** to avoid
creating dozens of huge daily CSV exports:

- `policy_sensitivity.csv`
- `policy_sensitivity_category.csv`
- `policy_sensitivity_vs_lightgbm.csv`

Run `python -m unittest discover -s tests -p 'test_*.py' -v` after copying the
supplied tests into `tests/` if desired.

## C. Dashboard

```powershell
streamlit run app.py
```

Open the new **Business Analysis** page. Existing dashboard pages stay intact.

## Scientific caveats

- The M5 dataset records **sales**, not verified unconstrained demand or actual
  store inventory. Stockouts and economic outcomes in this project are
  hypothetical policy-replay results.
- Historic forecasts are **not retrained on simulated censored sales**. Each
  sensitivity run therefore compares conditional policy outcomes, not full
  production-system dynamics.
- The original test period has already informed development: treat all these
  experiments as **exploratory** rather than an untouched confirmatory trial.
- An SKU can have lower forecasting MAE but more shortages because of lead
  times, prior stock, order priority, cost, or cumulative policy decisions.
- No category has an independent budget: the purchasing constraint is pooled
  across all categories. Category-level changes therefore reflect
  **cross-category competition** for limited purchasing resources.
- The original Foods-only run selected a *different* product portfolio and
  graph. Do not interpret differences across the two datasets as the effect
  of adding product categories.
