# Milestone 5: validate simulation and explore purchasing scenarios

This update adds `src/inventory_reporting.py`, `src/validate_inventory.py` and
replaces `pages/1_Inventory_Simulation.py`. It **does not replace** `app.py`,
retrain forecasting models or modify your existing scenario output folders.

## 1. Generate or regenerate all scenario exports

Run from the `retail-risk-gdl` project root (and use `--seeds 42`, as in your
previous scenarios). The dashboard needs the **full directory** for each
scenario, not just the downloaded `inventory_summary.csv` file.

```powershell
python src/simulate_inventory.py --seeds 42 --output-dir outputs/inventory_unconstrained
python src/simulate_inventory.py --seeds 42 --weekly-budget 14000 --output-dir outputs/inventory_budget_14000
python src/simulate_inventory.py --seeds 42 --weekly-budget 12000 --output-dir outputs/inventory_budget_12000
python src/simulate_inventory.py --seeds 42 --weekly-budget 10000 --output-dir outputs/inventory_budget_10000
```

If you **already have** these full scenario directories, do not rerun them.
Each should contain the five outputs: `inventory_summary.csv`,
`inventory_weekly.csv`, `inventory_products.csv`, `inventory_daily.csv`, and
`simulation_config.json`.

## 2. Audit the scenarios

```powershell
python src/validate_inventory.py --folders outputs/inventory_unconstrained outputs/inventory_budget_14000 outputs/inventory_budget_12000 outputs/inventory_budget_10000 --report outputs/inventory_audit.json
```

This checks daily demand and stock-flow conservation; per-model, per-week
purchasing limits; weekly, product and summary accounting; common demand and
initial stock across forecasting models and budget scenarios; and compatible
scenario assumptions (when saved configs are available). Any failure raises an
error and needs investigation before drawing conclusions. A six-week total
below the purchasing cap is **not sufficient** to verify weekly compliance.

## 3. Open the dashboard

```powershell
streamlit run app.py
```

Open the **Inventory Scenarios & Model Comparison** page in the sidebar.
Choose a purchasing scenario, compare forecast models and investigate which
SKU-store pairs experience more or fewer unmet units. The dashboard discovers
scenario folders automatically under `outputs/inventory*`. It displays
portfolio summaries from summary-only folders but requires the full five
exports for auditing, weekly analysis and product-level drilldown.

## Limitations

This is an open-loop policy replay using observed M5 sales as a proxy for
uncensored demand. Walmart's actual inventories, stockouts, purchasing costs
and supplier lead times are unavailable. Budget scenarios must differ **only**
in the purchasing cap for a controlled comparison. Results on the previously
inspected historical test period are exploratory; the interface's simulated
financial contribution is not Walmart's realized profit.

## Tests

```powershell
python -m unittest discover -s tests -p "test_inventory_reporting.py" -v
python -m unittest discover -s tests -p "test_inventory_simulation.py" -v
```
