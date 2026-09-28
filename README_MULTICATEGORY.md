# Multi-category MVP — FOODS, HOUSEHOLD, HOBBIES

This is a **separate expanded experiment**, not a drop-in replacement for the results
from your earlier 600-node FOODS_3 experiment. The expanded portfolio includes 200
items drawn separately from each of the three M5 `cat_id` categories, sold at
CA_1, CA_2 and TX_1 (1,800 SKU-store nodes). Product selection is deterministic
and limited to items recorded at all three selected stores.

## Start without overwriting the original project

1. Extract this archive into a **new** directory, e.g. `retail-risk-gdl-multicategory`.
2. Copy *only* `sales_train_evaluation.csv`, `calendar.csv`, and `sell_prices.csv`
   from your current project's `data/raw/` to this project's `data/raw/`.
3. Use your existing, working Python virtual environment. If a dependency is missing,
   install it in that environment; `requirements.txt` lists all of them. Check
   PyTorch/PyG compatibility if you install or upgrade either package.
4. Run every command below **from the new project root**. Do not copy your old
   `data/processed`, `models`, or `outputs` directories into this experiment.

## Pipeline

```powershell
python src/prepare_data.py
python src/build_graph.py
python src/train_model.py
python src/train_graphsage.py --batch-size 4
python src/evaluate_backtest.py
```

`build_graph.py` creates cross-store edges between the same product, and up to five
positive-correlation neighbors for each product **within its own store and broad category**.
It does not connect a Foods product to a Hobbies or Household product merely because
short historical sales series happen to correlate. The same forecasting periods
and seven-day targets as the original study remain in force.

The basic dashboard already has a dynamic category selector. Start it with:

```powershell
streamlit run app.py
```

## Ablations and the inventory simulator

The original ablation and inventory pipeline are retained. For the full expanded
experiment, first train and evaluate the frozen graph configurations:

```powershell
python src/run_graph_ablation.py --phase train --configs node_only cross_store product_similarity full --seeds 42 43 44 --epochs 80 --patience 12 --batch-size 4
python src/run_graph_ablation.py --phase test --configs node_only cross_store product_similarity full --seeds 42 43 44
```

Re-establish the **unconstrained** purchasing requirement for this different,
larger product mix before choosing constrained budgets:

```powershell
python src/simulate_inventory.py --seeds 42 --output-dir outputs/inventory_unconstrained
```

Inspect `outputs/inventory_unconstrained/inventory_weekly.csv` for spending
by model and week. Define several fixed *portfolio-wide* budgets from the new
unconstrained spending levels. Do not assume the original 10k/12k/14k weekly
budgets are comparable: both the number of products and category price mix changed.
For instance, substitute a budget value you actually intend to test:

```powershell
python src/simulate_inventory.py --seeds 42 --weekly-budget YOUR_BUDGET --output-dir outputs/inventory_budget_CUSTOM
```

For actual execution, replace `YOUR_BUDGET` with a number and
`inventory_budget_CUSTOM` with an informative folder name. To compare scenarios:

```powershell
python src/validate_inventory.py --folders outputs/inventory_unconstrained outputs/inventory_budget_CUSTOM
streamlit run app.py
```

The Inventory Scenarios page now includes a breakdown of fill rate, unmet units,
stockout SKU-days and spend **by product category**, plus a category filter for
the product-level LightGBM / GNN comparison. The weekly spending limit is shared
across all categories. Category-level spending is *not* independently capped.

## Tests

```powershell
python -m unittest discover -s tests -p 'test_*.py' -v
```

## Interpretation

This experiment is on a new collection of products. Differences in total units,
portfolio WAPE, number of stockouts and simulated financial results relative to
FOODS_3-only output do **not** measure the causal value of adding more categories.
For comparisons across categories, compare metrics within this new experiment using
identical test dates, models and purchasing assumptions. Inventory/lead times/costs
remain simulated; M5 observed sales are not verified unconstrained demand.
