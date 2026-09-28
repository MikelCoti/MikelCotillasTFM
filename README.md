# Retail Inventory Risk Intelligence — consolidated MVP

This project uses M5 historical **observed sales** to train a seven-day LightGBM baseline and construct an early-history product-store graph. Inventory and current stockout probabilities are simulated; neither actual inventory levels nor verified historical stockout labels are in M5.

## Input data

Download the M5 Forecasting - Accuracy evaluation files and place these in `data/raw/`:

- `sales_train_evaluation.csv`
- `calendar.csv`
- `sell_prices.csv`

## Run

Install Python dependencies from `requirements.txt` in your virtual environment. From the project root, execute **in order**:

```bash
python src/prepare_data.py
python src/build_graph.py
python src/train_model.py
python src/train_graphsage.py
python src/evaluate_backtest.py
python src/analyze_forecasts.py
streamlit run app.py
```

`src/config.py` holds the single source of truth for dates and product selection. The prepared panel starts at d_1370 (conventional lag_28 needs history back to d_1342); graph similarity uses d_1370–d_1549; training starts d_1550. The selected 200 FOODS_3 products across CA_1, CA_2 and TX_1 yield 600 product-store nodes. GraphSAGE training is implemented in `src/train_graphsage.py`. It saves a separate historical backtest and an optional app-compatible forecast file without replacing LightGBM dashboard predictions.

**Important:** `outputs/predictions.csv` contains the last historical *test* cutoff for demonstration, not a live forward forecast. The test partition has already been examined in earlier exploratory work; evaluate candidate graph architectures on validation rather than repeatedly using this test period for selection.

## GraphSAGE outputs

- `models/graphsage_7d.pt`: validation-selected model weights and training-fitted preprocessing metadata.
- `outputs/graphsage_training_history.csv`: loss and validation metrics by epoch.
- `outputs/graphsage_backtest_predictions.csv`: historical GNN test forecasts.
- `outputs/backtest_predictions_with_graphsage.csv`: aligned LightGBM, seasonal naive and GraphSAGE forecasts; `evaluate_backtest.py` detects this file automatically.
- `outputs/predictions_graphsage.csv`: same simulated inventory and prices as the LightGBM dashboard, with GNN point forecasts; `app.py` continues to use `outputs/predictions.csv` until a model selector is added.

GraphSAGE uses a point prediction, not a calibrated demand distribution. The provisional negative-binomial risk distribution remains an illustrative assumption.

## Milestone 4: historical inventory simulation

Run **after** `python src/train_model.py` and
`python src/run_graph_ablation.py --phase test --configs node_only cross_store product_similarity full --seeds 42 43 44`.
The simulator loads real test-period **forecast exports**, not model checkpoints;
no training is performed during inventory simulation.

```bash
# First establish what happens without a purchasing budget constraint:
python src/simulate_inventory.py --seeds 42 --output-dir outputs/inventory_unconstrained

# Compare all models under the SAME hypothetical weekly purchasing budget:
python src/simulate_inventory.py --seeds 42 --weekly-budget 20000

# View the original dashboard and the new Inventory Simulation page:
streamlit run app.py
```

You can adjust `--lead-days` (1–7), `--safety-days`, `--initial-cover-weeks`,
`--weekly-budget`, `--cost-ratio`, `--holding-rate` and `--order-fee`.
The default selected models are seasonal naive, LightGBM, node-only,
cross-store GraphSAGE, and full GraphSAGE. Add `product_similarity` with
`--models ...` if you wish to compare that configuration too. The default seed
42 is a single GraphSAGE model per configuration; specifying three seeds
averages their forecasts and **changes the evaluation into an ensemble**.

`outputs/inventory/` contains `inventory_summary.csv`, `inventory_weekly.csv`,
`inventory_products.csv`, `inventory_daily.csv`, and `simulation_config.json`.
The new Streamlit page reads only `outputs/inventory/`.

**Interpretation:** Decisions are made at d_1899, d_1906, d_1913, d_1920,
d_1927 and d_1934; subsequent historical M5 unit sales from d_1900 through
d_1941 serve as an imperfect **demand proxy**. The simulator uses the raw M5
sales file to obtain the final seven days, which are absent from the forecasting
cutoff panel. Sales may be censored by Walmart's actual stock availability.
Simulated inventory is not fed back into historical model features (an open-loop
policy replay). Weekly forecasts are extrapolated at a constant daily demand
rate across the supplier lead time, review period and safety allowance.
Economic outputs use *frozen pre-test prices* and assumed costs; neither actual
inventory, historical stockouts, suppliers, nor real profitability are observed.
Our original M5 test period has been inspected previously, so these are
**exploratory historical results**, not prospective operational validation.

Test the logic with a small synthetic two-SKU M5-shaped dataset:

```bash
python -m unittest discover -s tests -p 'test_inventory_simulation.py' -v
```

## Milestone 5: budget scenarios and business dashboard

See [README_MILESTONE5.md](README_MILESTONE5.md) for generation of the four
scenario folders, inventory accounting checks and the extended Streamlit
page. The original risk dashboard is unchanged.

## Expanded three-category edition

**This copy** is configured for 200 items from each of FOODS, HOUSEHOLD and HOBBIES
in CA_1, CA_2 and TX_1. See [README_MULTICATEGORY.md](README_MULTICATEGORY.md)
for the full sequence and instructions for preserving the earlier FOODS_3-only
experiment. The older single-category figure of 600 nodes in the text above
applies to the archived edition, not this expanded configuration.
