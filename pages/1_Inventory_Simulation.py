"""Milestone 5: compare inventory scenarios and examine SKU-level outcomes.

From project root: streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from inventory_reporting import (  # noqa: E402
    discover_scenarios, load_scenario, product_differences, scenario_label,
    validate_scenario,
)

st.set_page_config(page_title="Inventory Scenarios", page_icon="📦", layout="wide")
st.title("Inventory Scenarios & Model Comparison")
st.caption("Historical M5 sales replay · seven-day forecasts · hypothetical purchasing budgets")
st.warning(
    "These are simulated outcomes, not actual Walmart stockouts or profit. "
    "M5 sales serve as a proxy for demand; inventory, supplier lead times and costs are assumed. "
    "Forecast inputs are historical and the original test period has already been inspected."
)


def nice(name: str) -> str:
    labels = {"seasonal_naive": "Seasonal naive", "lightgbm": "LightGBM",
              "node_only": "Node-only NN", "cross_store": "Cross-store GNN",
              "product_similarity": "Similarity GNN", "full": "Full-graph GNN"}
    return labels.get(name, name)


def fmt_units(value: float) -> str:
    return f"{value:,.0f}"


found = discover_scenarios(ROOT)
if not found:
    st.info("No simulation results found. Generate scenarios using the commands in README_MILESTONE5.md.")
    st.stop()

# Stable scenario IDs are folder names, not budget numbers; this avoids silently
# combining repeated runs made with different lead times or random seeds.
by_folder = {x["folder"].name: x for x in found}
scenario_names = [x["folder"].name for x in found]
selected_name = st.selectbox(
    "Business scenario", scenario_names,
    format_func=lambda x: scenario_label(by_folder[x]),
)
chosen = by_folder[selected_name]
try:
    scenario = load_scenario(chosen["folder"], details=True)
    has_details = True
except FileNotFoundError:
    scenario = chosen
    has_details = False

summary = scenario["summary"]
selected_models = st.multiselect(
    "Compare forecasting models", summary.model.tolist(), default=summary.model.tolist(),
    format_func=nice,
)
if not selected_models:
    st.stop()
subset = summary.loc[summary.model.isin(selected_models)].copy()
subset["model_label"] = subset.model.map(nice)

# Display all scenarios only if they have the same modeled portfolio and horizon.
portfolio = pd.concat([
    s["summary"].assign(scenario=s["folder"].name,
                        scenario_label=scenario_label(s),
                        budget_value=s["budget"] if s["budget"] is not None else float("inf"))
    for s in found
], ignore_index=True)

st.subheader("Portfolio outcomes")
cols = st.columns(4)
cols[0].metric("SKU-store combinations", f"{int(subset.sku_store_nodes.iloc[0]):,}")
cols[1].metric("Simulated days", f"{int(subset.days.iloc[0])}")
cols[2].metric("Demand proxy", f"{subset.demand_proxy_units.iloc[0]:,.0f} units")
cols[3].metric("Weekly purchase limit", "Unrestricted" if scenario["budget"] is None
               else f"{scenario['budget']:,.0f}")

business_cols = ["model_label", "fill_rate_pct", "unmet_units", "stockout_sku_days",
                 "replenishment_spend", "mean_ending_inventory_units_per_sku", "contribution_proxy"]
st.dataframe(subset[business_cols].rename(columns={"model_label": "Model"}).round(2),
             hide_index=True, use_container_width=True)
left, right = st.columns(2)
with left:
    st.write("**Fill rate (%)**")
    st.bar_chart(subset.set_index("model_label")["fill_rate_pct"])
with right:
    st.write("**Unmet demand (units)**")
    st.bar_chart(subset.set_index("model_label")["unmet_units"])

if "lightgbm" in selected_models and len(selected_models) > 1:
    baseline = subset.set_index("model").loc["lightgbm"]
    comparison = subset.loc[subset.model.ne("lightgbm")].copy()
    comparison["fill_rate_change_pp"] = comparison.fill_rate_pct - baseline.fill_rate_pct
    comparison["unmet_reduction_units"] = baseline.unmet_units - comparison.unmet_units
    comparison["spend_change"] = comparison.replenishment_spend - baseline.replenishment_spend
    comparison["contribution_change"] = comparison.contribution_proxy - baseline.contribution_proxy
    st.write("**Differences relative to LightGBM (same scenario)**")
    st.dataframe(comparison[["model_label", "fill_rate_change_pp", "unmet_reduction_units",
                             "spend_change", "contribution_change"]]
                 .rename(columns={"model_label": "Model"}).round(2), hide_index=True,
                 use_container_width=True)
    st.caption("Positive unmet reduction = fewer unfulfilled units; negative spend change = lower purchasing outlay.")

st.subheader("Budget sensitivity across available scenarios")
model_across = st.multiselect("Models in budget comparison", selected_models,
                              default=selected_models, format_func=nice)
if model_across:
    sensitivity = portfolio.loc[portfolio.model.isin(model_across)].copy()
    # A scenario is not comparable just because its budget matches. Show explicit
    # directory labels so the user can see distinct simulation runs.
    sensitivity["scenario_name"] = sensitivity["scenario_label"]
    chart_data = sensitivity.pivot(index="scenario_name", columns="model", values="fill_rate_pct")
    chart_data = chart_data.rename(columns=nice)
    st.bar_chart(chart_data)
    st.caption("A scenario is a discrete run. These bars do not imply a continuous or causal budget response. "
               "Compare budgets only when all other simulation settings are identical.")
    st.dataframe(sensitivity[["scenario_name", "model", "fill_rate_pct", "unmet_units",
                             "replenishment_spend", "contribution_proxy"]]
                 .rename(columns={"model": "Model"}).round(2), hide_index=True,
                 use_container_width=True)


# Category-level statistics are calculated from SKU-day records so that
# service rates weight actual fulfilled demand rather than averaging SKU rates.
if has_details:
    st.subheader("Performance by product category")
    category_daily = scenario["daily"].copy()
    if "cat_id" not in category_daily.columns:
        # Compatible with older simulator exports, using M5 item_id prefixes.
        category_daily["cat_id"] = category_daily.item_id.str.split("_").str[0]
    category_daily = category_daily.loc[category_daily.model.isin(selected_models)]
    category_table = category_daily.groupby(["cat_id", "model"], as_index=False).agg(
        sku_store_nodes=("node_id", "nunique"),
        demand_proxy_units=("demand_proxy", "sum"),
        fulfilled_units=("fulfilled_units", "sum"),
        unmet_units=("unmet_units", "sum"),
        stockout_sku_days=("stockout_day", "sum"),
        replenishment_spend=("order_spend", "sum"),
    )
    category_table["fill_rate_pct"] = (
        100 * category_table.fulfilled_units
        / category_table.demand_proxy_units.replace(0, float("nan"))
    )
    category_table["model_label"] = category_table.model.map(nice)
    st.dataframe(category_table[["cat_id", "model_label", "sku_store_nodes",
        "demand_proxy_units", "fill_rate_pct", "unmet_units",
        "stockout_sku_days", "replenishment_spend"]].rename(
            columns={"cat_id": "Category", "model_label": "Model"}
        ).round(2), hide_index=True, use_container_width=True)
    st.caption("Category figures use the same inventory simulation and fixed budget "
               "across all categories; category-level spending is not independently budgeted.")

if not has_details:
    st.info(f"Only inventory_summary.csv was found in {chosen['folder'].name}. "
            "To enable weekly auditing and product analysis, run the inventory simulator "
            "with --output-dir pointing to that directory. See README_MILESTONE5.md.")
    st.stop()

st.divider()
st.subheader("Simulation accounting & purchasing-budget audit")
try:
    audit = validate_scenario(scenario)
except ValueError as exc:
    st.error(f"Audit failed: {exc}. Investigate this before interpreting the results.")
    st.stop()
st.success(f"Accounting checks passed: {audit['rows_checked']:,} SKU-day rows across "
           f"{audit['models']} models. Maximum model/week purchasing spend: "
           f"{audit['maximum_weekly_spend']:,.2f}.")
weekly = scenario["weekly"].loc[lambda f: f.model.isin(selected_models)].copy()
if scenario["budget"] is not None:
    weekly["budget_remaining"] = scenario["budget"] - weekly.replenishment_spend
    weekly["budget_used_pct"] = 100 * weekly.replenishment_spend / scenario["budget"] if scenario["budget"] else 0.0
st.write("**Weekly replenishment expenditure**")
st.line_chart(weekly.pivot(index="week", columns="model", values="replenishment_spend").rename(columns=nice))
if scenario["budget"] is not None:
    st.caption(f"Purchases must not exceed {scenario['budget']:,.2f} monetary units in any model/week.")
st.dataframe(weekly.round(2), hide_index=True, use_container_width=True)

st.subheader("SKU-store investigation")
products = scenario["products"]
if "lightgbm" not in products.model.unique():
    st.info("The SKU comparison requires LightGBM in this scenario.")
    st.stop()
comparison_models = [m for m in selected_models if m != "lightgbm"]
if not comparison_models:
    st.info("Select LightGBM and a second model above for a paired product comparison.")
    st.stop()
comparison_model = st.selectbox("Compare with LightGBM", comparison_models, format_func=nice)
product_categories = sorted(products.cat_id.unique()) if "cat_id" in products else sorted(
    products.item_id.str.split("_").str[0].unique()
)
selected_product_categories = st.multiselect(
    "Product categories for SKU investigation", product_categories,
    default=product_categories,
)
if not selected_product_categories:
    st.info("Select at least one category to investigate products.")
    st.stop()
paired = product_differences(products, comparison_model)
paired_category = paired.item_id.str.split("_").str[0]
paired = paired.loc[paired_category.isin(selected_product_categories)].copy()
better = int((paired.unmet_reduction_units > 0).sum())
worse = int((paired.unmet_reduction_units < 0).sum())
equal = int((paired.unmet_reduction_units == 0).sum())
a, b, c = st.columns(3)
a.metric("SKUs with less unmet demand", better)
b.metric("SKUs with more unmet demand", worse)
c.metric("SKUs with unchanged unmet demand", equal)
view = st.radio("Inspect", ["Largest shortage reductions", "Largest shortage increases", "All products"],
                horizontal=True)
if view == "Largest shortage reductions":
    displayed = paired.head(20)
elif view == "Largest shortage increases":
    displayed = paired.tail(20).sort_values("unmet_reduction_units")
else:
    displayed = paired
st.dataframe(displayed[["item_id", "store_id", "unmet_units_baseline", "unmet_units_model",
                            "unmet_reduction_units", "stockout_days_avoided", "spend_difference"]].round(2),
             hide_index=True, use_container_width=True)
st.caption("Positive shortage reduction means the selected graph model fulfilled more demand than LightGBM for that SKU-store pair.")

pairs = paired[["item_id", "store_id"]].drop_duplicates().sort_values(["item_id", "store_id"])
choices = [(str(x.item_id), str(x.store_id)) for x in pairs.itertuples(index=False)]
chosen_pair = st.selectbox("Inspect individual product and store", choices,
                           format_func=lambda pair: f"{pair[0]} | {pair[1]}")
item_id, store_id = chosen_pair
daily = scenario["daily"]
trace = daily.loc[daily.item_id.eq(item_id) & daily.store_id.eq(store_id)
                  & daily.model.isin(["lightgbm", comparison_model])].copy()
if trace.empty:
    st.warning("No daily records for the selected SKU-store pair.")
    st.stop()
left, right = st.columns(2)
with left:
    st.write("**Ending inventory by day**")
    st.line_chart(trace.pivot(index="d_num", columns="model", values="ending_inventory").rename(columns=nice))
with right:
    st.write("**Unmet demand by day**")
    st.line_chart(trace.pivot(index="d_num", columns="model", values="unmet_units").rename(columns=nice))
st.dataframe(trace[["model", "d_num", "forecast_7d", "demand_proxy", "received_units",
                    "ordered_units", "fulfilled_units", "unmet_units", "ending_inventory"]]
             .round(2), hide_index=True, use_container_width=True)
st.caption("Forecast values are shown on replenishment decision days only. Historical sales are "
           "treated as an imperfect demand proxy; simulated inventory is not fed back into forecasting features.")
