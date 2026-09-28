"""Read precomputed Milestone 5 business analyses. Run: streamlit run app.py"""
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "outputs" / "business_analysis"
st.set_page_config(page_title="Business Analysis", layout="wide")
st.title("Business impact of graph-based forecasting")
st.caption("Exploratory historical replay: assumed inventory and supplier costs; M5 sales proxy demand.")


@st.cache_data
def read_analysis(filename: str) -> pd.DataFrame:
    return pd.read_csv(ANALYSIS / filename)


required = ["category_comparison.csv", "product_comparison.csv", "category_forecasting.csv"]
missing = [name for name in required if not (ANALYSIS / name).exists()]
if missing:
    st.info("First generate business-analysis files with python src/analyze_inventory.py. "
            f"Missing: {', '.join(missing)}")
    st.stop()

cat = read_analysis("category_comparison.csv")
product = read_analysis("product_comparison.csv")
forecast = read_analysis("category_forecasting.csv")
models = sorted(product.model.unique())
scenarios = list(product.scenario.unique())
scenario = st.selectbox("Purchasing scenario", scenarios)
selected_model = st.selectbox("Compare with LightGBM", models)

category_tab, product_tab, policy_tab = st.tabs([
    "Category-level outcomes", "Product-level forecast vs. inventory", "Policy sensitivity"
])
with category_tab:
    st.subheader("Category-level inventory differences relative to LightGBM")
    selected = cat.loc[cat.scenario.eq(scenario) & cat.model.eq(selected_model)].copy()
    st.dataframe(selected[["cat_id", "fill_rate_change_pp", "unmet_reduction_units",
                           "stockout_days_avoided", "spend_difference", "contribution_difference"]]
                 .round(2), hide_index=True, use_container_width=True)
    st.caption("Positive fill-rate change and positive unmet reduction mean fewer shortages "
               "than LightGBM; positive spend difference means higher purchasing cost.")
    st.bar_chart(selected.set_index("cat_id")["unmet_reduction_units"])
    st.subheader("Forecasting by category (common forecasts across budget scenarios)")
    forecast_subset = forecast.loc[forecast.model.isin(["lightgbm", selected_model])]
    st.dataframe(forecast_subset[["cat_id", "model", "MAE", "WAPE_pct", "bias_pct"]]
                 .round(2), hide_index=True, use_container_width=True)

with product_tab:
    st.subheader("Did a more accurate forecast lead to fewer shortages?")
    st.caption("One point per SKU-store combination. Positive x = lower forecast MAE; "
               "positive y = fewer unmet units than LightGBM. The same six weekly forecasts "
               "are used under both models, irrespective of purchasing scenario.")
    subset = product.loc[product.scenario.eq(scenario) & product.model.eq(selected_model)].copy()
    if subset.empty:
        st.info("This model is unavailable in the selected scenario.")
    else:
        cats = sorted(subset.cat_id.unique())
        choice = st.multiselect("Categories", cats, default=cats)
        subset = subset.loc[subset.cat_id.isin(choice)]
        if not subset.empty:
            st.scatter_chart(subset, x="forecast_mae_improvement", y="unmet_reduction_units",
                             color="cat_id", size=None)
            st.subheader("Cases to investigate")
            st.dataframe(subset[["item_id", "store_id", "cat_id", "forecast_mae_improvement",
                                 "unmet_reduction_units", "stockout_days_avoided", "spend_difference",
                                 "forecast_result", "inventory_result"]]
                         .sort_values("unmet_reduction_units", ascending=False).round(2),
                         hide_index=True, use_container_width=True)
            st.caption("These associations do not prove that a smaller forecasting error causes "
                       "a lower shortage: stock carried over from prior weeks also matters.")

with policy_tab:
    p = ANALYSIS / "policy_sensitivity_vs_lightgbm.csv"
    if not p.exists():
        st.info("Run python src/run_policy_sensitivity.py to generate lead-time and safety-stock comparisons.")
    else:
        sensitivity = read_analysis("policy_sensitivity_vs_lightgbm.csv")
        policies = sensitivity.loc[sensitivity.model.eq(selected_model)].copy()
        if policies.empty:
            st.info("No sensitivity results for this model.")
        else:
            budget = st.selectbox("Sensitivity: weekly budget", sorted(policies.weekly_budget.unique()),
                                  format_func=lambda x: f"{x:,.0f}")
            policies = policies.loc[policies.weekly_budget.eq(budget)]
            st.dataframe(policies[["lead_days", "safety_days", "fill_rate_change_pp",
                                   "unmet_reduction_units", "stockout_days_avoided",
                                   "spend_difference", "contribution_difference"]].round(2),
                         hide_index=True, use_container_width=True)
            st.caption("Each row changes policy assumptions while holding the forecasting model, "
                       "initial stock, customer-demand proxy, and weekly budget fixed.")
