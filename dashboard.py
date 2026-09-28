"""Executive Summary for the retail inventory decision-support MVP.

Run from project root:
    streamlit run Executive_Summary.py

Before launching, calibrate the negative-binomial dispersion:
    python src/calibrate_demand_distribution.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import nbinom

from src.inventory_simulation import risk_from_forecasts, single_product_risk

ROOT = Path(__file__).resolve().parent
FORECAST_FILE = ROOT / "outputs" / "predictions.csv"
CALIBRATION_FILE = ROOT / "outputs" / "demand_dispersion_calibration.csv"

st.set_page_config(
    page_title="Executive Summary | Retail Inventory Risk",
    page_icon="📦",
    layout="wide",
)
st.title("Executive Summary — Retail Inventory Risk")
st.caption(
    "Seven-day demand forecasts, simulated inventory exposure, and replenishment what-if analysis"
)


@st.cache_data
def load_calibration(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing dispersion calibration file: {path}\n\n"
            "Run `python src/calibrate_demand_distribution.py` from the project root."
        )

    calibration = pd.read_csv(path)
    required = {"forecast_model", "level", "category", "alpha"}
    missing = required - set(calibration.columns)
    if missing:
        raise ValueError(
            f"Dispersion calibration is missing columns: {sorted(missing)}"
        )

    calibration = calibration.loc[
        calibration["forecast_model"].astype(str).eq("LightGBM")
    ].copy()
    if calibration.empty:
        raise ValueError("No LightGBM dispersion parameters were found.")

    global_rows = calibration.loc[calibration["level"].eq("global")]
    if len(global_rows) != 1:
        raise ValueError("Expected exactly one global LightGBM dispersion estimate.")
    global_alpha = float(global_rows["alpha"].iloc[0])

    category_rows = calibration.loc[calibration["level"].eq("category")]
    category_alpha = {
        str(row.category): float(row.alpha)
        for row in category_rows.itertuples(index=False)
    }

    alphas = np.array([global_alpha, *category_alpha.values()], dtype=float)
    if np.any(~np.isfinite(alphas)) or np.any(alphas <= 0):
        raise ValueError("All calibrated dispersion parameters must be finite and > 0.")

    return category_alpha, global_alpha


@st.cache_data
def load_data():
    if FORECAST_FILE.exists():
        df = pd.read_csv(FORECAST_FILE)
        category_alpha, global_alpha = load_calibration(CALIBRATION_FILE)

        # Ignore any old alpha=0.3 column in predictions.csv and replace it.
        df["alpha"] = (
            df["category"].astype(str).map(category_alpha).fillna(global_alpha)
        )
        df["alpha_source"] = np.where(
            df["category"].astype(str).isin(category_alpha),
            "validation-calibrated by category",
            "validation-calibrated global fallback",
        )
        source = "Historical M5 demand forecasts (LightGBM)"
        calibration_source = (
            "Negative-binomial dispersion estimated from validation-period "
            "seven-day forecast errors"
        )
    else:
        # UI-only fallback when the historical forecast export is absent.
        rng = np.random.default_rng(42)
        n = 180
        categories = ["FOODS", "HOUSEHOLD", "HOBBIES"]
        stores = ["CA_1", "CA_2", "TX_1"]
        df = pd.DataFrame({
            "sku": [f"{categories[i % 3]}_{i % 60:03d}" for i in range(n)],
            "store": [stores[i // 60] for i in range(n)],
            "category": [categories[i % 3] for i in range(n)],
            "forecast_7d": rng.uniform(8, 60, n),
            "inventory": rng.integers(3, 75, n),
            "unit_price": rng.uniform(4, 40, n),
            "alpha": 0.3,
            "alpha_source": "synthetic demonstration assumption",
        })
        source = "Synthetic demonstration data"
        calibration_source = (
            "Illustrative alpha=0.3 because historical forecasts are unavailable"
        )

    required = {
        "sku", "store", "category", "forecast_7d",
        "inventory", "unit_price", "alpha",
    }
    if required - set(df.columns):
        raise ValueError(
            f"Missing dashboard columns: {sorted(required - set(df.columns))}"
        )
    if df.duplicated(["sku", "store"]).any():
        raise ValueError("Duplicate SKU-store combinations in dashboard data")

    df = risk_from_forecasts(df)
    return df, source, calibration_source


try:
    df, source, calibration_source = load_data()
except (FileNotFoundError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

st.info(f"Data source: {source}")
st.caption(f"Risk model: {calibration_source}.")
st.caption(
    "M5 contains observed sales rather than verified unconstrained demand. "
    "Inventory is simulated, so the displayed probabilities are calibrated to "
    "historical sales variability around the forecast—not to observed Walmart "
    "stockout events."
)

if "forecast_cutoff" in df.columns:
    st.caption(f"Historical forecast cutoff: {df.forecast_cutoff.iloc[0]}")

st.sidebar.header("Inventory filters")
stores = st.sidebar.multiselect(
    "Store", sorted(df.store.unique()), default=sorted(df.store.unique())
)
categories = st.sidebar.multiselect(
    "Category",
    sorted(df.category.unique()),
    default=sorted(df.category.unique()),
)
threshold = st.sidebar.slider(
    "High-risk threshold (%)",
    min_value=0,
    max_value=100,
    value=70,
    step=1,
    help=(
        "Only SKU-store combinations at or above this estimated stockout-risk "
        "threshold are included in the high-risk KPIs and priority alert table."
    ),
)

filtered = df.loc[
    df.store.isin(stores) & df.category.isin(categories)
].copy()
if filtered.empty:
    st.warning("No products match these filters.")
    st.stop()

st.header("Inventory Risk Overview")

high_risk = filtered.loc[
    filtered.risk >= threshold / 100
].copy()

c1, c2, c3, c4 = st.columns(4)
c1.metric("SKU-store combinations", f"{len(filtered):,}")
c2.metric(f"At or above {threshold}% risk", f"{len(high_risk):,}")
c3.metric(
    "Expected unmet demand — high risk",
    f"{high_risk.expected_shortfall.sum():,.0f} units",
)
c4.metric(
    "Gross sales at risk — high risk",
    f"${high_risk.gross_sales_at_risk.sum():,.0f}",
)

st.caption(
    "The risk threshold filters the high-risk KPIs, category chart, and "
    "priority alerts below. Product Risk Analysis remains available for every "
    "product matching the Store and Category filters."
)

st.subheader("High-risk products by category")
category_counts = (
    high_risk.groupby("category")
    .size()
    .reindex(sorted(filtered.category.unique()), fill_value=0)
)
st.bar_chart(category_counts)

st.subheader("Priority inventory alerts")
cols = [
    "sku", "store", "category", "inventory", "forecast_7d",
    "risk", "expected_shortfall", "gross_sales_at_risk",
]

if high_risk.empty:
    st.info(
        f"No SKU-store combinations have an estimated stockout risk "
        f"of at least {threshold}%."
    )
else:
    alerts = high_risk.sort_values(
        "gross_sales_at_risk",
        ascending=False,
    )[cols].copy()
    alerts["risk"] = (alerts["risk"] * 100).round(1)
    st.dataframe(
        alerts.round(2),
        hide_index=True,
        use_container_width=True,
    )

st.caption(
    "Risk is displayed as a percentage. Gross sales exposure is not avoidable "
    "profit or measured lost sales."
)

st.header("Product Risk Analysis")
filtered["selection"] = (
    filtered.sku.astype(str) + " | " + filtered.store.astype(str)
)
choice = st.selectbox(
    "Select a product and store",
    filtered.selection.tolist(),
)
product = filtered.loc[filtered.selection.eq(choice)].iloc[0]

mean = float(product.forecast_7d)
stock = int(product.inventory)
alpha = float(product.alpha)
r = 1.0 / alpha
p = r / (r + mean)

p1, p2, p3, p4 = st.columns(4)
p1.metric("Current simulated inventory", f"{stock} units")
p2.metric("Seven-day forecast", f"{mean:.1f} units")
p3.metric("Estimated stockout risk", f"{product.risk:.1%}")
p4.metric("Calibrated dispersion (α)", f"{alpha:.3f}")

st.caption(
    f"Dispersion source for this product: {product.alpha_source}. "
    "A larger α represents more uncertainty around the same point forecast."
)

st.subheader("Expected cumulative demand")
days = np.arange(8)
curve = pd.DataFrame({
    "Day": days,
    "Expected cumulative demand": mean * days / 7,
    "Initial available inventory": stock,
}).set_index("Day")
st.line_chart(curve)
st.caption(
    "The demand curve is an illustrative constant-daily-rate projection, "
    "not a daily GNN forecast."
)

st.header("Replenishment Simulator")
st.write(
    "Test a scenario in which additional units are available immediately; "
    "supplier lead times are not modeled on this page."
)

additional = st.number_input(
    "Additional inventory (units)",
    min_value=0,
    max_value=10000,
    value=0,
    step=1,
)
new_risk = single_product_risk(mean, stock + additional, alpha)

a, b, c = st.columns(3)
a.metric("Original stockout risk", f"{product.risk:.1%}")
b.metric("Risk after replenishment", f"{new_risk:.1%}")
c.metric(
    "Risk reduction",
    f"{100 * (product.risk - new_risk):.1f} pp",
)

# P(D > inventory) <= 10% is approximated using the 90th NB demand quantile.
target_stock = int(nbinom.ppf(0.90, r, p))
required_units = max(0, target_stock - stock)
st.success(
    f"Additional units for a 90% modeled demand-coverage target: {required_units}"
)
st.caption(
    "The 90% target is the 90th percentile of the calibrated negative-binomial "
    "distribution around the seven-day LightGBM forecast. It is not a verified "
    "90% real-world Walmart service level because actual inventory and stockout "
    "labels are unavailable."
)
