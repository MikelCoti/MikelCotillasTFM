"""Business analyses for the M5 multi-category retail-risk MVP.

Run from the project root, after inventory simulations have exported their
five files per scenario. Reads existing simulation outputs; never trains or
modifies forecasting models. The original test period is exploratory.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from inventory_reporting import (
    compare_scenario_inputs,
    load_scenario,
    validate_scenario,
)

ROOT = Path(__file__).resolve().parents[1]
COMPARE_TO = "lightgbm"
KEY = ["node_id", "item_id", "store_id", "cat_id"]
TOL = 1e-6


def category_column(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    if "cat_id" not in daily:
        daily["cat_id"] = daily["item_id"].str.split("_").str[0]
    if daily["cat_id"].isna().any():
        raise ValueError("Missing product category")
    return daily


def check_identical_demand(frame: pd.DataFrame, keys: list[str]) -> None:
    if frame.duplicated(["model", *keys]).any():
        raise ValueError(f"Duplicate model/demand observations for {keys}")
    spread = frame.groupby(keys, observed=True)["demand_proxy"].agg(["min", "max"])
    if ((spread["max"] - spread["min"]).abs() > TOL).any():
        raise ValueError("Models do not share identical demand")


def category_tables(daily: pd.DataFrame, scenario_name: str, budget) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate numerators and denominators before computing percentage KPIs."""
    frame = category_column(daily)
    group_cols = ["cat_id", "model"]
    aggregations = dict(
        sku_store_nodes=("node_id", "nunique"),
        demand_proxy_units=("demand_proxy", "sum"),
        fulfilled_units=("fulfilled_units", "sum"),
        unmet_units=("unmet_units", "sum"),
        stockout_sku_days=("stockout_day", "sum"),
        replenishment_spend=("order_spend", "sum"),
        replenishment_units=("ordered_units", "sum"),
        mean_ending_inventory_units_per_sku=("ending_inventory", "mean"),
        gross_margin_proxy=("gross_margin_proxy", "sum"),
        holding_cost_proxy=("holding_cost_proxy", "sum"),
        ordering_fee_proxy=("ordering_fee_proxy", "sum"),
    )
    category = frame.groupby(group_cols, observed=True, as_index=False).agg(**aggregations)
    category["fill_rate_pct"] = 100 * category.fulfilled_units / category.demand_proxy_units.replace(0, np.nan)
    category["contribution_proxy"] = (category.gross_margin_proxy - category.holding_cost_proxy
                                      - category.ordering_fee_proxy)
    category.insert(0, "scenario", scenario_name)
    category.insert(1, "weekly_budget", budget)
    weekly = frame.groupby(["cat_id", "model", "week"], observed=True, as_index=False).agg(
        demand_proxy_units=("demand_proxy", "sum"),
        fulfilled_units=("fulfilled_units", "sum"),
        unmet_units=("unmet_units", "sum"),
        stockout_sku_days=("stockout_day", "sum"),
        replenishment_spend=("order_spend", "sum"),
    )
    weekly["fill_rate_pct"] = 100 * weekly.fulfilled_units / weekly.demand_proxy_units.replace(0, np.nan)
    weekly.insert(0, "scenario", scenario_name)
    weekly.insert(1, "weekly_budget", budget)
    return category, weekly


def category_comparison(category: pd.DataFrame, baseline: str = COMPARE_TO) -> pd.DataFrame:
    base = category.loc[category.model.eq(baseline)].copy()
    other = category.loc[category.model.ne(baseline)].copy()
    if base.empty or other.empty:
        raise ValueError(f"Category comparison needs {baseline} and at least one other model")
    fields = ["demand_proxy_units", "fill_rate_pct", "unmet_units", "stockout_sku_days",
              "replenishment_spend", "contribution_proxy"]
    paired = other.merge(
        base[["scenario", "cat_id", *fields]], on=["scenario", "cat_id"],
        how="left", suffixes=("", "_baseline"), validate="many_to_one"
    )
    if paired["demand_proxy_units_baseline"].isna().any():
        raise ValueError("Missing baseline category")
    if not np.allclose(paired.demand_proxy_units, paired.demand_proxy_units_baseline, atol=TOL):
        raise ValueError("Inconsistent demand between models by category")
    paired["fill_rate_change_pp"] = paired.fill_rate_pct - paired.fill_rate_pct_baseline
    paired["unmet_reduction_units"] = paired.unmet_units_baseline - paired.unmet_units
    paired["stockout_days_avoided"] = paired.stockout_sku_days_baseline - paired.stockout_sku_days
    paired["spend_difference"] = paired.replenishment_spend - paired.replenishment_spend_baseline
    paired["contribution_difference"] = paired.contribution_proxy - paired.contribution_proxy_baseline
    return paired


def weekly_forecasting_data(daily: pd.DataFrame) -> pd.DataFrame:
    """Use the *actual forecasts used to order* and subsequent seven daily demand observations."""
    frame = category_column(daily)
    required = {"model", "week", *KEY, "d_num", "demand_proxy", "forecast_7d"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"inventory_daily.csv lacks required forecast columns: {sorted(missing)}")
    forecast = frame.loc[frame.forecast_7d.notna(), ["model", "week", *KEY, "forecast_7d"]].copy()
    keys = ["model", "week", "node_id"]
    if forecast.duplicated(keys).any():
        raise ValueError("Multiple forecast decisions for the same model/week/node")
    counts = frame.groupby(keys, observed=True).d_num.nunique().rename("days_recorded").reset_index()
    totals = frame.groupby(keys, observed=True).demand_proxy.sum().rename("actual_7d").reset_index()
    totals = totals.merge(counts, on=keys, validate="one_to_one")
    if (totals.days_recorded != 7).any():
        raise ValueError("Every forecast must have exactly seven subsequent demand days")
    forecasts = forecast.merge(totals, on=keys, how="outer", indicator=True, validate="one_to_one")
    if (forecasts._merge != "both").any():
        raise ValueError("Missing seven-day forecast or subsequent demand for a node/week")
    forecasts = forecasts.drop(columns=["_merge", "days_recorded"])
    if not np.isfinite(forecasts[["forecast_7d", "actual_7d"]].to_numpy(dtype=float)).all():
        raise ValueError("Forecast or actual demand is not finite")
    if (forecasts[["forecast_7d", "actual_7d"]] < 0).any().any():
        raise ValueError("Negative forecast or demand")
    check = forecasts.rename(columns={"actual_7d": "demand_proxy"})
    check_identical_demand(check, ["week", "node_id"])
    forecasts["abs_error"] = (forecasts.forecast_7d - forecasts.actual_7d).abs()
    forecasts["signed_error"] = forecasts.forecast_7d - forecasts.actual_7d
    return forecasts


def forecast_summary(forecasts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    category = forecasts.groupby(["cat_id", "model"], observed=True, as_index=False).agg(
        predictions=("abs_error", "size"),
        absolute_error_units=("abs_error", "sum"),
        signed_error_units=("signed_error", "sum"),
        actual_units=("actual_7d", "sum"),
    )
    category["MAE"] = category.absolute_error_units / category.predictions
    category["WAPE_pct"] = 100 * category.absolute_error_units / category.actual_units.replace(0, np.nan)
    category["bias_pct"] = 100 * category.signed_error_units / category.actual_units.replace(0, np.nan)

    products = forecasts.groupby(["model", *KEY], observed=True, as_index=False).agg(
        predictions=("abs_error", "size"),
        forecast_abs_error_units=("abs_error", "sum"),
        predicted_units=("forecast_7d", "sum"),
        demand_proxy_units=("actual_7d", "sum"),
    )
    products["forecast_mae"] = products.forecast_abs_error_units / products.predictions
    products["forecast_bias_pct"] = (100 * (products.predicted_units - products.demand_proxy_units)
                                     / products.demand_proxy_units.replace(0, np.nan))
    return category, products


def product_comparison(products: pd.DataFrame, baseline: str = COMPARE_TO) -> pd.DataFrame:
    columns = ["scenario", "weekly_budget", "model", *KEY, "demand_proxy_units",
               "forecast_mae", "forecast_bias_pct", "unmet_units", "stockout_days",
               "replenishment_spend", "fill_rate_pct"]
    missing = set(columns) - set(products.columns)
    if missing:
        raise ValueError(f"Product metrics missing {sorted(missing)}")
    base = products.loc[products.model.eq(baseline), columns].drop(columns=["model", "weekly_budget"])
    comparison = products.loc[products.model.ne(baseline), columns].copy()
    if base.empty or comparison.empty:
        raise ValueError(f"Product comparison needs {baseline} and another model")
    join = ["scenario", *KEY]
    result = comparison.merge(base, on=join, how="left", suffixes=("", "_baseline"), validate="many_to_one")
    if result["forecast_mae_baseline"].isna().any():
        raise ValueError("Missing baseline product/forecast records")
    if not np.allclose(result.demand_proxy_units, result.demand_proxy_units_baseline, atol=TOL):
        raise ValueError("Models face different product-level demand")
    result["forecast_mae_improvement"] = result.forecast_mae_baseline - result.forecast_mae
    result["unmet_reduction_units"] = result.unmet_units_baseline - result.unmet_units
    result["stockout_days_avoided"] = result.stockout_days_baseline - result.stockout_days
    result["spend_difference"] = result.replenishment_spend - result.replenishment_spend_baseline
    result["forecast_result"] = np.select(
        [result.forecast_mae_improvement > TOL, result.forecast_mae_improvement < -TOL],
        ["Lower MAE", "Higher MAE"], default="Unchanged MAE"
    )
    result["inventory_result"] = np.select(
        [result.unmet_reduction_units > TOL, result.unmet_reduction_units < -TOL],
        ["Fewer unmet units", "More unmet units"], default="Same unmet units"
    )
    return result


def analyze(scenarios: list[dict], *, baseline: str = COMPARE_TO) -> dict[str, pd.DataFrame]:
    if not scenarios:
        raise ValueError("No scenario data")
    names = [s["folder"].name for s in scenarios]
    if len(names) != len(set(names)):
        raise ValueError("Scenario folder names must be distinct")
    categories, category_weeks, all_products, budget_rows = [], [], [], []
    forecast_reference = None
    forecast_by_category = None
    product_forecast = None
    for s in scenarios:
        name = s["folder"].name
        daily = category_column(s["daily"])
        c, cw = category_tables(daily, name, s["budget"])
        categories.append(c)
        category_weeks.append(cw)
        forecast = weekly_forecasting_data(daily)
        key_cols = ["model", "week", "node_id"]
        if forecast_reference is None:
            forecast_reference = forecast
            forecast_by_category, product_forecast = forecast_summary(forecast)
        else:
            left = forecast_reference.sort_values(key_cols).reset_index(drop=True)
            right = forecast.sort_values(key_cols).reset_index(drop=True)
            if not left[key_cols].equals(right[key_cols]) or not np.allclose(
                left[["forecast_7d", "actual_7d"]], right[["forecast_7d", "actual_7d"]], atol=1e-5
            ):
                raise ValueError(f"Forecasts or realized demand differ between scenarios: {name}")
        product = s["products"].copy()
        if "cat_id" not in product:
            product["cat_id"] = product.item_id.str.split("_").str[0]
        product = product.merge(
            product_forecast[["model", *KEY, "forecast_mae", "forecast_bias_pct", "predictions"]],
            on=["model", *KEY], validate="one_to_one", how="left"
        )
        if product.forecast_mae.isna().any():
            raise ValueError(f"Unmatched product forecasts in {name}")
        product.insert(0, "scenario", name)
        product.insert(1, "weekly_budget", s["budget"])
        all_products.append(product)
        weekly_spend = s["weekly"][["model", "week", "replenishment_spend"]].copy()
        weekly_spend.insert(0, "scenario", name)
        weekly_spend["weekly_budget"] = s["budget"]
        weekly_spend["remaining_budget"] = (np.nan if s["budget"] is None else
                                             s["budget"] - weekly_spend.replenishment_spend)
        weekly_spend["budget_binding"] = (False if s["budget"] is None else
                                           weekly_spend.remaining_budget < 1.0)
        budget_rows.append(weekly_spend)

    category = pd.concat(categories, ignore_index=True)
    product = pd.concat(all_products, ignore_index=True)
    comparison = product_comparison(product, baseline=baseline)
    patterns = comparison.groupby(
        ["scenario", "cat_id", "model", "forecast_result", "inventory_result"],
        observed=True, as_index=False
    ).agg(
        sku_store_nodes=("node_id", "size"),
        total_unmet_reduction_units=("unmet_reduction_units", "sum"),
    )
    return {
        "category_inventory": category,
        "category_comparison": category_comparison(category, baseline=baseline),
        "category_weekly": pd.concat(category_weeks, ignore_index=True),
        "category_forecasting": forecast_by_category,
        "product_model_metrics": product,
        "product_comparison": comparison,
        "product_patterns": patterns,
        "weekly_budget_audit": pd.concat(budget_rows, ignore_index=True),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--scenarios", nargs="+", type=Path,
                        default=[Path("outputs/inventory_budget_36000"),
                                 Path("outputs/inventory_budget_25000")])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/business_analysis"))
    parser.add_argument("--baseline", type=str, default=COMPARE_TO)
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    folders = [p if p.is_absolute() else root / p for p in args.scenarios]
    scenarios = [load_scenario(p, details=True) for p in folders]
    for scenario in scenarios:
        print(f"Auditing {scenario['folder'].name}...", flush=True)
        validate_scenario(scenario)
    compare_scenario_inputs(scenarios)
    result = analyze(scenarios, baseline=args.baseline)
    target = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    target.mkdir(parents=True, exist_ok=True)
    for name, table in result.items():
        destination = target / f"{name}.csv"
        table.to_csv(destination, index=False)
        print(f"Saved {destination} ({len(table):,} rows)")
    print("Business analysis complete. No forecasting model was retrained.")
    return result


if __name__ == "__main__":
    main()
