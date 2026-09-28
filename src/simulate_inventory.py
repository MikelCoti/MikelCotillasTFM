"""Historical, weekly-review inventory simulation for the M5 retail GDL MVP.

Run from project root after LightGBM and graph ablation test exports:
  python src/simulate_inventory.py --seeds 42 --weekly-budget 20000

This is a counterfactual *simulation*: M5 SALES are treated as a proxy for
customer demand; no real on-hand, supplier, stockout, or cost data are present.
Orders are decided only from forecasts available at each weekly cutoff.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import ROOT, TEST_FIRST_CUTOFF, TEST_LAST_CUTOFF, HORIZON

MODEL_COLS = {"seasonal_naive": "pred_naive", "lightgbm": "pred_lightgbm"}
GRAPH_MODELS = ("node_only", "cross_store", "product_similarity", "full")
ALL_MODELS = (*MODEL_COLS, *GRAPH_MODELS)
KEYS = ["item_id", "store_id"]


def require_columns(df: pd.DataFrame, names: list[str], label: str) -> None:
    absent = set(names) - set(df.columns)
    if absent:
        raise ValueError(f"{label}: missing columns {sorted(absent)}")


def align_frame(df: pd.DataFrame, keys: list[str], expected_index: pd.Index,
                columns: list[str], label: str) -> pd.DataFrame:
    require_columns(df, keys + columns, label)
    if df.duplicated(keys).any():
        raise ValueError(f"{label}: duplicate keys {keys}")
    aligned = df.set_index(keys).reindex(expected_index)
    if aligned[columns].isna().any().any():
        raise ValueError(f"{label}: missing keyed rows or values")
    return aligned


def load_problem(root: Path, models: list[str], seeds: list[int]):
    """Return ordered nodes, observed sales, pre-test prices/stock, and forecasts."""
    if len(set(models)) != len(models) or not models:
        raise ValueError("Provide distinct model names")
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("Provide distinct random seeds")
    if any(name not in ALL_MODELS for name in models):
        raise ValueError(f"Available models: {ALL_MODELS}")

    nodes = pd.read_csv(root / "data/processed/graph/nodes.csv")
    require_columns(nodes, ["node_id", *KEYS], "nodes.csv")
    nodes = nodes.sort_values("node_id").reset_index(drop=True)
    n = len(nodes)
    if not np.array_equal(nodes.node_id.to_numpy(), np.arange(n)):
        raise ValueError("nodes.csv must have contiguous node_id values 0..N-1")
    if nodes.duplicated(KEYS).any():
        raise ValueError("Duplicate SKU-store nodes")
    node_index = pd.MultiIndex.from_frame(nodes[KEYS])
    node_ids = pd.Index(np.arange(n), name="node_id")
    weekly_cutoffs = np.arange(TEST_FIRST_CUTOFF, TEST_LAST_CUTOFF + 1, HORIZON)
    if len(weekly_cutoffs) != 6 or int(weekly_cutoffs[-1]) != TEST_LAST_CUTOFF:
        raise ValueError("Expected exactly six weekly forecast cutoffs")
    first_sales_day, last_sales_day = int(weekly_cutoffs[0] + 1), int(weekly_cutoffs[-1] + HORIZON)

    base_path = root / "outputs/backtest_predictions.csv"
    baseline = pd.read_csv(base_path)
    require_columns(baseline, [*KEYS, "d_num", "target_7d", "sales_sum_7",
                              "sell_price", *MODEL_COLS.values()], str(base_path))
    base = baseline.loc[baseline.d_num.isin(weekly_cutoffs)].copy()
    if len(base) != n * len(weekly_cutoffs):
        raise ValueError("Baseline predictions do not cover all nodes and weekly cutoffs")
    reference = align_frame(base.loc[base.d_num.eq(weekly_cutoffs[0])], KEYS,
                            node_index, ["sales_sum_7"], "initial baseline")
    initial_sales = reference.sales_sum_7.to_numpy(dtype=float)
    if np.any(~np.isfinite(initial_sales)) or np.any(initial_sales < 0):
        raise ValueError("Invalid pre-test sales used to initialize inventory")

    # Prices are frozen using information at the first test cutoff, to prevent
    # future price observations from entering the budget or economic assumptions.
    starting = base.loc[base.d_num.eq(weekly_cutoffs[0])]
    starting_prices = align_frame(starting, KEYS, node_index, ["sales_sum_7"],
                                  "starting inventory")
    price_rows = starting.set_index(KEYS).reindex(node_index)
    # The copy is essential: pandas can expose a read-only NumPy view.
    # We replace missing/invalid prices below, so this must be writable.
    prices = pd.to_numeric(price_rows.sell_price, errors="coerce").to_numpy(
        dtype=float, copy=True
    )
    good = np.isfinite(prices) & (prices > 0)
    if not good.any():
        raise ValueError("No valid prices at the first forecast cutoff")
    price_imputations = int((~good).sum())
    prices[~good] = np.median(prices[good])

    # The processed panel ends at d_1934 because later days have no complete
    # seven-day forecast labels. Retrieve actual d_1900..d_1941 daily sales
    # from the raw file instead of trying to infer them from overlapping targets.
    day_columns = [f"d_{d}" for d in range(first_sales_day, last_sales_day + 1)]
    parts = []
    sales_path = root / "data/raw/sales_train_evaluation.csv"
    for chunk in pd.read_csv(sales_path, usecols=[*KEYS, *day_columns], chunksize=4000):
        chunk_idx = pd.MultiIndex.from_frame(chunk[KEYS])
        part = chunk.loc[chunk_idx.isin(node_index)]
        if not part.empty:
            parts.append(part)
    if not parts:
        raise ValueError("No matching product-store rows in the raw M5 sales file")
    raw_sales = align_frame(pd.concat(parts, ignore_index=True), KEYS,
                            node_index, day_columns, "raw daily sales")
    demand = raw_sales[day_columns].to_numpy(dtype=float)
    if np.any(~np.isfinite(demand)) or np.any(demand < 0):
        raise ValueError("Daily sales must be finite and nonnegative")

    forecasts = {name: [] for name in models}
    for cutoff in weekly_cutoffs:
        source = align_frame(base.loc[base.d_num.eq(cutoff)], KEYS, node_index,
                             ["target_7d", *MODEL_COLS.values()], f"baseline d_{cutoff}")
        # Verify all future target values against the raw sales, including
        # the final seven days beyond m5_panel.parquet's cutoff horizon.
        from_col = int(cutoff + 1 - first_sales_day)
        actual_7d = demand[:, from_col:from_col + HORIZON].sum(axis=1)
        if not np.allclose(source.target_7d.to_numpy(dtype=float), actual_7d,
                           atol=1e-3, rtol=1e-5):
            raise ValueError(f"M5 seven-day target mismatch at d_{cutoff}")
        for name, col in MODEL_COLS.items():
            if name in forecasts:
                forecasts[name].append(source[col].to_numpy(dtype=float))

    for config in GRAPH_MODELS:
        if config not in forecasts:
            continue
        by_seed = []
        for seed in seeds:
            path = root / f"outputs/ablation/test_{config}_seed{seed}.csv"
            frame = pd.read_csv(path)
            require_columns(frame, ["node_id", "d_num", "target_7d", "pred_graphsage"],
                            str(path))
            if len(frame) != n * (TEST_LAST_CUTOFF - TEST_FIRST_CUTOFF + 1):
                raise ValueError(f"Incomplete ablation test predictions: {path}")
            if frame.duplicated(["node_id", "d_num"]).any():
                raise ValueError(f"Duplicate ablation prediction keys: {path}")
            per_week = []
            for cutoff in weekly_cutoffs:
                aligned = align_frame(frame.loc[frame.d_num.eq(cutoff)], ["node_id"],
                                      node_ids, ["target_7d", "pred_graphsage"],
                                      f"{config} seed={seed}, d_{cutoff}")
                start = int(cutoff + 1 - first_sales_day)
                truth = demand[:, start:start + HORIZON].sum(axis=1)
                if not np.allclose(aligned.target_7d.to_numpy(dtype=float), truth,
                                   atol=1e-3, rtol=1e-5):
                    raise ValueError(f"Node/target misalignment: {config} seed={seed}")
                per_week.append(aligned.pred_graphsage.to_numpy(dtype=float))
            by_seed.append(np.stack(per_week))
        forecasts[config] = np.stack(by_seed).mean(axis=0)

    for model, prediction in forecasts.items():
        array = np.asarray(prediction, dtype=float)
        if array.shape != (len(weekly_cutoffs), n) or not np.isfinite(array).all() or (array < 0).any():
            raise ValueError(f"Invalid shape or values for {model}: {array.shape}")
        forecasts[model] = array

    print(f"Input: {n} SKU-store nodes, {len(weekly_cutoffs)} weekly decisions, "
          f"{demand.shape[1]} simulated days; imputed {price_imputations} pre-test prices")
    return nodes, demand, initial_sales, prices, forecasts, weekly_cutoffs, price_imputations


def simulate_model(model: str, nodes: pd.DataFrame, demand: np.ndarray,
                   initial_sales: np.ndarray, prices: np.ndarray,
                   weekly_forecasts: np.ndarray, cutoffs: np.ndarray,
                   *, lead_days: int, safety_days: float, initial_cover_weeks: float,
                   weekly_budget: float | None, cost_ratio: float,
                   holding_rate: float, order_fee: float) -> pd.DataFrame:
    """Weekly order-up-to policy with delayed receipts and lost sales."""
    n, total_days = demand.shape
    if weekly_forecasts.shape != (len(cutoffs), n) or total_days != HORIZON * len(cutoffs):
        raise ValueError("Forecast, node and daily demand dimensions do not match")
    if not (1 <= lead_days <= HORIZON):
        raise ValueError("Lead time must be between 1 and 7 days for this MVP")
    if min(safety_days, initial_cover_weeks, holding_rate, order_fee) < 0:
        raise ValueError("Safety, coverage and cost parameters cannot be negative")
    if not (0 < cost_ratio < 1):
        raise ValueError("cost_ratio must be strictly between zero and one")
    if weekly_budget is not None and weekly_budget < 0:
        raise ValueError("weekly_budget must be nonnegative")

    cost = prices * cost_ratio
    margin = prices - cost
    stock = np.ceil(initial_sales * initial_cover_weeks + 2).astype(np.int64)
    pipeline: dict[int, np.ndarray] = {}
    start_day = int(cutoffs[0] + 1)
    daily_frames = []

    for week, cutoff in enumerate(cutoffs):
        week_forecast = weekly_forecasts[week]
        if not np.isfinite(week_forecast).all() or (week_forecast < 0).any():
            raise ValueError("Forecasts must be finite nonnegative unit quantities")
        # Forecast covers seven days. Scaling to lead + weekly review + safety
        # assumes roughly constant mean DAILY demand beyond the forecast horizon.
        protection_days = HORIZON + lead_days + safety_days
        desired_position = np.ceil(week_forecast * protection_days / HORIZON).astype(np.int64)
        in_pipeline = sum(pipeline.values(), np.zeros(n, dtype=np.int64))
        inventory_position = stock + in_pipeline
        proposed = np.maximum(desired_position - inventory_position, 0)
        ordered = proposed.copy()

        if weekly_budget is not None:
            ordered.fill(0)
            remaining = float(weekly_budget)
            expected_shortage = np.maximum(
                week_forecast * (HORIZON + lead_days) / HORIZON - inventory_position, 0
            )
            # Business priority is *forecast-based*; future observed sales are
            # never used to rank or allocate replenishment.
            priority = expected_shortage * margin
            priority += np.maximum(week_forecast * (lead_days - 1) / HORIZON - stock, 0) * margin
            ranking = np.argsort(-priority, kind="stable")
            for j in ranking:
                if proposed[j] <= 0:
                    continue
                affordable = max(0, int(np.floor((remaining + 1e-8) / cost[j])))
                qty = min(int(proposed[j]), affordable)
                if qty:
                    ordered[j] = qty
                    remaining -= qty * cost[j]
            if float((ordered * cost).sum()) > weekly_budget + 1e-5:
                raise AssertionError("Weekly replenishment spending exceeded the budget")

        arrival_day = int(cutoff + lead_days)
        if arrival_day in pipeline:
            pipeline[arrival_day] += ordered
        else:
            pipeline[arrival_day] = ordered.copy()
        weekly_order_spend = ordered * cost

        for within_week in range(HORIZON):
            day = int(cutoff + 1 + within_week)
            receipts = pipeline.pop(day, np.zeros(n, dtype=np.int64))
            stock += receipts
            beginning = stock.copy()  # after due receipts, before demand
            observed = demand[:, day - start_day]
            filled = np.minimum(stock, observed)
            shortage = observed - filled
            stock = np.maximum(stock - filled, 0).astype(np.int64)
            ordering_today = within_week == 0
            daily_frames.append(pd.DataFrame({
                "model": model, "week": week + 1, "d_num": day,
                "node_id": nodes.node_id.to_numpy(dtype=int),
                "item_id": nodes.item_id.to_numpy(), "store_id": nodes.store_id.to_numpy(),
                "cat_id": (nodes["cat_id"] if "cat_id" in nodes.columns
                           else nodes.item_id.str.split("_").str[0]).to_numpy(),
                "demand_proxy": observed, "fulfilled_units": filled,
                "unmet_units": shortage, "stockout_day": shortage > 0,
                "received_units": receipts, "stock_before_demand": beginning,
                "ending_inventory": stock.copy(),
                "ordered_units": ordered if ordering_today else np.zeros(n, dtype=np.int64),
                "order_spend": weekly_order_spend if ordering_today else np.zeros(n),
                "forecast_7d": week_forecast if ordering_today else np.full(n, np.nan),
                "gross_sales_proxy": filled * prices,
                "gross_margin_proxy": filled * margin,
                "holding_cost_proxy": stock * cost * holding_rate / 365.0,
                "ordering_fee_proxy": (ordered > 0).astype(float) * order_fee
                if ordering_today else np.zeros(n),
                "lost_sales_value_proxy": shortage * prices,
            }))
    daily = pd.concat(daily_frames, ignore_index=True)
    if daily.duplicated(["model", "d_num", "node_id"]).any():
        raise AssertionError("Duplicate model/date/node events")
    if not np.allclose(daily.demand_proxy, daily.fulfilled_units + daily.unmet_units):
        raise AssertionError("Demand was not accounted for")
    return daily


def summarize(daily: pd.DataFrame, *, price_imputations: int,
              weekly_budget: float | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for model, frame in daily.groupby("model", sort=False):
        demand = float(frame.demand_proxy.sum())
        filled = float(frame.fulfilled_units.sum())
        shortage = float(frame.unmet_units.sum())
        holding = float(frame.holding_cost_proxy.sum())
        ordering_fee = float(frame.ordering_fee_proxy.sum())
        margin = float(frame.gross_margin_proxy.sum())
        rows.append({
            "model": model,
            "sku_store_nodes": int(frame.node_id.nunique()),
            "days": int(frame.d_num.nunique()),
            "demand_proxy_units": demand,
            "fulfilled_units": filled,
            "unmet_units": shortage,
            "fill_rate_pct": 100 * filled / demand if demand else np.nan,
            "stockout_sku_days": int(frame.stockout_day.sum()),
            "stockout_pct_of_positive_demand_days": 100 * frame.stockout_day.sum() / (frame.demand_proxy > 0).sum()
                if (frame.demand_proxy > 0).any() else np.nan,
            "mean_ending_inventory_units_per_sku": float(frame.ending_inventory.mean()),
            "ending_inventory_units": float(frame.loc[frame.d_num.eq(frame.d_num.max()), "ending_inventory"].sum()),
            "replenishment_units": float(frame.ordered_units.sum()),
            "replenishment_spend": float(frame.order_spend.sum()),
            "weekly_budget": weekly_budget,
            "purchase_budget_utilization_pct": 100 * float(frame.order_spend.sum()) / (
                weekly_budget * frame.week.nunique()) if weekly_budget and weekly_budget > 0 else np.nan,
            "gross_sales_proxy": float(frame.gross_sales_proxy.sum()),
            "gross_margin_proxy": margin,
            "holding_cost_proxy": holding,
            "ordering_fee_proxy": ordering_fee,
            "contribution_proxy": margin - holding - ordering_fee,
            "missing_starting_prices_imputed": price_imputations,
        })
    overall = pd.DataFrame(rows)
    weekly = daily.groupby(["model", "week"], as_index=False).agg(
        demand_proxy_units=("demand_proxy", "sum"),
        fulfilled_units=("fulfilled_units", "sum"),
        unmet_units=("unmet_units", "sum"),
        replenishment_spend=("order_spend", "sum"),
        replenishment_units=("ordered_units", "sum"),
        stockout_sku_days=("stockout_day", "sum"),
        average_ending_inventory=("ending_inventory", "mean"),
    )
    weekly["fill_rate_pct"] = 100 * weekly.fulfilled_units / weekly.demand_proxy_units.replace(0, np.nan)
    products = daily.groupby(["model", "node_id", "item_id", "store_id", "cat_id"], as_index=False).agg(
        demand_proxy_units=("demand_proxy", "sum"),
        fulfilled_units=("fulfilled_units", "sum"),
        unmet_units=("unmet_units", "sum"),
        stockout_days=("stockout_day", "sum"),
        ending_inventory=("ending_inventory", "last"),
        replenishment_units=("ordered_units", "sum"),
        replenishment_spend=("order_spend", "sum"),
        lost_sales_value_proxy=("lost_sales_value_proxy", "sum"),
    )
    products["fill_rate_pct"] = 100 * products.fulfilled_units / products.demand_proxy_units.replace(0, np.nan)
    return overall, weekly, products


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS,
                        default=["seasonal_naive", "lightgbm", "node_only", "cross_store", "full"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="Averaging several seeds makes an ENSEMBLE, not a comparable single model")
    parser.add_argument("--lead-days", type=int, default=3)
    parser.add_argument("--safety-days", type=float, default=2.0)
    parser.add_argument("--initial-cover-weeks", type=float, default=1.2)
    parser.add_argument("--weekly-budget", type=float, default=None,
                        help="Maximum replenishment spending per week; omit for unconstrained run")
    parser.add_argument("--cost-ratio", type=float, default=0.70,
                        help="Assumed unit purchasing cost as a fraction of selling price")
    parser.add_argument("--holding-rate", type=float, default=0.25,
                        help="Assumed annual holding cost as fraction of purchasing cost")
    parser.add_argument("--order-fee", type=float, default=2.0,
                        help="Assumed fixed ordering cost per SKU per weekly order")
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    out = (args.output_dir or root / "outputs/inventory").resolve()
    out.mkdir(parents=True, exist_ok=True)
    nodes, demand, initial_sales, prices, preds, cutoffs, imputations = load_problem(
        root, args.models, args.seeds
    )
    frames = []
    for model in args.models:
        print(f"Simulating {model}...", flush=True)
        frames.append(simulate_model(
            model, nodes, demand, initial_sales, prices, preds[model], cutoffs,
            lead_days=args.lead_days, safety_days=args.safety_days,
            initial_cover_weeks=args.initial_cover_weeks, weekly_budget=args.weekly_budget,
            cost_ratio=args.cost_ratio, holding_rate=args.holding_rate, order_fee=args.order_fee,
        ))
    daily = pd.concat(frames, ignore_index=True)
    overall, weekly, products = summarize(daily, price_imputations=imputations,
                                           weekly_budget=args.weekly_budget)
    daily.to_csv(out / "inventory_daily.csv", index=False)
    weekly.to_csv(out / "inventory_weekly.csv", index=False)
    products.to_csv(out / "inventory_products.csv", index=False)
    overall.to_csv(out / "inventory_summary.csv", index=False)
    metadata = {
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "test_cutoffs": [int(v) for v in cutoffs],
        "simulated_days": [int(cutoffs[0] + 1), int(cutoffs[-1] + HORIZON)],
        "assumptions": [
            "M5 observed unit sales serve as an imperfect proxy for uncensored customer demand.",
            "Stock, suppliers, costs, budgets and replenishment orders are hypothetical.",
            "Sales prices and unit costs are frozen at the pre-test cutoff; missing prices use its median.",
            "Weekly seven-day point forecasts are extrapolated at a constant rate for protection days.",
            "Unmet demand is lost, not backordered; demand is externally fixed, without substitution.",
            "Gross margin minus holding and fixed ordering fees is a contribution proxy, not measured profit.",
            "M5 test data were previously inspected; business simulation results are exploratory.",
        ],
    }
    (out / "simulation_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nSIMULATED INVENTORY RESULTS (all values conditional on stated assumptions)")
    print(overall[["model", "fill_rate_pct", "unmet_units", "stockout_sku_days",
                   "mean_ending_inventory_units_per_sku", "replenishment_spend",
                   "contribution_proxy"]].round(2).to_string(index=False))
    print(f"\nOutputs: {out}")
    return overall, weekly, products, daily


if __name__ == "__main__":
    main()
