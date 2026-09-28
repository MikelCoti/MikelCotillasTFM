"""Hold forecasts fixed; vary lead times and safety-stock allowances.

This runner reuses the *existing* simulator and its weekly predictions. It saves
portfolio- and category-level summaries (not dozens of huge daily CSV exports).
All budget/model combinations use the same initial inventory and M5 demand proxy.

Example:
  python src/run_policy_sensitivity.py --budgets 36000 25000 \
      --lead-days 1 3 5 --safety-days 0 2 4 --seeds 42
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_inventory import category_tables
from simulate_inventory import ALL_MODELS, load_problem, simulate_model, summarize
from config import ROOT


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/business_analysis"))
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS,
                        default=["lightgbm", "node_only", "cross_store", "full"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--budgets", nargs="+", type=float, default=[36000, 25000])
    parser.add_argument("--lead-days", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--safety-days", nargs="+", type=float, default=[0, 2, 4])
    parser.add_argument("--initial-cover-weeks", type=float, default=1.2)
    parser.add_argument("--cost-ratio", type=float, default=0.70)
    parser.add_argument("--holding-rate", type=float, default=0.25)
    parser.add_argument("--order-fee", type=float, default=2.0)
    args = parser.parse_args(argv)
    if len(set(args.models)) != len(args.models) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Models and seeds must each be unique")
    if any(b < 0 for b in args.budgets):
        parser.error("Budgets must be nonnegative")
    if any(not 1 <= lead <= 7 for lead in args.lead_days):
        parser.error("Simulator supports lead times from 1 through 7 days")
    if any(s < 0 for s in args.safety_days):
        parser.error("Safety days must be nonnegative")
    root = args.project_root.resolve()
    dest = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    dest.mkdir(parents=True, exist_ok=True)

    # Load sales and all model forecasts ONCE, avoiding repeated large CSV reads.
    nodes, demand, initial_sales, prices, forecasts, cutoffs, price_imputations = load_problem(
        root, args.models, args.seeds
    )
    overall_rows, category_rows = [], []
    for budget in args.budgets:
        for lead in args.lead_days:
            for safety in args.safety_days:
                label = f"budget_{budget:g}_lead_{lead}_safety_{safety:g}"
                print(f"Evaluating {label}...", flush=True)
                scenario_models = []
                for model in args.models:
                    daily = simulate_model(
                        model, nodes, demand, initial_sales, prices, forecasts[model], cutoffs,
                        lead_days=lead, safety_days=safety,
                        initial_cover_weeks=args.initial_cover_weeks,
                        weekly_budget=budget, cost_ratio=args.cost_ratio,
                        holding_rate=args.holding_rate, order_fee=args.order_fee,
                    )
                    summary, _, _ = summarize(daily, price_imputations=price_imputations,
                                               weekly_budget=budget)
                    category, _ = category_tables(daily, label, budget)
                    for df in (summary, category):
                        df["lead_days"] = lead
                        df["safety_days"] = safety
                        df["initial_cover_weeks"] = args.initial_cover_weeks
                    summary["scenario"] = label
                    overall_rows.append(summary)
                    category_rows.append(category)
                    scenario_models.append(model)
                    del daily
                if set(scenario_models) != set(args.models):
                    raise AssertionError("A model is missing from the sensitivity experiment")
    overall = pd.concat(overall_rows, ignore_index=True)
    category = pd.concat(category_rows, ignore_index=True)
    if "lightgbm" not in set(overall.model):
        raise ValueError("The LightGBM baseline is required for paired comparisons")
    compare_fields = ["fill_rate_pct", "unmet_units", "stockout_sku_days",
                      "replenishment_spend", "contribution_proxy"]
    keys = ["weekly_budget", "lead_days", "safety_days"]
    base = overall.loc[overall.model.eq("lightgbm"), keys + compare_fields]
    paired = overall.loc[overall.model.ne("lightgbm")].merge(
        base, on=keys, how="left", validate="many_to_one", suffixes=("", "_lightgbm")
    )
    if paired["fill_rate_pct_lightgbm"].isna().any():
        raise ValueError("Missing LightGBM comparison for a policy scenario")
    paired["fill_rate_change_pp"] = paired.fill_rate_pct - paired.fill_rate_pct_lightgbm
    paired["unmet_reduction_units"] = paired.unmet_units_lightgbm - paired.unmet_units
    paired["stockout_days_avoided"] = paired.stockout_sku_days_lightgbm - paired.stockout_sku_days
    paired["spend_difference"] = paired.replenishment_spend - paired.replenishment_spend_lightgbm
    paired["contribution_difference"] = paired.contribution_proxy - paired.contribution_proxy_lightgbm
    for filename, data in {
        "policy_sensitivity.csv": overall,
        "policy_sensitivity_category.csv": category,
        "policy_sensitivity_vs_lightgbm.csv": paired,
    }.items():
        output = dest / filename
        data.to_csv(output, index=False)
        print(f"Saved {output} ({len(data):,} rows)")
    print("Policy sensitivity complete. Demand forecasts and starting inventory were held fixed.")
    return overall, category, paired


if __name__ == "__main__":
    main()
