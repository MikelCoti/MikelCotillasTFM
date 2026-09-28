"""Synthetic, self-contained tests of the new business analysis functions."""
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

PATCH_SRC = Path(__file__).resolve().parents[1] / "src"
PROJECT_SRC = Path(__file__).resolve().parents[2] / "retail-risk-gdl-multicategory" / "src"
for folder in (PROJECT_SRC, PATCH_SRC):
    sys.path.insert(0, str(folder))

from analyze_inventory import (  # noqa: E402
    analyze, category_tables, forecast_summary, product_comparison,
    weekly_forecasting_data,
)
from inventory_reporting import validate_scenario, compare_scenario_inputs  # noqa: E402
from simulate_inventory import simulate_model, summarize  # noqa: E402
import run_policy_sensitivity  # noqa: E402


class BusinessAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.nodes = pd.DataFrame({
            "node_id": [0, 1, 2, 3],
            "item_id": ["FOODS_1_001", "FOODS_1_002", "HOUSEHOLD_1_001", "HOUSEHOLD_1_002"],
            "store_id": ["CA_1"] * 4,
            "cat_id": ["FOODS", "FOODS", "HOUSEHOLD", "HOUSEHOLD"],
        })
        self.cutoffs = np.arange(1899, 1935, 7)
        self.demand = np.tile([[3, 6, 2, 5, 4, 3, 2] * 6], (4, 1)).astype(float)
        self.demand[1] = self.demand[1] + 1
        self.demand[3] = self.demand[3] + 2
        self.initial_sales = np.array([20, 30, 15, 25], dtype=float)
        self.prices = np.array([10.0, 11.0, 25.0, 22.0])
        self.predictions = {
            "lightgbm": np.full((6, 4), 25.0),
            "node_only": np.full((6, 4), 23.0),
            "cross_store": np.full((6, 4), 26.0),
            "full": np.full((6, 4), 28.0),
        }

    def scenario(self, name, budget):
        frame = pd.concat([
            simulate_model(model, self.nodes, self.demand, self.initial_sales,
                           self.prices, forecast, self.cutoffs,
                           lead_days=3, safety_days=2, initial_cover_weeks=1.2,
                           weekly_budget=budget, cost_ratio=0.7,
                           holding_rate=0.25, order_fee=2.0)
            for model, forecast in self.predictions.items()
        ], ignore_index=True)
        summary, weekly, products = summarize(frame, price_imputations=0, weekly_budget=budget)
        return {"folder": Path(name), "budget": budget, "summary": summary,
                "weekly": weekly, "products": products, "daily": frame,
                "config": {"settings": {"lead_days": 3, "safety_days": 2}}}

    def test_forecasts_and_category_weighted_rates(self):
        scenario = self.scenario("budget_400", 400)
        audit = validate_scenario(scenario)
        self.assertEqual(audit["models"], 4)
        forecasts = weekly_forecasting_data(scenario["daily"])
        self.assertEqual(len(forecasts), 4 * 4 * 6)
        self.assertTrue((forecasts.predictions if "predictions" in forecasts else forecasts.actual_7d >= 0).all())
        cat, _ = category_tables(scenario["daily"], "budget_400", 400)
        actual = cat.loc[(cat.cat_id == "FOODS") & (cat.model == "lightgbm")].iloc[0]
        self.assertAlmostEqual(actual.fill_rate_pct,
                               100 * actual.fulfilled_units / actual.demand_proxy_units)
        _, product_fc = forecast_summary(forecasts)
        self.assertEqual(len(product_fc), 16)

    def test_analyses_and_scenario_alignment(self):
        one = self.scenario("budget_400", 400)
        two = self.scenario("budget_300", 300)
        compare_scenario_inputs([one, two])
        tables = analyze([one, two])
        self.assertEqual(len(tables["category_inventory"]), 2 * 2 * 4)
        self.assertEqual(len(tables["product_comparison"]), 2 * 3 * 4)
        pairs = tables["product_comparison"]
        self.assertTrue((pairs.demand_proxy_units == pairs.demand_proxy_units_baseline).all())
        lightgbm_forecast = tables["product_model_metrics"]
        self.assertTrue((lightgbm_forecast.predictions == 6).all())
        bycat = tables["category_inventory"].groupby(["scenario", "model"])[
            ["fulfilled_units", "unmet_units"]].sum()
        for s in (one, two):
            for row in s["summary"].itertuples(index=False):
                totals = bycat.loc[(s["folder"].name, row.model)]
                self.assertAlmostEqual(totals.fulfilled_units, row.fulfilled_units)
                self.assertAlmostEqual(totals.unmet_units, row.unmet_units)

    def test_policy_sensitivity_runner(self):
        def fake_problem(root, models, seeds):
            return (self.nodes, self.demand, self.initial_sales, self.prices,
                    {m: self.predictions[m] for m in models}, self.cutoffs, 0)

        with tempfile.TemporaryDirectory() as tempdir:
            with patch.object(run_policy_sensitivity, "load_problem", side_effect=fake_problem):
                overall, categories, paired = run_policy_sensitivity.main([
                    "--project-root", tempdir, "--output-dir", "results",
                    "--budgets", "300", "400", "--lead-days", "1", "3",
                    "--safety-days", "0", "2", "--seeds", "42",
                    "--models", "lightgbm", "node_only", "cross_store", "full",
                ])
            self.assertEqual(len(overall), 2 * 2 * 2 * 4)
            self.assertEqual(len(categories), 2 * 2 * 2 * 4 * 2)
            self.assertEqual(len(paired), 2 * 2 * 2 * 3)
            self.assertTrue((overall.replenishment_spend <= overall.weekly_budget * 6 + 1e-6).all())
            self.assertTrue((Path(tempdir) / "results" / "policy_sensitivity_vs_lightgbm.csv").exists())

    def test_detect_forecast_scenario_mismatch(self):
        one = self.scenario("a", 400)
        two = self.scenario("b", 300)
        mask = ((two["daily"].model == "full") & (two["daily"].node_id == 0)
                & (two["daily"].week == 1) & two["daily"].forecast_7d.notna())
        two["daily"].loc[mask, "forecast_7d"] += 1
        with self.assertRaisesRegex(ValueError, "Forecasts or realized demand differ"):
            analyze([one, two])


if __name__ == "__main__":
    unittest.main()
