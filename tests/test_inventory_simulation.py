"""Run: python -m unittest discover -s tests -p 'test_inventory_simulation.py' -v"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from simulate_inventory import load_problem, simulate_model, summarize


class InventorySimulationTests(unittest.TestCase):
    @staticmethod
    def fixture(tmp: Path) -> None:
        for folder in ("data/processed/graph", "data/raw", "outputs/ablation"):
            (tmp / folder).mkdir(parents=True, exist_ok=True)
        nodes = pd.DataFrame({
            "node_id": [0, 1], "item_id": ["A", "B"], "store_id": ["CA_1", "CA_1"]
        })
        nodes.to_csv(tmp / "data/processed/graph/nodes.csv", index=False)
        day_cols = {f"d_{d}": [3, 1] for d in range(1900, 1942)}
        pd.DataFrame({"item_id": ["A", "B"], "store_id": ["CA_1", "CA_1"],
                      **day_cols}).to_csv(tmp / "data/raw/sales_train_evaluation.csv", index=False)
        baseline = []
        graph = []
        for d in range(1899, 1935):
            for node, item in enumerate(("A", "B")):
                baseline.append({"item_id": item, "store_id": "CA_1", "d_num": d,
                                 "target_7d": 21 if node == 0 else 7,
                                 "sales_sum_7": 21 if node == 0 else 7,
                                 "sell_price": 10 if node == 0 else 8,
                                 "pred_naive": 21 if node == 0 else 7,
                                 "pred_lightgbm": 22 if node == 0 else 6})
                graph.append({"node_id": node, "d_num": d,
                              "target_7d": 21 if node == 0 else 7,
                              "pred_graphsage": 20 if node == 0 else 8})
        pd.DataFrame(baseline).to_csv(tmp / "outputs/backtest_predictions.csv", index=False)
        pd.DataFrame(graph).to_csv(tmp / "outputs/ablation/test_cross_store_seed42.csv", index=False)

    def test_end_to_end_alignment_budget_and_conservation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            nodes, demand, initial, prices, forecasts, cutoffs, missing = load_problem(
                root, ["seasonal_naive", "lightgbm", "cross_store"], [42]
            )
            self.assertEqual((2, 42), demand.shape)
            self.assertEqual(0, missing)
            result = {}
            for model in forecasts:
                result[model] = simulate_model(
                    model, nodes, demand, initial, prices, forecasts[model], cutoffs,
                    lead_days=3, safety_days=2, initial_cover_weeks=.5,
                    weekly_budget=80, cost_ratio=.7, holding_rate=.25, order_fee=2,
                )
                daily = result[model]
                self.assertEqual(84, len(daily))
                self.assertEqual(168, daily.demand_proxy.sum())
                self.assertTrue(np.allclose(daily.fulfilled_units + daily.unmet_units,
                                            daily.demand_proxy))
                self.assertTrue((daily.ending_inventory >= 0).all())
                weekly_spend = daily.groupby("week").order_spend.sum()
                self.assertTrue((weekly_spend <= 80 + 1e-5).all())
                # Orders decided on day 1899 with lead=3 cannot arrive before d_1902.
                self.assertEqual(0, daily.loc[daily.d_num.isin([1900, 1901]),
                                               "received_units"].sum())
                self.assertTrue((daily.loc[daily.d_num == 1900, "ordered_units"] >= 0).all())
            self.assertTrue((result["seasonal_naive"].demand_proxy.to_numpy()
                             == result["cross_store"].demand_proxy.to_numpy()).all())
            self.assertTrue((result["seasonal_naive"].stock_before_demand.iloc[:2].to_numpy()
                             == result["cross_store"].stock_before_demand.iloc[:2].to_numpy()).all())
            summary, weekly, products = summarize(pd.concat(result.values()),
                                                   price_imputations=0, weekly_budget=80)
            self.assertEqual(3, len(summary))
            self.assertEqual(18, len(weekly))
            self.assertEqual(6, len(products))
            self.assertTrue((summary.demand_proxy_units == 168).all())
            self.assertTrue((summary.fill_rate_pct.between(0, 100)).all())

    def test_future_demand_does_not_change_orders(self):
        nodes = pd.DataFrame({"node_id": [0], "item_id": ["A"], "store_id": ["CA_1"]})
        cutoffs = np.arange(1899, 1935, 7)
        forecasts = np.full((6, 1), 8.0)
        kwargs = dict(lead_days=3, safety_days=2, initial_cover_weeks=0,
                      weekly_budget=50, cost_ratio=.7, holding_rate=.25, order_fee=2)
        d1 = simulate_model("test", nodes, np.ones((1, 42)), np.array([2.]),
                            np.array([10.]), forecasts, cutoffs, **kwargs)
        d2 = simulate_model("test", nodes, np.ones((1, 42)) * 100, np.array([2.]),
                            np.array([10.]), forecasts, cutoffs, **kwargs)
        # The first week's order must be determined entirely by the common inputs.
        self.assertEqual(int(d1.ordered_units.iloc[0]), int(d2.ordered_units.iloc[0]))
        self.assertGreater(d2.unmet_units.sum(), d1.unmet_units.sum())

    def test_cli_writes_expected_csvs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            command = [sys.executable, str(ROOT / "src/simulate_inventory.py"),
                       "--project-root", str(root), "--models", "seasonal_naive", "lightgbm",
                       "cross_store", "--seeds", "42", "--weekly-budget", "80"]
            result = subprocess.run(command, capture_output=True, text=True,
                                    check=True, timeout=30)
            self.assertIn("SIMULATED INVENTORY RESULTS", result.stdout)
            for file in ("inventory_daily.csv", "inventory_weekly.csv", "inventory_products.csv",
                         "inventory_summary.csv", "simulation_config.json"):
                self.assertTrue((root / "outputs/inventory" / file).exists(), file)


if __name__ == "__main__":
    unittest.main()
