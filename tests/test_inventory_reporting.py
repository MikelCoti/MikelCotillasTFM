"""Milestone 5 audit tests. Run from root: python -m unittest discover -s tests -p 'test_inventory_reporting.py' -v"""
from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from simulate_inventory import simulate_model, summarize
from inventory_reporting import (
    compare_scenario_inputs, discover_scenarios, load_scenario, product_differences,
    validate_scenario,
)


class ReportingTests(unittest.TestCase):
    @staticmethod
    def write_scenario(root: Path, folder: str, budget: float | None):
        target = root / "outputs" / folder
        target.mkdir(parents=True, exist_ok=True)
        nodes = pd.DataFrame({"node_id": [0, 1], "item_id": ["A", "B"],
                              "store_id": ["CA_1", "CA_1"]})
        demand = np.array([[3.0] * 42, [1.0] * 42])
        initial = np.array([21.0, 7.0])
        prices = np.array([10.0, 8.0])
        cutoffs = np.arange(1899, 1935, 7)
        forecasts = {
            "lightgbm": np.tile([22.0, 6.0], (6, 1)),
            "cross_store": np.tile([20.0, 8.0], (6, 1)),
        }
        frames = [simulate_model(name, nodes, demand, initial, prices, pred, cutoffs,
                                 lead_days=3, safety_days=2, initial_cover_weeks=.5,
                                 weekly_budget=budget, cost_ratio=.7, holding_rate=.25,
                                 order_fee=2)
                  for name, pred in forecasts.items()]
        daily = pd.concat(frames, ignore_index=True)
        summary, weekly, products = summarize(daily, price_imputations=0,
                                               weekly_budget=budget)
        for frame, fname in ((summary, "inventory_summary.csv"),
                             (weekly, "inventory_weekly.csv"),
                             (products, "inventory_products.csv"),
                             (daily, "inventory_daily.csv")):
            frame.to_csv(target / fname, index=False)
        return target

    def test_validate_two_scenarios_and_comparison(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            a = self.write_scenario(root, "inventory_unconstrained", None)
            b = self.write_scenario(root, "inventory_budget_80", 80)
            discovered = discover_scenarios(root)
            self.assertEqual(2, len(discovered))
            scenarios = [load_scenario(folder, details=True) for folder in (a, b)]
            for scenario in scenarios:
                audit = validate_scenario(scenario)
                self.assertEqual(2, audit["models"])
                self.assertEqual(42, audit["days"])
                self.assertEqual(6, audit["weeks"])
            aligned = compare_scenario_inputs(scenarios)
            self.assertEqual(2, aligned["scenarios"])
            paired = product_differences(scenarios[1]["products"], "cross_store")
            self.assertEqual(2, len(paired))
            self.assertAlmostEqual(paired.unmet_reduction_units.sum(),
                                   float(scenarios[1]["summary"].set_index("model").loc["lightgbm", "unmet_units"]
                                         - scenarios[1]["summary"].set_index("model").loc["cross_store", "unmet_units"]))

    def test_detects_weekly_overspend(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = self.write_scenario(root, "inventory_budget_80", 80)
            scenario = load_scenario(folder, details=True)
            # Keep saved aggregate totals consistent with the altered daily row:
            # validation must still reject the illegal per-week expenditure.
            daily = scenario["daily"]
            marker = (daily.model.eq("lightgbm") & daily.week.eq(1)
                      & daily.node_id.eq(0) & daily.d_num.eq(1900))
            daily.loc[marker, "order_spend"] += 1000
            scenario["weekly"].loc[
                scenario["weekly"].model.eq("lightgbm") & scenario["weekly"].week.eq(1),
                "replenishment_spend"] += 1000
            scenario["products"].loc[
                scenario["products"].model.eq("lightgbm") & scenario["products"].node_id.eq(0),
                "replenishment_spend"] += 1000
            scenario["summary"].loc[
                scenario["summary"].model.eq("lightgbm"), "replenishment_spend"] += 1000
            with self.assertRaisesRegex(ValueError, "Weekly purchasing budget"):
                validate_scenario(scenario)

    def test_detects_inventory_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = self.write_scenario(Path(temp), "inventory_budget_80", 80)
            scenario = load_scenario(folder, details=True)
            scenario["daily"].loc[0, "ending_inventory"] += 1
            with self.assertRaisesRegex(ValueError, "Daily inventory conservation"):
                validate_scenario(scenario)

    def test_detects_cross_scenario_demand_shift(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            a = self.write_scenario(root, "inventory_unconstrained", None)
            b = self.write_scenario(root, "inventory_budget_80", 80)
            x, y = (load_scenario(p, details=True) for p in (a, b))
            y["daily"].loc[y["daily"].model.eq("cross_store"), "demand_proxy"] += 1
            with self.assertRaisesRegex(ValueError, "Cross-scenario observed demand"):
                compare_scenario_inputs([x, y])


if __name__ == "__main__":
    unittest.main()
