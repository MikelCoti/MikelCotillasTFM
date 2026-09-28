"""Audit Milestone 4 simulation exports from each scenario directory.

Run: python src/validate_inventory.py
Or:  python src/validate_inventory.py --folders outputs/inventory_unconstrained outputs/inventory_budget_12000
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from inventory_reporting import (
    compare_scenario_inputs, discover_scenarios, load_scenario, scenario_label,
    validate_scenario,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--folders", nargs="+", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None,
                        help="Optional JSON audit report location")
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    if args.folders:
        folders = [p if p.is_absolute() else root / p for p in args.folders]
        scenarios = [load_scenario(p, details=True) for p in folders]
    else:
        scenarios = [load_scenario(item["folder"], details=True)
                     for item in discover_scenarios(root)]
    if not scenarios:
        raise SystemExit("No scenario exports found under outputs/inventory*. Run simulator first.")
    report = []
    for scenario in scenarios:
        details = validate_scenario(scenario)
        print(f"PASS: {scenario_label(scenario)} | {details['models']} models | "
              f"max weekly spend {details['maximum_weekly_spend']:,.2f}")
        report.append({"scenario": scenario["folder"].name, **details})
    cross_scenario = compare_scenario_inputs(scenarios)
    print(f"PASS: scenario alignment ({cross_scenario['scenarios']} scenarios, "
          f"{cross_scenario['checked_nodes_days']} node-day records compared)")
    if args.report:
        dest = args.report if args.report.is_absolute() else root / args.report
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps({"scenarios": report, "alignment": cross_scenario}, indent=2),
                        encoding="utf-8")
        print(f"Saved audit report: {dest}")
    return report


if __name__ == "__main__":
    main()
