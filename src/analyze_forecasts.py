"""Inspect forecasting errors by RECENT demand state and by fixed segment.

Run after evaluate_backtest.py: python src/analyze_forecasts.py
"""
import numpy as np
import pandas as pd

from config import OUTPUTS

BINS = [-np.inf, 5, 20, 50, np.inf]
LABELS = ["Very low", "Low", "Medium", "High"]


def summarize(df, segment_column):
    rows = []
    for name, group in df.groupby(segment_column, observed=True):
        actual = group.target_7d.to_numpy(dtype=float)
        lgbm = group.pred_lightgbm.to_numpy(dtype=float)
        naive = group.pred_naive.to_numpy(dtype=float)
        mae_lgbm = np.abs(lgbm - actual).mean()
        mae_naive = np.abs(naive - actual).mean()
        rows.append({
            "Segment": str(name), "Observations": len(group),
            "MAE LightGBM": mae_lgbm, "MAE Naive": mae_naive,
            "MAE improvement (%)": 100 * (mae_naive - mae_lgbm) / mae_naive if mae_naive else np.nan,
            "Bias LightGBM (%)": 100 * (lgbm - actual).sum() / actual.sum() if actual.sum() > 0 else np.nan,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["Segment"] = pd.Categorical(out.Segment, categories=LABELS, ordered=True)
        out = out.sort_values("Segment").reset_index(drop=True)
    return out


def main():
    df = pd.read_csv(OUTPUTS / "backtest_predictions.csv")
    df["recent_demand_segment"] = pd.cut(
        df.sales_sum_7, bins=BINS, labels=LABELS,
    )
    recent = summarize(df, "recent_demand_segment")
    recent.to_csv(OUTPUTS / "evaluation" / "recent_demand_segments.csv", index=False)
    print("\nRECENT-DEMAND SEGMENTS (may change at each cutoff)")
    print(recent.round(2).to_string(index=False))

    weekly_file = OUTPUTS / "evaluation" / "weekly_predictions.csv"
    if weekly_file.exists():
        weekly = pd.read_csv(weekly_file)
        fixed = summarize(weekly, "demand_segment")
        fixed.to_csv(OUTPUTS / "evaluation" / "fixed_weekly_segments.csv", index=False)
        print("\nFIXED SEGMENTS — NON-OVERLAPPING WEEKLY FORECASTS")
        print(fixed.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
