"""Evaluate rolling and non-overlapping forecasts with fixed demand segments.

Run from root: python src/evaluate_backtest.py
"""
import numpy as np
import pandas as pd

from config import PROCESSED, OUTPUTS, TEST_FIRST_CUTOFF, TEST_LAST_CUTOFF, HORIZON

EVAL_DIR = OUTPUTS / "evaluation"
PRODUCT_KEYS = ["item_id", "store_id"]
GNN_BACKTEST = OUTPUTS / "backtest_predictions_with_graphsage.csv"
MODEL_COLUMNS = {"Seasonal naive": "pred_naive", "LightGBM": "pred_lightgbm"}
if GNN_BACKTEST.exists():
    MODEL_COLUMNS["GraphSAGE"] = "pred_graphsage"


def metrics(y_true, y_pred):
    actual = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    total = actual.sum()
    return {
        "MAE": float(np.mean(np.abs(actual - pred))),
        "WAPE (%)": float(100 * np.abs(actual - pred).sum() / total) if total > 0 else np.nan,
        "Bias (%)": float(100 * (pred - actual).sum() / total) if total > 0 else np.nan,
    }


def load_data():
    path = GNN_BACKTEST if GNN_BACKTEST.exists() else OUTPUTS / "backtest_predictions.csv"
    df = pd.read_csv(path)
    print(f"Evaluating backtest: {path}")
    df = df.loc[df.d_num.between(TEST_FIRST_CUTOFF, TEST_LAST_CUTOFF)].copy()
    required = PRODUCT_KEYS + ["d_num", "target_7d", *MODEL_COLUMNS.values()]
    if df.empty or df[required].isna().any().any():
        raise ValueError("Empty or incomplete test backtest")
    if df.duplicated(PRODUCT_KEYS + ["d_num"]).any():
        raise ValueError("Duplicate product-store forecasting cutoffs")
    all_cutoffs = set(range(TEST_FIRST_CUTOFF, TEST_LAST_CUTOFF + 1))
    if set(df.d_num.unique()) != all_cutoffs:
        raise ValueError("Missing test forecasting cutoffs")
    return df


def add_fixed_segments(df):
    panel = pd.read_parquet(
        PROCESSED / "m5_panel.parquet",
        columns=PRODUCT_KEYS + ["d_num", "sales_sum_7"],
    )
    # Pre-test 28-day average weekly demand yields more stable segment labels
    # than a single previous week. Information through d_1898 is allowed.
    reference_day = TEST_FIRST_CUTOFF - 1
    ref = panel.loc[panel.d_num.between(reference_day - 27, reference_day)]
    baseline = (ref.groupby(PRODUCT_KEYS, as_index=False)
                .agg(baseline_weekly_sales=("sales_sum_7", "mean")))
    if baseline.duplicated(PRODUCT_KEYS).any():
        raise ValueError("Duplicate demand-segment baselines")
    baseline["demand_segment"] = pd.cut(
        baseline.baseline_weekly_sales,
        bins=[-np.inf, 5, 20, 50, np.inf],
        labels=["Very low", "Low", "Medium", "High"],
    )
    df = df.merge(
        baseline[PRODUCT_KEYS + ["demand_segment"]],
        on=PRODUCT_KEYS, how="left", validate="many_to_one",
    )
    if df.demand_segment.isna().any():
        raise ValueError("Some products have no fixed demand segment")
    return df


def summarize(df, evaluation):
    overall, segments, products = [], [], []
    for model, column in MODEL_COLUMNS.items():
        overall.append({
            "Evaluation": evaluation, "Model": model, "Observations": len(df),
            "Forecast periods": df.d_num.nunique(),
            **metrics(df.target_7d, df[column]),
        })
        for segment, group in df.groupby("demand_segment", observed=True):
            segments.append({
                "Evaluation": evaluation, "Model": model, "Segment": str(segment),
                "Products": group[PRODUCT_KEYS].drop_duplicates().shape[0],
                "Observations": len(group),
                **metrics(group.target_7d, group[column]),
            })
        for (item, store), group in df.groupby(PRODUCT_KEYS, observed=True):
            products.append({
                "Evaluation": evaluation, "Model": model,
                "item_id": item, "store_id": store,
                "Segment": str(group.demand_segment.iloc[0]),
                "Observations": len(group), "Actual units": group.target_7d.sum(),
                **metrics(group.target_7d, group[column]),
            })
    return pd.DataFrame(overall), pd.DataFrame(segments), pd.DataFrame(products)


def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    rolling = add_fixed_segments(load_data())
    weekly_days = np.arange(TEST_FIRST_CUTOFF, TEST_LAST_CUTOFF + 1, HORIZON)
    weekly = rolling.loc[rolling.d_num.isin(weekly_days)].copy()
    if set(weekly.d_num.unique()) != set(weekly_days):
        raise ValueError("Missing non-overlapping weekly cutoffs")
    overall_list, segment_list, product_list, category_list = [], [], [], []
    for label, subset in (("Daily rolling", rolling), ("Non-overlapping weekly", weekly)):
        overall, segments, products = summarize(subset, label)
        overall_list.append(overall)
        segment_list.append(segments)
        product_list.append(products)
        # Each item_id carries its original M5 category prefix.
        category_frame = subset.copy()
        category_frame["cat_id"] = category_frame.item_id.str.split("_").str[0]
        for category, category_subset in category_frame.groupby("cat_id"):
            for model, column in MODEL_COLUMNS.items():
                category_list.append({"Evaluation": label, "Category": category,
                                      "Model": model, "Observations": len(category_subset),
                                      **metrics(category_subset.target_7d, category_subset[column])})
    overall = pd.concat(overall_list, ignore_index=True)
    segments = pd.concat(segment_list, ignore_index=True)
    products = pd.concat(product_list, ignore_index=True)
    pd.DataFrame(category_list).to_csv(EVAL_DIR / "category_metrics.csv", index=False)
    overall.to_csv(EVAL_DIR / "overall_metrics.csv", index=False)
    segments.to_csv(EVAL_DIR / "segment_metrics.csv", index=False)
    products.to_csv(EVAL_DIR / "product_metrics.csv", index=False)
    weekly.to_csv(EVAL_DIR / "weekly_predictions.csv", index=False)
    print("\nOVERALL MODEL EVALUATION")
    print(overall.round(3).to_string(index=False))
    print(f"\nSaved evaluation files to: {EVAL_DIR}")


if __name__ == "__main__":
    main()
