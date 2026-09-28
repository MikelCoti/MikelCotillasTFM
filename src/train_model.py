"""Train the matched-period LightGBM baseline and export Streamlit forecasts.

Run from root, after prepare_data.py and build_graph.py:
    python src/train_model.py
"""
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from config import (
    PROCESSED, GRAPH_DIR, OUTPUTS, MODELS, FEATURES, CATEGORICAL,
    HORIZON, SEED, TRAIN_FIRST_CUTOFF, TRAIN_LAST_CUTOFF,
    VALID_FIRST_CUTOFF, VALID_LAST_CUTOFF,
    TEST_FIRST_CUTOFF, TEST_LAST_CUTOFF, GRAPH_REFERENCE_LAST,
    EXPECTED_NODES, CATEGORIES, N_PRODUCTS_PER_CATEGORY,
)

TARGET = "target_7d"


def load_panel():
    path = PROCESSED / "m5_panel.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Run prepare_data.py first: {path}")
    df = pd.read_parquet(path).sort_values(["d_num", "store_id", "item_id"]).reset_index(drop=True)
    if df.duplicated(["item_id", "store_id", "d_num"]).any():
        raise ValueError("Duplicate item-store-cutoff rows")
    required = FEATURES + [TARGET, "d_num", "sales_sum_7"]
    if df[[c for c in required if c != "sell_price"]].isna().any().any():
        raise ValueError("Missing mandatory features/targets")
    for col in CATEGORICAL:
        df[col] = df[col].astype("category")
    if df.groupby(["item_id", "store_id"]).ngroups != EXPECTED_NODES:
        raise ValueError("Prepared panel has an incorrect SKU-store universe. Re-run prepare_data.py")
    return df


def assert_graph_alignment():
    path = GRAPH_DIR / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Run build_graph.py before training: {path}")
    meta = json.loads(path.read_text(encoding="utf-8"))
    if meta["reference_end"] != GRAPH_REFERENCE_LAST or meta["first_training_cutoff"] != TRAIN_FIRST_CUTOFF:
        raise ValueError("Graph and training dates differ. Rebuild graph with current config.py")
    expected = {cat: N_PRODUCTS_PER_CATEGORY for cat in CATEGORIES}
    if meta.get("num_nodes") != EXPECTED_NODES or meta.get("categories") != expected:
        raise ValueError("Graph is from a different product selection. Re-run build_graph.py")


def split(df):
    train = df.loc[df.d_num.between(TRAIN_FIRST_CUTOFF, TRAIN_LAST_CUTOFF)].copy()
    valid = df.loc[df.d_num.between(VALID_FIRST_CUTOFF, VALID_LAST_CUTOFF)].copy()
    test = df.loc[df.d_num.between(TEST_FIRST_CUTOFF, TEST_LAST_CUTOFF)].copy()
    if any(part.empty for part in (train, valid, test)):
        raise ValueError("Empty train/validation/test partition")
    if train.d_num.max() + HORIZON >= valid.d_num.min():
        raise ValueError("Training labels reach validation inputs")
    if valid.d_num.max() + HORIZON >= test.d_num.min():
        raise ValueError("Validation labels reach test inputs")
    print(f"Rows: train={len(train):,}, valid={len(valid):,}, test={len(test):,}")
    print(f"Training cutoffs d_{train.d_num.min()}–d_{train.d_num.max()}")
    return train, valid, test


def metrics(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    denominator = actual.sum()
    return {
        "MAE": float(np.abs(pred - actual).mean()),
        "WAPE": float(np.abs(pred - actual).sum() / denominator) if denominator > 0 else np.nan,
        "Bias": float((pred - actual).sum() / denominator) if denominator > 0 else np.nan,
    }


def train(train_df, val_df):
    model = lgb.LGBMRegressor(
        objective="regression", metric="l1", n_estimators=500,
        learning_rate=0.05, num_leaves=31, min_child_samples=40,
        random_state=SEED, n_jobs=-1, verbosity=-1,
    )
    model.fit(
        train_df[FEATURES], train_df[TARGET],
        eval_set=[(val_df[FEATURES], val_df[TARGET])],
        eval_metric="l1", categorical_feature=CATEGORICAL,
        callbacks=[lgb.early_stopping(40, first_metric_only=True, verbose=False)],
    )
    print("Best boosting iteration:", model.best_iteration_)
    return model


def evaluate_and_export(model, test):
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    pred = np.maximum(model.predict(test[FEATURES]), 0.0)
    naive = test.sales_sum_7.to_numpy(dtype=float)
    actual = test[TARGET].to_numpy(dtype=float)
    results = pd.DataFrame([
        {"model": "Seasonal naive", **metrics(actual, naive)},
        {"model": "LightGBM", **metrics(actual, pred)},
    ])
    results.to_csv(OUTPUTS / "model_metrics.csv", index=False)
    print("\nTEST RESULTS (exploratory holdout)")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    backtest = test[[
        "item_id", "store_id", "cat_id", "d_num", "date", "sell_price",
        "sales_sum_7", "target_7d",
    ]].copy()
    backtest["pred_lightgbm"] = pred
    backtest["pred_naive"] = naive
    backtest.to_csv(OUTPUTS / "backtest_predictions.csv", index=False)

    # Only the final historical test cutoff is displayed; this is NOT live data.
    last = backtest.loc[backtest.d_num.eq(TEST_LAST_CUTOFF)].copy()
    last = last.sort_values(["store_id", "item_id"]).reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    simulated_stock = np.rint(
        last.sales_sum_7.to_numpy() * rng.uniform(0.5, 1.5, len(last)) + 2
    ).clip(min=0).astype(int)
    dashboard = pd.DataFrame({
        "sku": last.item_id.to_numpy(), "store": last.store_id.to_numpy(),
        "category": last.cat_id.to_numpy(),
        "forecast_7d": last.pred_lightgbm.to_numpy(),
        "inventory": simulated_stock,
        "unit_price": last.sell_price.to_numpy(),
        "alpha": 0.3,  # Uncalibrated assumption: NOT learned from historical stockouts.
        "forecast_cutoff": last.date.to_numpy(),
        "forecast_model": "LightGBM",
        "inventory_source": "simulated",
    })
    before = len(dashboard)
    dashboard = dashboard.dropna(subset=["unit_price"]).copy()
    if dashboard.empty:
        raise ValueError("All dashboard prices are missing at the forecast cutoff")
    dashboard.to_csv(OUTPUTS / "predictions.csv", index=False)
    print(f"Exported {len(dashboard)} dashboard products; {before-len(dashboard)} lacked price")
    print(f"Historical forecast cutoff: {last.date.iloc[0]} (d_{TEST_LAST_CUTOFF})")
    print(f"Files written to: {OUTPUTS}")


def main():
    assert_graph_alignment()
    df = load_panel()
    train_df, valid_df, test_df = split(df)
    model = train(train_df, valid_df)
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS / "lightgbm_7d.joblib")
    evaluate_and_export(model, test_df)


if __name__ == "__main__":
    main()
