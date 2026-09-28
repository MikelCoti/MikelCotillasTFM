"""Create leakage-aware seven-day forecasting panel from the M5 CSV files.

Run from the project root: python src/prepare_data.py
"""
import numpy as np
import pandas as pd

from config import (
    RAW, PROCESSED, CATEGORIES, STORES, N_PRODUCTS_PER_CATEGORY, EXPECTED_NODES, SEED,
    FIRST_DAY, LAST_DAY, HORIZON, MAX_LAG, PANEL_FIRST_CUTOFF,
    NODE_COLUMNS,
)

DAY_COLUMNS = [f"d_{d}" for d in range(FIRST_DAY, LAST_DAY + 1)]
RAW_META = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]


def load_sales():
    """Sample the same item IDs across stores, separately within each category."""
    path = RAW / "sales_train_evaluation.csv"
    if not path.exists():
        raise FileNotFoundError(f"Download the M5 evaluation sales file: {path}")

    pieces = []
    for chunk in pd.read_csv(path, usecols=RAW_META + DAY_COLUMNS, chunksize=4000):
        sub = chunk.loc[
            chunk["cat_id"].isin(CATEGORIES) & chunk["store_id"].isin(STORES)
        ]
        if not sub.empty:
            pieces.append(sub)
    if not pieces:
        raise ValueError("No matching products and stores in the M5 sales file")
    sales = pd.concat(pieces, ignore_index=True)
    if sales.duplicated(["item_id", "store_id"]).any():
        raise ValueError("Duplicate product-store combinations in raw sales")

    selected = []
    # A separate reproducible random draw for each category means the
    # selection is not dominated by the category with the most SKUs.
    for category in CATEGORIES:
        sub = sales.loc[sales.cat_id.eq(category)]
        counts = sub.groupby("item_id")["store_id"].nunique()
        eligible = sorted(counts[counts.eq(len(STORES))].index)
        if len(eligible) < N_PRODUCTS_PER_CATEGORY:
            raise ValueError(
                f"{category}: only {len(eligible)} items occur in all "
                f"{len(STORES)} selected stores; need {N_PRODUCTS_PER_CATEGORY}"
            )
        # Category-specific seed: adding/reordering categories does not
        # inadvertently change a different category's selected items.
        cat_seed = SEED + CATEGORIES.index(category)
        chosen = np.random.default_rng(cat_seed).choice(
            eligible, size=N_PRODUCTS_PER_CATEGORY, replace=False
        )
        selected.append(sub.loc[sub.item_id.isin(chosen)])

    result = pd.concat(selected, ignore_index=True)
    if len(result) != EXPECTED_NODES:
        raise ValueError(f"Expected {EXPECTED_NODES} SKU-store rows, found {len(result)}")
    counts = result.groupby("cat_id")["item_id"].nunique()
    print("Selected items per category:\n" + counts.to_string())
    print(f"Selected {len(result):,} SKU-store nodes across {len(STORES)} stores")
    return result


def reshape(sales):
    daily = sales.melt(
        id_vars=RAW_META, value_vars=DAY_COLUMNS, var_name="d", value_name="sales"
    )
    daily["d_num"] = daily["d"].str[2:].astype("int32")
    daily["sales"] = daily["sales"].astype("float32")
    return daily


def add_calendar(daily):
    cal = pd.read_csv(RAW / "calendar.csv")
    cal["is_event"] = cal[["event_name_1", "event_name_2"]].notna().any(axis=1).astype("int8")
    cols = ["d", "date", "wm_yr_wk", "wday", "month", "is_event", "snap_CA", "snap_TX"]
    daily = daily.merge(cal[cols], on="d", how="left", validate="many_to_one")
    if daily["date"].isna().any():
        raise ValueError("Missing day identifiers in calendar.csv")
    daily["date"] = pd.to_datetime(daily["date"])
    daily["snap"] = np.where(daily["state_id"].eq("CA"), daily["snap_CA"], daily["snap_TX"]).astype("int8")
    return daily.drop(columns=["snap_CA", "snap_TX"])


def add_prices(daily):
    relevant = set(daily.item_id.unique())
    pieces = []
    for chunk in pd.read_csv(RAW / "sell_prices.csv", chunksize=250_000):
        sub = chunk.loc[chunk.item_id.isin(relevant) & chunk.store_id.isin(STORES)]
        if not sub.empty:
            pieces.append(sub)
    if not pieces:
        raise ValueError("No relevant selling prices in sell_prices.csv")
    prices = pd.concat(pieces, ignore_index=True)
    daily = daily.merge(
        prices[["store_id", "item_id", "wm_yr_wk", "sell_price"]],
        on=["store_id", "item_id", "wm_yr_wk"], how="left", validate="many_to_one",
    )
    daily = daily.sort_values(["item_id", "store_id", "d_num"]).reset_index(drop=True)
    # Forward fill ONLY: do not obtain a current price from a future retail week.
    daily["sell_price"] = daily.groupby(["item_id", "store_id"], sort=False)["sell_price"].ffill()
    return daily


def create_features(daily):
    group = daily.groupby(["item_id", "store_id"], sort=False)["sales"]
    # At a cutoff t the last observed day t is already available to rolling
    # statistics. These conventional lag names refer to t-1, t-7, t-28.
    for lag in (1, 7, 28):
        daily[f"sales_lag_{lag}"] = group.shift(lag)
    for window in (7, 28):
        daily[f"sales_mean_{window}"] = group.transform(
            lambda x: x.rolling(window, min_periods=window).mean()
        )
    daily["sales_std_28"] = group.transform(
        lambda x: x.rolling(28, min_periods=28).std(ddof=0)
    )
    daily["sales_sum_7"] = group.transform(
        lambda x: x.rolling(7, min_periods=7).sum()
    )
    daily["sales_trend"] = daily["sales_mean_7"] - daily["sales_mean_28"]
    return daily


def create_target(daily):
    group = daily.groupby(["item_id", "store_id"], sort=False)["sales"]
    # Exactly the next seven observed-sales days; sales at cutoff t excluded.
    daily["target_7d"] = sum(group.shift(-h) for h in range(1, HORIZON + 1))
    daily = daily.loc[
        daily["d_num"].between(PANEL_FIRST_CUTOFF, LAST_DAY - HORIZON)
    ].copy()
    feature_cols = [f"sales_lag_{k}" for k in (1, 7, 28)] + [
        "sales_mean_7", "sales_mean_28", "sales_std_28", "sales_sum_7",
        "sales_trend", "target_7d",
    ]
    if daily[feature_cols].isna().any().any():
        raise ValueError("Missing rolling features or incomplete target in prepared panel")
    expected_days = LAST_DAY - HORIZON - PANEL_FIRST_CUTOFF + 1
    counts = daily.groupby(["item_id", "store_id"])["d_num"].nunique()
    if not counts.eq(expected_days).all():
        raise ValueError("An item-store time series has missing panel dates")
    return daily


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    daily = create_target(create_features(add_prices(add_calendar(reshape(load_sales())))))
    daily = daily.drop(columns=["d", "wm_yr_wk"])
    out = PROCESSED / "m5_panel.parquet"
    daily.to_parquet(out, index=False)
    print(f"Saved {len(daily):,} rows; {daily.groupby(['item_id','store_id']).ngroups} nodes")
    print(f"Panel cutoff range: d_{daily.d_num.min()}–d_{daily.d_num.max()}")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
