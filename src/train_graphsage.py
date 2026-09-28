"""Train a full-graph, seven-day-demand GraphSAGE on the existing M5 panel.

Run from the project root, after prepare_data.py, build_graph.py, and
train_model.py:
    python src/train_graphsage.py

Writes a separate GNN backtest and app-compatible dashboard export. It does NOT
replace the LightGBM model or outputs/predictions.csv.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv

from config import (
    PROCESSED, GRAPH_DIR, OUTPUTS, MODELS, FEATURES, HORIZON, SEED,
    GRAPH_REFERENCE_LAST, TRAIN_FIRST_CUTOFF, TRAIN_LAST_CUTOFF,
    VALID_FIRST_CUTOFF, VALID_LAST_CUTOFF, TEST_FIRST_CUTOFF,
    TEST_LAST_CUTOFF, EXPECTED_NODES,
)

NUMERIC = [name for name in FEATURES if name not in ("item_id", "store_id")]
NONNEGATIVE = {
    "sales_lag_1", "sales_lag_7", "sales_lag_28", "sales_mean_7",
    "sales_mean_28", "sales_std_28", "sales_sum_7", "sell_price",
}
PAST_KEYS = ["item_id", "store_id", "d_num"]
BATCH_SIZE = 8
MAX_EPOCHS = 80
PATIENCE = 12
LR = 1e-3
WEIGHT_DECAY = 1e-4


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    a = np.asarray(actual, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    denominator = a.sum()
    return {
        "MAE": float(np.mean(np.abs(a - p))),
        "WAPE": float(np.abs(a - p).sum() / denominator) if denominator > 0 else float("nan"),
        "Bias": float((p - a).sum() / denominator) if denominator > 0 else float("nan"),
    }


def load_aligned_panel() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Rows are strictly sorted as (cutoff day, graph node_id)."""
    nodes_path = GRAPH_DIR / "nodes.csv"
    graph_path = GRAPH_DIR / "graph.pt"
    panel_path = PROCESSED / "m5_panel.parquet"
    for path in (nodes_path, graph_path, panel_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run the earlier milestones first.")

    nodes = pd.read_csv(nodes_path).sort_values("node_id").reset_index(drop=True)
    n = len(nodes)
    if n != EXPECTED_NODES:
        raise ValueError(f"Expected {EXPECTED_NODES} nodes; found {n}. Rebuild the dataset and graph")
    if not np.array_equal(nodes.node_id.to_numpy(), np.arange(n)):
        raise ValueError("nodes.csv requires contiguous node_ids from 0 to N-1")
    if nodes.duplicated(["item_id", "store_id"]).any():
        raise ValueError("Duplicate nodes in nodes.csv")

    graph = torch.load(graph_path, map_location="cpu", weights_only=True)
    if (graph["num_nodes"] != n
            or graph["reference_end"] != GRAPH_REFERENCE_LAST
            or graph["first_training_cutoff"] != TRAIN_FIRST_CUTOFF):
        raise ValueError("Graph metadata does not match nodes.csv/config.py; rebuild graph")
    edge_index = graph["edge_index"]
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("Invalid edge_index shape")
    if edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= n):
        raise ValueError("Graph contains invalid node indices")

    needed = ["item_id", "store_id", "d_num", "target_7d", "date", "cat_id", *NUMERIC]
    panel = pd.read_parquet(panel_path, columns=list(dict.fromkeys(needed)))
    panel = panel.merge(
        nodes[["item_id", "store_id", "node_id"]],
        on=["item_id", "store_id"], how="inner", validate="many_to_one",
    ).sort_values(["d_num", "node_id"]).reset_index(drop=True)
    if panel.duplicated(["d_num", "node_id"]).any():
        raise ValueError("Duplicate graph node observations at a cutoff")
    days = np.sort(panel.d_num.unique())
    if not np.array_equal(days, np.arange(days.min(), days.max() + 1)):
        raise ValueError("Missing days in processed panel")
    if len(panel) != len(days) * n:
        raise ValueError("At least one graph snapshot has missing SKU-store rows")
    if not np.array_equal(panel.node_id.to_numpy().reshape(-1, n),
                          np.broadcast_to(np.arange(n), (len(days), n))):
        raise ValueError("Node order differs between historical snapshots")
    if panel["target_7d"].isna().any():
        raise ValueError("Missing future sales targets")
    return panel, nodes, graph


def prepare_arrays(panel: pd.DataFrame, nodes: pd.DataFrame) -> tuple:
    """Fit preprocessing ONLY on the train partition; return daily tensors."""
    n = len(nodes)
    days = np.sort(panel.d_num.unique())
    raw = panel[NUMERIC].to_numpy(dtype=np.float64).reshape(len(days), n, -1)
    targets = panel.target_7d.to_numpy(dtype=np.float32).reshape(len(days), n)
    train_mask = (days >= TRAIN_FIRST_CUTOFF) & (days <= TRAIN_LAST_CUTOFF)
    if not train_mask.any():
        raise ValueError("No training snapshots in the specified range")
    raw_train = raw[train_mask]
    price_idx = NUMERIC.index("sell_price")
    known_prices = raw_train[:, :, price_idx]
    known_prices = known_prices[np.isfinite(known_prices)]
    if len(known_prices) == 0:
        raise ValueError("No historical selling prices in the training partition")
    price_fill = float(np.median(known_prices))
    missing_price = ~np.isfinite(raw[:, :, price_idx])
    raw[:, :, price_idx][missing_price] = price_fill
    if not np.isfinite(raw).all():
        raise ValueError("Non-price feature is missing or non-finite")

    # Log-transform right-skewed nonnegative features; preserve signed trends.
    for j, name in enumerate(NUMERIC):
        if name in NONNEGATIVE:
            raw[:, :, j] = np.log1p(np.clip(raw[:, :, j], 0, None))
        elif name == "sales_trend":
            raw[:, :, j] = np.arcsinh(raw[:, :, j])
    mean = raw[train_mask].mean(axis=(0, 1))
    std = raw[train_mask].std(axis=(0, 1))
    std = np.maximum(std, 1e-6)
    x = ((raw - mean) / std).astype(np.float32)

    # Loss is computed on scaled demand to keep its numerical magnitude stable.
    target_scale = max(1.0, float(targets[train_mask].mean()))

    item_names = sorted(nodes.item_id.unique().tolist())
    store_names = sorted(nodes.store_id.unique().tolist())
    item_map = {name: i for i, name in enumerate(item_names)}
    store_map = {name: i for i, name in enumerate(store_names)}
    item_idx = torch.tensor(nodes.item_id.map(item_map).to_numpy(), dtype=torch.long)
    store_idx = torch.tensor(nodes.store_id.map(store_map).to_numpy(), dtype=torch.long)
    preprocessing = {
        "numeric_features": NUMERIC,
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "missing_price_fill": price_fill,
        "target_scale": target_scale,
        "item_names": item_names,
        "store_names": store_names,
    }
    return days, x, targets, item_idx, store_idx, preprocessing


def make_snapshots(days, x, y, item_idx, store_idx, edge_index):
    """Each Data object is a separate full graph; batching keeps graphs disjoint."""
    snapshots = []
    n = len(item_idx)
    for k, day in enumerate(days):
        snapshots.append(Data(
            x=torch.from_numpy(x[k]),
            y=torch.from_numpy(y[k]),
            item_idx=item_idx.clone(),
            store_idx=store_idx.clone(),
            edge_index=edge_index,
            day=torch.full((n,), int(day), dtype=torch.long),
            node_id=torch.arange(n, dtype=torch.long),
            num_nodes=n,
        ))
    return snapshots


class DemandGraphSAGE(nn.Module):
    def __init__(self, num_features: int, num_items: int, num_stores: int):
        super().__init__()
        self.item_embed = nn.Embedding(num_items, 8)
        self.store_embed = nn.Embedding(num_stores, 4)
        input_dim = num_features + 8 + 4
        self.conv1 = SAGEConv(input_dim, 64, aggr="mean")
        self.conv2 = SAGEConv(64, 32, aggr="mean")
        self.dropout = 0.15
        self.head = nn.Linear(32, 1)

    def forward(self, batch: Batch) -> torch.Tensor:
        h = torch.cat((
            batch.x,
            self.item_embed(batch.item_idx),
            self.store_embed(batch.store_idx),
        ), dim=-1)
        h = F.dropout(F.relu(self.conv1(h, batch.edge_index)),
                      p=self.dropout, training=self.training)
        h = F.dropout(F.relu(self.conv2(h, batch.edge_index)),
                      p=self.dropout, training=self.training)
        # Units are normalized by a training-only scale outside this module.
        return F.softplus(self.head(h).squeeze(-1))


def evaluate(model, loader, device, target_scale: float, collect=False):
    model.eval()
    total_absolute = 0.0
    total_actual = 0.0
    total_predicted = 0.0
    observations = 0
    frames = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predicted = model(batch).mul(target_scale).cpu().numpy()
            actual = batch.y.cpu().numpy()
            total_absolute += float(np.abs(predicted - actual).sum())
            total_actual += float(actual.sum())
            total_predicted += float(predicted.sum())
            observations += len(actual)
            if collect:
                frames.append(pd.DataFrame({
                    "d_num": batch.day.cpu().numpy(),
                    "node_id": batch.node_id.cpu().numpy(),
                    "target_7d": actual,
                    "pred_graphsage": predicted,
                }))
    result = {
        "MAE": total_absolute / observations,
        "WAPE": total_absolute / total_actual if total_actual > 0 else float("nan"),
        "Bias": (total_predicted - total_actual) / total_actual if total_actual > 0 else float("nan"),
    }
    return result, (pd.concat(frames, ignore_index=True) if collect else None)


def export_results(predictions: pd.DataFrame, nodes: pd.DataFrame) -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    merged = predictions.merge(
        nodes[["node_id", "item_id", "store_id", "cat_id"]],
        on="node_id", validate="many_to_one",
    ).drop(columns="node_id")
    merged = merged.sort_values(["d_num", "store_id", "item_id"]).reset_index(drop=True)
    merged.to_csv(OUTPUTS / "graphsage_backtest_predictions.csv", index=False)

    baseline_path = OUTPUTS / "backtest_predictions.csv"
    if not baseline_path.exists():
        print("LightGBM backtest missing; GraphSAGE forecasts saved separately.")
        return
    baseline = pd.read_csv(baseline_path)
    if baseline.duplicated(PAST_KEYS).any() or merged.duplicated(PAST_KEYS).any():
        raise ValueError("Duplicate baseline or GraphSAGE forecast keys")
    added = merged[PAST_KEYS + ["target_7d", "pred_graphsage"]]
    combined = baseline.merge(added, on=PAST_KEYS, how="left",
                              validate="one_to_one", suffixes=("", "_gnn"))
    if combined.pred_graphsage.isna().any() or len(combined) != len(baseline):
        raise ValueError("Baseline and GraphSAGE forecasting keys do not match")
    if not np.allclose(combined.target_7d, combined.target_7d_gnn, atol=1e-4):
        raise ValueError("LightGBM and GraphSAGE use different target labels")
    combined = combined.drop(columns="target_7d_gnn")
    combined.to_csv(OUTPUTS / "backtest_predictions_with_graphsage.csv", index=False)

    # Keep the original LightGBM dashboard unchanged; export an ALTERNATIVE
    # dashboard file with the exact same simulated inventory and prices.
    app_path = OUTPUTS / "predictions.csv"
    if app_path.exists():
        app = pd.read_csv(app_path)
        last = merged.loc[merged.d_num.eq(TEST_LAST_CUTOFF),
                          ["item_id", "store_id", "pred_graphsage"]]
        app = app.merge(last, left_on=["sku", "store"],
                        right_on=["item_id", "store_id"],
                        how="left", validate="one_to_one")
        if app.pred_graphsage.isna().any():
            raise ValueError("Dashboard and GraphSAGE product-store keys do not match")
        app["forecast_7d"] = app.pop("pred_graphsage")
        app["forecast_model"] = "GraphSAGE"
        app = app.drop(columns=["item_id", "store_id"])
        app.to_csv(OUTPUTS / "predictions_graphsage.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    print(f"Device: {device}")
    panel, nodes, graph = load_aligned_panel()
    days, x, y, item_idx, store_idx, prep = prepare_arrays(panel, nodes)
    snapshots = make_snapshots(days, x, y, item_idx, store_idx, graph["edge_index"])

    def select(start, end):
        return [snapshot for snapshot, day in zip(snapshots, days) if start <= day <= end]

    train = select(TRAIN_FIRST_CUTOFF, TRAIN_LAST_CUTOFF)
    valid = select(VALID_FIRST_CUTOFF, VALID_LAST_CUTOFF)
    test = select(TEST_FIRST_CUTOFF, TEST_LAST_CUTOFF)
    if [len(train), len(valid), len(test)] != [290, 45, 36]:
        raise ValueError("Unexpected train/validation/test snapshot counts")
    print(f"Nodes: {len(nodes)} | train: {len(train)} | valid: {len(valid)} | test: {len(test)}")
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = DemandGraphSAGE(x.shape[-1], len(prep["item_names"]),
                            len(prep["store_names"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_mae = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, count = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred_scaled = model(batch)
            loss = F.smooth_l1_loss(pred_scaled, batch.y / prep["target_scale"], beta=0.5)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_sum += float(loss.item()) * batch.num_nodes
            count += batch.num_nodes
        validation, _ = evaluate(model, valid_loader, device, prep["target_scale"])
        history.append({"epoch": epoch, "train_loss": loss_sum / count,
                        "val_MAE": validation["MAE"], "val_WAPE": validation["WAPE"],
                        "val_Bias": validation["Bias"]})
        print(f"Epoch {epoch:03d} | train loss {loss_sum/count:.4f} "
              f"| val MAE {validation['MAE']:.4f} | val WAPE {validation['WAPE']:.2%}")
        if validation["MAE"] < best_mae - 1e-4:
            best_mae, best_epoch = validation["MAE"], epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"Early stopping after {epoch} epochs")
                break

    model.load_state_dict(best_state)
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": best_state,
        "preprocessing": prep,
        "architecture": "two_layer_graphsage",
        "num_features": x.shape[-1],
        "best_epoch": best_epoch,
        "best_validation_MAE": best_mae,
        "graph_reference_end": GRAPH_REFERENCE_LAST,
    }, MODELS / "graphsage_7d.pt")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(OUTPUTS / "graphsage_training_history.csv", index=False)
    final, predictions = evaluate(model, test_loader, device, prep["target_scale"], collect=True)
    pd.DataFrame([{"model": "GraphSAGE", **final}]).to_csv(
        OUTPUTS / "graphsage_metrics.csv", index=False)
    export_results(predictions, nodes)
    print(f"Best validation epoch: {best_epoch} | best validation MAE: {best_mae:.4f}")
    print(f"EXPLORATORY test: MAE={final['MAE']:.4f} WAPE={final['WAPE']:.2%} Bias={final['Bias']:.2%}")
    print(f"Model: {MODELS / 'graphsage_7d.pt'}")
    print(f"Backtest: {OUTPUTS / 'backtest_predictions_with_graphsage.csv'}")
    print("Existing LightGBM dashboard predictions were NOT overwritten.")


if __name__ == "__main__":
    main()
