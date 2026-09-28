"""Controlled graph ablations for the existing seven-day M5 GraphSAGE model.

Run from project root AFTER prepare_data.py and build_graph.py.

Training/validation ONLY (does not inspect test labels):
    python src/run_graph_ablation.py --phase train --seeds 42 43 44

Final exploratory test of frozen, validation-selected checkpoints:
    python src/run_graph_ablation.py --phase test --seeds 42 43 44

No existing model, dashboard, or baseline forecast files are overwritten.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.loader import DataLoader

from config import (
    GRAPH_DIR, MODELS, OUTPUTS, TRAIN_FIRST_CUTOFF, TRAIN_LAST_CUTOFF,
    VALID_FIRST_CUTOFF, VALID_LAST_CUTOFF, TEST_FIRST_CUTOFF, TEST_LAST_CUTOFF,
)
from train_graphsage import (
    DemandGraphSAGE, evaluate, load_aligned_panel, make_snapshots,
    prepare_arrays, seed_everything,
)

CONFIGS = ("node_only", "cross_store", "product_similarity", "full")
EDGE_TYPES = {"cross_store": 0, "product_similarity": 1}
DEFAULT_SEEDS = (42, 43, 44)


def graph_fingerprint(graph: dict) -> str:
    """Detect changes to graph connectivity between training and final testing."""
    digest = hashlib.sha256()
    for name in ("edge_index", "edge_type"):
        tensor = graph[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def choose_edges(graph: dict, config: str) -> torch.Tensor:
    """Keep the source graph immutable; preserve its node indexing and directions."""
    if config not in CONFIGS:
        raise ValueError(f"Unknown ablation: {config}")
    original = graph["edge_index"]
    types = graph["edge_type"]
    if original.ndim != 2 or original.shape[0] != 2:
        raise ValueError("Expected edge_index with shape (2, num_edges)")
    if types.ndim != 1 or original.shape[1] != types.numel():
        raise ValueError("edge_type must have one entry for every directed edge")
    if not torch.isin(types, torch.tensor([0, 1], dtype=types.dtype)).all():
        raise ValueError("Graph has an unexpected edge type")

    if config == "node_only":
        return torch.empty((2, 0), dtype=torch.long)
    if config == "full":
        result = original.clone()
    else:
        result = original[:, types == EDGE_TYPES[config]].clone()

    if result.size(1) == 0:
        raise ValueError(f"No edges for configuration {config}")
    return result


def node_fingerprint(nodes: pd.DataFrame) -> str:
    ordered = nodes.sort_values("node_id")[["node_id", "item_id", "store_id"]]
    payload = ordered.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def check_edge_semantics(graph: dict, nodes: pd.DataFrame) -> None:
    """Validate that both stored edge classes mean what the study claims."""
    index = graph["edge_index"].cpu().numpy()
    kinds = graph["edge_type"].cpu().numpy()
    items = nodes.sort_values("node_id")["item_id"].to_numpy()
    stores = nodes.sort_values("node_id")["store_id"].to_numpy()
    categories = nodes.sort_values("node_id")["cat_id"].to_numpy()
    n = len(nodes)
    if index.size == 0 or index.min() < 0 or index.max() >= n:
        raise ValueError("Invalid source graph indices")
    src, dst = index
    if np.any(src == dst):
        raise ValueError("Stored graph unexpectedly has self-loops")
    cross = kinds == 0
    similar = kinds == 1
    if not cross.any() or not similar.any():
        raise ValueError("Full graph must contain both relationship types")
    if not np.all((items[src[cross]] == items[dst[cross]]) &
                  (stores[src[cross]] != stores[dst[cross]])):
        raise ValueError("Invalid cross-store edge semantics")
    if not np.all((items[src[similar]] != items[dst[similar]]) &
                  (stores[src[similar]] == stores[dst[similar]]) &
                  (categories[src[similar]] == categories[dst[similar]])):
        raise ValueError("Invalid product similarity edge semantics")
    pairs = set(zip(src.tolist(), dst.tolist(), kinds.tolist()))
    if any((v, u, k) not in pairs for u, v, k in pairs):
        raise ValueError("Graph is not symmetrically directed")


def date_loader(snapshots, days, first, last, batch_size, shuffle, seed):
    selected = [s for s, d in zip(snapshots, days) if first <= int(d) <= last]
    expected = last - first + 1
    if len(selected) != expected:
        raise ValueError(f"Expected {expected} snapshots for {first}..{last}, "
                         f"got {len(selected)}")
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(selected, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, generator=generator)


def check_prediction_keys(frame: pd.DataFrame, expected_n: int, first: int, last: int):
    required = ["node_id", "d_num", "target_7d", "pred_graphsage"]
    if frame[required].isna().any().any():
        raise ValueError("Missing ablation predictions")
    if frame.duplicated(["d_num", "node_id"]).any():
        raise ValueError("Duplicate prediction keys")
    if len(frame) != (last - first + 1) * expected_n:
        raise ValueError("Incomplete graph snapshot predictions")


def training_run(args, config, seed, days, x, y, item_idx, store_idx,
                 prep, graph, nodes, output_dir: Path, model_dir: Path, device):
    seed_everything(seed)
    # Each run gets exactly the same node features and labels, changing only edges.
    edges = choose_edges(graph, config)
    snapshots = make_snapshots(days, x, y, item_idx, store_idx, edges)
    train_loader = date_loader(snapshots, days, TRAIN_FIRST_CUTOFF,
                               TRAIN_LAST_CUTOFF, args.batch_size, True, seed)
    val_loader = date_loader(snapshots, days, VALID_FIRST_CUTOFF,
                             VALID_LAST_CUTOFF, args.batch_size, False, seed)
    model = DemandGraphSAGE(x.shape[-1], len(prep["item_names"]),
                            len(prep["store_names"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_mae, best_epoch, best_state, stale = float("inf"), 0, None, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        weighted_loss, count = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            estimate = model(batch)
            loss = F.smooth_l1_loss(
                estimate, batch.y / prep["target_scale"], beta=0.5
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            weighted_loss += float(loss.item()) * batch.num_nodes
            count += batch.num_nodes
        validation, _ = evaluate(model, val_loader, device, prep["target_scale"])
        history.append({"epoch": epoch, "train_loss": weighted_loss / count,
                        "val_MAE": validation["MAE"],
                        "val_WAPE_pct": validation["WAPE"] * 100,
                        "val_Bias_pct": validation["Bias"] * 100})
        print(f"{config:18} seed={seed:3} epoch={epoch:03} "
              f"loss={weighted_loss/count:.4f} val_MAE={validation['MAE']:.4f}",
              flush=True)
        if validation["MAE"] < best_mae - 1e-4:
            best_mae, best_epoch = validation["MAE"], epoch
            best_state, stale = copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("No valid model checkpoint")
    model.load_state_dict(best_state)
    final_val, val_predictions = evaluate(
        model, val_loader, device, prep["target_scale"], collect=True
    )
    check_prediction_keys(val_predictions, len(nodes),
                          VALID_FIRST_CUTOFF, VALID_LAST_CUTOFF)
    val_predictions.to_csv(output_dir / f"validation_{config}_seed{seed}.csv",
                           index=False)
    pd.DataFrame(history).to_csv(output_dir / f"history_{config}_seed{seed}.csv",
                                 index=False)
    checkpoint = {
        "model_state_dict": best_state,
        "preprocessing": prep,
        "architecture": "two_layer_graphsage",
        "config": config,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_mae": best_mae,
        "graph_fingerprint": graph_fingerprint(graph),
        "node_fingerprint": node_fingerprint(nodes),
        "num_nodes": len(nodes),
        "num_features": x.shape[-1],
    }
    torch.save(checkpoint, model_dir / f"{config}_seed{seed}.pt")
    del model, snapshots, train_loader, val_loader
    gc.collect()
    return {"config": config, "seed": seed, "best_epoch": best_epoch,
            "MAE": final_val["MAE"], "WAPE_pct": final_val["WAPE"] * 100,
            "Bias_pct": final_val["Bias"] * 100}


def test_run(args, config, seed, days, x, y, item_idx, store_idx,
             prep, graph, nodes, output_dir: Path, model_dir: Path, device):
    ckpt_path = model_dir / f"{config}_seed{seed}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing validation-selected checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if (ckpt["config"] != config or int(ckpt["seed"]) != seed
            or ckpt["graph_fingerprint"] != graph_fingerprint(graph)
            or ckpt["node_fingerprint"] != node_fingerprint(nodes)
            or ckpt["num_nodes"] != len(nodes)
            or ckpt["num_features"] != x.shape[-1]
            or ckpt["preprocessing"] != prep):
        raise ValueError("Checkpoint no longer matches current dataset or graph")
    edges = choose_edges(graph, config)
    snapshots = make_snapshots(days, x, y, item_idx, store_idx, edges)
    test_loader = date_loader(snapshots, days, TEST_FIRST_CUTOFF,
                              TEST_LAST_CUTOFF, args.batch_size, False, seed)
    model = DemandGraphSAGE(x.shape[-1], len(prep["item_names"]),
                            len(prep["store_names"])).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics, predictions = evaluate(
        model, test_loader, device, prep["target_scale"], collect=True
    )
    check_prediction_keys(predictions, len(nodes),
                          TEST_FIRST_CUTOFF, TEST_LAST_CUTOFF)
    predictions.to_csv(output_dir / f"test_{config}_seed{seed}.csv", index=False)
    del snapshots, test_loader, model
    gc.collect()
    return {"config": config, "seed": seed, "best_epoch": ckpt["best_epoch"],
            "MAE": test_metrics["MAE"], "WAPE_pct": test_metrics["WAPE"] * 100,
            "Bias_pct": test_metrics["Bias"] * 100}


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    return (runs.groupby("config", sort=False)
            .agg(n_seeds=("seed", "nunique"),
                 MAE_mean=("MAE", "mean"), MAE_sd=("MAE", "std"),
                 WAPE_pct_mean=("WAPE_pct", "mean"),
                 Bias_pct_mean=("Bias_pct", "mean"))
            .reset_index())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("train", "test"))
    parser.add_argument("--configs", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS / "ablation")
    parser.add_argument("--model-dir", type=Path, default=MODELS / "ablation")
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size) < 1:
        parser.error("--epochs, --patience, and --batch-size must be positive")
    if len(set(args.configs)) != len(args.configs) or len(set(args.seeds)) != len(args.seeds):
        parser.error("--configs and --seeds must not contain duplicates")
    if args.phase == "test":
        validation_path = args.output_dir / "train_runs.csv"
        if not validation_path.exists():
            parser.error("Missing train_runs.csv; complete validation before testing")
        completed = pd.read_csv(validation_path)
        existing = set(zip(completed["config"], completed["seed"]))
        requested = {(cfg, seed) for cfg in args.configs for seed in args.seeds}
        if not requested.issubset(existing):
            parser.error(f"Test requested before validation completed for: {requested - existing}")
    if args.phase == "train" and args.output_dir.joinpath("test_runs.csv").exists():
        parser.error("Test outputs already exist in this output directory; "
                     "do not tune after inspecting them. Use a new directory.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    print(f"Device: {device} | phase={args.phase}", flush=True)
    panel, nodes, graph = load_aligned_panel()
    check_edge_semantics(graph, nodes)
    days, x, y, item_idx, store_idx, prep = prepare_arrays(panel, nodes)
    rows = []
    for config in args.configs:
        print(f"\nCONFIGURATION: {config}, edges={choose_edges(graph, config).shape[1]}",
              flush=True)
        for seed in args.seeds:
            if args.phase == "train":
                result = training_run(args, config, seed, days, x, y,
                                      item_idx, store_idx, prep, graph, nodes,
                                      args.output_dir, args.model_dir, device)
            else:
                result = test_run(args, config, seed, days, x, y,
                                  item_idx, store_idx, prep, graph, nodes,
                                  args.output_dir, args.model_dir, device)
            rows.append(result)
            print("FINISHED:", result, flush=True)
            # Preserve completed work if a later experiment fails.
            pd.DataFrame(rows).to_csv(args.output_dir / f"{args.phase}_runs.csv",
                                      index=False)
    all_runs = pd.DataFrame(rows)
    summary = summarize(all_runs)
    summary.to_csv(args.output_dir / f"{args.phase}_summary.csv", index=False)
    print("\nSUMMARY (descriptive means over seeds; no significance claim):")
    print(summary.round(4).to_string(index=False))
    if args.phase == "train":
        print("\nSelect architectures/settings using VALIDATION only. "
              "Run --phase test once the protocol is frozen.")
    else:
        print("\nTest results are exploratory because this M5 test period "
              "has already been inspected during prior model development.")


if __name__ == "__main__":
    main()
