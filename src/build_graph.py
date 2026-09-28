"""Build a fixed retail product-store graph using only early training history.

Run after prepare_data.py, BEFORE train_model.py:
    python src/build_graph.py

The frozen similarity graph covers the first 180 cutoffs of the *processed*
panel. Training starts strictly after its last day; validation/test are later.
"""
import json
from itertools import combinations

import numpy as np
import pandas as pd
import torch

from config import (
    PROCESSED, GRAPH_DIR, GRAPH_REFERENCE_DAYS, PANEL_FIRST_CUTOFF,
    GRAPH_REFERENCE_LAST, TRAIN_FIRST_CUTOFF, NODE_COLUMNS, STORES,
    CATEGORIES, N_PRODUCTS_PER_CATEGORY, EXPECTED_NODES,
)

TOP_K = 5
MIN_CORRELATION = 0.20
CROSS_STORE = 0
PRODUCT_SIMILARITY = 1


def build_similarity_edges(ref: pd.DataFrame, nodes: pd.DataFrame, lookup: dict) -> set:
    """Find top-K correlations only within the SAME category AND store.

    Correlations use the frozen, early-history reference period. Edges from
    Foods to Household/Hobbies are intentionally disallowed in this MVP.
    """
    similar = set()
    for store in sorted(nodes.store_id.unique()):
        for category in CATEGORIES:
            subset = ref.loc[ref.store_id.eq(store) & ref.cat_id.eq(category)]
            matrix = (subset.pivot(index="d_num", columns="item_id", values="sales")
                      .sort_index().sort_index(axis=1))
            if matrix.empty or matrix.isna().any().any():
                raise ValueError(f"Missing reference sales for {category} in {store}")
            corr = matrix.corr(method="pearson", min_periods=GRAPH_REFERENCE_DAYS)
            for item in corr.columns:
                scores = corr[item].drop(index=item)
                scores = scores[scores.notna() & (scores >= MIN_CORRELATION)]
                ordered = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
                for neighbor, _ in ordered[:TOP_K]:
                    a, b = sorted((lookup[(item, store)], lookup[(neighbor, store)]))
                    similar.add((a, b))
    return similar


def main():
    panel_path = PROCESSED / "m5_panel.parquet"
    if not panel_path.exists():
        raise FileNotFoundError(f"Run prepare_data.py first: {panel_path}")
    daily = pd.read_parquet(
        panel_path, columns=NODE_COLUMNS + ["d_num", "sales"]
    )
    if daily[NODE_COLUMNS + ["d_num", "sales"]].isna().any().any():
        raise ValueError("Graph source has missing required values")
    if daily.duplicated(["item_id", "store_id", "d_num"]).any():
        raise ValueError("Duplicate product-store-day observations")

    nodes = (daily[NODE_COLUMNS].drop_duplicates()
             .sort_values(["store_id", "item_id"]).reset_index(drop=True))
    if nodes.duplicated(["item_id", "store_id"]).any():
        raise ValueError("Inconsistent product-store metadata")
    nodes.insert(0, "node_id", np.arange(len(nodes), dtype="int64"))
    if len(nodes) != EXPECTED_NODES:
        raise ValueError(f"Expected {EXPECTED_NODES} nodes, found {len(nodes)}. Re-run prepare_data.py")
    observed = nodes.groupby("cat_id")["item_id"].nunique().to_dict()
    expected = {category: N_PRODUCTS_PER_CATEGORY for category in CATEGORIES}
    if observed != expected:
        raise ValueError(f"Unexpected category composition: {observed}; expected {expected}")
    lookup = {(r.item_id, r.store_id): int(r.node_id) for r in nodes.itertuples(index=False)}

    ref = daily.loc[daily.d_num.between(PANEL_FIRST_CUTOFF, GRAPH_REFERENCE_LAST)]
    expected = np.arange(PANEL_FIRST_CUTOFF, GRAPH_REFERENCE_LAST + 1)
    if not np.array_equal(np.sort(ref.d_num.unique()), expected):
        raise ValueError("Reference-period dates are incomplete")
    counts = ref.groupby(["item_id", "store_id"])["d_num"].nunique()
    if len(counts) != len(nodes) or not counts.eq(GRAPH_REFERENCE_DAYS).all():
        raise ValueError("Some graph nodes lack complete reference history")

    cross = set()
    for _, group in nodes.groupby("item_id"):
        ids = sorted(group.node_id.tolist())
        for a, b in combinations(ids, 2):
            cross.add((a, b))

    similar = build_similarity_edges(ref, nodes, lookup)

    edge_rows = [(a, b, CROSS_STORE) for a, b in sorted(cross)]
    edge_rows += [(a, b, PRODUCT_SIMILARITY) for a, b in sorted(similar)]
    edges = pd.DataFrame(edge_rows, columns=["source", "target", "edge_type"])
    if edges.empty:
        raise ValueError("Graph has no edges")
    reverse = edges.rename(columns={"source": "target", "target": "source"})
    directed = pd.concat([edges, reverse], ignore_index=True).sort_values(
        ["source", "target", "edge_type"]
    ).reset_index(drop=True)
    edge_index = torch.as_tensor(directed[["source", "target"]].to_numpy().T.copy(), dtype=torch.long)
    edge_type = torch.as_tensor(directed.edge_type.to_numpy().copy(), dtype=torch.long)
    if not (edge_index.min() >= 0 and edge_index.max() < len(nodes)):
        raise ValueError("Invalid node indices")
    if torch.any(edge_index[0] == edge_index[1]):
        raise ValueError("Unexpected graph self-loop")
    node_lookup = nodes.set_index("node_id")
    for row in edges.itertuples(index=False):
        a, b = node_lookup.loc[row.source], node_lookup.loc[row.target]
        if row.edge_type == CROSS_STORE:
            assert a.item_id == b.item_id and a.store_id != b.store_id
        else:
            assert (a.item_id != b.item_id and a.store_id == b.store_id
                    and a.cat_id == b.cat_id)

    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    nodes.to_csv(GRAPH_DIR / "nodes.csv", index=False)
    edges.to_csv(GRAPH_DIR / "edges.csv", index=False)
    directed.to_csv(GRAPH_DIR / "directed_edges.csv", index=False)
    torch.save({
        "edge_index": edge_index, "edge_type": edge_type,
        "num_nodes": len(nodes),
        "reference_start": PANEL_FIRST_CUTOFF,
        "reference_end": GRAPH_REFERENCE_LAST,
        "first_training_cutoff": TRAIN_FIRST_CUTOFF,
    }, GRAPH_DIR / "graph.pt")
    meta = {
        "num_nodes": len(nodes),
        "categories": {name: int(n) for name, n in nodes.groupby("cat_id")["item_id"].nunique().items()},
        "stores": list(STORES),
        "cross_store_edges": len(cross),
        "similarity_edges": len(similar), "directed_edges": edge_index.shape[1],
        "reference_start": PANEL_FIRST_CUTOFF,
        "reference_end": GRAPH_REFERENCE_LAST,
        "first_training_cutoff": TRAIN_FIRST_CUTOFF,
        "top_k": TOP_K, "min_correlation": MIN_CORRELATION,
        "edge_types": {"0": "cross_store", "1": "product_similarity"},
    }
    (GRAPH_DIR / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Nodes: {len(nodes)} | cross-store edges: {len(cross)} | similarity edges: {len(similar)}")
    print("Items per category:\n" + nodes.groupby("cat_id")["item_id"].nunique().to_string())
    print(f"Reference d_{PANEL_FIRST_CUTOFF}–d_{GRAPH_REFERENCE_LAST}; first train cutoff d_{TRAIN_FIRST_CUTOFF}")
    print(f"Saved: {GRAPH_DIR}")


if __name__ == "__main__":
    main()
