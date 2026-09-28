"""Tiny synthetic checks for category-balanced M5 sampling and similarity edges."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare_data
import build_graph


class MulticategoryTests(unittest.TestCase):
    def test_balanced_selection_same_items_all_stores(self):
        rows = []
        for category in ("FOODS", "HOUSEHOLD", "HOBBIES"):
            for item_index in range(4):
                item = f"{category}_1_{item_index:03d}"
                for store in ("CA_1", "CA_2", "TX_1"):
                    rows.append({
                        "item_id": item, "dept_id": f"{category}_1",
                        "cat_id": category, "store_id": store,
                        "state_id": store[:2],
                        "d_1342": 1, "d_1343": 2,
                    })
        with tempfile.TemporaryDirectory() as folder:
            pd.DataFrame(rows).to_csv(Path(folder) / "sales_train_evaluation.csv", index=False)
            with (patch.object(prepare_data, "RAW", Path(folder)),
                  patch.object(prepare_data, "DAY_COLUMNS", ["d_1342", "d_1343"]),
                  patch.object(prepare_data, "N_PRODUCTS_PER_CATEGORY", 2),
                  patch.object(prepare_data, "EXPECTED_NODES", 18)):
                chosen = prepare_data.load_sales()
        self.assertEqual(len(chosen), 18)
        self.assertEqual(chosen.groupby("cat_id").item_id.nunique().to_dict(),
                         {"FOODS": 2, "HOUSEHOLD": 2, "HOBBIES": 2})
        self.assertTrue(chosen.groupby("item_id").store_id.nunique().eq(3).all())

    def test_similarity_edges_never_cross_category_or_store(self):
        nodes = []
        ref = []
        seed = np.arange(180, dtype=float)
        for category in ("FOODS", "HOUSEHOLD", "HOBBIES"):
            for store in ("CA_1", "CA_2", "TX_1"):
                for index in range(2):
                    item = f"{category}_1_{index:03d}"
                    nodes.append({"item_id": item, "cat_id": category, "store_id": store})
                    # Exactly correlated products within each category/store.
                    for d, value in enumerate(seed):
                        ref.append({"item_id": item, "cat_id": category,
                                    "store_id": store, "d_num": 1370 + d,
                                    "sales": value + 5 * CATEGORIES.index(category)})
        nodes = pd.DataFrame(nodes).sort_values(["store_id", "item_id"]).reset_index(drop=True)
        nodes.insert(0, "node_id", np.arange(len(nodes)))
        lookup = {(r.item_id, r.store_id): int(r.node_id) for r in nodes.itertuples(index=False)}
        result = build_graph.build_similarity_edges(pd.DataFrame(ref), nodes, lookup)
        self.assertEqual(len(result), 9)
        for a, b in result:
            self.assertEqual(nodes.iloc[a].cat_id, nodes.iloc[b].cat_id)
            self.assertEqual(nodes.iloc[a].store_id, nodes.iloc[b].store_id)
            self.assertNotEqual(nodes.iloc[a].item_id, nodes.iloc[b].item_id)


CATEGORIES = ("FOODS", "HOUSEHOLD", "HOBBIES")

if __name__ == "__main__":
    unittest.main()
