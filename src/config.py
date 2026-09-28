"""Shared, reproducible experiment configuration for the retail risk MVP."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
GRAPH_DIR = PROCESSED / "graph"
OUTPUTS = ROOT / "outputs"
MODELS = ROOT / "models"

# Distinct items per broad M5 category. Sample each category separately,
# then keep each chosen item across all selected stores.
CATEGORIES = ("FOODS", "HOUSEHOLD", "HOBBIES")
STORES = ("CA_1", "CA_2", "TX_1")
N_PRODUCTS_PER_CATEGORY = 200
EXPECTED_NODES = len(CATEGORIES) * N_PRODUCTS_PER_CATEGORY * len(STORES)
SEED = 42
FIRST_DAY = 1342
LAST_DAY = 1941
HORIZON = 7
MAX_LAG = 28
GRAPH_REFERENCE_DAYS = 180
TRAIN_LAST_CUTOFF = 1839
VALID_FIRST_CUTOFF = 1847
VALID_LAST_CUTOFF = 1891
TEST_FIRST_CUTOFF = 1899
TEST_LAST_CUTOFF = 1934

# Panels begin at FIRST_DAY + MAX_LAG because lag_28 is sales at t-28.
# For a leakage-free fixed graph, build similarity edges from the first
# GRAPH_REFERENCE_DAYS rows of this panel, then train *after* that period.
PANEL_FIRST_CUTOFF = FIRST_DAY + MAX_LAG
GRAPH_REFERENCE_LAST = PANEL_FIRST_CUTOFF + GRAPH_REFERENCE_DAYS - 1
TRAIN_FIRST_CUTOFF = GRAPH_REFERENCE_LAST + 1

NODE_COLUMNS = ["item_id", "store_id", "dept_id", "cat_id", "state_id"]
CATEGORICAL = ["item_id", "store_id"]
FEATURES = [
    "item_id", "store_id", "sales_lag_1", "sales_lag_7",
    "sales_lag_28", "sales_mean_7", "sales_mean_28",
    "sales_std_28", "sales_sum_7", "sales_trend", "sell_price",
    "wday", "month", "is_event", "snap",
]

if not (TRAIN_FIRST_CUTOFF <= TRAIN_LAST_CUTOFF):
    raise ValueError("Graph reference period leaves no training cutoffs")
if not (TRAIN_LAST_CUTOFF + HORIZON < VALID_FIRST_CUTOFF):
    raise ValueError("Training targets overlap validation cutoffs")
if not (VALID_LAST_CUTOFF + HORIZON < TEST_FIRST_CUTOFF):
    raise ValueError("Validation targets overlap test cutoffs")
if not (TEST_LAST_CUTOFF + HORIZON <= LAST_DAY):
    raise ValueError("Test labels extend beyond available sales")
