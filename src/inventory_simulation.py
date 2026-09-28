"""Provisional seven-day stockout probabilities for the Streamlit MVP.

The forecast is observed SALES from M5 (not necessarily unconstrained demand).
Inventory is simulated, and alpha is an assumed, uncalibrated dispersion.
No orders arrive inside the seven-day horizon in this first version.
"""
import numpy as np
import pandas as pd
from scipy.stats import nbinom

DEFAULT_ALPHA = 0.3


def risk_from_forecasts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "alpha" not in out:
        out["alpha"] = DEFAULT_ALPHA
    required = ["forecast_7d", "inventory", "unit_price", "alpha"]
    if out[required].isna().any().any():
        raise ValueError("Missing risk inputs: forecast, inventory, price or alpha")
    mu = out.forecast_7d.to_numpy(dtype=float)
    stock = out.inventory.to_numpy(dtype=float)
    alpha = out.alpha.to_numpy(dtype=float)
    price = out.unit_price.to_numpy(dtype=float)
    if np.any(~np.isfinite(mu)) or np.any(~np.isfinite(stock)) or np.any(~np.isfinite(alpha)):
        raise ValueError("Non-finite risk inputs")
    if np.any(mu < 0) or np.any(stock < 0) or np.any(alpha <= 0) or np.any(price < 0):
        raise ValueError("Demand/stock/price must be >= 0 and alpha must be > 0")
    if np.any(stock != np.floor(stock)):
        raise ValueError("Inventory units must be whole numbers")
    r = 1 / alpha
    p = r / (r + mu)
    # Strict exceedance: exactly exhausting stock is not an unfulfilled sale.
    out["risk"] = nbinom.sf(stock, r, p)
    # Exact E[max(D - stock, 0)] for this negative-binomial model.
    shortage = mu * nbinom.sf(stock - 1, r + 1, p) - stock * nbinom.sf(stock, r, p)
    out["expected_shortfall"] = np.maximum(shortage, 0.0)
    out["gross_sales_at_risk"] = out.expected_shortfall * price
    return out


def single_product_risk(mean_7d: float, inventory: int, alpha: float = DEFAULT_ALPHA) -> float:
    frame = pd.DataFrame({
        "forecast_7d": [mean_7d], "inventory": [inventory],
        "unit_price": [0.0], "alpha": [alpha],
    })
    return float(risk_from_forecasts(frame).risk.iloc[0])
