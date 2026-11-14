"""Price download, caching, and the fixed-weight portfolio.

Conventions (used everywhere in the package)
--------------------------------------------
* Asset returns are *simple* daily total returns, ``R_{i,t} = P_{i,t}/P_{i,t-1} - 1``,
  from dividend- and split-adjusted closes.
* The book is rebalanced to target weights ``w`` at every close, so the portfolio
  return is exactly linear in asset returns: ``r_t = sum_i w_i R_{i,t}``.
* Loss is positive: ``L_t = -r_t``, a fraction of portfolio value.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PRICES_CSV = DATA_DIR / "prices.csv"

# Target weights, fixed ex ante: a 60/35/5 book with satellites.
WEIGHTS: dict[str, float] = {
    "SPY": 0.25,  # US large cap
    "QQQ": 0.10,  # US growth / tech
    "IWM": 0.05,  # US small cap
    "EFA": 0.10,  # developed ex-US equity
    "EEM": 0.05,  # emerging-market equity
    "VNQ": 0.05,  # US real estate
    "IEF": 0.15,  # 7-10y Treasuries
    "TLT": 0.10,  # 20y+ Treasuries
    "LQD": 0.10,  # investment-grade credit
    "GLD": 0.05,  # gold
}
ASSET_CLASS: dict[str, str] = {
    "SPY": "Equity", "QQQ": "Equity", "IWM": "Equity", "EFA": "Equity", "EEM": "Equity",
    "VNQ": "Real estate", "IEF": "Treasuries", "TLT": "Treasuries",
    "LQD": "IG credit", "GLD": "Gold",
}
START = "2004-01-01"
END = "2026-11-14"  # exclusive; last bar is Friday 13 Nov 2026


def download_prices(tickers: list[str] | None = None, retries: int = 5) -> pd.DataFrame:
    """Adjusted closes from Yahoo Finance in one batched request, with backoff."""
    import yfinance as yf

    tickers = tickers or list(WEIGHTS)
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            raw = yf.download(tickers, start=START, end=END, auto_adjust=True,
                              progress=False, threads=False)
            px = raw["Close"][tickers]
            if px.notna().sum().min() > 1000:
                px.index = pd.DatetimeIndex(px.index).tz_localize(None)
                px.index.name = "date"
                return px
            last_err = RuntimeError("incomplete download")
        except Exception as err:  # network errors are heterogeneous
            last_err = err
        time.sleep(5 * 2 ** attempt)
    raise RuntimeError(f"download failed after {retries} attempts: {last_err}")


def load_prices(refresh: bool = False) -> pd.DataFrame:
    """Cached adjusted closes; downloads on first use only."""
    if PRICES_CSV.exists() and not refresh:
        return pd.read_csv(PRICES_CSV, index_col="date", parse_dates=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    px = download_prices()
    px.to_csv(PRICES_CSV, float_format="%.6f")
    return px


def asset_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns from the first date on which every asset has a price."""
    px = prices.dropna(how="any")
    return (px / px.shift(1) - 1.0).iloc[1:]


def portfolio_returns(returns: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.Series:
    """Daily-rebalanced portfolio simple return ``r_t = w' R_t``."""
    w = pd.Series(weights or WEIGHTS)
    if abs(w.sum() - 1.0) > 1e-12:
        raise ValueError("weights must sum to one")
    return (returns[w.index] * w).sum(axis=1).rename("portfolio")


def weight_vector(columns: list[str]) -> np.ndarray:
    return np.array([WEIGHTS[c] for c in columns])
