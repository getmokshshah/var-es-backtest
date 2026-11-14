from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def real_returns() -> np.ndarray:
    """First 2000 daily returns of the project portfolio (Nov 2004 to Oct 2012), a fixed snapshot."""
    path = Path(__file__).parent / "data" / "portfolio_returns.csv"
    return pd.read_csv(path, index_col=0)["portfolio"].to_numpy()
