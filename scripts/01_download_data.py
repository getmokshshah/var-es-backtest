"""Download adjusted closes for the ten ETFs once and cache them under data/."""
from riskmodels.data import WEIGHTS, asset_returns, load_prices, portfolio_returns

if __name__ == "__main__":
    px = load_prices()
    rets = asset_returns(px)
    port = portfolio_returns(rets)
    print(f"prices: {px.shape[0]} rows, {px.index[0].date()} to {px.index[-1].date()}")
    print("first valid price per ticker:")
    for t in WEIGHTS:
        print(f"  {t}: {px[t].first_valid_index().date()}")
    print(f"common-sample returns: {len(rets)} days, {rets.index[0].date()} to {rets.index[-1].date()}")
    print(f"portfolio: mean {port.mean() * 252:.2%} p.a., vol {port.std() * 252 ** 0.5:.2%} p.a., "
          f"worst day {port.min():.2%} on {port.idxmin().date()}")
