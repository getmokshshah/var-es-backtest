"""Risk decomposition, stress windows and full-sample residual diagnostics.

Reads data/forecasts.csv from script 02. Writes sections "decomposition", "stress"
and "diagnostics" of results/summary.json, results/web/risk_contrib.json and
data/diagnostics.npz (inputs of the QQ and tail figures).
"""
import importlib
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import kurtosis, norm, skew

from riskmodels.data import (ASSET_CLASS, DATA_DIR, ROOT, WEIGHTS, asset_returns, load_prices,
                             portfolio_returns, weight_vector)
from riskmodels.decomp import ewma_covariance, historical_es_contributions, parametric_shares
from riskmodels.evt import fit_gpd_tail
from riskmodels.garch import fit_garch, loglik
from riskmodels.models import MODEL_NAMES, StudentTInnov

update_summary = importlib.import_module("02_forecast_and_backtest").update_summary

STRESS = {"GFC, Sep-Nov 2008": ("2008-09-01", "2008-11-28"),
          "COVID, 19 Feb-31 Mar 2020": ("2020-02-19", "2020-03-31"),
          "Rates sell-off, Jan-Oct 2022": ("2022-01-03", "2022-10-31"),
          "Tariff shock, Apr 2025": ("2025-04-02", "2025-04-30")}

if __name__ == "__main__":
    rets = asset_returns(load_prices())
    port = portfolio_returns(rets)
    fc = pd.read_csv(DATA_DIR / "forecasts.csv", index_col=0, parse_dates=True)
    start = len(port) - len(fc)
    tickers = list(rets.columns)
    w = weight_vector(tickers)
    R = rets.values

    # --- Euler decomposition -------------------------------------------------
    S = ewma_covariance(R, 0.94)[start:]
    shares = parametric_shares(w, S)                      # (n_oos, N)
    sigma_p = np.sqrt(np.einsum("i,tij,j->t", w, S, w))
    mean_share = shares.mean(axis=0)
    contrib, es_total = historical_es_contributions(R, w, 0.975)
    es_share = contrib / es_total
    classes = sorted(set(ASSET_CLASS.values()))
    by_class = {c: {"weight_pct": 100 * float(sum(w[i] for i, t in enumerate(tickers) if ASSET_CLASS[t] == c)),
                    "var_share_pct": 100 * float(sum(mean_share[i] for i, t in enumerate(tickers) if ASSET_CLASS[t] == c)),
                    "es_share_pct": 100 * float(sum(es_share[i] for i, t in enumerate(tickers) if ASSET_CLASS[t] == c))}
                for c in classes}
    tsy = [i for i, t in enumerate(tickers) if ASSET_CLASS[t] == "Treasuries"]
    tsy_share = pd.Series(shares[:, tsy].sum(axis=1), index=fc.index)
    pre, post = R[rets.index < "2022-01-01"], R[rets.index >= "2022-01-01"]
    c_pre, e_pre = historical_es_contributions(pre, w, 0.975)
    c_post, e_post = historical_es_contributions(post, w, 0.975)
    decomposition = {
        "method": {"var_share": "Euler share under the EWMA(0.94) covariance, averaged over out-of-sample days",
                   "es_share": "historical 97.5% ES contribution, full sample"},
        "assets": {t: {"class": ASSET_CLASS[t], "weight_pct": 100 * float(w[i]),
                       "var_share_pct": 100 * float(mean_share[i]),
                       "var_share_latest_pct": 100 * float(shares[-1, i]),
                       "es_share_pct": 100 * float(es_share[i]),
                       "es_contribution_pct": 100 * float(contrib[i])}
                   for i, t in enumerate(tickers)},
        "by_class": by_class,
        "hist_es975_pct": 100 * es_total,
        "mean_param_var99_pct": 100 * float(norm.ppf(0.99) * sigma_p.mean()),
        "treasuries": {
            "ewma_share_pct_2007_2021": 100 * float(tsy_share[:"2021-12-31"].mean()),
            "ewma_share_pct_2022_on": 100 * float(tsy_share["2022-01-01":].mean()),
            "ewma_share_pct_2022": 100 * float(tsy_share["2022-01-01":"2022-12-31"].mean()),
            "frac_days_negative_2007_2021_pct": 100 * float((tsy_share[:"2021-12-31"] < 0).mean()),
            "frac_days_negative_2022_on_pct": 100 * float((tsy_share["2022-01-01":] < 0).mean()),
            "es_share_pct_before_2022": 100 * float(c_pre[tsy].sum() / e_pre),
            "es_share_pct_2022_on": 100 * float(c_post[tsy].sum() / e_post)},
    }
    web = [{"asset": t, "class": ASSET_CLASS[t], "weight": round(float(w[i]), 4),
            "var_share": round(float(mean_share[i]), 4), "es_share": round(float(es_share[i]), 4)}
           for i, t in enumerate(tickers)]
    (ROOT / "results" / "web" / "risk_contrib.json").write_text(json.dumps(web, indent=1))

    # --- stress windows ------------------------------------------------------
    stress = {}
    for label, (lo, hi) in STRESS.items():
        rw = port[lo:hi]
        wealth = (1 + rw).cumprod()
        assets_cum = (1 + rets[lo:hi]).prod() - 1
        row = {"start": str(rw.index[0].date()), "end": str(rw.index[-1].date()), "n_days": len(rw),
               "return_pct": 100 * float(wealth.iloc[-1] - 1),
               "max_drawdown_pct": 100 * float((wealth / np.maximum(wealth.cummax(), 1.0) - 1).min()),
               "worst_day_pct": 100 * float(rw.min()), "worst_day": str(rw.idxmin().date()),
               "expected_exceedances99": 0.01 * len(rw),
               "asset_return_pct": {t: 100 * float(assets_cum[t]) for t in tickers}, "models": {}}
        f = fc[lo:hi]
        for name in MODEL_NAMES:
            v = f[f"{name}|var99"].values
            doubled = np.flatnonzero(v >= 2 * v[0])
            row["models"][name] = {
                "exceedances99": int((f["loss"].values > v).sum()),
                "var99_start_pct": 100 * float(v[0]), "var99_peak_pct": 100 * float(v.max()),
                "days_to_double": int(doubled[0]) if len(doubled) else None,
                "days_to_peak": int(v.argmax())}
        stress[label] = row

    # --- full-sample diagnostics of GARCH residuals --------------------------
    fit = fit_garch(port.values)
    z = fit.z
    nu = minimize_scalar(lambda v: -loglik(z, np.ones_like(z), v), bounds=(2.1, 100), method="bounded").x
    x = -z                                   # loss orientation
    tail = fit_gpd_tail(x, 0.10)
    t_innov = StudentTInnov(nu)
    diagnostics = {
        "n": len(z), "garch": {"mu_pct": 100 * fit.mu, "omega_pct2": 1e4 * fit.omega, "alpha": fit.alpha,
                               "beta": fit.beta, "persistence": fit.persistence},
        "resid_skew": float(skew(z)), "resid_excess_kurtosis": float(kurtosis(z)), "t_nu": float(nu),
        "gpd": {"threshold": tail.u, "xi": tail.xi, "beta": tail.beta, "k": int(round(tail.p_u * len(x)))},
        "loss_quantile_99": {"normal": float(norm.ppf(0.99)), "student_t": float(t_innov.ppf(0.99)),
                             "empirical": float(np.quantile(x, 0.99)), "gpd": float(tail.ppf(0.99))},
        "loss_quantile_999": {"normal": float(norm.ppf(0.999)), "student_t": float(t_innov.ppf(0.999)),
                              "empirical": float(np.quantile(x, 0.999)), "gpd": float(tail.ppf(0.999))},
        "gain_quantile_99_empirical": float(np.quantile(z, 0.99)),
        "es_975": {"normal": float(norm.pdf(norm.ppf(0.975)) / 0.025), "student_t": t_innov.es(0.975),
                   "empirical": float(np.sort(x)[int(np.ceil(0.975 * len(x))):].mean()), "gpd": tail.es(0.975)},
    }
    np.savez(DATA_DIR / "diagnostics.npz", z=z, nu=nu, u=tail.u, xi=tail.xi, beta=tail.beta, p_u=tail.p_u,
             tsy_share=tsy_share.values)
    update_summary({"decomposition": decomposition, "stress": stress, "diagnostics": diagnostics})

    print("asset   weight  VaR share  ES share")
    for d in web:
        print(f"{d['asset']:6s} {d['weight']:6.1%} {d['var_share']:9.1%} {d['es_share']:9.1%}")
    print(json.dumps(decomposition["by_class"], indent=1))
    print(json.dumps(decomposition["treasuries"], indent=1))
    for label, row in stress.items():
        print(f"\n{label}: return {row['return_pct']:.1f}%, maxDD {row['max_drawdown_pct']:.1f}%, worst day "
              f"{row['worst_day_pct']:.2f}% ({row['worst_day']}), {row['n_days']} days")
        for name, m in row["models"].items():
            print(f"  {name:8s} exc {m['exceedances99']:2d}  VaR start {m['var99_start_pct']:.2f}  peak "
                  f"{m['var99_peak_pct']:.2f}  days to double {m['days_to_double']}")
    print(json.dumps(diagnostics, indent=1))
