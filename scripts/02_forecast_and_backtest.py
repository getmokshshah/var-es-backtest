"""Run the seven models out of sample and apply every backtest.

Writes data/forecasts.csv (full precision, fractions of portfolio value),
results/summary.json (sections "design", "models", "dm", "subperiods", "parameters")
and results/web/var_series.json.
"""
import json
import time

import numpy as np
import pandas as pd

from riskmodels import backtest as bt
from riskmodels.data import DATA_DIR, ROOT, WEIGHTS, asset_returns, load_prices, portfolio_returns
from riskmodels.models import MODEL_NAMES, run_all_models

START, WINDOW, REFIT, LAMBDA, TAIL_FRACTION = 750, 1000, 20, 0.94, 0.10
A_VAR, A_ES = 0.99, 0.975
N_SIMS, N_BOOT, SEED = 5000, 10000, 2026
BASELINE = "HS-250"
SUBPERIODS = {"2007-11 to 2009-12": ("2007-01-01", "2009-12-31"),
              "2010 to 2019": ("2010-01-01", "2019-12-31"),
              "2020 to 2026-11": ("2020-01-01", "2026-12-31")}


def update_summary(section: dict) -> None:
    path = ROOT / "results" / "summary.json"
    summary = json.loads(path.read_text()) if path.exists() else {}
    summary.update(section)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    rets = asset_returns(load_prices())
    port = portfolio_returns(rets)
    r = port.values
    dates = port.index[START:]
    loss = -r[START:]
    n = len(loss)

    t0 = time.perf_counter()
    forecasts, fits = run_all_models(r, START, WINDOW, REFIT, LAMBDA, TAIL_FRACTION)
    print(f"forecasts for {n} days x {len(forecasts)} models in {time.perf_counter() - t0:.1f}s")

    rng = np.random.default_rng(SEED)
    frame = pd.DataFrame({"loss": loss}, index=dates)
    scores, models = {}, {}
    for name in MODEL_NAMES:
        f = forecasts[name]
        v99, v975, es = f.var(A_VAR), f.var(A_ES), f.es(A_ES)
        frame[f"{name}|var99"], frame[f"{name}|var975"], frame[f"{name}|es975"] = v99, v975, es
        h99, h975 = loss > v99, loss > v975
        k99, c99 = bt.kupiec(h99, 1 - A_VAR), bt.christoffersen(h99, 1 - A_VAR)
        k975, c975 = bt.kupiec(h975, 1 - A_ES), bt.christoffersen(h975, 1 - A_ES)
        zones = bt.basel_zones(bt.rolling_exceedances(h99, 250))
        z2 = bt.z2_statistic(loss, v975, es, A_ES)
        null = bt.z2_null(f.tail_ppf, es, A_ES, N_SIMS, rng)
        mf = bt.mcneil_frey(loss, v975, es, N_BOOT, rng)
        scores[name] = {"pinball99": bt.pinball_loss(loss, v99, A_VAR),
                        "fz0": bt.fz0_loss(loss, v975, es, A_ES)}
        models[name] = {
            "n_days": n,
            "var99": {"exceedances": int(h99.sum()), "expected": n * (1 - A_VAR),
                      "rate_pct": 100 * h99.mean(), "kupiec_lr": k99[0], "kupiec_p": k99[1],
                      "ind_lr": c99["lr_ind"], "ind_p": c99["p_ind"], "cc_lr": c99["lr_cc"],
                      "cc_p": c99["p_cc"], "pi01_pct": 100 * c99["pi01"], "pi11_pct": 100 * c99["pi11"]},
            "var975": {"exceedances": int(h975.sum()), "expected": n * (1 - A_ES),
                       "rate_pct": 100 * h975.mean(), "kupiec_lr": k975[0], "kupiec_p": k975[1],
                       "ind_p": c975["p_ind"], "cc_p": c975["p_cc"]},
            "basel": {"green_pct": 100 * zones["green"], "yellow_pct": 100 * zones["yellow"],
                      "red_pct": 100 * zones["red"], "max_250d_count": zones["max"]},
            "es975": {"z2": z2, "z2_p": float(np.mean(null <= z2)),
                      "z2_null_q05": float(np.quantile(null, 0.05)),
                      "z2_null_q01": float(np.quantile(null, 0.01)),
                      "mf_mean_resid": mf["mean_resid"], "mf_t": mf["t"], "mf_p": mf["p"]},
            "pinball99_bp": 1e4 * float(scores[name]["pinball99"].mean()),
            "fz0": float(scores[name]["fz0"].mean()),
            "level": {"mean_var99_pct": 100 * v99.mean(), "median_var99_pct": 100 * float(np.median(v99)),
                      "max_var99_pct": 100 * v99.max(), "max_over_median": float(v99.max() / np.median(v99)),
                      "mean_abs_daily_change_pct": 100 * float(np.mean(np.abs(np.diff(v99)) / v99[:-1])),
                      "mean_es975_pct": 100 * es.mean()},
        }

    # Diebold-Mariano: every model against the baseline, and all pairs for FZ0.
    dm = {"baseline": BASELINE, "vs_baseline": {}, "fz0_pairwise_dm": {}}
    for name in MODEL_NAMES:
        if name != BASELINE:
            dm["vs_baseline"][name] = {
                k: bt.diebold_mariano(scores[name][k] - scores[BASELINE][k]) for k in ("fz0", "pinball99")}
        dm["fz0_pairwise_dm"][name] = {
            other: bt.diebold_mariano(scores[name]["fz0"] - scores[other]["fz0"])["dm"]
            for other in MODEL_NAMES if other != name}

    sub = {}
    periods = dict(SUBPERIODS)
    periods[f"full {WINDOW}-day windows only"] = (str(port.index[WINDOW].date()), "2026-12-31")
    for label, (lo, hi) in periods.items():
        m = np.asarray((dates >= lo) & (dates <= hi))
        sub[label] = {"n_days": int(m.sum()), "expected99": float(m.sum() * (1 - A_VAR)),
                      "worst_loss_pct": 100 * float(loss[m].max())}
        for name in MODEL_NAMES:
            h = loss[m] > frame[f"{name}|var99"].values[m]
            sub[label][name] = {"exceedances": int(h.sum()), "kupiec_p": bt.kupiec(h, 1 - A_VAR)[1]}

    g, gt = fits["garch"], fits["gjr_t"]
    xi = np.array([inn.tail.xi for inn in forecasts["EVT"].innovations])
    params = {
        "n_refits": len(g),
        "garch": {"alpha_median": float(np.median([f.alpha for f in g])),
                  "beta_median": float(np.median([f.beta for f in g])),
                  "persistence_median": float(np.median([f.persistence for f in g]))},
        "gjr_t": {"alpha_median": float(np.median([f.alpha for f in gt])),
                  "gamma_median": float(np.median([f.gamma for f in gt])),
                  "beta_median": float(np.median([f.beta for f in gt])),
                  "nu_median": float(np.median([f.nu for f in gt])),
                  "nu_min": float(np.min([f.nu for f in gt])), "nu_max": float(np.max([f.nu for f in gt]))},
        "evt_xi": {"median": float(np.median(xi)), "min": float(xi.min()), "max": float(xi.max())},
    }

    design = {
        "tickers": list(WEIGHTS), "weights": WEIGHTS,
        "sample_start": str(port.index[0].date()), "sample_end": str(port.index[-1].date()),
        "n_returns": len(r), "oos_start": str(dates[0].date()), "oos_end": str(dates[-1].date()),
        "n_oos": n, "first_window": START, "window": WINDOW, "refit_every": REFIT,
        "ewma_lambda": LAMBDA, "evt_tail_fraction": TAIL_FRACTION, "z2_sims": N_SIMS,
        "mf_bootstrap": N_BOOT, "seed": SEED,
        "portfolio": {"ann_return_pct": 100 * 252 * float(r.mean()),
                      "ann_vol_pct": 100 * float(r.std() * np.sqrt(252)),
                      "worst_day_pct": 100 * float(r.min()), "worst_day": str(port.idxmin().date()),
                      "oos_mean_loss_pct": 100 * float(loss.mean())},
    }
    update_summary({"design": design, "models": models, "dm": dm, "subperiods": sub, "parameters": params})

    frame.to_csv(DATA_DIR / "forecasts.csv", float_format="%.10g")

    web = {"dates": [d.strftime("%Y-%m-%d") for d in dates],
           "loss": np.round(100 * loss, 3).tolist(), "units": "percent of portfolio value, loss positive",
           "var99": {name: np.round(100 * frame[f"{name}|var99"].values, 3).tolist() for name in MODEL_NAMES},
           "summary": {name: {"exc": m["var99"]["exceedances"], "exp": round(m["var99"]["expected"], 1),
                              "kupiec_p": round(m["var99"]["kupiec_p"], 4), "cc_p": round(m["var99"]["cc_p"], 4),
                              "pinball_bp": round(m["pinball99_bp"], 4), "fz0": round(m["fz0"], 4),
                              "es_z2_p": round(m["es975"]["z2_p"], 4), "es_mf_p": round(m["es975"]["mf_p"], 4)}
                       for name, m in models.items()}}
    out = ROOT / "results" / "web" / "var_series.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(web, separators=(",", ":")))
    print(f"var_series.json: {out.stat().st_size / 1024:.0f} KB")

    print(f"\n{'model':9s} exc99 (exp {n * 0.01:.0f})  kup_p   ind_p   cc_p  | exc975  kup_p  | Z2      p     MF p  | "
          f"pinball  FZ0    | red%  meanVaR")
    for name, m in models.items():
        a, b, e = m["var99"], m["var975"], m["es975"]
        print(f"{name:9s} {a['exceedances']:5d}          {a['kupiec_p']:.4f} {a['ind_p']:.4f} {a['cc_p']:.4f} | "
              f"{b['exceedances']:5d}  {b['kupiec_p']:.4f} | {e['z2']:+.3f} {e['z2_p']:.4f} {e['mf_p']:.4f} | "
              f"{m['pinball99_bp']:.4f} {m['fz0']:.4f} | {m['basel']['red_pct']:4.1f}  {m['level']['mean_var99_pct']:.3f}")
