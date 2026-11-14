"""Six figures from data/forecasts.csv, data/diagnostics.npz and results/summary.json."""
import json

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t

from riskmodels import backtest as bt
from riskmodels.data import DATA_DIR, ROOT
from riskmodels.evt import GPDTail
from riskmodels.models import MODEL_NAMES
from riskmodels.plotstyle import (FIGSIZE_WIDE, INK, INK_2, MUTED, RED, SERIES, SURFACE,
                                  apply_style, legend, save, title)

COLOR = dict(zip(MODEL_NAMES, SERIES))   # one colour per model in every figure
LONG = {"HS-250": "Historical simulation, 250 days", "HS-1000": "Historical simulation, 1000 days",
        "EWMA": "RiskMetrics EWMA", "GARCH-N": "GARCH(1,1) normal", "GJR-t": "GJR-GARCH Student-t",
        "FHS": "Filtered historical simulation", "EVT": "Conditional EVT"}


def panel_label(ax, text, x=0.02, y=0.95):
    ax.text(x, y, text, transform=ax.transAxes, ha="left", va="top", fontsize=9.5,
            fontweight="semibold", color=INK)


def fig_march_2020(fc, s):
    names = ["HS-1000", "EWMA", "EVT"]
    w = fc["2020-02-03":"2020-04-30"]
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.9), sharey=True)
    for ax, name in zip(axes, names):
        v = 100 * w[f"{name}|var99"]
        loss = 100 * w["loss"]
        hit = loss > v
        ax.vlines(w.index, 0, loss, color=MUTED, lw=1.3, label="Daily loss")
        ax.plot(w.index, v, color=COLOR[name], label="99% VaR forecast")
        ax.scatter(w.index[hit], loss[hit], color=RED, s=26, zorder=3, edgecolor=SURFACE, lw=0.8,
                   label="Exceedance")
        ax.axhline(0, color=INK_2, lw=0.6)
        panel_label(ax, f"{LONG[name]}\n{int(hit.sum())} exceedances in {len(w)} days")
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.set_xlim(pd.Timestamp("2020-01-30"), pd.Timestamp("2020-05-04"))
    axes[0].set_ylabel("Percent of portfolio value")
    axes[0].set_ylim(-6, 17.5)
    handles = [plt.Line2D([], [], color=MUTED, lw=1.3), plt.Line2D([], [], color=INK),
               plt.Line2D([], [], color=RED, marker="o", ls="")]
    labels = ["Daily loss", "99% VaR forecast (coloured by model)", "Exceedance"]
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.07), handlelength=1.4)
    st = s["stress"]["COVID, 19 Feb-31 Mar 2020"]["models"]
    title(axes[0], f"In March 2020 the 1000-day historical VaR peaked at {st['HS-1000']['var99_peak_pct']:.1f}%; "
                   f"the conditional EVT VaR reached {st['EVT']['var99_peak_pct']:.1f}%",
          "Daily portfolio loss (positive = loss, negative = gain) and the one-day 99% VaR forecast made the evening before, Feb-Apr 2020.")
    fig.subplots_adjust(wspace=0.06)
    save(fig, "var-march-2020")


def acceptance_band(n, p):
    xs = np.arange(0, int(4 * n * p))
    ok = [x for x in xs if bt.kupiec(np.arange(n) < x, p)[1] >= 0.05]
    return min(ok), max(ok)


def fig_coverage(s):
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharey=True)
    n = s["design"]["n_oos"]
    y = np.arange(len(MODEL_NAMES))[::-1]
    for ax, key, p, lab in zip(axes, ["var99", "var975"], [0.01, 0.025], ["99% VaR", "97.5% VaR"]):
        lo, hi = acceptance_band(n, p)
        ax.axvspan(lo, hi, color="#eeede8", zorder=0)
        ax.axvline(n * p, color=INK_2, lw=1.0)
        for yi, name in zip(y, MODEL_NAMES):
            m = s["models"][name][key]
            passed = m["kupiec_p"] >= 0.05
            ax.scatter(m["exceedances"], yi, s=70, zorder=3, color=COLOR[name] if passed else SURFACE,
                       edgecolor=COLOR[name], lw=2)
            ax.annotate(f"{m['exceedances']}", (m["exceedances"], yi), xytext=(9, 0),
                        textcoords="offset points", va="center", fontsize=9, color=INK_2)
        ax.set_yticks(y, [LONG[k] for k in MODEL_NAMES])
        ax.set_ylim(-0.7, len(MODEL_NAMES) - 0.3)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(f"Exceedances of the {lab} in {n:,} days")
        ax.annotate(f"expected {n * p:.0f}", (n * p, -0.55), xytext=(-5, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=8.5, color=INK_2)
    axes[0].set_xlim(30, 115)
    axes[1].set_xlim(95, 180)
    axes[0].scatter([], [], s=60, color=INK_2, edgecolor=INK_2, lw=2, label="Kupiec: not rejected at 5%")
    axes[0].scatter([], [], s=60, color=SURFACE, edgecolor=INK_2, lw=2, label="Kupiec: rejected")
    axes[0].legend(loc="lower right", handletextpad=0.3)
    title(axes[0], "Only filtered historical simulation and conditional EVT have correct coverage at both levels",
          f"Exceedance counts, {s['design']['oos_start'][:4]}-{s['design']['oos_end'][:4]}. "
          "Shaded band: counts the Kupiec test does not reject at 5%.")
    fig.subplots_adjust(wspace=0.05)
    save(fig, "coverage")


def fig_basel(fc, s):
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    loss = fc["loss"].values
    for name in ["HS-1000", "EWMA", "EVT"]:
        counts = bt.rolling_exceedances(loss > fc[f"{name}|var99"].values, 250)
        ax.plot(fc.index[249:], counts, color=COLOR[name], lw=1.6, label=LONG[name])
    for level, text in [(4.5, "yellow: 5 to 9"), (9.5, "red: 10 or more")]:
        ax.axhline(level, color=INK_2, lw=0.9)
        ax.text(1.01, level, text, transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=8.5, color=INK_2)
    ax.set_ylabel("Exceedances in trailing 250 days")
    ax.set_ylim(0, 30)
    legend(ax, loc="upper center", ncol=3)
    b = s["models"]
    title(ax, f"The 1000-day historical VaR logged {b['HS-1000']['basel']['max_250d_count']} exceedances in one year; "
              f"conditional EVT never exceeded {b['EVT']['basel']['max_250d_count']}",
          "Trailing 250-day count of 99% VaR exceedances against the Basel traffic-light zones (green is 0 to 4).")
    save(fig, "basel-traffic-light")


def fig_residual_tail(s):
    d = np.load(DATA_DIR / "diagnostics.npz")
    z, nu = np.sort(d["z"]), float(d["nu"])
    n = len(z)
    pp = (np.arange(1, n + 1) - 0.5) / n
    fig, (a1, a2) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    qn = norm.ppf(pp)
    qt = np.sqrt((nu - 2) / nu) * student_t.ppf(pp, nu)
    a1.plot([-6.6, 4.6], [-6.6, 4.6], color=INK_2, lw=0.9)
    a1.scatter(qn, z, s=9, color=COLOR["GARCH-N"], label="vs normal", zorder=3)
    a1.scatter(qt, z, s=9, color=COLOR["GJR-t"], label=f"vs Student-t, $\\nu$ = {nu:.1f}", zorder=2)
    a1.set_xlabel("Theoretical quantile")
    a1.set_ylabel("Standardised residual quantile")
    a1.set_xlim(-6.6, 4.6)
    a1.set_ylim(-6.6, 4.6)
    a1.grid(axis="x", visible=True)
    a1.legend(loc="upper left", markerscale=2, handletextpad=0.2)
    a1.annotate("losses are fatter than\neither symmetric law", (-3.6, -5.4), xytext=(-1.6, -5.9), fontsize=8.5,
                color=INK_2, va="center", arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.7))

    x = np.sort(-z)                         # loss residuals, ascending
    tail = GPDTail(float(d["u"]), float(d["xi"]), float(d["beta"]), float(d["p_u"]))
    surv = 1 - (np.arange(1, n + 1) - 0.5) / n
    m = x > tail.u
    grid = np.linspace(tail.u, 7.0, 200)
    a2.scatter(x[m], surv[m], s=10, color=MUTED, label="Empirical", zorder=2)
    a2.plot(grid, tail.sf(grid), color=COLOR["EVT"], label=f"GPD, $\\xi$ = {tail.xi:.2f}", zorder=3)
    a2.plot(grid, norm.sf(grid), color=COLOR["GARCH-N"], label="Normal")
    a2.plot(grid, student_t.sf(grid / np.sqrt((nu - 2) / nu), nu), color=COLOR["GJR-t"], label="Student-t")
    a2.set_yscale("log")
    a2.set_ylim(1e-4, 0.15)
    a2.set_xlim(tail.u, 7.0)
    a2.set_xlabel("Loss residual $-z$ (standard deviations)")
    a2.set_ylabel("Probability of a larger loss")
    legend(a2, loc="upper right")
    q = s["diagnostics"]["loss_quantile_99"]
    title(a1, f"GARCH residuals are skewed: the 99% loss quantile is {q['empirical']:.2f} sd, "
              f"not the normal {q['normal']:.2f} or Student-t {q['student_t']:.2f}",
          f"Left: QQ plot of {n:,} standardised residuals from a full-sample GARCH(1,1). "
          "Right: survival function of the loss tail above the 90th percentile.")
    fig.subplots_adjust(wspace=0.26)
    save(fig, "residual-tail")


def fig_risk_contributions(s):
    a = s["decomposition"]["assets"]
    names = list(a)
    y = np.arange(len(names))[::-1]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for yi, t in zip(y, names):
        vals = [a[t]["weight_pct"], a[t]["var_share_pct"], a[t]["es_share_pct"]]
        ax.plot([min(vals), max(vals)], [yi, yi], color="#d5d4cd", lw=2.2, zorder=1, solid_capstyle="butt")
    ax.scatter([a[t]["weight_pct"] for t in names], y, s=62, color=MUTED, zorder=3, label="Capital weight")
    ax.scatter([a[t]["var_share_pct"] for t in names], y, s=62, color=RED, zorder=3,
               label="Share of parametric VaR (EWMA covariance)")
    ax.scatter([a[t]["es_share_pct"] for t in names], y, s=62, marker="D", color=SURFACE, edgecolor=INK, lw=1.6,
               zorder=4, label="Share of historical 97.5% ES")
    ax.axvline(0, color=INK_2, lw=0.8)
    ax.set_yticks(y, [f"{t}  {a[t]['class']}" for t in names])
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("Percent of total")
    ax.set_xlim(-5, 42)
    legend(ax, loc="lower right")
    c = s["decomposition"]["by_class"]
    title(ax, f"Equities are {c['Equity']['weight_pct']:.0f}% of the capital and {c['Equity']['es_share_pct']:.0f}% "
              f"of the expected shortfall; Treasuries are {c['Treasuries']['weight_pct']:.0f}% and "
              f"{c['Treasuries']['es_share_pct']:.0f}%",
          "Euler risk contributions by asset against capital weights. VaR shares are averaged over the out-of-sample days.")
    save(fig, "risk-contributions")


def fig_es_and_scores(s):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.0, 4.0), sharey=True)
    y = np.arange(len(MODEL_NAMES))[::-1]
    for yi, name in zip(y, MODEL_NAMES):
        e = s["models"][name]["es975"]
        passed = e["z2_p"] >= 0.05
        a1.plot([e["z2_null_q05"], 0.35], [yi, yi], color="#e4e3dd", lw=9, zorder=1, solid_capstyle="butt")
        a1.scatter(e["z2"], yi, s=70, zorder=3, color=COLOR[name] if passed else SURFACE,
                   edgecolor=COLOR[name], lw=2)
        a1.text(0.04, yi, f"p = {e['z2_p']:.3f}" if e["z2_p"] >= 0.0005 else "p < 0.001",
                ha="left", va="center", fontsize=8.5, color=INK_2)
        if name != s["dm"]["baseline"]:
            d = s["dm"]["vs_baseline"][name]["fz0"]
            a2.plot([d["mean"] - 1.96 * d["se"], d["mean"] + 1.96 * d["se"]], [yi, yi], color=COLOR[name], lw=2)
            a2.scatter(d["mean"], yi, s=70, zorder=3, color=COLOR[name])
        else:
            a2.scatter(0, yi, s=70, zorder=3, color=COLOR[name])
            a2.annotate("baseline", (0, yi), xytext=(9, 0), textcoords="offset points", va="center",
                        fontsize=8.5, color=INK_2)
    for ax in (a1, a2):
        ax.axvline(0, color=INK_2, lw=0.9)
        ax.grid(axis="y", visible=False)
    a1.set_yticks(y, [LONG[k] for k in MODEL_NAMES])
    a1.set_ylim(-0.7, len(MODEL_NAMES) - 0.3)
    a1.set_xlim(-0.82, 0.35)
    a1.set_xlabel("Acerbi-Szekely $Z_2$ (negative: ES understated)")
    a2.set_xlabel("FZ0 loss minus HS-250, 95% interval (lower is better)")
    title(a1, "Only FHS and conditional EVT pass the ES backtest, and they have the lowest joint (VaR, ES) loss",
          "Left: 97.5% ES statistic; grey bar = values not rejected at 5% under the simulated null, hollow = rejected. "
          "Right: mean FZ0 loss relative to HS-250.")
    fig.subplots_adjust(wspace=0.08)
    save(fig, "es-backtest-and-scores")


if __name__ == "__main__":
    apply_style()
    summary = json.loads((ROOT / "results" / "summary.json").read_text())
    forecasts = pd.read_csv(DATA_DIR / "forecasts.csv", index_col=0, parse_dates=True)
    fig_march_2020(forecasts, summary)
    fig_coverage(summary)
    fig_basel(forecasts, summary)
    fig_residual_tail(summary)
    fig_risk_contributions(summary)
    fig_es_and_scores(summary)
    print("figures written")
