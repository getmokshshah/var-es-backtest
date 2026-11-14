"""Render the README tables from results/summary.json into results/tables.md."""
import json

from riskmodels.data import ROOT
from riskmodels.models import MODEL_NAMES


def p(x: float) -> str:
    return "<0.001" if x < 0.0005 else f"{x:.3f}"


if __name__ == "__main__":
    s = json.loads((ROOT / "results" / "summary.json").read_text())
    M, out = s["models"], []
    n = s["design"]["n_oos"]

    out += [f"**Table 1. VaR coverage, {n:,} out-of-sample days**", "",
            f"| Model | 99% exc. (exp. {M['EVT']['var99']['expected']:.1f}) | Kupiec p | Indep. p | Cond. cov. p "
            f"| 97.5% exc. (exp. {M['EVT']['var975']['expected']:.1f}) | Kupiec p | Cond. cov. p |",
            "|---|---|---|---|---|---|---|---|"]
    for k in MODEL_NAMES:
        a, b = M[k]["var99"], M[k]["var975"]
        out.append(f"| {k} | {a['exceedances']} | {p(a['kupiec_p'])} | {p(a['ind_p'])} | {p(a['cc_p'])} | "
                   f"{b['exceedances']} | {p(b['kupiec_p'])} | {p(b['cc_p'])} |")

    out += ["", "**Table 2. 97.5% ES backtests and scoring**", "",
            "| Model | Z2 | Z2 p | Mean exceedance residual | McNeil-Frey p | Pinball 99% (bp) | FZ0 | DM vs HS-250 (FZ0) |",
            "|---|---|---|---|---|---|---|---|"]
    for k in MODEL_NAMES:
        e = M[k]["es975"]
        dm = "baseline" if k == s["dm"]["baseline"] else f"{s['dm']['vs_baseline'][k]['fz0']['dm']:.2f}"
        out.append(f"| {k} | {e['z2']:.3f} | {p(e['z2_p'])} | {e['mf_mean_resid']:+.3f} | {p(e['mf_p'])} | "
                   f"{M[k]['pinball99_bp']:.3f} | {M[k]['fz0']:.4f} | {dm} |")

    out += ["", "**Table 3. Level and stability of the 99% VaR, and Basel zones**", "",
            "| Model | Mean VaR (%) | Max VaR (%) | Max / median | Mean abs. daily change (%) | Green (%) | Yellow (%) | Red (%) | Max 250-day count |",
            "|---|---|---|---|---|---|---|---|---|"]
    for k in MODEL_NAMES:
        l, b = M[k]["level"], M[k]["basel"]
        out.append(f"| {k} | {l['mean_var99_pct']:.2f} | {l['max_var99_pct']:.2f} | {l['max_over_median']:.1f} | "
                   f"{l['mean_abs_daily_change_pct']:.2f} | {b['green_pct']:.1f} | {b['yellow_pct']:.1f} | "
                   f"{b['red_pct']:.1f} | {b['max_250d_count']} |")

    out += ["", "**Table 4. 99% VaR exceedances by sub-period (Kupiec p in brackets)**", "",
            "| Period | Days | Expected | " + " | ".join(MODEL_NAMES) + " |", "|---|---|---|" + "---|" * len(MODEL_NAMES)]
    for label, v in s["subperiods"].items():
        cells = " | ".join(f"{v[k]['exceedances']} ({p(v[k]['kupiec_p'])})" for k in MODEL_NAMES)
        out.append(f"| {label} | {v['n_days']} | {v['expected99']:.1f} | {cells} |")

    out += ["", "**Table 5. Stress windows: book P&L and 99% VaR behaviour**", "",
            "| Window | Return (%) | Max drawdown (%) | Worst day (%) | Days | Model | Exceedances | VaR at start (%) | Peak VaR (%) | Days to double |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for label, v in s["stress"].items():
        for i, k in enumerate(MODEL_NAMES):
            m = v["models"][k]
            head = (f"| {label} | {v['return_pct']:.1f} | {v['max_drawdown_pct']:.1f} | {v['worst_day_pct']:.2f} | "
                    f"{v['n_days']} |") if i == 0 else "| | | | | |"
            dd = "never" if m["days_to_double"] is None else m["days_to_double"]
            out.append(f"{head} {k} | {m['exceedances99']} | {m['var99_start_pct']:.2f} | {m['var99_peak_pct']:.2f} | {dd} |")

    out += ["", "**Table 6. Euler risk contributions against capital weights (percent of total)**", "",
            "| Asset | Class | Capital weight | Parametric VaR share | Historical 97.5% ES share |", "|---|---|---|---|---|"]
    for t, a in s["decomposition"]["assets"].items():
        out.append(f"| {t} | {a['class']} | {a['weight_pct']:.0f} | {a['var_share_pct']:.1f} | {a['es_share_pct']:.1f} |")
    for c, a in s["decomposition"]["by_class"].items():
        out.append(f"| **{c}** | | **{a['weight_pct']:.0f}** | **{a['var_share_pct']:.1f}** | **{a['es_share_pct']:.1f}** |")

    (ROOT / "results" / "tables.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))
