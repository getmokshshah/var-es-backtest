"""Backtests and scoring rules for VaR and ES forecasts.

``hits`` are exceedance indicators ``I_t = 1{L_t > VaR_t}``; ``p = 1 - a`` is the
exceedance probability a correct model implies.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.special import xlogy
from scipy.stats import chi2, norm


def kupiec(hits: np.ndarray, p: float) -> tuple[float, float]:
    """Kupiec (1995) proportion-of-failures test of unconditional coverage.

    With ``x`` hits in ``n`` days and ``pi = x/n``,
    ``LR_uc = -2 log[ (1-p)^(n-x) p^x / ((1-pi)^(n-x) pi^x) ] ~ chi2(1)`` under H0.
    """
    n, x = len(hits), int(np.sum(hits))
    pi = x / n
    lr = -2.0 * (xlogy(n - x, 1.0 - p) + xlogy(x, p) - xlogy(n - x, 1.0 - pi) - xlogy(x, pi))
    return float(lr), float(chi2.sf(lr, 1))


def christoffersen(hits: np.ndarray, p: float) -> dict[str, float]:
    """Christoffersen (1998) independence and conditional-coverage tests.

    ``n_ij`` counts transitions from state ``i`` to ``j`` of the hit sequence;
    ``pi_01 = n01/(n00+n01)``, ``pi_11 = n11/(n10+n11)``, ``pi = (n01+n11)/(n-1)``.

    ``LR_ind = -2 log[ (1-pi)^(n00+n10) pi^(n01+n11)
                       / ((1-pi_01)^n00 pi_01^n01 (1-pi_11)^n10 pi_11^n11) ] ~ chi2(1)``
    ``LR_cc  = LR_uc + LR_ind ~ chi2(2)``.
    """
    h = np.asarray(hits, dtype=int)
    a, b = h[:-1], h[1:]
    n00, n01 = int(np.sum((a == 0) & (b == 0))), int(np.sum((a == 0) & (b == 1)))
    n10, n11 = int(np.sum((a == 1) & (b == 0))), int(np.sum((a == 1) & (b == 1)))
    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    ll0 = xlogy(n00 + n10, 1.0 - pi) + xlogy(n01 + n11, pi)
    ll1 = xlogy(n00, 1.0 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1.0 - pi11) + xlogy(n11, pi11)
    lr_ind = max(-2.0 * float(ll0 - ll1), 0.0)
    lr_uc, _ = kupiec(h, p)
    lr_cc = lr_uc + lr_ind
    return {"n00": n00, "n01": n01, "n10": n10, "n11": n11, "pi01": pi01, "pi11": pi11,
            "lr_ind": lr_ind, "p_ind": float(chi2.sf(lr_ind, 1)),
            "lr_cc": lr_cc, "p_cc": float(chi2.sf(lr_cc, 2))}


def rolling_exceedances(hits: np.ndarray, window: int = 250) -> np.ndarray:
    """Number of hits in each trailing ``window``-day period (length ``n - window + 1``)."""
    c = np.concatenate([[0], np.cumsum(np.asarray(hits, dtype=int))])
    return c[window:] - c[:-window]


def basel_zones(counts: np.ndarray) -> dict[str, float]:
    """Basel traffic light for 99% VaR over 250 days: green 0-4, yellow 5-9, red 10+."""
    return {"green": float(np.mean(counts <= 4)), "yellow": float(np.mean((counts >= 5) & (counts <= 9))),
            "red": float(np.mean(counts >= 10)), "max": int(counts.max())}


def z2_statistic(loss: np.ndarray, var: np.ndarray, es: np.ndarray, a: float) -> float:
    """Acerbi and Szekely (2014) second test statistic, in loss orientation.

    ``Z2 = 1 - (1/(T (1-a))) sum_t L_t 1{L_t > VaR_t} / ES_t``.
    ``E[Z2] = 0`` under H0; ``Z2 < 0`` when exceedances are too frequent or too large.
    """
    return float(1.0 - np.sum(loss * (loss > var) / es) / (len(loss) * (1.0 - a)))


def z2_null(tail_ppf: Callable[[int, np.ndarray], np.ndarray], es: np.ndarray, a: float,
            n_sims: int, rng: np.random.Generator) -> np.ndarray:
    """Simulated null distribution of Z2.

    For each day draw ``U ~ U(0,1)`` independently; the simulated loss is the model's own
    predictive quantile ``Q_t(U)``, which is only needed when ``U > a``. The forecasts
    ``(VaR_t, ES_t)`` are held fixed, as in Acerbi and Szekely.
    """
    total = np.zeros(n_sims)
    for i in range(len(es)):
        u = rng.random(n_sims)
        m = u > a
        if m.any():
            total[m] += tail_ppf(i, u[m]) / es[i]
    return 1.0 - total / (len(es) * (1.0 - a))


def mcneil_frey(loss: np.ndarray, var: np.ndarray, es: np.ndarray, n_boot: int,
                rng: np.random.Generator) -> dict[str, float]:
    """McNeil and Frey (2000) exceedance-residual test of the ES forecast.

    On exceedance days the residuals ``e_t = (L_t - ES_t)/ES_t`` have mean zero under
    H0. The one-sided alternative is ``E[e] > 0`` (ES understated). The p-value is from
    the bootstrap t-test of Efron and Tibshirani (1993, sec. 16.4): resample the
    centred residuals and compare studentised means. Scaling by the predictable ``ES_t``
    (McNeil and Frey scale by ``sigma_t``) lets the same test cover models with no
    volatility filter.
    """
    m = loss > var
    e = (loss[m] - es[m]) / es[m]
    k = len(e)

    def tstat(x: np.ndarray) -> np.ndarray:
        return x.mean(axis=-1) / (x.std(axis=-1, ddof=1) / np.sqrt(k))

    t_obs = float(tstat(e))
    centred = e - e.mean()
    t_star = tstat(centred[rng.integers(0, k, size=(n_boot, k))])
    return {"n_exceed": k, "mean_resid": float(e.mean()), "t": t_obs,
            "p": float(np.mean(t_star >= t_obs))}


def pinball_loss(loss: np.ndarray, var: np.ndarray, a: float) -> np.ndarray:
    """Quantile score ``a (L - v)^+ + (1 - a) (v - L)^+``, minimised in expectation by the a-quantile."""
    d = loss - var
    return np.where(d > 0.0, a * d, -(1.0 - a) * d)


def fz0_loss(loss: np.ndarray, var: np.ndarray, es: np.ndarray, a: float) -> np.ndarray:
    """FZ0 joint (VaR, ES) loss of Patton, Ziegel and Chen (2019), in loss orientation.

    ``1{L > v} (L - v) / ((1-a) e) + v/e + log(e) - 1``, for ``e > 0``. It is the
    zero-degree homogeneous member of the Fissler-Ziegel class, so differences between
    two models do not depend on the unit of ``L``.
    """
    return (loss > var) * (loss - var) / ((1.0 - a) * es) + var / es + np.log(es) - 1.0


def diebold_mariano(d: np.ndarray, lag: int | None = None) -> dict[str, float]:
    """Diebold and Mariano (1995) test that the mean loss differential ``d_t`` is zero.

    ``DM = dbar / sqrt(LRV / T) ~ N(0,1)`` with a Newey-West (Bartlett) long-run
    variance at lag ``floor(T^(1/3))``.
    """
    d = np.asarray(d, dtype=float)
    n = len(d)
    lag = int(n ** (1.0 / 3.0)) if lag is None else lag
    x = d - d.mean()
    lrv = x @ x / n
    for j in range(1, lag + 1):
        lrv += 2.0 * (1.0 - j / (lag + 1.0)) * (x[j:] @ x[:-j]) / n
    se = float(np.sqrt(lrv / n))
    stat = float(d.mean() / se)
    return {"mean": float(d.mean()), "se": se, "dm": stat, "p": float(2.0 * norm.sf(abs(stat)))}
