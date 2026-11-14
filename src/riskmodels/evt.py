"""Peaks-over-threshold tail estimation with the generalised Pareto distribution.

For excesses ``y = x - u > 0`` over a threshold ``u`` the GPD has distribution function

    G(y; xi, beta) = 1 - (1 + xi y / beta)^(-1/xi),      xi != 0
                   = 1 - exp(-y / beta),                 xi  = 0

on ``y >= 0`` (and ``y <= -beta/xi`` if ``xi < 0``). With ``k`` of ``n`` observations
above ``u``, the tail estimator of McNeil and Frey (2000, eq. 6-8) is

    1 - F(x) = (k/n) (1 + xi (x - u) / beta)^(-1/xi),    x > u

which inverts to the quantile and, for ``xi < 1``, the expected shortfall

    x_q  = u + (beta/xi) [ ((1 - q) n / k)^(-xi) - 1 ]
    ES_q = x_q / (1 - xi) + (beta - xi u) / (1 - xi).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

XI_MIN, XI_MAX = -0.5, 0.95  # MLE is regular for xi > -1/2; ES needs xi < 1


def gpd_nll(xi: float, beta: float, y: np.ndarray) -> float:
    """Negative log-likelihood ``k log beta + (1 + 1/xi) sum log(1 + xi y / beta)``."""
    if beta <= 0.0:
        return np.inf
    if abs(xi) < 1e-9:
        return y.shape[0] * np.log(beta) + y.sum() / beta
    a = 1.0 + xi * y / beta
    if a.min() <= 0.0:
        return np.inf
    return y.shape[0] * np.log(beta) + (1.0 + 1.0 / xi) * np.log(a).sum()


def fit_gpd(y: np.ndarray) -> tuple[float, float]:
    """Maximum-likelihood ``(xi, beta)`` for positive excesses ``y``.

    Nelder-Mead on ``(xi, log beta)`` started from the method-of-moments estimate
    ``xi = (1 - m^2/s^2)/2``, ``beta = m (1 + m^2/s^2)/2``; ``xi`` is restricted to
    ``(-0.5, 0.95)``.
    """
    y = np.asarray(y, dtype=float)
    m, s2 = y.mean(), y.var()
    xi0 = float(np.clip(0.5 * (1.0 - m * m / s2), -0.3, 0.6))
    beta0 = max(0.5 * m * (1.0 + m * m / s2), 1e-3 * m)

    def objective(p: np.ndarray) -> float:
        if not XI_MIN < p[0] < XI_MAX:
            return 1e12
        return min(gpd_nll(p[0], np.exp(p[1]), y), 1e12)

    best = None
    for start in ([xi0, np.log(beta0)], [0.1, np.log(m)]):
        res = minimize(objective, start, method="Nelder-Mead",
                       options={"xatol": 1e-9, "fatol": 1e-11, "maxiter": 2000})
        if best is None or res.fun < best.fun:
            best = res
    return float(best.x[0]), float(np.exp(best.x[1]))


@dataclass(frozen=True)
class GPDTail:
    """Fitted upper tail of a sample: threshold, GPD parameters, and ``k/n``."""
    u: float
    xi: float
    beta: float
    p_u: float  # k / n, the empirical probability of exceeding u

    def sf(self, x: np.ndarray | float) -> np.ndarray | float:
        """Survival function ``1 - F(x)`` for ``x > u``."""
        y = np.asarray(x) - self.u
        if abs(self.xi) < 1e-9:
            return self.p_u * np.exp(-y / self.beta)
        return self.p_u * (1.0 + self.xi * y / self.beta) ** (-1.0 / self.xi)

    def ppf(self, q: np.ndarray | float) -> np.ndarray | float:
        """Quantile ``x_q`` for ``q > 1 - k/n``."""
        ratio = (1.0 - np.asarray(q)) / self.p_u
        if abs(self.xi) < 1e-9:
            return self.u - self.beta * np.log(ratio)
        return self.u + self.beta / self.xi * (ratio ** (-self.xi) - 1.0)

    def es(self, q: float) -> float:
        """Expected shortfall ``E[X | X > x_q]``."""
        return float(self.ppf(q) / (1.0 - self.xi) + (self.beta - self.xi * self.u) / (1.0 - self.xi))


def fit_gpd_tail(x: np.ndarray, tail_fraction: float = 0.10) -> GPDTail:
    """Fit the upper tail of ``x`` by peaks over threshold.

    The threshold is the ``(k+1)``-th largest observation with ``k = floor(tail_fraction n)``,
    so exactly ``k`` observations exceed it (McNeil and Frey use ``k = 100`` of ``n = 1000``).
    """
    xs = np.sort(np.asarray(x, dtype=float))
    n = xs.shape[0]
    k = int(np.floor(tail_fraction * n))
    u = xs[n - k - 1]
    xi, beta = fit_gpd(xs[n - k:] - u)
    return GPDTail(u=float(u), xi=xi, beta=beta, p_u=k / n)
