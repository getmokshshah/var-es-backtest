"""One-day-ahead VaR and ES forecasts from seven models, strictly out of sample.

Every model is written as a predictive distribution for tomorrow's loss of the form

    L_t = loc_t + scale_t * X,      X ~ F_b(t)

where ``F_b`` is an "innovation" distribution in *loss* orientation (large X = large
loss) that is re-estimated on a fixed schedule (block ``b``), and ``loc_t``,
``scale_t`` use data up to ``t-1`` only. Then for confidence level ``a``

    VaR_a,t = loc_t + scale_t q_a(F_b),      ES_a,t = loc_t + scale_t ES_a(F_b).

| model            | loc   | scale            | innovation F                             |
|------------------|-------|------------------|------------------------------------------|
| HS-250 / HS-1000 | 0     | 1                | empirical law of the last n losses       |
| EWMA             | 0     | EWMA sigma_t     | N(0,1)                                   |
| GARCH-N          | -mu   | GARCH sigma_t    | N(0,1)                                   |
| GJR-t            | -mu   | GJR sigma_t      | unit-variance Student-t(nu)              |
| FHS              | -mu   | GARCH sigma_t    | empirical law of -z (GARCH residuals)    |
| EVT              | -mu   | GARCH sigma_t    | GPD tail fitted to the top 10% of -z     |

GARCH-N, FHS and EVT share one Gaussian quasi-ML GARCH(1,1) filter, so they differ
only in the innovation quantile.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, t as student_t

from .evt import GPDTail, fit_gpd_tail
from .garch import GarchFit, fit_garch

MODEL_NAMES = ["HS-250", "HS-1000", "EWMA", "GARCH-N", "GJR-t", "FHS", "EVT"]
_EPS = 1e-9  # guards ceil(n * a) against floating-point noise


class NormalInnov:
    def ppf(self, u):
        return norm.ppf(u)

    def es(self, a: float) -> float:
        """``phi(z_a) / (1 - a)``."""
        return float(norm.pdf(norm.ppf(a)) / (1.0 - a))


@dataclass(frozen=True)
class StudentTInnov:
    """Student-t scaled to unit variance: ``X = sqrt((nu-2)/nu) T_nu``."""
    nu: float

    def ppf(self, u):
        return np.sqrt((self.nu - 2.0) / self.nu) * student_t.ppf(u, self.nu)

    def es(self, a: float) -> float:
        """``sqrt((nu-2)/nu) f_nu(t_a)/(1-a) (nu + t_a^2)/(nu - 1)``."""
        q = student_t.ppf(a, self.nu)
        return float(np.sqrt((self.nu - 2.0) / self.nu) * student_t.pdf(q, self.nu) / (1.0 - a)
                     * (self.nu + q * q) / (self.nu - 1.0))


@dataclass(frozen=True)
class EmpiricalInnov:
    """Empirical distribution of a sorted sample ``x_(1) <= ... <= x_(n)``.

    Quantile: ``Q(u) = x_(ceil(n u))`` (the generalised inverse of the empirical CDF).
    Expected shortfall: ``ES_a = (1/(1-a)) int_a^1 Q(u) du``, which is
    ``[ (ceil(na) - na) x_(ceil(na)) + sum_{i > ceil(na)} x_(i) ] / (n (1-a))``.
    """
    x: np.ndarray

    def ppf(self, u):
        n = self.x.shape[0]
        idx = np.ceil(n * np.asarray(u) - _EPS).astype(int) - 1
        return self.x[np.clip(idx, 0, n - 1)]

    def es(self, a: float) -> float:
        n = self.x.shape[0]
        k = int(np.ceil(n * a - _EPS))
        return float(((k - n * a) * self.x[k - 1] + self.x[k:].sum()) / (n * (1.0 - a)))


@dataclass(frozen=True)
class GPDInnov:
    """GPD tail above the threshold; only valid for levels above ``1 - k/n``."""
    tail: GPDTail

    def ppf(self, u):
        return self.tail.ppf(u)

    def es(self, a: float) -> float:
        return self.tail.es(a)


@dataclass
class Forecast:
    """Predictive loss distributions for the out-of-sample days."""
    name: str
    loc: np.ndarray
    scale: np.ndarray
    block: np.ndarray        # index into ``innovations`` for each day
    innovations: list

    def var(self, a: float) -> np.ndarray:
        q = np.array([inn.ppf(a) for inn in self.innovations], dtype=float)
        return self.loc + self.scale * q[self.block]

    def es(self, a: float) -> np.ndarray:
        e = np.array([inn.es(a) for inn in self.innovations], dtype=float)
        return self.loc + self.scale * e[self.block]

    def tail_ppf(self, i: int, u: np.ndarray) -> np.ndarray:
        """Quantiles of day ``i``'s predictive loss distribution at levels ``u``."""
        return self.loc[i] + self.scale[i] * self.innovations[self.block[i]].ppf(u)


def historical_simulation(r: np.ndarray, start: int, window: int) -> Forecast:
    """Unfiltered historical simulation on the last ``window`` losses (fewer if unavailable)."""
    days = range(start, len(r))
    inns = [EmpiricalInnov(np.sort(-r[max(0, t - window):t])) for t in days]
    m = len(inns)
    return Forecast(f"HS-{window}", np.zeros(m), np.ones(m), np.arange(m), inns)


def ewma_variance(r: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """RiskMetrics variance forecast: ``s2_t = lam s2_{t-1} + (1-lam) r_{t-1}^2``, ``s2_1 = r_0^2``.

    Element ``t`` uses ``r_0 .. r_{t-1}`` only; element 0 is undefined (NaN).
    Equivalently ``s2_t = lam^(t-1) r_0^2 + (1-lam) sum_{k=1}^{t-1} lam^(k-1) r_{t-k}^2``.
    """
    s2 = np.full(len(r), np.nan)
    s2[1] = r[0] ** 2
    for t in range(2, len(r)):
        s2[t] = lam * s2[t - 1] + (1.0 - lam) * r[t - 1] ** 2
    return s2


def riskmetrics(r: np.ndarray, start: int, lam: float = 0.94) -> Forecast:
    sigma = np.sqrt(ewma_variance(r, lam)[start:])
    m = len(sigma)
    return Forecast("EWMA", np.zeros(m), sigma, np.zeros(m, dtype=int), [NormalInnov()])


def rolling_garch(r: np.ndarray, start: int, window: int = 1000, refit_every: int = 20,
                  gjr: bool = False, dist: str = "normal"
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[GarchFit]]:
    """Rolling-window GARCH forecasts ``(mu_t, sigma_t, block_t, fits)`` for ``t >= start``.

    Parameters are re-estimated every ``refit_every`` days on ``r[t-window:t]``
    (warm-started from the previous estimate). In between, the variance is updated
    daily with the latest residual and the parameters held fixed:
    ``sigma2_{t+1} = omega + (alpha + gamma 1{eps_t<0}) eps_t^2 + beta sigma2_t``.
    """
    m = len(r) - start
    mu, sigma, block = np.empty(m), np.empty(m), np.empty(m, dtype=int)
    fits: list[GarchFit] = []
    fit, s2 = None, np.nan
    for i in range(m):
        t = start + i
        if i % refit_every == 0:
            fit = fit_garch(r[max(0, t - window):t], gjr=gjr, dist=dist,
                            x0=None if fit is None else fit.theta)
            fits.append(fit)
            s2 = fit.sigma2_next
        else:
            eps = r[t - 1] - fit.mu
            s2 = fit.omega + (fit.alpha + fit.gamma * (eps < 0.0)) * eps * eps + fit.beta * s2
        mu[i], sigma[i], block[i] = fit.mu, np.sqrt(s2), len(fits) - 1
    return mu, sigma, block, fits


def run_all_models(r: np.ndarray, start: int = 750, window: int = 1000, refit_every: int = 20,
                   lam: float = 0.94, tail_fraction: float = 0.10
                   ) -> tuple[dict[str, Forecast], dict[str, list[GarchFit]]]:
    """Forecasts for days ``start .. len(r)-1``; day ``t`` uses ``r[:t]`` only.

    Also returns the sequence of GARCH and GJR-t fits (one per re-estimation).
    """
    r = np.asarray(r, dtype=float)
    out = {
        "HS-250": historical_simulation(r, start, 250),
        "HS-1000": historical_simulation(r, start, window),
        "EWMA": riskmetrics(r, start, lam),
    }
    mu, sigma, block, fits = rolling_garch(r, start, window, refit_every)
    m = len(mu)
    out["GARCH-N"] = Forecast("GARCH-N", -mu, sigma, np.zeros(m, dtype=int), [NormalInnov()])
    mu_t, sigma_t, block_t, fits_t = rolling_garch(r, start, window, refit_every, gjr=True, dist="t")
    out["GJR-t"] = Forecast("GJR-t", -mu_t, sigma_t, block_t, [StudentTInnov(f.nu) for f in fits_t])
    out["FHS"] = Forecast("FHS", -mu, sigma, block, [EmpiricalInnov(np.sort(-f.z)) for f in fits])
    out["EVT"] = Forecast("EVT", -mu, sigma, block,
                          [GPDInnov(fit_gpd_tail(-f.z, tail_fraction)) for f in fits])
    return out, {"garch": fits, "gjr_t": fits_t}
