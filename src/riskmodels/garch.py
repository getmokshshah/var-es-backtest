"""GARCH(1,1) and GJR-GARCH(1,1) by maximum likelihood, written from the recursion.

Model (returns ``r_t``, constant mean)::

    r_t      = mu + eps_t,            eps_t = sigma_t z_t,   z_t iid, E z = 0, Var z = 1
    sigma2_t = omega + (alpha + gamma 1{eps_{t-1} < 0}) eps_{t-1}^2 + beta sigma2_{t-1}

``gamma = 0`` is Bollerslev (1986) GARCH(1,1); ``gamma != 0`` is Glosten, Jagannathan
and Runkle (1993). ``z`` is standard normal or a Student-t scaled to unit variance,
``z = sqrt((nu-2)/nu) T_nu`` (Bollerslev 1987).

Initialisation. The pre-sample squared residual and variance are both set to a
"backcast" ``b``: an exponentially weighted mean (decay 0.94) of the first 75 squared
deviations from the sample mean, and the pre-sample indicator to its expectation 1/2::

    sigma2_0 = omega + (alpha + gamma/2 + beta) b

``b`` is computed once from the sample-mean residuals and held fixed during the
optimisation. This is the convention of the ``arch`` package, chosen so that the
estimates can be compared with it to optimiser precision.

Constraints. ``omega > 0``, ``alpha >= 0``, ``alpha + gamma >= 0``, ``beta >= 0``
(positivity of the variance), ``alpha + gamma/2 + beta <= 1`` (covariance
stationarity for a symmetric innovation), ``nu > 2.05``.

The recursion is linear in ``sigma2``, so it is evaluated with ``scipy.signal.lfilter``
rather than a Python loop.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import gammaln

_LOG_2PI = np.log(2.0 * np.pi)
_SCALE = 100.0  # the optimiser works on percent returns for conditioning


@dataclass(frozen=True)
class GarchFit:
    """Estimates in the units of the input series."""
    mu: float
    omega: float
    alpha: float
    gamma: float
    beta: float
    nu: float              # np.inf for normal innovations
    loglik: float          # at the optimum, in the units of the input series
    sigma2: np.ndarray     # in-sample conditional variances, length n
    sigma2_next: float     # one-step-ahead variance forecast
    z: np.ndarray          # standardised residuals (r - mu) / sigma
    theta: np.ndarray      # optimiser parameter vector (percent units), for warm starts

    @property
    def persistence(self) -> float:
        return self.alpha + 0.5 * self.gamma + self.beta


def backcast(eps: np.ndarray, decay: float = 0.94, span: int = 75) -> float:
    """Exponentially weighted mean of the first ``span`` squared residuals."""
    tau = min(span, eps.shape[0])
    w = decay ** np.arange(tau)
    return float(w @ (eps[:tau] ** 2) / w.sum())


def garch_variance(eps: np.ndarray, omega: float, alpha: float, beta: float,
                   gamma: float = 0.0, b: float | None = None) -> np.ndarray:
    """Conditional variances ``sigma2_0 .. sigma2_n`` for residuals ``eps_0 .. eps_{n-1}``.

    The last element is the one-step-ahead forecast. With
    ``x_t = (alpha + gamma 1{eps_t<0}) eps_t^2`` the recursion
    ``sigma2_t = omega + x_{t-1} + beta sigma2_{t-1}`` is an AR(1) filter of the
    input ``u = (sigma2_0, omega + x_0, ..., omega + x_{n-1})``.
    """
    if b is None:
        b = backcast(eps)
    e2 = eps * eps
    u = np.empty(eps.shape[0] + 1)
    u[0] = omega + (alpha + 0.5 * gamma + beta) * b
    u[1:] = omega + (alpha * e2 if gamma == 0.0 else (alpha + gamma * (eps < 0.0)) * e2)
    return lfilter([1.0], [1.0, -beta], u)


def loglik(eps: np.ndarray, sigma2: np.ndarray, nu: float = np.inf) -> float:
    """Log-likelihood of residuals given variances.

    Normal:     ``-1/2 sum [ log 2pi + log sigma2_t + eps_t^2 / sigma2_t ]``
    Student-t:  ``sum [ lgamma((nu+1)/2) - lgamma(nu/2) - 1/2 log(pi (nu-2))
                        - 1/2 log sigma2_t - (nu+1)/2 log(1 + eps_t^2 / (sigma2_t (nu-2))) ]``
    """
    if np.isinf(nu):
        return float(-0.5 * np.sum(_LOG_2PI + np.log(sigma2) + eps * eps / sigma2))
    const = gammaln(0.5 * (nu + 1.0)) - gammaln(0.5 * nu) - 0.5 * np.log(np.pi * (nu - 2.0))
    return float(eps.shape[0] * const - 0.5 * np.sum(np.log(sigma2))
                 - 0.5 * (nu + 1.0) * np.sum(np.log1p(eps * eps / (sigma2 * (nu - 2.0)))))


def _unpack(theta: np.ndarray, gjr: bool, t_dist: bool) -> tuple[float, float, float, float, float, float]:
    mu, omega, alpha = theta[0], theta[1], theta[2]
    gamma = theta[3] if gjr else 0.0
    beta = theta[3 + gjr]
    nu = theta[4 + gjr] if t_dist else np.inf
    return mu, omega, alpha, gamma, beta, nu


def _nll(theta: np.ndarray, y: np.ndarray, b: float, gjr: bool, t_dist: bool) -> float:
    mu, omega, alpha, gamma, beta, nu = _unpack(theta, gjr, t_dist)
    eps = y - mu
    s2 = garch_variance(eps, omega, alpha, beta, gamma, b)[:-1]
    if not np.all(s2 > 0.0):
        return 1e12
    val = -loglik(eps, s2, nu)
    return val if np.isfinite(val) else 1e12


def _starting_grid(y: np.ndarray, gjr: bool, t_dist: bool) -> list[np.ndarray]:
    mu, var = y.mean(), y.var()
    grid = []
    for alpha, persistence in [(0.05, 0.98), (0.10, 0.95), (0.10, 0.90), (0.20, 0.90), (0.03, 0.99)]:
        gamma = 0.08 if gjr else 0.0
        a = max(alpha - 0.5 * gamma, 0.01)
        beta = persistence - a - 0.5 * gamma
        theta = [mu, var * (1.0 - persistence), a] + ([gamma] if gjr else []) + [beta] + ([8.0] if t_dist else [])
        grid.append(np.array(theta))
    return grid


def fit_garch(r: np.ndarray, gjr: bool = False, dist: str = "normal",
              x0: np.ndarray | None = None) -> GarchFit:
    """Maximum-likelihood GARCH(1,1) or GJR-GARCH(1,1) with normal or Student-t innovations.

    Sequential least-squares programming (SLSQP) with numerical gradients on percent
    returns, under the bounds and the linear stationarity constraint in the module
    docstring. The optimiser is started from the best point of a small grid and, if
    given, from the warm start ``x0`` (the previous window's ``GarchFit.theta``); the
    higher likelihood wins.
    """
    if dist not in ("normal", "t"):
        raise ValueError("dist must be 'normal' or 't'")
    t_dist = dist == "t"
    y = _SCALE * np.asarray(r, dtype=float)
    n, var = y.shape[0], y.var()
    b = backcast(y - y.mean())

    bounds = [(None, None), (1e-8 * var, 10.0 * var), (0.0, 1.0)]
    if gjr:
        bounds.append((-1.0, 1.0))
    bounds.append((0.0, 1.0))
    if t_dist:
        bounds.append((2.05, 500.0))
    k = len(bounds)
    half_gamma = 0.5 if gjr else 0.0

    def stationarity(theta: np.ndarray) -> float:  # 1 - alpha - gamma/2 - beta >= 0
        return 1.0 - theta[2] - half_gamma * theta[3] - theta[3 + gjr]

    constraints = [{"type": "ineq", "fun": stationarity}]
    if gjr:
        constraints.append({"type": "ineq", "fun": lambda th: th[2] + th[3]})  # alpha + gamma >= 0

    args = (y, b, gjr, t_dist)
    grid = _starting_grid(y, gjr, t_dist)
    starts = [min(grid, key=lambda th: _nll(th, *args))]
    if x0 is not None and len(x0) == k:
        starts.append(np.asarray(x0, dtype=float))

    best = None
    for start in starts:
        res = minimize(_nll, start, args=args, method="SLSQP", bounds=bounds,
                       constraints=constraints, options={"ftol": 1e-12, "maxiter": 500})
        if best is None or res.fun < best.fun:
            best = res
    theta = np.asarray(best.x)
    mu, omega, alpha, gamma, beta, nu = _unpack(theta, gjr, t_dist)

    eps = y - mu
    s2 = garch_variance(eps, omega, alpha, beta, gamma, b)
    z = eps / np.sqrt(s2[:-1])
    return GarchFit(mu=mu / _SCALE, omega=omega / _SCALE ** 2, alpha=alpha, gamma=gamma, beta=beta,
                    nu=nu, loglik=-best.fun + n * np.log(_SCALE),
                    sigma2=s2[:-1] / _SCALE ** 2, sigma2_next=s2[-1] / _SCALE ** 2, z=z, theta=theta)


def simulate_garch(n: int, mu: float, omega: float, alpha: float, beta: float, gamma: float = 0.0,
                   nu: float = np.inf, rng: np.random.Generator | None = None,
                   burn: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Simulate ``(r, sigma2)`` from the model, started at the unconditional variance."""
    rng = rng or np.random.default_rng(0)
    m = n + burn
    z = rng.standard_normal(m) if np.isinf(nu) else rng.standard_t(nu, m) * np.sqrt((nu - 2.0) / nu)
    s2 = np.empty(m)
    eps = np.empty(m)
    s2[0] = omega / (1.0 - alpha - 0.5 * gamma - beta)
    eps[0] = np.sqrt(s2[0]) * z[0]
    for t in range(1, m):
        s2[t] = omega + (alpha + gamma * (eps[t - 1] < 0.0)) * eps[t - 1] ** 2 + beta * s2[t - 1]
        eps[t] = np.sqrt(s2[t]) * z[t]
    return mu + eps[burn:], s2[burn:]
