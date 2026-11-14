"""GARCH / GJR likelihoods and optimiser: against `arch`, an explicit loop, and simulated truth."""
import numpy as np
import pytest
from arch import arch_model
from scipy.integrate import quad
from scipy.stats import t as student_t

from riskmodels.garch import backcast, fit_garch, garch_variance, loglik, simulate_garch
from riskmodels.models import NormalInnov, StudentTInnov

SPECS = [(False, "normal"), (True, "normal"), (False, "t"), (True, "t")]


def _mine_vs_arch(r, gjr, dist):
    fit = fit_garch(r, gjr=gjr, dist=dist)
    ref = arch_model(100 * r, mean="Constant", vol="GARCH", p=1, o=int(gjr), q=1, dist=dist,
                     rescale=False).fit(disp="off", tol=1e-12, options={"maxiter": 2000})
    mine = [100 * fit.mu, 1e4 * fit.omega, fit.alpha] + ([fit.gamma] if gjr else []) + [fit.beta] \
        + ([fit.nu] if dist == "t" else [])
    return fit, np.array(mine), ref


@pytest.mark.parametrize("gjr,dist", SPECS)
@pytest.mark.parametrize("window", [slice(0, 1000), slice(1000, 2000), slice(0, 2000)])
def test_matches_arch_on_real_data(real_returns, gjr, dist, window):
    r = real_returns[window]
    fit, mine, ref = _mine_vs_arch(r, gjr, dist)
    assert fit.loglik - len(r) * np.log(100) >= ref.loglikelihood - 1e-6
    np.testing.assert_allclose(mine, ref.params.values, rtol=5e-4, atol=5e-5)
    np.testing.assert_allclose(1e4 * fit.sigma2, ref.conditional_volatility ** 2, rtol=1e-3)


@pytest.mark.parametrize("gjr,dist", SPECS)
def test_matches_arch_on_simulated_data(gjr, dist):
    r, _ = simulate_garch(3000, 4e-4, 1.5e-6, 0.05, 0.88, gamma=0.08 if gjr else 0.0,
                          nu=6.0 if dist == "t" else np.inf, rng=np.random.default_rng(7))
    fit, mine, ref = _mine_vs_arch(r, gjr, dist)
    assert abs(fit.loglik - len(r) * np.log(100) - ref.loglikelihood) < 1e-5
    np.testing.assert_allclose(mine, ref.params.values, rtol=1e-3, atol=1e-4)


def test_parameter_recovery_garch():
    r, _ = simulate_garch(20000, 5e-4, 2e-6, 0.08, 0.90, rng=np.random.default_rng(11))
    fit = fit_garch(r)
    assert abs(fit.alpha - 0.08) < 0.012 and abs(fit.beta - 0.90) < 0.02
    assert abs(fit.omega / 2e-6 - 1) < 0.25 and abs(fit.mu - 5e-4) < 2e-4


def test_parameter_recovery_gjr_t():
    r, _ = simulate_garch(20000, 3e-4, 2e-6, 0.03, 0.90, gamma=0.10, nu=6.0, rng=np.random.default_rng(12))
    fit = fit_garch(r, gjr=True, dist="t")
    assert abs(fit.alpha - 0.03) < 0.012 and abs(fit.gamma - 0.10) < 0.025
    assert abs(fit.beta - 0.90) < 0.02 and abs(fit.nu - 6.0) < 0.7


def test_filter_equals_explicit_recursion():
    rng = np.random.default_rng(3)
    eps = rng.standard_normal(300)
    omega, alpha, gamma, beta = 0.05, 0.04, 0.09, 0.88
    b = backcast(eps)
    s2 = np.empty(301)
    s2[0] = omega + (alpha + 0.5 * gamma + beta) * b
    for t in range(1, 301):
        s2[t] = omega + (alpha + gamma * (eps[t - 1] < 0)) * eps[t - 1] ** 2 + beta * s2[t - 1]
    np.testing.assert_allclose(garch_variance(eps, omega, alpha, beta, gamma, b), s2, rtol=1e-12)


def test_student_t_likelihood_and_moments():
    x = np.linspace(-3, 3, 7)
    nu, s2 = 5.0, 1.7
    scale = np.sqrt(s2 * (nu - 2) / nu)   # unit-variance t scaled to variance s2
    np.testing.assert_allclose(loglik(x, np.full(7, s2), nu), student_t.logpdf(x, nu, scale=scale).sum(), rtol=1e-12)
    np.testing.assert_allclose(loglik(x, np.full(7, s2), 1e7), loglik(x, np.full(7, s2)), rtol=1e-6)
    inn = StudentTInnov(nu)
    c = np.sqrt((nu - 2) / nu)
    var = quad(lambda y: (c * y) ** 2 * student_t.pdf(y, nu), -np.inf, np.inf)[0]
    assert abs(var - 1.0) < 1e-8
    for a in (0.975, 0.99):
        q = student_t.ppf(a, nu)
        es = quad(lambda y: c * y * student_t.pdf(y, nu), q, np.inf)[0] / (1 - a)
        assert abs(inn.es(a) - es) < 1e-8
        zq = NormalInnov().ppf(a)
        es_n = quad(lambda y: y * np.exp(-0.5 * y * y) / np.sqrt(2 * np.pi), zq, np.inf)[0] / (1 - a)
        assert abs(NormalInnov().es(a) - es_n) < 1e-8
