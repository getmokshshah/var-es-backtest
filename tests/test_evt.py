"""GPD maximum likelihood and the McNeil-Frey tail formulas."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import genpareto, t as student_t

from riskmodels.evt import GPDTail, fit_gpd, fit_gpd_tail


@pytest.mark.parametrize("xi,beta", [(0.25, 0.6), (0.0, 1.0), (-0.2, 2.0), (0.5, 0.3)])
def test_gpd_mle_recovers_parameters(xi, beta):
    y = genpareto.rvs(xi, scale=beta, size=40000, random_state=np.random.default_rng(5))
    xi_hat, beta_hat = fit_gpd(y)
    assert abs(xi_hat - xi) < 0.025 and abs(beta_hat / beta - 1) < 0.03


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gpd_mle_matches_scipy(seed):
    y = genpareto.rvs(0.15, scale=0.7, size=100, random_state=np.random.default_rng(seed))
    xi_hat, beta_hat = fit_gpd(y)
    xi_ref, _, beta_ref = genpareto.fit(y, floc=0)
    np.testing.assert_allclose([xi_hat, beta_hat], [xi_ref, beta_ref], rtol=2e-3, atol=2e-4)


def test_tail_formulas_exact_for_pareto():
    """Pareto(a): P(X > x) = x^-a, x >= 1. Excesses over u are GPD(xi = 1/a, beta = u/a) exactly."""
    a, u = 4.0, 1.8
    tail = GPDTail(u=u, xi=1 / a, beta=u / a, p_u=u ** -a)
    for q in (0.975, 0.99, 0.999):
        var = (1 - q) ** (-1 / a)
        assert abs(tail.ppf(q) - var) < 1e-12
        assert abs(tail.es(q) - a / (a - 1) * var) < 1e-12
        assert abs(tail.sf(var) - (1 - q)) < 1e-15


def test_tail_formulas_match_numerical_integration():
    tail = GPDTail(u=1.2, xi=0.2, beta=0.6, p_u=0.1)
    q = 0.975
    var = tail.ppf(q)

    def density(x):
        return tail.p_u / tail.beta * (1 + tail.xi * (x - tail.u) / tail.beta) ** (-1 / tail.xi - 1)

    es = quad(lambda x: x * density(x), var, np.inf)[0] / (1 - q)
    assert abs(tail.es(q) - es) < 1e-7
    zero = GPDTail(u=1.2, xi=1e-12, beta=0.6, p_u=0.1)    # exponential limit
    near = GPDTail(u=1.2, xi=1e-6, beta=0.6, p_u=0.1)
    assert abs(zero.ppf(q) - near.ppf(q)) < 1e-5 and abs(zero.es(q) - near.es(q)) < 1e-5


def test_pot_estimate_close_to_true_student_t_tail():
    nu = 4.0
    x = student_t.rvs(nu, size=200000, random_state=np.random.default_rng(9))
    tail = fit_gpd_tail(x, 0.05)
    assert abs(tail.xi - 1 / nu) < 0.08    # finite-threshold bias: the t tail is only asymptotically GPD
    for q in (0.99, 0.999):
        true_q = student_t.ppf(q, nu)
        true_es = quad(lambda y: y * student_t.pdf(y, nu), true_q, np.inf)[0] / (1 - q)
        assert abs(tail.ppf(q) / true_q - 1) < 0.03
        assert abs(tail.es(q) / true_es - 1) < 0.05
