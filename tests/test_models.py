"""Forecast engines: no look-ahead, estimator definitions, EWMA recursion, Euler identities."""
import numpy as np
from scipy.stats import norm

from riskmodels.decomp import ewma_covariance, historical_es_contributions, parametric_shares
from riskmodels.models import MODEL_NAMES, EmpiricalInnov, ewma_variance, run_all_models

KW = dict(start=300, window=400, refit_every=20)


def test_no_look_ahead(real_returns):
    """The forecast for day t is unchanged by truncating the sample after t or corrupting day t onwards."""
    r = real_returns[700:1300]
    cut = 517                                             # not a re-estimation day
    full, _ = run_all_models(r, **KW)
    truncated, _ = run_all_models(r[:cut + 1], **KW)
    corrupted = r.copy()
    corrupted[cut:] = np.random.default_rng(0).normal(0, 0.05, len(r) - cut)
    garbage, _ = run_all_models(corrupted, **KW)
    k = cut - KW["start"] + 1
    for name in MODEL_NAMES:
        for a, measure in ((0.99, "var"), (0.975, "var"), (0.975, "es")):
            ref = getattr(full[name], measure)(a)
            np.testing.assert_array_equal(getattr(truncated[name], measure)(a), ref[:k])
            changed = getattr(garbage[name], measure)(a)
            np.testing.assert_array_equal(changed[:k], ref[:k])
            assert not np.array_equal(changed[k:], ref[k:])   # the corruption does move later forecasts


def test_forecasts_are_ordered_and_consistent(real_returns):
    r = real_returns[700:1300]
    fc, fits = run_all_models(r, **KW)
    for name in MODEL_NAMES:
        v99, v975, es = fc[name].var(0.99), fc[name].var(0.975), fc[name].es(0.975)
        assert np.all(v99 >= v975) and np.all(es > v975) and np.all(v975 > 0)
        u = np.array([0.98, 0.99, 0.999])
        assert np.all(np.diff(fc[name].tail_ppf(5, u)) >= 0)
        assert abs(fc[name].tail_ppf(5, np.array([0.99]))[0] - v99[5]) < 1e-12
    g = fc["GARCH-N"]
    np.testing.assert_allclose(g.var(0.99), g.loc + g.scale * norm.ppf(0.99))
    np.testing.assert_array_equal(fc["FHS"].scale, g.scale)   # shared volatility filter
    np.testing.assert_array_equal(fc["EVT"].scale, g.scale)
    assert len(fits["garch"]) == 15
    # the day after a refit, sigma follows the one-step GARCH update with the new parameters
    f = fits["garch"][1]
    t = KW["start"] + 20
    s2 = f.omega + f.alpha * (r[t] - f.mu) ** 2 + f.beta * g.scale[20] ** 2
    assert abs(g.scale[21] - np.sqrt(s2)) < 1e-15


def test_historical_simulation_definitions():
    rng = np.random.default_rng(3)
    r = rng.standard_normal(260) * 0.01
    fc, _ = run_all_models(np.concatenate([r, [0.0]]), start=260, window=1000, refit_every=20)
    losses = np.sort(-r[-250:])
    assert fc["HS-250"].var(0.99)[0] == losses[-3]            # 3rd worst of 250
    assert fc["HS-250"].var(0.975)[0] == losses[-7]           # 7th worst of 250
    es = (losses[-6:].sum() + 0.25 * losses[-7]) / 6.25       # 6.25 tail observations
    assert abs(fc["HS-250"].es(0.975)[0] - es) < 1e-15
    assert fc["HS-1000"].var(0.99)[0] == np.sort(-r)[-3]      # 260 available: ceil(257.4) = 258th


def test_empirical_es_is_the_average_quantile():
    x = np.sort(np.random.default_rng(1).standard_normal(1000))
    inn = EmpiricalInnov(x)
    u = 0.975 + 0.025 * (np.arange(200000) + 0.5) / 200000    # midpoint rule on (a, 1)
    assert abs(inn.es(0.975) - inn.ppf(u).mean()) < 1e-6
    assert inn.ppf(0.975) == x[974] and abs(inn.es(0.975) - x[975:].mean()) < 1e-12


def test_ewma_recursion_equals_weighted_sum():
    r = np.random.default_rng(5).standard_normal(400) * 0.01
    lam, t = 0.94, 399
    s2 = ewma_variance(r, lam)
    k = np.arange(1, t)
    direct = lam ** (t - 1) * r[0] ** 2 + (1 - lam) * np.sum(lam ** (k - 1) * r[t - k] ** 2)
    assert abs(s2[t] / direct - 1) < 1e-12
    assert np.isnan(s2[0]) and s2[1] == r[0] ** 2


def test_euler_contributions_sum_to_total():
    rng = np.random.default_rng(7)
    mix = np.array([[1, .5, 0, 0], [0, 1, .2, 0], [0, 0, .4, -.1], [0, 0, 0, .3]])
    R = rng.standard_normal((1500, 4)) @ mix * 0.01
    w = np.array([0.4, 0.3, 0.2, 0.1])
    S = ewma_covariance(R)
    shares = parametric_shares(w, S[1:])
    np.testing.assert_allclose(shares.sum(axis=1), 1.0, atol=1e-12)
    # the portfolio variance from the covariance equals the univariate EWMA of the portfolio return
    np.testing.assert_allclose(np.einsum("i,tij,j->t", w, S[1:], w), ewma_variance(R @ w)[1:], rtol=1e-10)

    # component = w_i d sigma_p / d w_i, checked by central differences
    def sigma(v):
        return np.sqrt(v @ S[-1] @ v)

    grad = np.array([(sigma(w + 1e-6 * e) - sigma(w - 1e-6 * e)) / 2e-6 for e in np.eye(4)])
    np.testing.assert_allclose(w * grad / sigma(w), shares[-1], rtol=1e-7)
    contrib, es = historical_es_contributions(R, w, 0.975)
    assert abs(contrib.sum() - es) < 1e-15
    assert abs(es - EmpiricalInnov(np.sort(-(R @ w))).es(0.975)) < 1e-15

    # historical ES contribution = w_i dES/dw_i (the tail set is locally constant in w)
    def es_of(v):
        return EmpiricalInnov(np.sort(-(R @ v))).es(0.975)

    grad = np.array([(es_of(w + 1e-7 * e) - es_of(w - 1e-7 * e)) / 2e-7 for e in np.eye(4)])
    np.testing.assert_allclose(w * grad, contrib, rtol=1e-5)
