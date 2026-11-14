"""Backtest statistics against hand-computed values, and their size under the null."""
import math

import numpy as np
from scipy.stats import norm

from riskmodels import backtest as bt


def test_kupiec_hand_example():
    """5 exceptions in 250 days at 99%: LR = -2 [245 ln(0.99/0.98) + 5 ln(0.01/0.02)] = 1.9568."""
    hits = np.zeros(250, dtype=bool)
    hits[[10, 50, 90, 130, 170]] = True
    lr, p = bt.kupiec(hits, 0.01)
    assert abs(lr - (-2 * (245 * math.log(0.99 / 0.98) + 5 * math.log(0.5)))) < 1e-10
    assert abs(lr - 1.9568) < 1e-4 and abs(p - 0.1618) < 1e-3
    assert abs(bt.kupiec(np.zeros(250, dtype=bool), 0.01)[0] + 2 * 250 * math.log(0.99)) < 1e-12   # no hits


def test_christoffersen_hand_example():
    h = np.zeros(20, dtype=int)
    h[[3, 4, 10]] = 1          # transitions: n00 = 14, n01 = 2, n10 = 2, n11 = 1
    out = bt.christoffersen(h, 0.05)
    assert (out["n00"], out["n01"], out["n10"], out["n11"]) == (14, 2, 2, 1)
    ll0 = 16 * math.log(16 / 19) + 3 * math.log(3 / 19)
    ll1 = 14 * math.log(14 / 16) + 2 * math.log(2 / 16) + 2 * math.log(2 / 3) + 1 * math.log(1 / 3)
    assert abs(out["lr_ind"] - (-2 * (ll0 - ll1))) < 1e-10
    lr_uc = -2 * (17 * math.log(0.95) + 3 * math.log(0.05) - 17 * math.log(17 / 20) - 3 * math.log(3 / 20))
    assert abs(out["lr_cc"] - (lr_uc + out["lr_ind"])) < 1e-10


def test_coverage_tests_have_nominal_size():
    rng = np.random.default_rng(42)
    hits = rng.random((2000, 1500)) < 0.025
    p_uc = np.array([bt.kupiec(h, 0.025)[1] for h in hits])
    cc = [bt.christoffersen(h, 0.025) for h in hits]
    rate = {"uc": np.mean(p_uc < 0.05), "ind": np.mean(np.array([c["p_ind"] for c in cc]) < 0.05),
            "cc": np.mean(np.array([c["p_cc"] for c in cc]) < 0.05)}
    assert 0.03 < rate["uc"] < 0.075 and 0.03 < rate["cc"] < 0.075
    # With about 37 hits a hit-after-hit is rare, so the chi2(1) approximation for the
    # independence statistic is conservative: it rejects less often than 5%.
    assert 0.005 < rate["ind"] < 0.06


def test_independence_test_detects_clustering():
    rng = np.random.default_rng(1)
    h = np.zeros(3000, dtype=int)
    for t in range(1, 3000):                      # Markov chain: P(hit | hit) = 0.3, else 0.01
        h[t] = rng.random() < (0.30 if h[t - 1] else 0.01)
    assert bt.christoffersen(h, h.mean())["p_ind"] < 1e-4


def test_rolling_exceedances_and_zones():
    h = np.zeros(600, dtype=int)
    h[[5, 100, 101, 102, 103, 104, 400]] = 1
    counts = bt.rolling_exceedances(h, 250)
    assert np.array_equal(counts, [h[i:i + 250].sum() for i in range(351)])
    assert bt.basel_zones(np.array([0, 4, 5, 9, 10]))["red"] == 0.2


def test_z2_mean_zero_and_size_under_the_null():
    rng = np.random.default_rng(8)
    a, T = 0.975, 750
    sigma = np.exp(0.3 * rng.standard_normal(T))             # time-varying, known scale
    var, es = sigma * norm.ppf(a), sigma * norm.pdf(norm.ppf(a)) / (1 - a)
    null = np.sort(bt.z2_null(lambda i, u: sigma[i] * norm.ppf(u), es, a, 4000, rng))
    assert abs(null.mean()) < 0.01
    losses = sigma * rng.standard_normal((2000, T))
    z2 = np.array([bt.z2_statistic(L, var, es, a) for L in losses])
    p = np.searchsorted(null, z2, side="right") / len(null)
    assert 0.03 < np.mean(p < 0.05) < 0.07
    # ES understated by 20% (true scale 1.2 sigma): rejected most of the time
    z2_bad = np.array([bt.z2_statistic(1.2 * L, var, es, a) for L in losses[:500]])
    assert np.mean(np.searchsorted(null, z2_bad, side="right") / len(null) < 0.05) > 0.6


def test_mcneil_frey_size_and_power():
    rng = np.random.default_rng(4)
    a, T = 0.975, 4000
    var, es = np.full(T, norm.ppf(a)), np.full(T, norm.pdf(norm.ppf(a)) / (1 - a))
    p = np.array([bt.mcneil_frey(rng.standard_normal(T), var, es, 400, rng)["p"] for _ in range(400)])
    assert 0.02 < np.mean(p < 0.05) < 0.09
    heavy = rng.standard_t(3, T) / np.sqrt(3)                 # unit variance, fatter tail
    assert bt.mcneil_frey(heavy, np.full(T, np.quantile(heavy, a)), es, 2000, rng)["p"] < 0.01


def test_scores_are_minimised_by_the_truth():
    rng = np.random.default_rng(6)
    L = rng.standard_normal(1_000_000)
    a = 0.975
    q, e = norm.ppf(a), norm.pdf(norm.ppf(a)) / (1 - a)
    base_pin = bt.pinball_loss(L, q, a).mean()
    base_fz = bt.fz0_loss(L, q, e, a).mean()
    for d in (-0.2, 0.2):
        assert bt.pinball_loss(L, q + d, a).mean() > base_pin
        assert bt.fz0_loss(L, q + d, e, a).mean() > base_fz
        assert bt.fz0_loss(L, q, e + d, a).mean() > base_fz
    # FZ0 differences are invariant to the unit of the loss
    d1 = bt.fz0_loss(L, q, e, a) - bt.fz0_loss(L, q + 0.2, e + 0.1, a)
    d2 = bt.fz0_loss(100 * L, 100 * q, 100 * e, a) - bt.fz0_loss(100 * L, 100 * (q + 0.2), 100 * (e + 0.1), a)
    np.testing.assert_allclose(d1, d2, atol=1e-9)


def test_diebold_mariano():
    rng = np.random.default_rng(2)
    d = rng.standard_normal(500) + 0.1
    out = bt.diebold_mariano(d, lag=0)                        # lag 0 is the plain z-statistic
    assert abs(out["dm"] - d.mean() / (d.std() / math.sqrt(500))) < 1e-12
    stats = np.array([bt.diebold_mariano(x)["dm"] for x in rng.standard_normal((2000, 800))])
    assert 0.03 < np.mean(np.abs(stats) > 1.96) < 0.08
