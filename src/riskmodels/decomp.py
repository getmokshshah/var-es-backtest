"""Euler (component) risk contributions by asset.

For a risk measure ``rho(w)`` that is homogeneous of degree one in the weights,
Euler's theorem gives ``rho(w) = sum_i w_i d rho / d w_i``; the summands are the
component contributions (Tasche 1999).

Parametric (zero mean, covariance ``S``): ``sigma_p = sqrt(w'Sw)``, VaR and ES are
both multiples of ``sigma_p``, so every such measure has the same shares

    share_i = w_i (S w)_i / (w' S w).

Historical ES: ``d ES / d w_i = E[ -R_i | L >= VaR ]``, so the contribution of asset
``i`` is the average of its own loss ``-w_i R_i`` over the portfolio's tail days.
"""
from __future__ import annotations

import numpy as np

from .models import _EPS


def ewma_covariance(R: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """EWMA covariance forecasts ``S_t = lam S_{t-1} + (1-lam) R_{t-1} R_{t-1}'``, ``S_1 = R_0 R_0'``.

    Returns shape ``(T, N, N)``; element ``t`` uses rows ``0 .. t-1`` and element 0 is NaN.
    With this start ``w' S_t w`` equals the univariate EWMA variance of ``w'R`` exactly.
    """
    T, N = R.shape
    S = np.full((T, N, N), np.nan)
    S[1] = np.outer(R[0], R[0])
    for t in range(2, T):
        S[t] = lam * S[t - 1] + (1.0 - lam) * np.outer(R[t - 1], R[t - 1])
    return S


def parametric_shares(w: np.ndarray, S: np.ndarray) -> np.ndarray:
    """``w_i (S w)_i / (w'Sw)`` for one covariance or a stack of them; rows sum to one."""
    Sw = S @ w
    return w * Sw / (Sw @ w)[..., None]


def historical_es_contributions(R: np.ndarray, w: np.ndarray, a: float) -> tuple[np.ndarray, float]:
    """Component ES by asset from the empirical distribution, and the total ES.

    Uses the same weights on the order statistics as the empirical ES estimator
    (``models.EmpiricalInnov.es``), so the contributions sum to the ES exactly.
    """
    asset_loss = -R * w
    L = asset_loss.sum(axis=1)
    n = len(L)
    order = np.argsort(L)
    k = int(np.ceil(n * a - _EPS))
    weights = np.zeros(n)
    weights[order[k:]] = 1.0
    weights[order[k - 1]] = k - n * a
    weights /= n * (1.0 - a)
    contrib = weights @ asset_loss
    return contrib, float(contrib.sum())
