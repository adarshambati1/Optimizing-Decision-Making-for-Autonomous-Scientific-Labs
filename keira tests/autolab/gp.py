"""Gaussian-process belief over a finite candidate pool.

The belief b = (mu, Sigma) is a multivariate Gaussian over the K hidden
objective values f. Because the pool is finite, the GP posterior *is* this
K-dimensional Gaussian, and conditioning on one measurement is a rank-1
Kalman update:

    s  = Sigma[:, k]
    v  = Sigma[k, k] + sigma^2
    mu'    = mu + s (y - mu_k) / v
    Sigma' = Sigma - s s^T / v

Outliers: with probability rho a reading is a gross error,
y ~ N(f_k, sigma^2 + sigma_out^2). The agent computes P(reading is good | y)
and discards the reading if that is below 0.5.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

SQRT5 = np.sqrt(5.0)


def matern52(A: np.ndarray, B: np.ndarray, ls: np.ndarray, sf2: float) -> np.ndarray:
    """ARD Matern-5/2 kernel (the kernel Liang et al. found most robust)."""
    A = A / ls
    B = B / ls
    d2 = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T
    r = SQRT5 * np.sqrt(np.maximum(d2, 0.0))
    return sf2 * (1.0 + r + r * r / 3.0) * np.exp(-r)


@dataclass
class GPHyper:
    ls: np.ndarray   # (d,) lengthscales in normalized feature units
    sf2: float       # signal variance


def default_hyper(d: int) -> GPHyper:
    return GPHyper(ls=np.full(d, 0.5), sf2=1.0)


def fit_hyper(X: np.ndarray, y: np.ndarray, noise_var: float, rng: np.random.Generator,
              prev: GPHyper | None = None, n_restarts: int = 2) -> GPHyper:
    """MAP estimate of (lengthscales, signal variance).

    * The noise variance is not fitted: it is a known parameter of the
      observation model.
    * Informed prior: mean fixed at 0 and a tight prior on the signal variance
      around 1 (outputs are standardized), i.e. the lab knows the property's
      typical range. Without this, fits on 3-5 points are degenerate (e.g.
      sf2 = 0.5 -> "nothing can reach the spec" -> abandon at once).
    """
    n, d = X.shape
    eye = np.eye(n)
    log_ls0 = np.log(0.5)

    def neg_log_post(theta):
        ls, sf2 = np.exp(theta[:d]), np.exp(theta[d])
        Kxx = matern52(X, X, ls, sf2) + (noise_var + 1e-6) * eye
        try:
            L = np.linalg.cholesky(Kxx)
        except np.linalg.LinAlgError:
            return 1e10
        a = np.linalg.solve(L.T, np.linalg.solve(L, y))
        nll = 0.5 * y @ a + np.log(np.diag(L)).sum()
        prior = 0.5 * ((theta[:d] - log_ls0) ** 2).sum() + 0.5 * (theta[d] / 0.25) ** 2
        return nll + prior

    bounds = [(np.log(0.03), np.log(5.0))] * d + [(np.log(0.05), np.log(10.0))]
    starts = [np.r_[np.full(d, log_ls0), 0.0]]
    if prev is not None:
        starts.append(np.r_[np.log(prev.ls), np.log(prev.sf2)])
    else:
        for _ in range(n_restarts):
            starts.append(np.r_[rng.uniform(np.log(0.1), np.log(2.0), d), rng.uniform(-1, 1)])
    best = min((minimize(neg_log_post, s0, method="L-BFGS-B", bounds=bounds) for s0 in starts),
               key=lambda r: r.fun)
    return GPHyper(ls=np.exp(best.x[:d]), sf2=float(np.exp(best.x[d])))


class GaussianBelief:
    """Multivariate Gaussian belief over the pool's hidden objective values."""

    __slots__ = ("mu", "Sigma")

    def __init__(self, mu: np.ndarray, Sigma: np.ndarray):
        self.mu = mu
        self.Sigma = Sigma

    @classmethod
    def prior(cls, X: np.ndarray, hp: GPHyper) -> "GaussianBelief":
        K = matern52(X, X, hp.ls, hp.sf2)
        K[np.diag_indices_from(K)] += 1e-8
        return cls(np.zeros(len(X)), K)

    @property
    def var(self) -> np.ndarray:
        return np.maximum(np.diag(self.Sigma), 1e-12)

    def subset(self, idx: np.ndarray) -> "GaussianBelief":
        """Exact marginal over a subset of candidates."""
        return GaussianBelief(self.mu[idx].copy(), self.Sigma[np.ix_(idx, idx)].copy())

    def update(self, k: int, y: float, noise_var: float) -> None:
        """In-place rank-1 Kalman / GP update."""
        mu, Sigma = kalman_update(self.mu, self.Sigma, k, y, noise_var)
        self.mu, self.Sigma = mu, Sigma


def kalman_update(mu, Sigma, k, y, noise_var):
    s = Sigma[:, k].copy()
    v = s[k] + noise_var
    return mu + s * ((y - mu[k]) / v), Sigma - np.outer(s, s) / v


def is_good_reading(mu_k, var_k, y, noise_var, rho, out_var) -> bool:
    """P(reading is not an outlier | y) >= 0.5 under the current belief."""
    if rho <= 0:
        return True
    v_good = var_k + noise_var
    v_out = v_good + out_var
    r2 = (y - mu_k) ** 2
    log_good = np.log1p(-rho) - 0.5 * (np.log(v_good) + r2 / v_good)
    log_out = np.log(rho) - 0.5 * (np.log(v_out) + r2 / v_out)
    return bool(log_good >= log_out)
