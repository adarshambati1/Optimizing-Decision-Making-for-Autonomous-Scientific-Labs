"""The offline learners: behavioral cloning, batch Q-learning, and model estimation.

All of them learn from the fixed log written by logger.py and never run a new
experiment while learning. The two model-free learners produce a stopping
policy phi -> P(continue | phi) over the 8 belief features; the model
estimators produce the parameters of the antonio_tests/ abstraction, which a
planner then solves.

    BehavioralCloning     ADM 18.1   maximize the likelihood of an expert's logged decisions
    BatchQLearning        ADM 17.2 + 17.6 + 17.7: gradient Q-learning on a linear basis
                          (8.6), replaying minibatches from the fixed buffer
    FittedQ               the same Bellman backup, solved by least squares on the whole
                          buffer each sweep (a stable batch variant of 17.2 with 8.6)
    estimate_reading_model    ADM 16.1 maximum-likelihood counts, and 16.4 Beta posteriors
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import beta as beta_dist

from logger import FEAT_COLS, NEXT_COLS


# --------------------------------------------------------------------- basis --
class QuadraticBasis:
    """beta(phi): standardized features, their squares and pairwise products, and a bias.

    A linear model in beta is a quadratic in the features (ADM 8.6, polynomial basis).
    Standardization statistics are fitted once, on the training rows.
    """

    def __init__(self, degree: int = 2):
        self.degree = degree

    def fit(self, Phi: np.ndarray) -> "QuadraticBasis":
        X = Phi[:, 1:]                                   # drop the logged bias column
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        return self

    def __call__(self, Phi: np.ndarray) -> np.ndarray:
        Z = (np.atleast_2d(Phi)[:, 1:] - self.mu) / self.sd
        cols = [np.ones((len(Z), 1)), Z]
        if self.degree >= 2:
            iu = np.triu_indices(Z.shape[1])
            cols.append((Z[:, :, None] * Z[:, None, :])[:, iu[0], iu[1]])
        return np.hstack(cols)

    @property
    def dim(self) -> int:
        k = len(self.mu)
        return 1 + k + (k * (k + 1) // 2 if self.degree >= 2 else 0)


# ------------------------------------------------------- behavioral cloning --
class BehavioralCloning:
    """pi_theta(continue | phi) = sigmoid(theta . beta(phi)), fit by maximum likelihood (18.1),
    with a small L2 penalty. Labels are the expert's own (unrandomized) decisions."""

    def __init__(self, l2: float = 1e-3, degree: int = 2):
        self.l2, self.basis = l2, QuadraticBasis(degree)

    def fit(self, Phi: np.ndarray, y: np.ndarray) -> "BehavioralCloning":
        B = self.basis.fit(Phi)(Phi)
        y = np.asarray(y, float)

        def nll(th):
            z = B @ th
            p = expit(z)
            f = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)) + self.l2 * th[1:] @ th[1:]
            g = B.T @ (p - y) / len(y)
            g[1:] += 2 * self.l2 * th[1:]
            return f, g

        self.theta = minimize(nll, np.zeros(B.shape[1]), jac=True, method="L-BFGS-B").x
        return self

    def prob(self, Phi):
        return expit(self.basis(Phi) @ self.theta)

    def policy(self, Phi):
        """Deterministic, as deployed: continue iff P(continue) > 0.5."""
        return (self.prob(Phi) > 0.5).astype(float)


def expert_labels(tr: pd.DataFrame, behavior: str, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """The expert's own decisions in its campaigns: undo the eps mixing of the logged
    probability, p_base = (p_continue - eps/2) / (1 - eps). Forced stops are skipped."""
    d = tr[(tr.behavior == behavior) & (tr.forced == 0)]
    p_base = (d.p_continue.to_numpy() - eps / 2) / (1 - eps)
    return d[FEAT_COLS].to_numpy(), (p_base > 0.5).astype(float)


# ---------------------------------------------------- batch Q-learning --
def _arrays(tr: pd.DataFrame):
    tr = tr.sort_values(["campaign", "t"], kind="stable")
    return (tr[FEAT_COLS].to_numpy(), tr.action.to_numpy().astype(int), tr.reward.to_numpy(),
            tr[NEXT_COLS].to_numpy(), tr.done.to_numpy().astype(bool),
            # the next state may force a stop (nothing affordable): budget_left feature ~ 0
            tr.groupby("campaign").forced.shift(-1).fillna(0).to_numpy().astype(bool))


class _LinearQ:
    """Q_theta(phi, a) = theta_a . beta(phi), a in {0 = stop, 1 = continue}. gamma = 1:
    every campaign ends within the budget."""

    def q(self, Phi):
        B = self.basis(Phi)
        return B @ self.theta.T                       # (N, 2)

    def policy(self, Phi):
        Q = self.q(Phi)
        return (Q[:, 1] > Q[:, 0]).astype(float)

    def _next_value(self, Phi2, done, next_forced):
        Q2 = self.q(Phi2)
        v = np.where(next_forced, Q2[:, 0], Q2.max(1))  # a forced stop can only stop
        return np.where(done, 0.0, v)


class BatchQLearning(_LinearQ):
    """Gradient Q-learning (ADM 17.6, eq. 17.21) with experience replay (17.7).

    For each minibatch drawn uniformly from the FIXED buffer:
        theta_a += alpha * mean[(r + max_a' Q(s', a') - Q(s, a)) * beta(s)]  over rows with action a
    The book's update scales the gradient to unit norm at most; the same is done here.
    """

    def __init__(self, alpha: float = 0.05, batch: int = 256, n_updates: int = 40_000, degree: int = 2,
                 seed: int = 0, alpha_decay: float = 0.9999):
        self.alpha, self.batch, self.n_updates = alpha, batch, n_updates
        self.basis, self.rng, self.alpha_decay = QuadraticBasis(degree), np.random.default_rng(seed), alpha_decay

    def fit(self, tr: pd.DataFrame) -> "BatchQLearning":
        Phi, A, R, Phi2, done, nf = _arrays(tr)
        self.basis.fit(Phi)
        B, B2 = self.basis(Phi), self.basis(Phi2)
        self.theta = np.zeros((2, B.shape[1]))
        alpha = self.alpha
        self.td_trace = []
        for it in range(self.n_updates):
            idx = self.rng.integers(0, len(B), self.batch)
            Q2 = B2[idx] @ self.theta.T
            v2 = np.where(done[idx], 0.0, np.where(nf[idx], Q2[:, 0], Q2.max(1)))
            td = R[idx] + v2 - (B[idx] * self.theta[A[idx]]).sum(1)
            for a in (0, 1):
                m = A[idx] == a
                if m.any():
                    g = (td[m, None] * B[idx][m]).mean(0)
                    n = np.linalg.norm(g)
                    self.theta[a] += alpha * (g / n if n > 1 else g)
            alpha *= self.alpha_decay
            if it % 1000 == 0:
                self.td_trace.append(float(np.mean(td ** 2)))
        return self


class FittedQ(_LinearQ):
    """Batch Bellman backups by least squares: theta_a = argmin sum_{rows with a} (y - theta_a . beta)^2
    + ridge, with targets y = r + max_a' Q_k(s', a'), repeated until the policy stops changing."""

    def __init__(self, n_iter: int = 60, ridge: float = 1e-3, degree: int = 2):
        self.n_iter, self.ridge, self.basis = n_iter, ridge, QuadraticBasis(degree)

    def fit(self, tr: pd.DataFrame) -> "FittedQ":
        Phi, A, R, Phi2, done, nf = _arrays(tr)
        self.basis.fit(Phi)
        B = self.basis(Phi)
        self.theta = np.zeros((2, B.shape[1]))
        self.change = []
        for _ in range(self.n_iter):
            y = R + self._next_value(Phi2, done, nf)
            new = np.zeros_like(self.theta)
            for a in (0, 1):
                m = A == a
                Ba = B[m]
                new[a] = np.linalg.solve(Ba.T @ Ba + self.ridge * len(Ba) * np.eye(B.shape[1]), Ba.T @ y[m])
            self.change.append(float(np.abs(new - self.theta).max()))
            self.theta = new
        return self


# ---------------------------------------------- model estimation (16.1, 16.4) --
NOISE_BINS = [0.0, 0.25, 0.35, 0.45, 0.6, 1.0]


def reading_counts(me: pd.DataFrame, ca: pd.DataFrame, noise_lo: float, noise_hi: float) -> dict:
    """Counts of readings from logged experiments (initial design included: same physics),
    restricted to pools whose noise level lies in [noise_lo, noise_hi)."""
    camp = ca.set_index("campaign")
    nz = camp.noise_sd.loc[me.campaign].to_numpy()
    m = me[(nz >= noise_lo) & (nz < noise_hi)]
    ok = m[m.failed == 0]
    good, bad = ok[ok.meets_spec == 1], ok[ok.meets_spec == 0]
    return dict(n=len(m), n_fail=int(m.failed.sum()),
                good_high=int((good.high == 1).sum()), good_n=len(good),
                bad_high=int((bad.high == 1).sum()), bad_n=len(bad))


def estimate_reading_model(me, ca, noise_sd: float, kind: str = "mle", q: float = 0.05) -> dict:
    """Parameters of the abstraction's observation model for a pool with this noise level.

    mle          ADM 16.1: N(high | class) / N(class), N(fail) / N
    bayes_mean   ADM 16.4: Beta(1, 1) prior, posterior mean (N_high + 1) / (N + 2)
    bayes_pess   the same posterior, read pessimistically about how informative a
                 reading is: P(high | good) at its q-quantile, P(high | not good) at
                 its (1 - q)-quantile, P(fail) at its (1 - q)-quantile
    "not good" pools every candidate below spec (near misses and the rest), the
    population a shortlisted candidate is actually drawn from in the log.
    """
    i = int(np.clip(np.searchsorted(NOISE_BINS, noise_sd, side="right") - 1, 0, len(NOISE_BINS) - 2))
    c = reading_counts(me, ca, NOISE_BINS[i], NOISE_BINS[i + 1])
    a_f, b_f = c["n_fail"], c["n"] - c["n_fail"]
    a_g, b_g = c["good_high"], c["good_n"] - c["good_high"]
    a_b, b_b = c["bad_high"], c["bad_n"] - c["bad_high"]
    if kind == "mle":
        out = dict(fail_prob=a_f / c["n"], p_high_good=a_g / c["good_n"], p_high_bad=a_b / c["bad_n"])
    elif kind == "bayes_mean":
        out = dict(fail_prob=(a_f + 1) / (c["n"] + 2), p_high_good=(a_g + 1) / (c["good_n"] + 2),
                   p_high_bad=(a_b + 1) / (c["bad_n"] + 2))
    elif kind == "bayes_pess":
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # scipy's boost backend warns spuriously here
            out = dict(fail_prob=float(beta_dist.ppf(1 - q, a_f + 1, b_f + 1)),
                       p_high_good=float(beta_dist.ppf(q, a_g + 1, b_g + 1)),
                       p_high_bad=float(beta_dist.ppf(1 - q, a_b + 1, b_b + 1)))
        assert all(np.isfinite(v) and 0 < v < 1 for v in out.values()), out
    else:
        raise ValueError(kind)
    out.update(noise_bin=(NOISE_BINS[i], NOISE_BINS[i + 1]), counts=c)
    return out
