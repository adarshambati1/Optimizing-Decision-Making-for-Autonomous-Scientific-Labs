"""The autonomous-lab POMDP: config, ground-truth simulator and agent state.

  hidden state  s = (f, c): true objective of every pool candidate (static)
                and remaining budget c (fully observed)
  actions       measure(k) | commit(k) | abandon
  transitions   f static; c -= cost on measure; commit / abandon are terminal
  observations  fail (prob q, no information), or
                y = f_k + eps,  eps ~ N(0, sigma^2)                  w.p. 1 - rho
                y = f_k + eps,  eps ~ N(0, sigma^2 + sigma_out^2)    w.p. rho
  reward        measure: -cost ; commit(k): V 1[f_k >= tau] - C_commit ; abandon: 0
  belief        GP posterior b = (mu, Sigma) over f

`LabTask` is the simulator: it knows f and scores commits with it.
`AgentState` is everything the agent knows. The agent never sees f.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .datasets import Pool
from .gp import GaussianBelief, GPHyper, default_hyper, fit_hyper, is_good_reading


@dataclass
class TaskConfig:
    # observation model
    noise_scale: float = 1.0        # sigma = noise_scale * replicate sd of the dataset
    outlier_prob: float = 0.05      # rho
    outlier_sd: float = 3.0         # sigma_out: extra spread of a spurious reading
    fail_prob: float = 0.10         # q
    # economics (payoff units)
    cost: float = 0.01              # cost per measurement
    budget: float = 0.5             # total budget C (50 experiments)
    payoff: float = 1.0             # V, paid if the committed candidate meets the spec
    commit_cost: float = 0.10       # C_commit: scale-up / validation cost paid on commit
    tau_quantile: float = 0.95      # spec tau = this quantile of the pool ("top 5%")
    # protocol
    n_init: int = 3                 # random initial measurements, shared by all policies
    refit_every: int = 5            # refit GP hyperparameters every n successful readings
    min_fit_points: int = 5         # default hyperparameters until this many readings

    def with_(self, **kw) -> "TaskConfig":
        return replace(self, **kw)


class LabTask:
    """Ground-truth simulator for one episode (one lab campaign)."""

    def __init__(self, pool: Pool, cfg: TaskConfig):
        self.pool, self.cfg = pool, cfg
        self.f = pool.f
        self.noise_sd = cfg.noise_scale * pool.replicate_sd
        self.tau = pool.quantile(cfg.tau_quantile)

    def measure(self, k: int, rng: np.random.Generator) -> tuple[bool, float | None]:
        """Returns (failed, reading)."""
        if rng.random() < self.cfg.fail_prob:
            return True, None
        sd = self.noise_sd
        if rng.random() < self.cfg.outlier_prob:
            sd = np.sqrt(sd ** 2 + self.cfg.outlier_sd ** 2)
        return False, float(self.f[k] + sd * rng.standard_normal())

    def commit_reward(self, k: int) -> float:
        return self.cfg.payoff * float(self.f[k] >= self.tau) - self.cfg.commit_cost

    def oracle_value(self) -> float:
        """Knows f, makes no measurements, commits to the best candidate."""
        return max(0.0, self.commit_reward(int(np.argmax(self.f))))


@dataclass
class AgentModel:
    """The parts of the POMDP the agent knows."""
    noise_var: float
    outlier_prob: float
    outlier_var: float
    fail_prob: float
    cost: float
    budget: float
    tau: float
    payoff: float
    commit_cost: float
    refit_every: int
    min_fit_points: int

    @classmethod
    def from_task(cls, task: LabTask) -> "AgentModel":
        c = task.cfg
        return cls(noise_var=task.noise_sd ** 2, outlier_prob=c.outlier_prob,
                   outlier_var=c.outlier_sd ** 2, fail_prob=c.fail_prob, cost=c.cost,
                   budget=c.budget, tau=task.tau, payoff=c.payoff, commit_cost=c.commit_cost,
                   refit_every=c.refit_every, min_fit_points=c.min_fit_points)


class AgentState:
    """Belief state + history: everything a policy is allowed to look at."""

    def __init__(self, X: np.ndarray, model: AgentModel, rng: np.random.Generator):
        self.X, self.model, self.rng = X, model, rng
        self.obs: list[tuple[int, float]] = []   # successful readings (k, y)
        self.good: list[bool] = []               # was each reading accepted (not an outlier)?
        self.n_attempts = 0
        self.n_fail = 0
        self.spent = 0.0
        self.tested = np.zeros(len(X), bool)
        self.hyper: GPHyper = default_hyper(X.shape[1])
        self.belief = GaussianBelief.prior(X, self.hyper)
        self._since_refit = 0

    @property
    def remaining(self) -> float:
        return self.model.budget - self.spent

    def measurable(self) -> np.ndarray:
        """Untested candidates we can still afford."""
        if self.model.cost > self.remaining + 1e-12:
            return np.zeros(len(self.X), bool)
        return ~self.tested

    def observe(self, k: int, failed: bool, y: float | None) -> None:
        self.spent += self.model.cost
        self.tested[k] = True
        self.n_attempts += 1
        if failed:  # cost paid, belief unchanged
            self.n_fail += 1
            return
        self.obs.append((k, y))
        self._since_refit += 1
        if len(self.obs) == self.model.min_fit_points or self._since_refit >= self.model.refit_every:
            self.refit()
        else:
            self._absorb(k, y)

    def _absorb(self, k: int, y: float) -> None:
        m, b = self.model, self.belief
        ok = is_good_reading(b.mu[k], b.Sigma[k, k], y, m.noise_var, m.outlier_prob, m.outlier_var)
        self.good.append(ok)
        if ok:
            b.update(k, y, m.noise_var)

    def refit(self) -> None:
        """Refit GP hyperparameters on accepted readings, then replay all readings."""
        m = self.model
        self._since_refit = 0
        good = self.good + [True] * (len(self.obs) - len(self.good))
        ks = np.array([k for (k, _), g in zip(self.obs, good) if g], int)
        ys = np.array([y for (_, y), g in zip(self.obs, good) if g])
        if len(ks) >= m.min_fit_points:
            self.hyper = fit_hyper(self.X[ks], ys, m.noise_var, self.rng, prev=self.hyper)
        self.belief = GaussianBelief.prior(self.X, self.hyper)
        self.good = []
        for k, y in self.obs:
            self._absorb(k, y)
