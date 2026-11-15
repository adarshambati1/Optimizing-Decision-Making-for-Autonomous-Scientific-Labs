"""A learned stopping gate on top of the GP-EI baseline.

The agent makes one learned decision per step: run another experiment, or stop.

    continue -> the baseline's expected-improvement rule picks the experiment
    stop     -> the shared terminal rule (commit to the best candidate, or abandon)

The gate is a logistic policy over a few summary features of the belief:

    P(continue | s) = sigmoid(theta . phi(s))

It is trained by policy gradient (likelihood ratio with reward-to-go and
baseline subtraction; Algorithms for Decision Making, ch. 11) on SYNTHETIC
pools sampled from a GP prior, never on the three real datasets, so the real
pools are a held-out test.

There is no tree search and no lookahead anywhere in this agent.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KEIRA = os.path.join(os.path.dirname(HERE), "keira tests")
if KEIRA not in sys.path:
    sys.path.insert(0, KEIRA)

import numpy as np  # noqa: E402
from scipy.special import ndtr  # noqa: E402

from autolab.datasets import Pool  # noqa: E402
from autolab.gp import matern52  # noqa: E402
from autolab.mcts import target_ei  # noqa: E402
from autolab.policies import EIPolicy, Policy, terminal_action  # noqa: E402
from autolab.pomdp import AgentModel, AgentState, LabTask, TaskConfig  # noqa: E402
from autolab.utility import expected_commit_value  # noqa: E402

FEATURES = [
    "bias",
    "stop_value",          # value of stopping now: max(0, best expected commit value)
    "p_best",              # chance the best commit candidate meets spec
    "sd_best",             # posterior sd of that candidate
    "p_untested_max",      # best chance of meeting spec among untested candidates
    "target_ei_max",       # largest E[(f - tau)+] among untested candidates
    "budget_left",         # fraction of the budget remaining
    "voi_ratio",           # value of one more experiment / its cost (optional 8th input)
]
N_BASE = 7                 # gates trained with 7 weights do not use voi_ratio

_Z, _W = np.polynomial.hermite_e.hermegauss(15)
_W = _W / np.sqrt(2 * np.pi)


def voi_ratio(state: AgentState, n_top: int = 8, cap: float = 5.0) -> float:
    """One-step value of information, in units of the experiment cost.

    For each of the top untested candidates (by upper confidence bound), compute
    how much the value of stopping is expected to rise after one reading on it:

        VOI(k) = (1 - fail_prob) * ( E_y[ V_stop(belief after y) ] - V_stop(belief now) )

    The belief update is exact for a Gaussian belief (the posterior variance does
    not depend on y; the posterior mean shifts linearly in y), and the expectation
    over y is done by Gauss-Hermite quadrature. Outliers are ignored. Returns
    max_k VOI(k) / cost, capped. A ratio above 1 means some experiment is expected
    to pay for itself in one step; this is myopic, so it understates the value of
    experiments that only pay off in combination.
    """
    m, b = state.model, state.belief
    un = np.flatnonzero(state.measurable())
    if len(un) == 0:
        return 0.0
    mu, Sigma, var = b.mu, b.Sigma, b.var
    v0 = max(0.0, float(expected_commit_value(mu, var, m).max()))
    cand = un[np.argsort(-(mu[un] + 2.0 * np.sqrt(var[un])))[:n_top]]
    best = 0.0
    for k in cand:
        s = Sigma[:, k]
        v = s[k] + m.noise_var
        sd1 = np.sqrt(np.maximum(var - s * s / v, 1e-12))
        M = mu[None, :] + _Z[:, None] * (s / np.sqrt(v))[None, :]
        eu = m.payoff * ndtr((M - m.tau) / sd1[None, :]) - m.commit_cost
        vz = np.maximum(0.0, eu.max(axis=1))
        best = max(best, (1 - m.fail_prob) * float(_W @ vz - v0))
    return float(min(cap, max(0.0, best) / max(m.cost, 1e-12)))


def features(state: AgentState, n: int = N_BASE) -> np.ndarray:
    """The gate's inputs. n = 7 gives the base inputs; n = 8 adds voi_ratio."""
    m, b = state.model, state.belief
    mu, var = b.mu, b.var
    sd = np.sqrt(var)
    eu = expected_commit_value(mu, var, m)
    kb = int(np.argmax(eu))
    p = (eu + m.commit_cost) / m.payoff
    un = state.measurable()
    return np.array([
        1.0,
        max(0.0, float(eu[kb])),
        float(p[kb]),
        float(sd[kb]),
        float(p[un].max()) if un.any() else 0.0,
        float(target_ei(mu[un], sd[un], m.tau).max()) if un.any() else 0.0,
        state.remaining / m.budget,
    ] + ([voi_ratio(state)] if n > N_BASE else []))


def p_continue(theta: np.ndarray, phi: np.ndarray) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(theta @ phi, -30, 30))))


class GatePolicy(Policy):
    """EI picks the experiment; the learned gate decides whether to run it."""
    name = "gate"

    def __init__(self, theta, greedy: bool = True):
        self.theta = np.asarray(theta, float)
        self.greedy = greedy
        self._ei = EIPolicy()

    def act(self, state):
        if not state.measurable().any():
            return terminal_action(state)
        p = p_continue(self.theta, features(state, len(self.theta)))
        go = p > 0.5 if self.greedy else self.rng.random() < p
        return self._ei.act(state) if go else terminal_action(state)


class VOIRulePolicy(Policy):
    """No learning: run EI's pick while some experiment's one-step value of
    information exceeds its cost, then stop. A cost-aware stopping rule."""
    name = "voi_rule"

    def __init__(self):
        self._ei = EIPolicy()

    def act(self, state):
        if not state.measurable().any() or voi_ratio(state) <= 1.0:
            return terminal_action(state)
        return self._ei.act(state)


# ------------------------------------------------------------- training data --
def synthetic_pool(rng: np.random.Generator, hard: bool = False) -> Pool:
    """A made-up materials pool: random recipes, objective drawn from a GP prior.

    easy: smooth objective, moderate noise. Winners are nearly always findable.
    hard: also includes rough objectives, objectives that are partly unrelated
          to the recipe (so the GP cannot predict them), smaller pools and
          noisier measurements. Some of these campaigns are hopeless, which is
          what gives the gate a reason to learn to give up.
    """
    K = int(rng.integers(60 if hard else 80, 301))
    d = int(rng.integers(3, 6))
    X = rng.random((K, d))
    ls = np.exp(rng.uniform(np.log(0.08 if hard else 0.15), np.log(0.8), d))
    C = matern52(X, X, ls, 1.0)
    C[np.diag_indices_from(C)] += 1e-6
    f = np.linalg.cholesky(C) @ rng.standard_normal(K)
    if hard:   # fraction of the objective's variance that has no structure in X
        nu = rng.uniform(0.0, 0.7)
        f = np.sqrt(1 - nu) * (f - f.mean()) / f.std() + np.sqrt(nu) * rng.standard_normal(K)
    mu, sd = f.mean(), f.std()
    return Pool(name="synthetic", X=X, f=(f - mu) / sd,
                replicate_sd=float(rng.uniform(0.15, 0.9 if hard else 0.55)),
                n_replicates=np.ones(K, int), feature_names=[f"x{i + 1}" for i in range(d)],
                f_mean=float(mu), f_std=float(sd))


def rollout(args) -> dict:
    """One training campaign on a fresh synthetic pool with the stochastic gate.

    Returns per-decision features, actions (1 = continue), reward-to-go and the
    baseline value, plus the campaign's net utility.
    """
    theta, seed, hard = args
    theta = np.asarray(theta, float)
    cfg = TaskConfig()
    pool = synthetic_pool(np.random.default_rng([seed, 0]), hard)
    task = LabTask(pool, cfg)
    env_rng = np.random.default_rng([seed, 1])
    pol_rng = np.random.default_rng([seed, 3])
    state = AgentState(pool.X, AgentModel.from_task(task), np.random.default_rng([seed, 4]))
    ei = EIPolicy()

    for k in np.random.default_rng([seed, 2]).choice(pool.K, size=cfg.n_init, replace=False):
        state.observe(int(k), *task.measure(int(k), env_rng))

    phis, acts, base = [], [], []
    while state.measurable().any():
        phi = features(state, len(theta))
        go = pol_rng.random() < p_continue(theta, phi)
        phis.append(phi)
        acts.append(int(go))
        base.append(phi[1])                       # baseline b(s) = value of stopping now
        if not go:
            break
        k = ei.act(state)[1]
        state.observe(k, *task.measure(k, env_rng))

    action = terminal_action(state)
    final = task.commit_reward(action[1]) if action[0] == "commit" else 0.0
    n_cont = int(sum(acts))
    # reward-to-go of decision t: cost of every experiment from t on, plus the final payoff
    rtg = np.array([final - cfg.cost * (n_cont - sum(acts[:t])) for t in range(len(acts))])
    return dict(phi=np.array(phis).reshape(-1, len(theta)), a=np.array(acts), rtg=rtg,
                base=np.array(base), ret=final - state.spent, n=state.n_attempts)
