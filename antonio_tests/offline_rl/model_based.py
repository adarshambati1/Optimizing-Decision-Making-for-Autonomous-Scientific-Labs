"""Methods 4 and 5: estimate a model from the log, then plan on it.

The model is the discrete abstraction from antonio_tests/abstraction.py
(4 shortlisted candidates, up to 10 more experiments). Instead of assuming its
observation model, as antonio_tests/ did, it is estimated from the logged
experiments, separately for pools in each noise band:

    mle         ADM 16.1  counts: P(fail), P(high reading | meets spec), P(high | below spec)
    bayes_mean  ADM 16.4  Beta(1, 1) prior, posterior mean
    bayes_pess  ADM 16.4  the same posterior, read pessimistically (readings less informative,
                          failures more likely), so the planner does not over-trust rare data

"Below spec" pools near misses and weaker candidates, as they actually show up
in the log; the antonio_tests/ abstraction assumed every non-good slot was a
near miss. Costs and payoff are Keira's (the log's mean rewards recover them
exactly, so they are not re-estimated).

Each (noise band, estimator) model is solved offline with PBVI, the method
whose policies transferred best in antonio_tests/. Two ways to use it online:

    stop_only  the model only decides continue vs. stop; GP-EI picks the experiment
               (the same decision the model-free offline learners make)
    full       the model also picks which shortlisted candidate to measure
               (exactly how antonio_tests/ deployed its solvers)
"""
from __future__ import annotations

import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ANT = os.path.dirname(HERE)
if ANT not in sys.path:
    sys.path.insert(0, ANT)

import numpy as np  # noqa: E402
from scipy.special import ndtr  # noqa: E402

from abstraction import AbstractParams, DiscretePOMDP  # noqa: E402
from autolab.policies import EIPolicy, Policy, terminal_action  # noqa: E402
from autolab.pomdp import TaskConfig  # noqa: E402
from methods import NOISE_BINS, estimate_reading_model  # noqa: E402
from policy import pick_slots  # noqa: E402  (antonio_tests/policy.py)
from solvers import solve  # noqa: E402

MODELS = os.path.join(HERE, "models")
KINDS = ["mle", "bayes_mean", "bayes_pess"]


def band_centers():
    return [(lo + hi) / 2 for lo, hi in zip(NOISE_BINS[:-1], NOISE_BINS[1:])]


def build_params(est: dict, n_slots: int = 4, horizon: int = 10) -> AbstractParams:
    c = TaskConfig()
    return AbstractParams(n_slots=n_slots, horizon=horizon, cost=c.cost, payoff=c.payoff,
                          commit_cost=c.commit_cost, fail_prob=est["fail_prob"],
                          p_high_good=est["p_high_good"], p_high_bad=est["p_high_bad"])


def solve_one(args):
    """Estimate and solve one (noise band, estimator) model; returns its summary row."""
    me, ca, band, kind, method = args
    center = band_centers()[band]
    est = estimate_reading_model(me, ca, center, kind)
    P = DiscretePOMDP(build_params(est))
    vf, info = solve(P, method)
    path = os.path.join(MODELS, f"{kind}_band{band}_{method}.pkl")
    with open(path, "wb") as fh:
        pickle.dump((P.p, vf), fh)
    return dict(kind=kind, band=band, noise_lo=est["noise_bin"][0], noise_hi=est["noise_bin"][1],
                fail_prob=est["fail_prob"], p_high_good=est["p_high_good"], p_high_bad=est["p_high_bad"],
                n_readings=est["counts"]["n"], n_good=est["counts"]["good_n"], n_bad=est["counts"]["bad_n"],
                solve_sec=info["seconds"], path=path)


def model_path(kind: str, noise_sd: float, method: str = "pbvi") -> str:
    band = int(np.clip(np.searchsorted(NOISE_BINS, noise_sd, side="right") - 1, 0, len(NOISE_BINS) - 2))
    return os.path.join(MODELS, f"{kind}_band{band}_{method}.pkl")


_CACHE: dict = {}


def _load(path):
    if path not in _CACHE:
        with open(path, "rb") as fh:
            params, vf = pickle.load(fh)
        _CACHE[path] = (DiscretePOMDP(params), vf)
    return _CACHE[path]


class LearnedModelPolicy(Policy):
    """Plan on the learned abstraction for this pool's noise band.

    mode = "stop_only": continue iff some measure action beats committing / abandoning
    under the lookahead, and then run GP-EI's pick; "full": also use the model's slot.
    """
    name = "learned_model"

    def __init__(self, kind: str = "mle", mode: str = "stop_only", method: str = "pbvi"):
        self.kind, self.mode, self.method = kind, mode, method
        self._ei = EIPolicy()

    def act(self, state):
        m, b = state.model, state.belief
        meas = state.measurable()
        if not meas.any():
            return terminal_action(state)
        P, vf = _load(model_path(self.kind, float(np.sqrt(m.noise_var)), self.method))
        slots = pick_slots(state, P.n)
        p_good = ndtr((b.mu[slots] - m.tau) / np.sqrt(b.var[slots]))
        n_left = int(np.floor(state.remaining / m.cost + 1e-9))
        q = P.lookahead(P.product_belief(p_good, min(P.H, n_left)), vf.values)
        for i, k in enumerate(slots):
            if not meas[k] or n_left == 0:
                q[P.measure_actions[i]] = -np.inf
        a = int(q.argmax())
        if a in P.measure_actions:
            return self._ei.act(state) if self.mode == "stop_only" else ("measure", int(slots[a]))
        if self.mode == "stop_only":         # the shared terminal rule, like every stop-only agent
            return terminal_action(state)
        if a in P.commit_actions:
            return ("commit", int(slots[a - P.n]))
        return ("abandon",)
