"""Which experiment to run: exploration strategies from ADM ch. 15, on Keira's GP belief.

Every agent in the project decides two things each step: whether to keep
experimenting, and which candidate to measure. Most of them use GP-EI for the
second. This module swaps that choice for the book's exploration strategies,
adapted from bandits (pull an arm many times) to this problem (measure each
candidate at most once, under a GP belief):

    ei         expected improvement over the best tested posterior mean (Keira's baseline picker)
    greedy     argmax mu_k                                               (15.3, no exploration)
    quantile   argmax of the alpha-quantile of f_k: mu_k + Phi^-1(alpha) sd_k,
               alpha = 0.95                                              (15.4, algorithm 15.6)
    ucb1       argmax mu_k + c sqrt(log N / N(k))                        (15.4, eq. 15.3, algorithm 15.7)
               with the book's Bayesian adaptation of the count: N(k) is the number of
               noise-level readings the posterior on f_k is worth, sigma^2 / sd_k^2, and
               N the readings taken so far; so the bonus is c (sd_k / sigma) sqrt(log N).
               c = 1, not tuned.
    thompson   draw one joint sample f ~ GP posterior, measure argmax f_k (15.4, algorithm 15.8)

Each picker is paired with a stopping rule:

    never      run the whole budget (as GP-EI does)
    voi        Adarsh's one-step rule: stop when no experiment's one-step value of information beats its cost
    mcts       Keira's MCTS decides stop vs. continue (200 simulations); the picker chooses the experiment

never + ei is GP-EI, voi + ei is the one-step rule, and MCTS with its own choice is MCTS.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (os.path.join(ROOT, "rl tests"), os.path.join(ROOT, "keira tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
from scipy.special import ndtri  # noqa: E402

from autolab.mcts import BeliefMCTS  # noqa: E402
from autolab.policies import EIPolicy, Policy, terminal_action  # noqa: E402
from gate_policy import voi_ratio  # noqa: E402

PICKERS = ["ei", "greedy", "quantile", "ucb1", "thompson"]
STOPS = ["never", "voi", "mcts"]


def scores(state, picker: str, rng: np.random.Generator, alpha: float = 0.95, c: float = 1.0) -> np.ndarray:
    """Each candidate's score under the picker (higher is better); -inf if it cannot be measured."""
    b, m = state.belief, state.model
    meas = state.measurable()
    mu, sd = b.mu, np.sqrt(b.var)
    if picker == "greedy":
        s = mu.copy()
    elif picker == "quantile":
        s = mu + ndtri(alpha) * sd
    elif picker == "ucb1":
        n = max(2, len(state.obs))
        s = mu + c * (sd / np.sqrt(m.noise_var)) * np.sqrt(np.log(n))
    elif picker == "thompson":
        L = np.linalg.cholesky(b.Sigma + 1e-8 * np.eye(len(mu)))
        s = mu + L @ rng.standard_normal(len(mu))
    else:
        raise ValueError(picker)
    return np.where(meas, s, -np.inf)


class PickerPolicy(Policy):
    """A stopping rule plus an exploration strategy for which experiment to run."""
    name = "picker"

    def __init__(self, stop: str = "voi", picker: str = "ei", n_sims: int = 200, **kw):
        assert stop in STOPS and picker in PICKERS + ["mcts"], (stop, picker)
        self.stop, self.picker, self.kw = stop, picker, kw
        self._ei = EIPolicy()
        self._mcts = BeliefMCTS(n_sims=n_sims) if stop == "mcts" else None

    def reset(self, rng):
        super().reset(rng)
        if self._mcts is not None:
            self._mcts.reset(rng)

    def pick(self, state) -> int:
        if self.picker == "ei":
            return self._ei.act(state)[1]
        return int(np.argmax(scores(state, self.picker, self.rng, **self.kw)))

    def act(self, state):
        if not state.measurable().any():
            return terminal_action(state)
        if self.stop == "voi" and voi_ratio(state) <= 1.0:
            return terminal_action(state)
        if self.stop == "mcts":
            a = self._mcts.act(state)
            if a[0] != "measure":
                return a
            if self.picker == "mcts":
                return a
        return ("measure", self.pick(state))
