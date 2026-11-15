"""Policy interface and the baseline.

A policy maps an AgentState to an action:
    ("measure", k) | ("commit", k) | ("abandon",)

Baseline (Liang et al.-style Bayesian optimization): GP + expected improvement,
spend the whole budget, then take the optimal terminal action under the final
belief. This is the POMDP's MCTS agent with the stopping decision and
lookahead switched off.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from .pomdp import AgentState
from .utility import stop_value


def terminal_action(state: AgentState):
    """Commit to argmax_k E_b[U_k] if that beats abandoning (0), else abandon."""
    b = state.belief
    v, k = stop_value(b.mu, b.var, state.model)
    return ("commit", k) if v > 0 else ("abandon",)


class Policy:
    name = "policy"

    def reset(self, rng: np.random.Generator):
        self.rng = rng

    def act(self, state: AgentState):
        raise NotImplementedError


class EIPolicy(Policy):
    """GP expected improvement over the best tested posterior mean, fixed budget."""
    name = "ei"

    def __init__(self, xi: float = 0.01):
        self.xi = xi

    def act(self, state):
        cand = np.flatnonzero(state.measurable())
        if len(cand) == 0:
            return terminal_action(state)
        b = state.belief
        best = b.mu[state.tested].max() if state.tested.any() else b.mu.max()
        mu, sd = b.mu[cand], np.sqrt(b.var[cand])
        z = (mu - best - self.xi) / sd
        ei = (mu - best - self.xi) * ndtr(z) + sd * np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
        return ("measure", int(cand[np.argmax(ei)]))


def make_policy(name: str, **kw) -> Policy:
    from .mcts import BeliefMCTS
    return {"ei": EIPolicy, "mcts": BeliefMCTS}[name](**kw)
