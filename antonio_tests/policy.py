"""Use an offline-solved abstraction as a policy in Keira's simulator.

Each real decision:
  1. pick n slots from the GP belief: the best commit candidates first
     (`n_commit_slots`), then the untested candidates most likely to meet spec;
  2. build the abstract belief: slot i good with P(f_k >= tau) under the GP
     marginal, independent across slots, at budget level h = min(H, experiments left);
  3. one-step lookahead on the solver's value function, with measure(i) masked
     when slot i is already tested or no experiment is affordable;
  4. map the abstract action back: measure / commit that slot's candidate, or abandon.

The GP belief is rebuilt after every real reading, so the policy replans every step.
"""
from __future__ import annotations

import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KEIRA = os.path.join(os.path.dirname(HERE), "keira tests")
for p in (HERE, KEIRA):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
from scipy.special import ndtr  # noqa: E402

from autolab.policies import Policy, terminal_action  # noqa: E402
from autolab.utility import expected_commit_value  # noqa: E402

_CACHE: dict = {}


def load_solved(path: str):
    """(DiscretePOMDP, value function); cached per worker process."""
    if path not in _CACHE:
        from abstraction import DiscretePOMDP
        with open(path, "rb") as fh:
            params, vf = pickle.load(fh)
        _CACHE[path] = (DiscretePOMDP(params), vf)
    return _CACHE[path]


def pick_slots(state, n: int, n_commit_slots: int = 2) -> np.ndarray:
    m, b = state.model, state.belief
    eu = expected_commit_value(b.mu, b.var, m)
    p_good = ndtr((b.mu - m.tau) / np.sqrt(b.var))
    chosen = list(np.argsort(-eu)[:n_commit_slots])
    meas = state.measurable()
    for k in np.argsort(-np.where(meas, p_good, -np.inf)):
        if len(chosen) >= n or not meas[k]:
            break
        if k not in chosen:
            chosen.append(int(k))
    for k in np.argsort(-eu):                       # pad if too few untested candidates remain
        if len(chosen) >= n:
            break
        if k not in chosen:
            chosen.append(int(k))
    return np.array(chosen[:n], int)


class OfflinePOMDPPolicy(Policy):
    name = "offline"

    def __init__(self, path: str, n_commit_slots: int = 2):
        self.path, self.n_commit_slots = path, n_commit_slots

    def act(self, state):
        P, vf = load_solved(self.path)
        m, b = state.model, state.belief
        if not state.measurable().any():
            return terminal_action(state)
        slots = pick_slots(state, P.n, self.n_commit_slots)
        p_good = ndtr((b.mu[slots] - m.tau) / np.sqrt(b.var[slots]))
        n_left = int(np.floor(state.remaining / m.cost + 1e-9))
        h = min(P.H, n_left)
        q = P.lookahead(P.product_belief(p_good, h), vf.values)
        meas = state.measurable()
        for i, k in enumerate(slots):
            if not meas[k] or n_left == 0:
                q[P.measure_actions[i]] = -np.inf
        a = int(q.argmax())
        if a in P.measure_actions:
            return ("measure", int(slots[a]))
        if a in P.commit_actions:
            return ("commit", int(slots[a - P.n]))
        return ("abandon",)
