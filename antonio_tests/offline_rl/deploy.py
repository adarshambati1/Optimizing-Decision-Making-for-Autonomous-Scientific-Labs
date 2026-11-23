"""Run a learned stopping policy (behavioral cloning or batch Q-learning) in Keira's simulator.

A learned model is saved as JSON: its basis (degree, standardization) and weights.
At each decision the policy computes the same 8 belief features the log uses,
continues with GP-EI's pick if the learned rule says so, and otherwise takes
the shared terminal action.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
from scipy.special import expit  # noqa: E402

from behavior import features  # noqa: E402  (puts Keira's and Adarsh's code on the path)
from autolab.policies import EIPolicy, Policy, terminal_action  # noqa: E402
from methods import QuadraticBasis  # noqa: E402


def save(model, kind: str, path: str, **meta) -> None:
    b = model.basis
    with open(path, "w") as fh:
        json.dump(dict(kind=kind, degree=b.degree, mu=b.mu.tolist(), sd=b.sd.tolist(),
                       theta=np.asarray(model.theta).tolist(), **meta), fh, indent=1)


class LoadedRule:
    """phi -> continue (1) / stop (0), from a saved model."""

    def __init__(self, path: str):
        with open(path) as fh:
            d = json.load(fh)
        self.kind = d["kind"]
        self.basis = QuadraticBasis(d["degree"])
        self.basis.mu, self.basis.sd = np.array(d["mu"]), np.array(d["sd"])
        self.theta = np.array(d["theta"])

    def __call__(self, Phi: np.ndarray) -> np.ndarray:
        B = self.basis(np.atleast_2d(Phi))
        if self.kind == "bc":
            return (expit(B @ self.theta) > 0.5).astype(float)
        Q = B @ self.theta.T
        return (Q[:, 1] > Q[:, 0]).astype(float)


_RULES: dict = {}


class LearnedStopPolicy(Policy):
    name = "learned_stop"

    def __init__(self, path: str):
        if path not in _RULES:
            _RULES[path] = LoadedRule(path)
        self.rule = _RULES[path]
        self._ei = EIPolicy()

    def act(self, state):
        if not state.measurable().any():
            return terminal_action(state)
        go = self.rule(features(state, 8)[None])[0] > 0.5
        return self._ei.act(state) if go else terminal_action(state)
