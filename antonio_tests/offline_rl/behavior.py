"""Randomized behavior policies that generate the offline dataset.

Every agent here makes the same single decision at each step: continue (run
the experiment GP-EI picks) or stop (take the shared terminal action: commit
to the best candidate, or abandon). That is the decision the offline methods
learn, framed as in Adarsh's stopping gate (rl tests/gate_policy.py), so the
results compare directly.

Each behavior policy gives a base probability of continuing, then mixes in
uniform noise:

    P_behavior(continue | s) = (1 - eps) * p_base(s) + eps / 2

so both actions always have probability >= eps / 2. That is what makes
importance-sampling evaluation possible (every action a target policy might
take has a nonzero logged probability), and it spreads the stopping times so
the log covers stopping early, late and in between.

    name        p_base(s)                                         source
    ei          1 (never stops; GP-EI runs the full budget)       Keira's baseline
    voi_rule    1 if one-step value of information > cost, else 0 Adarsh's one-step rule
    gate        sigmoid(theta . features), his hard_voi weights   Adarsh's learned gate
    mcts        1 if MCTS would measure, 0 if it would stop       Keira's MCTS (stop decision only)
    hazard_q    1 - q: stop with fixed probability q each step    pure coverage, no belief used

For `mcts`, only its stop-or-continue decision is used; when it continues, the
experiment is still GP-EI's pick, so every logged "continue" means the same thing.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (os.path.join(ROOT, "rl tests"), os.path.join(ROOT, "keira tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

from autolab.mcts import BeliefMCTS  # noqa: E402
from gate_policy import features, p_continue, voi_ratio  # noqa: E402

GATE_WEIGHTS = os.path.join(ROOT, "rl tests", "results", "gate_weights_hard_voi.json")


class Behavior:
    """Base probability of continuing at a belief state, before the eps mixing."""
    name = "behavior"

    def reset(self, rng: np.random.Generator):
        self.rng = rng

    def p_base(self, state, phi: np.ndarray) -> float:
        raise NotImplementedError


class EIBehavior(Behavior):
    name = "ei"

    def p_base(self, state, phi):
        return 1.0


class VOIBehavior(Behavior):
    name = "voi_rule"

    def p_base(self, state, phi):
        return float(phi[7] > 1.0)            # phi[7] is voi_ratio = VOI / cost, the rule's test


class GateBehavior(Behavior):
    name = "gate"

    def __init__(self, path: str = GATE_WEIGHTS):
        with open(path) as fh:
            self.theta = np.asarray(json.load(fh)["theta"], float)

    def p_base(self, state, phi):
        return p_continue(self.theta, phi[: len(self.theta)])


class MCTSBehavior(Behavior):
    name = "mcts"

    def __init__(self, n_sims: int = 100):
        self.mcts = BeliefMCTS(n_sims=n_sims)

    def reset(self, rng):
        super().reset(rng)
        self.mcts.reset(rng)

    def p_base(self, state, phi):
        return float(self.mcts.act(state)[0] == "measure")


class HazardBehavior(Behavior):
    def __init__(self, q: float):
        self.q = q
        self.name = f"hazard_{q:g}"

    def p_base(self, state, phi):
        return 1.0 - self.q


class GreedyGateBehavior(GateBehavior):
    """Adarsh's gate exactly as deployed (continue iff P(continue) > 0.5)."""
    name = "gate_greedy"

    def p_base(self, state, phi):
        return float(super().p_base(state, phi) > 0.5)


def make_behavior(name: str) -> Behavior:
    if name.startswith("hazard_"):
        return HazardBehavior(float(name.split("_", 1)[1]))
    if name == "gate_greedy":
        return GreedyGateBehavior()
    return {"ei": EIBehavior, "voi_rule": VOIBehavior, "gate": GateBehavior, "mcts": MCTSBehavior}[name]()


def mix(p_base: float, eps: float) -> float:
    return (1 - eps) * p_base + eps / 2


__all__ = ["Behavior", "make_behavior", "mix", "features", "voi_ratio"]
