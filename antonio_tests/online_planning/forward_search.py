"""Belief-space forward search (ADM section 22.2, eq. 22.1) on Keira's GP belief.

    Q_d(b, a) = R(b, a) + sum_o P(o | b, a) U_{d-1}(Update(b, a, o))    (d > 0)
    U_d(b)    = max_a Q_d(b, a),       U_0(b) = V_stop(b)

with gamma = 1 (every campaign ends within its budget). The leaf value is the
exact value of stopping now, V_stop(b) = max(0, max_k V P(f_k >= tau) - C_commit),
the same terminal rule every agent in the project uses.

Actions at each belief node: stop, or measure one of the `n_actions` untested
candidates with the highest upper confidence bound mu + 2 sd (as MCTS does).
Observations of measure(k), using the agent's own model:
  * fail, with probability q: no information, the belief is unchanged;
  * a reading y ~ N(mu_k, Sigma_kk + sigma^2). The GP update is exact: Sigma'
    does not depend on y and mu' is linear in y, so the expectation over y is
    done by Gauss-Hermite quadrature (15 nodes at the last level, fewer above).
Outlier readings are left out of the lookahead, as in Adarsh's one-step rule.

Depth 1 is exactly Adarsh's one-step value-of-information test: measure some k
iff  -c + q V_stop(b) + (1 - q) E_y[V_stop(b')] > V_stop(b),  i.e. VOI(k) > c.
The tests check that equivalence. Deeper searches can see experiments that
only pay off in combination, which is what the one-step rule misses.

Searching works on an exact marginal of the GP over a working set (the top
candidates by upper confidence bound plus the best commit candidates), as MCTS does.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KEIRA = os.path.join(os.path.dirname(os.path.dirname(HERE)), "keira tests")
if KEIRA not in sys.path:
    sys.path.insert(0, KEIRA)

import numpy as np  # noqa: E402
from scipy.special import ndtr  # noqa: E402

from autolab.policies import EIPolicy, Policy, terminal_action  # noqa: E402
from autolab.utility import expected_commit_value  # noqa: E402


def _gh(n):
    z, w = np.polynomial.hermite_e.hermegauss(n)
    return z, w / np.sqrt(2 * np.pi)


class ForwardSearch:
    """The search itself, on (mu, Sigma) over a working set."""

    def __init__(self, m, n_actions: int = 5, n_quad_leaf: int = 15, n_quad_inner: int = 5, kappa: float = 2.0):
        self.m, self.n_actions, self.kappa = m, n_actions, kappa
        self.z_leaf, self.w_leaf = _gh(n_quad_leaf)
        self.z_in, self.w_in = _gh(n_quad_inner)
        self.nodes = 0

    # ---- values
    def stop_value(self, mu, var):
        m = self.m
        eu = m.payoff * ndtr((mu - m.tau) / np.sqrt(np.maximum(var, 1e-12))) - m.commit_cost
        return max(0.0, float(eu.max()))

    def actions(self, mu, Sigma, avail):
        idx = np.flatnonzero(avail)
        if len(idx) == 0:
            return idx
        ucb = mu[idx] + self.kappa * np.sqrt(np.maximum(np.diag(Sigma)[idx], 0.0))
        return idx[np.argsort(-ucb)[: self.n_actions]]

    def value(self, mu, Sigma, avail, d):
        """U_d(b)."""
        self.nodes += 1
        best = self.stop_value(mu, np.diag(Sigma))
        if d == 0:
            return best
        for k in self.actions(mu, Sigma, avail):
            best = max(best, self.q(mu, Sigma, avail, int(k), d))
        return best

    def q(self, mu, Sigma, avail, k, d):
        """Q_d(b, measure k)."""
        m = self.m
        s = Sigma[:, k]
        v = s[k] + m.noise_var
        Sigma2 = Sigma - np.outer(s, s) / v
        avail2 = avail.copy()
        avail2[k] = False
        fail = self.value(mu, Sigma, avail2, d - 1)            # a failed experiment teaches nothing
        shift = s / np.sqrt(v)                                 # mu' = mu + z * shift, z ~ N(0, 1)
        if d == 1:                                             # leaves: vectorized over the quadrature
            M = mu[None, :] + self.z_leaf[:, None] * shift[None, :]
            var2 = np.maximum(np.diag(Sigma2), 1e-12)
            eu = m.payoff * ndtr((M - m.tau) / np.sqrt(var2)[None, :]) - m.commit_cost
            ok = float(self.w_leaf @ np.maximum(0.0, eu.max(1)))
            self.nodes += len(self.z_leaf)
        else:
            ok = float(sum(w * self.value(mu + z * shift, Sigma2, avail2, d - 1)
                           for z, w in zip(self.z_in, self.w_in)))
        return -m.cost + m.fail_prob * fail + (1 - m.fail_prob) * ok


class ForwardSearchPolicy(Policy):
    """Plan with forward search at every real decision (receding horizon, ADM 9.1).

    picker = "self": measure the best action the search finds.
    picker = "ei":   use the search only to decide whether to continue; GP-EI picks
                     the experiment (isolates the stopping decision, as the one-step rule does).
    """
    name = "forward_search"

    def __init__(self, depth: int = 2, n_actions: int = 5, picker: str = "self", working_set: int = 40,
                 n_quad_inner: int = 5):
        self.depth, self.n_actions, self.picker = depth, n_actions, picker
        self.M, self.n_quad_inner = working_set, n_quad_inner
        self._ei = EIPolicy()
        self.nodes = 0

    def working_set(self, state):
        m, b = state.model, state.belief
        meas = state.measurable()
        ucb = np.where(meas, b.mu + 2.0 * np.sqrt(b.var), -np.inf)
        top_ucb = [i for i in np.argsort(-ucb)[: self.M] if np.isfinite(ucb[i])]
        top_commit = np.argsort(-expected_commit_value(b.mu, b.var, m))[:8]
        return np.unique(np.r_[top_ucb, top_commit]).astype(int)

    def act(self, state):
        meas = state.measurable()
        if not meas.any():
            return terminal_action(state)
        m, b = state.model, state.belief
        n_left = int(np.floor(state.remaining / m.cost + 1e-9))
        d = min(self.depth, n_left)
        S = self.working_set(state)
        mu, Sigma = b.mu[S].copy(), b.Sigma[np.ix_(S, S)].copy()
        avail = meas[S]
        fs = ForwardSearch(m, self.n_actions, n_quad_inner=self.n_quad_inner)
        v_stop = fs.stop_value(mu, np.diag(Sigma))
        best_k, best_q = None, v_stop
        for k in fs.actions(mu, Sigma, avail):
            qk = fs.q(mu, Sigma, avail, int(k), d)
            if qk > best_q + 1e-12:
                best_k, best_q = int(k), qk
        self.nodes += fs.nodes
        if best_k is None:
            return terminal_action(state)
        return ("measure", int(S[best_k])) if self.picker == "self" else self._ei.act(state)
