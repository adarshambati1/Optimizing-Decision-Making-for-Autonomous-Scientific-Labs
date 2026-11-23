"""Gap heuristic search (ADM section 22.7, algorithm 22.4) on Keira's GP belief.

Repeated searches from the current belief (up to k_max of them). Each one:
  * expands the node: every action's successor beliefs get initial bounds;
  * stops if the depth is used up or the gap U_hi(b) - U_lo(b) <= delta;
  * otherwise follows the action with the best upper-bound lookahead and, under it,
    the observation whose successor has the largest gap, and recurses;
  * on the way back, sets U_lo(b) and U_hi(b) to the best lower / upper lookahead.
The real action is the best under the LOWER-bound lookahead (algorithm 22.4),
so the planner only experiments when it can show experimenting beats stopping.

Bounds on the value of a GP belief b (net utility from here on, gamma = 1):
  lower  U_lo(b) = V_stop(b): stopping now is always available.
  upper  U_hi(b) = (V - C_commit) * (1 - prod_k (1 - p_k)),  p_k = P(f_k >= tau).
         An agent that saw every true score for free earns at most V - C_commit when
         some candidate meets spec and 0 otherwise, so the optimum is at most
         (V - C_commit) P(some f_k >= tau). For Gaussian scores with nonnegative
         correlations, Slepian's inequality gives P(some f_k >= tau) <= 1 - prod(1 - p_k).
         The GP prior's correlations are nonnegative; posterior ones nearly always
         are, but this is not guaranteed (the tests check how often the bound holds).
         The product runs over the working set, as everywhere else in the search.
Stop is an action whose value is known exactly (lower = upper = V_stop).

Observations of measure(k), as in forward search: fail (probability q, belief
unchanged) or one of `n_quad` Gauss-Hermite readings, each with its probability.
Outliers are left out, as in forward search.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from forward_search import ForwardSearch, ForwardSearchPolicy, _gh
from autolab.policies import terminal_action


class _Node:
    __slots__ = ("mu", "Sigma", "avail", "lo", "hi", "kids")

    def __init__(self, mu, Sigma, avail, lo, hi):
        self.mu, self.Sigma, self.avail, self.lo, self.hi = mu, Sigma, avail, lo, hi
        self.kids = None          # {k: (probs, [child nodes])}


class GapSearch(ForwardSearch):
    def __init__(self, m, n_actions: int = 5, n_quad: int = 5, delta: float = 1e-3):
        super().__init__(m, n_actions=n_actions)
        self.z, self.w = _gh(n_quad)
        self.delta = delta

    # ---- bounds
    def p_good(self, mu, var):
        return ndtr((mu - self.m.tau) / np.sqrt(np.maximum(var, 1e-12)))

    def lower(self, mu, var):
        return self.stop_value(mu, var)

    def upper(self, mu, var):
        m = self.m
        p_any = 1.0 - np.prod(1.0 - self.p_good(mu, var))
        return max(self.lower(mu, var), max(0.0, m.payoff - m.commit_cost) * p_any)

    def node(self, mu, Sigma, avail):
        var = np.diag(Sigma)
        return _Node(mu, Sigma, avail, self.lower(mu, var), self.upper(mu, var))

    # ---- tree
    def expand(self, nd):
        if nd.kids is not None:
            return
        nd.kids = {}
        m = self.m
        for k in self.actions(nd.mu, nd.Sigma, nd.avail):
            k = int(k)
            s = nd.Sigma[:, k]
            v = s[k] + m.noise_var
            Sigma2 = nd.Sigma - np.outer(s, s) / v
            avail2 = nd.avail.copy()
            avail2[k] = False
            probs = np.r_[m.fail_prob, (1 - m.fail_prob) * self.w]
            kids = [self.node(nd.mu, nd.Sigma, avail2)]                       # failed: belief unchanged
            kids += [self.node(nd.mu + z * s / np.sqrt(v), Sigma2, avail2) for z in self.z]
            nd.kids[k] = (probs, kids)
            self.nodes += len(kids)

    def q(self, nd, k, which):
        probs, kids = nd.kids[k]
        return -self.m.cost + float(probs @ np.array([getattr(c, which) for c in kids]))

    def search(self, nd, d):
        self.expand(nd)
        if d == 0 or nd.hi - nd.lo <= self.delta or not nd.kids:
            return
        v_stop = self.lower(nd.mu, np.diag(nd.Sigma))
        qs = {k: self.q(nd, k, "hi") for k in nd.kids}
        k = max(qs, key=qs.get)
        if qs[k] > v_stop:                                      # stop has no successor to explore
            probs, kids = nd.kids[k]
            child = max(kids, key=lambda c: c.hi - c.lo)        # largest gap (algorithm 22.4)
            self.search(child, d - 1)
        nd.lo = max(v_stop, max(self.q(nd, j, "lo") for j in nd.kids))
        nd.hi = max(v_stop, max(self.q(nd, j, "hi") for j in nd.kids))

    def plan(self, mu, Sigma, avail, depth, k_max):
        root = self.node(mu, Sigma, avail)
        self.gaps = [root.hi - root.lo]
        for _ in range(k_max):
            self.search(root, depth)
            self.gaps.append(root.hi - root.lo)
            if root.hi - root.lo < self.delta:
                break
        self.expand(root)
        v_stop = self.lower(mu, np.diag(Sigma))
        best_k, best_q = None, v_stop
        for k in root.kids:
            qk = self.q(root, k, "lo")
            if qk > best_q + 1e-12:
                best_k, best_q = k, qk
        return best_k, root


class GapSearchPolicy(ForwardSearchPolicy):
    """Receding-horizon gap heuristic search; same interface and pickers as ForwardSearchPolicy."""
    name = "gap_search"

    def __init__(self, depth: int = 3, k_max: int = 30, n_quad: int = 5, delta: float = 1e-3,
                 n_actions: int = 5, picker: str = "self", working_set: int = 40):
        super().__init__(depth=depth, n_actions=n_actions, picker=picker, working_set=working_set)
        self.k_max, self.n_quad, self.delta = k_max, n_quad, delta
        self.root_gaps = []

    def act(self, state):
        meas = state.measurable()
        if not meas.any():
            return terminal_action(state)
        m, b = state.model, state.belief
        d = min(self.depth, int(np.floor(state.remaining / m.cost + 1e-9)))
        S = self.working_set(state)
        gs = GapSearch(m, self.n_actions, self.n_quad, self.delta)
        k, root = gs.plan(b.mu[S].copy(), b.Sigma[np.ix_(S, S)].copy(), meas[S], d, self.k_max)
        self.nodes += gs.nodes
        self.root_gaps.append(root.hi - root.lo)
        if k is None:
            return terminal_action(state)
        return ("measure", int(S[k])) if self.picker == "self" else self._ei.act(state)
