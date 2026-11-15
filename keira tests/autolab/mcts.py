"""Monte Carlo tree search in belief space (online POMDP planning).

At every real decision the agent searches from scratch (receding horizon):

1. Actions. The pool has up to 600 candidates, too many to branch on, so the
   tree only considers measuring the top `n_measure` candidates by upper
   confidence bound, mu_k + kappa * sd_k (kappa = 2), plus one "stop" action. Stop's value is exact:
   V_stop(b) = max(0, max_k E_b[U_k]), since committing to anything but the
   argmax is dominated and abandoning gives 0.

2. Belief nodes. Each node holds an exact Gaussian belief (over a working set
   of promising candidates; the marginal of the full GP is exact). Simulating
   measure(k) samples a reading from the agent's own predictive distribution
   (fail w.p. q; outlier w.p. rho; otherwise y ~ N(mu_k, Sigma_kk + sigma^2)),
   then applies the same outlier check and Kalman update the real agent uses.

3. Double progressive widening. Readings are continuous, so an action node
   gets a new sampled reading only while #children <= k_o * N(a)^alpha_o;
   otherwise an existing child is revisited in proportion to its count.

4. UCB1 selection on measure actions (stop has an exact value, no bonus).

5. Leaf evaluation by rollout. From a leaf, a simple base policy measures the
   candidate with the largest expected improvement over the target
   E[(f - tau)^+] for (h* - depth) more steps, then stops. The horizon h* is
   picked once per decision at the root by averaging rollouts:
       G(h) = E[-cost of h measurements + V_stop(b_h)],   h* = argmax_h G(h).

The real action is the root action with the best mean return; stop is
chosen if no measurement (visited >= min_visits times) beats V_stop.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from .gp import is_good_reading, kalman_update
from .policies import Policy, terminal_action
from .utility import expected_commit_value


def target_ei(mu, sd, tau):
    """E[(f - tau)^+]: expected improvement over the *target*."""
    z = (mu - tau) / sd
    return (mu - tau) * ndtr(z) + sd * np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)


class _Node:
    """A belief node: Gaussian belief over the working set + remaining budget."""
    __slots__ = ("mu", "Sigma", "remaining", "avail", "stop_val", "N", "stats")

    def __init__(self, mu, Sigma, remaining, avail, m):
        self.mu, self.Sigma, self.remaining = mu, Sigma, remaining
        self.avail = avail if m.cost <= remaining + 1e-12 else np.zeros_like(avail)
        eu = expected_commit_value(mu, np.diag(Sigma), m)
        self.stop_val = max(0.0, float(eu.max()))
        self.N = 0
        self.stats = {}  # action -> [visits, mean return, [[child, count], ...]]


class BeliefMCTS(Policy):
    name = "mcts"

    def __init__(self, n_sims: int = 200, max_depth: int = 6, c_ucb: float = 0.5,
                 k_o: float = 2.0, alpha_o: float = 0.3, n_measure: int = 6,
                 working_set: int = 48, n_roll_root: int = 24, horizon: int = 30,
                 min_visits: int = 5, kappa: float = 2.0):
        self.n_sims, self.max_depth, self.c_ucb = n_sims, max_depth, c_ucb
        self.k_o, self.alpha_o = k_o, alpha_o
        self.n_measure, self.M = n_measure, working_set
        self.n_roll_root, self.H, self.min_visits = n_roll_root, horizon, min_visits
        self.kappa = kappa

    # ------------------------------------------------------------------ act --
    def act(self, state):
        meas = state.measurable()
        cand = np.flatnonzero(meas)
        if len(cand) == 0:
            return terminal_action(state)
        m, b = state.model, state.belief
        self.m = m

        # 1. tree actions: top candidates by upper confidence bound
        ucb = np.where(meas, b.mu + self.kappa * np.sqrt(b.var), -np.inf)
        top_meas = cand[np.argsort(-ucb[cand])[: self.n_measure]]

        # 2. working set: tree actions + more top-UCB (for rollouts) + top commit candidates
        top_ucb = [i for i in np.argsort(-ucb)[: self.M - 8] if np.isfinite(ucb[i])]
        top_commit = np.argsort(-expected_commit_value(b.mu, b.var, m))[:8]
        S = np.unique(np.r_[top_meas, top_ucb, top_commit]).astype(int)
        self.tree_mask = np.isin(S, top_meas)
        sub = b.subset(S)
        avail = meas[S]

        # 5. rollout horizon from root rollouts
        self.h_star = int(np.argmax(self._rollout_curve(sub.mu, sub.Sigma, avail, state.remaining)))

        root = _Node(sub.mu, sub.Sigma, state.remaining, avail, m)
        for _ in range(self.n_sims):
            self._simulate(root, 0)

        best_a, best_q = None, root.stop_val
        for a, (n, q, _) in root.stats.items():
            if n >= self.min_visits and q > best_q:
                best_a, best_q = a, q
        self.last_root = root
        if best_a is None:
            return terminal_action(state)
        return ("measure", int(S[best_a]))

    # ----------------------------------------------------------- simulation --
    def _select(self, node):
        """UCB1 over measure actions; returns None for stop."""
        best, best_u = None, node.stop_val
        logn = np.log(node.N + 1)
        for a in np.flatnonzero(node.avail & self.tree_mask):
            st = node.stats.get(a)
            if st is None or st[0] == 0:
                return int(a)
            u = st[1] + self.c_ucb * self.m.payoff * np.sqrt(logn / st[0])
            if u > best_u:
                best, best_u = int(a), u
        return best

    def _simulate(self, node, depth):
        if depth >= self.max_depth or not (node.avail & self.tree_mask).any():
            return self._rollout(node, depth)
        a = self._select(node)
        node.N += 1
        if a is None:
            return node.stop_val
        st = node.stats.setdefault(a, [0, 0.0, []])
        if len(st[2]) <= self.k_o * (st[0] + 1) ** self.alpha_o:   # widen: new reading
            child = self._step(node, a)
            st[2].append([child, 1])
            ret = -self.m.cost + self._rollout(child, depth + 1)
        else:                                                      # revisit a reading
            counts = np.array([e[1] for e in st[2]], float)
            i = self.rng.choice(len(counts), p=counts / counts.sum())
            st[2][i][1] += 1
            ret = -self.m.cost + self._simulate(st[2][i][0], depth + 1)
        st[0] += 1
        st[1] += (ret - st[1]) / st[0]
        return ret

    def _step(self, node, a):
        """Sample a reading from the agent's generative model and update the belief."""
        m, rng = self.m, self.rng
        avail = node.avail.copy()
        avail[a] = False
        rem = node.remaining - m.cost
        mu, Sigma = node.mu, node.Sigma
        if rng.random() >= m.fail_prob:
            var_y = Sigma[a, a] + m.noise_var
            if rng.random() < m.outlier_prob:
                var_y += m.outlier_var
            y = mu[a] + np.sqrt(var_y) * rng.standard_normal()
            if is_good_reading(mu[a], Sigma[a, a], y, m.noise_var, m.outlier_prob, m.outlier_var):
                mu, Sigma = kalman_update(mu, Sigma, a, y, m.noise_var)
        return _Node(mu, Sigma, rem, avail, m)

    def _base_action(self, node):
        sd = np.sqrt(np.maximum(np.diag(node.Sigma), 1e-12))
        return int(np.argmax(np.where(node.avail, target_ei(node.mu, sd, self.m.tau), -np.inf)))

    def _rollout(self, node, depth):
        """Base policy for (h* - depth) more steps, then stop."""
        ret = 0.0
        for _ in range(max(0, self.h_star - depth)):
            if not node.avail.any():
                break
            ret -= self.m.cost
            node = self._step(node, self._base_action(node))
        return ret + node.stop_val

    def _rollout_curve(self, mu, Sigma, avail, remaining):
        """G(h) for h = 0..H, averaged over rollouts of the base policy."""
        G = np.zeros(self.H + 1)
        for _ in range(self.n_roll_root):
            node = _Node(mu, Sigma, remaining, avail.copy(), self.m)
            spent = 0.0
            G[0] += node.stop_val
            for h in range(1, self.H + 1):
                if node.avail.any():
                    spent += self.m.cost
                    node = self._step(node, self._base_action(node))
                G[h] += node.stop_val - spent
        return G / self.n_roll_root
