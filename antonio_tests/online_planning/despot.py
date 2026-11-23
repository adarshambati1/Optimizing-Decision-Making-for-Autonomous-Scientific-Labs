"""Determinized sparse tree search (ADM section 22.6; DESPOT, Ye et al. 2017) on Keira's GP belief.

The book's version: fix m scenarios up front. Scenario i has a sampled hidden state
and a row of a determinizing matrix Phi (uniform random numbers), and Phi[i, j]
decides the outcome that scenario sees at depth j for whatever action is taken.
A belief node is the set of scenario particles that reached it, so the tree has
O(|A|^d m) nodes, and every action is evaluated on the *same* scenarios (common
random numbers), unlike sparse sampling, which draws fresh samples for each action.

Here, per decision:
  * hidden state of scenario i: f^(i) ~ GP posterior over the working set (an exact
    sample of the true scores, so readings generated from it are consistent with the belief);
  * Phi[i, j] = 3 uniforms: experiment fails? / reading is an outlier? / noise draw.
    measure(k) for particle i at depth j gives: fail, or y = f^(i)_k + noise
    (outlier spread with the agent's rho and sigma_out); the agent's outlier
    filter then accepts or discards the reading, as in the real agent;
  * observations are grouped so particles can share belief nodes: one child for
    "no usable reading" (failed or discarded; belief unchanged), and `n_bins`
    children for accepted readings, cut at quantiles of the predictive
    distribution of y. A child's belief is the exact GP update at the mean reading
    of its particles. (With continuous readings and no grouping, every node would
    hold one particle whose future is fixed, so the search could choose actions
    knowing their outcomes.)
  * a node with fewer than `min_particles` particles is a leaf: the search does not
    branch on what too few scenarios say (a simple form of the regularization the
    book notes DESPOT uses);
  * U(node) = max(V_stop(b), max_a Q(a)),  Q(a) = -c + sum_children (n_child / n) U(child);
    leaves use V_stop(b), as forward search and sparse sampling do.

Actions, working set and the "self" / "ei" pickers are shared with forward search.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtri

from forward_search import ForwardSearch, ForwardSearchPolicy
from autolab.gp import is_good_reading
from autolab.policies import terminal_action


class Despot(ForwardSearch):
    def __init__(self, m, F: np.ndarray, Phi: np.ndarray, n_actions: int = 5, n_bins: int = 3,
                 min_particles: int = 4):
        super().__init__(m, n_actions=n_actions)
        self.F, self.Phi = F, Phi                  # (n_particles, |S|), (n_particles, depth, 3)
        self.n_bins, self.min_particles = n_bins, min_particles

    def _observe(self, P, k, j, mu_k, var_k):
        """Determinized outcomes of measure(k) at depth j for particles P: (usable mask, y)."""
        m, U = self.m, self.Phi[P, j]
        failed = U[:, 0] < m.fail_prob
        outlier = U[:, 1] < m.outlier_prob
        sd = np.sqrt(m.noise_var + np.where(outlier, m.outlier_var, 0.0))
        y = self.F[P, k] + sd * ndtri(np.clip(U[:, 2], 1e-12, 1 - 1e-12))
        ok = np.array([(not f) and is_good_reading(mu_k, var_k, yi, m.noise_var, m.outlier_prob, m.outlier_var)
                       for f, yi in zip(failed, y)], bool)
        return ok, y

    def node_value(self, mu, Sigma, avail, P, j, d):
        self.nodes += 1
        best = self.stop_value(mu, np.diag(Sigma))
        if d == 0 or len(P) < self.min_particles:
            return best
        for k in self.actions(mu, Sigma, avail):
            best = max(best, self.q_particles(mu, Sigma, avail, P, int(k), j, d))
        return best

    def q_particles(self, mu, Sigma, avail, P, k, j, d):
        m = self.m
        s = Sigma[:, k]
        v = s[k] + m.noise_var
        Sigma2 = Sigma - np.outer(s, s) / v
        avail2 = avail.copy()
        avail2[k] = False
        ok, y = self._observe(P, k, j, mu[k], Sigma[k, k])
        total = 0.0
        if (~ok).any():                                     # no usable reading: belief unchanged
            total += (~ok).sum() * self.node_value(mu, Sigma, avail2, P[~ok], j + 1, d - 1)
        if ok.any():
            edges = mu[k] + np.sqrt(v) * ndtri(np.arange(1, self.n_bins) / self.n_bins)
            bins = np.searchsorted(edges, y[ok])
            Pok, yok = P[ok], y[ok]
            for c in np.unique(bins):
                sel = bins == c
                mu2 = mu + s * ((yok[sel].mean() - mu[k]) / v)
                total += sel.sum() * self.node_value(mu2, Sigma2, avail2, Pok[sel], j + 1, d - 1)
        return -m.cost + total / len(P)


class DespotPolicy(ForwardSearchPolicy):
    """Receding-horizon determinized sparse tree search; same interface as ForwardSearchPolicy."""
    name = "despot"

    def __init__(self, depth: int = 3, n_particles: int = 30, n_bins: int = 3, min_particles: int = 4,
                 n_actions: int = 5, picker: str = "self", working_set: int = 40):
        super().__init__(depth=depth, n_actions=n_actions, picker=picker, working_set=working_set)
        self.n_particles, self.n_bins, self.min_particles = n_particles, n_bins, min_particles

    def act(self, state):
        meas = state.measurable()
        if not meas.any():
            return terminal_action(state)
        m, b = state.model, state.belief
        d = min(self.depth, int(np.floor(state.remaining / m.cost + 1e-9)))
        S = self.working_set(state)
        mu, Sigma, avail = b.mu[S].copy(), b.Sigma[np.ix_(S, S)].copy(), meas[S]
        L = np.linalg.cholesky(Sigma + 1e-9 * np.eye(len(S)))
        F = mu[None, :] + self.rng.standard_normal((self.n_particles, len(S))) @ L.T   # scenario truths
        Phi = self.rng.random((self.n_particles, max(d, 1), 3))                        # determinizing matrix
        tree = Despot(m, F, Phi, self.n_actions, self.n_bins, self.min_particles)
        P = np.arange(self.n_particles)
        best_k, best_q = None, tree.stop_value(mu, np.diag(Sigma))
        for k in tree.actions(mu, Sigma, avail):
            qk = tree.q_particles(mu, Sigma, avail, P, int(k), 0, d)
            if qk > best_q + 1e-12:
                best_k, best_q = int(k), qk
        self.nodes += tree.nodes
        if best_k is None:
            return terminal_action(state)
        return ("measure", int(S[best_k])) if self.picker == "self" else self._ei.act(state)
