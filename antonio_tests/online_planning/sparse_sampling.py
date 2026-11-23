"""Belief-space sparse sampling (ADM section 22.4, eq. 22.2) on Keira's GP belief.

    Q_d(b, a) = (1/m) sum_{i=1..m} [ r_a^(i) + U_{d-1}(Update(b, a, o_a^(i))) ]    (d > 0)
    U_d(b)    = max_a Q_d(b, a),       U_0(b) = V_stop(b)

Forward search (forward_search.py) sums over every observation, which needs
the observation model written down in closed form, so it had to leave outlier
readings out. Sparse sampling only needs to *simulate* an observation, so here
each of the m samples of measure(k) is drawn from the agent's full generative
model, exactly as MCTS's simulator does (keira tests/autolab/mcts.py, _step):

  * fail with probability q: no reading, belief unchanged;
  * otherwise an outlier with probability rho, y ~ N(mu_k, Sigma_kk + sigma^2 + sigma_out^2),
    else y ~ N(mu_k, Sigma_kk + sigma^2);
  * the agent's outlier check (is_good_reading) runs on y, and the GP is updated only
    if the reading is accepted, as in the real agent.

Cost is O((|A| m)^d) per decision. Actions, working set, leaf value and the
"self" / "ei" pickers are the same as forward search, so the two differ only
in sampling vs. exact expectations (and in modeling outliers).
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from forward_search import ForwardSearch, ForwardSearchPolicy
from autolab.gp import is_good_reading  # noqa: E402  (keira tests is on the path via forward_search)
from autolab.policies import terminal_action


class SparseSampling(ForwardSearch):
    def __init__(self, m, rng: np.random.Generator, n_samples: int = 6, n_actions: int = 5, kappa: float = 2.0):
        super().__init__(m, n_actions=n_actions, kappa=kappa)
        self.rng, self.n_samples = rng, n_samples

    def _sample_outcomes(self, mu_k, var_k):
        """m samples: (accepted reading?, y). Rejected or failed -> belief unchanged."""
        m, n = self.m, self.n_samples
        failed = self.rng.random(n) < m.fail_prob
        outlier = self.rng.random(n) < m.outlier_prob
        sd = np.sqrt(var_k + m.noise_var + np.where(outlier, m.outlier_var, 0.0))
        y = mu_k + sd * self.rng.standard_normal(n)
        ok = np.array([(not f) and is_good_reading(mu_k, var_k, yi, m.noise_var, m.outlier_prob, m.outlier_var)
                       for f, yi in zip(failed, y)])
        return ok, y

    def q(self, mu, Sigma, avail, k, d):
        m = self.m
        s = Sigma[:, k]
        v = s[k] + m.noise_var
        Sigma2 = Sigma - np.outer(s, s) / v
        avail2 = avail.copy()
        avail2[k] = False
        ok, y = self._sample_outcomes(mu[k], Sigma[k, k])
        n_unchanged = int((~ok).sum())
        total = 0.0
        if n_unchanged:                                    # failed or rejected: same belief, less budget
            total += n_unchanged * self.value(mu, Sigma, avail2, d - 1)
        if ok.any():
            M = mu[None, :] + ((y[ok] - mu[k]) / v)[:, None] * s[None, :]
            if d == 1:                                     # leaves, vectorized over the samples
                var2 = np.maximum(np.diag(Sigma2), 1e-12)
                eu = m.payoff * ndtr((M - m.tau) / np.sqrt(var2)[None, :]) - m.commit_cost
                total += float(np.maximum(0.0, eu.max(1)).sum())
                self.nodes += len(M)
            else:
                total += sum(self.value(row, Sigma2, avail2, d - 1) for row in M)
        return -m.cost + total / self.n_samples


class SparseSamplingPolicy(ForwardSearchPolicy):
    """Receding-horizon sparse sampling; same interface and pickers as ForwardSearchPolicy."""
    name = "sparse_sampling"

    def __init__(self, depth: int = 2, n_samples: int = 6, n_actions: int = 5, picker: str = "self",
                 working_set: int = 40):
        super().__init__(depth=depth, n_actions=n_actions, picker=picker, working_set=working_set)
        self.n_samples = n_samples

    def act(self, state):
        meas = state.measurable()
        if not meas.any():
            return terminal_action(state)
        m, b = state.model, state.belief
        d = min(self.depth, int(np.floor(state.remaining / m.cost + 1e-9)))
        S = self.working_set(state)
        mu, Sigma, avail = b.mu[S].copy(), b.Sigma[np.ix_(S, S)].copy(), meas[S]
        ss = SparseSampling(m, self.rng, self.n_samples, self.n_actions)
        best_k, best_q = None, ss.stop_value(mu, np.diag(Sigma))
        for k in ss.actions(mu, Sigma, avail):
            qk = ss.q(mu, Sigma, avail, int(k), d)
            if qk > best_q + 1e-12:
                best_k, best_q = int(k), qk
        self.nodes += ss.nodes
        if best_k is None:
            return terminal_action(state)
        return ("measure", int(S[best_k])) if self.picker == "self" else self._ei.act(state)
