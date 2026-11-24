"""A small discrete POMDP that abstracts Keira's lab problem.

Keira's POMDP (keira tests/autolab/pomdp.py) has a continuous hidden state, the
true score of every one of up to 600 candidates. The offline solvers in
Algorithms for Decision Making ch. 21 need a discrete state space they can
write down as matrices, so they cannot solve it directly. This module builds a
discrete version that keeps the decision that matters (keep measuring, commit,
or give up) and that every solver can handle:

  slots        n shortlisted candidates (default 4)
  state        (h, g): h = experiments still affordable (0..H, fully observed)
               g = which slots truly meet spec, one bit per slot (2^n values)
               plus one absorbing terminal state
  actions      measure(i) | commit(i) | abandon            (2n + 1 actions)
  transitions  measure: h -> h - 1, g unchanged; at h = 0 it just ends the campaign
               commit / abandon -> terminal
  observations none/fail | low reading | high reading      (y below / above the spec)
  rewards      measure: -cost ; commit(i): V * g_i - C_commit ; abandon: 0

The observation model uses only what Keira's agent knows (AgentModel): the
failure rate, the noise and outlier levels, and the spec threshold tau. A
candidate that meets spec has a score drawn from N(0, 1) above tau; a "near
miss" has a score drawn from N(0, 1) between the `contender_q` quantile and
tau (shortlisted candidates are contenders, not average ones). P(high reading)
for each class is then integrated over that score and the noise mixture.

Every campaign ends within H + 2 steps, so the solvers use discount 1.

Online, a GP belief maps to a belief over this POMDP: pick the n slots from
the GP posterior, set h from the remaining budget, and give slot i
P(good) = P(f_k >= tau) under the GP marginal, independently across slots.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr, ndtri

OBS_NONE, OBS_LOW, OBS_HIGH = 0, 1, 2
N_OBS = 3
TIE_TOL = 1e-9


def best_action(q: np.ndarray, tol: float = TIE_TOL) -> int:
    """argmax with a deterministic tie-break: among actions within `tol` of the best,
    take the lowest index. Plain argmax lets 1e-16 roundoff decide between actions
    that are exactly tied in theory (e.g. QMDP and the fast informed bound, which are
    identical on this problem, would otherwise pick different candidates)."""
    q = np.asarray(q, float)
    return int(np.flatnonzero(q >= q.max() - tol)[0])


@dataclass(frozen=True)
class AbstractParams:
    n_slots: int
    horizon: int          # H: most experiments the abstraction looks ahead
    cost: float
    payoff: float
    commit_cost: float
    fail_prob: float
    p_high_good: float    # P(high reading | slot meets spec), given the experiment did not fail
    p_high_bad: float     # P(high reading | slot is a near miss)

    def key(self) -> tuple:
        return tuple(round(v, 10) if isinstance(v, float) else v for v in self.__dict__.values())


def reading_probs(tau: float, noise_sd: float, outlier_prob: float, outlier_sd: float,
                  contender_q: float = 0.75, n_grid: int = 4001) -> tuple[float, float]:
    """(P(y >= tau | good), P(y >= tau | near miss)) by numerical integration."""
    lo = min(float(ndtri(contender_q)), tau - 1e-3)
    f = np.linspace(lo, tau + 6.0, n_grid)
    dens = np.exp(-0.5 * f ** 2)
    sd_out = np.sqrt(noise_sd ** 2 + outlier_sd ** 2)
    p_hi = (1 - outlier_prob) * ndtr((f - tau) / max(noise_sd, 1e-9)) + outlier_prob * ndtr((f - tau) / sd_out)
    good, bad = f >= tau, f < tau
    return (float((p_hi * dens)[good].sum() / dens[good].sum()),
            float((p_hi * dens)[bad].sum() / dens[bad].sum()))


def params_from_model(m, n_slots: int = 4, horizon: int = 10, contender_q: float = 0.75) -> AbstractParams:
    """Build the abstraction from Keira's AgentModel (only agent-known quantities)."""
    hg, hb = reading_probs(m.tau, np.sqrt(m.noise_var), m.outlier_prob, np.sqrt(m.outlier_var), contender_q)
    return AbstractParams(n_slots=n_slots, horizon=horizon, cost=m.cost, payoff=m.payoff,
                          commit_cost=m.commit_cost, fail_prob=m.fail_prob, p_high_good=hg, p_high_bad=hb)


class DiscretePOMDP:
    """Explicit matrices for the abstraction.

    T[a, s, s']   transition probabilities
    O[a, s', o]   observation probabilities
    R[s, a]       expected rewards
    M[a, o, s, s'] = T[a, s, s'] * O[a, s', o]   (used for belief updates and backups)
    """

    def __init__(self, p: AbstractParams):
        self.p = p
        n, H = p.n_slots, p.horizon
        self.n, self.H, self.G = n, H, 2 ** n
        self.S = (H + 1) * self.G + 1
        self.term = self.S - 1
        self.A = 2 * n + 1
        self.O_n = N_OBS
        self.gamma = 1.0
        self.measure_actions = list(range(n))
        self.commit_actions = list(range(n, 2 * n))
        self.abandon = 2 * n
        bits = (np.arange(self.G)[:, None] >> np.arange(n)[None, :]) & 1   # (G, n): g_i
        self.bits = bits

        S, A, G = self.S, self.A, self.G
        T = np.zeros((A, S, S))
        O = np.zeros((A, S, N_OBS))
        R = np.zeros((S, A))
        T[:, self.term, self.term] = 1.0
        O[:, :, OBS_NONE] = 1.0                       # default: uninformative observation

        for h in range(H + 1):
            for g in range(G):
                s = self.idx(h, g)
                for i in range(n):
                    a = i                                  # measure slot i
                    R[s, a] = -p.cost
                    if h == 0:
                        T[a, s, self.term] = 1.0
                    else:
                        T[a, s, self.idx(h - 1, g)] = 1.0
                    c = n + i                              # commit slot i
                    R[s, c] = p.payoff * bits[g, i] - p.commit_cost
                    T[c, s, self.term] = 1.0
                T[self.abandon, s, self.term] = 1.0

        # informative observations of measure(i), as a function of the state reached
        for i in range(n):
            for h in range(H):                         # states reachable by measuring: h = 0..H-1
                for g in range(G):
                    s2 = self.idx(h, g)
                    ph = p.p_high_good if bits[g, i] else p.p_high_bad
                    O[i, s2] = [p.fail_prob, (1 - p.fail_prob) * (1 - ph), (1 - p.fail_prob) * ph]

        self.T, self.O, self.R = T, O, R
        self.M = np.einsum("ast,ato->aost", T, O)
        # transposes used to back up alpha vectors: alpha_a(s) = R + sum_o M[a,o] @ alpha_ao
        assert np.allclose(T.sum(-1), 1) and np.allclose(O.sum(-1), 1)

    def idx(self, h: int, g: int) -> int:
        return h * self.G + g

    # ------------------------------------------------------------ beliefs --
    def product_belief(self, p_good: np.ndarray, h: int) -> np.ndarray:
        """Belief at budget level h, slots independent with P(good_i) = p_good[i]."""
        p_good = np.clip(np.asarray(p_good, float), 0.0, 1.0)
        w = np.prod(np.where(self.bits == 1, p_good[None, :], 1 - p_good[None, :]), axis=1)
        b = np.zeros(self.S)
        b[h * self.G:(h + 1) * self.G] = w
        return b / b.sum()

    def level(self, b: np.ndarray) -> int:
        """Budget level of a belief (-1 for terminal). Beliefs never mix levels."""
        if b[self.term] > 0.5:
            return -1
        return int(np.argmax(b[:-1].reshape(self.H + 1, self.G).sum(1)))

    def successors(self, b: np.ndarray):
        """All (a, o) successor beliefs at once.

        Returns P (A, O) = P(o | b, a) and B2 (A, O, S) normalized successors
        (rows with P = 0 are left as zeros).
        """
        U = np.einsum("s,aost->aot", b, self.M)
        P = U.sum(-1)
        B2 = np.divide(U, P[..., None], out=np.zeros_like(U), where=P[..., None] > 0)
        return P, B2

    def update(self, b: np.ndarray, a: int, o: int) -> np.ndarray:
        u = b @ self.M[a, o]
        z = u.sum()
        return u / z if z > 0 else u

    def lookahead(self, b: np.ndarray, values) -> np.ndarray:
        """Q(b, a) = R(b, a) + sum_o P(o | b, a) U(b'), for every action.

        `values` maps an (N, S) array of beliefs to their (N,) values.
        """
        P, B2 = self.successors(b)
        V = values(B2.reshape(-1, self.S)).reshape(self.A, self.O_n)
        V = np.where(P > 0, V, 0.0)
        return b @ self.R + (P * V).sum(-1)

    # -------------------------------------------------------- simulation --
    def sample_state(self, b: np.ndarray, rng) -> int:
        return int(rng.choice(self.S, p=b))

    def step(self, s: int, a: int, rng) -> tuple[int, int, float]:
        s2 = int(rng.choice(self.S, p=self.T[a, s]))
        o = int(rng.choice(self.O_n, p=self.O[a, s2]))
        return s2, o, float(self.R[s, a])
