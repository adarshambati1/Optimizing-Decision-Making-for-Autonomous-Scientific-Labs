"""Offline POMDP solvers from Algorithms for Decision Making, chapter 21
(offline belief state planning).

Every solver returns a value function `vf` with `vf.values(B)` mapping an
(N, S) array of beliefs to (N,) values. The policy for every method is the
same one-step lookahead on that value function:

    pi(b) = argmax_a  R(b, a) + sum_o P(o | b, a) U(b')

so the methods differ only in how they approximate U.

    method             kind          what it is
    qmdp               upper bound   fully observable value approximation (QMDP)
    fib                upper bound   fast informed bound
    fast_lower         lower bound   fast lower bound (blind lower bound)
    pbvi               lower bound   point-based value iteration on sampled beliefs
    perseus            lower bound   randomized point-based value iteration
    sawtooth           upper bound   sawtooth upper bound (sawtooth iteration on sampled beliefs)
    point_selection    lower bound   PBVI on beliefs chosen by exploratory belief expansion
    sawtooth_search    lower+upper   sawtooth heuristic search (policy from its lower bound)
    triangulated       approximation triangulated value functions (Freudenthal)
"""
from __future__ import annotations

import time

import numpy as np

from abstraction import DiscretePOMDP
from triangulation import Freudenthal

ALL_METHODS = ["qmdp", "fib", "fast_lower", "pbvi", "perseus", "sawtooth",
               "point_selection", "sawtooth_search", "triangulated"]
KIND = {"qmdp": "upper", "fib": "upper", "fast_lower": "lower", "pbvi": "lower", "perseus": "lower",
        "sawtooth": "upper", "point_selection": "lower", "sawtooth_search": "lower",
        "triangulated": "approx"}


# ------------------------------------------------------------ value functions --
class AlphaVF:
    """Max over a set of alpha vectors."""

    def __init__(self, Gamma: np.ndarray):
        self.Gamma = np.atleast_2d(Gamma)

    def values(self, B):
        return (np.atleast_2d(B) @ self.Gamma.T).max(1)

    @property
    def size(self):
        return len(self.Gamma)


class SawtoothVF:
    """Sawtooth upper bound from corner values c and belief-value pairs (b_j, u_j)."""

    def __init__(self, corner: np.ndarray, B=None, u=None):
        self.c = np.asarray(corner, float).copy()
        self.S = len(self.c)
        self.Bp = np.zeros((0, self.S))
        self.u = np.zeros(0)
        self._supp = np.zeros((0, 1), int)
        self._sval = np.ones((0, 1))
        if B is not None:
            self.set_points(B, u)

    def set_points(self, B, u):
        self.Bp, self.u = np.atleast_2d(np.asarray(B, float)).copy(), np.asarray(u, float).copy()
        nz = self.Bp > 1e-15
        L = max(1, int(nz.sum(1).max())) if len(self.Bp) else 1
        supp = np.zeros((len(self.Bp), L), int)
        for j, row in enumerate(nz):
            idx = np.flatnonzero(row)
            supp[j] = np.r_[idx, np.full(L - len(idx), idx[0])]
        self._supp = supp
        self._sval = np.take_along_axis(self.Bp, supp, axis=1)

    def add_point(self, b, u):
        self.set_points(np.vstack([self.Bp, b]), np.r_[self.u, u])

    def values(self, B, chunk: int = 4096):
        B = np.atleast_2d(B)
        base = B @ self.c
        if len(self.Bp) == 0:
            return base
        off = self.u - self.Bp @ self.c                       # <= 0 once the points improve on the corners
        out = np.empty(len(B))
        for i in range(0, len(B), chunk):
            Q = B[i:i + chunk]
            delta = (Q[:, self._supp] / self._sval[None]).min(-1)   # (n, P)
            out[i:i + chunk] = np.minimum(base[i:i + chunk], (base[i:i + chunk, None] + delta * off[None]).min(1))
        return out

    @property
    def size(self):
        return len(self.Bp) + self.S


class TriangulatedVF:
    """Freudenthal interpolation of the value, one triangulation per budget level.

    The budget level h is fully observed, so each belief lives on the simplex
    over the 2^n slot states of one level; that is the simplex triangulated.
    """

    def __init__(self, P: DiscretePOMDP, m: int):
        self.H, self.G, self.F = P.H, P.G, Freudenthal(P.G, m)   # no reference to P: keeps pickles small
        self.vals = np.zeros((P.H + 1, self.F.V))

    def values(self, B):
        B = np.atleast_2d(B)
        out = np.zeros(len(B))
        blocks = B[:, :-1].reshape(len(B), self.H + 1, self.G)
        mass = blocks.sum(2)
        lv = mass.argmax(1)
        live = (B[:, -1] < 0.5) & (mass.max(1) > 0)
        for h in np.unique(lv[live]):
            r = np.flatnonzero(live & (lv == h))
            g = blocks[r, h] / mass[r, h][:, None]
            out[r] = self.F.interpolate(self.vals[h], g)
        return out

    @property
    def size(self):
        return self.vals.size


# --------------------------------------------------------------- the solvers --
def _converge(step, x, tol=1e-10, max_iter=500):
    for _ in range(max_iter):
        x2 = step(x)
        if np.abs(x2 - x).max() < tol:
            return x2
        x = x2
    return x


def qmdp(P: DiscretePOMDP) -> AlphaVF:
    """alpha_a(s) = R(s,a) + gamma sum_s' T(s'|s,a) max_a' alpha_a'(s')."""
    def step(al):
        return P.R.T + P.gamma * np.einsum("ast,t->as", P.T, al.max(0))
    return AlphaVF(_converge(step, np.zeros((P.A, P.S))))


def fib(P: DiscretePOMDP) -> AlphaVF:
    """alpha_a(s) = R(s,a) + gamma sum_o max_a' sum_s' O(o|a,s') T(s'|s,a) alpha_a'(s')."""
    def step(al):
        X = np.einsum("aost,bt->aobs", P.M, al)              # (A, O, A', S)
        return P.R.T + P.gamma * X.max(2).sum(1)
    return AlphaVF(_converge(step, np.zeros((P.A, P.S))))


def blind_lower_bound(P: DiscretePOMDP) -> np.ndarray:
    """alpha_a(s) = R(s,a) + gamma sum_s' T(s'|s,a) alpha_a(s'): always take action a.

    Every campaign ends within H + 2 steps, so iterating from zero converges to
    the exact value of each blind policy, a valid lower bound with gamma = 1.
    """
    def step(al):
        return P.R.T + P.gamma * np.einsum("ast,at->as", P.T, al)
    return _converge(step, np.zeros((P.A, P.S)))


def fast_lower(P: DiscretePOMDP) -> AlphaVF:
    return AlphaVF(blind_lower_bound(P))


def backup_batch(P: DiscretePOMDP, Gamma: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Point-based Bellman backup at every belief in B. Returns (N, S) alphas."""
    U = np.einsum("ns,aost->naot", B, P.M)                     # unnormalized successors
    best = (U @ Gamma.T).argmax(-1)                            # (N, A, O): best alpha at each b'
    al_ao = Gamma[best]                                        # (N, A, O, S)
    al_a = P.R.T[None] + P.gamma * np.einsum("aost,naot->nas", P.M, al_ao)
    a = np.einsum("ns,nas->na", B, al_a).argmax(1)
    return al_a[np.arange(len(B)), a]


def _unique_rows(G, tol=1e-9):
    G = np.round(G / tol) * tol
    return np.unique(G, axis=0)


def pbvi(P: DiscretePOMDP, B: np.ndarray, n_iter: int) -> AlphaVF:
    """Back up the alpha vectors at a fixed set of beliefs, starting from the blind lower bound."""
    blind = blind_lower_bound(P)
    Gamma = blind
    for _ in range(n_iter):
        Gamma = _unique_rows(np.vstack([blind, backup_batch(P, Gamma, B)]))
    return AlphaVF(Gamma)


def perseus(P: DiscretePOMDP, B: np.ndarray, n_iter: int, rng) -> AlphaVF:
    """Randomized point-based backups: back up random points until every point has improved."""
    blind = blind_lower_bound(P)
    Gamma = blind
    for _ in range(n_iter):
        U = (B @ Gamma.T).max(1)
        todo = np.ones(len(B), bool)
        new = []
        while todo.any():
            i = rng.choice(np.flatnonzero(todo))
            al = backup_batch(P, Gamma, B[i:i + 1])[0]
            if al @ B[i] < U[i]:
                al = Gamma[(Gamma @ B[i]).argmax()]
            new.append(al)
            todo &= (B @ np.array(new).T).max(1) < U - 1e-12
        Gamma = _unique_rows(np.vstack([blind, new]))
    return AlphaVF(Gamma)


def sample_beliefs(P: DiscretePOMDP, n: int, rng, h=None, a=0.7, b=1.5) -> np.ndarray:
    """Product beliefs: P(good_i) ~ Beta(a, b), budget level uniform (or fixed h)."""
    out = np.zeros((n, P.S))
    for j in range(n):
        lvl = int(rng.integers(0, P.H + 1)) if h is None else h
        out[j] = P.product_belief(rng.beta(a, b, P.n), lvl)
    return out


def exploratory_expansion(P: DiscretePOMDP, B: np.ndarray, n_target: int, rng) -> np.ndarray:
    """ADM exploratory belief expansion: for each b, add the successor (one sampled
    observation per action) that is farthest (L1) from the current set."""
    B = list(B)
    while len(B) < n_target:
        cur = np.array(B)
        for b in list(cur):
            Pr, B2 = P.successors(b)
            cands = []
            for a in range(P.A):
                if Pr[a].sum() <= 0:
                    continue
                o = rng.choice(P.O_n, p=Pr[a] / Pr[a].sum())
                b2 = B2[a, o]
                if P.level(b2) >= 0:
                    cands.append(b2)
            if not cands:
                continue
            cands = np.array(cands)
            dist = np.abs(cands[:, None, :] - np.array(B)[None]).sum(-1).min(1)
            if dist.max() > 1e-9:
                B.append(cands[dist.argmax()])
            if len(B) >= n_target:
                break
    return np.array(B[:n_target])


def point_selection(P: DiscretePOMDP, n_points: int, n_seed: int, n_iter: int, rng) -> tuple[AlphaVF, np.ndarray]:
    B0 = sample_beliefs(P, n_seed, rng, h=P.H)
    B = exploratory_expansion(P, B0, n_points, rng)
    return pbvi(P, B, n_iter), B


def sawtooth_iteration(P: DiscretePOMDP, B: np.ndarray, n_iter: int, init: AlphaVF) -> SawtoothVF:
    """Sawtooth upper bound: start from an upper bound (FIB) at the corners and the points,
    then repeatedly replace every value by its one-step lookahead under the sawtooth bound."""
    corners = np.eye(P.S)
    vf = SawtoothVF(init.values(corners), B, init.values(B))
    pts = np.vstack([corners[:-1], B])                         # all corners but terminal (value 0)
    for _ in range(n_iter):
        q = np.array([P.lookahead(b, vf.values).max() for b in pts])
        c = vf.c.copy()
        c[:-1] = np.minimum(c[:-1], q[:P.S - 1])
        u = np.minimum(vf.u, q[P.S - 1:])
        vf = SawtoothVF(c, B, u)
    return vf


class SawtoothSearchResult:
    def __init__(self, lower: AlphaVF, upper: SawtoothVF, gaps: list):
        self.lower, self.upper, self.gaps = lower, upper, gaps


def sawtooth_search(P: DiscretePOMDP, starts: np.ndarray, n_iter: int, depth: int, init_upper: AlphaVF,
                    delta: float = 1e-3) -> SawtoothSearchResult:
    """Sawtooth heuristic search: explore from each start belief, going
    down the action that is best under the upper bound and the observation with
    the largest probability-weighted gap between the bounds; on the way back up,
    tighten both bounds at the visited beliefs."""
    Gamma = [*blind_lower_bound(P)]
    upper = SawtoothVF(init_upper.values(np.eye(P.S)))
    lower_vals = lambda Bq: (np.atleast_2d(Bq) @ np.array(Gamma).T).max(1)   # noqa: E731

    def gap(Bq):
        return upper.values(Bq) - lower_vals(Bq)

    def explore(b, d):
        if d == 0 or P.level(b) < 0 or gap(b[None])[0] <= delta:
            return
        a = int(P.lookahead(b, upper.values).argmax())
        Pr, B2 = P.successors(b)
        score = Pr[a] * np.where(Pr[a] > 0, gap(B2[a]), 0.0)
        o = int(score.argmax())
        if score[o] > delta and P.level(B2[a, o]) >= 0:
            explore(B2[a, o], d - 1)
        Gamma.append(backup_batch(P, np.array(Gamma), b[None])[0])
        upper.add_point(b, min(upper.values(b[None])[0], P.lookahead(b, upper.values).max()))

    gaps = []
    for _ in range(n_iter):
        for b0 in starts:
            explore(b0, depth)
        gaps.append(float(gap(starts).mean()))
    return SawtoothSearchResult(AlphaVF(_unique_rows(np.array(Gamma))), upper, gaps)


def triangulated(P: DiscretePOMDP, m: int) -> TriangulatedVF:
    """Triangulated value iteration. Because h only goes down, one sweep from
    h = 0 to H computes every level from the level below it."""
    vf = TriangulatedVF(P, m)
    G = P.G
    for h in range(P.H + 1):
        Bf = np.zeros((vf.F.V, P.S))
        Bf[:, h * G:(h + 1) * G] = vf.F.beliefs
        U = np.einsum("vs,aost->vaot", Bf, P.M)
        Pr = U.sum(-1)
        B2 = np.divide(U, Pr[..., None], out=np.zeros_like(U), where=Pr[..., None] > 0)
        V2 = vf.values(B2.reshape(-1, P.S)).reshape(Pr.shape)
        Q = Bf @ P.R + (Pr * V2).sum(-1)
        vf.vals[h] = Q.max(1)
    return vf


# --------------------------------------------------------------------- driver --
DEFAULTS = dict(n_points=300, n_seed=30, n_iter=None, sawtooth_iter=None, search_starts=30,
                search_iter=3, tri_m=3, seed=0)


def solve(P: DiscretePOMDP, method: str, **kw):
    """Run one solver. Returns (value function for the policy, info dict)."""
    o = {**DEFAULTS, **kw}
    rng = np.random.default_rng(o["seed"])
    n_iter = o["n_iter"] or 2 * (P.H + 2)
    t0 = time.time()
    info = {}
    if method == "qmdp":
        vf = qmdp(P)
    elif method == "fib":
        vf = fib(P)
    elif method == "fast_lower":
        vf = fast_lower(P)
    elif method == "pbvi":
        vf = pbvi(P, sample_beliefs(P, o["n_points"], rng), n_iter)
    elif method == "perseus":
        vf = perseus(P, sample_beliefs(P, o["n_points"], rng), n_iter, rng)
    elif method == "point_selection":
        vf, _ = point_selection(P, o["n_points"], o["n_seed"], n_iter, rng)
    elif method == "sawtooth":
        vf = sawtooth_iteration(P, sample_beliefs(P, o["n_points"], rng), o["sawtooth_iter"] or (P.H + 2), fib(P))
    elif method == "sawtooth_search":
        res = sawtooth_search(P, sample_beliefs(P, o["search_starts"], rng, h=P.H), o["search_iter"],
                              P.H + 2, fib(P))
        vf = res.lower
        info.update(upper=res.upper, gaps=res.gaps)
    elif method == "triangulated":
        vf = triangulated(P, o["tri_m"])
    else:
        raise ValueError(method)
    info.update(seconds=time.time() - t0, size=vf.size, kind=KIND[method])
    return vf, info
