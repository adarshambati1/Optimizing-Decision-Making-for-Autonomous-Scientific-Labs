"""Freudenthal triangulation of the belief simplex (Lovejoy 1991; ADM section 21.8).

A belief b over d states maps to x in R^d with x_i = m * sum_{j >= i} b_j, so
m = x_1 >= x_2 >= ... >= x_d >= 0. The lattice points of this region (integer
x) are the vertices; there are C(m + d - 1, d - 1) of them. A belief lies in
exactly one Freudenthal simplex: with v = floor(x), r = x - v and the
coordinates sorted so that r_p1 >= r_p2 >= ..., the simplex has vertices
v, v + e_p1, v + e_p1 + e_p2, ... and barycentric weights
1 - r_p1, r_p1 - r_p2, ..., r_p(d-1).
"""
from __future__ import annotations

import numpy as np


class Freudenthal:
    def __init__(self, d: int, m: int):
        self.d, self.m = d, m
        self.base = (m + 1) ** np.arange(d - 1, dtype=np.int64)
        verts = []

        def rec(prefix, hi):
            if len(prefix) == d - 1:
                verts.append(prefix)
                return
            for v in range(hi, -1, -1):
                rec(prefix + [v], v)

        rec([], m)
        self.X = np.array(verts, dtype=np.int64)          # (V, d - 1): x_2 .. x_d
        keys = self.X @ self.base
        order = np.argsort(keys)
        self.X, self.keys = self.X[order], keys[order]
        self.V = len(self.X)
        full = np.c_[np.full(self.V, m), self.X, np.zeros(self.V, np.int64)]
        self.beliefs = (full[:, :-1] - full[:, 1:]) / m    # (V, d) vertex beliefs

    def weights(self, B: np.ndarray):
        """Vertex indices (N, d) and barycentric weights (N, d) for beliefs B (N, d)."""
        B = np.asarray(B, float)
        x = self.m * np.cumsum(B[:, ::-1], axis=1)[:, ::-1][:, 1:]      # (N, d - 1)
        x = np.clip(x, 0.0, self.m)
        v = np.floor(x + 1e-12)
        v = np.minimum(v, self.m)
        r = np.clip(x - v, 0.0, 1.0)
        perm = np.argsort(-r, axis=1, kind="stable")
        rs = np.take_along_axis(r, perm, axis=1)
        N, k = x.shape
        steps = np.zeros((N, k + 1, k), np.int64)
        rows = np.arange(N)[:, None]
        inc = np.zeros((N, k), np.int64)
        for j in range(k):
            inc[rows[:, 0], perm[:, j]] = 1
            steps[:, j + 1] = inc
        Vx = v[:, None, :].astype(np.int64) + steps                     # (N, d, d - 1)
        lam = np.empty((N, k + 1))
        lam[:, 0] = 1 - rs[:, 0]
        lam[:, 1:k] = rs[:, :-1] - rs[:, 1:]
        lam[:, k] = rs[:, -1]
        Vx = np.minimum(Vx, self.m)                                    # only zero-weight vertices can overflow
        keys = Vx @ self.base
        ids = np.searchsorted(self.keys, keys)
        ids = np.minimum(ids, self.V - 1)
        bad = self.keys[ids] != keys
        if bad.any():   # can only happen for zero-weight vertices pushed outside the region
            assert np.all(lam[bad] < 1e-9), "Freudenthal vertex outside the simplex"
            ids[bad] = 0
            lam[bad] = 0.0
        return ids, lam

    def interpolate(self, values: np.ndarray, B: np.ndarray) -> np.ndarray:
        ids, lam = self.weights(B)
        return (values[ids] * lam).sum(1)
