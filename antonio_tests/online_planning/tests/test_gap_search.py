"""Checks on gap heuristic search. Run:  python antonio_tests/online_planning/tests/test_gap_search.py"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tests"))

import numpy as np  # noqa: E402

from forward_search import ForwardSearch  # noqa: E402
from gap_search import GapSearch, GapSearchPolicy  # noqa: E402
from test_sparse_sampling import _belief  # noqa: E402

import autolab.runner as runner  # noqa: E402
from autolab.datasets import load_pool  # noqa: E402
from autolab.pomdp import TaskConfig  # noqa: E402


def _beliefs():
    for ds in ("p3ht", "perovskite", "crossed_barrel"):
        for seed in range(4):
            for n_obs in (3, 10):
                yield _belief(ds, n_obs=n_obs, seed=seed)


def test_upper_bound_covers_the_clairvoyant_value():
    """1 - prod(1 - p_k) >= P(some f_k >= tau), checked by sampling the GP posterior."""
    held, total = 0, 0
    for state, mu, Sig, av in _beliefs():
        gs = GapSearch(state.model)
        bound = 1 - np.prod(1 - gs.p_good(mu, np.diag(Sig)))
        L = np.linalg.cholesky(Sig + 1e-9 * np.eye(len(mu)))
        F = mu[None] + np.random.default_rng(0).standard_normal((20_000, len(mu))) @ L.T
        p_any = np.mean((F >= state.model.tau).any(1))
        held += bound >= p_any - 3 * np.sqrt(p_any * (1 - p_any) / 20_000) - 1e-9
        total += 1
    assert held == total, f"bound held on {held} of {total} beliefs"


def test_bounds_bracket_a_deeper_search():
    for state, mu, Sig, av in list(_beliefs())[::3]:
        gs, fs = GapSearch(state.model), ForwardSearch(state.model, n_actions=4)
        u2 = fs.value(mu, Sig, av, 2)
        var = np.diag(Sig)
        assert gs.lower(mu, var) <= u2 + 1e-12 <= gs.upper(mu, var) + 1e-9


def test_no_search_is_the_one_step_test():
    """With zero iterations the lower-bound decision is depth-1 forward search with the same quadrature."""
    for state, mu, Sig, av in list(_beliefs())[::2]:
        gs = GapSearch(state.model, n_actions=5, n_quad=5)
        fs = ForwardSearch(state.model, n_actions=5, n_quad_leaf=5)
        k, root = gs.plan(mu, Sig, av, depth=3, k_max=0)
        qs = {int(j): fs.q(mu, Sig, av, int(j), 1) for j in fs.actions(mu, Sig, av)}
        v_stop = fs.stop_value(mu, np.diag(Sig))
        best = max(qs, key=qs.get)
        expect = best if qs[best] > v_stop + 1e-12 else None
        assert k == expect, (k, expect)
        for j in qs:
            assert np.isclose(gs.q(root, j, "lo"), qs[j])


def test_searching_only_raises_the_lower_bound():
    for state, mu, Sig, av in list(_beliefs())[::4]:
        gs = GapSearch(state.model)
        root = gs.node(mu, Sig, av)
        los = [root.lo]
        for _ in range(15):
            gs.search(root, 3)
            los.append(root.lo)
            assert root.lo <= root.hi + 1e-12
        assert np.all(np.diff(los) >= -1e-12), los


def test_same_seed_same_campaign_and_budget():
    old = runner.make_policy
    runner.make_policy = lambda name, **kw: GapSearchPolicy(**kw)
    try:
        pool = load_pool("p3ht")
        a = runner.run_episode(pool, TaskConfig(), "g", 2, {"k_max": 10})
        b = runner.run_episode(pool, TaskConfig(), "g", 2, {"k_max": 10})
        assert a["net_utility"] == b["net_utility"] and a["n_measure"] == b["n_measure"]
        cfg = TaskConfig(budget=0.1)
        for depth in (1, 3):
            r = runner.run_episode(pool, cfg, "g", 0, {"depth": depth, "k_max": 5})
            assert r["spend"] <= cfg.budget + 1e-9
    finally:
        runner.make_policy = old


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
