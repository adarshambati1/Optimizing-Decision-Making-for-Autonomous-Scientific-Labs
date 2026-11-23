"""Checks on determinized sparse tree search. Run:  python antonio_tests/online_planning/tests/test_despot.py"""
import os
import sys
from dataclasses import replace

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tests"))

import numpy as np  # noqa: E402

from despot import Despot, DespotPolicy  # noqa: E402
from forward_search import ForwardSearch  # noqa: E402
from sparse_sampling import SparseSampling  # noqa: E402
from test_sparse_sampling import _belief  # noqa: E402

import autolab.runner as runner  # noqa: E402
from autolab.datasets import load_pool  # noqa: E402
from autolab.pomdp import TaskConfig  # noqa: E402


def _scenarios(mu, Sig, n, depth, rng):
    L = np.linalg.cholesky(Sig + 1e-9 * np.eye(len(mu)))
    return mu[None, :] + rng.standard_normal((n, len(mu))) @ L.T, rng.random((n, depth, 3))


def test_many_particles_and_fine_bins_approach_forward_search():
    for seed in range(3):
        state, mu, Sig, av = _belief(seed=seed)
        m = replace(state.model, outlier_prob=0.0)
        fs = ForwardSearch(m, n_actions=3)
        F, Phi = _scenarios(mu, Sig, 20_000, 1, np.random.default_rng(seed))
        tree = Despot(m, F, Phi, n_actions=3, n_bins=80, min_particles=1)
        P = np.arange(len(F))
        for k in fs.actions(mu, Sig, av):
            a, b = fs.q(mu, Sig, av, int(k), 1), tree.q_particles(mu, Sig, av, P, int(k), 0, 1)
            assert abs(a - b) < 6e-3, (seed, k, a, b)


def test_determinized_scenarios_steady_action_comparisons():
    """Common random numbers: the gap between two actions' Q values varies less across
    re-draws than with sparse sampling's independent samples, at a similar sample budget."""
    state, mu, Sig, av = _belief(seed=1)
    m = state.model
    ks = [int(k) for k in ForwardSearch(m, n_actions=2).actions(mu, Sig, av)]
    gd, gs = [], []
    for r in range(40):
        F, Phi = _scenarios(mu, Sig, 30, 2, np.random.default_rng(100 + r))
        t = Despot(m, F, Phi, n_actions=3, n_bins=3, min_particles=4)
        P = np.arange(30)
        gd.append(t.q_particles(mu, Sig, av, P, ks[0], 0, 2) - t.q_particles(mu, Sig, av, P, ks[1], 0, 2))
        ss = SparseSampling(m, np.random.default_rng(100 + r), n_samples=6, n_actions=3)
        gs.append(ss.q(mu, Sig, av, ks[0], 2) - ss.q(mu, Sig, av, ks[1], 2))
    assert np.std(gd) < np.std(gs), (np.std(gd), np.std(gs))


def test_same_seed_same_campaign_and_budget():
    old = runner.make_policy
    runner.make_policy = lambda name, **kw: DespotPolicy(**kw)
    try:
        pool = load_pool("p3ht")
        a = runner.run_episode(pool, TaskConfig(), "d", 3, {"depth": 2, "n_particles": 12})
        b = runner.run_episode(pool, TaskConfig(), "d", 3, {"depth": 2, "n_particles": 12})
        assert a["net_utility"] == b["net_utility"] and a["n_measure"] == b["n_measure"]
        cfg = TaskConfig(budget=0.1)
        for depth in (1, 2, 3):
            r = runner.run_episode(pool, cfg, "d", 0, {"depth": depth, "n_particles": 10, "n_actions": 3})
            assert r["spend"] <= cfg.budget + 1e-9
    finally:
        runner.make_policy = old


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
