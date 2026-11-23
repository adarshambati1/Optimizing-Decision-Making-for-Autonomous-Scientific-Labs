"""Checks on belief-space sparse sampling. Run:  python antonio_tests/online_planning/tests/test_sparse_sampling.py"""
import os
import sys
from dataclasses import replace

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

from forward_search import ForwardSearch, ForwardSearchPolicy  # noqa: E402
from sparse_sampling import SparseSampling, SparseSamplingPolicy  # noqa: E402

import autolab.runner as runner  # noqa: E402
from autolab.datasets import load_pool  # noqa: E402
from autolab.pomdp import AgentModel, AgentState, LabTask, TaskConfig  # noqa: E402


def _belief(ds="crossed_barrel", n_obs=6, seed=0):
    pool = load_pool(ds)
    task = LabTask(pool, TaskConfig())
    rng = np.random.default_rng(seed)
    state = AgentState(pool.X, AgentModel.from_task(task), np.random.default_rng(seed))
    for k in rng.choice(pool.K, n_obs, replace=False):
        state.observe(int(k), *task.measure(int(k), rng))
    S = ForwardSearchPolicy().working_set(state)
    return state, state.belief.mu[S], state.belief.Sigma[np.ix_(S, S)], state.measurable()[S]


def test_many_samples_converge_to_forward_search():
    """No outliers + many samples: sampled expectations match the exact quadrature."""
    for seed in range(3):
        state, mu, Sig, av = _belief(seed=seed)
        m = replace(state.model, outlier_prob=0.0)
        fs = ForwardSearch(m, n_actions=3)
        ss = SparseSampling(m, np.random.default_rng(seed), n_samples=40_000, n_actions=3)
        for k in fs.actions(mu, Sig, av):
            a, b = fs.q(mu, Sig, av, int(k), 1), ss.q(mu, Sig, av, int(k), 1)
            assert abs(a - b) < 3e-3, (seed, k, a, b)


def test_same_seed_same_campaign():
    old = runner.make_policy
    runner.make_policy = lambda name, **kw: SparseSamplingPolicy(**kw)
    try:
        pool = load_pool("p3ht")
        a = runner.run_episode(pool, TaskConfig(), "ss", 3, {"depth": 2, "n_samples": 4})
        b = runner.run_episode(pool, TaskConfig(), "ss", 3, {"depth": 2, "n_samples": 4})
        assert a["net_utility"] == b["net_utility"] and a["n_measure"] == b["n_measure"]
        cfg = TaskConfig(budget=0.1)
        for depth in (1, 2, 3):
            r = runner.run_episode(pool, cfg, "ss", 0, {"depth": depth, "n_samples": 3, "n_actions": 3})
            assert r["spend"] <= cfg.budget + 1e-9
    finally:
        runner.make_policy = old


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
