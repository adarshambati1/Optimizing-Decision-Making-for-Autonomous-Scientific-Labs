"""Checks on belief-space forward search. Run:  python antonio_tests/online_planning/tests/test_forward_search.py"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "rl tests"))

import numpy as np  # noqa: E402

from forward_search import ForwardSearch, ForwardSearchPolicy  # noqa: E402

import autolab.runner as runner  # noqa: E402
from autolab.datasets import load_pool  # noqa: E402
from autolab.pomdp import AgentModel, AgentState, LabTask, TaskConfig  # noqa: E402
from gate_policy import VOIRulePolicy  # noqa: E402


def _register():
    old = runner.make_policy

    def make(name, **kw):
        if name == "fs":
            return ForwardSearchPolicy(**kw)
        if name == "voi_rule":
            return VOIRulePolicy()
        return old(name, **kw)
    runner.make_policy = make
    return old


def test_depth_one_with_ei_picker_is_the_one_step_rule():
    """Same decisions, so the same campaigns, as Adarsh's VOIRulePolicy (n_top = 8)."""
    old = _register()
    try:
        for ds in ("p3ht", "perovskite"):
            pool = load_pool(ds)
            for seed in range(4):
                a = runner.run_episode(pool, TaskConfig(), "voi_rule", seed)
                b = runner.run_episode(pool, TaskConfig(), "fs", seed,
                                       {"depth": 1, "n_actions": 8, "picker": "ei", "working_set": 10_000})
                assert np.isclose(a["net_utility"], b["net_utility"]) and a["n_measure"] == b["n_measure"], (ds, seed)
    finally:
        runner.make_policy = old


def test_deeper_search_never_values_a_belief_lower():
    """Stopping is always available, so U_d(b) >= U_{d-1}(b) >= V_stop(b)."""
    pool = load_pool("crossed_barrel")
    task = LabTask(pool, TaskConfig())
    rng = np.random.default_rng(0)
    for trial in range(3):
        state = AgentState(pool.X, AgentModel.from_task(task), np.random.default_rng(trial))
        for k in rng.choice(pool.K, 6, replace=False):
            state.observe(int(k), *task.measure(int(k), rng))
        S = ForwardSearchPolicy().working_set(state)
        mu, Sig, av = state.belief.mu[S], state.belief.Sigma[np.ix_(S, S)], state.measurable()[S]
        fs = ForwardSearch(state.model, n_actions=4)
        u = [fs.value(mu, Sig, av, d) for d in range(3)]
        assert u[0] <= u[1] + 1e-12 <= u[2] + 2e-12, u


def test_policies_respect_the_budget():
    old = _register()
    try:
        cfg = TaskConfig(budget=0.1)
        for depth in (1, 2, 3):
            r = runner.run_episode(load_pool("p3ht"), cfg, "fs", 0, {"depth": depth, "n_actions": 3})
            assert r["spend"] <= cfg.budget + 1e-9
    finally:
        runner.make_policy = old


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
