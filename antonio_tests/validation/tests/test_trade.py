"""Checks on the trade-analysis knobs. Run:  python antonio_tests/validation/tests/test_trade.py"""
import json
import os
import sys
from dataclasses import replace

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from robustness import GATE, make_agent, run_mismatched  # noqa: E402
from trade import make, pareto  # noqa: E402

from autolab.datasets import load_pool  # noqa: E402
from autolab.pomdp import TaskConfig  # noqa: E402
from gate_policy import GatePolicy  # noqa: E402


def _run(pool, pol, seed):
    return run_mismatched(pool, TaskConfig(), TaskConfig(), pol, seed)


def test_default_knobs_are_the_agents():
    pool = load_pool("p3ht")
    pairs = [("ei_B47", make_agent("ei")), ("voi_L1", make_agent("voi_rule")),
             ("fs2_L1", make_agent("fs2_ei")), ("gate_T0.5", GatePolicy(json.load(open(GATE))["theta"]))]
    for name, ref in pairs:
        for seed in (0, 1):
            a, b = _run(pool, make(name), seed), _run(pool, ref, seed)
            assert np.isclose(a["net_utility"], b["net_utility"]) and a["n_measure"] == b["n_measure"], (name, seed)


def test_fixed_budget_and_planning_cost_move_the_right_way():
    pool = load_pool("crossed_barrel")
    for n in (2, 10):
        assert _run(pool, make(f"ei_B{n}"), 0)["n_measure"] == 3 + n
    exps = {lam: np.mean([_run(pool, make(f"voi_L{lam:g}"), s)["n_measure"] for s in range(12)])
            for lam in (0.25, 1.0, 4.0)}
    assert exps[0.25] >= exps[1.0] >= exps[4.0], exps


def test_rescaled_budget_keeps_the_affordable_experiment_count():
    from autolab.pomdp import AgentModel, AgentState, LabTask
    pool = load_pool("p3ht")
    task = LabTask(pool, TaskConfig())
    st = AgentState(pool.X, AgentModel.from_task(task), np.random.default_rng(0))
    st.spent = 0.17
    for lam in (0.25, 4.0):
        m2 = replace(st.model, cost=st.model.cost * lam, budget=st.spent + lam * (st.model.budget - st.spent))
        left_real = int(np.floor((st.model.budget - st.spent) / st.model.cost + 1e-9))
        left_seen = int(np.floor((m2.budget - st.spent) / m2.cost + 1e-9))
        assert left_real == left_seen


def test_pareto_frontier():
    pts = pd.DataFrame(dict(n_measure=[10, 20, 15, 30], hit_target=[0.5, 0.6, 0.4, 0.6]))
    assert list(pareto(pts)) == [True, True, False, False]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
