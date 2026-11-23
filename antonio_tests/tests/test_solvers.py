"""Correctness tests. Run:  python antonio_tests/tests/test_solvers.py   (or pytest)"""
import os
import sys
from math import comb

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "keira tests"))

import numpy as np  # noqa: E402

from abstraction import AbstractParams, DiscretePOMDP  # noqa: E402
from solvers import ALL_METHODS, KIND, sample_beliefs, solve  # noqa: E402
from triangulation import Freudenthal  # noqa: E402


def _tiny():
    return DiscretePOMDP(AbstractParams(n_slots=2, horizon=3, cost=0.02, payoff=1.0, commit_cost=0.1,
                                        fail_prob=0.1, p_high_good=0.85, p_high_bad=0.15))


def _exact(P, b):
    """Exact optimal value by expanding the whole belief tree (tiny POMDPs only)."""
    if P.level(b) < 0:
        return 0.0
    Pr, B2 = P.successors(b)
    q = b @ P.R
    for a in P.measure_actions:
        q[a] += sum(Pr[a, o] * _exact(P, B2[a, o]) for o in range(P.O_n) if Pr[a, o] > 0)
    return float(q.max())


def test_matrices_are_distributions():
    P = _tiny()
    assert np.allclose(P.T.sum(-1), 1) and np.allclose(P.O.sum(-1), 1)
    b = P.product_belief([0.3, 0.6], P.H)
    assert np.isclose(b.sum(), 1) and P.level(b) == P.H
    assert np.isclose((b @ P.R)[P.commit_actions[1]], 0.6 - 0.1)


def test_freudenthal_is_exact_on_vertices_and_linear_functions():
    rng = np.random.default_rng(0)
    for d, m in [(3, 2), (4, 3), (16, 3)]:
        F = Freudenthal(d, m)
        assert F.V == comb(m + d - 1, d - 1)
        vals = rng.random(F.V)
        assert np.allclose(F.interpolate(vals, F.beliefs), vals)
        B, c = rng.dirichlet(np.full(d, 0.5), 200), rng.random(d)
        assert np.allclose(F.interpolate(F.beliefs @ c, B), B @ c)


def test_bounds_bracket_the_exact_value():
    P = _tiny()
    test = sample_beliefs(P, 25, np.random.default_rng(1), h=P.H)
    exact = np.array([_exact(P, b) for b in test])
    for mt in ALL_METHODS:
        vf, _ = solve(P, mt, n_points=60, n_seed=10, search_starts=8, tri_m=4)
        v = vf.values(test)
        if KIND[mt] == "upper":
            assert np.all(v >= exact - 1e-9), mt
        elif KIND[mt] == "lower":
            assert np.all(v <= exact + 1e-9), mt
        assert np.all(np.abs(v - exact) < 0.25), mt
    # QMDP >= FIB always; equal here because transitions are deterministic
    q, _ = solve(P, "qmdp")
    f, _ = solve(P, "fib")
    assert np.all(q.values(test) >= f.values(test) - 1e-12)


def test_policy_runs_in_keiras_simulator():
    import pickle
    import tempfile

    from abstraction import params_from_model
    from autolab.datasets import load_pool
    from autolab.pomdp import AgentModel, LabTask, TaskConfig
    from autolab.runner import run_episode
    import autolab.runner as runner
    from policy import OfflinePOMDPPolicy

    pool, cfg = load_pool("perovskite"), TaskConfig(budget=0.2)
    P = DiscretePOMDP(params_from_model(AgentModel.from_task(LabTask(pool, cfg)), n_slots=2, horizon=4))
    vf, _ = solve(P, "fib")
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as fh:
        pickle.dump((P.p, vf), fh)
    old = runner.make_policy
    runner.make_policy = lambda name, **kw: OfflinePOMDPPolicy(**kw)
    try:
        r = run_episode(pool, cfg, "pomdp_fib", seed=0, policy_kwargs={"path": fh.name})
    finally:
        runner.make_policy = old
        os.unlink(fh.name)
    assert r["spend"] <= cfg.budget + 1e-9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
