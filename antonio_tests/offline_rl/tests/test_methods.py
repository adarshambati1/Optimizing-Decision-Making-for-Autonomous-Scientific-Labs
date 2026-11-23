"""Correctness tests for the offline methods on toy problems with known answers.

Run:  python antonio_tests/offline_rl/tests/test_methods.py
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from logger import FEAT_COLS, NEXT_COLS  # noqa: E402
from methods import BatchQLearning, BehavioralCloning, FittedQ, estimate_reading_model  # noqa: E402
from ope import evaluate  # noqa: E402

C = 0.01


def _toy_log(n=6000, p_behavior=0.5, seed=0):
    """A one-decision stopping problem with a known optimal policy.

    Decision 0: stop -> reward x;  continue -> reward -C, then a forced stop paying
    x + 0.1 z. Optimal: continue iff 0.1 z > C, i.e. z > 0.1 (representable by a quadratic basis).
    Features: [bias, x, z, 0, ...]. Behavior: continue with probability p_behavior.
    """
    rng = np.random.default_rng(seed)
    rows, camps = [], []
    for i in range(n):
        x, z = rng.uniform(0, 1), rng.normal()
        phi = np.r_[1.0, x, z, np.zeros(5)]
        a = int(rng.random() < p_behavior)
        base = dict(campaign=i, behavior="toy", t=0, **dict(zip(FEAT_COLS, phi)), action=a,
                    p_continue=p_behavior, p_action=p_behavior if a else 1 - p_behavior)
        if a:
            y = x + 0.1 * z
            phi2 = np.r_[1.0, y, z, np.zeros(5)]
            rows.append({**base, "forced": 0, "reward": -C, "done": 0, **dict(zip(NEXT_COLS, phi2))})
            rows.append(dict(campaign=i, behavior="toy", t=1, **dict(zip(FEAT_COLS, phi2)), action=0,
                             p_continue=0.0, p_action=1.0, forced=1, reward=y, done=1,
                             **dict(zip(NEXT_COLS, np.zeros(8)))))
        else:
            rows.append({**base, "forced": 0, "reward": x, "done": 1, **dict(zip(NEXT_COLS, np.zeros(8)))})
        camps.append(dict(campaign=i, init_cost=0.0))
    return pd.DataFrame(rows), pd.DataFrame(camps)


def _optimal(Phi):
    return (0.1 * Phi[:, 2] > C).astype(float)


def _true_value(policy, n=200_000, seed=1):
    rng = np.random.default_rng(seed)
    x, z = rng.uniform(0, 1, n), rng.normal(size=n)
    Phi = np.c_[np.ones(n), x, z, np.zeros((n, 5))]
    go = policy(Phi) > 0.5
    return float(np.mean(np.where(go, -C + x + 0.1 * z, x)))


def test_importance_sampling_is_unbiased_on_a_short_problem():
    tr, ca = _toy_log(n=20_000)
    for pol in (_optimal, lambda P: np.ones(len(P)), lambda P: np.zeros(len(P))):
        r = evaluate(tr, ca, pol)
        truth = _true_value(pol)
        assert abs(r.ordinary - truth) < 4 * r.se_ordinary + 1e-3, (r, truth)
        assert abs(r.weighted - truth) < 0.01 and abs(r.per_decision - truth) < 0.01, (r, truth)


def test_behavioral_cloning_recovers_an_expert():
    rng = np.random.default_rng(2)
    Phi = np.c_[np.ones(5000), rng.normal(size=(5000, 7))]
    y = (0.8 * Phi[:, 1] - 1.2 * Phi[:, 3] + 0.3 > 0).astype(float)          # a linear expert
    bc = BehavioralCloning().fit(Phi[:4000], y[:4000])
    assert np.mean(bc.policy(Phi[4000:]) == y[4000:]) > 0.97


def test_batch_q_learning_finds_the_optimal_stopping_rule():
    tr, _ = _toy_log(n=6000)
    test = np.c_[np.ones(4000), np.random.default_rng(3).uniform(0, 1, 4000),
                 np.random.default_rng(4).normal(size=4000), np.zeros((4000, 5))]
    for learner in (FittedQ(n_iter=10), BatchQLearning(n_updates=15_000)):
        q = learner.fit(tr)
        assert np.mean(q.policy(test) == _optimal(test)) > 0.95, type(learner).__name__
        assert abs(_true_value(q.policy) - _true_value(_optimal)) < 0.005, type(learner).__name__


def test_reading_model_counts_and_pessimism():
    rng = np.random.default_rng(5)
    n = 4000
    good = rng.random(n) < 0.2
    failed = rng.random(n) < 0.1
    high = np.where(failed, -1, (rng.random(n) < np.where(good, 0.8, 0.1)).astype(int))
    me = pd.DataFrame(dict(campaign=np.zeros(n, int), meets_spec=good.astype(int), failed=failed.astype(int), high=high))
    ca = pd.DataFrame(dict(campaign=[0], noise_sd=[0.3]))
    mle = estimate_reading_model(me, ca, 0.3, "mle")
    ok = ~failed
    assert np.isclose(mle["fail_prob"], failed.mean())
    assert np.isclose(mle["p_high_good"], (high[ok & good] == 1).mean())
    assert np.isclose(mle["p_high_bad"], (high[ok & ~good] == 1).mean())
    pess = estimate_reading_model(me, ca, 0.3, "bayes_pess")
    assert pess["p_high_good"] < mle["p_high_good"] and pess["p_high_bad"] > mle["p_high_bad"]
    assert pess["fail_prob"] > mle["fail_prob"]
    assert abs(estimate_reading_model(me, ca, 0.3, "bayes_mean")["p_high_good"] - mle["p_high_good"]) < 0.01


def test_learned_model_policy_runs_in_keiras_simulator():
    import model_based
    from autolab.datasets import load_pool
    from autolab.pomdp import TaskConfig
    import autolab.runner as runner
    from logger import load

    _, me, ca = load()
    with tempfile.TemporaryDirectory() as d:
        model_based.MODELS = d
        model_based._CACHE.clear()
        for band in range(len(model_based.band_centers())):
            model_based.solve_one((me, ca, band, "mle", "fast_lower"))
        old = runner.make_policy
        runner.make_policy = lambda name, **kw: model_based.LearnedModelPolicy(**kw)
        try:
            for mode in ("stop_only", "full"):
                cfg = TaskConfig()
                r = runner.run_episode(load_pool("p3ht"), cfg, "learned", 0,
                                       {"kind": "mle", "mode": mode, "method": "fast_lower"})
                assert r["spend"] <= cfg.budget + 1e-9
        finally:
            runner.make_policy = old
            model_based._CACHE.clear()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
