"""Checks on the exploration pickers. Run:  python antonio_tests/exploration/tests/test_pickers.py"""
import os
import sys
from types import SimpleNamespace

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
from scipy.special import ndtr  # noqa: E402

from pickers import PICKERS, PickerPolicy, scores  # noqa: E402

import autolab.runner as runner  # noqa: E402
from autolab.datasets import load_pool  # noqa: E402
from autolab.pomdp import TaskConfig  # noqa: E402
from gate_policy import VOIRulePolicy  # noqa: E402


def test_existing_agents_are_reproduced_exactly():
    """never_ei = GP-EI, voi_ei = the one-step rule, mcts_mcts = MCTS (same seeds, same campaigns)."""
    old = runner.make_policy
    pairs = [("ei", {}, dict(stop="never", picker="ei")),
             ("voi_rule", {}, dict(stop="voi", picker="ei")),
             ("mcts", {"n_sims": 30}, dict(stop="mcts", picker="mcts", n_sims=30))]

    def make(name, **kw):
        if name == "picker":
            return PickerPolicy(**kw)
        if name == "voi_rule":
            return VOIRulePolicy()
        return old(name, **kw)
    runner.make_policy = make
    try:
        pool = load_pool("perovskite")
        for name, kw, pkw in pairs:
            for seed in (0, 1):
                a = runner.run_episode(pool, TaskConfig(), name, seed, kw)
                b = runner.run_episode(pool, TaskConfig(), "picker", seed, pkw)
                assert np.isclose(a["net_utility"], b["net_utility"]) and a["n_measure"] == b["n_measure"], (name, seed)
    finally:
        runner.make_policy = old


def _state(mu, sd, measurable, n_obs=5, noise_sd=0.3):
    return SimpleNamespace(belief=SimpleNamespace(mu=np.array(mu, float), var=np.array(sd, float) ** 2,
                                                  Sigma=np.diag(np.array(sd, float) ** 2)),
                           model=SimpleNamespace(noise_var=noise_sd ** 2), obs=[None] * n_obs,
                           measurable=lambda: np.array(measurable))


def test_pickers_follow_their_definitions():
    st = _state([0.0, 0.5, 1.0], [1.0, 0.2, 0.1], [True, True, False])
    rng = np.random.default_rng(0)
    for p in PICKERS[1:]:
        assert np.isneginf(scores(st, p, rng)[2])                     # never pick an untestable candidate
    assert np.argmax(scores(st, "greedy", rng)) == 1                   # highest mean among the testable
    assert np.argmax(scores(st, "quantile", rng)) == 0                 # 0 + 1.645 > 0.5 + 0.33
    q = scores(st, "ucb1", rng)
    assert np.isclose(q[0], 0.0 + 1.0 / 0.3 * np.sqrt(np.log(5)))


def test_thompson_picks_each_candidate_with_its_probability_of_being_best():
    st = _state([0.0, 0.4], [1.0, 0.5], [True, True])
    rng = np.random.default_rng(1)
    picks = np.array([np.argmax(scores(st, "thompson", rng)) for _ in range(20_000)])
    p_second = ndtr((0.4 - 0.0) / np.sqrt(1.0 + 0.25))
    assert abs(np.mean(picks == 1) - p_second) < 0.015


def test_budget_respected():
    old = runner.make_policy
    runner.make_policy = lambda name, **kw: PickerPolicy(**kw)
    try:
        cfg = TaskConfig(budget=0.1)
        for p in PICKERS:
            r = runner.run_episode(load_pool("p3ht"), cfg, "picker", 0, {"stop": "never", "picker": p})
            assert r["spend"] <= cfg.budget + 1e-9 and r["n_measure"] == 10
    finally:
        runner.make_policy = old


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
