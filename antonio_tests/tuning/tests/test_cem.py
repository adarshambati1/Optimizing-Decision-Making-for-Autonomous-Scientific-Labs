"""Checks on the CEM tuner. Run:  python antonio_tests/tuning/tests/test_cem.py"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

from cem import DEFAULT, PARAMS, campaigns_for, cross_entropy, make_policy, to_params, to_z  # noqa: E402

from autolab.datasets import load_pool  # noqa: E402
from autolab.mcts import BeliefMCTS  # noqa: E402
from autolab.pomdp import TaskConfig  # noqa: E402
from robustness import run_mismatched  # noqa: E402


def test_cross_entropy_finds_the_optimum_of_a_toy_objective():
    target = np.array([1.0, -2.0, 0.5])
    rng = np.random.default_rng(0)
    noise = np.random.default_rng(1)
    score = lambda Z, it: -((Z - target) ** 2).sum(1) + 0.01 * noise.standard_normal(len(Z))  # noqa: E731
    mu, _ = cross_entropy(score, np.zeros(3), np.eye(3) * 4, k_max=25, m=40, m_elite=8, rng=rng)
    assert np.abs(mu - target).max() < 0.1, mu


def test_parameter_mapping():
    assert to_params(to_z(DEFAULT)) == DEFAULT
    wild = to_params(np.full(len(PARAMS), 10.0))           # far outside every range: clipped to the top
    assert all(wild[k] == v[2] for k, v in PARAMS.items())
    assert all(isinstance(wild[k], int) for k, v in PARAMS.items() if v[3])


def test_default_settings_are_keiras_mcts():
    pool = load_pool("perovskite")
    a = run_mismatched(pool, TaskConfig(), TaskConfig(), make_policy(DEFAULT), 0)
    b = run_mismatched(pool, TaskConfig(), TaskConfig(), BeliefMCTS(n_sims=200), 0)
    assert np.isclose(a["net_utility"], b["net_utility"]) and a["n_measure"] == b["n_measure"]


def test_training_campaigns_never_touch_benchmark_seeds():
    for kind, seed0 in (("real", 10_000), ("synthetic", 510_000)):
        for it in range(6):
            assert min(s for _, s, _ in campaigns_for(kind, it, 60, seed0)) >= 10_000


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
