"""Checks on the mismatch runner. Run:  python antonio_tests/validation/tests/test_robustness.py"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

from robustness import CONDITIONS, make_agent, run_mismatched  # noqa: E402

import autolab.runner as runner  # noqa: E402
from autolab.datasets import load_pool  # noqa: E402
from autolab.policies import EIPolicy, Policy  # noqa: E402
from autolab.pomdp import LabTask, TaskConfig  # noqa: E402
from gate_policy import VOIRulePolicy  # noqa: E402


def test_no_mismatch_reproduces_keiras_runner():
    old = runner.make_policy
    runner.make_policy = lambda name, **kw: VOIRulePolicy() if name == "voi_rule" else old(name, **kw)
    try:
        for ds in ("p3ht", "crossed_barrel"):
            pool = load_pool(ds)
            for agent in ("ei", "voi_rule"):
                for seed in (0, 1, 2):
                    a = runner.run_episode(pool, TaskConfig(), agent, seed)
                    b = run_mismatched(pool, TaskConfig(), TaskConfig(), make_agent(agent), seed)
                    assert np.isclose(a["net_utility"], b["net_utility"]) and a["n_measure"] == b["n_measure"]
    finally:
        runner.make_policy = old


class _Spy(Policy):
    """Records the noise level the agent believes, then behaves like GP-EI."""
    def __init__(self):
        self.seen, self._ei = [], EIPolicy()

    def act(self, state):
        self.seen.append(state.model.noise_var)
        return self._ei.act(state) if len(self.seen) < 3 else ("abandon",)


def test_agent_keeps_its_assumed_model_while_the_world_changes():
    pool = load_pool("perovskite")
    default_sd = LabTask(pool, TaskConfig()).noise_sd
    for cond in ("noise_x4", "known_noise_x4"):
        true_kw, assumed_kw = CONDITIONS[cond]
        spy = _Spy()
        run_mismatched(pool, TaskConfig(**true_kw), TaskConfig(**assumed_kw), spy, 0)
        believed = np.sqrt(spy.seen[0])
        assert np.isclose(believed, default_sd * (4 if cond.startswith("known") else 1))
    # and the world really is noisier: readings scatter about 4x more around the truth
    rng = np.random.default_rng(0)
    for scale in (1.0, 4.0):
        task = LabTask(pool, TaskConfig(noise_scale=scale, outlier_prob=0.0, fail_prob=0.0))
        ys = np.array([task.measure(0, rng)[1] for _ in range(4000)])
        assert abs(ys.std() / (scale * default_sd) - 1) < 0.05


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
