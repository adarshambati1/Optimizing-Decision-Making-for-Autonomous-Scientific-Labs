"""Checks on the offline dataset logger. Run:  python antonio_tests/offline_rl/tests/test_logger.py"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from logger import FEAT_COLS, NEXT_COLS, run_campaign  # noqa: E402  (puts Keira's and Adarsh's code on the path)

import autolab.runner as runner  # noqa: E402
from autolab.pomdp import TaskConfig  # noqa: E402
from gate_policy import VOIRulePolicy, synthetic_pool  # noqa: E402


def _campaigns(names=("ei", "voi_rule", "gate", "hazard_0.1"), eps=0.1, n=6):
    out = []
    for bi, name in enumerate(names):
        for j in range(n):
            out.append(run_campaign((name, eps, 9_000_000 + 1000 * bi + j, len(out))))
    tr = pd.DataFrame([r for t, _, _ in out for r in t])
    me = pd.DataFrame([r for _, m, _ in out for r in m])
    ca = pd.DataFrame([c for _, _, c in out])
    return tr, me, ca


def test_rewards_add_up_to_net_utility():
    tr, _, ca = _campaigns()
    ret = tr.groupby("campaign")["reward"].sum()
    assert np.allclose(ret.loc[ca.campaign].to_numpy() - ca.init_cost.to_numpy(), ca.net_utility.to_numpy())


def test_one_stop_row_at_the_end_and_next_features_line_up():
    tr, _, _ = _campaigns()
    for _, g in tr.groupby("campaign"):
        assert g.done.sum() == 1 and g.done.iloc[-1] == 1 and (g.action.iloc[:-1] == 1).all()
        cur, nxt = g[FEAT_COLS].to_numpy(), g[NEXT_COLS].to_numpy()
        assert np.allclose(nxt[:-1], cur[1:]) and np.allclose(nxt[-1], 0)


def test_logged_probabilities_support_importance_sampling():
    eps = 0.1
    tr, _, _ = _campaigns(eps=eps)
    free = tr[tr.forced == 0]
    p = np.where(free.action == 1, free.p_continue, 1 - free.p_continue)
    assert np.allclose(p, free.p_action)
    hz = free.behavior.str.startswith("hazard")
    assert (free.p_action[~hz] >= eps / 2 - 1e-12).all()   # both actions always possible
    assert (tr.p_action > 0).all() and (tr.p_action <= 1).all()


def test_measurements_match_campaigns():
    tr, me, ca = _campaigns()
    n = me.groupby("campaign").size()
    assert (n.loc[ca.campaign].to_numpy() == ca.n_measure.to_numpy()).all()
    assert ((me.initial == 1).groupby(me.campaign).sum() == TaskConfig().n_init).all()
    assert (me.high[me.failed == 1] == -1).all() and me.y[me.failed == 1].isna().all()
    assert set(me.meets_spec[me.near_miss == 1]) <= {0}


def test_unrandomized_logger_reproduces_the_team_agents():
    """eps = 0 must reproduce GP-EI and the one-step rule exactly, as run by Keira's run_episode."""
    old = runner.make_policy
    runner.make_policy = lambda name, **kw: VOIRulePolicy() if name == "voi_rule" else old(name, **kw)
    try:
        for name in ("ei", "voi_rule"):
            for seed in (11, 12, 13):
                _, _, camp = run_campaign((name, 0.0, seed, 0))
                pool = synthetic_pool(np.random.default_rng([seed, 0]), bool(seed % 2))
                r = runner.run_episode(pool, TaskConfig(), name, seed)
                assert np.isclose(camp["net_utility"], r["net_utility"]), (name, seed)
                assert camp["n_measure"] == r["n_measure"], (name, seed)
    finally:
        runner.make_policy = old


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
