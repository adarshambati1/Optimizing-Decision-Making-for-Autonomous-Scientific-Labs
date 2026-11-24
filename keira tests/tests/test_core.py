"""Sanity tests. Run:  python tests/test_core.py   (or python -m pytest -q tests)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autolab.datasets import DATASETS, load_pool  # noqa: E402
from autolab.gp import GaussianBelief, GPHyper, is_good_reading, matern52  # noqa: E402
from autolab.pomdp import TaskConfig  # noqa: E402
from autolab.runner import run_episode  # noqa: E402


def _toy(K=40, d=3, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((K, d)), GPHyper(ls=np.full(d, 0.4), sf2=1.0), rng


def test_sequential_update_matches_batch_gp():
    X, hp, rng = _toy()
    nv, idx = 0.05, np.array([3, 7, 11, 20])
    y = rng.standard_normal(len(idx))
    b = GaussianBelief.prior(X, hp)
    for k, yk in zip(idx, y):
        b.update(k, yk, nv)
    Kxx = matern52(X[idx], X[idx], hp.ls, hp.sf2) + nv * np.eye(len(idx))
    Kpx = matern52(X, X[idx], hp.ls, hp.sf2)
    assert np.allclose(b.mu, Kpx @ np.linalg.solve(Kxx, y), atol=1e-6)
    assert np.allclose(b.Sigma, matern52(X, X, hp.ls, hp.sf2) - Kpx @ np.linalg.solve(Kxx, Kpx.T),
                       atol=1e-6)


def test_outlier_check():
    assert not is_good_reading(0.0, 1.0, 8.0, 0.1, rho=0.1, out_var=9.0)
    assert is_good_reading(0.0, 1.0, 0.2, 0.1, rho=0.1, out_var=9.0)
    assert is_good_reading(0.0, 1.0, 50.0, 0.1, rho=0.0, out_var=9.0)


def test_rejected_reading_never_enters_the_hyperparameter_fit():
    """A reading that triggers a refit is outlier-checked before the fit, not after."""
    from autolab.pomdp import AgentModel, AgentState, LabTask
    pool = load_pool("p3ht")
    model = AgentModel.from_task(LabTask(pool, TaskConfig()))
    st = AgentState(pool.X, model, np.random.default_rng(0))
    default_ls = st.hyper.ls.copy()
    for k in range(4):                                  # four ordinary readings: no fit yet
        st.observe(k, False, float(pool.f[k]))
    st.observe(4, False, 50.0)                          # 5th reading triggers the first refit; it is absurd
    assert st.good[-1] is False or st.good[-1] == False  # noqa: E712  (rejected by the outlier check)
    # with only 4 accepted readings there is nothing to fit: hyperparameters stay at the default
    assert np.allclose(st.hyper.ls, default_ls)


def test_datasets_load():
    for n in DATASETS:
        p = load_pool(n)
        assert p.X.min() >= 0 and p.X.max() <= 1
        assert abs(p.f.mean()) < 1e-9 and abs(p.f.std() - 1) < 1e-9


def test_episodes_run_and_respect_budget():
    pool = load_pool("perovskite")
    cfg = TaskConfig(fail_prob=0.2, outlier_prob=0.1, budget=0.3)
    for pol, kw in [("ei", {}), ("mcts", {"n_sims": 30})]:
        r = run_episode(pool, cfg, pol, seed=1, policy_kwargs=kw)
        assert r["spend"] <= cfg.budget + 1e-9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
