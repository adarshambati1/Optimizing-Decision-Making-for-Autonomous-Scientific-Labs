"""Solve the abstraction with all nine methods and score them on the abstraction itself.

    python antonio_tests/solve_all.py
    python antonio_tests/solve_all.py --datasets p3ht --rollouts 2      # quick smoke run

For each dataset (its noise level and spec threshold set the abstraction) and
each method this:
  * solves offline and saves (POMDP, value function) to results/solved/, which
    benchmark.py then loads;
  * records offline time and size (alpha vectors / points / vertices);
  * evaluates the value function at held-out test beliefs (an upper bound
    should sit above every lower bound);
  * runs the method's lookahead policy inside the abstract POMDP from the same
    test beliefs, which measures its true value on the model it was solved for.

Writes results/solver_benchmark.csv and results/solver_summary.txt.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pickle
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "keira tests"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from abstraction import DiscretePOMDP, params_from_model  # noqa: E402
from autolab.datasets import DATASETS, load_pool  # noqa: E402
from autolab.pomdp import AgentModel, LabTask, TaskConfig  # noqa: E402
from solvers import ALL_METHODS, sample_beliefs, solve  # noqa: E402

RESULTS = os.path.join(HERE, "results")
SOLVED = os.path.join(RESULTS, "solved")


def build_pomdp(ds: str, n_slots: int, horizon: int) -> DiscretePOMDP:
    m = AgentModel.from_task(LabTask(load_pool(ds), TaskConfig()))
    return DiscretePOMDP(params_from_model(m, n_slots=n_slots, horizon=horizon))


def rollout_value(P, vf, b0, rng) -> float:
    """Play the lookahead policy in the abstract POMDP from belief b0."""
    s, b, total = P.sample_state(b0, rng), b0.copy(), 0.0
    for _ in range(P.H + 3):
        if s == P.term:
            break
        a = int(P.lookahead(b, vf.values).argmax())
        s, o, r = P.step(s, a, rng)
        total += r
        b = P.update(b, a, o)
    return total


def job(args):
    ds, method, n_slots, horizon, n_test, rollouts = args
    P = build_pomdp(ds, n_slots, horizon)
    vf, info = solve(P, method)
    with open(os.path.join(SOLVED, f"{ds}_{method}.pkl"), "wb") as fh:
        pickle.dump((P.p, vf), fh)          # parameters only; the matrices are rebuilt on load
    test = sample_beliefs(P, n_test, np.random.default_rng(12345), h=P.H)   # same for every method
    bound = vf.values(test)
    rng = np.random.default_rng(777)
    ret = np.array([rollout_value(P, vf, b, rng) for b in test for _ in range(rollouts)])
    row = dict(dataset=ds, method=method, kind=info["kind"], offline_sec=info["seconds"], size=info["size"],
               bound_at_test=bound.mean(), policy_value=ret.mean(), policy_se=ret.std(ddof=1) / len(ret) ** 0.5,
               p_high_good=P.p.p_high_good, p_high_near_miss=P.p.p_high_bad)
    if "upper" in info:   # sawtooth heuristic search also keeps an upper bound
        row["search_upper_at_test"] = info["upper"].values(test).mean()
        row["search_final_gap"] = info["gaps"][-1]
    print(f"  {ds:15s} {method:16s} {info['seconds']:6.1f}s  bound {bound.mean():.4f}  "
          f"policy {ret.mean():.4f} +- {row['policy_se']:.4f}", flush=True)
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS)
    ap.add_argument("--n_slots", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--n_test", type=int, default=100, help="held-out test beliefs")
    ap.add_argument("--rollouts", type=int, default=20, help="abstract-POMDP rollouts per test belief")
    a = ap.parse_args()
    os.makedirs(SOLVED, exist_ok=True)

    jobs = [(ds, mt, a.n_slots, a.horizon, a.n_test, a.rollouts) for ds in a.datasets for mt in a.methods]
    with mp.Pool(min(len(jobs), os.cpu_count())) as pool:
        df = pd.DataFrame(pool.map(job, jobs, chunksize=1))
    df.to_csv(os.path.join(RESULTS, "solver_benchmark.csv"), index=False)
    cols = ["dataset", "method", "kind", "offline_sec", "size", "bound_at_test", "policy_value", "policy_se"]
    text = (f"Abstraction: {a.n_slots} slots, horizon {a.horizon} "
            f"({(a.horizon + 1) * 2 ** a.n_slots + 1} states, {2 * a.n_slots + 1} actions, 3 observations)\n"
            f"{a.n_test} test beliefs at the full horizon, {a.rollouts} rollouts each\n\n"
            + df[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}") + "\n")
    if "search_final_gap" in df:
        s = df.dropna(subset=["search_final_gap"])
        text += ("\nSawtooth heuristic search, both bounds at the test beliefs:\n"
                 + s[["dataset", "bound_at_test", "search_upper_at_test", "search_final_gap"]]
                 .to_string(index=False, float_format=lambda v: f"{v:.4f}") + "\n")
    with open(os.path.join(RESULTS, "solver_summary.txt"), "w") as fh:
        fh.write(text)
    print("\n" + text)
