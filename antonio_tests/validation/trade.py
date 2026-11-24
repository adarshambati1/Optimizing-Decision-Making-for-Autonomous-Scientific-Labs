"""Trade analysis (ADM section 14.4): experiment spend vs. success, and the Pareto frontier.

    python antonio_tests/validation/trade.py --seeds 100

Section 14.4: vary one parameter of a policy at a time to trace a trade-off curve
between two competing metrics, then keep the policies no other policy beats on
both (Pareto optimal). The two metrics here:

    experiments run                       (cost; fewer is better)
    P(committed candidate meets spec)     (success; abandoning counts as a miss)

One knob per agent family:

    ei_B{n}       GP-EI with a fixed budget of n experiments after the 3 initial ones;
                  n = 47 is GP-EI as everywhere else
    voi_L{x}      the one-step rule, planning as if each experiment cost x times its real cost
    fs2_L{x}      depth-2 forward search (GP-EI picks), same planning-cost multiplier
    mcts_L{x}     MCTS, same planning-cost multiplier
    gate_T{t}     Adarsh's gate (hard_voi), continue iff P(continue) > t (t = 0.5 as deployed)

The planning-cost multiplier changes only what the planner believes: it sees cost
x c and a budget rescaled so it can still afford exactly the experiments the real
budget allows. Real costs and budget are unchanged. x = 1 is the agent as
everywhere else (the tests check this).

Writes results/trade.csv and results/trade_summary.txt.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from dataclasses import replace

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from robustness import GATE, run_mismatched  # noqa: E402  (also puts the other folders on the path)
from autolab.datasets import DATASETS, load_pool  # noqa: E402
from autolab.mcts import BeliefMCTS  # noqa: E402
from autolab.policies import EIPolicy, Policy, terminal_action  # noqa: E402
from autolab.pomdp import TaskConfig  # noqa: E402

RESULTS = os.path.join(HERE, "results")
LAMBDAS = [0.25, 0.5, 1.0, 2.0, 4.0]
FAMILIES = {
    "ei": [2, 5, 10, 20, 30, 47],
    "voi": LAMBDAS, "fs2": LAMBDAS, "mcts": LAMBDAS,
    "gate": [0.2, 0.35, 0.5, 0.65, 0.8],
}
KNOB = {"ei": "B", "voi": "L", "fs2": "L", "mcts": "L", "gate": "T"}


def policy_names(families=FAMILIES):
    return [f"{f}_{KNOB[f]}{v:g}" for f, vals in families.items() for v in vals]


class FixedBudgetEI(Policy):
    """GP-EI that stops after n experiments (beyond the initial design)."""

    def __init__(self, n):
        self.n, self._ei = n, EIPolicy()

    def reset(self, rng):
        super().reset(rng)
        self.done = 0

    def act(self, state):
        if self.done >= self.n or not state.measurable().any():
            return terminal_action(state)
        self.done += 1
        return self._ei.act(state)


class PlanningCost(Policy):
    """Wrap a planner so it plans as if each experiment cost lam times its real cost."""

    def __init__(self, base, lam):
        self.base, self.lam = base, lam

    def reset(self, rng):
        super().reset(rng)
        self.base.reset(rng)

    def act(self, state):
        if self.lam == 1.0:
            return self.base.act(state)
        m = state.model
        real = m
        # cost x lam, and a budget with the same number of affordable experiments left
        state.model = replace(m, cost=m.cost * self.lam, budget=state.spent + self.lam * (m.budget - state.spent))
        try:
            return self.base.act(state)
        finally:
            state.model = real


def make(name):
    fam, knob = name.split("_", 1)
    v = float(knob[1:])
    if fam == "ei":
        return FixedBudgetEI(int(v))
    if fam == "gate":
        from gate_policy import GatePolicy, features, p_continue

        class Gate(GatePolicy):
            def act(self, state):
                if not state.measurable().any():
                    return terminal_action(state)
                go = p_continue(self.theta, features(state, len(self.theta))) > v
                return self._ei.act(state) if go else terminal_action(state)
        return Gate(json.load(open(GATE))["theta"])
    if fam == "voi":
        from gate_policy import VOIRulePolicy
        return PlanningCost(VOIRulePolicy(), v)
    if fam == "fs2":
        from forward_search import ForwardSearchPolicy
        return PlanningCost(ForwardSearchPolicy(depth=2, picker="ei"), v)
    if fam == "mcts":
        return PlanningCost(BeliefMCTS(n_sims=200), v)
    raise ValueError(name)


_POOLS: dict = {}


def job(args):
    ds, name, seed = args
    if ds not in _POOLS:
        _POOLS[ds] = load_pool(ds)
    r = run_mismatched(_POOLS[ds], TaskConfig(), TaskConfig(), make(name), seed)
    fam, knob = name.split("_", 1)
    return dict(dataset=ds, policy=name, family=fam, knob=float(knob[1:]), seed=seed, **r)


def pareto(points: pd.DataFrame) -> pd.Series:
    """True for points no other point beats on both metrics (fewer experiments, higher hit rate)."""
    x, y = points.n_measure.to_numpy(), points.hit_target.to_numpy()
    keep = []
    for i in range(len(x)):
        dominated = np.any((x <= x[i]) & (y >= y[i]) & ((x < x[i]) | (y > y[i])))
        keep.append(not dominated)
    return pd.Series(keep, index=points.index)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    jobs = [(ds, p, s) for ds in a.datasets for p in policy_names() for s in range(a.seeds)]
    jobs.sort(key=lambda j: not j[1].startswith("mcts"))
    out = []
    with mp.Pool(os.cpu_count()) as pool:
        for i, r in enumerate(pool.imap_unordered(job, jobs, chunksize=4)):
            out.append(r)
            if (i + 1) % 500 == 0 or i + 1 == len(jobs):
                print(f"\r{i + 1}/{len(jobs)} campaigns", end="", flush=True)
    print()
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(RESULTS, "trade.csv"), index=False)

    g = df.groupby(["dataset", "family", "knob", "policy"])
    s = g.agg(experiments=("n_measure", "mean"), hit=("hit_target", "mean"), abandon=("committed", lambda c: 1 - c.mean()),
              net_utility=("net_utility", "mean"), net_se=("net_utility", "sem")).reset_index()
    s["pareto"] = False
    for ds, d in s.groupby("dataset"):
        s.loc[d.index, "pareto"] = pareto(d.rename(columns={"experiments": "n_measure", "hit": "hit_target"})).to_numpy()
    s.to_csv(os.path.join(RESULTS, "trade_points.csv"), index=False)
    fmt = lambda v: f"{v:.3f}"  # noqa: E731
    text = f"{a.seeds} paired seeds per point. pareto = not beaten on both fewer experiments and higher hit rate.\n"
    for ds, d in s.groupby("dataset"):
        text += f"\n=== {ds}\n" + d.drop(columns="dataset").sort_values(["family", "knob"]) \
            .to_string(index=False, float_format=fmt) + "\n"
    with open(os.path.join(RESULTS, "trade_summary.txt"), "w") as fh:
        fh.write(text)
    print(text)
