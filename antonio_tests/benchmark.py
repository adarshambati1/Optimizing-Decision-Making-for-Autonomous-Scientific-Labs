"""The nine offline solvers vs. GP-EI, MCTS and the one-step rule in Keira's simulator.

    python antonio_tests/solve_all.py              # first: solve offline (writes results/solved/)
    python antonio_tests/benchmark.py --seeds 250

Same simulator, TaskConfig (default costs) and seeds as keira tests/ and
rl tests/, so every comparison is paired. Writes results/benchmark.csv and
results/benchmark_summary.txt.
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "keira tests", "experiments"))
sys.path.insert(0, os.path.join(ROOT, "rl tests"))
sys.path.insert(0, HERE)

from common import DATASETS, TaskConfig, paired_difference, run_grid, summarize  # noqa: E402

import autolab.runner as runner  # noqa: E402
from gate_policy import VOIRulePolicy  # noqa: E402
from policy import OfflinePOMDPPolicy  # noqa: E402
from solvers import ALL_METHODS  # noqa: E402

RESULTS = os.path.join(HERE, "results")
SOLVED = os.path.join(RESULTS, "solved")

# Keira's run_episode builds policies by name; add ours without editing her code.
# This runs at import time, so it also applies inside the worker processes.
_make_policy = runner.make_policy


def make_policy(name, **kw):
    if name.startswith("pomdp_"):
        return OfflinePOMDPPolicy(**kw)
    if name == "voi_rule":
        return VOIRulePolicy()
    return _make_policy(name, **kw)


runner.make_policy = make_policy

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=250)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS)
    ap.add_argument("--n_sims", type=int, default=200, help="MCTS simulations per decision")
    a = ap.parse_args()

    cfg = TaskConfig()
    jobs = []
    for ds in a.datasets:
        for s in range(a.seeds):
            jobs += [(ds, cfg, "ei", s, None), (ds, cfg, "mcts", s, {"n_sims": a.n_sims}),
                     (ds, cfg, "voi_rule", s, None)]
            for mt in a.methods:
                path = os.path.join(SOLVED, f"{ds}_{mt}.pkl")
                if not os.path.exists(path):
                    sys.exit(f"missing {path}: run antonio_tests/solve_all.py first")
                jobs.append((ds, cfg, f"pomdp_{mt}", s, {"path": path}))
    df = run_grid(jobs)
    df.to_csv(os.path.join(RESULTS, "benchmark.csv"), index=False)

    fmt = lambda v: f"{v:.3f}"  # noqa: E731
    cols = ["dataset", "policy", "net_utility", "net_se", "hit_target", "abandon", "n_measure", "sec", "n"]
    summ = summarize(df)[cols].sort_values(["dataset", "net_utility"], ascending=[True, False])
    text = summ.to_string(index=False, float_format=fmt) + "\n"
    for other, label in (("mcts", "MCTS"), ("voi_rule", "one-step rule"), ("ei", "GP-EI")):
        rows = []
        for mt in a.methods:
            d = paired_difference(df, f"pomdp_{mt}", other)
            d.insert(1, "method", mt)
            rows.append(d)
        import pandas as pd
        text += (f"\nPaired difference in net utility, method - {label} (same seeds):\n"
                 + pd.concat(rows).sort_values(["dataset", "diff"], ascending=[True, False])
                 .to_string(index=False, float_format=fmt) + "\n")
    with open(os.path.join(RESULTS, "benchmark_summary.txt"), "w") as fh:
        fh.write(text)
    print(text)
