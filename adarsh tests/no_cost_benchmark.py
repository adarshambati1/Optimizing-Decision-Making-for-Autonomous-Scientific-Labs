"""What happens to MCTS vs. the GP-EI baseline when experiments are free?

    python "adarsh tests/no_cost_benchmark.py" --seeds 50
    python "adarsh tests/no_cost_benchmark.py" --seeds 5 --n_sims 50     # quick smoke run

Runs Keira's benchmark (keira tests/) under two conditions on the same seeds:

  with_cost : her default TaskConfig (0.01 per experiment, budget 0.5 = 50 experiments)
  no_cost   : experiments are free, but still capped at 50 per campaign

"Free" is implemented as a negligible cost (1e-9) with a budget of 50 of them,
so the 50-experiment cap still binds while total spend is ~5e-8. Everything
else is unchanged: 10% failures, 5% outliers, commit cost 0.1, payoff 1 for a
top-5% candidate.

Results go to adarsh tests/results/.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "keira tests", "experiments"))

from common import DATASETS, POLICIES, TaskConfig, paired_difference, run_grid, summarize  # noqa: E402

RESULTS = os.path.join(HERE, "results")
MAX_EXPERIMENTS = 50
FREE = 1e-9

CONDITIONS = {
    "with_cost": TaskConfig(),
    "no_cost": TaskConfig(cost=FREE, budget=MAX_EXPERIMENTS * FREE),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--n_sims", type=int, default=200, help="MCTS simulations per decision")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    fmt = lambda v: f"{v:.3f}"  # noqa: E731
    cols = ["dataset", "policy", "net_utility", "net_se", "hit_target", "abandon", "n_measure", "n"]
    text = ""
    for name, cfg in CONDITIONS.items():
        jobs = [(ds, cfg, p, s, {"n_sims": a.n_sims} if p == "mcts" else None)
                for ds in a.datasets for p in POLICIES for s in range(a.seeds)]
        df = run_grid(jobs, a.workers)
        df.to_csv(os.path.join(RESULTS, f"{name}.csv"), index=False)
        text += (f"=== {name} ===\n"
                 + summarize(df)[cols].to_string(index=False, float_format=fmt)
                 + "\n\nPaired difference in net utility, MCTS - GP-EI (same seeds):\n"
                 + paired_difference(df).to_string(index=False, float_format=fmt) + "\n\n")
    with open(os.path.join(RESULTS, "no_cost_summary.txt"), "w") as fh:
        fh.write(text)
    print(text)
