"""MCTS vs. the GP-EI baseline on the three datasets.

    python experiments/benchmark.py --seeds 50
    python experiments/benchmark.py --seeds 5 --n_sims 50     # quick smoke run

Default conditions (TaskConfig): replicate-calibrated noise, 10% failed
experiments, 5% outlier readings, cost 0.01 per experiment, commit cost 0.1,
payoff 1 for meeting the top-5% spec, budget 0.5 (50 experiments).
"""
import argparse

from common import DATASETS, POLICIES, RESULTS, TaskConfig, paired_difference, run_grid, summarize

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--n_sims", type=int, default=200, help="MCTS simulations per decision")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()

    cfg = TaskConfig()
    jobs = [(ds, cfg, p, s, {"n_sims": a.n_sims} if p == "mcts" else None)
            for ds in a.datasets for p in POLICIES for s in range(a.seeds)]
    df = run_grid(jobs, a.workers)
    df.to_csv(f"{RESULTS}/benchmark.csv", index=False)

    fmt = lambda v: f"{v:.3f}"  # noqa: E731
    text = (summarize(df).to_string(index=False, float_format=fmt)
            + "\n\nPaired difference in net utility, MCTS - GP-EI (same seeds):\n"
            + paired_difference(df).to_string(index=False, float_format=fmt))
    with open(f"{RESULTS}/benchmark_summary.txt", "w") as fh:
        fh.write(text + "\n")
    print(text)
