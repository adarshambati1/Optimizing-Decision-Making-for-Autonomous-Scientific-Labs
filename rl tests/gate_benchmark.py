"""Learned stopping gate vs. MCTS vs. the GP-EI baseline on the three real pools.

    python "rl tests/train_gate.py" --pools easy     # first: train on synthetic pools
    python "rl tests/train_gate.py" --pools hard
    python "rl tests/gate_benchmark.py" --seeds 50   # benchmarks every trained gate it finds

Same simulator, TaskConfig and seeds as keira tests/experiments/benchmark.py.
The gate never saw these pools during training.
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "keira tests", "experiments"))
sys.path.insert(0, HERE)

from common import DATASETS, TaskConfig, paired_difference, run_grid, summarize  # noqa: E402

import autolab.runner as runner  # noqa: E402
from gate_policy import GatePolicy, VOIRulePolicy  # noqa: E402

RESULTS = os.path.join(HERE, "results")

# Keira's run_episode builds policies by name; teach it "gate" without editing her code.
# This runs at import time, so it also applies inside the worker processes.
_make_policy = runner.make_policy


def make_policy(name, **kw):
    if name.startswith("gate"):
        return GatePolicy(**kw)
    if name == "voi_rule":
        return VOIRulePolicy()
    return _make_policy(name, **kw)


runner.make_policy = make_policy

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--n_sims", type=int, default=200, help="MCTS simulations per decision")
    a = ap.parse_args()

    cfg = TaskConfig()
    kw = {"ei": None, "mcts": {"n_sims": a.n_sims}, "voi_rule": None}
    for path in sorted(glob.glob(os.path.join(RESULTS, "gate_weights_*.json"))):
        w = json.load(open(path))
        kw[f"gate_{w.get('tag', w['pools'])}"] = {"theta": w["theta"]}
    gates = ["voi_rule"] + [p for p in kw if p.startswith("gate")]
    jobs = [(ds, cfg, p, s, kw[p]) for ds in a.datasets for p in kw for s in range(a.seeds)]
    df = run_grid(jobs)
    df.to_csv(os.path.join(RESULTS, "gate_benchmark.csv"), index=False)

    fmt = lambda v: f"{v:.3f}"  # noqa: E731
    cols = ["dataset", "policy", "net_utility", "net_se", "hit_target", "abandon", "n_measure", "sec", "n"]
    text = summarize(df)[cols].to_string(index=False, float_format=fmt) + "\n"
    for g in gates:
        for other, label in (("mcts", "MCTS"), ("ei", "GP-EI")):
            text += (f"\nPaired difference in net utility, {g} - {label} (same seeds):\n"
                     + paired_difference(df, g, other).to_string(index=False, float_format=fmt) + "\n")
    with open(os.path.join(RESULTS, "gate_benchmark_summary.txt"), "w") as fh:
        fh.write(text)
    print(text)
