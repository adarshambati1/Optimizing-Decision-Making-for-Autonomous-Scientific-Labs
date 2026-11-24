"""Exploration strategies (ADM 15.4) for choosing the experiment, under three stopping rules.

    python antonio_tests/exploration/benchmark.py --seeds 250

Every (stopping rule, picker) pair in pickers.py on the three real datasets,
same TaskConfig and seeds as every other benchmark in the repo (paired).
never_ei is GP-EI, voi_ei is the one-step rule and mcts_mcts is MCTS; the tests
check these reproduce those agents exactly.

Writes results/benchmark.csv and results/benchmark_summary.txt.
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "keira tests", "experiments"))
sys.path.insert(0, HERE)

from common import DATASETS, TaskConfig, paired_difference, run_grid, summarize  # noqa: E402

import pandas as pd  # noqa: E402

import autolab.runner as runner  # noqa: E402
from pickers import PICKERS, STOPS, PickerPolicy  # noqa: E402

RESULTS = os.path.join(HERE, "results")
POLICIES = [f"{s}_{p}" for s in STOPS for p in PICKERS] + ["mcts_mcts"]
REFERENCE = {"never": "never_ei", "voi": "voi_ei", "mcts": "mcts_mcts"}   # GP-EI, one-step rule, MCTS

_make_policy = runner.make_policy


def make_policy(name, **kw):
    stop, picker = name.split("_", 1)
    if stop in STOPS:
        return PickerPolicy(stop=stop, picker=picker)
    return _make_policy(name, **kw)


runner.make_policy = make_policy          # at import time, so worker processes see it too


def write_summary(df: pd.DataFrame) -> str:
    fmt = lambda v: f"{v:.3f}"  # noqa: E731
    cols = ["dataset", "policy", "net_utility", "net_se", "hit_target", "abandon", "n_measure", "sec", "n"]
    text = summarize(df)[cols].sort_values(["dataset", "net_utility"], ascending=[True, False]) \
        .to_string(index=False, float_format=fmt) + "\n"
    for stop, ref in REFERENCE.items():
        rows = []
        for p in [q for q in df.policy.unique() if q.startswith(stop + "_") and q != ref]:
            d = paired_difference(df, p, ref)
            d.insert(1, "policy", p)
            rows.append(d)
        if rows:
            text += (f"\nPaired difference in net utility, picker - {ref} (same stopping rule '{stop}', same seeds):\n"
                     + pd.concat(rows).sort_values(["dataset", "diff"], ascending=[True, False])
                     .to_string(index=False, float_format=fmt) + "\n")
    rows = []
    for p in [q for q in df.policy.unique() if q != "voi_ei"]:
        d = paired_difference(df, p, "voi_ei")
        d.insert(1, "policy", p)
        rows.append(d)
    text += ("\nPaired difference in net utility, policy - one-step rule (voi_ei):\n"
             + pd.concat(rows).sort_values(["dataset", "diff"], ascending=[True, False])
             .to_string(index=False, float_format=fmt) + "\n")
    with open(os.path.join(RESULTS, "benchmark_summary.txt"), "w") as fh:
        fh.write(text)
    return text


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=250)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--policies", nargs="+", default=POLICIES)
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    cfg = TaskConfig()
    jobs = [(ds, cfg, p, s, None) for ds in a.datasets for s in range(a.seeds) for p in a.policies]
    jobs.sort(key=lambda j: not j[2].startswith("mcts_"))      # slow ones first
    new = run_grid(jobs)
    path = os.path.join(RESULTS, "benchmark.csv")
    if os.path.exists(path):
        old = pd.read_csv(path)
        old = old[~(old.policy.isin(a.policies) & old.dataset.isin(a.datasets))]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)
    print(write_summary(new))
