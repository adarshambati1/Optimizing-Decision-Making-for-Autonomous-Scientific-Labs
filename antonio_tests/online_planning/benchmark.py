"""Online belief-state planners (ADM ch. 22) vs. GP-EI, MCTS and the one-step rule.

    python antonio_tests/online_planning/benchmark.py --seeds 250
    python antonio_tests/online_planning/benchmark.py --seeds 250 --policies fs2 fs3   # a subset

Same simulator, default TaskConfig and seeds as every other benchmark in the
repo, so all comparisons are paired. Results are appended per policy to
results/benchmark.csv, so later planners can be added without rerunning these.

Policies:
    fs1, fs2, fs3        forward search to depth 1-3; the search also picks the experiment
    fs2_ei, fs3_ei       forward search decides only whether to continue; GP-EI picks
                         (fs1_ei is exactly the one-step rule, see the tests)
    ss2, ss3             sparse sampling to depth 2 (10 samples per action) / 3 (5 samples)
    ss2_ei, ss3_ei       the same, deciding only whether to continue
    ss2_m3               depth 2 with only 3 samples, to show the optimism of small samples
    ds2, ds3             determinized sparse tree search, depth 2 / 3, 30 scenarios
    ds2_ei, ds3_ei       the same, deciding only whether to continue
    ds3_p100             depth 3 with 100 scenarios (nodes need >= 8 to branch)
    gs3, gs3_ei          gap heuristic search, depth 3, 30 searches per decision
    gs6                  gap heuristic search, depth 6, 60 searches per decision
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "keira tests", "experiments"))
sys.path.insert(0, os.path.join(ROOT, "rl tests"))
sys.path.insert(0, HERE)

from common import DATASETS, TaskConfig, paired_difference, run_grid, summarize  # noqa: E402

import pandas as pd  # noqa: E402

import autolab.runner as runner  # noqa: E402
from forward_search import ForwardSearchPolicy  # noqa: E402
from sparse_sampling import SparseSamplingPolicy  # noqa: E402
from despot import DespotPolicy  # noqa: E402
from gap_search import GapSearchPolicy  # noqa: E402
from gate_policy import VOIRulePolicy  # noqa: E402

RESULTS = os.path.join(HERE, "results")

PLANNERS = {
    "fs1": ("fs", {"depth": 1}), "fs2": ("fs", {"depth": 2}), "fs3": ("fs", {"depth": 3}),
    "fs2_ei": ("fs", {"depth": 2, "picker": "ei"}), "fs3_ei": ("fs", {"depth": 3, "picker": "ei"}),
    "ss2": ("ss", {"depth": 2, "n_samples": 10}), "ss3": ("ss", {"depth": 3, "n_samples": 5}),
    "ss2_ei": ("ss", {"depth": 2, "n_samples": 10, "picker": "ei"}),
    "ss3_ei": ("ss", {"depth": 3, "n_samples": 5, "picker": "ei"}),
    "ss2_m3": ("ss", {"depth": 2, "n_samples": 3}),
    "ds2": ("ds", {"depth": 2}), "ds3": ("ds", {"depth": 3}),
    "ds2_ei": ("ds", {"depth": 2, "picker": "ei"}), "ds3_ei": ("ds", {"depth": 3, "picker": "ei"}),
    "ds3_p100": ("ds", {"depth": 3, "n_particles": 100, "min_particles": 8}),
    "gs3": ("gs", {"depth": 3, "k_max": 30}), "gs3_ei": ("gs", {"depth": 3, "k_max": 30, "picker": "ei"}),
    "gs6": ("gs", {"depth": 6, "k_max": 60}),
}
BASELINES = {"ei": None, "mcts": {"n_sims": 200}, "voi_rule": None}

_make_policy = runner.make_policy


def make_policy(name, **kw):
    if name in PLANNERS:
        kind, base = PLANNERS[name]
        return {"fs": ForwardSearchPolicy, "ss": SparseSamplingPolicy, "ds": DespotPolicy, "gs": GapSearchPolicy}[kind](**base)
    if name == "voi_rule":
        return VOIRulePolicy()
    return _make_policy(name, **kw)


runner.make_policy = make_policy          # at import time, so worker processes see it too


def write_summary(df: pd.DataFrame) -> str:
    fmt = lambda v: f"{v:.3f}"  # noqa: E731
    cols = ["dataset", "policy", "net_utility", "net_se", "hit_target", "abandon", "n_measure", "sec", "n"]
    text = summarize(df)[cols].sort_values(["dataset", "net_utility"], ascending=[True, False]) \
        .to_string(index=False, float_format=fmt) + "\n"
    planners = [p for p in df.policy.unique() if p not in BASELINES]
    for other, label in (("voi_rule", "one-step rule"), ("mcts", "MCTS"), ("ei", "GP-EI")):
        rows = []
        for p in planners:
            have = set(df.dataset[df.policy == p]) & set(df.dataset[df.policy == other])
            d = paired_difference(df[df.dataset.isin(have)], p, other)   # only datasets both were run on
            d.insert(1, "policy", p)
            rows.append(d)
        text += (f"\nPaired difference in net utility, policy - {label} (same seeds):\n"
                 + pd.concat(rows).sort_values(["dataset", "diff"], ascending=[True, False])
                 .to_string(index=False, float_format=fmt) + "\n")
    with open(os.path.join(RESULTS, "benchmark_summary.txt"), "w") as fh:
        fh.write(text)
    return text


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=250)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--policies", nargs="+", default=list(BASELINES) + list(PLANNERS))
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    cfg = TaskConfig()
    kw = {**BASELINES, **{p: None for p in PLANNERS}}
    jobs = [(ds, cfg, p, s, kw[p]) for ds in a.datasets for s in range(a.seeds) for p in a.policies]
    # slow planners first so the pool stays busy at the end
    jobs.sort(key=lambda j: j[2] not in ("mcts", "fs3", "fs3_ei", "ss3", "ss3_ei", "ds3_p100", "ds3", "ds3_ei", "gs6"))
    new = run_grid(jobs)

    path = os.path.join(RESULTS, "benchmark.csv")
    if os.path.exists(path):                 # keep other policies' rows, replace these
        old = pd.read_csv(path)
        old = old[~(old.policy.isin(a.policies) & old.dataset.isin(a.datasets))]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)
    print(write_summary(new))
