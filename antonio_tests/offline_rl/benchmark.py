"""Phase 3: the offline-learned policies on the three REAL datasets.

    python antonio_tests/offline_rl/logger.py     # the log (made-up pools only)
    python antonio_tests/offline_rl/train.py      # train the five methods
    python antonio_tests/offline_rl/benchmark.py --seeds 250

Nothing here was trained on these pools: every learned policy saw only the log
of made-up pools. Same simulator, TaskConfig and seeds as keira tests/,
rl tests/ and antonio_tests/benchmark.py, so every comparison is paired.

Also reports each model-free policy's importance-sampling estimate from the
(made-up-pool) log next to its true score here, which shows how far an
offline estimate can be trusted when the deployment pools differ from the log.

Writes results/benchmark.csv, results/benchmark_summary.txt.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "keira tests", "experiments"))
sys.path.insert(0, HERE)

from common import DATASETS, TaskConfig, paired_difference, run_grid, summarize  # noqa: E402

import pandas as pd  # noqa: E402

import train  # noqa: E402,F401  (registers the learned policies with Keira's runner, also in workers)
from behavior import GATE_WEIGHTS  # noqa: E402
from model_based import KINDS, MODELS  # noqa: E402

RESULTS = os.path.join(HERE, "results")

LEARNED = ["bc_voi", "bc_mcts", "batch_q", "fitted_q"]


def policy_specs(n_sims: int) -> dict:
    specs = {"ei": None, "mcts": {"n_sims": n_sims}, "voi_rule": None,
             "gate": {"theta": json.load(open(GATE_WEIGHTS))["theta"]}}
    for name in LEARNED:
        specs[f"learned_{name}"] = {"path": os.path.join(MODELS, f"{name}.json")}
    for kind in KINDS:
        for mode in ("stop_only", "full"):
            specs[f"model_{kind}_{mode}"] = {"kind": kind, "mode": mode}
    return specs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=250)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--n_sims", type=int, default=200, help="MCTS simulations per decision")
    a = ap.parse_args()

    specs = policy_specs(a.n_sims)
    cfg = TaskConfig()
    jobs = [(ds, cfg, p, s, kw) for ds in a.datasets for s in range(a.seeds) for p, kw in specs.items()]
    df = run_grid(jobs)
    df.to_csv(os.path.join(RESULTS, "benchmark.csv"), index=False)

    fmt = lambda v: f"{v:.3f}"  # noqa: E731
    cols = ["dataset", "policy", "net_utility", "net_se", "hit_target", "abandon", "n_measure", "sec", "n"]
    summ = summarize(df)[cols].sort_values(["dataset", "net_utility"], ascending=[True, False])
    text = f"{a.seeds} paired seeds per dataset; learned policies trained on made-up pools only\n\n"
    text += summ.to_string(index=False, float_format=fmt) + "\n"
    learned = [p for p in specs if p.startswith(("learned_", "model_"))]
    for other, label in (("voi_rule", "one-step rule"), ("mcts", "MCTS"), ("gate", "Adarsh's gate"), ("ei", "GP-EI")):
        rows = []
        for p in learned:
            d = paired_difference(df, p, other)
            d.insert(1, "policy", p)
            rows.append(d)
        text += (f"\nPaired difference in net utility, policy - {label} (same seeds):\n"
                 + pd.concat(rows).sort_values(["dataset", "diff"], ascending=[True, False])
                 .to_string(index=False, float_format=fmt) + "\n")

    off = pd.read_csv(os.path.join(RESULTS, "offline_eval.csv")).set_index("policy")
    s = summarize(df).pivot(index="policy", columns="dataset", values="net_utility")
    s["made_up_pools_true"] = off.true_net
    s["log_estimate_wis"] = off.is_weighted
    text += ("\nOffline estimate (weighted IS on the made-up-pool log) vs. true net utility,\n"
             "on made-up pools and on each real dataset:\n"
             + s.loc[[p for p in s.index if p in off.index and pd.notna(off.is_weighted.get(p))]]
             .to_string(float_format=fmt) + "\n")
    with open(os.path.join(RESULTS, "benchmark_summary.txt"), "w") as fh:
        fh.write(text)
    print(text)
