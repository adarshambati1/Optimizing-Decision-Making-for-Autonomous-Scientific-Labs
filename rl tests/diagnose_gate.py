"""Why don't the learned gates give up on perovskite?

    python "rl tests/diagnose_gate.py" --seeds 1000

Two possible explanations, which this script separates:

  A. Training failed: the one-step rule would also have scored higher on the
     gate's own SYNTHETIC training pools, and policy gradient did not find it.
  B. The gate is right for its training pools and wrong for the real ones:
     on synthetic pools the gate scores at least as well as the one-step rule,
     so the failure is a mismatch between made-up and real pools.

It plays every trained gate (greedy), the one-step rule and the baseline on
fresh synthetic pools of both kinds, never seen in training, on shared seeds.
Writes results/diagnose_synthetic.csv and results/diagnose_summary.txt.
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gate_policy import GatePolicy, VOIRulePolicy, synthetic_pool  # noqa: E402  (adds keira tests to the path)

import autolab.runner as runner  # noqa: E402
from autolab.pomdp import TaskConfig  # noqa: E402

RESULTS = os.path.join(HERE, "results")
_make_policy = runner.make_policy


def make_policy(name, **kw):
    if name.startswith("gate"):
        return GatePolicy(**kw)
    if name == "voi_rule":
        return VOIRulePolicy()
    return _make_policy(name, **kw)


runner.make_policy = make_policy


def job(args):
    policy, kw, seed, hard = args
    pool = synthetic_pool(np.random.default_rng([seed, 0]), hard)
    r = runner.run_episode(pool, TaskConfig(), policy, seed, kw)
    return dict(pools="hard" if hard else "easy", policy=policy, seed=seed,
                net_utility=r["net_utility"], n_measure=r["n_measure"],
                hit_target=r["hit_target"], abandon=not r["committed"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1000)
    a = ap.parse_args()

    kw = {"ei": {}, "voi_rule": {}}
    for path in sorted(glob.glob(os.path.join(RESULTS, "gate_weights_*.json"))):
        w = json.load(open(path))
        kw[f"gate_{w.get('tag', w['pools'])}"] = {"theta": w["theta"]}

    # seeds far from the training seeds (training used 1,000,000 and up)
    jobs = [(p, kw[p], 50_000_000 + s, hard) for hard in (False, True) for p in kw for s in range(a.seeds)]
    with mp.Pool(os.cpu_count()) as pool:
        df = pd.DataFrame(pool.map(job, jobs, chunksize=8))
    df.to_csv(os.path.join(RESULTS, "diagnose_synthetic.csv"), index=False)

    fmt = lambda v: f"{v:.3f}"  # noqa: E731
    g = df.groupby(["pools", "policy"])
    summ = g.agg(net_utility=("net_utility", "mean"),
                 net_se=("net_utility", lambda x: x.std(ddof=1) / len(x) ** 0.5),
                 hit_target=("hit_target", "mean"), abandon=("abandon", "mean"),
                 n_measure=("n_measure", "mean"), n=("seed", "count")).reset_index()
    text = f"Held-out SYNTHETIC pools, {a.seeds} seeds each\n\n" + summ.to_string(index=False, float_format=fmt) + "\n"
    text += "\nPaired difference in net utility, gate - voi_rule (same synthetic pools):\n"
    rows = []
    for pools, d in df.groupby("pools"):
        piv = d.pivot_table(index="seed", columns="policy", values="net_utility")
        for p in [c for c in piv.columns if c.startswith("gate")]:
            diff = piv[p] - piv["voi_rule"]
            se = diff.std(ddof=1) / len(diff) ** 0.5
            rows.append(dict(pools=pools, gate=p, diff=diff.mean(), se=se, z=diff.mean() / se))
    text += pd.DataFrame(rows).to_string(index=False, float_format=fmt) + "\n"
    with open(os.path.join(RESULTS, "diagnose_summary.txt"), "w") as fh:
        fh.write(text)
    print(text)
