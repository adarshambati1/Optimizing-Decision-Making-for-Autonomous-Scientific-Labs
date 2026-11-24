"""Robustness analysis (ADM section 14.3): agents in worlds that differ from the model they assume.

    python antonio_tests/validation/robustness.py --seeds 100

Section 14.3: keep the policy's planning model fixed, and evaluate it in
environments that deviate from that model (its Figure 14.4 plots performance
as one true parameter moves away from the assumed value). Here the agent's
model (AgentModel) is always built from Keira's default TaskConfig:
noise = 1 x replicate noise, 5% outliers, 10% failed experiments. The
simulated world (LabTask) varies one thing at a time:

    noise_x0.5, noise_x2, noise_x4   true reading noise, as a multiple of the replicate noise
    outlier_0, outlier_0.2           true share of outlier readings
    fail_0, fail_0.3                 true share of failed experiments
    init_1, init_10                  random initial experiments (default 3). A protocol change,
                                     shared by the world and the agent, not a mismatch
    known_noise_x0.5/x2/x4           the same noise levels, but the agent is told the truth:
                                     separates "the world got harder" from "the model is wrong"
    default                          no change (the benchmark every other folder runs)

Everything else follows Keira's run_episode exactly (same seeds, same initial
design, same scoring); with no mismatch it reproduces run_episode, which the
tests check.

Writes results/robustness.csv and results/robustness_summary.txt.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import asdict

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
ANT = os.path.dirname(HERE)
ROOT = os.path.dirname(ANT)
for p in (os.path.join(ANT, "offline_rl"), os.path.join(ANT, "online_planning"), os.path.join(ROOT, "rl tests"),
          os.path.join(ROOT, "keira tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from autolab.datasets import DATASETS, load_pool  # noqa: E402
from autolab.mcts import BeliefMCTS  # noqa: E402
from autolab.policies import EIPolicy, terminal_action  # noqa: E402
from autolab.pomdp import AgentModel, AgentState, LabTask, TaskConfig  # noqa: E402

RESULTS = os.path.join(HERE, "results")
GATE = os.path.join(ROOT, "rl tests", "results", "gate_weights_hard_voi.json")
BC_MCTS = os.path.join(ANT, "offline_rl", "models", "bc_mcts.json")

# condition -> (changes to the TRUE world, changes to what the agent ASSUMES)
CONDITIONS = {
    "default": ({}, {}),
    "noise_x0.5": ({"noise_scale": 0.5}, {}), "noise_x2": ({"noise_scale": 2.0}, {}),
    "noise_x4": ({"noise_scale": 4.0}, {}),
    "known_noise_x0.5": ({"noise_scale": 0.5}, {"noise_scale": 0.5}),
    "known_noise_x2": ({"noise_scale": 2.0}, {"noise_scale": 2.0}),
    "known_noise_x4": ({"noise_scale": 4.0}, {"noise_scale": 4.0}),
    "outlier_0": ({"outlier_prob": 0.0}, {}), "outlier_0.2": ({"outlier_prob": 0.2}, {}),
    "fail_0": ({"fail_prob": 0.0}, {}), "fail_0.3": ({"fail_prob": 0.3}, {}),
    "init_1": ({"n_init": 1}, {"n_init": 1}), "init_10": ({"n_init": 10}, {"n_init": 10}),
}
AGENTS = ["ei", "mcts", "voi_rule", "gate", "fs2_ei", "ds3_ei", "bc_mcts"]


def make_agent(name):
    if name == "ei":
        return EIPolicy()
    if name == "mcts":
        return BeliefMCTS(n_sims=200)
    if name == "voi_rule":
        from gate_policy import VOIRulePolicy
        return VOIRulePolicy()
    if name == "gate":
        from gate_policy import GatePolicy
        return GatePolicy(theta=json.load(open(GATE))["theta"])
    if name == "fs2_ei":
        from forward_search import ForwardSearchPolicy
        return ForwardSearchPolicy(depth=2, picker="ei")
    if name == "ds3_ei":
        from despot import DespotPolicy
        return DespotPolicy(depth=3, picker="ei")
    if name == "bc_mcts":
        from deploy import LearnedStopPolicy
        return LearnedStopPolicy(path=BC_MCTS)
    raise ValueError(name)


def run_mismatched(pool, true_cfg: TaskConfig, assumed_cfg: TaskConfig, policy, seed: int, max_steps: int = 500):
    """Keira's run_episode (keira tests/autolab/runner.py), except that the agent's model
    comes from `assumed_cfg` while the simulated world follows `true_cfg`."""
    t0 = time.time()
    task = LabTask(pool, true_cfg)
    model = AgentModel.from_task(LabTask(pool, assumed_cfg))
    env_rng = np.random.default_rng([seed, 1])
    policy.reset(np.random.default_rng([seed, 3]))
    state = AgentState(pool.X, model, np.random.default_rng([seed, 4]))

    def do_measure(k):
        state.observe(k, *task.measure(k, env_rng))

    for k in np.random.default_rng([seed, 2]).choice(pool.K, size=true_cfg.n_init, replace=False):
        do_measure(int(k))
    for _ in range(max_steps):
        action = policy.act(state)
        if action[0] != "measure":
            break
        do_measure(action[1])
    else:
        action = terminal_action(state)
    committed = action[0] == "commit"
    k = action[1] if committed else -1
    reward = task.commit_reward(k) if committed else 0.0
    return dict(net_utility=reward - state.spent, spend=state.spent, n_measure=state.n_attempts,
                committed=committed, hit_target=bool(committed and pool.f[k] >= task.tau),
                wall_time=time.time() - t0)


_POOLS: dict = {}


def job(args):
    ds, cond, agent, seed = args
    if ds not in _POOLS:
        _POOLS[ds] = load_pool(ds)
    true_kw, assumed_kw = CONDITIONS[cond]
    r = run_mismatched(_POOLS[ds], TaskConfig(**true_kw), TaskConfig(**assumed_kw), make_agent(agent), seed)
    return dict(dataset=ds, condition=cond, agent=agent, seed=seed, **r)


def paired(df, agent, cond, base="default"):
    """Per dataset: mean change in net utility from `base` to `cond` for one agent, paired by seed."""
    out = {}
    for ds, g in df[df.agent == agent].groupby("dataset"):
        piv = g.pivot_table(index="seed", columns="condition", values="net_utility")
        if cond in piv and base in piv:
            d = piv[cond] - piv[base]
            out[ds] = (d.mean(), d.std(ddof=1) / np.sqrt(len(d)))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--agents", nargs="+", default=AGENTS)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    jobs = [(ds, c, ag, s) for ds in a.datasets for c in a.conditions for ag in a.agents for s in range(a.seeds)]
    jobs.sort(key=lambda j: j[2] != "mcts")
    out = []
    with mp.Pool(os.cpu_count()) as pool:
        for i, r in enumerate(pool.imap_unordered(job, jobs, chunksize=4)):
            out.append(r)
            if (i + 1) % 500 == 0 or i + 1 == len(jobs):
                print(f"\r{i + 1}/{len(jobs)} campaigns", end="", flush=True)
    print()
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(RESULTS, "robustness.csv"), index=False)

    fmt = lambda v: f"{v:.3f}"  # noqa: E731
    tab = df.groupby(["dataset", "condition", "agent"]).net_utility.mean().unstack("agent")[a.agents]
    exp = df.groupby(["dataset", "condition", "agent"]).n_measure.mean().unstack("agent")[a.agents]
    text = (f"{a.seeds} paired seeds per (dataset, condition, agent). Agent's model: Keira's defaults unless the\n"
            "condition says otherwise (see robustness.py). Mean net utility:\n\n"
            + tab.to_string(float_format=fmt) + "\n\nMean experiments run:\n\n" + exp.to_string(float_format=fmt) + "\n")
    rows = []
    for ag in a.agents:
        for c in a.conditions:
            if c == "default":
                continue
            for ds, (m, se) in paired(df, ag, c).items():
                rows.append(dict(agent=ag, condition=c, dataset=ds, change=m, se=se, z=m / se if se > 0 else np.nan))
    ch = pd.DataFrame(rows)
    ch.to_csv(os.path.join(RESULTS, "robustness_changes.csv"), index=False)
    text += ("\nChange in net utility from the default world, paired by seed:\n\n"
             + ch.pivot_table(index=["dataset", "condition"], columns="agent", values="change")[a.agents]
             .to_string(float_format=fmt) + "\n")
    with open(os.path.join(RESULTS, "robustness_summary.txt"), "w") as fh:
        fh.write(text)
    print(text)
