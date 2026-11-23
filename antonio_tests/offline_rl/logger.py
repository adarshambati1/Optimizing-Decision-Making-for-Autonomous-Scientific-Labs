"""Generate the offline dataset: logged campaigns on SYNTHETIC pools.

    python antonio_tests/offline_rl/logger.py                       # full dataset (a few minutes)
    python antonio_tests/offline_rl/logger.py --scale 0.05          # quick smoke run

Campaigns run on made-up pools from Adarsh's generator (rl tests/gate_policy.py,
half "easy", half "hard"), never on the three real datasets, so the real pools
stay a held-out test, as in his gate experiments. Task settings are Keira's
defaults (TaskConfig()).

Three tables are written to antonio_tests/offline_rl/data/ (git-ignored;
regenerate with this script):

transitions.csv.gz   one row per decision
    campaign, behavior, seed, hard, t (decision index),
    f_bias .. f_voi_ratio   the 8 belief features (gate_policy.FEATURES)
    action                  1 = continue, 0 = stop
    p_continue              behavior's probability of continuing here (after eps mixing)
    p_action                probability of the logged action (for importance sampling)
    forced                  1 if stopping was forced (budget spent or nothing left to test)
    reward                  -cost for continue; the true terminal payoff for stop
    done                    1 on the stop row
    next_f_*                features of the next decision (zeros when done)

measurements.csv.gz  one row per experiment, including the shared initial design
    campaign, step, initial, k, f_true, tau, meets_spec, near_miss,
    mu_before, sd_before, p_good_before, failed, y, high (y >= tau), accepted

campaigns.csv.gz     one row per campaign
    campaign, behavior, eps, seed, hard, net_utility, logged_return, init_cost,
    n_measure, committed, hit_target, K, d,
    noise_sd, tau            the reading noise and spec threshold (both known to the agent)

Reward bookkeeping: logged_return (the sum of `reward` over the campaign's
rows) - init_cost (the initial design's cost, paid before any decision)
equals net_utility exactly, which the tests check. init_cost is the same for
every policy, so it does not affect which policy is better.
"""
from __future__ import annotations

import argparse
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
from scipy.special import ndtr  # noqa: E402

from behavior import make_behavior, mix  # noqa: E402  (also puts Keira's and Adarsh's code on the path)
from autolab.policies import EIPolicy, terminal_action  # noqa: E402
from autolab.pomdp import AgentModel, AgentState, LabTask, TaskConfig  # noqa: E402
from gate_policy import FEATURES, features, synthetic_pool  # noqa: E402

DATA = os.path.join(HERE, "data")
FEAT_COLS = [f"f_{n}" for n in FEATURES]
NEXT_COLS = [f"next_{c}" for c in FEAT_COLS]

# behavior -> (campaigns at scale 1, eps). Seeds are disjoint across behaviors.
PLAN = {
    "ei": (400, 0.10),
    "voi_rule": (800, 0.10),
    "gate": (800, 0.10),
    "mcts": (400, 0.10),
    "hazard_0.02": (300, 0.0),
    "hazard_0.05": (300, 0.0),
    "hazard_0.1": (300, 0.0),
    "hazard_0.2": (300, 0.0),
}
SEED0 = 70_000_000          # far from Adarsh's training (1e6+) and diagnostic (5e7+) seeds
CONTENDER_Q = 0.75          # "near miss": score between this pool quantile and the spec


def run_campaign(args) -> tuple[list, list, dict]:
    behavior_name, eps, seed, campaign = args
    hard = bool(seed % 2)
    cfg = TaskConfig()
    pool = synthetic_pool(np.random.default_rng([seed, 0]), hard)
    task = LabTask(pool, cfg)
    env_rng = np.random.default_rng([seed, 1])
    beh_rng = np.random.default_rng([seed, 5])
    state = AgentState(pool.X, AgentModel.from_task(task), np.random.default_rng([seed, 4]))
    beh = make_behavior(behavior_name)
    beh.reset(np.random.default_rng([seed, 3]))
    ei = EIPolicy()
    near_lo = pool.quantile(CONTENDER_Q)
    meas_rows = []

    def measure(k, step, initial):
        b, m = state.belief, state.model
        mu, sd = float(b.mu[k]), float(np.sqrt(b.var[k]))
        failed, y = task.measure(k, env_rng)
        state.observe(k, failed, y)
        # state.good[-1] is the outlier check's verdict on this reading (also after a refit,
        # which re-checks every reading in order)
        accepted = (not failed) and bool(state.good[-1])
        f = float(pool.f[k])
        meas_rows.append(dict(campaign=campaign, step=step, initial=int(initial), k=int(k), f_true=f,
                              tau=task.tau, meets_spec=int(f >= task.tau),
                              near_miss=int(near_lo <= f < task.tau), mu_before=mu, sd_before=sd,
                              p_good_before=float(ndtr((mu - task.tau) / sd)), failed=int(failed),
                              y=np.nan if failed else y, high=-1 if failed else int(y >= task.tau),
                              accepted=int(accepted)))

    for k in np.random.default_rng([seed, 2]).choice(pool.K, size=cfg.n_init, replace=False):
        measure(int(k), -1, True)
    init_cost = state.spent

    rows, t = [], 0
    while True:
        phi = features(state, len(FEATURES))
        if not state.measurable().any():
            p_cont, a, forced = 0.0, 0, 1
        else:
            p_cont = mix(beh.p_base(state, phi), eps)
            a, forced = int(beh_rng.random() < p_cont), 0
        row = dict(campaign=campaign, behavior=behavior_name, seed=seed, hard=int(hard), t=t,
                   **dict(zip(FEAT_COLS, phi)), action=a, p_continue=p_cont,
                   p_action=p_cont if a else 1.0 - p_cont, forced=forced)
        if a:
            row.update(reward=-cfg.cost, done=0)
            rows.append(row)
            measure(ei.act(state)[1], t, False)
            t += 1
            continue
        action = terminal_action(state)
        final = task.commit_reward(action[1]) if action[0] == "commit" else 0.0
        row.update(reward=final, done=1)
        rows.append(row)
        break

    for i, r in enumerate(rows):                                   # next-state features
        nxt = rows[i + 1] if i + 1 < len(rows) else None
        for c, nc in zip(FEAT_COLS, NEXT_COLS):
            r[nc] = nxt[c] if (nxt is not None and not r["done"]) else 0.0

    committed = action[0] == "commit"
    camp = dict(campaign=campaign, behavior=behavior_name, eps=eps, seed=seed, hard=int(hard),
                net_utility=final - state.spent, logged_return=float(sum(r["reward"] for r in rows)),
                init_cost=init_cost, n_measure=state.n_attempts, committed=int(committed),
                hit_target=int(committed and pool.f[action[1]] >= task.tau), K=pool.K, d=pool.X.shape[1],
                noise_sd=float(task.noise_sd), tau=float(task.tau))
    return rows, meas_rows, camp


def build_jobs(scale: float) -> list:
    jobs, c = [], 0
    for bi, (name, (n, eps)) in enumerate(PLAN.items()):
        for j in range(max(1, int(round(n * scale)))):
            jobs.append((name, eps, SEED0 + 100_000 * bi + j, c))
            c += 1
    return jobs


def load(data_dir: str = DATA) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(transitions, measurements, campaigns)."""
    return tuple(pd.read_csv(os.path.join(data_dir, f"{n}.csv.gz"))
                 for n in ("transitions", "measurements", "campaigns"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every campaign count")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default=DATA)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    jobs = build_jobs(a.scale)
    # MCTS campaigns are slow, so start them first
    jobs.sort(key=lambda j: j[0] != "mcts")
    tr, me, ca = [], [], []
    with mp.Pool(a.workers or os.cpu_count()) as pool:
        for i, (r, m, c) in enumerate(pool.imap_unordered(run_campaign, jobs, chunksize=2)):
            tr += r
            me += m
            ca.append(c)
            print(f"\r{i + 1}/{len(jobs)} campaigns", end="", flush=True)
    print()
    for name, rows in (("transitions", tr), ("measurements", me), ("campaigns", ca)):
        df = pd.DataFrame(rows).sort_values(["campaign"] + (["t"] if name == "transitions" else
                                                           ["step"] if name == "measurements" else []),
                                            kind="stable")
        df.to_csv(os.path.join(a.out, f"{name}.csv.gz"), index=False)
        print(f"  {name}: {len(df)} rows")
