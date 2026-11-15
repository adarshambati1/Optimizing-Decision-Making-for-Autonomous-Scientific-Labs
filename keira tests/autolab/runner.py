"""Run one episode (one lab campaign) and score it against ground truth.

Common random numbers: for a given seed, both policies see the same random
initial design and the same stream of measurement noise / failures, so the
comparison between policies is paired.
"""
from __future__ import annotations

import time

import numpy as np

from .datasets import Pool
from .policies import make_policy, terminal_action
from .pomdp import AgentModel, AgentState, LabTask, TaskConfig


def run_episode(pool: Pool, cfg: TaskConfig, policy_name: str, seed: int,
                policy_kwargs: dict | None = None, max_steps: int = 500) -> dict:
    t0 = time.time()
    task = LabTask(pool, cfg)
    env_rng = np.random.default_rng([seed, 1])
    policy = make_policy(policy_name, **(policy_kwargs or {}))
    policy.reset(np.random.default_rng([seed, 3]))
    state = AgentState(pool.X, AgentModel.from_task(task), np.random.default_rng([seed, 4]))

    def do_measure(k):
        state.observe(k, *task.measure(k, env_rng))

    # shared random initial design
    for k in np.random.default_rng([seed, 2]).choice(pool.K, size=cfg.n_init, replace=False):
        do_measure(int(k))

    for _ in range(max_steps):
        action = policy.act(state)
        if action[0] != "measure":
            break
        do_measure(action[1])
    else:  # safety net: force a terminal decision
        action = terminal_action(state)

    committed = action[0] == "commit"
    k = action[1] if committed else -1
    reward = task.commit_reward(k) if committed else 0.0
    oracle = task.oracle_value()
    return dict(
        dataset=pool.name, policy=policy_name, seed=seed,
        net_utility=reward - state.spent,          # payoff - commit cost - experiment spend
        frac_oracle=(reward - state.spent) / oracle,
        spend=state.spent,
        n_measure=state.n_attempts,
        n_fail=state.n_fail,
        committed=committed,
        hit_target=bool(committed and pool.f[k] >= task.tau),
        regret=float(pool.f.max() - pool.f[k]) if committed else np.nan,
        wall_time=time.time() - t0,
    )
