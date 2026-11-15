"""Shared parallel experiment runner."""
from __future__ import annotations

import os
import sys

# one BLAS thread per worker process (set before numpy is imported)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import multiprocessing as mp  # noqa: E402
from dataclasses import asdict  # noqa: E402

import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

from autolab.datasets import DATASETS, load_pool  # noqa: E402
from autolab.pomdp import TaskConfig  # noqa: E402
from autolab.runner import run_episode  # noqa: E402

POLICIES = ["ei", "mcts"]
_POOLS = {}


def _job(args):
    ds, cfg_dict, policy, seed, pkw = args
    if ds not in _POOLS:
        _POOLS[ds] = load_pool(ds)
    return run_episode(_POOLS[ds], TaskConfig(**cfg_dict), policy, seed, pkw)


def run_grid(jobs, workers: int | None = None) -> pd.DataFrame:
    """jobs: list of (dataset, TaskConfig, policy, seed, policy_kwargs)."""
    jobs = [(d, asdict(c), p, s, k or {}) for d, c, p, s, k in jobs]
    out = []
    with mp.Pool(workers or os.cpu_count()) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, jobs)):
            out.append(r)
            print(f"\r{i + 1}/{len(jobs)} episodes", end="", flush=True)
    print()
    return pd.DataFrame(out)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["dataset", "policy"])
    return g.agg(net_utility=("net_utility", "mean"),
                 net_se=("net_utility", lambda x: x.std(ddof=1) / len(x) ** 0.5),
                 frac_oracle=("frac_oracle", "mean"),
                 hit_target=("hit_target", "mean"),
                 abandon=("committed", lambda x: 1 - x.mean()),
                 regret=("regret", "mean"),
                 spend=("spend", "mean"),
                 n_measure=("n_measure", "mean"),
                 sec=("wall_time", "mean"),
                 n=("seed", "count")).reset_index()


def paired_difference(df: pd.DataFrame, a: str = "mcts", b: str = "ei") -> pd.DataFrame:
    """Per-dataset mean of (a - b) net utility on identical seeds, with SE."""
    rows = []
    for ds, g in df.groupby("dataset"):
        piv = g.pivot_table(index="seed", columns="policy", values="net_utility")
        d = piv[a] - piv[b]
        se = d.std(ddof=1) / len(d) ** 0.5
        rows.append(dict(dataset=ds, diff=d.mean(), se=se, z=d.mean() / se, n=len(d)))
    return pd.DataFrame(rows)
