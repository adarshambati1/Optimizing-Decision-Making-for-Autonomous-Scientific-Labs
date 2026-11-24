"""Tune MCTS with the cross-entropy method (ADM section 10.4, algorithm 10.4).

    python antonio_tests/tuning/cem.py --pools real        # tune on the real pools, training seeds
    python antonio_tests/tuning/cem.py --pools synthetic   # tune on made-up pools only
    python antonio_tests/tuning/cem.py --test              # evaluate the tuned settings on held-out seeds

The cross-entropy method keeps a Gaussian search distribution over policy
parameters. Each iteration draws m parameter vectors, scores each by
simulation, and refits the Gaussian (mean and full covariance, by maximum
likelihood) to the m_elite best. Here the policy is Keira's MCTS with its
compute budget fixed at 200 simulations per decision, so tuning cannot win by
searching more, and six settings are searched (in log space; integers rounded):

    c_ucb        exploration bonus in the tree                    default 0.5   range 0.05-5
    kappa        UCB width used to shortlist candidates           default 2     range 0.5-4
    n_measure    candidates the tree branches on                  default 6     range 2-12
    max_depth    tree depth                                       default 6     range 2-10
    min_visits   visits before a measure action can beat stopping default 5     range 1-20
    lam          planning-cost multiplier (validation/trade.py)   default 1     range 0.25-4

Score of a parameter vector: mean net utility over the iteration's campaigns.
Every vector in an iteration plays the same campaigns (common random numbers),
and campaigns are redrawn every iteration so the search cannot overfit a fixed set.

    --pools real        campaigns on P3HT, Perovskite and Crossed barrel with seeds
                        10,000 and up: never the benchmark seeds 0-249
    --pools synthetic   campaigns on Adarsh's made-up pools (half easy, half hard)

--test runs MCTS with the default and the two tuned settings on the three real
datasets, seeds 0-249, paired with every other benchmark in the repo.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
ANT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ANT, "validation"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from robustness import run_mismatched  # noqa: E402  (also puts the other folders on the path)
from trade import PlanningCost  # noqa: E402
from autolab.datasets import DATASETS, load_pool  # noqa: E402
from autolab.mcts import BeliefMCTS  # noqa: E402
from autolab.pomdp import TaskConfig  # noqa: E402
from gate_policy import synthetic_pool  # noqa: E402

RESULTS = os.path.join(HERE, "results")
PARAMS = {  # name: (default, low, high, integer?)
    "c_ucb": (0.5, 0.05, 5.0, False),
    "kappa": (2.0, 0.5, 4.0, False),
    "n_measure": (6, 2, 12, True),
    "max_depth": (6, 2, 10, True),
    "min_visits": (5, 1, 20, True),
    "lam": (1.0, 0.25, 4.0, False),
}
NAMES = list(PARAMS)
DEFAULT = {k: v[0] for k, v in PARAMS.items()}


def to_params(z: np.ndarray) -> dict:
    """Search space (log of each setting) -> clipped, rounded settings."""
    out = {}
    for zi, (k, (_, lo, hi, integer)) in zip(z, PARAMS.items()):
        v = float(np.clip(np.exp(zi), lo, hi))
        out[k] = int(round(v)) if integer else v
    return out


def to_z(params: dict) -> np.ndarray:
    return np.log(np.array([float(params[k]) for k in NAMES]))


def make_policy(params: dict):
    p = dict(params)
    lam = p.pop("lam")
    return PlanningCost(BeliefMCTS(n_sims=200, **p), lam)


_POOLS: dict = {}


def campaign(args):
    params, kind, seed, ds = args
    if kind == "synthetic":
        pool = synthetic_pool(np.random.default_rng([seed, 0]), bool(seed % 2))
    else:
        if ds not in _POOLS:
            _POOLS[ds] = load_pool(ds)
        pool = _POOLS[ds]
    r = run_mismatched(pool, TaskConfig(), TaskConfig(), make_policy(params), seed)
    return r["net_utility"]


def campaigns_for(kind: str, it: int, n: int, seed0: int) -> list:
    """The iteration's campaigns: (kind, seed, dataset), shared by every sample in the iteration."""
    base = seed0 + 1000 * it
    if kind == "synthetic":
        return [(kind, base + j, None) for j in range(n)]
    ds = list(DATASETS)
    return [(kind, base + j, ds[j % len(ds)]) for j in range(n)]


def cross_entropy(score_batch, mu, Sigma, k_max: int, m: int, m_elite: int, rng, log=None):
    """Algorithm 10.4: sample m points from N(mu, Sigma), score them, refit to the m_elite best.

    score_batch(Z, iteration) -> scores (higher is better). The current mean is always
    scored as sample 0. The refit is the maximum-likelihood Gaussian of the elites
    (plus a small floor on the covariance so it stays positive definite).
    """
    for it in range(k_max):
        Z = rng.multivariate_normal(mu, Sigma, size=m)
        Z[0] = mu
        score = np.asarray(score_batch(Z, it), float)
        elite = np.argsort(-score)[:m_elite]
        if log is not None:
            log(it, Z, score, elite)
        mu = Z[elite].mean(0)
        Sigma = np.cov(Z[elite].T) + 1e-4 * np.eye(len(mu))
    return mu, Sigma


def cem(kind: str, k_max: int, m: int, m_elite: int, n_campaigns: int, seed: int, pool) -> tuple[dict, list]:
    """Tune MCTS: cross_entropy over log-parameters, scored by simulated campaigns."""
    rng = np.random.default_rng(seed)
    lo = np.log([v[1] for v in PARAMS.values()])
    hi = np.log([v[2] for v in PARAMS.values()])
    Sigma0 = np.diag(((hi - lo) / 4) ** 2)               # +-2 sd spans each setting's range
    seed0 = 10_000 + (0 if kind == "real" else 500_000)
    history, se = [], {}

    def score_batch(Z, it):
        cs = campaigns_for(kind, it, n_campaigns, seed0)  # shared by every sample this iteration
        jobs = [(to_params(z), k, s, ds) for z in Z for (k, s, ds) in cs]
        u = np.array(pool.map(campaign, jobs, chunksize=2)).reshape(len(Z), n_campaigns)
        se[it] = u.std(1, ddof=1) / np.sqrt(n_campaigns)
        return u.mean(1)

    def log(it, Z, score, elite):
        for i, z in enumerate(Z):
            history.append(dict(iteration=it, sample=i, score=score[i], se=se[it][i], elite=bool(i in elite),
                                is_mean=(i == 0), **to_params(z)))
        print(f"  [{kind}] iter {it}: mean-params score {score[0]:+.3f}, best {score.max():+.3f}, "
              f"elite mean {score[elite].mean():+.3f}  -> {to_params(Z[elite].mean(0))}", flush=True)

    mu, _ = cross_entropy(score_batch, to_z(DEFAULT), Sigma0, k_max, m, m_elite, rng, log)
    return to_params(mu), history


def test_job(args):
    name, params, ds, seed = args
    if ds not in _POOLS:
        _POOLS[ds] = load_pool(ds)
    r = run_mismatched(_POOLS[ds], TaskConfig(), TaskConfig(), make_policy(params), seed)
    return dict(dataset=ds, policy=name, seed=seed, **r)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", choices=["real", "synthetic"])
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--elite", type=int, default=4)
    ap.add_argument("--campaigns", type=int, default=60, help="campaigns per sample per iteration")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--test_seeds", type=int, default=250)
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    with mp.Pool(os.cpu_count()) as pool:
        if a.pools:
            best, hist = cem(a.pools, a.iters, a.samples, a.elite, a.campaigns, a.seed, pool)
            pd.DataFrame(hist).to_csv(os.path.join(RESULTS, f"cem_history_{a.pools}.csv"), index=False)
            with open(os.path.join(RESULTS, f"cem_best_{a.pools}.json"), "w") as fh:
                json.dump(dict(params=best, default=DEFAULT, pools=a.pools, iters=a.iters, samples=a.samples,
                               elite=a.elite, campaigns=a.campaigns), fh, indent=1)
            print("tuned:", best)
        if a.test:
            specs = {"mcts_default": DEFAULT}
            for kind in ("real", "synthetic"):
                p = os.path.join(RESULTS, f"cem_best_{kind}.json")
                if os.path.exists(p):
                    specs[f"mcts_cem_{kind}"] = json.load(open(p))["params"]
            jobs = [(n, pr, ds, s) for n, pr in specs.items() for ds in DATASETS for s in range(a.test_seeds)]
            df = pd.DataFrame(pool.map(test_job, jobs, chunksize=2))
            df.to_csv(os.path.join(RESULTS, "cem_test.csv"), index=False)
            g = df.groupby(["dataset", "policy"])
            summ = g.agg(net_utility=("net_utility", "mean"), net_se=("net_utility", "sem"),
                         hit=("hit_target", "mean"), experiments=("n_measure", "mean"),
                         sec=("wall_time", "mean")).reset_index()
            text = summ.to_string(index=False, float_format=lambda v: f"{v:.3f}") + "\n\nPaired vs. default MCTS:\n"
            for n in specs:
                if n == "mcts_default":
                    continue
                for ds, gg in df.groupby("dataset"):
                    piv = gg.pivot_table(index="seed", columns="policy", values="net_utility")
                    d = piv[n] - piv["mcts_default"]
                    text += f"  {n:18s} {ds:15s} {d.mean():+.3f}  z = {d.mean() / (d.std(ddof=1) / np.sqrt(len(d))):+.1f}\n"
            text += "\nSettings:\n" + "\n".join(f"  {n}: {json.dumps(p)}" for n, p in specs.items()) + "\n"
            with open(os.path.join(RESULTS, "cem_test_summary.txt"), "w") as fh:
                fh.write(text)
            print(text)
