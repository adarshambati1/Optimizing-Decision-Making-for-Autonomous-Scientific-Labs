"""Train the stopping gate by policy gradient on synthetic pools.

    python "rl tests/train_gate.py" --pools easy          # smooth pools (a few minutes)
    python "rl tests/train_gate.py" --pools hard          # also rough, noisy, partly unpredictable pools
    python "rl tests/train_gate.py" --iters 2 --batch 28  # smoke run

Each iteration:
  1. play `batch` campaigns on fresh synthetic pools with the current stochastic gate;
  2. estimate the policy gradient with the likelihood-ratio trick, reward-to-go
     and baseline subtraction (book sections 11.3-11.5):

         g = mean over campaigns of  sum_t  grad log pi(a_t | s_t) * (R_t - b(s_t))

     where R_t is the reward-to-go and b(s) is the value of stopping now;
  3. take a fixed-length step in the gradient direction (restricted gradient
     update, section 12.2), with the step length decaying over training.

Add --voi to give the gate the value-of-information input.
Writes results/gate_weights_<tag>.json and results/training_curve_<tag>.csv.
"""
import argparse
import json
import multiprocessing as mp
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from gate_policy import FEATURES, HERE, N_BASE, p_continue, rollout  # noqa: E402

RESULTS = os.path.join(HERE, "results")


def gradient(theta, batch):
    g = np.zeros_like(theta)
    for ep in batch:
        for phi, a, rtg, b in zip(ep["phi"], ep["a"], ep["rtg"], ep["base"]):
            p = p_continue(theta, phi)
            g += (a - p) * phi * (rtg - b)       # grad log pi = (a - p) phi for a logistic policy
    return g / len(batch)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--batch", type=int, default=280, help="campaigns per iteration")
    ap.add_argument("--step0", type=float, default=0.4)
    ap.add_argument("--step1", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pools", default="easy", choices=["easy", "hard"],
                    help="which synthetic training pools to use (see gate_policy.synthetic_pool)")
    ap.add_argument("--voi", action="store_true",
                    help="give the gate the value-of-information input (8 weights, not 7)")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    feats = FEATURES if a.voi else FEATURES[:N_BASE]
    tag = a.pools + ("_voi" if a.voi else "")
    theta = np.zeros(len(feats))
    theta[0] = 2.0                               # start by almost always continuing
    curve, tail = [], []
    with mp.Pool(a.workers or os.cpu_count()) as pool:
        for it in range(a.iters):
            seeds = [1_000_000 * (a.seed + 1) + it * a.batch + j for j in range(a.batch)]
            batch = pool.map(rollout, [(theta.tolist(), s, a.pools == "hard") for s in seeds])
            ret = np.array([ep["ret"] for ep in batch])
            n = np.array([ep["n"] for ep in batch])
            g = gradient(theta, batch)
            step = a.step0 * (a.step1 / a.step0) ** (it / max(1, a.iters - 1))
            curve.append(dict(iter=it, mean_return=ret.mean(), se=ret.std(ddof=1) / len(ret) ** 0.5,
                              mean_experiments=n.mean(), step=step,
                              **{f"theta_{f}": t for f, t in zip(feats, theta)}))
            print(f"iter {it:3d}  return {ret.mean():+.3f} +- {curve[-1]['se']:.3f}  "
                  f"experiments {n.mean():5.1f}  step {step:.3f}", flush=True)
            theta = theta + step * g / (np.linalg.norm(g) + 1e-12)
            if it >= a.iters - 10:
                tail.append(theta.copy())

    theta = np.mean(tail, axis=0)                # average the last iterates to smooth the noise
    import pandas as pd
    pd.DataFrame(curve).to_csv(os.path.join(RESULTS, f"training_curve_{tag}.csv"), index=False)
    with open(os.path.join(RESULTS, f"gate_weights_{tag}.json"), "w") as fh:
        json.dump(dict(features=feats, theta=theta.tolist(), tag=tag, pools=a.pools, iters=a.iters, batch=a.batch,
                       seed=a.seed), fh, indent=2)
    print("\nlearned weights:")
    for f, t in zip(feats, theta):
        print(f"  {f:16s} {t:+.3f}")
