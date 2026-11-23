"""Train all five offline methods on the log, then evaluate them two ways.

    python antonio_tests/offline_rl/logger.py      # first: the dataset
    python antonio_tests/offline_rl/train.py

1. Train, on 70% of the logged campaigns (split by campaign, within each behavior):
     bc_voi, bc_mcts       behavioral cloning of the one-step rule / of MCTS (18.1)
     batch_q, fitted_q     batch Q-learning with experience replay, and its least-squares variant (17.2-17.7)
     mle, bayes_mean, bayes_pess   reading-model estimates per noise band, solved with PBVI (16.1, 16.4)
   Model-free models go to models/*.json; solved models to models/*.pkl (git-ignored).

2. Evaluate on made-up pools only (the real datasets are for Phase 3):
     offline  importance sampling on the held-out 30% of the log (model-free policies only:
              the model-based ones need the GP state, which the log does not store)
     online   1,000 fresh campaigns per policy in Keira's simulator, as ground truth
              for checking the offline estimates

Writes results/train_summary.txt, results/offline_eval.csv and results/models.csv.
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
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import deploy  # noqa: E402
import model_based  # noqa: E402
from behavior import GATE_WEIGHTS  # noqa: E402
from logger import PLAN, load  # noqa: E402
from methods import BatchQLearning, BehavioralCloning, FittedQ, expert_labels  # noqa: E402
from ope import always_continue, evaluate, gate, voi_rule  # noqa: E402

import autolab.runner as runner  # noqa: E402
from autolab.pomdp import TaskConfig  # noqa: E402
from gate_policy import GatePolicy, VOIRulePolicy, synthetic_pool  # noqa: E402

MODELS, RESULTS = model_based.MODELS, os.path.join(HERE, "results")
ONLINE_SEED0 = 81_000_000


# ------------------------------------------------- policies for the simulator --
def make_policy(name, **kw):
    if name.startswith("learned_"):
        return deploy.LearnedStopPolicy(**kw)
    if name.startswith("model_"):
        return model_based.LearnedModelPolicy(**kw)
    if name == "voi_rule":
        return VOIRulePolicy()
    if name == "gate":
        return GatePolicy(**kw)
    return _make_policy(name, **kw)


_make_policy = runner.make_policy
runner.make_policy = make_policy          # at import time, so worker processes see it too


def online_job(args):
    name, kw, seed = args
    pool = synthetic_pool(np.random.default_rng([seed, 0]), bool(seed % 2))
    r = runner.run_episode(pool, TaskConfig(), name, seed, kw)
    return dict(policy=name, seed=seed, net_utility=r["net_utility"], n_measure=r["n_measure"],
                committed=r["committed"], hit_target=r["hit_target"])


def split(ca: pd.DataFrame, frac: float = 0.7, seed: int = 0) -> set:
    rng = np.random.default_rng(seed)
    train = set()
    for _, g in ca.groupby("behavior"):
        ids = g.campaign.to_numpy()
        train |= set(rng.choice(ids, int(round(frac * len(ids))), replace=False).tolist())
    return train


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--online", type=int, default=1000, help="fresh campaigns per policy for ground truth")
    a = ap.parse_args()
    os.makedirs(MODELS, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)

    tr, me, ca = load()
    train_ids = split(ca)
    tr_tr, tr_te = tr[tr.campaign.isin(train_ids)], tr[~tr.campaign.isin(train_ids)]
    ca_tr, ca_te = ca[ca.campaign.isin(train_ids)], ca[~ca.campaign.isin(train_ids)]
    me_tr = me[me.campaign.isin(train_ids)]
    print(f"train {len(ca_tr)} campaigns / {len(tr_tr)} decisions; held out {len(ca_te)} / {len(tr_te)}")

    # ---- 1. model-free learners
    info = {}
    for name, beh in (("bc_voi", "voi_rule"), ("bc_mcts", "mcts")):
        Phi, y = expert_labels(tr_tr, beh, PLAN[beh][1])
        bc = BehavioralCloning().fit(Phi, y)
        Phi_te, y_te = expert_labels(tr_te, beh, PLAN[beh][1])
        info[name] = dict(train_rows=len(y), heldout_accuracy=float(np.mean(bc.policy(Phi_te) == y_te)),
                          expert_continue_rate=float(y.mean()))
        deploy.save(bc, "bc", os.path.join(MODELS, f"{name}.json"), expert=beh, **info[name])
    for name, learner in (("batch_q", BatchQLearning()), ("fitted_q", FittedQ())):
        q = learner.fit(tr_tr)
        conv = q.td_trace[-1] if name == "batch_q" else q.change[-1]
        info[name] = dict(train_rows=len(tr_tr), last_td_mse_or_change=float(conv))
        deploy.save(q, "q", os.path.join(MODELS, f"{name}.json"), **info[name])
    print("model-free:", json.dumps(info, indent=1))

    # ---- 1b. model-based: estimate and solve every (estimator, noise band)
    jobs = [(me_tr, ca_tr, band, kind, "pbvi") for kind in model_based.KINDS
            for band in range(len(model_based.band_centers()))]
    with mp.Pool(min(len(jobs), os.cpu_count())) as pool:
        models = pd.DataFrame(pool.map(model_based.solve_one, jobs, chunksize=1))
    models.drop(columns="path").to_csv(os.path.join(RESULTS, "models.csv"), index=False)

    # ---- 2a. offline evaluation: importance sampling on held-out campaigns
    theta = json.load(open(GATE_WEIGHTS))["theta"]
    phi_policies = {"ei": always_continue, "voi_rule": voi_rule, "gate": gate(theta)}
    for name in ("bc_voi", "bc_mcts", "batch_q", "fitted_q"):
        phi_policies[f"learned_{name}"] = deploy.LoadedRule(os.path.join(MODELS, f"{name}.json"))
    off = []
    for name, pol in phi_policies.items():
        r = evaluate(tr_te, ca_te, pol)
        off.append(dict(policy=name, is_ordinary=r.ordinary, is_se=r.se_ordinary, is_weighted=r.weighted,
                        is_per_decision=r.per_decision, ess=r.ess, n=r.n, frac_nonzero=r.frac_nonzero))
    off = pd.DataFrame(off)

    # ---- 2b. ground truth: fresh campaigns in the simulator
    specs = {"ei": None, "voi_rule": None, "gate": {"theta": theta}}
    for name in ("bc_voi", "bc_mcts", "batch_q", "fitted_q"):
        specs[f"learned_{name}"] = {"path": os.path.join(MODELS, f"{name}.json")}
    for kind in model_based.KINDS:
        for mode in ("stop_only", "full"):
            specs[f"model_{kind}_{mode}"] = {"kind": kind, "mode": mode}
    jobs = [(name, kw, ONLINE_SEED0 + s) for name, kw in specs.items() for s in range(a.online)]
    with mp.Pool(os.cpu_count()) as pool:
        on = pd.DataFrame(pool.map(online_job, jobs, chunksize=8))
    g = on.groupby("policy")
    truth = pd.DataFrame(dict(true_net=g.net_utility.mean(), true_se=g.net_utility.sem(),
                              experiments=g.n_measure.mean(), hit=g.hit_target.mean(),
                              abandon=1 - g.committed.mean())).reset_index()
    ev = truth.merge(off, on="policy", how="left").sort_values("true_net", ascending=False)
    ev.to_csv(os.path.join(RESULTS, "offline_eval.csv"), index=False)

    fmt = lambda v: f"{v:.3f}"  # noqa: E731
    text = (f"Trained on {len(ca_tr)} logged campaigns; offline estimates on {len(ca_te)} held-out campaigns;\n"
            f"ground truth from {a.online} fresh campaigns per policy (made-up pools, seeds {ONLINE_SEED0}+).\n\n"
            "Model-free learners:\n" + json.dumps(info, indent=1) + "\n\n"
            "Reading models estimated from the log (per noise band):\n"
            + models[["kind", "noise_lo", "noise_hi", "fail_prob", "p_high_good", "p_high_bad", "n_readings",
                      "solve_sec"]].to_string(index=False, float_format=fmt) + "\n\n"
            "Every policy, true net utility vs. its importance-sampling estimate (blank: not estimable from the log):\n"
            + ev[["policy", "true_net", "true_se", "is_ordinary", "is_se", "is_weighted", "is_per_decision", "ess",
                  "experiments", "hit", "abandon"]].to_string(index=False, float_format=fmt, na_rep="") + "\n")
    with open(os.path.join(RESULTS, "train_summary.txt"), "w") as fh:
        fh.write(text)
    print(text)
