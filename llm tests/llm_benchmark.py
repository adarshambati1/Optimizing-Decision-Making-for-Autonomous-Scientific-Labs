"""LLM agent vs. MCTS vs. the GP-EI baseline, on the same seeds.

    python "llm tests/llm_benchmark.py" --mock --seeds 3      # no API key, no cost: tests the pipeline
    python "llm tests/llm_benchmark.py" --seeds 5             # real run (needs ANTHROPIC_API_KEY)
    python "llm tests/llm_benchmark.py" --seeds 5 --names real   # LLM also sees real feature names

Requires: numpy scipy pandas anthropic.

All three agents play Keira's POMDP (keira tests/autolab) through her
run_episode, with her default TaskConfig. Only the decision rule differs.
Every LLM decision and its stated reason is logged to results/llm_decisions.jsonl.

A real run makes one API call per decision: up to 47 per campaign after the
3 shared initial measurements, so seeds x datasets x ~10-47 calls in total.
"""
import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "keira tests", "experiments"))
sys.path.insert(0, HERE)

import pandas as pd  # noqa: E402
from common import DATASETS, TaskConfig, paired_difference, run_grid, summarize  # noqa: E402

import autolab.runner as runner  # noqa: E402
from autolab.datasets import load_pool  # noqa: E402
from llm_policy import DEFAULT_MODEL, LLMPolicy, MockClient  # noqa: E402

RESULTS = os.path.join(HERE, "results")

# Keira's run_episode builds policies by name; teach it the extra name without editing her code.
_make_policy = runner.make_policy
_made = threading.local()


def make_policy(name, **kw):
    if name in ("llm", "llm_mock"):
        _made.policy = LLMPolicy(**kw)
        return _made.policy
    return _make_policy(name, **kw)


runner.make_policy = make_policy


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--names", default="anonymized", choices=["anonymized", "real"],
                    help="anonymized hides feature names and the material from the LLM")
    ap.add_argument("--mock", action="store_true", help="fake LLM: tests the pipeline with no API calls")
    ap.add_argument("--n_sims", type=int, default=200, help="MCTS simulations per decision")
    ap.add_argument("--threads", type=int, default=4, help="concurrent LLM campaigns")
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    cfg = TaskConfig()
    llm_name = "llm_mock" if a.mock else "llm"
    tag = "mock" if a.mock else f"{a.model}_{a.names}"

    # 1. GP-EI and MCTS (Keira's parallel runner)
    jobs = [(ds, cfg, p, s, {"n_sims": a.n_sims} if p == "mcts" else None)
            for ds in a.datasets for p in ("ei", "mcts") for s in range(a.seeds)]
    df = run_grid(jobs)

    # 2. the LLM agent, same seeds
    if a.mock:
        client = MockClient()
    else:
        import anthropic
        client = anthropic.Anthropic()

    log_path = os.path.join(RESULTS, f"llm_decisions_{tag}.jsonl")
    log_lock = threading.Lock()
    open(log_path, "w").close()

    def log(row):
        with log_lock, open(log_path, "a") as fh:
            fh.write(json.dumps(row) + "\n")

    pools = {ds: load_pool(ds) for ds in a.datasets}

    def one(job):
        ds, seed = job
        pool = pools[ds]
        r = runner.run_episode(pool, cfg, llm_name, seed, policy_kwargs=dict(
            model=a.model, effort=a.effort, names=a.names, feature_names=pool.feature_names,
            dataset=ds, client=client, log=log, log_meta={"dataset": ds, "seed": seed}))
        p = _made.policy
        r.update(llm_calls=p.n_calls, llm_invalid=p.n_invalid,
                 in_tokens=p.in_tokens, out_tokens=p.out_tokens)
        print(f"  {ds} seed {seed}: net {r['net_utility']:+.2f} after {r['n_measure']} experiments", flush=True)
        return r

    with ThreadPoolExecutor(a.threads) as ex:
        llm_rows = list(ex.map(one, [(ds, s) for ds in a.datasets for s in range(a.seeds)]))
    llm = pd.DataFrame(llm_rows)
    df = pd.concat([df, llm], ignore_index=True)
    df.to_csv(os.path.join(RESULTS, f"llm_benchmark_{tag}.csv"), index=False)

    fmt = lambda v: f"{v:.3f}"  # noqa: E731
    cols = ["dataset", "policy", "net_utility", "net_se", "hit_target", "abandon", "n_measure", "n"]
    text = (f"LLM: {'MOCK (not a real LLM)' if a.mock else a.model}, effort {a.effort}, "
            f"feature names {a.names}, {a.seeds} seeds\n\n"
            + summarize(df)[cols].to_string(index=False, float_format=fmt)
            + f"\n\nPaired difference in net utility, {llm_name} - MCTS (same seeds):\n"
            + paired_difference(df, llm_name, "mcts").to_string(index=False, float_format=fmt)
            + f"\n\nPaired difference in net utility, {llm_name} - GP-EI (same seeds):\n"
            + paired_difference(df, llm_name, "ei").to_string(index=False, float_format=fmt)
            + f"\n\nLLM calls: {int(llm.llm_calls.sum())}, invalid or refused: {int(llm.llm_invalid.sum())}, "
              f"tokens in/out: {int(llm.in_tokens.sum())}/{int(llm.out_tokens.sum())}\n")
    with open(os.path.join(RESULTS, f"llm_summary_{tag}.txt"), "w") as fh:
        fh.write(text)
    print("\n" + text)
