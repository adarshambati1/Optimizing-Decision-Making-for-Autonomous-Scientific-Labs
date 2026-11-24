# antonio_tests: decision-making methods from the textbook on the autonomous-lab problem

Antonio's part of the project. It implements and benchmarks methods from *Algorithms for
Decision Making* (Kochenderfer, Wheeler & Wray; free at
<https://algorithmsbook.com/files/dm.pdf>) on Keira's autonomous-lab POMDP, against the
team's existing agents:
- **GP-EI**, the baseline (Keira, `keira tests/`)
- **MCTS** (Keira)
- **the one-step value-of-information rule** (Adarsh, `rl tests/`)
- **the learned stopping gate** (Adarsh)

**The problem in one paragraph.** Each campaign looks for one material that meets spec (top
5% of a pool). The agent runs noisy experiments that sometimes fail or give outlier
readings, keeps a Gaussian-process belief over every candidate's true score, and at each
step either measures another candidate or stops. Stopping means committing to the best
candidate or abandoning the campaign. The score is **net utility**: 1 if the committed
candidate truly meets spec, minus 0.1 to commit, minus 0.01 per experiment. The real
datasets are P3HT/CNT, Perovskite and Crossed barrel.

**How everything here is compared.** Every agent runs in Keira's simulator with her
default task, on the same 250 random seeds per dataset (100 in `validation/`). So every
comparison is **paired**: same pools, same noise, same failures. Differences are reported
with a paired z-score; |z| ≥ 2 is treated as significant.

## Contents

| Folder | Book | What it does | Headline result |
|---|---|---|---|
| [this folder (top level)](#chapter-21-offline-pomdp-solvers) | ch. 21 | 9 offline POMDP solvers (QMDP, fast informed bound, fast lower bound, PBVI, randomized PBVI, sawtooth bound, point selection, sawtooth heuristic search, triangulated value functions) on a 177-state discrete version of the problem | Valid bounds in seconds, but no solver beats the one-step rule, and 8 of 9 fail on P3HT by over-experimenting |
| [`offline_rl/`](offline_rl/README.md) | 16, 17, 18, A.14 | 5 offline RL methods learned from a fixed log of 3,600 campaigns: importance-sampling evaluation, behavioral cloning, batch Q-learning, maximum-likelihood and Bayesian learned models | Cloning MCTS beats MCTS on Crossed barrel (not significantly elsewhere). Batch Q-learning beats its data on made-up pools but inherits the gate's Perovskite failure. Importance sampling is unreliable over 10–50-decision campaigns |
| [`online_planning/`](online_planning/README.md) | ch. 22 | Forward search, sparse sampling, determinized sparse tree search, gap heuristic search | Deeper lookahead than one step barely helps. Forward, determinized and gap search tie or beat MCTS; sparse sampling loses to it on Perovskite |
| [`exploration/`](exploration/README.md) | 15.3–15.4 | Greedy, quantile, UCB1 and Thompson sampling as the way to *pick* the experiment, under three stopping rules | None beats GP-EI as the picker |
| [`validation/`](validation/README.md) | 14.3–14.4 | Robustness (agents in worlds that differ from their model) and trade analysis (experiments vs success, Pareto frontier) | The simple planners are the most robust. Adaptive stopping beats any fixed budget |
| [`tuning/`](tuning/README.md) | 10.4 | The cross-entropy method tuning six MCTS settings, tested on held-out seeds | Tuning does not significantly improve MCTS; its defaults sit on a flat optimum |

Each folder's README has its methods, tests, full result tables with significance, and
caveats.

## What the project shows, across all of it

1. **Deciding *when to stop* is what matters.**
   - Every agent that beats GP-EI does it by stopping early, or by giving up on hopeless
     pools like Perovskite. GP-EI always runs all 50 experiments.
   - *Which* experiment to run is effectively solved by GP-EI. No exploration strategy
     from ch. 15 beats it under any stopping rule ([`exploration/`](exploration/README.md)).

2. **The one-step value-of-information rule is very hard to beat.**
   - Deeper exact or sampled lookahead ties it on P3HT and Perovskite; the largest
     consistent gain is +0.04 ([`online_planning/`](online_planning/README.md)).
   - Offline solvers don't beat it ([ch. 21 below](#chapter-21-offline-pomdp-solvers)),
     learned models don't ([`offline_rl/`](offline_rl/README.md)), and tuned MCTS doesn't
     ([`tuning/`](tuning/README.md)).
   - It is also among the most robust agents when the world differs from its model
     ([`validation/`](validation/README.md)).

3. **Pairing MCTS's stop decision with GP-EI's pick does at least as well as MCTS,
   but this is significant only on Crossed barrel.**
   - The behavioral clone of MCTS's stopping decision beats MCTS on Crossed barrel
     (+0.10, z = 2.4) and leans ahead on P3HT (+0.07, z = 1.6)
     ([`offline_rl/`](offline_rl/README.md)).
   - MCTS stopping with GP-EI picking leans ahead on P3HT and Crossed barrel, but not
     significantly (z = 1.8 and 1.1) ([`exploration/`](exploration/README.md)). The clone
     imitates this same policy, so these two are not independent; they score the same on
     P3HT (0.350).
   - Separately, no ch. 22 planner is significantly worse when GP-EI picks its experiment,
     and 2 of 21 comparisons are significantly better
     ([`online_planning/`](online_planning/README.md)).

   The direction is consistent, but most individual comparisons are within noise.

4. **The right amount of experimenting depends on the pool.**
   - Crossed barrel (600 candidates) rewards experimenting more; P3HT and Perovskite reward
     stopping early.
   - Sparse sampling's 0.41 on Crossed barrel came from running more experiments, not from
     better planning. Determinizing the same search removes both the extra experiments and
     the gain ([`online_planning/`](online_planning/README.md)).
   - Trade analysis measures it directly: planners do best on Crossed barrel treating
     experiments as half price, and on Perovskite as 4× the price
     ([`validation/`](validation/README.md)).
   - Still, adaptive stopping beats every fixed budget, even one tuned per dataset.

5. **Learned stopping rules don't transfer well; planners and imitators do.**
   - Adarsh's gate and batch Q-learning, both trained on made-up pools, fail on Perovskite.
   - Behavioral clones of agents that reason about the belief transfer fine.
   - In robustness tests, telling the learned rules the true noise level makes them
     *worse*, while planners use the information correctly.
   - Learned rules read the belief through features calibrated to their training
     conditions; planners recompute from the model.

6. **Overestimating noise is the safer modeling mistake.** Readings noisier than assumed
   hurt everyone (up to −0.55). Readings more precise than assumed are mostly harmless,
   except on Perovskite, where knowing the precision would have turned a hopeless pool
   winnable ([`validation/`](validation/README.md)).

7. **Two methodological cautions for the write-up.**
   - Importance-sampling estimates from a log can't replace real runs here. Campaigns are
     too long, and the made-up pools differ from the real ones
     ([`offline_rl/`](offline_rl/README.md)).
   - The discrete abstraction behind ch. 21 and the learned models fails on P3HT even when
     its reading model is re-estimated from data, so the cause is its structure, not
     miscalibration.

## Results status: fixes after an external review

An outside review found five issues; all were confirmed and fixed. **The saved results in
the folders below were produced before these fixes. Reruns are pending.**

1. **Shared simulator (Keira's `pomdp.py`):** a reading that triggered a GP
   hyperparameter refit went into the fit before its outlier check. It affected 2–8% of
   refits (5–10% of one-step-rule campaigns, 15–57% of GP-EI campaigns). Now it is
   checked first; regression test in `keira tests/tests/test_core.py`. **Measured
   impact on GP-EI, MCTS and the one-step rule (250 paired seeds): every change within
   noise (|z| ≤ 1.8).** MCTS and the one-step rule move by ≤ 0.007; GP-EI on P3HT moves
   most (+0.05, z = 1.8). Rankings and conclusions are unchanged. The 20%-outlier
   robustness test is the result most likely to shift.
2. **QMDP vs. FIB (ch. 21):** they differed only through tie-breaking. Ties are now
   deterministic, with a test (see the ch. 21 section).
3. **Gap search's upper bound** could undershoot under negative correlations. It is now
   valid for any correlation structure, with a test (`online_planning/`).
4. **`load_data.split_data`** puts replicates of the same recipe in both train and test.
   It was never used downstream; its call in `main.py` is commented out with a note.
5. **README claims** that overstated the tables were corrected: "every planner beats or
   ties MCTS", "cloning MCTS beats MCTS", and finding 3 above.

## Open threads and untested hypotheses

These were raised along the way and not yet run:
- **P3HT abstraction failure:** count re-shortlisted candidates as already tested, to check
  whether refilling the shortlist after every reading causes the over-experimenting
  (ch. 21, `offline_rl/`).
- **Sparse sampling:** its gains mix small-sample optimism with modeling outliers.
  Leaving outliers out of the samples would separate the two (`online_planning/`).
- **Gap heuristic search:** a tighter upper bound, one that charges for the experiments
  needed to learn the scores, would let the gap close (`online_planning/`).
- **Bad pickers make the stopping rule run longer**, probably because they don't sharpen
  the belief about the candidates that matter (`exploration/`).
- **Learned stopping on Perovskite:** training on made-up pools that include hopeless
  campaigns should fix the failure (`offline_rl/`, `rl tests/`).
- **Tuning MCTS:** a budget about 10× larger (several hundred campaigns per sample) would
  be needed to resolve differences of about 0.03 (`tuning/`).
- **Optional methods not built:** actor-critic with MCTS (13.4), a particle-filter belief
  (19.6).
- **Small run-to-run differences in Keira's simulator** (e.g. GP-EI 0.201 vs 0.209 on
  Crossed barrel between runs). Report exact numbers from one run.
- **Terminology:** "offline RL" can mean offline *planning* (ch. 21 here) or learning from
  logged data (`offline_rl/`). Say which.

## Rerunning everything

Requirements: numpy, scipy, pandas and matplotlib, the same as `keira tests/`. The `cs238`
conda environment has them.

Tests (44, all passing, about 45 seconds in total; Keira's `keira tests/tests/test_core.py` has 5 more):

```bash
for t in antonio_tests/tests/test_solvers.py antonio_tests/offline_rl/tests/test_logger.py antonio_tests/offline_rl/tests/test_methods.py antonio_tests/online_planning/tests/test_*.py antonio_tests/exploration/tests/test_pickers.py antonio_tests/validation/tests/test_*.py antonio_tests/tuning/tests/test_cem.py; do python "$t"; done
```

Experiments, from the repo root. Times are on a 10-core laptop.

| Folder | Commands | Time |
|---|---|---|
| ch. 21 (top level) | `bash antonio_tests/run_all.sh` | ~15 min |
| `offline_rl/` | `bash antonio_tests/offline_rl/run_all.sh` | ~15 min |
| `online_planning/` | `python antonio_tests/online_planning/benchmark.py --seeds 250`, then `plots.py` | ~45 min |
| `exploration/` | `python antonio_tests/exploration/benchmark.py --seeds 250`, then `plots.py` | ~15 min |
| `validation/` | `python antonio_tests/validation/robustness.py --seeds 100`, `trade.py --seeds 100`, then `plots.py` | ~35 min |
| `tuning/` | `python antonio_tests/tuning/cem.py --pools real`, `--pools synthetic`, `--test`, then `plots.py` | ~50 min |

Generated data that git ignores, and rebuilds:
- `results/solved/` (`solve_all.py`, ~1 min)
- `offline_rl/data/` (`offline_rl/logger.py`, ~30 s)
- `offline_rl/models/*.pkl` (`offline_rl/train.py`)

Every folder reuses Keira's simulator (`keira tests/autolab/`) and, where needed, Adarsh's
code (`rl tests/gate_policy.py`) without modifying either.

---

## Chapter 21: offline POMDP solvers

*(Code in this folder's top level.)*

Benchmarks the nine offline belief-state planning methods from ch. 21 against GP-EI, MCTS
and the one-step rule, on 250 paired seeds.

| Method | Kind |
|---|---|
| Fully observable value approximation (QMDP) | upper bound |
| Fast informed bound (FIB) | upper bound |
| Fast lower bound (blind lower bound) | lower bound |
| Point-based value iteration (PBVI) | lower bound |
| Randomized point-based value iteration (Perseus) | lower bound |
| Sawtooth upper bound | upper bound |
| Point selection (exploratory belief expansion, then PBVI) | lower bound |
| Sawtooth heuristic search | lower + upper bound (policy from the lower) |
| Triangulated value functions (Freudenthal) | interpolation |

### Why an abstraction

These are offline solvers for **discrete** POMDPs: they need the state space written down
as matrices. Keira's POMDP has a continuous hidden state (the true score of each of up to
600 candidates), so they cannot be run on it directly. `abstraction.py` builds a small
discrete POMDP that keeps the decision that matters (keep measuring, commit, or give up):

- **slots:** 4 shortlisted candidates
- **state:** experiments still affordable h ∈ {0..10} (fully observed) × which slots truly
  meet spec (2⁴), plus a terminal state: **177 states**
- **actions:** measure(i), commit(i), abandon: **9 actions**
- **observations:** fail / reading below spec / reading above spec: **3 observations**
- **rewards:** Keira's costs: −0.01 per experiment, payoff 1 for a candidate that meets
  spec, −0.1 to commit

The reading probabilities come only from what Keira's agent knows (noise, outlier and
failure rates, the spec threshold). A "near miss" slot is modeled as a score between the
75th percentile and the spec.

Each method is **solved once per dataset, offline**. Online, every decision maps the
current GP belief onto the abstraction (top 2 commit candidates + top 2 untested
candidates by P(meets spec), slots treated as independent), and takes the one-step
lookahead action on that method's value function. So the nine methods differ only in how
they approximate the value function.

### Results on the abstraction itself (`results/solver_summary.txt`, `results/figures/solvers.png`)

- **Every solver is correct on a POMDP small enough to solve exactly** (`tests/`): upper
  bounds sit above the exact value, lower bounds below it. PBVI, Perseus, point selection
  and sawtooth search land within ~0.003 of it.
- **QMDP and FIB are identical here.** FIB only improves on QMDP when transitions are
  random, and in this problem they are deterministic. *Their different rows in the
  simulator table below are tie-breaking noise, not a real difference.* The policies'
  top actions were often exactly tied, and plain `argmax` broke the ties by 1e-16
  roundoff (outcomes differed on 148–167 of 250 seeds per dataset, never significantly).
  Ties now break deterministically (`abstraction.best_action`), and a test checks that
  the two play identical campaigns; the table is pending a rerun.
- **The sawtooth bound is a little looser than FIB at the test beliefs:** away from its
  sampled points it falls back on interpolating the corner values, which is looser than FIB.
- **Lower-bound methods make the better policies.** Their lookahead policies earn
  0.58–0.65 on the abstraction, vs 0.51–0.62 for the upper-bound and triangulated methods,
  which overvalue experimenting.
- Offline cost: under 25 s per method per dataset.

### Results in Keira's simulator (`results/benchmark_summary.txt`, `results/figures/benchmark.png`)

Net utility, 250 paired seeds (± ≈ 0.03):

| | P3HT | Perovskite | Crossed barrel |
|---|---|---|---|
| GP-EI (baseline) | −0.05 | −0.44 | 0.20 |
| MCTS | 0.28 | −0.08 | 0.29 |
| One-step rule | **0.39** | −0.06 | 0.30 |
| QMDP | 0.04 | −0.13 | 0.29 |
| Fast informed bound | 0.08 | −0.11 | 0.30 |
| Fast lower bound (blind) | **0.38** | **−0.05** | 0.19 |
| PBVI | 0.04 | −0.06 | 0.28 |
| Randomized PBVI | 0.06 | −0.07 | 0.23 |
| Sawtooth upper bound | 0.02 | −0.16 | 0.27 |
| Point selection + PBVI | 0.06 | −0.06 | 0.28 |
| Sawtooth heuristic search | 0.06 | −0.08 | **0.33** |
| Triangulated | 0.02 | −0.16 | 0.27 |

- **Every method beats GP-EI on P3HT and Perovskite** (z from 1.9 to 22). On Crossed
  barrel, all but the blind lower bound beat it.
- **Perovskite:** the lower-bound methods (blind, PBVI, point selection, Perseus, sawtooth
  search) tie MCTS and the one-step rule. All give up early, after 6–9 experiments. The
  upper-bound methods keep experimenting and do significantly worse (z ≤ −2).
- **Crossed barrel:** most methods tie MCTS and the one-step rule. Sawtooth heuristic
  search scores highest (0.33), but not significantly (z = 0.8 vs MCTS).
- **P3HT is the failure case:** every method except the blind lower bound runs 35–43
  experiments and nets ~0.05, far below MCTS (0.28) and the one-step rule (0.39), at
  z ≈ −5 to −9. The blind lower bound stops after ~12 experiments and ties the one-step
  rule. It is the only lookahead that does not count on future experiments.

**Cause of the P3HT failure.** The original guess was a miscalibrated reading model.
[`offline_rl/`](offline_rl/README.md) later re-estimated the reading model from logged data
and the failure stayed, so the more likely cause is the abstraction's structure: slots
treated as independent, and the shortlist refilled with a fresh contender after every
reading, so the model always sees one more experiment worth running.

**Overall:** solving a discrete abstraction offline is fast and gives valid bounds, but no
method beats the one-step rule, which needs no solver. The lower-bound methods (the ones
pessimistic about future information) transfer best to the real simulator.

### Files

| File | What it is |
|---|---|
| `abstraction.py` | the discrete POMDP and belief mapping |
| `solvers.py` | the nine methods |
| `triangulation.py` | Freudenthal triangulation |
| `policy.py` | solved value function → policy in Keira's simulator |
| `solve_all.py` | offline solves and the abstraction-level benchmark |
| `benchmark.py` | end-to-end benchmark |
| `plots.py` | both figures |
| `run_all.sh` | tests, solves, benchmark and figures in one command |
| `tests/test_solvers.py` | correctness tests |
