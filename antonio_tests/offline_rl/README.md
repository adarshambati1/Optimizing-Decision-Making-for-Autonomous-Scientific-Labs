# Offline RL on the autonomous-lab problem

Offline RL means learning a policy **from a fixed log of past campaigns**, with no
model handed to the learner and no new experiments while it learns.
*Algorithms for Decision Making* calls this setting "batch reinforcement
learning" (ch. 15, footnote 2) but has no chapter on it. The five methods
planned here are the book's methods that work on a fixed dataset:

| # | Method | Book section | Status |
|---|---|---|---|
| 1 | Off-policy evaluation by importance sampling | A.14, 14.2 | done (`ope.py`) |
| 2 | Behavioral cloning | 18.1 | done (`methods.py`) |
| 3 | Batch Q-learning with experience replay | 17.2, 17.6, 17.7, 8.6 | done (`methods.py`) |
| 4 | Maximum-likelihood model, then plan | 16.1 | done (`methods.py`, `model_based.py`) |
| 5 | Bayesian model estimation | 16.4 | done (`methods.py`, `model_based.py`) |

The textbook is free online at <https://algorithmsbook.com/files/dm.pdf>.

## The decision being learned

At every step: **continue** (run the experiment GP-EI picks) or **stop** (commit
to the best candidate, or abandon). This is the same framing as Adarsh's
stopping gate (`rl tests/`), so results compare directly.

## Phase 1: the dataset (done)

`logger.py` runs 3,600 campaigns on **made-up pools** (Adarsh's generator, half
"easy", half "hard"), so the three real datasets stay a held-out test. Each
campaign is driven by a randomized version of one existing agent:

| Behavior | Base rule | Randomization | Campaigns |
|---|---|---|---|
| `ei` | never stops | ε = 0.1 | 400 |
| `voi_rule` | one-step rule | ε = 0.1 | 800 |
| `gate` | Adarsh's `hard_voi` gate | ε = 0.1 | 800 |
| `mcts` | Keira's MCTS stop decision (100 simulations) | ε = 0.1 | 400 |
| `hazard_q` | stop with probability q each step, q ∈ {0.02, 0.05, 0.1, 0.2} | none | 4 × 300 |

With ε = 0.1, each step continues with probability (1 − ε)·p_base + ε/2. Both
actions therefore always have probability ≥ 0.05, which importance sampling needs.
When MCTS continues, the experiment is still GP-EI's pick, so "continue" means the
same thing in every row.

```bash
python antonio_tests/offline_rl/logger.py               # about 30 s on 10 cores
python antonio_tests/offline_rl/tests/test_logger.py    # checks
```

The data goes to `offline_rl/data/` (git-ignored, ~5.5 MB; regenerate with the
command above). The tables are documented at the top of `logger.py`:
- `transitions`: 44k decisions, with features, action, logged probability, reward
  and next features
- `measurements`: 51k experiments, with the true score, meets-spec and near-miss
  labels, and the reading
- `campaigns`: one row per campaign, with its outcome

**The tests check that:**
- per-step rewards add up to the simulator's net utility exactly;
- logged probabilities are consistent;
- with randomization off, the logger reproduces GP-EI and the one-step rule
  exactly, as run by Keira's `run_episode`.

### What the log covers

| Behavior | Net utility | Experiments (10th–90th pct) |
|---|---|---|
| gate | 0.287 ± 0.017 | 12 (4–23) |
| mcts | 0.254 ± 0.023 | 11 (3–20) |
| voi_rule | 0.213 ± 0.016 | 10 (4–19) |
| hazard_0.05 | 0.191 ± 0.026 | 20 (4–46) |
| ei | 0.167 ± 0.022 | 20 (5–46) |
| hazard_0.02 | 0.151 ± 0.027 | 33 (6–50) |
| hazard_0.1 | 0.149 ± 0.025 | 11 (3–22) |
| hazard_0.2 | 0.114 ± 0.023 | 7 (3–13) |

- **Thin coverage of late stopping:** 80% of voluntary stops happen with more than
  60% of the budget left, and only 93 happen in the last 20%. Methods that learn
  the value of stopping late will lean on generalization there.
- **Readings, estimated from the log** (synthetic pools, so not directly comparable
  to any one real dataset): 10% failures; P(high reading | meets spec) = 0.76;
  P(high | near miss) = 0.18. Also, **61% of measured candidates are neither**: they
  score below the near-miss band. The `antonio_tests/` abstraction assumes every
  shortlisted candidate either meets spec or is a near miss, and this is the kind
  of miscalibration method 4 will estimate and correct.

## Phase 2: the five methods (done)

```bash
python antonio_tests/offline_rl/tests/test_methods.py   # toy problems with known answers
python antonio_tests/offline_rl/train.py                # train + evaluate, about 3 min on 10 cores
```

**How each method learns from the log** (training uses 70% of the campaigns, split by
campaign within each behavior; the other 30% are held out for offline evaluation):

- **Behavioral cloning:** logistic regression on a quadratic feature basis, imitating
  the expert's own decisions (the logged probabilities with the ε-mixing undone).
  `bc_voi` copies the one-step rule (99.3% held-out accuracy). `bc_mcts` copies MCTS's
  stop decision (95.8%).
- **Batch Q-learning:** Q(features, continue/stop) is linear in the same quadratic basis,
  learned only from the fixed buffer, two ways:
  - `batch_q`: gradient Q-learning (eq. 17.21) on minibatches replayed from the buffer
  - `fitted_q`: the same Bellman backup, solved by least squares each sweep
  The two agree.
- **Model estimation:** counts of reading outcomes, split by pool noise level:
  - `mle`: maximum-likelihood counts (16.1)
  - `bayes_mean`: a Beta(1, 1) posterior mean (16.4)
  - `bayes_pess`: the same posterior read pessimistically (16.4)
  Each model is solved with PBVI on the `antonio_tests/` abstraction and used either to
  decide stopping only (`stop_only`) or also to pick the experiment (`full`).

**Tests** (`tests/test_methods.py`, on toy problems with known answers):
- importance sampling is unbiased on a short problem;
- behavioral cloning recovers a known expert;
- both Q-learners find the optimal stopping rule;
- the counts and the posterior's pessimism are right;
- the model-based policy runs in Keira's simulator.

### Results on made-up pools (`results/train_summary.txt`)

True net utility comes from 1,000 fresh campaigns per policy in Keira's simulator. The
offline estimate is weighted importance sampling on the held-out log.

| Policy | True net utility | Offline estimate (WIS) | Experiments |
|---|---|---|---|
| Adarsh's gate (trained online, ~22k campaigns) | 0.383 ± 0.016 | 0.483 | 24 |
| **fitted Q** (offline, 2,520 logged campaigns) | **0.378 ± 0.016** | 0.511 | 24 |
| **batch Q-learning** (offline) | **0.370 ± 0.016** | 0.475 | 25 |
| behavioral cloning of MCTS | 0.339 ± 0.015 | 0.288 | 22 |
| behavioral cloning of the one-step rule | 0.308 ± 0.015 | 0.351 | 15 |
| one-step rule | 0.288 ± 0.016 | 0.318 | 14 |
| learned model, Bayes mean (full) | 0.236 ± 0.015 | n/a | 25 |
| learned model, MLE (stop only) | 0.229 ± 0.015 | n/a | 26 |
| learned model, Bayes pessimistic (stop only) | 0.224 ± 0.015 | n/a | 24 |
| GP-EI | 0.138 ± 0.014 | 0.030 | 50 |

- **Batch Q-learning works as offline RL should: it beats the policies that generated
  its data.** Both Q-learners beat the one-step rule by ~0.08–0.09 (z ≈ 4, unpaired) and
  tie Adarsh's gate. The gate learned online from ~22k fresh campaigns; the Q-learners
  had 2,520 logged ones and ran no new experiments.
- **Behavioral cloning matches its expert, as expected.** The MCTS clone scores 0.34,
  above the randomized MCTS that generated its data (0.25 in the log). It is
  deterministic and drops the ε-noise.
- **The learned models are the weakest offline methods** (0.21–0.24): better than
  GP-EI (z ≈ 4.5), worse than the one-step rule (z ≈ −2.7). With ~7k readings per noise
  band, the Bayesian posterior equals the MLE. Reading it pessimistically costs a little.
  Estimating the reading model does not rescue the abstraction, so its other assumptions
  (independent slots, a refilled shortlist) are the more likely problem.
- **Importance sampling ranks policies roughly but does not give their values.**
  Weighted IS gets the top group (gate, fitted Q, batch Q) and the bottom (GP-EI) right,
  but overestimates the top group by ~0.1 and misplaces the MCTS clone (effective sample
  size 11). Ordinary IS underestimates every policy. Campaigns run 10–50 decisions, so a
  deterministic policy matches few logged campaigns exactly (effective sample sizes 11–44
  of 1,080). On a one-decision toy problem the same estimators are accurate (tests), so
  this is the method's known long-horizon weakness, not a bug.

All of this is on made-up pools, the log's own distribution. Phase 3 tests the
three real datasets.

## Phase 3: the real datasets (done)

```bash
python antonio_tests/offline_rl/benchmark.py --seeds 250   # about 10 min on 10 cores
python antonio_tests/offline_rl/plots.py
```

Every learned policy was trained on the made-up-pool log only; none of them saw these
pools. 250 paired seeds per dataset, the same seeds as `keira tests/`, `rl tests/` and
`antonio_tests/`. The GP-EI, MCTS, one-step rule and gate numbers match those folders.
Figures: `results/figures/benchmark.png`, `results/figures/ope.png`.

| Net utility (± ≈ 0.03) | P3HT | Perovskite | Crossed barrel |
|---|---|---|---|
| GP-EI | −0.05 | −0.44 | 0.20 |
| MCTS | 0.28 | −0.08 | 0.29 |
| One-step rule | **0.39** | **−0.06** | 0.30 |
| Adarsh's gate (online RL) | 0.31 | −0.30 | 0.38 |
| Behavioral cloning: one-step rule | **0.38** | **−0.06** | 0.32 |
| Behavioral cloning: MCTS | 0.35 | −0.08 | **0.39** |
| Batch Q-learning (replay) | 0.28 | −0.28 | 0.36 |
| Batch Q-learning (fitted) | 0.28 | −0.29 | 0.34 |
| Learned models (6 variants) | 0.02 – 0.08 | −0.09 – −0.10 | 0.28 – 0.32 |

- **Behavioral cloning of MCTS does better than MCTS itself on Crossed barrel**
  (+0.10, z = 2.4, significant). On P3HT it leans the same way but is not significant
  (+0.07, z = 1.6), and it ties on Perovskite. It is also the best policy
  overall on Crossed barrel (+0.09 over the one-step rule, z = 3.3) while running
  8–16× faster than MCTS. A possible reason, not tested: the clone copies only
  MCTS's *stopping* decision, and pairs it with GP-EI's pick of *which* experiment, a
  better pick than MCTS's own.
- **Behavioral cloning of the one-step rule ties the rule on every dataset**
  (|z| < 1.5). It is a faithful, learned stand-in for it.
- **Batch Q-learning inherits the gate's Perovskite failure.** It tracks Adarsh's gate
  on every dataset (|z| < 1.7), including Perovskite, where both keep experimenting
  (~41 experiments) and score −0.28 to −0.30, against −0.06 for the one-step rule
  (z = −8.6). Both learned from made-up pools that rarely contain hopeless campaigns
  like Perovskite (Adarsh's diagnosis in `rl tests/`), so learning offline instead of
  online does not fix that. Imitation of MCTS or of the one-step rule transfers fine,
  because those experts reason about the actual belief instead of learning when to give up.
- **The learned models still fail on P3HT** (0.02–0.08 vs 0.39, z ≈ −7 to −10). They run
  35–38 experiments, as the `antonio_tests/` solvers did. This settles the open question
  from `antonio_tests/`: **re-estimating the reading model from data does not fix
  P3HT**, so a miscalibrated reading model was not the cause. What remains is the
  abstraction itself: independent slots and a shortlist refilled after every reading.
  On Crossed barrel the models tie the one-step rule. On Perovskite they are slightly
  worse (z ≈ −2.4).
- **Offline estimates from the made-up-pool log do not predict real-dataset scores.**
  Weighted IS rates batch Q-learning 0.48 and the gate 0.48. On Perovskite both score
  −0.28 to −0.30. An offline evaluation is only as good as the match between the
  logged pools and the deployment pools.

**Bottom line:** the cheapest offline method, behavioral cloning, transfers best. It
matches or beats its experts on every real dataset. Offline Q-learning beats its data
on the logged distribution but not under distribution shift, and model-based offline
RL is limited by the abstraction it plans on.
