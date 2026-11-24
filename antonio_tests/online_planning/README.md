# Online belief-state planning (ADM ch. 22)

Planners that search from the current GP belief at every decision, on Keira's
simulator. All four of the chapter's search methods beyond MCTS are here:

| Method | Book section | Status |
|---|---|---|
| Forward search, depth 1–3 | 22.2 | done (`forward_search.py`) |
| Sparse sampling | 22.4 | done (`sparse_sampling.py`) |
| Determinized sparse tree search | 22.6 | done (`despot.py`) |
| Gap heuristic search | 22.7 | done (`gap_search.py`) |

```bash
python antonio_tests/online_planning/tests/test_forward_search.py
python antonio_tests/online_planning/tests/test_sparse_sampling.py
python antonio_tests/online_planning/tests/test_despot.py
python antonio_tests/online_planning/tests/test_gap_search.py
python antonio_tests/online_planning/benchmark.py --seeds 250     # about 10 min on 10 cores
python antonio_tests/online_planning/plots.py
```

## Forward search (22.2)

Equation 22.1, on the GP belief:

- **Leaf value:** the exact value of stopping now, U₀(b) = V_stop(b).
- **Actions at each node:** stop, or measure one of the 5 untested candidates with the
  highest upper confidence bound.
- **Observations:** a failed experiment (no information), or a reading. The GP update is
  exact and linear in the reading, so the expectation over readings uses Gauss-Hermite
  quadrature: 15 nodes at the last level, 5 above it.
- **Outliers** are left out of the lookahead, as in the one-step rule.
- **Working set:** search runs on an exact marginal of the GP over about 40–48
  candidates, as MCTS does.

**Depth 1 is exactly Adarsh's one-step rule.** "Measure k if Q₁(b, k) > V_stop(b)" is
the same as "VOI(k) > cost". The tests check that depth 1, with GP-EI picking the
experiment, reproduces his `VOIRulePolicy` campaign for campaign. So depth 2 and 3
answer the open question: **does looking further ahead than one step help?**

Two ways to use the search:
- `fs1`, `fs2`, `fs3`: the search also picks the experiment
- `fs2_ei`, `fs3_ei`: the search decides only whether to continue, and GP-EI picks

### Results (250 paired seeds, `results/benchmark_summary.txt`)

| Net utility (± ≈ 0.03) | P3HT | Perovskite | Crossed barrel | Seconds / campaign |
|---|---|---|---|---|
| GP-EI | −0.05 | −0.44 | 0.20 | 0.1–0.5 |
| MCTS | 0.28 | −0.08 | 0.29 | 0.3–1.2 |
| One-step rule | 0.39 | −0.06 | 0.30 | 0.01–0.13 |
| Forward search, depth 1 | 0.38 | −0.06 | 0.29 | 0.01–0.10 |
| Forward search, depth 2 | 0.38 | −0.06 | 0.31 | 0.04–0.21 |
| Forward search, depth 3 | 0.38 | **−0.04** | 0.30 | 1.0–2.9 |
| Depth 2, GP-EI picks | **0.40** | −0.06 | **0.34** | 0.06–0.24 |
| Depth 3, GP-EI picks | 0.39 | −0.06 | 0.33 | 1.4–3.3 |

- **Looking further ahead barely helps.** Every depth-2 and depth-3 variant lands within
  0.04 of the one-step rule. The only significant gain is depth 2 with GP-EI picking, on
  Crossed barrel (+0.042, z = 2.4). Depth 3 never beats depth 2 (|z| ≤ 1.3) and costs
  10–30× more time. On P3HT it is slightly worse with GP-EI picking (−0.017, z = −2.5),
  probably from its coarser inner quadrature. **So the one-step rule is not losing much
  by being myopic.** In this problem one good reading is usually enough to decide.
- **Every forward-search variant beats MCTS on P3HT** (+0.09 to +0.12, z = 2.2–3.1) and
  ties or beats it elsewhere (depth 3 on Perovskite: +0.034, z = 2.0). That fits the
  one-step rule's earlier lead over MCTS. Exact expectations over readings seem to
  serve this problem better than MCTS's sampled tree.
- **Letting GP-EI pick the experiment is never significantly worse** and tends to be
  better (+0.03 at depth 2, not significant). This lines up with the offline-RL result
  that a clone of MCTS's *stopping* decision, paired with GP-EI's pick, did better than
  MCTS on Crossed barrel.

## Sparse sampling (22.4)

Equation 22.2: forward search with the sum over observations replaced by an average
over m sampled ones. Sampling only needs to *simulate* an experiment, so each sample
uses the agent's full model, the same one MCTS simulates: failures, outlier readings,
and the agent's outlier filter. Everything else (actions, working set, leaf value,
pickers) is shared with forward search. The tests check that with outliers off and
many samples it converges to forward search's exact values.

Variants:
- `ss2`: depth 2, m = 10 samples per action
- `ss3`: depth 3, m = 5
- `ss2_ei`, `ss3_ei`: the same, with GP-EI picking the experiment
- `ss2_m3`: depth 2 with only m = 3, to expose small-sample effects

### Results (250 paired seeds, same seeds as above)

| Net utility (± ≈ 0.03) | P3HT | Perovskite | Crossed barrel | Experiments (P3HT / Perov. / C.b.) |
|---|---|---|---|---|
| One-step rule | 0.39 | −0.06 | 0.30 | 14 / 10 / 17 |
| Forward search, depth 2, GP-EI picks | **0.40** | **−0.06** | 0.34 | 14 / 10 / 19 |
| Sparse, depth 2 (m = 10) | 0.34 | −0.12 | 0.31 | 20 / 13 / 26 |
| Sparse, depth 3 (m = 5) | 0.25 | −0.12 | 0.33 | 27 / 14 / 29 |
| Sparse, depth 2, GP-EI picks | 0.34 | −0.10 | 0.38 | 20 / 15 / 26 |
| Sparse, depth 3, GP-EI picks | 0.30 | −0.14 | **0.41** | 24 / 17 / 29 |
| Sparse, depth 2 (m = 3) | 0.24 | −0.11 | 0.38 | 25 / 14 / 29 |

- **Sparse sampling runs more experiments than forward search on every dataset**
  (20–29 vs 14–19). This is the known optimism of sampled lookahead. Taking the max over
  noisy sampled estimates of "measure" while the value of stopping is exact makes
  continuing look better than it is. Fewer samples make it worse: m = 3 runs the most
  experiments on P3HT and loses 0.10 there against m = 10 (z = −2.4).
- **That extra experimenting helps on Crossed barrel and hurts elsewhere.** Sparse depth
  3 with GP-EI picking scores **0.41 on Crossed barrel, the best of any planner here**
  (vs the one-step rule +0.11, z = 3.6; vs MCTS +0.12, z = 2.8). Only one agent in the
  repo does better there: Adarsh's easy-pools gate with the value-of-information input
  (0.43, `rl tests/`), which also runs many experiments on this dataset (28). The same planners
  lose 0.04–0.14 to the one-step rule on P3HT and Perovskite (z = −2 to −4).
- **This doesn't make sparse sampling the better planner. Crossed barrel rewards more
  experiments.** It has the largest pool (600 candidates), and across every agent there
  more experiments buy a higher hit rate. GP-EI, with all 50, hits spec 80% of the time.
  Sparse sampling's bias happens to push that way. On the two smaller pools, stopping
  early is right, and the bias costs it.
- Two effects are mixed together: the small-sample optimism, and the fact that sparse
  sampling models outliers while forward search does not. A version that leaves outliers
  out of the samples would separate them; that is not run yet.

## Determinized sparse tree search (22.6)

The book's method, from DESPOT (Ye et al., 2017):
- **m scenarios, fixed up front.** Each has a sampled hidden state (here, an exact
  draw of the true scores from the GP posterior) and a row of random numbers that
  decides every outcome it will see: fail?, outlier?, noise.
- **A belief node is the set of scenarios that reach it.**
- **Every action is evaluated on the same scenarios** (common random numbers), and
  the tree has O(|A|^d·m) nodes.

Two adaptations, both documented in `despot.py`:
1. **Readings are grouped.** One "no usable reading" child, plus 3 bins cut at
   predictive quantiles. Readings are continuous, so without grouping every node would
   hold one scenario with a fixed future, and the search would choose experiments
   already knowing their results.
2. **Nodes with fewer than 4 scenarios are leaves.** This is a simple version of the
   regularization the book notes DESPOT uses.

On a fixed belief, the estimated gap between two actions' values is **4× less noisy**
than sparse sampling's (s.d. 0.020 vs 0.086 over 60 re-plans, `tests/test_despot.py`
checks the direction). With many scenarios and fine bins it approaches forward
search's exact values.

Variants:
- `ds2`, `ds3`: depth 2 / 3, 30 scenarios
- `ds2_ei`, `ds3_ei`: the same, with GP-EI picking the experiment
- `ds3_p100`: depth 3 with 100 scenarios

### Results (250 paired seeds)

| Net utility (± ≈ 0.03) | P3HT | Perovskite | Crossed barrel | Experiments | Seconds / campaign |
|---|---|---|---|---|---|
| One-step rule | 0.39 | −0.06 | 0.30 | 10–17 | 0.01–0.13 |
| Forward search, depth 3, GP-EI picks | 0.39 | −0.06 | 0.33 | 10–20 | 1.4–3.3 |
| Sparse, depth 3, GP-EI picks | 0.30 | −0.14 | **0.41** | 17–29 | 2.8–4.9 |
| Determinized, depth 2 | 0.29 | **−0.05** | 0.25 | 7–18 | 0.05–0.27 |
| Determinized, depth 3 | 0.28 | −0.07 | 0.30 | 8–22 | 0.2–0.8 |
| Determinized, depth 2, GP-EI picks | 0.36 | −0.05 | 0.27 | 8–17 | 0.07–0.25 |
| Determinized, depth 3, GP-EI picks | 0.35 | −0.06 | 0.32 | 9–21 | 0.3–0.8 |
| Determinized, depth 3, 100 scenarios | 0.33 | −0.06 | 0.31 | 8–20 | 0.5–1.8 |

- **Determinization removes sparse sampling's over-experimenting.** Experiment counts
  fall back to forward search's range (7–22 instead of 17–29). That fixes Perovskite:
  determinized beats sparse sampling there by 0.05–0.08 (z = 3.2–5.6). It also gives
  up sparse sampling's lucky win on Crossed barrel (−0.09 with GP-EI picking, z = −3.3).
  More evidence that the Crossed barrel win came from experimenting more, not from
  better planning.
- **With GP-EI picking, it ties the one-step rule and exact forward search everywhere**
  (|z| ≤ 1.7), at about **4–5× less time than depth-3 forward search** (0.3–0.8 s vs
  1.4–3.3 s per campaign). Its value here is cheap, steady deep lookahead, not a higher score.
- **Letting it pick the experiment itself costs ~0.1 on P3HT** (z ≈ −2.6 vs the one-step
  rule), as with the other planners.
- **100 scenarios instead of 30** helps a little on P3HT (+0.06, z = 1.5), with no
  change elsewhere.

## Gap heuristic search (22.7)

Algorithm 22.4 on the GP belief:
- **Repeated searches from the root** (30 per decision). Each follows the action with
  the best upper-bound lookahead, then the observation whose successor has the largest
  gap between the bounds. On the way back it replaces both bounds at every visited node
  by their lookahead backups.
- **The real action is the best under the lower bound**, so the planner only
  experiments when it can show that experimenting beats stopping.

Bounds on a belief's value:
- **Lower:** V_stop(b), since stopping is always available.
- **Upper:** (V − C_commit) × a bound on P(some candidate meets spec). Even an agent that
  saw every true score for free could not earn more than V − C_commit times that chance.
  - By Slepian's inequality, 1 − ∏ₖ(1 − pₖ) bounds the chance when no two scores are
    negatively correlated, with pₖ = P(candidate k meets spec).
  - GP posteriors can have negative correlations, and then the product can undershoot.
    With two coin-flip candidates at correlation −0.9, a real policy earned 0.68 against
    a product bound of 0.675.
  - **So the bound now applies the product only within groups of candidates with no
    negatively correlated pair, and adds the groups up (union bound).** That is valid for
    any correlation structure, and identical to the product when nothing is negatively
    correlated.
  - On real posteriors it is rarely looser: median ratio 1.0, at most 24% looser.
  - The tests cover the clairvoyant value on 24 sampled posteriors and on the
    negative-correlation counterexample.
  - *The gap-search numbers below were produced with the earlier product-only bound;
    rerunning with the valid bound is pending. The new bound also makes the search 2–3×
    slower (0.9–1.6 s per campaign).*

Observations are the same as forward search (fail, or 5 quadrature readings).
**With zero searches, the decision is exactly depth-1 forward search**, which the tests
check. So everything beyond the one-step rule comes from searching.

### Results (250 paired seeds)

| Net utility (± ≈ 0.03) | P3HT | Perovskite | Crossed barrel | Seconds / campaign |
|---|---|---|---|---|
| One-step rule | 0.39 | −0.06 | 0.30 | 0.01–0.13 |
| Forward search, depth 3 | 0.38 | −0.04 | 0.30 | 1.0–2.9 |
| Gap search, depth 3 | 0.32 | −0.05 | 0.31 | 0.2–0.6 |
| Gap search, depth 3, GP-EI picks | 0.38 | **−0.05** | 0.32 | 0.2–0.7 |
| Gap search, depth 6 (60 searches) | 0.31 | −0.06 | 0.25 | 1.1–2.9 |

- **The depth-3 variants tie the one-step rule and depth-3 forward search** (|z| ≤ 1.6,
  everywhere but one cell). The one exception is a small, significant gain on Perovskite with
  GP-EI picking: +0.015 over the one-step rule (z = 2.7) and over depth-3 forward search
  (z = 3.1).
- **The gap does not close.** On the two beliefs inspected (one P3HT, one Crossed barrel),
  the root gap shrinks from 0.55 and 0.37 to 0.37 and 0.33, then stalls, so the gap
  threshold never ends a search early and every decision uses all 30 searches. The upper bound
  assumes free, perfect information, which is far from what a few noisy experiments
  can deliver, so it stays loose and the searches mostly lift the lower bound along
  the most optimistic path. The book notes that the bounds drive this method's
  performance, and a tighter upper bound (e.g. one that charges for the experiments
  needed to learn the scores) is the obvious improvement.
- **Deeper is not better:** depth 6 with 60 searches is no better than depth 3
  (Crossed barrel −0.06, z = −1.5) and 4–5× slower.

## What chapter 22 shows for this project

1. **Looking further ahead than one step does not pay here.** Forward search, sparse
   sampling, determinized search and gap search all tie the one-step rule on P3HT and
   Perovskite to within a few hundredths. The largest consistent lookahead gain is
   +0.04 (depth-2 forward search on Crossed barrel, z = 2.4).
2. **The one big result, sparse sampling's 0.41 on Crossed barrel, comes from
   experimenting more, not from planning better.** Determinizing the same search
   removes both the extra experiments and the gain, and the extra experiments cost
   it on the two smaller pools. How much to experiment depends on the dataset.
3. **Letting GP-EI pick the experiment is never significantly worse for any planner.**
   Across 21 paired comparisons (7 planners × 3 datasets), letting the planner pick its
   own experiment instead never wins significantly (lowest z = −1.0). GP-EI picking is
   significantly better in only 2 of them (sparse depth 3 on Crossed barrel, z = 2.0;
   determinized depth 2 on P3HT, z = 2.0); the rest are within noise. The planners'
   value comes mainly from the stop-or-continue decision.
4. **Forward search, determinized search and gap search tie or beat MCTS; sparse
   sampling does not.** Forward search beats MCTS on P3HT (+0.09 to +0.12, z = 2.2–3.1)
   and ties or beats it elsewhere. **Four sparse-sampling variants lose to MCTS on
   Perovskite** (−0.04 to −0.06, z = −2.4 to −3.7), because their over-experimenting
   costs the most where giving up is right. Exact or determinized expectations over
   readings serve this problem better than MCTS's sampled tree; sampled ones do not.
5. **Cost:** the one-step rule is the cheapest. It is 1.6–5× cheaper than depth-2 forward
   search and up to ~230× cheaper than depth-3 sparse sampling (Perovskite).
