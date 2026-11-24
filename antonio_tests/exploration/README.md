# Which experiment to run: exploration strategies (ADM 15.4)

Every agent in the project makes two decisions each step: **whether** to keep
experimenting, and **which** candidate to measure. Everything so far has tested the
first. Most agents just use GP-EI for the second. This folder tests the second: it swaps
GP-EI for the book's directed-exploration strategies, adapted from bandits (pull arms
repeatedly) to this problem (measure each candidate at most once, under a GP belief):

| Picker | Rule | Book |
|---|---|---|
| GP-EI | expected improvement over the best tested mean | Keira's baseline |
| Greedy | highest posterior mean (no exploration) | 15.3 |
| Quantile | highest 95th percentile, μ + 1.645σ | 15.4, alg. 15.6 |
| UCB1 | μ + c·√(log N / N(k)), with the book's Bayesian count adaptation N(k) = σ²_noise / σ_k², so the bonus is c·(σ_k/σ_noise)·√(log N); c = 1, not tuned | 15.4, eq. 15.3, alg. 15.7 |
| Thompson sampling | one joint draw of all scores from the GP posterior; measure the highest | 15.4, alg. 15.8 |

Each picker runs under three stopping rules:
- **never stop** (the full budget, like GP-EI)
- **the one-step rule**
- **MCTS's stop decision**, with the picker choosing the experiment whenever MCTS continues

Never-stop + GP-EI is GP-EI, one-step rule + GP-EI is the one-step rule, and MCTS + its own
choice is MCTS. **The tests check that these three reproduce those agents exactly**, and
the benchmark's numbers for them match every other folder.

```bash
python antonio_tests/exploration/tests/test_pickers.py
python antonio_tests/exploration/benchmark.py --seeds 250     # about 15 min on 10 cores
python antonio_tests/exploration/plots.py
```

## Results (250 paired seeds, `results/benchmark_summary.txt`, `results/figures/benchmark.png`)

Net utility (± ≈ 0.03):

| Stopping rule → | Never stop | | | One-step rule | | | MCTS stops | | |
|---|---|---|---|---|---|---|---|---|---|
| Picker ↓ / dataset → | P3HT | Perov. | C. barrel | P3HT | Perov. | C. barrel | P3HT | Perov. | C. barrel |
| **GP-EI** | **−0.05** | −0.44 | **0.20** | **0.39** | −0.06 | **0.30** | **0.35** | **−0.07** | **0.34** |
| Greedy | −0.24 | **−0.38** | 0.08 | 0.01 | −0.17 | 0.28 | 0.02 | −0.16 | 0.30 |
| Quantile (95%) | −0.11 | −0.45 | 0.15 | 0.32 | **−0.05** | 0.26 | 0.34 | −0.08 | 0.28 |
| UCB1 | −0.11 | −0.43 | 0.12 | 0.23 | −0.07 | 0.29 | 0.07 | −0.09 | 0.27 |
| Thompson | −0.06 | −0.46 | 0.09 | 0.27 | −0.05 | 0.25 | 0.24 | −0.09 | 0.21 |
| MCTS's own choice | | | | | | | 0.28 | −0.08 | 0.29 |

- **GP-EI is the best picker.** Under every stopping rule, on every dataset, no
  exploration strategy beats it significantly. The one exception is greedy under
  never-stop on Perovskite (+0.06, z = 2.2), a setting where every agent loses badly
  anyway. Elsewhere they tie it at best or lose by up to 0.38, many significantly
  (e.g. Thompson under the one-step rule on P3HT: −0.11, z = −2.8; UCB1 under MCTS on
  P3HT: −0.28, z = −6.1).
- **Quantile exploration is the only one that keeps up** (within 0.07 of GP-EI
  everywhere, |z| ≤ 1.9). It is the closest in spirit: both are optimistic about
  uncertain candidates near the top.
- **A worse picker also makes the stopping rule run longer.** Greedy, UCB1 and Thompson
  lead the one-step rule and MCTS to run 20–32 experiments on P3HT, against 14–15 with
  GP-EI. A plausible reason, not tested: those pickers do not sharpen the belief about
  the candidates that matter for committing, so the value of one more experiment stays
  high and the stopping rule keeps going. The picker affects the score twice: through
  what each experiment finds, and through when the campaign stops.
- **Greedy is by far the worst on P3HT** (−0.18 to −0.38). Measuring the highest mean
  never tests the uncertain candidates that might meet spec.
- **MCTS's stop decision with GP-EI picking leans ahead of MCTS's own choice**, but not
  significantly (P3HT +0.07, z = 1.8; Crossed barrel +0.05, z = 1.1). It points the same
  way as two related results: the behavioral clone of MCTS's stop decision
  (`offline_rl/`; significant on Crossed barrel only), which imitates this same policy
  and scores the same on P3HT (0.350 and 0.350), and the chapter 22 planners, which are
  never significantly worse with GP-EI picking (`online_planning/`).

**Bottom line:** for this problem, which experiment to run is a solved part. GP-EI is
hard to beat, and the gains in this project all come from deciding when to stop.
