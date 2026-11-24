# Tuning MCTS with the cross-entropy method (ADM 10.4)

This answers the obvious objection to the project's results, "MCTS just wasn't tuned",
and the future-work item "play with the parameters of MCTS".

```bash
python antonio_tests/tuning/tests/test_cem.py
python antonio_tests/tuning/cem.py --pools real          # about 20 min on 10 cores
python antonio_tests/tuning/cem.py --pools synthetic     # about 20 min
python antonio_tests/tuning/cem.py --test                # held-out test, about 10 min
python antonio_tests/tuning/plots.py
```

**The method (algorithm 10.4):** keep a Gaussian over policy parameters. Each iteration
draws 16 parameter vectors, scores each by simulation, and refits the Gaussian (mean and
full covariance) to the 4 best. Six iterations.

**What is tuned:** Keira's MCTS with its compute **fixed at 200 simulations per decision**,
so tuning cannot win just by searching more. Six settings are searched in log space:

| Setting | Default | Range | Tuned on real pools | Tuned on made-up pools |
|---|---|---|---|---|
| `c_ucb` (exploration bonus in the tree) | 0.5 | 0.05–5 | 0.27 | 0.83 |
| `kappa` (UCB width for the shortlist) | 2 | 0.5–4 | 2.13 | 2.15 |
| `n_measure` (candidates the tree branches on) | 6 | 2–12 | 8 | 6 |
| `max_depth` | 6 | 2–10 | 4 | 6 |
| `min_visits` | 5 | 1–20 | 4 | 7 |
| planning-cost multiplier λ (`validation/trade.py`) | 1 | 0.25–4 | 1.14 | 1.01 |

**Two tunings:**
- **Real pools:** P3HT, Perovskite and Crossed barrel, with **training seeds 10,000+**
  only, never the benchmark's 0–249.
- **Made-up pools:** Adarsh's generator only.

Every sample in an iteration plays the same 60 campaigns (common random numbers), and the
campaigns are redrawn each iteration so the search cannot overfit a fixed set.

**Then held out:** default and both tuned settings on the three real datasets, seeds
0–249, paired with every other benchmark in the repo.

**Tests (`tests/test_cem.py`):**
- the cross-entropy loop finds the optimum of a toy objective;
- the parameter mapping round-trips and clips correctly;
- the default settings reproduce Keira's MCTS exactly;
- the training campaigns never touch benchmark seeds.

## Results

Figures: `results/figures/cem_progress.png` (the search) and `results/figures/cem_test.png`
(the held-out test).

| Held-out net utility (250 seeds, ± ≈ 0.03) | P3HT | Perovskite | Crossed barrel |
|---|---|---|---|
| MCTS, default settings | 0.28 | −0.08 | 0.29 |
| MCTS, tuned on real pools | 0.26 (−0.02, z = −0.5) | −0.06 (+0.01, z = 0.9) | 0.27 (−0.03, z = −0.7) |
| MCTS, tuned on made-up pools | **0.35** (+0.07, z = 1.7) | −0.09 (−0.02, z = −1.3) | **0.34** (+0.05, z = 1.2) |
| *for reference: one-step rule* | *0.39* | *−0.06* | *0.30* |
| *for reference: forward search d2, GP-EI picks* | *0.40* | *−0.06* | *0.34* |

- **Tuning does not significantly improve MCTS.** Neither tuned version beats the default on
  any dataset (|z| ≤ 1.7). The made-up-pool tuning moves in the right direction on P3HT and
  Crossed barrel (+0.05 to +0.07) and slightly the wrong way on Perovskite. The real-pool
  tuning is a wash.
- **The search was limited by noise, not by the method.** With 60 campaigns per sample, a
  score's standard error is 0.05–0.07, about as large as the spread between the 16
  samples (s.d. ≈ 0.05). The search mean's score wanders within its error bars from one iteration to the
  next (0.20–0.27 on real pools), and each iteration's "best" sample scores lower once the
  next iteration redraws campaigns: a winner's curse. Telling apart settings that differ by
  about 0.03 would take several hundred campaigns per sample, roughly 10× this budget.
- **The settings barely moved.** Every setting ended within a factor of 2 of the default,
  most within 1.5. The exploration bonus `c_ucb` moved most, and in opposite directions in
  the two tunings (0.27 vs 0.83). The planning-cost multiplier stayed near 1 (1.01–1.14), consistent with
  trade analysis: Crossed barrel wants λ ≈ 0.5 and Perovskite wants λ ≈ 4, so one setting
  shared across datasets averages back to about 1. The defaults look like they sit on a
  broad, flat optimum.
- **For the write-up:** at a fixed 200 simulations, retuning MCTS's settings does not close
  its gap to the one-step rule on P3HT (0.35 at best vs 0.39) or Perovskite. The project's
  conclusion that simple value-of-information stopping is as good as or better than MCTS
  holds up against a tuned MCTS too, within the precision this tuning budget allows.
