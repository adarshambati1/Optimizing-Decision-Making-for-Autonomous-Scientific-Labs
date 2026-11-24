# Policy validation (ADM ch. 14)

| Method | Book section | Status |
|---|---|---|
| Robustness analysis | 14.3 | done (`robustness.py`) |
| Trade analysis | 14.4 | done (`trade.py`) |

```bash
python antonio_tests/validation/tests/test_robustness.py
python antonio_tests/validation/tests/test_trade.py
python antonio_tests/validation/robustness.py --seeds 100    # about 25 min on 10 cores
python antonio_tests/validation/trade.py --seeds 100         # about 10 min on 10 cores
python antonio_tests/validation/plots.py
```

## Robustness analysis (14.3)

Section 14.3: keep each policy's planning model fixed, and evaluate it in worlds that
deviate from that model. Its Figure 14.4 plots performance as one true parameter moves
away from the assumed value. Here every agent assumes Keira's defaults: noise = 1× the
replicate noise, 5% outliers, 10% failures. The simulated world changes one thing at a
time:

- **Reading noise** 0.5×, 2× or 4×. Each level also runs with the agent **told** the true
  noise, which separates "the world got harder" from "the agent's model is wrong".
- **Outliers** 0% or 20%.
- **Failures** 0% or 30%.
- **Initial random experiments** 1 or 10 instead of 3. This is a protocol change applied to
  agent and world alike, the "start values" item from the future-work list.

Agents:
- GP-EI, MCTS, and the one-step rule
- Adarsh's gate (`hard_voi`)
- depth-2 forward search and depth-3 determinized search, both with GP-EI picking
- the behavioral clone of MCTS (`offline_rl/`)

That is 13 worlds × 7 agents × 3 datasets × **100 paired seeds**, so about 27,000
campaigns. The runner is Keira's `run_episode` with the agent's model built separately
from the world's. **The tests check that with no mismatch it reproduces `run_episode`
exactly**, and that the agent really keeps its assumed noise while the world's readings
scatter 4× more.

With 100 seeds (0–99) instead of 250, the default-world numbers differ a little from other
folders (for example, the one-step rule scores 0.31 vs 0.39 on P3HT). **Compare within this
grid**, where every world uses the same seeds.

Figures:
- `results/figures/robustness_noise.png`: net utility vs. the true noise, as in Fig. 14.4.
  Solid lines: the agent assumes 1×. Dashed: the agent is told the truth.
- `results/figures/robustness_other.png`: the change from the default world for the other
  mismatches.

Full tables: `results/robustness_summary.txt`. Paired changes with standard errors:
`results/robustness_changes.csv`.

### What happens

**1. Noisier readings than assumed hurt everyone, and only the planners can use the truth.**
- At 4× the noise, every agent loses 0.06–0.29 on P3HT and 0.33–0.55 on Crossed barrel,
  where every agent goes negative.
- **Telling the agent the true noise helps the agents that plan from their model:**
  - on P3HT: the one-step rule +0.14 (z = 2.7), forward search +0.14 (z = 2.6),
    GP-EI +0.33 (z = 5.7), MCTS +0.11 (z = 1.7)
  - smaller gains on the other two datasets
- **It does not help the learned stopping rules, and often hurts them.** Adarsh's gate
  loses 0.25 on Perovskite (z = −6.8) and 0.16 on Crossed barrel (z = −2.7). The MCTS clone
  loses 0.12 on Crossed barrel (z = −2.2). Told the truth, the gate keeps experimenting
  (32–48 experiments instead of 11–18). Learned rules read the belief through features
  calibrated to the noise levels they were trained on, so they cannot adjust to a new noise
  regime even with correct information. Planners recompute the value of an experiment from
  the model, so they adjust automatically.

**2. Readings more precise than assumed are mostly harmless.**
- On P3HT and Crossed barrel, assuming 1× when the truth is 0.5× costs nothing significant
  (|z| ≤ 2.0, with MCTS the one exception, +0.12 for knowing on Crossed barrel).
- **Perovskite is the exception.** There, knowing the readings are twice as precise is
  worth +0.07 to +0.60 (z up to 9.7). It turns a hopeless pool into a winnable one, and
  the gate in particular goes from −0.40 to +0.20.
- **For the write-up:** overestimating the noise is the safer mistake.

**3. More outliers than assumed cost 0.05–0.22 on P3HT and Crossed barrel.** GP-EI suffers
most (−0.22, z = −4.0). MCTS and the one-step rule suffer least, and their losses are not
significant (−0.05 to −0.09, |z| ≤ 1.9). Perovskite is unaffected, because agents give up there anyway. Fewer outliers than
assumed helps by up to 0.11.

**4. Failed experiments barely matter.** At 30% failures instead of 10%, nothing changes
significantly (|z| ≤ 1.7, at most −0.12). With no failures, MCTS gains 0.10–0.12 (z ≈ 2)
and the gate gains 0.13 on P3HT (z = 2.3).

**5. The starting design barely matters.** With 1 or 10 random initial experiments instead
of 3, nearly every change is insignificant. The exceptions all involve MCTS:
- MCTS gains 0.15 on P3HT with a single initial experiment (z = 2.1).
- MCTS and its clone lose 0.06–0.09 on Perovskite with 10 (z = −2.9). The extra random
  experiments are pure cost on a pool where the right move is to give up.

**6. Which agents stay near the best, across all 39 worlds:**

| Agent | Median gap to the best agent | Worst gap | Best in |
|---|---|---|---|
| One-step rule | 0.016 | 0.16 | 9 of 39 |
| Forward search d2 (GP-EI picks) | 0.019 | 0.15 | 12 of 39 |
| MCTS | 0.027 | 0.16 | 9 of 39 |
| Determinized d3 (GP-EI picks) | 0.043 | 0.20 | 2 of 39 |
| Behavioral clone of MCTS | 0.059 | 0.20 | 1 of 39 |
| Adarsh's gate | 0.086 | 0.41 | 6 of 39 |
| GP-EI | 0.364 | 0.61 | 0 of 39 |

The simple planners (the one-step rule and depth-2 forward search) are the most robust:
never more than 0.16 behind the best agent in any world. The learned policies have the
worst worst cases, because they do not adapt when the world moves away from their
training conditions.

## Trade analysis (14.4)

Section 14.4: vary one parameter of each policy at a time to trace a trade-off curve
between two competing metrics, then keep the Pareto-optimal policies, the ones no other
policy beats on both. The two metrics:
- **experiments run** (cost)
- **P(committed candidate meets spec)** (success; giving up counts as a miss)

One knob per agent family:
- **GP-EI:** a fixed budget of 2, 5, 10, 20, 30 or 47 experiments after the initial 3
  (47 is GP-EI as deployed).
- **One-step rule, depth-2 forward search (GP-EI picks), MCTS:** a *planning-cost
  multiplier* λ ∈ {0.25, 0.5, 1, 2, 4}. The planner believes each experiment costs λ times
  its real cost. Its budget is rescaled so it can still afford exactly the experiments the
  real budget allows. Real costs are unchanged.
- **Adarsh's gate:** the threshold t on P(continue), from 0.2 to 0.8 (0.5 as deployed).

That is 26 settings × 3 datasets × 100 paired seeds. **The tests check that the default
settings reproduce each agent exactly**, that the knobs move experiment counts the right
way, that the rescaled budget leaves the affordable experiment count unchanged, and that
the Pareto filter is right.

Figure: `results/figures/trade.png` (one curve per family; large dots mark each agent as
deployed; dashed: the Pareto frontier). Points: `results/trade_points.csv`.

### What happens

**1. The adaptive planners own the frontier.** Most Pareto-optimal points come from MCTS,
the one-step rule and forward search (23 of 32 across the three datasets). Fixed-budget
GP-EI reaches the frontier only at the extremes: its cheapest setting, or, on Crossed
barrel, all 50 experiments.

**2. Adaptive stopping beats even a well-chosen fixed budget.**
- **No single fixed budget works everywhere.** The best per dataset is 10 experiments
  (P3HT), 30 (Crossed barrel) and 5 (Perovskite).
- **The best single fixed budget across all three**, 10, totals 0.35 net utility summed
  over the datasets. That is against 0.47 for the one-step rule and 0.56 for depth-2
  forward search as deployed.
- **Even tuned per dataset, a fixed budget loses to forward search on P3HT** (−0.10,
  z = −2.4) and ties it elsewhere (|z| ≤ 1.4). Deciding when to stop campaign by campaign
  is worth more than any fixed rule of thumb.

**3. The deployed planning cost is not the best one for every dataset, and the direction
matches what every earlier folder found:**
- **Crossed barrel** rewards planning as if experiments cost *half* as much (λ = 0.5).
  The gains are +0.06 to +0.08 for all three planners (z = 1.1–1.9 each). MCTS at λ = 0.5
  scores 0.38, the best of any setting there.
- **Perovskite** rewards planning as if experiments cost *4×* as much, i.e. giving up
  sooner. The gains are +0.04 to +0.05 for all three planners (z = 1.6–1.9 each).
- **P3HT** is best near λ = 1 (forward search at λ = 1 is the best setting there, 0.35).

Each of these is individually short of significance, but the three planners agree on the
direction on both datasets. Tuning one knob per dataset could add about 0.05. It needs to
know in advance which kind of pool you are in, which a real lab would not.

**4. Adarsh's gate is too eager to continue.** Raising its threshold from 0.5 to 0.8 helps
on P3HT (+0.09, z = 2.3) and Perovskite (+0.10, z = 2.5). Even then, it still loses 0.20
on Perovskite, where it runs about 38 experiments against about 10 for the planners. The
threshold does not fix the gate's Perovskite failure.

**5. A caveat about the metric.** On Perovskite, the gate sits on the Pareto frontier
(the highest hit rate, 0.22) while having the worst net utility. Hit rate ignores the
cost of committing and the value of giving up, so on a pool where giving up is right, the
hit-rate frontier is not the right target. Net utility remains the score that matters.
