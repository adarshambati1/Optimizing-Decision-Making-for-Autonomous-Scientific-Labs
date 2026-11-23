# Offline POMDP solvers on the autonomous-lab problem

Benchmarks the nine offline belief-state planning methods from *Algorithms for
Decision Making*, ch. 21, against GP-EI, MCTS (Keira) and the one-step rule
(Adarsh), in Keira's simulator with her default task and the same 250 seeds
as `rl tests/`.

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

## Why an abstraction

These are offline solvers for **discrete** POMDPs: they need the state space
written down as matrices. Keira's POMDP has a continuous hidden state (the
true score of each of up to 600 candidates), so they cannot be run on it
directly. `abstraction.py` builds a small discrete POMDP that keeps the
decision that matters (keep measuring, commit, or give up):

- **slots:** 4 shortlisted candidates
- **state:** experiments still affordable h ∈ {0..10} (fully observed) × which slots
  truly meet spec (2⁴), plus a terminal state: **177 states**
- **actions:** measure(i), commit(i), abandon: **9 actions**
- **observations:** fail / reading below spec / reading above spec: **3 observations**
- **rewards:** Keira's costs: −0.01 per experiment, payoff 1 for a candidate that meets spec, −0.1 to commit

The reading probabilities come only from what Keira's agent knows (noise, outlier
and failure rates, the spec threshold). A "near miss" slot is modeled as a score
between the 75th percentile and the spec.

Each method is **solved once per dataset, offline**. Online, every decision maps
the current GP belief onto the abstraction (top 2 commit candidates + top 2
untested candidates by P(meets spec), slots treated as independent), and takes
the one-step lookahead action on that method's value function. So the nine methods
differ only in how they approximate the value function.

## Results

### 1. On the abstraction itself (`results/solver_summary.txt`, `figures/solvers.png`)

- **Every solver is correct on a POMDP small enough to solve exactly** (`tests/`):
  upper bounds sit above the exact value, lower bounds below it. PBVI, Perseus,
  point selection and sawtooth search land within ~0.003 of it.
- **QMDP and FIB are identical here.** FIB only improves on QMDP when transitions
  are random, and in this problem they are deterministic.
- **The sawtooth bound is a little looser than FIB at the test beliefs:** away
  from its sampled points it falls back on interpolating the corner values, which
  is looser than FIB.
- **Lower-bound methods make the better policies.** Their lookahead policies earn
  0.58–0.65 on the abstraction, vs 0.51–0.62 for the upper-bound and
  triangulated methods, which overvalue experimenting.
- Offline cost: under 25 s per method per dataset on 10 cores.

### 2. In Keira's simulator (`results/benchmark_summary.txt`, `figures/benchmark.png`)

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

- **Every method beats GP-EI on P3HT and Perovskite** (z from 1.9 to 22). On
  Crossed barrel, all but the blind lower bound beat it.
- **Perovskite:** the lower-bound methods (blind, PBVI, point selection, Perseus,
  sawtooth search) tie MCTS and the one-step rule. All give up early, after 6–9
  experiments. The upper-bound methods keep experimenting and do significantly
  worse (z ≤ −2).
- **Crossed barrel:** most methods tie MCTS and the one-step rule. Sawtooth heuristic
  search scores highest (0.33), but not significantly (z = 0.8 vs MCTS).
- **P3HT is the failure case:** every method except the blind lower bound runs
  35–43 experiments and nets ~0.05, far below MCTS (0.28) and the one-step rule
  (0.39), at z ≈ −5 to −9. The blind lower bound stops after ~12 experiments and
  ties the one-step rule. It is the only lookahead that does not count on future
  experiments.

**Likely cause of the P3HT failure (a hypothesis, not tested):** the abstraction
overvalues experimenting. It treats slots as independent and readings as very
informative (P3HT readings separate meets-spec from near-miss 89% vs 10%). After
each reading the shortlist is refilled with a fresh untested contender, so the
model always sees one more experiment worth running. Tests worth running:
calibrating the near-miss band (`contender_q`), counting re-shortlisted candidates
as already tested, or a shorter horizon.

**Overall:** solving a discrete abstraction offline is fast and gives valid
bounds, but no method beats the one-step rule, which needs no solver. The lower-bound
methods (the ones that are pessimistic about future information) transfer best to
the real simulator.

## Run it

```bash
bash antonio_tests/run_all.sh
```

That runs the tests, `solve_all.py` (offline solves, ~1 min, writes
`results/solved/`, which git ignores), `benchmark.py --seeds 250` (~10 min on
10 cores) and `plots.py`. Needs the same packages as `keira tests/`.

GP-EI and MCTS were rerun here, so the comparison is paired. Their numbers match
`rl tests/` closely but not exactly (e.g. GP-EI on crossed barrel 0.201 vs
0.209), which suggests a little run-to-run nondeterminism in Keira's simulator
(GP fits or BLAS threading).

| File | What it is |
|---|---|
| `abstraction.py` | the discrete POMDP and belief mapping |
| `solvers.py` | the nine methods |
| `triangulation.py` | Freudenthal triangulation |
| `policy.py` | solved value function → policy in Keira's simulator |
| `solve_all.py` | offline solves and the abstraction-level benchmark |
| `benchmark.py` | end-to-end benchmark |
| `plots.py` | both figures |
| `tests/test_solvers.py` | correctness tests |
