"""Figures for the offline-solver experiments.

    python antonio_tests/plots.py

  results/figures/benchmark.png  every agent in Keira's simulator (from benchmark.csv)
  results/figures/solvers.png    bounds and policy values on the abstraction (from solver_benchmark.csv)

Same style as keira tests/ and rl tests/; GP-EI, MCTS and the one-step rule keep their colors.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "keira tests", "experiments"))

from common import summarize  # noqa: E402  (sets BLAS threads first)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

RESULTS = os.path.join(HERE, "results")
FIG = os.path.join(RESULTS, "figures")
os.makedirs(FIG, exist_ok=True)

METHODS = ["qmdp", "fib", "fast_lower", "pbvi", "perseus", "sawtooth", "point_selection",
           "sawtooth_search", "triangulated"]
POLICIES = ["ei", "mcts", "voi_rule"] + [f"pomdp_{m}" for m in METHODS]
COLOR = {"ei": "#2a78d6", "mcts": "#eb6834", "voi_rule": "#1baf7a",
         "pomdp_qmdp": "#9ecae9", "pomdp_fib": "#6baed6", "pomdp_sawtooth": "#3182bd",
         "pomdp_triangulated": "#08519c",
         "pomdp_fast_lower": "#fdd0a2", "pomdp_pbvi": "#fdae6b", "pomdp_perseus": "#e6550d",
         "pomdp_point_selection": "#a63603", "pomdp_sawtooth_search": "#7f2704"}
LABEL = {"ei": "GP-EI (baseline)", "mcts": "MCTS (belief)", "voi_rule": "One-step rule",
         "pomdp_qmdp": "QMDP", "pomdp_fib": "Fast informed bound", "pomdp_fast_lower": "Fast lower bound (blind)",
         "pomdp_pbvi": "PBVI", "pomdp_perseus": "Randomized PBVI", "pomdp_sawtooth": "Sawtooth upper bound",
         "pomdp_point_selection": "Point selection + PBVI", "pomdp_sawtooth_search": "Sawtooth heuristic search",
         "pomdp_triangulated": "Triangulated"}
DS_LABEL = {"p3ht": "P3HT/CNT", "perovskite": "Perovskite", "crossed_barrel": "Crossed barrel"}
DATASETS = ["p3ht", "perovskite", "crossed_barrel"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def benchmark_figure():
    df = pd.read_csv(os.path.join(RESULTS, "benchmark.csv"))
    s = summarize(df).set_index(["dataset", "policy"])
    n = int(s["n"].max())
    rows = [("net_utility", "Net utility\n(payoff − costs)"),
            ("hit_target", "P(committed candidate\nmeets spec)"),
            ("n_measure", "Experiments run\n(of 50 allowed)")]
    fig, axes = plt.subplots(3, 3, figsize=(13, 8.5), sharex=True)
    x = np.arange(len(POLICIES))
    for j, ds in enumerate(DATASETS):
        for i, (col, ylab) in enumerate(rows):
            ax = axes[i, j]
            vals = [s.loc[(ds, p), col] for p in POLICIES]
            err = [s.loc[(ds, p), "net_se"] for p in POLICIES] if col == "net_utility" else None
            ax.bar(x, vals, color=[COLOR[p] for p in POLICIES], width=0.75,
                   yerr=err, error_kw=dict(ecolor=MUTED, lw=0.8))
            ax.axhline(0, color=MUTED, lw=0.6)
            if i == 0:
                ax.set_title(DS_LABEL[ds], fontsize=11)
            if j == 0:
                ax.set_ylabel(ylab)
            ax.set_xticks([])
    fig.legend([Patch(color=COLOR[p]) for p in POLICIES], [LABEL[p] for p in POLICIES],
               loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"Offline POMDP solvers in Keira's simulator ({n} seeds; bars on net utility are ±1 standard error)\n"
                 "blues: upper-bound methods · oranges: lower-bound methods", fontsize=10)
    fig.savefig(os.path.join(FIG, "benchmark.png"))
    plt.close(fig)


def solver_figure():
    df = pd.read_csv(os.path.join(RESULTS, "solver_benchmark.csv")).set_index(["dataset", "method"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    x = np.arange(len(METHODS))
    for j, ds in enumerate(DATASETS):
        ax = axes[j]
        d = df.loc[ds].loc[METHODS]
        ax.bar(x - 0.2, d["bound_at_test"], width=0.4, color="#bdbdbd", label="Value function at test beliefs")
        ax.bar(x + 0.2, d["policy_value"], width=0.4, yerr=d["policy_se"], color=[COLOR[f"pomdp_{m}"] for m in METHODS],
               error_kw=dict(ecolor=MUTED, lw=0.8), label="Policy value (rollouts)")
        ax.set_xticks(x)
        ax.set_xticklabels([LABEL[f"pomdp_{m}"] for m in METHODS], rotation=55, ha="right", fontsize=8)
        ax.set_title(DS_LABEL[ds], fontsize=11)
        ax.set_ylim(0.45, 0.72)
        if j == 0:
            ax.set_ylabel("Value on the abstraction")
            ax.legend([Patch(color="#bdbdbd"), Patch(color="#525252")],
                      ["Value function at test beliefs", "Policy value from rollouts (colored by method)"],
                      frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Solvers on the discrete abstraction: what each value function claims (grey) "
                 "vs. what its lookahead policy earns (color)", fontsize=10)
    fig.savefig(os.path.join(FIG, "solvers.png"))
    plt.close(fig)


if __name__ == "__main__":
    solver_figure()
    benchmark_figure()
    print("wrote", FIG)
