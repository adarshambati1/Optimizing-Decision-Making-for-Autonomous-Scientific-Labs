"""Figure for the exploration-strategy experiments: results/figures/benchmark.png

    python antonio_tests/exploration/plots.py

Grouped by stopping rule (never stop / one-step rule / MCTS), one bar per
picker, so each picker's effect can be read within a group.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "keira tests", "experiments"))

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

STOPS = [("never", "Never stop\n(full budget)"), ("voi", "One-step rule\nstops"), ("mcts", "MCTS\nstops")]
PICKERS = ["ei", "greedy", "quantile", "ucb1", "thompson", "mcts"]
COLOR = {"ei": "#2a78d6", "greedy": "#9e9e9e", "quantile": "#1baf7a", "ucb1": "#eda100",
         "thompson": "#c2185b", "mcts": "#eb6834"}
LABEL = {"ei": "GP-EI", "greedy": "Greedy (highest mean)", "quantile": "Quantile (95%)",
         "ucb1": "UCB1 (Bayesian counts)", "thompson": "Thompson sampling", "mcts": "MCTS's own choice"}
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
    fig, axes = plt.subplots(3, 3, figsize=(14, 8.5), sharex=True)
    width = 0.13
    for j, ds in enumerate(DATASETS):
        for i, (col, ylab) in enumerate(rows):
            ax = axes[i, j]
            for g, (stop, _) in enumerate(STOPS):
                for b, pk in enumerate(PICKERS):
                    pol = f"{stop}_{pk}"
                    if (ds, pol) not in s.index:
                        continue
                    x = g + (b - 2.5) * width
                    err = s.loc[(ds, pol), "net_se"] if col == "net_utility" else None
                    ax.bar(x, s.loc[(ds, pol), col], width=width * 0.92, color=COLOR[pk],
                           yerr=err, error_kw=dict(ecolor=MUTED, lw=0.7))
            ax.axhline(0, color=MUTED, lw=0.6)
            ax.set_xticks(range(len(STOPS)))
            ax.set_xticklabels([lab for _, lab in STOPS])
            if i == 0:
                ax.set_title(DS_LABEL[ds], fontsize=11)
            if j == 0:
                ax.set_ylabel(ylab)
    fig.legend([Patch(color=COLOR[p]) for p in PICKERS], [LABEL[p] for p in PICKERS],
               loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f"Which experiment to run: exploration strategies (ADM 15.4) under three stopping rules "
                 f"({n} paired seeds; ±1 SE)", fontsize=10)
    fig.savefig(os.path.join(FIG, "benchmark.png"))
    plt.close(fig)


if __name__ == "__main__":
    benchmark_figure()
    print("wrote", FIG)
