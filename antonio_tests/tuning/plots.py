"""Figures for the CEM tuning of MCTS.

    python antonio_tests/tuning/plots.py

  results/figures/cem_progress.png   each iteration's samples (grey), elites (color) and
                                     the search mean's score (line); and how the search
                                     mean's settings move, for both tunings
  results/figures/cem_test.png       default vs. tuned MCTS on held-out seeds 0-249
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIG = os.path.join(RESULTS, "figures")
os.makedirs(FIG, exist_ok=True)

KINDS = {"real": ("Tuned on real pools (training seeds)", "#c2185b"),
         "synthetic": ("Tuned on made-up pools", "#6250d6")}
PARAMS = ["c_ucb", "kappa", "n_measure", "max_depth", "min_visits", "lam"]
DEFAULT = {"c_ucb": 0.5, "kappa": 2.0, "n_measure": 6, "max_depth": 6, "min_visits": 5, "lam": 1.0}
DS_LABEL = {"p3ht": "P3HT/CNT", "perovskite": "Perovskite", "crossed_barrel": "Crossed barrel"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def progress_figure():
    kinds = [k for k in KINDS if os.path.exists(os.path.join(RESULTS, f"cem_history_{k}.csv"))]
    fig, axes = plt.subplots(2, len(kinds), figsize=(6.5 * len(kinds), 7), squeeze=False)
    for j, k in enumerate(kinds):
        h = pd.read_csv(os.path.join(RESULTS, f"cem_history_{k}.csv"))
        lab, col = KINDS[k]
        ax = axes[0, j]
        jit = np.random.default_rng(0).uniform(-0.15, 0.15, len(h))
        ax.scatter(h.iteration + jit, h.score, s=10, color=np.where(h.elite, col, "#bdbdbd"), zorder=2)
        mean = h[h.is_mean]
        ax.errorbar(mean.iteration, mean.score, yerr=mean.se, color=INK, lw=1.2, marker="o", ms=4, capsize=2,
                    label="search mean's settings (±1 SE)")
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Net utility on that iteration's campaigns")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        ax = axes[1, j]
        for p in PARAMS:
            v = mean.set_index("iteration")[p] / DEFAULT[p]
            ax.plot(v.index, v.values, marker="o", ms=3, label=p)
        ax.axhline(1, color=MUTED, lw=0.8, ls=":")
        ax.set_yscale("log", base=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Search mean / default (log scale)")
        ax.legend(frameon=False, fontsize=8, ncol=3)
    fig.suptitle("Cross-entropy method tuning MCTS (ADM 10.4): 16 samples per iteration, 4 elites (colored)",
                 fontsize=10)
    fig.savefig(os.path.join(FIG, "cem_progress.png"))
    plt.close(fig)


def test_figure():
    p = os.path.join(RESULTS, "cem_test.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    pols = [x for x in ["mcts_default", "mcts_cem_real", "mcts_cem_synthetic"] if x in set(df.policy)]
    col = {"mcts_default": "#eb6834", "mcts_cem_real": KINDS["real"][1], "mcts_cem_synthetic": KINDS["synthetic"][1]}
    lab = {"mcts_default": "MCTS, default settings", "mcts_cem_real": KINDS["real"][0],
           "mcts_cem_synthetic": KINDS["synthetic"][0]}
    g = df.groupby(["dataset", "policy"]).net_utility
    m, se = g.mean(), g.sem()
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for i, ds in enumerate(["p3ht", "perovskite", "crossed_barrel"]):
        for j, p in enumerate(pols):
            ax.bar(i + (j - 1) * 0.26, m[(ds, p)], 0.24, yerr=se[(ds, p)], color=col[p], label=lab[p] if i == 0 else None,
                   error_kw=dict(ecolor=MUTED, lw=0.8))
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xticks(range(3))
    ax.set_xticklabels([DS_LABEL[d] for d in ["p3ht", "perovskite", "crossed_barrel"]])
    ax.set_ylabel("Net utility (held-out seeds 0–249)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("MCTS before and after CEM tuning, on the benchmark's seeds (never used in tuning)", fontsize=9)
    fig.savefig(os.path.join(FIG, "cem_test.png"))
    plt.close(fig)


if __name__ == "__main__":
    progress_figure()
    test_figure()
    print("wrote", FIG)
