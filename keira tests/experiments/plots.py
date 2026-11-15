"""Benchmark figure: results/figures/benchmark.png

    python experiments/plots.py
"""
import os

from common import RESULTS, summarize  # noqa: E402  (sets BLAS threads first)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

FIG = os.path.join(RESULTS, "figures")
os.makedirs(FIG, exist_ok=True)

POLICIES = ["ei", "mcts"]
COLOR = {"ei": "#2a78d6", "mcts": "#eb6834"}
LABEL = {"ei": "GP-EI (baseline)", "mcts": "MCTS (belief)"}
DS_LABEL = {"p3ht": "P3HT/CNT", "perovskite": "Perovskite", "crossed_barrel": "Crossed barrel"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.bbox": "tight",
})

if __name__ == "__main__":
    s = summarize(pd.read_csv(os.path.join(RESULTS, "benchmark.csv")))
    dss = [d for d in DS_LABEL if d in set(s.dataset)]
    metrics = [("net_utility", "Net utility\n(payoff − costs)", "net_se"),
               ("hit_target", "P(committed candidate\nmeets spec)", None),
               ("spend", "Experiment spend", None)]
    fig, axes = plt.subplots(len(metrics), len(dss), figsize=(2.6 * len(dss), 2.1 * len(metrics)),
                             squeeze=False)
    for j, ds in enumerate(dss):
        sd = s[s.dataset == ds].set_index("policy")
        for i, (m, ylabel, se) in enumerate(metrics):
            ax = axes[i, j]
            ax.bar(range(len(POLICIES)), [sd.loc[p, m] for p in POLICIES],
                   yerr=[sd.loc[p, se] for p in POLICIES] if se else None,
                   color=[COLOR[p] for p in POLICIES], width=0.6, edgecolor="white",
                   error_kw=dict(ecolor=MUTED, lw=0.8))
            ax.axhline(0, color=MUTED, lw=0.6)
            ax.set_xticks(range(len(POLICIES)))
            ax.set_xticklabels([LABEL[p] for p in POLICIES] if i == len(metrics) - 1 else [])
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == 0:
                ax.set_title(DS_LABEL[ds], color=INK)
    fig.savefig(f"{FIG}/benchmark.png")
    print("wrote", f"{FIG}/benchmark.png")
