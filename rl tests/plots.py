"""Results figure for the stopping-rule experiments: results/figures/results.png

    python "rl tests/plots.py"

Same layout and style as keira tests/experiments/plots.py, with all seven
agents from gate_benchmark.csv. GP-EI and MCTS keep the colors they have in
Keira's figure.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "keira tests", "experiments"))

from common import summarize  # noqa: E402  (sets BLAS threads first)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

RESULTS = os.path.join(HERE, "results")
FIG = os.path.join(RESULTS, "figures")
os.makedirs(FIG, exist_ok=True)

POLICIES = ["ei", "mcts", "voi_rule", "gate_easy", "gate_easy_voi", "gate_hard", "gate_hard_voi"]
COLOR = {"ei": "#2a78d6", "mcts": "#eb6834", "voi_rule": "#1baf7a", "gate_easy": "#eda100",
         "gate_easy_voi": "#e87ba4", "gate_hard": "#008300", "gate_hard_voi": "#6250d6"}
LABEL = {"ei": "GP-EI (baseline)", "mcts": "MCTS (belief)", "voi_rule": "One-step rule (no learning)",
         "gate_easy": "Learned gate, easy pools", "gate_easy_voi": "Learned gate, easy pools + one-step input",
         "gate_hard": "Learned gate, hard pools", "gate_hard_voi": "Learned gate, hard pools + one-step input"}
DS_LABEL = {"p3ht": "P3HT/CNT", "perovskite": "Perovskite", "crossed_barrel": "Crossed barrel"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.bbox": "tight",
})

if __name__ == "__main__":
    s = summarize(pd.read_csv(os.path.join(RESULTS, "gate_benchmark.csv")))
    pols = [p for p in POLICIES if p in set(s.policy)]
    dss = [d for d in DS_LABEL if d in set(s.dataset)]
    n = int(s.n.min())
    metrics = [("net_utility", "Net utility\n(payoff − costs)", "net_se"),
               ("hit_target", "P(committed candidate\nmeets spec)", None),
               ("n_measure", "Experiments run\n(of 50 allowed)", None)]
    fig, axes = plt.subplots(len(metrics), len(dss), figsize=(3.3 * len(dss), 2.1 * len(metrics)),
                             squeeze=False)
    for j, ds in enumerate(dss):
        sd = s[s.dataset == ds].set_index("policy")
        for i, (m, ylabel, se) in enumerate(metrics):
            ax = axes[i, j]
            ax.bar(range(len(pols)), [sd.loc[p, m] for p in pols],
                   yerr=[sd.loc[p, se] for p in pols] if se else None,
                   color=[COLOR[p] for p in pols], width=0.8, edgecolor="white", linewidth=1.2,
                   error_kw=dict(ecolor=MUTED, lw=0.8))
            ax.axhline(0, color=MUTED, lw=0.6)
            ax.set_xticks([])
            ax.grid(axis="x", visible=False)
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == 0:
                ax.set_title(DS_LABEL[ds], color=INK)
    fig.legend(handles=[Patch(facecolor=COLOR[p], label=LABEL[p]) for p in pols], loc="lower center",
               ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.09), labelcolor=INK, columnspacing=1.6,
               handlelength=1.2)
    fig.suptitle(f"Stopping rules on the three real pools ({n} seeds; bars on net utility are ±1 standard error)",
                 color=INK, fontsize=10, y=0.995)
    fig.tight_layout()
    fig.savefig(f"{FIG}/results.png")
    print("wrote", f"{FIG}/results.png")
