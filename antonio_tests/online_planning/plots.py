"""Figure for the online-planning experiments: results/figures/benchmark.png

    python antonio_tests/online_planning/plots.py

Plots every policy in results/benchmark.csv, baselines first. GP-EI, MCTS and
the one-step rule keep the colors they have everywhere else in the repo.
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

ORDER = ["ei", "mcts", "voi_rule", "fs1", "fs2", "fs3", "fs2_ei", "fs3_ei",
         "ss2", "ss3", "ss2_ei", "ss3_ei", "ss2_m3",
         "ds2", "ds3", "ds2_ei", "ds3_ei", "ds3_p100",
         "gs3", "gs3_ei", "gs6"]
COLOR = {"ei": "#2a78d6", "mcts": "#eb6834", "voi_rule": "#1baf7a",
         "fs1": "#c6dbef", "fs2": "#6baed6", "fs3": "#08519c", "fs2_ei": "#bcbddc", "fs3_ei": "#54278f",
         "ss2": "#e6ab02", "ss3": "#a6761d", "ss2_ei": "#f2d479", "ss3_ei": "#5c3f0c", "ss2_m3": "#fbeec1",
         "ds2": "#fcc5c0", "ds3": "#dd3497", "ds2_ei": "#fde0dd", "ds3_ei": "#7a0177", "ds3_p100": "#f768a1",
         "gs3": "#9e9e9e", "gs3_ei": "#d4d4d4", "gs6": "#424242"}
LABEL = {"ei": "GP-EI (baseline)", "mcts": "MCTS", "voi_rule": "One-step rule",
         "fs1": "Forward search, depth 1", "fs2": "Forward search, depth 2", "fs3": "Forward search, depth 3",
         "fs2_ei": "Depth 2, GP-EI picks", "fs3_ei": "Depth 3, GP-EI picks",
         "ss2": "Sparse sampling, depth 2 (m=10)", "ss3": "Sparse sampling, depth 3 (m=5)",
         "ss2_ei": "Sparse depth 2, GP-EI picks", "ss3_ei": "Sparse depth 3, GP-EI picks",
         "ss2_m3": "Sparse sampling, depth 2 (m=3)",
         "ds2": "Determinized, depth 2", "ds3": "Determinized, depth 3",
         "ds2_ei": "Determinized depth 2, GP-EI picks", "ds3_ei": "Determinized depth 3, GP-EI picks",
         "ds3_p100": "Determinized depth 3, 100 scenarios",
         "gs3": "Gap heuristic search, depth 3", "gs3_ei": "Gap search depth 3, GP-EI picks",
         "gs6": "Gap heuristic search, depth 6"}
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
    pols = [p for p in ORDER if p in set(df.policy)] + sorted(set(df.policy) - set(ORDER))
    n = int(s["n"].max())
    rows = [("net_utility", "Net utility\n(payoff − costs)"),
            ("hit_target", "P(committed candidate\nmeets spec)"),
            ("n_measure", "Experiments run\n(of 50 allowed)"),
            ("sec", "Seconds per campaign\n(log scale)")]
    fig, axes = plt.subplots(len(rows), 3, figsize=(16, 10), sharex=True)
    x = np.arange(len(pols))
    for j, ds in enumerate(DATASETS):
        for i, (col, ylab) in enumerate(rows):
            ax = axes[i, j]
            vals = [s.loc[(ds, p), col] for p in pols]
            err = [s.loc[(ds, p), "net_se"] for p in pols] if col == "net_utility" else None
            ax.bar(x, vals, color=[COLOR.get(p, "#999999") for p in pols], width=0.75,
                   yerr=err, error_kw=dict(ecolor=MUTED, lw=0.8))
            if col == "sec":
                ax.set_yscale("log")
            else:
                ax.axhline(0, color=MUTED, lw=0.6)
            if i == 0:
                ax.set_title(DS_LABEL[ds], fontsize=11)
            if j == 0:
                ax.set_ylabel(ylab)
            ax.set_xticks([])
    fig.legend([Patch(color=COLOR.get(p, "#999999")) for p in pols], [LABEL.get(p, p) for p in pols],
               loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(f"Online belief-state planning (ADM ch. 22) on the real datasets "
                 f"({n} paired seeds; bars on net utility are ±1 standard error)", fontsize=10)
    fig.savefig(os.path.join(FIG, "benchmark.png"))
    plt.close(fig)


if __name__ == "__main__":
    benchmark_figure()
    print("wrote", FIG)
