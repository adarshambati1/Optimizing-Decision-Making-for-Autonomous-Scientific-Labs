"""Results figure for the LLM comparison: results/figures/results.png

    python "llm tests/plots.py"
    python "llm tests/plots.py" --csv "llm tests/results/<another run>.csv"

Same layout and style as keira tests/experiments/plots.py, with the LLM agent
added. GP-EI and MCTS keep the colors they have in Keira's figure.
"""
import argparse
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "keira tests", "experiments"))

from common import summarize  # noqa: E402  (sets BLAS threads first)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

RESULTS = os.path.join(HERE, "results")
FIG = os.path.join(RESULTS, "figures")
os.makedirs(FIG, exist_ok=True)

POLICIES = ["ei", "mcts", "llm"]
COLOR = {"ei": "#2a78d6", "mcts": "#eb6834", "llm": "#6250d6"}
LABEL = {"ei": "GP-EI\n(baseline)", "mcts": "MCTS\n(belief)", "llm": "LLM\n(Opus 5.5)"}
DS_LABEL = {"p3ht": "P3HT/CNT", "perovskite": "Perovskite", "crossed_barrel": "Crossed barrel"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.bbox": "tight",
})

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="benchmark csv (default: the only real run in results/)")
    a = ap.parse_args()
    path = a.csv or sorted(p for p in glob.glob(os.path.join(RESULTS, "llm_benchmark_*.csv"))
                           if "mock" not in p)[0]
    s = summarize(pd.read_csv(path))
    pols = [p for p in POLICIES if p in set(s.policy)]
    dss = [d for d in DS_LABEL if d in set(s.dataset)]
    n = int(s.n.min())
    metrics = [("net_utility", "Net utility\n(payoff − costs)", "net_se"),
               ("hit_target", "P(committed candidate\nmeets spec)", None),
               ("n_measure", "Experiments run\n(of 50 allowed)", None)]
    fig, axes = plt.subplots(len(metrics), len(dss), figsize=(2.6 * len(dss), 2.1 * len(metrics)),
                             squeeze=False)
    for j, ds in enumerate(dss):
        sd = s[s.dataset == ds].set_index("policy")
        for i, (m, ylabel, se) in enumerate(metrics):
            ax = axes[i, j]
            ax.bar(range(len(pols)), [sd.loc[p, m] for p in pols],
                   yerr=[sd.loc[p, se] for p in pols] if se else None,
                   color=[COLOR[p] for p in pols], width=0.6, edgecolor="white",
                   error_kw=dict(ecolor=MUTED, lw=0.8))
            ax.axhline(0, color=MUTED, lw=0.6)
            ax.set_xticks(range(len(pols)))
            ax.set_xticklabels([LABEL[p] for p in pols] if i == len(metrics) - 1 else [])
            ax.grid(axis="x", visible=False)
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == 0:
                ax.set_title(DS_LABEL[ds], color=INK)
    fig.suptitle(f"LLM agent vs. planner vs. baseline ({n} seeds only; bars on net utility are ±1 standard error)",
                 color=INK, fontsize=10, y=0.995)
    fig.tight_layout()
    fig.savefig(f"{FIG}/results.png")
    print("wrote", f"{FIG}/results.png")
