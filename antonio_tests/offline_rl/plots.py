"""Figures for the offline RL experiments.

    python antonio_tests/offline_rl/plots.py

  results/figures/benchmark.png  every policy on the three real datasets (benchmark.csv)
  results/figures/ope.png        offline (importance-sampling) estimates vs. true values

Same style as keira tests/, rl tests/ and antonio_tests/. GP-EI, MCTS, the
one-step rule and Adarsh's gate keep their colors.
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

POLICIES = ["ei", "mcts", "voi_rule", "gate",
            "learned_bc_voi", "learned_bc_mcts", "learned_batch_q", "learned_fitted_q",
            "model_mle_stop_only", "model_bayes_mean_stop_only", "model_bayes_pess_stop_only",
            "model_mle_full", "model_bayes_mean_full", "model_bayes_pess_full"]
COLOR = {"ei": "#2a78d6", "mcts": "#eb6834", "voi_rule": "#1baf7a", "gate": "#6250d6",
         "learned_bc_voi": "#9fe0c4", "learned_bc_mcts": "#f7b48f",
         "learned_batch_q": "#c2185b", "learned_fitted_q": "#7b1040",
         "model_mle_stop_only": "#d9d9d9", "model_bayes_mean_stop_only": "#bdbdbd",
         "model_bayes_pess_stop_only": "#969696", "model_mle_full": "#737373",
         "model_bayes_mean_full": "#525252", "model_bayes_pess_full": "#252525"}
LABEL = {"ei": "GP-EI (baseline)", "mcts": "MCTS", "voi_rule": "One-step rule", "gate": "Adarsh's gate (online RL)",
         "learned_bc_voi": "Behavioral cloning: one-step rule", "learned_bc_mcts": "Behavioral cloning: MCTS",
         "learned_batch_q": "Batch Q-learning (replay)", "learned_fitted_q": "Batch Q-learning (fitted)",
         "model_mle_stop_only": "Learned model, MLE (stop only)",
         "model_bayes_mean_stop_only": "Learned model, Bayes mean (stop only)",
         "model_bayes_pess_stop_only": "Learned model, Bayes pessimistic (stop only)",
         "model_mle_full": "Learned model, MLE (full)", "model_bayes_mean_full": "Learned model, Bayes mean (full)",
         "model_bayes_pess_full": "Learned model, Bayes pessimistic (full)"}
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
    x = np.arange(len(POLICIES))
    for j, ds in enumerate(DATASETS):
        for i, (col, ylab) in enumerate(rows):
            ax = axes[i, j]
            vals = [s.loc[(ds, p), col] for p in POLICIES]
            err = [s.loc[(ds, p), "net_se"] for p in POLICIES] if col == "net_utility" else None
            ax.bar(x, vals, color=[COLOR[p] for p in POLICIES], width=0.78,
                   yerr=err, error_kw=dict(ecolor=MUTED, lw=0.8))
            ax.axhline(0, color=MUTED, lw=0.6)
            for sep in (3.5, 7.5):
                ax.axvline(sep, color=GRID, lw=1.2, zorder=0)
            if i == 0:
                ax.set_title(DS_LABEL[ds], fontsize=11)
            if j == 0:
                ax.set_ylabel(ylab)
            ax.set_xticks([])
    fig.legend([Patch(color=COLOR[p]) for p in POLICIES], [LABEL[p] for p in POLICIES],
               loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(f"Offline RL on the real datasets ({n} paired seeds; bars on net utility are ±1 standard error)\n"
                 "left: existing agents · middle: model-free offline RL · right: learned models + PBVI. "
                 "Learned policies never saw these pools.", fontsize=10)
    fig.savefig(os.path.join(FIG, "benchmark.png"))
    plt.close(fig)


def ope_figure():
    off = pd.read_csv(os.path.join(RESULTS, "offline_eval.csv")).dropna(subset=["is_weighted"]).set_index("policy")
    order = [p for p in POLICIES if p in off.index]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    x = np.arange(len(order))
    ax.bar(x - 0.27, off.loc[order, "true_net"], width=0.27, yerr=off.loc[order, "true_se"],
           color=[COLOR[p] for p in order], error_kw=dict(ecolor=MUTED, lw=0.8))
    ax.bar(x, off.loc[order, "is_weighted"], width=0.27, color="none", edgecolor=INK, hatch="///", lw=0.8)
    ax.bar(x + 0.27, off.loc[order, "is_ordinary"], width=0.27, color="none", edgecolor=MUTED, hatch="...", lw=0.8)
    for xi, p in zip(x, order):
        ax.text(xi, max(off.loc[p, ["true_net", "is_weighted", "is_ordinary"]].max(), 0) + 0.03,
                f"ESS {off.loc[p, 'ess']:.0f}", ha="center", fontsize=7, color=MUTED)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[p] for p in order], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Net utility on made-up pools")
    ax.legend([Patch(color="#9e9e9e"), Patch(facecolor="none", edgecolor=INK, hatch="///"),
               Patch(facecolor="none", edgecolor=MUTED, hatch="...")],
              ["True (1,000 fresh campaigns)", "Weighted IS from the log", "Ordinary IS from the log"],
              frameon=False, fontsize=8, loc="upper left", ncol=3)
    ax.set_ylim(top=0.68)
    ax.set_title("Importance sampling from 1,080 held-out logged campaigns vs. the truth\n"
                 "(ESS: effective number of campaigns behind each estimate)", fontsize=9)
    fig.savefig(os.path.join(FIG, "ope.png"))
    plt.close(fig)


if __name__ == "__main__":
    ope_figure()
    benchmark_figure()
    print("wrote", FIG)
