"""Figures for the validation experiments (ADM ch. 14).

    python antonio_tests/validation/plots.py

  results/figures/robustness_noise.png   net utility vs. the TRUE noise level, like ADM fig. 14.4.
                                          solid: the agent keeps assuming the default noise;
                                          dashed: the agent is told the true noise
  results/figures/robustness_other.png   change in net utility from the default world for the
                                          other mismatches (outliers, failures, initial design)
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIG = os.path.join(RESULTS, "figures")
os.makedirs(FIG, exist_ok=True)

AGENTS = ["ei", "mcts", "voi_rule", "gate", "fs2_ei", "ds3_ei", "bc_mcts"]
COLOR = {"ei": "#2a78d6", "mcts": "#eb6834", "voi_rule": "#1baf7a", "gate": "#6250d6",
         "fs2_ei": "#6baed6", "ds3_ei": "#7a0177", "bc_mcts": "#f7b48f"}
LABEL = {"ei": "GP-EI", "mcts": "MCTS", "voi_rule": "One-step rule", "gate": "Adarsh's gate",
         "fs2_ei": "Forward search d2 (GP-EI picks)", "ds3_ei": "Determinized d3 (GP-EI picks)",
         "bc_mcts": "Behavioral clone of MCTS"}
DS_LABEL = {"p3ht": "P3HT/CNT", "perovskite": "Perovskite", "crossed_barrel": "Crossed barrel"}
DATASETS = ["p3ht", "perovskite", "crossed_barrel"]
OTHER = [("outlier_0", "no outliers"), ("outlier_0.2", "20% outliers"), ("fail_0", "no failures"),
         ("fail_0.3", "30% failures"), ("init_1", "1 initial exp."), ("init_10", "10 initial exp.")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def noise_figure(df):
    g = df.groupby(["dataset", "condition", "agent"]).net_utility
    mean, se = g.mean(), g.sem()
    scales = [0.5, 1.0, 2.0, 4.0]
    mis = {0.5: "noise_x0.5", 1.0: "default", 2.0: "noise_x2", 4.0: "noise_x4"}
    known = {0.5: "known_noise_x0.5", 1.0: "default", 2.0: "known_noise_x2", 4.0: "known_noise_x4"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, ds in zip(axes, DATASETS):
        for ag in AGENTS:
            for conds, ls in ((mis, "-"), (known, "--")):
                y = np.array([mean.get((ds, conds[s], ag), np.nan) for s in scales])
                e = np.array([se.get((ds, conds[s], ag), np.nan) for s in scales])
                ax.plot(scales, y, ls, color=COLOR[ag], lw=1.6 if ls == "-" else 1.1, marker="o", ms=3)
                if ls == "-":
                    ax.fill_between(scales, y - e, y + e, color=COLOR[ag], alpha=0.08, lw=0)
        ax.axvline(1.0, color=MUTED, lw=0.8, ls=":")
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_xscale("log", base=2)
        ax.set_xticks(scales)
        ax.set_xticklabels(["0.5×", "1× (assumed)", "2×", "4×"])
        ax.set_title(DS_LABEL[ds], fontsize=11)
        ax.set_xlabel("True reading noise (× the replicate noise)")
    axes[0].set_ylabel("Net utility")
    handles = [Line2D([], [], color=COLOR[a], lw=2) for a in AGENTS] + \
              [Line2D([], [], color=MUTED, ls="-"), Line2D([], [], color=MUTED, ls="--")]
    labels = [LABEL[a] for a in AGENTS] + ["agent assumes 1× (mismatched)", "agent told the true noise"]
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.12))
    n = int(df.groupby(["dataset", "condition", "agent"]).size().max())
    fig.suptitle(f"Robustness to reading noise (ADM 14.3; {n} paired seeds per point, band ±1 SE on the solid lines)",
                 fontsize=10)
    fig.savefig(os.path.join(FIG, "robustness_noise.png"))
    plt.close(fig)


def other_figure(ch):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    width = 0.11
    for ax, ds in zip(axes, DATASETS):
        d = ch[ch.dataset == ds].set_index(["condition", "agent"])
        for gi, (cond, _) in enumerate(OTHER):
            for ai, ag in enumerate(AGENTS):
                if (cond, ag) in d.index:
                    r = d.loc[(cond, ag)]
                    ax.bar(gi + (ai - 3) * width, r.change, width * 0.92, color=COLOR[ag], yerr=r.se,
                           error_kw=dict(ecolor=MUTED, lw=0.6))
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_xticks(range(len(OTHER)))
        ax.set_xticklabels([lab for _, lab in OTHER], rotation=30, ha="right")
        ax.set_title(DS_LABEL[ds], fontsize=11)
    axes[0].set_ylabel("Change in net utility\nfrom the default world")
    fig.legend([Patch(color=COLOR[a]) for a in AGENTS], [LABEL[a] for a in AGENTS], loc="lower center",
               ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("Other mismatches: agents assume 5% outliers, 10% failures; "
                 "initial design is shared by agent and world (±1 SE, paired by seed)", fontsize=10)
    fig.savefig(os.path.join(FIG, "robustness_other.png"))
    plt.close(fig)


FAM = {"ei": ("GP-EI, fixed budget", "#2a78d6", 47), "voi": ("One-step rule, planning cost ×λ", "#1baf7a", 1.0),
       "fs2": ("Forward search d2, planning cost ×λ", "#6baed6", 1.0), "mcts": ("MCTS, planning cost ×λ", "#eb6834", 1.0),
       "gate": ("Adarsh's gate, threshold t", "#6250d6", 0.5)}


def trade_figure(s):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, ds in zip(axes, DATASETS):
        d = s[s.dataset == ds]
        xs = np.linspace(0, 52, 2)
        for u in np.arange(-0.4, 0.8, 0.1):            # iso net utility, if every campaign commits
            ax.plot(xs, u + 0.1 + 0.01 * xs, color=GRID, lw=0.7, zorder=0)
        for fam, (lab, col, default) in FAM.items():
            f = d[d.family == fam].sort_values("experiments")
            ax.plot(f.experiments, f.hit, "-o", color=col, ms=3.5, lw=1.4, label=lab)
            dd = f[np.isclose(f.knob, default)]
            ax.plot(dd.experiments, dd.hit, "o", color=col, ms=8, mec=INK, mew=0.8)
        p = d[d.pareto].sort_values("experiments")
        ax.step(p.experiments, p.hit, where="post", color=INK, lw=1.0, ls="--", label="Pareto frontier")
        ax.set_xlim(0, 52)
        ax.set_ylim(0, min(1.0, d.hit.max() + 0.08))
        ax.set_title(DS_LABEL[ds], fontsize=11)
        ax.set_xlabel("Experiments run (mean per campaign)")
    axes[0].set_ylabel("P(committed candidate meets spec)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("Trade analysis (ADM 14.4): experiment spend vs. success. Large dots: each agent as deployed. "
                 "Grey lines: equal net utility if every campaign commits (+0.1 per line).", fontsize=10)
    fig.savefig(os.path.join(FIG, "trade.png"))
    plt.close(fig)


if __name__ == "__main__":
    if os.path.exists(os.path.join(RESULTS, "robustness.csv")):
        df = pd.read_csv(os.path.join(RESULTS, "robustness.csv"))
        noise_figure(df)
        other_figure(pd.read_csv(os.path.join(RESULTS, "robustness_changes.csv")))
    if os.path.exists(os.path.join(RESULTS, "trade_points.csv")):
        trade_figure(pd.read_csv(os.path.join(RESULTS, "trade_points.csv")))
    print("wrote", FIG)
