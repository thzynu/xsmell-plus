"""Step 12 - Figures for the new contributions, drawn only from files already produced by steps 05-09.

Usage: python 12_paper_figures.py      (takes a few seconds, no model is trained)
Outputs (figures/):
  fig1_pipeline.png               four-stage XSmell+ pipeline (replaces the mislabelled Fig. 1 of the conference paper)
  fig_ablation_feasibility.png    share of ALL flagged samples that get a feasible top-1 recommendation (= coverage x feasibility)
  fig_ablation_impact.png         simulated drop in P(smelly) of the top-1 recommendation (model-based estimate)
  fig_fidelity_curves.png         mean drop in P(smelly) when the top-k features are removed (SHAP / global / random)
The simulated impact is a model-based estimate, NOT evidence that real code improves (see 11_before_after.py).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "figures"; os.makedirs(OUT, exist_ok=True)
SMELLS = ["blob", "data_class", "long_method", "feature_envy"]


# ---------------------------------------------------------------- Fig. 1: pipeline
def pipeline():
    stages = [
        ("1  Data construction", ["MLCQ labels (reviewers)", "source at the labelled commit", "CK metrics\n(class 48 / method 27)", "label merge, de-duplication"]),
        ("2  Cross-project detection", ["project-wise grouped CV", "impute + scale (train only)", "SMOTE on training folds only", "LogReg / RF / XGBoost\n5 seeds, mean ± SD"]),
        ("3  Explanation", ["out-of-fold SHAP values", "group by metric family", "stability (Spearman)", "fidelity (remove top-k)"]),
        ("4  XSmell+ recommendation", ["direction filter", "refactoring preconditions", "what-if impact estimate", "score S(r|x), conflict-free\nselection, abstain if weak"]),
    ]
    fig, ax = plt.subplots(figsize=(12.4, 4.2)); ax.set_xlim(0, 12.4); ax.set_ylim(0, 4.2); ax.axis("off")
    w, gap = 2.6, 0.5
    for i, (title, items) in enumerate(stages):
        x0 = 0.2 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x0, 0.2), w, 3.8, boxstyle="round,pad=0.02,rounding_size=0.12", fc="#f4f6fa", ec="#33415c", lw=1.4))
        ax.text(x0 + w / 2, 3.7, title, ha="center", va="center", fontsize=10, fontweight="bold", color="#1d2d50")
        for j, t in enumerate(items):
            y = 3.0 - j * 0.78
            ax.add_patch(FancyBboxPatch((x0 + 0.15, y - 0.3), w - 0.3, 0.6, boxstyle="round,pad=0.01,rounding_size=0.08", fc="white", ec="#8d99ae", lw=0.8))
            ax.text(x0 + w / 2, y, t, ha="center", va="center", fontsize=7.5)
        if i < 3:
            ax.add_patch(FancyArrowPatch((x0 + w + 0.03, 2.1), (x0 + w + gap - 0.03, 2.1), arrowstyle="-|>", mutation_scale=16, lw=1.6, color="#33415c"))
    fig.tight_layout(); fig.savefig(f"{OUT}/fig1_pipeline.png", dpi=200); plt.close(fig)


# ---------------------------------------------------------------- ablation
LABEL = {"rule_only": "Rule-only", "shap_only": "SHAP-only", "xsmell_noF": "XSmell+ w/o feasibility",
         "xsmell_noDir": "XSmell+ w/o direction", "xsmell_E": "XSmell+ (evidence only)", "xsmell_full": "XSmell+ (full)"}
ORDER = ["rule_only", "shap_only", "xsmell_noF", "xsmell_noDir", "xsmell_E", "xsmell_full"]


def ablation_bars(col, ylabel, fname, title, times_coverage=False):
    f = "results/recommender_ablation.csv"
    if not os.path.exists(f):
        print("skip", fname, "(run 08 first)"); return
    d = pd.read_csv(f)
    d = d[(d.rho == d.rho.iloc[0]) & (d.lam == d.lam.iloc[0])].copy()
    if times_coverage:      # feasible_top1 is measured among samples that RECEIVE a recommendation; coverage is the share that do
        d[col] = d[col] * d["coverage"]
    fig, ax = plt.subplots(figsize=(9, 4)); n = len(ORDER); w = 0.8 / n
    for i, c in enumerate(ORDER):
        v = [d[(d.smell == s) & (d.config == c)][col].iloc[0] if len(d[(d.smell == s) & (d.config == c)]) else np.nan for s in SMELLS]
        ax.bar(np.arange(len(SMELLS)) + (i - (n - 1) / 2) * w, v, w, label=LABEL[c])
    labels = []
    for s in SMELLS:
        ab = bool(d[d.smell == s].abstained.iloc[0]) if (d.smell == s).any() else False
        labels.append(s + ("\n(abstains: weak detector)" if ab else ""))
    ax.set_xticks(range(len(SMELLS))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10); ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout(); fig.savefig(f"{OUT}/{fname}", dpi=200, bbox_inches="tight"); plt.close(fig)


def fidelity():
    f = "results/fidelity_summary.csv"
    if not os.path.exists(f):
        print("skip fidelity (run 09 first)"); return
    d = pd.read_csv(f)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    for ax, s in zip(axes, SMELLS):
        g = d[d.smell == s].sort_values("k")
        ax.plot(g.k, g.shap_local, "o-", label="SHAP (local)")
        ax.plot(g.k, g.global_importance, "s--", label="Global importance")
        ax.plot(g.k, g.random, "^:", label="Random")
        ax.set_title(s + (" (weak detector)" if s == "feature_envy" else ""), fontsize=9); ax.set_xlabel("k features removed")
    axes[0].set_ylabel("Mean drop in P(smelly)"); axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_fidelity_curves.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    pipeline()
    ablation_bars("feasible_top1", "Flagged samples with a feasible top-1 (share)", "fig_ablation_feasibility.png",
                  "Share of flagged samples that receive an applicable top-1 recommendation", times_coverage=True)
    ablation_bars("sim_dp_top1", "Simulated drop in P(smelly)", "fig_ablation_impact.png",
                  "Simulated impact of the top-1 recommendation (model-based estimate, not real refactoring)")
    fidelity()
    print("figures written to", OUT)
