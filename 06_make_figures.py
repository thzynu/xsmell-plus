"""Step 6 - Figures regenerated from the real experiment (replaces Fig. 2, 4, 5 of the conference paper).

Usage:  python 06_make_figures.py
Needs:  results/per_seed_project.csv (from 05_run_experiments.py), data/dataset_*.csv, pip install shap

For each smell the best configuration (model x SMOTE) is chosen by mean Macro-F1 over the
5 seeds of the project-wise split. SHAP is computed on a project-wise HOLD-OUT set
(projects never seen in training), never on the training data.
Outputs: figures/*.png and figures/figure_manifest.csv (which model/seed/sample each figure uses).
"""
import importlib.util, os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore")
spec = importlib.util.spec_from_file_location("exp", "05_run_experiments.py")
exp = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)

OUT = "figures"; os.makedirs(OUT, exist_ok=True)
res = pd.read_csv("results/per_seed_project.csv")
cm = pd.read_csv("results/confusion_project.csv")
manifest = []
WEAK_MCC = 0.30          # same threshold as the recommender's abstention rule (08 --mcc-min)

# ---- best configuration per smell (by mean Macro-F1 over seeds) ----
best = (res.groupby(["smell", "model", "smote"])["macro_f1"].mean().reset_index()
        .sort_values("macro_f1", ascending=False).groupby("smell").head(1).set_index("smell"))
print(best)

# ---- Fig. 2 replacement: F1 per smell (mean +- SD over seeds) ----
fig, ax = plt.subplots(figsize=(7, 4))
smells = list(exp.TASKS)
xs = np.arange(len(smells)); w = 0.38
for i, (col, lab) in enumerate([("macro_f1", "Macro-F1"), ("f1_smelly", "F1 (smelly class)")]):
    m, s = [], []
    for sm in smells:
        b = best.loc[sm]
        d = res[(res.smell == sm) & (res.model == b["model"]) & (res.smote == b["smote"])][col]
        m.append(d.mean()); s.append(d.std())
    ax.bar(xs + (i - 0.5) * w, m, w, yerr=s, capsize=3, label=lab)
ax.set_xticks(xs)
ax.set_xticklabels([f"{sm}\n({best.loc[sm,'model']}{'+SMOTE' if best.loc[sm,'smote'] else ''})" for sm in smells], fontsize=8)
ax.set_ylabel("Score (mean ± SD, 5 seeds)"); ax.set_ylim(0, 1); ax.legend(fontsize=8)
ax.set_title("Per-smell detection performance (project-wise split)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_f1_per_smell.png", dpi=200); plt.close(fig)
manifest.append(("fig2_f1_per_smell.png", "res/per_seed_project.csv", "best config per smell, mean±SD over seeds"))

# ---- evaluation-protocol comparison (random vs project-wise vs leave-one-organisation-out) ----
if os.path.exists("results/split_comparison.csv"):
    sc = pd.read_csv("results/split_comparison.csv")
    splits_present = [x for x in ["random", "project", "org", "project_dedup"] if x in set(sc.split)]
    fig, ax = plt.subplots(figsize=(8, 4)); w = 0.8 / max(1, len(splits_present))
    for i, sp in enumerate(splits_present):
        x = sc[sc.split == sp].set_index("smell").reindex(smells)
        ax.bar(np.arange(len(smells)) + (i - (len(splits_present) - 1) / 2) * w, x["macro_f1_mean"], w,
               yerr=x["macro_f1_sd"].fillna(0), capsize=2, label=sp)
    ax.set_xticks(range(len(smells))); ax.set_xticklabels(smells, fontsize=8)
    ax.set_ylabel("Macro-F1 (mean ± SD over seeds)"); ax.set_ylim(0, 1); ax.legend(fontsize=7, ncol=2)
    ax.set_title("Same configuration under different evaluation protocols")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_split_comparison.png", dpi=200); plt.close(fig)
    manifest.append(("fig_split_comparison.png", "results/split_comparison.csv", "random / project / org / dedup"))

# ---- class distribution (the old Fig. 2 content, now correctly captioned) ----
aud = pd.read_csv("results/dataset_audit.csv")
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.bar(aud["smell"], aud["n_smelly"], label="smelly"); ax.bar(aud["smell"], aud["n"] - aud["n_smelly"], bottom=aud["n_smelly"], label="clean", alpha=.5)
for i, r in aud.iterrows():
    ax.text(i, r["n"] + 20, f"{r['smelly_rate']:.1%}", ha="center", fontsize=8)
ax.set_ylabel("Samples"); ax.set_title("Class distribution after CK join"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_class_distribution.png", dpi=200); plt.close(fig)
manifest.append(("fig_class_distribution.png", "results/dataset_audit.csv", "smelly vs clean per smell"))

# ---- confusion matrices (summed over folds, averaged over seeds) ----
fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
for ax, sm in zip(axes, smells):
    b = best.loc[sm]
    d = cm[(cm.smell == sm) & (cm.model == b["model"]) & (cm.smote == b["smote"])][["tn", "fp", "fn", "tp"]].mean()
    mat = np.array([[d.tn, d.fp], [d.fn, d.tp]])
    ax.imshow(mat, cmap="Blues")
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, f"{v:.0f}", ha="center", va="center", color="black" if v < mat.max() * .6 else "white")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xticklabels(["clean", "smelly"]); ax.set_yticklabels(["clean", "smelly"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(sm, fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_confusion_matrices.png", dpi=200); plt.close(fig)
manifest.append(("fig_confusion_matrices.png", "results/confusion_project.csv", "summed over folds, mean over seeds"))

# ---- SHAP: out-of-fold over the 5 project-wise folds (each sample is explained by a model
#      that never saw its project), summary + waterfall + importance + stability ----
from scipy.stats import spearmanr
SEED = 0
imp_rows, stab_rows = [], []
for sm in smells:
    b = best.loc[sm]
    df, feats, *_ = exp.load(sm, "data")
    y = df["label"].astype(int).values; X = df[feats].values; g = df["project"].values
    V = np.zeros(X.shape); B = np.zeros(len(y)); P = np.zeros(len(y)); fold_imp = []
    for tr, te in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y, g):
        pipe = exp.make_model(b["model"], SEED, bool(b["smote"]))
        pipe.fit(X[tr], y[tr])
        def prep(A):                       # impute/scale only; SMOTE is a training-time step
            for name, step in pipe.steps[:-1]:
                if name != "smote":
                    A = step.transform(A)
            return A
        Xtr_t, Xte_t = prep(X[tr]), prep(X[te])
        clf = pipe.named_steps["clf"]
        expl = shap.LinearExplainer(clf, Xtr_t) if b["model"] == "logreg" else shap.TreeExplainer(clf)
        sv = expl(Xte_t)
        vals = sv.values[:, :, 1] if sv.values.ndim == 3 else sv.values
        base = sv.base_values[:, 1] if sv.base_values.ndim == 2 else sv.base_values
        V[te] = vals; B[te] = base; P[te] = pipe.predict_proba(X[te])[:, 1]
        fold_imp.append(np.abs(vals).mean(0))
    imp = np.abs(V).mean(0)
    order = np.argsort(-imp)
    for r, j in enumerate(order[:15], 1):
        imp_rows.append({"smell": sm, "model": b["model"], "rank": r, "feature": feats[j], "mean_abs_shap": imp[j]})
    rhos = [spearmanr(fold_imp[i], fold_imp[k])[0] for i in range(len(fold_imp)) for k in range(i + 1, len(fold_imp))]
    stab_rows.append({"smell": sm, "model": b["model"], "mean_pairwise_spearman": float(np.nanmean(rhos)),
                      "min_pairwise_spearman": float(np.nanmin(rhos)), "n_positives": int(y.sum())})
    print(f"\n{sm} ({b['model']}) top features by mean|SHAP|:", ", ".join(feats[j] for j in order[:8]))

    mcc_sm = float(res[(res.smell == sm) & (res.model == b["model"]) & (res.smote == b["smote"])]["mcc"].mean())
    weak = mcc_sm < WEAK_MCC
    tag = f"\n[reference only - weak detector, MCC={mcc_sm:.2f}]" if weak else ""
    plt.figure()
    shap.summary_plot(V, X, feature_names=feats, max_display=12, show=False)
    plt.title(f"SHAP summary - {sm} ({b['model']}{'+SMOTE' if b['smote'] else ''}, out-of-fold, project-wise)" + tag, fontsize=9)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig4_shap_summary_{sm}.png", dpi=200, bbox_inches="tight"); plt.close()
    manifest.append((f"fig4_shap_summary_{sm}.png", f"{b['model']} seed={SEED}, 5 project-wise folds",
                     f"n={len(y)}, positives={int(y.sum())}"))

    cand = np.where((y == 1) & (P >= 0.5))[0]
    if len(cand):
        k = cand[np.argmax(P[cand])]
        plt.figure()
        shap.plots.waterfall(shap.Explanation(values=V[k], base_values=B[k], data=X[k], feature_names=feats),
                             max_display=10, show=False)
        if weak:
            plt.gcf().text(0.5, 0.995, f"[reference only - weak detector, MCC={mcc_sm:.2f}]", ha="center", va="top", fontsize=8, color="firebrick")
        plt.tight_layout(); plt.savefig(f"{OUT}/fig5_shap_waterfall_{sm}.png", dpi=200, bbox_inches="tight"); plt.close()
        manifest.append((f"fig5_shap_waterfall_{sm}.png", f"{b['model']} seed={SEED}",
                         f"sample_id={df.iloc[k]['sample_id']}, code_name={df.iloc[k]['code_name']}, p={P[k]:.2f}"))

pd.DataFrame(imp_rows).to_csv(f"{OUT}/shap_importance.csv", index=False)
st = pd.DataFrame(stab_rows); st.to_csv(f"{OUT}/shap_stability.csv", index=False)
print("\nSHAP rank stability across folds (Spearman):\n", st.round(2).to_string(index=False))

# ---- correlation heatmap (kept as its own figure, for the leakage / collinearity discussion) ----
df, feats, *_ = exp.load("blob", "data")
corr = df[feats].corr()
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(feats))); ax.set_yticks(range(len(feats)))
ax.set_xticklabels(feats, rotation=90, fontsize=5); ax.set_yticklabels(feats, fontsize=5)
fig.colorbar(im, fraction=0.046); ax.set_title("Feature correlation (class-level, blob subset)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_correlation_heatmap.png", dpi=200); plt.close(fig)
manifest.append(("fig_correlation_heatmap.png", "data/dataset_class.csv", "Pearson, blob subset"))

pd.DataFrame(manifest, columns=["figure", "source", "details"]).to_csv(f"{OUT}/figure_manifest.csv", index=False)
print("\nDone. Figures in", OUT, "(see figure_manifest.csv for the exact model/sample behind every figure)")
