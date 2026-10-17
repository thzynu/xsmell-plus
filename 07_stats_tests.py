"""Step 7 - Statistical comparison of configurations (exploratory) + split comparison table + per-organisation breakdown.

Usage: python 07_stats_tests.py [--min-pos 10]
Reads results/per_fold_<split>.csv produced by 05_run_experiments.py.

Only the SMOTE-vs-none family is inferential. "best vs others" rows are labelled DESCRIPTIVE ONLY
because the best configuration is selected on the same folds (circular).
Tests: paired Wilcoxon signed-rank on matched (seed, fold) results, Holm-corrected within each
family of comparisons.  (a) SMOTE vs no SMOTE for every model;  (b) the best configuration vs
every other configuration.  Caveat to state in the paper: fold results from different seeds
re-use the same samples, so they are not independent; p-values are exploratory, and effect sizes
(mean paired difference) matter more than p.
Also writes results/split_comparison.csv: best configuration per smell under each split.
For the leave-one-organisation-out split the mean over held-out organisations is weighted by test size
(a simple mean is inflated by tiny folds).

Part 3 (was 07b): leave-one-organisation-out breakdown -> results/org_breakdown.csv. For the best configuration of each
smell it lists the score of every held-out organisation (with its number of test positives) and three summaries:
unweighted mean, mean over organisations with >= --min-pos test positives, and mean weighted by test size.
Folds with very few positives are unstable (one prediction moves Macro-F1 a lot), so the unweighted mean can mislead.
"""
import os, warnings
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")
R = "results"
METRICS = ["macro_f1", "mcc", "pr_auc"]


def holm(p):
    p = np.asarray(p, float); n = len(p); order = np.argsort(p); adj = np.empty(n); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (n - rank) * p[i]); adj[i] = min(1.0, run)
    return adj


def paired(a, b):
    d = (a - b).dropna()
    if len(d) < 6 or np.allclose(d, 0):
        return np.nan, d.mean() if len(d) else np.nan, len(d)
    try:
        return wilcoxon(d, zero_method="wilcox").pvalue, d.mean(), len(d)
    except ValueError:
        return np.nan, d.mean(), len(d)


def main():
    out = []
    for split in ["project", "org", "random", "project_dedup"]:
        f = os.path.join(R, f"per_fold_{split}.csv")
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        d["cfg"] = d["model"] + np.where(d["smote"], "+SMOTE", "")
        d["key"] = d["seed"].astype(str) + "|" + d["fold"].astype(str)
        for smell, g in d.groupby("smell"):
            piv = {m: g.pivot_table(index="key", columns="cfg", values=m) for m in METRICS}
            # (a) SMOTE effect per model
            fam = []
            for model in sorted(g["model"].unique()):
                for m in METRICS:
                    p, eff, n = paired(piv[m][model + "+SMOTE"], piv[m][model])
                    fam.append(dict(split=split, smell=smell, family="SMOTE vs none", comparison=model, metric=m,
                                    mean_diff=eff, n_pairs=n, p=p))
            # (b) best config vs the others (best by mean macro-F1)
            best = piv["macro_f1"].mean().idxmax()
            for other in piv["macro_f1"].columns:
                if other == best:
                    continue
                for m in METRICS:
                    p, eff, n = paired(piv[m][best], piv[m][other])
                    fam.append(dict(split=split, smell=smell, family=f"DESCRIPTIVE ONLY best ({best}) vs others",
                                    comparison=other, metric=m, mean_diff=eff, n_pairs=n, p=p))
            fam = pd.DataFrame(fam)
            for _, sub in fam.groupby(["family", "metric"]):
                fam.loc[sub.index, "p_holm"] = holm(sub["p"].fillna(1.0).values)
            out.append(fam)
    if out:
        st = pd.concat(out, ignore_index=True)
        st.to_csv(os.path.join(R, "stats_tests.csv"), index=False)
        # "best vs others" is circular (the best config was chosen on the same folds): never inferential
        st.loc[st["family"].str.startswith("DESCRIPTIVE"), "p_holm"] = np.nan
        st.to_csv(os.path.join(R, "stats_tests.csv"), index=False)
        sig = st[(st["metric"] == "macro_f1") & (st["p_holm"] < 0.05) & (st["family"] == "SMOTE vs none")]
        print(f"{len(st)} comparisons written to results/stats_tests.csv; "
              f"{len(sig)} pre-specified SMOTE-vs-none comparisons significant on Macro-F1 after Holm (alpha=0.05):")
        print(sig[["split", "smell", "family", "comparison", "mean_diff", "p_holm"]].round(4).to_string(index=False)
              if len(sig) else "  none")

    # split comparison: best configuration (chosen under the project split) evaluated under every split
    rows = []
    ref = pd.read_csv(os.path.join(R, "per_seed_project.csv"))
    best = (ref.groupby(["smell", "model", "smote"])["macro_f1"].mean().reset_index()
            .sort_values("macro_f1", ascending=False).groupby("smell").head(1))
    for split in ["random", "project", "org", "project_dedup"]:
        f = os.path.join(R, f"per_seed_{split}.csv")
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        if split == "org":
            # leave-one-organisation-out: folds are organisations of very different size (some have only a
            # handful of smelly samples), so a simple mean over folds is inflated/unstable. Weight each held-out
            # organisation by its number of test samples, per seed, then average over seeds.
            pf = pd.read_csv(os.path.join(R, "per_fold_org.csv"))
            d = (pf.groupby(["smell", "model", "smote", "seed"])
                   .apply(lambda g: pd.Series({m: np.average(g[m], weights=g["n_test"]) for m in ["macro_f1", "mcc", "pr_auc"]}))
                   .reset_index())
        for _, b in best.iterrows():
            x = d[(d.smell == b.smell) & (d.model == b.model) & (d.smote == b.smote)]
            rows.append(dict(smell=b.smell, config=b.model + ("+SMOTE" if b.smote else ""), split=split,
                             macro_f1_mean=x.macro_f1.mean(), macro_f1_sd=x.macro_f1.std(),
                             mcc_mean=x.mcc.mean(), pr_auc_mean=x.pr_auc.mean(), n_seeds=len(x)))
    sc = pd.DataFrame(rows)
    sc.to_csv(os.path.join(R, "split_comparison.csv"), index=False)
    print("\nBest configuration per smell under each evaluation protocol (Macro-F1 mean ± SD):")
    sc["macro_f1"] = sc.macro_f1_mean.round(3).astype(str) + " ± " + sc.macro_f1_sd.fillna(0).round(3).astype(str)
    print(sc.pivot(index=["smell", "config"], columns="split", values="macro_f1").to_string())


def org_breakdown(min_pos):
    """Per held-out organisation scores of the best configuration (needs per_fold_org.csv and per_seed_project.csv)."""
    f_org, f_ref = os.path.join(R, "per_fold_org.csv"), os.path.join(R, "per_seed_project.csv")
    if not (os.path.exists(f_org) and os.path.exists(f_ref)):
        print("\nskip org breakdown (needs per_fold_org.csv and per_seed_project.csv)"); return
    ref, d = pd.read_csv(f_ref), pd.read_csv(f_org)
    best = (ref.groupby(["smell", "model", "smote"])["macro_f1"].mean().reset_index()
            .sort_values("macro_f1", ascending=False).groupby("smell").head(1))
    rows, summ = [], []
    for _, b in best.iterrows():
        x = d[(d.smell == b.smell) & (d.model == b.model) & (d.smote == b.smote)]
        per = x.groupby("fold").agg(macro_f1=("macro_f1", "mean"), mcc=("mcc", "mean"),
                                    n_test=("n_test", "first"), n_pos_test=("n_pos_test", "first")).reset_index()
        per.insert(0, "config", b.model + ("+SMOTE" if b.smote else "")); per.insert(0, "smell", b.smell)
        rows.append(per)
        ok = per[per.n_pos_test >= min_pos]
        summ.append(dict(smell=b.smell, config=per.config.iloc[0], n_orgs=len(per),
                         mean_all=per.macro_f1.mean(),
                         **{f"mean_orgs_with_{min_pos}+_pos": ok.macro_f1.mean() if len(ok) else np.nan},
                         n_orgs_kept=len(ok), weighted_by_n=np.average(per.macro_f1, weights=per.n_test)))
    out = pd.concat(rows); out.to_csv(os.path.join(R, "org_breakdown.csv"), index=False)
    print("\nLeave-one-organisation-out breakdown (best configuration per smell):")
    print(out.round(3).to_string(index=False))
    print("\nSummary (Macro-F1):"); print(pd.DataFrame(summ).round(3).to_string(index=False))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-pos", type=int, default=10, help="min test positives for an organisation to count in the filtered mean")
    a = ap.parse_args()
    main()
    org_breakdown(a.min_pos)
