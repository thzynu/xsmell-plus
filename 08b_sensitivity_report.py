"""Step 8b - Sensitivity of the recommender (RQ6): does the ablation conclusion survive other rho / lambda?

Reads results/recommender_ablation.csv produced by `python 08_recommender.py --sensitivity` (no retraining).
  rho    = fraction of the code a refactoring is assumed to touch in the what-if (0.3 / 0.5 / 0.7)
  lambda = weight of SHAP evidence vs simulated impact in the score (0.25 / 0.5 / 0.75)
Output: results/recommender_sensitivity.csv (+ printed summary).

How to read it: the ABSOLUTE simulated impact grows with rho by construction (a bigger assumed change lowers the
predicted probability more), so only the ORDER of the configurations and the trends are meaningful.
"""
import pandas as pd

a = pd.read_csv("results/recommender_ablation.csv")
if a[["rho", "lam"]].drop_duplicates().shape[0] < 2:
    raise SystemExit("Only one (rho, lambda) setting found. Run: python 08_recommender.py --sensitivity")
pd.set_option("display.width", 220)

settings = a[["rho", "lam"]].drop_duplicates().sort_values(["rho", "lam"]).values.tolist()
rows = []
for sm, g in a.groupby("smell"):
    for (rho, lam), h in g.groupby(["rho", "lam"]):
        d = h.set_index("config")
        best = d.sim_dp_top1.idxmax()
        rows.append(dict(smell=sm, rho=rho, lam=lam, best_by_sim_dp=best,
                         xsmell_full=d.loc["xsmell_full", "sim_dp_top1"], rule_only=d.loc["rule_only", "sim_dp_top1"],
                         shap_only=d.loc["shap_only", "sim_dp_top1"],
                         full_minus_rule=round(d.loc["xsmell_full", "sim_dp_top1"] - d.loc["rule_only", "sim_dp_top1"], 3),
                         full_minus_shap=round(d.loc["xsmell_full", "sim_dp_top1"] - d.loc["shap_only", "sim_dp_top1"], 3),
                         coverage_full=d.loc["xsmell_full", "coverage"],
                         agreement_full=d.loc["xsmell_full", "top1_agreement_across_seeds"],
                         abstained=bool(d.loc["xsmell_full", "abstained"])))
out = pd.DataFrame(rows)
out.to_csv("results/recommender_sensitivity.csv", index=False)

print("=== Simulated impact of the top-1 recommendation, XSmell+ (full) vs baselines, all settings ===")
print(out[["smell", "rho", "lam", "xsmell_full", "rule_only", "shap_only", "full_minus_rule", "full_minus_shap",
           "best_by_sim_dp"]].to_string(index=False))
n = len(settings)
print("\n=== Stability of the conclusions over", n, "settings ===")
for sm, g in out.groupby("smell"):
    tag = "  [detector too weak: reference only]" if g.abstained.iloc[0] else ""
    print(f"{sm}{tag}: XSmell+ >= rule-only in {(g.full_minus_rule >= 0).sum()}/{n} settings, "
          f">= SHAP-only in {(g.full_minus_shap >= 0).sum()}/{n}; best by simulated impact: "
          f"{g.best_by_sim_dp.value_counts().to_dict()}; coverage {g.coverage_full.min():.2f}-{g.coverage_full.max():.2f}")
lam = a[(a.rho == a.rho.median()) & (a.config == "xsmell_full")].pivot_table(index="smell", columns="lam",
        values=["sim_dp_top1", "top1_agreement_across_seeds"])
print("\n=== Effect of lambda on XSmell+ (rho fixed): lower lambda = more weight on simulated impact ===")
print(lam.round(3).to_string())
print("\nSaved results/recommender_sensitivity.csv")
