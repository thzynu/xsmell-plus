"""Step 9 - Explanation fidelity of SHAP (RQ3): deletion / insertion tests.

For samples the detector flags as smelly (out-of-fold P >= 0.5, project-wise split) we ask:
does removing the features SHAP says matter really reduce the predicted smell probability?

  deletion     (comprehensiveness): replace the top-k features by their population MEDIAN, measure the
               mean drop in P(smelly) and the share of samples whose decision flips to 'clean';
  insertion    (sufficiency): keep ONLY the top-k features, all others replaced by the median, measure the
               fraction of the original probability that is retained;
compared with  (a) the same number of RANDOM features (10 draws) and (b) the top-k GLOBAL features
(mean |SHAP|, identical for every sample).  Fidelity gain = drop(SHAP local) - drop(random).

Caveat (state it in the paper): median replacement creates inputs that may lie off the data manifold.

Usage: python 09_fidelity.py [--seeds 5]      Output: results/fidelity.csv, results/fidelity_summary.csv
"""
import argparse
import numpy as np
import pandas as pd
import xs_common as xc

KS = (1, 2, 3, 5, 8)


def run(res, rng, n_random=10):
    X, V, P, feats = res["X"], res["V"], res["P"], res["feats"]
    flagged = np.where(P >= 0.5)[0]
    if len(flagged) < 10:
        return []
    med = np.nanmedian(X, axis=0)
    glob = np.argsort(-np.abs(V).mean(0))
    order_local = np.argsort(-V, axis=1)                  # most smell-increasing features first
    rows = []
    for k in KS:
        k = min(k, len(feats))
        variants = {}
        Xd = X.copy(); Xi = np.tile(med, (len(X), 1))
        for i in flagged:
            top = order_local[i, :k]
            Xd[i, top] = med[top]; Xi[i, top] = X[i, top]
        variants["shap_local"] = (Xd, Xi)
        Xd = X.copy(); Xi = np.tile(med, (len(X), 1)); Xi[:, glob[:k]] = X[:, glob[:k]]; Xd[:, glob[:k]] = med[glob[:k]]
        variants["global_importance"] = (Xd, Xi)
        drops_r, ins_r, flips_r = [], [], []
        for _ in range(n_random):
            Xd = X.copy(); Xi = np.tile(med, (len(X), 1))
            for i in flagged:
                top = rng.choice(len(feats), k, replace=False)
                Xd[i, top] = med[top]; Xi[i, top] = X[i, top]
            pd_, pi_ = xc.repredict(res, Xd)[flagged], xc.repredict(res, Xi)[flagged]
            drops_r.append((P[flagged] - pd_).mean()); flips_r.append((pd_ < 0.5).mean())
            ins_r.append((pi_ / np.maximum(P[flagged], 1e-9)).mean())
        for name, (Xd, Xi) in variants.items():
            pd_, pi_ = xc.repredict(res, Xd)[flagged], xc.repredict(res, Xi)[flagged]
            rows.append(dict(smell=res["smell"], seed=res["seed"], k=k, method=name, n=len(flagged),
                             mean_drop=(P[flagged] - pd_).mean(), flip_rate=(pd_ < 0.5).mean(),
                             retained_prob=(pi_ / np.maximum(P[flagged], 1e-9)).mean()))
        rows.append(dict(smell=res["smell"], seed=res["seed"], k=k, method="random", n=len(flagged),
                         mean_drop=np.mean(drops_r), flip_rate=np.mean(flips_r), retained_prob=np.mean(ins_r)))
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    rows = []
    for smell in xc.TASKS:
        for seed in range(a.seeds):
            r = run(xc.oof_explain(smell, seed), rng)
            rows += r
            print(f"{smell} seed {seed} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("results/fidelity.csv", index=False)
    g = df.groupby(["smell", "k", "method"])[["mean_drop", "flip_rate", "retained_prob"]].mean().round(3).reset_index()
    piv = g.pivot_table(index=["smell", "k"], columns="method", values="mean_drop")
    piv["gain_vs_random"] = (piv["shap_local"] - piv["random"]).round(3)
    piv.round(3).to_csv("results/fidelity_summary.csv")
    print("\nMean drop in P(smelly) after deleting top-k features (higher = explanation more faithful):")
    print(piv.round(3).to_string())
    aopc = df.groupby(["smell", "method"])["mean_drop"].mean().unstack().round(3)
    print("\nAOPC (mean drop averaged over k):\n", aopc.to_string())
    aopc.to_csv("results/fidelity_aopc.csv")


if __name__ == "__main__":
    main()
