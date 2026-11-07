"""Step 15 - Automated final checks of the experimental design (Reviewer item 5 of round 2).

  A  project-wise folds: no repository appears in both train and test (outer folds and the inner folds of 05b).
  B  preprocessing is fitted on the TRAINING part only: imputer medians and scaler means equal the training statistics,
     and differ from the statistics of the whole dataset.
  C  positives per outer test fold (5 seeds x 5 folds) - how reliable are the scores of rare smells (Feature Envy)?
  D  label-permutation sanity check: with shuffled labels the whole protocol must give MCC ~ 0.
Writes results/design_checks.txt (also printed).   Usage: python 15_leakage_checks.py [--perm-repeats 3]
"""
import argparse, io, sys, contextlib
import numpy as np, pandas as pd
import xs_common as xc
from sklearn.model_selection import StratifiedGroupKFold

exp = xc.exp
ap = argparse.ArgumentParser(); ap.add_argument("--perm-repeats", type=int, default=3); a = ap.parse_args()
buf = io.StringIO()
def out(*s):
    line = " ".join(str(x) for x in s); print(line); buf.write(line + "\n")
bad = 0
SEEDS, FOLDS = 5, 5

for smell in exp.TASKS:
    df, feats, *_ = exp.load(smell, "data")
    y = df["label"].astype(int).values; X = df[feats].values.astype(float); g = df["project"].values
    orgs = df["org"].values
    out(f"\n=== {smell}: n={len(y)}, positives={y.sum()} ({y.mean():.1%}), repositories={len(set(g))}, features={len(feats)}")

    # A: group disjointness (outer + inner)
    viol = 0; pos = []
    for seed in range(SEEDS):
        for tr, te, name in exp.make_splits("project", X, y, g, orgs, seed, FOLDS, 6):
            viol += len(set(g[tr]) & set(g[te])); pos.append(int(y[te].sum()))
            for itr, ite in StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed).split(X[tr], y[tr], g[tr]):
                viol += len(set(g[tr][itr]) & set(g[tr][ite]))
    out(f"A  repositories shared between train and test (outer+inner, {SEEDS} seeds): {viol}  -> {'OK' if viol == 0 else 'LEAK'}")
    bad += viol > 0

    # B: preprocessing statistics come from the training part only
    tr, te, _ = exp.make_splits("project", X, y, g, orgs, 0, FOLDS, 6)[0]
    pipe = exp.make_model("logreg", 0, True); pipe.fit(X[tr], y[tr])
    med_tr = np.nanmedian(X[tr], axis=0); med_all = np.nanmedian(X, axis=0)
    ok_tr = np.allclose(pipe.named_steps["impute"].statistics_, med_tr, equal_nan=True)
    differs = not np.allclose(med_tr, med_all, equal_nan=True)
    Xi = np.where(np.isnan(X[tr]), med_tr, X[tr])
    ok_sc = np.allclose(pipe.named_steps["scale"].mean_, Xi.mean(0))
    out(f"B  imputer = training medians: {ok_tr}; training medians differ from whole-data medians: {differs}; scaler mean = training mean: {ok_sc}")
    bad += not (ok_tr and ok_sc)

    # C: positives per outer test fold
    p = np.array(pos)
    out(f"C  positives per outer test fold: min={p.min()}, median={int(np.median(p))}, max={p.max()}; folds with 0 positives: {(p == 0).sum()}/{len(p)}; with <5: {(p < 5).sum()}/{len(p)}")

    # D: permutation sanity (labels shuffled across repositories' rows; the protocol is otherwise unchanged)
    mccs = []
    for rep in range(a.perm_repeats):
        rng = np.random.default_rng(100 + rep); yp = rng.permutation(y); fold_m = []
        for tr, te, _ in exp.make_splits("project", X, yp, g, orgs, rep, FOLDS, 6):
            if yp[te].sum() == 0 or yp[tr].sum() < 2: continue
            m = exp.make_model("logreg", rep, True); m.fit(X[tr], yp[tr])
            pr = m.predict_proba(X[te])[:, 1]; fold_m.append(exp.metrics(yp[te], (pr >= 0.5).astype(int), pr)["mcc"])
        mccs.append(np.mean(fold_m))
    out(f"D  MCC with shuffled labels (logreg+SMOTE, {a.perm_repeats} repeats): mean={np.mean(mccs):+.3f} (|MCC|<0.1 expected)  -> {'OK' if abs(np.mean(mccs)) < 0.1 else 'CHECK'}")
    bad += abs(np.mean(mccs)) >= 0.1

out("\nRESULT:", "all checks passed" if bad == 0 else f"{bad} check(s) need attention")
open("results/design_checks.txt", "w", encoding="utf-8").write(buf.getvalue())
