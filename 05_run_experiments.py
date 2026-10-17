"""Step 5 - Detection experiments (RQ1/RQ2) on the rebuilt MLCQ+CK dataset.

Usage:
    pip install scikit-learn imbalanced-learn xgboost
    python 05_run_experiments.py                 # project-wise split (main result)
    python 05_run_experiments.py --split random  # random split (to quantify the gap)
    python 05_run_experiments.py --split org     # leave-one-organisation-out (strict cross-project)
    python 05_run_experiments.py --dedup         # drop rows with identical features+label first
    python 05_run_experiments.py --seeds 5 --folds 5

Design (follows the supervisor's checklist):
  * one binary task per smell (smelly=1 vs clean=0), as in MLCQ; class-level for blob /
    data_class, method-level for feature_envy / long_method;
  * project-wise split: StratifiedGroupKFold, group = repository, so no project appears
    in both train and test; test folds keep the natural class imbalance;
  * org split (--split org): leave-one-organisation-out for the --top-orgs largest GitHub
    organisations (apache, eclipse, ...): train on all other organisations, test on the held-out
    one. This is the strict cross-project test (MLCQ has ~400 repos but only 29 organisations);
  * SMOTE is applied INSIDE the training fold only (imblearn Pipeline);
  * 5 random seeds, Mean +- SD over seeds (fold results are averaged within a seed first);
  * metrics: Macro-F1, Weighted-F1, Precision/Recall/F1 of the smelly class, MCC, PR-AUC,
    ROC-AUC, plus a confusion matrix summed over the folds of each seed;
  * `severity_median` is NEVER used as a feature (it is derived from the same reviews as
    the label -> leakage).
Outputs go to results/.
"""
import argparse, os, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import (f1_score, precision_score, recall_score, matthews_corrcoef,
                             average_precision_score, roc_auc_score, confusion_matrix)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ID_COLS = {"sample_id", "smell", "label", "severity_median", "project", "commit", "path", "code_name"}
TASKS = {"blob": "class", "data_class": "class", "feature_envy": "method", "long_method": "method"}


class SafeSMOTE(SMOTE):
    """SMOTE whose k_neighbors adapts when the training fold has very few positives."""
    def fit_resample(self, X, y):
        n_min = int(min(np.bincount(y)))
        if n_min < 2:                       # cannot interpolate: return data unchanged
            return X, y
        self.k_neighbors = max(1, min(5, n_min - 1))
        return super().fit_resample(X, y)


def make_model(name, seed, smote):
    if name == "logreg":
        clf = LogisticRegression(max_iter=2000, class_weight=None if smote else "balanced", random_state=seed)
        scale = True
    elif name == "rf":
        clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed,
                                     class_weight=None if smote else "balanced_subsample")
        scale = False
    elif name == "xgb":
        clf = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, eval_metric="logloss", random_state=seed, n_jobs=-1)
        scale = False
    else:
        raise ValueError(name)
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    if smote:
        steps.append(("smote", SafeSMOTE(random_state=seed)))
    steps.append(("clf", clf))
    return Pipeline(steps)


def metrics(y, pred, proba):
    out = {
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y, pred, average="weighted", zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1_smelly": f1_score(y, pred, zero_division=0),
        "mcc": matthews_corrcoef(y, pred),
    }
    if len(np.unique(y)) == 2:
        out["pr_auc"] = average_precision_score(y, proba)
        out["roc_auc"] = roc_auc_score(y, proba)
    else:
        out["pr_auc"] = out["roc_auc"] = np.nan
    return out


def load(smell, data_dir, dedup=False):
    df = pd.read_csv(os.path.join(data_dir, f"dataset_{TASKS[smell]}.csv"))
    df = df[df["smell"] == smell].copy()
    df["org"] = df["project"].str.split("/").str[3]
    feats = [c for c in df.columns if c not in ID_COLS and c != "org" and pd.api.types.is_numeric_dtype(df[c])]
    feats = [c for c in feats if df[c].nunique(dropna=True) > 1]        # drop constant columns
    # leakage guard: no single feature may separate the label (near-)perfectly (AUC ~ 1 or 0)
    yb = df["label"].astype(int).values
    if yb.min() != yb.max():
        for c in feats:
            auc = roc_auc_score(yb, df[c].fillna(df[c].median()).values)
            if auc > 0.995 or auc < 0.005:
                raise RuntimeError(f"feature '{c}' separates the label (AUC={auc:.3f}) - leakage, remove it")
    assert "severity_median" not in feats and "label" not in feats
    n_before = len(df)
    conflicts = int(df.duplicated(subset=feats, keep=False).sum()
                    - df.duplicated(subset=feats + ["label"], keep=False).sum())
    if dedup:
        df = df.drop_duplicates(subset=feats + ["label"]).reset_index(drop=True)
    return df, feats, n_before, conflicts


def make_splits(split, X, y, groups, orgs, seed, folds, top_orgs):
    """Return list of (train_idx, test_idx, name)."""
    if split == "project":
        cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        return [(tr, te, f"fold{i}") for i, (tr, te) in enumerate(cv.split(X, y, groups))]
    if split == "random":
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        return [(tr, te, f"fold{i}") for i, (tr, te) in enumerate(cv.split(X, y))]
    # org: leave-one-organisation-out for the largest organisations (deterministic split)
    top = pd.Series(orgs).value_counts().head(top_orgs).index
    return [(np.where(orgs != o)[0], np.where(orgs == o)[0], str(o)) for o in top]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--split", choices=["project", "random", "org"], default="project")
    ap.add_argument("--top-orgs", type=int, default=6, help="orgs held out one by one (--split org)")
    ap.add_argument("--dedup", action="store_true", help="drop duplicate (features, label) rows")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--models", default="logreg,rf,xgb")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    models = a.models.split(",")

    tag = a.split + ("_dedup" if a.dedup else "")
    rows, fold_rows, cms, audit = [], [], [], []
    for smell in TASKS:
        df, feats, n_before, conflicts = load(smell, a.data, a.dedup)
        y_all = df["label"].astype(int).values
        X_all = df[feats].values
        groups = df["project"].values
        orgs = df["org"].values
        dup = int(df.duplicated(subset=feats + ["label"]).sum())
        audit.append({"smell": smell, "split": tag, "n_before_dedup": n_before, "n": len(df),
                      "n_projects": df["project"].nunique(), "n_orgs": df["org"].nunique(),
                      "n_smelly": int(y_all.sum()), "smelly_rate": round(y_all.mean(), 4),
                      "n_features": len(feats), "duplicate_rows_remaining": dup,
                      "rows_same_features_conflicting_label": conflicts})
        print(f"\n== {smell} [{tag}]: n={len(df)}, projects={df['project'].nunique()}, orgs={df['org'].nunique()}, "
              f"smelly={y_all.sum()} ({y_all.mean():.1%}), features={len(feats)}, dup rows={dup}, "
              f"conflicting-label dups={conflicts}", flush=True)

        for seed in range(a.seeds):
            splits = make_splits(a.split, X_all, y_all, groups, orgs, seed, a.folds, a.top_orgs)
            for mname in models:
                for smote in (False, True):
                    fold_res, tot_cm = [], np.zeros((2, 2), dtype=int)
                    for tr, te, name in splits:
                        if y_all[te].sum() == 0 or y_all[tr].sum() < 2:
                            continue                      # fold without positives: skip
                        m = make_model(mname, seed, smote)
                        m.fit(X_all[tr], y_all[tr])
                        proba = m.predict_proba(X_all[te])[:, 1]
                        pred = (proba >= 0.5).astype(int)
                        mt = metrics(y_all[te], pred, proba)
                        fold_res.append(mt)
                        fold_rows.append(dict(mt, smell=smell, model=mname, smote=smote, seed=seed,
                                              split=tag, fold=name, n_test=len(te), n_pos_test=int(y_all[te].sum())))
                        tot_cm += confusion_matrix(y_all[te], pred, labels=[0, 1])
                    if not fold_res:
                        continue
                    r = pd.DataFrame(fold_res).mean().to_dict()
                    r.update(smell=smell, model=mname, smote=smote, seed=seed, split=tag,
                             n_valid_folds=len(fold_res))
                    rows.append(r)
                    cms.append({"smell": smell, "model": mname, "smote": smote, "seed": seed,
                                "tn": tot_cm[0, 0], "fp": tot_cm[0, 1], "fn": tot_cm[1, 0], "tp": tot_cm[1, 1]})
            print(f"   seed {seed} done", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(a.out, f"per_seed_{tag}.csv"), index=False)
    pd.DataFrame(fold_rows).to_csv(os.path.join(a.out, f"per_fold_{tag}.csv"), index=False)
    pd.DataFrame(cms).to_csv(os.path.join(a.out, f"confusion_{tag}.csv"), index=False)
    pd.DataFrame(audit).to_csv(os.path.join(a.out, f"dataset_audit_{tag}.csv"), index=False)
    if a.split == "project" and not a.dedup:          # keep the file name used by 06_make_figures.py
        pd.DataFrame(audit).to_csv(os.path.join(a.out, "dataset_audit.csv"), index=False)

    mcols = ["macro_f1", "weighted_f1", "precision", "recall", "f1_smelly", "mcc", "pr_auc", "roc_auc"]
    g = res.groupby(["smell", "model", "smote"])[mcols]
    mean, sd = g.mean(), g.std().fillna(0.0)   # sd=0 when only 1 seed
    summ = mean.round(3).astype(str) + " ± " + sd.round(3).astype(str)
    summ.to_csv(os.path.join(a.out, f"summary_{tag}.csv"))
    print("\n=== Mean ± SD over seeds (", tag, "split ) ===")
    print(summ[["macro_f1", "mcc", "pr_auc", "recall"]].to_string())
    print("\nDataset audit:\n", pd.DataFrame(audit).to_string(index=False))
    print(f"\nSaved to {a.out}/ (per_seed_{tag}.csv, per_fold_{tag}.csv, summary_{tag}.csv, confusion_{tag}.csv)")


if __name__ == "__main__":
    main()
