"""Shared helpers for steps 08-11 (recommender, fidelity, cost, before/after).

* loads 05_run_experiments.py as a module (same data loading / model factory / splits);
* picks the best configuration per smell from results/per_seed_project.csv (same rule as 06);
* `oof_explain` returns OUT-OF-FOLD predictions and SHAP values under the project-wise split, together
  with the fitted fold models, so that later steps can re-predict modified inputs with the model
  that never saw the sample's project (no leakage). Results are cached in results/cache/.
"""
import importlib.util, os, pickle, sys, warnings
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("exp", os.path.join(_HERE, "05_run_experiments.py"))
exp = importlib.util.module_from_spec(_spec)
sys.modules["exp"] = exp          # lets pickle find exp.SafeSMOTE when caching fitted pipelines
_spec.loader.exec_module(exp)
TASKS = exp.TASKS
CACHE = os.path.join("results", "cache")


def best_configs(path="results/per_fold_nested.csv"):
    """smell -> modal (model, smote) chosen by inner CV across outer folds and seeds."""
    r = pd.read_csv(path)
    out = {}
    for smell, d in r.groupby("smell"):
        c = d.groupby(["model", "smote"]).size().sort_values(ascending=False)
        (model, smote), _ = next(iter(c.items()))
        sub = d[(d.model == model) & (d.smote == smote)]
        out[smell] = {"model": model, "smote": bool(smote),
                      "mcc": float(sub["mcc"].mean()), "macro_f1": float(sub["macro_f1"].mean())}
    return out


def _prep(pipe, A):
    """Apply imputer/scaler of a fitted imblearn pipeline (SMOTE is a training-time step only)."""
    for name, step in pipe.steps[:-1]:
        if name != "smote":
            A = step.transform(A)
    return A


def explain(pipe, model_name, Xtrain_raw, Xeval_raw):
    """SHAP values (class 'smelly') for Xeval_raw using the fitted pipeline."""
    clf = pipe.named_steps["clf"]
    Xtr, Xev = _prep(pipe, Xtrain_raw), _prep(pipe, Xeval_raw)
    ex = shap.LinearExplainer(clf, Xtr) if model_name == "logreg" else shap.TreeExplainer(clf)
    sv = ex(Xev)
    vals = sv.values[:, :, 1] if sv.values.ndim == 3 else sv.values
    base = sv.base_values[:, 1] if sv.base_values.ndim == 2 else sv.base_values
    return vals, base


def oof_explain(smell, seed=0, data="data", folds=5, cache=True):
    """Out-of-fold probabilities + SHAP + fold models for one smell and one seed."""
    os.makedirs(CACHE, exist_ok=True)
    cf = os.path.join(CACHE, f"oof_{smell}_seed{seed}.pkl")
    if cache and os.path.exists(cf):
        with open(cf, "rb") as f:
            return pickle.load(f)
    cfg = best_configs()[smell]
    df, feats, *_ = exp.load(smell, data)
    y = df["label"].astype(int).values
    X = df[feats].values.astype(float)
    g = df["project"].values
    V = np.zeros(X.shape); B = np.zeros(len(y)); P = np.zeros(len(y)); fold_models = []; train_idx = []
    for tr, te in StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed).split(X, y, g):
        pipe = exp.make_model(cfg["model"], seed, cfg["smote"])
        pipe.fit(X[tr], y[tr])
        V[te], B[te] = explain(pipe, cfg["model"], X[tr], X[te])
        P[te] = pipe.predict_proba(X[te])[:, 1]
        fold_models.append((te, pipe))
        train_idx.append(tr)
    out = dict(smell=smell, seed=seed, cfg=cfg, df=df, feats=feats, X=X, y=y, groups=g,
               V=V, B=B, P=P, folds=fold_models, train_idx=train_idx)
    if cache:
        with open(cf, "wb") as f:
            pickle.dump(out, f)
    return out


def repredict(res, Xmod):
    """Predicted P(smelly) for a modified matrix, each row scored by the fold model that held it out."""
    P = np.zeros(len(Xmod))
    for te, pipe in res["folds"]:
        P[te] = pipe.predict_proba(Xmod[te])[:, 1]
    return P


def bootstrap_ci(v, n=1000, seed=0):
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = [rng.choice(v, len(v), replace=True).mean() for _ in range(n)]
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def bootstrap_ci_cluster(v, groups, n=1000, seed=0):
    """Percentile bootstrap that resamples whole groups (repositories), not single instances."""
    v = np.asarray(v, dtype=float); groups = np.asarray(groups)
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    ug, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=v); cnt = np.bincount(inv)
    rng = np.random.default_rng(seed); k = len(ug)
    est = []
    for _ in range(n):
        s = rng.integers(0, k, k)
        est.append(sums[s].sum() / cnt[s].sum())
    return (float(v.mean()), float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5)))
