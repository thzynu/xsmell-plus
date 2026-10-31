import numpy as np, pandas as pd
import xs_common as xc
from sklearn.model_selection import StratifiedGroupKFold

exp = xc.exp
SEEDS, FOLDS = 5, 5
CONFIGS = [(m, s) for m in ("logreg", "rf", "xgb") for s in (False, True)]


def select_config(X, y, g, seed, inner_folds=3):
    """Choose (model, smote) by mean Macro-F1 over an inner group CV on TRAINING data only."""
    splits = list(StratifiedGroupKFold(n_splits=inner_folds, shuffle=True, random_state=seed).split(X, y, g))
    scores = {}
    for m, s in CONFIGS:
        f1 = []
        for tr, te in splits:
            if y[te].sum() == 0 or y[tr].sum() < 2:
                continue
            p = exp.make_model(m, seed, s)
            p.fit(X[tr], y[tr])
            proba = p.predict_proba(X[te])[:, 1]
            f1.append(exp.metrics(y[te], (proba >= 0.5).astype(int), proba)["macro_f1"])
        scores[(m, s)] = (float(np.mean(f1)) if f1 else -1.0, len(f1))
    best = max(scores, key=lambda k: scores[k][0])
    return dict(model=best[0], smote=best[1], inner_f1=scores[best][0], n_inner=scores[best][1])


rows, fold_rows = [], []
for smell in exp.TASKS:
    df, feats, *_ = exp.load(smell, "data")
    y = df["label"].astype(int).values
    X = df[feats].values.astype(float)
    g = df["project"].values
    orgs = df["org"].values
    for seed in range(SEEDS):
        res = []
        for tr, te, name in exp.make_splits("project", X, y, g, orgs, seed, FOLDS, 6):
            if y[te].sum() == 0 or y[tr].sum() < 2:
                continue
            cfg = select_config(X[tr], y[tr], g[tr], seed)
            m = exp.make_model(cfg["model"], seed, cfg["smote"])
            m.fit(X[tr], y[tr])
            proba = m.predict_proba(X[te])[:, 1]
            mt = exp.metrics(y[te], (proba >= 0.5).astype(int), proba)
            res.append(mt)
            fold_rows.append(dict(mt, smell=smell, seed=seed, fold=name, model=cfg["model"],
                                  smote=cfg["smote"], inner_f1=cfg["inner_f1"], n_inner=cfg["n_inner"],
                                  n_test=len(te), n_pos_test=int(y[te].sum())))
        if res:
            rows.append(dict(pd.DataFrame(res).mean().to_dict(), smell=smell, seed=seed, n_valid_folds=len(res)))
        print(smell, "seed", seed, "done", flush=True)

pd.DataFrame(rows).to_csv("results/per_seed_nested.csv", index=False)
fr = pd.DataFrame(fold_rows)
fr.to_csv("results/per_fold_nested.csv", index=False)
print(pd.DataFrame(rows).groupby("smell")[["macro_f1", "mcc"]].agg(["mean", "std"]).round(3))
print(fr.groupby(["smell", "model", "smote"]).size())
