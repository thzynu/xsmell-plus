"""Step 8 - XSmell+ : SHAP-guided, constraint-aware, quality-aware refactoring recommendation (Contribution 2).

Usage:
    python 08_recommender.py                    # all smells, seeds 0-4, rho = 0.5
    python 08_recommender.py --seeds 5 --rho 0.5 --lam 0.5
    python 08_recommender.py --sensitivity      # also repeat for rho in {0.3, 0.5, 0.7} and lambda in {0.25, 0.5, 0.75}

For every sample that the detector flags as smelly (out-of-fold P >= tau) the algorithm
  1. aggregates SHAP values into semantic metric groups and keeps only the evidence that points in the
     direction a refactoring can act on (e.g. LCOM high = low cohesion, TCC low = low cohesion);
  2. checks structural PRECONDITIONS of every candidate refactoring on the CK metrics (feasibility F);
  3. estimates the IMPACT I = p(x) - p(T_r(x)) with a what-if transformation T_r of the metrics
     (a MODEL-BASED estimate under stated assumptions, NOT observed refactoring outcomes);
  4. scores S(r|x) = F * (lam*E + (1-lam)*I) / (1 + kappa*(C_r - 1)), E = SHAP evidence share, C_r = cost;
  5. selects the top-k set without CONFLICTING refactorings (exact search, <= 5 candidates);
  6. ABSTAINS when the detector for the smell is too weak (cross-validated MCC < mcc_min) or P < tau.

Ablation (the three configurations requested by the supervisor + component ablations):
  rule_only      static smell -> refactoring table, no SHAP, no checks
  shap_only      top-1 raw SHAP feature -> fixed feature -> refactoring table (style of the conference paper)
  xsmell_E       groups + direction + feasibility, evidence only (lam = 1)
  xsmell_noF     full score without the feasibility filter
  xsmell_noDir   full score without the direction check
  xsmell_full    full algorithm
Evaluation here is INTERNAL (feasibility, simulated impact, stability, conflicts). Whether a recommendation
really improves code needs the before/after study (step 11).
Outputs: results/recommender_ablation.csv, results/recommendations.csv, results/recommender_examples.txt
"""
import argparse, itertools, os, time
import numpy as np
import pandas as pd
import xs_common as xc

# ----------------------------------------------------------------------------------------------
# 1. Metric groups (disjoint) and the direction in which each metric is "worse"
# ----------------------------------------------------------------------------------------------
GROUPS = {
    "size": ["loc", "uniqueWordsQty", "assignmentsQty", "variablesQty", "stringLiteralsQty", "numbersQty",
             "returnQty", "returnsQty", "parenthesizedExpsQty", "mathOperationsQty", "logStatementsQty",
             "totalMethodsQty", "staticMethodsQty", "publicMethodsQty", "privateMethodsQty",
             "protectedMethodsQty", "defaultMethodsQty", "visibleMethodsQty", "abstractMethodsQty",
             "finalMethodsQty", "synchronizedMethodsQty", "nosi"],
    "complexity": ["wmc", "comparisonsQty", "loopQty", "maxNestedBlocksQty", "tryCatchQty"],
    "coupling": ["cbo", "cboModified", "fanout", "fanin", "rfc", "methodsInvokedQty",
                 "methodsInvokedLocalQty", "methodsInvokedIndirectLocalQty"],
    "cohesion": ["lcom", "lcom*", "tcc", "lcc"],
    "state": ["totalFieldsQty", "staticFieldsQty", "publicFieldsQty", "privateFieldsQty",
              "protectedFieldsQty", "defaultFieldsQty", "finalFieldsQty", "synchronizedFieldsQty"],
    "parameters": ["parametersQty"],
    "structure": ["anonymousClassesQty", "innerClassesQty", "lambdasQty"],
    "hierarchy": ["dit", "noc"],
}
# CK: LCOM / LCOM* higher = LESS cohesive; TCC / LCC higher = MORE cohesive.  +1: high is bad, -1: low is bad.
BAD_DIR = {"tcc": -1, "lcc": -1}

# ----------------------------------------------------------------------------------------------
# 2. Refactoring catalogue per level: preconditions, effect model, cost
#    precondition = list of clauses (AND); a clause = list of (feature, op, kind, value) alternatives (OR)
#    kind "q": value is a quantile of the feature over ALL samples of the level (no labels used); "abs": absolute value
#    effect = list of (target, op, value); target = feature or group; ops: mul, set, add, improve, transfer, add_prop
# ----------------------------------------------------------------------------------------------
R = 0.5   # default rho = fraction of the code affected by the transformation (--rho)
CATALOG = {
    "class": {
        "Extract Class": dict(cost=3,
            pre=[[("totalMethodsQty", ">=", "q", .75)], [("lcom", ">=", "q", .5), ("tcc", "<=", "q", .5)]],
            eff=lambda r: [("size", "mul", 1 - r), ("complexity", "mul", 1 - r), ("state", "mul", 1 - r),
                           ("cohesion", "improve", 1 - r), ("coupling", "mul", 1 - r / 2)]),
        "Move Method": dict(cost=2,
            pre=[[("cbo", ">=", "q", .5), ("fanout", ">=", "q", .5)], [("totalMethodsQty", ">=", "q", .5)]],
            eff=lambda r: [("size", "mul", 1 - r / 2), ("coupling", "mul", 1 - r / 2),
                           ("cohesion", "improve", 1 - r / 2), ("complexity", "mul", 1 - r / 2)]),
        "Replace Conditional with Polymorphism": dict(cost=3,
            pre=[[("comparisonsQty", ">=", "q", .75)], [("wmc", ">=", "q", .75)]],
            eff=lambda r: [("complexity", "mul", 1 - r)]),
        "Encapsulate Field": dict(cost=1,
            pre=[[("publicFieldsQty", ">=", "abs", 1)]],
            eff=lambda r: [("publicFieldsQty->privateFieldsQty", "transfer", 1.0),
                           ("totalMethodsQty", "add_prop:publicFieldsQty", 2.0),
                           ("publicMethodsQty", "add_prop:publicFieldsQty", 2.0),
                           ("wmc", "add_prop:publicFieldsQty", 2.0), ("loc", "add_prop:publicFieldsQty", 6.0)]),
        "Move Behavior to Data Class": dict(cost=2,
            pre=[[("totalMethodsQty", "<=", "q", .5)], [("fanin", ">=", "q", .5)]],
            eff=lambda r: [("totalMethodsQty", "add", 6 * r), ("publicMethodsQty", "add", 6 * r),
                           ("wmc", "add", 6 * r), ("rfc", "add", 6 * r), ("loc", "add", 40 * r)]),
    },
    "method": {
        "Extract Method": dict(cost=1,
            pre=[[("loc", ">=", "q", .75)]],
            eff=lambda r: [("size", "mul", 1 - r), ("complexity", "mul", 1 - r), ("coupling", "mul", 1 - r / 2)]),
        "Decompose Conditional": dict(cost=1,
            pre=[[("comparisonsQty", ">=", "q", .75)], [("maxNestedBlocksQty", ">=", "abs", 2)]],
            eff=lambda r: [("comparisonsQty", "mul", 1 - r), ("maxNestedBlocksQty", "mul", 1 - r),
                           ("wmc", "mul", 1 - r / 2)]),
        "Introduce Parameter Object": dict(cost=1,
            pre=[[("parametersQty", ">=", "abs", 4)]],
            eff=lambda r: [("parametersQty", "set", 1.0)]),
        "Replace Method with Method Object": dict(cost=2,
            pre=[[("variablesQty", ">=", "q", .75)], [("loc", ">=", "q", .75)]],
            eff=lambda r: [("size", "mul", 1 - r), ("complexity", "mul", 1 - r)]),
        "Move Method": dict(cost=2,
            pre=[[("foreign_calls", ">=", "q", .5)], [("foreign_ratio", ">", "abs", 1.0)]],
            eff=lambda r: [("coupling", "mul", 1 - r)]),
    },
}
# (smell, refactoring) -> list of (group, side, weight); side: "high" | "low" | "bad" (uses BAD_DIR)
WEIGHTS = {
    ("blob", "Extract Class"): [("cohesion", "bad", 1.0), ("size", "high", .8), ("state", "high", .5), ("complexity", "high", .5)],
    ("blob", "Move Method"): [("coupling", "high", 1.0), ("size", "high", .6), ("cohesion", "bad", .6)],
    ("blob", "Replace Conditional with Polymorphism"): [("complexity", "high", 1.0)],
    ("data_class", "Encapsulate Field"): [("state", "high", 1.0)],
    ("data_class", "Move Behavior to Data Class"): [("size", "low", .8), ("complexity", "low", .6), ("state", "high", .4)],
    ("long_method", "Extract Method"): [("size", "high", 1.0), ("complexity", "high", .8)],
    ("long_method", "Decompose Conditional"): [("complexity", "high", 1.0)],
    ("long_method", "Introduce Parameter Object"): [("parameters", "high", 1.0)],
    ("long_method", "Replace Method with Method Object"): [("size", "high", .7), ("complexity", "high", .7)],
    ("feature_envy", "Move Method"): [("coupling", "high", 1.0)],
    ("feature_envy", "Extract Method"): [("size", "high", .4), ("coupling", "high", .4)],
}
# refactorings that must not be recommended together (alternative / overlapping solutions to the same problem)
CONFLICTS = {
    "class": [("Extract Class", "Move Method"), ("Extract Class", "Replace Conditional with Polymorphism"),
              ("Encapsulate Field", "Extract Class")],
    "method": [("Extract Method", "Replace Method with Method Object"),
               ("Decompose Conditional", "Replace Method with Method Object"),
               ("Extract Method", "Decompose Conditional")],
}
# baselines
STATIC_RULE = {"blob": "Extract Class", "data_class": "Encapsulate Field",
               "long_method": "Extract Method", "feature_envy": "Move Method"}
# fixed feature -> refactoring table (style of the conference paper: "High LOC -> Extract Method", ...)
FEATURE_RULE = {
    "class": {"loc": "Extract Class", "wmc": "Replace Conditional with Polymorphism", "rfc": "Move Method",
              "cbo": "Move Method", "fanout": "Move Method", "fanin": "Move Method", "lcom": "Extract Class",
              "lcom*": "Extract Class", "comparisonsQty": "Replace Conditional with Polymorphism",
              "totalMethodsQty": "Extract Class", "totalFieldsQty": "Extract Class",
              "publicFieldsQty": "Encapsulate Field", "privateFieldsQty": "Encapsulate Field"},
    "method": {"loc": "Extract Method", "wmc": "Extract Method", "parametersQty": "Introduce Parameter Object",
               "comparisonsQty": "Decompose Conditional", "maxNestedBlocksQty": "Decompose Conditional",
               "cbo": "Move Method", "fanout": "Move Method", "rfc": "Move Method",
               "variablesQty": "Replace Method with Method Object", "methodsInvokedQty": "Move Method"},
}


# ----------------------------------------------------------------------------------------------
def derived(X, feats):
    """dict feature -> column, including derived 'foreign_calls' and 'foreign_ratio' (method level)."""
    cols = {f: X[:, i] for i, f in enumerate(feats)}
    if "methodsInvokedQty" in cols and "methodsInvokedLocalQty" in cols:
        fc = np.maximum(cols["methodsInvokedQty"] - cols["methodsInvokedLocalQty"], 0)
        cols["foreign_calls"] = fc
        cols["foreign_ratio"] = fc / (cols["methodsInvokedLocalQty"] + 1.0)
    return cols


def feasibility(cols, pre, quant):
    n = len(next(iter(cols.values())))
    ok = np.ones(n, dtype=bool)
    for clause in pre:
        any_ok = np.zeros(n, dtype=bool)
        for f, op, kind, val in clause:
            if f not in cols:
                continue
            thr = quant[f][val] if kind == "q" else val
            c = cols[f]
            any_ok |= {">=": c >= thr, "<=": c <= thr, ">": c > thr, "<": c < thr}[op]
        ok &= any_ok
    return ok


def group_index(feats):
    return {g: [feats.index(f) for f in fs if f in feats] for g, fs in GROUPS.items()}


def side_mask(X, med, feats, idx, side):
    """boolean (n, len(idx)): feature value on the side a refactoring can act on."""
    sub = X[:, idx] - med[idx]
    if side == "high":
        return sub > 0
    if side == "low":
        return sub < 0
    sign = np.array([BAD_DIR.get(feats[j], 1) for j in idx])
    return (sub * sign) > 0


def evidence(smell, rname, V, X, med, feats, gidx, use_dir=True):
    """E(r|x) in [0,1]: weighted share of the positive SHAP mass that points to the metric groups r acts on."""
    total = np.maximum(V, 0).sum(1)
    num = np.zeros(len(X))
    for g, side, w in WEIGHTS[(smell, rname)]:
        idx = gidx[g]
        if not idx:
            continue
        phi = np.maximum(V[:, idx], 0)
        if use_dir:
            phi = phi * side_mask(X, med, feats, idx, side)
        num += w * phi.sum(1)
    return np.where(total > 0, num / np.maximum(total, 1e-12), 0.0)


def apply_effect(X, feats, gidx, eff):
    """Vectorised what-if transformation of the metric matrix (assumption-based, see README)."""
    Xm = X.copy()
    col = {f: i for i, f in enumerate(feats)}

    def targets(t):
        return [col[t]] if t in col else gidx.get(t, [])
    for t, op, v in eff:
        if op == "transfer":
            a, b = t.split("->")
            if a in col and b in col:
                Xm[:, col[b]] += Xm[:, col[a]]; Xm[:, col[a]] = 0
            continue
        if op.startswith("add_prop"):
            src = op.split(":")[1]
            if src in col:
                for j in targets(t):
                    Xm[:, j] += v * X[:, col[src]]
            continue
        for j in targets(t):
            f = feats[j]
            if op == "mul":
                Xm[:, j] = Xm[:, j] * v
            elif op == "set":
                Xm[:, j] = np.minimum(Xm[:, j], v) if v <= 1 else v
            elif op == "add":
                Xm[:, j] = Xm[:, j] + v
            elif op == "improve":                       # move cohesion metrics towards "better"
                if BAD_DIR.get(f, 1) == 1:
                    Xm[:, j] = Xm[:, j] * v
                else:
                    Xm[:, j] = np.minimum(1.0, Xm[:, j] / max(v, 1e-6))
    return Xm


def best_conflict_free(scores, names, conflicts, k):
    """exact max-weight conflict-free subset of size <= k among candidates with score > 0."""
    cand = [i for i, s in enumerate(scores) if s > 0]
    best, best_val = [], 0.0
    cset = {frozenset(c) for c in conflicts}
    for size in range(1, min(k, len(cand)) + 1):
        for comb in itertools.combinations(cand, size):
            if any(frozenset((names[a], names[b])) in cset for a, b in itertools.combinations(comb, 2)):
                continue
            val = sum(scores[i] for i in comb)
            if val > best_val:
                best, best_val = list(comb), val
    return sorted(best, key=lambda i: -scores[i])


def run_smell(res, rho, lam, kappa, tau, topk, mcc_min):
    smell, level = res["smell"], xc.TASKS[res["smell"]]
    X, V, P, feats = res["X"], res["V"], res["P"], res["feats"]
    cat = CATALOG[level]
    names = list(cat)
    cols = derived(X, feats)
    quant = {f: {q: float(np.nanquantile(c, q)) for q in (.25, .5, .75)} for f, c in cols.items()}
    med = np.nanmedian(X, axis=0)
    gidx = group_index(feats)
    n = len(X)
    cand_mask = np.array([(smell, r) in WEIGHTS for r in names])

    F = np.zeros((n, len(names))); I = np.zeros((n, len(names)))
    E = np.zeros((n, len(names))); Edir0 = np.zeros((n, len(names)))
    for j, r in enumerate(names):
        F[:, j] = feasibility(cols, cat[r]["pre"], quant)
        Xm = apply_effect(X, feats, gidx, cat[r]["eff"](rho))
        I[:, j] = np.maximum(0, P - xc.repredict(res, Xm))
        if (smell, r) in WEIGHTS:
            E[:, j] = evidence(smell, r, V, X, med, feats, gidx, use_dir=True)
            Edir0[:, j] = evidence(smell, r, V, X, med, feats, gidx, use_dir=False)
    cost = np.array([cat[r]["cost"] for r in names], dtype=float)
    pen = 1 + kappa * (cost - 1)
    flagged = P >= tau
    abstain_detector = res["cfg"]["mcc"] < mcc_min

    def score(lmbd, use_F=True, use_dir=True):
        base = lmbd * (E if use_dir else Edir0) + (1 - lmbd) * I
        s = base / pen
        if use_F:
            s = s * F
        s[:, ~cand_mask] = -1                       # not a candidate for this smell
        return s

    S = {"xsmell_full": score(lam), "xsmell_E": score(1.0), "xsmell_noF": score(lam, use_F=False),
         "xsmell_noDir": score(lam, use_dir=False)}
    top1 = {k: np.where(flagged & (v.max(1) > 0), v.argmax(1), -1) for k, v in S.items()}

    # baselines
    top1["rule_only"] = np.where(flagged, names.index(STATIC_RULE[smell]), -1)
    shap_top = np.full(n, -1)
    order = np.argsort(-V, axis=1)
    for i in np.where(flagged)[0]:
        for j in order[i]:
            if V[i, j] <= 0:
                break
            r = FEATURE_RULE[level].get(feats[j])
            if r:
                shap_top[i] = names.index(r); break
        if shap_top[i] < 0:
            shap_top[i] = names.index(STATIC_RULE[smell])
    top1["shap_only"] = shap_top

    # conflict-free top-k set for the full algorithm; naive top-k (no conflict handling) for comparison
    sets_full, naive_conflict = [], np.full(n, np.nan)
    cset = {frozenset(c) for c in CONFLICTS[level]}
    for i in range(n):
        if not flagged[i]:
            sets_full.append([]); continue
        sets_full.append(best_conflict_free(list(S["xsmell_full"][i]), names, CONFLICTS[level], topk))
        pos = [j for j in np.argsort(-S["xsmell_full"][i])[:topk] if S["xsmell_full"][i, j] > 0]
        naive_conflict[i] = float(any(frozenset((names[a], names[b])) in cset
                                      for a, b in itertools.combinations(pos, 2))) if len(pos) > 1 else 0.0
    return dict(names=names, F=F, I=I, E=E, S=S, top1=top1, flagged=flagged, sets=sets_full,
                naive_conflict=naive_conflict, abstain_detector=abstain_detector, cand_mask=cand_mask)


def explain_text(res, out, i, rho):
    smell, feats, X, V = res["smell"], res["feats"], res["X"], res["V"]
    gidx = group_index(feats)
    med = np.nanmedian(X, axis=0)
    names = out["names"]
    lines = [f"[{smell}] {res['df'].iloc[i]['code_name']}  (detector P={res['P'][i]:.2f}, "
             f"true label={'smelly' if res['y'][i] else 'clean'})"]
    ev = []
    for g, idx in gidx.items():
        if idx:
            mass = np.maximum(V[i, idx], 0).sum()
            if mass > 0:
                top = idx[int(np.argmax(V[i, idx]))]
                ev.append((mass, g, feats[top], X[i, top], med[top]))
    for mass, g, f, v, m in sorted(ev, reverse=True)[:3]:
        lines.append(f"  evidence: {g} (SHAP +{mass:.3f}); strongest metric {f}={v:g} vs median {m:g}")
    for rank, j in enumerate(out["sets"][i], 1):
        lines.append(f"  #{rank} {names[j]}: score={out['S']['xsmell_full'][i, j]:.3f} "
                     f"(evidence={out['E'][i, j]:.2f}, feasible, simulated dP=-{out['I'][i, j]:.2f} "
                     f"[model-based what-if, rho={rho}])")
    rejected = [names[j] for j in range(len(names))
                if out["cand_mask"][j] and out["F"][i, j] == 0]
    if rejected:
        lines.append("  rejected (preconditions not met): " + ", ".join(rejected))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--kappa", type=float, default=0.25)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--mcc-min", type=float, default=0.30)
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--smells", default="blob,data_class,long_method,feature_envy")
    a = ap.parse_args()
    os.makedirs("results", exist_ok=True)

    settings = [(a.rho, a.lam)]
    if a.sensitivity:
        settings = [(r, a.lam) for r in (0.3, 0.5, 0.7)] + [(a.rho, l) for l in (0.25, 0.75)]
    abl_rows, rec_rows, examples = [], [], []
    for smell in a.smells.split(","):
        outs = {}
        for seed in range(a.seeds):
            t0 = time.time()
            res = xc.oof_explain(smell, seed)
            for rho, lam in settings:
                o = run_smell(res, rho, lam, a.kappa, a.tau, a.topk, a.mcc_min)
                outs[(seed, rho, lam)] = (res, o)
            print(f"{smell} seed {seed}: {int(o['flagged'].sum())} flagged samples "
                  f"({time.time() - t0:.0f}s)", flush=True)
        for rho, lam in settings:
            # stability of top-1 across seeds on samples flagged in all seeds
            tops = {c: np.array([outs[(s, rho, lam)][1]["top1"][c] for s in range(a.seeds)])
                    for c in outs[(0, rho, lam)][1]["top1"]}
            all_flag = np.all([outs[(s, rho, lam)][1]["flagged"] for s in range(a.seeds)], axis=0)
            for cfg in tops:
                res, o = outs[(0, rho, lam)]
                t1 = o["top1"][cfg]
                m = o["flagged"] & (t1 >= 0)
                feas = o["F"][np.arange(len(t1))[m], t1[m]]
                dp = o["I"][np.arange(len(t1))[m], t1[m]]
                mf, lo_f, hi_f = xc.bootstrap_ci(feas)
                md, lo_d, hi_d = xc.bootstrap_ci(dp)
                if a.seeds > 1 and all_flag.any():
                    ag = np.mean([np.mean(tops[cfg][s1][all_flag] == tops[cfg][s2][all_flag])
                                  for s1, s2 in itertools.combinations(range(a.seeds), 2)])
                else:
                    ag = np.nan
                nc = o["naive_conflict"][o["flagged"]]
                abl_rows.append(dict(
                    smell=smell, config=cfg, rho=rho, lam=lam, n_flagged=int(o["flagged"].sum()),
                    coverage=round(float(m.sum() / max(1, o["flagged"].sum())), 3),
                    feasible_top1=round(mf, 3), feasible_ci=f"[{lo_f:.3f},{hi_f:.3f}]",
                    sim_dp_top1=round(md, 3), sim_dp_ci=f"[{lo_d:.3f},{hi_d:.3f}]",
                    top1_agreement_across_seeds=round(float(ag), 3) if not np.isnan(ag) else np.nan,
                    conflict_rate_naive_top3=round(float(np.nanmean(nc)), 3) if cfg == "xsmell_full" and len(nc) else np.nan,
                    detector_mcc=round(res["cfg"]["mcc"], 3), abstained=bool(o["abstain_detector"])))
        # per-sample output for the main setting
        res, o = outs[(0, a.rho, a.lam)]
        for i in np.where(o["flagged"])[0]:
            rec_rows.append(dict(smell=smell, sample_id=res["df"].iloc[i]["sample_id"],
                                 code_name=res["df"].iloc[i]["code_name"], p_smelly=round(float(res["P"][i]), 3),
                                 true_label=int(res["y"][i]), abstained=bool(o["abstain_detector"]),
                                 recommended=" | ".join(o["names"][j] for j in o["sets"][i]),
                                 shap_only_top1=o["names"][o["top1"]["shap_only"][i]],
                                 rule_only_top1=o["names"][o["top1"]["rule_only"][i]]))
        tp = [i for i in np.where(o["flagged"] & (res["y"] == 1))[0] if o["sets"][i]]
        for i in sorted(tp, key=lambda i: -res["P"][i])[:3]:
            examples.append(explain_text(res, o, i, a.rho))
    pd.DataFrame(abl_rows).to_csv("results/recommender_ablation.csv", index=False)
    pd.DataFrame(rec_rows).to_csv("results/recommendations.csv", index=False)
    with open("results/recommender_examples.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(examples))
    main_rows = pd.DataFrame(abl_rows)
    main_rows = main_rows[(main_rows.rho == a.rho) & (main_rows.lam == a.lam)]
    print("\n=== Ablation (internal metrics, see header for caveats) ===")
    print(main_rows[["smell", "config", "n_flagged", "coverage", "feasible_top1", "sim_dp_top1",
                     "top1_agreement_across_seeds", "conflict_rate_naive_top3", "abstained"]].to_string(index=False))
    print("\nSaved results/recommender_ablation.csv, recommendations.csv, recommender_examples.txt")


if __name__ == "__main__":
    main()
