"""Step 16 - Assumed effect (Table 4, rho = 0.5) versus the effect observed after the real refactorings.

Reads results/before_after.csv (element level for the target class/method) and, if available, results/recommendations.csv
(the simulated impact I_r of the same instance). Writes results/assumed_vs_observed.csv and prints a summary.
  * multiplicative assumptions of Table 4 are compared with observed ratios after/before
    (Extract Class: size, complexity x(1-rho), coupling x(1-rho/2); Move Method: size, coupling, complexity x(1-rho/2);
     Extract Method: size, complexity x(1-rho), coupling x(1-rho/2));
  * additive assumptions with observed differences (Move Behavior to Data Class: WMC/RFC +6 rho, LOC +40 rho).
  * for Blob cases the simulated impact I_r (drop of P(smell) predicted by the out-of-fold model) is set against the
    observed drop of the detector probability after the real change. Both are model outputs, not software quality.
Usage: python 16_assumed_vs_observed.py [--rho 0.5]
"""
import argparse
import numpy as np, pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("--rho", type=float, default=0.5)
ap.add_argument("--ba", default="results/before_after.csv"); ap.add_argument("--rec", default="results/recommendations.csv")
ap.add_argument("--out", default="results/assumed_vs_observed.csv"); a = ap.parse_args()
r = a.rho
MUL = {  # refactoring (level) -> {metric: assumed factor}
    ("Extract Class", "class"): {"wmc": 1 - r, "loc": 1 - r, "cbo": 1 - r / 2},
    ("Move Method", "class"): {"wmc": 1 - r / 2, "loc": 1 - r / 2, "cbo": 1 - r / 2},
    ("Extract Method", "method"): {"wmc": 1 - r, "loc": 1 - r, "cbo": 1 - r / 2},
}
ADD = {("Move Behavior to Data Class", "class"): {"wmc": 6 * r, "rfc": 6 * r, "loc": 40 * r}}
ACTION_COL = {"Extract Class": "I_extract_class", "Move Method": "I_move_method"}

ba = pd.read_csv(a.ba)
rows = []
for _, c in ba.iterrows():
    key = (c["refactoring"], c["level"])
    if key in MUL:
        for m, f in MUL[key].items():
            b, af = c.get(m + "_before"), c.get(m + "_after")
            if pd.notna(b) and pd.notna(af) and b > 0:
                rows.append(dict(case_id=c["case_id"], refactoring=c["refactoring"], metric=m, kind="ratio",
                                 before=b, after=af, assumed=round(f, 3), observed=round(af / b, 3)))
    if key in ADD:
        for m, d in ADD[key].items():
            b, af = c.get(m + "_before"), c.get(m + "_after")
            if pd.notna(b) and pd.notna(af):
                rows.append(dict(case_id=c["case_id"], refactoring=c["refactoring"], metric=m, kind="difference",
                                 before=b, after=af, assumed=round(d, 3), observed=round(af - b, 3)))
res = pd.DataFrame(rows)

# simulated versus observed change of the detector probability (Blob cases)
dp = []
try:
    rec = pd.read_csv(a.rec).sort_values("p_smelly", ascending=False).drop_duplicates(["smell", "code_name"])
    for _, c in ba.iterrows():
        col = ACTION_COL.get(c["refactoring"])
        if col is None or col not in rec.columns or "p_blob_before" not in ba.columns or pd.isna(c.get("p_blob_before")):
            continue
        m = rec[(rec.smell == "blob") & (rec.code_name == c["code_name"])]
        if m.empty:
            continue
        dp.append(dict(case_id=c["case_id"], refactoring=c["refactoring"], simulated_dP=round(float(m.iloc[0][col]), 3),
                       observed_dP=round(float(c["p_blob_before"] - c["p_blob_after"]), 3)))
except FileNotFoundError:
    pass
dp = pd.DataFrame(dp)

pd.set_option("display.width", 220)
if not res.empty:
    s = (res.groupby(["refactoring", "metric", "kind"])
            .agg(n=("case_id", "count"), assumed=("assumed", "first"), observed_min=("observed", "min"),
                 observed_median=("observed", "median"), observed_max=("observed", "max")).round(3).reset_index())
    print("ASSUMED (Table 4) vs OBSERVED at the target element (ratio after/before, or difference after-before)\n")
    print(s.to_string(index=False))
    res.to_csv(a.out, index=False, lineterminator="\n")
if not dp.empty:
    print("\nSIMULATED vs OBSERVED drop of P(Blob) (model outputs, not code quality)\n")
    print(dp.to_string(index=False))
    dp.to_csv(a.out.replace(".csv", "_dP.csv"), index=False, lineterminator="\n")
print("\nwritten", a.out)
