"""Step 14 - Paired comparison table (XSmell+ action vs static-rule action on the SAME code).

Reads results/before_after.csv and results/mi_touched.csv; writes results/pairs_table.csv and prints it.
Pairs are given as  label=case_xs,case_rule  (default: the two completed pairs).
Usage: python 14_pair_table.py [--pairs blob_03=blob_03,blob_03_rule blob_12=blob_12,blob_12_rule]
"""
import argparse, os
import pandas as pd

METRICS = [("wmc", "class WMC"), ("loc", "class LOC"), ("rfc", "class RFC"), ("cbo", "class CBO"),
           ("touched_loc", "touched LOC"), ("touched_wmc", "touched WMC"), ("touched_cbo", "touched CBO"),
           ("touched_rfc", "touched RFC"), ("p_blob", "P(Blob) detector")]


def fmt(b, a, nd=0):
    try:
        b, a = float(b), float(a)
    except (TypeError, ValueError):
        return "n/a"
    return f"{b:.{nd}f} -> {a:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ba", default="results/before_after.csv"); ap.add_argument("--mi", default="results/mi_touched.csv")
    ap.add_argument("--out", default="results/pairs_table.csv")
    ap.add_argument("--pairs", nargs="*", default=["blob_03=blob_03,blob_03_rule", "blob_12=blob_12,blob_12_rule"])
    a = ap.parse_args()
    ba = pd.read_csv(a.ba).set_index("case_id")
    mi = pd.read_csv(a.mi).set_index("case_id") if os.path.exists(a.mi) else pd.DataFrame()
    rows = []
    for spec in a.pairs:
        label, ids = spec.split("=")
        for role, cid in zip(("XSmell+", "rule"), ids.split(",")):
            if cid not in ba.index:
                print("missing in before_after.csv:", cid); continue
            r = ba.loc[cid]
            row = {"pair": label, "role": role, "case_id": cid, "action": r.get("refactoring", "")}
            for m, name in METRICS:
                nd = 3 if m == "p_blob" else 0
                row[name] = fmt(r.get(m + "_before"), r.get(m + "_after"), nd)
            row["MI touched"] = (fmt(mi.loc[cid, "mi_touched_before"], mi.loc[cid, "mi_touched_after"], 2)
                                 if cid in mi.index else "n/a")
            row["verdict class"] = r.get("verdict", ""); row["verdict touched"] = r.get("verdict_touched", "")
            row["tests before/after (1 = pass)"] = f'{r.get("tests_before", "")}/{r.get("tests_after", "")}'
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False, lineterminator="\n")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(out.T.to_string(header=False))
    print("written", a.out)


if __name__ == "__main__":
    main()
