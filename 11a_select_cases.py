"""Step 11a - Pick candidate samples for the refactoring case study (approach B) from the recommender output.

Usage: python 11a_select_cases.py [--per-smell 20] [--max-class-loc 500] [--max-method-loc 150] [--min-p 0.6] [--max-per-repo 2]
Needs: results/recommendations.csv (step 08), data/labels.csv, data/dataset_class.csv, data/dataset_method.csv
Output: cases_todo.csv  (one row per candidate; fill the empty columns by hand, then run 11_before_after.py --cases cases_todo.csv)

Selection rules (all stated so they can be reported): the detector flagged the sample, reviewers also labelled it smelly,
the recommender did NOT abstain and produced a recommendation, p(smelly) >= --min-p, the element is small enough to
refactor by hand, at most --max-per-repo samples per repository (diversity). Feature Envy is excluded (detector too weak).
Within a smell the candidates are ranked by p(smelly) then by size; a random shuffle with a fixed seed breaks ties
so the choice is not hand-picked. Candidates are a pool: a case that cannot be built or refactored is kept in the
table with status = skipped and a reason (report these; do not silently drop them).
"""
import argparse
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--per-smell", type=int, default=20)
ap.add_argument("--max-class-loc", type=int, default=500)
ap.add_argument("--max-method-loc", type=int, default=150)
ap.add_argument("--min-p", type=float, default=0.6)
ap.add_argument("--max-per-repo", type=int, default=2)
ap.add_argument("--out", default="cases_todo.csv")
ap.add_argument("--force", action="store_true", help="overwrite the output file (WARNING: erases what you filled in)")
a = ap.parse_args()

import os, sys
for f in ["results/recommendations.csv", "data/labels.csv", "data/dataset_class.csv", "data/dataset_method.csv"]:
    if not os.path.exists(f):
        sys.exit(f"Missing {f}." + ("  Run step 08 first:  python 08_recommender.py  (or python run_all.py --skip-05)" if "recommendations" in f else ""))
if os.path.exists(a.out) and not a.force:
    sys.exit(f"{a.out} already exists and may contain your work. Not overwriting. Use --out other.csv, or --force to replace it.")

rec = pd.read_csv("results/recommendations.csv")
lab = pd.read_csv("data/labels.csv")
size = pd.concat([pd.read_csv("data/dataset_class.csv", usecols=["sample_id", "smell", "loc", "wmc"]),
                  pd.read_csv("data/dataset_method.csv", usecols=["sample_id", "smell", "loc", "wmc"])])
keep = lab[["sample_id", "smell", "repo_url", "commit_hash", "path", "start_line", "end_line", "org", "level"]]

d = rec[(~rec.abstained.astype(bool)) & (rec.true_label == 1) & (rec.p_smelly >= a.min_p)
        & rec.recommended.notna() & (rec.recommended.astype(str).str.strip() != "")
        & rec.smell.isin(["blob", "data_class", "long_method"])]
d = d.merge(keep, on=["sample_id", "smell"], how="left").merge(size, on=["sample_id", "smell"], how="left")
d = d[d.repo_url.notna() & d["loc"].notna()]
d = d[((d.level == "class") & (d["loc"] <= a.max_class_loc)) | ((d.level == "method") & (d["loc"] <= a.max_method_loc))]

rng = np.random.RandomState(0)
d = d.assign(_tie=rng.rand(len(d))).sort_values(["smell", "p_smelly", "_tie"], ascending=[True, False, True])
picked = []
for smell, g in d.groupby("smell"):
    per_repo, rows = {}, []
    for r in g.itertuples():
        if per_repo.get(r.repo_url, 0) >= a.max_per_repo:
            continue
        per_repo[r.repo_url] = per_repo.get(r.repo_url, 0) + 1
        rows.append(r.Index)
        if len(rows) >= a.per_smell:
            break
    picked.append(d.loc[rows])
out = pd.concat(picked)

out["case_id"] = [f"{s}_{i+1:02d}" for s, i in zip(out.smell, out.groupby("smell").cumcount())]
out["commit_before"] = out["commit_hash"]
def _mname(cn):                       # "pkg.Cls#name p1 p2" or "pkg.Cls.name p1" -> "name"
    head = str(cn).split(" ")[0]
    return head.split("#")[-1] if "#" in head else head.split(".")[-1]
out["method_name"] = [(_mname(c) if l == "method" else "") for c, l in zip(out.code_name, out.level)]
out["refactoring"] = out.recommended.astype(str).str.split(" \\| ").str[0]       # change it if you apply something else
out = out.rename(columns={"recommended": "recommended_all", "loc": "loc_before", "wmc": "wmc_before_ck"})
for c in ["repo_url_after", "commit_after", "test_cmd", "status", "notes"]:
    out[c] = ""
cols = ["case_id", "smell", "level", "repo_url", "commit_before", "path", "code_name", "method_name", "start_line", "end_line",
        "refactoring", "recommended_all", "p_smelly", "loc_before", "wmc_before_ck", "org",
        "repo_url_after", "commit_after", "test_cmd", "status", "notes"]
out[cols].to_csv(a.out, index=False)
print(f"{len(out)} candidates written to {a.out}")
print(out.groupby("smell").agg(n=("case_id", "size"), repos=("repo_url", "nunique"), median_loc=("loc_before", "median")).to_string())
print("\nCandidates available per smell before the cap:", d.groupby("smell").size().to_dict())
