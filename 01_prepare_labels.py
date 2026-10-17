"""Step 1 - Aggregate MLCQ reviews into one label per code sample.

Usage:
    python 01_prepare_labels.py --csv MLCQCodeSmellSamples.csv --out data/labels.csv

Design choice (state it in the paper, Section 4.2):
  * MLCQ has several reviewers per sample, each giving a severity in
    {none, minor, major, critical}.
  * A sample is POSITIVE if more than half of its reviewers gave severity != none,
    NEGATIVE if fewer than half did. Exact ties are dropped as ambiguous.
  * Each (sample, smell) pair is one detection instance. We train one binary
    detector per smell (Blob / Data Class at class level, Feature Envy / Long
    Method at method level) instead of a 16-class problem.
"""
import argparse, csv, os
import pandas as pd

SEV = {"none": 0, "minor": 1, "major": 2, "critical": 3}
REQUIRED = ["sample_id", "smell", "severity", "code_name", "repository",
            "commit_hash", "path", "start_line", "end_line"]


def read_mlcq(path):
    with open(path, newline="", encoding="utf-8") as f:
        head = f.readline()
    sep = ";" if head.count(";") > head.count(",") else ","
    df = pd.read_csv(path, sep=sep)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns {missing}. Columns found: {list(df.columns)}\n"
                         "Rename them in read_mlcq() to match your file.")
    return df


def repo_url(r):
    """git@github.com:owner/repo.git | https://github.com/owner/repo(.git) | owner/repo -> https URL"""
    r = str(r).strip().rstrip("/")
    if r.endswith(".git"):
        r = r[:-4]
    if r.startswith("git@github.com:"):
        r = "https://github.com/" + r.split(":", 1)[1]
    return r if r.startswith("http") else f"https://github.com/{r}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="data/labels.csv")
    a = ap.parse_args()

    df = read_mlcq(a.csv)
    df["smell"] = df["smell"].str.strip().str.lower().str.replace(" ", "_")
    df["sev_num"] = df["severity"].str.strip().str.lower().map(SEV)
    df = df.dropna(subset=["sev_num"])
    df["smelly"] = (df["sev_num"] > 0).astype(int)
    n_raw = len(df)

    keys = ["sample_id", "smell"]
    g = df.groupby(keys)
    agg = g.agg(n_reviewers=("smelly", "size"), frac_smelly=("smelly", "mean"),
                severity_median=("sev_num", "median")).reset_index()
    meta = df.drop_duplicates(keys)[keys + ["code_name", "repository", "commit_hash",
                                            "path", "start_line", "end_line"]]
    out = agg.merge(meta, on=keys)
    out["repo_url"] = out["repository"].map(repo_url)
    n_all = len(out)
    out = out[out["frac_smelly"] != 0.5].copy()          # drop ties
    out["label"] = (out["frac_smelly"] > 0.5).astype(int)
    out["org"] = out["repo_url"].str.extract(r"github\.com/([^/]+)/")[0]
    out["level"] = out["smell"].map({"blob": "class", "data_class": "class",
                                     "feature_envy": "method", "long_method": "method"})
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out.to_csv(a.out, index=False)

    print(f"reviews: {n_raw} | samples (sample x smell): {n_all} | dropped ties: {n_all-len(out)}")
    print(f"repositories: {out.repo_url.nunique()} | (repo, commit) pairs: "
          f"{out.groupby(['repo_url','commit_hash']).ngroups}")
    print("\nSamples per smell x label (0 = clean, 1 = smelly):")
    print(pd.crosstab(out["smell"], out["label"]).to_string())
    print(f"\nwritten: {a.out}")


if __name__ == "__main__":
    main()
