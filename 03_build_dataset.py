"""Step 3 - Join labels with CK metrics -> model-ready tables.

Usage:
    python 03_build_dataset.py --labels data/labels.csv --ck data/ck_raw --out data

Outputs: data/dataset_class.csv (blob, data_class), data/dataset_method.csv
(feature_envy, long_method) and data/join_report.csv (matched vs unmatched).
Column `project` (= repository URL) is the grouping key for project-wise splits.
"""
import argparse, os, re, glob
import pandas as pd
from step_utils import tag


def norm(p):
    p = str(p).replace("\\", "/")
    return re.sub(r"^(\./|/)+", "", p)


def load_ck(d, kind):
    f = glob.glob(os.path.join(d, f"*{kind}.csv"))
    if not f or os.path.getsize(f[0]) == 0:
        return None
    try:
        df = pd.read_csv(f[0])
    except pd.errors.EmptyDataError:
        return None
    if df.empty or "file" not in df.columns:
        return None
    root = ""
    rf = os.path.join(d, "ROOT.txt")
    if os.path.exists(rf):
        root = norm(open(rf).read().strip())
    df["_file"] = df["file"].map(norm).map(
        lambda x: x[len(root):].lstrip("/") if root and x.startswith(root) else x)
    return df


def same_file(ck_file, mlcq_path):
    m = norm(mlcq_path)
    return ck_file == m or ck_file.endswith("/" + m) or m.endswith("/" + ck_file)


def fn_name(code_name):
    s = str(code_name)
    if "#" not in s:
        return None
    return re.split(r"[ (]", s.split("#", 1)[1])[0]



def candidates(ck_by_file, path, memo):
    """Các dòng CK thuộc file trùng với path của mẫu (tra dict, rồi mới quét tên file)."""
    if path in memo:
        return memo[path]
    m = norm(path)
    if m in ck_by_file:
        res = ck_by_file[m]
    else:
        keys = [k for k in ck_by_file if same_file(k, m)]
        res = pd.concat([ck_by_file[k] for k in keys]) if keys else None
    memo[path] = res
    return res


def match_class(c, r):
    # MLCQ viet lop long nhau bang '.', CK bang '$' -> chuan hoa ve '.'
    want = str(r["code_name"]).split("[")[0].split("#")[0].replace("$", ".")
    names = c["class"].astype(str).str.replace("$", ".", regex=False)
    exact = c[names == want]
    if len(exact):
        return exact.iloc[0]
    # nhan khong co package: chap nhan neu khop duoi ".<want>" va chi co dung 1 ung vien
    suf = c[names.str.endswith("." + want)]
    if len(suf) == 1:
        return suf.iloc[0]
    return None


def match_method(c, r):
    c = c[(c["line"] >= r["start_line"] - 2) & (c["line"] <= r["end_line"])]
    if c.empty:
        return None
    fn = fn_name(r["code_name"])
    if fn:
        byname = c[c["method"].astype(str).str.startswith(fn + "/")]
        if len(byname):
            c = byname
    return c.iloc[(c["line"] - r["start_line"]).abs().argsort().iloc[0]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--ck", default="data/ck_raw")
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    lab = pd.read_csv(a.labels)
    lab["_tag"] = [tag(u, s) for u, s in zip(lab["repo_url"], lab["commit_hash"])]
    recs = {"class": [], "method": []}
    stat = []
    groups = list(lab.groupby("_tag", sort=False))
    for gi, (t, g) in enumerate(groups, 1):
        # mỗi repo: nạp CK 1 lần, ghép xong thì giải phóng
        cks = {}
        for lvl in g["level"].unique():
            ck = load_ck(os.path.join(a.ck, t), lvl)
            if ck is not None:
                ck = {k: v for k, v in ck.groupby("_file")}
            cks[lvl] = ck
        memo = {"class": {}, "method": {}}
        for idx, r in g.iterrows():
            lvl = r["level"]
            ck = cks[lvl]
            m = None
            if ck is not None:
                c = candidates(ck, r["path"], memo[lvl])
                if c is not None:
                    m = match_class(c, r) if lvl == "class" else match_method(c, r)
            stat.append({"smell": r["smell"], "matched": m is not None, "ck_missing": ck is None})
            if m is None:
                continue
            feats = {k: v for k, v in m.items()
                     if k not in ("file", "class", "method", "_file", "line", "type")
                     and pd.api.types.is_number(v)}
            recs[lvl].append((idx, {"sample_id": r["sample_id"], "smell": r["smell"], "label": r["label"],
                              "severity_median": r["severity_median"], "project": r["repo_url"],
                              "commit": r["commit_hash"], "path": r["path"],
                              "code_name": r["code_name"], **feats}))
        del cks
        if gi % 20 == 0 or gi == len(groups):
            print(f"[{gi}/{len(groups)}] repos done", flush=True)
    for lvl, rows in recs.items():
        rows.sort(key=lambda x: x[0])
        df = pd.DataFrame([x[1] for x in rows])
        df.to_csv(os.path.join(a.out, f"dataset_{lvl}.csv"), index=False)
        print(f"{lvl}: {len(df)} rows, {df['project'].nunique() if len(df) else 0} projects")
    rep = pd.DataFrame(stat).groupby("smell").agg(total=("matched", "size"), matched=("matched", "sum"),
                                                  ck_missing=("ck_missing", "sum"))
    rep["with_ck"] = rep["total"] - rep["ck_missing"]
    rep["match_rate"] = (rep["matched"] / rep["with_ck"].where(rep["with_ck"] > 0)).round(3)
    rep.to_csv(os.path.join(a.out, "join_report.csv"))
    print(rep.to_string())
    print("match_rate = matched / samples whose repo has CK output (should be > 0.9).")


if __name__ == "__main__":
    main()
