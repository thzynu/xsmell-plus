"""Chẩn đoán vì sao mẫu không ghép được với CK.
Usage: python 04_diagnose_unmatched.py
Ghi data/unmatched_class.csv và in thống kê lý do + 15 ví dụ."""
import os, importlib.util
import pandas as pd
from step_utils import tag

spec = importlib.util.spec_from_file_location("b", "03_build_dataset.py")
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)

lab = pd.read_csv("data/labels.csv")
lab = lab[lab["level"] == "class"].copy()
lab["_tag"] = [tag(u, s) for u, s in zip(lab["repo_url"], lab["commit_hash"])]

rows = []
for t, g in lab.groupby("_tag", sort=False):
    d = os.path.join("data/ck_raw", t)
    logrow = None
    if not os.path.isdir(d):
        for _, r in g.iterrows():
            rows.append(dict(reason="no_ck_folder", tag=t, smell=r["smell"], path=r["path"], code_name=r["code_name"], ck_classes=""))
        continue
    ck = b.load_ck(d, "class")
    if ck is None:
        for _, r in g.iterrows():
            rows.append(dict(reason="ck_class_csv_missing_or_empty", tag=t, smell=r["smell"], path=r["path"], code_name=r["code_name"], ck_classes=""))
        continue
    by = {k: v for k, v in ck.groupby("_file")}
    memo = {}
    for _, r in g.iterrows():
        c = b.candidates(by, r["path"], memo)
        if c is None:
            reason, names = "file_not_in_ck", ""
        elif b.match_class(c, r) is None:
            reason, names = "class_name_mismatch", "; ".join(c["class"].astype(str).head(8))
        else:
            continue
        rows.append(dict(reason=reason, tag=t, smell=r["smell"], path=r["path"], code_name=r["code_name"], ck_classes=names))

df = pd.DataFrame(rows)
df.to_csv("data/unmatched_class.csv", index=False)
print("Lý do không ghép được (mẫu cấp class):")
print(df["reason"].value_counts().to_string())
print("\n15 ví dụ class_name_mismatch (code_name của nhãn  vs  tên lớp CK trong cùng file):")
ex = df[df["reason"] == "class_name_mismatch"].sample(min(15, (df["reason"] == "class_name_mismatch").sum()), random_state=0)
for _, r in ex.iterrows():
    print(f"  nhãn: {r['code_name']!r:55}  CK: {r['ck_classes'][:110]}")
print("\n10 ví dụ file_not_in_ck (path của nhãn):")
ex = df[df["reason"] == "file_not_in_ck"]
for _, r in ex.sample(min(10, len(ex)), random_state=0).iterrows():
    print("  ", r["path"])
