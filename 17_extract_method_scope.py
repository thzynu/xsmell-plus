"""Step 17 - Extract Method evaluated on the original method, the NEW method(s) and the containing class.

For every case with refactoring == "Extract Method" (level method) it compares the CK output of the commit before and
the commit after:
  original method   : the target method before and after (as in Table 18);
  new method(s)     : methods of the same class that exist after the change and not before (by CK method key);
  original + new    : sum of LOC, WMC, CBO, RFC over the original method after the change and the new methods;
  containing class  : the class of the target, WMC, LOC, CBO, RFC, LCOM*, before and after.
Writes results/extract_method_scope.csv.  It needs the CK output of both commits (data/ck_ba/<tag>), which
11_before_after.py has created.     Usage: python 17_extract_method_scope.py --cases cases_todo.csv
"""
import argparse, importlib.util, os, sys
import numpy as np, pandas as pd

M = ["loc", "wmc", "cbo", "rfc"]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod); return mod


def scope(mb, ma, cb, ca, tb, ta):
    """mb/ma: CK method tables before/after; cb/ca: CK class tables; tb/ta: target method rows before/after."""
    cls_b, cls_a = tb["class"], ta["class"]
    kb = set(mb[mb["class"] == cls_b]["method"]); a_rows = ma[ma["class"] == cls_a]
    new = a_rows[~a_rows["method"].isin(kb)]
    new = new[new["method"] != ta["method"]]
    out = {"n_new_methods": len(new)}
    for m in M:
        out[f"orig_{m}_before"] = float(tb[m]); out[f"orig_{m}_after"] = float(ta[m])
        out[f"new_{m}_sum"] = float(new[m].sum()) if len(new) else 0.0
        out[f"new_{m}_max"] = float(new[m].max()) if len(new) else 0.0
        out[f"orig_plus_new_{m}"] = float(ta[m]) + out[f"new_{m}_sum"]
    for tag, ck, cls in (("before", cb, cls_b), ("after", ca, cls_a)):
        r = ck[ck["class"] == cls]
        if r.empty:
            continue
        r = r.iloc[0]
        for m in M + ["lcom*"]:
            if m in r.index:
                out[f"class_{m}_{tag}"] = float(r[m])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="cases_todo.csv"); ap.add_argument("--out-ck", default="data/ck_ba")
    ap.add_argument("--out", default="results/extract_method_scope.csv")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    s02 = _load("s02", os.path.join(here, "02_fetch_and_run_ck.py")); s03 = _load("s03", os.path.join(here, "03_build_dataset.py"))
    ba = _load("ba11", os.path.join(here, "11_before_after.py"))
    df = pd.read_csv(a.cases, dtype=str, keep_default_na=False)
    df = df[(df["refactoring"] == "Extract Method") & (df["commit_after"] != "")]
    rows = []
    for c in df.itertuples():
        try:
            sides = {}
            for tag, url, sha in (("before", c.repo_url, c.commit_before),
                                  ("after", (getattr(c, "repo_url_after", "") or c.repo_url), c.commit_after)):
                out = os.path.join(a.out_ck, s02.tag(url, sha))
                mk, ck = s03.load_ck(out, "method"), s03.load_ck(out, "class")
                t = ba.find_element(mk, "method", c.path, c.code_name, getattr(c, "method_name", None), s03,
                                    start_line=(float(c.start_line) if getattr(c, "start_line", "") not in ("", None) else None))
                if t is None:
                    raise RuntimeError("target not found " + tag)
                sides[tag] = (mk, ck, t)
            rec = dict(case_id=c.case_id, code_name=c.code_name)
            rec.update(scope(sides["before"][0], sides["after"][0], sides["before"][1], sides["after"][1],
                             sides["before"][2], sides["after"][2]))
            rows.append(rec)
            print(f'{c.case_id}: new methods {rec["n_new_methods"]}; original LOC {rec["orig_loc_before"]:.0f} -> {rec["orig_loc_after"]:.0f}, '
                  f'new methods LOC {rec["new_loc_sum"]:.0f}, original+new {rec["orig_plus_new_loc"]:.0f}; '
                  f'WMC original+new {rec["orig_plus_new_wmc"]:.0f} (before {rec["orig_wmc_before"]:.0f}); '
                  f'class WMC {rec.get("class_wmc_before", float("nan")):.0f} -> {rec.get("class_wmc_after", float("nan")):.0f}')
        except Exception as e:
            print(c.case_id, "FAILED:", e)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False, lineterminator="\n")
    print("written", a.out)


if __name__ == "__main__":
    main()
