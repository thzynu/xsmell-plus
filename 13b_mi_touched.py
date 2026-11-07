"""Step 13b - Maintainability Index over ALL files touched by a refactoring (before vs after).

13_mi_metrics.py measures the MI of the original class only. For Extract Class that is misleading: moving small,
maintainable methods out of a class lowers the LOC-weighted mean MI of the REMAINING methods, and the new class is
not counted. Here the MI is the LOC-weighted mean over the methods declared directly in the primary class of every
file touched by the patch (patches/<case_id>.patch): the files that exist before the change are measured at
commit_before, the files that exist after the change (including new files) at commit_after.

Read-only: it does not modify cases_todo.csv. Output: results/mi_touched.csv
Usage:  python 13b_mi_touched.py --cases cases_todo.csv [--workdir work_ba] [--patches patches]
"""
import argparse, importlib.util, os, re, sys
import pandas as pd
from javalang import tokenizer as T


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m


def patch_files(patch_text):
    before, after = [], []
    for line in patch_text.splitlines():
        if line.startswith("--- ") and not line.startswith("--- /dev/null"):
            before.append(line[4:].strip().split("\t")[0][2:])
        elif line.startswith("+++ ") and not line.startswith("+++ /dev/null"):
            after.append(line[4:].strip().split("\t")[0][2:])
    keep = lambda L: [p for p in dict.fromkeys(L) if p.endswith(".java")]
    return keep(before), keep(after)


def mi_files(mi, root, paths):
    """(LOC-weighted MI, number of methods, total LOC) over the primary-class methods of the given files."""
    ms = []
    for p in paths:
        fp = os.path.join(root, p)
        if not os.path.exists(fp):
            continue
        src = open(fp, encoding="utf-8", errors="replace").read()
        toks = list(T.tokenize(src))
        cls = os.path.splitext(os.path.basename(p))[0]
        for _name, _par, sl, _ln in mi.class_members(toks, cls):
            v, loc = mi.mi_of(sl)
            ms.append((v, loc))
    w = sum(l for _, l in ms)
    return (sum(v * l for v, l in ms) / w if w else float("nan")), len(ms), w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="cases_todo.csv"); ap.add_argument("--workdir", default="work_ba")
    ap.add_argument("--patches", default="patches"); ap.add_argument("--out", default="results/mi_touched.csv")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    s02 = load("s02", os.path.join(here, "02_fetch_and_run_ck.py"))
    mi = load("mi13", os.path.join(here, "13_mi_metrics.py"))
    df = pd.read_csv(a.cases, dtype=str, keep_default_na=False)
    rows = []
    for _, r in df.iterrows():
        if not r.get("commit_after") or str(r.get("status", "")).lower().startswith("skip"):
            continue
        pf = os.path.join(a.patches, r["case_id"] + ".patch")
        if not os.path.exists(pf):
            print(r["case_id"], "no patch file - skipped"); continue
        b, af = patch_files(open(pf, encoding="utf-8", errors="replace").read())
        try:
            dirs = {}
            for tag, url, sha in (("before", r["repo_url"], r["commit_before"]),
                                  ("after", r.get("repo_url_after") or r["repo_url"], r["commit_after"])):
                dest = os.path.join(a.workdir, s02.tag(url, sha))
                if not os.path.exists(dest):
                    ok, err = s02.fetch(url, sha, dest)
                    if not ok:
                        raise RuntimeError("fetch failed: " + str(err))
                dirs[tag] = dest
            mb, nb, lb = mi_files(mi, dirs["before"], b)
            ma, na, la = mi_files(mi, dirs["after"], af)
            rows.append(dict(case_id=r["case_id"], refactoring=r.get("refactoring", ""), files_before=len(b), files_after=len(af),
                             methods_before=nb, methods_after=na, loc_before=lb, loc_after=la,
                             mi_touched_before=round(mb, 2), mi_touched_after=round(ma, 2), mi_touched_change=round(ma - mb, 2)))
            print(f'{r["case_id"]:14s} files {len(b)}->{len(af)}  methods {nb}->{na}  MI(touched) {mb:.2f} -> {ma:.2f}  ({ma - mb:+.2f})')
        except Exception as e:
            print(r["case_id"], "FAILED:", e)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False, lineterminator="\n")
    print("written", a.out)


if __name__ == "__main__":
    main()
