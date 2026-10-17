"""Step 11 - Refactoring effectiveness: measure code before vs after a real refactoring (Contribution 3).

MLCQ only has ONE snapshot per sample, so effectiveness must be measured on pairs of commits
(before, after) in which a developer really applied a refactoring.  Typical sources:
  * refactoring commits mined with RefactoringMiner (https://github.com/tsantalis/RefactoringMiner),
  * a small case study where the recommended refactoring is applied by hand / with an IDE.

Input CSV (--cases), one row per case:
  case_id, repo_url, commit_before, commit_after, path, level (class|method), code_name,
  method_name (method level only; name as in source), refactoring (label from the recommender or the miner),
  test_cmd (optional; e.g. "mvn -q -o test -Dtest=FooTest" run inside the checkout),
  repo_url_after (optional; your fork when commit_after exists only there)
  (11a_select_cases.py writes cases_todo.csv with these columns; rows without commit_after or with status 'skipped...' are ignored)

For each case the script
  1. fetches both commits, runs CK (same code as step 02);
  2. finds the target class/method before and after and reports the change in WMC (cyclomatic), CBO, RFC, LOC
     and, for classes, LCOM* (lower = better) and TCC (higher = better);
  3. (optional --score-detector) re-scores the element with the trained XSmell+ detector: observed change in
     P(smelly);
  4. (optional --run-tests) runs test_cmd before and after: behaviour preserved = passes in both;
  5. labels the case improved / neutral / degraded and writes results/before_after.csv and a summary.

Not covered (state this in the paper): Maintainability Index and technical debt need external tools
(e.g. SonarQube); 13_mi_metrics.py fills mi_before/mi_after/mi_class_before/mi_class_after (approximate MI),
technical debt (debt_before/debt_after, SonarQube sqale_index in minutes) is typed in by hand; both are included in
the summary automatically.

Usage:
  python 11_before_after.py --cases cases.csv --ck-jar tools/ck.jar [--score-detector] [--run-tests]
  python 11_before_after.py --self-test       # checks the summary logic on synthetic data
"""
import argparse, importlib.util, os, subprocess, sys
import numpy as np
import pandas as pd


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


# metrics for which LOWER is better / HIGHER is better
LOWER = {"wmc": "WMC (cyclomatic)", "cbo": "CBO", "rfc": "RFC", "loc": "LOC", "lcom*": "LCOM*"}
HIGHER = {"tcc": "TCC"}


def verdict(delta, tol=0.10):
    """improved: no metric degrades by more than tol (relative) and at least one improves by more than tol;
    degraded: some metric degrades by more than tol and none improves; mixed otherwise; neutral if all within tol."""
    better = [m for m, (b, a) in delta.items() if _rel(b, a, m) > tol]
    worse = [m for m, (b, a) in delta.items() if _rel(b, a, m) < -tol]
    if better and not worse:
        return "improved"
    if worse and not better:
        return "degraded"
    if better and worse:
        return "mixed"
    return "neutral"


def _rel(b, a, m):
    """relative improvement (positive = better)."""
    if b is None or a is None or np.isnan(b) or np.isnan(a):
        return 0.0
    d = (b - a) if m in LOWER else (a - b)
    return d / max(abs(b), 1.0)


def find_element(ck, level, path, code_name, method_name, step3):
    row = {"path": path, "code_name": code_name, "start_line": 0, "end_line": 10 ** 9}
    if level == "class":
        return step3.match_class(ck, row)
    c = ck[ck["_file"].map(lambda x: step3.same_file(x, path))]
    if method_name:
        c = c[c["method"].astype(str).str.startswith(method_name + "/")]
    return None if c.empty else c.iloc[0]


def ck_for(repo_url, sha, a, s02, tmp):
    tag = s02.tag(repo_url, sha)
    out = os.path.join(a.out_ck, tag)
    repo = os.path.join(a.workdir, tag)
    if not os.path.exists(os.path.join(out, "class.csv")):
        s02.rm(repo)
        ok, err = s02.fetch(repo_url, sha, repo)
        if not ok:
            raise RuntimeError(f"fetch failed: {err}")
        os.makedirs(out, exist_ok=True)
        s02.run(["java", "-jar", a.ck_jar, repo, "true", "0", "false", out + os.sep], timeout=3600)
        open(os.path.join(out, "ROOT.txt"), "w").write(os.path.abspath(repo))
    return out, repo


def tests_pass(repo, cmd, log_path=None):
    """Run the test command inside the checkout. Returns 1.0 (exit code 0), 0.0 (non-zero) or NaN (no command).
    The exit code and the last lines of the output go to log_path so that a failure (wrong Java version,
    missing dependency, real test failure) can be told apart."""
    if not isinstance(cmd, str) or not cmd.strip():
        return np.nan
    try:
        r = subprocess.run(cmd, cwd=repo, shell=True, capture_output=True, text=True, errors="replace", timeout=3600)
        code, tail = r.returncode, "\n".join((r.stdout + "\n" + r.stderr).splitlines()[-60:])
    except subprocess.TimeoutExpired:
        code, tail = -9, "TIMEOUT after 3600 s"
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"cmd: {cmd}\ncwd: {repo}\nexit code: {code}\n\n{tail}\n")
    if code != 0:
        print(f"   tests FAILED (exit {code}); see {log_path}", flush=True)
    return float(code == 0)


def detector_scorer(smell_by_level):
    """Train the best configuration per smell on ALL data (for scoring unseen before/after code)."""
    xc = _load("xs_common", "xs_common.py")
    models = {}
    for smell, cfg in xc.best_configs().items():
        df, feats, *_ = xc.exp.load(smell, "data")
        m = xc.exp.make_model(cfg["model"], 0, cfg["smote"]); m.fit(df[feats].values.astype(float), df["label"].astype(int).values)
        models[smell] = (m, feats)
    return models


def _metric_table(df):
    mets = [c.replace("_before", "") for c in df.columns if c.endswith("_before") and not c.startswith("tests")]
    rows = []
    for m in mets:
        b, a = df[m + "_before"], df[m + "_after"]
        ok = b.notna() & a.notna()
        if ok.sum():
            rows.append(dict(metric=m, n=int(ok.sum()), mean_before=round(b[ok].mean(), 3), mean_after=round(a[ok].mean(), 3),
                             mean_change=round((a[ok] - b[ok]).mean(), 3),
                             share_improved=round(float((_better_series(b[ok], a[ok], m)).mean()), 3)))
    return pd.DataFrame(rows)


def summarize(df):
    """Verdicts and metric changes, reported separately for class-level and method-level cases."""
    if "level" not in df:
        df = df.assign(level="all")
    s = df.groupby(["level", "verdict"]).size().rename("n_cases").reset_index()
    print("\n=== Verdicts by level ===\n", s.to_string(index=False))
    tables = []
    for lvl, g in df.groupby("level"):
        t = _metric_table(g)
        t.insert(0, "level", lvl)
        tables.append(t)
        print(f"\n=== Metric change, {lvl}-level cases (n={len(g)}) ===\n", t.drop(columns="level").to_string(index=False))
    t = pd.concat(tables) if tables else pd.DataFrame()
    if "tests_before" in df and df["tests_after"].notna().any():
        ran = df[df.tests_after.notna() | df.tests_before.notna()]
        both = int(((ran.tests_before == 1) & (ran.tests_after == 1)).sum())
        print(f"\nUnit tests: {len(ran)} case(s) with a test command; pass before: {int((ran.tests_before == 1).sum())}, "
              f"pass after: {int((ran.tests_after == 1).sum())}, pass in both (behaviour preserved): {both}.")
        bad = ran[(ran.tests_before != 1) | (ran.tests_after != 1)]
        for r in bad.itertuples():
            print(f"  NOT usable as evidence: {r.case_id} (before={r.tests_before}, after={r.tests_after}); "
                  f"read results/test_logs/{r.case_id}_*.log (a build/Java problem is not a behaviour change)")
    os.makedirs("results", exist_ok=True)
    t.to_csv("results/before_after_summary.csv", index=False)
    return s, t


def _lower_is_better(m):
    return m in LOWER or m.startswith("p_") or m == "debt"


def _better_series(b, a, m):
    return (b > a) if _lower_is_better(m) else (a > b)


def self_test():
    d = {"wmc": (30, 12), "cbo": (20, 18), "rfc": (50, 31), "loc": (400, 250)}
    assert verdict(d) == "improved", verdict(d)
    assert verdict({"wmc": (10, 25), "loc": (100, 140)}) == "degraded"
    assert verdict({"wmc": (30, 12), "loc": (100, 160)}) == "mixed"
    assert verdict({"wmc": (30, 29), "loc": (100, 101)}) == "neutral"
    assert verdict({"tcc": (0.2, 0.5)}) == "improved"
    df = pd.DataFrame({"case_id": [1, 2], "level": ["method", "class"], "wmc_before": [30, 10], "wmc_after": [12, 25], "verdict": ["improved", "degraded"],
                       "tests_before": [1, 1], "tests_after": [1, 0]})
    summarize(df)
    print("\nself-test OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases"); ap.add_argument("--ck-jar", default="tools/ck.jar")
    ap.add_argument("--workdir", default="work_ba"); ap.add_argument("--out-ck", default="data/ck_ba")
    ap.add_argument("--score-detector", action="store_true"); ap.add_argument("--run-tests", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.cases or not os.path.exists(a.cases):
        sys.exit("Step 11 needs a cases file with real before/after refactoring commits (--cases cases.csv).\n"
                 "See cases_template.csv for the columns. To only check the logic, run:  python 11_before_after.py --self-test")
    s02 = _load("s02", "02_fetch_and_run_ck.py"); s03 = _load("s03", "03_build_dataset.py")
    cases = pd.read_csv(a.cases)
    if "commit_after" in cases.columns:                    # cases_todo.csv from 11a: only rows already refactored
        n0 = len(cases)
        cases = cases[cases["commit_after"].notna() & (cases["commit_after"].astype(str).str.strip() != "")]
        if "status" in cases.columns:
            cases = cases[~cases["status"].astype(str).str.lower().str.startswith("skip")]
        print(f"{len(cases)} of {n0} rows have a commit_after and are not skipped")
    if cases.empty:
        sys.exit("No case to measure yet: fill commit_after (and repo_url_after) for at least one row of the cases file "
                 "after applying the refactoring, then run this again.")
    smell_of = None
    if a.score_detector:
        models = detector_scorer(None)
    rows = []
    for c in cases.itertuples():
        rec = {"case_id": c.case_id, "repo_url": c.repo_url, "refactoring": c.refactoring, "level": c.level,
               "code_name": c.code_name}
        try:
            kind = "class" if c.level == "class" else "method"
            sides = {}
            for tag, sha in (("before", c.commit_before), ("after", c.commit_after)):
                url = c.repo_url
                ua = getattr(c, "repo_url_after", None)          # the refactored commit usually lives in YOUR fork
                if tag == "after" and isinstance(ua, str) and ua.strip():
                    url = ua.strip()
                out, repo = ck_for(url, sha, a, s02, None)
                ck = s03.load_ck(out, kind)
                el = find_element(ck, c.level, c.path, c.code_name, getattr(c, "method_name", None), s03)
                if el is None:
                    raise RuntimeError(f"element not found {tag}")
                sides[tag] = el
                if a.run_tests:
                    rec[f"tests_{tag}"] = tests_pass(repo, getattr(c, "test_cmd", None),
                                                 os.path.join("results", "test_logs", f"{c.case_id}_{tag}.log"))
            delta = {}
            for m in list(LOWER) + list(HIGHER):
                if m in sides["before"].index and m in sides["after"].index:
                    rec[m + "_before"], rec[m + "_after"] = sides["before"][m], sides["after"][m]
                    delta[m] = (sides["before"][m], sides["after"][m])
            for extra in ("mi", "mi_class", "debt"):                       # optional external numbers
                for tag in ("before", "after"):
                    col = f"{extra}_{tag}"
                    if col in cases.columns:
                        rec[col] = getattr(c, col)
            if a.score_detector:
                smells = [s for s in models if (s in ("blob", "data_class")) == (c.level == "class")]
                for sm in smells:
                    m, feats = models[sm]
                    for tag in ("before", "after"):
                        v = np.array([[sides[tag].get(f, np.nan) for f in feats]], dtype=float)
                        rec[f"p_{sm}_{tag}"] = float(m.predict_proba(v)[0, 1])
            rec["verdict"] = verdict(delta)
        except Exception as e:                               # keep going; report failures in the summary
            rec["verdict"] = "failed"; rec["error"] = str(e)[:200]
        rows.append(rec)
        print(rec["case_id"], rec["verdict"], rec.get("error", ""), flush=True)
    df = pd.DataFrame(rows)
    if df.empty or "verdict" not in df:
        sys.exit("No case was processed.")
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/before_after.csv", index=False)
    summarize(df[df.verdict != "failed"])
    print(f"\nfailed cases: {(df.verdict == 'failed').sum()} / {len(df)}  (report them; do not drop silently)")


if __name__ == "__main__":
    main()
