"""Step 14 - Technical debt (SonarQube remediation effort) of the refactored element, before vs after.

WHY THIS DESIGN. SonarQube reports technical debt per file/project (metric sqale_index, in minutes) and gives every code smell
a line and a remediation effort, but NOT a debt per method. We therefore (1) analyse ONLY the one source file of each case
(before and after) as its own tiny SonarQube project, and (2) sum the effort of the CODE_SMELL issues whose line lies inside the
class/method that was refactored. Analysing a single file is fast, needs no build, and works on huge repositories
(hadoop, cloudstack, lucene-solr). Limitation to report: without compiled classes / dependencies SonarJava analyses with
reduced semantic information, so some type-based rules do not fire; the SAME handicap applies to before and after, so the
comparison stays fair, but absolute minutes are not those of a full-project scan.

TWO STEPS
  python 14_sonar_debt.py prepare --cases cases_todo.csv --org YOUR_SONAR_ORG
      -> sonar_in/<case>_<before|after>/ (the single .java file + sonar-project.properties), sonar_in/run_all.bat, ranges.json
  (run sonar_in\\run_all.bat, see README_sonar.md)
  python 14_sonar_debt.py collect --cases cases_todo.csv --org YOUR_SONAR_ORG        (token in env SONAR_TOKEN)
      -> writes debt_before / debt_after / issues_before / issues_after into the cases file; 11_before_after.py then
         reports them in results/before_after.csv

  python 14_sonar_debt.py --self-test      # checks the effort parser and the line filtering
"""
import argparse
import importlib.util
import json
import os
import re
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


def effort_minutes(s):
    """SonarQube effort string -> minutes.  '5min', '1h', '1h30min', '2d', '1d4h' ; 1 day = 8 hours (Sonar default)."""
    if s is None or s == "":
        return 0
    total = 0
    for num, unit in re.findall(r"(\d+)\s*(d|h|min)", str(s)):
        n = int(num)
        total += n * 480 if unit == "d" else n * 60 if unit == "h" else n
    return total


def debt_in_range(issues, first, last):
    """-> (n_issues, minutes) of CODE_SMELL issues whose start line is inside [first, last]."""
    n = m = 0
    for it in issues:
        if it.get("type") not in (None, "CODE_SMELL"):
            continue
        line = it.get("line") or (it.get("textRange") or {}).get("startLine")
        if line is None or not (first <= int(line) <= last):
            continue
        n += 1
        m += effort_minutes(it.get("effort") or it.get("debt"))
    return n, m


def cases_done(cases):
    c = cases[cases["commit_after"].notna() & (cases["commit_after"].astype(str).str.strip() != "")]
    if "status" in c.columns:
        c = c[~c["status"].astype(str).str.lower().str.startswith("skip")]
    return c


def locate(a, row, tag, s02, s03, q, f11):
    """-> (java_file_path, first_line, last_line) for one side of one case (uses the checkouts/CK output of step 11)."""
    sha = row.commit_before if tag == "before" else row.commit_after
    url = row.repo_url
    ua = getattr(row, "repo_url_after", None)
    if tag == "after" and isinstance(ua, str) and ua.strip():
        url = ua.strip()
    out, repo = f11.ck_for(url, sha, a, s02, None)           # reuses data/ck_ba and work_ba, fetches only if missing
    kind = "class" if row.level == "class" else "method"
    ck = s03.load_ck(out, kind)
    el = f11.find_element(ck, row.level, row.path, row.code_name, getattr(row, "method_name", None), s03,
                          start_line=getattr(row, "start_line", None))
    if el is None:
        raise RuntimeError(f"element not found ({tag})")
    path = os.path.join(repo, el["_file"])
    start = int(el["line"]) if kind == "method" else 0
    _, l0, l1 = q.element_range(path, row.level, el["class"] if kind == "class" else row.code_name,
                                getattr(row, "method_name", None), start)
    return path, l0, l1


def prepare(a):
    s02 = _load("s02", "02_fetch_and_run_ck.py"); s03 = _load("s03", "03_build_dataset.py")
    f11 = _load("f11", "11_before_after.py"); q = _load("xs_quality", "xs_quality.py")
    cases = cases_done(pd.read_csv(a.cases))
    root = a.indir
    os.makedirs(root, exist_ok=True)
    cmds, ranges = [], {}
    for row in cases.itertuples():
        for tag in ("before", "after"):
            key = f"{a.prefix}_{row.case_id}_{tag}".replace("-", "_")
            pkey = f"{a.org}_{key}" if a.org else key
            path, l0, l1 = locate(a, row, tag, s02, s03, q, f11)
            d = os.path.join(root, f"{row.case_id}_{tag}")
            shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
            shutil.copy(path, os.path.join(d, os.path.basename(path)))
            open(os.path.join(d, "sonar-project.properties"), "w", encoding="utf-8").write(
                f"sonar.host.url={a.host}\n" + (f"sonar.organization={a.org}\n" if a.org else "") + f"sonar.projectKey={pkey}\n"
                f"sonar.projectName={key}\nsonar.sources=.\nsonar.sourceEncoding=UTF-8\nsonar.java.binaries=.\n"
                f"sonar.java.source=8\n")
            ranges[f"{row.case_id}_{tag}"] = dict(file=os.path.basename(path), first=l0, last=l1, key=pkey)
            cmds.append(os.path.abspath(d))
            print(f"{row.case_id} {tag}: {os.path.basename(path)} lines {l0}-{l1}")
    json.dump(ranges, open(os.path.join(root, "ranges.json"), "w"), indent=1)
    open(os.path.join(root, "commands.txt"), "w").write("\n".join(cmds) + "\n")
    with open(os.path.join(root, "run_all.bat"), "w") as f:            # Windows: one scanner run per folder
        f.write("@echo off\n")
        for d in cmds:
            f.write(f'pushd "{d}"\ncall sonar-scanner\nif errorlevel 1 echo FAILED in {d}\npopd\n')
    print(f"\n{len(cmds)} single-file projects written under {root}/ . Run the scanner in each folder (see README_sonar.md).")


def api_get(host, token, path, params):
    url = f"{host.rstrip('/')}/api/{path}?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def all_issues(host, token, key, org):
    """Maintainability issues of a project. SonarQube in MQR mode has no 'code smell' type: use the software quality
    (impactSoftwareQualities=MAINTAINABILITY), which is what sqale_index adds up; older servers fall back to types=CODE_SMELL."""
    for flt in ({"impactSoftwareQualities": "MAINTAINABILITY"}, {"types": "CODE_SMELL"}):
        out, p = [], 1
        try:
            while True:
                params = {"componentKeys": key, "resolved": "false", "ps": 500, "p": p, **flt}
                if org:
                    params["organization"] = org
                d = api_get(host, token, "issues/search", params)
                out += d.get("issues", [])
                if p * 500 >= d.get("paging", {}).get("total", 0) or not d.get("issues"):
                    return out
                p += 1
        except urllib.error.HTTPError as e:
            if e.code != 400:
                raise
    return []


FILE_METRICS = ["sqale_index", "cognitive_complexity", "complexity", "ncloc", "violations"]


def file_measures(host, token, key):
    m = api_get(host, token, "measures/component", {"component": key, "metricKeys": ",".join(FILE_METRICS)})
    got = {x["metric"]: float(x["value"]) for x in m["component"]["measures"] if "value" in x}
    return {k: got.get(k, 0.0) for k in FILE_METRICS}


def collect(a):
    token = os.environ.get("SONAR_TOKEN")
    if not token:
        sys.exit("Set your token first:  set SONAR_TOKEN=xxxxxxxx   (SonarQube Cloud > My Account > Security)")
    ranges = json.load(open(os.path.join(a.indir, "ranges.json")))
    cases = pd.read_csv(a.cases, dtype={"commit_after": str, "commit_before": str})
    for col in ("debt_before", "debt_after", "issues_before", "issues_after"):
        if col not in cases.columns:
            cases[col] = float("nan")
    fm = {}
    for k, r in ranges.items():
        case_id, tag = k.rsplit("_", 1)
        issues = all_issues(a.host, token, r["key"], a.org)
        n, minutes = debt_in_range(issues, r["first"], r["last"])
        try:
            fm[(case_id, tag)] = file_measures(a.host, token, r["key"])
            file_debt = int(fm[(case_id, tag)]["sqale_index"])
        except Exception:
            file_debt = None
        whole = sum(effort_minutes(i.get("effort")) for i in issues)
        note = "" if file_debt in (None, whole) else f"  [WARNING whole-file sum {whole} != sqale_index {file_debt}]"
        print(f"{case_id} {tag}: element lines {r['first']}-{r['last']}: {n} maintainability issues, {minutes} min; "
              f"whole file: {len(issues)} issues, {file_debt} min{note}")
        i = cases.index[cases.case_id == case_id]
        cases.loc[i, f"debt_{tag}"] = minutes
        cases.loc[i, f"issues_{tag}"] = n
    rows = []
    for case_id in sorted({c for c, _ in fm}):
        if (case_id, "before") in fm and (case_id, "after") in fm:
            row = {"case_id": case_id}
            for m in FILE_METRICS:
                b, af = fm[(case_id, "before")][m], fm[(case_id, "after")][m]
                row.update({f"{m}_before": b, f"{m}_after": af, f"{m}_change": af - b})
            rows.append(row)
    if rows:
        os.makedirs("results", exist_ok=True)
        pd.DataFrame(rows).to_csv("results/sonar_file_metrics.csv", index=False)
        print("\nSonarQube file-level metrics (one file per project, so the change comes from the refactoring only):")
        t = pd.DataFrame(rows)
        print(t[["case_id", "cognitive_complexity_before", "cognitive_complexity_after", "complexity_before", "complexity_after",
                 "sqale_index_before", "sqale_index_after"]].to_string(index=False))
        print("-> results/sonar_file_metrics.csv")
    cases.to_csv(a.cases, index=False)
    print(f"\nwritten to {a.cases}: debt_before/after (minutes) and issues_before/after. Now run 11_before_after.py again.")


def self_test():
    assert effort_minutes("5min") == 5 and effort_minutes("1h30min") == 90 and effort_minutes("2d") == 960
    assert effort_minutes("1d4h") == 720 and effort_minutes(None) == 0 and effort_minutes("") == 0
    iss = [dict(type="CODE_SMELL", line=10, effort="5min"), dict(type="CODE_SMELL", textRange=dict(startLine=20), effort="1h"),
           dict(type="BUG", line=12, effort="30min"), dict(type="CODE_SMELL", line=99, effort="10min"),
           dict(type="CODE_SMELL", effort="5min")]
    assert debt_in_range(iss, 5, 25) == (2, 65), debt_in_range(iss, 5, 25)
    print("self-test OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", choices=["prepare", "collect"])
    ap.add_argument("--cases", default="cases_todo.csv"); ap.add_argument("--org", default="")
    ap.add_argument("--host", default="https://sonarcloud.io"); ap.add_argument("--prefix", default="xsmell")
    ap.add_argument("--indir", default="sonar_in"); ap.add_argument("--ck-jar", default="tools/ck.jar")
    ap.add_argument("--workdir", default="work_ba"); ap.add_argument("--out-ck", default="data/ck_ba")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
    elif a.mode == "prepare":
        if not a.org and "sonarcloud.io" in a.host:
            sys.exit("--org YOUR_SONAR_ORGANIZATION_KEY is required for sonarcloud.io (for a local SonarQube use --host http://localhost:9000 and no --org)")
        prepare(a)
    elif a.mode == "collect":
        collect(a)
    else:
        ap.print_help()
