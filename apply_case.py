"""Apply one prepared refactoring patch to YOUR fork and register the commit in cases_todo.csv.

Prerequisites: you forked the upstream repo on GitHub (button Fork) and `git` is logged in (git push works).
Usage (from the xsmell_mlcq folder):
    python apply_case.py --case long_method_06 --user thzynu
    python apply_case.py --case long_method_06 --user thzynu --no-push      # dry run: commit locally only
What it does: shallow-fetches commit_before, creates branch xsmell-<case_id>, applies patches/<case_id>.patch,
commits, pushes, then writes repo_url_after, commit_after, status and notes into cases_todo.csv.
"""
import argparse, os, re, shutil, stat, subprocess, sys
import pandas as pd

NOTES = {
    "long_method_06": ("Extract Method", "Extract Method checkZkConnection(req, rb, cc) from SearchHandler.getAndPrepShardHandler. "
                       "Repo builds with ant/ivy: no test run, metrics only (CK, MI)."),
    "long_method_19": ("Extract Method", "Extract Method checkFromLeaderRequest(...) from DistributedZkUpdateProcessor.doDefensiveChecks. "
                       "Repo builds with ant/ivy: no test run, metrics only (CK, MI)."),
    "long_method_09": ("Extract Method", "Extract Method checkClusterIsReadyForDiscovery(clusterId) from ResourceManagerImpl.discoverHosts(AddHostCmd). "
                       "Multi-module maven build too heavy for this machine: no test run, metrics only (CK, MI)."),
    "blob_15": ("Extract Class", "Extract Class MedianFinder (readAndFindMedian moved out of WordMedian, new file MedianFinder.java). "
                "Legacy hadoop-common build not reproducible here: no test run, metrics only (CK, MI)."),
}


def run(cmd, cwd=None, check=True):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if check and r.returncode:
        sys.exit(f"command failed: {' '.join(cmd)}\n{r.stdout}\n{r.stderr}")
    return r.stdout.strip()


def rmtree(p):
    def onerr(f, path, e):
        os.chmod(path, stat.S_IWRITE); f(path)
    shutil.rmtree(p, onerror=onerr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True); ap.add_argument("--user", help="your GitHub user name")
    ap.add_argument("--fork-url", help="override the fork URL (tests)"); ap.add_argument("--cases", default="cases_todo.csv")
    ap.add_argument("--patches", default="patches"); ap.add_argument("--work", default="work_cases")
    ap.add_argument("--no-push", action="store_true")
    a = ap.parse_args()
    df = pd.read_csv(a.cases, dtype=str, keep_default_na=False)
    row = df[df.case_id == a.case]
    if row.empty:
        sys.exit(f"{a.case} not in {a.cases}")
    r = row.iloc[0]
    patch = os.path.abspath(os.path.join(a.patches, a.case + ".patch"))
    if not os.path.exists(patch):
        sys.exit(f"missing {patch}")
    name = r.repo_url.rstrip("/").split("/")[-1]
    fork = a.fork_url or f"https://github.com/{a.user}/{name}"
    if not (a.fork_url or a.user):
        sys.exit("give --user (your GitHub name)")
    if not run(["git", "config", "user.name"], check=False):
        sys.exit('set your identity first: git config --global user.name "NAME" ; git config --global user.email "EMAIL"')
    d = os.path.join(a.work, a.case)
    if os.path.exists(d):
        rmtree(d)
    os.makedirs(d)
    run(["git", "init", "-q"], d); run(["git", "config", "core.longpaths", "true"], d)
    run(["git", "remote", "add", "origin", fork], d)
    f = subprocess.run(["git", "fetch", "-q", "--depth", "1", "origin", r.commit_before], cwd=d, capture_output=True, text=True)
    if f.returncode:                                  # fork may not contain the commit: fall back to upstream objects
        print("commit not found in the fork, fetching it from upstream ...")
        run(["git", "remote", "add", "upstream", r.repo_url], d)
        run(["git", "fetch", "-q", "--depth", "1", "upstream", r.commit_before], d)
    run(["git", "checkout", "-q", "-b", f"xsmell-{a.case}", "FETCH_HEAD"], d)
    run(["git", "apply", "--whitespace=nowarn", patch], d)
    run(["git", "add", "-A"], d)
    run(["git", "commit", "-q", "-m", f"{NOTES[a.case][0]} ({a.case})"], d)
    sha = run(["git", "rev-parse", "HEAD"], d)
    print("commit_after =", sha)
    if not a.no_push:
        run(["git", "push", "-u", "origin", f"xsmell-{a.case}"], d)
        print("pushed to", fork)
    i = df.index[df.case_id == a.case][0]
    df.at[i, "repo_url_after"] = fork if not a.fork_url else r.repo_url
    df.at[i, "commit_after"] = sha; df.at[i, "status"] = "done"
    df.at[i, "refactoring"] = NOTES[a.case][0]; df.at[i, "notes"] = NOTES[a.case][1]; df.at[i, "test_cmd"] = ""
    df.to_csv(a.cases, index=False, lineterminator="\n")
    print(a.cases, "updated")


if __name__ == "__main__":
    main()
