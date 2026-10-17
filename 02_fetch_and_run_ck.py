"""Step 2 - For every (repository, commit) in labels.csv: fetch the source at that
commit and run the CK metric tool. Resumable: finished commits are skipped.

Usage:
    python 02_fetch_and_run_ck.py --labels data/labels.csv --ck-jar tools/ck.jar \
        --workdir work --out data/ck_raw --jobs 2

Requires: git, Java (11+), and the CK "jar-with-dependencies" file
(https://github.com/mauricioaniche/ck - see README).
"""
import argparse, glob, os, shutil, stat, subprocess, re
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from step_utils import tag


def rm(path):
    """rmtree that also works on Windows (git marks .git files read-only)."""
    def onerr(func, p, exc):
        try:
            os.chmod(p, stat.S_IWRITE); func(p)
        except Exception:
            pass
    shutil.rmtree(path, onerror=onerr) if os.path.exists(path) else None


def run(cmd, cwd=None, timeout=1800):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def fetch(url, sha, dest):
    os.makedirs(dest, exist_ok=True)
    for c in (["git", "init", "-q"], ["git", "config", "core.longpaths", "true"],
              ["git", "remote", "add", "origin", url]):
        run(c, cwd=dest)
    r = run(["git", "fetch", "-q", "--depth", "1", "origin", sha], cwd=dest)
    if r.returncode != 0:                       # server refused fetch-by-sha: full clone
        rm(dest)
        r = run(["git", "clone", "-q", "-c", "core.longpaths=true", url, dest])
        if r.returncode != 0:
            return False, r.stderr[-300:]
        r = run(["git", "checkout", "-q", sha], cwd=dest)
        return r.returncode == 0, r.stderr[-300:]
    r = run(["git", "checkout", "-q", "FETCH_HEAD"], cwd=dest)
    return r.returncode == 0, r.stderr[-300:]


def job(args):
    url, sha, a = args
    t = tag(url, sha)
    out = os.path.join(a.out, t)
    if glob.glob(os.path.join(out, "*class.csv")):
        return t, "skipped(done)", ""
    repo = os.path.join(a.workdir, t)
    rm(repo)
    ok, err = fetch(url, sha, repo)
    if not ok:
        rm(repo)
        return t, "FETCH_FAILED", err
    os.makedirs(out, exist_ok=True)
    try:
        # CK args: <project dir> <use jars> <max files per partition, 0=auto> <variable metrics> <output dir/>
        r = run(["java", "-jar", a.ck_jar, repo, "true", "0", "false", out + os.sep],
                timeout=a.timeout)
        status = "ok" if glob.glob(os.path.join(out, "*class.csv")) else "CK_NO_OUTPUT"
        err = r.stderr[-300:] if status != "ok" else ""
    except subprocess.TimeoutExpired:
        status, err = "CK_TIMEOUT", ""
    if status == "ok":                          # remember the checkout root to strip from CK paths
        with open(os.path.join(out, "ROOT.txt"), "w") as f:
            f.write(os.path.abspath(repo))
    if not a.keep:
        rm(repo)
    return t, status, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--ck-jar", required=True)
    ap.add_argument("--workdir", default="work")
    ap.add_argument("--out", default="data/ck_raw")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--keep", action="store_true", help="keep cloned sources")
    ap.add_argument("--limit", type=int, default=0, help="only first N commits (for a test run)")
    a = ap.parse_args()
    os.makedirs(a.workdir, exist_ok=True); os.makedirs(a.out, exist_ok=True)

    lab = pd.read_csv(a.labels)
    pairs = lab[["repo_url", "commit_hash"]].drop_duplicates().values.tolist()
    if a.limit:
        pairs = pairs[:a.limit]
    print(f"{len(pairs)} (repo, commit) pairs to process")
    rows = []
    with ThreadPoolExecutor(a.jobs) as ex:
        for i, (t, st, err) in enumerate(ex.map(job, [(u, s, a) for u, s in pairs]), 1):
            print(f"[{i}/{len(pairs)}] {t}: {st}")
            rows.append({"tag": t, "status": st, "error": err})
    log = pd.DataFrame(rows)
    log.to_csv(os.path.join(a.out, "_run_log.csv"), index=False)
    print(log["status"].value_counts().to_string())
    print("Failures (repos deleted/force-pushed, or CK errors) are listed in _run_log.csv;"
          " report how many samples were lost in the paper.")


if __name__ == "__main__":
    main()
