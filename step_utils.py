import re


def tag(url, sha):
    """Folder name for a (repo, commit) pair. Must equal the one used in 02_fetch_and_run_ck.py."""
    name = re.sub(r"[^A-Za-z0-9]+", "_", url.split("github.com/")[-1])
    return f"{name}__{str(sha)[:10]}"
