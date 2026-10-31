"""Git guard: keep match data, frame-level files and changes to the official scorer out of git.

    python -m saha.tools.guard               check the staged changes (the pre-commit hook)
    python -m saha.tools.guard --pre-push    refuse pushes to main; scan every blob being pushed
    python -m saha.tools.guard --all         check every tracked file
    python -m saha.tools.guard --range A..B  check every blob introduced by commits in A..B (CI)
    python -m saha.tools.guard --history     check every blob ever committed (before going public)

A blob is flagged when it is under data/, cache/ or submissions/; has a data-dump extension
(parquet, npy, pickle, video, archives...); is named like a match asset (BAR-ATH_...); is larger
than MAX_KB; or is a text table with more than MAX_TABLE_LINES lines (a frame-level dump).
Staged changes to evaluation/ or games.py are refused too: that is the official scorer.

Blobs that came from the upstream repository are public by definition and always pass. An upstream
sync (WS1 only) sets SAHA_UPSTREAM_SYNC=1 to let changes to the scorer through; the one-off push
of main sets SAHA_ALLOW_MAIN_PUSH=1. Standard library only: runs under any Python 3.
"""

import argparse
import os
import subprocess
import sys

FORBIDDEN_DIRS = ("data/", "cache/", "submissions/")
ALLOWED_PATHS = {"data/README.md"}
FORBIDDEN_SUFFIXES = (".parquet", ".feather", ".arrow", ".npy", ".npz", ".pkl", ".pickle",
                      ".joblib", ".h5", ".hdf5", ".mp4", ".mov", ".avi", ".mkv", ".zip", ".7z",
                      ".tar", ".tgz", ".gz", ".gzip", ".bz2", ".xz", ".pl3container")
DATA_NAMES = ("bar-ath", "bar_ath")
TABLE_SUFFIXES = (".csv", ".tsv", ".txt", ".json", ".jsonl", ".xml", ".dat")
PROTECTED = ("evaluation/", "games.py")
MAX_KB = 500
MAX_TABLE_LINES = 5000

# The upstream commit this repository was created from. Its blobs (and those of upstream/main,
# when the remote exists) are the official public files: the scorer, the public GT, baselines.
UPSTREAM_BASE = "f8f2629ff1e0985f81d7e59d8413678514ee1530"
UPSTREAM_REFS = ("refs/remotes/upstream/main",)
MAIN_REFS = ("refs/heads/main", "refs/heads/master")
ZERO = "0" * 40


def git(*args, binary=False, check=True):
    done = subprocess.run(("git",) + args, capture_output=True, check=False)
    if check and done.returncode:
        raise RuntimeError("git {} failed: {}".format(" ".join(args),
                                                      done.stderr.decode(errors="replace")))
    return done.stdout if binary else done.stdout.decode("utf-8", errors="replace")


def ref_exists(ref):
    return subprocess.run(["git", "rev-parse", "--verify", "--quiet", ref + "^{commit}"],
                          capture_output=True).returncode == 0


def upstream_blobs():
    refs = [ref for ref in (UPSTREAM_BASE,) + UPSTREAM_REFS if ref_exists(ref)]
    if not refs:
        return set()
    return {line.split()[0] for line in git("rev-list", "--objects", *refs).splitlines()
            if line.strip()}


def path_problems(path):
    lower = path.lower()
    name = lower.rsplit("/", 1)[-1]
    found = []
    if path not in ALLOWED_PATHS and lower.startswith(FORBIDDEN_DIRS):
        found.append("lives under data/, cache/ or submissions/")
    if lower.endswith(FORBIDDEN_SUFFIXES):
        found.append("data-dump file type")
    if any(token in name for token in DATA_NAMES) and path not in ALLOWED_PATHS:
        found.append("named like a match asset")
    return found


def count_lines(blob_sha):
    return git("cat-file", "-p", blob_sha, binary=True).count(b"\n")


def blob_problems(path, sha, size):
    found = path_problems(path)
    if size > MAX_KB * 1024:
        found.append("{} KB > {} KB".format(size // 1024, MAX_KB))
    if path.lower().endswith(TABLE_SUFFIXES) and size > 10 * 1024:
        lines = count_lines(sha)
        if lines > MAX_TABLE_LINES:
            found.append("{} lines: looks like a frame-level table".format(lines))
    return found


def blob_sizes(shas):
    """{sha: (type, size)} through one cat-file --batch-check call."""
    if not shas:
        return {}
    done = subprocess.run(["git", "cat-file", "--batch-check"], input="\n".join(shas) + "\n",
                          capture_output=True, text=True, check=True)
    out = {}
    for line in done.stdout.splitlines():
        parts = line.split()
        if len(parts) == 3:
            out[parts[0]] = (parts[1], int(parts[2]))
    return out


def scan_objects(rev_args):
    """[(path, problems)] for every non-upstream blob reachable from rev_args."""
    listed = [line.split(" ", 1) for line in git("rev-list", "--objects", *rev_args).splitlines()
              if " " in line]
    info = blob_sizes([sha for sha, _ in listed])
    allowed = upstream_blobs()
    report = []
    for sha, path in listed:
        kind, size = info.get(sha, ("", 0))
        if kind != "blob" or sha in allowed:
            continue
        found = blob_problems(path, sha, size)
        if found:
            report.append((path, found))
    return report


def staged_problems(allow_upstream_sync):
    entries = git("diff", "--cached", "--name-status", "--no-renames", "-z").split("\0")
    allowed = upstream_blobs()
    report = []
    for status, path in zip(entries[0::2], entries[1::2]):
        if not path:
            continue
        found = []
        if path.startswith(PROTECTED) and not allow_upstream_sync:
            found.append("changes the official scorer (evaluation/, games.py)")
        if status != "D":
            sha = git("rev-parse", ":" + path).strip()
            if sha not in allowed:
                size = int(git("cat-file", "-s", sha).strip())
                found += blob_problems(path, sha, size)
        if found:
            report.append((path, found))
    return report


def tracked_problems():
    allowed = upstream_blobs()
    report = []
    for line in git("ls-files", "-s").splitlines():
        meta, path = line.split("\t", 1)
        sha = meta.split()[1]
        if sha in allowed:
            continue
        found = blob_problems(path, sha, int(git("cat-file", "-s", sha).strip()))
        if found:
            report.append((path, found))
    return report


def pushed_ranges():
    """[(remote_ref, rev-list args)] for this push, from pre-commit's env or git's stdin."""
    remote_ref = os.environ.get("PRE_COMMIT_REMOTE_BRANCH")
    if remote_ref is not None:
        to_ref = os.environ.get("PRE_COMMIT_TO_REF", "HEAD")
        from_ref = os.environ.get("PRE_COMMIT_FROM_REF")
        args = ["{}..{}".format(from_ref, to_ref)] if from_ref else [to_ref, "--not",
                                                                     "--remotes"]
        return [(remote_ref, args)]
    ranges = []
    for line in sys.stdin.read().splitlines():
        parts = line.split()
        if len(parts) != 4 or parts[1] == ZERO:
            continue
        local_sha, remote_ref, remote_sha = parts[1], parts[2], parts[3]
        args = (["{}..{}".format(remote_sha, local_sha)] if remote_sha != ZERO
                else [local_sha, "--not", "--remotes"])
        ranges.append((remote_ref, args))
    return ranges


def print_report(title, report):
    if not report:
        return 0
    print("saha guard: {} -- refused".format(title))
    for path, found in report:
        print("  {}  ({})".format(path, "; ".join(found)))
    print("Match data and frame-level files never go into git (CLAUDE.md). Unstage them "
          "(git restore --staged <path>) or, if a commit already holds them, ask WS1.")
    return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--pre-push", action="store_true")
    mode.add_argument("--all", action="store_true")
    mode.add_argument("--range")
    mode.add_argument("--history", action="store_true")
    args = parser.parse_args(argv)

    if args.pre_push:
        status = 0
        for remote_ref, rev_args in pushed_ranges():
            if remote_ref in MAIN_REFS and os.environ.get("SAHA_ALLOW_MAIN_PUSH") != "1":
                print("saha guard: direct push to {} refused. Push your branch and open a PR "
                      "(CLAUDE.md, git flow).".format(remote_ref))
                status = 1
            status |= print_report("blobs in the commits being pushed", scan_objects(rev_args))
        return status
    if args.all:
        return print_report("tracked files", tracked_problems())
    if args.range:
        return print_report("commits in " + args.range, scan_objects([args.range]))
    if args.history:
        report = scan_objects(["--all"])
        if not report:
            print("saha guard: history clean ({} rules, upstream blobs excluded)".format(
                "path/type/name/size/table"))
        return print_report("full history", report)
    return print_report("staged changes",
                        staged_problems(os.environ.get("SAHA_UPSTREAM_SYNC") == "1"))


if __name__ == "__main__":
    sys.exit(main())
