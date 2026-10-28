"""Has the official repository changed since we last merged it?

    python -m saha.tools.upstream_check

Fetches upstream/main and lists the commits our main does not have yet, with the files they
touch. Changes to the scorer (evaluation/, games.py) or to the rules and data documents are
flagged: they can change what we optimise or what we are allowed to do. WS1 runs this twice a
week and merges upstream into main through a PR (SAHA_UPSTREAM_SYNC=1 lets the guard accept
the scorer changes in that PR only).
"""

import subprocess
import sys

from saha.common import paths

WATCH = ("evaluation/", "games.py", "RULES.md", "FAQ.md", "EVALUATION.md", "SUBMISSION.md",
         "DATA.md", "CHALLENGE.md", "OVERVIEW.md", "README.md", "starter/")


def git(*args):
    return subprocess.run(["git"] + list(args), cwd=paths.REPO_ROOT, capture_output=True,
                          text=True)


def main(argv=None):
    if git("remote", "get-url", "upstream").returncode:
        print("no 'upstream' remote; run: python -m saha.tools.setup_dev")
        return 1
    fetched = git("fetch", "-q", "upstream", "main")
    if fetched.returncode:
        print("could not fetch upstream: {}".format(fetched.stderr.strip()))
        return 1
    base = next((ref for ref in ("main", "origin/main") if not git(
        "rev-parse", "--verify", "--quiet", ref).returncode), "HEAD")
    commits = git("log", "--format=%h %ad %s", "--date=short",
                  "{}..upstream/main".format(base)).stdout.strip()
    if not commits:
        print("upstream/main has nothing new for {}".format(base))
        return 0
    print("upstream/main has commits that {} does not:\n{}".format(base, commits))
    files = git("diff", "--name-only", "{}...upstream/main".format(base)).stdout.split()
    flagged = [f for f in files if f.startswith(WATCH)]
    print("\nfiles changed: {}".format(len(files)))
    for name in files:
        print("  {}{}".format(name, "   <-- scorer/rules/docs" if name in flagged else ""))
    return 2 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())
