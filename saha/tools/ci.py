"""Run the checks of .github/workflows/ci.yml locally. GitHub Actions does not run on this
repository (DECISIONS D023), so this is the CI now.

    python -m saha.tools.ci                  # the checks, on the current checkout
    python -m saha.tools.ci --base main      # the branch range: origin/<base>..HEAD (default main)

The same steps as the workflow, in the same order:

  1. the guard on every tracked file;
  2. the guard on every commit of this branch (origin/<base>..HEAD; run `git fetch` first);
  3. the experiment records;
  4. the tests;
  5. the official scorer on the shipped baselines;
  6. docs/METRIC_NOTES.md against the scorer.

Synthetic data and the repository's public files only: no match data is read, so it may run
anywhere, a cloud session included. Install the pins first (`requirements.txt` and
`requirements-saha.txt`). One line per step; the tail of a failing step's output; exit 1 if any
step fails.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from saha.common import paths

TAIL = 25


def steps(base, scratch):
    """[(name, command)] in the workflow's order."""
    py = sys.executable
    return [
        ("guard: every tracked file", [py, "-m", "saha.tools.guard", "--all"]),
        ("guard: commits of this branch",
         [py, "-m", "saha.tools.guard", "--range", "origin/{}..HEAD".format(base)]),
        ("experiment records", [py, "-m", "saha.common.experiments", "check"]),
        ("tests", [py, "-m", "pytest", "-q"]),
        ("official scorer on the baselines",
         [py, "evaluation/run_evaluation.py", "--no-detail", "--out", str(scratch / "eval")]),
        ("docs/METRIC_NOTES.md matches the scorer",
         [py, "-m", "saha.tools.metric_behaviour", "--out", str(scratch / "METRIC_NOTES.md")]),
    ]


def same_text(generated, committed):
    """True when the regenerated METRIC_NOTES.md is byte for byte the committed one."""
    return Path(generated).read_bytes() == Path(committed).read_bytes()


def has_ref(ref):
    return subprocess.run(["git", "rev-parse", "--verify", "--quiet", ref],
                          cwd=paths.REPO_ROOT, capture_output=True).returncode == 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="main")
    args = parser.parse_args(argv)
    if not has_ref("origin/" + args.base):
        print("origin/{} is missing: run `git fetch origin {}` first".format(args.base, args.base))
        return 1
    failed = []
    with tempfile.TemporaryDirectory() as folder:
        scratch = Path(folder)
        for name, command in steps(args.base, scratch):
            done = subprocess.run(command, cwd=paths.REPO_ROOT, capture_output=True, text=True)
            ok = done.returncode == 0
            if ok and name.startswith("docs/METRIC_NOTES.md"):
                ok = same_text(scratch / "METRIC_NOTES.md", paths.REPO_ROOT / "docs" /
                               "METRIC_NOTES.md")
            last = (done.stdout.strip().splitlines() or [""])[-1]
            print("{:<5} {:<42} {}".format("PASS" if ok else "FAIL", name, last[:60]),
                  flush=True)
            if not ok:
                failed.append(name)
                output = (done.stdout + done.stderr).strip().splitlines()
                print("\n".join("      " + line for line in output[-TAIL:]))
    print("\n{} of {} steps passed{}".format(len(steps(args.base, Path("."))) - len(failed),
                                            len(steps(args.base, Path("."))),
                                            ": failed " + ", ".join(failed) if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
