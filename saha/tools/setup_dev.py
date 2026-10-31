"""Set up a member's clone once, and check it any time.

    python -m saha.tools.setup_dev            check, then install git hooks and the upstream remote
    python -m saha.tools.setup_dev --check    check only (exit code 1 if something is wrong)

Checks Python 3.11, the pinned package versions (requirements-saha.txt), the git hooks, and the
data bundle (MTAH_DATA_DIR or ./data: the four tactical files, by size). Prints no data.
"""

import argparse
import re
import subprocess
import sys
from importlib import metadata

from saha.common import paths

UPSTREAM_URL = "https://github.com/metrica-sports/more-than-a-hack-2027"
REQUIREMENTS = paths.REPO_ROOT / "requirements-saha.txt"
# Sizes of the tactical files as delivered, to catch a truncated download. Videos are optional.
EXPECTED_BYTES = {"atd_raw": 195714285, "atd_meta": 59057, "homography": 23187043,
                  "smart_tagging": 261055}


def git(*args, check=False):
    return subprocess.run(["git"] + list(args), cwd=paths.REPO_ROOT, capture_output=True,
                          text=True, check=check)


def pins():
    out = {}
    for line in REQUIREMENTS.read_text().splitlines():
        match = re.match(r"^\s*([A-Za-z0-9_.\-]+)==([^\s#]+)", line)
        if match:
            out[match.group(1)] = match.group(2)
    return out


def check(report):
    ok = True
    version = "{}.{}.{}".format(*sys.version_info[:3])
    good = sys.version_info[:2] == (3, 11)
    report.append(("python 3.11", good, version))
    ok &= good

    wrong = []
    for name, wanted in sorted(pins().items()):
        try:
            have = metadata.version(name)
        except metadata.PackageNotFoundError:
            have = "missing"
        if have != wanted:
            wrong.append("{} {} (want {})".format(name, have, wanted))
    report.append(("pinned packages", not wrong, "; ".join(wrong) or "all match"))
    ok &= not wrong

    hooks = paths.REPO_ROOT / ".git" / "hooks"
    installed = all((hooks / name).is_file() for name in ("pre-commit", "pre-push"))
    report.append(("git hooks", installed, "installed" if installed else "not installed"))
    ok &= installed

    remote = git("remote", "get-url", "upstream").stdout.strip()
    report.append(("upstream remote", bool(remote), remote or "missing"))

    data = paths.data_dir()
    problems = []
    for asset, expected in EXPECTED_BYTES.items():
        path = paths.asset_path(asset)
        if not path.is_file():
            problems.append("{} missing".format(path.name))
        elif path.stat().st_size != expected:
            problems.append("{} has {} bytes, expected {}".format(
                path.name, path.stat().st_size, expected))
    report.append(("data bundle", not problems, "; ".join(problems) or "OK at {}".format(data)))
    return ok and not problems


def install(report):
    if not git("remote", "get-url", "upstream").stdout.strip():
        git("remote", "add", "upstream", UPSTREAM_URL)
    git("remote", "set-url", "--push", "upstream", "DISABLED")  # nobody pushes to the organisers
    git("fetch", "-q", "upstream")
    done = subprocess.run([sys.executable, "-m", "pre_commit", "install"], cwd=paths.REPO_ROOT,
                          capture_output=True, text=True)
    report.append(("install hooks", done.returncode == 0,
                   (done.stdout or done.stderr).strip().replace("\n", "; ")))
    for folder in (paths.cache_dir(), paths.submissions_dir()):
        folder.mkdir(parents=True, exist_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="check only, change nothing")
    args = parser.parse_args(argv)
    report = []
    if not args.check:
        install(report)
    ok = check(report)
    for name, good, detail in report:
        print("{:<5} {:<16} {}".format("OK" if good else "FAIL", name, detail))
    if not ok:
        print("\nSetup is incomplete; see TEAM.md (checklist).")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
