"""The whole submission in one command: data in, submissions/SAHA.csv out, validated and scored.

    python -m saha.pipeline --data <path-to-data-bundle>
    python -m saha.pipeline                      # MTAH_DATA_DIR or ./data
    python -m saha.pipeline --reuse-evidence     # skip builders whose evidence file exists
    python -m saha.pipeline --fresh              # delete the cache first (what the final run does)

Steps: frame table (cached, rebuilt when the data or its version changes) -> every evidence
builder in the config, as `python -m <module>` exactly as members run them -> interface validation
-> combiner -> CSV -> evaluation/validate_submission.py (the official checks) -> public-window
score with a 1000-draw bootstrap and match-wide statistics -> submissions/SAHA.summary.json.

Everything printed is aggregate: scores, counts, hashes. No frame-level output, ever (CLAUDE.md).
"""

import argparse
import json
import platform
import shutil
import subprocess
import sys
import tomllib
from importlib import metadata
from pathlib import Path

from saha.combine.decode import CombineParams, decode
from saha.common import evidence, frame_table, paths, submission

DEFAULT_CONFIG = paths.CONFIGS_DIR / "pipeline.toml"
TEAM = "SAHA"
PINNED = ("numpy", "pandas", "pyarrow", "scipy", "kloppy")
# Evidence pairs that must not both give p_home on one frame (TEAM.md, boundaries).
BOUNDARY_PAIRS = (("visible", "occluded"),)


def load_config(path):
    with open(path, "rb") as handle:
        config = tomllib.load(handle)
    entries = config.get("evidence", [])
    names = [e["name"] for e in entries]
    if len(set(names)) != len(names):
        raise ValueError("duplicate evidence names in {}".format(path))
    for entry in entries:
        if not evidence.NAME_PATTERN.match(entry["name"]) or "module" not in entry:
            raise ValueError("bad evidence entry {!r} in {}".format(entry, path))
    return CombineParams.from_dict(config.get("combine")), entries


def environment():
    versions = {}
    for name in PINNED:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": versions}


def git_commit():
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=paths.REPO_ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=paths.REPO_ROOT, capture_output=True, text=True,
                               check=True).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def run_builder(module):
    """`python -m module` from the repo root, inheriting MTAH_DATA_DIR / SAHA_CACHE_DIR."""
    print("  $ python -m {}".format(module), flush=True)
    done = subprocess.run([sys.executable, "-m", module], cwd=paths.REPO_ROOT)
    if done.returncode:
        raise SystemExit("builder {} failed with exit code {}".format(module, done.returncode))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", help="data bundle root (default: MTAH_DATA_DIR or ./data)")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out", default=None, help="default: submissions/SAHA.csv")
    parser.add_argument("--reuse-evidence", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="delete the cache before running")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--no-score", action="store_true")
    args = parser.parse_args(argv)

    if sys.version_info[:2] != (3, 11):
        print("WARNING: Python {}.{}; the team's reproducible runs use 3.11 (CLAUDE.md)".format(
            *sys.version_info[:2]))
    if args.data:
        paths.use_data_dir(args.data)
    params, entries = load_config(args.config)
    out = Path(args.out) if args.out else paths.submissions_dir() / "{}.csv".format(TEAM)

    if args.fresh and paths.cache_dir().exists():
        shutil.rmtree(paths.cache_dir())
        print("deleted {}".format(paths.cache_dir()))

    print("[1/6] frame table")
    table = frame_table.load()
    print("  " + frame_table.summary(table).replace("\n", "\n  "))
    del table

    print("[2/6] evidence builders")
    for entry in entries:
        if args.reuse_evidence and paths.evidence_path(entry["name"]).is_file():
            print("  reusing {}".format(paths.evidence_path(entry["name"]).name))
            continue
        run_builder(entry["module"])

    print("[3/6] evidence validation")
    sources, hashes = [], {}
    for entry in entries:
        ok, lines = evidence.validate_file(entry["name"])
        print("  " + "\n  ".join(lines))
        if not ok:
            raise SystemExit("evidence {} is invalid".format(entry["name"]))
        loaded = evidence.read_evidence(entry["name"])
        sources.append((loaded, float(entry.get("scale", 1.0))))
        hashes[entry["name"]] = loaded.content_hash()
    by_name = {source.name: source for source, _ in sources}
    for first, second in BOUNDARY_PAIRS:
        if first in by_name and second in by_name:
            shared = evidence.home_overlap(by_name[first], by_name[second])
            if shared:
                print("  WARNING: {} and {} both give p_home on {} graded frames; they should "
                      "split the frames by ball visibility (TEAM.md, boundaries)".format(
                          first, second, shared))

    print("[4/6] combine")
    codes = decode(sources, params)
    submission.write_submission(out, codes)
    digest = submission.sha256_of(out)
    print("  wrote {}  sha256 {}".format(out, digest))

    print("[5/6] official validation")
    check = subprocess.run([sys.executable, str(paths.VALIDATOR), str(out)],
                           cwd=paths.REPO_ROOT, capture_output=True, text=True)
    print("  " + check.stdout.strip().replace("\n", "\n  "))
    if check.returncode:
        raise SystemExit("validate_submission.py rejected the file")

    summary = {"team": TEAM, "csv": str(out), "sha256": digest, "config": args.config,
               "combine": params.to_dict(), "evidence": hashes, "git_commit": git_commit(),
               "environment": environment(), "match": submission.match_stats(codes)}
    print("[6/6] statistics")
    m = summary["match"]
    print("  match-wide: {} possessions, mean {:.1f} s (median {:.1f} s), {} shorter than 2 s, "
          "D {:.1%}, X {:.1%}, home share {:.1%}".format(
              m["n_possessions"], m["mean_duration_s"], m["median_duration_s"],
              m["short_possessions_lt_2s"], m["d_rate"], m["x_rate"], m["home_share"]))
    if not args.no_score:
        s = submission.score(codes, bootstrap=args.bootstrap)
        summary["public"] = s
        print("  public 5 min (pipeline check only, not for choosing): spellF1 {:.4f} "
              "[{}-{}]  macroF1 {:.4f}  possessions {} vs {}".format(
                  s["spell_f1_mean"],
                  "--" if s["spell_bootstrap_low"] is None else "{:.2f}".format(
                      s["spell_bootstrap_low"]),
                  "--" if s["spell_bootstrap_high"] is None else "{:.2f}".format(
                      s["spell_bootstrap_high"]),
                  s["macro_f1"], s["n_pred_spells"], s["n_gt_spells"]))
    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("  wrote {}".format(summary_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
