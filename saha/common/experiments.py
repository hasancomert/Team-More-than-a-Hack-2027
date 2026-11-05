"""Experiment records: one Markdown file per experiment, compiled into one table on demand.

    python -m saha.common.experiments                       # the table of every experiment
    python -m saha.common.experiments new ws2 restarts-v1   # a new record, numbers pre-filled
    python -m saha.common.experiments check                 # validate every record (CI runs it)

Files live in experiments/ as YYYYMMDD_<package>_<short-name>.md. One file per experiment, so
two people never edit the same file and there are no merge conflicts. `new` copies the numbers of
the last pipeline run from submissions/SAHA.summary.json, so run the pipeline first.
"""

import argparse
import datetime
import json
import re
import sys
from pathlib import Path

from saha.common import paths

PACKAGES = ("ws1", "ws2", "ws3", "ws4", "ws5")
FILENAME = re.compile(r"^(\d{8})_(ws[1-5])_([a-z0-9][a-z0-9-]*)\.md$")
VERDICTS = ("open", "adopted", "rejected")
FIELDS = ("date", "package", "name", "author", "commit", "config", "public_spell_f1",
          "public_spell_low", "public_spell_high", "public_macro_f1", "n_possessions",
          "mean_duration_s", "d_rate", "verdict")
COLUMNS = (("date", "date"), ("package", "pkg"), ("name", "experiment"),
           ("public_spell_f1", "spellF1"), ("public_spell_low", "lo"),
           ("public_spell_high", "hi"), ("public_macro_f1", "macroF1"),
           ("n_possessions", "poss"), ("mean_duration_s", "dur s"), ("d_rate", "D"),
           ("verdict", "verdict"), ("commit", "commit"))

TEMPLATE = """---
{front}
---

## What changed

<!-- One paragraph: the code or config change, with the PR link. -->

## Why

<!-- The football or physics reason, or the result on WS5's external data. Parameters are tuned
on external data only; the public score never chooses (DECISIONS D016). -->

## Result

<!-- WS5 external results first, when they exist. Then, as checks only: public spellF1 with its
bootstrap interval, match-wide possession count, mean duration and D rate, against the previous
adopted version. On the full match, gaps below ~0.04 or inside overlapping intervals are
noise. -->

## Decision

<!-- adopted / rejected / open, and why. -->
"""


def experiments_dir():
    return paths.EXPERIMENTS_DIR


def read_front_matter(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("{}: no front matter".format(Path(path).name))
    fields = {}
    for line in lines[1:]:
        if line.strip() == "---":
            return fields
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
    raise ValueError("{}: front matter is not closed".format(Path(path).name))


def records():
    return sorted(p for p in experiments_dir().glob("*.md") if FILENAME.match(p.name))


def problems():
    found = []
    for path in sorted(experiments_dir().glob("*.md")):
        if path.name in ("README.md", "TEMPLATE.md"):
            continue
        match = FILENAME.match(path.name)
        if not match:
            found.append("{}: name must be YYYYMMDD_ws<N>_<short-name>.md".format(path.name))
            continue
        try:
            fields = read_front_matter(path)
        except ValueError as error:
            found.append(str(error))
            continue
        missing = [k for k in FIELDS if k not in fields]
        if missing:
            found.append("{}: missing fields {}".format(path.name, missing))
        if fields.get("package") != match.group(2):
            found.append("{}: package field does not match the file name".format(path.name))
        if fields.get("verdict") not in VERDICTS:
            found.append("{}: verdict must be one of {}".format(path.name, VERDICTS))
    return found


def table():
    rows = []
    for path in records():
        fields = read_front_matter(path)
        rows.append([fields.get(key, "") for key, _ in COLUMNS])
    header = [title for _, title in COLUMNS]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines), len(rows)


def from_summary():
    """Numbers of the last pipeline run, as front-matter strings."""
    path = paths.submissions_dir() / "SAHA.summary.json"
    if not path.is_file():
        return {}
    summary = json.loads(path.read_text())
    public, match = summary.get("public", {}), summary.get("match", {})

    def fmt(value, digits):
        return "" if value is None else "{:.{}f}".format(value, digits)

    config = summary.get("config", "")
    try:
        config = str(Path(config).resolve().relative_to(paths.REPO_ROOT).as_posix())
    except ValueError:
        pass
    return {
        "commit": (summary.get("git_commit") or "")[:12],
        "config": config,
        "public_spell_f1": fmt(public.get("spell_f1_mean"), 4),
        "public_spell_low": fmt(public.get("spell_bootstrap_low"), 3),
        "public_spell_high": fmt(public.get("spell_bootstrap_high"), 3),
        "public_macro_f1": fmt(public.get("macro_f1"), 4),
        "n_possessions": str(match.get("n_possessions", "")),
        "mean_duration_s": fmt(match.get("mean_duration_s"), 1),
        "d_rate": fmt(match.get("d_rate"), 3),
    }


def new(package, name, author="", today=None):
    if package not in PACKAGES:
        raise ValueError("package must be one of {}".format(PACKAGES))
    if not re.match(r"^[a-z0-9][a-z0-9-]*$", name):
        raise ValueError("short name: lower case, digits and dashes")
    today = today or datetime.date.today()
    path = experiments_dir() / "{}_{}_{}.md".format(today.strftime("%Y%m%d"), package, name)
    if path.exists():
        raise FileExistsError(path)
    fields = {key: "" for key in FIELDS}
    fields.update(from_summary())
    fields.update({"date": today.isoformat(), "package": package, "name": name,
                   "author": author, "verdict": "open"})
    front = "\n".join("{}: {}".format(key, fields[key]).rstrip() for key in FIELDS)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(TEMPLATE.format(front=front), encoding="utf-8")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command")
    make = sub.add_parser("new", help="create a record for a new experiment")
    make.add_argument("package", choices=PACKAGES)
    make.add_argument("name", help="short-name-with-dashes")
    make.add_argument("--author", default="")
    sub.add_parser("check", help="validate every record")
    args = parser.parse_args(argv)

    if args.command == "new":
        print(new(args.package, args.name, args.author))
        return 0
    found = problems()
    if args.command == "check" or found:
        print("\n".join(found) if found else "experiment records OK")
        if found:
            return 1
        return 0
    text, count = table()
    print(text)
    print("\n{} experiments".format(count))
    return 0


if __name__ == "__main__":
    sys.exit(main())
