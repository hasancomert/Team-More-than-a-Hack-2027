"""The evidence interface: how every package hands its opinion to the combiner.

One file per producer, cache/evidence/<name>.parquet (gitignored), exactly 199,319 rows:

    frame    int32    0 .. 199318, in order
    p_dead   float32  P(ball is dead); NaN = no opinion
    p_home   float32  P(home team has the ball | ball alive); NaN = no opinion
    weight   float32  >= 0, how much to trust this row; 0 = no opinion

Exactly these four columns. Changing the interface is a team decision (DECISIONS.md).

Write files with write_evidence(), which validates first and quantises probabilities and weights
to 1e-4 so that last-bit floating-point differences between machines cannot reach the combiner.

    python -m saha.common.evidence            # validate every file in cache/evidence
    python -m saha.common.evidence deadball   # just one

The validator prints coverage statistics, never rows.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from saha.common import alignment, paths
from saha.common.constants import N_FRAMES

COLUMNS = ("frame", "p_dead", "p_home", "weight")
SCHEMA_VERSION = 1
QUANTUM = 1e-4
META_KEY = b"saha_evidence"
NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class EvidenceError(ValueError):
    pass


@dataclass
class Evidence:
    name: str
    p_dead: np.ndarray   # (N,) float32
    p_home: np.ndarray   # (N,) float32
    weight: np.ndarray   # (N,) float32
    meta: dict

    def content_hash(self):
        return content_hash(self.p_dead, self.p_home, self.weight)


def empty(n_frames=N_FRAMES):
    """(p_dead, p_home, weight) meaning 'no opinion anywhere' -- a valid starting point."""
    return (np.full(n_frames, np.nan, dtype=np.float32),
            np.full(n_frames, np.nan, dtype=np.float32),
            np.zeros(n_frames, dtype=np.float32))


def quantise(values):
    """Round to QUANTUM and store as float32; NaN stays NaN."""
    values = np.asarray(values, dtype=np.float64)
    return (np.round(values / QUANTUM) * QUANTUM).astype(np.float32)


def content_hash(p_dead, p_home, weight):
    """SHA256 of the values, independent of parquet encoding. Compare it across machines."""
    digest = hashlib.sha256()
    for array in (p_dead, p_home, weight):
        digest.update(np.ascontiguousarray(array, dtype="<f4").tobytes())
    return digest.hexdigest()


def problems(frame, p_dead, p_home, weight, n_frames=N_FRAMES):
    """Every way the arrays break the interface, as readable strings. Empty means valid."""
    out = []
    for name, array in (("frame", frame), ("p_dead", p_dead), ("p_home", p_home),
                        ("weight", weight)):
        if len(array) != n_frames:
            out.append("{}: {} rows, expected {}".format(name, len(array), n_frames))
    if out:
        return out
    if not np.issubdtype(np.asarray(frame).dtype, np.integer):
        out.append("frame: dtype {} is not an integer type".format(np.asarray(frame).dtype))
    elif not np.array_equal(frame, np.arange(n_frames)):
        out.append("frame: not 0..{} in order".format(n_frames - 1))
    for name, array in (("p_dead", p_dead), ("p_home", p_home), ("weight", weight)):
        if np.asarray(array).dtype != np.float32:
            out.append("{}: dtype {} is not float32".format(name, np.asarray(array).dtype))
    for name, array in (("p_dead", p_dead), ("p_home", p_home)):
        finite = np.isfinite(array)
        if np.isinf(array).any():
            out.append("{}: infinite values".format(name))
        outside = finite & ((array < 0) | (array > 1))
        if outside.any():
            out.append("{}: {} values outside [0, 1]".format(name, int(outside.sum())))
    if not np.isfinite(weight).all():
        out.append("weight: NaN or infinite values (use 0 for no opinion)")
    elif (weight < 0).any():
        out.append("weight: {} negative values".format(int((weight < 0).sum())))
    return out


def stats(p_dead, p_home, weight):
    """Coverage on graded frames, as fractions. Aggregates only."""
    graded = alignment.graded_mask(len(weight))
    has_dead = np.isfinite(p_dead) & (weight > 0)
    has_home = np.isfinite(p_home) & (weight > 0)
    either = has_dead | has_home
    return {
        "dead_coverage": float(has_dead[graded].mean()),
        "home_coverage": float(has_home[graded].mean()),
        "mean_weight": float(weight[graded & either].mean()) if (graded & either).any() else 0.0,
        "mean_p_dead": float(np.nanmean(np.where(has_dead, p_dead, np.nan)[graded]))
        if (has_dead & graded).any() else None,
        "extreme": int(((p_dead == 0) | (p_dead == 1) | (p_home == 0) | (p_home == 1)).sum()),
    }


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=paths.REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_evidence(name, p_dead, p_home, weight, producer=None, params=None):
    """Validate, quantise and write cache/evidence/<name>.parquet. Returns the path."""
    if not NAME_PATTERN.match(name):
        raise EvidenceError("evidence name {!r} must match {}".format(name, NAME_PATTERN.pattern))
    p_dead, p_home, weight = quantise(p_dead), quantise(p_home), quantise(weight)
    frame = np.arange(len(weight), dtype=np.int32)
    found = problems(frame, p_dead, p_home, weight)
    if found:
        raise EvidenceError("evidence {!r} is invalid:\n  {}".format(name, "\n  ".join(found)))

    info = {"name": name, "schema_version": SCHEMA_VERSION, "producer": producer,
            "params": params or {}, "git_commit": _git_commit(),
            "content_sha256": content_hash(p_dead, p_home, weight)}
    table = pa.table({"frame": frame, "p_dead": p_dead, "p_home": p_home, "weight": weight})
    table = table.replace_schema_metadata({META_KEY: json.dumps(info, sort_keys=True,
                                                                default=str).encode()})
    path = paths.evidence_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")
    return path


def read_evidence(name_or_path, check=True):
    """Load one evidence file; with check=True an invalid file raises EvidenceError."""
    path = Path(name_or_path)
    if path.suffix != ".parquet":
        path = paths.evidence_path(str(name_or_path))
    if not path.is_file():
        raise EvidenceError("no evidence file at {}".format(path))
    table = pq.read_table(path)
    missing = [c for c in COLUMNS if c not in table.column_names]
    extra = [c for c in table.column_names if c not in COLUMNS]
    if check and (missing or extra):
        raise EvidenceError("{}: columns must be exactly {} (missing {}, extra {})".format(
            path.name, list(COLUMNS), missing, extra))
    arrays = {c: table.column(c).to_numpy() for c in COLUMNS if c in table.column_names}
    if check:
        found = problems(arrays["frame"], arrays["p_dead"], arrays["p_home"], arrays["weight"])
        if found:
            raise EvidenceError("{} is invalid:\n  {}".format(path.name, "\n  ".join(found)))
    meta = json.loads((table.schema.metadata or {}).get(META_KEY, b"{}"))
    return Evidence(path.stem, arrays["p_dead"], arrays["p_home"], arrays["weight"], meta)


def validate_file(path):
    """(ok, message lines) for one file, for the CLI and the pipeline."""
    try:
        evidence = read_evidence(path)
    except (EvidenceError, OSError, pa.ArrowException) as error:
        return False, ["FAIL  {}".format(Path(path).name), "      {}".format(error)]
    s = stats(evidence.p_dead, evidence.p_home, evidence.weight)
    lines = ["OK    {:<16} graded coverage: dead {:5.1%}  home {:5.1%}   mean weight {:.3f}"
             "   sha {}".format(evidence.name, s["dead_coverage"], s["home_coverage"],
                               s["mean_weight"], evidence.content_hash()[:12])]
    if s["extreme"]:
        lines.append("      note: {} probabilities are exactly 0 or 1; the combiner clips them"
                     .format(s["extreme"]))
    if not s["dead_coverage"] and not s["home_coverage"]:
        lines.append("      note: no opinion on any graded frame (a stub?)")
    return True, lines


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validate evidence files against the interface.")
    parser.add_argument("names", nargs="*", help="evidence names or paths (default: all)")
    args = parser.parse_args(argv)
    targets = args.names or sorted(p.stem for p in paths.evidence_dir().glob("*.parquet"))
    if not targets:
        print("no evidence files in {}".format(paths.evidence_dir()))
        return 1
    failed = 0
    for target in targets:
        ok, lines = validate_file(target)
        failed += not ok
        print("\n".join(lines))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
