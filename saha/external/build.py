"""WS5: build the external validation sets.

    python -m saha.external.build                  -> cache/external/dfl/J03WPY.npz
    python -m saha.external.build --all            all seven DFL/IDSSE matches (~5 GB download)
    python -m saha.external.build --matches J03WMX J03WN1
    python -m saha.external.score                  then score the pipeline on them

Downloads and converts DFL/Sportec open tracking data (saha.external.dfl), then prints, per
match, the same aggregate statistics of its labels as for the public ground truth of our match.
That comparison answers TEAM.md's question "do their possessions mean what ours mean?" (similar
durations, D rate, short spells). It only checks the definitions agree; nothing is tuned on it.
"""

import argparse

import numpy as np

from saha.common import submission
from saha.common.constants import D, FPS
from saha.external import dfl


def label_stats(codes):
    """Aggregate shape of a label array (X = not scored), the scorer's way."""
    pm = submission.metrics()
    codes = np.asarray(codes, dtype=np.int8)
    runs = pm.valid_runs(codes)
    scored = sum(stop - start for start, stop in runs)
    spells = pm.possession_spells(codes, runs)
    lengths = np.array([b - a for a, b, _ in spells], dtype=np.float64) / FPS
    dead = [b - a for a, b, c in pm.segments(codes, runs) if c == D]
    alive = np.isin(codes, (0, 1))
    return {
        "scored_minutes": scored / FPS / 60.0,
        "possessions": len(spells),
        "possessions_per_minute": len(spells) / (scored / FPS / 60.0) if scored else 0.0,
        "mean_possession_s": float(lengths.mean()) if len(lengths) else 0.0,
        "median_possession_s": float(np.median(lengths)) if len(lengths) else 0.0,
        "share_possessions_lt_2s": float((lengths < 2).mean()) if len(lengths) else 0.0,
        "d_rate": float((codes == D).sum() / scored) if scored else 0.0,
        "stoppages": len(dead),
        "median_stoppage_s": float(np.median(dead) / FPS) if dead else 0.0,
        "home_share": float((codes == 0).sum() / alive.sum()) if alive.any() else 0.0,
    }


ROWS = (("scored_minutes", "scored minutes", ".1f"), ("possessions", "possessions", "d"),
        ("possessions_per_minute", "possessions per minute", ".2f"),
        ("mean_possession_s", "mean possession s", ".1f"),
        ("median_possession_s", "median possession s", ".1f"),
        ("share_possessions_lt_2s", "possessions < 2 s", ".1%"),
        ("d_rate", "D rate", ".1%"), ("stoppages", "stoppages", "d"),
        ("median_stoppage_s", "median stoppage s", ".1f"), ("home_share", "home share", ".1%"))


def print_comparison(columns):
    names = list(columns)
    print("  {:<24}".format("") + "".join("{:>14}".format(n) for n in names))
    for key, label, spec in ROWS:
        print("  {:<24}".format(label) +
              "".join("{:>14}".format(format(columns[n][key], spec)) for n in names))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matches", nargs="*", help="DFL match ids (default: {})".format(
        " ".join(dfl.DEFAULT_MATCHES)))
    parser.add_argument("--all", action="store_true", help="all seven DFL matches")
    parser.add_argument("--rebuild", action="store_true", help="convert again")
    args = parser.parse_args(argv)
    matches = sorted(dfl.MATCHES) if args.all else (args.matches or list(dfl.DEFAULT_MATCHES))

    columns = {}
    for match_id in matches:
        match = dfl.prepare(match_id, rebuild=args.rebuild)
        unlabelled = float((match.labels[match.half > 0] == 3).mean())
        print("{}: {} frames, {} slots, {:.2%} of in-half frames unlabelled".format(
            match.name, match.n_frames, len(match.slot_team), unlabelled))
        columns[match_id] = label_stats(match.labels)
    # The public 5 minutes of our match, as aggregates only (CLAUDE.md): never a tuning target.
    columns["public 5 min"] = label_stats(submission.public_gt())
    print("\nlabel statistics (definition check, not a target):")
    print_comparison(columns)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
