"""Label-free check of the combiner: does it recover possession where it was not told?

    python -m saha.tools.selfcheck                                   # sweep cost_switch
    python -m saha.tools.selfcheck --param cost_to_dead --values 10 20 35 60
    python -m saha.tools.selfcheck --config configs/<experiment>.toml

A *touch* is a run of at least MIN_TOUCH frames in a graded half where the ball is tracked, slow and
within TOUCH_M of one player: a team at the ball. Slow means under TOUCH_SPEED, measured from the
ball's positions, because the ATD's ball speed channel does not follow them (DECISIONS D015).
Touches are numbered in time and split into two folds, even and odd. For each fold and each value,
every evidence source loses its p_home opinion on that fold's touches, the match is decoded with the
value, and a held-out touch counts as recovered when most of its frames come out as the touching
team (D counts as wrong). The score is the share recovered. The match-wide statistics of the full
decode are printed beside it, because a setting that wins on touches by inventing possessions should
not win (docs/METRIC_NOTES.md).

No ground truth is used anywhere, so this may inform combiner settings (CLAUDE.md, rule 2). Its
touches come from the same signal as the nearest-player evidence, though, so on its own it favours
settings that follow that evidence closely: it is a sanity check, not the selector (experiment
20261104_ws1_cost-switch-sweep-1). It reads the evidence files of the last pipeline run and prints
aggregates only.
"""

import argparse
import dataclasses
import sys

import numpy as np

from saha.combine.decode import decode
from saha.common import alignment, evidence, frame_table, kinematics, paths, submission
from saha.pipeline import DEFAULT_CONFIG, load_config

TOUCH_M = 1.5        # metres between the ball and the nearest player
TOUCH_SPEED = 5.0    # m/s; a faster ball is passing or flying, not at someone's feet
MIN_TOUCH = 6        # frames (0.24 s)
DEFAULT_VALUES = (5.0, 10.0, 20.0, 35.0, 60.0)
PARAMS = ("cost_switch", "cost_to_dead", "cost_from_dead", "prior_dead", "eps")


def at_ball(table):
    """(mask, team, counts): the frames with a team at a slow, tracked ball, the nearest team on
    every frame, and how many graded frames pass each condition in turn (for the report)."""
    ball = table.ball_xy_m()
    distance, team = kinematics.nearest_player(ball, table.players_xy_m(), table.slot_team)
    speed = kinematics.displacement_speed(ball)
    seen = table.graded & table.ball_visible
    near = seen & (distance <= TOUCH_M)
    mask = near & (speed <= TOUCH_SPEED)
    counts = {"graded": int(table.graded.sum()), "ball tracked": int(seen.sum()),
              "player within {:g} m".format(TOUCH_M): int(near.sum()),
              "ball under {:g} m/s".format(TOUCH_SPEED): int(mask.sum())}
    return mask, team, counts


def touch_runs(mask, team):
    """[(start, stop, team)]: maximal runs of at least MIN_TOUCH masked frames with one team,
    inside the graded halves."""
    team = np.where(mask, team, -1)
    out = []
    for _, start, stop in alignment.half_slices():
        window = team[start:stop]
        breaks = np.flatnonzero(window[1:] != window[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [len(window)]))
        for a, b in zip(bounds[:-1], bounds[1:]):
            if window[a] >= 0 and b - a >= MIN_TOUCH:
                out.append((start + int(a), start + int(b), int(window[a])))
    return out


def without_home(sources, mask):
    """The sources with every p_home opinion removed on the masked frames."""
    out = []
    for source, scale in sources:
        p_home = source.p_home.copy()
        p_home[mask] = np.nan
        out.append((evidence.Evidence(source.name, source.p_dead, p_home, source.weight,
                                      source.meta), scale))
    return out


def run(config=DEFAULT_CONFIG, param="cost_switch", values=DEFAULT_VALUES, table=None):
    """One dict per value: recovered share, frame agreement and match-wide statistics."""
    base, entries = load_config(config)
    sources = []
    for entry in entries:
        sources.append((evidence.read_evidence(entry["name"]), float(entry.get("scale", 1.0))))
    table = table or frame_table.load()
    at, at_team, counts = at_ball(table)
    found = touch_runs(at, at_team)
    folds = [[t for i, t in enumerate(found) if i % 2 == fold] for fold in (0, 1)]
    masks = []
    for fold in folds:
        mask = np.zeros(table.n_frames, dtype=bool)
        for start, stop, _ in fold:
            mask[start:stop] = True
        masks.append(mask)

    rows = []
    for value in values:
        params = dataclasses.replace(base, **{param: float(value)})
        stats = submission.match_stats(decode(sources, params))
        hits = frames = agree = 0
        for fold, mask in zip(folds, masks):
            labels = decode(without_home(sources, mask), params)
            for start, stop, team in fold:
                same = int((labels[start:stop] == team).sum())
                hits += same * 2 > stop - start
                frames += stop - start
                agree += same
        rows.append({"value": float(value), "touches": len(found), "counts": counts,
                     "recovered": hits / len(found) if found else float("nan"),
                     "frame_agreement": agree / frames if frames else float("nan"), **stats})
    return base, rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--param", default="cost_switch", choices=PARAMS)
    parser.add_argument("--values", type=float, nargs="+", default=list(DEFAULT_VALUES))
    parser.add_argument("--data", help="data bundle root (default: MTAH_DATA_DIR or ./data)")
    args = parser.parse_args(argv)
    if args.data:
        paths.use_data_dir(args.data)
    try:
        base, rows = run(args.config, args.param, args.values)
    except evidence.EvidenceError as error:
        print("{}\nRun python -m saha.pipeline first: the self-check reads its evidence files."
              .format(error))
        return 1

    current = getattr(base, args.param)
    if rows:
        print("touch frames: " + ", then ".join("{} {}".format(n, name)
                                               for name, n in rows[0]["counts"].items()))
    print("self-check of {} ({} touches held out in two folds; config value {:g})".format(
        args.param, rows[0]["touches"] if rows else 0, current))
    print("{:>8}  {:>9}  {:>9}  {:>11}  {:>6}  {:>8}  {:>8}  {:>5}".format(
        "value", "recovered", "frames", "possessions", "mean s", "median s", "short<2s", "D"))
    for row in rows:
        print("{:>8g}  {:>9.3f}  {:>9.3f}  {:>11}  {:>6.1f}  {:>8.1f}  {:>8}  {:>5.1%}".format(
            row["value"], row["recovered"], row["frame_agreement"], row["n_possessions"],
            row["mean_duration_s"], row["median_duration_s"], row["short_possessions_lt_2s"],
            row["d_rate"]))
    print("Read the recovered share together with the possession counts: a value that gains a "
          "few touches\nby inventing many short possessions is not better (docs/METRIC_NOTES.md).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
