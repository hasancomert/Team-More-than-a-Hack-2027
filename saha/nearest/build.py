"""The official nearest-player baseline, re-expressed as evidence.

    python -m saha.nearest.build          -> cache/evidence/nearest.parquet

Same information as starter/baseline.py, handed to the combiner as probabilities instead of
labels, so that the pipeline has one real source from day one:

  p_home  where the ball is tracked: P_TEAM for the team of the nearest tracked player
  p_dead  P_DEAD_NO_BALL where no ball is tracked within +/- DEAD_WINDOW frames (the official
          baseline's "D when the window has no ball"), else P_DEAD_BALL_SEEN

Distances are in metres (pitch from games.py), ties go to the lower slot, as in the official code.
WS3 (ball visible) and WS2 (dead ball) exist to beat both halves of this; it stays as the
reference point and as the fallback.
"""

import argparse

import numpy as np

from saha.common import evidence, frame_table, paths

P_TEAM = 0.8
DEAD_WINDOW = 50
P_DEAD_NO_BALL = 0.9
P_DEAD_BALL_SEEN = 0.1
NOBODY = -1


def nearest_team(ball_m, players_m, slot_team):
    """Team code (0 home, 1 away) of the tracked player nearest the ball, NOBODY if none.

    Float64 metres in, first minimum wins -- the same arithmetic as starter/baseline.py.
    """
    dx = players_m[..., 0] - ball_m[:, None, 0]
    dy = players_m[..., 1] - ball_m[:, None, 1]
    distance = dx * dx + dy * dy
    known = np.isfinite(distance)
    has = known.any(axis=1) & np.isfinite(ball_m).all(axis=1)
    slot = np.argmin(np.where(known, distance, np.inf), axis=1)
    return np.where(has, slot_team[slot], NOBODY).astype(np.int8)


def window_count(mask, window):
    """For each frame, how many frames within +/- window are True (clipped at the ends)."""
    cumulative = np.concatenate(([0], np.cumsum(mask, dtype=np.int64)))
    index = np.arange(len(mask))
    low = np.maximum(0, index - window)
    high = np.minimum(len(mask), index + window + 1)
    return cumulative[high] - cumulative[low]


def build(table):
    team = nearest_team(table.ball_xy_m(), table.players_xy_m(), table.slot_team)
    p_dead, p_home, weight = evidence.empty(table.n_frames)
    seen = team != NOBODY
    p_home[seen] = np.where(team[seen] == 0, P_TEAM, 1.0 - P_TEAM)
    p_dead[:] = np.where(window_count(seen, DEAD_WINDOW) > 0, P_DEAD_BALL_SEEN, P_DEAD_NO_BALL)
    weight[:] = 1.0
    return p_dead, p_home, weight


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", help="data bundle root (default: MTAH_DATA_DIR or ./data)")
    args = parser.parse_args(argv)
    if args.data:
        paths.use_data_dir(args.data)
    p_dead, p_home, weight = build(frame_table.load())
    path = evidence.write_evidence(
        "nearest", p_dead, p_home, weight, producer="saha.nearest.build",
        params={"P_TEAM": P_TEAM, "DEAD_WINDOW": DEAD_WINDOW,
                "P_DEAD_NO_BALL": P_DEAD_NO_BALL, "P_DEAD_BALL_SEEN": P_DEAD_BALL_SEEN})
    print("\n".join(evidence.validate_file(path)[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
