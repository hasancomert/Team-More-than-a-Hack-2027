"""Ball-and-player features for the team in possession where the ball is tracked.

The same code runs on our match and on WS5's DFL matches: it reads a FrameTable.

- **Per team T**, in T's attacking frame (`saha.shape.features.directions`):
    - the distance from the ball to T's nearest and second-nearest player (capped at 30 m);
    - T's players within 5 m and within 10 m;
    - the nearest player's speed towards the ball;
    - the ball's x, and its velocity towards T's attacking goal.
- **Global:** the ball's speed and acceleration, and its distance from the halfway line and from
  the middle of the pitch's width.
- **Context.** Each team feature is averaged over a centred 0.5 s and a centred 2 s window, the
  second before and the second after. The global features are taken raw and over a centred 1 s.
  The averages skip frames without a ball, so a window near a gap uses the frames it has.
- **The two teams' shape** (`saha.shape.features`, home minus away) is added as context.

`design(table)` returns `(x, x_swapped, ok, n_shape)`. `x` is home first, `x_swapped` is the same
frames with the teams exchanged, and `ok` marks the frames where every input is finite. A model
`m` is made symmetric as `p = (m(x) + 1 - m(x_swapped)) / 2`.
"""

import numpy as np

from saha.common.constants import FPS
from saha.common.frame_table import PITCH_LENGTH_M, PITCH_WIDTH_M
from saha.shape import features as shape_features

K_PLAYER = 6        # frames: a player's velocity over +/- 0.24 s
K_BALL = 3          # frames: the ball's velocity over +/- 0.12 s
CAP_M = 30.0        # metres: a farther player says nothing more about the ball
MAX_PLAYER_SPEED = 12.0
TEAM = ("nearest", "second", "within5", "within10", "approach", "ball_x", "ball_vx")
GLOBAL = ("ball_speed", "ball_acc", "from_halfway", "from_middle")


def velocity(xy, k):
    """(N, ..., 2) m/s from positions +/- k frames; NaN near gaps and at the ends."""
    v = np.full_like(xy, np.nan)
    v[k:-k] = (xy[2 * k:] - xy[:-2 * k]) / (2.0 * k / FPS)
    return v


def frame_features(table):
    """(team (N, 2, len(TEAM)), global (N, len(GLOBAL))): raw per-frame features."""
    ball = table.ball_xy_m()
    players = table.players_xy_m()
    direction = shape_features.directions(table)
    pv = velocity(players, K_PLAYER)
    with np.errstate(invalid="ignore"):
        pv[np.linalg.norm(pv, axis=-1) > MAX_PLAYER_SPEED] = np.nan
    bv = velocity(ball, K_BALL)
    speed = np.linalg.norm(bv, axis=-1)
    acc = np.linalg.norm(velocity(bv, K_BALL), axis=-1)
    rel = ball[:, None, :] - players
    dist = np.linalg.norm(rel, axis=-1)
    n = table.n_frames
    rows = np.arange(n)
    team = np.full((n, 2, len(TEAM)), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        for t in (0, 1):
            cols = np.flatnonzero(table.slot_team == t)
            d = dist[:, cols]
            dd = np.where(np.isfinite(d), d, np.inf)
            order = np.argsort(dd, axis=1, kind="stable")
            first = dd[rows, order[:, 0]]
            second = dd[rows, order[:, 1]] if len(cols) > 1 else np.full(n, np.inf)
            has = np.isfinite(first)
            team[:, t, 0] = np.where(has, np.minimum(first, CAP_M), np.nan)
            team[:, t, 1] = np.where(has, np.minimum(np.where(np.isfinite(second), second, CAP_M),
                                                     CAP_M), np.nan)
            team[:, t, 2] = np.where(has, (d <= 5.0).sum(axis=1), np.nan)
            team[:, t, 3] = np.where(has, (d <= 10.0).sum(axis=1), np.nan)
            j = cols[order[:, 0]]
            r = rel[rows, j]
            unit = r / np.linalg.norm(r, axis=-1, keepdims=True)
            team[:, t, 4] = np.where(has, (pv[rows, j] * unit).sum(axis=-1), np.nan)
            team[:, t, 5] = direction[:, t] * (ball[:, 0] - PITCH_LENGTH_M / 2.0)
            team[:, t, 6] = direction[:, t] * bv[:, 0]
    glob = np.stack([np.minimum(speed, 40.0), np.minimum(acc, 400.0),
                     np.abs(ball[:, 0] - PITCH_LENGTH_M / 2.0),
                     np.abs(ball[:, 1] - PITCH_WIDTH_M / 2.0)], axis=1)
    return team, glob


def context(team, glob):
    """(team context (N, 2, 4 * len(TEAM)), global context (N, 2 * len(GLOBAL)))."""
    mm = shape_features.moving_mean
    second = mm(team, FPS)
    parts = [mm(team, FPS // 2), mm(team, 2 * FPS),
             shape_features.shifted(second, -FPS // 2),      # the second before
             shape_features.shifted(second, FPS // 2)]       # the second after
    return np.concatenate(parts, axis=2), np.concatenate([glob, mm(glob, FPS)], axis=1)


def names():
    """The input names, in the order of `design`'s columns."""
    team = ["{}_{}".format(f, w) for w in ("0.5s", "2s", "before", "after") for f in TEAM]
    glob = list(GLOBAL) + ["{}_1s".format(g) for g in GLOBAL]
    shape = (["{}_{}s".format(b, w // FPS) for w in shape_features.WINDOWS
              for b in shape_features.BASE] +
             ["{}_delta".format(b) for b in shape_features.BASE])
    return (["home_" + t for t in team] + ["away_" + t for t in team] + glob +
            ["shape_{} (home minus away)".format(s) for s in shape])


def design(table, shape=None):
    """(x, x_swapped, ok, n_shape): model inputs for every frame (float64), and the frames where
    every input is finite. `shape` is `saha.shape.features.pair_features(table)` if already made.
    """
    tf, gf = context(*frame_features(table))
    sf = (shape or shape_features.pair_features(table))[0]
    sdiff = sf[:, 0, :] - sf[:, 1, :]
    x = np.concatenate([tf[:, 0, :], tf[:, 1, :], gf, sdiff], axis=1)
    xs = np.concatenate([tf[:, 1, :], tf[:, 0, :], gf, -sdiff], axis=1)
    ok = np.isfinite(x).all(axis=1)
    return x, xs, ok, sdiff.shape[1]
