"""Inputs of the second opinion on WS2's dead calls where the ball is missing.

The same code runs on our match and on WS5's DFL matches: it reads a FrameTable and WS2's
decision. No input depends on which team is which.

- **The frames** (`population`): graded, the ball missing, inside the played part of the half
  (between WS2's first and last alive frame), and called dead by WS2.
- **Where the frame is:** seconds since the start, until the end, and the length of its gap in the
  ball's tracking, and of WS2's dead spell (each capped at 60 s).
- **The edges of the gap,** over 1 s of tracking on each side: the ball's speed, its distance from
  the halfway line and from the nearest touchline, and the nearest player's distance to it.
- **The players,** over 1 s and 4 s: the mean speed, the lower of the two teams' mean speeds, the
  mean forward speed (absolute), the teams' mean length and width, and the sum of the two teams'
  advancement; and the change of the mean speed over 2 s.
"""

import numpy as np

from saha.common.constants import FPS
from saha.common.frame_table import PITCH_LENGTH_M, PITCH_WIDTH_M
from saha.onball.features import K_BALL, velocity
from saha.shape import features as shape_features

CAP_S = 60.0        # seconds: a longer gap or dead spell says nothing more
CAP_M = 30.0        # metres: as the on-ball features
EDGE = FPS          # frames of tracking read on each side of a gap
MAX_BALL_SPEED = 40.0
PLAYER = ("speed", "slower_team_speed", "forward_speed", "length", "width", "advancement")


def names():
    """The input names, in the order of `inputs`' columns."""
    spans = ["{}_{}".format(span, part) for span in ("gap", "dead_spell")
             for part in ("since", "until", "length")]
    edge = ["{}_{}".format(side, f) for side in ("before", "after")
            for f in ("ball_speed", "ball_from_halfway", "ball_from_touchline", "nearest_player")]
    players = (["{}_{}".format(f, w) for w in ("1s", "4s") for f in PLAYER] + ["speed_change_2s"])
    return spans + edge + players


def runs(mask):
    """[(start, stop)] of the True runs of a boolean array, half-open, in order."""
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(edges[0::2], edges[1::2])]


def edge_raw(table):
    """(N, 4): the ball's speed, its distance from the halfway line and from the nearest touchline,
    and the nearest player's distance to it (any team); NaN where the ball is missing."""
    ball = table.ball_xy_m()
    players = table.players_xy_m()
    speed = np.minimum(np.linalg.norm(velocity(ball, K_BALL), axis=-1), MAX_BALL_SPEED)
    from_halfway = np.abs(ball[:, 0] - PITCH_LENGTH_M / 2.0)
    from_touchline = np.minimum(ball[:, 1], PITCH_WIDTH_M - ball[:, 1])
    dist = np.linalg.norm(ball[:, None, :] - players, axis=-1)
    nearest = np.minimum(np.where(np.isfinite(dist), dist, np.inf).min(axis=1), CAP_M)
    nearest = np.where(np.isfinite(ball[:, 0]), nearest, np.nan)
    return np.stack([speed, from_halfway, from_touchline, nearest], axis=1)


def player_inputs(table):
    """(N, 13): the six player summaries over 1 s and 4 s, and the change of the mean speed."""
    base = shape_features.base_features(table)    # centroid, deep, front, length, width,
    speed = base[:, :, 7]                          # in_opp_half, vel_fwd, speed (both teams)
    raw = np.stack([speed.mean(axis=1), speed.min(axis=1), np.abs(base[:, :, 6]).mean(axis=1),
                    base[:, :, 3].mean(axis=1), base[:, :, 4].mean(axis=1),
                    base[:, 0, 0] + base[:, 1, 0]], axis=1)
    mm = shape_features.moving_mean
    window = mm(raw[:, :1], 2 * FPS)
    change = shape_features.shifted(window, FPS) - shape_features.shifted(window, -FPS)
    return np.concatenate([mm(raw, FPS), mm(raw, 4 * FPS), change], axis=1)


def positions(spans, n):
    """(N, 3): seconds since the start, until the end, and the length of the span holding each
    frame (capped); NaN outside every span."""
    out = np.full((n, 3), np.nan)
    for a, b in spans:
        frames = np.arange(a, b)
        out[a:b, 0] = np.minimum((frames - a) / FPS, CAP_S)
        out[a:b, 1] = np.minimum((b - frames) / FPS, CAP_S)
        out[a:b, 2] = min((b - a) / FPS, CAP_S)
    return out


def nanmean_rows(block):
    finite = np.isfinite(block)
    count = finite.sum(axis=0)
    return np.where(count > 0, np.where(finite, block, 0.0).sum(axis=0) / np.maximum(count, 1),
                    np.nan)


def inputs(table, ws2_dead):
    """(index of the population's frames, (M, 27) float32 inputs). `ws2_dead`: (N,) bool, WS2's
    decision. The inputs are rounded to float32, as the model was trained, so that a last-bit
    difference between machines almost never reaches the model."""
    n = table.n_frames
    ws2_dead = np.asarray(ws2_dead, dtype=bool)
    visible = table.ball_visible
    raw = edge_raw(table)
    gap_spans, dead_spans = [], []
    edges = np.full((n, 8), np.nan)
    played = np.zeros(n, dtype=bool)
    for start, stop in shape_features.half_runs(table.half):
        alive = np.flatnonzero(~ws2_dead[start:stop])
        if len(alive):
            played[start + alive[0]:start + alive[-1] + 1] = True
        for g0, g1 in runs(~visible[start:stop]):
            g0, g1 = g0 + start, g1 + start
            gap_spans.append((g0, g1))
            before, after = raw[max(start, g0 - EDGE):g0], raw[g1:min(stop, g1 + EDGE)]
            edges[g0:g1, :4] = nanmean_rows(before) if len(before) else np.nan
            edges[g0:g1, 4:] = nanmean_rows(after) if len(after) else np.nan
        dead_spans += [(a + start, b + start) for a, b in runs(ws2_dead[start:stop])]
    index = np.flatnonzero(table.graded & ~visible & played & ws2_dead)
    x = np.concatenate([positions(gap_spans, n)[index], positions(dead_spans, n)[index],
                        edges[index], player_inputs(table)[index]], axis=1)
    return index, x.astype(np.float32)
