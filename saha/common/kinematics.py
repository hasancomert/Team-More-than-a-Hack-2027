"""Speeds and nearest players from positions, so that every package measures them the same way.

    from saha.common import frame_table, kinematics
    table = frame_table.load()
    ball = table.ball_xy_m()
    speed = kinematics.displacement_speed(ball)          # m/s, NaN where it cannot be measured
    distance, team = kinematics.nearest_player(ball, table.players_xy_m(), table.slot_team)

The ATD's own speed channel follows the displacements for players (m/s, DECISIONS D012), but the
ball's does not (D015), so the ball's speed always comes from here.
"""

import numpy as np

FPS = 25


def displacement_speed(xy_m, span=5):
    """Speed in m/s from positions in metres, over a centred window of `span` frames (odd, >= 3).

    xy_m is (N, ..., 2). Frame t gets |p[t + h] - p[t - h]| / (2h / FPS), with h = span // 2.
    It is NaN where either end is missing, and on the first and last h frames.
    """
    if span < 3 or span % 2 == 0:
        raise ValueError("span must be odd and at least 3, not {}".format(span))
    half = span // 2
    speed = np.full(xy_m.shape[:-1], np.nan)
    if len(xy_m) > 2 * half:
        moved = xy_m[2 * half:] - xy_m[:-2 * half]
        speed[half:len(xy_m) - half] = np.hypot(moved[..., 0], moved[..., 1]) * FPS / (2 * half)
    return speed


def nearest_player(ball_m, players_m, slot_team):
    """(distance in metres, team) of the tracked player nearest the ball, per frame.

    ball_m is (N, 2), players_m (N, S, 2), slot_team (S,). Frames without a ball or without a
    tracked player get distance inf and team -1. The arithmetic is that of saha.nearest and the
    official baseline (squared distances, first minimum wins), so the team always agrees with
    the nearest-player evidence.
    """
    dx = players_m[..., 0] - ball_m[:, None, 0]
    dy = players_m[..., 1] - ball_m[:, None, 1]
    squared = dx * dx + dy * dy
    squared = np.where(np.isfinite(squared), squared, np.inf)
    slot = np.argmin(squared, axis=1)
    closest = np.sqrt(squared[np.arange(len(slot)), slot])
    return closest, np.where(np.isfinite(closest), slot_team[slot], -1).astype(np.int8)
