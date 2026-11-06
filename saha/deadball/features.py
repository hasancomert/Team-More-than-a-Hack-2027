"""Per-frame features computed from the tracking, all indexed by video frame.

Everything here is collective or ball-relative, because the tracking has no player identities:
a slot is a tracklet, not a person.
"""

import warnings

import numpy as np

from saha.common.frame_table import AWAY, HOME

FPS = 25


def to_metres(xy, pitch):
    """(..., 2) normalised pitch coordinates -> metres."""
    return xy * np.asarray(pitch, dtype=np.float32)


def nearest_by_team(ball_m, players_m, team):
    """(n, 2) distance from the ball to the nearest home / away player, NaN where unknown."""
    out = np.full((len(ball_m), 2), np.nan, dtype=np.float32)
    delta = players_m - ball_m[:, None, :]
    dist = np.hypot(delta[..., 0], delta[..., 1])
    for side in (HOME, AWAY):
        sub = dist[:, team == side]
        has = ~np.all(np.isnan(sub), axis=1)
        out[has, side] = np.nanmin(sub[has], axis=1)
    return out


def central_diff(series, step):
    """d(series)/dt per second by central differences over +/- step frames; NaN at the edges."""
    n = len(series)
    out = np.full(series.shape, np.nan, dtype=np.float32)
    out[step:n - step] = (series[2 * step:] - series[:n - 2 * step]) * (FPS / (2.0 * step))
    return out


def rolling_nanmean(values, half_width):
    """Centred moving average ignoring NaN, same length as values."""
    values = np.asarray(values, dtype=np.float64)
    ok = ~np.isnan(values)
    filled = np.where(ok, values, 0.0)
    csum = np.concatenate(([0.0], np.cumsum(filled)))
    ccount = np.concatenate(([0], np.cumsum(ok)))
    index = np.arange(len(values))
    low = np.maximum(0, index - half_width)
    high = np.minimum(len(values), index + half_width + 1)
    count = ccount[high] - ccount[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, (csum[high] - csum[low]) / np.maximum(count, 1), np.nan)


def rolling_count(mask, half_width):
    """Centred count of True over +/- half_width frames."""
    csum = np.concatenate(([0], np.cumsum(mask.astype(np.int64))))
    index = np.arange(len(mask))
    low = np.maximum(0, index - half_width)
    high = np.minimum(len(mask), index + half_width + 1)
    return csum[high] - csum[low]


def attack_sign(players_m, team, blocks, pitch_length):
    """{(half, side): +1 if that side attacks towards x = pitch_length in that half, else -1}.

    Read off the data: over a half, a team's average position sits in its own half of the pitch.
    """
    out = {}
    for half, first, last in blocks:
        for side in (HOME, AWAY):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean_x = np.nanmean(players_m[first:last + 1, team == side, 0])
            out[(half, side)] = 1.0 if mean_x < pitch_length / 2.0 else -1.0
    return out


def frame_attack_sign(n, blocks, signs, side):
    """(n,) attack sign of `side` per frame; first half's sign outside the graded blocks."""
    out = np.full(n, signs[(blocks[0][0], side)], dtype=np.float32)
    for half, first, last in blocks:
        out[first:last + 1] = signs[(half, side)]
    # halftime and after the match belong to neither half; give them the closer half's sign
    out[blocks[0][2] + 1:blocks[1][1]] = signs[(blocks[1][0], side)]
    return out
