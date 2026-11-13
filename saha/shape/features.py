"""Team-shape features from player positions only, never the ball.

The same code runs on our match and on WS5's degraded DFL matches: it reads a FrameTable.

- **Orientation.** In each graded half, take each team's player nearest the low-x goal, with the
  median over the half. For the team that defends that goal, this is its goalkeeper or last
  defender; for the other team, it is a forward, who stays further out to be onside. So the team
  with the lower median defends the low-x goal and attacks towards high x. This is an algorithm,
  not a constant (D009).
- **Per team and frame.** Every value is taken in that team's own attacking frame: x' is metres
  from the halfway line, positive in the opponent's half.
    - the centroid;
    - the 15th and 85th percentiles of x' (the deep and the advanced players);
    - the length and width (standard deviations);
    - the share of players in the opponent's half;
    - the mean velocity towards the opponent's goal;
    - the mean speed.
- **Context.** Each value is smoothed over 1 s and over 4 s. The change from the 2 s before to the
  2 s after is added as well.

`pair_features(table)` returns `(features, ok, names)`. `features` has shape (N, 2, K): team 0 is
home (H) and team 1 is away (A). `ok` marks the graded frames where every feature is finite.
"""

import warnings

import numpy as np

from saha.common.constants import FPS
from saha.common.frame_table import PITCH_LENGTH_M

VEL_HALF = FPS // 2           # frames: a player's velocity over +/- 0.5 s
MAX_SPEED = 12.0              # m/s: faster than any sprint, so a tracking jump, not a run
WINDOWS = (FPS, 4 * FPS)      # centred moving averages: 1 s and 4 s
DELTA = 2 * FPS               # change from the 2 s before to the 2 s after
BASE = ("centroid", "deep", "front", "length", "width", "in_opp_half", "vel_fwd", "speed")


def half_runs(half):
    """[(start, stop)] of the runs of each half number > 0, in order."""
    half = np.asarray(half)
    out = []
    for number in sorted(set(half[half > 0].tolist())):
        index = np.flatnonzero(half == number)
        breaks = np.flatnonzero(np.diff(index) > 1)
        starts = np.concatenate(([index[0]], index[breaks + 1]))
        stops = np.concatenate((index[breaks] + 1, [index[-1] + 1]))
        out += [(int(a), int(b)) for a, b in zip(starts, stops)]
    return sorted(out)


def deepest_medians(table):
    """[(start, stop, median deepest x of team 0, of team 1)] per graded half, metres."""
    x = table.players_xy_m()[..., 0]
    out = []
    for start, stop in half_runs(table.half):
        medians = []
        for team in (0, 1):
            block = x[start:stop][:, table.slot_team == team]
            present = np.isfinite(block).any(axis=1)
            low = np.where(present, np.nanmin(np.where(np.isfinite(block), block, np.inf),
                                              axis=1), np.nan)
            medians.append(float(np.nanmedian(low)) if present.any() else np.nan)
        out.append((start, stop, medians[0], medians[1]))
    return out


def directions(table):
    """(N, 2): +1 or -1, the direction of attack along x of each team; 0 outside the halves."""
    out = np.zeros((table.n_frames, 2))
    for start, stop, home, away in deepest_medians(table):
        home_attacks_high = home < away
        out[start:stop, 0] = 1.0 if home_attacks_high else -1.0
        out[start:stop, 1] = -out[start:stop, 0]
    return out


def velocity_x(x):
    """(N, S) m/s along x from positions +/- VEL_HALF frames; NaN near gaps, edges and jumps."""
    v = np.full_like(x, np.nan)
    k = VEL_HALF
    v[k:-k] = (x[2 * k:] - x[:-2 * k]) / (2.0 * k / FPS)
    v[np.abs(v) > MAX_SPEED] = np.nan
    return v


def moving_mean(values, width):
    """Centred moving average over `width` frames along axis 0, NaN-aware."""
    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.0)
    pad = np.zeros((1,) + values.shape[1:])
    csum = np.cumsum(np.concatenate((pad, filled)), axis=0)
    ccnt = np.cumsum(np.concatenate((pad, finite)), axis=0)
    n = len(values)
    index = np.arange(n)
    lo = np.clip(index - width // 2, 0, n)
    hi = np.clip(index + width // 2 + 1, 0, n)
    total, count = csum[hi] - csum[lo], ccnt[hi] - ccnt[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def shifted(values, offset):
    """values[t + offset] at t, NaN past the ends."""
    out = np.full_like(values, np.nan)
    if offset > 0:
        out[:-offset] = values[offset:]
    elif offset < 0:
        out[-offset:] = values[:offset]
    else:
        out[:] = values
    return out


def base_features(table):
    """(N, 2, len(BASE)) per-team features, oriented; NaN where a team has no tracked player."""
    xy = table.players_xy_m()
    x, y = xy[..., 0], xy[..., 1]
    direction = directions(table)
    vx = velocity_x(x)
    speed = table.players_speed.astype(np.float64)
    out = np.full((table.n_frames, 2, len(BASE)), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # all-NaN rows: no tracked player
        for team in (0, 1):
            cols = table.slot_team == team
            d = direction[:, team][:, None]
            xo = d * (x[:, cols] - PITCH_LENGTH_M / 2.0)
            present = np.isfinite(xo)
            count = present.sum(axis=1)
            out[:, team, 0] = np.nanmean(xo, axis=1)
            out[:, team, 1] = np.nanpercentile(xo, 15, axis=1)
            out[:, team, 2] = np.nanpercentile(xo, 85, axis=1)
            out[:, team, 3] = np.nanstd(xo, axis=1)
            out[:, team, 4] = np.nanstd(y[:, cols], axis=1)
            out[:, team, 5] = np.where(count > 0, (xo > 0).sum(axis=1) / np.maximum(count, 1),
                                       np.nan)
            out[:, team, 6] = np.nanmean(d * vx[:, cols], axis=1)
            out[:, team, 7] = np.nanmean(np.where(present, speed[:, cols], np.nan), axis=1)
    out[table.half <= 0] = np.nan
    return out


def pair_features(table):
    """(features (N, 2, K), ok (N,), names): smoothed and change features of both teams."""
    base = base_features(table)
    parts, names = [], []
    for width in WINDOWS:
        parts.append(moving_mean(base, width))
        names += ["{}_{}s".format(b, width // FPS) for b in BASE]
    window = moving_mean(base, DELTA)
    parts.append(shifted(window, DELTA // 2) - shifted(window, -DELTA // 2))
    names += ["{}_delta".format(b) for b in BASE]
    features = np.concatenate(parts, axis=2)
    ok = np.isfinite(features).all(axis=(1, 2)) & (table.half > 0)
    return features, ok, names
