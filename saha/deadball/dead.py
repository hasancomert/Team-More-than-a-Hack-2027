"""Is the ball in play? A classifier trained on labelled external matches.

What separates a stoppage from play, in tracking with no event data:

  - the players slow down -- everyone, not one or two;
  - the tracker drops the ball: it leaves the pitch, sits in someone's hands, or gets carried to
    the restart spot. Apart from blips of a few frames, ball gaps in this data last five seconds
    or more, and most of them sit on slow players;
  - when the tracker can see the ball, it is outside the lines, or lying still where a restart is
    about to be taken.

None of these is decisive alone: the ball also vanishes in crowded boxes and aerial duels while
play goes on, and teams also walk the ball around the back. So a gradient-boosted classifier
learns what dead ball looks like from twenty SkillCorner matches, whose phases of play say when
the ball was in play (saha/deadball/train.py). Nothing about our match goes into it: it is only
applied here.
"""

import os
import pickle
import warnings
from pathlib import Path

import numpy as np

from saha.common import paths

from .features import FPS, central_diff, nearest_by_team, rolling_count, rolling_nanmean, to_metres

OOB_MARGIN = 0.5        # metres past a line before the ball counts as out
STILL_SPEED = 1.0       # m/s; slower than this the ball is lying still
CAP_S = 30.0            # cap on time-since/time-until features, seconds

# The decoder settings that go with the classifier, both from leaving SkillCorner matches out:
# the residual correlation time of its logit, and the evidence weight with the best spellF1.
EXTERNAL_TAU = 204.0
EXTERNAL_WEIGHT = 32.0

# Where the classifier lives: a file path, or "none" for no classifier at all (build.py then has
# no opinion, as the Phase 0 stub had; the tests run that way). Unset: cache/deadball/.
MODEL_ENV = "SAHA_DEADBALL_MODEL"


def model_disabled():
    return os.environ.get(MODEL_ENV) == "none"


def model_path():
    """Where train.py writes the classifier: under cache/, like everything derived from data."""
    env = os.environ.get(MODEL_ENV)
    if env and env != "none":
        return Path(env).expanduser().resolve()
    return paths.cache_dir() / "deadball" / "dead_skillcorner.pkl"


def _runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def _since_last(mask, cap):
    """Frames since mask was last True (0 where True), capped."""
    n = len(mask)
    last = np.where(mask, np.arange(n), -10 ** 9)
    last = np.maximum.accumulate(last)
    return np.minimum(np.arange(n) - last, cap).astype(np.float32)


def _until_next(mask, cap):
    return _since_last(mask[::-1], cap)[::-1]


class Signals:
    """The per-frame quantities the features and the event detectors are both built from."""

    def __init__(self, ball_xy, players, team, pitch):
        length, width = pitch
        self.length, self.width = length, width
        speed = players[..., 2]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.mean_speed = np.nanmean(speed, axis=1)
            self.fast_speed = np.nanpercentile(speed, 90, axis=1)
            self.tracked = np.sum(~np.isnan(speed), axis=1).astype(np.float32)
        self.visible = ~np.isnan(ball_xy[:, 0])
        ball_m = to_metres(ball_xy, pitch)
        self.ball_m = ball_m
        velocity = central_diff(ball_m, 2)
        self.ball_speed = np.hypot(velocity[:, 0], velocity[:, 1])
        with np.errstate(invalid="ignore"):
            self.out = self.visible & (
                (ball_m[:, 0] < -OOB_MARGIN) | (ball_m[:, 0] > length + OOB_MARGIN) |
                (ball_m[:, 1] < -OOB_MARGIN) | (ball_m[:, 1] > width + OOB_MARGIN))
            self.still = self.visible & (self.ball_speed < STILL_SPEED)
            self.line_distance = np.minimum.reduce([
                ball_m[:, 0], length - ball_m[:, 0], ball_m[:, 1], width - ball_m[:, 1]])
        players_m = to_metres(players[..., :2], pitch)
        delta = players_m - ball_m[:, None, :]
        with np.errstate(invalid="ignore"):
            self.crowd = np.sum(np.hypot(delta[..., 0], delta[..., 1]) < 10.0, axis=1)
        self.crowd = np.where(self.visible, self.crowd, np.nan).astype(np.float32)
        # At every restart the laws push the non-taking side away from the ball (9.15 m, 2 m at a
        # throw-in), so the second-nearest side's distance to the ball separates a set piece
        # being prepared from slow open play, where somebody is always pressing.
        near = nearest_by_team(ball_m, players_m, team)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.near_side = np.nanmin(near, axis=1)
            self.far_side = np.nanmax(near, axis=1)
        moved = np.full(len(ball_m), np.nan, dtype=np.float32)
        moved[:-5] = np.hypot(*(ball_m[5:] - ball_m[:-5]).T)
        self.ball_moved = moved   # metres per 0.2 s, less noisy than the frame speed
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.spread = (np.nanstd(players_m[..., 0], axis=1) +
                           np.nanstd(players_m[..., 1], axis=1))

        # ball gaps: the length of the gap each invisible frame sits in
        starts, stops = _runs(~self.visible)
        self.gap_length = np.zeros(len(ball_xy), dtype=np.float32)
        self.gap_mean_speed = np.full(len(ball_xy), np.nan, dtype=np.float32)
        for a, b in zip(starts, stops):
            self.gap_length[a:b] = b - a
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                self.gap_mean_speed[a:b] = np.nanmean(self.mean_speed[a:b])
        # still-ball runs: the length of the run each still frame sits in
        starts, stops = _runs(self.still)
        self.still_length = np.zeros(len(ball_xy), dtype=np.float32)
        for a, b in zip(starts, stops):
            self.still_length[a:b] = b - a


def dead_features(sig):
    """(n, 24) what each frame looks like, averaged over a few seconds around it."""
    def logspeed(values, half):
        return np.log(np.maximum(rolling_nanmean(values, half), 0.1))

    before = rolling_nanmean(np.concatenate(([np.nan] * 25, sig.mean_speed[:-25])), 25)
    after = rolling_nanmean(np.concatenate((sig.mean_speed[25:], [np.nan] * 25)), 25)
    cap = int(CAP_S * FPS)
    cols = [
        logspeed(sig.mean_speed, 25),
        logspeed(sig.mean_speed, 75),
        logspeed(sig.fast_speed, 25),
        after - before,
        rolling_count(sig.visible, 12) / 25.0,
        rolling_count(sig.visible, 50) / 101.0,
        rolling_count(sig.visible, 125) / 251.0,
        sig.gap_length / FPS,
        sig.gap_mean_speed,
        _since_last(sig.visible, cap) / FPS,
        _until_next(sig.visible, cap) / FPS,
        rolling_count(sig.out, 25) / 51.0,
        _since_last(sig.out, cap) / FPS,
        _until_next(sig.out, cap) / FPS,
        rolling_nanmean(sig.line_distance, 12),
        rolling_count(sig.still, 25) / 51.0,
        rolling_nanmean(sig.ball_speed, 5),
        rolling_nanmean(sig.crowd, 12),
        rolling_nanmean(sig.near_side, 12),
        rolling_nanmean(sig.far_side, 12),
        rolling_nanmean(sig.far_side, 50),
        rolling_nanmean(sig.ball_moved, 25),
        rolling_nanmean(sig.spread, 25),
        rolling_nanmean(sig.tracked, 25),
    ]
    return np.stack(cols, axis=1).astype(np.float32)


CONTEXT_WINDOWS = ((-25, 0), (0, 25), (25, 75))   # frames: the second before, the second after,
                                                   # and the two seconds after that


def window_mean(x, a, b):
    """Mean of x over [t + a, t + b) for every frame t, NaN-aware; NaN where the window is empty."""
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(x)
    c = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    k = np.concatenate(([0], np.cumsum(ok)))
    n = len(x)
    lo = np.clip(np.arange(n) + a, 0, n)
    hi = np.clip(np.arange(n) + b, 0, n)
    cnt = k[hi] - k[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, (c[hi] - c[lo]) / np.maximum(cnt, 1), np.nan)


def context_features(sig):
    """(n, 55) what changes across each frame: eleven signals averaged over CONTEXT_WINDOWS, and
    the two after-minus-before differences. A stop or a restart is a change; centred averages
    alone smear it over several seconds."""
    base = [sig.mean_speed, sig.fast_speed, sig.visible.astype(float), sig.out.astype(float),
            sig.ball_speed, sig.far_side, sig.near_side, sig.line_distance, sig.crowd, sig.spread,
            sig.still.astype(float)]
    cols = []
    for x in base:
        means = [window_mean(x, a, b) for a, b in CONTEXT_WINDOWS]
        cols += means + [means[1] - means[0], means[2] - means[0]]
    return np.stack(cols, axis=1).astype(np.float32)


def external_features(sig):
    """(n, 79) the classifier's features: dead_features, then context_features."""
    return np.concatenate([dead_features(sig), context_features(sig)], axis=1)


class ExternalDeadModel:
    """P(dead | collective signals), trained on labelled external matches and loaded from disk.

    The logit is a likelihood ratio: the classifiers' mean log-odds with the training prior
    removed, so the decoder's own priors decide how often the ball is dead.
    """

    def __init__(self, path=None):
        path = path or model_path()
        if not path.is_file():
            raise FileNotFoundError("no dead-ball classifier at {} -- run: python -m "
                                    "saha.deadball.train".format(path))
        with open(path, "rb") as handle:
            saved = pickle.load(handle)
        self.models, self.prior_logit = saved["models"], saved["prior_logit"]
        self.n_features = saved["n_features"]

    def logit(self, features, rows):
        if features.shape[1] != self.n_features:
            raise ValueError("expected {} features, got {}".format(self.n_features,
                                                                   features.shape[1]))
        out = np.zeros(len(rows))
        for model in self.models:
            p = np.clip(model.predict_proba(features[rows])[:, 1], 1e-4, 1 - 1e-4)
            out += np.log(p / (1.0 - p))
        return out / len(self.models) - self.prior_logit
