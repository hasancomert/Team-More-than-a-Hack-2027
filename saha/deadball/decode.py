"""Segmentation into home possession, away possession and dead ball: a three-state Viterbi.

The frame evidence is the dead-ball classifier's (segment.py; WS2 gives it no team evidence, so H
and A are the same alive state twice), and the transition costs are football priors on how long
possessions and stoppages last, gated by the stop and restart events. The result is a path of
whole stoppages rather than per-frame votes, which is what the ranking metric counts.
"""

import numpy as np

from .features import FPS

H, A, D = 0, 1, 2

# Priors, seconds. A stoppage follows about every half-minute of play and lasts about as long;
# within play the ball changes hands every 15-20 s -- a match has a few hundred possessions in
# under an hour of live ball.
MEAN_ALIVE_S = 30.0
MEAN_DEAD_S = 25.0
MEAN_SWITCH_S = 18.0

# Event-gated transitions (gated_log_trans). Near a detected stop event the ball dies within about
# a second; away from one, a stoppage can still begin -- a whistle too quiet to detect, a ball that
# leaves the pitch unseen -- but the dead-ball evidence has to carry it alone, as if stoppages
# without a detected cause came once every two minutes of play. Restarts the same way round.
NEAR_EVENT_S = 1.0
AWAY_FROM_EVENT_S = 120.0
# Between a stop event and the next confirmed restart the ball is dead by the laws, and the decoder
# holds it dead: inside a hold, play can only resume at a confirmed restart. A soft hold was tried
# first and lost exactly the stoppages it was for -- while a set piece is prepared the dead-ball
# classifier sees slow players and the ball at someone's feet, and its "this is play" evidence
# outweighs any finite price. The hold is capped instead (pipeline.HOLD_CAP), so a restart the
# detector missed cannot lock a stoppage for longer than a set piece takes to prepare.


def transition_logprob():
    """(3, 3) log P(to | from) per frame."""
    stop = 1.0 / (FPS * MEAN_ALIVE_S)
    switch = 1.0 / (FPS * MEAN_SWITCH_S)
    restart = 1.0 / (FPS * MEAN_DEAD_S)
    t = np.array([
        [1.0 - stop - switch, switch, stop],
        [switch, 1.0 - stop - switch, stop],
        [restart / 2.0, restart / 2.0, 1.0 - restart],
    ])
    return np.log(t)


def gated_log_trans(stop_near, restart_near, hold=None, stop_rate=None, restart_rate=None):
    """(n, 3, 3) per-frame log P(to | from): stops and restarts are likely only near events.

    hold marks frames between a stop event and the next confirmed restart: a stoppage that has
    begun there cannot end. stop_rate / restart_rate, when given, are per-frame probabilities
    from a learned model; each frame takes the larger of those and the event gating.
    """
    n = len(stop_near)
    near, away = 1.0 / (FPS * NEAR_EVENT_S), 1.0 / (FPS * AWAY_FROM_EVENT_S)
    stop = np.where(stop_near, near, away)
    restart = np.where(restart_near, near, away)
    if stop_rate is not None:
        stop = np.maximum(stop, stop_rate)
    if restart_rate is not None:
        restart = np.maximum(restart, restart_rate)
    if hold is not None:
        restart = np.where(hold & ~restart_near, 0.0, restart)
    switch = 1.0 / (FPS * MEAN_SWITCH_S)
    t = np.empty((n, 3, 3))
    t[:, H, H] = t[:, A, A] = 1.0 - stop - switch
    t[:, H, A] = t[:, A, H] = switch
    t[:, H, D] = t[:, A, D] = stop
    t[:, D, H] = t[:, D, A] = restart / 2.0
    t[:, D, D] = 1.0 - restart
    with np.errstate(divide="ignore"):
        return np.log(t)


def viterbi_varying(emission, log_trans):
    """Viterbi with a transition matrix per frame: log_trans[t] scores the step into frame t."""
    n, k = emission.shape
    back = np.zeros((n, k), dtype=np.int8)
    score = emission[0] - np.log(k)
    cols = np.arange(k)
    for t in range(1, n):
        cand = score[:, None] + log_trans[t]
        back[t] = np.argmax(cand, axis=0)
        score = cand[back[t], cols] + emission[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def viterbi(emission, log_trans):
    """Most likely state path for one run. emission is (n, k) log-likelihoods."""
    n, k = emission.shape
    back = np.zeros((n, k), dtype=np.int8)
    score = emission[0] - np.log(k)
    for t in range(1, n):
        cand = score[:, None] + log_trans
        back[t] = np.argmax(cand, axis=0)
        score = cand[back[t], np.arange(k)] + emission[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def integrated_autocorr_time(residual, max_lag=500):
    """Frames per effectively independent observation, from the autocorrelation of a residual.

    `residual` is the evidence minus what the true state predicts, NaN where that is unknown;
    pairs are only formed where both frames are known, so separate stretches never mix. Summed
    until the autocorrelation first drops below zero. Used to discount per-frame evidence that is
    really one observation repeated over many frames -- without it the decoder treats a smoothed
    signal as dozens of independent votes and switches far too readily.

    It has to be the residual, not the signal: the signal stays put for as long as a possession
    does, and its autocorrelation would measure that instead of the noise.
    """
    x = np.asarray(residual, dtype=np.float64)
    known = ~np.isnan(x)
    x = np.where(known, x - np.nanmean(x), 0.0)
    var = np.dot(x, x) / known.sum()
    tau = 1.0
    for lag in range(1, max_lag):
        pairs = np.sum(known[:-lag] & known[lag:])
        if not pairs:
            break
        rho = np.dot(x[:-lag], x[lag:]) / pairs / var
        if rho <= 0:
            break
        tau += 2.0 * rho
    return tau


def run_lengths(mask):
    """Lengths of the True runs in a boolean array."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
