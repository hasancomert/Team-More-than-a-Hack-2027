"""From tracking to stoppages: stop and restart events, the dead-ball classifier, and a decoder.

evidence() turns one match into per-frame evidence and decode() into dead/alive. They work on any
match given its arrays (build.py passes our frame table; WS5's harness passes external tables),
so nothing here names our match.

The decoder is an HMM over H, A and D whose transitions are gated by the events: a stoppage can
open readily only near a stop event and close readily only near a restart event (decode.py), and
after a whistle in live play it holds until a confirmed restart. It gets no team evidence: the
team package only reports dead or alive, and the prototype's team-shape model was fitted on this
match, which the organizers' rules do not allow. H and A are then two copies of the same alive
state, and the path's dead stretches are those of a dead/alive decoder. On twenty SkillCorner
matches this costs 0.006 spellF1 against the prototype's team evidence (docs/tech/ws2_deadball.md).
"""

import numpy as np

from .dead import EXTERNAL_TAU, EXTERNAL_WEIGHT, ExternalDeadModel, Signals, external_features
from .decode import A, D, H, gated_log_trans, viterbi_varying
from .events import (event_mask, hold_mask, implied_stops, restart_events, stop_events,
                     strong_restarts)
from .features import FPS, attack_sign, frame_attack_sign, to_metres
from .kickoff import kickoff_dead_mask, kickoffs, own_halves

HOME, AWAY = 0, 1

# Windows around events, in frames, inside which the decoder lets a stoppage open or close. Players
# can stop a moment before the whistle and the decision lands at it; a ball can leave the pitch a
# few seconds before the tracker gives up on it; a restart is the ball clearly moving, give or take
# tracking noise.
STOP_WINDOWS = {"whistle": (25, 50), "exit": (12, 25), "vanish": (75, 12), "implied": (12, 25)}
RESTART_WINDOW = (12, 25)
WHISTLE_DEAD = 25          # frames after a whistle in live play that are dead by the laws
HOLD_CAP = 60 * FPS        # a stoppage is held for at most its first minute (decode.gated_log_trans)
# The stop events that open a hard hold. By the laws, play stopped by the whistle restarts only
# with a restart. On twenty external matches a hold after the ball leaves the pitch cost more than
# it saved, because a restart the detector misses lets the stoppage run on.
HOLD_FROM = ("whistle",)


def log(message):
    print(message, flush=True)


def blocks_of(half):
    """[(half, first, last)] inclusive frames of each run of half > 0, in order."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], half, [0]))) != 0)
    return [(int(half[a]), int(a), int(b) - 1) for a, b in zip(edges[:-1], edges[1:])
            if half[a] > 0]


def evidence(ball_xy, players, team, pitch, blocks, kickoff_windows=(), whistles=(),
             verbose=True):
    """Everything the decoder needs, as a dict of per-frame arrays and scalars.

    ball_xy (n, 2) and players (n, k, 3: x, y, speed m/s) in 0-1 pitch units, team (k,) 0 home /
    1 away, pitch (length, width) in metres, blocks [(half, first, last)] of the graded halves,
    kickoff_windows [(start, stop)] frames where a tagger saw a kick-off, whistles [(onset, end)]
    as from audio.whistle_events.
    """
    say = log if verbose else (lambda message: None)
    length = pitch[0]
    n = len(ball_xy)
    runs = [(first, last + 1) for _, first, last in blocks]
    graded = np.zeros(n, dtype=bool)
    for a, b in runs:
        graded[a:b] = True
    rows = np.flatnonzero(graded)
    players_m = to_metres(players[..., :2], pitch)

    # -- kickoffs: dead ball by the laws, enforced rather than estimated ----------------------
    signs = attack_sign(players_m, team, blocks, length)
    by_side = {side: frame_attack_sign(n, blocks, signs, side) for side in (HOME, AWAY)}
    found = kickoffs(own_halves(players_m, team, by_side, length), kickoff_windows)
    forced_dead = kickoff_dead_mask(n, found, runs)
    say("kickoffs confirmed by formation: {} of {} tagged; {} graded frames forced dead".format(
        len(found), len(kickoff_windows), int(np.sum(forced_dead & graded))))

    # -- stop and restart events ---------------------------------------------------------------
    sig = Signals(ball_xy, players, team, pitch)
    events = event_evidence(sig, list(whistles), pitch, [kick for _, kick in found], runs, n)
    stops = events["stops"]
    for start, restart in events["implied_dead"]:
        forced_dead[start:restart] = True
    live_whistles = [e for e in stops if e[1] == "whistle"]
    for onset, _ in live_whistles:
        forced_dead[onset:onset + WHISTLE_DEAD] = True
    say("events: {} whistles ({} in live play), {} stops, {} restart candidates of which {} "
        "confirmed".format(len(whistles), len(live_whistles), len(stops),
                           len(events["candidates"]), len(events["restarts"])))

    # -- dead ball -----------------------------------------------------------------------------
    # The events stay out of the classifier on purpose. It judges how a frame looks; the memory of
    # "a stoppage opened and has not been closed" lives in the decoder's gating.
    dead_logit = np.zeros(n)
    dead_logit[rows] = ExternalDeadModel().logit(external_features(sig), rows)
    say("dead ball: external classifier, evidence correlation time {:.0f} frames, weight "
        "{:.0f}".format(EXTERNAL_TAU, EXTERNAL_WEIGHT))
    return {"n": n, "runs": runs, "graded": graded, "dead_logit": dead_logit,
            "tau_dead": EXTERNAL_TAU / EXTERNAL_WEIGHT, "forced_dead": forced_dead,
            "stop_near": events["stop_near"], "restart_near": events["restart_near"],
            "hold": events["hold"], "stops": stops, "restarts": events["restarts"],
            "whistles": list(whistles), "kickoff_frames": [kick for _, kick in found]}


def event_evidence(sig, whistles, pitch, kickoff_frames, runs, n, hold_from=HOLD_FROM):
    """Stop and restart events, and the decoder's gating masks built from them."""
    stops = stop_events(sig, whistles, pitch)
    candidates = restart_events(sig, kickoff_frames)
    restarts = strong_restarts(sig, candidates)
    implied, implied_dead = implied_stops(sig, restarts)
    stops = sorted(stops + implied)
    stop_near = np.zeros(n, dtype=bool)
    for kind, (before, after) in STOP_WINDOWS.items():
        stop_near |= event_mask(n, [e for e in stops if e[1] == kind], before, after)
    return {"stops": stops, "candidates": candidates, "restarts": restarts,
            "implied_dead": implied_dead, "stop_near": stop_near,
            "restart_near": event_mask(n, restarts, *RESTART_WINDOW),
            "hold": hold_mask(n, [e for e in stops if e[1] in hold_from], restarts, runs,
                              cap_frames=HOLD_CAP)}


def decode(ev):
    """(n,) bool: dead ball on graded frames, False elsewhere."""
    dead_llr = ev["dead_logit"] / ev["tau_dead"]
    # Without team evidence the factorised emissions of the prototype reduce to these: H and A
    # alike, and whether the ball is alive decided by the dead-ball evidence alone.
    split = np.logaddexp(0.0, 0.0)
    emission = np.zeros((ev["n"], 3))
    emission[:, H] = -0.5 * dead_llr - split
    emission[:, A] = -0.5 * dead_llr - split
    emission[:, D] = 0.5 * dead_llr - split
    emission[ev["forced_dead"], H] = -1e9
    emission[ev["forced_dead"], A] = -1e9

    log_trans = gated_log_trans(ev["stop_near"], ev["restart_near"], ev["hold"])
    dead = np.zeros(ev["n"], dtype=bool)
    for a, b in ev["runs"]:
        dead[a:b] = viterbi_varying(emission[a:b], log_trans[a:b]) == D
    return dead
