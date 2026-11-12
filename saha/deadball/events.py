"""Stop and restart events: the moments a stoppage opens and closes.

The dead-ball classifier judges each frame by how it looks, and that is exactly where it fails:
for a few seconds after a whistle the match still looks like play, and while a set piece is being
prepared the ball sits at someone's feet with players walking around it, which looks like slow
build-up. What decides those frames is not how they look but what happened before them and what
happens after: a stoppage lasts from the event that stops play to the event that restarts it.

So these detectors find the events, and the decoder only lets the ball die near a stop event and
come back to life near a restart event (pipeline.decode). They are written from the laws of the
game and from physics, not fitted to anything:

  stops     a whistle blown while the match is moving; the ball crossing a line (guarded against
            high balls, whose ground projection can land outside the pitch while play goes on);
            the ball vanishing while everyone slows down.
  restarts  the ball struck after lying still, or after being out of sight or out of the pitch
            (a placed free kick or goal kick, a throw-in, a ball fetched and played); a kickoff.

A detector firing in open play is harmless: a restart candidate can only end a stoppage, and a
stop candidate only lets one begin if the dead-ball evidence agrees.
"""

import warnings

import numpy as np

from .features import FPS, rolling_nanmean

# -- stops ----------------------------------------------------------------------------------------
LIVE_PACE = 1.5            # m/s over the 2 s before a whistle: the match was moving
OUT_MARGIN = 1.0           # m past a line: clear of tracking noise
OUT_MIN = 5                # frames beyond the line
OUT_STAYS = 38             # frames out (1.5 s): a ball that stays out is out, not a high ball
OUT_BACK = 25              # frames: a ball not back inside within 1 s went out
GAP_SOON = 50              # frames: or a long ball gap starts within 2 s of the exit
LONG_GAP = 5 * FPS         # a ball gap this long is the tracker losing the ball for good
SLOW_GAP = 1.6             # m/s mean over a long gap: players have stopped

# -- restarts -------------------------------------------------------------------------------------
REST_FRAMES = 25           # look back 1 s for the ball at rest
REST_MOVE = 1.5            # m moved over that second: still, placed, or carried at walking pace
DEPART = 0.3               # m in the next 0.2 s: the ball starts to move
TRAVEL = 3.0               # m within the next second: and it was played, not nudged
SLOW_STEP = 0.6            # m per 0.2 s (3 m/s) at most over the 0.6 s before a kick: dribbled or placed
KICK_STEP = 1.2            # m per 0.2 s (6 m/s) reached within 0.4 s: struck
HIDDEN_FRAMES = 25         # out of sight for at least 1 s before it reappears
REAPPEAR_MOVING = 1.0      # m in the 0.2 s after reappearing: struck while out of sight
# A real restart is followed by play: players speed up. Mean player speed 1-4 s after over the 3 s
# before. Across the match the candidates inside stoppages fall in two groups, below about 1.1 and
# above about 1.5; the cut sits between them.
RESUMES = 1.2
# A ball coming back into play from beyond or on a line, and fast, is a throw-in or a corner taken
# where the tracker had lost the ball. A ball fetched from behind the line rolls in slowly.
LINE_NEAR = 1.0            # m from a line, or beyond it
FROM_LINE_STEP = 1.6       # m per 0.2 s (8 m/s) within 0.4 s of the departure
GOAL_AREA = (5.5 + 1.0, 9.16 + 1.0)   # m deep, m either side of the goal centre, plus noise
CORNER = 2.0                           # m from a corner


def _runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def _ahead(ball_m, k):
    """Distance from each frame's ball position to the position k frames later."""
    out = np.full(len(ball_m), np.nan, dtype=np.float32)
    out[:-k] = np.hypot(*(ball_m[k:] - ball_m[:-k]).T)
    return out


def stop_events(sig, whistles, pitch):
    """[(frame, kind)] where play may have stopped."""
    length, width = pitch
    out = []
    pace = rolling_nanmean(sig.mean_speed, 25)

    for onset, _ in whistles:
        before = np.nanmean(pace[max(onset - 50, 0):onset]) if onset > 0 else np.nan
        if before >= LIVE_PACE:
            out.append((onset, "whistle"))

    ball = sig.ball_m
    with np.errstate(invalid="ignore"):
        beyond = sig.visible & ((ball[:, 0] < -OUT_MARGIN) | (ball[:, 0] > length + OUT_MARGIN) |
                                (ball[:, 1] < -OUT_MARGIN) | (ball[:, 1] > width + OUT_MARGIN))
        inside = sig.visible & (ball[:, 0] >= 0) & (ball[:, 0] <= length) & \
            (ball[:, 1] >= 0) & (ball[:, 1] <= width)
    starts, stops = _runs(beyond)
    for a, b in zip(starts, stops):
        if b - a < OUT_MIN:
            continue
        back = np.flatnonzero(inside[b:b + OUT_BACK])
        gap_soon = bool(np.any(sig.gap_length[b:b + GAP_SOON] >= LONG_GAP))
        if (b - a >= OUT_STAYS) or not len(back) or gap_soon:
            # the crossing: the last frame the ball was seen inside, if that was just before
            seen = np.flatnonzero(inside[max(a - OUT_BACK, 0):a])
            crossing = max(a - OUT_BACK, 0) + int(seen[-1]) + 1 if len(seen) else a
            out.append((crossing, "exit"))

    starts, stops = _runs(~sig.visible)
    for a, b in zip(starts, stops):
        if b - a >= LONG_GAP and np.nanmean(sig.mean_speed[a:b]) < SLOW_GAP:
            out.append((int(a), "vanish"))
    return sorted(out)


def restart_events(sig, kickoff_frames):
    """[(frame, kind)] where play may have restarted.

    A restart is the ball starting to move after lying still, being carried at walking pace (a
    keeper walking to the edge of his area), or sitting beyond a line, and then travelling at
    least twice as far in the next second as in the last: a goal kick or a free kick is often rolled gently before it picks
    up speed, so the restart is where the ball departs, not where it is fastest. A ball that was
    struck while out of the tracker's sight reappears already moving.
    """
    ball = sig.ball_m
    moved_02 = _ahead(ball, 5)
    moved_1s = _ahead(ball, 25)
    back_1s = np.full(len(ball), np.nan, dtype=np.float32)
    back_1s[REST_FRAMES:] = np.hypot(*(ball[REST_FRAMES:] - ball[:-REST_FRAMES]).T)
    hidden_before = np.zeros(len(ball), dtype=np.int64)       # frames of gap just before
    starts, stops = _runs(~sig.visible)
    for a, b in zip(starts, stops):
        if b < len(ball):
            hidden_before[b] = b - a
    with np.errstate(invalid="ignore"):
        # throw-in: the ball was beyond a touchline (not a goal line -- a ball fetched from behind
        # the goal for a goal kick comes back in the same way) and is now in play
        beyond_touch = sig.out & ((ball[:, 1] < 0) | (ball[:, 1] > sig.width))
        held_out = np.zeros(len(ball), dtype=bool)
        held_out[1:] = beyond_touch[:-1]
        at_rest = sig.visible & ((back_1s < REST_MOVE) | held_out)
        departs = at_rest & (moved_02 >= DEPART) & (moved_1s >= np.maximum(TRAVEL, 2.0 * back_1s))
        # or a kick of a ball being dribbled into place: slow for the 0.6 s before, then fast
        step = moved_02                                       # m over the next 0.2 s
        slow_before = np.ones(len(ball), dtype=bool)
        for k in (5, 10, 15):
            prev = np.full(len(ball), np.nan, dtype=np.float32)
            prev[k:] = step[:-k]
            slow_before &= prev <= SLOW_STEP
        fast_soon = np.zeros(len(ball), dtype=bool)
        for k in (0, 5):
            nxt = np.full(len(ball), np.nan, dtype=np.float32)
            nxt[:len(ball) - k] = step[k:]
            fast_soon |= nxt >= KICK_STEP
        departs |= sig.visible & slow_before & fast_soon & (moved_1s >= TRAVEL)
        reappears = sig.visible & (hidden_before >= HIDDEN_FRAMES) & (moved_02 >= REAPPEAR_MOVING)
    out, last = [], -10 ** 9
    for f in np.flatnonzero(departs | reappears):
        if f - last > FPS:                                    # one restart per second at most
            kind = "reappear" if reappears[f] else ("throw" if held_out[f] else "strike")
            out.append((int(f), kind))
        last = f
    out.extend((int(f), "kickoff") for f in kickoff_frames)
    return sorted(out)


def strong_restarts(sig, restarts):
    """The restart candidates that the laws or the players say are real.

    A kickoff or a throw-in is a restart wherever it happens. A ball played from the goal area or a
    corner is a goal kick or a corner, and a ball sent fast into the pitch from a line is a throw-in
    or a corner. Anywhere else -- a free kick, a ball fetched and played -- the candidate counts
    only if players speed up afterwards: touches while a set piece is being prepared leave
    everyone standing.
    """
    ball = sig.ball_m
    step = _ahead(ball, 5)
    out = []
    for f, kind in restarts:
        if kind in ("kickoff", "throw"):
            out.append((f, kind))
            continue
        with np.errstate(invalid="ignore"):
            at_line = np.nanmin(sig.line_distance[max(f - 10, 0):f + 1]) <= LINE_NEAR \
                if np.isfinite(sig.line_distance[max(f - 10, 0):f + 1]).any() else False
            fast = np.nanmax(step[f:f + 6]) >= FROM_LINE_STEP if np.isfinite(step[f:f + 6]).any() else False
        if at_line and fast:
            out.append((f, "from line"))
            continue
        # where the ball sat for the second before it moved: a goal kick is taken from a ball
        # placed in the goal area, a corner from one placed at the flag -- a ball fetched from
        # behind the line passes through those places on its way, so passing through is not enough
        recent = ball[max(f - REST_FRAMES, 0):f + 1]
        recent = recent[~np.isnan(recent[:, 0])]
        if len(recent) >= REST_FRAMES // 2:
            depth = np.minimum(recent[:, 0], sig.length - recent[:, 0])
            in_area = (depth >= 0) & (depth <= GOAL_AREA[0]) & \
                (np.abs(recent[:, 1] - sig.width / 2.0) <= GOAL_AREA[1])
            if np.all(in_area):
                out.append((f, "goal kick"))
                continue
            corner_dist = np.min([np.hypot(recent[:, 0] - cx, recent[:, 1] - cy)
                                  for cx in (0.0, sig.length) for cy in (0.0, sig.width)], axis=0)
            if np.all(corner_dist <= CORNER):
                out.append((f, "corner"))
                continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            before = np.nanmean(sig.mean_speed[max(f - 75, 0):f])
            after = np.nanmean(sig.mean_speed[f + 25:f + 100])
        if before > 0 and after / before >= RESUMES:
            out.append((f, kind))
    return out


SET_PIECE_KINDS = ("goal kick", "corner", "throw", "from line")
SPOT_RADIUS = 12.0         # m: the ball is being brought to, or kept at, the restart spot
BROUGHT_MIN = 3 * FPS      # frames: at least this long
BROUGHT_PACE = 1.5         # m/s: carried, rolled or placed, not dribbled at speed
PLAYED_SPEED = 5.0         # m/s: a ball this fast going back in time was still being played
LOOK_BACK = 60 * FPS


def implied_stops(sig, restarts):
    """([(frame, "implied")], [(start, restart)]): stoppages a set-piece restart proves happened.

    A goal kick, a corner or a throw-in can only follow a dead ball. Before it, the ball is
    fetched, carried or rolled to the restart spot and kept there. Walking back from the restart
    while the ball stays near that spot (or beyond a line, or out of sight) finds where the
    stoppage began, even when nothing else marked it -- a shot gone wide that the players jog
    back from looks like play to everything but the ball.

    The second list is the subset that began with the ball at or beyond a line, or out of sight:
    the ball went out, and everything from there to the restart is dead by the laws. The others
    could also be a keeper on the ball in his own area, so they only open a stoppage, they do not
    force one.
    """
    ball = sig.ball_m
    out, certain = [], []
    for f, kind in restarts:
        if kind not in SET_PIECE_KINDS:
            continue
        seen = np.flatnonzero(sig.visible[f:f + 25])
        if not len(seen):
            continue
        spot = ball[f + seen[0]]
        t = f
        while t > max(f - LOOK_BACK, 0):
            if sig.visible[t - 1]:
                if sig.ball_speed[t - 1] > PLAYED_SPEED:
                    break                                     # the shot or pass that ended play
                if not sig.out[t - 1] and np.hypot(*(ball[t - 1] - spot)) > SPOT_RADIUS:
                    break
            t -= 1
        if f - t < BROUGHT_MIN:
            continue
        # median, not path length: frame-to-frame jitter adds up along a path
        speeds = sig.ball_speed[t:f]
        if np.isfinite(speeds).any() and np.nanmedian(speeds) > BROUGHT_PACE:
            continue
        out.append((int(t), "implied"))
        if not sig.visible[t] or sig.line_distance[t] <= LINE_NEAR:
            certain.append((int(t), int(f)))
    return out, certain


def hold_mask(n, stops, restarts, runs, cap_frames=None):
    """(n,) bool: from each stop event up to the next confirmed restart, within its half.

    cap_frames keeps only the first that many frames of each interval: past that, a restart the
    detector missed is likelier than a stoppage still being prepared.
    """
    out = np.zeros(n, dtype=bool)
    starts = np.array(sorted(f for f, _ in restarts))
    for f, _ in stops:
        run = next(((a, b) for a, b in runs if a <= f < b), None)
        if run is None:
            continue
        later = starts[starts > f] if len(starts) else starts
        end = min(int(later[0]) if len(later) else run[1], run[1])
        if cap_frames is not None:
            end = min(end, f + cap_frames)
        out[f:end] = True
    return out


def event_mask(n, events, before, after):
    """(n,) bool: within [frame - before, frame + after] of any event."""
    mask = np.zeros(n, dtype=bool)
    for f, _ in events:
        mask[max(f - before, 0):min(f + after + 1, n)] = True
    return mask


def since_and_until(n, events, cap):
    """(since, until): frames since the last event and until the next one, capped."""
    marks = np.zeros(n, dtype=bool)
    for f, _ in events:
        if 0 <= f < n:
            marks[f] = True
    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(marks, idx, -10 ** 9))
    nxt = np.minimum.accumulate(np.where(marks, idx, 10 ** 9)[::-1])[::-1]
    return np.minimum(idx - last, cap).astype(np.float32), np.minimum(nxt - idx, cap).astype(np.float32)


def open_stoppage(n, stops, restarts):
    """(n,) bool: a stop event has happened and no restart event since."""
    marks = np.zeros(n, dtype=np.int8)
    for f, _ in stops:
        if 0 <= f < n:
            marks[f] = 1
    for f, _ in restarts:
        if 0 <= f < n:
            marks[f] = -1                                     # a restart on the same frame wins
    state = np.zeros(n, dtype=bool)
    # forward fill of the last mark
    idx = np.arange(n)
    last_mark = np.maximum.accumulate(np.where(marks != 0, idx, -1))
    has = last_mark >= 0
    state[has] = marks[last_mark[has]] == 1
    return state
