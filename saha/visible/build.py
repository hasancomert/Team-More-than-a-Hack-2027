"""WS3: Possession while the ball is tracked.

    python -m saha.visible.build          -> cache/evidence/visible.parquet

v2.1, touches confirmed by the team that takes the ball, written only on graded frames where the
ball is tracked (TEAM.md, boundaries). p_dead stays NaN everywhere: that question is WS2's.

1. Touches. A frame is a touch frame for a team when that team's nearest player is the closest to
   the ball and within TOUCH_M of it, at any ball speed and duels included. A touch is a run of
   touch frames of one team; runs of the same team at most JOIN frames apart are one touch (the
   tracker jitters around the 1.5 m line, and no pass comes back that fast).

2. Hand-overs. CHALLENGE.md: possession changes only when the opponent gains control, and a touch
   or a deflection is not enough. So a touch by the team without the ball is a candidate, and it is
   confirmed when, before the team with the ball touches it again,
     - that touch lasts CONFIRM_FRAMES or more (the player keeps the ball), or
     - the same team touches the ball a second time (a pass completed, a carry).
   An unconfirmed candidate is a deflection or a lost duel: the ball stays with the last team. A
   confirmed hand-over is placed at the candidate's first frame, the new team's first touch after
   the old team's last one. On WS5's DFL match that is where the labelled change is (median 2
   frames off), where v1's start of a control run was 19 frames late.

3. Probabilities. On each tracked graded frame after the first confirmed possession of a stretch,
   for the team that has the ball:
     - P_TOUCH on its own touches;
     - P_HOLD while the ball travels between two of its touches (a pass, a dribble);
     - P_LOOSE while the ball travels to the other team, on the other team's unconfirmed touches,
       and after the last touch of a stretch.
   With FIRST_TOUCH_OWNS, a stretch's first touch starts a possession without confirmation: a
   restart is taken by the team with the ball, and after a gap nobody else has touched it yet.
   Frames before a stretch's first possession have no opinion. Every frame with an opinion has
   WEIGHT.

4. Resets. The state is forgotten at the start of each graded half and across a ball gap longer than
   RESET_GAP: what happened while the ball was missing is WS4's question.

The parameters come from football and physics reasoning, not from the public 5 minutes (CLAUDE.md,
rule 2). They are checked and tuned on WS5's external data (`python -m saha.external.score`),
never on this match (D016). v1 (control runs with hysteresis) and why it changed:
experiments/20261106_ws3_control-hysteresis-v1.md, experiments/20261111_ws3_touch-confirm-v2.md;
v2.1's two settings: experiments/20261111_ws3_first-touch-weight-v21.md.

Read the match only through saha.common and write only through saha.common.evidence. Everything
printed is aggregate: counts and shares. Rules: CLAUDE.md.
"""

import argparse

import numpy as np

from saha.common import evidence, frame_table, paths
from saha.common.constants import FPS

# -- touches -----------------------------------------------------------------------------------
TOUCH_M = 1.5            # metres: a stride plus the ball's reach; the touch of selfcheck and WS4
JOIN = 3                 # frames (0.12 s): same-team touches this close are one touch
# -- hand-overs --------------------------------------------------------------------------------
CONFIRM_FRAMES = 6       # frames (0.24 s): longer than a deflection (1-3 frames at 25 fps)
CONFIRM_TOUCHES = 2      # a second touch by the same team: a completed pass or a carry
FIRST_TOUCH_OWNS = True  # a stretch's first touch starts a possession unconfirmed (a restart is
                         # taken by the team with the ball; after a gap, nobody else has touched)
# -- resets ------------------------------------------------------------------------------------
RESET_GAP = FPS // 2     # frames: a longer ball gap forgets the last team (WS4's frames)
# -- output ------------------------------------------------------------------------------------
P_TOUCH = 0.9            # the team with the ball touches it; not 1: the tracker swaps players
P_HOLD = 0.8             # the ball travels between two touches of the same team
P_LOOSE = 0.6            # the ball is on its way to the other team, or nobody touches it again
WEIGHT = 2.0             # twice nearest's per-frame weight: v2.1 reads the same player, with the
                         # possession rules on top (experiments/20261111_ws3_first-touch-weight-v21.md)

HOME, AWAY, NOBODY = 0, 1, -1


def params():
    """The parameters, as recorded in the evidence file's metadata."""
    return {"version": "2.1", "TOUCH_M": TOUCH_M, "JOIN": JOIN, "CONFIRM_FRAMES": CONFIRM_FRAMES,
            "CONFIRM_TOUCHES": CONFIRM_TOUCHES, "FIRST_TOUCH_OWNS": FIRST_TOUCH_OWNS,
            "RESET_GAP": RESET_GAP, "P_TOUCH": P_TOUCH,
            "P_HOLD": P_HOLD, "P_LOOSE": P_LOOSE, "WEIGHT": WEIGHT}


def nearest_by_team(ball_m, players_m, slot_team):
    """(N, 2) metres from the ball to the nearest tracked home and away player; inf if none."""
    dx = players_m[..., 0] - ball_m[:, None, 0]
    dy = players_m[..., 1] - ball_m[:, None, 1]
    distance = np.sqrt(dx * dx + dy * dy)
    distance = np.where(np.isfinite(distance), distance, np.inf)
    out = np.full((len(ball_m), 2), np.inf)
    for team in (HOME, AWAY):
        columns = slot_team == team
        if columns.any():
            out[:, team] = distance[:, columns].min(axis=1)
    return out


def touch_team(ball_m, players_m, slot_team, usable):
    """(N,) int8: the team touching the ball on each frame, NOBODY where no team is."""
    by_team = nearest_by_team(ball_m, players_m, slot_team)
    team = np.argmin(by_team, axis=1)          # equal distances go to home; both touch anyway
    touching = usable & (by_team.min(axis=1) <= TOUCH_M)
    return np.where(touching, team, NOBODY).astype(np.int8)


def runs(values, start, stop):
    """[(a, b, value)]: maximal runs of equal values in values[start:stop], absolute, half-open."""
    window = values[start:stop]
    if not len(window):
        return []
    breaks = np.flatnonzero(window[1:] != window[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(window)]))
    return [(start + int(a), start + int(b), int(window[a]))
            for a, b in zip(bounds[:-1], bounds[1:])]


def stretches(usable, half):
    """[(start, stop)]: runs of usable frames within one half, joined across gaps of at most
    RESET_GAP frames."""
    index = np.flatnonzero(usable)
    if not len(index):
        return []
    step = np.diff(index)
    cut = np.flatnonzero((step > RESET_GAP + 1) | (np.diff(half[index]) != 0)) + 1
    return [(int(part[0]), int(part[-1]) + 1) for part in np.split(index, cut)]


def touches(team, start, stop):
    """[(a, b, team)]: the touches in team[start:stop], same-team runs within JOIN joined."""
    out = []
    for a, b, k in runs(team, start, stop):
        if k == NOBODY:
            continue
        if out and out[-1][2] == k and a - out[-1][1] <= JOIN:
            out[-1] = (out[-1][0], b, k)
        else:
            out.append((a, b, k))
    return out


def hand_overs(events):
    """[(frame, team)]: where each confirmed possession starts, from a stretch's touches."""
    owner, candidate, count, out = NOBODY, None, 0, []
    for a, b, k in events:
        if k == owner:
            candidate, count = None, 0     # the team with the ball touched it again
            continue
        if candidate is None or candidate[1] != k:
            candidate, count = (a, k), 0
        count += 1
        first = FIRST_TOUCH_OWNS and not out
        if first or b - a >= CONFIRM_FRAMES or count >= CONFIRM_TOUCHES:
            out.append(candidate)
            owner, candidate, count = k, None, 0
    return out


def toward(team, p):
    """P(home) when `team` has the ball with probability p."""
    return p if team == HOME else 1.0 - p


def possession(team, usable, half):
    """(p_home, counts): P(home) on usable frames from the confirmed touches, NaN = no opinion."""
    p_home = np.full(len(team), np.nan)
    counts = {"stretches": 0, "touches": 0, "possessions": 0, "hand-overs": 0,
              "unconfirmed touches": 0}
    for start, stop in stretches(usable, half):
        counts["stretches"] += 1
        events = touches(team, start, stop)
        starts = hand_overs(events)
        counts["touches"] += len(events)
        counts["possessions"] += len(starts)
        counts["hand-overs"] += max(len(starts) - 1, 0)
        if not starts:
            continue
        begin = dict(starts)
        owner = NOBODY
        for i, (a, b, k) in enumerate(events):
            if a in begin:
                owner = begin[a]
            if owner == NOBODY:
                continue
            if k == owner:
                p_home[a:b] = toward(owner, P_TOUCH)
            else:
                p_home[a:b] = toward(owner, P_LOOSE)
                counts["unconfirmed touches"] += 1
            # the frames until the next touch: a pass between teammates, or the ball on its way
            # to the other team (whose touch then starts a possession or is a deflection)
            if i + 1 < len(events):
                a2, _, k2 = events[i + 1]
                held = k == owner and k2 == owner
                p_home[b:a2] = toward(owner, P_HOLD if held else P_LOOSE)
            else:
                p_home[b:stop] = toward(owner, P_LOOSE)
    p_home[~usable] = np.nan
    return p_home, counts


def evidence_and_counts(table):
    """((p_dead, p_home, weight), counts) over every video frame of the table."""
    usable = table.graded & table.ball_visible
    team = touch_team(table.ball_xy_m(), table.players_xy_m(), table.slot_team, usable)
    p_home, counts = possession(team, usable, table.half)
    p_dead, _, weight = evidence.empty(table.n_frames)
    has = np.isfinite(p_home)
    weight[has] = WEIGHT
    counts = {"graded frames, ball tracked": int(usable.sum()),
              "touch frames": int((team != NOBODY).sum()), **counts,
              "frames with an opinion": int(has.sum())}
    return (p_dead, p_home.astype(np.float32), weight), counts


def build(table):
    """(p_dead, p_home, weight) arrays over every video frame."""
    return evidence_and_counts(table)[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", help="data bundle root (default: MTAH_DATA_DIR or ./data)")
    args = parser.parse_args(argv)
    if args.data:
        paths.use_data_dir(args.data)
    (p_dead, p_home, weight), counts = evidence_and_counts(frame_table.load())
    path = evidence.write_evidence("visible", p_dead, p_home, weight,
                                   producer="saha.visible.build", params=params())
    tracked = max(counts["graded frames, ball tracked"], 1)
    for name, value in counts.items():
        print("{:<28} {:>8}".format(name, value))
    print("{:<28} {:>8.3f}".format("coverage of tracked frames",
                                   counts["frames with an opinion"] / tracked))
    print("\n".join(evidence.validate_file(path)[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
