"""WS3: Possession while the ball is tracked.

    python -m saha.visible.build          -> cache/evidence/visible.parquet

v1, control with hysteresis, written only on graded frames where the ball is tracked (TEAM.md,
boundaries). p_dead stays NaN everywhere: that question is WS2's.

1. Control. A frame is a control frame for a team when its nearest player is within CONTROL_M of a
   slow ball (CONTROL_SPEED, the ball's speed from its positions, D015) and the nearest opponent is
   at least CONTEST_M further away. A duel, with both teams at the ball, is nobody's control. A
   control run is MIN_CONTROL or more consecutive control frames of one team. Shorter runs (a
   deflection, a touch in passing) are ignored: CHALLENGE.md says a touch without control is not a
   change of possession.

2. Hysteresis. CHALLENGE.md: a loose or contested ball stays with the last team, and possession
   changes only when the opponent gains control. So between two control runs the ball belongs to
   the team of the first one:
     - P_HOLD when the next run is the same team: a pass, a dribble, a first touch.
     - P_LOOSE when the next run is the other team: the hand-over happens at the start of the
       opponent's run, and before it the ball may already have been lost.
   On a control run itself, P_CONTROL for its team.

3. Resets. The state is forgotten at the start of each graded half and across a ball gap longer than
   RESET_GAP: what happened while the ball was missing is WS4's question. Frames before the first
   control run of a stretch have no opinion; frames after its last one get P_LOOSE for that team.

The parameters come from football and physics reasoning, not from the public 5 minutes (CLAUDE.md,
rule 2); the control definition is the touch of saha.tools.selfcheck and WS4, so that the packages
agree on what "at the ball" means. They are checked and tuned on WS5's external data
(`python -m saha.external.score`), never on this match (D016).

Read the match only through saha.common and write only through saha.common.evidence. Everything
printed is aggregate: counts and shares. Rules: CLAUDE.md.
"""

import argparse

import numpy as np

from saha.common import evidence, frame_table, kinematics, paths
from saha.common.constants import FPS

# -- control (the touch of saha.tools.selfcheck and WS4) ---------------------------------------
CONTROL_M = 1.5          # metres: a player this close to the ball has it at his feet
CONTROL_SPEED = 5.0      # m/s; faster is a pass, a shot or a clearance, not a ball under control
CONTEST_M = 1.0          # metres: an opponent within this much of the controller makes it a duel
MIN_CONTROL = 6          # frames (0.24 s): receiving and keeping the ball, not a deflection
# -- hysteresis --------------------------------------------------------------------------------
RESET_GAP = FPS // 2     # frames: a longer ball gap forgets the last team (WS4's frames)
P_CONTROL = 0.9          # a team controls the ball; not 1: the tracker swaps and merges players
P_HOLD = 0.8             # between two runs of the same team: the ball travels between teammates
P_LOOSE = 0.6            # between runs of different teams, or after the last run: still the last
                         # team's, but the hand-over is near
WEIGHT = 1.0

HOME, AWAY, NOBODY = 0, 1, -1


def params():
    """The parameters, as recorded in the evidence file's metadata."""
    return {"CONTROL_M": CONTROL_M, "CONTROL_SPEED": CONTROL_SPEED, "CONTEST_M": CONTEST_M,
            "MIN_CONTROL": MIN_CONTROL, "RESET_GAP": RESET_GAP, "P_CONTROL": P_CONTROL,
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


def control_team(ball_m, players_m, slot_team, usable):
    """(N,) int8: the team in control of the ball on each frame, NOBODY where no team is."""
    by_team = nearest_by_team(ball_m, players_m, slot_team)
    team = np.argmin(by_team, axis=1)          # equal distances are a duel, removed below
    own = by_team.min(axis=1)
    other = by_team.max(axis=1)
    speed = kinematics.displacement_speed(ball_m)
    with np.errstate(invalid="ignore"):        # NaN speed or inf - inf: not control
        control = (usable & (own <= CONTROL_M) & (speed <= CONTROL_SPEED)
                   & (other - own >= CONTEST_M))
    return np.where(control, team, NOBODY).astype(np.int8)


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


def toward(team, p):
    """P(home) when `team` has the ball with probability p."""
    return p if team == HOME else 1.0 - p


def hysteresis(control, usable, half):
    """(p_home, counts): P(home) on usable frames from the control runs, NaN where no opinion."""
    p_home = np.full(len(control), np.nan)
    counts = {"stretches": 0, "control runs": 0, "hand-overs": 0}
    for start, stop in stretches(usable, half):
        counts["stretches"] += 1
        state, cursor = NOBODY, start
        for a, b, team in runs(control, start, stop):
            if team == NOBODY or b - a < MIN_CONTROL:
                continue
            if state != NOBODY:
                p_home[cursor:a] = toward(state, P_HOLD if team == state else P_LOOSE)
                counts["hand-overs"] += int(team != state)
            p_home[a:b] = toward(team, P_CONTROL)
            counts["control runs"] += 1
            state, cursor = team, b
        if state != NOBODY:
            p_home[cursor:stop] = toward(state, P_LOOSE)
    p_home[~usable] = np.nan
    return p_home, counts


def evidence_and_counts(table):
    """((p_dead, p_home, weight), counts) over every video frame of the table."""
    usable = table.graded & table.ball_visible
    control = control_team(table.ball_xy_m(), table.players_xy_m(), table.slot_team, usable)
    p_home, counts = hysteresis(control, usable, table.half)
    p_dead, _, weight = evidence.empty(table.n_frames)
    has = np.isfinite(p_home)
    weight[has] = WEIGHT
    counts = {"graded frames, ball tracked": int(usable.sum()),
              "control frames": int((control != NOBODY).sum()), **counts,
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
