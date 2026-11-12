"""WS4: Possession while the ball is missing (41% of graded frames).

    python -m saha.occluded.build          -> cache/evidence/occluded.parquet

v2, two sources fused in log-odds, written only on graded frames where the ball is not tracked
(TEAM.md, boundaries). p_dead stays NaN everywhere: that question is WS2's.

1. The bridge. Every missing-ball stretch has a sighting before it and one after it. On each side
   the team at the ball is voted over the tracked frames within ANCHOR_HORIZON of the gap: a frame
   votes when a player is within CONTROL_M of a slow ball (the same "touch" as saha.tools.selfcheck),
   and recent frames count more (half-life ANCHOR_HALF_LIFE). Inside the gap, possession is a
   two-state Markov chain that changes team at rate 1 / MEAN_HOLD_S per second; the forward-backward
   posterior between the two sides gives P(home) on every missing frame. Near a side it follows that
   side; in the middle of a long gap it fades to 0.5 (no opinion); when the two sides disagree it
   crosses 0.5 where a hand-over is most likely. Gaps and anchors are cut at the runs of the
   table's own `half` array, so the same code runs on our match and on WS5's external tables.

2. Smart Tagging phases. Each code's meaning comes from its official definition (OVERVIEW.md: the
   tag Team is "the team performing the event") and is fixed in TAG_SIDE: the tagged team has the
   ball (BUILD UP, PROGRESSION ...), or it does not (DEFENDING IN ..., DEFENSIVE TRANSITION), or
   the code gives no opinion (SET PIECES, GOALS, team N/A, any other code). P(tagged team has the
   ball) is TAG_P on the first side and 1 - TAG_P on the second. Nothing is measured on this match
   (DECISIONS D017). On a missing frame the overlapping tags are averaged in log-odds and added to
   the bridge at TAG_SCALE (weak evidence). An external table (WS5) has no Smart Tagging, so it
   gets no tags: never this match's.

The result is clipped to [P_MIN, P_MAX]. Every parameter below is set by football and physics
reasoning, with no fitting, and never on the public 5 minutes (CLAUDE.md, rule 2). WS5's external
data and `python -m saha.occluded.selfcheck` (label-free) check them; they never choose them
(DECISIONS D016, D017).

Read the match only through saha.common and write only through saha.common.evidence. Everything
printed is aggregate: counts and shares. Rules: CLAUDE.md.
"""

import argparse

import numpy as np

from saha.common import evidence, frame_table, kinematics, paths, smart_tagging
from saha.common.constants import FPS

# -- the team at the ball (same definition as the touches of saha.tools.selfcheck) -------------
CONTROL_M = 1.5          # metres: a player this close to the ball has it at his feet
CONTROL_SPEED = 5.0      # m/s, from the ball's positions (D015); faster is a pass or a shot
# -- the bridge --------------------------------------------------------------------------------
ANCHOR_HORIZON = 2 * FPS     # frames: look at most 2 s past each side of a gap
ANCHOR_HALF_LIFE = FPS / 2   # frames: a touch 0.5 s further from the gap counts half as much
ANCHOR_MAX = 0.9             # the most one side can say: the touch team is not always right
MEAN_HOLD_S = 12.0           # seconds between possession changes in open play: team possessions
                             # last 10-20 s in top-flight football, and the short contested ones
                             # count here too (docs/METRIC_NOTES.md), so the short end
# -- Smart Tagging -------------------------------------------------------------------------------
# Each code's side comes from its official definition (OVERVIEW.md, "Smart Tagging"), where the
# tag Team is "the team performing the event". It is fixed: nothing is measured on this match
# (DECISIONS D017). Codes are upper case after smart_tagging.load().
WITH_BALL, WITHOUT_BALL = 1, -1
TAG_SIDE = {
    "BUILD UP": WITH_BALL,                   # "offensive actions"
    "PROGRESSION": WITH_BALL,                # "offensive actions"
    "CREATING CHANCES": WITH_BALL,           # "offensive actions"
    "ATTACKING TRANSITION": WITH_BALL,       # "immediately after recovering the ball"
    "BALL IN FINAL THIRD": WITH_BALL,        # "the team maintains or plays the ball"
    "BALL IN THE BOX": WITH_BALL,            # the ball is played into the rival box
    "PLAYERS IN THE BOX": WITH_BALL,         # "attacking players ... during an offensive action"
    "LONG BALL": WITH_BALL,                  # the passer's team: a loose ball stays with the last
                                             # team that had it (CHALLENGE.md); the bridge's far
                                             # side says who got it
    "DEFENDING IN ATTACKING THIRD": WITHOUT_BALL,   # "defensive actions"
    "DEFENDING IN MIDDLE THIRD": WITHOUT_BALL,      # "defensive actions"
    "DEFENDING IN DEFENSIVE THIRD": WITHOUT_BALL,   # "defensive actions"
    "DEFENSIVE TRANSITION": WITHOUT_BALL,           # "immediately after losing possession"
}
# No opinion: SET PIECES (a restart's team is WS2's question, TEAM.md boundaries), GOALS (a
# stoppage and the other team's kick-off follow), team N/A, and any code not in TAG_SIDE (counted
# in the report, never an error).
NO_OPINION_CODES = (smart_tagging.SET_PIECES, "GOALS")
# P(tagged team has the ball) for a with-the-ball code; 1 - TAG_P for a without-the-ball code.
# More likely than not, but far from sure: Smart Tagging is automatic, with false positives and
# negatives, and its windows are phases, not possessions (DATA.md, OVERVIEW.md). No external
# dataset has Smart Tagging, so this cannot be tuned (D016): it stays weak on purpose.
TAG_P = 0.7
# Tags are weak evidence: half of their log-odds. With overlapping tags averaged, not summed, the
# tags add at most 0.5 x logit(0.7) ~ 0.42 nats per frame. Not tunable either, for the same reason.
TAG_SCALE = 0.5
# -- output ------------------------------------------------------------------------------------
# At most ~2.2 nats per frame. Since WS1's exact emissions a confident p_home no longer pushes
# against D (experiments/20261105_ws1_exact-emissions.md); the clip stays because the opinions
# inside one gap all come from the same two sides, so they are not independent, and no single
# frame should decide alone.
P_MIN, P_MAX = 0.1, 0.9
WEIGHT = 1.0

HOME, AWAY, NOBODY = 0, 1, -1


def params():
    """The parameters, as recorded in the evidence file's metadata."""
    return {"CONTROL_M": CONTROL_M, "CONTROL_SPEED": CONTROL_SPEED,
            "ANCHOR_HORIZON": ANCHOR_HORIZON, "ANCHOR_HALF_LIFE": ANCHOR_HALF_LIFE,
            "ANCHOR_MAX": ANCHOR_MAX, "MEAN_HOLD_S": MEAN_HOLD_S,
            "TAG_SIDE": dict(sorted(TAG_SIDE.items())),
            "NO_OPINION_CODES": list(NO_OPINION_CODES), "TAG_P": TAG_P,
            "TAG_SCALE": TAG_SCALE, "P_MIN": P_MIN, "P_MAX": P_MAX, "WEIGHT": WEIGHT}


def logit(p):
    p = np.asarray(p, dtype=np.float64)
    return np.log(p / (1.0 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


# -- the team at the ball ----------------------------------------------------------------------

def control_team(table):
    """int8 per frame: 0 home / 1 away when a player is within CONTROL_M of a slow, tracked ball
    on a graded frame, -1 elsewhere."""
    ball = table.ball_xy_m()
    distance, team = kinematics.nearest_player(ball, table.players_xy_m(), table.slot_team)
    speed = kinematics.displacement_speed(ball)
    held = (table.graded & table.ball_visible & (team >= 0) & (distance <= CONTROL_M)
            & (speed <= CONTROL_SPEED))
    return np.where(held, team, NOBODY).astype(np.int8)


def runs(mask):
    """[(start, stop)] of the True runs of a boolean array, half-open, in order."""
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(edges[0::2], edges[1::2])]


def half_runs(half):
    """[(start, stop)] of the graded halves of a `half` array (1 or 2 inside a graded half, 0
    elsewhere), in order. Two halves that touch stay apart."""
    half = np.asarray(half)
    found = [run for number in sorted(set(half[half > 0].tolist()))
             for run in runs(half == number)]
    return sorted(found)


# -- 1. the bridge -----------------------------------------------------------------------------

def anchor(control, edge, step, lo, hi):
    """P(home) on one side of a gap, from the touches in the ANCHOR_HORIZON frames that start at
    `edge` and go in direction `step` (-1 back in time, +1 forward), inside [lo, hi).

    0.5 + (ANCHOR_MAX - 0.5) * (home - away) / (home + away + 1), with recency-weighted touch
    counts: one touch right at the gap gives about half of the maximum, several give nearly all
    of it, and a side without touches gives 0.5 (no opinion).
    """
    k = np.arange(ANCHOR_HORIZON)
    frames = edge + step * k
    inside = (frames >= lo) & (frames < hi)
    teams = control[frames[inside]]
    w = np.exp2(-k[inside] / ANCHOR_HALF_LIFE)
    home, away = float(w[teams == HOME].sum()), float(w[teams == AWAY].sum())
    return 0.5 + (ANCHOR_MAX - 0.5) * (home - away) / (home + away + 1.0)


def stay_probability(seconds, rate):
    """P(same team after `seconds`) in a symmetric two-state chain that switches at `rate`/s."""
    return 0.5 * (1.0 + np.exp(-2.0 * rate * np.asarray(seconds, dtype=np.float64)))


def bridge(p_before, p_after, length, rate=1.0 / MEAN_HOLD_S):
    """P(home) on the `length` missing frames between two sightings.

    p_before belongs to the last tracked frame before the gap, p_after to the first one after it.
    Frame i of the gap is (i + 1) frames after the first and (length - i) frames before the second.
    """
    i = np.arange(length)
    stay_before = stay_probability((i + 1) / FPS, rate)
    stay_after = stay_probability((length - i) / FPS, rate)
    forward = p_before * stay_before + (1.0 - p_before) * (1.0 - stay_before)
    backward = p_after * stay_after + (1.0 - p_after) * (1.0 - stay_after)
    home = forward * backward
    return home / (home + (1.0 - forward) * (1.0 - backward))


def bridge_gaps(control, missing, half):
    """(p, informed, gaps): the bridge on every missing stretch of each graded half.

    `half` is the table's half array (1 or 2 inside a graded half, 0 elsewhere): on our match
    alignment.half_of_frames(), on an external table its own halves. p is P(home) on missing
    graded frames (NaN elsewhere); informed marks the frames of gaps with at least one side that
    has an opinion; gaps is [(start, stop, p_before, p_after)]. A gap never reaches across a half:
    the side beyond the half's edge has no opinion.
    """
    p = np.full(len(control), np.nan)
    informed = np.zeros(len(control), dtype=bool)
    gaps = []
    for lo, hi in half_runs(half):
        for start, stop in runs(missing[lo:hi]):
            start, stop = start + lo, stop + lo
            before = anchor(control, start - 1, -1, lo, hi)
            after = anchor(control, stop, +1, lo, hi)
            p[start:stop] = bridge(before, after, stop - start)
            informed[start:stop] = before != 0.5 or after != 0.5
            gaps.append((start, stop, before, after))
    return p, informed, gaps


# -- 2. Smart Tagging phases ---------------------------------------------------------------------

def tag_rows(tags):
    """The rows of a Smart Tagging table; none for no table (an external match)."""
    return [] if tags is None else list(tags.itertuples(index=False))


def tag_instances(tags, n_frames):
    """[(code, team code, start, stop)] of the tags with an opinion: a code in TAG_SIDE, team H or
    A, and a window that overlaps the video. Sorted, so nothing depends on file order."""
    out = []
    for row in tag_rows(tags):
        if row.code not in TAG_SIDE or row.team not in ("H", "A"):
            continue
        start, stop = max(int(row.start_frame), 0), min(int(row.end_frame), n_frames)
        if start < stop:
            out.append((row.code, HOME if row.team == "H" else AWAY, start, stop))
    return sorted(out)


def tag_counts(tags):
    """({code: {"instances", "side"}}, {reason: count}): how many instances each code of TAG_SIDE
    has with a team, and how many instances were left out, by reason (SET PIECES, GOALS, N/A,
    unknown). Counts only: nothing about agreement with the touches (DECISIONS D017)."""
    codes = {code: {"instances": 0, "side": "with" if side == WITH_BALL else "without"}
             for code, side in TAG_SIDE.items()}
    left_out = {code: 0 for code in NO_OPINION_CODES}
    left_out.update({"N/A": 0, "unknown": 0})
    for row in tag_rows(tags):
        if row.code in NO_OPINION_CODES:
            left_out[row.code] += 1
        elif row.code not in TAG_SIDE:
            left_out["unknown"] += 1
        elif row.team not in ("H", "A"):
            left_out["N/A"] += 1
        else:
            codes[row.code]["instances"] += 1
    return codes, left_out


def tag_logit(instances, n_frames):
    """Mean log-odds for home over the tags covering each frame; NaN where no tag covers it.

    A tag says logit(TAG_P) for the tagged team when its code is a with-the-ball code, and the
    same against it when it is a without-the-ball code. Overlapping tags are averaged, not summed.
    """
    value = float(logit(TAG_P))
    total = np.zeros(n_frames)
    count = np.zeros(n_frames, dtype=np.int64)
    for code, team, start, stop in instances:
        for_tagged = value * TAG_SIDE[code]
        total[start:stop] += for_tagged if team == HOME else -for_tagged
        count[start:stop] += 1
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


# -- the evidence ------------------------------------------------------------------------------

def opinion(control, missing, instances, half):
    """(p_home, report): P(home) on missing graded frames where we have an opinion, NaN elsewhere.

    `half` is the table's half array (bridge_gaps). Split from build() so that the self-check can
    run it on a ball it hid itself.
    """
    n = len(control)
    p_bridge, informed, gaps = bridge_gaps(control, missing, half)
    tags = tag_logit(instances, n)
    tagged = missing & np.isfinite(tags)
    has = missing & (informed | tagged)
    total = logit(np.where(has, p_bridge, 0.5)) + TAG_SCALE * np.where(tagged, tags, 0.0)
    p_home = np.full(n, np.nan)
    p_home[has] = np.clip(sigmoid(total[has]), P_MIN, P_MAX)
    report = {"gaps": gaps, "missing": int(missing.sum()),
              "covered": int(has.sum()), "tagged": int(tagged.sum())}
    return p_home, report


def build(table, tags=None, report=None):
    """(p_dead, p_home, weight) over every video frame; fills `report` (a dict) if given.

    `tags` defaults to this match's Smart Tagging, except on an external table (WS5's harness,
    "external" in table.meta, the same test as saha.deadball.build), which has none: it never gets
    this match's tags.
    """
    if tags is None and "external" not in table.meta:
        tags = smart_tagging.load()
    control = control_team(table)
    missing = table.graded & ~table.ball_visible
    p, found = opinion(control, missing, tag_instances(tags, table.n_frames), table.half)
    p_dead, p_home, weight = evidence.empty(table.n_frames)
    has = np.isfinite(p)
    p_home[has] = p[has]
    weight[has] = WEIGHT
    if report is not None:
        report.update(found)
        report["codes"], report["left_out"] = tag_counts(tags)
        report["touches"] = int((control >= 0).sum())
        report["tracked"] = int((table.graded & table.ball_visible).sum())
    return p_dead, p_home, weight


def summary(report, p_home):
    """Aggregate lines for the console: counts and shares, never frames."""
    gaps = report["gaps"]
    lengths = np.array([stop - start for start, stop, _, _ in gaps], dtype=np.float64) / FPS

    def side(p):
        return 0 if p == 0.5 else (1 if p > 0.5 else -1)

    both = [g for g in gaps if side(g[2]) and side(g[3])]
    agree = sum(side(g[2]) == side(g[3]) for g in both)
    none = sum(not side(g[2]) and not side(g[3]) for g in gaps)
    covered = np.isfinite(p_home)
    confident = covered & ((p_home >= 0.7) | (p_home <= 0.3))
    lines = [
        "touches (team at the ball) on {} of {} tracked graded frames ({:.1%})".format(
            report["touches"], report["tracked"], report["touches"] / max(report["tracked"], 1)),
        "missing-ball gaps {}  length s: p50 {:.1f}  p90 {:.1f}  max {:.1f}".format(
            len(gaps), *(np.percentile(lengths, [50, 90, 100]) if len(gaps) else (0, 0, 0))),
        "  both sides with a team {} (same team {}, a hand-over inside {}), neither side {}"
        .format(len(both), agree, len(both) - agree, none),
        "missing graded frames {}: opinion on {:.1%}, a tag on {:.1%}, |p - 0.5| >= 0.2 on {:.1%}"
        .format(report["missing"], report["covered"] / max(report["missing"], 1),
                report["tagged"] / max(report["missing"], 1),
                confident.sum() / max(report["missing"], 1)),
        "Smart Tagging codes (side fixed by the official definitions; P(tagged team has the "
        "ball) = {:.2f} with, {:.2f} without; TAG_SCALE {:g})".format(TAG_P, 1.0 - TAG_P,
                                                                      TAG_SCALE),
    ]
    for code, entry in sorted(report["codes"].items()):
        lines.append("  {:<32} instances {:3d}  side {}".format(
            code, entry["instances"], entry["side"]))
    left_out = report["left_out"]
    lines.append("  left out: {} instances ({})".format(
        sum(left_out.values()),
        ", ".join("{} {}".format(reason, n) for reason, n in left_out.items())))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", help="data bundle root (default: MTAH_DATA_DIR or ./data)")
    args = parser.parse_args(argv)
    if args.data:
        paths.use_data_dir(args.data)
    report = {}
    p_dead, p_home, weight = build(frame_table.load(), report=report)
    print(summary(report, p_home))
    path = evidence.write_evidence("occluded", p_dead, p_home, weight,
                                   producer="saha.occluded.build", params=params())
    print("\n".join(evidence.validate_file(path)[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
