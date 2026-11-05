"""WS4: Possession while the ball is missing (41% of graded frames).

    python -m saha.occluded.build          -> cache/evidence/occluded.parquet

v1, two sources fused in log-odds, written only on graded frames where the ball is not tracked
(TEAM.md, boundaries). p_dead stays NaN everywhere: that question is WS2's.

1. The bridge. Every missing-ball stretch has a sighting before it and one after it. On each side
   the team at the ball is voted over the tracked frames within ANCHOR_HORIZON of the gap: a frame
   votes when a player is within CONTROL_M of a slow ball (the same "touch" as saha.tools.selfcheck),
   and recent frames count more (half-life ANCHOR_HALF_LIFE). Inside the gap, possession is a
   two-state Markov chain that changes team at rate 1 / MEAN_HOLD_S per second; the forward-backward
   posterior between the two sides gives P(home) on every missing frame. Near a side it follows that
   side; in the middle of a long gap it fades to 0.5 (no opinion); when the two sides disagree it
   crosses 0.5 where a hand-over is most likely.

2. Smart Tagging phases. Every tag with a team, except SET PIECES (a restart's team is WS2's
   question, TEAM.md), says something about who has the ball. Whether the tagged team is the team
   in possession (BUILD UP, PROGRESSION ...) or out of it (DEFENDING IN ...) is not assumed: for
   each code the algorithm measures, on the tracked frames inside its instances, how often the team
   at the ball is the tagged team. No ground truth is used. That share, shrunk towards 0.5 by
   TAG_PRIOR pseudo-instances, is P(tagged team has the ball). On a missing frame the overlapping
   tags are averaged in log-odds and added to the bridge at TAG_SCALE (weak evidence).

The result is clipped to [P_MIN, P_MAX]. The parameters below come from football and physics
reasoning, not from the public 5 minutes (CLAUDE.md, rule 2); WS5's external data and
`python -m saha.occluded.selfcheck` (label-free) are where they get checked.

Read the match only through saha.common and write only through saha.common.evidence. Everything
printed is aggregate: counts, shares, per-code rates. Rules: CLAUDE.md.
"""

import argparse

import numpy as np

from saha.common import alignment, evidence, frame_table, kinematics, paths, smart_tagging
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
EXCLUDED_CODES = (smart_tagging.SET_PIECES,)   # restart windows: WS2 (TEAM.md, boundaries)
TAG_PRIOR = 5.0          # pseudo-instances on each side when measuring a code's reliability
TAG_SCALE = 0.5          # tags are weak evidence: half of their log-odds
# -- output ------------------------------------------------------------------------------------
P_MIN, P_MAX = 0.1, 0.9  # at most ~2.2 nats per frame, so no single frame of ours decides alone
WEIGHT = 1.0

HOME, AWAY, NOBODY = 0, 1, -1


def params():
    """The parameters, as recorded in the evidence file's metadata."""
    return {"CONTROL_M": CONTROL_M, "CONTROL_SPEED": CONTROL_SPEED,
            "ANCHOR_HORIZON": ANCHOR_HORIZON, "ANCHOR_HALF_LIFE": ANCHOR_HALF_LIFE,
            "ANCHOR_MAX": ANCHOR_MAX, "MEAN_HOLD_S": MEAN_HOLD_S,
            "EXCLUDED_CODES": list(EXCLUDED_CODES), "TAG_PRIOR": TAG_PRIOR,
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


def bridge_gaps(control, missing):
    """(p, informed, gaps): the bridge on every missing stretch of each graded half.

    p is P(home) on missing graded frames (NaN elsewhere); informed marks the frames of gaps with
    at least one side that has an opinion; gaps is [(start, stop, p_before, p_after)]. A gap never
    reaches across a half: the side beyond the half's edge has no opinion.
    """
    p = np.full(len(control), np.nan)
    informed = np.zeros(len(control), dtype=bool)
    gaps = []
    for _, lo, hi in alignment.half_slices():
        hi = min(hi, len(control))
        if lo >= hi:
            continue
        for start, stop in runs(missing[lo:hi]):
            start, stop = start + lo, stop + lo
            before = anchor(control, start - 1, -1, lo, hi)
            after = anchor(control, stop, +1, lo, hi)
            p[start:stop] = bridge(before, after, stop - start)
            informed[start:stop] = before != 0.5 or after != 0.5
            gaps.append((start, stop, before, after))
    return p, informed, gaps


# -- 2. Smart Tagging phases ---------------------------------------------------------------------

def tag_instances(tags, n_frames):
    """[(code, team code, start, stop)] of the usable tags: a team, a code not excluded, and a
    window that overlaps the video. Sorted, so nothing depends on file order."""
    out = []
    for row in tags.itertuples(index=False):
        if row.code in EXCLUDED_CODES or row.team not in ("H", "A"):
            continue
        start, stop = max(int(row.start_frame), 0), min(int(row.end_frame), n_frames)
        if start < stop:
            out.append((row.code, HOME if row.team == "H" else AWAY, start, stop))
    return sorted(out)


def tag_reliability(instances, control):
    """{code: {"instances", "informative", "agree", "p"}}: P(the tagged team has the ball).

    An instance is informative when its touches have a majority team; it agrees when that team is
    the tagged one. p = (agree + TAG_PRIOR) / (informative + 2 TAG_PRIOR), so a code that is
    rarely informative stays near 0.5, and a code whose team is the defending side comes out
    below 0.5 by itself.
    """
    codes = {}
    for code, team, start, stop in instances:
        entry = codes.setdefault(code, {"instances": 0, "informative": 0, "agree": 0})
        entry["instances"] += 1
        window = control[start:stop]
        home, away = int((window == HOME).sum()), int((window == AWAY).sum())
        if home == away:
            continue
        entry["informative"] += 1
        entry["agree"] += int((HOME if home > away else AWAY) == team)
    for entry in codes.values():
        entry["p"] = (entry["agree"] + TAG_PRIOR) / (entry["informative"] + 2.0 * TAG_PRIOR)
    return codes


def tag_logit(instances, codes, n_frames):
    """Mean log-odds for home over the tags covering each frame; NaN where no tag covers it."""
    total = np.zeros(n_frames)
    count = np.zeros(n_frames, dtype=np.int64)
    for code, team, start, stop in instances:
        value = float(logit(codes[code]["p"]))
        total[start:stop] += value if team == HOME else -value
        count[start:stop] += 1
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


# -- the evidence ------------------------------------------------------------------------------

def opinion(control, missing, instances):
    """(p_home, report): P(home) on missing graded frames where we have an opinion, NaN elsewhere.

    Split from build() so that the self-check can run it on a ball it hid itself.
    """
    n = len(control)
    p_bridge, informed, gaps = bridge_gaps(control, missing)
    codes = tag_reliability(instances, control)
    tags = tag_logit(instances, codes, n)
    tagged = missing & np.isfinite(tags)
    has = missing & (informed | tagged)
    total = logit(np.where(has, p_bridge, 0.5)) + TAG_SCALE * np.where(tagged, tags, 0.0)
    p_home = np.full(n, np.nan)
    p_home[has] = np.clip(sigmoid(total[has]), P_MIN, P_MAX)
    report = {"gaps": gaps, "codes": codes, "missing": int(missing.sum()),
              "covered": int(has.sum()), "tagged": int(tagged.sum())}
    return p_home, report


def build(table, tags=None, report=None):
    """(p_dead, p_home, weight) over every video frame; fills `report` (a dict) if given."""
    if tags is None:
        tags = smart_tagging.load()
    control = control_team(table)
    missing = table.graded & ~table.ball_visible
    p, found = opinion(control, missing, tag_instances(tags, table.n_frames))
    p_dead, p_home, weight = evidence.empty(table.n_frames)
    has = np.isfinite(p)
    p_home[has] = p[has]
    weight[has] = WEIGHT
    if report is not None:
        report.update(found)
        report["touches"] = int((control >= 0).sum())
        report["tracked"] = int((table.graded & table.ball_visible).sum())
    return p_dead, p_home, weight


def summary(report, p_home):
    """Aggregate lines for the console: counts, shares and per-code rates, never frames."""
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
        "Smart Tagging codes: P(tagged team has the ball), measured on touches, no ground truth",
    ]
    for code, entry in sorted(report["codes"].items()):
        lines.append("  {:<32} instances {:3d}  informative {:3d}  agree {:3d}  p {:.3f}".format(
            code, entry["instances"], entry["informative"], entry["agree"], entry["p"]))
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
