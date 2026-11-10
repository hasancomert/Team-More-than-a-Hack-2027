"""WS2's dead-ball decision on WS5's DFL matches: frames, stoppages, and the frozen-ball artefact.

    python -m saha.external.build --all          # once: WS5's DFL matches
    python -m saha.deadball.external_check

For each converted DFL match, degraded to our conditions exactly as saha.external.score does
it, the decision of saha.deadball.build is compared with DFL's BallStatus (0 = dead):
dead-frame precision, recall and F1, stoppage F1 (one-to-one IoU matching averaged over
0.3-0.7, like the ranking metric), and how far matched stoppages start and end from the truth.

DFL's tracking freezes the ball whenever play is dead: its position stops changing to the last
bit. That is a cue no real tracker gives, and our classifier reads the ball. So every match is
scored twice: as delivered, and with the ball removed wherever its position has not changed for
FROZEN_FRAMES frames or more (no labels involved; in our tracking the ball never sits still to
the last bit for that long). The difference is what the artefact is worth.

It also splits the live frames we call dead by whether the harness hid the ball there: the
degradation hides it in long stretches placed at random, whatever the play.

External data: the numbers may be shared. Nothing here chooses a parameter.
"""

import argparse
from dataclasses import replace

import numpy as np

from saha.common.constants import D, X
from saha.external import degrade, dfl

from . import build

FROZEN_FRAMES = 5
IOUS = (0.3, 0.4, 0.5, 0.6, 0.7)


def frozen(ball_xy, min_frames=FROZEN_FRAMES):
    """(n,) bool: frames inside a run of >= min_frames with the ball at exactly one position."""
    same = np.zeros(len(ball_xy), dtype=bool)
    same[1:] = np.all(ball_xy[1:] == ball_xy[:-1], axis=1)
    out = np.zeros(len(ball_xy), dtype=bool)
    edges = np.diff(np.concatenate(([0], same.astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if b - a + 1 >= min_frames:
            out[a - 1:b] = True      # the first frame of the run is frozen too
    return out


def runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def stoppage_scores(true, pred):
    """(stoppage F1 over IOUS, median start and end offsets in s of the pairs matched at 0.3)."""
    T, P = runs(true), runs(pred)
    pairs = []
    for i, (a, b) in enumerate(T):
        for j, (c, e) in enumerate(P):
            inter = min(b, e) - max(a, c)
            if inter > 0:
                pairs.append((inter / (max(b, e) - min(a, c)), i, j))
    pairs.sort(key=lambda x: (-x[0], x[1], x[2]))
    f1s, offsets = [], []
    for thr in IOUS:
        used_t, used_p, matched = set(), set(), []
        for iou, i, j in pairs:
            if iou >= thr and i not in used_t and j not in used_p:
                used_t.add(i)
                used_p.add(j)
                matched.append((i, j))
        f1s.append(2 * len(matched) / max(len(T) + len(P), 1))
        if thr == IOUS[0]:
            offsets = [(P[j][0] - T[i][0], P[j][1] - T[i][1]) for i, j in matched]
    off = np.array(offsets, dtype=float).reshape(-1, 2) / 25.0
    return (float(np.mean(f1s)),) + tuple(np.median(off, axis=0)) if len(off) else (
        float(np.mean(f1s)), np.nan, np.nan)


def check(match, unfreeze):
    if unfreeze:
        ball = match.ball_xy.copy()
        ball[frozen(ball)] = np.nan
        match = replace(match, ball_xy=ball)
    table, labels = degrade.degrade(match)
    dead = build.dead_ball(table, verbose=False)
    scored = labels != X
    true, pred = (labels == D) & scored, dead & scored
    tp = float(np.sum(true & pred))
    p, r = tp / max(pred.sum(), 1), tp / max(true.sum(), 1)
    stop_f1, start, end = stoppage_scores(true, pred)
    live = scored & ~true
    hidden = ~table.ball_visible
    return {"P": p, "R": r, "F1": 2 * p * r / max(p + r, 1e-9), "stop_F1": stop_f1,
            "dead_true": float(true[scored].mean()), "dead_pred": float(pred[scored].mean()),
            "start_s": start, "end_s": end,
            "live_dead_hidden": float(pred[live & hidden].mean()),
            "live_dead_visible": float(pred[live & ~hidden].mean())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matches", nargs="*", help="DFL match ids (default: converted ones)")
    args = parser.parse_args(argv)
    matches = args.matches or dfl.converted_matches()
    if not matches:
        raise SystemExit("no converted DFL match: run python -m saha.external.build --all")
    fields = ("P", "R", "F1", "stop_F1", "dead_true", "dead_pred", "start_s", "end_s",
              "live_dead_hidden", "live_dead_visible")
    totals = {False: [], True: []}
    print("%-8s %-10s %6s %6s %6s %7s %9s %9s %7s %7s %9s %9s" % (
        "match", "ball", "P", "R", "F1", "stopF1", "D true", "D ours", "start s", "end s",
        "live→D h", "live→D v"))
    for match_id in matches:
        match = dfl.prepare(match_id)
        scored = match.labels != X
        is_dead = (match.labels == D) & scored
        ice = frozen(match.ball_xy)
        for unfreeze in (False, True):
            row = check(match, unfreeze)
            totals[unfreeze].append(row)
            print("%-8s %-10s %6.3f %6.3f %6.3f %7.3f %8.1f%% %8.1f%% %+7.1f %+7.1f %8.1f%% "
                  "%8.1f%%" % ((match_id, "unfrozen" if unfreeze else "delivered") + tuple(
                      row[k] * (100 if k.startswith(("dead", "live")) else 1) for k in fields)))
        print("%-8s frozen ball on %.1f%% of dead frames, %.1f%% of live ones" % (
            "", 100 * ice[is_dead].mean(), 100 * ice[scored & ~is_dead].mean()))
    print("live→D h / v: share of live frames called dead where the harness hid the ball / "
          "where the ball is in view")
    for unfreeze in (False, True):
        rows = totals[unfreeze]
        print("%-8s %-10s " % ("mean", "unfrozen" if unfreeze else "delivered") + " ".join(
            "%s %.3f" % (k, np.nanmean([r[k] for r in rows])) for k in fields))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
