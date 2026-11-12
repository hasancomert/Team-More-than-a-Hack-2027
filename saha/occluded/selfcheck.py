"""Label-free check of WS4: hide the ball where it was tracked, and see whether the team comes back.

    python -m saha.occluded.selfcheck
    python -m saha.occluded.selfcheck --seconds 2 5 10 20

For each length, a stretch of that many seconds is hidden in the middle of every tracked-ball run
of a graded half that is long enough to keep ANCHOR_HORIZON of tracked ball on both sides. The
WS4 opinion is recomputed as if the ball were missing there, and compared with the team at the
ball (the touches of saha.occluded.build) on the hidden frames. Two variants: the bridge alone,
and the bridge with the Smart Tagging phases.

It reports, per length: how many stretches qualified, coverage, accuracy, and calibration (for
each band of confidence max(p, 1 - p), the share right). A p of 0.8 should be right about 80% of
the time (TEAM.md, WS4 success).

No ground truth is used. It is a check that catches errors and never chooses a value (DECISIONS
D016, D017): no WS4 setting is changed because of its output. Its caveats: the target is the
touch team, not the truth, and a tracked ball is easier than a real gap (the real gaps are longer
and happen in crowds and in the air). Everything printed is aggregate.
"""

import argparse

import numpy as np

from saha.common import frame_table, paths, smart_tagging
from saha.common.constants import FPS
from saha.occluded import build as occluded

DEFAULT_SECONDS = (2.0, 5.0, 10.0, 20.0)
BANDS = (0.5, 0.6, 0.7, 0.8, 0.9 + 1e-9)


def hidden_stretches(visible_graded, length):
    """Boolean mask: `length` frames hidden in the middle of every tracked run long enough."""
    hidden = np.zeros(len(visible_graded), dtype=bool)
    count = 0
    for start, stop in occluded.runs(visible_graded):
        if stop - start < length + 2 * occluded.ANCHOR_HORIZON:
            continue
        middle = start + (stop - start - length) // 2
        hidden[middle:middle + length] = True
        count += 1
    return hidden, count


def score(p, target, hidden):
    """(coverage, accuracy, [(band low, band high, n, mean confidence, share right)])."""
    judged = hidden & (target >= 0)
    has = judged & np.isfinite(p)
    if not has.any():
        return 0.0, None, []
    right = (p[has] > 0.5) == (target[has] == occluded.HOME)
    confidence = np.maximum(p[has], 1.0 - p[has])
    bands = []
    for low, high in zip(BANDS[:-1], BANDS[1:]):
        inside = (confidence >= low) & (confidence < high)
        if inside.any():
            bands.append((low, min(high, 0.9), int(inside.sum()),
                          float(confidence[inside].mean()), float(right[inside].mean())))
    return float(has.sum() / judged.sum()), float(right.mean()), bands


def run(table, tags, seconds):
    control = occluded.control_team(table)
    missing = table.graded & ~table.ball_visible
    instances = occluded.tag_instances(tags, table.n_frames)
    lines = ["touches on {} graded frames; target = the touch team on the hidden frames".format(
        int((control >= 0).sum()))]
    saved = occluded.TAG_SCALE
    try:
        for length_s in seconds:
            length = int(round(length_s * FPS))
            hidden, count = hidden_stretches(table.graded & table.ball_visible, length)
            lines.append("hide {:g} s: {} stretches, {} frames with a touch team".format(
                length_s, count, int((hidden & (control >= 0)).sum())))
            if not count:
                continue
            reduced = np.where(hidden, occluded.NOBODY, control).astype(np.int8)
            for name, scale in (("bridge", 0.0), ("bridge + tags", saved)):
                occluded.TAG_SCALE = scale
                p, _ = occluded.opinion(reduced, missing | hidden, instances, table.half)
                coverage, accuracy, bands = score(p, control, hidden)
                lines.append("  {:<14} coverage {:5.1%}  accuracy {}".format(
                    name, coverage, "--" if accuracy is None else "{:.3f}".format(accuracy)))
                for low, high, n, mean, share in bands:
                    lines.append("    confidence {:.1f}-{:.1f}: {:6d} frames, mean {:.3f}, "
                                 "right {:.3f}".format(low, high, n, mean, share))
    finally:
        occluded.TAG_SCALE = saved
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", help="data bundle root (default: MTAH_DATA_DIR or ./data)")
    parser.add_argument("--seconds", type=float, nargs="+", default=list(DEFAULT_SECONDS))
    args = parser.parse_args(argv)
    if args.data:
        paths.use_data_dir(args.data)
    print(run(frame_table.load(), smart_tagging.load(), args.seconds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
