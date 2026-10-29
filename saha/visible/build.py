"""WS3: Possession while the ball is tracked.

    python -m saha.visible.build          -> cache/evidence/visible.parquet

STUB written by WS1 in Phase 0: a valid file that has no opinion anywhere (all NaN, weight 0), so
the pipeline runs this package from day one. The owner replaces build() and keeps the interface:

    p_home only on frames where the ball is tracked (frame_table ball_visible); NaN and
    weight 0 elsewhere. p_dead stays NaN. Boundaries: TEAM.md.

Starting ideas (TEAM.md, not binding):
    nearest player + a control condition (ball near one player for several frames, low ball
    speed). Loose or contested balls keep the last team; possession changes only when the
    opponent controls the ball -- model it with hysteresis.

Read the match only through saha.common (frame_table, smart_tagging, homography, alignment) and
write only through saha.common.evidence.write_evidence. Rules: CLAUDE.md.
"""

import argparse

from saha.common import evidence, frame_table, paths


def build(table):
    """(p_dead, p_home, weight) arrays over every video frame. Stub: no opinion anywhere."""
    return evidence.empty(table.n_frames)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", help="data bundle root (default: MTAH_DATA_DIR or ./data)")
    args = parser.parse_args(argv)
    if args.data:
        paths.use_data_dir(args.data)
    p_dead, p_home, weight = build(frame_table.load())
    path = evidence.write_evidence("visible", p_dead, p_home, weight,
                                   producer="saha.visible.build", params={"stub": True})
    print("\n".join(evidence.validate_file(path)[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
