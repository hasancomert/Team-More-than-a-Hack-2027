"""WS2: Dead ball (D) detection.

    python -m saha.deadball.build          -> cache/evidence/deadball.parquet

STUB written by WS1 in Phase 0: a valid file that has no opinion anywhere (all NaN, weight 0), so
the pipeline runs this package from day one. The owner replaces build() and keeps the interface:

    p_dead on every frame you can judge; p_home only inside a restart window whose team
    you know. Boundaries: TEAM.md.

Starting ideas (TEAM.md, not binding):
    ball out of the pitch; ball unseen for long while players slow down; set-piece formations;
    Smart Tagging restarts (Throw-in, Corner Kick, Goal Kick, Free Kick, Kick Off) end a
    dead-ball period; after a goal, D until the kick-off.

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
    path = evidence.write_evidence("deadball", p_dead, p_home, weight,
                                   producer="saha.deadball.build", params={"stub": True})
    print("\n".join(evidence.validate_file(path)[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
