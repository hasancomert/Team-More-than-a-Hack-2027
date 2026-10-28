"""WS4: Possession while the ball is missing (41% of graded frames).

    python -m saha.occluded.build          -> cache/evidence/occluded.parquet

STUB written by WS1 in Phase 0: a valid file that has no opinion anywhere (all NaN, weight 0), so
the pipeline runs this package from day one. The owner replaces build() and keeps the interface:

    p_home on frames where the ball is not tracked; p_dead stays NaN.

Starting ideas (TEAM.md, not binding):
    interpolate the ball across short gaps; over long gaps use team shape (who attacks,
    defensive block, pressing) and Smart Tagging phase teams; treat Team = N/A (every throw-in)
    separately. Look at PathCRF, minding that our slots carry no identity.

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
    path = evidence.write_evidence("occluded", p_dead, p_home, weight,
                                   producer="saha.occluded.build", params={"stub": True})
    print("\n".join(evidence.validate_file(path)[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
