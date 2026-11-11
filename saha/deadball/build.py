"""WS2: Dead ball (D) detection.

    python -m saha.deadball.build              -> cache/evidence/deadball.parquet
    python -m saha.deadball.build --whistle    the referee's whistle too (not the default, below)

p_dead is WS2's decision, dead or alive, on every graded frame: the stoppages that the dead-ball
classifier (trained on SkillCorner, train.py) and the stop and restart events (events.py) give
through WS2's event-gated decoder (segment.py). It is written as a near-certain probability with
a weight high enough that the combiner keeps every stoppage and every stretch of play exactly as
WS2 decided them at the current costs (WEIGHT below). p_home is not written: WS2 gives no team.

The classifier is committed in saha/deadball/models/ with its manifest (DECISIONS D020);
saha/deadball/train.py rebuilds it from the external data. SAHA_DEADBALL_MODEL names another
file, or "none" for no classifier and no opinion, as the Phase 0 stub had.

The whistle is off by default. Its frequency bands were measured on this match's audio, which
the organizers' rules do not allow, and it needs the tactical video and ffmpeg on every machine
that reproduces the submission, a team decision (STATUS.md, WS2 items 5 and 6; D006).

build(table) runs on any frame table: ours, or an external one from WS5's harness, where there
is no Smart Tagging (so no tagged kick-offs) and no audio. Read the match only through
saha.common and write only through saha.common.evidence.write_evidence. Rules: CLAUDE.md.
Design and external results: docs/tech/ws2_deadball.md.
"""

import argparse

import numpy as np

from saha.common import evidence, frame_table, paths, smart_tagging
from saha.common.constants import PITCH_LENGTH_M, PITCH_WIDTH_M

from . import segment
from .dead import model_disabled

P_DEAD = 0.99       # the decision as a probability: 1 - the combiner's eps, so nothing is clipped
# One frame of p = 0.99 at this weight is 20 x 4.6 = 92 nats of dead-vs-alive evidence, doubled in
# the combiner's scores to 184 > 2 x 2 x 35 (its doubled costs of a stoppage's two transitions), so
# the combiner reproduces WS2's path frame for frame at the current costs.
WEIGHT = 20.0


def kickoff_windows(table):
    """[(start, stop)] frames of the Smart Tagging kick-offs; none on an external table."""
    if "external" in table.meta:
        return []
    tags = smart_tagging.load()
    return [(int(row.start_frame), int(row.end_frame)) for row in tags.itertuples()
            if isinstance(row.type, str) and "kick off" in row.type.lower()]


def whistles():
    from .audio import load_whistle_score, whistle_events
    return whistle_events(load_whistle_score())


def dead_ball(table, whistle=False, verbose=True):
    """(n,) bool: WS2's dead-ball decision on graded frames, False elsewhere."""
    players = np.concatenate([table.players_xy, table.players_speed[..., None]], axis=-1)
    ev = segment.evidence(table.ball_xy, players, table.slot_team,
                          (PITCH_LENGTH_M, PITCH_WIDTH_M), segment.blocks_of(table.half),
                          kickoff_windows(table), whistles() if whistle else (), verbose)
    return segment.decode(ev)


def build(table, whistle=False, verbose=False):
    """(p_dead, p_home, weight) arrays over every video frame."""
    if model_disabled():
        return evidence.empty(table.n_frames)
    dead = dead_ball(table, whistle, verbose)
    graded = table.half > 0
    p_dead = np.where(graded, np.where(dead, P_DEAD, 1.0 - P_DEAD), np.nan).astype(np.float32)
    p_home = np.full(table.n_frames, np.nan, dtype=np.float32)
    weight = np.where(graded, WEIGHT, 0.0).astype(np.float32)
    return p_dead, p_home, weight


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", help="data bundle root (default: MTAH_DATA_DIR or ./data)")
    parser.add_argument("--whistle", action="store_true",
                        help="use the referee's whistle (needs the video and ffmpeg)")
    args = parser.parse_args(argv)
    if args.data:
        paths.use_data_dir(args.data)
    p_dead, p_home, weight = build(frame_table.load(), whistle=args.whistle, verbose=True)
    path = evidence.write_evidence("deadball", p_dead, p_home, weight,
                                   producer="saha.deadball.build",
                                   params={"whistle": args.whistle, "p_dead": P_DEAD,
                                           "weight": WEIGHT})
    print("\n".join(evidence.validate_file(path)[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
