"""WS1: who has the ball while it is missing, from the two teams' shape and motion.

    python -m saha.shape.build          -> cache/evidence/shape.parquet

- **Where:** `p_home` only on graded frames where the ball is not tracked, the frames WS4 also
  covers. `p_dead` stays NaN everywhere. DECISIONS D026 records the exception to D011 (one
  opinion per question per frame).
- **What:** the logistic model of `saha/shape/model.json`, applied to `saha.shape.features`. It was
  trained on WS5's DFL matches only (`python -m saha.shape.train`), and nothing of our match is
  fitted.
- **Strength:** the probability is clipped to [P_MIN, P_MAX] and written at weight 1.
  `configs/pipeline.toml` sets its scale.

It runs on any frame table: ours, or one of WS5's external tables. Everything printed is
aggregate.
"""

import argparse
import hashlib
import json

import numpy as np

from saha.common import evidence, frame_table, paths
from saha.shape import features as shape_features
from saha.shape.train import MODEL_PATH, predict

P_MIN, P_MAX = 0.1, 0.9     # at most ~2.2 nats per frame and unit of scale: one frame never
                            # decides alone, and the frames of one gap are not independent
WEIGHT = 1.0
MIN_MARGIN_M = 10.0         # metres between the two teams' players nearest the low-x goal (median
                            # over a half). The defending side's is its goalkeeper or last
                            # defender; the other side's is a forward, who stays further out to
                            # be onside: 21-27 m apart on WS5's DFL matches. Below 10 m the
                            # direction is unclear and the half gets no opinion.


def load_model(path=MODEL_PATH):
    model = json.loads(path.read_text())
    if len(model["coef"]) != len(model["mean"]) or len(model["coef"]) != len(model["std"]):
        raise ValueError("model.json: coef, mean and std differ in length")
    return model


def oriented_frames(table, halves=None):
    """The frames of the halves whose direction of attack is clear (MIN_MARGIN_M)."""
    oriented = np.zeros(table.n_frames, dtype=bool)
    for start, stop, home, away in (halves or shape_features.deepest_medians(table)):
        # The two teams' deepest players must sit at opposite ends; otherwise the direction of
        # attack is a guess, and a wrong one would invert every opinion of the half.
        oriented[start:stop] = abs(away - home) >= MIN_MARGIN_M
    return oriented


def build(table, report=None, model=None):
    """(p_dead, p_home, weight) over every video frame; fills `report` (a dict) if given.

    `model` defaults to saha/shape/model.json; the leave-one-match-out check passes its own.
    """
    model = model or load_model()
    feats, ok, names = shape_features.pair_features(table)
    if len(names) != len(model["coef"]):
        raise ValueError("model.json has {} features, the code makes {}".format(
            len(model["coef"]), len(names)))
    p_dead, p_home, weight = evidence.empty(table.n_frames)
    halves = shape_features.deepest_medians(table)
    hidden = table.graded & ~table.ball_visible & ok & oriented_frames(table, halves)
    if hidden.any():
        p = predict(model, feats[hidden].astype(np.float64))
        p_home[hidden] = np.clip(p, P_MIN, P_MAX)
        weight[hidden] = WEIGHT
    if report is not None:
        report["halves"] = halves
        report["hidden"] = int((table.graded & ~table.ball_visible).sum())
        report["covered"] = int(hidden.sum())
        report["home_share"] = float((p_home[hidden] > 0.5).mean()) if hidden.any() else 0.0
        report["confident"] = (float((np.abs(p_home[hidden] - 0.5) >= 0.2).mean())
                               if hidden.any() else 0.0)
    return p_dead, p_home, weight


def summary(report):
    """Aggregate lines: the orientation check, coverage and shares."""
    lines = []
    for number, (start, stop, home, away) in enumerate(report["halves"], 1):
        margin = abs(away - home)
        lines.append("half {}: home attacks {} x; deepest players' medians {:.1f} m and {:.1f} m "
                     "(margin {:.1f} m){}".format(
                         number, "high" if home < away else "low", home, away, margin,
                         "" if margin >= MIN_MARGIN_M else
                         "  WARNING: below {:g} m, no opinion in this half".format(MIN_MARGIN_M)))
    lines.append("missing-ball graded frames {}: opinion on {:.1%}, home on {:.1%} of them, "
                 "|p - 0.5| >= 0.2 on {:.1%}".format(
                     report["hidden"], report["covered"] / max(report["hidden"], 1),
                     report["home_share"], report["confident"]))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", help="data bundle root (default: MTAH_DATA_DIR or ./data)")
    args = parser.parse_args(argv)
    if args.data:
        paths.use_data_dir(args.data)
    report = {}
    p_dead, p_home, weight = build(frame_table.load(), report=report)
    print(summary(report))
    path = evidence.write_evidence(
        "shape", p_dead, p_home, weight, producer="saha.shape.build",
        params={"P_MIN": P_MIN, "P_MAX": P_MAX, "WEIGHT": WEIGHT,
                "model_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()})
    print("\n".join(evidence.validate_file(path)[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
