"""WS1: who has the ball while it is tracked, from a model trained on WS5's DFL matches.

    python -m saha.onball.build          -> cache/evidence/onball.parquet

- **Where:** `p_home` only on graded frames where the ball is tracked, the frames WS3 and
  `nearest` also cover, in halves whose direction of attack is clear (`saha.shape.build`'s
  10 m margin). `p_dead` stays NaN. DECISIONS D027 records the exception to D011.
- **What:** the gradient-boosting model of `saha/onball/models/onball.pkl`, trained by
  `python -m saha.onball.train` on WS5's DFL matches only, applied to `saha.onball.features`. It
  is symmetric between the teams: the mean of its answers with the teams in both orders.
- **Strength:** the probability is clipped to [P_MIN, P_MAX] and written at weight 1.
  `configs/pipeline.toml` sets its scale.

It runs on any frame table: ours, or one of WS5's external tables. Everything printed is
aggregate.
"""

import argparse
import hashlib
import pickle

import numpy as np

from saha.common import evidence, frame_table, paths
from saha.onball import features as onball_features
from saha.onball.train import MODEL_PATH, predict
from saha.shape.build import oriented_frames

P_MIN, P_MAX = 0.1, 0.9
WEIGHT = 1.0


def load_model(path=MODEL_PATH):
    with open(path, "rb") as handle:
        saved = pickle.load(handle)
    return saved


def build(table, report=None, model=None):
    """(p_dead, p_home, weight) over every video frame; fills `report` (a dict) if given.

    `model` defaults to the committed one; the leave-one-match-out check passes its own.
    """
    saved = {"model": model} if model is not None else load_model()
    x, xs, ok, _ = onball_features.design(table)
    if "n_features" in saved and saved["n_features"] != x.shape[1]:
        raise ValueError("the model has {} features, the code makes {}".format(
            saved["n_features"], x.shape[1]))
    p_dead, p_home, weight = evidence.empty(table.n_frames)
    tracked = table.graded & table.ball_visible & ok & oriented_frames(table)
    if tracked.any():
        p = predict(saved["model"], x[tracked], xs[tracked])
        p_home[tracked] = np.clip(p, P_MIN, P_MAX)
        weight[tracked] = WEIGHT
    if report is not None:
        report["tracked"] = int((table.graded & table.ball_visible).sum())
        report["covered"] = int(tracked.sum())
        report["home_share"] = float((p_home[tracked] > 0.5).mean()) if tracked.any() else 0.0
        report["confident"] = (float((np.abs(p_home[tracked] - 0.5) >= 0.2).mean())
                               if tracked.any() else 0.0)
    return p_dead, p_home, weight


def summary(report):
    return ("tracked graded frames {}: opinion on {:.1%}, home on {:.1%} of them, "
            "|p - 0.5| >= 0.2 on {:.1%}".format(
                report["tracked"], report["covered"] / max(report["tracked"], 1),
                report["home_share"], report["confident"]))


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
        "onball", p_dead, p_home, weight, producer="saha.onball.build",
        params={"P_MIN": P_MIN, "P_MAX": P_MAX, "WEIGHT": WEIGHT,
                "model_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()})
    print("\n".join(evidence.validate_file(path)[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
