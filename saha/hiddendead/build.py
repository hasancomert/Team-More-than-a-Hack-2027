"""WS1: a second opinion on WS2's dead calls where the ball is missing (DECISIONS D028).

    python -m saha.hiddendead.build      -> cache/evidence/hiddendead.parquet  (after deadball)

- **Where:** graded frames where the ball is missing, inside the played part of the half (between
  WS2's first and last alive frame), and called dead by WS2. Every frame before a half's kick-off
  stays as WS2 decided.
- **What:** P(dead) from the gradient-boosting model of `saha/hiddendead/models/hiddendead.pkl`,
  trained by `python -m saha.hiddendead.train` on WS5's DFL matches only. It is clipped to
  [P_MIN, P_MAX] and written as `p_dead` at weight 1; `p_home` stays NaN.
- **How it counts:** `configs/pipeline.toml` gives it `overrides = "deadball"`, so the pipeline
  drops WS2's `p_dead` on these frames and the decoder weighs this one instead, with its usual
  costs. It can only soften WS2's dead calls: it never adds one where WS2 says alive.

It reads WS2's decision from cache/evidence/deadball.parquet, so the config lists it after
`deadball`. It runs on any frame table: ours, or one of WS5's external tables. Everything printed
is aggregate.
"""

import argparse
import hashlib
import pickle

import numpy as np

from saha.common import evidence, frame_table, paths
from saha.hiddendead import features as hd_features
from saha.hiddendead.train import MODEL_PATH

P_MIN, P_MAX = 0.1, 0.9
WEIGHT = 1.0


def load_model(path=MODEL_PATH):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def build(table, ws2_dead, report=None, model=None):
    """(p_dead, p_home, weight) over every video frame; fills `report` (a dict) if given.

    `ws2_dead`: (N,) bool, WS2's decision. `model` defaults to the committed one.
    """
    saved = {"model": model} if model is not None else load_model()
    index, x = hd_features.inputs(table, ws2_dead)
    if "n_features" in saved and saved["n_features"] != x.shape[1]:
        raise ValueError("the model has {} features, the code makes {}".format(
            saved["n_features"], x.shape[1]))
    p_dead, p_home, weight = evidence.empty(table.n_frames)
    if len(index):
        p_dead[index] = np.clip(saved["model"].predict_proba(x)[:, 1], P_MIN, P_MAX)
        weight[index] = WEIGHT
    if report is not None:
        graded_dead = np.asarray(ws2_dead, dtype=bool) & table.graded
        report["ws2_dead"] = int(graded_dead.sum())
        report["covered"] = int(len(index))
        report["alive_share"] = float((p_dead[index] < 0.5).mean()) if len(index) else 0.0
    return p_dead, p_home, weight


def summary(report):
    return ("WS2's dead calls on graded frames {}: second opinion on {} of them (ball missing, "
            "inside play), {:.1%} of those rather alive".format(
                report["ws2_dead"], report["covered"], report["alive_share"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", help="data bundle root (default: MTAH_DATA_DIR or ./data)")
    args = parser.parse_args(argv)
    if args.data:
        paths.use_data_dir(args.data)
    if not paths.evidence_path("deadball").is_file():
        raise SystemExit("cache/evidence/deadball.parquet is missing: run saha.deadball.build "
                         "first (configs/pipeline.toml lists hiddendead after deadball)")
    ws2 = evidence.read_evidence("deadball")
    ws2_dead = np.isfinite(ws2.p_dead) & (ws2.p_dead > 0.5)
    report = {}
    p_dead, p_home, weight = build(frame_table.load(), ws2_dead, report=report)
    print(summary(report))
    path = evidence.write_evidence(
        "hiddendead", p_dead, p_home, weight, producer="saha.hiddendead.build",
        params={"P_MIN": P_MIN, "P_MAX": P_MAX, "WEIGHT": WEIGHT,
                "model_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()})
    print("\n".join(evidence.validate_file(path)[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
