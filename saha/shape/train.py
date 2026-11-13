"""Train the shape model on WS5's DFL matches and write saha/shape/model.json.

    python -m saha.shape.train            # all seven matches -> saha/shape/model.json
    python -m saha.shape.train --lomo     # leave-one-match-out accuracy, writes nothing

External open data only: DFL/IDSSE (CC BY 4.0), downloaded by saha.external.dfl into
cache/external/dfl/ on first use. Nothing of our match is read, fitted or looked at (D016, D017).

- **Features:** `saha.shape.features` on each match, degraded by WS5's harness (seed 0). The
  features use no ball, so the degradation only splits the slots by half.
- **Labels:** DFL's frame-level possession, made closer to CHALLENGE.md's controlled possession.
  Every possession shorter than 2 s that sits between two possessions of the other team is merged
  into them (`controlled`). Only frames where the ball is alive are used.
- **Model:** logistic regression on the difference between the two teams' features (home minus
  away), with every example also added with the teams swapped and no intercept. So the model
  cannot learn a home or an away side: swapping the teams turns p into 1 - p.
- **Missing players:** DFL tracks all 22 players on every frame; our tracking about 20.5. So
  every match is also added once with players removed in runs (mean DROP_MEAN_S missing, share
  DROP_SHARE, seeded per match). Rule fixed before the run, in
  experiments/20261113_ws1_shape-model.md: held out at our level, the accuracy rose from 0.856 to
  0.876 and fell by 0.002 with every player present.
- **Fitting:** features standardised, every STEP-th alive frame, scikit-learn's lbfgs, C = 1.
  The fit is deterministic. The committed model.json is the artifact the pipeline reads.
"""

import argparse
import json
import platform
from dataclasses import replace

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression

from saha.common import paths
from saha.common.constants import A, FPS, H
from saha.shape import features as shape_features

MATCHES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
CONTROLLED_S = 2.0
STEP = 5
C = 1.0
DROP_SHARE = 0.10           # share of player-frames removed in the second copy of each match
DROP_MEAN_S = 5.0           # mean length of a removed run, seconds
DROP_SEED = 101
MODEL_PATH = paths.REPO_ROOT / "saha" / "shape" / "model.json"


def drop_players(table, share, rng):
    """A copy of `table` with each slot missing in runs: mean DROP_MEAN_S long, `share` of the
    player-frames overall (alternating exponential runs, starting state drawn per slot)."""
    if share <= 0:
        return table
    xy = table.players_xy.copy()
    speed = table.players_speed.copy()
    n, slots = xy.shape[:2]
    mean_off = DROP_MEAN_S * FPS
    mean_on = mean_off * (1.0 - share) / share
    for slot in range(slots):
        t, missing = 0, rng.random() < share
        while t < n:
            length = int(rng.exponential(mean_off if missing else mean_on)) + 1
            if missing:
                xy[t:t + length, slot] = np.nan
                speed[t:t + length, slot] = np.nan
            t += length
            missing = not missing
    return replace(table, players_xy=xy, players_speed=speed)


def match_seed(match_id):
    return DROP_SEED + sum(map(ord, match_id))


def controlled(labels, seconds=CONTROLLED_S):
    """Merge every H or A run shorter than `seconds` whose neighbours on both sides are runs of
    the other team (A H A -> A A A). One pass over the original runs; D and X never change."""
    labels = np.asarray(labels)
    out = labels.copy()
    change = np.flatnonzero(np.diff(labels)) + 1
    starts = np.concatenate(([0], change))
    stops = np.concatenate((change, [len(labels)]))
    codes = labels[starts]
    limit = int(round(seconds * FPS))
    for i in range(1, len(starts) - 1):
        code = codes[i]
        if code not in (H, A) or stops[i] - starts[i] >= limit:
            continue
        other = A if code == H else H
        if codes[i - 1] == other and codes[i + 1] == other:
            out[starts[i]:stops[i]] = other
    return out


def match_data(match_id):
    """(features (N, 2, K), ok, labels (controlled), names, features and ok with players
    removed) of one DFL match."""
    from saha.external import degrade, dfl
    match = dfl.prepare(match_id)
    table, labels = degrade.degrade(match, replace(degrade.Degradation(), seed=0))
    feats, ok, names = shape_features.pair_features(table)
    dropped = drop_players(table, DROP_SHARE, np.random.default_rng(match_seed(match_id)))
    feats_drop, ok_drop, _ = shape_features.pair_features(dropped)
    return feats, ok, controlled(labels), names, feats_drop, ok_drop


def rows_of(data):
    """Training rows of one match: the match as delivered, and its copy with players removed."""
    feats, ok, labels, _, feats_drop, ok_drop = data
    first = training_rows(feats, ok, labels)
    second = training_rows(feats_drop, ok_drop, labels)
    return np.concatenate([first[0], second[0]]), np.concatenate([first[1], second[1]])


def training_rows(feats, ok, labels):
    """(difference features, y) on every STEP-th frame where the ball is alive."""
    alive = ok & ((labels == H) | (labels == A))
    index = np.flatnonzero(alive)[::STEP]
    diff = feats[index, 0, :] - feats[index, 1, :]
    return diff, (labels[index] == H).astype(int)


def fit(diff, y):
    """{'mean', 'std', 'coef'}: the symmetric logistic model on difference features."""
    x = np.concatenate([diff, -diff])
    target = np.concatenate([y, 1 - y])
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    model = LogisticRegression(C=C, fit_intercept=False, max_iter=5000)
    model.fit((x - mean) / std, target)
    return {"mean": mean.tolist(), "std": std.tolist(), "coef": model.coef_[0].tolist()}


def predict(model, feats):
    """P(home has the ball) for (N, 2, K) features; NaN where a feature is missing."""
    diff = feats[:, 0, :] - feats[:, 1, :]
    z = ((diff - np.asarray(model["mean"])) / np.asarray(model["std"])) @ np.asarray(model["coef"])
    return 1.0 / (1.0 + np.exp(-z))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lomo", action="store_true", help="leave-one-match-out accuracy only")
    args = parser.parse_args(argv)
    data = {m: match_data(m) for m in MATCHES}
    names = data[MATCHES[0]][3]
    if args.lomo:
        for held in MATCHES:
            rows = [rows_of(data[m]) for m in MATCHES if m != held]
            model = fit(np.concatenate([r[0] for r in rows]), np.concatenate([r[1] for r in rows]))
            feats, ok, labels, _, feats_drop, ok_drop = data[held]
            out = []
            for name, f, o in (("as delivered", feats, ok), ("players removed", feats_drop,
                                                             ok_drop)):
                alive = o & ((labels == H) | (labels == A))
                p = predict(model, f[alive])
                out.append("{} {:.3f}".format(name, ((p > 0.5) == (labels[alive] == H)).mean()))
            print("held out {}: accuracy on alive frames, {}".format(held, ", ".join(out)))
        return 0
    rows = [rows_of(data[m]) for m in MATCHES]
    diff = np.concatenate([r[0] for r in rows])
    y = np.concatenate([r[1] for r in rows])
    model = fit(diff, y)
    model.update({
        "version": 1,
        "features": ["{} (home minus away)".format(n) for n in names],
        "label": "DFL ball possession on alive frames, possessions under {:g} s between two of "
                 "the other team merged into them".format(CONTROLLED_S),
        "training": {
            "script": "saha/shape/train.py",
            "data": "DFL/IDSSE open data (Bassek, Weber, Rein and Memmert, 2025, Scientific Data), "
                    "matches " + ", ".join(MATCHES),
            "licence": "CC BY 4.0",
            "frames": int(len(y)), "step": STEP, "C": C, "solver": "lbfgs",
            "missing_players": {"share": DROP_SHARE, "mean_s": DROP_MEAN_S,
                                "seed": "{} + the sum of the match id's characters".format(
                                    DROP_SEED)},
            "seed": "the missing-player copies are seeded; the fit itself is deterministic",
            "libraries": {"python": platform.python_version(), "numpy": np.__version__,
                          "scikit-learn": sklearn.__version__},
        },
    })
    MODEL_PATH.write_text(json.dumps(model, indent=1, sort_keys=True) + "\n")
    print("wrote {} from {} frames of {} matches".format(MODEL_PATH, len(y), len(MATCHES)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
