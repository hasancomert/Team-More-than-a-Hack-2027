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
MODEL_PATH = paths.REPO_ROOT / "saha" / "shape" / "model.json"


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
    """(features (N, 2, K), ok, labels (controlled), names) of one DFL match."""
    from saha.external import degrade, dfl
    match = dfl.prepare(match_id)
    table, labels = degrade.degrade(match, replace(degrade.Degradation(), seed=0))
    feats, ok, names = shape_features.pair_features(table)
    return feats, ok, controlled(labels), names


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
            rows = [training_rows(*data[m][:3]) for m in MATCHES if m != held]
            model = fit(np.concatenate([r[0] for r in rows]), np.concatenate([r[1] for r in rows]))
            feats, ok, labels, _ = data[held]
            alive = ok & ((labels == H) | (labels == A))
            p = predict(model, feats[alive])
            right = (p > 0.5) == (labels[alive] == H)
            print("held out {}: alive frames {}, accuracy {:.3f}".format(held, int(alive.sum()),
                                                                         right.mean()))
        return 0
    rows = [training_rows(*data[m][:3]) for m in MATCHES]
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
            "seed": "none needed: the fit is deterministic",
            "libraries": {"python": platform.python_version(), "numpy": np.__version__,
                          "scikit-learn": sklearn.__version__},
        },
    })
    MODEL_PATH.write_text(json.dumps(model, indent=1, sort_keys=True) + "\n")
    print("wrote {} from {} frames of {} matches".format(MODEL_PATH, len(y), len(MATCHES)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
