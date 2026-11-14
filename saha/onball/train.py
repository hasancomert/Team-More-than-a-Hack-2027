"""Train the on-ball model on WS5's DFL matches: saha/onball/models/onball.pkl (+ .json).

    python -m saha.onball.train            # all seven matches -> the model and its manifest
    python -m saha.onball.train --lomo     # leave-one-match-out accuracy, writes nothing

External open data only: DFL/IDSSE (CC BY 4.0), downloaded by saha.external.dfl into
cache/external/dfl/ on first use. Nothing of our match is read, fitted or looked at (D016, D017).

- **Inputs:** `saha.onball.features.design` on each match as delivered, with the ball on every
  frame.
- **Labels:** DFL's frame-level possession on alive frames, with every possession under 2 s
  between two of the other team merged into them (`saha.shape.train.controlled`).
- **Model:** scikit-learn's HistGradientBoostingClassifier on every 5th alive frame. Every
  example is also added with the teams swapped, and the prediction is symmetrised in build.
- **Determinism.** Training runs on one thread with random_state 0 and no early stopping, as WS2
  does: scikit-learn stores the thread count in the pickle. The manifest records the data, the
  library versions and the file's SHA256, and the guard checks it (D020).
"""

import argparse
import hashlib
import json
import pickle
import platform
import statistics

import numpy as np

from saha.common import paths
from saha.common.constants import A, H
from saha.onball import features as onball_features
from saha.shape.train import MATCHES, controlled

STEP = 5
MODEL_PATH = paths.REPO_ROOT / "saha" / "onball" / "models" / "onball.pkl"
PARAMS = {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 200,
          "l2_regularization": 1.0, "early_stopping": False, "random_state": 0}


def classifier():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(**PARAMS)


def match_rows(match_id, step=STEP):
    """(x, x_swapped, y) on every `step`-th alive frame of one DFL match, as delivered."""
    from saha.external import degrade, dfl
    table, labels = degrade.degrade(dfl.prepare(match_id), degrade.NONE)
    labels = controlled(labels)
    x, xs, ok, _ = onball_features.design(table)
    alive = ok & ((labels == H) | (labels == A))
    index = np.flatnonzero(alive)[::step]
    return x[index], xs[index], (labels[index] == H).astype(int)


def fit(rows):
    from threadpoolctl import threadpool_limits
    x = np.concatenate([np.concatenate([r[0], r[1]]) for r in rows])
    y = np.concatenate([np.concatenate([r[2], 1 - r[2]]) for r in rows])
    with threadpool_limits(limits=1):
        return classifier().fit(x, y), int(len(y))


def predict(model, x, xs):
    """Symmetric P(home has the ball): the mean of the two orderings of the teams."""
    return 0.5 * (model.predict_proba(x)[:, 1] + 1.0 - model.predict_proba(xs)[:, 1])


def lomo(data):
    from saha.external import degrade, dfl
    accuracies = []
    for held in MATCHES:
        model, _ = fit([data[m] for m in MATCHES if m != held])
        table, labels = degrade.degrade(dfl.prepare(held), degrade.Degradation())
        labels = controlled(labels)
        x, xs, ok, _ = onball_features.design(table)
        judge = ok & table.ball_visible & ((labels == H) | (labels == A))
        p = predict(model, x[judge], xs[judge])
        accuracies.append(float(((p > 0.5) == (labels[judge] == H)).mean()))
        print("held out {}: accuracy where the ball is tracked (seed 0) {:.3f}".format(
            held, accuracies[-1]), flush=True)
    print("mean {:.3f}".format(statistics.mean(accuracies)))


def main(argv=None):
    import sklearn
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lomo", action="store_true", help="leave-one-match-out accuracy only")
    args = parser.parse_args(argv)
    data = {m: match_rows(m) for m in MATCHES}
    if args.lomo:
        lomo(data)
        return 0
    model, frames = fit([data[m] for m in MATCHES])
    names = onball_features.names()
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as handle:
        pickle.dump({"model": model, "n_features": len(names), "names": names,
                     "sklearn": sklearn.__version__}, handle)
    digest = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    manifest = {
        "model": MODEL_PATH.name,
        "training_script": "saha/onball/train.py",
        "command": "python -m saha.onball.train",
        "data": "DFL/IDSSE open data (Bassek, Weber, Rein and Memmert, 2025, Scientific Data), "
                "matches " + ", ".join(MATCHES) + "; labels: DFL ball possession on alive frames, "
                "possessions under 2 s between two of the other team merged into them",
        "licence": "CC BY 4.0, (c) DFL / Sportec Solutions",
        "augmentation": "every example also with the teams swapped",
        "seed": PARAMS["random_state"],
        "params": PARAMS,
        "libraries": {"python": platform.python_version(), "numpy": np.__version__,
                      "scikit-learn": sklearn.__version__, "threads": 1},
        "frames": frames,
        "features": len(names),
        "sha256": digest,
    }
    with open(str(MODEL_PATH) + ".json", "w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print("on-ball model: {} frames, {} features, sha256 {}".format(frames, len(names), digest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
