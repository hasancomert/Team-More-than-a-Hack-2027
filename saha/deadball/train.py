"""Train the dead-ball classifier on the twenty SkillCorner matches.

    python -m saha.deadball.train      -> saha/deadball/models/dead_skillcorner.pkl (+ .json)

The model is committed (DECISIONS D020), so the pipeline never runs this; it is here to rebuild
the model from the external data. The first run downloads and converts the matches (about
1.8 GB, saha/deadball/skillcorner.py); training then takes about two minutes. Trained on external
data only: SkillCorner open data, MIT licence, (c) SkillCorner.

Training runs on one thread: the trees come out the same on 1, 4 and 12 threads, but
scikit-learn stores the thread count in the pickle, so pinning it makes the file the same on
every machine. The manifest next to the model records the data, the seeds, the library versions
and the file's SHA256, which the guard checks against the file.

The settings (classifier, frame subsampling, features) are the ones checked by leaving
SkillCorner matches out, with EXTERNAL_TAU and EXTERNAL_WEIGHT in dead.py: docs/tech/ws2_deadball.md.
"""

import argparse
import hashlib
import json
import pickle
import platform

import numpy as np

from . import skillcorner
from .dead import Signals, external_features, model_path

SUBSAMPLE = 5           # train on every 5th tracked frame: neighbours are near copies
SEEDS = (0, 1, 2)       # one classifier per seed, logits averaged
SCRIPT = "saha/deadball/train.py"


def classifier(seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          min_samples_leaf=200, l2_regularization=1.0,
                                          random_state=seed)


def training_set(matches=skillcorner.MATCHES):
    """(X, y): features and dead (1) / alive (0) on every SUBSAMPLE-th tracked frame."""
    Xs, ys = [], []
    for match in matches:
        d = skillcorner.load(match)
        sig = Signals(d["ball"][:, :2], d["players"], d["team"],
                      tuple(float(v) for v in d["pitch"]))
        rows = np.flatnonzero(d["valid"])[::SUBSAMPLE]
        Xs.append(external_features(sig)[rows])
        ys.append((d["alive"][rows] == 0).astype(np.int8))
    return np.concatenate(Xs), np.concatenate(ys)


def manifest(path, frames, n_features, rate):
    """The D020 manifest of the model file at path."""
    import scipy
    import sklearn
    with open(path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    return {
        "model": path.name,
        "training_script": SCRIPT,
        "command": "python -m saha.deadball.train",
        "data": "SkillCorner open data, {} A-League 2024/25 matches ({}), github.com/{} at "
                "commit {}; labels from its phases of play (alive inside a phase, dead "
                "between phases)".format(len(skillcorner.MATCHES),
                                         ", ".join(skillcorner.MATCHES),
                                         skillcorner.REPOSITORY, skillcorner.COMMIT),
        "licence": "MIT, (c) SkillCorner",
        "seed": list(SEEDS),
        "libraries": {"python": platform.python_version(), "numpy": np.__version__,
                      "scipy": scipy.__version__, "scikit-learn": sklearn.__version__,
                      "threads": 1},
        "frames": frames,
        "features": n_features,
        "dead_rate": round(rate, 6),
        "sha256": digest,
    }


def train(path=None):
    """Fit the classifiers, write them to path (default dead.model_path()) with the manifest
    next to them. Returns the path."""
    import sklearn
    from threadpoolctl import threadpool_limits

    path = path or model_path()
    X, y = training_set()
    with threadpool_limits(limits=1):
        models = [classifier(seed).fit(X, y) for seed in SEEDS]
    rate = float(y.mean())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump({"models": models, "prior_logit": float(np.log(rate / (1 - rate))),
                     "n_features": X.shape[1], "sklearn": sklearn.__version__,
                     "trained_on": "SkillCorner open data, {} A-League 2024/25 matches, commit "
                                   "{}".format(len(skillcorner.MATCHES), skillcorner.COMMIT),
                     "frames": int(len(y)), "dead_rate": rate}, handle)
    info = manifest(path, int(len(y)), X.shape[1], rate)
    with open(str(path) + ".json", "w") as handle:
        json.dump(info, handle, indent=2)
        handle.write("\n")
    print("dead-ball classifier: {} frames, dead rate {:.3f}, {} features, sklearn {}, "
          "sha256 {}".format(len(y), rate, X.shape[1], sklearn.__version__, info["sha256"]),
          flush=True)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args(argv)
    train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
