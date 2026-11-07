"""Train the dead-ball classifier on the twenty SkillCorner matches.

    python -m saha.deadball.train      -> saha/deadball/models/dead_skillcorner.pkl (+ .json)

The model is committed (DECISIONS D020), so the pipeline never runs this; it is here to rebuild
the model from the external data.

Every match is used twice: as delivered, and with the ball hidden the way WS5's degradation hides
it (saha/external/degrade.py, its default settings, seed 0: long stretches placed at random,
whatever the play). On SkillCorner the ball is missing on 6% of live frames and 55% of dead
ones, so a classifier trained on the delivered matches alone learns that a missing ball means a
stoppage; on WS5's DFL matches it then called 92% of live frames under a hidden stretch dead.
With both copies, held out: +0.158 spellF1 on SkillCorner under WS5's hiding, −0.003 as
delivered; DFL dead-ball F1 0.790 to 0.917 (experiments/20261107_ws2_hidden-training.md). The first run downloads and converts the matches (about
1.8 GB, saha/deadball/skillcorner.py); training then takes about seven minutes. Trained on external
data only: SkillCorner open data, MIT licence, (c) SkillCorner.

Training runs on one thread: the trees come out the same on 1, 4 and 12 threads, but
scikit-learn stores the thread count in the pickle, so pinning it makes the file the same on
every machine. Run it with the pinned numpy (requirements-saha.txt): numpy 2.5.4 computes a
percentile of the players' speeds one float32 step away from 2.3.5, and most trees change. The manifest next to the model records the data, the seeds, the library versions
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

from saha.common.constants import X
from saha.external.degrade import Degradation, hide_mask

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


def hidden_ball(d, config=Degradation()):
    """The match's ball with WS5's hidden stretches (saha/external/degrade.hide_mask)."""
    half = np.zeros(len(d["alive"]), dtype=np.int8)
    for number, (a, b) in enumerate(d["runs"]):
        half[a:b] = number + 1
    labels = np.where(d["valid"], 0, X).astype(np.int8)    # only "scored or not" is read
    hidden = hide_mask(half, labels, config, np.random.default_rng(config.seed))
    ball = d["ball"].copy()
    ball[hidden] = np.nan
    return ball


def training_set(matches=skillcorner.MATCHES):
    """(X, y): features and dead (1) / alive (0) on every SUBSAMPLE-th tracked frame, every
    match as delivered and then every match with WS5's hidden ball."""
    copies = {"delivered": ([], []), "hidden": ([], [])}
    for match in matches:
        d = skillcorner.load(match)
        rows = np.flatnonzero(d["valid"])[::SUBSAMPLE]
        y = (d["alive"][rows] == 0).astype(np.int8)
        for name, ball in (("delivered", d["ball"]), ("hidden", hidden_ball(d))):
            sig = Signals(ball[:, :2], d["players"], d["team"],
                          tuple(float(v) for v in d["pitch"]))
            copies[name][0].append(external_features(sig)[rows])
            copies[name][1].append(y)
    Xs = copies["delivered"][0] + copies["hidden"][0]
    ys = copies["delivered"][1] + copies["hidden"][1]
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
        "augmentation": "every match twice: as delivered, and with the ball hidden by "
                        "saha.external.degrade.hide_mask, Degradation() defaults, seed 0",
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
