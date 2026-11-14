"""Train the second opinion on WS5's DFL matches: saha/hiddendead/models/hiddendead.pkl (+ .json).

    python -m saha.hiddendead.train            # all seven matches -> the model and its manifest
    python -m saha.hiddendead.train --lomo     # leave-one-match-out frame rates, writes nothing

External open data only: DFL/IDSSE (CC BY 4.0), downloaded by saha.external.dfl into
cache/external/dfl/ on first use. Nothing of our match is read, fitted or looked at (D016, D017).

- **Data:** each match is degraded by WS5's harness at seeds 100, 101 and 102 (the ball hidden in
  long stretches, as ours; never the evaluation seeds 0-2), and WS2's decoder runs on it.
- **Frames:** `saha.hiddendead.features.inputs`: graded, ball missing, inside the played part of
  the half, and called dead by WS2; every 3rd of those with a known label.
- **Target:** DFL's own ball status: dead or alive.
- **Model:** scikit-learn's HistGradientBoostingClassifier with the on-ball model's parameters, on
  one thread with no early stopping, so the pickle is the same on every run (as WS2's and
  `saha.onball`'s). The manifest records the data, the library versions and the file's SHA256,
  and the guard checks it (D020).
"""

import argparse
import hashlib
import json
import pickle
import platform
import statistics
from dataclasses import replace

import numpy as np

from saha.common import paths
from saha.common.constants import A, D, H
from saha.hiddendead import features as hd_features
from saha.onball.train import PARAMS
from saha.shape.train import MATCHES

SEEDS = (100, 101, 102)
STEP = 3
MODEL_PATH = paths.REPO_ROOT / "saha" / "hiddendead" / "models" / "hiddendead.pkl"


def classifier():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(**PARAMS)


def ws2_dead(table):
    """WS2's decision on every frame of a table, as the pipeline reads it (p_dead > 0.5)."""
    from saha.deadball import build as ws2
    p_dead = ws2.build(table)[0]
    return np.isfinite(p_dead) & (p_dead > 0.5)


def match_rows(match_id, seed, step=STEP):
    """(x, y) on every `step`-th frame of the population with a known label, one degraded match."""
    from saha.external import degrade, dfl
    config = replace(degrade.Degradation(), seed=seed)
    table, labels = degrade.degrade(dfl.prepare(match_id), config)
    index, x = hd_features.inputs(table, ws2_dead(table))
    label = labels[index]
    rows = np.flatnonzero((label == D) | (label == H) | (label == A))[::step]
    return x[rows], (label[rows] == D).astype(int)


def fit(rows):
    from threadpoolctl import threadpool_limits
    x = np.concatenate([r[0] for r in rows])
    y = np.concatenate([r[1] for r in rows])
    with threadpool_limits(limits=1):
        return classifier().fit(x, y), int(len(y))


def lomo(data):
    from saha.external import degrade, dfl
    rates = []
    for held in MATCHES:
        model, _ = fit([data[(m, s)] for m in MATCHES for s in SEEDS if m != held])
        table, labels = degrade.degrade(dfl.prepare(held), degrade.Degradation())
        index, x = hd_features.inputs(table, ws2_dead(table))
        label = labels[index]
        live = (label == H) | (label == A)
        p = model.predict_proba(x)[:, 1]
        rates.append(float((p[live] > 0.5).mean()) if live.any() else 0.0)
        print("held out {}: of WS2's dead calls on live frames with the ball missing, {:.1%} stay "
              "dead (seed 0)".format(held, rates[-1]), flush=True)
    print("mean {:.3f}".format(statistics.mean(rates)))


def main(argv=None):
    import sklearn
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lomo", action="store_true", help="leave-one-match-out rates only")
    args = parser.parse_args(argv)
    data = {(m, s): match_rows(m, s) for m in MATCHES for s in SEEDS}
    if args.lomo:
        lomo(data)
        return 0
    model, frames = fit([data[(m, s)] for m in MATCHES for s in SEEDS])
    names = hd_features.names()
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as handle:
        pickle.dump({"model": model, "n_features": len(names), "names": names,
                     "sklearn": sklearn.__version__}, handle)
    digest = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    manifest = {
        "model": MODEL_PATH.name,
        "training_script": "saha/hiddendead/train.py",
        "command": "python -m saha.hiddendead.train",
        "data": "DFL/IDSSE open data (Bassek, Weber, Rein and Memmert, 2025, Scientific Data), "
                "matches " + ", ".join(MATCHES) + ", each degraded by saha.external.degrade at "
                "seeds " + ", ".join(map(str, SEEDS)) + "; frames: ball missing and called dead by "
                "WS2 (saha.deadball); target: DFL's ball status",
        "licence": "CC BY 4.0, (c) DFL / Sportec Solutions",
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
    print("second-opinion model: {} frames, {} features, sha256 {}".format(frames, len(names),
                                                                         digest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
