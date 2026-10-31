"""Writing, reading and scoring label arrays, through the official code wherever it exists.

The official scorer (evaluation/possession_metrics.py) is loaded from its file, never copied, so
every number we print is defined by the same code that ranks the competition.
"""

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np

from saha.common import alignment, paths
from saha.common.constants import FPS, LABELS, N_FRAMES, X


def metrics():
    """The official possession_metrics module, imported from evaluation/ unmodified."""
    if "possession_metrics" not in sys.modules:
        location = paths.EVALUATION_DIR / "possession_metrics.py"
        spec = importlib.util.spec_from_file_location("possession_metrics", location)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules["possession_metrics"] = module
    return sys.modules["possession_metrics"]


def write_submission(path, codes):
    """frame,label CSV with LF line endings, byte-identical on every OS. Returns the path."""
    codes = np.asarray(codes)
    if len(codes) != N_FRAMES:
        raise ValueError("{} labels, expected {}".format(len(codes), N_FRAMES))
    names = np.array(LABELS)[codes]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "frame,label\n" + "".join("{},{}\n".format(i, n) for i, n in enumerate(names))
    path.write_bytes(body.encode("ascii"))
    return path


def read_codes(path):
    pm = metrics()
    return pm.label_codes(pm.read_label_csv(str(path)))


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def public_gt():
    return read_codes(paths.PUBLIC_GT)


def score(codes, gt_codes=None, bootstrap=1000):
    """The official evaluate() against the public GT (or a given one), headline numbers only."""
    pm = metrics()
    gt = public_gt() if gt_codes is None else gt_codes
    result = pm.evaluate(gt, np.asarray(codes, dtype=np.int8), bootstrap_n=bootstrap)
    keys = ("macro_f1", "spell_f1_mean", "spell_bootstrap_low", "spell_bootstrap_high",
            "n_pred_spells", "n_gt_spells", "mean_spell_duration_s", "gt_mean_spell_duration_s",
            "alive_dead_f1", "team_acc_both_alive")
    return {k: result[k] for k in keys}


def match_stats(codes):
    """Match-wide description of a submission over the graded halves, the scorer's way.

    X is forward-filled inside each half before possessions are counted, exactly as spellF1 sees
    it. These numbers need no ground truth: compare them across versions and against football
    sense (CLAUDE.md), never against the public window.
    """
    pm = metrics()
    codes = np.asarray(codes, dtype=np.int8)
    runs = [(start, stop) for _, start, stop in alignment.half_slices()]
    graded = alignment.graded_mask(len(codes))
    filled = pm.fill_abstentions(codes, runs)
    spells = pm.possession_spells(filled, runs)
    lengths = np.array([b - a for a, b, _ in spells], dtype=np.float64)
    alive = (filled[graded] == 0) | (filled[graded] == 1)
    return {
        "n_possessions": len(spells),
        "mean_duration_s": float(lengths.mean() / FPS) if len(lengths) else 0.0,
        "median_duration_s": float(np.median(lengths) / FPS) if len(lengths) else 0.0,
        "short_possessions_lt_2s": int((lengths < 2 * FPS).sum()),
        "d_rate": float((filled[graded] == 2).mean()),
        "x_rate": float((codes[graded] == X).mean()),
        "home_share": float((filled[graded] == 0).sum() / alive.sum()) if alive.any() else 0.0,
    }
