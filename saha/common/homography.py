"""Per-frame homography: normalised pitch coordinates (0-1) <-> tactical video pixels.

Row i is video frame i. Check `valid` before using a matrix (the tactical file is valid on every
frame, but the code should not assume it). See starter/load_homography.py for the full notes.
"""

import numpy as np
import pandas as pd

from saha.common import paths

MATRIX_COLUMNS = ["h{}{}".format(r, c) for r in range(3) for c in range(3)]


def load(data=None, source=paths.SOURCE):
    """(matrices (N, 3, 3) float64, reprojection_error (N,), valid (N,) bool)."""
    path = paths.require_assets(["homography"], data, source)[0]
    table = pd.read_csv(path, float_precision="round_trip")
    frames = table["frame"].to_numpy()
    if not np.array_equal(frames, np.arange(len(frames))):
        raise ValueError("{}: 'frame' is not 0..n-1 in order".format(path.name))
    matrices = table[MATRIX_COLUMNS].to_numpy(dtype=np.float64).reshape(-1, 3, 3)
    return (matrices, table["reprojection_error"].to_numpy(dtype=np.float64),
            table["valid"].to_numpy().astype(bool))


def project(matrix, x, y):
    """Normalised pitch (0-1) -> pixels. NaN in, NaN out."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    w = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
    return ((matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]) / w,
            (matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]) / w)


def unproject(matrix, px, py):
    """Pixels -> normalised pitch (0-1)."""
    return project(np.linalg.inv(matrix), px, py)
