"""The frame alignment, defined once. Every package converts through these functions.

    submission row / video frame   0 .. 199318
    ATD frame_id                   1 .. 199319     video_frame = frame_id - 1
    Smart Tagging seconds          from the first frame of the tactical video
                                                   video_frame = round(seconds * 25)
    homography `frame`             0 .. 199318     already a video frame

See DATA.md ("Frame alignment"). The per-half offsets in games.py describe the counter the ground
truth was derived from; participants never see that counter, so nothing here uses them.

Rounding follows the starter code exactly: Python's round(), i.e. half-to-even on the exact
binary value of seconds * fps. np.rint() is the same operation, which is what the vectorised
version uses.
"""

import numpy as np

from saha.common.constants import FPS, GRADED_HALVES, N_FRAMES


def video_frame_of_atd(frame_id):
    """ATD frame_id (1-based) -> video frame (0-based). Scalars or arrays."""
    if np.isscalar(frame_id):
        return int(frame_id) - 1
    return np.asarray(frame_id, dtype=np.int64) - 1


def atd_frame_id_of(video_frame):
    """Video frame -> ATD frame_id. The inverse of video_frame_of_atd."""
    if np.isscalar(video_frame):
        return int(video_frame) + 1
    return np.asarray(video_frame, dtype=np.int64) + 1


def video_frame_of_seconds(seconds, fps=FPS):
    """Smart Tagging seconds -> video frame, exactly as starter/load_smart_tagging.py does it."""
    if np.isscalar(seconds):
        return int(round(float(seconds) * fps))
    return np.rint(np.asarray(seconds, dtype=np.float64) * fps).astype(np.int64)


def half_slices():
    """[(half, start, stop)] for the graded halves, half-open, ready for array slicing."""
    return [(half, first, last + 1) for half, first, last in GRADED_HALVES]


def half_of_frames(n_frames=N_FRAMES):
    """int8 array over video frames: 1 or 2 inside a graded half, 0 elsewhere."""
    half = np.zeros(n_frames, dtype=np.int8)
    for number, start, stop in half_slices():
        half[start:min(stop, n_frames)] = number
    return half


def graded_mask(n_frames=N_FRAMES):
    """True on the 158,857 frames the scorer grades."""
    return half_of_frames(n_frames) > 0
