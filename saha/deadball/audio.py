"""The referee's whistle, from the tactical video's audio track.

The tactical recording carries stadium audio, and the referee's whistle is in it: two steady tones
near 3.8 kHz and 4.3 kHz, 25-40 dB above the crowd. The two bands below were measured on the
whistle before the first-half kickoff, which is outside the graded frames.

A whistle is the one direct trace of a decision the tracking cannot see: a foul or an offside
stops play with no ball event at all. It is also blown to restart play after long breaks, so a
whistle alone is a cue, not a stoppage; events.py decides which is which.

The audio is decoded with the ffmpeg command-line tool and the per-frame score is cached under
cache/deadball/.

Not used by default: the bands were measured on this match's audio, which the organizers' rules
do not allow (STATUS.md, WS2 item 5), and the whistle needs the video and ffmpeg on every machine
that reproduces the submission, which is a team decision (D006). build.py --whistle turns it on.
"""

import subprocess

import numpy as np

from saha.common import paths
from saha.common.constants import N_FRAMES

SAMPLE_RATE = 16000
FPS = 25
NFFT = 1024
BAND_LOW = (3700.0, 3960.0)    # Hz, first whistle tone
BAND_HIGH = (4180.0, 4400.0)   # Hz, second whistle tone
# Reference bands either side, for the crowd level the tones are measured against
REFERENCE = ((3000.0, 3600.0), (4500.0, 5200.0))
# dB above the reference, both tones at once. The score distribution falls steeply up to about
# 15 dB and flattens into a tail from about 20 dB: that tail is the whistles.
WHISTLE_DB = 20.0
MIN_FRAMES = 3      # 0.12 s
BRIDGE = 3          # gaps this short inside one blast are dropouts, not two blasts


def decode_audio(video_path):
    """Mono 16 kHz float32 samples of the video's audio track."""
    command = ["ffmpeg", "-v", "error", "-i", video_path, "-vn", "-ac", "1",
               "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"]
    raw = subprocess.run(command, check=True, stdout=subprocess.PIPE).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def whistle_score(samples, n_frames):
    """(n_frames,) dB by which the weaker of the two whistle tones stands above the crowd."""
    hop = SAMPLE_RATE // FPS
    freqs = np.fft.rfftfreq(NFFT, 1.0 / SAMPLE_RATE)
    low = (freqs >= BAND_LOW[0]) & (freqs <= BAND_LOW[1])
    high = (freqs >= BAND_HIGH[0]) & (freqs <= BAND_HIGH[1])
    ref = np.zeros_like(low)
    for a, b in REFERENCE:
        ref |= (freqs >= a) & (freqs <= b)
    window = np.hanning(NFFT).astype(np.float32)
    usable = max(0, min(n_frames, (len(samples) - NFFT) // hop))
    score = np.zeros(n_frames, dtype=np.float32)
    step = 20000
    for first in range(0, usable, step):
        idx = np.arange(first, min(usable, first + step))
        start = np.clip(idx * hop + hop // 2 - NFFT // 2, 0, len(samples) - NFFT)
        spectrum = np.abs(np.fft.rfft(samples[start[:, None] + np.arange(NFFT)] * window, axis=1))
        db = 20.0 * np.log10(spectrum + 1e-9)
        reference = np.median(db[:, ref], axis=1)
        score[idx] = np.minimum(db[:, low].max(axis=1), db[:, high].max(axis=1)) - reference
    return score


def load_whistle_score(refresh=False):
    """Per-video-frame whistle score of the tactical video, cached."""
    path = paths.cache_dir() / "deadball" / "whistle_score.npy"
    if path.is_file() and not refresh:
        return np.load(path)
    video = paths.require_assets(["video"])[0]
    score = whistle_score(decode_audio(str(video)), N_FRAMES)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, score)
    return score


def whistle_events(score, threshold=WHISTLE_DB):
    """[(onset, end)] frames of whistle blasts."""
    hot = score > threshold
    edges = np.diff(np.concatenate(([0], hot.astype(np.int8), [0])))
    blasts = []
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if blasts and a - blasts[-1][1] <= BRIDGE:
            blasts[-1][1] = b
        else:
            blasts.append([a, b])
    return [(int(a), int(b)) for a, b in blasts if b - a >= MIN_FRAMES]
