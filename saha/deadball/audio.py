"""The referee's whistle, from the tactical video's audio track.

A whistle is the one direct trace of a decision the tracking cannot see: a foul or an offside
stops play with no ball event at all. It is also blown to restart play after long breaks, so a
whistle alone is a cue, not a stoppage; events.py decides which is which.

A whistle is one steady tone; a crowd spreads its sound over every frequency. So the score of a
frame is how far the strongest frequency in the whistle band stands above the band's own median
level. It needs no assumption about which whistle the referee used, only that its tone lies in
the band. Nothing here comes from this match's audio:

  - the band holds the tones that referee-whistle makers publish for their whistles (Dolfin PRO
    3.67 and 4.15 kHz, Acme Tornado 3.74 and 4.10 kHz, Fox 40 about 3.8 kHz on average) and the
    whistles of TU Dortmund's labelled recordings (peaks at 4.7-5.3 kHz);
  - the analysis (44.1 kHz, 1024-sample Hamming windows, 20 log10 |FFT|) is the one of those
    recordings (Kleingarn and Brämer, Whistle Detection Dataset, TUDOdata 2024,
    doi:10.17877/tudodata-2024-m0f6bmi1, CC BY-SA 4.0), so that the threshold, chosen on them,
    means the same here. docs/tech/ws2_deadball.md has the numbers.

The audio is decoded with the ffmpeg command-line tool, streamed so that memory stays small, and
the per-frame score is cached under cache/deadball/.

Not used by default: the whistle needs the video and ffmpeg on every machine that reproduces the
submission, which is a team decision (D006). build.py --whistle turns it on.
"""

import subprocess

import numpy as np

from saha.common import paths
from saha.common.constants import N_FRAMES

SAMPLE_RATE = 44100
FPS = 25
NFFT = 1024
BAND = (3000.0, 6000.0)    # Hz: where a referee's whistle has its tone
# dB by which the band's strongest frequency stands above the band's median. On TU Dortmund's
# labelled test recordings, read like here (one window per 1/25 s, the blast rules below), 38 dB
# is the lowest threshold with no false blast in their 0.45 h of background, and it finds 73% of
# their 249 blasts. The rule was set before the numbers were seen: a false whistle holds a
# stoppage until a confirmed restart, a missed one leaves the other evidence.
WHISTLE_DB = 38.0
MIN_FRAMES = 3      # 0.12 s: a referee's blast is longer than that
BRIDGE = 3          # gaps this short inside one blast are dropouts, not two blasts
BLOCK = 5000        # frames per step of the stream


def band_mask():
    freqs = np.fft.rfftfreq(NFFT, 1.0 / SAMPLE_RATE)
    return (freqs >= BAND[0]) & (freqs <= BAND[1])


def frame_scores(samples, starts, band):
    """dB by which the band's strongest frequency stands above the band's median, for windows
    of `samples` starting at `starts`."""
    window = np.hamming(NFFT)
    frames = samples[starts[:, None] + np.arange(NFFT)] * window
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(np.abs(np.fft.rfft(frames, axis=1)))
    db = np.maximum(db, -300.0)   # digital silence
    db = db[:, band]
    return (db.max(axis=1) - np.median(db, axis=1)).astype(np.float32)


def whistle_score(read, n_frames):
    """(n_frames,) score per video frame, from a reader of mono s16le samples at SAMPLE_RATE.

    Frame i is scored on the window centred on the middle of its 1/25 s; frames past the end of
    the audio score 0.
    """
    hop = SAMPLE_RATE // FPS
    centre = hop // 2 - NFFT // 2       # window start relative to the frame's first sample
    band = band_mask()
    score = np.zeros(n_frames, dtype=np.float32)
    buffer, base, ended = np.zeros(0), 0, False   # buffer[0] is sample number `base`
    odd = b""                                     # half a sample left over by the last read
    for first in range(0, n_frames, BLOCK):
        idx = np.arange(first, min(n_frames, first + BLOCK))
        starts = idx * hop + centre
        need = int(starts[-1]) + NFFT
        while not ended and base + len(buffer) < need:
            raw = read(2 * (need - base - len(buffer)))
            ended = not raw
            raw, odd = odd + raw, b""
            if len(raw) % 2:
                raw, odd = raw[:-1], raw[-1:]
            if raw:
                buffer = np.concatenate([buffer, np.frombuffer(raw, dtype="<i2") / 32768.0])
        ok = starts + NFFT <= base + len(buffer)
        if ok.any():
            score[idx[ok]] = frame_scores(buffer, starts[ok] - base, band)
        cut = min(max(0, int(starts[-1]) + hop - base), len(buffer))   # only what was read
        buffer, base = buffer[cut:], base + cut
    return score


def decode_score(video_path, n_frames):
    """whistle_score of a video's audio track, decoded by ffmpeg as it streams."""
    command = ["ffmpeg", "-v", "error", "-i", str(video_path), "-vn", "-ac", "1",
               "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    try:
        score = whistle_score(process.stdout.read, n_frames)
    finally:
        process.stdout.close()
        process.wait()
    return score


def load_whistle_score(refresh=False):
    """Per-video-frame whistle score of the tactical video, cached."""
    path = paths.cache_dir() / "deadball" / "whistle_score_v2.npy"
    if path.is_file() and not refresh:
        return np.load(path)
    score = decode_score(paths.require_assets(["video"])[0], N_FRAMES)
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
