"""Match constants, read from games.py rather than re-typed, so there is one source of truth.

Label codes are the scorer's own (evaluation/possession_metrics.py): H=0, A=1, D=2, X=3. Use
these, never your own numbering, so that arrays can go straight into the official metrics.
"""

from saha.common.paths import games

GAME = games.GAME
N_FRAMES = games.GAMES[GAME]["n_video_frames"]          # 199319 rows in a submission
FPS = games.GAMES[GAME]["fps"]                           # 25
PITCH_LENGTH_M, PITCH_WIDTH_M = games.GAMES[GAME]["pitch"]  # 105.3 x 68.0, see DATA.md

# Inclusive video-frame ranges of the two graded halves: [(1, 11185, 90651), (2, 110808, 190197)].
# Published by the organisers; everything outside them is ungraded and costs nothing as X.
GRADED_HALVES = tuple(games.half_video_blocks(GAME))

LABELS = ("H", "A", "D", "X")
H, A, D, X = 0, 1, 2, 3

# The only ground truth the team may use, and only to check that the pipeline is correct.
PUBLIC_WINDOW = (11185, 18684)  # inclusive video frames, see evaluation/gt/*_public_meta.json
